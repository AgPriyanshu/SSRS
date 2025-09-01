#!/usr/bin/env python3
"""
Inference script for UNetFormer building detection.
Loads trained model and runs inference on RGB + DEM data to generate building masks.
"""

import os
import sys
import torch
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.features import shapes
import argparse
from datetime import datetime
from typing import Tuple, List
from pathlib import Path
import warnings
import geopandas as gpd
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import unary_union

# Add current directory to path for imports
sys.path.append('.')
from unetformer_mmsam import UNetFormer

warnings.filterwarnings('ignore')
# from model_wrapper import ModelWrapper  # Has conflicting argparser
# from config import config  # May have conflicting argparser


class BuildingInference:
    """Building detection inference pipeline using trained UNetFormer."""
    
    def __init__(self, model_path: str, device: str = 'cuda', training_resolution: float = 0.25, 
                 probability_threshold: float = 0.7, min_building_area: float = 25.0, 
                 overlap_ratio: float = 0.25, debug_tiles: bool = False, auto_resolution: bool = False):
        """Initialize inference pipeline for vector boundary generation.
        
        Args:
            model_path: Path to trained model weights
            device: Device to run inference on ('cuda' or 'cpu')
            training_resolution: Resolution in meters used during training (default: 0.25m for MEGA model)
            probability_threshold: Threshold for converting probability to binary mask
            min_building_area: Minimum building area in square meters to keep
            overlap_ratio: Overlap ratio between tiles (0.0 = no overlap, 0.5 = 50% overlap)
            debug_tiles: If True, save individual tile outputs for debugging
            auto_resolution: If True, automatically choose optimal resolution based on input data
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = None
        self.window_size = (256, 256)  # Training window size
        self.overlap_ratio = overlap_ratio
        self.overlap = int(self.window_size[0] * overlap_ratio)  # Calculate overlap in pixels
        # Auto-detect training resolution if requested
        if auto_resolution:
            self.training_resolution = self._detect_training_resolution()
        else:
            self.training_resolution = training_resolution
            
        self.probability_threshold = probability_threshold
        self.min_building_area = min_building_area
        self.debug_tiles = debug_tiles
        self.debug_dir = None
        self.auto_resolution = auto_resolution
        
        print(f"🖥️  Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"🎯 Training resolution: {self.training_resolution}m per pixel")
        print("🔄 Vector-per-tile mode: Both ortho and DSM resampled to training resolution")
        print("🎯 HIGH CONFIDENCE MODE: Enhanced filtering for reliable building detection")
        if self.overlap_ratio > 0:
            print(f"📐 Overlapping inference: {self.overlap_ratio*100:.0f}% overlap ({self.overlap} pixels)")
            print("   ✨ Better building boundary detection across tile edges")
        else:
            print("📐 Non-overlapping inference: Sharp tile boundaries")
        print(f"📊 Probability threshold: {self.probability_threshold} (HIGH CONFIDENCE)")
        print(f"📏 Min building area: {self.min_building_area} m²")
        
        if self.debug_tiles:
            print("🐛 Debug mode: Individual tile outputs will be saved")
        
        self._load_model()
    
    def _setup_debug_directory(self, base_output_path: str) -> None:
        """Setup debug directory for tile outputs."""
        if not self.debug_tiles:
            return
            
        base_path = Path(base_output_path).parent
        debug_name = Path(base_output_path).stem + "_debug_tiles"
        self.debug_dir = base_path / debug_name
        self.debug_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        (self.debug_dir / "rgb_tiles").mkdir(exist_ok=True)
        (self.debug_dir / "dem_tiles").mkdir(exist_ok=True)  
        (self.debug_dir / "prediction_tiles").mkdir(exist_ok=True)
        (self.debug_dir / "binary_tiles").mkdir(exist_ok=True)
        
        print(f"🐛 Debug directory created: {self.debug_dir}")
    
    def _save_debug_tile(self, tile_idx: int, rgb_patch: np.ndarray, dem_patch: np.ndarray, 
                        prediction: np.ndarray, y: int, x: int, y_end: int, x_end: int) -> None:
        """Save debug information for a single tile."""
        if not self.debug_tiles or self.debug_dir is None:
            return
            
        try:
            from PIL import Image
            
            tile_name = f"tile_{tile_idx:04d}_{y}_{x}_{y_end}_{x_end}"
            
            # Save RGB tile (convert from CHW to HWC and normalize)
            if rgb_patch.shape[0] == 3:  # CHW format
                rgb_viz = np.transpose(rgb_patch, (1, 2, 0))
                rgb_viz = np.clip(rgb_viz * 255, 0, 255).astype(np.uint8)
                rgb_path = self.debug_dir / "rgb_tiles" / f"{tile_name}_rgb.png"
                Image.fromarray(rgb_viz).save(rgb_path)
            
            # Save DEM tile (single channel)
            if dem_patch.shape[0] == 1:  # CHW format
                dem_viz = dem_patch[0]
                # Normalize DEM for visualization
                dem_min, dem_max = dem_viz.min(), dem_viz.max()
                if dem_max > dem_min:
                    dem_viz = ((dem_viz - dem_min) / (dem_max - dem_min) * 255).astype(np.uint8)
                else:
                    dem_viz = np.zeros_like(dem_viz, dtype=np.uint8)
                dem_path = self.debug_dir / "dem_tiles" / f"{tile_name}_dem.png"
                Image.fromarray(dem_viz, mode='L').save(dem_path)
            
            # Save probability prediction
            prob_viz = (np.clip(prediction, 0, 1) * 255).astype(np.uint8)
            prob_path = self.debug_dir / "prediction_tiles" / f"{tile_name}_prob.png"
            Image.fromarray(prob_viz, mode='L').save(prob_path)
            
            # Save binary prediction (threshold at 0.5)
            binary_viz = ((prediction > 0.5) * 255).astype(np.uint8)
            binary_path = self.debug_dir / "binary_tiles" / f"{tile_name}_binary.png"
            Image.fromarray(binary_viz, mode='L').save(binary_path)
            
        except Exception as e:
            print(f"   Warning: Failed to save debug tile {tile_idx}: {e}")
    
    def _load_model(self) -> None:
        """Load the trained model from checkpoint."""
        print(f"📂 Loading model from: {self.model_path}")
        
        # Initialize model with 2 classes (background + buildings)
        self.model = UNetFormer(num_classes=2)
        
        # Load weights
        try:
            if self.device.type == 'cpu':
                checkpoint = torch.load(self.model_path, map_location='cpu')
            else:
                checkpoint = torch.load(self.model_path)
                self.model = self.model.cuda()
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            print("✅ Model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def _load_image(self, image_path: str) -> Tuple[np.ndarray, dict]:
        """Load image and return array with metadata.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (image_array, metadata_dict)
        """
        print(f"📁 Loading: {image_path}")
        
        with rasterio.open(image_path) as src:
            # Read all bands
            image = src.read()  # Shape: (bands, height, width)
            
            # Get metadata
            metadata = {
                'transform': src.transform,
                'crs': src.crs,
                'height': src.height,
                'width': src.width,
                'count': src.count,
                'dtype': src.dtypes[0],  # Get dtype of first band
                'bounds': src.bounds
            }
            
            print(f"   Shape: {image.shape}, Bands: {src.count}, CRS: {src.crs}")
            
        return image, metadata
    
    def _resample_to_resolution(self, source_image: np.ndarray, source_meta: dict, 
                               target_resolution: float) -> Tuple[np.ndarray, dict]:
        """Resample source image to target resolution.
        
        Args:
            source_image: Source image array (C, H, W)
            source_meta: Source image metadata
            target_resolution: Target pixel resolution in meters
            
        Returns:
            Tuple of (resampled_image, new_metadata)
        """
        from rasterio.crs import CRS
        from rasterio.warp import calculate_default_transform
        
        # Calculate current resolution from transform
        current_resolution = abs(source_meta['transform'][0])
        
        # Check if we're in geographic coordinates (degrees)
        is_geographic = source_meta['crs'].is_geographic if hasattr(source_meta['crs'], 'is_geographic') else 'EPSG:4326' in str(source_meta['crs'])
        
        if is_geographic:
            print("   Input in geographic coordinates (degrees)")
            print(f"   Current pixel size: {current_resolution:.8f} degrees")
            
            # For geographic data, we need to reproject to a projected CRS first
            # Use UTM zone based on center longitude
            bounds = source_meta['bounds']
            center_lon = (bounds.left + bounds.right) / 2
            center_lat = (bounds.top + bounds.bottom) / 2
            
            # Determine UTM zone
            utm_zone = int((center_lon + 180) / 6) + 1
            utm_epsg = 32600 + utm_zone if center_lat >= 0 else 32700 + utm_zone  # North/South hemisphere
            target_crs = CRS.from_epsg(utm_epsg)
            
            print(f"   Reprojecting to UTM Zone {utm_zone}: EPSG:{utm_epsg}")
            
            # Calculate transform for target resolution in projected CRS
            dst_transform, dst_width, dst_height = calculate_default_transform(
                source_meta['crs'], target_crs,
                source_meta['width'], source_meta['height'], 
                *bounds,
                resolution=target_resolution
            )
            
        else:
            # Data already in projected coordinates
            print("   Input in projected coordinates")
            print(f"   Current resolution: {current_resolution:.3f}m")
            
            if abs(current_resolution - target_resolution) < 1e-6:
                print(f"   Already at target resolution ({current_resolution:.3f}m)")
                return source_image, source_meta
            
            target_crs = source_meta['crs']
            
            # Calculate new dimensions
            scale_factor = current_resolution / target_resolution
            dst_height = max(1, int(source_image.shape[1] * scale_factor))
            dst_width = max(1, int(source_image.shape[2] * scale_factor))
            
            # Create new transform for target resolution
            bounds = source_meta['bounds']
            dst_transform = from_bounds(
                bounds.left, bounds.bottom, bounds.right, bounds.top,
                dst_width, dst_height
            )
        
        print(f"   Target resolution: {target_resolution:.3f}m")
        print(f"   Output dimensions: {dst_height} x {dst_width}")
        
        # Create output array
        resampled = np.zeros((source_image.shape[0], dst_height, dst_width), 
                           dtype=source_image.dtype)
        
        # Resample each band
        for band in range(source_image.shape[0]):
            reproject(
                source=source_image[band],
                destination=resampled[band],
                src_transform=source_meta['transform'],
                src_crs=source_meta['crs'],
                dst_transform=dst_transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear
            )
        
        # Update metadata
        new_meta = source_meta.copy()
        new_meta.update({
            'transform': dst_transform,
            'height': dst_height,
            'width': dst_width,
            'crs': target_crs
        })
        
        return resampled, new_meta
    
    def _analyze_resolution_strategy(self, rgb_resolution: float, dem_resolution: float) -> dict:
        """Analyze input resolutions and recommend optimal strategy.
        
        Args:
            rgb_resolution: RGB pixel resolution in meters
            dem_resolution: DEM pixel resolution in meters
            
        Returns:
            Dictionary with resolution analysis and recommendations
        """
        analysis = {
            'rgb_resolution': rgb_resolution,
            'dem_resolution': dem_resolution,
            'training_resolution': self.training_resolution,
            'strategy': 'resample_both',
            'recommended_resolution': self.training_resolution,
            'warnings': []
        }
        
        # Check if input resolutions are significantly better than training resolution
        rgb_better = rgb_resolution < (self.training_resolution * 0.8)  # 20% better threshold
        dem_better = dem_resolution < (self.training_resolution * 0.8)
        
        # Check if input resolutions are significantly worse than training resolution  
        rgb_worse = rgb_resolution > (self.training_resolution * 1.5)  # 50% worse threshold
        dem_worse = dem_resolution > (self.training_resolution * 1.5)
        
        if rgb_better and dem_better:
            # Both inputs are higher resolution - consider using native resolution
            native_res = max(rgb_resolution, dem_resolution)  # Use worse of the two good resolutions
            analysis['strategy'] = 'use_native'
            analysis['recommended_resolution'] = native_res
            analysis['warnings'].append(f"Input data is higher resolution than training ({native_res:.3f}m vs {self.training_resolution:.3f}m)")
            analysis['warnings'].append("Using native resolution to preserve detail - may affect model accuracy")
            
        elif rgb_worse or dem_worse:
            # One or both inputs are much lower resolution
            analysis['strategy'] = 'resample_both' 
            analysis['warnings'].append("Input data has lower resolution than training - upsampling may create artifacts")
            
        elif abs(rgb_resolution - dem_resolution) > 0.1:  # Significant resolution mismatch
            analysis['warnings'].append(f"RGB and DEM have different resolutions ({rgb_resolution:.3f}m vs {dem_resolution:.3f}m)")
            
        # Check for very high resolution that might be overkill
        if min(rgb_resolution, dem_resolution) < 0.1:
            analysis['warnings'].append("Very high resolution detected (<0.1m) - consider if this level of detail is needed")
            
        return analysis
    
    def _detect_training_resolution(self) -> float:
        """Attempt to detect the training resolution from available information.
        
        Returns:
            Detected training resolution in meters, or default if not found
        """
        # Try to find dataset info for MEGA multiscale dataset
        mega_info_path = Path("prepared_datasets/MEGA_multiscale/dataset_info.json")
        if mega_info_path.exists():
            try:
                import json
                with open(mega_info_path, 'r') as f:
                    info = json.load(f)
                    if 'base_resolution' in info:
                        detected_res = info['base_resolution']
                        print(f"🔍 Detected training resolution from MEGA dataset: {detected_res:.3f}m")
                        return detected_res
            except Exception:
                pass
        
        # Try MOPR multiscale dataset (another main dataset)
        mopr_info_path = Path("prepared_datasets/MOPR_multiscale_multiscale/dataset_info.json")
        if mopr_info_path.exists():
            try:
                import json
                with open(mopr_info_path, 'r') as f:
                    info = json.load(f)
                    if 'base_resolution' in info:
                        detected_res = info['base_resolution']
                        print(f"🔍 Detected training resolution from MOPR dataset: {detected_res:.3f}m")
                        return detected_res
            except Exception:
                pass
        
        # Try other dataset info files as fallback
        for info_file in Path("prepared_datasets").glob("*/dataset_info.json"):
            if 'MEGA' in str(info_file) or 'MOPR' in str(info_file):
                continue  # Already tried these above
            try:
                import json
                with open(info_file, 'r') as f:
                    info = json.load(f)
                    if 'base_resolution' in info:
                        detected_res = info['base_resolution']
                        print(f"🔍 Detected training resolution from {info_file.parent.name}: {detected_res:.3f}m")
                        return detected_res
            except Exception:
                continue
        
        # Default fallback
        print(f"⚠️  Could not detect training resolution, using default: {self.training_resolution:.3f}m")
        return self.training_resolution
    
    def _resample_to_match(self, source_image: np.ndarray, source_meta: dict, 
                          target_shape: Tuple[int, int], target_meta: dict) -> np.ndarray:
        """Resample source image to match target shape and geospatial properties.
        
        Args:
            source_image: Source image array (C, H, W)
            source_meta: Source image metadata
            target_shape: Target shape (H, W)
            target_meta: Target image metadata
            
        Returns:
            Resampled image array
        """
        # Create output array
        resampled = np.zeros((source_image.shape[0], target_shape[0], target_shape[1]), 
                           dtype=source_image.dtype)
        
        # Resample each band
        for band in range(source_image.shape[0]):
            reproject(
                source=source_image[band],
                destination=resampled[band],
                src_transform=source_meta['transform'],
                src_crs=source_meta['crs'],
                dst_transform=target_meta['transform'],
                dst_crs=target_meta['crs'],
                resampling=Resampling.bilinear
            )
        
        return resampled
    
    def _normalize_image(self, image: np.ndarray, is_rgb: bool = True, nodata_value: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize image for model input and return nodata mask.
        
        Args:
            image: Input image array
            is_rgb: Whether this is RGB image (True) or DEM (False)
            nodata_value: Value representing nodata pixels
            
        Returns:
            Tuple of (normalized_image, nodata_mask)
        """
        # Create nodata mask
        if nodata_value is not None:
            nodata_mask = np.any(image == nodata_value, axis=0) if len(image.shape) > 2 else (image == nodata_value)
        else:
            nodata_mask = np.zeros(image.shape[-2:], dtype=bool)
        
        # Create a copy for processing
        image_copy = image.copy().astype(np.float32)
        
        if is_rgb:
            # RGB normalization - match training exactly: scale to [0,1] then ImageNet normalization
            if image.dtype == np.uint8:
                image_copy = image_copy / 255.0
            elif image.dtype == np.uint16:
                image_copy = image_copy / 65535.0
            else:
                # Normalize by percentiles, excluding nodata
                if np.any(~nodata_mask):
                    valid_pixels = image_copy[:, ~nodata_mask] if len(image_copy.shape) > 2 else image_copy[~nodata_mask]
                    p2, p98 = np.percentile(valid_pixels, (2, 98))
                    image_copy = np.clip((image_copy - p2) / (p98 - p2), 0, 1)
                else:
                    image_copy = np.clip(image_copy, 0, 1)
            
            # Apply ImageNet normalization (like training)
            imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]  # (C, 1, 1)
            imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]   # (C, 1, 1)
            
            if len(image_copy.shape) == 3 and image_copy.shape[0] == 3:  # (C, H, W)
                image_copy = (image_copy - imagenet_mean) / imagenet_std
        else:
            # DEM - keep raw values for per-patch normalization later in _predict_patch
            # No global normalization needed since training uses per-patch normalization
            image_copy = image_copy.astype(np.float32)  # Ensure float32 format
            
        # Set nodata pixels to 0 (they'll be masked anyway)
        if len(image_copy.shape) > 2:
            image_copy[:, nodata_mask] = 0
        else:
            image_copy[nodata_mask] = 0
                
        return image_copy, nodata_mask
    
    def _get_sliding_windows(self, height: int, width: int) -> list:
        """Calculate sliding window positions without extracting patches.
        
        Args:
            height: Image height
            width: Image width
            
        Returns:
            List of window positions (y, x, y_end, x_end)
        """
        window_h, window_w = self.window_size
        stride_h = window_h - self.overlap
        stride_w = window_w - self.overlap
        
        windows = []
        
        # Calculate regular grid windows
        for y in range(0, height, stride_h):
            for x in range(0, width, stride_w):
                y_end = min(y + window_h, height)
                x_end = min(x + window_w, width)
                
                # Ensure minimum window size
                if (y_end - y) >= window_h // 2 and (x_end - x) >= window_w // 2:
                    # Adjust start position if we're at the edge
                    if y_end == height and y_end - y < window_h:
                        y = max(0, height - window_h)
                        y_end = height
                    if x_end == width and x_end - x < window_w:
                        x = max(0, width - window_w)
                        x_end = width
                    
                    windows.append((y, x, y_end, x_end))
        
        print(f"   Generated {len(windows)} sliding windows")
        return windows
    
    def _predict_patch(self, rgb_patch: np.ndarray, dem_patch: np.ndarray) -> np.ndarray:
        """Run inference on a single patch.
        
        Args:
            rgb_patch: RGB patch (C, H, W) - already normalized
            dem_patch: DEM patch (1, H, W) - NOT yet normalized per-patch 
            
        Returns:
            Prediction array (H, W)
        """
        with torch.no_grad():
            # Apply per-patch DSM normalization to match training approach
            dem_patch_norm = dem_patch.copy().astype(np.float32)
            dem_single = dem_patch_norm[0]  # Extract (H, W) from (1, H, W)
            
            # Per-patch min-max normalization [0,1] - EXACTLY like training
            if dem_single.max() > dem_single.min():
                dem_single = (dem_single - dem_single.min()) / (dem_single.max() - dem_single.min())
            else:
                # Constant elevation - set to 0 (like training fallback)
                dem_single = np.zeros_like(dem_single)
            
            # Convert to tensors and add batch dimension
            rgb_tensor = torch.from_numpy(rgb_patch).unsqueeze(0).to(self.device)  # (1, C, H, W)
            dem_tensor = torch.from_numpy(dem_single).unsqueeze(0).to(self.device)  # (1, H, W)
            
            # Run inference
            output = self.model(rgb_tensor, dem_tensor)  # (1, num_classes, H, W)
            
            # Get probability for building class (class 1)
            prob = torch.softmax(output, dim=1)  # (1, num_classes, H, W)
            building_prob = prob[0, 1].cpu().numpy()  # (H, W)
            
        return building_prob
    
    def _create_tile_vectors(self, prediction: np.ndarray, tile_transform: rasterio.Affine, crs: str) -> List[Polygon]:
        """Create vector polygons from a single tile's prediction with high confidence filtering.
        
        Args:
            prediction: Prediction array (H, W) with probabilities [0-1]
            tile_transform: Geospatial transform for this tile
            crs: Coordinate reference system
            
        Returns:
            List of Polygon objects for buildings in this tile
        """
        # Convert probability to binary mask with high confidence threshold
        binary_mask = (prediction >= self.probability_threshold).astype(np.uint8)
        
        # Skip if no buildings detected
        if not np.any(binary_mask):
            return []
        
        polygons = []
        
        try:
            # Convert raster to vector polygons
            for geom, value in shapes(binary_mask, mask=binary_mask > 0, transform=tile_transform):
                if value == 1:  # Building pixel
                    polygon = shape(geom)
                    
                    # Calculate average confidence for this polygon
                    # Get pixel coordinates within the polygon bounds
                    minx, miny, maxx, maxy = polygon.bounds
                    
                    # Convert to pixel coordinates
                    inv_transform = ~tile_transform
                    px_minx, px_maxy = inv_transform * (minx, miny)
                    px_maxx, px_miny = inv_transform * (maxx, maxy)
                    
                    px_minx, px_miny = max(0, int(px_minx)), max(0, int(px_miny))
                    px_maxx, px_maxy = min(prediction.shape[1], int(px_maxx)+1), min(prediction.shape[0], int(px_maxy)+1)
                    
                    if px_minx < px_maxx and px_miny < px_maxy:
                        # Get prediction values within polygon bounds
                        polygon_region = prediction[px_miny:px_maxy, px_minx:px_maxx]
                        polygon_mask = binary_mask[px_miny:px_maxy, px_minx:px_maxx]
                        
                        if np.any(polygon_mask):
                            # Calculate average confidence for building pixels in this polygon
                            building_pixels = polygon_region[polygon_mask > 0]
                            avg_confidence = np.mean(building_pixels)
                            
                            # Apply confidence-based area filtering
                            # Higher confidence buildings can be smaller, lower confidence need larger area
                            confidence_multiplier = max(0.5, (1.0 - avg_confidence) * 2.0)  # Scale 0.7-1.0 conf to 0.5-1.0 multiplier
                            adjusted_min_area = self.min_building_area * confidence_multiplier
                            
                            # Filter by confidence-adjusted minimum area
                            if polygon.area >= adjusted_min_area:
                                # Simplify polygon slightly to reduce complexity
                                simplified = polygon.simplify(tolerance=0.5, preserve_topology=True)
                                if simplified.is_valid and not simplified.is_empty:
                                    polygons.append(simplified)
            
        except Exception as e:
            print(f"   Warning: Vector creation failed for tile: {e}")
        
        return polygons
    
    def _merge_tile_vectors(self, all_polygons: List[Polygon], crs: str) -> gpd.GeoDataFrame:
        """Merge overlapping polygons from all tiles into clean building boundaries.
        
        Args:
            all_polygons: List of all polygons from all tiles
            crs: Coordinate reference system
            
        Returns:
            GeoDataFrame with merged building polygons
        """
        if not all_polygons:
            return gpd.GeoDataFrame(columns=['geometry'], crs=crs)
        
        print(f"🔗 Merging {len(all_polygons)} tile polygons...")
        
        try:
            # Union all polygons to merge overlapping buildings
            merged_geometry = unary_union(all_polygons)
            
            # Handle different geometry types
            final_polygons = []
            if isinstance(merged_geometry, Polygon):
                final_polygons = [merged_geometry]
            elif isinstance(merged_geometry, MultiPolygon):
                final_polygons = list(merged_geometry.geoms)
            
            # Filter by area again after merging
            final_polygons = [
                p for p in final_polygons 
                if p.area >= self.min_building_area and p.is_valid
            ]
            
            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(geometry=final_polygons, crs=crs)
            
            # Add building attributes
            gdf['building_id'] = range(1, len(gdf) + 1)
            gdf['area_m2'] = gdf.geometry.area
            gdf['perimeter_m'] = gdf.geometry.length
            
            print(f"✅ Created {len(gdf)} merged building polygons")
            return gdf
            
        except Exception as e:
            print(f"❌ Polygon merging failed: {e}")
            # Fallback: return individual polygons
            gdf = gpd.GeoDataFrame(geometry=all_polygons, crs=crs)
            gdf['building_id'] = range(1, len(gdf) + 1)
            gdf['area_m2'] = gdf.geometry.area
            gdf['perimeter_m'] = gdf.geometry.length
            return gdf
    
    def _tile_by_tile_vector_inference(self, rgb_image: np.ndarray, dem_image: np.ndarray, 
                                      base_transform: rasterio.Affine, crs: str) -> List[Polygon]:
        """Run inference tile-by-tile and create vector boundaries for each tile.
        
        Args:
            rgb_image: RGB image array (C, H, W)
            dem_image: DEM image array (1, H, W)
            base_transform: Geospatial transform of the resampled image
            crs: Coordinate reference system
            
        Returns:
            List of all polygon objects from all tiles
        """
        height, width = rgb_image.shape[1:]
        
        # Get sliding window positions
        windows = self._get_sliding_windows(height, width)
        
        overlap_msg = f"overlapping ({self.overlap_ratio*100:.0f}% overlap)" if self.overlap_ratio > 0 else "non-overlapping"
        debug_msg = f"🔮 Running tile-by-tile vector inference on {len(windows)} {overlap_msg} tiles..."
        if self.debug_tiles:
            debug_msg += f" (Debug mode: saving individual tiles to {self.debug_dir})"
        print(debug_msg)
        
        all_polygons = []
        
        for i, (y, x, y_end, x_end) in enumerate(windows):
            if i % 50 == 0 or self.debug_tiles:
                status = "(vectorizing)" if not self.debug_tiles else "(debug+vectorizing)"
                print(f"   Processing tile {i+1}/{len(windows)} [{y}:{y_end}, {x}:{x_end}] {status}")
            
            # Extract patches on-demand
            window_h = y_end - y
            window_w = x_end - x
            
            # Extract RGB and DEM patches
            rgb_patch = rgb_image[:, y:y_end, x:x_end]
            dem_patch = dem_image[:, y:y_end, x:x_end]
            
            # Pad if needed to reach minimum window size
            pad_h = max(0, self.window_size[0] - window_h)
            pad_w = max(0, self.window_size[1] - window_w)
            
            if pad_h > 0 or pad_w > 0:
                # Pad with reflection
                rgb_patch = np.pad(rgb_patch, 
                                 ((0, 0), (0, pad_h), (0, pad_w)), 
                                 mode='reflect')
                dem_patch = np.pad(dem_patch, 
                                 ((0, 0), (0, pad_h), (0, pad_w)), 
                                 mode='reflect')
            
            # Run inference on patch
            prediction = self._predict_patch(rgb_patch, dem_patch)
            
            # Remove padding from prediction if added
            if pad_h > 0 or pad_w > 0:
                prediction = prediction[:window_h, :window_w]
            
            # Create tile-specific geospatial transform
            tile_transform = rasterio.Affine(
                base_transform.a,  # pixel width
                base_transform.b,  # row rotation (typically 0)
                base_transform.c + x * base_transform.a,  # x offset
                base_transform.d,  # column rotation (typically 0)
                base_transform.e,  # pixel height (typically negative)
                base_transform.f + y * base_transform.e   # y offset
            )
            
            # Create vector boundaries for this tile
            tile_polygons = self._create_tile_vectors(prediction, tile_transform, crs)
            all_polygons.extend(tile_polygons)
            
            # Save debug information for this tile
            self._save_debug_tile(i, rgb_patch, dem_patch, prediction, y, x, y_end, x_end)
        
        if self.debug_tiles:
            print(f"🐛 Debug: Saved {len(windows)} tile outputs to {self.debug_dir}")
        
        print(f"🏗️  Collected {len(all_polygons)} building polygons from {len(windows)} tiles")
        return all_polygons
    
    def predict(self, rgb_path: str, dem_path: str, output_path: str) -> str:
        """Run vector-based inference pipeline with resampling to training resolution.
        
        Args:
            rgb_path: Path to RGB/ortho image
            dem_path: Path to DEM image
            output_path: Path to save vector buildings (GeoJSON/Shapefile)
            
        Returns:
            Path to the created vector file
        """
        start_time = datetime.now()
        print("="*80)
        print("🏗️  VECTOR-BASED BUILDING DETECTION")
        print("="*80)
        print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load images
        rgb_image, rgb_meta = self._load_image(rgb_path)
        dem_image, dem_meta = self._load_image(dem_path)
        
        print("📏 Original resolutions:")
        rgb_resolution = abs(rgb_meta['transform'][0])
        dem_resolution = abs(dem_meta['transform'][0])
        
        # Check if data is in geographic coordinates
        rgb_is_geo = rgb_meta['crs'].is_geographic if hasattr(rgb_meta['crs'], 'is_geographic') else 'EPSG:4326' in str(rgb_meta['crs'])
        dem_is_geo = dem_meta['crs'].is_geographic if hasattr(dem_meta['crs'], 'is_geographic') else 'EPSG:4326' in str(dem_meta['crs'])
        
        if rgb_is_geo:
            print(f"   RGB: {rgb_resolution:.8f} degrees ({rgb_meta['crs']})")
        else:
            print(f"   RGB: {rgb_resolution:.3f}m per pixel ({rgb_meta['crs']})")
            
        if dem_is_geo:
            print(f"   DEM: {dem_resolution:.8f} degrees ({dem_meta['crs']})")
        else:
            print(f"   DEM: {dem_resolution:.3f}m per pixel ({dem_meta['crs']})")
        
        # Determine target resolution
        if self.auto_resolution and not rgb_is_geo and not dem_is_geo:
            # Analyze resolution strategy
            print("\n🧠 Analyzing optimal resolution strategy...")
            resolution_analysis = self._analyze_resolution_strategy(rgb_resolution, dem_resolution)
            
            print("📊 Resolution Analysis:")
            print(f"   Training resolution: {resolution_analysis['training_resolution']:.3f}m")
            print(f"   Recommended strategy: {resolution_analysis['strategy']}")
            print(f"   Target resolution: {resolution_analysis['recommended_resolution']:.3f}m")
            
            if resolution_analysis['warnings']:
                print("⚠️  Warnings:")
                for warning in resolution_analysis['warnings']:
                    print(f"   • {warning}")
            
            # Use the recommended resolution
            target_resolution = resolution_analysis['recommended_resolution']
        else:
            # Use specified training resolution
            target_resolution = self.training_resolution
            if rgb_is_geo or dem_is_geo:
                print(f"📍 Geographic coordinates detected - using training resolution: {target_resolution:.3f}m")
            else:
                print(f"🎯 Using specified training resolution: {target_resolution:.3f}m")
        
        # Resample both ortho and DSM to target resolution
        print(f"\n🔄 Resampling both ortho and DSM to target resolution ({target_resolution:.3f}m)...")
        
        rgb_resampled, rgb_resampled_meta = self._resample_to_resolution(
            rgb_image, rgb_meta, target_resolution
        )
        dem_resampled, dem_resampled_meta = self._resample_to_resolution(
            dem_image, dem_meta, target_resolution  
        )
        
        # Ensure both are aligned to exactly the same grid
        print("🔧 Ensuring perfect grid alignment...")
        dem_resampled = self._resample_to_match(
            dem_resampled, dem_resampled_meta,
            rgb_resampled.shape[1:], rgb_resampled_meta
        )
        dem_resampled_meta = rgb_resampled_meta.copy()
        
        actual_resolution = abs(rgb_resampled_meta['transform'][0])
        print(f"✅ Both images aligned to {actual_resolution:.3f}m: {rgb_resampled.shape[1:]} pixels")
        
        # Handle RGB channels (ensure 3 channels)
        if rgb_resampled.shape[0] == 4:  # RGBA to RGB
            rgb_resampled = rgb_resampled[:3]
            print("   Converted RGBA to RGB")
        elif rgb_resampled.shape[0] == 1:  # Grayscale to RGB
            rgb_resampled = np.repeat(rgb_resampled, 3, axis=0)
            print("   Converted grayscale to RGB")
        elif rgb_resampled.shape[0] > 4:  # Multispectral - take first 3 bands
            rgb_resampled = rgb_resampled[:3]
            print(f"   Using first 3 bands from {rgb_resampled.shape[0]} bands")
        
        # Handle DEM (ensure single channel)
        if dem_resampled.shape[0] > 1:
            dem_resampled = dem_resampled[:1]  # Take first band
            print("   Using first band from DEM")
        
        # Normalize images and get nodata masks
        print("🔧 Detecting and handling nodata pixels...")
        rgb_resampled, rgb_nodata_mask = self._normalize_image(rgb_resampled, is_rgb=True, nodata_value=0)
        dem_resampled, dem_nodata_mask = self._normalize_image(dem_resampled, is_rgb=False, nodata_value=-9999)
        
        # Combine nodata masks
        combined_nodata_mask = rgb_nodata_mask | dem_nodata_mask
        nodata_pixel_count = np.sum(combined_nodata_mask)
        
        if nodata_pixel_count > 0:
            nodata_percentage = (nodata_pixel_count / combined_nodata_mask.size) * 100
            print(f"   Found {nodata_pixel_count:,} nodata pixels ({nodata_percentage:.2f}%)")
        
        # Setup debug directory if debug mode is enabled
        if self.debug_tiles:
            self._setup_debug_directory(output_path)
        
        # Run tile-by-tile vector inference
        print("🔮 Running tile-by-tile vector inference...")
        all_polygons = self._tile_by_tile_vector_inference(
            rgb_resampled, dem_resampled, 
            rgb_resampled_meta['transform'], 
            str(rgb_resampled_meta['crs'])
        )
        
        # Merge polygons from all tiles
        print("🔗 Merging tile polygons into final building boundaries...")
        buildings_gdf = self._merge_tile_vectors(all_polygons, str(rgb_resampled_meta['crs']))
        
        # Save vector results
        vector_output = output_path if output_path.endswith(('.geojson', '.shp')) else output_path.replace('.tif', '_buildings.geojson')
        
        print(f"💾 Saving vector buildings to: {vector_output}")
        os.makedirs(os.path.dirname(vector_output), exist_ok=True)
        
        if vector_output.endswith('.shp'):
            buildings_gdf.to_file(vector_output, driver='ESRI Shapefile')
        else:
            buildings_gdf.to_file(vector_output, driver='GeoJSON')
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Statistics
        total_buildings = len(buildings_gdf)
        total_building_area = buildings_gdf['area_m2'].sum() if total_buildings > 0 else 0
        avg_building_area = buildings_gdf['area_m2'].mean() if total_buildings > 0 else 0
        
        print("\n" + "="*80)
        print("🎉 HIGH CONFIDENCE VECTOR INFERENCE COMPLETED!")
        print("="*80)
        print(f"Processing time: {str(duration).split('.')[0]}")
        print(f"Training resolution: {actual_resolution:.3f}m per pixel")
        print(f"Total buildings detected: {total_buildings:,}")
        if total_buildings > 0:
            print(f"Total building area: {total_building_area:,.1f} m²")
            print(f"Average building area: {avg_building_area:.1f} m²")
            print(f"Largest building: {buildings_gdf['area_m2'].max():.1f} m²")
            print(f"Smallest building: {buildings_gdf['area_m2'].min():.1f} m²")
        print(f"Vector file saved: {vector_output}")
        print("="*80)
        
        return vector_output


def main():
    """Main vector-based inference function."""
    parser = argparse.ArgumentParser(description='Vector-based building detection inference')
    parser.add_argument('--rgb', default='./data/Shahada/ortho.tif',
                      help='Path to RGB/ortho image')
    parser.add_argument('--dem', default='./data/Shahada/dsm.tif',
                      help='Path to DEM image')
    parser.add_argument('--model', default='./weights/UNetformer_run20250831_120522_epoch14_87.1615.pth',
                      help='Path to model weights')
    parser.add_argument('--output', default='./results/Shahada_buildings.geojson',
                      help='Output path for vector buildings (GeoJSON/Shapefile)')
    parser.add_argument('--training-resolution', type=float, default=0.25,
                      help='Training resolution in meters (both ortho and DSM resampled to this) - MEGA model trained at 0.25m')
    parser.add_argument('--auto-resolution', action='store_true', 
                      help='Automatically choose optimal resolution based on input data')
    parser.add_argument('--threshold', type=float, default=0.7,
                      help='Probability threshold for building detection (default: 0.7 for high confidence)')
    parser.add_argument('--min-area', type=float, default=25.0,
                      help='Minimum building area in square meters')
    parser.add_argument('--overlap', type=float, default=0.25,
                      help='Tile overlap ratio (0.0=no overlap, 0.25=25%% overlap, 0.5=50%% overlap)')
    parser.add_argument('--debug-tiles', action='store_true',
                      help='Save individual tile outputs for debugging')
    parser.add_argument('--device', default='cuda',
                      help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.rgb):
        print(f"❌ RGB image not found: {args.rgb}")
        return
    
    if not os.path.exists(args.dem):
        print(f"❌ DEM image not found: {args.dem}")
        return
    
    if not os.path.exists(args.model):
        print(f"❌ Model weights not found: {args.model}")
        return
    
    # Run vector-based inference
    try:
        inference = BuildingInference(
            model_path=args.model, 
            device=args.device, 
            training_resolution=args.training_resolution,
            probability_threshold=args.threshold,
            min_building_area=args.min_area,
            overlap_ratio=args.overlap,
            debug_tiles=args.debug_tiles,
            auto_resolution=args.auto_resolution
        )
        
        vector_file = inference.predict(args.rgb, args.dem, args.output)
        print(f"\n✅ Vector buildings saved to: {vector_file}")
        
    except Exception as e:
        print(f"\n❌ Vector inference failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
