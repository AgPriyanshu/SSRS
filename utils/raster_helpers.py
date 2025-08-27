"""
Raster processing utilities for SSRS semantic segmentation training.

This module provides functions to tile large orthophotos, DSMs, and masks
into training patches suitable for the UNetFormer model.
"""

from typing import Tuple, List, Dict
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.transform import from_bounds
import os
from pathlib import Path
import json
from PIL import Image
from shapely.geometry import box

from constants import WINDOW_SIZE, DATASET_DIR


def analyze_raster_resolutions(
    raster_paths: List[str],
    labels: List[str] = None
) -> Dict:
    """
    Analyze the resolutions of multiple raster files.
    
    Args:
        raster_paths: List of paths to raster files
        labels: Optional labels for the rasters (e.g., ['ortho', 'dsm', 'mask'])
        
    Returns:
        Dictionary with resolution information for each raster
    """
    if labels is None:
        labels = [f"raster_{i}" for i in range(len(raster_paths))]
    
    resolution_info = {}
    
    for path, label in zip(raster_paths, labels):
        with rasterio.open(path) as src:
            # Get pixel size in x and y directions
            pixel_size_x = abs(src.transform[0])
            pixel_size_y = abs(src.transform[4])
            
            resolution_info[label] = {
                'path': path,
                'pixel_size_x': pixel_size_x,
                'pixel_size_y': pixel_size_y,
                'width': src.width,
                'height': src.height,
                'bounds': src.bounds,
                'crs': src.crs,
                'transform': src.transform
            }
    
    # Find finest and coarsest resolutions
    pixel_sizes = [(info['pixel_size_x'] + info['pixel_size_y']) / 2 
                   for info in resolution_info.values()]
    
    finest_idx = pixel_sizes.index(min(pixel_sizes))
    coarsest_idx = pixel_sizes.index(max(pixel_sizes))
    
    resolution_info['summary'] = {
        'finest_resolution': {
            'label': labels[finest_idx],
            'pixel_size': pixel_sizes[finest_idx]
        },
        'coarsest_resolution': {
            'label': labels[coarsest_idx], 
            'pixel_size': pixel_sizes[coarsest_idx]
        },
        'resolutions_match': len(set(pixel_sizes)) == 1
    }
    
    return resolution_info


def resample_raster_to_target(
    input_path: str,
    output_path: str,
    target_transform: rasterio.Affine,
    target_width: int,
    target_height: int,
    target_crs: rasterio.crs.CRS,
    resampling_method: Resampling = Resampling.bilinear
) -> None:
    """
    Resample a raster to match target spatial properties.
    
    Args:
        input_path: Path to input raster
        output_path: Path for output resampled raster
        target_transform: Target affine transform
        target_width: Target width in pixels
        target_height: Target height in pixels
        target_crs: Target coordinate reference system
        resampling_method: Resampling algorithm to use
    """
    print(f"Resampling {os.path.basename(input_path)} to target resolution...")
    
    with rasterio.open(input_path) as src:
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Set up the output raster
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs,
            'transform': target_transform,
            'width': target_width,
            'height': target_height,
            'compress': 'lzw',
            'tiled': True,
            'blockxsize': 512,
            'blockysize': 512
        })
        
        with rasterio.open(output_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    resampling=resampling_method
                )
    
    print(f"  Saved resampled raster to: {output_path}")


def align_rasters_to_common_resolution(
    ortho_path: str,
    dsm_path: str,
    mask_path: str,
    output_dir: str,
    target_resolution: str = 'coarsest',
    custom_pixel_size: float = None
) -> Tuple[str, str, str]:
    """
    Resample all rasters to a common resolution.
    
    Args:
        ortho_path: Path to orthophoto
        dsm_path: Path to DSM
        mask_path: Path to mask
        output_dir: Directory to save resampled rasters
        target_resolution: 'coarsest', 'finest', or 'custom' (defaults to coarsest)
        custom_pixel_size: Custom pixel size if target_resolution='custom'
        
    Returns:
        Tuple of paths to resampled (ortho, dsm, mask) files
    """
    print("Analyzing raster resolutions...")
    
    # Analyze current resolutions
    resolution_info = analyze_raster_resolutions(
        [ortho_path, dsm_path, mask_path],
        ['ortho', 'dsm', 'mask']
    )
    
    print("Current resolutions:")
    for label in ['ortho', 'dsm', 'mask']:
        info = resolution_info[label]
        avg_pixel_size = (info['pixel_size_x'] + info['pixel_size_y']) / 2
        print(f"  {label}: {avg_pixel_size:.3f} units/pixel ({info['width']}x{info['height']})")
    
    # Check if resolutions already match
    if resolution_info['summary']['resolutions_match']:
        print("✅ All rasters already have the same resolution!")
        return ortho_path, dsm_path, mask_path
    
    print(f"\\nResolutions don't match. Resampling to {target_resolution} resolution...")
    
    # Determine target resolution
    if target_resolution == 'finest':
        target_label = resolution_info['summary']['finest_resolution']['label']
        target_pixel_size = resolution_info['summary']['finest_resolution']['pixel_size']
        print(f"Target: {target_label} resolution ({target_pixel_size:.3f} units/pixel)")
    elif target_resolution == 'coarsest':
        target_label = resolution_info['summary']['coarsest_resolution']['label'] 
        target_pixel_size = resolution_info['summary']['coarsest_resolution']['pixel_size']
        print(f"Target: {target_label} resolution ({target_pixel_size:.3f} units/pixel)")
    elif target_resolution == 'custom':
        if custom_pixel_size is None:
            raise ValueError("custom_pixel_size must be provided when target_resolution='custom'")
        target_pixel_size = custom_pixel_size
        target_label = 'custom'
        print(f"Target: Custom resolution ({target_pixel_size:.3f} units/pixel)")
    else:
        raise ValueError("target_resolution must be 'finest', 'coarsest', or 'custom'")
    
    # Use the reference raster to define target spatial properties
    if target_resolution == 'custom':
        # Use the first raster as reference for extent but with custom resolution
        ref_info = resolution_info['ortho']
    else:
        ref_info = resolution_info[target_label]
    
    # Calculate target transform and dimensions
    bounds = ref_info['bounds']
    
    if target_resolution == 'custom':
        # Calculate new dimensions based on custom pixel size
        width = int((bounds.right - bounds.left) / custom_pixel_size)
        height = int((bounds.top - bounds.bottom) / custom_pixel_size)
        target_transform = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, width, height)
    else:
        # Use the target raster's properties
        target_transform = ref_info['transform']
        width = ref_info['width']
        height = ref_info['height']
    
    target_crs = ref_info['crs']
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output paths
    ortho_resampled = output_dir / f"ortho_resampled_{target_resolution}.tif"
    dsm_resampled = output_dir / f"dsm_resampled_{target_resolution}.tif"  
    mask_resampled = output_dir / f"mask_resampled_{target_resolution}.tif"
    
    # Resample each raster
    print("\\nResampling rasters:")
    
    # Ortho (bilinear interpolation for continuous data)
    resample_raster_to_target(
        ortho_path, str(ortho_resampled), target_transform,
        width, height, target_crs, Resampling.bilinear
    )
    
    # DSM (bilinear interpolation for continuous elevation data)
    resample_raster_to_target(
        dsm_path, str(dsm_resampled), target_transform,
        width, height, target_crs, Resampling.bilinear
    )
    
    # Mask (nearest neighbor for categorical data)
    resample_raster_to_target(
        mask_path, str(mask_resampled), target_transform,
        width, height, target_crs, Resampling.nearest
    )
    
    print("\\n✅ All rasters resampled to common resolution!")
    print(f"Resampled files saved in: {output_dir}")
    
    return str(ortho_resampled), str(dsm_resampled), str(mask_resampled)


def tile_large_raster(
    input_raster_path: str,
    output_dir: str,
    tile_size: Tuple[int, int] = WINDOW_SIZE,
    overlap: int = 0,
    prefix: str = "tile",
    save_format: str = "tif",
    min_valid_pixels: float = 0.1
) -> List[Dict]:
    """
    Tile a large raster into smaller patches for training.
    
    Args:
        input_raster_path: Path to the input raster file
        output_dir: Directory to save the tiles
        tile_size: Size of each tile (height, width)
        overlap: Overlap between tiles in pixels
        prefix: Prefix for output tile names
        save_format: Output format ('tif', 'png', 'jpg')
        min_valid_pixels: Minimum fraction of valid (non-nodata) pixels required
        
    Returns:
        List of dictionaries containing tile information
    """
    os.makedirs(output_dir, exist_ok=True)
    tile_info = []
    
    with rasterio.open(input_raster_path) as src:
        height, width = src.height, src.width
        tile_height, tile_width = tile_size
        step_y = tile_height - overlap
        step_x = tile_width - overlap
        
        print(f"Tiling {input_raster_path}")
        print(f"  Source size: {width}x{height}")
        print(f"  Tile size: {tile_width}x{tile_height}")
        print(f"  Overlap: {overlap} pixels")
        print(f"  Step size: {step_x}x{step_y}")
        
        tile_count = 0
        
        for row in range(0, height - tile_height + 1, step_y):
            for col in range(0, width - tile_width + 1, step_x):
                # Define the window
                window = Window(col, row, tile_width, tile_height)
                
                # Read the tile data
                tile_data = src.read(window=window)
                
                # Check for valid data (skip tiles with mostly nodata)
                if src.nodata is not None:
                    valid_pixels = np.sum(tile_data != src.nodata) / tile_data.size
                    if valid_pixels < min_valid_pixels:
                        continue
                
                # Generate tile filename
                tile_id = f"{prefix}_{tile_count:06d}"
                if save_format.lower() == 'tif':
                    tile_filename = f"{tile_id}.tif"
                    tile_path = os.path.join(output_dir, tile_filename)
                    
                    # Save as GeoTIFF with proper georeferencing
                    tile_transform = rasterio.windows.transform(window, src.transform)
                    
                    with rasterio.open(
                        tile_path, 'w',
                        driver='GTiff',
                        height=tile_height,
                        width=tile_width,
                        count=src.count,
                        dtype=src.dtypes[0],
                        crs=src.crs,
                        transform=tile_transform,
                        compress='lzw'
                    ) as dst:
                        dst.write(tile_data)
                        
                else:
                    # Save as image format (PNG/JPG)
                    tile_filename = f"{tile_id}.{save_format.lower()}"
                    tile_path = os.path.join(output_dir, tile_filename)
                    
                    if tile_data.ndim == 3 and tile_data.shape[0] > 1:
                        # Multi-band image (RGB/RGBA)
                        tile_image = np.transpose(tile_data, (1, 2, 0))
                    else:
                        # Single band
                        tile_image = tile_data[0] if tile_data.ndim == 3 else tile_data
                    
                    # Normalize if needed
                    if tile_image.dtype != np.uint8:
                        if tile_image.max() <= 1.0:
                            tile_image = (tile_image * 255).astype(np.uint8)
                        else:
                            tile_image = np.clip(tile_image, 0, 255).astype(np.uint8)
                    
                    # Handle RGBA to RGB conversion for JPEG (only needed for JPEG format)
                    if save_format.lower() in ['jpg', 'jpeg'] and tile_image.ndim == 3 and tile_image.shape[2] == 4:
                        # Convert RGBA to RGB by removing alpha channel
                        tile_image = tile_image[:, :, :3]
                        print(f"  Converting RGBA to RGB for JPEG: {tile_filename}")
                    
                    # Create PIL Image and save
                    pil_image = Image.fromarray(tile_image)
                    pil_image.save(tile_path)
                
                # Store tile information
                tile_info.append({
                    'tile_id': tile_id,
                    'filename': tile_filename,
                    'path': tile_path,
                    'window': {
                        'col': col,
                        'row': row,
                        'width': tile_width,
                        'height': tile_height
                    },
                    'bounds': rasterio.windows.bounds(window, src.transform),
                    'crs': str(src.crs) if src.crs else None
                })
                
                tile_count += 1
        
        print(f"  Created {tile_count} tiles")
        
    return tile_info


def prepare_training_dataset(
    ortho_path: str,
    dsm_path: str,
    mask_path: str,
    output_root: str,
    dataset_name: str,
    tile_size: Tuple[int, int] = WINDOW_SIZE,
    overlap: int = 32,
    train_ratio: float = 0.8,
    min_valid_pixels: float = 0.1,
    target_resolution: str = 'coarsest',
    custom_pixel_size: float = None
) -> Dict:
    """
    Prepare a complete training dataset from large orthophoto, DSM, and mask.
    
    Args:
        ortho_path: Path to the orthophoto
        dsm_path: Path to the DSM
        mask_path: Path to the mask
        output_root: Root directory for the prepared dataset
        dataset_name: Name of the dataset
        tile_size: Size of tiles
        overlap: Overlap between tiles
        train_ratio: Ratio of tiles to use for training (rest for testing)
        min_valid_pixels: Minimum valid pixel ratio per tile
        target_resolution: 'coarsest', 'finest', or 'custom' for resolution alignment (defaults to coarsest)
        custom_pixel_size: Custom pixel size if target_resolution='custom'
        
    Returns:
        Dictionary with dataset information
    """
    # Create output directory structure
    dataset_dir = Path(output_root) / dataset_name
    images_dir = dataset_dir / "images"
    dsm_dir = dataset_dir / "dsm"
    labels_dir = dataset_dir / "labels"
    resampled_dir = dataset_dir / "resampled"
    
    for dir_path in [images_dir, dsm_dir, labels_dir, resampled_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Preparing dataset: {dataset_name}")
    print(f"Output directory: {dataset_dir}")
    
    # Step 0: Align rasters to common resolution
    print("\\n" + "="*50)
    print("STEP 0: RESOLUTION ALIGNMENT")
    print("="*50)
    
    ortho_aligned, dsm_aligned, mask_aligned = align_rasters_to_common_resolution(
        ortho_path, dsm_path, mask_path,
        str(resampled_dir), target_resolution, custom_pixel_size
    )
    
    # Use aligned rasters for tiling
    ortho_path = ortho_aligned
    dsm_path = dsm_aligned  
    mask_path = mask_aligned
    
    # Tile the orthophoto
    print("\\nTiling orthophoto...")
    ortho_tiles = tile_large_raster(
        ortho_path, str(images_dir), tile_size, overlap,
        prefix="img", save_format="png", min_valid_pixels=min_valid_pixels
    )
    
    # Tile the DSM
    print("\\nTiling DSM...")
    dsm_tiles = tile_large_raster(
        dsm_path, str(dsm_dir), tile_size, overlap,
        prefix="dsm", save_format="tif", min_valid_pixels=min_valid_pixels
    )
    
    # Tile the mask
    print("\\nTiling mask...")
    mask_tiles = tile_large_raster(
        mask_path, str(labels_dir), tile_size, overlap,
        prefix="label", save_format="png", min_valid_pixels=min_valid_pixels
    )
    
    # Ensure all have the same number of tiles
    min_tiles = min(len(ortho_tiles), len(dsm_tiles), len(mask_tiles))
    ortho_tiles = ortho_tiles[:min_tiles]
    dsm_tiles = dsm_tiles[:min_tiles]
    mask_tiles = mask_tiles[:min_tiles]
    
    print(f"\\nTotal valid tiles: {min_tiles}")
    
    # Split into train/test
    train_count = int(min_tiles * train_ratio)
    train_indices = list(range(train_count))
    test_indices = list(range(train_count, min_tiles))
    
    # Generate train and test IDs
    train_ids = [f"img_{i:06d}" for i in train_indices]
    test_ids = [f"img_{i:06d}" for i in test_indices]
    
    print(f"Training tiles: {len(train_ids)}")
    print(f"Testing tiles: {len(test_ids)}")
    
    # Create dataset configuration
    dataset_info = {
        'name': dataset_name,
        'dataset_dir': str(dataset_dir),
        'total_tiles': min_tiles,
        'train_ids': train_ids,
        'test_ids': test_ids,
        'tile_size': tile_size,
        'overlap': overlap,
        'data_pattern': "images/img_{}.png",
        'dsm_pattern': "dsm/dsm_{}.tif",
        'label_pattern': "labels/label_{}.png",
        'source_files': {
            'ortho': ortho_path,
            'dsm': dsm_path,
            'mask': mask_path
        }
    }
    
    # Save dataset info
    info_path = dataset_dir / "dataset_info.json"
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"\\nDataset info saved to: {info_path}")
    
    return dataset_info


def create_overlap_tiles(
    input_raster_path: str,
    output_dir: str,
    tile_size: Tuple[int, int] = WINDOW_SIZE,
    overlap_ratio: float = 0.25
) -> List[Dict]:
    """
    Create overlapping tiles to avoid edge artifacts during training.
    
    Args:
        input_raster_path: Path to input raster
        output_dir: Output directory
        tile_size: Size of each tile
        overlap_ratio: Ratio of overlap (0.25 = 25% overlap)
        
    Returns:
        List of tile information dictionaries
    """
    overlap_pixels = int(min(tile_size) * overlap_ratio)
    return tile_large_raster(
        input_raster_path, output_dir, tile_size, 
        overlap=overlap_pixels, min_valid_pixels=0.1
    )


def validate_dataset_spatial_compatibility(
    ortho_path: str, 
    dsm_path: str, 
    mask_path: str
) -> Dict:
    """
    Validate that ortho, DSM, and mask are spatially compatible for processing.
    
    Args:
        ortho_path: Path to orthophoto
        dsm_path: Path to DSM
        mask_path: Path to mask
        
    Returns:
        Dictionary with spatial compatibility information
    """
    compatibility_info = {}
    
    # Open all rasters and gather information
    with rasterio.open(ortho_path) as ortho:
        with rasterio.open(dsm_path) as dsm:
            with rasterio.open(mask_path) as mask:
                
                # Check dimensions
                compatibility_info['dimensions'] = {
                    'ortho': (ortho.width, ortho.height),
                    'dsm': (dsm.width, dsm.height),
                    'mask': (mask.width, mask.height)
                }
                
                # Check CRS
                compatibility_info['crs'] = {
                    'ortho': str(ortho.crs),
                    'dsm': str(dsm.crs),
                    'mask': str(mask.crs)
                }
                
                # Check bounds
                compatibility_info['bounds'] = {
                    'ortho': list(ortho.bounds),
                    'dsm': list(dsm.bounds),
                    'mask': list(mask.bounds)
                }
                
                # Check resolutions
                compatibility_info['resolutions'] = {
                    'ortho': (abs(ortho.transform[0]) + abs(ortho.transform[4])) / 2,
                    'dsm': (abs(dsm.transform[0]) + abs(dsm.transform[4])) / 2,
                    'mask': (abs(mask.transform[0]) + abs(mask.transform[4])) / 2
                }
                
                # Validation checks
                crs_match = ortho.crs == dsm.crs == mask.crs
                
                # Check spatial overlap (all rasters should overlap)
                def bounds_overlap(bounds1, bounds2):
                    return not (bounds1.right <= bounds2.left or 
                              bounds1.left >= bounds2.right or
                              bounds1.top <= bounds2.bottom or 
                              bounds1.bottom >= bounds2.top)
                
                ortho_dsm_overlap = bounds_overlap(ortho.bounds, dsm.bounds)
                ortho_mask_overlap = bounds_overlap(ortho.bounds, mask.bounds)
                dsm_mask_overlap = bounds_overlap(dsm.bounds, mask.bounds)
                spatial_overlap = ortho_dsm_overlap and ortho_mask_overlap and dsm_mask_overlap
                
                # Calculate overlap areas for more detailed analysis
                ortho_box = box(*ortho.bounds)
                dsm_box = box(*dsm.bounds)
                mask_box = box(*mask.bounds)
                
                try:
                    intersection = ortho_box.intersection(dsm_box).intersection(mask_box)
                    overlap_area = intersection.area if intersection.is_valid else 0
                    total_area = ortho_box.union(dsm_box).union(mask_box).area
                    overlap_ratio = overlap_area / total_area if total_area > 0 else 0
                except Exception:
                    # Handle any geometric computation errors
                    overlap_ratio = 0
                
                compatibility_info['validation'] = {
                    'crs_match': crs_match,
                    'spatial_overlap': spatial_overlap,
                    'overlap_ratio': overlap_ratio,
                    'spatially_compatible': crs_match and spatial_overlap and overlap_ratio > 0.1,
                    'resolutions_vary': len(set(compatibility_info['resolutions'].values())) > 1
                }
        
    return compatibility_info


def calculate_dataset_statistics(dataset_dir: str) -> Dict:
    """
    Calculate statistics for the prepared dataset.
    
    Args:
        dataset_dir: Path to the dataset directory
        
    Returns:
        Dictionary with dataset statistics
    """
    dataset_path = Path(dataset_dir)
    
    # Count files in each directory
    images_dir = dataset_path / "images"
    dsm_dir = dataset_path / "dsm" 
    labels_dir = dataset_path / "labels"
    
    stats = {
        'image_count': len(list(images_dir.glob("*.png"))) if images_dir.exists() else 0,
        'dsm_count': len(list(dsm_dir.glob("*.tif"))) if dsm_dir.exists() else 0,
        'label_count': len(list(labels_dir.glob("*.png"))) if labels_dir.exists() else 0
    }
    
    # Calculate total dataset size
    total_size = 0
    for dir_path in [images_dir, dsm_dir, labels_dir]:
        if dir_path.exists():
            for file_path in dir_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
    
    stats['total_size_mb'] = total_size / (1024 * 1024)
    
    return stats


if __name__ == "__main__":
    # Example usage
    print("Example: Preparing training dataset from large rasters")
    
    # Paths to your large files
    ortho_path = DATASET_DIR / "ortho_cog.tif"
    dsm_path = DATASET_DIR / "dsm_cog.tif" 
    mask_path = DATASET_DIR / "mask_cog.tif"
    
    # Check if files exist before processing
    if all(Path(p).exists() for p in [ortho_path, dsm_path, mask_path]):
        # Validate spatial compatibility first
        print("Validating spatial compatibility...")
        compatibility = validate_dataset_spatial_compatibility(
            str(ortho_path), str(dsm_path), str(mask_path)
        )
        print(f"Spatially compatible: {compatibility['validation']['spatially_compatible']}")
        print(f"Different resolutions: {compatibility['validation']['resolutions_vary']}")
        
        if compatibility['validation']['spatially_compatible']:
            # Prepare the training dataset with resolution alignment
            dataset_info = prepare_training_dataset(
                ortho_path=str(ortho_path),
                dsm_path=str(dsm_path),
                mask_path=str(mask_path),
                output_root="./prepared_datasets",
                dataset_name="my_building_dataset",
                tile_size=(256, 256),
                overlap=32,
                train_ratio=0.8,
                target_resolution='coarsest'  # Align to coarsest resolution
            )
            
            # Calculate statistics
            stats = calculate_dataset_statistics(dataset_info['dataset_dir'])
            print("\\nDataset Statistics:")
            print(f"  Images: {stats['image_count']}")
            print(f"  DSMs: {stats['dsm_count']}")
            print(f"  Labels: {stats['label_count']}")
            print(f"  Total size: {stats['total_size_mb']:.2f} MB")
            
        else:
            print("Warning: Input files are not spatially compatible!")
            print("Please ensure ortho, DSM, and mask have:")
            print("- Same coordinate reference system")
            print("- Overlapping geographic areas") 
            print("- Sufficient spatial overlap")
    else:
        print("Example files not found. Update paths in the script.")
