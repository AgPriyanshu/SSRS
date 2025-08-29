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
from rasterio.windows import Window
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from pathlib import Path
import argparse
from datetime import datetime
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append('.')

from unetformer_mmsam import UNetFormer
from model_wrapper import ModelWrapper
from config import config


class BuildingInference:
    """Building detection inference pipeline using trained UNetFormer."""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        """Initialize inference pipeline.
        
        Args:
            model_path: Path to trained model weights
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = None
        self.window_size = (256, 256)  # Training window size
        self.overlap = 32  # Reduced overlap for efficiency
        
        print(f"🖥️  Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        
        self._load_model()
    
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
    
    def _normalize_image(self, image: np.ndarray, is_rgb: bool = True) -> np.ndarray:
        """Normalize image for model input.
        
        Args:
            image: Input image array
            is_rgb: Whether this is RGB image (True) or DEM (False)
            
        Returns:
            Normalized image array
        """
        if is_rgb:
            # RGB normalization - scale to 0-1
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            elif image.dtype == np.uint16:
                image = image.astype(np.float32) / 65535.0
            else:
                # Assume already float, normalize by percentiles
                image = image.astype(np.float32)
                p2, p98 = np.percentile(image, (2, 98))
                image = np.clip((image - p2) / (p98 - p2), 0, 1)
        else:
            # DEM normalization - standardize
            image = image.astype(np.float32)
            mean = np.mean(image)
            std = np.std(image)
            if std > 0:
                image = (image - mean) / std
            else:
                image = image - mean
                
        return image
    
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
            rgb_patch: RGB patch (C, H, W)
            dem_patch: DEM patch (1, H, W)
            
        Returns:
            Prediction array (H, W)
        """
        with torch.no_grad():
            # Convert to tensors and add batch dimension
            rgb_tensor = torch.from_numpy(rgb_patch).unsqueeze(0).to(self.device)  # (1, C, H, W)
            dem_tensor = torch.from_numpy(dem_patch[0]).unsqueeze(0).to(self.device)  # (1, H, W)
            
            # Run inference
            output = self.model(rgb_tensor, dem_tensor)  # (1, num_classes, H, W)
            
            # Get probability for building class (class 1)
            prob = torch.softmax(output, dim=1)  # (1, num_classes, H, W)
            building_prob = prob[0, 1].cpu().numpy()  # (H, W)
            
        return building_prob
    
    def _sliding_window_inference(self, rgb_image: np.ndarray, dem_image: np.ndarray) -> np.ndarray:
        """Run inference using sliding window approach without pre-generating tiles.
        
        Args:
            rgb_image: RGB image array (C, H, W)
            dem_image: DEM image array (1, H, W)
            
        Returns:
            Full prediction array (H, W)
        """
        height, width = rgb_image.shape[1:]
        prediction = np.zeros((height, width), dtype=np.float32)
        weight_map = np.zeros((height, width), dtype=np.float32)
        
        # Get sliding window positions
        windows = self._get_sliding_windows(height, width)
        
        print(f"🔮 Running sliding window inference on {len(windows)} windows...")
        
        for i, (y, x, y_end, x_end) in enumerate(windows):
            if i % 10 == 0:
                print(f"   Processing window {i+1}/{len(windows)} [{y}:{y_end}, {x}:{x_end}]")
            
            # Extract patches on-demand
            window_h = y_end - y
            window_w = x_end - x
            
            # Extract RGB patch
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
            pred = self._predict_patch(rgb_patch, dem_patch)
            
            # Remove padding from prediction if added
            if pad_h > 0 or pad_w > 0:
                pred = pred[:window_h, :window_w]
            
            # Create weight mask (higher weight in center, lower at edges)
            weight = np.ones((window_h, window_w), dtype=np.float32)
            if self.overlap > 0:
                fade = min(self.overlap // 2, 16)
                # Apply fade at edges
                for f in range(fade):
                    if f < window_h and f < window_w:
                        weight[f, :] *= (f + 1) / fade
                        weight[-(f+1), :] *= (f + 1) / fade
                        weight[:, f] *= (f + 1) / fade
                        weight[:, -(f+1)] *= (f + 1) / fade
            
            # Accumulate prediction with weights
            prediction[y:y_end, x:x_end] += pred * weight
            weight_map[y:y_end, x:x_end] += weight
        
        # Normalize by weight map
        weight_map[weight_map == 0] = 1
        prediction = prediction / weight_map
        
        return prediction
    
    def predict(self, rgb_path: str, dem_path: str, output_path: str, 
                threshold: float = 0.5) -> None:
        """Run full inference pipeline.
        
        Args:
            rgb_path: Path to RGB image
            dem_path: Path to DEM image
            output_path: Path to save binary mask
            threshold: Threshold for binary classification
        """
        start_time = datetime.now()
        print("="*80)
        print("🏢 BUILDING DETECTION INFERENCE")
        print("="*80)
        print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load images
        rgb_image, rgb_meta = self._load_image(rgb_path)
        dem_image, dem_meta = self._load_image(dem_path)
        
        # Handle different resolutions by resampling DEM to match RGB
        if rgb_image.shape[1:] != dem_image.shape[1:]:
            print(f"⚠️  Resolution mismatch: RGB {rgb_image.shape[1:]} vs DEM {dem_image.shape[1:]}")
            print("🔧 Resampling DEM to match RGB resolution...")
            
            # Resample DEM to match RGB resolution
            dem_image = self._resample_to_match(
                dem_image, dem_meta, 
                rgb_image.shape[1:], rgb_meta
            )
            print(f"✅ DEM resampled to: {dem_image.shape[1:]}")
        
        # Handle RGB channels (ensure 3 channels)
        if rgb_image.shape[0] == 4:  # RGBA to RGB
            rgb_image = rgb_image[:3]
            print("   Converted RGBA to RGB")
        elif rgb_image.shape[0] == 1:  # Grayscale to RGB
            rgb_image = np.repeat(rgb_image, 3, axis=0)
            print("   Converted grayscale to RGB")
        elif rgb_image.shape[0] > 4:  # Multispectral - take first 3 bands
            rgb_image = rgb_image[:3]
            print(f"   Using first 3 bands from {rgb_image.shape[0]} bands")
        
        # Handle DEM (ensure single channel)
        if dem_image.shape[0] > 1:
            dem_image = dem_image[:1]  # Take first band
            print(f"   Using first band from DEM")
        
        print(f"Processing image: {rgb_image.shape[1:]} pixels")
        
        # Normalize images
        rgb_image = self._normalize_image(rgb_image, is_rgb=True)
        dem_image = self._normalize_image(dem_image, is_rgb=False)
        
        # Run sliding window inference (no tile pre-generation)
        print("🔧 Running sliding window inference...")
        full_prediction = self._sliding_window_inference(rgb_image, dem_image)
        
        # Create binary mask
        binary_mask = (full_prediction >= threshold).astype(np.uint8)
        
        # Save results
        print(f"💾 Saving results to: {output_path}")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save binary mask
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=binary_mask.shape[0],
            width=binary_mask.shape[1],
            count=1,
            dtype=binary_mask.dtype,
            crs=rgb_meta['crs'],
            transform=rgb_meta['transform'],
            compress='lzw'
        ) as dst:
            dst.write(binary_mask, 1)
        
        # Save probability map
        prob_output = output_path.replace('.tif', '_probability.tif')
        with rasterio.open(
            prob_output,
            'w',
            driver='GTiff',
            height=full_prediction.shape[0],
            width=full_prediction.shape[1],
            count=1,
            dtype=np.float32,
            crs=rgb_meta['crs'],
            transform=rgb_meta['transform'],
            compress='lzw'
        ) as dst:
            dst.write(full_prediction, 1)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Statistics
        total_pixels = binary_mask.size
        building_pixels = np.sum(binary_mask)
        building_percentage = (building_pixels / total_pixels) * 100
        
        print("\n" + "="*80)
        print("🎉 INFERENCE COMPLETED!")
        print("="*80)
        print(f"Processing time: {str(duration).split('.')[0]}")
        print(f"Image size: {binary_mask.shape[1]} x {binary_mask.shape[0]} pixels")
        print(f"Building pixels: {building_pixels:,} ({building_percentage:.2f}%)")
        print(f"Binary mask saved: {output_path}")
        print(f"Probability map saved: {prob_output}")
        print("="*80)


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Building detection inference')
    parser.add_argument('--rgb', default='./data/Shahada/ortho.tif',
                      help='Path to RGB image')
    parser.add_argument('--dem', default='./data/Shahada/dsm.tif',
                      help='Path to DEM image')
    parser.add_argument('--model', default='./weights/UNetformer_epoch8_80.6478',
                      help='Path to model weights')
    parser.add_argument('--output', default='./results/Shahada_building_mask.tif',
                      help='Output path for binary mask')
    parser.add_argument('--threshold', type=float, default=0.7,
                      help='Threshold for binary classification')
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
    
    # Run inference
    try:
        inference = BuildingInference(args.model, args.device)
        inference.predict(args.rgb, args.dem, args.output, args.threshold)
        
    except Exception as e:
        print(f"\n❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
