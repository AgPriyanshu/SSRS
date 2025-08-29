#!/usr/bin/env python3
"""
Multi-scale dataset loader for Deep Fusion Module training.

This dataset loader handles tiles generated at multiple overview levels,
enabling the Deep Fusion Module to learn from multi-scale features.
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import rasterio
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
from skimage import transform

from dataset import SemanticSegmentationDataset  # Inherit from existing dataset


class MultiScaleSegmentationDataset(SemanticSegmentationDataset):
    """
    Multi-scale dataset loader that provides tiles from different overview levels.
    
    This dataset loader extends the base SegmentationDataset to handle tiles
    generated at multiple scale levels, perfect for Deep Fusion Module training.
    """
    
    def __init__(
        self,
        tile_ids: List[str],
        multiscale_mapping_file: str,
        window_size: Tuple[int, int] = (256, 256),
        stride_size: int = 64,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        augment: bool = True,
        cache: bool = False,
        debug: bool = False
    ):
        """
        Initialize multi-scale dataset.
        
        Args:
            tile_ids: List of tile IDs with scale prefixes (e.g., "scale1_img_000001")
            multiscale_mapping_file: Path to JSON file mapping tile IDs to datasets
            window_size: Size of input patches
            stride_size: Stride for patch extraction
            mean: Normalization mean for RGB channels
            std: Normalization std for RGB channels
            augment: Whether to apply data augmentation
            cache: Whether to cache loaded tiles in memory
            debug: Enable debug mode
        """
        
        # Don't call parent __init__ yet, we need to process the mapping first
        self.tile_ids = tile_ids
        self.window_size = window_size
        self.stride_size = stride_size
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.augment = augment
        self.cache = cache
        self.debug = debug
        
        # Load multi-scale mapping
        with open(multiscale_mapping_file, 'r') as f:
            self.multiscale_mapping = json.load(f)
        
        # Validate all tile IDs exist in mapping
        missing_ids = [tid for tid in tile_ids if tid not in self.multiscale_mapping]
        if missing_ids:
            raise ValueError(f"Missing tile IDs in mapping: {missing_ids[:5]}...")
        
        # Cache for loaded tiles if caching is enabled
        self._tile_cache = {} if cache else None
        
        # Store scale level distribution for analysis
        self.scale_distribution = self._analyze_scale_distribution()
        
        if debug:
            print(f"MultiScaleDataset initialized:")
            print(f"  Total tiles: {len(tile_ids)}")
            print(f"  Scale distribution: {self.scale_distribution}")
            print(f"  Caching: {cache}")
            print(f"  Augmentation: {augment}")
    
    def _analyze_scale_distribution(self) -> Dict[int, int]:
        """Analyze the distribution of scale levels in the dataset."""
        distribution = {}
        for tile_id in self.tile_ids:
            scale_level = self.multiscale_mapping[tile_id]['scale_level']
            distribution[scale_level] = distribution.get(scale_level, 0) + 1
        return distribution
    
    def get_tile_info(self, tile_id: str) -> Dict:
        """Get detailed information about a specific tile."""
        return self.multiscale_mapping[tile_id]
    
    def _load_tile_data(self, tile_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load RGB, DSM, and label data for a specific tile.
        
        Returns:
            Tuple of (rgb_data, dsm_data, label_data) as numpy arrays
        """
        
        # Check cache first
        if self.cache and tile_id in self._tile_cache:
            return self._tile_cache[tile_id]
        
        # Get tile mapping info
        tile_info = self.multiscale_mapping[tile_id]
        dataset_path = tile_info['dataset_path']
        original_id = tile_info['original_id']
        scale_level = tile_info['scale_level']
        
        # Construct file paths
        dataset_dir = Path(dataset_path)
        rgb_path = dataset_dir / "images" / f"{original_id}.png"
        dsm_path = dataset_dir / "dsm" / f"dsm_{original_id.replace('img_', '')}.tif"
        label_path = dataset_dir / "labels" / f"label_{original_id.replace('img_', '')}.png"
        
        # Load RGB image
        rgb_image = Image.open(rgb_path).convert('RGB')
        rgb_data = np.array(rgb_image).transpose(2, 0, 1)  # HWC -> CHW
        
        # Load DSM
        with rasterio.open(dsm_path) as src:
            dsm_data = src.read(1)  # Read first band
        
        # Load labels
        label_image = Image.open(label_path)
        label_data = np.array(label_image)
        
        # Handle grayscale labels
        if len(label_data.shape) == 3:
            label_data = label_data[:, :, 0]  # Take first channel
        
        # Store in cache if caching is enabled
        if self.cache:
            self._tile_cache[tile_id] = (rgb_data, dsm_data, label_data)
        
        return rgb_data, dsm_data, label_data
    
    def __len__(self) -> int:
        """Return total number of patches across all tiles."""
        # Estimate patches per tile based on window and stride size
        patches_per_tile = ((256 - self.window_size[0]) // self.stride_size + 1) * \
                          ((256 - self.window_size[1]) // self.stride_size + 1)
        return len(self.tile_ids) * patches_per_tile
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample (patch) from the multi-scale dataset.
        
        Returns:
            Dictionary containing:
            - 'image': RGB+DSM tensor [4, H, W]
            - 'label': Label tensor [H, W]  
            - 'scale_level': Scale level of the source tile
            - 'scale_factor': Scale factor relative to base resolution
            - 'tile_id': Source tile ID
        """
        
        # Calculate which tile and which patch within that tile
        patches_per_tile = ((256 - self.window_size[0]) // self.stride_size + 1) * \
                          ((256 - self.window_size[1]) // self.stride_size + 1)
        tile_idx = idx // patches_per_tile
        patch_idx = idx % patches_per_tile
        
        # Get tile ID and load data
        tile_id = self.tile_ids[tile_idx]
        rgb_data, dsm_data, label_data = self._load_tile_data(tile_id)
        
        # Calculate patch coordinates
        patches_per_row = (256 - self.window_size[1]) // self.stride_size + 1
        patch_row = patch_idx // patches_per_row
        patch_col = patch_idx % patches_per_row
        
        start_h = patch_row * self.stride_size
        start_w = patch_col * self.stride_size
        end_h = start_h + self.window_size[0]
        end_w = start_w + self.window_size[1]
        
        # Extract patches
        data_patch = rgb_data[:, start_h:end_h, start_w:end_w]
        dsm_patch = dsm_data[start_h:end_h, start_w:end_w]
        label_patch = label_data[start_h:end_h, start_w:end_w]
        
        # Apply augmentation if enabled
        if self.augment:
            data_patch, dsm_patch, label_patch = self._apply_augmentation(
                data_patch, dsm_patch, label_patch
            )
        
        # Normalize RGB channels
        data_patch = data_patch.astype(np.float32) / 255.0
        for c in range(3):  # Only normalize RGB channels
            data_patch[c] = (data_patch[c] - self.mean[c]) / self.std[c]
        
        # Normalize DSM
        dsm_patch = dsm_patch.astype(np.float32)
        if dsm_patch.max() > dsm_patch.min():
            dsm_patch = (dsm_patch - dsm_patch.min()) / (dsm_patch.max() - dsm_patch.min())
        
        # Convert to tensors (keep RGB and DSM separate as expected by trainer)
        rgb_tensor = torch.from_numpy(data_patch.astype(np.float32))  # [3, H, W]
        dsm_tensor = torch.from_numpy(dsm_patch.astype(np.float32))   # [H, W]
        label_tensor = torch.from_numpy(label_patch.astype(np.int64))  # [H, W]
        
        # Return format expected by trainer: (data, dsm, target)
        return rgb_tensor, dsm_tensor, label_tensor
    
    def _apply_augmentation(
        self, 
        data: np.ndarray, 
        dsm: np.ndarray, 
        label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply data augmentation to a sample.
        Enhanced for multi-scale training.
        """
        
        # Random horizontal flip
        if random.random() > 0.5:
            data = np.flip(data, axis=2).copy()
            dsm = np.flip(dsm, axis=1).copy()
            label = np.flip(label, axis=1).copy()
        
        # Random vertical flip  
        if random.random() > 0.5:
            data = np.flip(data, axis=1).copy()
            dsm = np.flip(dsm, axis=0).copy()
            label = np.flip(label, axis=0).copy()
        
        # Random rotation (90°, 180°, 270°)
        if random.random() > 0.5:
            k = random.randint(1, 3)
            data = np.rot90(data, k=k, axes=(1, 2)).copy()
            dsm = np.rot90(dsm, k=k, axes=(0, 1)).copy()
            label = np.rot90(label, k=k, axes=(0, 1)).copy()
        
        # Scale-aware augmentation: adjust based on scale level
        # Higher scale levels (more zoomed out) get more aggressive augmentation
        
        # Random brightness/contrast (more aggressive for coarser scales)
        if random.random() > 0.3:
            brightness_factor = random.uniform(0.8, 1.2)
            contrast_factor = random.uniform(0.8, 1.2)
            
            data_p = data.astype(np.float32)
            for c in range(3):  # Only adjust RGB channels
                # Apply brightness
                data_p[c] = data_p[c] * brightness_factor
                # Apply contrast
                mean_val = data_p[c].mean()
                data_p[c] = (data_p[c] - mean_val) * contrast_factor + mean_val
                # Clip to valid range
                data_p[c] = np.clip(data_p[c], 0, 255)
            
            data = data_p.astype(data.dtype)
        
        # Scale-specific scale augmentation (simulate zoom variations)
        if random.random() > 0.5:
            scale_factor = random.uniform(0.9, 1.1)  # Subtle scale changes
            
            orig_h, orig_w = data.shape[1], data.shape[2]
            new_h, new_w = int(orig_h * scale_factor), int(orig_w * scale_factor)
            
            if new_h > 0 and new_w > 0:
                # Scale RGB data
                data_scaled = np.zeros((data.shape[0], new_h, new_w), dtype=data.dtype)
                for c in range(data.shape[0]):
                    data_scaled[c] = transform.resize(
                        data[c], (new_h, new_w), 
                        preserve_range=True, anti_aliasing=True
                    ).astype(data.dtype)
                
                # Scale DSM and labels
                dsm_scaled = transform.resize(
                    dsm, (new_h, new_w), 
                    preserve_range=True, anti_aliasing=True
                ).astype(dsm.dtype)
                
                label_scaled = transform.resize(
                    label, (new_h, new_w), 
                    preserve_range=True, anti_aliasing=False, order=0
                ).astype(label.dtype)
                
                # Crop or pad to original size
                if new_h >= orig_h and new_w >= orig_w:
                    # Crop from center
                    start_h = (new_h - orig_h) // 2
                    start_w = (new_w - orig_w) // 2
                    data = data_scaled[:, start_h:start_h+orig_h, start_w:start_w+orig_w]
                    dsm = dsm_scaled[start_h:start_h+orig_h, start_w:start_w+orig_w]
                    label = label_scaled[start_h:start_h+orig_h, start_w:start_w+orig_w]
                else:
                    # Pad to center
                    pad_h = (orig_h - new_h) // 2
                    pad_w = (orig_w - new_w) // 2
                    
                    data_temp = np.zeros_like(data)
                    dsm_temp = np.zeros_like(dsm)
                    label_temp = np.zeros_like(label)
                    
                    data_temp[:, pad_h:pad_h+new_h, pad_w:pad_w+new_w] = data_scaled
                    dsm_temp[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = dsm_scaled
                    label_temp[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = label_scaled
                    
                    data, dsm, label = data_temp, dsm_temp, label_temp
        
        return data, dsm, label
    
    def get_scale_level_stats(self) -> Dict:
        """Get statistics about scale level distribution in the dataset."""
        return {
            'distribution': self.scale_distribution,
            'total_tiles': len(self.tile_ids),
            'unique_scales': len(self.scale_distribution),
            'scale_range': (min(self.scale_distribution.keys()), max(self.scale_distribution.keys()))
        }
    
    def filter_by_scale_level(self, scale_levels: List[int]) -> 'MultiScaleSegmentationDataset':
        """
        Create a new dataset with only specific scale levels.
        
        Args:
            scale_levels: List of scale levels to include
            
        Returns:
            New MultiScaleSegmentationDataset with filtered tiles
        """
        filtered_ids = [
            tile_id for tile_id in self.tile_ids
            if self.multiscale_mapping[tile_id]['scale_level'] in scale_levels
        ]
        
        return MultiScaleSegmentationDataset(
            tile_ids=filtered_ids,
            multiscale_mapping_file=None,  # Pass None since we already have the mapping
            window_size=self.window_size,
            stride_size=self.stride_size,
            mean=self.mean.tolist(),
            std=self.std.tolist(),
            augment=self.augment,
            cache=self.cache,
            debug=self.debug
        )


def create_multiscale_datasets(
    dataset_info_file: str,
    window_size: Tuple[int, int] = (256, 256),
    stride_size: int = 64,
    augment_train: bool = True,
    cache: bool = False,
    debug: bool = False
) -> Tuple[MultiScaleSegmentationDataset, MultiScaleSegmentationDataset]:
    """
    Create train and test datasets from multiscale dataset info.
    
    Args:
        dataset_info_file: Path to the multiscale dataset info JSON file
        window_size: Size of input patches
        stride_size: Stride for patch extraction
        augment_train: Whether to apply augmentation to training set
        cache: Whether to cache loaded tiles
        debug: Enable debug mode
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    
    # Load dataset info
    with open(dataset_info_file, 'r') as f:
        dataset_info = json.load(f)
    
    mapping_file = dataset_info['multiscale_mapping_file']
    train_ids = dataset_info['train_ids']
    test_ids = dataset_info['test_ids']
    
    # Create datasets
    train_dataset = MultiScaleSegmentationDataset(
        tile_ids=train_ids,
        multiscale_mapping_file=mapping_file,
        window_size=window_size,
        stride_size=stride_size,
        augment=augment_train,
        cache=cache,
        debug=debug
    )
    
    test_dataset = MultiScaleSegmentationDataset(
        tile_ids=test_ids,
        multiscale_mapping_file=mapping_file,
        window_size=window_size,
        stride_size=stride_size,
        augment=False,  # No augmentation for test set
        cache=cache,
        debug=debug
    )
    
    if debug:
        print(f"Created multiscale datasets:")
        print(f"  Train: {len(train_dataset)} patches from {len(train_ids)} tiles")
        print(f"  Test:  {len(test_dataset)} patches from {len(test_ids)} tiles")
        print(f"  Train scale stats: {train_dataset.get_scale_level_stats()}")
        print(f"  Test scale stats:  {test_dataset.get_scale_level_stats()}")
    
    return train_dataset, test_dataset

