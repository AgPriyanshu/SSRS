import os
import random
import numpy as np
import torch
import json
from skimage import io, transform
from scipy.ndimage import rotate, gaussian_filter
from utils2 import convert_from_color, get_random_pos

class MultiDatasetSemanticSegmentation(torch.utils.data.Dataset):
    """
    Multi-dataset loader that combines multiple prepared datasets for maximum diversity.
    Handles loading from different dataset directories based on ID mapping.
    """
    
    def __init__(
        self,
        ids,
        mapping_file="multi_dataset_mapping.json",
        data_root="./prepared_datasets/",
        cache=False,
        augmentation=True,
    ):
        super().__init__()

        self.augmentation = augmentation
        self.cache = cache
        self.data_root = data_root
        
        # Load dataset mapping
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                self.dataset_mapping = json.load(f)
        else:
            raise FileNotFoundError(f"Dataset mapping file not found: {mapping_file}")
        
        # Filter IDs to only those in mapping
        self.ids = [id for id in ids if id in self.dataset_mapping]
        
        if len(self.ids) != len(ids):
            missing = len(ids) - len(self.ids)
            print(f"Warning: {missing} IDs not found in dataset mapping")
        
        print(f"Multi-dataset loader initialized with {len(self.ids)} samples")
        
        # Initialize cache dicts
        self.data_cache_ = {}
        self.dsm_cache_ = {}
        self.label_cache_ = {}

    def __len__(self):
        # Return a fixed number for training (can be adjusted based on your needs)
        return 10 * 1000

    def _get_file_paths(self, id):
        """Get file paths for a given ID from the mapping."""
        if id not in self.dataset_mapping:
            raise KeyError(f"ID {id} not found in dataset mapping")
        
        mapping = self.dataset_mapping[id]
        dataset_path = mapping['dataset_path']
        original_id = mapping['original_id']
        
        data_file = os.path.join(dataset_path, "images", f"img_{original_id}.png")
        dsm_file = os.path.join(dataset_path, "dsm", f"dsm_{original_id}.tif")
        label_file = os.path.join(dataset_path, "labels", f"label_{original_id}.png")
        
        return data_file, dsm_file, label_file

    @classmethod
    def data_augmentation(cls, data_p, dsm_p, label_p):
        """
        AGGRESSIVE data augmentation to combat severe overfitting.
        Applies multiple random transformations to increase dataset diversity.
        """
        # Convert to numpy arrays for processing
        if torch.is_tensor(data_p):
            data_p = data_p.numpy()
        if torch.is_tensor(dsm_p):
            dsm_p = dsm_p.numpy()
        if torch.is_tensor(label_p):
            label_p = label_p.numpy()
        
        # 1. GEOMETRIC TRANSFORMATIONS (Applied to all: data, dsm, labels)
        
        # Random rotation (0-360 degrees)
        if random.random() < 0.7:  # 70% chance
            angle = random.uniform(-30, 30)  # Limit to ±30° for building detection
            
            # Rotate RGB data (handle channel dimension)
            if len(data_p.shape) == 3:  # (C, H, W)
                data_rotated = np.zeros_like(data_p)
                for c in range(data_p.shape[0]):
                    data_rotated[c] = rotate(data_p[c], angle, reshape=False, mode='reflect')
                data_p = data_rotated
            else:  # (H, W)
                data_p = rotate(data_p, angle, reshape=False, mode='reflect')
            
            # Rotate DSM and labels
            dsm_p = rotate(dsm_p, angle, reshape=False, mode='reflect')
            label_p = rotate(label_p, angle, reshape=False, mode='nearest', order=0)
        
        # Random horizontal flip
        if random.random() < 0.5:
            if len(data_p.shape) == 3:
                data_p = data_p[:, :, ::-1]
            else:
                data_p = data_p[:, ::-1]
            dsm_p = dsm_p[:, ::-1]
            label_p = label_p[:, ::-1]
        
        # Random vertical flip
        if random.random() < 0.5:
            if len(data_p.shape) == 3:
                data_p = data_p[:, ::-1, :]
            else:
                data_p = data_p[::-1, :]
            dsm_p = dsm_p[::-1, :]
            label_p = label_p[::-1, :]
        
        # Random scaling + crop back to original size
        if random.random() < 0.6:  # 60% chance
            scale_factor = random.uniform(0.8, 1.2)  # Scale 80%-120%
            
            if len(data_p.shape) == 3:  # (C, H, W)
                new_h, new_w = int(data_p.shape[1] * scale_factor), int(data_p.shape[2] * scale_factor)
                data_scaled = np.zeros((data_p.shape[0], new_h, new_w), dtype=data_p.dtype)
                for c in range(data_p.shape[0]):
                    data_scaled[c] = transform.resize(data_p[c], (new_h, new_w), 
                                                    preserve_range=True, anti_aliasing=True)
                
                # Crop or pad back to original size
                orig_h, orig_w = data_p.shape[1], data_p.shape[2]
                if new_h >= orig_h and new_w >= orig_w:
                    # Crop from center
                    start_h = (new_h - orig_h) // 2
                    start_w = (new_w - orig_w) // 2
                    data_p = data_scaled[:, start_h:start_h+orig_h, start_w:start_w+orig_w]
                else:
                    # Pad to original size
                    data_p = np.zeros_like(data_p)
                    start_h = (orig_h - new_h) // 2
                    start_w = (orig_w - new_w) // 2
                    data_p[:, start_h:start_h+new_h, start_w:start_w+new_w] = data_scaled
            
            # Apply same scaling to DSM and labels
            dsm_scaled = transform.resize(dsm_p, (new_h, new_w), preserve_range=True, anti_aliasing=True)
            label_scaled = transform.resize(label_p, (new_h, new_w), preserve_range=True, 
                                          anti_aliasing=False, order=0)
            
            # Crop/pad back
            orig_h, orig_w = dsm_p.shape
            if new_h >= orig_h and new_w >= orig_w:
                start_h = (new_h - orig_h) // 2
                start_w = (new_w - orig_w) // 2
                dsm_p = dsm_scaled[start_h:start_h+orig_h, start_w:start_w+orig_w]
                label_p = label_scaled[start_h:start_h+orig_h, start_w:start_w+orig_w]
            else:
                dsm_temp = np.zeros_like(dsm_p)
                label_temp = np.zeros_like(label_p)
                start_h = (orig_h - new_h) // 2
                start_w = (orig_w - new_w) // 2
                dsm_temp[start_h:start_h+new_h, start_w:start_w+new_w] = dsm_scaled
                label_temp[start_h:start_h+new_h, start_w:start_w+new_w] = label_scaled
                dsm_p, label_p = dsm_temp, label_temp
        
        # 2. PHOTOMETRIC TRANSFORMATIONS (Applied only to RGB data, not labels)
        
        # Random brightness adjustment
        if random.random() < 0.8:  # 80% chance
            brightness_factor = random.uniform(0.6, 1.4)  # ±40% brightness
            data_p = np.clip(data_p * brightness_factor, 0, 1)
        
        # Random contrast adjustment
        if random.random() < 0.8:  # 80% chance
            contrast_factor = random.uniform(0.7, 1.3)  # ±30% contrast
            mean_val = np.mean(data_p)
            data_p = np.clip((data_p - mean_val) * contrast_factor + mean_val, 0, 1)
        
        # Random hue/saturation shift (for RGB only)
        if random.random() < 0.6 and len(data_p.shape) == 3 and data_p.shape[0] == 3:
            # Simple color channel shuffling/scaling
            channel_factors = np.random.uniform(0.8, 1.2, 3)
            for c in range(3):
                data_p[c] = np.clip(data_p[c] * channel_factors[c], 0, 1)
        
        # Random Gaussian noise
        if random.random() < 0.7:  # 70% chance
            noise_std = random.uniform(0.01, 0.05)  # 1-5% noise
            noise = np.random.normal(0, noise_std, data_p.shape).astype(data_p.dtype)
            data_p = np.clip(data_p + noise, 0, 1)
        
        # Random Gaussian blur
        if random.random() < 0.5:  # 50% chance
            sigma = random.uniform(0.5, 1.5)
            if len(data_p.shape) == 3:
                for c in range(data_p.shape[0]):
                    data_p[c] = gaussian_filter(data_p[c], sigma=sigma)
            else:
                data_p = gaussian_filter(data_p, sigma=sigma)
        
        # 3. DSM-SPECIFIC AUGMENTATION
        
        # Random DSM noise (elevation noise)
        if random.random() < 0.6:  # 60% chance
            dsm_noise_std = random.uniform(0.02, 0.08)  # 2-8% DSM noise
            dsm_noise = np.random.normal(0, dsm_noise_std, dsm_p.shape).astype(dsm_p.dtype)
            dsm_p = np.clip(dsm_p + dsm_noise, 0, 1)
        
        # Random DSM smoothing
        if random.random() < 0.4:  # 40% chance
            dsm_sigma = random.uniform(0.3, 1.0)
            dsm_p = gaussian_filter(dsm_p, sigma=dsm_sigma)
        
        # Ensure all data is in valid ranges
        data_p = np.clip(data_p, 0, 1)  # RGB data must be [0, 1]
        dsm_p = np.clip(dsm_p, 0, 1)    # DSM data must be [0, 1]
        
        # Ensure labels remain integers and in valid range
        label_p = np.round(label_p).astype(np.int64)
        label_p = np.clip(label_p, 0, 1)  # For binary classification (background=0, buildings=1)
        
        return np.copy(data_p), np.copy(dsm_p), np.copy(label_p)

    def __getitem__(self, i):
        # Pick a random ID
        random_idx = random.randint(0, len(self.ids) - 1)
        selected_id = self.ids[random_idx]

        # If the tile hasn't been loaded yet, put in cache
        if selected_id in self.data_cache_.keys():
            data = self.data_cache_[selected_id]
        else:
            # Get file paths for this ID
            data_file, _, _ = self._get_file_paths(selected_id)
            
            # Load and normalize data in [0, 1]
            try:
                img_data = io.imread(data_file)
                
                # Handle different image formats and channels
                if len(img_data.shape) == 3:
                    # Multi-channel image - use first 3 channels (RGB)
                    if img_data.shape[2] >= 3:
                        data = img_data[:, :, :3].transpose((2, 0, 1))
                    else:
                        # Grayscale or single channel - repeat to 3 channels
                        data = np.repeat(img_data[:, :, :1].transpose((2, 0, 1)), 3, axis=0)
                else:
                    # Grayscale image - repeat to 3 channels
                    data = np.repeat(img_data[None, :, :], 3, axis=0)
                
                data = 1 / 255 * np.asarray(data, dtype="float32")
            except Exception as e:
                print(f"Error loading data file {data_file}: {e}")
                # Create dummy data as fallback
                data = np.zeros((3, 256, 256), dtype="float32")
                
            if self.cache:
                self.data_cache_[selected_id] = data

        if selected_id in self.dsm_cache_.keys():
            dsm = self.dsm_cache_[selected_id]
        else:
            # Get file paths for this ID
            _, dsm_file, _ = self._get_file_paths(selected_id)
            
            # Load and normalize DSM in [0, 1]
            try:
                # Use rasterio for TIF files, skimage for others
                if dsm_file.endswith('.tif') or dsm_file.endswith('.tiff'):
                    import rasterio
                    with rasterio.open(dsm_file) as src:
                        dsm = src.read(1).astype("float32")  # Read first band
                else:
                    dsm = np.asarray(io.imread(dsm_file), dtype="float32")
                    # Handle different DSM formats
                    if len(dsm.shape) > 2:
                        dsm = dsm[:, :, 0]  # Take first channel if multi-channel
                
                # Normalize to [0, 1]
                dsm_min = np.min(dsm)
                dsm_max = np.max(dsm)
                if dsm_max > dsm_min:
                    dsm = (dsm - dsm_min) / (dsm_max - dsm_min)
                else:
                    dsm = np.zeros_like(dsm)
            except Exception as e:
                print(f"Error loading DSM file {dsm_file}: {e}")
                # Create dummy DSM as fallback
                dsm = np.zeros((256, 256), dtype="float32")
                
            if self.cache:
                self.dsm_cache_[selected_id] = dsm

        if selected_id in self.label_cache_.keys():
            label = self.label_cache_[selected_id]
        else:
            # Get file paths for this ID
            _, _, label_file = self._get_file_paths(selected_id)
            
            # Load labels
            try:
                label_img = io.imread(label_file)
                
                # Handle different label formats
                if len(label_img.shape) == 3:
                    # RGB labels - convert to class indices
                    label = np.asarray(convert_from_color(label_img), dtype="int64")
                else:
                    # Grayscale labels - assume already class indices
                    label = np.asarray(label_img, dtype="int64")
            except Exception as e:
                print(f"Error loading label file {label_file}: {e}")
                # Create dummy label as fallback
                label = np.zeros((256, 256), dtype="int64")
                
            if self.cache:
                self.label_cache_[selected_id] = label

        # Since our tiles are already 256x256 (WINDOW_SIZE), use the full tile
        from constants import WINDOW_SIZE
        
        # Check if tile is already the right size
        if data.shape[1:] == WINDOW_SIZE:
            # Use full tile - no cropping needed
            data_p = data
            dsm_p = dsm
            label_p = label
        else:
            # Fallback: crop if tile is larger than expected
            x1, x2, y1, y2 = get_random_pos(data, WINDOW_SIZE)
            data_p = data[:, x1:x2, y1:y2]
            dsm_p = dsm[x1:x2, y1:y2]
            label_p = label[x1:x2, y1:y2]

        # Data augmentation
        if self.augmentation:
            data_p, dsm_p, label_p = self.data_augmentation(data_p, dsm_p, label_p)

        # Return the torch.Tensor values
        return (
            torch.from_numpy(data_p),
            torch.from_numpy(dsm_p),
            torch.from_numpy(label_p),
        )
