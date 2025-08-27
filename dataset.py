import os
import random
import numpy as np
import torch
from skimage import io
from utils import convert_from_color, get_random_pos

class SemanticSegmentationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        ids,
        data_pattern=None,
        dsm_pattern=None,
        label_pattern=None,
        data_root="./data/",
        cache=False,
        augmentation=True,
    ):
        super(SemanticSegmentationDataset, self).__init__()

        self.augmentation = augmentation
        self.cache = cache
        self.data_root = data_root
        
        # Set default patterns if not provided
        if data_pattern is None:
            data_pattern = "images/img_{}.jpg"
        if dsm_pattern is None:
            dsm_pattern = "dsm/dsm_{}.tif" 
        if label_pattern is None:
            label_pattern = "labels/label_{}.png"
            
        # List of files using patterns
        self.data_files = [os.path.join(data_root, data_pattern.format(id)) for id in ids]
        self.dsm_files = [os.path.join(data_root, dsm_pattern.format(id)) for id in ids]
        self.label_files = [os.path.join(data_root, label_pattern.format(id)) for id in ids]

        # Sanity check: warn if files don't exist (but don't fail immediately)
        missing_files = []
        for f in self.data_files + self.dsm_files + self.label_files:
            if not os.path.isfile(f):
                missing_files.append(f)
        
        if missing_files:
            print(f"Warning: {len(missing_files)} files not found:")
            for f in missing_files[:5]:  # Show first 5 missing files
                print(f"  - {f}")
            if len(missing_files) > 5:
                print(f"  ... and {len(missing_files) - 5} more")

        # Initialize cache dicts
        self.data_cache_ = {}
        self.dsm_cache_ = {}
        self.label_cache_ = {}

    def __len__(self):
        # Return a fixed number for training (can be adjusted based on your needs)
        return 10 * 1000

    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True

        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))

        return tuple(results)

    def __getitem__(self, i):
        # Pick a random image
        random_idx = random.randint(0, len(self.data_files) - 1)

        # If the tile hasn't been loaded yet, put in cache
        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
        else:
            # Load and normalize data in [0, 1]
            try:
                img_data = io.imread(self.data_files[random_idx])
                
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
                print(f"Error loading data file {self.data_files[random_idx]}: {e}")
                # Create dummy data as fallback
                data = np.zeros((3, 256, 256), dtype="float32")
                
            if self.cache:
                self.data_cache_[random_idx] = data

        if random_idx in self.dsm_cache_.keys():
            dsm = self.dsm_cache_[random_idx]
        else:
            # Load and normalize DSM in [0, 1]
            try:
                dsm = np.asarray(io.imread(self.dsm_files[random_idx]), dtype="float32")
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
                print(f"Error loading DSM file {self.dsm_files[random_idx]}: {e}")
                # Create dummy DSM as fallback
                dsm = np.zeros((256, 256), dtype="float32")
                
            if self.cache:
                self.dsm_cache_[random_idx] = dsm

        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
        else:
            # Load labels
            try:
                label_img = io.imread(self.label_files[random_idx])
                
                # Handle different label formats
                if len(label_img.shape) == 3:
                    # RGB labels - convert to class indices
                    label = np.asarray(convert_from_color(label_img), dtype="int64")
                else:
                    # Grayscale labels - assume already class indices
                    label = np.asarray(label_img, dtype="int64")
            except Exception as e:
                print(f"Error loading label file {self.label_files[random_idx]}: {e}")
                # Create dummy label as fallback
                label = np.zeros((256, 256), dtype="int64")
                
            if self.cache:
                self.label_cache_[random_idx] = label

        # Get a random patch
        from constants import WINDOW_SIZE
        x1, x2, y1, y2 = get_random_pos(data, WINDOW_SIZE)
        data_p = data[:, x1:x2, y1:y2]
        dsm_p = dsm[x1:x2, y1:y2]
        label_p = label[x1:x2, y1:y2]

        # Data augmentation
        data_p, dsm_p, label_p = self.data_augmentation(data_p, dsm_p, label_p)

        # Return the torch.Tensor values
        return (
            torch.from_numpy(data_p),
            torch.from_numpy(dsm_p),
            torch.from_numpy(label_p),
        )

