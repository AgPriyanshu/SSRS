"""Configuration module for SSRS training."""

import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch


@dataclass
class DatasetConfig:
    """Configuration for dataset parameters."""
    name: str
    train_ids: List[str]
    test_ids: List[str]
    stride_size: int
    epochs: int
    save_epoch: int
    data_root: str
    data_pattern: str  # Pattern for data files, e.g. "images/img_{}.png"
    dsm_pattern: str   # Pattern for DSM files, e.g. "dsm/dsm_{}.tif"
    label_pattern: str # Pattern for label files, e.g. "labels/label_{}.png"
    eroded_pattern: Optional[str] = None  # Optional eroded labels pattern


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    base_lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0005
    scheduler_milestones: List[int] = None
    scheduler_gamma: float = 0.1
    batch_size: int = 10
    window_size: Tuple[int, int] = (256, 256)
    n_classes: int = 6
    cache: bool = True
    
    def __post_init__(self):
        if self.scheduler_milestones is None:
            self.scheduler_milestones = [25, 35, 45]


class Config:
    """Main configuration class for the SSRS project."""
    
    def __init__(self):
        # Model configuration
        self.model_name = "UNetformer"
        self.mode = "Train"  # "Train" or "Test"
        self.if_sam = True
        self.rgb_only = False  # Set to True to use RGB-only model
        self.fusion_strategy = "dual_rgb"  # For RGB-only: "single_encoder", "dual_rgb", "self_fusion"
        
        # Hardware configuration
        self.cuda_device = "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_device
        
        # Data folder path - modify this to your dataset path
        self.data_root = "./data/"
        
        # Training configuration
        self.training = TrainingConfig()
        
        # Labels and classes - configured for building detection
        self.labels = ["background", "buildings"]
        self.n_classes = len(self.labels)
        
        # Color palette for visualization - can be customized
        self.color_palette = {
            0: (0, 0, 0),        # Background (black)
            1: (255, 0, 0),      # Class 1 (red)
            2: (0, 255, 0),      # Class 2 (green)
            3: (0, 0, 255),      # Class 3 (blue)
            4: (255, 255, 0),    # Class 4 (yellow)
            5: (255, 0, 255),    # Class 5 (magenta)
            6: (0, 255, 255),    # Additional class (cyan)
        }
        
        # Initialize with default dataset configuration
        self.dataset = None
        self._setup_default_dataset()
    
    def set_dataset_config(self, dataset_config: DatasetConfig):
        """Set the dataset configuration."""
        self.dataset = dataset_config
    
    def set_mode(self, mode: str):
        """Set the mode (Train/Test)."""
        if mode not in ["Train", "Test"]:
            raise ValueError("Mode must be 'Train' or 'Test'")
        self.mode = mode
    
    def set_rgb_only(self, rgb_only: bool, fusion_strategy: str = None):
        """Set RGB-only mode and fusion strategy.
        
        Args:
            rgb_only: Whether to use RGB-only model
            fusion_strategy: Fusion strategy for RGB-only model
        """
        self.rgb_only = rgb_only
        if fusion_strategy is not None:
            if fusion_strategy not in ["single_encoder", "dual_rgb", "self_fusion"]:
                raise ValueError("Invalid fusion strategy")
            self.fusion_strategy = fusion_strategy
    
    def set_labels(self, labels: List[str], color_palette: dict = None):
        """Set custom labels and color palette.
        
        Args:
            labels: List of class labels
            color_palette: Optional color palette dict {class_id: (r, g, b)}
        """
        self.labels = labels
        self.n_classes = len(labels)
        if color_palette:
            self.color_palette = color_palette
    
    def _setup_default_dataset(self):
        """Setup default dataset configuration."""
        self.dataset = DatasetConfig(
            name="default",
            train_ids=["train_001", "train_002", "train_003"],
            test_ids=["test_001", "test_002"],
            stride_size=64,
            epochs=50,
            save_epoch=1,
            data_root=self.data_root,
            data_pattern="images/img_{}.png",
            dsm_pattern="dsm/dsm_{}.tif",
            label_pattern="labels/label_{}.png",
            eroded_pattern="labels_eroded/label_{}_eroded.png"
        )
    
    def get_output_dir(self) -> str:
        """Get the output directory for saving models."""
        if self.dataset and self.dataset.name:
            return f"./results_{self.dataset.name}/"
        else:
            return "./results/"
    
    def print_config(self):
        """Print the current configuration."""
        print(f"Model: {self.model_name}")
        print(f"Mode: {self.mode}")
        print(f"Dataset: {self.dataset.name if self.dataset else 'None'}")
        print(f"SAM enabled: {self.if_sam}")
        print(f"RGB-only mode: {self.rgb_only}")
        if self.rgb_only:
            print(f"Fusion strategy: {self.fusion_strategy}")
        print(f"Window size: {self.training.window_size}")
        print(f"Batch size: {self.training.batch_size}")
        if self.dataset:
            print(f"Stride size: {self.dataset.stride_size}")
            print(f"Epochs: {self.dataset.epochs}")
            print(f"Save epoch: {self.dataset.save_epoch}")
        print(f"Learning rate: {self.training.base_lr}")
        print(f"Number of classes: {self.n_classes}")
        print(f"Class labels: {self.labels}")


# Create a global config instance
config = Config()
