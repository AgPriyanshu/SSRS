"""
Clean training script for SSRS semantic segmentation.

This script provides a simplified and organized approach to training and testing
the UNetFormer model for semantic segmentation on custom datasets.
"""

import os
import argparse
from typing import Optional, Tuple

import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.transform import from_bounds

from config import config
from model_wrapper import ModelWrapper
from trainer import Trainer
from evaluator import Evaluator




def setup_output_directories() -> None:
    """Create output directories if they don't exist."""
    output_dir = config.get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory ready: {output_dir}")


def train_model(epochs: Optional[int] = None) -> None:
    """Train the model.
    
    Args:
        epochs: Number of epochs to train (uses config default if None)
    """
    print("="*60)
    print("INITIALIZING TRAINING")
    print("="*60)
    
    # Set mode and print configuration
    config.set_mode("Train")
    config.print_config()
    
    # Setup output directories
    setup_output_directories()
    
    # Initialize components
    print("\\nInitializing model...")
    model_wrapper = ModelWrapper(num_classes=config.n_classes)
    model_wrapper.print_parameter_summary()
    
    print("\\nInitializing evaluator...")
    evaluator = Evaluator()
    
    print("\\nInitializing trainer...")
    trainer = Trainer(model_wrapper, evaluator)
    
    # Start training
    print("\\nStarting training process...")
    trainer.train(epochs=epochs)


def test_model(model_path: str = "YOUR_MODEL") -> None:
    """Test the model.
    
    Args:
        model_path: Path to the trained model weights
    """
    print("="*60)
    print("INITIALIZING TESTING")
    print("="*60)
    
    # Set mode and print configuration
    config.set_mode("Test")
    config.print_config()
    
    # Initialize components
    print("\\nInitializing model...")
    model_wrapper = ModelWrapper(num_classes=config.n_classes)
    model_wrapper.print_parameter_summary()
    
    # Load trained weights
    if model_path == "YOUR_MODEL":
        output_dir = config.get_output_dir()
        model_path = os.path.join(output_dir, model_path)
        print(f"\\nWarning: Using placeholder model path: {model_path}")
        print("Please specify a valid model path for testing.")
        return
    
    print(f"\\nLoading model weights from: {model_path}")
    try:
        model_wrapper.load_weights(model_path, strict=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Initialize evaluator
    print("\\nInitializing evaluator...")
    evaluator = Evaluator()
    
    # Run evaluation
    print("\\nStarting evaluation...")
    miou = evaluator.evaluate_and_save(
        model_wrapper=model_wrapper,
        test_ids=config.dataset.test_ids,
        stride=32,
        save_predictions=True
    )
    
    print("\\n" + "="*60)
    print("TESTING COMPLETED")
    print(f"Final mIoU: {miou:.4f}")
    print("="*60)


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="SSRS Semantic Segmentation Training/Testing"
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["train", "test"], 
        default="train",
        help="Mode to run: train or test"
    )
    
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="default",
        help="Name of the dataset (used for output directory)"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=None,
        help="Number of epochs to train (uses config default if not specified)"
    )
    
    parser.add_argument(
        "--model-path", 
        type=str, 
        default="YOUR_MODEL",
        help="Path to model weights for testing"
    )
    
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data/",
        help="Root directory of the dataset"
    )
    
    parser.add_argument(
        "--data-pattern",
        type=str,
        default="images/img_{}.jpg",
        help="Pattern for data files (e.g., 'images/img_{}.jpg')"
    )
    
    parser.add_argument(
        "--dsm-pattern",
        type=str,
        default="dsm/dsm_{}.tif",
        help="Pattern for DSM files (e.g., 'dsm/dsm_{}.tif')"
    )
    
    parser.add_argument(
        "--label-pattern",
        type=str,
        default="labels/label_{}.png",
        help="Pattern for label files (e.g., 'labels/label_{}.png')"
    )
    
    parser.add_argument(
        "--train-ids",
        type=str,
        nargs="+",
        default=["train_001", "train_002", "train_003"],
        help="List of training image IDs"
    )
    
    parser.add_argument(
        "--test-ids",
        type=str,
        nargs="+",
        default=["test_001", "test_002"],
        help="List of test image IDs"
    )
    
    args = parser.parse_args()
    
    # Configure dataset
    from config import DatasetConfig
    dataset_config = DatasetConfig(
        name=args.dataset_name,
        train_ids=args.train_ids,
        test_ids=args.test_ids,
        stride_size=64,
        epochs=args.epochs or 50,
        save_epoch=1,
        data_root=args.data_root,
        data_pattern=args.data_pattern,
        dsm_pattern=args.dsm_pattern,
        label_pattern=args.label_pattern
    )
    
    config.set_dataset_config(dataset_config)
    print(f"Dataset configured: {args.dataset_name}")
    print(f"Data root: {args.data_root}")
    
    try:
        if args.mode == "train":
            train_model(epochs=args.epochs)
        elif args.mode == "test":
            test_model(model_path=args.model_path)
    except KeyboardInterrupt:
        print("\\nTraining/Testing interrupted by user.")
    except Exception as e:
        print(f"\\nError: {e}")
        raise


# Main entry point
if __name__ == "__main__":
    # Example usage:
    # python main.py --mode train --dataset-name my_dataset --data-root ./my_data/ --epochs 50
    # python main.py --mode test --dataset-name my_dataset --data-root ./my_data/ --model-path ./results_my_dataset/model.pth
    # 
    # With custom patterns:
    # python main.py --mode train --data-pattern "rgb/{}.png" --label-pattern "gt/{}_mask.png"
    #
    # With custom train/test splits:
    # python main.py --mode train --train-ids img001 img002 img003 --test-ids img004 img005
    
    # Example usage of binary mask creation from shapefile:
    # 
    # Option 1: Using a reference raster (recommended)
    # mask = create_binary_mask_from_shapefile(
    #     shapefile_path="buildings.shp",
    #     reference_raster_path="satellite_image.tif"
    # )
    #
    # Option 2: Specifying parameters manually  
    # mask = create_binary_mask_from_shapefile(
    #     shapefile_path="buildings.shp",
    #     output_shape=(1024, 1024),
    #     bounds=(left, bottom, right, top),  # in CRS units
    #     pixel_size=0.5,  # meters per pixel
    #     crs="EPSG:32633"  # UTM Zone 33N
    # )
    #
    # Save the mask:
    # from PIL import Image
    # Image.fromarray(mask * 255).save("building_mask.png")
    
    main()
