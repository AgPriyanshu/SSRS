#!/usr/bin/env python3
"""
Simple training script for building detection using prepared dataset.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import config, DatasetConfig
from model_wrapper import ModelWrapper
from trainer import Trainer
from evaluator import Evaluator

def main():
    print("="*60)
    print("BUILDING DETECTION TRAINING")
    print("="*60)
    print("Starting training script...")
    
    # Configure dataset for buildings
    print("Configuring dataset...")
    buildings_dataset = DatasetConfig(
        name="buildings",
        train_ids=[f"{i:06d}" for i in range(788)],  # 80% of 986 tiles - just numbers
        test_ids=[f"{i:06d}" for i in range(788, 986)],  # 20% of 986 tiles - just numbers
        stride_size=64,
        epochs=10,  # Start with fewer epochs for testing
        save_epoch=2,
        data_root="./prepared_datasets/buildings",
        data_pattern="images/img_{}.png",  # This will add "img_" prefix
        dsm_pattern="dsm/dsm_{}.tif", 
        label_pattern="labels/label_{}.png"
    )
    print("Dataset configuration created.")
    
    # Apply configuration
    config.set_dataset_config(buildings_dataset)
    
    # Print configuration
    print("Configuration:")
    print(f"  Dataset: {config.dataset.name}")
    print(f"  Classes: {config.n_classes} ({config.labels})")
    print(f"  Training tiles: {len(config.dataset.train_ids)}")
    print(f"  Test tiles: {len(config.dataset.test_ids)}")
    print(f"  Data root: {config.dataset.data_root}")
    print(f"  Epochs: {config.dataset.epochs}")
    
    # Initialize model
    print("\\nInitializing UNetFormer model...")
    model_wrapper = ModelWrapper(num_classes=config.n_classes)
    model_wrapper.print_parameter_summary()
    
    # Initialize evaluator
    print("\\nInitializing evaluator...")
    evaluator = Evaluator()
    
    # Initialize trainer
    print("\\nInitializing trainer...")
    trainer = Trainer(model_wrapper, evaluator)
    
    # Start training
    print("\\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    try:
        trainer.train(epochs=config.dataset.epochs)
        print("\\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\\nTraining interrupted by user.")
    except Exception as e:
        print(f"\\nTraining error: {e}")
        raise

if __name__ == "__main__":
    main()
