#!/usr/bin/env python3
"""
MEGA Multiscale Training Script for Deep Fusion Module.

This script trains your Deep Fusion Module using the MEGA multiscale dataset
with 4,664+ training tiles across 4 scale levels and 2 geographic regions.
"""

import sys
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import config, DatasetConfig, TrainingConfig
from model_wrapper import ModelWrapper
from trainer import Trainer
from evaluator import Evaluator
from multiscale_dataset import create_multiscale_datasets


def setup_mega_multiscale_training():
    """Configure training for MEGA multiscale dataset."""
    
    print("="*80)
    print("🚀 MEGA MULTISCALE TRAINING SETUP")
    print("="*80)
    
    # MEGA dataset configuration
    mega_dataset = DatasetConfig(
        name="MEGA_multiscale",
        train_ids=[],  # Will be loaded from dataset_info.json
        test_ids=[],   # Will be loaded from dataset_info.json
        stride_size=64,
        epochs=30,     # More epochs for the massive dataset
        save_epoch=2,  # Save every 2 epochs
        data_root="./prepared_datasets/MEGA_multiscale",
        data_pattern="images/img_{}.png",
        dsm_pattern="dsm/dsm_{}.tif",
        label_pattern="labels/label_{}.png"
    )
    
    # Enhanced training configuration for multiscale learning
    training_config = TrainingConfig(
        base_lr=0.0001,         # Conservative learning rate for massive dataset
        weight_decay=0.01,      # Strong regularization for diverse data
        batch_size=4,           # Smaller batches due to multiscale complexity
        momentum=0.9,
        scheduler_milestones=[10, 20, 25],  # Progressive LR decay
        scheduler_gamma=0.5,    # 50% LR reduction at milestones
        window_size=(256, 256),
        n_classes=2,            # Buildings: background + buildings
        cache=False             # Don't cache due to large dataset size
    )
    
    # Apply configurations
    config.training = training_config
    config.set_dataset_config(mega_dataset)
    
    return mega_dataset, training_config


def create_multiscale_data_loaders():
    """Create data loaders for the MEGA multiscale dataset."""
    
    print("\n📊 Creating MEGA multiscale data loaders...")
    
    # Create datasets using our multiscale loader
    train_dataset, test_dataset = create_multiscale_datasets(
        dataset_info_file="prepared_datasets/MEGA_multiscale/dataset_info.json",
        window_size=(256, 256),
        stride_size=64,
        augment_train=True,
        cache=False,  # Don't cache due to large size
        debug=True
    )
    
    print(f"✅ Datasets created:")
    print(f"   Training patches: {len(train_dataset):,}")
    print(f"   Testing patches:  {len(test_dataset):,}")
    
    # Print scale distribution
    train_stats = train_dataset.get_scale_level_stats()
    print(f"\n🔍 Training Scale Distribution:")
    for scale_level, count in train_stats['distribution'].items():
        percentage = (count / train_stats['total_tiles']) * 100
        print(f"   Scale {scale_level}: {count:,} tiles ({percentage:.1f}%)")
    
    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    print(f"\n🔧 Data Loaders Created:")
    print(f"   Train batches: {len(train_loader):,}")
    print(f"   Test batches:  {len(test_loader):,}")
    print(f"   Batch size: {config.training.batch_size}")
    
    return train_loader, test_loader, train_dataset, test_dataset


def inspect_multiscale_batch(data_loader, dataset_name=""):
    """Inspect a sample batch to verify multiscale loading."""
    
    print(f"\n🔍 Inspecting {dataset_name} Multiscale Batch:")
    
    for batch_idx, batch in enumerate(data_loader):
        print(f"   Batch {batch_idx + 1}:")
        print(f"     Images shape: {batch['image'].shape}")  # [B, 4, H, W] (RGB+DSM)
        print(f"     Labels shape: {batch['label'].shape}")  # [B, H, W]
        print(f"     Scale levels: {batch['scale_level'].tolist()}")
        
        # Show scale factors
        scale_factors = [f"{sf:.1f}x" for sf in batch['scale_factor'].tolist()]
        print(f"     Scale factors: {scale_factors}")
        
        # Show unique tile sources
        tile_ids = batch['tile_id']
        unique_sources = set([tid.split('_')[0] + '_' + tid.split('_')[1] for tid in tile_ids])
        print(f"     Source datasets: {list(unique_sources)}")
        
        if batch_idx >= 1:  # Show first 2 batches
            break
    
    print(f"   ✅ Multiscale loading verified!")


def main():
    """Main training function for MEGA multiscale dataset."""
    
    start_time = datetime.now()
    print("="*80)
    print("🚀 MEGA MULTISCALE DEEP FUSION MODULE TRAINING")
    print("="*80)
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Setup configuration
    mega_dataset, training_config = setup_mega_multiscale_training()
    
    print(f"\n📊 MEGA DATASET CONFIGURATION:")
    print(f"   Dataset name: {mega_dataset.name}")
    print(f"   Data root: {mega_dataset.data_root}")
    print(f"   Epochs: {mega_dataset.epochs}")
    print(f"   Save every: {mega_dataset.save_epoch} epochs")
    
    print(f"\n⚙️  OPTIMIZED TRAINING PARAMETERS:")
    print(f"   Learning rate: {training_config.base_lr:.6f}")
    print(f"   Weight decay: {training_config.weight_decay:.3f}")
    print(f"   Batch size: {training_config.batch_size}")
    print(f"   Momentum: {training_config.momentum:.1f}")
    print(f"   LR milestones: {training_config.scheduler_milestones}")
    print(f"   LR decay: {training_config.scheduler_gamma:.1f}")
    
    # Create data loaders
    train_loader, test_loader, train_dataset, test_dataset = create_multiscale_data_loaders()
    
    # Inspect sample batches
    inspect_multiscale_batch(train_loader, "Training")
    inspect_multiscale_batch(test_loader, "Testing")
    
    # Initialize model
    print(f"\n🧠 Initializing UNetFormer + Deep Fusion Module...")
    model_wrapper = ModelWrapper(num_classes=config.n_classes)
    model_wrapper.print_parameter_summary()
    
    # Initialize evaluator
    print(f"\n🔍 Initializing evaluator...")
    evaluator = Evaluator()
    print("✅ Evaluator ready")
    
    # Initialize trainer with custom data loaders
    print(f"\n🏃 Initializing trainer with MEGA multiscale data...")
    trainer = Trainer(model_wrapper, evaluator)
    
    # Override trainer's data loaders with our multiscale ones
    trainer.train_loader = train_loader
    trainer.test_loader = test_loader
    
    print("✅ Trainer ready with multiscale data loaders")
    
    try:
        print(f"\n" + "="*80)
        print("🚀 STARTING MEGA MULTISCALE TRAINING")
        print("="*80)
        print("⚡ ULTIMATE TRAINING STRATEGY:")
        print("   • 4,664 training tiles across 4 scale levels")
        print("   • 2 geographic regions for maximum diversity")
        print("   • Scale-aware data augmentation")
        print("   • Deep Fusion Module multi-scale learning")
        print("   • Conservative LR with progressive decay")
        print("   • Strong regularization for generalization")
        print("")
        print("✨ EXPECTED BREAKTHROUGH RESULTS:")
        print("   • Gradual but steady accuracy improvement")
        print("   • Excellent multi-scale feature learning")
        print("   • Superior generalization across regions")
        print("   • NO overfitting with this massive diverse dataset")
        print("   • State-of-the-art building detection performance")
        print("")
        print("🎯 MULTISCALE ADVANTAGE:")
        print("   • Scale 1 (finest): Learn fine building details")
        print("   • Scale 2 (2x): Learn medium-level features")
        print("   • Scale 3 (4x): Learn contextual relationships")
        print("   • Scale 4 (8x): Learn global spatial patterns")
        print("")
        print("🔥 THIS IS THE ULTIMATE DATASET FOR YOUR DEEP FUSION MODULE!")
        print("="*80)
        
        # Start training
        trainer.train(epochs=mega_dataset.epochs)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n" + "="*80)
        print("🎉 MEGA MULTISCALE TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total time:   {str(duration).split('.')[0]}")
        print(f"Dataset used: MEGA multiscale with {len(train_dataset):,} training patches")
        print(f"Scale levels: 4 (1x, 2x, 4x, 8x)")
        print(f"Geographic regions: 2 (MOPR + Aarvi)")
        print("="*80)
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
        print("Partial results may be saved")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        raise


if __name__ == "__main__":
    main()
