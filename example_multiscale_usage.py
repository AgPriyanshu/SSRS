#!/usr/bin/env python3
"""
Example usage of the multi-scale dataset generation and training system.

This example shows how to:
1. Generate multi-scale tiles from your existing datasets
2. Train your Deep Fusion Module with multi-scale data
3. Leverage the enhanced dataset for better generalization
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from prepare_multiscale_dataset import prepare_multiscale_dataset
from multiscale_dataset import create_multiscale_datasets
from config import config
import json


def generate_multiscale_for_existing_datasets():
    """
    Generate multi-scale versions of your existing datasets.
    """
    
    print("="*70)
    print("🚀 GENERATING MULTI-SCALE DATASETS")
    print("="*70)
    
    # Define your existing datasets and their source files
    datasets_to_process = [
        {
            'name': 'aarvi_multiscale',
            'ortho': 'path/to/aarvi/ortho.tif',  # Update with your actual paths
            'dsm': 'path/to/aarvi/dsm.tif',
            'mask': 'path/to/aarvi/mask.tif'
        },
        {
            'name': 'buildings_multiscale', 
            'ortho': 'path/to/buildings/ortho.tif',  # Update with your actual paths
            'dsm': 'path/to/buildings/dsm.tif',
            'mask': 'path/to/buildings/mask.tif'
        }
    ]
    
    all_multiscale_datasets = []
    
    for dataset_config in datasets_to_process:
        print(f"\\n🔄 Processing {dataset_config['name']}...")
        
        # Generate multi-scale dataset
        multiscale_info = prepare_multiscale_dataset(
            ortho_path=dataset_config['ortho'],
            dsm_path=dataset_config['dsm'],
            mask_path=dataset_config['mask'],
            output_root="./prepared_datasets",
            dataset_name=dataset_config['name'],
            tile_size=(256, 256),
            overlap=32,
            train_ratio=0.8,
            min_valid_pixels=0.1,
            num_scale_levels=4  # Generate 4 scale levels
        )
        
        all_multiscale_datasets.append(multiscale_info)
        
        total_tiles = len(multiscale_info['train_ids']) + len(multiscale_info['test_ids'])
        print(f"✅ {dataset_config['name']}: {total_tiles:,} total tiles generated!")
    
    return all_multiscale_datasets


def combine_multiscale_datasets(datasets_info_list):
    """
    Combine multiple multiscale datasets into one mega-dataset.
    """
    
    print("\\n" + "="*70)
    print("🔗 COMBINING MULTI-SCALE DATASETS")
    print("="*70)
    
    combined_train_ids = []
    combined_test_ids = []
    combined_mapping = {}
    
    for i, dataset_info in enumerate(datasets_info_list):
        dataset_name = dataset_info['name']
        
        # Load the dataset's mapping
        with open(dataset_info['multiscale_mapping_file'], 'r') as f:
            mapping = json.load(f)
        
        # Add train IDs with dataset prefix
        for train_id in dataset_info['train_ids']:
            prefixed_id = f"{dataset_name}_{train_id}"
            combined_train_ids.append(prefixed_id)
            combined_mapping[prefixed_id] = mapping[train_id]
        
        # Add test IDs with dataset prefix  
        for test_id in dataset_info['test_ids']:
            prefixed_id = f"{dataset_name}_{test_id}"
            combined_test_ids.append(prefixed_id)
            combined_mapping[prefixed_id] = mapping[test_id]
    
    # Save combined mapping
    output_dir = Path("./prepared_datasets/combined_multiscale")
    output_dir.mkdir(exist_ok=True)
    
    mapping_file = output_dir / "combined_multiscale_mapping.json"
    with open(mapping_file, 'w') as f:
        json.dump(combined_mapping, f, indent=2)
    
    # Create combined dataset info
    combined_info = {
        'name': 'combined_multiscale',
        'dataset_dir': str(output_dir),
        'total_tiles': len(combined_train_ids) + len(combined_test_ids),
        'train_ids': combined_train_ids,
        'test_ids': combined_test_ids,
        'tile_size': [256, 256],
        'overlap': 32,
        'num_datasets': len(datasets_info_list),
        'source_datasets': [d['name'] for d in datasets_info_list],
        'data_pattern': "images/img_{}.png",
        'dsm_pattern': "dsm/dsm_{}.tif", 
        'label_pattern': "labels/label_{}.png",
        'multiscale_mapping_file': str(mapping_file)
    }
    
    # Save combined dataset info
    info_file = output_dir / "dataset_info.json"
    with open(info_file, 'w') as f:
        json.dump(combined_info, f, indent=2)
    
    print(f"🎉 Combined Dataset Created:")
    print(f"  📊 Total training tiles: {len(combined_train_ids):,}")
    print(f"  📊 Total testing tiles:  {len(combined_test_ids):,}")
    print(f"  📊 Total tiles:         {combined_info['total_tiles']:,}")
    print(f"  📁 Location: {output_dir}")
    
    return combined_info


def train_with_multiscale_dataset(dataset_info_file):
    """
    Example of how to train with the multi-scale dataset.
    """
    
    print("\\n" + "="*70)
    print("🧠 TRAINING WITH MULTI-SCALE DATASET")
    print("="*70)
    
    # Create multi-scale datasets
    train_dataset, test_dataset = create_multiscale_datasets(
        dataset_info_file=dataset_info_file,
        window_size=(256, 256),
        stride_size=64,
        augment_train=True,
        cache=False,  # Set to True if you have enough RAM
        debug=True
    )
    
    print(f"\\n📈 Dataset Statistics:")
    print(f"  Training patches: {len(train_dataset):,}")
    print(f"  Testing patches:  {len(test_dataset):,}")
    
    # Print scale distribution
    train_stats = train_dataset.get_scale_level_stats()
    print(f"\\n🔍 Training Scale Distribution:")
    for scale_level, count in train_stats['distribution'].items():
        percentage = (count / train_stats['total_tiles']) * 100
        print(f"  Scale {scale_level}: {count:,} tiles ({percentage:.1f}%)")
    
    # Create data loaders
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=6,  # Adjust based on your GPU memory
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=6,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"\\n🔧 Data Loaders Created:")
    print(f"  Train batches: {len(train_loader):,}")
    print(f"  Test batches:  {len(test_loader):,}")
    
    # Example: inspect a batch to see multi-scale features
    print(f"\\n🔍 Inspecting Multi-Scale Batch:")
    for batch_idx, batch in enumerate(train_loader):
        print(f"  Batch {batch_idx + 1}:")
        print(f"    Images shape: {batch['image'].shape}")  # [B, 4, H, W] (RGB+DSM)
        print(f"    Labels shape: {batch['label'].shape}")  # [B, H, W]
        print(f"    Scale levels: {batch['scale_level'].tolist()}")
        print(f"    Scale factors: {[f'{sf:.1f}x' for sf in batch['scale_factor'].tolist()]}")
        
        if batch_idx >= 2:  # Just show first 3 batches
            break
    
    print(f"\\n✅ Ready for training with {len(train_dataset):,} multi-scale patches!")
    print(f"\\n🚀 Your Deep Fusion Module will now learn from:")
    print(f"   • Multiple geographic regions")
    print(f"   • Multiple scale levels (zoom levels)")
    print(f"   • Diverse building types and contexts")
    print(f"   • Enhanced data augmentation")
    
    return train_loader, test_loader


def estimate_tile_counts_for_your_data():
    """
    Estimate how many tiles you'll get with multi-scale generation.
    """
    
    print("\\n" + "="*70)
    print("📊 TILE COUNT PROJECTIONS FOR YOUR DATASETS")
    print("="*70)
    
    # Your current datasets
    current_datasets = {
        'aarvi_clean': 562,
        'buildings_clean': 785, 
        'buildings_massive': 2380
    }
    
    scale_levels = [1, 2, 3, 4, 5]
    
    print(f"\\n📈 Projected Tile Counts by Scale Levels:")
    print(f"{'Dataset':<20} {'Current':<10} " + "".join([f"{'Scale ' + str(s):<12}" for s in scale_levels]))
    print("-" * 80)
    
    total_current = sum(current_datasets.values())
    
    for dataset_name, current_count in current_datasets.items():
        row = f"{dataset_name:<20} {current_count:<10,}"
        for scale_level in scale_levels:
            projected = current_count * scale_level
            row += f"{projected:<12,}"
        print(row)
    
    print("-" * 80)
    row = f"{'TOTAL':<20} {total_current:<10,}"
    for scale_level in scale_levels:
        projected_total = total_current * scale_level
        row += f"{projected_total:<12,}"
    print(row)
    
    print(f"\\n🎯 RECOMMENDATIONS:")
    print(f"  • Use 4 scale levels for optimal balance")
    print(f"  • Expected total: {total_current * 4:,} tiles")
    print(f"  • That's {4}x more training data!")
    print(f"  • Perfect for your Deep Fusion Module architecture")
    
    print(f"\\n💡 BENEFITS:")
    print(f"  ✅ Multi-scale feature learning")
    print(f"  ✅ Better generalization across zoom levels")
    print(f"  ✅ Reduced overfitting")
    print(f"  ✅ Enhanced robustness")


def main():
    """
    Main example showing the complete multi-scale workflow.
    """
    
    print("="*70)
    print("🚀 MULTI-SCALE DATASET GENERATION EXAMPLE")
    print("="*70)
    
    # Step 1: Estimate tile counts
    estimate_tile_counts_for_your_data()
    
    # Step 2: Show how to generate (commented out - requires actual file paths)
    print(f"\\n" + "="*50)
    print("📝 TO GENERATE MULTI-SCALE DATASETS:")
    print("="*50)
    print("1. Update the file paths in generate_multiscale_for_existing_datasets()")
    print("2. Run: python prepare_multiscale_dataset.py --ortho your_ortho.tif --dsm your_dsm.tif --mask your_mask.tif --dataset-name your_name --num-scales 4")
    print("3. Use the multiscale_dataset.py loader in your training")
    
    # Step 3: Show training setup (commented out - requires generated dataset)
    print(f"\\n" + "="*50)
    print("📝 TO TRAIN WITH MULTI-SCALE DATA:")
    print("="*50)
    print("1. Generate your multi-scale datasets")
    print("2. Use create_multiscale_datasets() to create train/test loaders")
    print("3. Your existing training loop will work with enhanced data")
    
    print(f"\\n🎉 EXPECTED IMPROVEMENTS:")
    print(f"  • 4x more training data")
    print(f"  • Better multi-scale feature learning")
    print(f"  • Improved generalization")
    print(f"  • Reduced overfitting")
    print(f"  • Enhanced model robustness")


if __name__ == "__main__":
    main()

