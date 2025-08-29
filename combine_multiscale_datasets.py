#!/usr/bin/env python3
"""
Script to combine multiple multiscale datasets into one mega training dataset.

This script combines the MOPR and Aarvi multiscale datasets to create the ultimate
training dataset with maximum diversity across scale levels and geographic regions.
"""

import json
import shutil
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))


def combine_multiscale_datasets(
    dataset_paths: list,
    output_name: str = "MEGA_multiscale",
    output_root: str = "./prepared_datasets"
) -> dict:
    """
    Combine multiple multiscale datasets into one mega dataset.
    
    Args:
        dataset_paths: List of paths to multiscale dataset info files
        output_name: Name for the combined dataset
        output_root: Root directory for output
        
    Returns:
        Combined dataset info dictionary
    """
    
    print("="*80)
    print("🔗 COMBINING MULTISCALE DATASETS INTO MEGA DATASET")
    print("="*80)
    
    combined_train_ids = []
    combined_test_ids = []
    combined_mapping = {}
    dataset_stats = []
    
    total_datasets = 0
    total_scale_levels = 0
    
    # Process each dataset
    for i, dataset_path in enumerate(dataset_paths):
        print(f"\n📂 Processing dataset {i+1}/{len(dataset_paths)}: {dataset_path}")
        
        # Load dataset info
        with open(dataset_path, 'r') as f:
            dataset_info = json.load(f)
        
        dataset_name = dataset_info['name']
        mapping_file = dataset_info['multiscale_mapping_file']
        
        print(f"   Name: {dataset_name}")
        print(f"   Train tiles: {len(dataset_info['train_ids']):,}")
        print(f"   Test tiles: {len(dataset_info['test_ids']):,}")
        print(f"   Scale levels: {dataset_info.get('num_scale_levels', 4)}")
        
        # Load the dataset's mapping
        with open(mapping_file, 'r') as f:
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
        
        # Store dataset stats
        dataset_stats.append({
            'name': dataset_name,
            'source_path': dataset_path,
            'train_tiles': len(dataset_info['train_ids']),
            'test_tiles': len(dataset_info['test_ids']),
            'total_tiles': len(dataset_info['train_ids']) + len(dataset_info['test_ids']),
            'scale_levels': dataset_info.get('num_scale_levels', 4)
        })
        
        total_datasets += 1
        total_scale_levels = max(total_scale_levels, dataset_info.get('num_scale_levels', 4))
        
        print(f"   ✅ Added {len(dataset_info['train_ids']) + len(dataset_info['test_ids']):,} tiles")
    
    # Create output directory
    output_dir = Path(output_root) / output_name
    output_dir.mkdir(exist_ok=True)
    
    # Save combined mapping
    mapping_file = output_dir / "multiscale_mapping.json"
    with open(mapping_file, 'w') as f:
        json.dump(combined_mapping, f, indent=2)
    
    # Calculate scale distribution
    scale_distribution = {}
    for tile_id, tile_info in combined_mapping.items():
        scale_level = tile_info['scale_level']
        scale_distribution[scale_level] = scale_distribution.get(scale_level, 0) + 1
    
    # Create combined dataset info
    combined_info = {
        'name': output_name,
        'dataset_dir': str(output_dir),
        'total_tiles': len(combined_train_ids) + len(combined_test_ids),
        'train_ids': combined_train_ids,
        'test_ids': combined_test_ids,
        'tile_size': [256, 256],
        'overlap': 32,
        'num_datasets': total_datasets,
        'num_scale_levels': total_scale_levels,
        'source_datasets': [ds['name'] for ds in dataset_stats],
        'dataset_statistics': dataset_stats,
        'scale_distribution': scale_distribution,
        'data_pattern': "images/img_{}.png",
        'dsm_pattern': "dsm/dsm_{}.tif",
        'label_pattern': "labels/label_{}.png",
        'multiscale_mapping_file': str(mapping_file),
        'geographic_diversity': True,
        'multiscale_fusion': True,
        'creation_info': {
            'created_from': [str(p) for p in dataset_paths],
            'combination_method': 'prefix_based_mapping',
            'maintains_scale_hierarchy': True,
            'cross_dataset_training': True
        }
    }
    
    # Save combined dataset info
    info_file = output_dir / "dataset_info.json"
    with open(info_file, 'w') as f:
        json.dump(combined_info, f, indent=2)
    
    # Print results
    print(f"\n" + "="*80)
    print("🎉 MEGA MULTISCALE DATASET CREATION COMPLETE!")
    print("="*80)
    
    print(f"\n📊 FINAL MEGA DATASET STATISTICS:")
    print(f"  🔢 Total training tiles: {len(combined_train_ids):,}")
    print(f"  🔢 Total testing tiles:  {len(combined_test_ids):,}")
    print(f"  🔢 Total tiles:         {combined_info['total_tiles']:,}")
    print(f"  📈 Source datasets:     {total_datasets}")
    print(f"  📈 Scale levels:        {total_scale_levels}")
    print(f"  🌍 Geographic regions:  {total_datasets}")
    
    print(f"\n📋 BREAKDOWN BY SOURCE DATASET:")
    for stats in dataset_stats:
        print(f"  {stats['name']:<25}: {stats['train_tiles']:4,} train + {stats['test_tiles']:3,} test = {stats['total_tiles']:4,} total")
    
    print(f"\n📋 BREAKDOWN BY SCALE LEVEL:")
    for scale_level in sorted(scale_distribution.keys()):
        count = scale_distribution[scale_level]
        percentage = (count / combined_info['total_tiles']) * 100
        print(f"  Scale {scale_level}: {count:,} tiles ({percentage:.1f}%)")
    
    # Calculate improvements
    single_scale_estimate = combined_info['total_tiles'] // total_scale_levels
    improvement_vs_single = combined_info['total_tiles'] / single_scale_estimate
    
    print(f"\n🚀 MEGA IMPROVEMENTS:")
    print(f"  📈 vs single scale: {improvement_vs_single:.1f}x more data!")
    print(f"  🌍 Geographic diversity: {total_datasets} different regions")
    print(f"  🔍 Scale diversity: {total_scale_levels} different zoom levels")
    print(f"  🎯 Perfect for Deep Fusion Module!")
    
    print(f"\n📁 MEGA DATASET LOCATION:")
    print(f"  📂 Directory: {output_dir}")
    print(f"  📄 Info file: {info_file}")
    print(f"  📄 Mapping file: {mapping_file}")
    
    print(f"\n✨ DEEP FUSION MODULE BENEFITS:")
    print(f"  ✅ Multi-scale feature learning across {total_scale_levels} zoom levels")
    print(f"  ✅ Multi-geographic generalization across {total_datasets} regions")
    print(f"  ✅ Enhanced robustness and reduced overfitting")
    print(f"  ✅ Ultimate training dataset diversity!")
    
    return combined_info


def main():
    """Main function to combine multiscale datasets."""
    
    print("="*80)
    print("🚀 MEGA MULTISCALE DATASET COMBINER")
    print("="*80)
    
    # Define dataset paths to combine
    dataset_paths = [
        "prepared_datasets/MOPR_multiscale_multiscale/dataset_info.json",
        "prepared_datasets/Aarvi_multiscale_multiscale/dataset_info.json"
    ]
    
    # Verify all paths exist
    missing_paths = []
    for path in dataset_paths:
        if not Path(path).exists():
            missing_paths.append(path)
    
    if missing_paths:
        print(f"❌ ERROR: Missing dataset files:")
        for path in missing_paths:
            print(f"   {path}")
        return 1
    
    print(f"📂 Found {len(dataset_paths)} multiscale datasets to combine:")
    for path in dataset_paths:
        print(f"   ✅ {path}")
    
    # Combine datasets
    combined_info = combine_multiscale_datasets(
        dataset_paths=dataset_paths,
        output_name="MEGA_multiscale",
        output_root="./prepared_datasets"
    )
    
    print(f"\n🎊 SUCCESS! MEGA MULTISCALE DATASET READY!")
    print(f"   📊 {combined_info['total_tiles']:,} tiles from {combined_info['num_datasets']} regions")
    print(f"   🔍 {combined_info['num_scale_levels']} scale levels for multi-scale learning")
    print(f"   🚀 Perfect for your Deep Fusion Module training!")
    
    return 0


if __name__ == "__main__":
    exit(main())
