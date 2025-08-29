#!/usr/bin/env python3
"""
Script to prepare multi-scale training dataset for Deep Fusion Module.

This script generates tiles at multiple overview levels (zoom levels) to leverage
the multi-scale capabilities of your Deep Fusion Module architecture.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import List, Dict, Tuple

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from utils.raster_helpers import (
    prepare_training_dataset,
    validate_dataset_spatial_compatibility,
    calculate_dataset_statistics,
    analyze_raster_resolutions
)
from config import config, DatasetConfig


def get_overview_resolutions(base_resolution: float, num_levels: int = 4) -> List[float]:
    """
    Generate overview resolutions for multi-scale training.
    
    Args:
        base_resolution: Base (finest) resolution in units/pixel
        num_levels: Number of overview levels to generate
        
    Returns:
        List of resolutions from finest to coarsest
    """
    resolutions = []
    for i in range(num_levels):
        # Each level is 2x coarser than the previous
        resolution = base_resolution * (2 ** i)
        resolutions.append(resolution)
    
    return resolutions


def prepare_multiscale_dataset(
    ortho_path: str,
    dsm_path: str, 
    mask_path: str,
    output_root: str,
    dataset_name: str,
    tile_size: Tuple[int, int] = (256, 256),
    overlap: int = 32,
    train_ratio: float = 0.8,
    min_valid_pixels: float = 0.1,
    num_scale_levels: int = 4
) -> Dict:
    """
    Prepare dataset with multiple scale levels for Deep Fusion Module training.
    """
    
    print("="*70)
    print("🔄 MULTI-SCALE DATASET PREPARATION FOR DEEP FUSION MODULE")
    print("="*70)
    
    # Analyze base resolution
    resolution_info = analyze_raster_resolutions(
        [ortho_path, dsm_path, mask_path],
        ['ortho', 'dsm', 'mask']
    )
    
    # Use finest resolution as base
    base_resolution = min([
        resolution_info['ortho']['pixel_size_x'],
        resolution_info['dsm']['pixel_size_x'], 
        resolution_info['mask']['pixel_size_x']
    ])
    
    print(f"Base (finest) resolution: {base_resolution:.3f} units/pixel")
    
    # Generate overview resolutions
    target_resolutions = get_overview_resolutions(base_resolution, num_scale_levels)
    
    print(f"\n📊 Will generate {num_scale_levels} scale levels:")
    for i, res in enumerate(target_resolutions):
        scale_factor = res / base_resolution
        print(f"  Level {i+1}: {res:.3f} units/pixel (scale: {scale_factor:.1f}x)")
    
    # Prepare datasets for each scale level
    all_datasets = {}
    total_train_tiles = 0
    total_test_tiles = 0
    
    for level, target_res in enumerate(target_resolutions, 1):
        print(f"\n" + "="*50)
        print(f"🔍 PROCESSING SCALE LEVEL {level}/{num_scale_levels}")
        print(f"Resolution: {target_res:.3f} units/pixel")
        print("="*50)
        
        # Create scale-specific dataset name
        scale_dataset_name = f"{dataset_name}_scale{level}"
        
        # Prepare dataset for this scale
        dataset_info = prepare_training_dataset(
            ortho_path=ortho_path,
            dsm_path=dsm_path,
            mask_path=mask_path,
            output_root=output_root,
            dataset_name=scale_dataset_name,
            tile_size=tile_size,
            overlap=overlap,
            train_ratio=train_ratio,
            min_valid_pixels=min_valid_pixels,
            target_resolution='custom',
            custom_pixel_size=target_res
        )
        
        # Store dataset info
        all_datasets[f"scale_{level}"] = {
            'resolution': target_res,
            'scale_factor': target_res / base_resolution,
            'dataset_info': dataset_info,
            'train_tiles': len(dataset_info['train_ids']),
            'test_tiles': len(dataset_info['test_ids'])
        }
        
        total_train_tiles += len(dataset_info['train_ids'])
        total_test_tiles += len(dataset_info['test_ids'])
        
        print(f"✅ Scale {level} complete: {len(dataset_info['train_ids'])} train + {len(dataset_info['test_ids'])} test tiles")
    
    # Create combined dataset configuration
    print(f"\n" + "="*70)
    print("🔗 CREATING COMBINED MULTI-SCALE DATASET")
    print("="*70)
    
    # Combine all train/test IDs with scale prefixes
    combined_train_ids = []
    combined_test_ids = []
    scale_mapping = {}
    
    for scale_key, scale_data in all_datasets.items():
        level = scale_key.split('_')[1]
        scale_dataset_name = f"{dataset_name}_scale{level}"
        
        # Add train IDs with scale prefix
        for train_id in scale_data['dataset_info']['train_ids']:
            combined_id = f"scale{level}_{train_id}"
            combined_train_ids.append(combined_id)
            scale_mapping[combined_id] = {
                'dataset_path': scale_data['dataset_info']['dataset_dir'],
                'original_id': train_id,
                'scale_level': int(level),
                'resolution': scale_data['resolution'],
                'scale_factor': scale_data['scale_factor']
            }
        
        # Add test IDs with scale prefix
        for test_id in scale_data['dataset_info']['test_ids']:
            combined_id = f"scale{level}_{test_id}"
            combined_test_ids.append(combined_id)
            scale_mapping[combined_id] = {
                'dataset_path': scale_data['dataset_info']['dataset_dir'],
                'original_id': test_id,
                'scale_level': int(level),
                'resolution': scale_data['resolution'],
                'scale_factor': scale_data['scale_factor']
            }
    
    # Save combined mapping
    multiscale_output_dir = Path(output_root) / f"{dataset_name}_multiscale"
    multiscale_output_dir.mkdir(exist_ok=True)
    
    mapping_file = multiscale_output_dir / "multiscale_mapping.json"
    with open(mapping_file, 'w') as f:
        json.dump(scale_mapping, f, indent=2)
    
    # Create combined dataset info
    combined_dataset_info = {
        'name': f"{dataset_name}_multiscale",
        'dataset_dir': str(multiscale_output_dir),
        'total_tiles': total_train_tiles + total_test_tiles,
        'train_ids': combined_train_ids,
        'test_ids': combined_test_ids,
        'tile_size': list(tile_size),
        'overlap': overlap,
        'num_scale_levels': num_scale_levels,
        'base_resolution': base_resolution,
        'scale_levels': {
            f"scale_{level}": {
                'resolution': scale_data['resolution'],
                'scale_factor': scale_data['scale_factor'],
                'train_tiles': scale_data['train_tiles'],
                'test_tiles': scale_data['test_tiles']
            }
            for level, (scale_key, scale_data) in enumerate(all_datasets.items(), 1)
        },
        'data_pattern': "images/img_{}.png",
        'dsm_pattern': "dsm/dsm_{}.tif",
        'label_pattern': "labels/label_{}.png",
        'multiscale_mapping_file': str(mapping_file)
    }
    
    # Save combined dataset info
    info_file = multiscale_output_dir / "dataset_info.json"
    with open(info_file, 'w') as f:
        json.dump(combined_dataset_info, f, indent=2)
    
    return combined_dataset_info


def main():
    parser = argparse.ArgumentParser(
        description="Prepare multi-scale dataset for Deep Fusion Module training"
    )
    
    # Input files
    parser.add_argument("--ortho", required=True, 
                       help="Path to large orthophoto file")
    parser.add_argument("--dsm", required=True,
                       help="Path to large DSM file")
    parser.add_argument("--mask", required=True,
                       help="Path to large mask file")
    
    # Output configuration
    parser.add_argument("--output-root", default="./prepared_datasets",
                       help="Root directory for prepared datasets")
    parser.add_argument("--dataset-name", required=True,
                       help="Base name for the dataset")
    
    # Multi-scale parameters
    parser.add_argument("--num-scales", type=int, default=4,
                       help="Number of scale levels to generate (default: 4)")
    
    # Tiling parameters
    parser.add_argument("--tile-size", nargs=2, type=int, default=[256, 256],
                       help="Tile size (height width)")
    parser.add_argument("--overlap", type=int, default=32,
                       help="Overlap between tiles in pixels")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="Ratio of tiles for training (rest for testing)")
    parser.add_argument("--min-valid-pixels", type=float, default=0.1,
                       help="Minimum valid pixel ratio per tile")
    
    args = parser.parse_args()
    
    # Convert paths to absolute paths
    ortho_path = Path(args.ortho).resolve()
    dsm_path = Path(args.dsm).resolve()
    mask_path = Path(args.mask).resolve()
    
    # Validate input files exist
    for path, name in [(ortho_path, "ortho"), (dsm_path, "DSM"), (mask_path, "mask")]:
        if not path.exists():
            print(f"Error: {name} file not found: {path}")
            return 1
    
    print("="*70)
    print("🚀 MULTI-SCALE DATASET PREPARATION")
    print("="*70)
    print(f"Orthophoto: {ortho_path}")
    print(f"DSM: {dsm_path}")
    print(f"Mask: {mask_path}")
    print(f"Dataset name: {args.dataset_name}")
    print(f"Number of scales: {args.num_scales}")
    print(f"Tile size: {args.tile_size[0]}x{args.tile_size[1]}")
    print(f"Overlap: {args.overlap} pixels")
    print(f"Train ratio: {args.train_ratio}")
    
    # Validate spatial compatibility first
    print("\\n🔍 Validating spatial compatibility...")
    compatibility = validate_dataset_spatial_compatibility(
        str(ortho_path), str(dsm_path), str(mask_path)
    )
    
    if not compatibility['validation']['spatially_compatible']:
        print("❌ ERROR: Input files are not spatially compatible!")
        return 1
    
    print("✅ Files are spatially compatible!")
    
    # Prepare multi-scale dataset
    combined_info = prepare_multiscale_dataset(
        ortho_path=str(ortho_path),
        dsm_path=str(dsm_path),
        mask_path=str(mask_path),
        output_root=args.output_root,
        dataset_name=args.dataset_name,
        tile_size=tuple(args.tile_size),
        overlap=args.overlap,
        train_ratio=args.train_ratio,
        min_valid_pixels=args.min_valid_pixels,
        num_scale_levels=args.num_scales
    )
    
    # Print final results
    print("\\n" + "="*70)
    print("🎉 MULTI-SCALE DATASET PREPARATION COMPLETE!")
    print("="*70)
    
    total_train = len(combined_info['train_ids'])
    total_test = len(combined_info['test_ids'])
    total_tiles = total_train + total_test
    
    print(f"\\n📊 FINAL STATISTICS:")
    print(f"  🔢 Total training tiles: {total_train:,}")
    print(f"  🔢 Total testing tiles:  {total_test:,}")
    print(f"  🔢 Total tiles:         {total_tiles:,}")
    print(f"  📈 Scale levels:        {args.num_scales}")
    
    print(f"\\n📋 BREAKDOWN BY SCALE LEVEL:")
    for scale_key, scale_info in combined_info['scale_levels'].items():
        level = scale_key.split('_')[1]
        print(f"  Scale {level}: {scale_info['train_tiles']:4d} train + {scale_info['test_tiles']:3d} test = {scale_info['train_tiles'] + scale_info['test_tiles']:4d} total")
        print(f"           Resolution: {scale_info['resolution']:.3f} units/pixel (scale: {scale_info['scale_factor']:.1f}x)")
    
    # Calculate improvement
    single_scale_estimate = total_tiles // args.num_scales
    improvement_factor = total_tiles / single_scale_estimate
    print(f"\\n🚀 IMPROVEMENT:")
    print(f"  vs single scale: {improvement_factor:.1f}x more training data!")
    print(f"  Deep Fusion Module can now learn multi-scale features!")
    
    print(f"\\n📁 Dataset location: {combined_info['dataset_dir']}")
    print(f"📄 Mapping file: {combined_info['multiscale_mapping_file']}")
    
    print("\\n🔧 TO USE THIS DATASET:")
    print("1. Update your multi_dataset.py to handle the multiscale mapping")
    print("2. Modify your training script to use the combined dataset")
    print("3. Your Deep Fusion Module will automatically benefit from multi-scale features!")
    
    return 0


if __name__ == "__main__":
    exit(main())

