#!/usr/bin/env python3
"""
Script to prepare training dataset from large orthophoto, DSM, and mask files.

This script integrates the raster_helpers utilities with the SSRS configuration
system to prepare datasets for semantic segmentation training.
"""

import argparse
from pathlib import Path
import sys

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from utils.raster_helpers import (
    prepare_training_dataset,
    validate_dataset_spatial_compatibility,
    calculate_dataset_statistics
)
from config import config, DatasetConfig


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training dataset from large raster files"
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
                       help="Name for the dataset")
    
    # Tiling parameters
    parser.add_argument("--tile-size", nargs=2, type=int, default=[256, 256],
                       help="Tile size (height width)")
    parser.add_argument("--overlap", type=int, default=32,
                       help="Overlap between tiles in pixels")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="Ratio of tiles for training (rest for testing)")
    parser.add_argument("--min-valid-pixels", type=float, default=0.1,
                       help="Minimum valid pixel ratio per tile")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--stride-size", type=int, default=64,
                       help="Stride size for training")
    
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
    
    print("="*60)
    print("PREPARING LARGE DATASET FOR SSRS TRAINING")
    print("="*60)
    print(f"Orthophoto: {ortho_path}")
    print(f"DSM: {dsm_path}")
    print(f"Mask: {mask_path}")
    print(f"Dataset name: {args.dataset_name}")
    print(f"Tile size: {args.tile_size[0]}x{args.tile_size[1]}")
    print(f"Overlap: {args.overlap} pixels")
    print(f"Train ratio: {args.train_ratio}")
    print("Resolution: Automatically downsample to lowest resolution")
    
    # Step 1: Validate spatial compatibility
    print("\\n" + "="*60)
    print("STEP 1: VALIDATING SPATIAL COMPATIBILITY")
    print("="*60)
    
    compatibility = validate_dataset_spatial_compatibility(
        str(ortho_path), str(dsm_path), str(mask_path)
    )
    
    print("Spatial Compatibility Results:")
    print(f"  CRS match: {compatibility['validation']['crs_match']}")
    print(f"  Spatial overlap: {compatibility['validation']['spatial_overlap']}")
    print(f"  Overlap ratio: {compatibility['validation']['overlap_ratio']:.3f}")
    print(f"  Different resolutions: {compatibility['validation']['resolutions_vary']}")
    print(f"  Compatible for processing: {compatibility['validation']['spatially_compatible']}")
    
    print("\\nFile Details:")
    for key in ['ortho', 'dsm', 'mask']:
        dims = compatibility['dimensions'][key]
        resolution = compatibility['resolutions'][key]
        crs = compatibility['crs'][key]
        print(f"  {key}: {dims[0]}x{dims[1]}, {resolution:.3f} units/pixel, {crs}")
    
    if not compatibility['validation']['spatially_compatible']:
        print("\\n❌ ERROR: Input files are not spatially compatible!")
        print("\\nPossible issues:")
        if not compatibility['validation']['crs_match']:
            print("  - Files have different coordinate reference systems")
        if not compatibility['validation']['spatial_overlap']:
            print("  - Files don't spatially overlap")
        if compatibility['validation']['overlap_ratio'] <= 0.1:
            print("  - Files have insufficient spatial overlap (<10%)")
        print("\\nPlease ensure all files:")
        print("  - Use the same coordinate reference system")
        print("  - Cover overlapping geographic areas")
        return 1
    
    if compatibility['validation']['resolutions_vary']:
        print("\\n⚠️  Files have different resolutions - will be aligned to coarsest resolution")
    else:
        print("\\n✅ All files have the same resolution!")
    
    print("✅ Files are spatially compatible for processing!")
    
    # Step 2: Prepare dataset
    print("\\n" + "="*60)
    print("STEP 2: TILING AND PREPARING DATASET")
    print("="*60)
    
    dataset_info = prepare_training_dataset(
        ortho_path=str(ortho_path),
        dsm_path=str(dsm_path),
        mask_path=str(mask_path),
        output_root=args.output_root,
        dataset_name=args.dataset_name,
        tile_size=tuple(args.tile_size),
        overlap=args.overlap,
        train_ratio=args.train_ratio,
        min_valid_pixels=args.min_valid_pixels,
        target_resolution='coarsest'
    )
    
    # Step 3: Calculate statistics
    print("\\n" + "="*60)
    print("STEP 3: DATASET STATISTICS")
    print("="*60)
    
    stats = calculate_dataset_statistics(dataset_info['dataset_dir'])
    print(f"Images: {stats['image_count']}")
    print(f"DSMs: {stats['dsm_count']}")  
    print(f"Labels: {stats['label_count']}")
    print(f"Total size: {stats['total_size_mb']:.2f} MB")
    
    # Step 4: Configure SSRS
    print("\\n" + "="*60)
    print("STEP 4: CONFIGURING SSRS")
    print("="*60)
    
    # Create SSRS dataset configuration
    dataset_config = DatasetConfig(
        name=args.dataset_name,
        train_ids=dataset_info['train_ids'],
        test_ids=dataset_info['test_ids'],
        stride_size=args.stride_size,
        epochs=args.epochs,
        save_epoch=1,
        data_root=dataset_info['dataset_dir'],
        data_pattern=dataset_info['data_pattern'],
        dsm_pattern=dataset_info['dsm_pattern'],
        label_pattern=dataset_info['label_pattern']
    )
    
    # Update global config
    config.set_dataset_config(dataset_config)
    
    print("SSRS Configuration:")
    print(f"  Dataset name: {dataset_config.name}")
    print(f"  Data root: {dataset_config.data_root}")
    print(f"  Training tiles: {len(dataset_config.train_ids)}")
    print(f"  Testing tiles: {len(dataset_config.test_ids)}")
    print(f"  Epochs: {dataset_config.epochs}")
    print(f"  Stride size: {dataset_config.stride_size}")
    
    # Step 5: Training instructions
    print("\\n" + "="*60)
    print("DATASET PREPARATION COMPLETE!")
    print("="*60)
    print("\\nTo start training, run:")
    print(f"python main.py --mode train --dataset-name {args.dataset_name} \\\\")
    print(f"              --data-root {dataset_info['dataset_dir']} \\\\")
    print(f"              --epochs {args.epochs}")
    
    print("\\nTo test the model after training:")
    print(f"python main.py --mode test --dataset-name {args.dataset_name} \\\\")
    print(f"              --data-root {dataset_info['dataset_dir']} \\\\")
    print(f"              --model-path ./results_{args.dataset_name}/YOUR_MODEL.pth")
    
    print(f"\\nDataset files are located in: {dataset_info['dataset_dir']}")
    print("\\n✅ Ready for training!")
    
    return 0


if __name__ == "__main__":
    exit(main())
