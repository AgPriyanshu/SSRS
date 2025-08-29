"""
Post-processing script to convert building probability masks to refined polygons.
Handles thresholding, noise removal, morphological operations, and polygon extraction.
"""

import os
import sys
import numpy as np
import cv2
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape, Polygon
from shapely.ops import unary_union
from shapely import simplify, buffer
from skimage import morphology, measure, filters
from skimage.segmentation import clear_border
import argparse
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class BuildingPostProcessor:
    """Post-processing pipeline for building probability masks."""
    
    def __init__(self, probability_threshold: float = 0.5, min_area: int = 50, 
                 max_area: int = 50000, simplify_tolerance: float = 1.0):
        """Initialize post-processor.
        
        Args:
            probability_threshold: Threshold for converting probabilities to binary
            min_area: Minimum building area in pixels
            max_area: Maximum building area in pixels  
            simplify_tolerance: Polygon simplification tolerance
        """
        self.prob_threshold = probability_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.simplify_tolerance = simplify_tolerance
        
    def load_probability_mask(self, mask_path: str) -> tuple:
        """Load probability mask and metadata."""
        print(f"📂 Loading probability mask: {mask_path}")
        
        with rasterio.open(mask_path) as src:
            prob_mask = src.read(1).astype(np.float32)
            transform = src.transform
            crs = src.crs
            
        # Normalize if needed
        if prob_mask.max() > 1.0:
            prob_mask = prob_mask / 255.0
            
        print(f"   Shape: {prob_mask.shape}")
        print(f"   Probability range: {prob_mask.min():.3f} - {prob_mask.max():.3f}")
        print(f"   Mean probability: {prob_mask.mean():.3f}")
        
        return prob_mask, transform, crs
    
    def adaptive_threshold(self, prob_mask: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding for better building extraction."""
        print(f"🎯 Applying adaptive thresholding...")
        
        # Method 1: Fixed threshold
        binary_fixed = (prob_mask >= self.prob_threshold).astype(np.uint8)
        
        # Method 2: Otsu's threshold on probability mask
        # Convert to 8-bit for Otsu
        prob_8bit = (prob_mask * 255).astype(np.uint8)
        otsu_thresh, binary_otsu = cv2.threshold(prob_8bit, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_thresh_norm = otsu_thresh / 255.0
        
        # Method 3: Local adaptive threshold
        prob_8bit_blur = cv2.GaussianBlur(prob_8bit, (15, 15), 0)
        binary_adaptive = cv2.adaptiveThreshold(
            prob_8bit_blur, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2
        )
        
        print(f"   Fixed threshold: {self.prob_threshold:.3f}")
        print(f"   Otsu threshold: {otsu_thresh_norm:.3f}")
        print(f"   Pixels above fixed: {np.sum(binary_fixed):,}")
        print(f"   Pixels above Otsu: {np.sum(binary_otsu):,}")
        print(f"   Pixels above adaptive: {np.sum(binary_adaptive):,}")
        
        # Use the method that gives reasonable coverage (adjust as needed)
        coverage_fixed = np.sum(binary_fixed) / binary_fixed.size
        coverage_otsu = np.sum(binary_otsu) / binary_otsu.size
        
        if 0.005 <= coverage_fixed <= 0.15:  # 0.5% - 15% coverage seems reasonable
            print("   Using fixed threshold")
            return binary_fixed
        elif 0.005 <= coverage_otsu <= 0.15:
            print("   Using Otsu threshold")
            return binary_otsu
        else:
            print("   Using adaptive threshold")
            return binary_adaptive
    
    def clean_binary_mask(self, binary_mask: np.ndarray) -> np.ndarray:
        """Clean binary mask with morphological operations."""
        print("🧹 Cleaning binary mask...")
        
        # Remove small noise with opening
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_small)
        
        # Fill small holes with closing
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_medium)
        
        # Remove border artifacts
        cleaned = clear_border(cleaned)
        
        # Filter by area
        labeled = measure.label(cleaned)
        regions = measure.regionprops(labeled)
        
        for region in regions:
            if region.area < self.min_area or region.area > self.max_area:
                coords = region.coords
                cleaned[coords[:, 0], coords[:, 1]] = 0
        
        removed_pixels = np.sum(binary_mask) - np.sum(cleaned)
        print(f"   Removed {removed_pixels:,} pixels during cleaning")
        
        return cleaned
    
    def extract_polygons_rasterio(self, binary_mask: np.ndarray, transform: rasterio.Affine) -> list:
        """Extract polygons using rasterio.features.shapes."""
        print("🗺️  Extracting polygons with rasterio...")
        
        polygons = []
        
        # Extract shapes (polygons) from binary mask
        for geom, value in shapes(binary_mask.astype(np.uint8), mask=(binary_mask > 0), transform=transform):
            if value == 1:  # Building pixels
                polygon = shape(geom)
                if polygon.is_valid and polygon.area > self.min_area:
                    polygons.append(polygon)
        
        print(f"   Extracted {len(polygons)} polygons")
        return polygons
    
    def refine_polygons(self, polygons: list) -> list:
        """Refine polygons with geometric operations."""
        print("✨ Refining polygons...")
        
        refined = []
        
        for poly in polygons:
            try:
                # Simplify polygon
                simplified = simplify(poly, tolerance=self.simplify_tolerance)
                
                # Small buffer to smooth edges
                buffered = simplified.buffer(0.5)
                unbuffered = buffered.buffer(-0.5)
                
                if unbuffered.area > self.min_area and unbuffered.is_valid:
                    if hasattr(unbuffered, 'geoms'):
                        # MultiPolygon - take largest part
                        largest = max(unbuffered.geoms, key=lambda x: x.area)
                        refined.append(largest)
                    else:
                        refined.append(unbuffered)
                        
            except Exception as e:
                # Keep original if processing fails
                if poly.is_valid and poly.area > self.min_area:
                    refined.append(poly)
        
        print(f"   Refined to {len(refined)} polygons")
        return refined
    
    def save_results(self, polygons: list, output_path: str, crs: str = None) -> None:
        """Save polygons to GeoJSON/Shapefile."""
        print(f"💾 Saving {len(polygons)} polygons...")
        
        if not polygons:
            print("   No polygons to save!")
            return
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
        
        # Add attributes
        gdf['building_id'] = range(1, len(polygons) + 1)
        gdf['area_sqm'] = gdf.geometry.area
        gdf['perimeter_m'] = gdf.geometry.length
        gdf['compactness'] = (4 * np.pi * gdf['area_sqm']) / (gdf['perimeter_m'] ** 2)
        
        # Save
        output_path = Path(output_path)
        if output_path.suffix.lower() == '.shp':
            gdf.to_file(output_path, driver='ESRI Shapefile')
        else:
            output_path = output_path.with_suffix('.geojson')
            gdf.to_file(output_path, driver='GeoJSON')
        
        print(f"   Saved to: {output_path}")
        
        # Statistics
        print(f"\n📊 Building Statistics:")
        print(f"   Total buildings: {len(polygons)}")
        print(f"   Total area: {gdf['area_sqm'].sum():.1f} m²")
        print(f"   Average area: {gdf['area_sqm'].mean():.1f} m²")
        print(f"   Largest: {gdf['area_sqm'].max():.1f} m²")
        print(f"   Smallest: {gdf['area_sqm'].min():.1f} m²")
    
    def create_visualization(self, prob_mask: np.ndarray, binary_mask: np.ndarray, 
                           polygons: list, output_path: str, transform: rasterio.Affine) -> None:
        """Create side-by-side visualization."""
        print(f"📊 Creating visualization...")
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original probability mask
        axes[0].imshow(prob_mask, cmap='viridis')
        axes[0].set_title('Probability Mask')
        axes[0].axis('off')
        
        # Binary mask
        axes[1].imshow(binary_mask, cmap='gray')
        axes[1].set_title('Binary Mask')
        axes[1].axis('off')
        
        # Polygons overlay
        axes[2].imshow(prob_mask, cmap='gray', alpha=0.7)
        
        # Draw polygon boundaries
        for poly in polygons:
            if poly.exterior:
                coords = np.array(poly.exterior.coords)
                # Convert to pixel coordinates
                pixel_coords = []
                for x_geo, y_geo in coords:
                    col, row = ~transform * (x_geo, y_geo)
                    pixel_coords.append([col, row])
                
                if pixel_coords:
                    pixel_coords = np.array(pixel_coords)
                    axes[2].plot(pixel_coords[:, 0], pixel_coords[:, 1], 'r-', linewidth=1)
        
        axes[2].set_title(f'Extracted Polygons ({len(polygons)})')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   Visualization saved: {output_path}")
    
    def process(self, prob_mask_path: str, output_dir: str = "./polygon_results/") -> str:
        """Complete processing pipeline."""
        start_time = datetime.now()
        print("="*80)
        print("🏗️  BUILDING POLYGON EXTRACTION FROM PROBABILITY MASK")
        print("="*80)
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Load probability mask
        prob_mask, transform, crs = self.load_probability_mask(prob_mask_path)
        
        # Convert to binary
        binary_mask = self.adaptive_threshold(prob_mask)
        
        # Clean binary mask
        cleaned_mask = self.clean_binary_mask(binary_mask)
        
        # Extract polygons
        polygons = self.extract_polygons_rasterio(cleaned_mask, transform)
        
        # Refine polygons
        final_polygons = self.refine_polygons(polygons)
        
        # Generate output files
        mask_name = Path(prob_mask_path).stem
        polygon_output = output_dir / f"{mask_name}_buildings.geojson"
        vis_output = output_dir / f"{mask_name}_processing_steps.png"
        
        # Save results
        self.save_results(final_polygons, polygon_output, crs)
        self.create_visualization(prob_mask, cleaned_mask, final_polygons, vis_output, transform)
        
        duration = datetime.now() - start_time
        print(f"\n🎉 Completed in {str(duration).split('.')[0]}")
        print(f"📁 Polygons: {polygon_output}")
        print(f"📊 Visualization: {vis_output}")
        
        return str(polygon_output)


def main():
    parser = argparse.ArgumentParser(description='Extract building polygons from probability mask')
    parser.add_argument('--mask', required=True, help='Path to probability mask')
    parser.add_argument('--output', default='./polygon_results/', help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold')
    parser.add_argument('--min-area', type=int, default=50, help='Minimum area in pixels')
    parser.add_argument('--max-area', type=int, default=50000, help='Maximum area in pixels')
    parser.add_argument('--simplify', type=float, default=1.0, help='Simplification tolerance')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = BuildingPostProcessor(
        probability_threshold=args.threshold,
        min_area=args.min_area,
        max_area=args.max_area,
        simplify_tolerance=args.simplify
    )
    
    # Process
    try:
        result = processor.process(args.mask, args.output)
        print(f"\n✅ Success! Polygons saved to: {result}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()