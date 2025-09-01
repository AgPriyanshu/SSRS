"""
Post-processing script to convert building probability masks to refined polygons.
Handles thresholding, noise removal, morphological operations, and polygon extraction.
"""

import numpy as np
import cv2
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape, Polygon
from shapely import simplify
from skimage import measure
from skimage.segmentation import clear_border
import argparse
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class BuildingPostProcessor:
    """Enhanced post-processing pipeline for building probability masks with precise boundary extraction."""
    
    def __init__(self, probability_threshold: float = 0.5, min_area_pixels: int = 50, 
                 max_area_pixels: int = 50000, simplify_tolerance: float = 1.0, 
                 use_watershed: bool = True, use_contours: bool = True):
        """Initialize post-processor.
        
        Args:
            probability_threshold: Threshold for converting probabilities to binary
            min_area_pixels: Minimum building area in pixels (for binary mask filtering)
            max_area_pixels: Maximum building area in pixels (for binary mask filtering)
            simplify_tolerance: Polygon simplification tolerance
            use_watershed: Whether to use watershed segmentation for better separation
            use_contours: Whether to use contour-based polygon extraction for cleaner boundaries
        """
        self.prob_threshold = probability_threshold
        self.min_area_pixels = min_area_pixels
        self.max_area_pixels = max_area_pixels
        self.simplify_tolerance = simplify_tolerance
        self.use_watershed = use_watershed
        self.use_contours = use_contours
        
        # Will be calculated based on transform when processing
        self.min_area_geo = None
        self.max_area_geo = None
        
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
        print("🎯 Applying adaptive thresholding...")
        
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
            if region.area < self.min_area_pixels or region.area > self.max_area_pixels:
                coords = region.coords
                cleaned[coords[:, 0], coords[:, 1]] = 0
        
        removed_pixels = int(np.sum(binary_mask.astype(np.int32))) - int(np.sum(cleaned.astype(np.int32)))
        print(f"   Removed {removed_pixels:,} pixels during cleaning")
        
        return cleaned
    
    def watershed_segmentation(self, prob_mask: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
        """Apply watershed segmentation to separate touching buildings."""
        print("🌊 Applying watershed segmentation for better building separation...")
        
        from scipy import ndimage
        from skimage.segmentation import watershed
        from skimage.feature import peak_local_maxima
        
        # Create distance transform
        distance = ndimage.distance_transform_edt(binary_mask)
        
        # Find local maxima as seeds
        min_distance = max(5, min(distance.shape) // 20)  # Adaptive minimum distance
        local_maxima = peak_local_maxima(distance, min_distance=min_distance, threshold_abs=0.3*distance.max())
        
        # Create markers
        markers = np.zeros_like(distance, dtype=np.int32)
        for i, (y, x) in enumerate(local_maxima):
            markers[y, x] = i + 1
        
        # Use probability mask as elevation map for watershed
        elevation = 1.0 - prob_mask  # Invert so buildings are valleys
        
        # Apply watershed
        labels = watershed(elevation, markers, mask=binary_mask)
        
        print(f"   Found {len(local_maxima)} building seeds")
        print(f"   Watershed created {np.max(labels)} segments")
        
        return labels
    
    def extract_contour_polygons(self, binary_mask: np.ndarray, transform: rasterio.Affine) -> list:
        """Extract polygons using contour detection for cleaner boundaries."""
        print("🗺️  Extracting polygons using contour detection...")
        
        # Find contours using OpenCV for better boundary precision
        contours, _ = cv2.findContours(binary_mask.astype(np.uint8), 
                                       cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        polygons = []
        
        for contour in contours:
            # Skip very small contours
            if len(contour) < 3:
                continue
                
            # Convert contour to polygon coordinates
            contour_coords = []
            for point in contour:
                x_pixel, y_pixel = point[0]
                # Convert pixel coordinates to geographic coordinates
                x_geo, y_geo = transform * (x_pixel, y_pixel)
                contour_coords.append([x_geo, y_geo])
            
            if len(contour_coords) >= 3:
                # Close the polygon if not already closed
                if contour_coords[0] != contour_coords[-1]:
                    contour_coords.append(contour_coords[0])
                
                try:
                    polygon = Polygon(contour_coords)
                    if polygon.is_valid and not polygon.is_empty:
                        polygons.append(polygon)
                except Exception:
                    continue
        
        print(f"   Extracted {len(polygons)} polygons from contours")
        return polygons
    
    def extract_watershed_polygons(self, labels: np.ndarray, transform: rasterio.Affine) -> list:
        """Extract polygons from watershed labels."""
        print("🗺️  Extracting polygons from watershed segments...")
        
        polygons = []
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels > 0]  # Skip background (0)
        
        for label_id in unique_labels:
            # Create binary mask for this label
            label_mask = (labels == label_id).astype(np.uint8)
            
            # Find contours for this label
            contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if len(contour) < 3:
                    continue
                    
                # Convert to geographic coordinates
                contour_coords = []
                for point in contour:
                    x_pixel, y_pixel = point[0]
                    x_geo, y_geo = transform * (x_pixel, y_pixel)
                    contour_coords.append([x_geo, y_geo])
                
                if len(contour_coords) >= 3:
                    # Close the polygon
                    if contour_coords[0] != contour_coords[-1]:
                        contour_coords.append(contour_coords[0])
                    
                    try:
                        polygon = Polygon(contour_coords)
                        if polygon.is_valid and not polygon.is_empty:
                            polygons.append(polygon)
                    except Exception:
                        continue
        
        print(f"   Extracted {len(polygons)} polygons from watershed")
        return polygons
    
    def extract_polygons_rasterio(self, binary_mask: np.ndarray, transform: rasterio.Affine) -> list:
        """Extract polygons using rasterio.features.shapes."""
        print("🗺️  Extracting polygons with rasterio...")
        
        # Calculate geographic area thresholds from pixel area
        if self.min_area_geo is None:
            # Get pixel size in each direction
            pixel_size_x = abs(transform.a)  # pixel width in map units
            pixel_size_y = abs(transform.e)  # pixel height in map units
            pixel_area_geo = pixel_size_x * pixel_size_y  # pixel area in geographic units
            
            # For very small pixels (geographic coordinates), use reasonable defaults
            if pixel_area_geo < 1e-10:
                print(f"   Warning: Very small pixel area ({pixel_area_geo:.2e}), using geographic coordinate mode")
                # For geographic coordinates, calculate reasonable area thresholds
                # Assume ~1 meter per pixel approximately and convert to degrees²
                approx_meter_per_degree = 111000  # rough approximation at equator
                pixel_size_approx_meters = 1.0  # assume 1m pixel resolution  
                pixel_size_degrees = pixel_size_approx_meters / approx_meter_per_degree
                pixel_area_degrees_sq = pixel_size_degrees * pixel_size_degrees
                
                self.min_area_geo = self.min_area_pixels * pixel_area_degrees_sq
                self.max_area_geo = self.max_area_pixels * pixel_area_degrees_sq
            else:
                self.min_area_geo = self.min_area_pixels * pixel_area_geo
                self.max_area_geo = self.max_area_pixels * pixel_area_geo
            
            print(f"   Transform: {transform}")
            print(f"   Pixel size X: {pixel_size_x:.8f}")
            print(f"   Pixel size Y: {pixel_size_y:.8f}")
            print(f"   Pixel area in geo units: {pixel_area_geo:.8f}")
            print(f"   Min area threshold: {self.min_area_geo:.8f} geo units²")
            print(f"   Max area threshold: {self.max_area_geo:.8f} geo units²")
        
        polygons = []
        total_extracted = 0
        
        # Extract shapes (polygons) from binary mask
        for geom, value in shapes(binary_mask.astype(np.uint8), mask=(binary_mask > 0), transform=transform):
            if value == 1:  # Building pixels
                polygon = shape(geom)
                total_extracted += 1
                if polygon.is_valid and self.min_area_geo <= polygon.area <= self.max_area_geo:
                    polygons.append(polygon)
        
        print(f"   Total shapes extracted: {total_extracted}")
        print(f"   Polygons after area filtering: {len(polygons)}")
        return polygons
    
    def refine_polygons(self, polygons: list) -> list:
        """Refine polygons with geometric operations."""
        print("✨ Refining polygons...")
        
        refined = []
        
        # Calculate appropriate buffer size based on coordinate system
        if self.min_area_geo is not None and self.min_area_geo < 1e-6:
            # Geographic coordinates - use very small buffer
            buffer_size = 1e-6  # ~0.1 meter in degrees
            simplify_tol = max(self.simplify_tolerance * 1e-6, 1e-7)
        else:
            # Projected coordinates - use original buffer
            buffer_size = 0.5
            simplify_tol = self.simplify_tolerance
        
        print(f"   Using buffer size: {buffer_size:.8f}")
        print(f"   Using simplify tolerance: {simplify_tol:.8f}")
        
        for poly in polygons:
            try:
                # Simplify polygon
                simplified = simplify(poly, tolerance=simplify_tol)
                
                # Small buffer to smooth edges (adjusted for coordinate system)
                buffered = simplified.buffer(buffer_size)
                unbuffered = buffered.buffer(-buffer_size)
                
                if unbuffered.area >= self.min_area_geo and unbuffered.is_valid:
                    if hasattr(unbuffered, 'geoms'):
                        # MultiPolygon - take largest part
                        largest = max(unbuffered.geoms, key=lambda x: x.area)
                        if largest.area >= self.min_area_geo:
                            refined.append(largest)
                    else:
                        refined.append(unbuffered)
                        
            except Exception:
                # Keep original if processing fails
                if poly.is_valid and poly.area >= self.min_area_geo:
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
        print("📊 Creating visualization...")
        
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
                    # Use inverse transform to convert from geographic to pixel coordinates
                    col, row = ~transform * (x_geo, y_geo)
                    pixel_coords.append([col, row])
                
                if pixel_coords:
                    pixel_coords = np.array(pixel_coords)
                    # Clip to image bounds for visualization
                    pixel_coords[:, 0] = np.clip(pixel_coords[:, 0], 0, prob_mask.shape[1]-1)
                    pixel_coords[:, 1] = np.clip(pixel_coords[:, 1], 0, prob_mask.shape[0]-1)
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
        
        # Extract polygons using enhanced methods
        if self.use_watershed and self.use_contours:
            print("🎯 Using watershed + contour method for precise boundaries...")
            # Apply watershed segmentation
            labels = self.watershed_segmentation(prob_mask, cleaned_mask)
            polygons = self.extract_watershed_polygons(labels, transform)
        elif self.use_contours:
            print("🎯 Using contour-based extraction for clean boundaries...")
            polygons = self.extract_contour_polygons(cleaned_mask, transform)
        else:
            print("🎯 Using standard rasterio extraction...")
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
    parser.add_argument('--mask', default='./results/Shahada_building_mask_probability.tif',  help='Path to probability mask')
    parser.add_argument('--output', default='./polygon_results/', help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold')
    parser.add_argument('--min-area', type=int, default=50, help='Minimum area in pixels')
    parser.add_argument('--max-area', type=int, default=50000, help='Maximum area in pixels')
    parser.add_argument('--simplify', type=float, default=1.0, help='Simplification tolerance')
    parser.add_argument('--watershed', action='store_true', default=True, help='Use watershed segmentation for better building separation')
    parser.add_argument('--contours', action='store_true', default=True, help='Use contour-based polygon extraction for cleaner boundaries')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = BuildingPostProcessor(
        probability_threshold=args.threshold,
        min_area_pixels=args.min_area,
        max_area_pixels=args.max_area,
        simplify_tolerance=args.simplify,
        use_watershed=args.watershed,
        use_contours=args.contours
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