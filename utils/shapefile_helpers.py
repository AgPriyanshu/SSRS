from typing import Optional, Tuple, List
import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.transform import from_bounds
from rasterio.windows import Window
import os
from pathlib import Path

from constants import DATASET_DIR


def save_mask_as_tif(
    mask: np.ndarray,
    output_path: str,
    transform: rasterio.Affine,
    crs: rasterio.crs.CRS,
    compress: str = 'lzw'
) -> None:
    """
    Save a binary mask as a georeferenced TIF file.
    
    Args:
        mask: Binary mask array (0s and 1s)
        output_path: Path where to save the TIF file
        transform: Rasterio affine transform for georeferencing
        crs: Coordinate reference system
        compress: Compression method ('lzw', 'deflate', 'none')
    """
    try:
        print(f"Saving mask to: {output_path}")
        
        # Create output directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write the mask as a GeoTIFF
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=mask.shape[0],
            width=mask.shape[1],
            count=1,
            dtype=mask.dtype,
            crs=crs,
            transform=transform,
            compress=compress,
            tiled=True,
            blockxsize=512,
            blockysize=512
        ) as dst:
            dst.write(mask, 1)
            dst.set_band_description(1, "Building binary mask (1=building, 0=background)")
        
        print(f"Mask saved successfully as: {output_path}")
        
    except Exception as e:
        print(f"Error saving mask as TIF: {e}")
        raise


def create_binary_mask_from_shapefile(
    shapefile_path: str,
    reference_raster_path: Optional[str] = None,
    output_shape: Optional[Tuple[int, int]] = None,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    pixel_size: Optional[float] = None,
    crs: Optional[str] = None,
    output_tif_path: Optional[str] = None
) -> Tuple[np.ndarray, rasterio.Affine, rasterio.crs.CRS]:
    """
    Create a binary mask from building polygons in a shapefile.
    
    Args:
        shapefile_path: Path to the shapefile containing building polygons
        reference_raster_path: Path to a reference raster to match spatial properties
        output_shape: Tuple of (height, width) for output mask dimensions
        bounds: Tuple of (left, bottom, right, top) spatial bounds
        pixel_size: Pixel size in the same units as the CRS
        crs: Coordinate reference system (e.g., 'EPSG:4326')
        output_tif_path: Optional path to save the mask as a georeferenced TIF file
        
    Returns:
        Tuple containing:
        - np.ndarray: Binary mask where 1 = inside building polygon, 0 = outside
        - rasterio.Affine: Georeferencing transform
        - rasterio.crs.CRS: Coordinate reference system
        
    Note:
        Either provide reference_raster_path OR (output_shape, bounds, pixel_size, crs)
    """
    try:
        # Read the shapefile
        print(f"Reading shapefile: {shapefile_path}")
        gdf = gpd.read_file(shapefile_path)
        
        if gdf.empty:
            raise ValueError("Shapefile contains no geometries")
        
        print(f"Loaded {len(gdf)} polygons from shapefile")
        
        # Option 1: Use reference raster for spatial properties
        if reference_raster_path:
            print(f"Using reference raster: {reference_raster_path}")
            with rasterio.open(reference_raster_path) as ref_raster:
                transform = ref_raster.transform
                width = ref_raster.width
                height = ref_raster.height
                ref_crs = ref_raster.crs
                
        # Option 2: Use provided parameters
        elif all([output_shape, bounds, pixel_size, crs]):
            print("Using provided spatial parameters")
            height, width = output_shape
            left, bottom, right, top = bounds
            transform = from_bounds(left, bottom, right, top, width, height)
            ref_crs = rasterio.crs.CRS.from_string(crs)
            
        else:
            raise ValueError(
                "Either provide reference_raster_path OR "
                "(output_shape, bounds, pixel_size, crs)"
            )
        
        # Reproject shapefile to match reference CRS if needed
        if gdf.crs != ref_crs:
            print(f"Reprojecting from {gdf.crs} to {ref_crs}")
            gdf = gdf.to_crs(ref_crs)
        
        # Create the binary mask
        print(f"Creating binary mask of size {height}x{width}")
        
        # Extract polygon geometries
        shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]
        
        if not shapes:
            print("Warning: No valid geometries found")
            return np.zeros((height, width), dtype=np.uint8)
        
        # Rasterize the polygons
        mask = features.rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,  # Background value
            dtype=np.uint8
        )
        
        # Count the number of pixels inside buildings
        building_pixels = np.sum(mask == 1)
        total_pixels = mask.size
        coverage_percent = (building_pixels / total_pixels) * 100
        
        print("Mask created successfully:")
        print(f"  - Shape: {mask.shape}")
        print(f"  - Building pixels: {building_pixels:,}")
        print(f"  - Total pixels: {total_pixels:,}")
        print(f"  - Building coverage: {coverage_percent:.2f}%")
        
        # Optionally save as TIF file
        if output_tif_path:
            save_mask_as_tif(mask, output_tif_path, transform, ref_crs)
        
        return mask, transform, ref_crs
        
    except Exception as e:
        print(f"Error creating binary mask: {e}")
        raise

if __name__ == "__main__":
    # Example 1: Create mask and save as TIF in one step
    mask, transform, crs = create_binary_mask_from_shapefile(
        shapefile_path=DATASET_DIR / "buildings_cropped.shp",
        reference_raster_path=DATASET_DIR / "ortho_cog_cropped.tif",
        output_tif_path=DATASET_DIR / "building_mask.tif"
    )
    print(f"Mask shape: {mask.shape}")
    
    