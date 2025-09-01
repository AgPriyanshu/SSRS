from typing import Optional, Tuple
import os
from pathlib import Path
import math
import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from rasterio.transform import from_bounds
from shapely.geometry import box
from tqdm import tqdm

def save_mask_as_tif_header(
    output_path: str,
    height: int,
    width: int,
    transform: rasterio.Affine,
    crs: rasterio.crs.CRS,
    compress: str = "ZSTD",
    blocksize: int = 512,
):
    os.makedirs(os.path.dirname(str(output_path)), exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",  # Keep uint8 for rasterio compatibility
        "crs": crs,
        "transform": transform,
        "tiled": True,
        "blockxsize": blocksize,
        "blockysize": blocksize,
        "compress": compress,
        "predictor": 2,
        "BIGTIFF": "IF_NEEDED",
        "nodata": 255,  # Use 255 as nodata (outside valid class range 0-1)
    }
    return rasterio.open(output_path, "w", **profile)

def create_binary_mask_streaming(
    shapefile_path: str,
    reference_raster_path: Optional[str] = None,
    output_tif_path: Optional[str] = None,
    # If you don’t have a reference raster, provide these:
    output_shape: Optional[Tuple[int, int]] = None,  # (height, width)
    bounds: Optional[Tuple[float, float, float, float]] = None,  # (left, bottom, right, top)
    pixel_size: Optional[float] = None,
    crs: Optional[str] = None,
    # Streaming params:
    tile: int = 8192,             # tile/window size in pixels (safe for 22GB RAM)
    halo: int = 32,               # overlap to avoid seam artifacts
    all_touched: bool = True,     # set False for stricter boundaries
    clip_geoms: bool = True,      # clip polygons to tile bounds to reduce rasterize work
    simplify: bool = True,        # simplify polygons to ~pixel size
    compress: str = "ZSTD",
) -> None:
    """
    Stream-rasterize polygons to a binary mask aligned to reference_raster_path,
    using small memory. Writes directly to GeoTIFF on disk.
    
    Output mask values:
    - 255: nodata (invalid/missing pixels)  
    - 0: background (valid non-building pixels)
    - 1: building (valid building pixels)
    
    Uses uint8 dtype with nodata=255 to avoid conflict with valid class labels (0,1).
    """
    shp_path = Path(shapefile_path)
    if not shp_path.exists():
        raise FileNotFoundError(f"Shapefile not found: {shp_path}")

    print(f"Reading polygons: {shp_path}")
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError("Shapefile contains no geometries")

    # Determine target grid (from reference raster OR provided params)
    if reference_raster_path:
        with rasterio.open(reference_raster_path) as ref:
            ref_transform = ref.transform
            ref_crs = ref.crs
            width, height = ref.width, ref.height
            target_bounds = rasterio.windows.bounds(Window(0, 0, width, height), ref_transform)
            px_size_x = ref_transform.a
            # px_size_y = -ref_transform.e  # not used; grid square assumed
    else:
        if not (output_shape and bounds and pixel_size and crs):
            raise ValueError("Provide reference_raster_path OR (output_shape, bounds, pixel_size, crs).")
        height, width = output_shape
        left, bottom, right, top = bounds
        ref_transform = from_bounds(left, bottom, right, top, width, height)
        ref_crs = rasterio.crs.CRS.from_string(crs)
        target_bounds = (left, bottom, right, top)
        px_size_x = pixel_size

    # Reproject polygons if needed
    if gdf.crs is None:
        raise ValueError("Input shapefile has no CRS. Set its CRS first.")
    if gdf.crs != ref_crs:
        print(f"Reprojecting polygons {gdf.crs} → {ref_crs}")
        gdf = gdf.to_crs(ref_crs)

    # Optional: simplify to ~pixel size for speed (keeps topology)
    if simplify:
        tol = max(px_size_x * 0.5, 0.0)
        print(f"Simplifying geometries with tolerance ~ {tol:.6f} (≈ half pixel)")
        gdf["geometry"] = gdf.geometry.simplify(tol, preserve_topology=True)

    # Keep only polygons that intersect the target extent
    target_poly = box(*target_bounds)
    gdf = gdf[gdf.intersects(target_poly)]
    if gdf.empty:
        raise ValueError("No polygons intersect the target raster extent.")

    # Build spatial index
    print("Building spatial index…")
    sindex = gdf.sindex  # rtree/pygeos index

    # Prepare output
    assert output_tif_path, "output_tif_path is required for streaming write."
    print(f"Writing mask to: {output_tif_path}")
    with save_mask_as_tif_header(output_tif_path, height, width, ref_transform, ref_crs, compress=compress) as dst:

        tiles_x = math.ceil(width / tile)
        tiles_y = math.ceil(height / tile)
        total_tiles = tiles_x * tiles_y

        with tqdm(total=total_tiles, unit="tile") as pbar:
            for y0 in range(0, height, tile):
                for x0 in range(0, width, tile):
                    # Define write window (no halo)
                    w = min(tile, width - x0)
                    h = min(tile, height - y0)
                    win_write = Window(x0, y0, w, h)

                    # Define read window with halo (expanded bounds for clipping)
                    x_read = max(0, x0 - halo)
                    y_read = max(0, y0 - halo)
                    w_read = min(width - x_read, tile + 2 * halo)
                    h_read = min(height - y_read, tile + 2 * halo)
                    win_read = Window(x_read, y_read, w_read, h_read)

                    # Bounds in target CRS for the read window
                    read_bounds = rasterio.windows.bounds(win_read, ref_transform)
                    read_box = box(*read_bounds)

                    # Query sindex for candidates
                    candidate_idx = list(sindex.intersection(read_bounds))
                    if not candidate_idx:
                        # No polygons → write zeros quickly
                        dst.write(np.zeros((h, w), dtype=np.uint8), 1, window=win_write)
                        pbar.update(1)
                        continue

                    cand = gdf.iloc[candidate_idx]
                    # Optionally clip to window bounds for speed
                    if clip_geoms:
                        cand = gpd.clip(cand, read_box)

                    # Filter empty geometries
                    cand = cand[~cand.geometry.is_empty & cand.geometry.notnull()]
                    if cand.empty:
                        dst.write(np.zeros((h, w), dtype=np.uint8), 1, window=win_write)
                        pbar.update(1)
                        continue

                    # Rasterize on the read window grid, then crop halo back to write window
                    win_tfm = window_transform(win_read, ref_transform)
                    out_arr = features.rasterize(
                        shapes=((geom, 1) for geom in cand.geometry),
                        out_shape=(h_read, w_read),
                        transform=win_tfm,
                        fill=0,
                        all_touched=all_touched,
                        dtype="uint8",  # Back to uint8 for rasterio compatibility
                    )

                    # Crop from read window (with halo) down to write window area
                    y_in = y0 - y_read
                    x_in = x0 - x_read
                    tile_arr = out_arr[y_in:y_in + h, x_in:x_in + w]

                    dst.write(tile_arr, 1, window=win_write)
                    pbar.update(1)

    print("Done. (Tip: build overviews for snappy QGIS browsing)")
    print(f"gdaladdo -r nearest {output_tif_path} 2 4 8 16")

if __name__ == "__main__":
    # Example usage: align exactly to a huge reference ortho
    create_binary_mask_streaming(
        shapefile_path="../data/Aarvi/Building_shp/polygons.shp",
        reference_raster_path="../data/Aarvi/ortho.tif",
        output_tif_path="../data/Aarvi/building_mask.tif",
        tile=8192,         # safe for 22 GB RAM (≈64 MB/tile for uint8)
        halo=32,           # robust seams
        all_touched=True,  # or False for stricter edges
        clip_geoms=True,
        simplify=True,     # big speedup; disable if you need pixel-perfect edges
        compress="ZSTD",   # or "LZW"
    )
