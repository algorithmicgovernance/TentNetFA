from pathlib import Path

import rasterio
from rasterio.transform import from_bounds
from rasterio.merge import merge
from glob import glob
import numpy as np
from contextlib import ExitStack

from displacement_tracker.util.logging_config import setup_logging

LOGGER = setup_logging("tiff-predictions")

def save_prediction_tiff(probs, bounds, out_path):
    """
    Save a single-band GeoTIFF from a probability array.
    probs: 2D numpy array (H, W)
    bounds: dict with lat_min, lat_max, lon_min, lon_max (may be in any order)
    """
    height, width = probs.shape

    # Defensive: coerce to floats and make sure min/max ordering is correct
    lat_min = float(bounds.get("lat_min"))
    lat_max = float(bounds.get("lat_max"))
    lon_min = float(bounds.get("lon_min"))
    lon_max = float(bounds.get("lon_max"))

    left = min(lon_min, lon_max)
    right = max(lon_min, lon_max)
    bottom = min(lat_min, lat_max)
    top = max(lat_min, lat_max)

    transform = from_bounds(left, bottom, right, top, width, height)

    with rasterio.open(
        str(out_path),
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        compress="lzw",
        nodata=np.nan,
    ) as dst:
        dst.write(probs.astype(np.float32), 1)


def merge_prediction_tiffs(tiff_dir, out_path, dst_crs="EPSG:4326"):
    """
    Merge all *_pred.tif files in tiff_dir into a single georeferenced GeoTIFF.
    """
    tiff_paths = sorted(glob(str(Path(tiff_dir) / "*_pred.tif")))
    if not tiff_paths:
        LOGGER.warning("No prediction TIFFs found to merge.")
        return

    with ExitStack() as stack:
        src_files = [stack.enter_context(rasterio.open(p)) for p in tiff_paths]

        # Todo: Can we do better than a hard coded 9 tiles?
        mosaic, out_trans = merge(src_files, method="sum")
        mosaic = mosaic / 9
        mosaic = mosaic.astype(np.float32)

        out_meta = src_files[0].meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "crs": dst_crs,
                "count": mosaic.shape[0],
                "dtype": "float32",
            }
        )

        with rasterio.open(out_path, "w", **out_meta) as dest:
            dest.write(mosaic)

    LOGGER.info(f"Mosaic written to {out_path}")
