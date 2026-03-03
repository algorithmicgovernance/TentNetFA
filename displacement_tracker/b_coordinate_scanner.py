from __future__ import annotations

import glob
import json
import math
import os
from collections import defaultdict
from datetime import datetime, date
from typing import Any
from tqdm import tqdm

import h5py
import numpy as np
import rasterio
import yaml
import fiona
import click
from rasterio.errors import RasterioIOError
from rasterio.warp import transform, transform_geom
from rasterio.transform import xy
from rasterio import mask
from rasterio.io import MemoryFile

from displacement_tracker.util.logging_config import setup_logging

LOGGER = setup_logging("coordinate_scanner")

# The original code enforced a single tile WIDTH and HEIGHT across runs using module globals.
# Keep that behaviour but encapsulate access through helpers so code is shorter and clearer.
WIDTH: int | None = None
HEIGHT: int | None = None


def _group_coords(
    features: list[dict[str, Any]], step: float
) -> dict[tuple[float, float], list[dict[str, Any]]]:
    grouped = defaultdict(list)
    for feat in features:
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates") or []
        if len(coords) != 2:
            continue
        lon, lat = coords
        base_lon = math.floor(lon / step) * step
        base_lat = math.floor(lat / step) * step
        # include 3x3 neighbourhood (original behaviour)
        for i in (-1, 0, 1):
            # Slightly dirty fix, easiest way to ensure all subtiles are labelled in the dataset
            # Previously, upper three subtiles had labels = 0, which caused underfitting
            for j in (-2, -1, 0, 1, 2):
                grouped[
                    (round(base_lon + i * step, 5), round(base_lat + j * step, 5))
                ].append(feat)
    return grouped


def _parse_date_safe(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def _filter_tents_by_target_date(
    features: list[dict[str, Any]], target: date
) -> list[dict[str, Any]]:
    """
    Keep features that match the TIFF date by either:
      - start == target, OR
      - start <= target <= end (when end exists)

    This removes the old 'end is None and within 30 days' fallback.
    """
    out: list[dict[str, Any]] = []
    for f in features:
        props = f.get("properties", {}) or {}
        start = _parse_date_safe(props.get("date_start"))
        end = _parse_date_safe(props.get("date_end"))

        if start is None:
            continue
        # primary match: start date equals TIFF date
        if start == target:
            out.append(f)
            continue
        # secondary match: feature interval covers TIFF date
        if end and start <= target <= end:
            out.append(f)
    return out


def _extract_date_from_filename(path: str) -> str | None:
    name = os.path.splitext(os.path.basename(path))[0]
    for part in name.split("_"):
        if part.isdigit() and len(part) == 8:
            return part
    return None


def _world_window(
    src: rasterio.io.DatasetReader,
    lon: float,
    lat: float,
    step: float,
    src_crs: Any,
    wgs84_crs: Any = "EPSG:4326",
):
    global WIDTH, HEIGHT
    try:
        # get 4 corner coordinates (paired properly)
        lons = [lon - step, lon - step, lon + 2 * step, lon + 2 * step]
        lats = [lat - step, lat + 2 * step, lat - step, lat + 2 * step]
        xs, ys = transform(wgs84_crs, src_crs, lons, lats)

        rows = []
        cols = []
        for x, y in zip(xs, ys):
            try:
                r, c = src.index(x, y)
                rows.append(int(r))
                cols.append(int(c))
            except Exception:
                continue

        if not rows or not cols:
            return None

        # clip to raster bounds first
        r0 = max(0, min(rows))
        r1 = min(src.height, max(rows))
        c0 = max(0, min(cols))
        c1 = min(src.width, max(cols))

        if r0 >= r1 or c0 >= c1:
            return None

        w = c1 - c0
        h = r1 - r0

        # --- fixed canonical tile size (height, width) ---
        WIDTH = 191
        HEIGHT = 224

        # ensure tile size never exceeds source dimensions
        tile_w = min(WIDTH, src.width)
        tile_h = min(HEIGHT, src.height)

        # enforce fixed size by expanding or shifting the clipped window
        if w != tile_w:
            desired_c1 = c0 + tile_w
            if desired_c1 <= src.width:
                c1 = desired_c1
            else:
                c0 = max(0, src.width - tile_w)
                c1 = c0 + tile_w
            w = c1 - c0

        if h != tile_h:
            desired_r1 = r0 + tile_h
            if desired_r1 <= src.height:
                r1 = desired_r1
            else:
                r0 = max(0, src.height - tile_h)
                r1 = r0 + tile_h
            h = r1 - r0

        if r0 >= r1 or c0 >= c1:
            return None

        return (r0, r1, c0, c1)
    except Exception:
        return None


def _read_rgb_and_grey(src: rasterio.io.DatasetReader, window):
    """
    Read RGB bands from src for the given window and return greyscale (float32) or None
    if the window contains no valid data.
    """
    data = src.read([1, 2, 3], window=((window[0], window[1]), (window[2], window[3])))
    if data.size == 0 or np.all(np.isnan(data)) or np.all(data == 0):
        return None
    r = data[0].astype(float)
    g = data[1].astype(float)
    b = data[2].astype(float)
    grey = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.float32)
    return grey


def _create_label_from_feats(
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    feats: list[dict[str, Any]],
    label_shape,
):
    label = np.zeros(label_shape, dtype=np.uint8)
    h, w = label.shape
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min
    if lon_span == 0 or lat_span == 0:
        return label

    for feat in feats:
        feat_lon, feat_lat = feat["geometry"]["coordinates"]
        # column: left->right
        local_col = int(round((feat_lon - lon_min) / lon_span * (w - 1)))
        # row: lat decreases downward; map lat_max to row 0
        local_row = int(round((lat_max - feat_lat) / lat_span * (h - 1)))

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = local_row + dr
                cc = local_col + dc
                if 0 <= rr < h and 0 <= cc < w:
                    label[rr, cc] = 255
    return label


def process_group(
    src: rasterio.io.DatasetReader,
    feats: list[dict[str, Any]],
    lon: float,
    lat: float,
    step: float,
    origin_image: str,
    origin_date: str,
    src_crs: Any,
    wgs84_crs: Any,
    prewar_src: rasterio.io.DatasetReader | None = None,
    min_valid_fraction: float = 0.0,  # << added param
) -> tuple[
    np.ndarray | None, np.ndarray | None, dict[str, Any] | None, np.ndarray | None
]:
    try:
        window = _world_window(src, lon, lat, step, src_crs, wgs84_crs)
        if not window:
            return None, None, None, None

        grey = _read_rgb_and_grey(src, window)
        if grey is None:
            return None, None, None, None

        # ----- new: drop tiles with too much missing data -----
        try:
            nodata = src.nodata
            # grey is single-band float32; use nodata if present, otherwise treat NaN and zero as missing
            if nodata is not None:
                valid_mask = (~np.isnan(grey)) & (grey != nodata)
            else:
                valid_mask = (~np.isnan(grey)) & (grey != 0)
            valid_fraction = np.count_nonzero(valid_mask) / grey.size
        except Exception:
            valid_fraction = 0.0

        if valid_fraction < float(min_valid_fraction):
            LOGGER.debug(
                "Skipping tile %s at lon=%s lat=%s due to low valid_fraction %.3f (< %.3f)",
                origin_image,
                lon,
                lat,
                valid_fraction,
                min_valid_fraction,
            )
            return None, None, None, None

        # window is (r0,r1,c0,c1) with r1/c1 exclusive-like indices from _world_window
        r0, r1, c0, c1 = window

        # compute geographic corners of the pixel window in source CRS, then transform to WGS84
        # use upper-left and lower-right pixel corners consistently
        x_ul, y_ul = xy(src.transform, r0, c0, offset="ul")
        x_lr, y_lr = xy(src.transform, r1 - 1, c1 - 1, offset="lr")

        lons, lats = transform(src.crs, "EPSG:4326", [x_ul, x_lr], [y_ul, y_lr])
        lon_min, lon_max = min(lons), max(lons)
        lat_min, lat_max = min(lats), max(lats)

        # create label using the exact window bounds (function expects lon_min, lat_min, lon_max, lat_max)
        label = _create_label_from_feats(
            lon_min, lat_min, lon_max, lat_max, feats, grey.shape
        )
        if grey.shape != label.shape:
            LOGGER.warning(
                "shape mismatch for %s: grey %s vs label %s; skipping tile",
                origin_image,
                grey.shape,
                label.shape,
            )
            return None, None, None, None

        meta = {
            "origin_image": origin_image,
            "origin_date": origin_date,
            "lon_min": lon_min,
            "lon_max": lon_max,
            "lat_min": lat_min,
            "lat_max": lat_max,
        }

        prewar_tile = None
        if prewar_src:
            try:
                # ensure canonical defaults if _world_window hasn't set module globals
                global WIDTH, HEIGHT
                if WIDTH is None:
                    WIDTH = 191
                if HEIGHT is None:
                    HEIGHT = 224

                # map the same WGS84 bbox into prewar CRS and index there
                pxs, pys = transform(
                    "EPSG:4326", prewar_src.crs, [lon_min, lon_max], [lat_min, lat_max]
                )
                rpre_a, cpre_a = prewar_src.index(pxs[0], pys[0])
                rpre_b, cpre_b = prewar_src.index(pxs[1], pys[1])
                rpre0, rpre1 = sorted([int(rpre_a), int(rpre_b)])
                cpre0, cpre1 = sorted([int(cpre_a), int(cpre_b)])
                # clip
                rpre0 = max(0, rpre0)
                rpre1 = min(prewar_src.height, rpre1)
                cpre0 = max(0, cpre0)
                cpre1 = min(prewar_src.width, cpre1)
                if rpre0 < rpre1 and cpre0 < cpre1:
                    # enforce the same fixed canonical tile size for prewar tiles
                    w = cpre1 - cpre0
                    h = rpre1 - rpre0
                    tile_w = min(WIDTH, prewar_src.width)
                    tile_h = min(HEIGHT, prewar_src.height)

                    # align/expand horizontally
                    if w != tile_w:
                        desired_c1 = cpre0 + tile_w
                        if desired_c1 <= prewar_src.width:
                            cpre1 = desired_c1
                        else:
                            cpre0 = max(0, prewar_src.width - tile_w)
                            cpre1 = cpre0 + tile_w
                        w = cpre1 - cpre0

                    # align/expand vertically
                    if h != tile_h:
                        desired_r1 = rpre0 + tile_h
                        if desired_r1 <= prewar_src.height:
                            rpre1 = desired_r1
                        else:
                            rpre0 = max(0, prewar_src.height - tile_h)
                            rpre1 = rpre0 + tile_h
                        h = rpre1 - rpre0

                    if rpre0 < rpre1 and cpre0 < cpre1:
                        pre_data = prewar_src.read(
                            1, window=((rpre0, rpre1), (cpre0, cpre1))
                        )
                        prewar_tile = pre_data.astype(np.float32)
            except Exception:
                LOGGER.exception("Failed to build prewar tile for %s", origin_image)

        return grey, label, meta, prewar_tile
    except Exception:
        LOGGER.exception("Failed to process group")
        return None, None, None, None


def is_high_quality_tile(
    feats: list[dict[str, Any]],
    date_target_str: str,
    src: rasterio.io.DatasetReader,
    lon: float,
    lat: float,
    step: float,
    start_threshold: float,
    max_missing_end: float,
    min_valid_fraction: float,
) -> bool:
    """
    Checks date distributions and raster valid-pixel fraction for the tile.
    Mirrors original logic but uses shared helpers to keep the function concise.
    """
    if not feats or not date_target_str:
        return False
    try:
        date_target = datetime.strptime(date_target_str, "%Y%m%d").date()
    except Exception:
        return False

    start_matches = sum(
        _parse_date_safe(f.get("properties", {}).get("date_start")) == date_target
        for f in feats
    )
    missing_end = sum(
        _parse_date_safe(f.get("properties", {}).get("date_end")) is None for f in feats
    )
    n = len(feats)
    if n == 0:
        return False
    if (start_matches / n) < start_threshold or (missing_end / n) > max_missing_end:
        return False

    window = _world_window(src, lon, lat, step, src.crs, "EPSG:4326")
    if not window:
        return False
    try:
        data = src.read(
            [1, 2, 3], window=((window[0], window[1]), (window[2], window[3]))
        )
        if data.size == 0:
            return False
        nodata = src.nodata
        # data shape is (bands, h, w); compare element-wise against nodata if available
        if nodata is not None:
            valid_mask = (~np.isnan(data)) & (data != nodata)
        else:
            valid_mask = (~np.isnan(data)) & (data != 0)
        valid_fraction = np.count_nonzero(valid_mask) / data.size
        return valid_fraction >= min_valid_fraction
    except Exception:
        return False


class HDF5Writer:
    """
    Writes tiles by appending into extensible chunked datasets:
      - feature : (N, H, W) dtype=float32 (or grey.dtype)
      - label   : (N, H, W) dtype=uint8
      - prewar  : (N, Hp, Wp) optional
      - meta    : (N,) variable-length JSON strings

    Datasets are created lazily on the first add_entry (so we know tile shape).
    After each append we call flush() to push data to disk and release internal buffers.
    """

    def __init__(self, path: str):
        self.path = path
        # keep file open for entire run. h5py keeps some metadata in memory; that is OK.
        self._file = h5py.File(self.path, "w")
        self._created = False
        self.tile_idx = 0
        # placeholders for datasets
        self._d_feature = None
        self._d_label = None
        self._d_prewar = None
        self._d_meta = None

    def _create_datasets(
        self, grey: np.ndarray, label: np.ndarray, prewar: np.ndarray | None
    ):
        h, w = grey.shape
        # chunk 1 tile per chunk so appends are efficient
        chunks = (1, h, w)
        maxshape = (None, h, w)
        # dtype preservation
        feat_dtype = grey.dtype
        label_dtype = label.dtype

        self._d_feature = self._file.create_dataset(
            "feature",
            shape=(0, h, w),
            maxshape=maxshape,
            chunks=chunks,
            dtype=feat_dtype,
            compression="gzip",
        )
        self._d_label = self._file.create_dataset(
            "label",
            shape=(0, h, w),
            maxshape=maxshape,
            chunks=chunks,
            dtype=label_dtype,
            compression="gzip",
        )

        # Always create prewar dataset using canonical tile size
        global WIDTH, HEIGHT
        if WIDTH is None:
            WIDTH = 191
        if HEIGHT is None:
            HEIGHT = 224

        self._d_prewar = self._file.create_dataset(
            "prewar",
            shape=(0, HEIGHT, WIDTH),
            maxshape=(None, HEIGHT, WIDTH),
            chunks=(1, HEIGHT, WIDTH),
            dtype=np.float32,
            compression="gzip",
        )

        # metadata: store as variable-length JSON strings
        dt = h5py.special_dtype(vlen=str)
        self._d_meta = self._file.create_dataset(
            "meta", shape=(0,), maxshape=(None,), dtype=dt, chunks=(1,)
        )
        self._created = True

    def add_entry(
        self,
        grey: np.ndarray,
        label: np.ndarray,
        meta: dict[str, Any],
        prewar: np.ndarray | None = None,
    ):
        # lazy create on first tile
        if not self._created:
            self._create_datasets(grey, label, prewar)

        # append index
        idx = self._d_feature.shape[0]

        # resize then write each dataset
        self._d_feature.resize(idx + 1, axis=0)
        self._d_feature[idx, :, :] = grey

        self._d_label.resize(idx + 1, axis=0)
        self._d_label[idx, :, :] = label

        # handle prewar dataset (may exist or not)
        if self._d_prewar is not None:
            # always resize first so dataset length matches other datasets
            self._d_prewar.resize(idx + 1, axis=0)

            if prewar is None:
                # write an all-zero tile
                zero_tile = np.zeros(
                    self._d_prewar.shape[1:], dtype=self._d_prewar.dtype
                )
                self._d_prewar[idx, :, :] = zero_tile
            else:
                ph, pw = prewar.shape
                target_h, target_w = self._d_prewar.shape[1], self._d_prewar.shape[2]
                if (ph, pw) != (target_h, target_w):
                    # pad/crop to target shape (top-left aligned). Change alignment if desired.
                    adjusted = np.zeros(
                        (target_h, target_w), dtype=self._d_prewar.dtype
                    )
                    copy_h = min(ph, target_h)
                    copy_w = min(pw, target_w)
                    # cast source to dataset dtype before copy to avoid unexpected dtype promotion
                    adjusted[:copy_h, :copy_w] = prewar[:copy_h, :copy_w].astype(
                        self._d_prewar.dtype, copy=False
                    )
                    prewar_to_write = adjusted
                else:
                    # shapes match — ensure dtype matches dataset
                    prewar_to_write = prewar.astype(self._d_prewar.dtype, copy=False)

                self._d_prewar[idx, :, :] = prewar_to_write

        # store metadata as a compact JSON string
        meta_json = json.dumps(meta, ensure_ascii=False)
        self._d_meta.resize(idx + 1, axis=0)
        self._d_meta[idx] = meta_json

        self.tile_idx += 1
        # flush to push buffers to disk and reduce memory usage
        try:
            self._file.flush()
        except Exception:
            # non-fatal: continue running; flush best-effort
            pass

    def write(self):
        self._file.close()


def _open_raster(path: str):
    try:
        return rasterio.open(path)
    except RasterioIOError:
        LOGGER.exception(f"Error opening GeoTIFF: {path}")
        return None


def _crop_src_to_boundaries(
    src: rasterio.io.DatasetReader, boundaries_path: str
) -> rasterio.io.DatasetReader | None:
    try:
        with fiona.open(boundaries_path, "r") as shp:
            shp_geoms = [feat["geometry"] for feat in shp]
            shp_crs = shp.crs
    except Exception:
        LOGGER.exception(
            "Failed to read boundaries shapefile; proceeding without cropping"
        )
        return src

    transformed = []
    for g in shp_geoms:
        try:
            transformed.append(transform_geom(shp_crs, src.crs, g))
        except Exception:
            try:
                transformed.append(transform_geom("EPSG:4326", src.crs, g))
            except Exception:
                LOGGER.exception("Failed to transform a boundary geometry; skipping it")
    if not transformed:
        LOGGER.warning(
            "No valid transformed geometries found in boundaries; skipping cropping."
        )
        return src
    try:
        out_image, out_transform = mask.mask(src, transformed, crop=True)
    except ValueError:
        LOGGER.info(
            f"No overlap between {os.path.basename(src.name)} and boundaries; skipping file."
        )
        src.close()
        return None
    except Exception:
        LOGGER.exception(
            "Unexpected error while cropping raster; proceeding with original raster"
        )
        return src

    out_meta = src.meta.copy()
    out_meta.update(
        {
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
        }
    )
    mem = MemoryFile()
    with mem.open(**out_meta) as ds:
        ds.write(out_image)
    src.close()
    new_src = mem.open()
    new_src._memfile = mem

    orig_close = new_src.close

    def _close():
        try:
            orig_close()
        finally:
            mem.close()

    LOGGER.info(
        f"Cropped {os.path.basename(new_src.name)} to provided boundaries; new bounds: {new_src.bounds}"
    )
    new_src.close = _close
    return new_src


def scan_grouped_coordinates(
    geotiff_path: str,
    geojson_path: str,
    hdf5_writer: HDF5Writer,
    quality_thresholds: dict[str, Any],
    step: float,
    date_target: str | None,
    prewar_path: str | None = None,
    boundaries_path: str | None = None,
    complete_list: list[str] | None = None,
) -> None:
    src = _open_raster(geotiff_path)
    if src is None:
        return

    if boundaries_path:
        cropped = _crop_src_to_boundaries(src, boundaries_path)
        if cropped is None:
            return
        src = cropped

    prewar_src = _open_raster(prewar_path) if prewar_path else None

    try:
        with open(geojson_path, "r") as f:
            features = json.load(f).get("features", [])
    except Exception:
        LOGGER.exception(f"Failed to read geojson: {geojson_path}")
        src.close()
        if prewar_src:
            prewar_src.close()
        return

    base_name = os.path.basename(geotiff_path)
    is_complete = base_name in (complete_list or [])

    # minimal valid-fraction threshold (0.0 preserves old behaviour)
    min_valid = (
        quality_thresholds.get("min_valid_fraction", 0.0)
        if isinstance(quality_thresholds, dict)
        else 0.0
    )

    # always ensure tents match the TIFF date using start (and interval when end exists)
    if date_target:
        try:
            date_obj = datetime.strptime(date_target, "%Y%m%d").date()
            features = _filter_tents_by_target_date(features, date_obj)
            LOGGER.info(
                f"[{base_name}] Filtered to {len(features)} features for date {date_target}."
            )
        except Exception:
            LOGGER.exception(f"Date parsing error for {geotiff_path}")
            src.close()
            if prewar_src:
                prewar_src.close()
            return

    # If the TIFF is marked COMPLETE, process the entire raster grid (including tiles with no tents)
    grouped = _group_coords(features, step)

    # Ensure tiffs intended for prediction only are always fully processed
    prediction_only = params.get("processing", {}).get("prediction_only", False)
    if prediction_only:
        is_complete = True

    if is_complete:
        LOGGER.info(
            f"[{base_name}] marked as COMPLETE - quality gating disabled; scanning entire raster grid."
        )
        bounds = src.bounds
        lon_bounds, lat_bounds = transform(
            src.crs,
            "EPSG:4326",
            [bounds.left, bounds.right],
            [bounds.bottom, bounds.top],
        )
        LOGGER.info(
            f"[{os.path.basename(geotiff_path)}] Raster bounds (wgs84): {lon_bounds[0]}..{lon_bounds[1]} x {lat_bounds[0]}..{lat_bounds[1]}"
        )

        processed_count = 0
        high_quality_found = False

        lon_iter = np.arange(lon_bounds[0], lon_bounds[1], step)
        lat_iter = np.arange(lat_bounds[0], lat_bounds[1], step)
        # iterate in a deterministic order
        for lon in tqdm(lon_iter, desc=f"{base_name} lon"):
            for lat in lat_iter:
                # lookup features for this tile (keys were rounded to 5 decimals in _group_coords)
                base_lon = round(math.floor(lon / step) * step, 5)
                base_lat = round(math.floor(lat / step) * step, 5)
                key = (base_lon, base_lat)
                feats_for_tile = grouped.get(key, [])

                grey, label, meta, prewar_tile = process_group(
                    src,
                    feats_for_tile,
                    lon,
                    lat,
                    step,
                    base_name,
                    date_target or "",
                    src.crs,
                    "EPSG:4326",
                    prewar_src,
                    min_valid_fraction=min_valid,
                )

                if (
                        grey is not None
                        and label is not None
                        and meta is not None
                        and prewar_tile is not None
                ):
                    hdf5_writer.add_entry(grey, label, meta, prewar_tile)
                    processed_count += 1
                    high_quality_found = True

        LOGGER.info(f"[{base_name}] wrote {processed_count} tiles (complete=True).")
        if not high_quality_found:
            LOGGER.warning(f"No valid tiles written for COMPLETE TIFF {base_name}")

        src.close()
        if prewar_src:
            prewar_src.close()
        return

    # Incomplete TIFF: group by tent locations and apply quality gating
    grouped = _group_coords(features, step)
    LOGGER.info(
        f"[{base_name}] marked as INCOMPLETE - applying quality gating from YAML."
    )
    LOGGER.info(
        f"[{os.path.basename(geotiff_path)}] Found {len(grouped)} coordinate groups with tents."
    )
    LOGGER.info(
        f"GeoTIFF bounds: min_lon={src.bounds.left}, min_lat={src.bounds.bottom}, max_lon={src.bounds.right}, max_lat={src.bounds.top}"
    )

    high_quality_found = False
    processed_count = 0

    for (lon, lat), feats in tqdm(
        grouped.items(), desc=f"{base_name} processing tiles"
    ):
        # incomplete TIFF: apply quality filter
        if not is_high_quality_tile(
            feats, date_target, src, lon, lat, step, **quality_thresholds
        ):
            continue
        grey, label, meta, prewar_tile = process_group(
            src,
            feats,
            lon,
            lat,
            step,
            base_name,
            date_target or "",
            src.crs,
            "EPSG:4326",
            prewar_src,
            min_valid_fraction=min_valid,
        )
        if (
                grey is not None
                and label is not None
                and meta is not None
                and prewar_tile is not None
        ):
            hdf5_writer.add_entry(grey, label, meta, prewar_tile)
            high_quality_found = True
            processed_count += 1

    LOGGER.info(f"[{base_name}] wrote {processed_count} tiles (complete=False).")
    if not high_quality_found:
        LOGGER.warning(f"No valid high-quality tiles found in {base_name}")

    src.close()
    if prewar_src:
        prewar_src.close()


def scan_all_coordinates(
    geotiff_path: str,
    hdf5_writer: HDF5Writer,
    date_target: str | None,
    step: float = 0.001,
    prewar_path: str | None = None,
    min_valid_fraction: float = 0.0,
):
    src = _open_raster(geotiff_path)
    if src is None:
        return

    bounds = src.bounds
    lon_bounds, lat_bounds = transform(
        src.crs, "EPSG:4326", [bounds.left, bounds.right], [bounds.bottom, bounds.top]
    )
    LOGGER.info(
        f"GeoTIFF bounds: min_lon={lon_bounds[0]}, min_lat={lat_bounds[0]}, max_lon={lon_bounds[1]}, max_lat={lat_bounds[1]}"
    )

    prewar_src = _open_raster(prewar_path) if prewar_path else None

    lon_iter = np.arange(lon_bounds[0], lon_bounds[1], step)
    lat_iter = np.arange(lat_bounds[0], lat_bounds[1], step)
    for lon in lon_iter:
        for lat in lat_iter:
            grey, label, meta, prewar_img = process_group(
                src,
                [],
                lon,
                lat,
                step,
                os.path.basename(geotiff_path),
                date_target or "",
                src.crs,
                "EPSG:4326",
                prewar_src,
                min_valid_fraction=min_valid_fraction,
            )
            if grey is not None and label is not None and meta is not None:
                hdf5_writer.add_entry(grey, label, meta, prewar_img)

    src.close()
    if prewar_src:
        prewar_src.close()


@click.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
def cli(config):
    with open(config, "r") as f:
        params = yaml.safe_load(f)
    for key in ("geotiff_dir", "hdf5", "processing"):
        if key not in params:
            raise click.ClickException(f"Missing required config key: {key}")

    proc = params["processing"]
    step = proc["step"]
    quality_thresholds = proc["quality_thresholds"]
    complete_list = proc.get("complete", [])  # exact filenames listed in YAML

    coordinate_scanner(
        params["geotiff_dir"],
        params.get("geojson"),
        params["hdf5"],
        step,
        quality_thresholds,
        prewar_path=params.get("prewar_gaza"),
        boundaries_path=params.get("boundaries"),
        complete_list=complete_list,
        config=config,
    )


def coordinate_scanner(
    geotiff_dir: str,
    geojson: str | None,
    hdf5: str,
    step: float,
    quality_thresholds: dict[str, Any],
    prewar_path: str | None = None,
    boundaries_path: str | None = None,
    complete_list: list[str] | None = None,
    config: str | None = None,
) -> None:
    if config:
        with open(config, "r") as f:
            cfg = yaml.safe_load(f)
        search_files = cfg.get("loading", {}).get("files", [])
        tif_files = [
            os.path.join(geotiff_dir, f)
            for f in search_files
            if os.path.exists(os.path.join(geotiff_dir, f))
        ]
    else:
        tif_files = glob.glob(os.path.join(geotiff_dir, "*.tif"))

    if not tif_files:
        LOGGER.error(f"No .tif files found in {geotiff_dir}")
        return

    with rasterio.Env(GDAL_CACHEMAX=256):
        for tif_path in tif_files:
            date_target = _extract_date_from_filename(tif_path)
            if not date_target:
                LOGGER.warning(f"Skipping {tif_path} (no date found in filename).")
                continue

            base = os.path.splitext(os.path.basename(tif_path))[0]
            out_h5 = os.path.join(os.path.dirname(hdf5), f"{base}.h5")

            LOGGER.info(f"Processing {tif_path} → {out_h5}")

            writer = HDF5Writer(out_h5)

            if geojson:
                scan_grouped_coordinates(
                    tif_path,
                    geojson,
                    writer,
                    quality_thresholds,
                    step,
                    date_target,
                    prewar_path,
                    boundaries_path,
                    complete_list or [],
                )
            else:
                scan_all_coordinates(tif_path, writer, date_target, step, prewar_path)

            writer.write()
            LOGGER.info(f"Saved dataset to {out_h5}")


if __name__ == "__main__":
    cli()

# EXAMPLE CLI USAGE:
# poetry run coordinate_scanner config.yaml
