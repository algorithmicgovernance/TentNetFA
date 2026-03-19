import yaml
import click
import torch
import json
from pathlib import Path
from scipy.ndimage import label, center_of_mass, gaussian_filter
import geopandas as gpd
from shapely.geometry import Point
from torch.utils.data import DataLoader
import gc
import os
import psutil
import tempfile
from tqdm.auto import tqdm

from displacement_tracker.paired_image_dataset import PairedImageDataset
from displacement_tracker.simple_cnn import SimpleCNN
from displacement_tracker.util.logging_config import setup_logging
from displacement_tracker.util.distance import interpolate_centroid
from displacement_tracker.util.deduplication import merge_close_points_global
from displacement_tracker.util.tiff_predictions import (
    save_prediction_tiff,
    merge_prediction_tiffs,
)

LOGGER = setup_logging("predict_json")


def extract_tile_centroids(probs_np, bounds, threshold, min_area):
    """Threshold a tile prediction map and return interpolated region centroids."""
    mask = probs_np > threshold
    labeled, num_features = label(mask)

    coords = []
    shape = mask.shape

    for region_id in range(1, num_features + 1):
        region_mask = labeled == region_id
        area = region_mask.sum()
        if area > min_area:
            centroid = center_of_mass(region_mask)
            peak_value = float(probs_np[region_mask].max())
            adjusted_peak = None
            try:
                lat, lon = interpolate_centroid(centroid, bounds, shape)
                coords.append((lat, lon, peak_value, adjusted_peak))
            except Exception as exc:
                LOGGER.warning(f"Interpolation error: {exc}")

    return coords


def extract_tile_nms(probs_np, bounds, threshold):
    """Return interpolated local maxima (lat, lon, peak_value) above threshold."""
    probs_t = torch.as_tensor(probs_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    blurred_np = gaussian_filter(probs_np, sigma=50) * 3
    blurred_t = torch.as_tensor(blurred_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    score_t = probs_t + blurred_t

    max_pooled = torch.nn.functional.max_pool2d(
        score_t,
        kernel_size=5,
        stride=1,
        padding=2,
    )
    local_max_mask = (score_t == max_pooled) & (score_t > threshold)

    rows, cols = torch.where(local_max_mask[0, 0])
    shape = probs_np.shape
    coords = []

    for row, col in zip(rows.tolist(), cols.tolist()):
        peak_value = float(probs_t[0, 0, row, col])
        adjusted_peak = float(score_t[0, 0, row, col])
        try:
            lat, lon = interpolate_centroid((row, col), bounds, shape)
            coords.append((lat, lon, peak_value, adjusted_peak))
        except Exception as exc:
            LOGGER.warning(f"Interpolation error: {exc}")

    return coords


def predict(
    dataset,
    model,
    device,
    selection_cfg,
    sample_cfg=None,
    validation_tifs=False,
    batch_size=12,
    num_workers=4,
    progress_label=None,
):
    """
    Run predictions and extract centroids of labeled regions above min_area for each tile.
    (No intra-tile spatial deduplication here; deduplication is performed globally after prediction.)
    Streaming: write per-tile centroids to a tmp NDJSON to avoid accumulating Python objects.
    """
    threshold = selection_cfg.get("threshold", 0.5)
    min_area = selection_cfg.get("min_area", 20)
    crop_pixels = selection_cfg.get("crop_pixels", 0)
    agreement = selection_cfg.get("agreement", False)
    min_distance_m = selection_cfg.get("min_distance_m", 2.0)
    method = str(selection_cfg.get("method", "centroid")).strip().lower()
    batch_size = max(1, int(batch_size))

    if method not in {"centroid", "nms"}:
        raise click.ClickException(
            "selection.method must be one of: 'centroid', 'nms'"
        )

    if sample_cfg and sample_cfg.get("enable", True):
        total = len(dataset)
        size = min(sample_cfg.get("size", total), total)
        seed = sample_cfg.get("seed", None)
        frac = float(size) / float(total)
        subsets, _ = dataset.create_subsets([frac, 1 - frac], seed=seed)
        subset = subsets[0]
        LOGGER.info(f"🔹 Using random sample of {size}/{total} tiles for prediction.")
    else:
        subset = dataset
        LOGGER.info(f"🔹 Using all {len(dataset)} tiles for prediction.")

    # streaming path for temporary per-tile points
    configured_tmp_path = selection_cfg.get("tmp_points_ndjson")
    if configured_tmp_path:
        tmp_ndjson = Path(configured_tmp_path)
        if tmp_ndjson.exists():
            try:
                tmp_ndjson.unlink()
            except Exception:
                LOGGER.warning(f"Could not remove existing tmp file {tmp_ndjson}")
    else:
        tmp_handle = tempfile.NamedTemporaryFile(
            prefix="pred_points_", suffix=".ndjson", delete=False
        )
        tmp_handle.close()
        tmp_ndjson = Path(tmp_handle.name)

    # DataLoader as before
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )

    process = psutil.Process(os.getpid())
    progress_desc = progress_label or Path(dataset.hdf5_path).name
    total_predicted_tents = 0
    max_tile_tents = 0

    # Open temp file once and append per-tile points as they are computed
    with tmp_ndjson.open("a", encoding="utf-8") as tmp_f:
        progress_bar = tqdm(
            loader,
            total=len(loader),
            desc=progress_desc,
            unit="batch",
            leave=True,
            dynamic_ncols=True,
            position=0,
        )
        status_bar = tqdm(
            total=0,
            bar_format="{desc}",
            leave=False,
            dynamic_ncols=True,
            position=1,
        )
        try:
            with torch.no_grad():
                for i, entry in enumerate(progress_bar):
                    if i % 200 == 0:
                        mem_gb = process.memory_info().rss / (1024**3)
                        if device.type == "cuda":
                            progress_bar.set_postfix(
                                rss_gb=f"{mem_gb:.2f}",
                                cuda_gb=f"{torch.cuda.memory_allocated(device)/(1024**3):.2f}",
                            )
                        else:
                            progress_bar.set_postfix(rss_gb=f"{mem_gb:.2f}")

                    try:
                        feature = entry["feature"]
                        prewar = entry["prewar"]
                        diff = feature - prewar

                        feats = torch.cat((feature, prewar, diff), dim=1)
                        feats = feats.to(device)

                        outputs = model(feats)
                        probs = outputs.detach().cpu().numpy()

                        del outputs
                        del feats
                        del feature
                        del prewar
                        del diff

                        batch_points = 0
                        batch_max_tile_tents = 0
                        batch_size = probs.shape[0]

                        for b in range(batch_size):
                            probs_np = probs[b, 0]
                            bounds = json.loads(entry["meta"][b])

                            if crop_pixels > 0:
                                probs_np[:crop_pixels, :] = 0
                                probs_np[-crop_pixels:, :] = 0
                                probs_np[:, :crop_pixels] = 0
                                probs_np[:, -crop_pixels:] = 0

                            if method == "nms":
                                coords = extract_tile_nms(
                                    probs_np, bounds, threshold
                                )
                            else:
                                coords = extract_tile_centroids(
                                    probs_np, bounds, threshold, min_area
                                )

                            tile_tent_count = len(coords)
                            batch_points += tile_tent_count
                            batch_max_tile_tents = max(
                                batch_max_tile_tents, tile_tent_count
                            )

                            for pt in coords:
                                tmp_f.write(json.dumps(pt) + "\n")

                            del coords

                        total_predicted_tents += batch_points
                        max_tile_tents = max(max_tile_tents, batch_max_tile_tents)
                        status_bar.set_description_str(
                            "current tent predictions: "
                            f"{total_predicted_tents} | max tents in tile: {max_tile_tents}"
                        )

                    except Exception:
                        LOGGER.exception("Prediction error during batch processing")
                        raise

                    if i % 500 == 0:
                        tmp_f.flush()
                        gc.collect()
        finally:
            status_bar.close()
            progress_bar.close()

        # ensure final flush
        tmp_f.flush()

    # After loop: read streamed points back into memory as flat list (tuples)
    flat_results = []
    with tmp_ndjson.open("r", encoding="utf-8") as fh:
        for line in fh:
            flat_results.append(tuple(json.loads(line)))

    # optionally remove temp file
    try:
        tmp_ndjson.unlink()
    except Exception:
        pass
    
    print("")  # add new line after tqdm bars

    LOGGER.info(f"Total number of tents (pre-merge): {len(flat_results)}")

    # Only keep points with agreement between overlaps
    if isinstance(agreement, bool):  # convert to int, False -> 1 point agreement, True -> 2 point agreement
        agreement = 2 if agreement else 1

    # global deduplication in meters
    merged_coords = merge_close_points_global(
        flat_results, min_distance_m=min_distance_m, agreement=agreement
    )
    LOGGER.info(f"Total number of tents (post-merge): {len(merged_coords)}")
    return merged_coords


def save_geojson(points, out_path, boundaries_path=None):
    """
    Save results to a GeoJSON file.
    Only keep points inside Gaza municipal boundaries.
    """

    # Load Gaza boundaries once
    gaza_path = Path(
        boundaries_path or "gaza_boundaries/GazaStrip_MunicipalBoundaries.shp"
    )
    gaza_gdf = gpd.read_file(gaza_path)
    gaza_gdf = gaza_gdf.to_crs("EPSG:4326")

    # Ensure single unified polygon
    gaza_union = gaza_gdf.geometry.union_all()

    features = []

    for lat, lon, peak, peak_adj in points:
        pt = Point(lon, lat)

        # Keep only points inside boundary
        if not gaza_union.contains(pt):
            continue

        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "name": "tents",
                    "peak_value": peak,
                    "adjusted_peak": peak_adj,
                },
            }
        )

    geojson = {"type": "FeatureCollection", "features": features}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)

    LOGGER.info(f"GeoJSON saved to {out_path}")


def resolve_prediction_jobs(pred_cfg):
    input_path = pred_cfg.get("input")
    output_path = pred_cfg.get("output")
    input_folder = pred_cfg.get("input_folder")
    output_folder = pred_cfg.get("output_folder")

    if input_folder or output_folder:
        if not input_folder or not output_folder:
            raise click.ClickException(
                "prediction.input_folder and prediction.output_folder must be set together"
            )

        input_dir = Path(input_folder)
        output_dir = Path(output_folder)
        if not input_dir.exists() or not input_dir.is_dir():
            raise click.ClickException(
                f"Prediction input folder not found: {input_dir}"
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        input_files = sorted(input_dir.glob("*.hdf5")) + sorted(input_dir.glob("*.h5"))
        if not input_files:
            raise click.ClickException(
                f"No HDF5 files found in prediction input folder: {input_dir}"
            )

        return [
            (input_file, output_dir / f"{input_file.stem}.json")
            for input_file in input_files
        ]

    if not input_path or not output_path:
        raise click.ClickException(
            "prediction.input and prediction.output are required unless folder mode is used"
        )

    return [(Path(input_path), Path(output_path))]


def run_prediction_job(
    input_path,
    output_path,
    model,
    device,
    selection_cfg,
    sample_cfg,
    validation_tifs,
    boundaries_path,
    batch_size,
    num_workers,
):
    dataset = PairedImageDataset(str(input_path))
    try:
        results = predict(
            dataset,
            model,
            device,
            selection_cfg,
            sample_cfg,
            validation_tifs=validation_tifs,
            batch_size=batch_size,
            num_workers=num_workers,
            progress_label=input_path.stem,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_geojson(results, output_path, boundaries_path=boundaries_path)
        del results
        gc.collect()
    finally:
        dataset.close()


@click.command()
@click.argument("config", type=click.Path(exists=True))
def cli(config) -> None:
    with open(config, "r") as f:
        params = yaml.safe_load(f)

    if "prediction" not in params:
        raise click.ClickException("Missing required config key: prediction")

    pred_cfg = params["prediction"]
    sample_cfg = pred_cfg.get("sample", {})
    selection_cfg = pred_cfg.get("selection", {})
    device = pred_cfg.get("device", None)
    batch_size = pred_cfg.get("batch_size", 12)
    num_workers = pred_cfg.get("num_workers", 4)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SimpleCNN.from_pth(
        pred_cfg["model"], model_args={"n_channels": 3, "n_classes": 1}
    )
    validation_tifs = pred_cfg.get("validation_tifs", False)
    boundaries_path = params.get("boundaries")

    device = torch.device(device)
    model.to(device)
    model.eval()

    prediction_jobs = resolve_prediction_jobs(pred_cfg)
    for input_path, output_path in prediction_jobs:
        run_prediction_job(
            input_path,
            output_path,
            model,
            device,
            selection_cfg,
            sample_cfg,
            validation_tifs,
            boundaries_path,
            batch_size,
            num_workers,
        )

    if validation_tifs:
        tiff_dir = Path(selection_cfg.get("tiff_output_dir", "prediction_tiffs"))
        mosaic_out = Path(pred_cfg.get("tiff_mosaic_output", "predictions_mosaic.tif"))
        merge_prediction_tiffs(tiff_dir, str(mosaic_out))


if __name__ == "__main__":
    cli()

# poetry run python displacement_tracker/04_predict_json.py predict_config.yaml