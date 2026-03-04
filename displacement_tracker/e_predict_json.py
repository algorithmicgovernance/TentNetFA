import yaml
import click
import torch
import json
from pathlib import Path
from scipy.ndimage import label, center_of_mass
import geopandas as gpd
from shapely.geometry import Point
from torch.utils.data import DataLoader
import gc
import os
import psutil

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


def predict(
    dataset, model, device, processing_cfg, sample_cfg=None, validation_tifs=False
):
    """
    Run predictions and extract centroids of labeled regions above min_area for each tile.
    (No intra-tile spatial deduplication here; deduplication is performed globally after prediction.)
    Streaming: write per-tile centroids to a tmp NDJSON to avoid accumulating Python objects.
    """
    import psutil  # <-- ensure you added `import psutil` at top of file as well
    threshold = processing_cfg.get("threshold", 0.5)
    min_area = processing_cfg.get("min_area", 20)
    crop_pixels = processing_cfg.get("crop_pixels", 0)
    agreement = processing_cfg.get("agreement", False)
    min_distance_m = processing_cfg.get("min_distance_m", 2.0)

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
    tmp_ndjson = Path(processing_cfg.get("tmp_points_ndjson", "pred_points.ndjson"))
    if tmp_ndjson.exists():
        try:
            tmp_ndjson.unlink()
        except Exception:
            LOGGER.warning(f"Could not remove existing tmp file {tmp_ndjson}")

    # DataLoader as before
    loader = DataLoader(
        subset,
        batch_size=6,
        shuffle=False,
        num_workers=6,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )

    process = psutil.Process(os.getpid())

    # Open temp file once and append per-tile points as they are computed
    with tmp_ndjson.open("a", encoding="utf-8") as tmp_f:
        with torch.no_grad():
            for i, entry in enumerate(loader):
                # periodic memory logging
                if i % 200 == 0:
                    mem_gb = process.memory_info().rss / (1024 ** 3)
                    LOGGER.info(f"Iteration {i}/{len(loader)} — RSS {mem_gb:.2f} GB")
                    if device.type == "cuda":
                        LOGGER.info(
                            f"CUDA alloc: {torch.cuda.memory_allocated(device)/(1024**3):.2f} GB, "
                            f"cached: {torch.cuda.memory_reserved(device)/(1024**3):.2f} GB"
                        )

                try:
                    LOGGER.info(f"Predicting image {i + 1}/{len(loader)}")

                    feature = entry["feature"]  # [B, C, H, W]
                    prewar = entry["prewar"]  # [B, C, H, W]
                    diff = feature - prewar

                    feats = torch.cat((feature, prewar, diff), dim=1)  # concat channels
                    feats = feats.to(device)

                    outputs = model(feats)  # [B, 1, H, W]
                    probs = outputs.detach().cpu().numpy()

                    # free GPU/PyTorch objects immediately
                    del outputs
                    del feats
                    del feature
                    del prewar
                    del diff

                    B = probs.shape[0]

                    for b in range(B):
                        probs_np = probs[b, 0]  # [H, W]
                        bounds = json.loads(entry["meta"][b])

                        if crop_pixels > 0:
                            probs_np[:crop_pixels, :] = 0
                            probs_np[-crop_pixels:, :] = 0
                            probs_np[:, :crop_pixels] = 0
                            probs_np[:, -crop_pixels:] = 0

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
                                try:
                                    lat, lon = interpolate_centroid(centroid, bounds, shape)
                                    coords.append((lat, lon, peak_value))
                                except Exception as exc:
                                    LOGGER.warning(f"Interpolation error: {exc}")

                        # STREAM: write each point as a JSON tuple line instead of keeping in RAM
                        for pt in coords:
                            tmp_f.write(json.dumps(pt) + "\n")
                        # release coords list
                        del coords

                    LOGGER.info(f"Found {num_features} tents on batch (pre-merge).")

                except Exception:
                    # log full traceback to help debug; do not silence errors
                    LOGGER.exception("Prediction error during batch processing")
                    raise

                # periodic flush + collect to reduce spike risk
                if i % 500 == 0:
                    tmp_f.flush()
                    gc.collect()

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

def save_geojson(points, out_path):
    """
    Save results to a GeoJSON file.
    Only keep points inside Gaza municipal boundaries.
    """

    # Load Gaza boundaries once
    gaza_path = Path("gaza_boundaries/GazaStrip_MunicipalBoundaries.shp")
    gaza_gdf = gpd.read_file(gaza_path)
    gaza_gdf = gaza_gdf.to_crs("EPSG:4326")

    # Ensure single unified polygon
    gaza_union = gaza_gdf.geometry.union_all()

    features = []

    for lat, lon, peak in points:
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
                },
            }
        )

    geojson = {"type": "FeatureCollection", "features": features}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)

    LOGGER.info(f"GeoJSON saved to {out_path}")


@click.command()
@click.argument("config", type=click.Path(exists=True))
def cli(config) -> None:
    with open(config, "r") as f:
        params = yaml.safe_load(f)

    if "prediction" not in params:
        raise click.ClickException("Missing required config key: prediction")

    pred_cfg = params["prediction"]
    sample_cfg = pred_cfg.get("sample", {})
    out_path = pred_cfg.get("output", "predictions.geojson")
    processing_cfg = pred_cfg.get("processing", {})
    device = pred_cfg.get("device", None)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = PairedImageDataset(pred_cfg["input"])
    model = SimpleCNN.from_pth(
        pred_cfg["model"], model_args={"n_channels": 3, "n_classes": 1}
    )
    validation_tifs = pred_cfg.get("validation_tifs", False)

    device = torch.device(device)
    model.to(device)

    # run predictions (per-tile centroids, no intra-tile dedupe)
    results = predict(
        ds, model, device, processing_cfg, sample_cfg, validation_tifs=validation_tifs
    )
    # Save only the cleaned/deduplicated points (no raw points, no deduplicated flag)
    save_geojson(results, out_path)
    del results
    gc.collect()

    if validation_tifs:
        tiff_dir = Path(processing_cfg.get("tiff_output_dir", "prediction_tiffs"))
        mosaic_out = Path(pred_cfg.get("tiff_mosaic_output", "predictions_mosaic.tif"))
        merge_prediction_tiffs(tiff_dir, str(mosaic_out))


if __name__ == "__main__":
    cli()

# poetry run python displacement_tracker/04_predict_json.py predict_config.yaml