import yaml
import copy
import torch
import click
from pathlib import Path

import importlib

module = importlib.import_module("displacement_tracker.04_predict_json")
predict = module.predict
save_geojson = module.save_geojson

from displacement_tracker.paired_image_dataset import PairedImageDataset
from displacement_tracker.simple_cnn import SimpleCNN


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def main(config_path):

    with open(config_path, "r") as f:
        base_params = yaml.safe_load(f)

    # Load geotiff_dir from top level
    geotiff_dir = base_params.get("geotiff_dir")
    if not geotiff_dir:
        raise ValueError("No top-level 'geotiff_dir' found in config.")

    tif_dir = Path(geotiff_dir)
    if not tif_dir.exists():
        raise FileNotFoundError(f"GeoTIFF directory not found: {tif_dir}")

    pred_cfg = base_params["prediction"]
    processing_cfg = pred_cfg.get("processing", {})
    sample_cfg = pred_cfg.get("sample", {})
    device_name = pred_cfg.get("device", None)

    device = torch.device(
        device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = SimpleCNN.from_pth(
        pred_cfg["model"], model_args={"n_channels": 3, "n_classes": 1}
    )
    model.to(device)
    model.eval()

    # === Changed block: accept prediction.files OR loading.files, fallback to glob ===
    tif_files = pred_cfg.get("files") or base_params.get("loading", {}).get("files")
    if not tif_files:
        # fall back to scanning geotiff_dir for .tif files (filenames only)
        tif_files = [p.name for p in sorted(tif_dir.glob("*.tif"))]
    if not tif_files:
        raise ValueError(
            "No file list found: neither 'prediction.files' nor 'loading.files' present and no .tif files in geotiff_dir."
        )
    # === end changed block ===

    out_root = Path("predictions") / tif_dir.name
    out_root.mkdir(parents=True, exist_ok=True)

    for tif_name in tif_files:
        tif_path = tif_dir / tif_name

        if not tif_path.exists():
            raise FileNotFoundError(f"TIF file not found: {tif_path}")

        name = tif_path.stem

        params = copy.deepcopy(base_params)
        # override input/output for this run (in-memory only)
        # Base hdf5 template path from config
        hdf5_template = Path(base_params.get("hdf5"))
        if not hdf5_template:
            raise ValueError("No top-level 'hdf5' path found in config.")

        # Replace the filename part with current TIF stem
        h5_path = hdf5_template.parent / f"{name}.h5"

        if not h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

        params["prediction"]["input"] = str(h5_path)

        output_path = out_root / f"{name}.geojson"
        params["prediction"]["output"] = str(output_path)

        print(f"Running prediction for {name}")

        ds = PairedImageDataset(params["prediction"]["input"])

        results = predict(
            ds,
            model,
            device,
            processing_cfg,
            sample_cfg,
            validation_tifs=params["prediction"].get("validation_tifs", False),
        )

        save_geojson(results, output_path)


if __name__ == "__main__":
    main()

# poetry run python displacement_tracker/util/run_batch_predictions.py predict_config.yaml