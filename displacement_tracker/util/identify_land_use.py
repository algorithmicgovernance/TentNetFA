#!/usr/bin/env python

import sys
import os
import yaml
import numpy as np
import rasterio
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from rasterio.warp import transform
import joblib

# ------------------------------------------------------------------
# SET THIS ONCE
# ------------------------------------------------------------------
PREWAR_TIF_PATH = "tif_files/prewar_gaza.tif"
SAMPLE_PIXELS_PER_TIF = 10000
BATCH_SIZE = 50000
# ------------------------------------------------------------------

def build_features(post_pixels, pre_pixels):
    features = []

    for i in range(post_pixels.shape[0]):
        features.append(post_pixels[i])

    for i in range(post_pixels.shape[0]):
        features.append(np.abs(post_pixels[i] - pre_pixels[i]))

    return np.stack(features, axis=1)


def sample_from_tif(post_path):
    with rasterio.open(post_path) as post_src, rasterio.open(PREWAR_TIF_PATH) as pre_src:

        height = post_src.height
        width = post_src.width
        bands = post_src.count

        collected = []
        total_collected = 0

        while total_collected < SAMPLE_PIXELS_PER_TIF:

            rows = np.random.randint(0, height, BATCH_SIZE)
            cols = np.random.randint(0, width, BATCH_SIZE)

            # Read post pixels efficiently
            post_pixels = post_src.read()[:, rows, cols].astype(np.float32)

            # Convert to geographic coords
            xs, ys = post_src.transform * (cols, rows)

            # Transform CRS if needed
            if post_src.crs != pre_src.crs:
                xs, ys = transform(post_src.crs, pre_src.crs, xs, ys)

            coords = list(zip(xs, ys))

            # Sample prewar only at these coordinates
            pre_samples = np.array(list(pre_src.sample(coords))).T.astype(np.float32)

            # Ensure correct shape
            if pre_samples.shape[0] != bands:
                continue

            # Mask invalid overlap
            valid_mask = (
                ~np.any(np.isnan(post_pixels), axis=0)
                & ~np.any(np.isnan(pre_samples), axis=0)
            )

            if not np.any(valid_mask):
                continue

            post_valid = post_pixels[:, valid_mask]
            pre_valid = pre_samples[:, valid_mask]

            # Build features
            features = []
            for i in range(bands):
                features.append(post_valid[i])
            for i in range(bands):
                features.append(np.abs(post_valid[i] - pre_valid[i]))

            X = np.stack(features, axis=1)

            collected.append(X)
            total_collected += X.shape[0]

        X_all = np.vstack(collected)

        if X_all.shape[0] > SAMPLE_PIXELS_PER_TIF:
            idx = np.random.choice(X_all.shape[0], SAMPLE_PIXELS_PER_TIF, replace=False)
            X_all = X_all[idx]

        return X_all


def main():
    if len(sys.argv) != 3:
        print("Usage: python identify_land_use.py <predict_config.yaml> <n_clusters>")
        sys.exit(1)

    config_path = sys.argv[1]
    n_clusters = int(sys.argv[2])

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    geotiff_dir = cfg["geotiff_dir"]
    file_list = cfg["loading"]["files"]

    scaler = StandardScaler()
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=10000,
        n_init=10,
    )

    scaler_initialized = False

    for fname in file_list:
        tif_name = os.path.splitext(fname)[0] + ".tif"
        post_path = os.path.join(geotiff_dir, tif_name)

        if not os.path.exists(post_path):
            print(f"Skipping missing file: {post_path}")
            continue

        print(f"Sampling from {tif_name}...")
        X = sample_from_tif(post_path)

        if X is None:
            continue

        if not scaler_initialized:
            scaler.partial_fit(X)
            scaler_initialized = True
        else:
            scaler.partial_fit(X)

    if not scaler_initialized:
        raise RuntimeError("No valid pixels found across all TIFFs.")

    print("Training KMeans incrementally...")

    for fname in file_list:
        tif_name = os.path.splitext(fname)[0] + ".tif"
        post_path = os.path.join(geotiff_dir, tif_name)

        if not os.path.exists(post_path):
            continue

        X = sample_from_tif(post_path)
        if X is None:
            continue

        X_scaled = scaler.transform(X)
        model.partial_fit(X_scaled)

    model_output_path = "runs/land_use_model.pkl"

    joblib.dump(
        {
            "scaler": scaler,
            "kmeans": model,
            "n_clusters": n_clusters,
        },
        model_output_path,
    )

    print(f"Saved global model to {model_output_path}")


if __name__ == "__main__":
    main()