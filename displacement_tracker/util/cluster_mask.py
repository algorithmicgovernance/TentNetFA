from pathlib import Path
import numpy as np
import joblib
import rasterio

def compute_cluster4_mask(pred_cfg, prewar_path, geotiff_dir):
    """
    Load original GeoTIFF (derived from input .h5 name),
    apply saved clustering model, return boolean mask of cluster 4.
    """

    # derive original tif path
    h5_path = Path(pred_cfg["input"])
    tif_name = h5_path.stem + ".tif"
    tif_path = Path(geotiff_dir) / tif_name

    cluster_model_path = pred_cfg.get("cluster_model")
    if cluster_model_path is None:
        raise ValueError("prediction.cluster_model must be provided in config")

    bundle = joblib.load(cluster_model_path)
    scaler = bundle["scaler"]
    kmeans = bundle["kmeans"]
    TARGET_CLUSTER = 4

    with rasterio.open(tif_path) as post_src, rasterio.open(prewar_path) as pre_src:

        post = post_src.read().astype(np.float32)
        pre = pre_src.read(
            out_shape=post.shape,
            resampling=rasterio.enums.Resampling.bilinear,
        ).astype(np.float32)

        bands, h, w = post.shape

        features = []
        for i in range(bands):
            features.append(post[i].ravel())
        for i in range(bands):
            features.append(np.abs(post[i] - pre[i]).ravel())

        X = np.stack(features, axis=1)
        X_scaled = scaler.transform(X)
        labels = kmeans.predict(X_scaled)

        cluster_map = labels.reshape(h, w)
        mask = cluster_map == TARGET_CLUSTER

        transform = post_src.transform

    return mask, transform


def point_in_cluster_mask(lat, lon, mask, transform):
    """
    Check if geographic point falls inside cluster 4 mask.
    """
    col, row = ~transform * (lon, lat)
    row = int(round(row))
    col = int(round(col))

    if 0 <= row < mask.shape[0] and 0 <= col < mask.shape[1]:
        return bool(mask[row, col])
    return False
