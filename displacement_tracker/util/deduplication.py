import math
import time

import numpy as np
from scipy.spatial import cKDTree

from displacement_tracker.util.distance import haversine_m
from displacement_tracker.util.logging_config import setup_logging

LOGGER = setup_logging("deduplication")


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def merge_close_points_global(flat, min_distance_m=2.0, agreement: int = 1):
    """
    Merge points across all flat (lat, lon, peak) centroids ...
    Returns a flat list of merged (lat, lon, peak) centroids.
    Peak is the max peak within each cluster.
    """
    n = len(flat)
    if n == 0:
        LOGGER.info("No points provided for global deduplication.")
        return []

    start_t = time.perf_counter()
    LOGGER.info(
        "Starting global deduplication: points=%d, min_distance_m=%.3f, agreement=%d",
        n,
        float(min_distance_m),
        int(agreement),
    )

    uf = UnionFind(n)

    # Build a local metric projection so spatial indexing can cheaply find
    # candidate neighbors within min_distance_m.
    lats = np.array([pt[0] for pt in flat], dtype=np.float64)
    lons = np.array([pt[1] for pt in flat], dtype=np.float64)

    lat0_rad = math.radians(float(lats.mean()))
    meters_per_deg = (math.pi / 180.0) * 6371000.0
    x = lons * meters_per_deg * max(math.cos(lat0_rad), 1e-12)
    y = lats * meters_per_deg

    tree = cKDTree(np.column_stack((x, y)))
    LOGGER.info("Spatial index built for %d points.", n)

    # Candidate pairs from the index; verify with haversine for exactness.
    candidate_pairs = tree.query_pairs(r=max(0.0, float(min_distance_m)))
    candidate_count = len(candidate_pairs)
    LOGGER.info("Candidate neighbor pairs from KD-tree: %d", candidate_count)

    unions = 0
    if candidate_count > 0:
        pair_log_every = max(10_000, candidate_count // 10)
    else:
        pair_log_every = 10_000

    for idx, (i, j) in enumerate(candidate_pairs, start=1):
        lat_i, lon_i, _, _ = flat[i]
        lat_j, lon_j, _, _ = flat[j]
        if haversine_m(lat_i, lon_i, lat_j, lon_j) <= min_distance_m:
            uf.union(i, j)
            unions += 1

        if idx % pair_log_every == 0 or idx == candidate_count:
            LOGGER.info(
                "Pair check progress: %d/%d (%.1f%%), unions=%d",
                idx,
                candidate_count,
                (100.0 * idx / candidate_count) if candidate_count else 100.0,
                unions,
            )

    # collect clusters
    clusters = {}
    cluster_log_every = max(10_000, n // 10)
    for idx in range(n):
        root = uf.find(idx)
        clusters.setdefault(root, []).append(idx)

        checked = idx + 1
        if checked % cluster_log_every == 0 or checked == n:
            LOGGER.info(
                "Cluster assignment progress: %d/%d (%.1f%%)",
                checked,
                n,
                100.0 * checked / n,
            )

    LOGGER.info("Clusters formed: %d", len(clusters))

    # compute centroid for each cluster (simple average of lat/lon)
    merged = []
    for members in clusters.values():
        sum_lat = 0.0
        sum_lon = 0.0
        max_peak = 0.0
        max_adj_peak = 0.0

        if len(members) < agreement:
            continue

        for m in members:
            lat, lon, peak, adj_peak = flat[m]
            sum_lat += lat
            sum_lon += lon
            if peak > max_peak:
                max_peak = peak
                max_adj_peak = adj_peak

        cnt = len(members)
        merged.append((sum_lat / cnt, sum_lon / cnt, max_peak, max_adj_peak))

    elapsed_s = time.perf_counter() - start_t
    LOGGER.info(
        "Deduplication complete: input=%d, output=%d, elapsed=%.2fs",
        n,
        len(merged),
        elapsed_s,
    )
    return merged
