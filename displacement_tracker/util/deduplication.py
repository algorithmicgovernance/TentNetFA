from displacement_tracker.util.distance import haversine_m


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
        return []

    uf = UnionFind(n)

    # O(n^2) pairwise merge; OK for moderate n (few thousands).
    for i in range(n):
        lat_i, lon_i, _ = flat[i]
        for j in range(i + 1, n):
            lat_j, lon_j, _ = flat[j]
            if haversine_m(lat_i, lon_i, lat_j, lon_j) <= min_distance_m:
                uf.union(i, j)

    # collect clusters
    clusters = {}
    for idx in range(n):
        root = uf.find(idx)
        clusters.setdefault(root, []).append(idx)

    # compute centroid for each cluster (simple average of lat/lon)
    merged = []
    for members in clusters.values():
        sum_lat = 0.0
        sum_lon = 0.0
        max_peak = 0.0

        if len(members) < agreement:
            continue

        for m in members:
            lat, lon, peak = flat[m]
            sum_lat += lat
            sum_lon += lon
            if peak > max_peak:
                max_peak = peak

        cnt = len(members)
        merged.append((sum_lat / cnt, sum_lon / cnt, max_peak))
    return merged
