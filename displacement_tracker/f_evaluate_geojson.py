import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import click
import math


@dataclass(frozen=True)
class Bounds:
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    # width/height no longer used; kept for backward compatibility of parsing
    width: int = 0
    height: int = 0

    def contains(self, lon: float, lat: float) -> bool:
        return (self.lon_min <= lon <= self.lon_max) and (
            self.lat_min <= lat <= self.lat_max
        )


def lonlat_dist(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Euclidean distance in degrees between two lon/lat points (not meters)."""
    return math.hypot(lon2 - lon1, lat2 - lat1)


def load_geojson(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def is_point_feature(feat: Dict) -> bool:
    return feat.get("geometry", {}).get("type") == "Point"


def is_polygon_feature(feat: Dict) -> bool:
    return feat.get("geometry", {}).get("type") in {"Polygon", "MultiPolygon"}


def feature_coords(feat: Dict) -> List:
    return feat.get("geometry", {}).get("coordinates", [])


def extract_bounds_from_polygon_feature(
    feat: Dict, default_width: int = 0, default_height: int = 0
) -> Bounds:
    geom = feat.get("geometry", {})
    coords = geom.get("coordinates")
    if geom.get("type") == "Polygon":
        rings = coords or []
        # exterior ring first
        exterior = rings[0] if rings else []
        xs = [c[0] for c in exterior]
        ys = [c[1] for c in exterior]
    elif geom.get("type") == "MultiPolygon":
        # take the first polygon's exterior ring
        exterior = (coords[0][0]) if coords and coords[0] else []
        xs = [c[0] for c in exterior]
        ys = [c[1] for c in exterior]
    else:
        raise ValueError("Feature is not a polygon type")

    lon_min, lon_max = min(xs), max(xs)
    lat_min, lat_max = min(ys), max(ys)

    # width/height not used anymore, but we read if provided
    props = feat.get("properties", {})
    width = (
        int(props.get("width", props.get("image_width", default_width))) if props else 0
    )
    height = (
        int(props.get("height", props.get("image_height", default_height)))
        if props
        else 0
    )

    return Bounds(
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        width=width,
        height=height,
    )


def collect_points(fc: Dict) -> List[Tuple[float, float, Dict]]:
    pts: List[Tuple[float, float, Dict]] = []
    for feat in fc.get("features", []):
        if is_point_feature(feat):
            lon, lat = feature_coords(feat)
            pts.append((lon, lat, feat.get("properties", {})))
    return pts


def collect_bounds(fc: Dict) -> List[Bounds]:
    bounds_list: List[Bounds] = []
    for feat in fc.get("features", []):
        if is_polygon_feature(feat):
            try:
                b = extract_bounds_from_polygon_feature(feat)
                bounds_list.append(b)
            except Exception:
                # Ignore malformed polygon features
                pass
    return bounds_list


def group_points_by_bounds(
    points: List[Tuple[float, float, Dict]], bounds_list: List[Bounds]
) -> Dict[int, List[Tuple[float, float, Dict]]]:
    groups: Dict[int, List[Tuple[float, float, Dict]]] = {
        i: [] for i in range(len(bounds_list))
    }
    for lon, lat, props in points:
        for i, b in enumerate(bounds_list):
            if b.contains(lon, lat):
                groups[i].append((lon, lat, props))
                break
    return groups


@dataclass
class TileStats:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: "TileStats") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    def to_dict(self) -> Dict:
        total_pred = self.tp + self.fp
        total_gt = self.tp + self.fn
        denom = self.tp + self.fp + self.fn
        precision = (self.tp / total_pred) if total_pred > 0 else 0.0
        recall = (self.tp / total_gt) if total_gt > 0 else 0.0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )
        accuracy = (self.tp / denom) if denom > 0 else 0.0
        fp_rate = (self.fp / total_pred) if total_pred > 0 else 0.0  # 1 - precision
        fn_rate = (self.fn / total_gt) if total_gt > 0 else 0.0  # 1 - recall
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "fp_rate": fp_rate,
            "fn_rate": fn_rate,
        }


def match_points_per_tile_lonlat(
    gt_pts: List[Tuple[float, float, Dict]],
    pred_pts: List[Tuple[float, float, Dict]],
    dist_deg: float,
) -> TileStats:
    # Prepare coordinate lists
    gt_coords = [(lon, lat) for lon, lat, _ in gt_pts]
    pr_coords = [(lon, lat) for lon, lat, _ in pred_pts]

    # Build list of (distance, gt_idx, pred_idx) for pairs within threshold (degrees)
    pairs: List[Tuple[float, int, int]] = []
    near_gt_for_pred: List[List[int]] = [[] for _ in range(len(pr_coords))]

    for gi, (glon, glat) in enumerate(gt_coords):
        for pi, (plon, plat) in enumerate(pr_coords):
            d = lonlat_dist(glon, glat, plon, plat)
            if d <= dist_deg:
                pairs.append((d, gi, pi))
                near_gt_for_pred[pi].append(gi)

    # Greedy assignment by increasing distance
    pairs.sort(key=lambda x: x[0])
    assigned_gt = set()
    assigned_pred = set()
    for d, gi, pi in pairs:
        if gi in assigned_gt or pi in assigned_pred:
            continue
        assigned_gt.add(gi)
        assigned_pred.add(pi)

    tp = len(assigned_gt)
    fn = len(gt_coords) - tp

    # False positives: any prediction not assigned
    fp = 0
    for pi in range(len(pr_coords)):
        if pi in assigned_pred:
            continue
        # Whether near any GT or far, it's an FP when unassigned
        fp += 1

    return TileStats(tp=tp, fp=fp, fn=fn)


@click.command()
@click.argument("gt_geojson", type=click.Path(exists=True))
@click.argument("pred_geojson", type=click.Path(exists=True))
@click.option(
    "--dist-deg",
    type=float,
    default=0.0005,
    help="Distance threshold in lon/lat degrees for matching",
)
@click.option(
    "--per-tile",
    is_flag=True,
    default=False,
    help="Print per-tile stats as well as overall",
)
@click.option(
    "--save-report",
    type=click.Path(),
    default=None,
    help="Optional path to save JSON report",
)
@click.option(
    "--global-match",
    is_flag=True,
    default=False,
    help="Force global evaluation of all points, ignoring tile bounds.",
)
def cli(
    gt_geojson: str,
    pred_geojson: str,
    dist_deg: float,
    per_tile: bool,
    save_report: Optional[str],
    global_match: bool,
):
    """
    Evaluate predictions (points + rectangular bounds) against ground-truth points using lon/lat degrees.

    - If bounds exist and global-match is NOT set:
       - Filters GT points to only those within any rectangle in prediction GeoJSON.
       - Performs per-rectangle matching using Euclidean distance in degrees (lon/lat space).
    - If bounds do NOT exist or global-match IS set:
       - Matches all GT points against all predicted points globally.

    Computes TP/FP/FN and reports accuracy, precision, recall, F1, FP rate, FN rate.
    """
    gt_fc = load_geojson(gt_geojson)
    pr_fc = load_geojson(pred_geojson)

    gt_points = collect_points(gt_fc)
    pred_points = collect_points(pr_fc)
    bounds_list = collect_bounds(pr_fc)

    # Determine evaluation mode
    do_global = global_match
    if not bounds_list and not global_match:
        click.echo(
            "No rectangular bounds (Polygon features) found in prediction GeoJSON. Falling back to global matching.",
            err=True,
        )
        do_global = True

    if do_global:
        # Global matching: Match all GT points against all Pred points directly
        stats = match_points_per_tile_lonlat(gt_points, pred_points, dist_deg)
        report = {
            "mode": "global",
            "threshold_degrees": dist_deg,
            "overall": stats.to_dict(),
            "counts": {"gt_total": len(gt_points), "pred_total": len(pred_points)},
        }
    else:
        # Tile-based matching: Only consider GT points inside prediction tiles
        gt_by_tile = group_points_by_bounds(gt_points, bounds_list)
        pr_by_tile = group_points_by_bounds(pred_points, bounds_list)

        overall = TileStats()
        per_tile_stats: List[Dict] = []

        for i, bounds in enumerate(bounds_list):
            gt_tile = gt_by_tile.get(i, [])
            pr_tile = pr_by_tile.get(i, [])
            stats = match_points_per_tile_lonlat(gt_tile, pr_tile, dist_deg)
            overall.add(stats)
            if per_tile:
                per_tile_stats.append(
                    {
                        "tile_index": i,
                        "bounds": {
                            "lon_min": bounds.lon_min,
                            "lon_max": bounds.lon_max,
                            "lat_min": bounds.lat_min,
                            "lat_max": bounds.lat_max,
                        },
                        "counts": stats.to_dict(),
                        "num_gt_points": len(gt_tile),
                        "num_pred_points": len(pr_tile),
                    }
                )

        report = {
            "mode": "tiled",
            "threshold_degrees": dist_deg,
            "overall": overall.to_dict(),
        }
        if per_tile:
            report["per_tile"] = per_tile_stats

    click.echo(json.dumps(report, indent=2))

    if save_report:
        with open(save_report, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    cli()
