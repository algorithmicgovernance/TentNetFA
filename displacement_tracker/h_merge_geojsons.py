"""
merge_geojsons.py

Merge multiple point-cloud GeoJSON files from a folder into a single deduplicated
GeoPackage (.gpkg), using the same merge_close_points_global function used during
prediction.

Usage:
    poetry run merge-geojsons <input_folder> <output.gpkg> [OPTIONS]
"""

import json
from pathlib import Path

import click
import geopandas as gpd
from shapely.geometry import Point
import yaml

from displacement_tracker.util.deduplication import merge_close_points_global
from displacement_tracker.util.logging_config import setup_logging

LOGGER = setup_logging("merge_geojsons")


def load_thresholds(thresholds_config: str | None) -> dict[str, float]:
    """
    Load per-file adj_peak thresholds from a YAML file.

    Expected format::

        default: 0.05          # optional global fallback (overrides --min-adj-peak)
        per_file:
          some_file.json: 0.10
          other_file.geojson: 0.20
    """
    if not thresholds_config:
        return {}
    path = Path(thresholds_config)
    if not path.exists():
        raise click.ClickException(f"Thresholds config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def resolve_threshold(
    filename: str,
    thresholds_data: dict,
    global_threshold: float,
) -> float:
    """Return the effective adj_peak threshold for a given filename."""
    per_file = thresholds_data.get("per_file") or {}
    if filename in per_file:
        return float(per_file[filename])
    if "default" in thresholds_data:
        return float(thresholds_data["default"])
    return global_threshold


def load_points_from_geojson(path: Path) -> list[tuple]:
    """
    Load all Point features from a GeoJSON file.
    Returns a list of (lat, lon, peak_value, adjusted_peak) tuples.
    adjusted_peak defaults to 0.0 when the field is absent.
    """
    with path.open("r", encoding="utf-8") as f:
        gj = json.load(f)

    points = []
    for feat in gj.get("features", []):
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates")
        if not coords or len(coords) < 2:
            continue
        lon, lat = float(coords[0]), float(coords[1])
        props = feat.get("properties") or {}
        peak = float(props.get("peak_value", 0.0))
        adj_peak = float(props.get("adjusted_peak", 0.0))
        points.append((lat, lon, peak, adj_peak))

    return points


def load_exclusion_geometry(exclusion_zones_gpkg: str | None):
    """Load and unify exclusion geometries from a GeoPackage file."""
    if not exclusion_zones_gpkg:
        return None

    path = Path(exclusion_zones_gpkg)
    if not path.exists():
        raise click.ClickException(f"Exclusion zones GPKG not found: {path}")

    try:
        zones_gdf = gpd.read_file(path)
    except Exception as exc:
        raise click.ClickException(
            f"Failed to read exclusion zones GPKG {path}: {exc}"
        ) from exc

    if zones_gdf.empty:
        LOGGER.warning("Exclusion zones file is empty: %s", path)
        return None

    if zones_gdf.crs is None:
        LOGGER.warning(
            "Exclusion zones CRS missing, assuming EPSG:4326: %s", path
        )
        zones_gdf = zones_gdf.set_crs("EPSG:4326")
    else:
        zones_gdf = zones_gdf.to_crs("EPSG:4326")

    exclusion_geom = zones_gdf.geometry.union_all()
    LOGGER.info("Loaded exclusion zones from %s", path)
    return exclusion_geom


def filter_points_by_exclusion(points: list[tuple], exclusion_geom) -> list[tuple]:
    """Drop points that lie inside exclusion geometry."""
    if exclusion_geom is None:
        return points

    kept = []
    for lat, lon, peak, adj_peak in points:
        pt = Point(lon, lat)
        if exclusion_geom.contains(pt):
            continue
        kept.append((lat, lon, peak, adj_peak))
    return kept


def save_merged_gpkg(points: list[tuple], out_path: Path) -> None:
    """Save merged (lat, lon, peak, adj_peak) tuples to a GeoPackage file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "geometry": Point(lon, lat),
            "name": "tents",
            "peak_value": peak,
            "adjusted_peak": adj_peak,
        }
        for lat, lon, peak, adj_peak in points
    ]

    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    gdf.to_file(str(out_path), driver="GPKG")

    LOGGER.info("Merged GeoPackage saved to %s (%d points)", out_path, len(gdf))


@click.command()
@click.argument("input_folder", type=click.Path(exists=True, file_okay=False))
@click.argument("output_gpkg", type=click.Path(dir_okay=False))
@click.option(
    "--min-distance-m",
    default=3.0,
    show_default=True,
    help="Merge points within this distance (metres).",
)
@click.option(
    "--agreement",
    default=1,
    show_default=True,
    help="Minimum cluster size to keep after merging.",
)
@click.option(
    "--min-adj-peak",
    default=0.0,
    show_default=True,
    help="Global minimum adjusted_peak threshold; points below this are dropped before merging.",
)
@click.option(
    "--thresholds-config",
    default=None,
    type=click.Path(dir_okay=False),
    help=(
        "Optional YAML file with per-file adj_peak thresholds. "
        "Keys: 'default' (overrides --min-adj-peak) and 'per_file' (filename -> threshold)."
    ),
)
@click.option(
    "--exclusion-zones-gpkg",
    default=None,
    type=click.Path(dir_okay=False),
    help="Optional GPKG file with exclusion geometries; points inside are dropped before merge.",
)
def cli(
    input_folder: str,
    output_gpkg: str,
    min_distance_m: float,
    agreement: int,
    min_adj_peak: float,
    thresholds_config: str | None,
    exclusion_zones_gpkg: str | None,
) -> None:
    input_dir = Path(input_folder)
    geojson_files = sorted(input_dir.glob("*.geojson")) + sorted(input_dir.glob("*.json"))

    if not geojson_files:
        raise click.ClickException(f"No GeoJSON files found in {input_dir}")

    LOGGER.info("Found %d GeoJSON files in %s", len(geojson_files), input_dir)

    thresholds_data = load_thresholds(thresholds_config)
    exclusion_geom = load_exclusion_geometry(exclusion_zones_gpkg)

    flat: list[tuple] = []
    for path in geojson_files:
        pts = load_points_from_geojson(path)
        threshold = resolve_threshold(path.name, thresholds_data, min_adj_peak)
        if threshold > 0.0:
            filtered = [p for p in pts if p[3] >= threshold]
            LOGGER.info(
                "  %s: %d points loaded, %d kept (adj_peak >= %.4f)",
                path.name, len(pts), len(filtered), threshold,
            )
            pts = filtered
        else:
            LOGGER.info("  %s: %d points", path.name, len(pts))

        if exclusion_geom is not None:
            before_exclusion = len(pts)
            pts = filter_points_by_exclusion(pts, exclusion_geom)
            LOGGER.info(
                "  %s: %d kept after exclusion filtering",
                path.name,
                len(pts),
            )
            if before_exclusion != len(pts):
                LOGGER.info(
                    "  %s: %d points removed by exclusion zones",
                    path.name,
                    before_exclusion - len(pts),
                )

        flat.extend(pts)

    LOGGER.info("Total points before merge: %d", len(flat))

    merged = merge_close_points_global(
        flat, min_distance_m=min_distance_m, agreement=agreement
    )

    LOGGER.info("Total points after merge: %d", len(merged))
    save_merged_gpkg(merged, Path(output_gpkg))


if __name__ == "__main__":
    cli()
