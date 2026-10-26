"""Shared command-line options and export utilities; plotting lives in figures/."""
from pathlib import Path
import argparse
import json
import sys

from package_paths import PACKAGE_ROOT, output_dir


def prepare_figure(label, status):
    parser = argparse.ArgumentParser(description="Draw " + label)
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--external-inputs", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    args = parser.parse_args()
    if args.check_inputs:
        mapping = json.loads((PACKAGE_ROOT / "code/input_paths.json").read_text(encoding="utf-8"))
        missing = sorted({value for value in mapping.values() if not (PACKAGE_ROOT / value).is_file()})
        if missing:
            raise SystemExit("Missing packaged inputs: " + ", ".join(missing))
        print(label + ": " + status)
        raise SystemExit(0)
    if status != "analysis_tables" and args.external_inputs is None:
        raise SystemExit(label + ": " + status + ". See figure_index.csv and provide local files with --external-inputs.")
    if args.external_inputs:
        import package_paths
        package_paths.EXTERNAL_ROOT = args.external_inputs.resolve()
    import matplotlib
    matplotlib.use("Agg")
    sys.argv = [sys.argv[0]]
    return args


def save_figure(figure, bbox, label, args):
    import matplotlib.pyplot as plt
    if figure is not None:
        for extension in args.formats:
            figure.savefig(output_dir(label) / (label + "." + extension), dpi=args.dpi,
                           bbox_inches=bbox if bbox is not None else "tight", facecolor="white")
        plt.close(figure)
    print(output_dir(label))


def read_table(relative):
    import pandas as pd
    return pd.read_csv(PACKAGE_ROOT / "data" / relative)


def model_tables(zone=False):
    folder = "shared/zone_association_models" if zone else "shared/global_association_models"
    prefix = "Zone_model" if zone else "Global_model"
    return lambda name: read_table(f"{folder}/{prefix}_{name}.csv")


def configure_land_functions(module):
    land = PACKAGE_ROOT / "data/shared/World_land_polygons.geojson"
    for name in ["find_world_land", "find_naturalearth_shp", "find_land_path", "find_world_shapefile"]:
        if hasattr(module, name):
            setattr(module, name, lambda: land)
