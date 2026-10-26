"""Plot Supplementary_Fig_S2 using the packaged numerical tables or supplied local inputs."""
from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_support"))
from figure_cli import prepare_figure as _prepare_figure, save_figure as _save_figure
from figure_cli import read_table as _read_table, model_tables as _model_tables
from figure_cli import configure_land_functions as _configure_land_functions

_figure_options = _prepare_figure('Supplementary_Fig_S2', 'external_inputs_required') if __name__ == "__main__" else None

from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
import argparse
from io import BytesIO
import json
import math
import os
from pathlib import Path
from typing import Iterable
os.environ.setdefault('WINDIR', 'external_inputs/Windows')
os.environ.setdefault('SystemRoot', 'external_inputs/Windows')
_MPL_CONFIG_DIR = Path(__file__).resolve().parents[2] / 'figure_outputs_q3' / '.matplotlib_cache'
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(_MPL_CONFIG_DIR))
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator, MultipleLocator
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, geometry_window
from rasterio.transform import from_bounds
from rasterio.warp import transform_geom
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'], 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': 7.0, 'font.weight': 'regular', 'axes.titleweight': 'regular', 'axes.labelweight': 'regular', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.65, 'xtick.major.width': 0.55, 'ytick.major.width': 0.55, 'xtick.major.size': 2.2, 'ytick.major.size': 2.2, 'legend.frameon': False, 'axes.unicode_minus': False, 'text.color': '#000000', 'axes.labelcolor': '#000000', 'axes.titlecolor': '#000000', 'xtick.color': '#000000', 'ytick.color': '#000000'})
PROJECT_ROOT = PACKAGE_ROOT
DATA_DIR = PROJECT_ROOT / '影响变量数据' / '环境风险' / '热暴露' / '全球降尺度LST'
OUT_DIR = output_dir('Supplementary_Fig_S2')
OUT_DIR.mkdir(parents=True, exist_ok=True)
CV_CSV = DATA_DIR / '5fold_CV_predictions.csv'
INDEP_CSV = DATA_DIR / 'independent_validation_predictions.csv'
MODIS_LST_TIF = DATA_DIR / 'MODIS.tif'
DOWNSCALED_LST_TIF = DATA_DIR / 'LSTdownscale.tif'
LANDSAT_OFFICIAL_LST_TIF = DATA_DIR / 'landsat_c2_l2_lst_2019_2024_median_2000m_C_mosaic.tif'
OUT_BASE = OUT_DIR / 'fig_lst_downscaling_validation'
REP_CITY_DIR = DATA_DIR.parent / '代表性城市'
CITY_MODIS_DIR = REP_CITY_DIR / '城市MODIS LST'
CITY_DOWNSCALED_DIR = REP_CITY_DIR / '城市降尺度LST'
CITY_OFFICIAL_DIR = REP_CITY_DIR / '城市官方LST'
CITY_CIRCLE_DIR = REP_CITY_DIR / '城市行政边界'
CITY_CLIP_DIR = OUT_DIR / 'lst_city_circle_clips'
CITY_CLIP_DIR.mkdir(parents=True, exist_ok=True)
CITY_GOOGLE_CACHE_DIR = OUT_DIR / 'lst_city_google_image_cache'
CITY_GOOGLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
CITY_GOOGLE_DIR_CANDIDATES = [REP_CITY_DIR / '城市Google影像', REP_CITY_DIR / '城市谷歌影像', REP_CITY_DIR / '城市高分辨率谷歌影像', REP_CITY_DIR / 'Google影像', REP_CITY_DIR / '谷歌影像', REP_CITY_DIR / '高分辨率谷歌影像']
CITY_GOOGLE_TILE_ZOOM_BY_CITY = {'Beijing': 11, 'Los_Angeles': 11, 'Rome': 11, 'Cape_Town': 12, 'Rio_de_Janeiro': 12, 'Singapore': 13}
CITY_GOOGLE_TILE_TIMEOUT = 2.5
CITY_ORDER = [{'key': 'Beijing', 'label': 'Beijing', 'zh': '北京市'}, {'key': 'Los_Angeles', 'label': 'Los Angeles', 'zh': '洛杉矶'}, {'key': 'Singapore', 'label': 'Singapore', 'zh': '新加坡'}, {'key': 'Rio_de_Janeiro', 'label': 'Rio de Janeiro', 'zh': '里约热内卢'}, {'key': 'Rome', 'label': 'Rome', 'zh': '罗马'}, {'key': 'Cape_Town', 'label': 'Cape Town', 'zh': '开普敦'}]
CITY_LST_LIMITS_C = {'Beijing': (21.5, 27.0), 'Los_Angeles': (24.7, 36.7), 'Singapore': (24.7, 33.7), 'Rio_de_Janeiro': (24.7, 33.7), 'Rome': (21.3, 29.3), 'Cape_Town': (22.5, 32.5)}
CITY_LST_LIMITS_K = {key: (limits[0] + 273.15, limits[1] + 273.15) for key, limits in CITY_LST_LIMITS_C.items()}
PRODUCT_TAGS = {'MODIS': 'modis', 'Downscaled': 'downscaled', 'Landsat': 'landsat'}
CITY_CLIP_NODATA = -9999.0
TEXT = '#000000'
MUTED = '#5F6B7A'
LIGHT = '#E9EBF1'
CV = '#3B82D6'
INDEP = '#F27C2B'
NEUTRAL = '#30343B'
FONT_PANEL = 9.2
FONT_TITLE = 7.7
FONT_LABEL = 7.0
FONT_TICK = 6.2
FONT_NOTE = 5.9
EXPORT_DPI = 600
CITY_MAP_TARGET_HEIGHT_PX = 529
CITY_GRID_BOTTOM = 0.04
CITY_GRID_TOP_PAD = 0.03
CITY_GRID_HEADER_H = 0.038
CITY_GRID_ROW_GAP = 0.006
CITY_GRID_COL_GAP = 0.012
CITY_GRID_CONTENT_X_SHIFT = -0.02
_CITY_CIRCLE_CACHE: dict[str, tuple[list[dict], Path]] = {}
_CITY_GOOGLE_ERRORS: list[dict] = []

def lst_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list('lst_distribution', ['#2C5AA8', '#8FB8E8', '#F6F1D3', '#FDB46C', '#B72C2C'])

def require_columns(df: pd.DataFrame, path: Path, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f'{path} is missing required columns: {missing}')

def load_validation_data(cv_path: Path, indep_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cv = pd.read_csv(resource(cv_path))
    indep = pd.read_csv(resource(indep_path))
    require_columns(cv, cv_path, ['fold', 'actual_LST', 'predicted_LST', 'residual', 'abs_error', 'sq_error'])
    require_columns(indep, indep_path, ['actual_LST', 'predicted_LST', 'residual', 'abs_error', 'sq_error'])
    cv = cv.copy()
    indep = indep.copy()
    cv['dataset'] = '5-fold CV'
    indep['dataset'] = 'Independent'
    for df in (cv, indep):
        df['actual_C'] = df['actual_LST'] - 273.15
        df['predicted_C'] = df['predicted_LST'] - 273.15
        df['mean_C'] = (df['actual_C'] + df['predicted_C']) / 2.0
        df['calc_residual'] = df['predicted_LST'] - df['actual_LST']
        if not np.allclose(df['calc_residual'], df['residual'], atol=1e-08):
            raise ValueError('Residual column is not predicted_LST - actual_LST.')
        if df[['actual_LST', 'predicted_LST', 'residual', 'abs_error']].isna().any().any():
            raise ValueError('Validation CSV contains missing values in key columns.')
    return (cv, indep)

def metrics(df: pd.DataFrame) -> dict[str, float]:
    y = df['actual_LST'].to_numpy(dtype=float)
    yhat = df['predicted_LST'].to_numpy(dtype=float)
    residual = yhat - y
    y_centered = y - y.mean()
    yhat_centered = yhat - yhat.mean()
    r_denom = float(np.sqrt(np.sum(y_centered ** 2) * np.sum(yhat_centered ** 2)))
    r = float(np.sum(y_centered * yhat_centered) / r_denom) if r_denom > 0 else float('nan')
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {'n': int(len(df)), 'r': r, 'r2': float(1.0 - ss_res / ss_tot), 'rmse': float(np.sqrt(np.mean(residual ** 2))), 'mae': float(np.mean(np.abs(residual))), 'bias': float(np.mean(residual)), 'q01': float(np.quantile(residual, 0.01)), 'q05': float(np.quantile(residual, 0.05)), 'q50': float(np.quantile(residual, 0.5)), 'q95': float(np.quantile(residual, 0.95)), 'q99': float(np.quantile(residual, 0.99)), 'within_1k': float(np.mean(np.abs(residual) <= 1.0)), 'within_2k': float(np.mean(np.abs(residual) <= 2.0)), 'within_3k': float(np.mean(np.abs(residual) <= 3.0)), 'within_5k': float(np.mean(np.abs(residual) <= 5.0)), 'within_10k': float(np.mean(np.abs(residual) <= 10.0))}

def per_fold_metrics(cv: pd.DataFrame, indep: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for fold, group in cv.groupby('fold', sort=True):
        row = metrics(group)
        row['split'] = f'Fold {int(fold)}'
        rows.append(row)
    row = metrics(indep)
    row['split'] = 'Indep.'
    rows.append(row)
    return pd.DataFrame(rows)

def set_clean_axis(ax: mpl.axes.Axes) -> None:
    ax.grid(False)
    ax.tick_params(labelsize=FONT_TICK, colors=TEXT)
    for spine in ax.spines.values():
        spine.set_color(TEXT)
        spine.set_linewidth(0.65)

def add_panel_label(ax: mpl.axes.Axes, letter: str) -> None:
    ax._panel_letter = letter

def set_title(ax: mpl.axes.Axes, letter: str, title: str) -> None:
    ax._panel_letter = letter
    ax._panel_title = title
    ax.set_title('')

def add_panel_header(fig: mpl.figure.Figure, bbox: mpl.transforms.Bbox, letter: str, title: str, y_pad: float=0.01, letter_dx: float=0.022) -> None:
    y = bbox.y1 + y_pad
    fig.text(bbox.x0 - letter_dx, y, letter, ha='left', va='bottom', fontsize=FONT_PANEL, fontweight='bold', color=TEXT)
    fig.text(bbox.x0, y, title, ha='left', va='bottom', fontsize=FONT_TITLE, color=TEXT)

def add_axis_panel_header(fig: mpl.figure.Figure, ax: mpl.axes.Axes) -> None:
    add_panel_header(fig, ax.get_position(), str(getattr(ax, '_panel_letter')), str(getattr(ax, '_panel_title')))

def draw_summary_panel(ax: mpl.axes.Axes, cv_metrics: dict[str, float], indep_metrics: dict[str, float]) -> None:
    set_title(ax, 'a', 'Validation summary')
    ax.set_axis_off()
    rows = [('5-fold CV', CV, cv_metrics), ('Independent', INDEP, indep_metrics)]
    cols = [('n', lambda m: f"{int(m['n']):,}"), ('R2', lambda m: f"{m['r2']:.3f}"), ('RMSE', lambda m: f"{m['rmse']:.2f}"), ('MAE', lambda m: f"{m['mae']:.2f}"), ('Bias', lambda m: f"{m['bias']:+.2f}"), ('<=5 K', lambda m: f"{m['within_5k'] * 100:.1f}%")]
    x0, y0 = (0.02, 0.78)
    label_w = 0.3
    cell_w = (0.96 - label_w) / len(cols)
    row_h = 0.24
    ax.text(x0, 0.95, 'Internal 5-fold CV and independent holdout validation.', transform=ax.transAxes, fontsize=FONT_NOTE, ha='left', va='top', color=TEXT, wrap=True)
    for j, (label, _) in enumerate(cols):
        ax.text(x0 + label_w + j * cell_w + cell_w / 2, y0, label, transform=ax.transAxes, ha='center', va='center', fontsize=FONT_NOTE, color=TEXT)
    for i, (name, color, met) in enumerate(rows):
        y = y0 - (i + 1) * row_h
        ax.scatter([x0 + 0.0175], [y], transform=ax.transAxes, marker='s', s=48, color=color, alpha=0.9, linewidths=0)
        ax.text(x0 + 0.05, y, name, transform=ax.transAxes, ha='left', va='center', fontsize=FONT_LABEL, color=TEXT)
        for j, (_, fmt) in enumerate(cols):
            ax.text(x0 + label_w + j * cell_w + cell_w / 2, y, fmt(met), transform=ax.transAxes, ha='center', va='center', fontsize=FONT_LABEL, color=TEXT)
    ax.text(x0, 0.08, 'Residual = predicted LST - observed LST; error magnitudes are in K.', transform=ax.transAxes, fontsize=FONT_NOTE, ha='left', va='bottom', color=TEXT)

def density_cmap(color: str, name: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, ['#FFFFFF', '#EAF2FB', color])

def read_lst_preview(path: Path, max_width: int=3200, max_height: int=1500) -> dict:
    if not resource(path).exists():
        raise FileNotFoundError(path)
    with rasterio.open(resource(path)) as src:
        if src.crs is None or src.crs.to_epsg() != 4326:
            raise ValueError(f'{path} must be WGS 1984 / EPSG:4326; found {src.crs}.')
        scale = max(1, int(np.ceil(max(src.width / max_width, src.height / max_height))))
        out_width = max(1, int(np.ceil(src.width / scale)))
        out_height = max(1, int(np.ceil(src.height / scale)))
        data = src.read(1, out_shape=(out_height, out_width), masked=True, resampling=Resampling.average).astype('float32')
        arr = np.asarray(data.filled(np.nan), dtype='float32')
        arr[arr > 1e+20] = np.nan
        arr[arr < -100] = np.nan
        finite = np.isfinite(arr)
        unit = 'deg C'
        if finite.any() and float(np.nanmedian(arr)) > 100.0:
            arr = arr - 273.15
            unit = 'K converted to deg C'
        arr[(arr < -60.0) | (arr > 70.0)] = np.nan
        bounds = src.bounds
        return {'path': str(path), 'array': arr, 'extent': (bounds.left, bounds.right, bounds.bottom, bounds.top), 'crs': str(src.crs), 'unit': unit, 'source_width': int(src.width), 'source_height': int(src.height), 'preview_width': int(out_width), 'preview_height': int(out_height)}

def find_city_product_file(folder: Path, city: dict, suffixes: tuple[str, ...]) -> Path | None:
    if not resource(folder).exists():
        return None
    needles = {str(city['key']).lower(), str(city['key']).replace('_', ' ').lower(), str(city['label']).lower(), str(city['zh']).lower()}
    files = [p for p in resource_glob(folder, '*') if resource(p).is_file() and p.suffix.lower() in suffixes]
    for path in sorted(files):
        haystack = path.stem.lower()
        if any((needle in haystack for needle in needles)):
            return path
    return None

def find_city_google_image(city: dict) -> Path | None:
    image_suffixes = ('.png', '.jpg', '.jpeg', '.webp', '.tif', '.tiff')
    for folder in CITY_GOOGLE_DIR_CANDIDATES:
        found = find_city_product_file(folder, city, image_suffixes)
        if found is not None:
            return found
    if resource(REP_CITY_DIR).exists():
        needles = {str(city['key']).lower(), str(city['key']).replace('_', ' ').lower(), str(city['label']).lower(), str(city['zh']).lower()}
        image_words = ('google', '谷歌', 'satellite', 'image', 'imagery', '影像', '高分辨率')
        for path in sorted(resource_glob(REP_CITY_DIR, '*')):
            if not resource(path).is_file() or path.suffix.lower() not in image_suffixes:
                continue
            haystack = f'{path.parent.name} {path.stem}'.lower()
            if any((needle in haystack for needle in needles)) and any((word in haystack for word in image_words)):
                return path
    return None

def find_city_circle_file(city: dict) -> Path | None:
    if not resource(CITY_CIRCLE_DIR).exists():
        return None
    exact_names = [f"{city['zh']}C.shp", f"{city['label']}C.shp", f"{city['key']}C.shp"]
    for name in exact_names:
        path = CITY_CIRCLE_DIR / name
        if resource(path).exists():
            return path
    needles = {f"{city['zh']}c".lower(), f"{city['label']}c".lower(), f"{city['key']}c".lower(), f"{str(city['key']).replace('_', ' ')}c".lower()}
    for path in sorted(resource_glob(CITY_CIRCLE_DIR, '*C.shp')):
        if path.stem.lower() in needles:
            return path
    return None

def read_city_circle_geometries(city: dict, dst_crs: CRS) -> tuple[list[dict], Path]:
    import shapefile
    circle_path = find_city_circle_file(city)
    if circle_path is None:
        raise FileNotFoundError(f"Missing circular mask shapefile for {city['label']} in {CITY_CIRCLE_DIR}")
    reader = shapefile.Reader(resource(str(circle_path)))
    geoms = [shape.__geo_interface__ for shape in reader.shapes()]
    prj_path = circle_path.with_suffix('.prj')
    src_crs = CRS.from_wkt(resource(prj_path).read_text(encoding='utf-8')) if resource(prj_path).exists() else CRS.from_epsg(4326)
    if not geoms:
        raise ValueError(f'{circle_path} contains no valid geometries.')
    if src_crs != dst_crs:
        geoms = [transform_geom(src_crs, dst_crs, geom) for geom in geoms]
    return (geoms, circle_path)

def get_city_circle_geometries_4326(city: dict) -> tuple[list[dict], Path]:
    key = str(city['key'])
    if key not in _CITY_CIRCLE_CACHE:
        _CITY_CIRCLE_CACHE[key] = read_city_circle_geometries(city, CRS.from_epsg(4326))
    return _CITY_CIRCLE_CACHE[key]

def iter_geojson_exterior_rings(geom: dict):
    geom_type = geom.get('type')
    coordinates = geom.get('coordinates', [])
    if geom_type == 'Polygon':
        if coordinates:
            yield coordinates[0]
    elif geom_type == 'MultiPolygon':
        for polygon in coordinates:
            if polygon:
                yield polygon[0]

def iter_geojson_boundary_rings(geom: dict):
    geom_type = geom.get('type')
    coordinates = geom.get('coordinates', [])
    if geom_type == 'Polygon':
        for ring in coordinates:
            yield ring
    elif geom_type == 'MultiPolygon':
        for polygon in coordinates:
            for ring in polygon:
                yield ring

def city_circle_extent(city: dict, pad_ratio: float=0.03) -> tuple[float, float, float, float]:
    geoms, _ = get_city_circle_geometries_4326(city)
    xs: list[float] = []
    ys: list[float] = []
    for geom in geoms:
        for ring in iter_geojson_exterior_rings(geom):
            for x, y, *_ in ring:
                xs.append(float(x))
                ys.append(float(y))
    if not xs or not ys:
        raise ValueError(f"Could not derive circular extent for {city['label']}.")
    left, right = (min(xs), max(xs))
    bottom, top = (min(ys), max(ys))
    pad_x = max((right - left) * pad_ratio, 1e-06)
    pad_y = max((top - bottom) * pad_ratio, 1e-06)
    return (left - pad_x, right + pad_x, bottom - pad_y, top + pad_y)

def square_extent(extent: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    left, right, bottom, top = extent
    cx = (left + right) / 2.0
    cy = (bottom + top) / 2.0
    side = max(right - left, top - bottom)
    return (cx - side / 2.0, cx + side / 2.0, cy - side / 2.0, cy + side / 2.0)

def city_circle_square_extent(city: dict, pad_ratio: float=0.05) -> tuple[float, float, float, float]:
    return square_extent(city_circle_extent(city, pad_ratio=pad_ratio))

def draw_city_circle_boundary(ax: mpl.axes.Axes, city: dict) -> None:
    geoms, _ = get_city_circle_geometries_4326(city)
    for geom in geoms:
        for ring in iter_geojson_boundary_rings(geom):
            if len(ring) < 2:
                continue
            xs = [float(point[0]) for point in ring]
            ys = [float(point[1]) for point in ring]
            ax.plot(xs, ys, color=TEXT, lw=0.42, alpha=0.88, zorder=30, clip_on=False)

def mask_rgb_to_city_circle(rgb: np.ndarray, extent: tuple[float, float, float, float], city: dict) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        return rgb
    geoms, _ = get_city_circle_geometries_4326(city)
    height, width = rgb.shape[:2]
    left, right, bottom, top = extent
    transform = from_bounds(left, bottom, right, top, width, height)
    inside = geometry_mask(geoms, out_shape=(height, width), transform=transform, invert=True)
    rgba = np.dstack([rgb[..., :3], inside.astype('float32')])
    return rgba

def lonlat_to_global_pixel(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    lat = min(max(float(lat), -85.05112878), 85.05112878)
    sin_lat = math.sin(math.radians(lat))
    scale = 256 * 2 ** zoom
    x = (float(lon) + 180.0) / 360.0 * scale
    y = (0.5 - math.log((1.0 + sin_lat) / (1.0 - sin_lat)) / (4.0 * math.pi)) * scale
    return (x, y)

def city_google_tile_zoom(city: dict) -> int:
    return CITY_GOOGLE_TILE_ZOOM_BY_CITY.get(str(city['key']), 11)

def fetch_esri_satellite_preview(city: dict, max_size: int=900) -> dict | None:
    zoom = city_google_tile_zoom(city)
    cache_path = CITY_GOOGLE_CACHE_DIR / f"{city['key']}_esri_world_imagery_z{zoom}.png"
    extent = city_circle_square_extent(city, pad_ratio=0.05)
    if resource(cache_path).exists():
        image = Image.open(resource(cache_path)).convert('RGB')
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        arr = np.asarray(image, dtype='float32') / 255.0
        return {'array': arr, 'extent': extent, 'source': str(cache_path)}
    left, right, bottom, top = extent
    px_left, py_top = lonlat_to_global_pixel(left, top, zoom)
    px_right, py_bottom = lonlat_to_global_pixel(right, bottom, zoom)
    tile_left = int(math.floor(px_left / 256))
    tile_right = int(math.floor(px_right / 256))
    tile_top = int(math.floor(py_top / 256))
    tile_bottom = int(math.floor(py_bottom / 256))
    tile_count = max(0, tile_right - tile_left + 1) * max(0, tile_bottom - tile_top + 1)
    if tile_count <= 0 or tile_count > 180:
        _CITY_GOOGLE_ERRORS.append({'city': str(city['label']), 'source': 'Esri World Imagery', 'error': f'tile count {tile_count}'})
        return None
    mosaic = Image.new('RGB', ((tile_right - tile_left + 1) * 256, (tile_bottom - tile_top + 1) * 256), 'white')
    fetched = 0
    raise FileNotFoundError('A local imagery file is required for this panel.')

def city_clip_path(city: dict, product: str) -> Path:
    return CITY_CLIP_DIR / f"{city['key']}_{PRODUCT_TAGS[product]}_circle_clip.tif"

def shapefile_component_paths(path: Path) -> list[Path]:
    suffixes = ['.shp', '.shx', '.dbf', '.prj', '.cpg']
    return [path.with_suffix(suffix) for suffix in suffixes if resource(path.with_suffix(suffix)).exists()]

def clip_is_current(clip_path: Path, source_path: Path, circle_path: Path) -> bool:
    if not resource(clip_path).exists():
        return False
    clip_mtime = clip_path.stat().st_mtime
    dependencies = [source_path, *shapefile_component_paths(circle_path)]
    return all((clip_mtime >= dep.stat().st_mtime for dep in dependencies if resource(dep).exists()))

def clip_city_lst_raster(source_path: Path, city: dict, product: str, overwrite: bool=False) -> dict:
    clip_path = city_clip_path(city, product)
    with rasterio.open(resource(source_path)) as src:
        if src.crs is None or src.crs.to_epsg() != 4326:
            raise ValueError(f'{source_path} must be WGS 1984 / EPSG:4326; found {src.crs}.')
        geoms, circle_path = read_city_circle_geometries(city, src.crs)
        if not overwrite and clip_is_current(clip_path, source_path, circle_path):
            with rasterio.open(resource(clip_path)) as clipped:
                arr = clipped.read(1, masked=True)
                return {'city': str(city['label']), 'product': product, 'source_raster': str(source_path), 'clipped_raster': str(clip_path), 'circle_mask': str(circle_path), 'crs': str(clipped.crs), 'width': int(clipped.width), 'height': int(clipped.height), 'finite_pixels': int(np.ma.count(arr)), 'recreated': False}
        window = geometry_window(src, geoms)
        data = src.read(1, window=window, masked=True).astype('float32')
        out_transform = src.window_transform(window)
        arr = np.asarray(data.filled(np.nan), dtype='float32')
        inside_circle = geometry_mask(geoms, out_shape=arr.shape, transform=out_transform, invert=True)
        arr[~inside_circle] = np.nan
        arr[arr > 1e+20] = np.nan
        arr[arr < -100] = np.nan
        finite_pixels = int(np.isfinite(arr).sum())
        out_arr = np.where(np.isfinite(arr), arr, CITY_CLIP_NODATA).astype('float32')
        profile = src.profile.copy()
        profile.update({'driver': 'GTiff', 'height': int(out_arr.shape[0]), 'width': int(out_arr.shape[1]), 'count': 1, 'dtype': 'float32', 'crs': src.crs, 'transform': out_transform, 'nodata': CITY_CLIP_NODATA, 'compress': 'lzw'})
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(resource(clip_path), 'w', **profile) as dst:
        dst.write(out_arr, 1)
    return {'city': str(city['label']), 'product': product, 'source_raster': str(source_path), 'clipped_raster': str(clip_path), 'circle_mask': str(circle_path), 'crs': 'EPSG:4326', 'width': int(out_arr.shape[1]), 'height': int(out_arr.shape[0]), 'finite_pixels': finite_pixels, 'recreated': True}

def prepare_city_circle_clips(overwrite: bool=False) -> list[dict]:
    product_sources = [('MODIS', CITY_MODIS_DIR), ('Downscaled', CITY_DOWNSCALED_DIR), ('Landsat', CITY_OFFICIAL_DIR)]
    records = []
    for city in CITY_ORDER:
        for product, folder in product_sources:
            source_path = find_city_product_file(folder, city, ('.tif', '.tiff'))
            if source_path is None:
                records.append({'city': str(city['label']), 'product': product, 'source_raster': None, 'clipped_raster': str(city_clip_path(city, product)), 'circle_mask': str(find_city_circle_file(city)), 'error': f'missing source raster in {folder}'})
                continue
            records.append(clip_city_lst_raster(source_path, city, product, overwrite=overwrite))
    return records

def read_city_lst_preview(path: Path, max_size: int=360) -> dict:
    with rasterio.open(resource(path)) as src:
        if src.crs is None or src.crs.to_epsg() != 4326:
            raise ValueError(f'{path} must be WGS 1984 / EPSG:4326; found {src.crs}.')
        scale = max(1, int(np.ceil(max(src.width, src.height) / max_size)))
        out_width = max(1, int(np.ceil(src.width / scale)))
        out_height = max(1, int(np.ceil(src.height / scale)))
        data = src.read(1, out_shape=(out_height, out_width), masked=True, resampling=Resampling.average).astype('float32')
        arr = np.asarray(data.filled(np.nan), dtype='float32')
        arr[arr > 1e+20] = np.nan
        arr[arr < -100] = np.nan
        finite = np.isfinite(arr)
        if finite.any() and float(np.nanmedian(arr)) > 100.0:
            arr = arr - 273.15
        arr[(arr < -60.0) | (arr > 70.0)] = np.nan
        bounds = src.bounds
        return {'array': arr, 'path': str(path), 'extent': (bounds.left, bounds.right, bounds.bottom, bounds.top), 'crs': str(src.crs), 'masked_finite_pixels': int(np.isfinite(arr).sum()), 'source_width': int(src.width), 'source_height': int(src.height), 'preview_width': int(out_width), 'preview_height': int(out_height)}

def read_city_rgb_image(path: Path, max_size: int=360) -> np.ndarray:
    if path.suffix.lower() in ('.tif', '.tiff'):
        with rasterio.open(resource(path)) as src:
            count = min(src.count, 3)
            scale = max(1, int(np.ceil(max(src.width, src.height) / max_size)))
            out_width = max(1, int(np.ceil(src.width / scale)))
            out_height = max(1, int(np.ceil(src.height / scale)))
            data = src.read(list(range(1, count + 1)), out_shape=(count, out_height, out_width), masked=True, resampling=Resampling.bilinear)
            arr = np.moveaxis(np.asarray(data.filled(0), dtype='float32'), 0, -1)
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            lo, hi = np.nanpercentile(arr, [2, 98])
            if hi > lo:
                arr = np.clip((arr - lo) / (hi - lo), 0, 1)
            return arr
    image = Image.open(resource(path)).convert('RGB')
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype='float32') / 255.0

def read_city_google_preview(city: dict, max_size: int=900) -> dict | None:
    extent = city_circle_square_extent(city, pad_ratio=0.05)
    path = find_city_google_image(city)
    if path is not None:
        arr = read_city_rgb_image(path, max_size=max_size)
        return {'array': arr, 'extent': extent, 'source': str(path)}
    return fetch_esri_satellite_preview(city, max_size=max_size)

def draw_city_temperature_legend(fig: mpl.figure.Figure, x0: float, y0: float, width: float, height: float, cmap: LinearSegmentedColormap, vmin: float, vmax: float) -> None:
    ax_leg = fig.add_axes([x0, y0, width, height])
    gradient = np.linspace(0.0, 1.0, 128, dtype='float32')[None, :]
    ax_leg.imshow(gradient, cmap=cmap, aspect='auto', origin='lower', extent=[vmin, vmax, 0, 1])
    ax_leg.set_yticks([])
    ax_leg.set_xticks([vmin, vmax])
    ax_leg.set_xticklabels([f'{vmin:.1f}', f'{vmax:.1f}'], fontsize=FONT_NOTE * 0.86, color=TEXT)
    ax_leg.tick_params(axis='x', length=1.4, width=0.35, pad=0.6, colors=TEXT)
    for spine in ax_leg.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.35)
        spine.set_color(TEXT)

def draw_city_lsts_grid(fig: mpl.figure.Figure, bounds: tuple[float, float, float, float], cmap: LinearSegmentedColormap, vmin: float, vmax: float) -> dict:
    left, bottom, right, top = bounds
    row_labels = ['MODIS', 'Downscaled', 'Google image']
    city_lst_layers = []
    city_google_layers = []
    row_label_w = 0.057
    header_h = CITY_GRID_HEADER_H
    col_gap = CITY_GRID_COL_GAP
    row_gap = CITY_GRID_ROW_GAP
    grid_left = left + row_label_w
    grid_top = top - header_h
    grid_width = right - grid_left
    grid_height = grid_top - bottom
    cell_w = (grid_width - 5 * col_gap) / 6.0
    cell_h = (grid_height - 3 * row_gap) / 4.0
    fig_ratio = fig.get_figheight() / fig.get_figwidth()
    map_w = cell_h * fig_ratio
    map_h = cell_h
    if map_w > cell_w:
        map_w = cell_w
        map_h = cell_w / fig_ratio
    fig.text(left + row_label_w * 0.46, top - 0.007, 'City', ha='center', va='top', fontsize=FONT_LABEL, color=TEXT)
    for col, city in enumerate(CITY_ORDER):
        cell_x0 = grid_left + col * (cell_w + col_gap)
        map_x0 = cell_x0 + (cell_w - map_w) / 2.0 + CITY_GRID_CONTENT_X_SHIFT
        fig.text(map_x0 + map_w / 2, top - 0.006, str(city['label']), ha='center', va='top', fontsize=FONT_LABEL, color=TEXT)
        city_vmin, city_vmax = CITY_LST_LIMITS_C[str(city['key'])]
        draw_city_temperature_legend(fig, map_x0 + map_w * 0.16, top - 0.025, map_w * 0.68, 0.0038, cmap, city_vmin, city_vmax)
    for row, label in enumerate(row_labels):
        y0 = grid_top - (row + 1) * cell_h - row * row_gap
        fig.text(left + row_label_w * 0.46, y0 + cell_h / 2, label, ha='center', va='center', fontsize=FONT_LABEL, color=TEXT, rotation=90, rotation_mode='anchor')
        for col, city in enumerate(CITY_ORDER):
            cell_x0 = grid_left + col * (cell_w + col_gap)
            map_x0 = cell_x0 + (cell_w - map_w) / 2.0 + CITY_GRID_CONTENT_X_SHIFT
            map_y0 = y0 + (cell_h - map_h) / 2.0
            display_extent = city_circle_square_extent(city, pad_ratio=0.05)
            ax = fig.add_axes([map_x0, map_y0, map_w, map_h])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_facecolor('#FFFFFF')
            if label != 'Google image':
                product = row_labels[row]
                path = city_clip_path(city, product)
                if not resource(path).exists():
                    ax.set_xlim(display_extent[0], display_extent[1])
                    ax.set_ylim(display_extent[2], display_extent[3])
                    ax.set_aspect('equal', adjustable='box')
                    ax.text(0.5, 0.5, 'missing\nclip', transform=ax.transAxes, ha='center', va='center', fontsize=FONT_LABEL, color=TEXT)
                else:
                    preview = read_city_lst_preview(path)
                    city_lst_layers.append({'city': str(city['label']), 'product': product, 'raster': preview['path'], 'crs': preview['crs'], 'masked_finite_pixels': preview['masked_finite_pixels'], 'source_width': preview['source_width'], 'source_height': preview['source_height']})
                    city_vmin, city_vmax = CITY_LST_LIMITS_C[str(city['key'])]
                    ax.imshow(preview['array'], extent=preview['extent'], origin='upper', cmap=cmap, vmin=city_vmin, vmax=city_vmax, interpolation='nearest', rasterized=True)
                    ax.set_xlim(display_extent[0], display_extent[1])
                    ax.set_ylim(display_extent[2], display_extent[3])
                    ax.set_aspect('equal', adjustable='box')
            else:
                google = read_city_google_preview(city)
                if google is None:
                    ax.set_xlim(display_extent[0], display_extent[1])
                    ax.set_ylim(display_extent[2], display_extent[3])
                    ax.set_aspect('equal', adjustable='box')
                    ax.set_facecolor('#F7F8FA')
                    ax.text(0.5, 0.5, 'image\npending', transform=ax.transAxes, ha='center', va='center', fontsize=FONT_LABEL, color=TEXT, linespacing=0.92)
                else:
                    rgb = mask_rgb_to_city_circle(google['array'], google['extent'], city)
                    ax.imshow(rgb, extent=google['extent'], origin='upper', interpolation='lanczos', rasterized=True)
                    ax.set_xlim(display_extent[0], display_extent[1])
                    ax.set_ylim(display_extent[2], display_extent[3])
                    ax.set_aspect('equal', adjustable='box')
                    city_google_layers.append({'city': str(city['label']), 'source': str(google['source']), 'extent': [float(v) for v in google['extent']], 'tile_zoom': city_google_tile_zoom(city)})
            draw_city_circle_boundary(ax, city)
    return {'city_grid_bounds': [left, bottom, right, top], 'city_circle_mask_dir': str(CITY_CIRCLE_DIR), 'city_lst_layers': city_lst_layers, 'city_google_layers': city_google_layers, 'google_image_dirs_checked': [str(path) for path in CITY_GOOGLE_DIR_CANDIDATES], 'google_image_tile_source': 'Esri World Imagery', 'google_image_errors': _CITY_GOOGLE_ERRORS, 'city_map_axis_size_fraction': [map_w, map_h], 'city_map_target_height_px': CITY_MAP_TARGET_HEIGHT_PX}

def shared_lst_limits(previews: list[dict]) -> tuple[float, float]:
    values = []
    for preview in previews:
        arr = preview['array']
        finite = arr[np.isfinite(arr)]
        if finite.size:
            values.append(finite)
    if not values:
        return (-15.0, 45.0)
    merged = np.concatenate(values)
    lo, hi = np.nanpercentile(merged, [2, 98])
    lo = float(np.floor(lo / 5.0) * 5.0)
    hi = float(np.ceil(hi / 5.0) * 5.0)
    if np.isclose(lo, hi):
        lo -= 5.0
        hi += 5.0
    return (lo, hi)

def draw_lst_distribution_map(ax: mpl.axes.Axes, preview: dict, title: str, cmap: LinearSegmentedColormap, vmin: float, vmax: float, show_crs: bool=False) -> mpl.image.AxesImage:
    image = ax.imshow(preview['array'], extent=preview['extent'], origin='upper', cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest', rasterized=True)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('#FFFFFF')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(-0.075, 1.06, title, transform=ax.transAxes, ha='left', va='top', fontsize=5.8, color=TEXT, clip_on=False, bbox=dict(facecolor='white', edgecolor='none', alpha=0.72, pad=1.1))
    if show_crs:
        ax.text(0.98, 0.08, 'WGS 1984 / EPSG:4326', transform=ax.transAxes, ha='right', va='bottom', fontsize=4.8, color=TEXT)
    return image

def draw_lst_density_inset(ax: mpl.axes.Axes, preview: dict, xlim: tuple[float, float], label_y: float=1.1) -> None:
    values = preview['array']
    values = values[np.isfinite(values)]
    values = values[(values >= xlim[0]) & (values <= xlim[1])]
    if values.size < 50:
        return
    x, den = smoothed_density(values, xlim, bins=150)
    if not np.isfinite(den).any() or float(np.nanmax(den)) <= 0:
        return
    den = den / float(np.nanmax(den))
    inset = ax.inset_axes([-0.04, 0.185, 0.235, 0.26], transform=ax.transAxes)
    inset.set_facecolor((1, 1, 1, 0.78))
    inset.fill_between(x, 0, den, color='#E1EEFA', alpha=1.0, linewidth=0)
    inset.set_xlim(xlim)
    inset.set_ylim(0, 1.08)
    inset.set_yticks([])
    inset.set_xticks([0, 20, 40])
    inset.tick_params(axis='x', labelsize=4.8, length=1.6, width=0.4, pad=1.0, colors=TEXT)
    inset.set_xlabel('LST (deg C)', fontsize=4.8, labelpad=0.7, color=TEXT)
    inset.text(-0.07, label_y, 'pixel density', transform=inset.transAxes, ha='left', va='top', fontsize=4.8, color=TEXT, clip_on=False)
    for side in ('left', 'right', 'top'):
        inset.spines[side].set_visible(False)
    inset.spines['bottom'].set_linewidth(0.45)
    inset.spines['bottom'].set_color(TEXT)

def draw_observed_predicted(ax: mpl.axes.Axes, df: pd.DataFrame, title: str, letter: str, color: str, lim: tuple[float, float], met: dict[str, float]) -> None:
    set_title(ax, letter, title)
    hb = ax.hexbin(df['actual_C'], df['predicted_C'], gridsize=52, extent=(lim[0], lim[1], lim[0], lim[1]), mincnt=1, bins='log', cmap=density_cmap(color, f'density_{letter}'), linewidths=0, rasterized=True)
    ax.plot(lim, lim, ls='--', lw=0.75, color=NEUTRAL, alpha=0.7)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Observed LST (deg C)', fontsize=FONT_LABEL, labelpad=1.8)
    ax.set_ylabel('Predicted LST (deg C)', fontsize=FONT_LABEL, labelpad=1.8)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    set_clean_axis(ax)
    text = f"n={int(met['n']):,}\nR2={met['r2']:.3f}\nRMSE={met['rmse']:.2f} K\nBias={met['bias']:+.2f} K"
    ax.text(0.04, 0.96, text, transform=ax.transAxes, ha='left', va='top', fontsize=FONT_NOTE, linespacing=1.6, color=TEXT)
    grad = np.linspace(0, 1, 160, dtype='float32').reshape(1, -1)
    gx0 = lim[0] + 0.72 * (lim[1] - lim[0])
    gx1 = lim[0] + 0.96 * (lim[1] - lim[0])
    gy0 = lim[0] + 0.075 * (lim[1] - lim[0])
    gy1 = lim[0] + 0.115 * (lim[1] - lim[0])
    ax.imshow(grad, extent=(gx0, gx1, gy0, gy1), origin='lower', cmap=hb.cmap, aspect='auto', interpolation='nearest', zorder=5)
    ax.plot([gx0, gx1, gx1, gx0, gx0], [gy0, gy0, gy1, gy1, gy0], color=TEXT, lw=0.45, zorder=6)
    ax.text((gx0 + gx1) / 2.0, gy1 + 0.18, 'point density', ha='center', va='bottom', fontsize=5.0, color=TEXT, zorder=6)
    tick_y0 = gy0
    tick_y1 = gy0 - 0.45
    label_y = tick_y1 - 0.16
    for tick_x, tick_label in ((gx0, 'Low'), (gx1, 'High')):
        ax.plot([tick_x, tick_x], [tick_y0, tick_y1], color=TEXT, lw=0.42, zorder=6)
        ax.text(tick_x, label_y, tick_label, ha='center', va='top', fontsize=4.8, color=TEXT, zorder=6)
    ax.set_xlim(lim)
    ax.set_ylim(lim)

def column_scaled_matrix(values: pd.DataFrame) -> np.ndarray:
    arr = values.to_numpy(dtype=float)
    scaled = np.zeros_like(arr)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        lo, hi = (np.nanmin(col), np.nanmax(col))
        if np.isclose(lo, hi):
            scaled[:, j] = 0.5
        else:
            scaled[:, j] = (col - lo) / (hi - lo)
        if values.columns[j] in {'RMSE', 'MAE', '|Bias|'}:
            scaled[:, j] = 1.0 - scaled[:, j]
    return scaled

def draw_fold_matrix(ax: mpl.axes.Axes, fold_df: pd.DataFrame, letter: str='b') -> None:
    set_title(ax, letter, 'Fold-level skill stability')
    show = pd.DataFrame({'R2': fold_df['r2'].to_numpy(dtype=float), 'RMSE': fold_df['rmse'].to_numpy(dtype=float), 'MAE': fold_df['mae'].to_numpy(dtype=float), '|Bias|': fold_df['bias'].abs().to_numpy(dtype=float)}, index=fold_df['split'].to_list())
    cmap = LinearSegmentedColormap.from_list('skill', ['#F7F0E6', '#DCE6F2', '#3B82D6'])
    ax.imshow(column_scaled_matrix(show), cmap=cmap, vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(np.arange(show.shape[1]))
    ax.set_yticks(np.arange(show.shape[0]))
    ax.set_xticklabels(show.columns, fontsize=FONT_TICK, rotation=35, ha='right')
    ax.set_yticklabels(show.index, fontsize=FONT_TICK)
    ax.tick_params(length=0)
    for i in range(show.shape[0]):
        for j in range(show.shape[1]):
            val = show.iloc[i, j]
            label = f'{val:.3f}' if show.columns[j] == 'R2' else f'{val:.2f}'
            ax.text(j, i, label, ha='center', va='center', fontsize=5.6, color=TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, show.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, show.shape[0], 1), minor=True)
    ax.grid(which='minor', color='#FFFFFF', linewidth=0.8)
    ax.tick_params(which='minor', bottom=False, left=False)

def smoothed_density(data: np.ndarray, xlim: tuple[float, float], bins: int=180) -> tuple[np.ndarray, np.ndarray]:
    counts, edges = np.histogram(data, bins=bins, range=xlim, density=True)
    x = (edges[:-1] + edges[1:]) / 2.0
    kernel_x = np.linspace(-3, 3, 31)
    kernel = np.exp(-0.5 * kernel_x ** 2)
    kernel = kernel / kernel.sum()
    y = np.convolve(counts, kernel, mode='same')
    return (x, y)

def draw_residual_distribution(ax: mpl.axes.Axes, cv: pd.DataFrame, indep: pd.DataFrame, letter: str='e') -> None:
    set_title(ax, letter, 'Residual distribution')
    xlim = (-12.0, 12.0)
    rows = [('5-fold CV', CV, cv), ('Independent', INDEP, indep)]
    for idx, (label, color, df) in enumerate(rows):
        ybase = 1 - idx
        x, den = smoothed_density(df['residual'].to_numpy(dtype=float), xlim)
        den = den / den.max() * 0.55
        ax.fill_between(x, ybase, ybase + den, color=color, alpha=0.22, linewidth=0)
        ax.plot(x, ybase + den, color=color, lw=1.15)
        med = float(np.median(df['residual']))
        q05, q95 = np.quantile(df['residual'], [0.05, 0.95])
        ax.plot([q05, q95], [ybase - 0.08, ybase - 0.08], color=color, lw=1.1)
        ax.plot([med], [ybase - 0.08], marker='o', ms=3.1, color=color, mec='white', mew=0.3)
        ax.text(xlim[0] + 0.2, ybase + 0.11, label, ha='left', va='center', fontsize=FONT_LABEL, color=TEXT)
    ax.set_xlim(xlim)
    ax.set_ylim(-0.35, 1.72)
    ax.plot([0, 0], [-0.35, 1.72], color=NEUTRAL, lw=0.75, ls='--', alpha=0.75)
    ax.set_yticks([])
    ax.set_xlabel('Residual (K)', fontsize=FONT_LABEL, labelpad=1.8)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    set_clean_axis(ax)
    ax.spines['left'].set_visible(False)

def bin_stats(df: pd.DataFrame, x_col: str, y_col: str, edges: np.ndarray) -> pd.DataFrame:
    sub = df[[x_col, y_col]].dropna().copy()
    sub['bin'] = pd.cut(sub[x_col], bins=edges, include_lowest=True)
    rows = []
    for interval, group in sub.groupby('bin', observed=True):
        if len(group) < 12:
            continue
        rows.append({'x': float(group[x_col].median()), 'median': float(group[y_col].median()), 'q25': float(group[y_col].quantile(0.25)), 'q75': float(group[y_col].quantile(0.75)), 'n': int(len(group))})
    return pd.DataFrame(rows)

def draw_residual_gradient(ax: mpl.axes.Axes, cv: pd.DataFrame, indep: pd.DataFrame, letter: str='f') -> None:
    set_title(ax, letter, 'Error along LST gradient')
    combined = pd.concat([cv, indep], ignore_index=True)
    edges = np.linspace(combined['actual_C'].min(), combined['actual_C'].max(), 20)
    for label, color, df in [('5-fold CV', CV, cv), ('Independent', INDEP, indep)]:
        stat = bin_stats(df, 'actual_C', 'residual', edges)
        ax.fill_between(stat['x'], stat['q25'], stat['q75'], color=color, alpha=0.16, linewidth=0)
        line_style = '--' if label == 'Independent' else '-'
        ax.plot(stat['x'], stat['median'], color=color, lw=1.25, ls=line_style, label=label)
    ax.set_xlabel('Observed LST (deg C)', fontsize=FONT_LABEL, labelpad=1.8)
    ax.set_ylabel('Residual (K)', fontsize=FONT_LABEL, labelpad=1.8)
    ax.set_ylim(-8.0, 6.5)
    ax.plot([float(combined['actual_C'].min()), float(combined['actual_C'].max())], [0, 0], color=NEUTRAL, lw=0.75, ls='--', alpha=0.75)
    ax.yaxis.set_major_locator(MultipleLocator(3))
    ax.xaxis.set_major_locator(MaxNLocator(4))
    set_clean_axis(ax)
    ax.legend(loc='lower left', fontsize=FONT_NOTE, handlelength=1.5, borderaxespad=0.2)

def draw_bland_altman(ax: mpl.axes.Axes, cv: pd.DataFrame, indep: pd.DataFrame, letter: str='g') -> None:
    set_title(ax, letter, 'Bland-Altman diagnostic')
    rng = np.random.default_rng(20260823)
    for color, df in [(CV, cv), (INDEP, indep)]:
        n = min(len(df), 1800)
        idx = rng.choice(len(df), size=n, replace=False)
        sample = df.iloc[idx]
        ax.scatter(sample['mean_C'], sample['residual'], s=3.2, color=color, alpha=0.16, linewidths=0, rasterized=True)
    all_df = pd.concat([cv, indep], ignore_index=True)
    bias = float(all_df['residual'].mean())
    sd = float(all_df['residual'].std(ddof=1))
    loa_lo = bias - 1.96 * sd
    loa_hi = bias + 1.96 * sd
    xlo, xhi = ax.get_xlim()
    ax.plot([xlo, xhi], [bias, bias], color=TEXT, lw=0.85)
    ax.plot([xlo, xhi], [loa_lo, loa_lo], color=MUTED, lw=0.75, ls='--')
    ax.plot([xlo, xhi], [loa_hi, loa_hi], color=MUTED, lw=0.75, ls='--')
    ax.set_xlim(xlo, xhi)
    ax.text(0.02, 0.94, f'Bias {bias:+.2f} K', transform=ax.transAxes, ha='left', va='top', fontsize=FONT_NOTE)
    ax.text(0.98, 0.09, '95% limits', transform=ax.transAxes, ha='right', va='bottom', fontsize=FONT_NOTE)
    ax.set_xlabel('Mean LST (deg C)', fontsize=FONT_LABEL, labelpad=1.8)
    ax.set_ylabel('Residual (K)', fontsize=FONT_LABEL, labelpad=1.8)
    ax.set_ylim(-14.5, 14.5)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    set_clean_axis(ax)

def draw_exceedance(ax: mpl.axes.Axes, cv: pd.DataFrame, indep: pd.DataFrame) -> None:
    set_title(ax, 'h', 'Absolute-error tail')
    thresholds = np.linspace(0, 12, 121)
    for label, color, df in [('5-fold CV', CV, cv), ('Independent', INDEP, indep)]:
        err = df['abs_error'].to_numpy(dtype=float)
        exceed = np.array([(err > t).mean() * 100 for t in thresholds])
        ax.plot(thresholds, exceed, color=color, lw=1.35, label=label)
        val5 = float((err > 5).mean() * 100)
        ax.plot([5], [val5], marker='o', ms=3.2, color=color, mec='white', mew=0.3)
        y_text = 17.0 if label == '5-fold CV' else 9.5
        ax.text(5.45, y_text, f'{100 - val5:.1f}% <=5 K', fontsize=FONT_NOTE, color=TEXT, ha='left')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Absolute error threshold (K)', fontsize=FONT_LABEL, labelpad=1.8)
    ax.set_ylabel('Samples exceeding threshold (%)', fontsize=FONT_LABEL, labelpad=1.8)
    ax.xaxis.set_major_locator(MultipleLocator(3))
    ax.yaxis.set_major_locator(MultipleLocator(25))
    set_clean_axis(ax)
    ax.legend(loc='upper right', fontsize=FONT_NOTE, handlelength=1.5, borderaxespad=0.2)

def draw_figure(cv: pd.DataFrame, indep: pd.DataFrame) -> tuple[mpl.figure.Figure, dict]:
    cv_met = metrics(cv)
    indep_met = metrics(indep)
    fold_df = per_fold_metrics(cv, indep)
    lim = (0.0, 40.0)
    map_specs = [('MODIS LST', MODIS_LST_TIF), ('Downscaled LST', DOWNSCALED_LST_TIF), ('Landsat LST', LANDSAT_OFFICIAL_LST_TIF)]
    map_previews = []
    for title, path in map_specs:
        preview = read_lst_preview(path)
        preview['title'] = title
        map_previews.append(preview)
    map_vmin, map_vmax = shared_lst_limits(map_previews)
    city_clip_records = prepare_city_circle_clips(overwrite=False)
    base_fig_height = 5.2
    city_map_target_in = CITY_MAP_TARGET_HEIGHT_PX / EXPORT_DPI
    city_grid_fixed_fraction = CITY_GRID_BOTTOM + CITY_GRID_TOP_PAD + CITY_GRID_HEADER_H + 3 * CITY_GRID_ROW_GAP
    fig_height = (base_fig_height + 4 * city_map_target_in) / (1.0 - city_grid_fixed_fraction)
    height_scale = base_fig_height / fig_height
    extra_bottom = (fig_height - base_fig_height) / fig_height
    fig = plt.figure(figsize=(7.2, fig_height), dpi=220)
    gs = fig.add_gridspec(3, 4, left=0.052, right=0.992, top=extra_bottom + 0.94 * height_scale, bottom=extra_bottom + 0.085 * height_scale, wspace=0.36, hspace=0.46, height_ratios=[0.78, 1.32, 0.98])
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[1, 0:2])
    ax_d = fig.add_subplot(gs[1, 2:4])
    ax_e = fig.add_subplot(gs[2, 0:2])
    ax_f = fig.add_subplot(gs[2, 2:4])
    draw_summary_panel(ax_a, cv_met, indep_met)
    draw_residual_distribution(ax_b, cv, indep, letter='b')
    draw_observed_predicted(ax_c, cv, '5-fold CV prediction', 'c', CV, lim, cv_met)
    draw_observed_predicted(ax_d, indep, 'Independent prediction', 'd', INDEP, lim, indep_met)
    draw_residual_gradient(ax_e, cv, indep, letter='e')
    draw_bland_altman(ax_f, cv, indep, letter='f')
    pos_a = ax_a.get_position()
    pos_b = ax_b.get_position()
    top_width = min(pos_a.width, pos_b.width)
    base_height = min(pos_a.height, pos_b.height)
    summary_height = base_height * 1.3
    ax_a.set_position([pos_a.x0, pos_a.y0 - (summary_height - base_height), top_width, summary_height])
    ax_b.set_position([pos_b.x0 + 0.018, pos_a.y0, top_width * 0.94, base_height])
    fig.canvas.draw()
    pos_c = ax_c.get_position()
    pos_d = ax_d.get_position()
    pos_e = ax_e.get_position()
    pos_f = ax_f.get_position()
    diag_left = ax_a.get_position().x0
    diag_gap = 0.075
    diag_width = min(pos_c.width, pos_c.height * fig_height / 7.2)
    diag_right = diag_left + diag_width + diag_gap
    ax_c.set_position([diag_left, pos_c.y0, diag_width, pos_c.height])
    ax_d.set_position([diag_right, pos_d.y0, diag_width, pos_d.height])
    ax_e.set_position([diag_left, pos_e.y0, diag_width, pos_e.height])
    ax_f.set_position([diag_right, pos_f.y0, diag_width, pos_f.height])
    fig.canvas.draw()
    pos_d = ax_d.get_position()
    pos_f = ax_f.get_position()
    map_left_base = pos_d.x1 + 0.045
    map_top_base = pos_d.y1
    map_bottom_base = pos_f.y0
    map_right = 0.985
    cbar_width = 0.006
    cbar_gap = 0.01
    map_gap = 0.024
    map_gap_size_reference = 0.032
    map_scale = 1.034
    cbar_left = map_right - cbar_width
    base_map_width = map_right - map_left_base - cbar_width - cbar_gap
    base_map_height = (map_top_base - map_bottom_base - 2 * map_gap_size_reference) / 3.0
    map_width = base_map_width * map_scale
    map_height = base_map_height * map_scale
    map_left = cbar_left - cbar_gap - map_width
    map_group_center = (map_top_base + map_bottom_base) / 2.0
    map_group_height = 3 * map_height + 2 * map_gap
    map_top = map_group_center + map_group_height / 2.0
    map_bottom = map_group_center - map_group_height / 2.0
    map_panel_header = mpl.transforms.Bbox.from_extents(ax_b.get_position().x0, pos_d.y0, ax_b.get_position().x0 + map_width, pos_d.y1)
    map_y_shift = -0.018 * height_scale
    map_top += map_y_shift
    map_draw_gap = 0.036
    map_bottom = map_top - 3 * map_height - 2 * map_draw_gap
    map_image = None
    cmap = lst_cmap()
    for idx, preview in enumerate(map_previews):
        y0 = map_top - (idx + 1) * map_height - idx * map_draw_gap
        ax_map = fig.add_axes([map_left, y0, map_width, map_height])
        map_image = draw_lst_distribution_map(ax_map, preview, str(preview['title']), cmap, map_vmin, map_vmax, show_crs=False)
        draw_lst_density_inset(ax_map, preview, (map_vmin, map_vmax), label_y=1.22 if idx == 2 else 1.1)
    if map_image is not None:
        cbar_x_shift = -0.04
        cax = fig.add_axes([map_left + map_width + cbar_gap + cbar_x_shift, map_bottom, cbar_width, map_top - map_bottom])
        cbar = fig.colorbar(map_image, cax=cax, orientation='vertical')
        cbar.outline.set_linewidth(0.45)
        cbar.ax.tick_params(labelsize=FONT_TICK, width=0.45, length=2.0, colors=TEXT)
        cbar.set_label('LST (deg C)', fontsize=FONT_NOTE, color=TEXT, labelpad=2.5)
        cbar.ax.yaxis.set_major_locator(MaxNLocator(5))
    for ax in (ax_e, ax_f):
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 - 0.012, pos.width, pos.height])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    e_ylabel_bbox = ax_e.yaxis.label.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
    city_row_label_w = 0.057
    city_grid_left = float(e_ylabel_bbox.x0 + e_ylabel_bbox.width / 2.0 - city_row_label_w * 0.46)
    city_grid_bounds = (city_grid_left, CITY_GRID_BOTTOM, 0.985, extra_bottom - CITY_GRID_TOP_PAD)
    city_grid_info = draw_city_lsts_grid(fig, city_grid_bounds, cmap, map_vmin, map_vmax)
    for ax in (ax_a, ax_b, ax_c, ax_d, ax_e, ax_f):
        add_axis_panel_header(fig, ax)
    add_panel_header(fig, map_panel_header, 'g', 'Global all-pixel LST distributions')
    h_header_x0 = ax_e.get_position().x0
    city_panel_header = mpl.transforms.Bbox.from_extents(h_header_x0, city_grid_bounds[1], h_header_x0 + 0.2, city_grid_bounds[3])
    add_panel_header(fig, city_panel_header, 'h', 'Representative-city LST comparison')
    audit = {'figure': 'fig_lst_downscaling_validation', 'data_sources': {'cross_validation': str(CV_CSV), 'independent_validation': str(INDEP_CSV), 'raster_maps': [{key: value for key, value in preview.items() if key != 'array'} for preview in map_previews]}, 'map_color_limits_deg_c': [map_vmin, map_vmax], 'city_map_color_limits_deg_c': CITY_LST_LIMITS_C, 'metrics': {'cross_validation': cv_met, 'independent_validation': indep_met}, 'fold_metrics': fold_df.to_dict(orient='records'), 'city_lsts_grid': city_grid_info, 'city_circle_clips': city_clip_records, 'notes': ['LST axes are shown in degrees C after subtracting 273.15 from Kelvin values.', 'Residual and error metrics are temperature differences, so K and degrees C have the same numerical magnitude.', 'Raster maps are read as WGS 1984 / EPSG:4326 previews.', 'The MODIS distribution map uses the original global all-pixel raster, not the urban-mask derivative.', 'Representative-city LST thumbnails are drawn from locally pre-clipped circular-mask GeoTIFFs.']}
    return (fig, audit)

def save_outputs(fig: mpl.figure.Figure, audit: dict, out_base: Path) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for ext in ('png', 'pdf', 'svg'):
        path = out_base.with_suffix(f'.{ext}')
        fig.savefig(path, dpi=EXPORT_DPI, bbox_inches='tight', pad_inches=0.025)
        outputs[ext] = str(path)
    tiff_path = out_base.with_suffix('.tiff')
    fig.savefig(tiff_path, dpi=EXPORT_DPI, bbox_inches='tight', pad_inches=0.025, pil_kwargs={'compression': 'tiff_lzw'})
    outputs['tiff'] = str(tiff_path)
    gray_path = out_base.with_name(f'{out_base.name}_grayscale.png')
    with Image.open(resource(out_base.with_suffix('.png'))) as image:
        ImageOps.grayscale(image).save(gray_path)
    outputs['grayscale_png'] = str(gray_path)
    audit = dict(audit)
    audit['outputs'] = outputs
    audit_path = out_base.with_name(f'{out_base.name}_audit.json')
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    outputs['audit_json'] = str(audit_path)
    return outputs

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cv', type=Path, default=CV_CSV, help='5-fold CV prediction CSV.')
    parser.add_argument('--independent', type=Path, default=INDEP_CSV, help='Independent validation CSV.')
    parser.add_argument('--out-base', type=Path, default=OUT_BASE, help='Output basename without extension.')
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    cv, indep = load_validation_data(args.cv, args.independent)
    fig, audit = draw_figure(cv, indep)
    outputs = save_outputs(fig, audit, args.out_base)
    plt.close(fig)
    for key, value in outputs.items():
        print(f'{key}: {value}')
    return 0

def _script_entry():
    raise SystemExit(main())

def _build_manuscript_figure():
    _script_entry()
    return None, None


if __name__ == "__main__":
    _configure_land_functions(sys.modules[__name__])
    _figure, _bbox = _build_manuscript_figure()
    _save_figure(_figure, _bbox, 'Supplementary_Fig_S2', _figure_options)
