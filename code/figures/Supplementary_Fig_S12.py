"""Plot Supplementary_Fig_S12 using the packaged numerical tables or supplied local inputs."""
from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_support"))
from figure_cli import prepare_figure as _prepare_figure, save_figure as _save_figure
from figure_cli import read_table as _read_table, model_tables as _model_tables
from figure_cli import configure_land_functions as _configure_land_functions

_figure_options = _prepare_figure('Supplementary_Fig_S12', 'external_inputs_required') if __name__ == "__main__" else None

from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
import json
import math
from io import BytesIO
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import shapefile
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
mpl.rcParams.update({'font.family': 'Arial', 'font.sans-serif': ['Arial'], 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'axes.unicode_minus': False, 'text.color': '#000000', 'axes.labelcolor': '#000000', 'axes.titlecolor': '#000000', 'xtick.color': '#000000', 'ytick.color': '#000000', 'savefig.facecolor': 'white', 'figure.facecolor': 'white'})
PROJECT_ROOT = PACKAGE_ROOT
ACCESS_DIR = PROJECT_ROOT / '影响变量数据' / '公共服务' / '城市可达性数据'
FRICTION_DIR = PROJECT_ROOT / '影响变量数据' / '公共服务' / '城市摩擦面数据'
BOUNDARY_DIR = PROJECT_ROOT / '影响变量数据' / '环境风险' / '热暴露' / '代表性城市' / '城市行政边界'
OUT_DIR = output_dir('Supplementary_Fig_S12')
OUT_DIR.mkdir(parents=True, exist_ok=True)
SATELLITE_CACHE_DIR = OUT_DIR / 'lst_city_google_image_cache'
SATELLITE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_BASE = OUT_DIR / 'fig_six_city_accessibility_mode_comparison'
OUT_AUDIT = OUT_DIR / 'fig_six_city_accessibility_mode_comparison_audit.json'
TEXT = '#000000'
LAND = '#F3F5F8'
MASKED_PIXEL = '#FFFFFF'
SATELLITE_BOUNDARY = '#FEED20'
FONT_PANEL = 11.2
FONT_TITLE = 9.8
FONT_LABEL = 8.8
FONT_TICK = 8.0
FONT_NOTE = 7.8
FIG_W = 11.6
FIG_H = 12.6
EXPORT_DPI = 600
CITY_ORDER = [{'key': 'Beijing', 'label': 'Beijing'}, {'key': 'Los_Angeles', 'label': 'Los Angeles'}, {'key': 'Singapore', 'label': 'Singapore'}, {'key': 'Rio_de_Janeiro', 'label': 'Rio de Janeiro'}, {'key': 'Rome', 'label': 'Rome'}, {'key': 'Cape_Town', 'label': 'Cape Town'}]
CITY_BOUNDARY_FILES = {'Beijing': '北京市.shp', 'Los_Angeles': '洛杉矶.shp', 'Singapore': '新加坡.shp', 'Rio_de_Janeiro': '里约热内卢.shp', 'Rome': '罗马.shp', 'Cape_Town': '开普敦.shp'}
BAND_ORDER = ['T_life_min', 'T_production_min', 'T_transport_min', 'T_education_min', 'T_health_min', 'T_leisure_min']
MODE_SETTINGS = {'walk': {'label': 'Non-motorized', 'vmin': 0.0, 'vmax': 60.0, 'ticks': [0, 20, 40, 60]}, 'car': {'label': 'Motorized', 'vmin': 0.0, 'vmax': 15.0, 'ticks': [0, 5, 10, 15]}}
FRICTION_SETTINGS = {'walk': {'label': 'Non-motorized friction', 'vmin': 0.0, 'vmax': 0.05, 'ticks': [0.0, 0.025, 0.05]}, 'car': {'label': 'Motorized friction', 'vmin': 0.0, 'vmax': 0.05, 'ticks': [0.0, 0.025, 0.05]}}
CITY_TILE_ZOOM = {'Beijing': 11, 'Los_Angeles': 11, 'Rome': 11, 'Cape_Town': 12, 'Rio_de_Janeiro': 12, 'Singapore': 13}
MAINLAND_ONLY_DISPLAY = {'Los_Angeles'}
_SATELLITE_ERRORS: list[dict] = []
_SATELLITE_FALLBACKS: list[dict] = []
_BOUNDARY_CACHE: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}

def access_cmap() -> LinearSegmentedColormap:
    cmap = LinearSegmentedColormap.from_list('poi_access_time', ['#FFF51D', '#F7B13C', '#EE744F', '#C23880', '#8600A8', '#2B006F'])
    cmap.set_bad(MASKED_PIXEL)
    return cmap

def access_raster_path(mode: str, city: dict) -> Path:
    path = ACCESS_DIR / f"poi_access_{mode}_{city['key']}_30m_multiband.tif"
    if not resource(path).exists():
        raise FileNotFoundError(path)
    return path

def friction_raster_path(mode: str, city: dict) -> Path:
    path = FRICTION_DIR / f"friction_{mode}_{city['key']}_30m_min_per_m.tif"
    if not resource(path).exists():
        raise FileNotFoundError(path)
    return path

def band_indices(ds: rasterio.io.DatasetReader) -> list[int]:
    descriptions = list(ds.descriptions)
    indices: list[int] = []
    for band_name in BAND_ORDER:
        if band_name in descriptions:
            indices.append(descriptions.index(band_name) + 1)
        else:
            fallback_idx = BAND_ORDER.index(band_name) + 1
            if fallback_idx <= ds.count:
                indices.append(fallback_idx)
    if not indices:
        raise ValueError(f'No expected access-time bands found in {ds.name}; descriptions={descriptions}')
    return indices

def read_mean_access_time(raster_path: Path, max_size: int=1100) -> dict:
    with rasterio.open(resource(raster_path)) as src:
        if src.crs is None or src.crs.to_epsg() != 4326:
            raise ValueError(f'{raster_path} must be WGS 1984 / EPSG:4326; found {src.crs}.')
        out_scale = max(1, int(math.ceil(max(src.width, src.height) / max_size)))
        out_width = max(1, int(math.ceil(src.width / out_scale)))
        out_height = max(1, int(math.ceil(src.height / out_scale)))
        data = src.read(band_indices(src), out_shape=(len(band_indices(src)), out_height, out_width), masked=True, resampling=Resampling.bilinear).astype('float32')
        arr = data.filled(np.nan) if np.ma.isMaskedArray(data) else np.asarray(data, dtype='float32')
        arr[~np.isfinite(arr)] = np.nan
        valid = np.isfinite(arr)
        counts = valid.sum(axis=0)
        sums = np.nansum(arr, axis=0)
        mean = np.full((out_height, out_width), np.nan, dtype='float32')
        np.divide(sums, counts, out=mean, where=counts > 0)
        bounds = src.bounds
    finite_values = mean[np.isfinite(mean)]
    return {'array': mean, 'extent': (bounds.left, bounds.right, bounds.bottom, bounds.top), 'source': str(raster_path), 'finite_pixels_preview': int(finite_values.size), 'p02_p50_p98': [float(v) for v in np.nanpercentile(finite_values, [2, 50, 98])] if finite_values.size else []}

def read_single_band_raster(raster_path: Path, max_size: int=1100) -> dict:
    with rasterio.open(resource(raster_path)) as src:
        if src.crs is None or src.crs.to_epsg() != 4326:
            raise ValueError(f'{raster_path} must be WGS 1984 / EPSG:4326; found {src.crs}.')
        out_scale = max(1, int(math.ceil(max(src.width, src.height) / max_size)))
        out_width = max(1, int(math.ceil(src.width / out_scale)))
        out_height = max(1, int(math.ceil(src.height / out_scale)))
        data = src.read(1, out_shape=(out_height, out_width), masked=True, resampling=Resampling.bilinear).astype('float32')
        arr = data.filled(np.nan) if np.ma.isMaskedArray(data) else np.asarray(data, dtype='float32')
        arr[~np.isfinite(arr)] = np.nan
        bounds = src.bounds
    finite_values = arr[np.isfinite(arr)]
    return {'array': arr, 'extent': (bounds.left, bounds.right, bounds.bottom, bounds.top), 'source': str(raster_path), 'finite_pixels_preview': int(finite_values.size), 'p02_p50_p98': [float(v) for v in np.nanpercentile(finite_values, [2, 50, 98])] if finite_values.size else []}

def city_full_extent(city: dict) -> tuple[float, float, float, float]:
    with rasterio.open(resource(access_raster_path('walk', city))) as src:
        if src.crs is None or src.crs.to_epsg() != 4326:
            raise ValueError(f'{src.name} must be WGS 1984 / EPSG:4326; found {src.crs}.')
        bounds = src.bounds
    return (bounds.left, bounds.right, bounds.bottom, bounds.top)

def city_display_extent(city: dict, full_extent: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if str(city['key']) not in MAINLAND_ONLY_DISPLAY:
        return full_extent
    lines = load_city_boundary(city)
    largest: tuple[np.ndarray, np.ndarray] | None = None
    largest_area = -1.0
    for xs, ys in lines:
        if len(xs) < 3:
            continue
        area = 0.5 * abs(float(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))))
        if area > largest_area:
            largest_area = area
            largest = (xs, ys)
    if largest is None:
        return full_extent
    xs, ys = largest
    return pad_extent((float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())), pad_ratio=0.012)

def city_boundary_path(city: dict) -> Path:
    path = BOUNDARY_DIR / CITY_BOUNDARY_FILES[str(city['key'])]
    if not resource(path).exists():
        raise FileNotFoundError(path)
    return path

def load_city_boundary(city: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    key = str(city['key'])
    if key in _BOUNDARY_CACHE:
        return _BOUNDARY_CACHE[key]
    path = city_boundary_path(city)
    source_crs = CRS.from_epsg(4326)
    prj_path = path.with_suffix('.prj')
    if resource(prj_path).exists():
        source_crs = CRS.from_wkt(resource(prj_path).read_text(encoding='utf-8', errors='ignore'))
    transformer = None
    if source_crs.to_epsg() != 4326:
        transformer = Transformer.from_crs(source_crs, CRS.from_epsg(4326), always_xy=True)
    reader = shapefile.Reader(resource(str(path)))
    lines: list[tuple[np.ndarray, np.ndarray]] = []
    for shp in reader.shapes():
        points = np.asarray(shp.points, dtype='float64')
        if points.size == 0:
            continue
        part_starts = list(shp.parts) + [len(points)]
        for start, end in zip(part_starts[:-1], part_starts[1:]):
            part = points[start:end]
            if len(part) >= 2:
                xs = part[:, 0]
                ys = part[:, 1]
                if transformer is not None:
                    xs, ys = transformer.transform(xs, ys)
                    xs = np.asarray(xs, dtype='float64')
                    ys = np.asarray(ys, dtype='float64')
                lines.append((xs, ys))
    _BOUNDARY_CACHE[key] = lines
    return lines

def draw_city_boundary(ax: mpl.axes.Axes, city: dict, lw: float=0.48, color: str='#202020', alpha: float=0.86, glow: bool=False) -> None:
    for xs, ys in load_city_boundary(city):
        if glow:
            ax.plot(xs, ys, color='#00E5FF', linewidth=lw * 3.8, alpha=0.22, zorder=8)
            ax.plot(xs, ys, color='#D7FF2F', linewidth=lw * 2.2, alpha=0.46, zorder=9)
        ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha, zorder=10)

def pad_extent(extent: tuple[float, float, float, float], pad_ratio: float=0.015) -> tuple[float, float, float, float]:
    left, right, bottom, top = extent
    pad_x = max((right - left) * pad_ratio, 1e-07)
    pad_y = max((top - bottom) * pad_ratio, 1e-07)
    return (left - pad_x, right + pad_x, bottom - pad_y, top + pad_y)

def lonlat_to_global_pixel(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    lat = min(max(float(lat), -85.05112878), 85.05112878)
    sin_lat = math.sin(math.radians(lat))
    scale = 256 * 2 ** zoom
    x = (float(lon) + 180.0) / 360.0 * scale
    y = (0.5 - math.log((1.0 + sin_lat) / (1.0 - sin_lat)) / (4.0 * math.pi)) * scale
    return (x, y)

def load_existing_city_satellite(city: dict, extent: tuple[float, float, float, float], max_size: int=1100) -> dict | None:
    key = str(city['key'])
    preferred_zoom = CITY_TILE_ZOOM.get(key, 11)
    candidates = [SATELLITE_CACHE_DIR / f'{key}_esri_world_imagery_z{preferred_zoom}.png']
    candidates.extend(sorted(resource_glob(SATELLITE_CACHE_DIR, f'{key}_esri_world_imagery_z*.png'), reverse=True))
    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not resource(path).exists():
            continue
        seen.add(path)
        image = Image.open(resource(path)).convert('RGB')
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        _SATELLITE_FALLBACKS.append({'city': city['label'], 'source': str(path), 'note': 'Existing local Esri cache shown without circular masking.'})
        return {'array': np.asarray(image, dtype='float32') / 255.0, 'extent': extent, 'source': str(path)}
    return None

def fetch_esri_satellite(city: dict, extent: tuple[float, float, float, float], max_size: int=1100) -> dict | None:
    left, right, bottom, top = extent
    dx = max(right - left, 1e-09)
    dy = max(top - bottom, 1e-09)
    if dx >= dy:
        width = max_size
        height = max(1, int(round(max_size * dy / dx)))
    else:
        height = max_size
        width = max(1, int(round(max_size * dx / dy)))
    cache_path = SATELLITE_CACHE_DIR / f"{city['key']}_accessibility_wgs84_extent.png"
    if resource(cache_path).exists():
        image = Image.open(resource(cache_path)).convert('RGB')
        return {'array': np.asarray(image, dtype='float32') / 255.0, 'extent': extent, 'source': str(cache_path)}
    raise FileNotFoundError('A local imagery file is required for this panel.')

def draw_map_common(ax: mpl.axes.Axes, extent: tuple[float, float, float, float], facecolor: str='white') -> None:
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    ax.set_facecolor(facecolor)

def draw_access_map(ax: mpl.axes.Axes, city: dict, preview: dict, cmap: LinearSegmentedColormap, vmin: float, vmax: float, display_extent: tuple[float, float, float, float]) -> None:
    ax.imshow(np.ma.masked_invalid(preview['array']), extent=preview['extent'], origin='upper', cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest', rasterized=True)
    draw_map_common(ax, display_extent, facecolor=MASKED_PIXEL)
    draw_city_boundary(ax, city)

def draw_friction_map(ax: mpl.axes.Axes, city: dict, preview: dict, cmap: LinearSegmentedColormap, vmin: float, vmax: float, display_extent: tuple[float, float, float, float]) -> None:
    ax.imshow(np.ma.masked_invalid(preview['array']), extent=preview['extent'], origin='upper', cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest', rasterized=True)
    draw_map_common(ax, display_extent, facecolor=MASKED_PIXEL)
    draw_city_boundary(ax, city)

def draw_satellite_map(ax: mpl.axes.Axes, city: dict, full_extent: tuple[float, float, float, float], display_extent: tuple[float, float, float, float]) -> dict | None:
    satellite = fetch_esri_satellite(city, full_extent)
    if satellite is None:
        ax.set_facecolor(LAND)
        ax.text(0.5, 0.5, 'image\npending', transform=ax.transAxes, ha='center', va='center', fontsize=FONT_NOTE, color=TEXT, linespacing=0.9)
        draw_map_common(ax, display_extent)
        draw_city_boundary(ax, city, lw=0.72, color=SATELLITE_BOUNDARY, alpha=0.98, glow=False)
        return None
    ax.imshow(satellite['array'], extent=satellite['extent'], origin='upper', interpolation='lanczos', rasterized=True)
    draw_map_common(ax, display_extent)
    draw_city_boundary(ax, city, lw=0.72, color=SATELLITE_BOUNDARY, alpha=0.98, glow=False)
    return satellite

def make_axes_grid(fig: mpl.figure.Figure) -> tuple[list[list[mpl.axes.Axes]], dict]:
    left = 0.095
    right = 0.025
    top = 0.9
    col_gap = 0.02
    row_gap = 0.01
    nrows, ncols = (len(CITY_ORDER), 5)
    map_w = (1.0 - left - right - col_gap * (ncols - 1)) / ncols
    row_aspects: list[float] = []
    row_heights: list[float] = []
    for city in CITY_ORDER:
        extent = city_display_extent(city, city_full_extent(city))
        dx = max(extent[1] - extent[0], 1e-09)
        dy = max(extent[3] - extent[2], 1e-09)
        aspect = dy / dx
        row_aspects.append(aspect)
        row_heights.append(map_w * FIG_W / FIG_H * aspect)
    bottom = top - sum(row_heights) - row_gap * (nrows - 1)
    axes: list[list[mpl.axes.Axes]] = []
    y = top
    for row in range(nrows):
        row_axes = []
        map_h = row_heights[row]
        y = y - map_h
        for col in range(ncols):
            x = left + col * (map_w + col_gap)
            row_axes.append(fig.add_axes([x, y, map_w, map_h]))
        axes.append(row_axes)
        y = y - row_gap
    layout = {'left': left, 'right': right, 'top': top, 'bottom': bottom, 'map_w': map_w, 'row_heights': row_heights, 'row_aspects': row_aspects, 'col_gap': col_gap, 'row_gap': row_gap}
    return (axes, layout)

def add_titles_and_labels(fig: mpl.figure.Figure, axes: list[list[mpl.axes.Axes]]) -> None:
    column_labels = ['Non-motorized', 'Motorized', 'Non-motorized friction', 'Motorized friction', 'Satellite image']
    for col, label in enumerate(column_labels):
        pos = axes[0][col].get_position()
        fig.text(pos.x0 + pos.width / 2.0, pos.y1 + 0.014, label, ha='center', va='bottom', fontsize=FONT_TITLE, color=TEXT)
    for row, city in enumerate(CITY_ORDER):
        pos = axes[row][0].get_position()
        fig.text(pos.x0 - 0.018, pos.y0 + pos.height / 2.0, city['label'], ha='right', va='center', fontsize=FONT_TITLE, color=TEXT)

def add_colorbar(fig: mpl.figure.Figure, ax_ref: mpl.axes.Axes, cmap: LinearSegmentedColormap, vmin: float, vmax: float, ticks: list[float], label: str) -> None:
    pos = ax_ref.get_position()
    cbar_y = max(0.02, pos.y0 - 0.03)
    cax = fig.add_axes([pos.x0 + 0.018, cbar_y, pos.width - 0.036, 0.012])
    sm = mpl.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_ticks(ticks)
    cb.set_ticklabels([str(int(t)) if float(t).is_integer() else f'{t:g}' for t in ticks])
    cb.set_label(label, fontsize=FONT_LABEL, color=TEXT, labelpad=2.0)
    cb.ax.tick_params(axis='x', labelsize=FONT_TICK, length=2.4, width=0.6, pad=1.5, colors=TEXT, direction='out')
    cb.outline.set_visible(True)
    cb.outline.set_linewidth(0.8)
    cb.outline.set_edgecolor(TEXT)

def plot() -> dict:
    cmap = access_cmap()
    friction_cmap = access_cmap()
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=EXPORT_DPI)
    axes, layout = make_axes_grid(fig)
    add_titles_and_labels(fig, axes)
    records: list[dict] = []
    for row, city in enumerate(CITY_ORDER):
        city_extent = city_full_extent(city)
        display_extent = city_display_extent(city, city_extent)
        for col, mode in enumerate(['walk', 'car']):
            settings = MODE_SETTINGS[mode]
            preview = read_mean_access_time(access_raster_path(mode, city))
            draw_access_map(axes[row][col], city, preview, cmap, float(settings['vmin']), float(settings['vmax']), display_extent)
            records.append({'city': city['label'], 'mode': mode, 'source': preview['source'], 'extent': [float(v) for v in preview['extent']], 'display_extent': [float(v) for v in display_extent], 'composite': 'pixelwise mean of six POI access-time bands', 'finite_pixels_preview': preview['finite_pixels_preview'], 'p02_p50_p98': preview['p02_p50_p98']})
        for offset, mode in enumerate(['walk', 'car'], start=2):
            settings = FRICTION_SETTINGS[mode]
            preview = read_single_band_raster(friction_raster_path(mode, city))
            draw_friction_map(axes[row][offset], city, preview, friction_cmap, float(settings['vmin']), float(settings['vmax']), display_extent)
            records.append({'city': city['label'], 'mode': f'friction_{mode}', 'source': preview['source'], 'extent': [float(v) for v in preview['extent']], 'display_extent': [float(v) for v in display_extent], 'unit': 'min per meter', 'finite_pixels_preview': preview['finite_pixels_preview'], 'p02_p50_p98': preview['p02_p50_p98']})
        satellite = draw_satellite_map(axes[row][4], city, city_extent, display_extent)
        records.append({'city': city['label'], 'mode': 'satellite', 'source': None if satellite is None else satellite['source'], 'extent': [float(v) for v in city_extent], 'display_extent': [float(v) for v in display_extent]})
    add_colorbar(fig, axes[-1][0], cmap, MODE_SETTINGS['walk']['vmin'], MODE_SETTINGS['walk']['vmax'], MODE_SETTINGS['walk']['ticks'], 'Mean non-motorized access time (min)')
    add_colorbar(fig, axes[-1][1], cmap, MODE_SETTINGS['car']['vmin'], MODE_SETTINGS['car']['vmax'], MODE_SETTINGS['car']['ticks'], 'Mean motorized access time (min)')
    add_colorbar(fig, axes[-1][2], friction_cmap, FRICTION_SETTINGS['walk']['vmin'], FRICTION_SETTINGS['walk']['vmax'], FRICTION_SETTINGS['walk']['ticks'], 'Non-motorized friction (min m$^{-1}$)')
    add_colorbar(fig, axes[-1][3], friction_cmap, FRICTION_SETTINGS['car']['vmin'], FRICTION_SETTINGS['car']['vmax'], FRICTION_SETTINGS['car']['ticks'], 'Motorized friction (min m$^{-1}$)')
    saved = []
    for suffix in ('.png', '.tiff', '.pdf'):
        out_path = OUT_BASE.with_suffix(suffix)
        fig.savefig(out_path, dpi=EXPORT_DPI, bbox_inches='tight', pad_inches=0.035)
        saved.append(str(out_path))
    plt.close(fig)
    audit = {'project_root': str(PROJECT_ROOT), 'access_dir': str(ACCESS_DIR), 'friction_dir': str(FRICTION_DIR), 'boundary_dir': str(BOUNDARY_DIR), 'crs_requirement': 'WGS 1984 / EPSG:4326', 'spatial_display': 'full raster extent; no circular mask', 'city_outline': 'administrative-boundary shapefile transformed to WGS84 and overlaid on every map', 'layout': layout, 'columns': ['Non-motorized mean access time', 'Motorized mean access time', 'Non-motorized friction', 'Motorized friction', 'Satellite image'], 'rows': [city['label'] for city in CITY_ORDER], 'mode_settings': MODE_SETTINGS, 'friction_settings': FRICTION_SETTINGS, 'records': records, 'satellite_errors': _SATELLITE_ERRORS, 'satellite_fallbacks': _SATELLITE_FALLBACKS, 'saved': saved}
    with OUT_AUDIT.open('w', encoding='utf-8') as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    return audit

def main() -> None:
    audit = plot()
    print('Saved accessibility mode comparison:')
    for path in audit['saved']:
        print(path)
    print(OUT_AUDIT)

def _script_entry():
    main()

def _build_manuscript_figure():
    _script_entry()
    return None, None


if __name__ == "__main__":
    _configure_land_functions(sys.modules[__name__])
    _figure, _bbox = _build_manuscript_figure()
    _save_figure(_figure, _bbox, 'Supplementary_Fig_S12', _figure_options)
