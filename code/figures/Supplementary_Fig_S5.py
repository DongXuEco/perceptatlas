"""Plot Supplementary_Fig_S5 using the packaged numerical tables or supplied local inputs."""
from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_support"))
from figure_cli import prepare_figure as _prepare_figure, save_figure as _save_figure
from figure_cli import read_table as _read_table, model_tables as _model_tables
from figure_cli import configure_land_functions as _configure_land_functions

_figure_options = _prepare_figure('Supplementary_Fig_S5', 'external_inputs_required') if __name__ == "__main__" else None

from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
import os
import math
import warnings
from pathlib import Path
from io import BytesIO
os.environ.setdefault('WINDIR', 'external_inputs/Windows')
os.environ.setdefault('SystemRoot', 'external_inputs/Windows')
_MPL_CONFIG_DIR = Path(__file__).resolve().parents[2] / '.mplconfig_dpython'
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(_MPL_CONFIG_DIR))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import RegularPolygon
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import ScaledTranslation
from affine import Affine
from PIL import Image
import rasterio
from rasterio.crs import CRS
from rasterio.features import geometry_mask, geometry_window
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling, transform_geom
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ModuleNotFoundError:
    ccrs = None
    cfeature = None
    HAS_CARTOPY = False
warnings.filterwarnings('ignore')
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[1]
base_dir = script_dir / '数据'
veg_tif = os.path.join(base_dir, 'ngreenspace.tif')
soil_tif = os.path.join(base_dir, 'nimpervious.tif')
wat_tif = os.path.join(base_dir, 'nwater.tif')
output_png = str(output_dir('Supplementary_Fig_S5') / 'Supplementary_Fig_S5.png')
output_pdf = str(output_dir('Supplementary_Fig_S5') / 'Supplementary_Fig_S5.pdf')
CITY_FRACTION_DIR = script_dir / '六城市数据'
CITY_CIRCLE_DIR = project_root / '影响变量数据' / '环境风险' / '热暴露' / '代表性城市' / '城市行政边界'
CITY_CLIP_DIR = CITY_FRACTION_DIR / 'circle_clips'
CITY_CLIP_DIR.mkdir(parents=True, exist_ok=True)
CITY_GOOGLE_CACHE_DIR = project_root / 'figure_outputs_q3' / 'lst_city_google_image_cache'
CITY_GOOGLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
TERNARY_TITLE_FONTSIZE = 22
SINGLE_MAP_TITLE_FONTSIZE = 22
SINGLE_CBAR_TICK_FONTSIZE = 16
SINGLE_CBAR_LABEL_FONTSIZE = 18
SINGLE_CBAR_LABEL_PAD = 6
DPI = 600
EXPORT_DPI = int(os.environ.get('GREENSPACE_DPI', DPI))
SAVE_PDF = os.environ.get('GREENSPACE_SAVE_PDF', '1') != '0'
TERNARY_LEGEND_LEFT = -0.19
TERNARY_LEGEND_BOTTOM = -0.17
TERNARY_LEGEND_WIDTH = 0.6
TERNARY_LEGEND_HEIGHT = 0.672
TERNARY_LEGEND_TICK_FONTSIZE = 16
TERNARY_LEGEND_LABEL_FONTSIZE = 18
TERNARY_LEGEND_LABEL_GAP = 0.41
LON_MIN, LON_MAX = (-180, 180)
LAT_MIN, LAT_MAX = (-60, 85)
DRAW_EDGE_STRIP_LEGEND = True
EDGE_STRIP_LEFT = 0.43
EDGE_STRIP_BOTTOM = -0.15
EDGE_STRIP_WIDTH = 0.34
EDGE_STRIP_HEIGHT = 0.26
EDGE_STRIP_LABEL_FONTSIZE = 18
EDGE_STRIP_HEX_RADIUS = 0.42
EDGE_STRIP_ROW_GAP = 1.5
EDGE_STRIP_TEXT_GAP = 1.5
EDGE_STRIP_EDGE_COLOR = '#666666'
EDGE_STRIP_EDGE_WIDTH = 0.2
VEG_NAME = 'Vegetation'
SOIL_NAME = 'Bright substrate'
WAT_NAME = 'Water'
LAND_BG_COLOR = '#F7F7F7'
OCEAN_COLOR = '#FFFFFF'
COAST_COLOR = '#BEBEBE'
BORDER_COLOR = '#D0D0D0'
SINGLE_VMIN = 0.0
SINGLE_VMAX = 1
TOTAL_TARGET = 1.0
SINGLE_CBAR_TICKS = [0, 0.25, 0.5, 0.75, 1.0]
SINGLE_CBAR_WIDTH_RATIO = 0.68
SINGLE_CBAR_LEFT_RATIO = 0.16
SINGLE_CBAR_HEIGHT = 0.02
SINGLE_CBAR_BOTTOM_OFFSET = 0.03
DISPLAY_STRIDE = 8
CITY_ORDER = [{'key': 'Beijing', 'label': 'Beijing', 'zh': '北京市'}, {'key': 'Los_Angeles', 'label': 'Los Angeles', 'zh': '洛杉矶'}, {'key': 'Singapore', 'label': 'Singapore', 'zh': '新加坡'}, {'key': 'Rio_de_Janeiro', 'label': 'Rio de Janeiro', 'zh': '里约热内卢'}, {'key': 'Rome', 'label': 'Rome', 'zh': '罗马'}, {'key': 'Cape_Town', 'label': 'Cape Town', 'zh': '开普敦'}]
CITY_ROW_LABELS = ['Vegetation', 'Bright substrate', 'Water', 'Google image']
CITY_CLIP_NODATA = -9999.0
CITY_GOOGLE_TILE_ZOOM_BY_CITY = {'Beijing': 11, 'Los_Angeles': 11, 'Rome': 11, 'Cape_Town': 12, 'Rio_de_Janeiro': 12, 'Singapore': 13}
CITY_GOOGLE_TILE_TIMEOUT = 2.5
CITY_GRID_LEFT = 0.03
CITY_GRID_RIGHT = 0.985
CITY_GRID_BOTTOM = 0.025
CITY_GRID_TOP = 0.515
CITY_GRID_HEADER_H = 0.04
CITY_GRID_ROW_GAP = 0.006
CITY_GRID_COL_GAP = 0.012
CITY_ROW_LABEL_W = 0.068
CITY_MAP_SIZE_IN = 2.4
CITY_GRID_BODY_Y_SHIFT = 0.02
CITY_CIRCLE_EDGE_COLOR = '#30343B'
CITY_CIRCLE_EDGE_WIDTH = 0.75
CITY_CITY_LABEL_FONTSIZE = 20
CITY_ROW_LABEL_FONTSIZE = 20
CLEAN_TO_0_1 = True
RENORMALIZE = True
HONEY_N = 9
HEX_RADIUS_FACTOR = 1.0
DRAW_TRIANGLE_FRAME = True
DRAW_INNER_GRID = True
HONEY_ROWS = [['#b113dd'], ['#b633c1', '#9e1adc'], ['#bb51a8', '#a23cbf', '#8c23db'], ['#bf6c93', '#a65ba5', '#8f45bd', '#7a2cdb'], ['#c5857f', '#aa778f', '#9265a3', '#7c4fbc', '#6736db'], ['#cb9c6d', '#ae917b', '#95828b', '#7e70a1', '#695abb', '#5541da'], ['#d3b35c', '#b3a968', '#989d77', '#808e89', '#6a7b9e', '#5765b9', '#444cd9'], ['#dbc84b', '#b9c056', '#9bb663', '#81a973', '#6b9a86', '#57879c', '#4471b7', '#3259d7'], ['#e6dd3a', '#bfd745', '#9fce51', '#83c45f', '#6bb670', '#56a683', '#44949a', '#327eb4', '#2165d4'], ['#f2f227', '#c7ed33', '#a3e63f', '#85dc4c', '#6bd15b', '#55c36c', '#42b380', '#31a196', '#218bb1', '#1173d0']]
VEG_VERTEX_COLOR = HONEY_ROWS[-1][0]
SOIL_VERTEX_COLOR = HONEY_ROWS[0][0]
WAT_VERTEX_COLOR = HONEY_ROWS[-1][-1]
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['savefig.facecolor'] = 'white'
mpl.rcParams['text.color'] = '#000000'
mpl.rcParams['axes.labelcolor'] = '#000000'
mpl.rcParams['axes.titlecolor'] = '#000000'
mpl.rcParams['xtick.color'] = '#000000'
mpl.rcParams['ytick.color'] = '#000000'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

def hex_to_rgb255(hex_color):
    hex_color = hex_color.lstrip('#')
    return np.array([int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)], dtype=np.uint8)

def hex_to_rgb01(hex_color):
    return hex_to_rgb255(hex_color).astype(float) / 255.0

def make_single_cmap(hex_color, name='single_cmap'):
    return LinearSegmentedColormap.from_list(name, ['#FFFFFF', '#F2F2F2', hex_color], N=256)

def read_raster_for_display(path, stride=8, clean_to_0_1=True):
    if not os.path.exists(resource(path)):
        raise FileNotFoundError(f'文件不存在: {path}')
    with rasterio.open(resource(path)) as src:
        out_h = max(1, int(math.ceil(src.height / stride)))
        out_w = max(1, int(math.ceil(src.width / stride)))
        arr = src.read(1, out_shape=(out_h, out_w), masked=True, resampling=Resampling.average).astype(np.float32)
        arr = arr.filled(np.nan)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        arr[~np.isfinite(arr)] = np.nan
        if clean_to_0_1:
            arr[(arr < 0.0) | (arr > 1.0)] = np.nan
        scale_x = src.width / out_w
        scale_y = src.height / out_h
        out_transform = src.transform * Affine.scale(scale_x, scale_y)
        meta = {'crs': src.crs, 'transform': out_transform, 'width': out_w, 'height': out_h, 'shape': arr.shape, 'nodata': src.nodata}
    return (arr, meta)

def same_grid(meta1, meta2):
    return meta1['shape'] == meta2['shape'] and meta1['crs'] == meta2['crs'] and np.allclose(tuple(meta1['transform']), tuple(meta2['transform']))

def reproject_to_match(src_arr, src_meta, dst_meta, resampling=Resampling.bilinear):
    src_fill = -9999.0
    dst_fill = -9999.0
    src_tmp = np.where(np.isfinite(src_arr), src_arr, src_fill).astype(np.float32)
    dst_arr = np.full(dst_meta['shape'], dst_fill, dtype=np.float32)
    reproject(source=src_tmp, destination=dst_arr, src_transform=src_meta['transform'], src_crs=src_meta['crs'], src_nodata=src_fill, dst_transform=dst_meta['transform'], dst_crs=dst_meta['crs'], dst_nodata=dst_fill, resampling=resampling)
    dst_arr[dst_arr == dst_fill] = np.nan
    dst_arr[~np.isfinite(dst_arr)] = np.nan
    if CLEAN_TO_0_1:
        dst_arr[(dst_arr < 0.0) | (dst_arr > 1.0)] = np.nan
    return dst_arr

def get_extent_from_transform(transform, width, height):
    left = transform.c
    top = transform.f
    right = left + transform.a * width
    bottom = top + transform.e * height
    return [left, right, bottom, top]
_CITY_CIRCLE_CACHE = {}
_CITY_GOOGLE_ERRORS = []

def find_city_fraction_file(city):
    candidates = [CITY_FRACTION_DIR / f"fraction_2024_30m_{city['key']}.tif", CITY_FRACTION_DIR / f"fraction_2024_30m_{city['label']}.tif"]
    for path in candidates:
        if resource(path).exists():
            return path
    needles = {city['key'].lower(), city['label'].replace(' ', '_').lower(), city['label'].lower()}
    for path in sorted(resource_glob(CITY_FRACTION_DIR, '*.tif')):
        stem = path.stem.lower()
        if any((needle in stem for needle in needles)):
            return path
    return None

def find_city_circle_file(city):
    exact_names = [f"{city['zh']}C.shp", f"{city['label']}C.shp", f"{city['key']}C.shp"]
    for name in exact_names:
        path = CITY_CIRCLE_DIR / name
        if resource(path).exists():
            return path
    needles = {f"{city['zh']}c".lower(), f"{city['label']}c".lower(), f"{city['key']}c".lower(), f"{city['key'].replace('_', ' ')}c".lower()}
    for path in sorted(resource_glob(CITY_CIRCLE_DIR, '*C.shp')):
        if path.stem.lower() in needles:
            return path
    return None

def read_city_circle_geometries(city, dst_crs):
    import shapefile
    circle_path = find_city_circle_file(city)
    if circle_path is None:
        raise FileNotFoundError(f"Missing circular mask shapefile for {city['label']} in {CITY_CIRCLE_DIR}")
    reader = shapefile.Reader(resource(str(circle_path)))
    geoms = [shape.__geo_interface__ for shape in reader.shapes()]
    if not geoms:
        raise ValueError(f'{circle_path} contains no valid geometries.')
    prj_path = circle_path.with_suffix('.prj')
    src_crs = CRS.from_wkt(resource(prj_path).read_text(encoding='utf-8')) if resource(prj_path).exists() else CRS.from_epsg(4326)
    if src_crs != dst_crs:
        geoms = [transform_geom(src_crs, dst_crs, geom) for geom in geoms]
    return (geoms, circle_path)

def get_city_circle_geometries_4326(city):
    key = city['key']
    if key not in _CITY_CIRCLE_CACHE:
        _CITY_CIRCLE_CACHE[key] = read_city_circle_geometries(city, CRS.from_epsg(4326))
    return _CITY_CIRCLE_CACHE[key]

def iter_geojson_exterior_rings(geom):
    geom_type = geom.get('type')
    coordinates = geom.get('coordinates', [])
    if geom_type == 'Polygon':
        if coordinates:
            yield coordinates[0]
    elif geom_type == 'MultiPolygon':
        for polygon in coordinates:
            if polygon:
                yield polygon[0]

def iter_geojson_boundary_rings(geom):
    geom_type = geom.get('type')
    coordinates = geom.get('coordinates', [])
    if geom_type == 'Polygon':
        for ring in coordinates:
            yield ring
    elif geom_type == 'MultiPolygon':
        for polygon in coordinates:
            for ring in polygon:
                yield ring

def city_circle_extent(city, pad_ratio=0.03):
    geoms, _ = get_city_circle_geometries_4326(city)
    xs = []
    ys = []
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

def square_extent(extent):
    left, right, bottom, top = extent
    cx = (left + right) / 2.0
    cy = (bottom + top) / 2.0
    side = max(right - left, top - bottom)
    return (cx - side / 2.0, cx + side / 2.0, cy - side / 2.0, cy + side / 2.0)

def city_circle_square_extent(city, pad_ratio=0.05):
    return square_extent(city_circle_extent(city, pad_ratio=pad_ratio))

def draw_city_circle_boundary(ax, city):
    geoms, _ = get_city_circle_geometries_4326(city)
    for geom in geoms:
        for ring in iter_geojson_boundary_rings(geom):
            if len(ring) < 2:
                continue
            xs = [float(point[0]) for point in ring]
            ys = [float(point[1]) for point in ring]
            ax.plot(xs, ys, color=CITY_CIRCLE_EDGE_COLOR, lw=CITY_CIRCLE_EDGE_WIDTH, alpha=0.95, zorder=30, clip_on=False)

def mask_rgb_to_city_circle(rgb, extent, city):
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        return rgb
    geoms, _ = get_city_circle_geometries_4326(city)
    height, width = rgb.shape[:2]
    left, right, bottom, top = extent
    transform = from_bounds(left, bottom, right, top, width, height)
    inside = geometry_mask(geoms, out_shape=(height, width), transform=transform, invert=True)
    return np.dstack([rgb[..., :3], inside.astype('float32')])

def lonlat_to_global_pixel(lon, lat, zoom):
    lat = min(max(float(lat), -85.05112878), 85.05112878)
    sin_lat = math.sin(math.radians(lat))
    scale = 256 * 2 ** zoom
    x = (float(lon) + 180.0) / 360.0 * scale
    y = (0.5 - math.log((1.0 + sin_lat) / (1.0 - sin_lat)) / (4.0 * math.pi)) * scale
    return (x, y)

def city_google_tile_zoom(city):
    return CITY_GOOGLE_TILE_ZOOM_BY_CITY.get(str(city['key']), 11)

def fetch_esri_satellite_preview(city, max_size=900):
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

def shapefile_component_paths(path):
    suffixes = ['.shp', '.shx', '.dbf', '.prj', '.cpg']
    return [path.with_suffix(suffix) for suffix in suffixes if resource(path.with_suffix(suffix)).exists()]

def city_clip_path(city):
    return CITY_CLIP_DIR / f"{city['key']}_fraction_2024_30m_circle_clip.tif"

def clip_is_current(clip_path, source_path, circle_path):
    if not resource(clip_path).exists():
        return False
    clip_mtime = clip_path.stat().st_mtime
    dependencies = [source_path, *shapefile_component_paths(circle_path)]
    return all((clip_mtime >= dep.stat().st_mtime for dep in dependencies if resource(dep).exists()))

def clip_city_fraction_raster(city, overwrite=False):
    source_path = find_city_fraction_file(city)
    if source_path is None:
        raise FileNotFoundError(f"Missing fraction raster for {city['label']} in {CITY_FRACTION_DIR}")
    clip_path = city_clip_path(city)
    with rasterio.open(resource(source_path)) as src:
        if src.count < 3:
            raise ValueError(f'{source_path} must contain at least three bands.')
        if src.crs is None or src.crs.to_epsg() != 4326:
            raise ValueError(f'{source_path} must be WGS 1984 / EPSG:4326; found {src.crs}.')
        geoms, circle_path = read_city_circle_geometries(city, src.crs)
        if not overwrite and clip_is_current(clip_path, source_path, circle_path):
            return {'city': city['label'], 'source_raster': str(source_path), 'clipped_raster': str(clip_path), 'circle_mask': str(circle_path), 'recreated': False}
        window = geometry_window(src, geoms)
        data = src.read([1, 2, 3], window=window, masked=True).astype('float32')
        out_transform = src.window_transform(window)
        arr = np.asarray(data.filled(np.nan), dtype=np.float32)
        inside_circle = geometry_mask(geoms, out_shape=(arr.shape[1], arr.shape[2]), transform=out_transform, invert=True)
        for band in range(arr.shape[0]):
            arr[band, ~inside_circle] = np.nan
        arr[(arr < 0.0) | (arr > 1.0)] = np.nan
        out_arr = np.where(np.isfinite(arr), arr, CITY_CLIP_NODATA).astype('float32')
        profile = src.profile.copy()
        profile.update({'driver': 'GTiff', 'height': int(out_arr.shape[1]), 'width': int(out_arr.shape[2]), 'count': 3, 'dtype': 'float32', 'crs': src.crs, 'transform': out_transform, 'nodata': CITY_CLIP_NODATA, 'compress': 'lzw'})
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(resource(clip_path), 'w', **profile) as dst:
        dst.write(out_arr)
        dst.set_band_description(1, 'Vegetation')
        dst.set_band_description(2, 'Bright_substrate')
        dst.set_band_description(3, 'Water')
    return {'city': city['label'], 'source_raster': str(source_path), 'clipped_raster': str(clip_path), 'circle_mask': str(circle_path), 'recreated': True}

def prepare_city_fraction_clips(overwrite=False):
    records = []
    for city in CITY_ORDER:
        records.append(clip_city_fraction_raster(city, overwrite=overwrite))
    return records

def read_city_fraction_preview(path, max_size=360):
    with rasterio.open(resource(path)) as src:
        scale = max(1, int(np.ceil(max(src.width, src.height) / max_size)))
        out_width = max(1, int(np.ceil(src.width / scale)))
        out_height = max(1, int(np.ceil(src.height / scale)))
        data = src.read([1, 2, 3], out_shape=(3, out_height, out_width), masked=True, resampling=Resampling.average).astype('float32')
        arr = np.asarray(data.filled(np.nan), dtype=np.float32)
        arr[(arr < 0.0) | (arr > 1.0)] = np.nan
        bounds = src.bounds
        return {'array': arr, 'extent': [bounds.left, bounds.right, bounds.bottom, bounds.top]}

def build_honeycomb_lookup_uint8():
    lookup = {}
    for row_idx, row_colors in enumerate(HONEY_ROWS):
        soil_i = HONEY_N - row_idx
        for col_idx, hex_color in enumerate(row_colors):
            veg_i = row_idx - col_idx
            wat_i = col_idx
            lookup[veg_i, soil_i, wat_i] = hex_to_rgb255(hex_color)
    return lookup

def build_color_lut():
    size = HONEY_N + 1
    lut = np.zeros((size, size, size, 4), dtype=np.uint8)
    lookup = build_honeycomb_lookup_uint8()
    for (vi, si, wi), rgb in lookup.items():
        lut[vi, si, wi, 0] = rgb[0]
        lut[vi, si, wi, 1] = rgb[1]
        lut[vi, si, wi, 2] = rgb[2]
        lut[vi, si, wi, 3] = 255
    return lut

def quantize_to_simplex_memory_safe(wv, ws, ww, n=9):
    sv = wv * n
    ss = ws * n
    sw = ww * n
    vi = np.floor(sv).astype(np.int16)
    si = np.floor(ss).astype(np.int16)
    wi = np.floor(sw).astype(np.int16)
    fv = sv - vi
    fs = ss - si
    fw = sw - wi
    deficit = n - (vi + si + wi)
    deficit = np.clip(deficit, 0, 3).astype(np.int8)
    for _ in range(3):
        mask = deficit > 0
        if not np.any(mask):
            break
        choose_v = (fv >= fs) & (fv >= fw) & mask
        choose_s = (fs > fv) & (fs >= fw) & mask
        choose_w = mask & ~choose_v & ~choose_s
        vi[choose_v] += 1
        si[choose_s] += 1
        wi[choose_w] += 1
        fv[choose_v] = -1
        fs[choose_s] = -1
        fw[choose_w] = -1
        deficit[mask] -= 1
    vi = np.clip(vi, 0, n)
    si = np.clip(si, 0, n)
    wi = np.clip(wi, 0, n)
    return (vi, si, wi)

def build_ternary_rgba_memory_safe(veg, soil, wat, renormalize=True):
    total = veg + soil + wat
    valid = np.isfinite(veg) & np.isfinite(soil) & np.isfinite(wat) & np.isfinite(total) & (total > 0)
    wv = np.zeros_like(veg, dtype=np.float32)
    ws = np.zeros_like(soil, dtype=np.float32)
    ww = np.zeros_like(wat, dtype=np.float32)
    if renormalize:
        wv[valid] = veg[valid] / total[valid]
        ws[valid] = soil[valid] / total[valid]
        ww[valid] = wat[valid] / total[valid]
    else:
        wv[valid] = veg[valid]
        ws[valid] = soil[valid]
        ww[valid] = wat[valid]
    vi, si, wi = quantize_to_simplex_memory_safe(wv, ws, ww, n=HONEY_N)
    color_lut = build_color_lut()
    rgba = color_lut[vi, si, wi]
    rgba[~valid, :] = 0
    return (rgba, valid)

def unit_vector(v):
    v = np.asarray(v, dtype=float)
    n = np.hypot(v[0], v[1])
    if n == 0:
        return v
    return v / n

def edge_point(P0, P1, t):
    return (1 - t) * P0 + t * P1

def draw_honeycomb_ternary_legend(ax, top_label='Bright substrate', left_label='Vegetation', right_label='Water', tick_values=(0, 0.2, 0.4, 0.6, 0.8, 1.0), tick_fs=10, label_fs=11, label_gap=0.28):
    h = np.sqrt(3) / 2.0
    V_top = np.array([0.5, h], dtype=float)
    V_left = np.array([0.0, 0.0], dtype=float)
    V_right = np.array([1.0, 0.0], dtype=float)

    def bary_to_xy(v, s, w):
        return v * V_left + s * V_top + w * V_right
    lookup = build_honeycomb_lookup_uint8()
    dx = 1.0 / HONEY_N
    hex_r = dx / np.sqrt(3) * HEX_RADIUS_FACTOR
    for (veg_i, soil_i, wat_i), rgb255 in lookup.items():
        wv = veg_i / HONEY_N
        ws = soil_i / HONEY_N
        ww = wat_i / HONEY_N
        center = bary_to_xy(wv, ws, ww)
        hex_patch = RegularPolygon((center[0], center[1]), numVertices=6, radius=hex_r, orientation=0.0, facecolor=rgb255.astype(float) / 255.0, edgecolor='none', linewidth=0.0, antialiased=False, zorder=2)
        ax.add_patch(hex_patch)
    if DRAW_TRIANGLE_FRAME:
        tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, h], [0.0, 0.0]])
        ax.plot(tri[:, 0], tri[:, 1], color='#D6D6D6', lw=0.85, ls=(0, (1.2, 2.2)), zorder=4)
    if DRAW_INNER_GRID:
        for k in range(1, HONEY_N):
            p1 = bary_to_xy((HONEY_N - k) / HONEY_N, k / HONEY_N, 0)
            p2 = bary_to_xy(0, k / HONEY_N, (HONEY_N - k) / HONEY_N)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#D6D6D6', lw=0.7, ls=(0, (1.2, 2.2)), zorder=5)
            p3 = bary_to_xy(k / HONEY_N, (HONEY_N - k) / HONEY_N, 0)
            p4 = bary_to_xy(k / HONEY_N, 0, (HONEY_N - k) / HONEY_N)
            ax.plot([p3[0], p4[0]], [p3[1], p4[1]], color='#D6D6D6', lw=0.7, ls=(0, (1.2, 2.2)), zorder=5)
            p5 = bary_to_xy((HONEY_N - k) / HONEY_N, 0, k / HONEY_N)
            p6 = bary_to_xy(0, (HONEY_N - k) / HONEY_N, k / HONEY_N)
            ax.plot([p5[0], p6[0]], [p5[1], p6[1]], color='#D6D6D6', lw=0.7, ls=(0, (1.2, 2.2)), zorder=5)
    tick_color = '#000000'
    bottom_gap = 0.105
    side_gap = 0.1
    d_left = unit_vector(V_top - V_left)
    n_left = unit_vector(np.array([-d_left[1], d_left[0]]))
    d_right = unit_vector(V_top - V_right)
    n_right = unit_vector(np.array([d_right[1], -d_right[0]]))
    for v in tick_values:
        p = edge_point(V_left, V_right, v)
        ax.text(p[0], p[1] - bottom_gap, f'{v:g}', ha='center', va='top', fontsize=tick_fs, color=tick_color)
    for val in tick_values:
        t = 1 - val
        p = edge_point(V_left, V_top, t)
        pos = p + n_left * side_gap
        ax.text(pos[0], pos[1], f'{val:g}', ha='right', va='center', fontsize=tick_fs, color=tick_color)
    for val in tick_values:
        t = val
        p = edge_point(V_right, V_top, t)
        pos = p + n_right * side_gap
        ax.text(pos[0], pos[1], f'{val:g}', ha='left', va='center', fontsize=tick_fs, color=tick_color)
    label_color = '#000000'
    LABEL_GAP = label_gap
    veg_edge_mid = 0.5 * (V_left + V_top)
    veg_label_pos = veg_edge_mid + n_left * LABEL_GAP
    ax.text(veg_label_pos[0], veg_label_pos[1], left_label, rotation=60, ha='center', va='center', fontsize=label_fs, color=label_color)
    water_label_pos = 0.5 * (V_left + V_right) + np.array([0.0, -LABEL_GAP])
    ax.text(water_label_pos[0], water_label_pos[1], right_label, rotation=0, ha='center', va='center', fontsize=label_fs, color=label_color)
    soil_edge_mid = 0.5 * (V_right + V_top)
    soil_label_pos = soil_edge_mid + n_right * LABEL_GAP
    ax.text(soil_label_pos[0], soil_label_pos[1], top_label, rotation=-60, ha='center', va='center', fontsize=label_fs, color=label_color)
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim(-0.28, h + 0.26)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

def draw_outer_edge_hex_strip_legend(ax):
    ax.set_axis_off()
    veg_edge_colors = [row[0] for row in HONEY_ROWS]
    soil_edge_colors = [row[-1] for row in HONEY_ROWS][::-1]
    wat_edge_colors = HONEY_ROWS[-1]
    strip_items = [(VEG_NAME, veg_edge_colors), (SOIL_NAME, soil_edge_colors), (WAT_NAME, wat_edge_colors)]
    hex_r = EDGE_STRIP_HEX_RADIUS
    dx = np.sqrt(3) * hex_r
    row_gap = EDGE_STRIP_ROW_GAP
    n_cols = HONEY_N + 1
    for row_idx, (label, colors) in enumerate(strip_items):
        y = (len(strip_items) - 1 - row_idx) * row_gap
        for j, color in enumerate(colors):
            x = j * dx
            hex_patch = RegularPolygon((x, y), numVertices=6, radius=hex_r, orientation=0.0, facecolor=color, edgecolor=EDGE_STRIP_EDGE_COLOR, linewidth=EDGE_STRIP_EDGE_WIDTH, antialiased=True, zorder=2)
            ax.add_patch(hex_patch)
        text_x = (n_cols - 1) * dx + EDGE_STRIP_TEXT_GAP
        ax.text(text_x, y, label, ha='left', va='center', fontsize=EDGE_STRIP_LABEL_FONTSIZE, color='#000000')
    ax.set_xlim(-hex_r * 1.5, (n_cols - 1) * dx + EDGE_STRIP_TEXT_GAP + 4.5)
    ax.set_ylim(-0.8, (len(strip_items) - 1) * row_gap + 0.8)
    ax.set_aspect('equal')

def draw_outer_edge_strip_legend(ax):
    ax.set_axis_off()
    vegetation_edge = [row[0] for row in HONEY_ROWS]
    bright_edge = [row[-1] for row in HONEY_ROWS]
    water_edge = HONEY_ROWS[-1]
    strip_items = [(VEG_NAME, vegetation_edge), (SOIL_NAME, bright_edge), (WAT_NAME, water_edge)]
    n_cols = HONEY_N + 1
    strip_h = 0.62
    row_gap = 0.42
    max_label_x = n_cols + 0.75
    for i, (label, colors) in enumerate(strip_items):
        y = (len(strip_items) - 1 - i) * (strip_h + row_gap)
        for j, color in enumerate(colors):
            rect = mpl.patches.Rectangle((j, y), 1.0, strip_h, facecolor=color, edgecolor=EDGE_STRIP_BORDER_COLOR, linewidth=EDGE_STRIP_BORDER_WIDTH)
            ax.add_patch(rect)
        ax.text(max_label_x, y + strip_h / 2, label, ha='left', va='center', fontsize=EDGE_STRIP_FONTSIZE, color='#000000')
    ax.set_xlim(-0.1, n_cols + 6.5)
    ax.set_ylim(-0.15, 3 * strip_h + 2 * row_gap + 0.15)

def setup_map_ax(ax):
    ax.set_facecolor(OCEAN_COLOR)
    if HAS_CARTOPY:
        ax.add_feature(cfeature.OCEAN.with_scale('110m'), facecolor=OCEAN_COLOR, edgecolor='none', zorder=0)
        ax.add_feature(cfeature.LAND.with_scale('110m'), facecolor=LAND_BG_COLOR, edgecolor='none', zorder=0.5)
        ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0, edgecolor=COAST_COLOR, zorder=3)
        ax.add_feature(cfeature.BORDERS.with_scale('110m'), linewidth=0.15, edgecolor=BORDER_COLOR, zorder=3)
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    else:
        ax.set_xlim(LON_MIN, LON_MAX)
        ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_xticks([])
    ax.set_yticks([])
    if 'geo' in ax.spines:
        ax.spines['geo'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

def panel_title_transform(fig, transform, panel_label, fontsize):
    prefix_width, _, _ = fig.canvas.get_renderer().get_text_width_height_descent(f'{panel_label.upper()}. ', FontProperties(size=fontsize), ismath=False)
    return transform + ScaledTranslation(prefix_width / fig.dpi, 0, fig.dpi_scale_trans)

def add_panel_title(ax, title, panel_label=None, fontsize=14):
    title_transform = ax.transAxes
    if panel_label is not None:
        ax.text(0.01, 1.08, panel_label.lower(), transform=ax.transAxes, ha='left', va='top', fontsize=fontsize, fontweight='bold', color='#000000', zorder=10, clip_on=False)
        title_transform = panel_title_transform(ax.figure, ax.transAxes, panel_label, fontsize)
    ax.text(0.01, 1.08, title, transform=title_transform, ha='left', va='top', fontsize=fontsize, color='#000000', zorder=10, clip_on=False)

def plot_ternary_panel(ax, rgba, extent, panel_label='d'):
    setup_map_ax(ax)
    imshow_kwargs = {'origin': 'upper', 'extent': extent, 'interpolation': 'nearest', 'zorder': 2}
    if HAS_CARTOPY:
        imshow_kwargs['transform'] = ccrs.PlateCarree()
    ax.imshow(rgba, **imshow_kwargs)
    add_panel_title(ax, 'Ternary map of vegetation, bright substrate, and water', panel_label=panel_label, fontsize=TERNARY_TITLE_FONTSIZE)
    ax_leg = ax.inset_axes([TERNARY_LEGEND_LEFT, TERNARY_LEGEND_BOTTOM, TERNARY_LEGEND_WIDTH, TERNARY_LEGEND_HEIGHT])
    ax_leg.set_facecolor((1, 1, 1, 0))
    draw_honeycomb_ternary_legend(ax_leg, top_label=SOIL_NAME, left_label=VEG_NAME, right_label=WAT_NAME, tick_values=(0, 0.2, 0.4, 0.6, 0.8, 1.0), tick_fs=TERNARY_LEGEND_TICK_FONTSIZE, label_fs=TERNARY_LEGEND_LABEL_FONTSIZE, label_gap=TERNARY_LEGEND_LABEL_GAP)
    if DRAW_EDGE_STRIP_LEGEND:
        ax_edge = ax.inset_axes([EDGE_STRIP_LEFT, EDGE_STRIP_BOTTOM, EDGE_STRIP_WIDTH, EDGE_STRIP_HEIGHT])
        ax_edge.set_facecolor((1, 1, 1, 0))
        draw_outer_edge_hex_strip_legend(ax_edge)

def plot_single_variable_panel(fig, ax, data, extent, title, panel_label, cmap, vmin=0, vmax=1, ticks=None, cbar_label='Value'):
    setup_map_ax(ax)
    data_ma = np.ma.masked_invalid(data)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    imshow_kwargs = {'origin': 'upper', 'extent': extent, 'cmap': cmap, 'norm': norm, 'interpolation': 'nearest', 'zorder': 2}
    if HAS_CARTOPY:
        imshow_kwargs['transform'] = ccrs.PlateCarree()
    im = ax.imshow(data_ma, **imshow_kwargs)
    add_panel_title(ax, title, panel_label=panel_label, fontsize=SINGLE_MAP_TITLE_FONTSIZE)
    bbox = ax.get_position()
    cbar_width = bbox.width * SINGLE_CBAR_WIDTH_RATIO
    cbar_height = SINGLE_CBAR_HEIGHT
    cbar_left = bbox.x0 + bbox.width * SINGLE_CBAR_LEFT_RATIO
    cbar_bottom = bbox.y0 - SINGLE_CBAR_BOTTOM_OFFSET
    cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    if ticks is None:
        ticks = SINGLE_CBAR_TICKS
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=ticks)
    cbar.outline.set_linewidth(1.1)
    cbar.outline.set_edgecolor('#555555')
    cbar.ax.tick_params(labelsize=SINGLE_CBAR_TICK_FONTSIZE, width=0.5, length=3.5, pad=2, colors='#000000')
    cbar.set_label(cbar_label, fontsize=SINGLE_CBAR_LABEL_FONTSIZE, labelpad=SINGLE_CBAR_LABEL_PAD, color='#000000')
    return im

def add_figure_panel_title(fig, x, y, title, panel_label, fontsize=15):
    fig.text(x, y, panel_label.lower(), ha='left', va='top', fontsize=fontsize, fontweight='bold', color='#000000')
    fig.text(x, y, title, transform=panel_title_transform(fig, fig.transFigure, panel_label, fontsize), ha='left', va='top', fontsize=fontsize, color='#000000')

def extend_canvas_preserving_existing_axes(fig, new_height_in, bottom_extra_in):
    old_width_in, old_height_in = fig.get_size_inches()
    old_positions = [(ax, ax.get_position().frozen()) for ax in fig.axes]
    fig.set_size_inches(old_width_in, new_height_in, forward=True)
    for ax, pos in old_positions:
        ax.set_position([pos.x0, (pos.y0 * old_height_in + bottom_extra_in) / new_height_in, pos.width, pos.height * old_height_in / new_height_in])

def draw_city_endmember_grid(fig, bounds, veg_cmap, soil_cmap, wat_cmap):
    left, bottom, right, top = bounds
    add_figure_panel_title(fig, left, top + 0.032, 'Representative-city endmember fractions', 'e', fontsize=SINGLE_MAP_TITLE_FONTSIZE)
    body_bottom = bottom + CITY_GRID_BODY_Y_SHIFT
    body_top = top + CITY_GRID_BODY_Y_SHIFT
    grid_left = left + CITY_ROW_LABEL_W
    grid_right = right
    grid_top = body_top - CITY_GRID_HEADER_H
    grid_bottom = body_bottom
    grid_w = grid_right - grid_left
    grid_h = grid_top - grid_bottom
    n_cols = len(CITY_ORDER)
    n_rows = len(CITY_ROW_LABELS)
    cell_w = (grid_w - (n_cols - 1) * CITY_GRID_COL_GAP) / n_cols
    cell_h = (grid_h - (n_rows - 1) * CITY_GRID_ROW_GAP) / n_rows
    fig_w_in, fig_h_in = fig.get_size_inches()
    map_w = CITY_MAP_SIZE_IN / fig_w_in
    map_h = CITY_MAP_SIZE_IN / fig_h_in
    map_scale = min(1.0, cell_w * 0.96 / map_w, cell_h * 0.96 / map_h)
    map_w *= map_scale
    map_h *= map_scale
    fig.text(left + CITY_ROW_LABEL_W * 0.43, body_top - CITY_GRID_HEADER_H * 0.52, 'City', ha='center', va='center', fontsize=CITY_CITY_LABEL_FONTSIZE, color='#000000')
    for col, city in enumerate(CITY_ORDER):
        cell_x0 = grid_left + col * (cell_w + CITY_GRID_COL_GAP)
        fig.text(cell_x0 + cell_w / 2, body_top - CITY_GRID_HEADER_H * 0.52, city['label'], ha='center', va='center', fontsize=CITY_CITY_LABEL_FONTSIZE, color='#000000')
    previews = {}
    for city in CITY_ORDER:
        previews[city['key']] = read_city_fraction_preview(city_clip_path(city))
    scalar_cmaps = [veg_cmap, soil_cmap, wat_cmap]
    city_layers = []
    for row, row_label in enumerate(CITY_ROW_LABELS):
        y0 = grid_top - (row + 1) * cell_h - row * CITY_GRID_ROW_GAP
        fig.text(left + CITY_ROW_LABEL_W * 0.43, y0 + cell_h / 2, row_label, ha='center', va='center', fontsize=CITY_ROW_LABEL_FONTSIZE, color='#000000', rotation=90, rotation_mode='anchor')
        for col, city in enumerate(CITY_ORDER):
            cell_x0 = grid_left + col * (cell_w + CITY_GRID_COL_GAP)
            map_x0 = cell_x0 + (cell_w - map_w) / 2.0
            map_y0 = y0 + (cell_h - map_h) / 2.0
            display_extent = city_circle_square_extent(city, pad_ratio=0.05)
            ax = fig.add_axes([map_x0, map_y0, map_w, map_h])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor('#FFFFFF')
            for spine in ax.spines.values():
                spine.set_visible(False)
            preview = previews[city['key']]
            arr = preview['array']
            if row < 3:
                ax.imshow(arr[row], extent=preview['extent'], origin='upper', cmap=scalar_cmaps[row], vmin=0.0, vmax=1.0, interpolation='nearest', rasterized=True)
            else:
                google = fetch_esri_satellite_preview(city)
                if google is None:
                    ax.set_facecolor('#F7F8FA')
                    ax.text(0.5, 0.5, 'image\npending', transform=ax.transAxes, ha='center', va='center', fontsize=CITY_ROW_LABEL_FONTSIZE, color='#000000', linespacing=0.92)
                else:
                    rgb = mask_rgb_to_city_circle(google['array'], google['extent'], city)
                    ax.imshow(rgb, extent=google['extent'], origin='upper', interpolation='lanczos', rasterized=True)
            ax.set_xlim(display_extent[0], display_extent[1])
            ax.set_ylim(display_extent[2], display_extent[3])
            ax.set_aspect('equal', adjustable='box')
            draw_city_circle_boundary(ax, city)
            city_layers.append({'city': city['label'], 'row': row_label, 'clip': str(city_clip_path(city)), 'google_tile_source': 'Esri World Imagery' if row_label == 'Google image' else None})
    return {'city_grid_bounds': [left, bottom, right, top], 'city_fraction_layers': city_layers, 'city_circle_mask_dir': str(CITY_CIRCLE_DIR), 'google_image_cache_dir': str(CITY_GOOGLE_CACHE_DIR), 'google_image_errors': _CITY_GOOGLE_ERRORS}
print('正在读取降采样后的栅格，用于绘图...')
veg, veg_meta = read_raster_for_display(veg_tif, stride=DISPLAY_STRIDE, clean_to_0_1=CLEAN_TO_0_1)
soil, soil_meta = read_raster_for_display(soil_tif, stride=DISPLAY_STRIDE, clean_to_0_1=CLEAN_TO_0_1)
wat, wat_meta = read_raster_for_display(wat_tif, stride=DISPLAY_STRIDE, clean_to_0_1=CLEAN_TO_0_1)
if not same_grid(soil_meta, veg_meta):
    print('Bare soil 栅格重采样到 Vegetation 显示网格...')
    soil = reproject_to_match(soil, soil_meta, veg_meta)
if not same_grid(wat_meta, veg_meta):
    print('Water 栅格重采样到 Vegetation 显示网格...')
    wat = reproject_to_match(wat, wat_meta, veg_meta)
print('栅格读取完成。')
print(f'显示数组尺寸: {veg.shape}')
total_raw = veg + soil + wat
valid_raw = np.isfinite(veg) & np.isfinite(soil) & np.isfinite(wat)
if np.any(valid_raw):
    diff = np.abs(total_raw[valid_raw] - TOTAL_TARGET)
    print('\n逐像素总和与 1 的偏差统计：')
    print(f'  最小偏差: {np.nanmin(diff):.6f}')
    print(f'  中位偏差: {np.nanmedian(diff):.6f}')
    print(f'  平均偏差: {np.nanmean(diff):.6f}')
    print(f'  最大偏差: {np.nanmax(diff):.6f}')
print('正在生成三元图 RGBA...')
rgba, valid = build_ternary_rgba_memory_safe(veg, soil, wat, renormalize=RENORMALIZE)
extent = get_extent_from_transform(veg_meta['transform'], veg_meta['width'], veg_meta['height'])
print('三元图生成完成。')
veg_cmap = plt.get_cmap('YlGn').copy()
soil_cmap = plt.get_cmap('magma_r').copy()
wat_cmap = plt.get_cmap('GnBu').copy()
veg_cmap.set_bad((1, 1, 1, 0))
soil_cmap.set_bad((1, 1, 1, 0))
wat_cmap.set_bad((1, 1, 1, 0))
print('正在裁剪六城市圆圈范围的三端元 fraction 数据...')
city_clip_records = prepare_city_fraction_clips(overwrite=False)
print('六城市圆圈裁剪完成。')
subplot_kw = {'projection': ccrs.PlateCarree()} if HAS_CARTOPY else {}
FIG_WIDTH_IN = 18.0
TOP_PANEL_HEIGHT_IN = 10.0
CITY_PANEL_HEIGHT_IN = 13.0
FIG_HEIGHT_IN = TOP_PANEL_HEIGHT_IN + CITY_PANEL_HEIGHT_IN
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(FIG_WIDTH_IN, TOP_PANEL_HEIGHT_IN), subplot_kw=subplot_kw, facecolor='white')
plt.subplots_adjust(left=0.025, right=0.985, top=0.955, bottom=0.08, wspace=0.05, hspace=0.31)
ax1 = axes[0, 0]
ax2 = axes[0, 1]
ax3 = axes[1, 0]
ax4 = axes[1, 1]
plot_single_variable_panel(fig=fig, ax=ax1, data=veg, extent=extent, title=f'{VEG_NAME}', panel_label='a', cmap=veg_cmap, vmin=SINGLE_VMIN, vmax=SINGLE_VMAX, ticks=SINGLE_CBAR_TICKS, cbar_label='Endmember fraction')
plot_single_variable_panel(fig=fig, ax=ax2, data=soil, extent=extent, title=f'{SOIL_NAME}', panel_label='b', cmap=soil_cmap, vmin=SINGLE_VMIN, vmax=SINGLE_VMAX, ticks=SINGLE_CBAR_TICKS, cbar_label='Endmember fraction')
plot_single_variable_panel(fig=fig, ax=ax3, data=wat, extent=extent, title=f'{WAT_NAME}', panel_label='c', cmap=wat_cmap, vmin=SINGLE_VMIN, vmax=SINGLE_VMAX, ticks=SINGLE_CBAR_TICKS, cbar_label='Endmember fraction')
plot_ternary_panel(ax4, rgba=rgba, extent=extent, panel_label='d')
extend_canvas_preserving_existing_axes(fig, new_height_in=FIG_HEIGHT_IN, bottom_extra_in=CITY_PANEL_HEIGHT_IN)
city_grid_info = draw_city_endmember_grid(fig=fig, bounds=(CITY_GRID_LEFT, CITY_GRID_BOTTOM, CITY_GRID_RIGHT, CITY_GRID_TOP), veg_cmap=veg_cmap, soil_cmap=soil_cmap, wat_cmap=wat_cmap)
fig.savefig(output_png, dpi=EXPORT_DPI, bbox_inches='tight', facecolor='white')
if SAVE_PDF:
    fig.savefig(output_pdf, dpi=EXPORT_DPI, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('输出完成：')
print(output_png)
print(output_pdf)

def _build_manuscript_figure():
    return None, None


if __name__ == "__main__":
    _configure_land_functions(sys.modules[__name__])
    _figure, _bbox = _build_manuscript_figure()
    _save_figure(_figure, _bbox, 'Supplementary_Fig_S5', _figure_options)
