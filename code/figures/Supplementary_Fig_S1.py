"""Plot Supplementary_Fig_S1 using the packaged numerical tables or supplied local inputs."""
from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_support"))
from figure_cli import prepare_figure as _prepare_figure, save_figure as _save_figure
from figure_cli import read_table as _read_table, model_tables as _model_tables
from figure_cli import configure_land_functions as _configure_land_functions

_figure_options = _prepare_figure('Supplementary_Fig_S1', 'external_inputs_required') if __name__ == "__main__" else None

from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import array_bounds
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
warnings.filterwarnings('ignore')
PROJECT_ROOT = PACKAGE_ROOT
INPUT_SHP = os.path.join(PROJECT_ROOT, '全球城市数据', 'global_city_point.shp')
OUTPUT_DIR = output_dir('Supplementary_Fig_S1')
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PNG = str(output_dir('Supplementary_Fig_S1') / 'Supplementary_Fig_S1.png')
OUTPUT_PDF = str(output_dir('Supplementary_Fig_S1') / 'Supplementary_Fig_S1.pdf')
RASTER_FILES = {'beauty': os.path.join(OUTPUT_DIR, 'beauty.tif'), 'safer': os.path.join(OUTPUT_DIR, 'safer.tif'), 'livelier': os.path.join(OUTPUT_DIR, 'livelier.tif'), 'wealthy': os.path.join(OUTPUT_DIR, 'wealthy.tif'), 'boring': os.path.join(OUTPUT_DIR, 'boring.tif'), 'depression': os.path.join(OUTPUT_DIR, 'depression.tif')}
VARIABLES = [('beauty', 'Beauty'), ('safer', 'Safety'), ('livelier', 'Liveliness'), ('wealthy', 'Wealth'), ('boring', 'Boredom'), ('depression', 'Depression')]
REVERSE_NEGATIVE = False
NEGATIVE_FIELDS = ['boring', 'depression']
VAR_LABEL_DICT = {'beauty': 'Beauty', 'safer': 'Safety', 'livelier': 'Liveliness', 'wealthy': 'Wealth', 'boring': 'Boredom', 'depression': 'Depression'}
NCLASS = 10
PANEL_RATIO_W = 5
PANEL_RATIO_H = 3.0
PANEL_BOX_ASPECT = PANEL_RATIO_H / PANEL_RATIO_W
CLASS_MODE = 'fixed_auto'
MANUAL_RANGE = None
EMOTION_RENDER_RANGES = {'beauty': (0.2, 0.7), 'safer': (0.3, 0.85), 'livelier': (0.1, 0.6), 'wealthy': (0.2, 0.85), 'boring': (0.3, 0.55), 'depression': (0.5, 0.9)}
LEGEND_DISPLAY_RANGE = (0, 0.8)
RASTER_RENDER_RANGES = {'beauty': (0.2, 0.9), 'safer': (0.6, 1.0), 'livelier': (0.2, 0.8), 'wealthy': (0.3, 0.85), 'boring': (0.3, 0.5), 'depression': (0.55, 0.9)}
RASTER_AUTO_PERCENTILES = (2, 98)
LAT_MIN, LAT_MAX = (-60, 83)
LON_MIN, LON_MAX = (-180, 180)
POINT_SIZE = 1
POINT_ALPHA = 1
DPI = 600
OCEAN_COLOR = '#FFFFFF'
LAND_COLOR = '#E9EBF1'
PANEL_FACE = '#FFFFFF'
PANEL_EDGE = '#2A2A2A'
PANEL_LW = 1.0
SHOW_GRIDLINES = False
LEGEND_SCALE = 1.0
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Roboto', 'Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['savefig.facecolor'] = 'white'
mpl.rcParams['axes.unicode_minus'] = False
WGS84_GLOBE = ccrs.Globe(datum='WGS84')
WGS84_PROJ = ccrs.PlateCarree(globe=WGS84_GLOBE)
palette_rgb = np.array([[(114, 47, 118), (155, 66, 136), (180, 83, 151), (200, 106, 163), (216, 122, 175), (230, 143, 178), (236, 130, 147), (241, 118, 116), (246, 106, 84), (252, 93, 53)], [(139, 72, 150), (174, 95, 166), (199, 118, 180), (215, 135, 189), (229, 158, 197), (235, 168, 194), (239, 156, 161), (242, 145, 129), (246, 134, 96), (250, 123, 64)], [(151, 100, 175), (181, 127, 189), (201, 151, 201), (223, 169, 208), (234, 186, 213), (240, 192, 210), (242, 182, 176), (244, 173, 142), (246, 163, 108), (248, 154, 74)], [(161, 128, 194), (191, 159, 210), (214, 180, 219), (229, 195, 224), (240, 208, 229), (245, 216, 226), (245, 208, 191), (245, 200, 156), (245, 192, 120), (245, 184, 85)], [(167, 152, 214), (200, 183, 225), (221, 207, 235), (234, 216, 235), (243, 231, 241), (250, 241, 242), (248, 234, 206), (246, 228, 169), (245, 221, 132), (243, 214, 96)], [(152, 163, 221), (180, 190, 231), (205, 213, 239), (224, 226, 243), (238, 239, 246), (249, 246, 232), (250, 246, 214), (245, 240, 180), (238, 233, 147), (222, 212, 83)], [(130, 148, 212), (159, 177, 225), (191, 205, 236), (211, 219, 241), (228, 234, 244), (244, 244, 225), (242, 241, 192), (232, 233, 170), (218, 223, 134), (190, 203, 74)], [(106, 131, 204), (134, 160, 217), (165, 188, 230), (186, 205, 236), (211, 224, 241), (231, 237, 222), (227, 232, 194), (216, 223, 162), (197, 211, 126), (165, 191, 68)], [(90, 113, 193), (112, 143, 209), (139, 170, 225), (158, 190, 230), (184, 207, 236), (213, 224, 214), (204, 219, 180), (189, 208, 146), (169, 197, 112), (128, 169, 63)], [(72, 91, 179), (96, 124, 199), (115, 147, 214), (128, 167, 222), (157, 188, 229), (188, 208, 202), (177, 201, 160), (163, 191, 128), (142, 177, 97), (88, 140, 52)]], dtype=float) / 255.0

def load_city_points(shp_path, variables, reverse_negative=False):
    if not os.path.exists(resource(shp_path)):
        raise FileNotFoundError(f'未找到 shp 文件: {shp_path}')
    gdf = gpd.read_file(resource(shp_path))
    if gdf.crs is None:
        print('[提示] 当前 shp 没有 CRS 信息，默认指定为 WGS 1984 / EPSG:4326。')
        gdf = gdf.set_crs(epsg=4326)
    else:
        print('原始坐标系:', gdf.crs)
    gdf = gdf.to_crs(epsg=4326)
    print('已统一转换为 WGS 1984 / EPSG:4326。')
    gdf = gdf[gdf.geometry.notna()].copy()
    geom_types = set(gdf.geometry.geom_type.unique())
    if not geom_types.issubset({'Point'}):
        print('[提示] 输入数据不是纯 Point，将使用 representative_point() 作为城市点位置。')
        pts = gdf.geometry.representative_point()
    else:
        pts = gdf.geometry
    gdf['lon'] = pts.x
    gdf['lat'] = pts.y
    gdf = gdf[(gdf['lon'] >= LON_MIN) & (gdf['lon'] <= LON_MAX) & (gdf['lat'] >= LAT_MIN) & (gdf['lat'] <= LAT_MAX)].copy()
    fields = [v[0] for v in variables]
    missing = [f for f in fields if f not in gdf.columns]
    if missing:
        raise ValueError(f'以下字段在 shp 中不存在: {missing}\n当前 shp 字段包括: {list(gdf.columns)}')
    for f in fields:
        gdf[f] = pd.to_numeric(gdf[f], errors='coerce')
    if reverse_negative:
        for f in NEGATIVE_FIELDS:
            if f in gdf.columns:
                max_val = np.nanmax(gdf[f].values)
                min_val = np.nanmin(gdf[f].values)
                if min_val >= 0 and max_val <= 1.2:
                    gdf[f] = 1.0 - gdf[f]
                elif min_val >= 0 and max_val <= 100:
                    gdf[f] = 100.0 - gdf[f]
                else:
                    gdf[f] = -1.0 * gdf[f]
                print(f'[提示] 已反向处理字段: {f}')
    gdf = gdf.dropna(subset=fields, how='all').copy()
    gdf = gdf.sample(frac=1, random_state=42).reset_index(drop=True)
    print('WGS84 经度范围:', gdf['lon'].min(), '~', gdf['lon'].max())
    print('WGS84 纬度范围:', gdf['lat'].min(), '~', gdf['lat'].max())
    return gdf

def determine_global_fixed_range(gdf, fields):
    if MANUAL_RANGE is not None:
        return MANUAL_RANGE
    vals = []
    for f in fields:
        arr = gdf[f].values.astype(float)
        arr = arr[np.isfinite(arr)]
        vals.append(arr)
    vals = np.concatenate(vals)
    vmin = np.nanmin(vals)
    vmax = np.nanmax(vals)
    if vmin >= 0 and vmax <= 1.2:
        return (0.0, 1.0)
    elif vmin >= 0 and vmax <= 100:
        return (0.0, 100.0)
    else:
        return (float(vmin), float(vmax))

def classify_fixed(values, vmin, vmax, nclass=10):
    values = np.asarray(values, dtype=float)
    idx = np.full(values.shape, -1, dtype=int)
    valid = np.isfinite(values)
    if valid.sum() == 0:
        return idx
    edges = np.linspace(vmin, vmax, nclass + 1)
    clipped = np.clip(values[valid], vmin, vmax)
    idx_valid = np.digitize(clipped, bins=edges[1:-1], right=False)
    idx_valid = np.clip(idx_valid, 0, nclass - 1)
    idx[valid] = idx_valid
    return idx

def classify_quantile(values, nclass=10):
    values = np.asarray(values, dtype=float)
    idx = np.full(values.shape, -1, dtype=int)
    valid = np.isfinite(values)
    if valid.sum() == 0:
        return idx
    s = pd.Series(values[valid])
    try:
        cls = pd.qcut(s, q=nclass, labels=False, duplicates='drop').astype(float).values
        cmin, cmax = (np.nanmin(cls), np.nanmax(cls))
        if cmax > cmin:
            cls = np.round((cls - cmin) / (cmax - cmin) * (nclass - 1)).astype(int)
        else:
            cls = np.zeros_like(cls, dtype=int)
        idx[valid] = np.clip(cls, 0, nclass - 1)
    except Exception:
        vmin, vmax = (np.nanmin(values[valid]), np.nanmax(values[valid]))
        idx[valid] = classify_fixed(values[valid], vmin, vmax, nclass)
    return idx

def build_class_columns(gdf, variables):
    fields = [v[0] for v in variables]
    if CLASS_MODE == 'fixed_auto':
        global_vmin, global_vmax = determine_global_fixed_range(gdf, fields)
        print('[分类方式] fixed_auto + variable-specific render ranges')
        print(f'[默认全局范围] {global_vmin} ~ {global_vmax}')
        render_range_info = {}
        for f in fields:
            if f in EMOTION_RENDER_RANGES:
                vmin, vmax = EMOTION_RENDER_RANGES[f]
            elif MANUAL_RANGE is not None:
                vmin, vmax = MANUAL_RANGE
            else:
                vmin, vmax = (global_vmin, global_vmax)
            if vmax <= vmin:
                raise ValueError(f'{f} 的渲染范围设置错误：vmax 必须大于 vmin，目前为 {vmin} ~ {vmax}')
            gdf[f'{f}_cls'] = classify_fixed(gdf[f].values, vmin, vmax, NCLASS)
            render_range_info[f] = (vmin, vmax)
            print(f'[全球点渲染范围] {f}: {vmin} ~ {vmax}')
        legend_mode = 'fixed'
        if LEGEND_DISPLAY_RANGE is not None:
            legend_range = LEGEND_DISPLAY_RANGE
        elif MANUAL_RANGE is not None:
            legend_range = MANUAL_RANGE
        else:
            legend_range = (global_vmin, global_vmax)
        print(f'[全球点图例显示范围] {legend_range[0]} ~ {legend_range[1]}')
    elif CLASS_MODE == 'quantile':
        print('[分类方式] quantile: 每个情绪变量按自身分位数分为10类')
        render_range_info = {}
        for f in fields:
            gdf[f'{f}_cls'] = classify_quantile(gdf[f].values, NCLASS)
            render_range_info[f] = 'quantile'
        legend_mode = 'quantile'
        legend_range = None
    else:
        raise ValueError("CLASS_MODE 只能是 'fixed_auto' 或 'quantile'")
    return (gdf, legend_mode, legend_range, render_range_info)

def make_point_rgba(x_cls, y_cls, palette, alpha=0.88):
    x_cls = np.asarray(x_cls, dtype=int)
    y_cls = np.asarray(y_cls, dtype=int)
    rgba = np.zeros((len(x_cls), 4), dtype=float)
    valid = (x_cls >= 0) & (y_cls >= 0)
    rows = (NCLASS - 1 - y_cls[valid]).astype(int)
    cols = x_cls[valid].astype(int)
    rgba[valid, :3] = palette[rows, cols]
    rgba[valid, 3] = alpha
    rgba[~valid, 3] = 0.0
    return (rgba, valid)

def read_single_band_raster(path):
    if not os.path.exists(resource(path)):
        raise FileNotFoundError(f'未找到栅格文件: {path}')
    with rasterio.open(resource(path)) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    arr[~np.isfinite(arr)] = np.nan
    return (arr, profile, transform, crs)

def load_all_rasters(raster_files):
    data = {}
    meta_ref = None
    for k, fp in raster_files.items():
        arr, profile, transform, crs = read_single_band_raster(fp)
        if meta_ref is None:
            meta_ref = {'shape': arr.shape, 'transform': transform, 'crs': crs, 'width': profile['width'], 'height': profile['height']}
        else:
            if arr.shape != meta_ref['shape']:
                raise ValueError(f"{k} 的栅格 shape 与参考栅格不一致：{arr.shape} vs {meta_ref['shape']}")
            if transform != meta_ref['transform']:
                raise ValueError(f'{k} 的 transform 与参考栅格不一致，请先重采样/对齐。')
            if crs != meta_ref['crs']:
                raise ValueError(f'{k} 的 CRS 与参考栅格不一致，请先统一投影。')
        data[k] = arr
    return (data, meta_ref)

def get_raster_render_range(arr, manual_range=None, pct=(2, 98), var_name=''):
    if manual_range is not None:
        vmin, vmax = manual_range
        if vmax < vmin:
            print(f'[提醒] 栅格 {var_name} 渲染范围写反：({vmin}, {vmax})，已自动交换。')
            vmin, vmax = (vmax, vmin)
        if vmax == vmin:
            vmax = vmin + 1e-06
    else:
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            vmin, vmax = (0.0, 1.0)
        else:
            vmin, vmax = (np.nanpercentile(valid, pct[0]), np.nanpercentile(valid, pct[1]))
        if vmax <= vmin:
            vmax = vmin + 1e-06
    return (float(vmin), float(vmax))

def normalize_clip(arr, vmin, vmax):
    out = (arr - vmin) / (vmax - vmin)
    out = np.clip(out, 0, 1)
    out[~np.isfinite(arr)] = np.nan
    return out

def classify_raster_to_nclass(arr_norm, nclass=10):
    cls = np.full(arr_norm.shape, -1, dtype=np.int16)
    valid = np.isfinite(arr_norm)
    cls_valid = np.floor(arr_norm[valid] * nclass).astype(np.int16)
    cls_valid = np.clip(cls_valid, 0, nclass - 1)
    cls[valid] = cls_valid
    return cls

def make_raster_bivariate_rgba(x_cls, y_cls, palette):
    h, w = x_cls.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    valid = (x_cls >= 0) & (y_cls >= 0)
    rows = (NCLASS - 1 - y_cls[valid]).astype(int)
    cols = x_cls[valid].astype(int)
    rgba[valid, :3] = palette[rows, cols, :]
    rgba[valid, 3] = 1.0
    return rgba

def raster_extent_from_meta(meta):
    h = meta['height']
    w = meta['width']
    transform = meta['transform']
    left, bottom, right, top = array_bounds(h, w, transform)
    return [left, right, bottom, top]

def build_raster_class_data(raster_files):
    raster_data, raster_meta = load_all_rasters(raster_files)
    raster_extent = raster_extent_from_meta(raster_meta)
    raster_cls = {}
    raster_render_info = {}
    for var in [v[0] for v in VARIABLES]:
        arr = raster_data[var]
        manual_range = RASTER_RENDER_RANGES.get(var, None)
        vmin, vmax = get_raster_render_range(arr, manual_range=manual_range, pct=RASTER_AUTO_PERCENTILES, var_name=var)
        arr_norm = normalize_clip(arr, vmin, vmax)
        arr_cls = classify_raster_to_nclass(arr_norm, NCLASS)
        raster_cls[var] = arr_cls
        raster_render_info[var] = (vmin, vmax)
        print(f'[栅格渲染范围] {var}: {vmin:.4f} ~ {vmax:.4f}')
    return (raster_cls, raster_extent, raster_render_info)

def style_panel_box(ax_box):
    ax_box.set_facecolor(PANEL_FACE)
    ax_box.set_xticks([])
    ax_box.set_yticks([])
    ax_box.set_xlim(0, 1)
    ax_box.set_ylim(0, 1)
    ax_box.set_box_aspect(PANEL_BOX_ASPECT)
    for spine in ax_box.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(PANEL_LW)
        spine.set_edgecolor(PANEL_EDGE)
    return ax_box

def create_boxed_map_panel(fig, gs_cell, proj, gdf, x_field, y_field, label_text=None):
    ax_box = fig.add_subplot(gs_cell)
    style_panel_box(ax_box)
    x_label = VAR_LABEL_DICT.get(x_field, x_field)
    y_label = VAR_LABEL_DICT.get(y_field, y_field)
    ax_box.text(0.03, 0.95, f'X: {x_label}   Y: {y_label}', ha='left', va='top', fontsize=10, color='#2A2A2A', linespacing=1.12)
    pos = ax_box.get_position()
    left = pos.x0 + 0.005 * pos.width
    bottom = pos.y0 + 0.04 * pos.height
    width = pos.width * 0.99
    height = pos.height * 0.72
    ax_map = fig.add_axes([left, bottom, width, height], projection=proj)
    draw_pair_map(ax_map, gdf, x_field, y_field)
    return (ax_box, ax_map)

def create_boxed_raster_panel(fig, gs_cell, raster_cls, raster_extent, x_field, y_field):
    ax_box = fig.add_subplot(gs_cell)
    style_panel_box(ax_box)
    x_label = VAR_LABEL_DICT.get(x_field, x_field)
    y_label = VAR_LABEL_DICT.get(y_field, y_field)
    ax_box.text(0.03, 0.95, f'X: {x_label}   Y: {y_label}', ha='left', va='top', fontsize=10, color='#2A2A2A', linespacing=1.12)
    pos = ax_box.get_position()
    left = pos.x0 + 0.005 * pos.width
    bottom = pos.y0 + 0.04 * pos.height
    width = pos.width * 0.99
    height = pos.height * 0.78
    ax_raster = fig.add_axes([left, bottom, width, height])
    x_cls = raster_cls[x_field]
    y_cls = raster_cls[y_field]
    rgba = make_raster_bivariate_rgba(x_cls=x_cls, y_cls=y_cls, palette=palette_rgb)
    ax_raster.imshow(rgba, extent=raster_extent, origin='upper', interpolation='nearest')
    ax_raster.set_xticks([])
    ax_raster.set_yticks([])
    ax_raster.set_aspect('equal')
    for sp in ax_raster.spines.values():
        sp.set_visible(False)
    return (ax_box, ax_raster)

def create_diag_panel(fig, gs_cell, var_field, var_label, raster_render_info=None):
    ax = fig.add_subplot(gs_cell)
    style_panel_box(ax)
    ax.text(0.5, 0.65, var_label, ha='center', va='center', fontsize=13, fontweight='normal', color='#5A71C1')
    if raster_render_info is not None and var_field in raster_render_info:
        vmin, vmax = raster_render_info[var_field]
        range_text = f'{vmin:.2f}–{vmax:.2f}'
        ax.text(0.5, 0.45, 'Raster range', ha='center', va='center', fontsize=10, color='#4A4F54')
        ax.text(0.5, 0.3, range_text, ha='center', va='center', fontsize=10, color='#4A4F54')
    return ax

def add_bivariate_legend(fig, palette, box=(0.81, 0.36, 0.17, 0.17), legend_mode='fixed', legend_range=(0, 100), scale=1.0):
    ax_leg = fig.add_axes(box)
    ax_leg.imshow(palette, origin='upper', extent=[0, NCLASS, 0, NCLASS], interpolation='nearest')
    ax_leg.set_aspect('equal', adjustable='box')
    ax_leg.set_xlim(0, NCLASS)
    ax_leg.set_ylim(0, NCLASS)
    for k in range(NCLASS + 1):
        ax_leg.axhline(k, color='white', lw=0.38 * scale, alpha=0.85)
        ax_leg.axvline(k, color='white', lw=0.38 * scale, alpha=0.85)
    for spine in ax_leg.spines.values():
        spine.set_linewidth(0.7 * scale)
        spine.set_edgecolor('#4A4F54')
    if legend_mode == 'fixed' and legend_range is not None:
        vmin, vmax = legend_range
        mid1 = vmin + (vmax - vmin) * 0.25
        mid2 = vmin + (vmax - vmin) * 0.5
        mid3 = vmin + (vmax - vmin) * 0.75

        def fmt(x):
            if vmax <= 1.2:
                return f'{x:.2f}'.rstrip('0').rstrip('.')
            else:
                return f'{x:.2f}'.rstrip('0').rstrip('.')
        labels = [fmt(vmin), fmt(mid1), fmt(mid2), fmt(mid3), fmt(vmax)]
    else:
        labels = ['Low', '', 'Mid', '', 'High']
    ax_leg.set_xticks([0, 2.5, 5, 7.5, 10])
    ax_leg.set_xticklabels(labels, fontsize=10.2, color='#4A4F54')
    ax_leg.set_yticks([0, 2.5, 5, 7.5, 10])
    ax_leg.set_yticklabels(labels, fontsize=10.2, color='#4A4F54')
    ax_leg.tick_params(axis='both', which='major', labelsize=10.2 * scale, width=0.6 * scale, length=3.0 * scale, colors='#4A4F54')
    ax_leg.set_xlabel('X emotion', fontsize=11.5, labelpad=6 * scale, color='#2D3134')
    ax_leg.set_ylabel('Y emotion', fontsize=11.5, labelpad=7 * scale, color='#2D3134')
    ax_leg.text(0.35, 9.65, 'Y emotion ↑\nX emotion ↓', ha='left', va='top', fontsize=13, color='white')
    ax_leg.text(9.65, 9.65, 'Co-high', ha='right', va='top', fontsize=13, color='white')
    ax_leg.text(0.35, 0.35, 'Co-low', ha='left', va='bottom', fontsize=13, color='white')
    ax_leg.text(9.65, 0.35, 'X emotion ↑\nY emotion ↓', ha='right', va='bottom', fontsize=13, color='white')
    return ax_leg

def add_two_band_explanation(fig, palette, box=(0.79, 0.66, 0.21, 0.16), scale=1.0):
    ax_note = fig.add_axes(box)
    ax_note.set_xlim(0, 1)
    ax_note.set_ylim(0, 1)
    ax_note.axis('off')
    band_x0, band_x1 = (0.37, 0.9)
    band_h = 0.145
    column_strip = palette[-1, :, :]
    y0, y1 = (0.6, 0.6 + band_h)
    ax_note.imshow(column_strip[np.newaxis, :, :], extent=[band_x0, band_x1, y0, y1], origin='lower', aspect='auto', interpolation='nearest')
    for k in range(1, column_strip.shape[0]):
        xx = band_x0 + (band_x1 - band_x0) * k / column_strip.shape[0]
        ax_note.plot([xx, xx], [y0, y1], color='white', lw=0.85 * scale, alpha=0.7)
    ax_note.text(0.02, (y0 + y1) / 2, 'X emotion', ha='left', va='center', fontsize=11.5, color='#444444')
    row_strip = palette[::-1, 0, :]
    y0p, y1p = (0.38, 0.38 + band_h)
    ax_note.imshow(row_strip[np.newaxis, :, :], extent=[band_x0, band_x1, y0p, y1p], origin='lower', aspect='auto', interpolation='nearest')
    for k in range(1, row_strip.shape[0]):
        xx = band_x0 + (band_x1 - band_x0) * k / row_strip.shape[0]
        ax_note.plot([xx, xx], [y0p, y1p], color='white', lw=0.85 * scale, alpha=0.7)
    ax_note.text(0.02, (y0p + y1p) / 2, 'Y emotion', ha='left', va='center', fontsize=11.5, color='#444444')
    ax_note.annotate('', xy=(0.91, 0.16), xytext=(0.45, 0.16), xycoords='axes fraction', textcoords='axes fraction', arrowprops=dict(arrowstyle='-|>', lw=1.2 * scale, color='#333333'), annotation_clip=False)
    ax_note.text(0.51, 0.06, 'Lower', ha='center', va='top', fontsize=11.5, color='#333333', transform=ax_note.transAxes)
    ax_note.text(0.83, 0.06, 'Higher', ha='center', va='top', fontsize=11.5, color='#333333', transform=ax_note.transAxes)
    return ax_note

def add_raster_bivariate_legend(fig, palette, box=(0.74, 0.475, 0.3, 0.3), scale=1.0):
    ax_leg = fig.add_axes(box)
    ax_leg.imshow(palette, origin='upper', extent=[0, NCLASS, 0, NCLASS], interpolation='nearest')
    ax_leg.set_aspect('equal', adjustable='box')
    ax_leg.set_xlim(0, NCLASS)
    ax_leg.set_ylim(0, NCLASS)
    for k in range(NCLASS + 1):
        ax_leg.axhline(k, color='white', lw=0.38 * scale, alpha=0.85)
        ax_leg.axvline(k, color='white', lw=0.38 * scale, alpha=0.85)
    ax_leg.set_xticks([0, NCLASS / 2, NCLASS])
    ax_leg.set_xticklabels(['Low', 'Middle', 'High'], fontsize=10.2 * scale, color='#4A4F54')
    ax_leg.set_yticks([0, NCLASS / 2, NCLASS])
    ax_leg.set_yticklabels(['Low', 'Middle', 'High'], fontsize=10.2 * scale, color='#4A4F54')
    ax_leg.tick_params(axis='both', which='major', labelsize=10.2 * scale, width=0.6 * scale, length=3.0 * scale, colors='#4A4F54')
    ax_leg.set_xlabel('X emotion', fontsize=11.5, labelpad=6 * scale, color='#2D3134')
    ax_leg.set_ylabel('Y emotion', fontsize=11.5, labelpad=7 * scale, color='#2D3134')
    ax_leg.text(0.35, 9.65, 'Y emotion ↑\nX emotion ↓', ha='left', va='top', fontsize=13, color='white')
    ax_leg.text(9.65, 9.65, 'Co-high', ha='right', va='top', fontsize=11, color='white')
    ax_leg.text(0.35, 0.35, 'Co-low', ha='left', va='bottom', fontsize=11, color='white')
    ax_leg.text(9.65, 0.35, 'X emotion ↑\nY emotion ↓', ha='right', va='bottom', fontsize=13, color='white')
    for sp in ax_leg.spines.values():
        sp.set_linewidth(0.7 * scale)
        sp.set_edgecolor('#AEB5BA')
    return ax_leg

def draw_pair_map(ax, gdf, x_field, y_field):
    data_crs = WGS84_PROJ
    ax.set_facecolor('white')
    ax.set_anchor('C')
    ax.add_feature(cfeature.OCEAN.with_scale('110m'), facecolor=OCEAN_COLOR, edgecolor='none', zorder=0)
    ax.add_feature(cfeature.LAND.with_scale('110m'), facecolor=LAND_COLOR, edgecolor='none', zorder=1)
    x_cls = gdf[f'{x_field}_cls'].values
    y_cls = gdf[f'{y_field}_cls'].values
    rgba, valid = make_point_rgba(x_cls=x_cls, y_cls=y_cls, palette=palette_rgb, alpha=POINT_ALPHA)
    sub = gdf.loc[valid].copy()
    rgba_valid = rgba[valid]
    ax.scatter(sub['lon'].values, sub['lat'].values, c=rgba_valid, s=POINT_SIZE, marker='o', linewidths=0, transform=data_crs, zorder=4, rasterized=True)
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=data_crs)
    if SHOW_GRIDLINES:
        gl = ax.gridlines(crs=data_crs, draw_labels=False, linewidth=0.25, color='#D6DADD', alpha=0.7, linestyle='-')
        gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
        gl.ylocator = mticker.FixedLocator([-60, -30, 0, 30, 60])
    else:
        ax.gridlines(crs=data_crs, draw_labels=False, linewidth=0.0, color='none', alpha=0)
    if 'geo' in ax.spines:
        ax.spines['geo'].set_visible(False)
    return ax

def get_consistent_xy_fields(i, j, variables):
    idx_x = min(i, j)
    idx_y = max(i, j)
    x_field, x_label = variables[idx_x]
    y_field, y_label = variables[idx_y]
    return (x_field, x_label, y_field, y_label)

def main():
    print('读取全球城市点数据:', INPUT_SHP)
    gdf = load_city_points(INPUT_SHP, VARIABLES, reverse_negative=REVERSE_NEGATIVE)
    fields = [v[0] for v in VARIABLES]
    labels = [v[1] for v in VARIABLES]
    print('城市点数量:', len(gdf))
    print('变量字段:', fields)
    gdf, legend_mode, legend_range, render_range_info = build_class_columns(gdf, VARIABLES)
    print('\n读取并处理栅格数据...')
    raster_cls, raster_extent, raster_render_info = build_raster_class_data(RASTER_FILES)
    n = len(VARIABLES)
    proj = WGS84_PROJ
    fig = plt.figure(figsize=(16.5, 9), facecolor='white')
    gs = fig.add_gridspec(nrows=n, ncols=n, left=0.04, right=0.74, bottom=0.06, top=0.925, wspace=0.035, hspace=-0.25)
    for i in range(n):
        for j in range(n):
            if i == j:
                create_diag_panel(fig, gs[i, j], var_field=fields[i], var_label=labels[i], raster_render_info=raster_render_info)
                continue
            x_field, x_label, y_field, y_label = get_consistent_xy_fields(i, j, VARIABLES)
            if i > j:
                create_boxed_map_panel(fig=fig, gs_cell=gs[i, j], proj=proj, gdf=gdf, x_field=x_field, y_field=y_field, label_text=f'{x_label} × {y_label}')
            else:
                create_boxed_raster_panel(fig=fig, gs_cell=gs[i, j], raster_cls=raster_cls, raster_extent=raster_extent, x_field=x_field, y_field=y_field)
    legend_axes_start = len(fig.axes)
    add_two_band_explanation(fig, palette_rgb, box=(0.765, 0.815, 0.19, 0.135), scale=1.35)
    add_raster_bivariate_legend(fig, palette_rgb, box=(0.803, 0.463, 0.13, 0.2383333333), scale=LEGEND_SCALE)
    add_bivariate_legend(fig, palette_rgb, box=(0.803, 0.106, 0.13, 0.2383333333), legend_mode=legend_mode, legend_range=legend_range, scale=LEGEND_SCALE)
    legend_headers = []
    for y, title in [(0.733, 'Raster scale legend:'), (0.376, 'Global city scale legend:')]:
        header_box = mpl.patches.FancyBboxPatch((0.803, y - 0.013), 0.13, 0.026, boxstyle='round,pad=0,rounding_size=0.001', transform=fig.transFigure, facecolor='#3F89C1', edgecolor='none')
        fig.add_artist(header_box)
        header_text = fig.text(0.868, y, title, ha='center', va='center', fontsize=11.5, color='white')
        legend_headers.append((header_box, header_text))
    fig.canvas.draw()
    export_bbox = fig.get_tightbbox(fig.canvas.get_renderer()).padded(mpl.rcParams['savefig.pad_inches'])
    legend_factor = 0.85
    anchor_x, anchor_y = (0.955, 0.95)
    for ax in fig.axes[legend_axes_start:]:
        x, y, width, height = ax.get_position().bounds
        ax.set_position([anchor_x + legend_factor * (x - anchor_x), anchor_y + legend_factor * (y - anchor_y), width * legend_factor, height * legend_factor])
        for text in ax.findobj(mpl.text.Text):
            text.set_fontsize(text.get_fontsize() * legend_factor)
        for axis in (ax.xaxis, ax.yaxis):
            axis.labelpad *= legend_factor
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                tick.set_pad(tick.get_pad() * legend_factor)
                for line in (tick.tick1line, tick.tick2line):
                    line.set_markersize(line.get_markersize() * legend_factor)
                    line.set_markeredgewidth(line.get_markeredgewidth() * legend_factor)
        for line in ax.lines:
            line.set_linewidth(line.get_linewidth() * legend_factor)
        for spine in ax.spines.values():
            spine.set_linewidth(spine.get_linewidth() * legend_factor)
        for text in ax.texts:
            arrow = getattr(text, 'arrow_patch', None)
            if arrow is not None:
                arrow.set_linewidth(arrow.get_linewidth() * legend_factor)
                arrow.set_mutation_scale(arrow.get_mutation_scale() * legend_factor)
    header_transform = mpl.transforms.Affine2D().translate(-anchor_x, -anchor_y).scale(legend_factor).translate(anchor_x, anchor_y) + fig.transFigure
    for header_box, header_text in legend_headers:
        header_box.set_transform(header_transform)
        header_text.set_transform(header_transform)
        header_text.set_fontsize(header_text.get_fontsize() * legend_factor)
    text_factor = 1.1
    for text in fig.findobj(mpl.text.Text):
        text.set_fontsize(text.get_fontsize() * text_factor)
    fig.set_dpi(DPI)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fitted_labels = 0
    for ax in fig.axes[:legend_axes_start]:
        for text in ax.texts:
            if text.get_text().startswith('X:'):
                available = ax.get_window_extent(renderer).width * 0.94
                actual = text.get_window_extent(renderer).width
                if actual > available:
                    text.set_fontsize(text.get_fontsize() * available / actual)
                    fitted_labels += 1
    for header_box, header_text in legend_headers:
        available = header_box.get_window_extent(renderer).width * 0.96
        actual = header_text.get_window_extent(renderer).width
        if actual > available:
            header_text.set_fontsize(header_text.get_fontsize() * available / actual)
    print(f'Long panel labels fitted to width: {fitted_labels}')
    matrix_bottom = min((ax.get_window_extent(renderer).y0 for ax in fig.axes[:legend_axes_start]))
    legend_bottom = min((ax.get_tightbbox(renderer).y0 for ax in fig.axes[legend_axes_start:]))
    legend_shift = (matrix_bottom - legend_bottom) / fig.bbox.height
    legend_shift_x = -0.015
    for ax in fig.axes[legend_axes_start:]:
        x, y, width, height = ax.get_position().bounds
        ax.set_position([x + legend_shift_x, y + legend_shift, width, height])
    aligned_header_transform = mpl.transforms.Affine2D().translate(-anchor_x, -anchor_y).scale(legend_factor).translate(anchor_x + legend_shift_x, anchor_y + legend_shift) + fig.transFigure
    for header_box, header_text in legend_headers:
        header_box.set_transform(aligned_header_transform)
        header_text.set_transform(aligned_header_transform)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    alignment_error = min((ax.get_tightbbox(renderer).y0 for ax in fig.axes[legend_axes_start:])) - matrix_bottom
    assert abs(alignment_error) < 1.0, alignment_error
    print(f'Legend bottom alignment error: {alignment_error:.3f} pixels')
    fig.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches=export_bbox, facecolor='white')
    fig.savefig(OUTPUT_PDF, dpi=DPI, bbox_inches=export_bbox, facecolor='white')
    plt.show()
    print('\n绘图完成。')
    print('PNG:', OUTPUT_PNG)
    print('PDF:', OUTPUT_PDF)

def _script_entry():
    main()

def _build_manuscript_figure():
    _script_entry()
    return None, None


if __name__ == "__main__":
    _configure_land_functions(sys.modules[__name__])
    _figure, _bbox = _build_manuscript_figure()
    _save_figure(_figure, _bbox, 'Supplementary_Fig_S1', _figure_options)
