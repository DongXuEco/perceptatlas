"""Plot Supplementary_Fig_S13 using the packaged numerical tables or supplied local inputs."""
from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_support"))
from figure_cli import prepare_figure as _prepare_figure, save_figure as _save_figure
from figure_cli import read_table as _read_table, model_tables as _model_tables
from figure_cli import configure_land_functions as _configure_land_functions

_figure_options = _prepare_figure('Supplementary_Fig_S13', 'external_inputs_required') if __name__ == "__main__" else None

from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling
import cartopy.crs as ccrs
import cartopy.feature as cfeature
warnings.filterwarnings('ignore')
base_dir = 'external_inputs/可达性&poi&路网双变量'
accessibility_tif = os.path.join(base_dir, 'global_comprehensive_accessibility_score_100.tif')
road_density_tif = os.path.join(base_dir, 'road.tif')
poi_density_tif = os.path.join(base_dir, 'poi.tif')
output_png = str(output_dir('Supplementary_Fig_S13') / 'Supplementary_Fig_S13.png')
output_pdf = str(output_dir('Supplementary_Fig_S13') / 'Supplementary_Fig_S13.pdf')
ROAD_MIN, ROAD_MAX = (0.0, 8000.0)
POI_MIN, POI_MAX = (0.0, 8.0)
ACCESS_MIN, ACCESS_MAX = (60.0, 100.0)
ACCESS_LOWER_IS_BETTER = False
NCLASS = 10
LAT_MIN, LAT_MAX = (-60, 83)
LON_MIN, LON_MAX = (-180, 180)
DPI = 600
MASK_ZERO_ROAD = False
MASK_ZERO_POI = False
MASK_ZERO_ACCESS = False
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Roboto', 'Arial', 'Helvetica', 'DejaVu Sans']
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['savefig.facecolor'] = 'white'
mpl.rcParams['axes.unicode_minus'] = False
OCEAN_COLOR = '#FFFFFF'
LAND_COLOR = '#252525'
COAST_COLOR = '#BFC5CA'
BORDER_COLOR = '#4E5459'
palette_rgb = np.array([[(114, 47, 118), (155, 66, 136), (180, 83, 151), (200, 106, 163), (216, 122, 175), (230, 143, 178), (236, 130, 147), (241, 118, 116), (246, 106, 84), (252, 93, 53)], [(139, 72, 150), (174, 95, 166), (199, 118, 180), (215, 135, 189), (229, 158, 197), (235, 168, 194), (239, 156, 161), (242, 145, 129), (246, 134, 96), (250, 123, 64)], [(151, 100, 175), (181, 127, 189), (201, 151, 201), (223, 169, 208), (234, 186, 213), (240, 192, 210), (242, 182, 176), (244, 173, 142), (246, 163, 108), (248, 154, 74)], [(161, 128, 194), (191, 159, 210), (214, 180, 219), (229, 195, 224), (240, 208, 229), (245, 216, 226), (245, 208, 191), (245, 200, 156), (245, 192, 120), (245, 184, 85)], [(167, 152, 214), (200, 183, 225), (221, 207, 235), (234, 216, 235), (243, 231, 241), (250, 241, 242), (248, 234, 206), (246, 228, 169), (245, 221, 132), (243, 214, 96)], [(152, 163, 221), (180, 190, 231), (205, 213, 239), (224, 226, 243), (238, 239, 246), (249, 246, 232), (250, 246, 214), (245, 240, 180), (238, 233, 147), (222, 212, 83)], [(130, 148, 212), (159, 177, 225), (191, 205, 236), (211, 219, 241), (228, 234, 244), (244, 244, 225), (242, 241, 192), (232, 233, 170), (218, 223, 134), (190, 203, 74)], [(106, 131, 204), (134, 160, 217), (165, 188, 230), (186, 205, 236), (211, 224, 241), (231, 237, 222), (227, 232, 194), (216, 223, 162), (197, 211, 126), (165, 191, 68)], [(90, 113, 193), (112, 143, 209), (139, 170, 225), (158, 190, 230), (184, 207, 236), (213, 224, 214), (204, 219, 180), (189, 208, 146), (169, 197, 112), (128, 169, 63)], [(72, 91, 179), (96, 124, 199), (115, 147, 214), (128, 167, 222), (157, 188, 229), (188, 208, 202), (177, 201, 160), (163, 191, 128), (142, 177, 97), (88, 140, 52)]], dtype=float) / 255.0

def read_raster_crop(path, lon_min, lat_min, lon_max, lat_max, mask_zero=False):
    if not os.path.exists(resource(path)):
        raise FileNotFoundError(f'文件不存在: {path}')
    with rasterio.open(resource(path)) as src:
        win = from_bounds(lon_min, lat_min, lon_max, lat_max, src.transform)
        win = win.round_offsets().round_lengths()
        data = src.read(1, window=win, masked=True)
        transform = src.window_transform(win)
        arr = data.filled(np.nan).astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        arr[~np.isfinite(arr)] = np.nan
        arr[np.abs(arr) > 1e+20] = np.nan
        arr[arr < 0] = np.nan
        if mask_zero:
            arr[arr <= 0] = np.nan
        meta = {'shape': arr.shape, 'transform': transform, 'crs': src.crs, 'width': arr.shape[1], 'height': arr.shape[0]}
    return (arr, meta)

def same_grid(meta1, meta2):
    return meta1['shape'] == meta2['shape'] and meta1['crs'] == meta2['crs'] and (tuple(meta1['transform']) == tuple(meta2['transform']))

def reproject_to_match(src_arr, src_meta, dst_meta, resampling=Resampling.bilinear):
    src_fill = -9999.0
    dst_fill = -9999.0
    src_tmp = np.where(np.isfinite(src_arr), src_arr, src_fill).astype(np.float32)
    dst_arr = np.full(dst_meta['shape'], dst_fill, dtype=np.float32)
    reproject(source=src_tmp, destination=dst_arr, src_transform=src_meta['transform'], src_crs=src_meta['crs'], src_nodata=src_fill, dst_transform=dst_meta['transform'], dst_crs=dst_meta['crs'], dst_nodata=dst_fill, resampling=resampling)
    dst_arr[dst_arr == dst_fill] = np.nan
    return dst_arr

def classify_fixed(arr, vmin, vmax, nclass, invert=False):
    edges = np.linspace(vmin, vmax, nclass + 1)
    arr_clip = np.clip(arr, vmin, vmax)
    idx = np.digitize(arr_clip, bins=edges[1:-1], right=False)
    idx = np.clip(idx, 0, nclass - 1)
    idx[~np.isfinite(arr)] = -1
    if invert:
        valid = idx >= 0
        idx[valid] = nclass - 1 - idx[valid]
    return (idx, edges)

def make_rgba_from_classes(x_idx, y_idx, palette):
    h, w = x_idx.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    valid = (x_idx >= 0) & (y_idx >= 0)
    rows = (NCLASS - 1 - y_idx[valid]).astype(int)
    cols = x_idx[valid].astype(int)
    rgba[valid, :3] = palette[rows, cols]
    rgba[valid, 3] = 1.0
    return rgba

def format_tick_labels(vmin, vmax, invert=False, nticks=3):
    vals = np.linspace(vmin, vmax, nticks)
    if invert:
        vals = vals[::-1]
    labels = []
    for v in vals:
        if abs(v) >= 100:
            labels.append(f'{v:.0f}')
        elif abs(v) >= 10:
            labels.append(f'{v:.1f}'.rstrip('0').rstrip('.'))
        else:
            labels.append(f'{v:.2f}'.rstrip('0').rstrip('.'))
    return labels

def add_bivariate_legend_inset(parent_ax, palette, x_tick_labels, y_tick_labels, x_label, y_label, title, corner_texts=None, box=(0.78, 0.03, 0.19, 0.29)):
    ax_leg = parent_ax.inset_axes(box)
    ax_leg.imshow(palette, origin='upper', extent=[0, NCLASS, 0, NCLASS], interpolation='nearest')
    ax_leg.set_aspect('equal', adjustable='box')
    ax_leg.set_xlim(0, NCLASS)
    ax_leg.set_ylim(0, NCLASS)
    for k in range(NCLASS + 1):
        ax_leg.axhline(k, color='white', lw=0.35, alpha=0.85)
        ax_leg.axvline(k, color='white', lw=0.35, alpha=0.85)
    for spine in ax_leg.spines.values():
        spine.set_linewidth(0.6)
        spine.set_edgecolor('#000000')
    ax_leg.set_xticks([0, 5, 10])
    ax_leg.set_xticklabels(x_tick_labels, fontsize=9, color='#000000')
    ax_leg.set_yticks([0, 5, 10])
    ax_leg.set_yticklabels(y_tick_labels, fontsize=9, color='#000000')
    ax_leg.tick_params(axis='both', which='major', labelsize=9, width=0.5, length=2.5, colors='#000000', pad=1.5)
    ax_leg.set_xlabel(x_label, fontsize=9, labelpad=5, color='#000000')
    ax_leg.set_ylabel(y_label, fontsize=9, labelpad=5, color='#000000')
    ax_leg.set_title(title, fontsize=9, pad=5, color='#000000')
    if corner_texts is None:
        corner_texts = {'tl': 'High Y\nLow X', 'tr': 'Co-high', 'bl': 'Co-low', 'br': 'High X\nLow Y'}
    ax_leg.text(0.55, 9.6, corner_texts['tl'], ha='left', va='top', fontsize=7, color='white')
    ax_leg.text(9.45, 9.6, corner_texts['tr'], ha='right', va='top', fontsize=7, color='white')
    ax_leg.text(0.55, 0.45, corner_texts['bl'], ha='left', va='bottom', fontsize=7, color='white')
    ax_leg.text(9.45, 0.45, corner_texts['br'], ha='right', va='bottom', fontsize=7, color='white')
    return ax_leg
print('当前 base_dir:', base_dir)
print('文件夹是否存在:', os.path.exists(resource(base_dir)))
for p in [accessibility_tif, road_density_tif, poi_density_tif]:
    print(p, '->', os.path.exists(resource(p)))
    if not os.path.exists(resource(p)):
        raise FileNotFoundError(f'文件不存在: {p}')
access_arr, access_meta = read_raster_crop(accessibility_tif, LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, mask_zero=MASK_ZERO_ACCESS)
road_arr, road_meta = read_raster_crop(road_density_tif, LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, mask_zero=MASK_ZERO_ROAD)
poi_arr, poi_meta = read_raster_crop(poi_density_tif, LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, mask_zero=MASK_ZERO_POI)
if not same_grid(road_meta, access_meta):
    print('[提示] 路网密度栅格正在重采样到综合可达性参考网格...')
    road_arr = reproject_to_match(road_arr, road_meta, access_meta, resampling=Resampling.bilinear)
if not same_grid(poi_meta, access_meta):
    print('[提示] POI密度栅格正在重采样到综合可达性参考网格...')
    poi_arr = reproject_to_match(poi_arr, poi_meta, access_meta, resampling=Resampling.bilinear)
road_idx, _ = classify_fixed(road_arr, ROAD_MIN, ROAD_MAX, NCLASS, invert=False)
poi_idx, _ = classify_fixed(poi_arr, POI_MIN, POI_MAX, NCLASS, invert=False)
access_idx, _ = classify_fixed(access_arr, ACCESS_MIN, ACCESS_MAX, NCLASS, invert=ACCESS_LOWER_IS_BETTER)
rgba_road_access = make_rgba_from_classes(road_idx, access_idx, palette_rgb)
rgba_poi_access = make_rgba_from_classes(poi_idx, access_idx, palette_rgb)
proj = ccrs.PlateCarree()
pc = ccrs.PlateCarree()
fig = plt.figure(figsize=(15.5, 7.5), facecolor='white')
data_maps = {'road': {'rgba': rgba_road_access, 'title': 'Accessibility × Road density', 'panel': 'A', 'pos': [0.025, 0.55, 0.62, 0.38]}, 'poi': {'rgba': rgba_poi_access, 'title': 'Accessibility × POI density', 'panel': 'B', 'pos': [0.025, 0.08, 0.62, 0.38]}}
h, w = access_arr.shape
transform = access_meta['transform']
left = transform.c
top = transform.f
right = left + transform.a * w
bottom = top + transform.e * h
extent = [left, right, bottom, top]
axes_dict = {}
for key in ['road', 'poi']:
    item = data_maps[key]
    ax = fig.add_axes(item['pos'], projection=proj)
    axes_dict[key] = ax
    ax.set_facecolor('white')
    ax.add_feature(cfeature.OCEAN.with_scale('110m'), facecolor=OCEAN_COLOR, edgecolor='none', zorder=0)
    ax.add_feature(cfeature.LAND.with_scale('110m'), facecolor=LAND_COLOR, edgecolor='none', zorder=1)
    ax.imshow(item['rgba'], origin='upper', extent=extent, transform=pc, interpolation='nearest', zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.18, edgecolor=COAST_COLOR, zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale('110m'), linewidth=0.1, edgecolor=BORDER_COLOR, zorder=3)
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=pc)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    if 'geo' in ax.spines:
        ax.spines['geo'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.0, 1.08, f"{item['panel']}.  {item['title']}", transform=ax.transAxes, ha='left', va='top', fontsize=10.5, fontweight='normal', color='#111111')
access_y_labels = format_tick_labels(ACCESS_MIN, ACCESS_MAX, invert=ACCESS_LOWER_IS_BETTER, nticks=3)
road_x_labels = format_tick_labels(ROAD_MIN, ROAD_MAX, invert=False, nticks=3)
poi_x_labels = format_tick_labels(POI_MIN, POI_MAX, invert=False, nticks=3)
if ACCESS_LOWER_IS_BETTER:
    y_label = 'Accessibility'
    tl_text = 'High access.\nLow density'
    tr_text = 'Co-high'
    bl_text = 'Co-low'
    br_text = 'Low access.\nHigh density'
else:
    y_label = 'Accessibility score (%)'
    tl_text = 'High access.\nLow density'
    tr_text = 'Co-high'
    bl_text = 'Co-low'
    br_text = 'High density\nLow access.'
corner_texts = {'tl': tl_text, 'tr': tr_text, 'bl': bl_text, 'br': br_text}
add_bivariate_legend_inset(parent_ax=axes_dict['road'], palette=palette_rgb, x_tick_labels=road_x_labels, y_tick_labels=access_y_labels, x_label='Road density (m km$^{-2}$)', y_label=y_label, title='', corner_texts=corner_texts, box=(0.01, 0.07, 0.28, 0.4))
add_bivariate_legend_inset(parent_ax=axes_dict['poi'], palette=palette_rgb, x_tick_labels=poi_x_labels, y_tick_labels=access_y_labels, x_label='POI density (count km$^{-2}$)', y_label=y_label, title='', corner_texts=corner_texts, box=(0.01, 0.07, 0.28, 0.4))
fig.savefig(output_png, dpi=DPI, bbox_inches='tight', facecolor='white')
fig.savefig(output_pdf, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.show()
print('\n绘图完成。')
print('PNG:', output_png)
print('PDF:', output_pdf)

def _build_manuscript_figure():
    return None, None


if __name__ == "__main__":
    _configure_land_functions(sys.modules[__name__])
    _figure, _bbox = _build_manuscript_figure()
    _save_figure(_figure, _bbox, 'Supplementary_Fig_S13', _figure_options)
