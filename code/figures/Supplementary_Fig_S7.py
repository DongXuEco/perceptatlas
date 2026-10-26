"""Plot Supplementary_Fig_S7 using the packaged numerical tables or supplied local inputs."""
from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_support"))
from figure_cli import prepare_figure as _prepare_figure, save_figure as _save_figure
from figure_cli import read_table as _read_table, model_tables as _model_tables
from figure_cli import configure_land_functions as _configure_land_functions

_figure_options = _prepare_figure('Supplementary_Fig_S7', 'external_inputs_required') if __name__ == "__main__" else None

from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Wedge
input_dir = 'external_inputs/全球路网长度栅格'
output_png = str(output_dir('Supplementary_Fig_S7') / 'Supplementary_Fig_S7.png')
output_pdf = str(output_dir('Supplementary_Fig_S7') / 'Supplementary_Fig_S7.pdf')
raster_files = [('bridleway.tif', 'Bridleway'), ('cycleway.tif', 'Cycleway'), ('footway.tif', 'Footway'), ('living_street.tif', 'Living street'), ('motorway.tif', 'Motorway'), ('path.tif', 'Path'), ('pedestrian.tif', 'Pedestrian'), ('primary.tif', 'Primary'), ('residential.tif', 'Residential'), ('secondary.tif', 'Secondary'), ('service.tif', 'Service'), ('steps.tif', 'Steps'), ('tertiary.tif', 'Tertiary'), ('track.tif', 'Track'), ('trunk.tif', 'Trunk'), ('unknown.tif', 'Unknown')]
GRID_AREA_KM2 = 25.0
CLASS_RANGES = {'Bridleway': (0, 20), 'Cycleway': (40, 80), 'Footway': (32, 120), 'Living street': (20, 40), 'Motorway': (60, 180), 'Path': (40, 240), 'Pedestrian': (1, 4), 'Primary': (60, 240), 'Residential': (60, 640), 'Secondary': (80, 260), 'Service': (0, 280), 'Steps': (3, 8), 'Tertiary': (60, 300), 'Track': (40, 720), 'Trunk': (80, 320), 'Unknown': (20, 600)}
MAP_EXTENT = [-180, 180, -60, 85]
DPI = 600
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 9
mpl.rcParams['axes.titlesize'] = 11
mpl.rcParams['axes.titleweight'] = 'normal'
mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['savefig.bbox'] = 'tight'
UNIFIED_CMAP = plt.get_cmap('plasma').copy()
UNIFIED_CMAP.set_bad(color=(1, 1, 1, 0))

def read_raster(raster_path, vmin, vmax):
    if not os.path.exists(resource(raster_path)):
        raise FileNotFoundError(f'找不到栅格文件：{raster_path}')
    with rasterio.open(resource(raster_path)) as src:
        data = src.read(1).astype(np.float32)
        bounds = src.bounds
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        data[data <= 0] = np.nan
        data = data / GRID_AREA_KM2
        data = np.clip(data, vmin, vmax)
        data = np.ma.masked_invalid(data)
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        return (data, extent)

def sum_positive_raster_values(raster_path, unit_scale=1.0):
    if not os.path.exists(resource(raster_path)):
        raise FileNotFoundError(f'找不到栅格文件：{raster_path}')
    total_value = 0.0
    with rasterio.open(resource(raster_path)) as src:
        nodata = src.nodata
        for _, window in src.block_windows(1):
            arr = src.read(1, window=window).astype(np.float64)
            valid_mask = np.isfinite(arr)
            if nodata is not None:
                valid_mask &= arr != nodata
            valid_mask &= arr > 0
            if np.any(valid_mask):
                total_value += np.sum(arr[valid_mask])
    return total_value / unit_scale

def calculate_road_length_totals(unit_scale=1.0):
    road_lengths = {}
    for fname, title in raster_files:
        raster_path = os.path.join(input_dir, fname)
        road_lengths[title] = sum_positive_raster_values(raster_path=raster_path, unit_scale=unit_scale)
    total_road_length = sum(road_lengths.values())
    if total_road_length <= 0:
        raise ValueError('全球路网总长度为0，请检查输入栅格是否存在有效正值。')
    print('\n========== 全球路网长度统计 ==========')
    print(f'Total road length: {total_road_length:.4e}')
    for title, value in road_lengths.items():
        pct = value / total_road_length * 100
        print(f'{title:<15s}: {value:.4e}  ({pct:.2f}%)')
    return (road_lengths, total_road_length)

def remove_geo_outline(ax):
    if hasattr(ax, 'outline_patch'):
        ax.outline_patch.set_visible(False)
    if 'geo' in ax.spines:
        ax.spines['geo'].set_visible(False)
    ax.set_frame_on(False)

def add_land_background(ax):
    ax.set_facecolor('white')
    ax.add_feature(cfeature.LAND.with_scale('110m'), facecolor='#252525', edgecolor='none', zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), edgecolor='#333333', linewidth=0.25, zorder=3)

def add_panel_colorbar(fig, ax, mappable, vmin, vmax):
    bbox = ax.get_position()
    cb_width = bbox.width * 0.4
    cb_height = bbox.height * 0.07
    cb_left = bbox.x0 + (bbox.width - cb_width) / 2 + bbox.width * 0.07
    cb_bottom = bbox.y0 - bbox.height * 0.1
    cax = fig.add_axes([cb_left, cb_bottom, cb_width, cb_height])
    cbar = fig.colorbar(mappable, cax=cax, orientation='horizontal', ticks=[vmin, vmax])
    cbar.outline.set_linewidth(0.55)
    cbar.outline.set_edgecolor('#252525')
    cbar.ax.tick_params(labelsize=10, width=0.55, length=3.5, pad=1, colors='#252525')
    cbar.ax.set_xticklabels([f'{vmin:g}', f'{vmax:g}'])
    cbar.set_label('Length density (m km$^{-2}$)', fontsize=10, labelpad=6, color='#252525')
    cbar.ax.xaxis.set_label_position('top')

def format_sci_value(value, digits=1):
    if value == 0:
        return '0'
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10 ** exponent
    superscript_map = str.maketrans('0123456789-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻')
    exponent_str = str(exponent).translate(superscript_map)
    return f'{mantissa:.{digits}f}×10{exponent_str}'

def add_donut_chart(fig, ax, length_value, total_value):
    if total_value <= 0:
        return
    pct = length_value / total_value
    pct = max(0, min(1, pct))
    pct_display = pct * 100
    bbox = ax.get_position()
    donut_w = bbox.width * 0.49
    donut_h = bbox.height * 0.49
    donut_left = bbox.x0 + bbox.width * -0.088
    donut_bottom = bbox.y0 + bbox.height * 0.005
    iax = fig.add_axes([donut_left, donut_bottom, donut_w, donut_h])
    iax.set_zorder(20)
    iax.set_facecolor('none')
    donut_color = '#9613A1'
    remainder_color = '#BDC3D2'
    bg_r = 1.0
    bg_width = 0.1
    bg_ring = Wedge(center=(0, 0), r=bg_r, theta1=0, theta2=360, width=bg_width, facecolor=remainder_color, edgecolor='none')
    iax.add_patch(bg_ring)
    fg_width = 0.24
    ring_center_radius = bg_r - bg_width / 2
    fg_r = ring_center_radius + fg_width / 2
    theta_start = 90
    theta_end = 90 - 360 * pct
    fg_ring = Wedge(center=(0, 0), r=fg_r, theta1=theta_end, theta2=theta_start, width=fg_width, facecolor=donut_color, edgecolor='none')
    iax.add_patch(fg_ring)
    if 0 < pct_display < 0.1:
        pct_text = '<0.1%'
    else:
        pct_text = f'{pct_display:.1f}%'
    iax.text(0, 0.2, pct_text, ha='center', va='center', fontsize=9, color='#4d4d4d', fontweight='normal', family='Arial')
    iax.text(0, -0.32, format_sci_value(length_value, digits=1), ha='center', va='center', fontsize=9, color=donut_color, fontweight='normal', family='Arial')
    iax.set_aspect('equal')
    iax.set_xlim(-1.15, 1.15)
    iax.set_ylim(-1.15, 1.15)
    iax.set_xticks([])
    iax.set_yticks([])
    for spine in iax.spines.values():
        spine.set_visible(False)

def plot_global_road_maps():
    missing_titles = [title for _, title in raster_files if title not in CLASS_RANGES]
    if len(missing_titles) > 0:
        raise ValueError(f'CLASS_RANGES 中缺少这些类别的显示范围：{missing_titles}')
    road_lengths, total_road_length = calculate_road_length_totals(unit_scale=1.0)
    fig = plt.figure(figsize=(14.5, 9.2), facecolor='white')
    gs = fig.add_gridspec(nrows=4, ncols=4, left=0.035, right=0.99, top=0.965, bottom=0.04, wspace=0.03, hspace=0.1)
    panel_labels = ['A.', 'B.', 'C.', 'D.', 'E.', 'F.', 'G.', 'H.', 'I.', 'J.', 'K.', 'L.', 'M.', 'N.', 'O.', 'P.']
    for i, (fname, title) in enumerate(raster_files):
        row = i // 4
        col = i % 4
        ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
        remove_geo_outline(ax)
        ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
        add_land_background(ax)
        class_vmin, class_vmax = CLASS_RANGES[title]
        class_norm = mpl.colors.Normalize(vmin=class_vmin, vmax=class_vmax)
        raster_path = os.path.join(input_dir, fname)
        data, extent = read_raster(raster_path=raster_path, vmin=class_vmin, vmax=class_vmax)
        im = ax.imshow(data, origin='upper', extent=extent, transform=ccrs.PlateCarree(), cmap=UNIFIED_CMAP, norm=class_norm, interpolation='nearest', zorder=2, alpha=1)
        ax.set_title(f'{panel_labels[i]}  {title}', loc='left', pad=6, fontsize=12, color='#111111', fontweight='normal')
        ax.set_xticks([])
        ax.set_yticks([])
        add_panel_colorbar(fig=fig, ax=ax, mappable=im, vmin=class_vmin, vmax=class_vmax)
        add_donut_chart(fig=fig, ax=ax, length_value=road_lengths[title], total_value=total_road_length)
    plt.savefig(output_png, dpi=DPI, facecolor='white')
    plt.savefig(output_pdf, dpi=DPI, facecolor='white')
    plt.show()
    print('绘图完成！')
    print('PNG输出：', output_png)
    print('PDF输出：', output_pdf)

def _script_entry():
    plot_global_road_maps()

def _build_manuscript_figure():
    _script_entry()
    return None, None


if __name__ == "__main__":
    _configure_land_functions(sys.modules[__name__])
    _figure, _bbox = _build_manuscript_figure()
    _save_figure(_figure, _bbox, 'Supplementary_Fig_S7', _figure_options)
