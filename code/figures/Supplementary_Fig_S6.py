"""Plot Supplementary_Fig_S6 using the packaged numerical tables or supplied local inputs."""
from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_support"))
from figure_cli import prepare_figure as _prepare_figure, save_figure as _save_figure
from figure_cli import read_table as _read_table, model_tables as _model_tables
from figure_cli import configure_land_functions as _configure_land_functions

_figure_options = _prepare_figure('Supplementary_Fig_S6', 'external_inputs_required') if __name__ == "__main__" else None

from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cmcrameri import cm as cmc
from matplotlib.patches import Wedge
input_dir = 'external_inputs/全球POI密度栅格'
output_png = str(output_dir('Supplementary_Fig_S6') / 'Supplementary_Fig_S6.png')
output_pdf = str(output_dir('Supplementary_Fig_S6') / 'Supplementary_Fig_S6.pdf')
raster_files = [('lifeImg.tif', 'Living'), ('productionImg.tif', 'Production'), ('transportImg.tif', 'Public transit'), ('educationImg.tif', 'Education'), ('healthImg.tif', 'Healthcare'), ('entertainmentImg.tif', 'Entertainment')]
POI_COUNTS = {'Living': 63151666, 'Production': 16401010, 'Public transit': 4189034, 'Education': 5465191, 'Healthcare': 4181500, 'Entertainment': 5554500}
VMIN = 0
VMAX = 8
MAP_EXTENT = [-180, 180, -60, 85]
DPI = 600
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.titleweight'] = 'normal'
mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Arial'
mpl.rcParams['mathtext.it'] = 'Arial:italic'
mpl.rcParams['mathtext.bf'] = 'Arial:normal'
UNIFIED_CMAP = plt.get_cmap('plasma').copy()
UNIFIED_CMAP.set_bad(color=(1, 1, 1, 0))
NORM = mpl.colors.Normalize(vmin=VMIN, vmax=VMAX)

def read_raster(raster_path):
    if not os.path.exists(resource(raster_path)):
        raise FileNotFoundError(f'找不到栅格文件：{raster_path}')
    with rasterio.open(resource(raster_path)) as src:
        data = src.read(1).astype(np.float32)
        bounds = src.bounds
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        data = np.clip(data, VMIN, VMAX)
        data[data <= 0] = np.nan
        data = np.ma.masked_invalid(data)
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        return (data, extent)

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

def format_sci_count(value, digits=2):
    if value == 0:
        return '0'
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10 ** exponent
    superscript_map = str.maketrans('0123456789-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻')
    exponent_str = str(exponent).translate(superscript_map)
    return f'{mantissa:.{digits}f}×10{exponent_str}'

def add_donut_chart(fig, ax, count_value, total_value):
    if total_value <= 0:
        return
    pct = count_value / total_value
    pct = max(0, min(1, pct))
    pct_display = pct * 100
    bbox = ax.get_position()
    donut_w = bbox.width * 0.4
    donut_h = bbox.height * 0.4
    donut_left = bbox.x1 - donut_w - 0.31
    donut_bottom = bbox.y1 - donut_h - 0.11
    iax = fig.add_axes([donut_left, donut_bottom, donut_w, donut_h])
    iax.set_zorder(10)
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
    iax.text(0, 0.18, f'{pct_display:.1f}%', ha='center', va='center', fontsize=10.5, color='#4d4d4d', fontweight='normal')
    iax.text(0, -0.3, format_sci_count(count_value, digits=1), ha='center', va='center', fontsize=10.5, color='#9613A1', fontweight='normal')
    iax.set_aspect('equal')
    iax.set_xlim(-1.15, 1.15)
    iax.set_ylim(-1.15, 1.15)
    iax.set_xticks([])
    iax.set_yticks([])
    for spine in iax.spines.values():
        spine.set_visible(False)

def plot_global_poi_maps():
    missing_titles = [title for _, title in raster_files if title not in POI_COUNTS]
    if len(missing_titles) > 0:
        raise ValueError(f'POI_COUNTS 中缺少这些类别的数量：{missing_titles}')
    total_count = sum(POI_COUNTS.values())
    if total_count <= 0:
        raise ValueError('POI_COUNTS 的总数不能为 0，请填写真实的全局 POI 数量。')
    fig = plt.figure(figsize=(11.5, 9.5), facecolor='white')
    gs = fig.add_gridspec(nrows=3, ncols=2, left=0.045, right=0.985, top=0.915, bottom=0.115, wspace=0.045, hspace=0.135)
    axes = []
    panel_labels = ['A.', 'B.', 'C.', 'D.', 'E', 'F.']
    for i, (fname, title) in enumerate(raster_files):
        row = i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
        axes.append(ax)
        remove_geo_outline(ax)
        ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
        add_land_background(ax)
        raster_path = os.path.join(input_dir, fname)
        data, extent = read_raster(raster_path)
        ax.imshow(data, origin='upper', extent=extent, transform=ccrs.PlateCarree(), cmap=UNIFIED_CMAP, norm=NORM, interpolation='nearest', zorder=2, alpha=1)
        ax.set_title(f'{panel_labels[i]}  {title}', loc='left', pad=5, fontsize=12, color='#111111', fontweight='normal')
        ax.set_xticks([])
        ax.set_yticks([])
        add_donut_chart(fig=fig, ax=ax, count_value=POI_COUNTS[title], total_value=total_count)
    sm = mpl.cm.ScalarMappable(norm=NORM, cmap=UNIFIED_CMAP)
    sm.set_array([])
    cax = fig.add_axes([0.23, 0.085, 0.54, 0.023])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal', ticks=[0, 2, 4, 6, 8])
    cbar.outline.set_linewidth(0.6)
    cbar.outline.set_edgecolor('#252525')
    cbar.ax.tick_params(labelsize=12, width=0.6, length=3.5, pad=2, colors='#252525')
    cbar.set_label('POI density (count per 5-km grid cell)', fontsize=12, labelpad=4, color='#252525')
    plt.savefig(output_png, dpi=DPI, facecolor='white')
    plt.savefig(output_pdf, dpi=DPI, facecolor='white')
    plt.show()
    print('绘图完成！')
    print('PNG输出：', output_png)
    print('PDF输出：', output_pdf)

def _script_entry():
    plot_global_poi_maps()

def _build_manuscript_figure():
    _script_entry()
    return None, None


if __name__ == "__main__":
    _configure_land_functions(sys.modules[__name__])
    _figure, _bbox = _build_manuscript_figure()
    _save_figure(_figure, _bbox, 'Supplementary_Fig_S6', _figure_options)
