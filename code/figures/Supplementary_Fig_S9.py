"""Plot Supplementary_Fig_S9 using the packaged numerical tables or supplied local inputs."""
from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_support"))
from figure_cli import prepare_figure as _prepare_figure, save_figure as _save_figure
from figure_cli import read_table as _read_table, model_tables as _model_tables
from figure_cli import configure_land_functions as _configure_land_functions

_figure_options = _prepare_figure('Supplementary_Fig_S9', 'external_inputs_required') if __name__ == "__main__" else None

from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from rasterio.enums import Resampling
from matplotlib.patches import FancyBboxPatch
input_dir = 'external_inputs/机动车'
output_png = str(output_dir('Supplementary_Fig_S9') / 'Supplementary_Fig_S9.png')
output_pdf = str(output_dir('Supplementary_Fig_S9') / 'Supplementary_Fig_S9.pdf')
raster_files = [{'file': 'living.tif', 'title': 'Living services', 'north': 0.8355955121965022, 'south': 1.939993249140368}, {'file': 'production.tif', 'title': 'Production services', 'north': 1.9490652900899705, 'south': 2.2995038085542627}, {'file': 'transport.tif', 'title': 'Public transit', 'north': 7.017140908280096, 'south': 5.864443750960431}, {'file': 'education.tif', 'title': 'Education', 'north': 1.7280134222884374, 'south': 2.4146081088739435}, {'file': 'health.tif', 'title': 'Healthcare', 'north': 3.776423742331042, 'south': 3.297378835687257}, {'file': 'entertainment.tif', 'title': 'Entertainment', 'north': 1.6849160608801166, 'south': 2.799566314773902}]
VMIN = 0
VMAX = 30
MASK_ZERO_AS_NODATA = False
MAP_EXTENT = [-180, 180, -60, 85]
DPI = 600
MAX_DISPLAY_WIDTH = 15000
MAX_DISPLAY_HEIGHT = 7500
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
UNIFIED_CMAP = plt.get_cmap('plasma_r').copy()
UNIFIED_CMAP.set_bad(color=(1, 1, 1, 0))
NORM = mpl.colors.Normalize(vmin=VMIN, vmax=VMAX)

def read_raster(raster_path):
    if not os.path.exists(resource(raster_path)):
        raise FileNotFoundError(f'找不到栅格文件：{raster_path}')
    with rasterio.open(resource(raster_path)) as src:
        src_height = src.height
        src_width = src.width
        scale_factor = max(src_width / MAX_DISPLAY_WIDTH, src_height / MAX_DISPLAY_HEIGHT, 1)
        out_width = int(src_width / scale_factor)
        out_height = int(src_height / scale_factor)
        print(f'读取栅格: {os.path.basename(raster_path)} | 原始尺寸: {src_width}×{src_height} | 绘图尺寸: {out_width}×{out_height}')
        data = src.read(1, out_shape=(out_height, out_width), masked=True, resampling=Resampling.bilinear)
        bounds = src.bounds
        nodata = src.nodata
        if np.ma.isMaskedArray(data):
            data = data.astype(np.float32).filled(np.nan)
        else:
            data = data.astype(np.float32)
        if nodata is not None:
            data[data == nodata] = np.nan
        data[~np.isfinite(data)] = np.nan
        data[np.abs(data) > 1e+20] = np.nan
        data[data < 0] = np.nan
        if MASK_ZERO_AS_NODATA:
            data[data <= 0] = np.nan
        data = np.clip(data, VMIN, VMAX)
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

def add_north_south_annotation(ax, north, south, xpos=-0.09, ypos=0.16, box_width=0.265, box_height=0.17):
    if south == 0:
        ratio_text = 'North/South: NA'
    else:
        ratio = north / south
        ratio_text = f'North/South: {ratio:.2f}×'
    box = FancyBboxPatch((xpos, ypos), box_width, box_height, boxstyle='square,pad=0.012', transform=ax.transAxes, facecolor='white', edgecolor='#222222', linewidth=0, alpha=1, zorder=9)
    ax.add_patch(box)
    text_x = xpos - 0.0
    text_y = ypos + box_height - 0.02
    ax.text(text_x, text_y, f'North: {north:.2f} min', transform=ax.transAxes, ha='left', va='top', fontsize=10.5, color='#222222', zorder=10)
    ax.text(text_x, text_y - 0.1, f'South: {south:.2f} min', transform=ax.transAxes, ha='left', va='top', fontsize=10.5, color='#222222', zorder=10)
    ax.text(text_x, text_y - 0.2, ratio_text, transform=ax.transAxes, ha='left', va='top', fontsize=10.5, color='#9613A1', fontweight='normal', zorder=10)

def plot_global_accessibility_maps():
    fig = plt.figure(figsize=(11.5, 9.5), facecolor='white')
    gs = fig.add_gridspec(nrows=3, ncols=2, left=0.045, right=0.985, top=0.915, bottom=0.115, wspace=0.045, hspace=0.135)
    panel_labels = ['A.', 'B.', 'C.', 'D.', 'E.', 'F.']
    for i, item in enumerate(raster_files):
        row = i // 2
        col = i % 2
        fname = item['file']
        title = item['title']
        north = item['north']
        south = item['south']
        ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
        remove_geo_outline(ax)
        ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
        add_land_background(ax)
        raster_path = os.path.join(input_dir, fname)
        data, extent = read_raster(raster_path)
        ax.imshow(data, origin='upper', extent=extent, transform=ccrs.PlateCarree(), cmap=UNIFIED_CMAP, norm=NORM, interpolation='nearest', zorder=2, alpha=1)
        ax.set_title(f'{panel_labels[i]}  {title}', loc='left', pad=7, fontsize=12, color='#111111', fontweight='normal')
        ax.set_xticks([])
        ax.set_yticks([])
        add_north_south_annotation(ax=ax, north=north, south=south, xpos=0.025, ypos=0.135, box_width=0.23, box_height=0.265)
    sm = mpl.cm.ScalarMappable(norm=NORM, cmap=UNIFIED_CMAP)
    sm.set_array([])
    cax = fig.add_axes([0.23, 0.085, 0.54, 0.023])
    tick_values = [VMIN, VMAX * 0.25, VMAX * 0.5, VMAX * 0.75, VMAX]
    tick_values = [round(x, 1) for x in tick_values]
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal', ticks=tick_values)
    cbar.outline.set_linewidth(0.6)
    cbar.outline.set_edgecolor('#252525')
    cbar.ax.tick_params(labelsize=12, width=0.6, length=4, pad=2, colors='#252525')
    cbar.set_label('Motorized travel time to nearest service facility (min)', fontsize=12, labelpad=4, color='#252525')
    plt.savefig(output_png, dpi=DPI, facecolor='white')
    plt.savefig(output_pdf, dpi=DPI, facecolor='white')
    plt.show()
    print('绘图完成！')
    print('PNG输出：', output_png)
    print('PDF输出：', output_pdf)

def _script_entry():
    plot_global_accessibility_maps()

def _build_manuscript_figure():
    _script_entry()
    return None, None


if __name__ == "__main__":
    _configure_land_functions(sys.modules[__name__])
    _figure, _bbox = _build_manuscript_figure()
    _save_figure(_figure, _bbox, 'Supplementary_Fig_S9', _figure_options)
