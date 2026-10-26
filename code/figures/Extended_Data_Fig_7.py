"""Plot Extended_Data_Fig_7 using the packaged numerical tables or supplied local inputs."""
from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_support"))
from figure_cli import prepare_figure as _prepare_figure, save_figure as _save_figure
from figure_cli import read_table as _read_table, model_tables as _model_tables
from figure_cli import configure_land_functions as _configure_land_functions

_figure_options = _prepare_figure('Extended_Data_Fig_7', 'external_inputs_required') if __name__ == "__main__" else None

from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
input_dir = os.path.dirname(os.path.abspath(__file__))
output_png = str(output_dir('Extended_Data_Fig_7') / 'Extended_Data_Fig_7.png')
output_pdf = str(output_dir('Extended_Data_Fig_7') / 'Extended_Data_Fig_7.pdf')
raster_items = [{'file': 'streetview_density_10km_wgs1984_per_km2.tif', 'title': 'Street-view density', 'unit_label': 'Street-view density (count km$^{-2}$)', 'scale': 'log', 'vmin': 0.1, 'vmax': 32, 'ticks': [0.1, 1, 10, 32], 'cmap': 'plasma', 'north': 62.34829218, 'south': 24.41543978}, {'file': 'poi.tif', 'title': 'POI density', 'unit_label': 'POI density (count km$^{-2}$)', 'vmin': 0, 'vmax': 16, 'ticks': [0, 4, 8, 12, 16], 'cmap': 'plasma', 'north': 23.484072, 'south': 59.450742}, {'file': 'road.tif', 'title': 'Road length density', 'unit_label': 'Road length density (m km$^{-2}$)', 'vmin': 0, 'vmax': 8000, 'ticks': [0, 2000, 4000, 6000, 8000], 'cmap': 'plasma', 'north': 8945.586, 'south': 6783.164}, {'file': 'landsat8.tif', 'title': 'Landsat image density', 'unit_label': 'Landsat-8 image count', 'vmin': 60, 'vmax': 360, 'ticks': [60, 135, 210, 285, 360], 'cmap': 'plasma', 'north': 231.634011, 'south': 205.011636}]
MAP_EXTENT = [-180, 180, -60, 85]
DPI = 600
ZERO_TO_NAN = True
REMOVE_NEGATIVE = True
CLIP_TO_RANGE = True
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

def get_raster_path(file_value):
    if os.path.isabs(file_value):
        return file_value
    return os.path.join(input_dir, file_value)

def read_raster(raster_path, vmin, vmax):
    if not os.path.exists(resource(raster_path)):
        raise FileNotFoundError(f'找不到栅格文件：{raster_path}')
    with rasterio.open(resource(raster_path)) as src:
        data = src.read(1).astype(np.float32)
        bounds = src.bounds
        nodata = src.nodata
        print('\n读取栅格：', raster_path)
        print('CRS:', src.crs)
        print('Size:', src.width, '×', src.height)
        print('Resolution:', src.res)
        print('NoData:', nodata)
        print('Raw min/max:', np.nanmin(data), np.nanmax(data))
        if nodata is not None:
            data[data == nodata] = np.nan
        data[~np.isfinite(data)] = np.nan
        if REMOVE_NEGATIVE:
            data[data < 0] = np.nan
        if ZERO_TO_NAN:
            data[data <= 0] = np.nan
        if CLIP_TO_RANGE:
            data = np.clip(data, vmin, vmax)
        print('Plot min/max:', np.nanmin(data), np.nanmax(data))
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

def build_cmap(cmap_name):
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color=(1, 1, 1, 0))
    return cmap

def add_north_south_annotation(ax, north, south, xpos=0.25, ypos=0.3):
    if south == 0:
        ratio_text = 'North/South: NA'
    else:
        ratio = north / south
        ratio_text = f'North/South: {ratio:.2f}×'
    ax.text(xpos, ypos, 'North: 000000\nSouth: 000000\nNorth/South: 00000', transform=ax.transAxes, ha='left', va='top', fontsize=10, color=(1, 1, 1, 0), linespacing=1.5, zorder=9, bbox=dict(boxstyle='square,pad=0.28', facecolor='white', edgecolor='#222222', linewidth=0, alpha=1))
    ax.text(xpos, ypos, f'North: {north:.2f}', transform=ax.transAxes, ha='left', va='top', fontsize=10, color='#222222', zorder=10)
    ax.text(xpos, ypos - 0.1, f'South: {south:.2f}', transform=ax.transAxes, ha='left', va='top', fontsize=10, color='#222222', zorder=10)
    ax.text(xpos, ypos - 0.2, ratio_text, transform=ax.transAxes, ha='left', va='top', fontsize=10, color='#9613A1', fontweight='normal', zorder=10)

def plot_global_four_density_maps():
    fig = plt.figure(figsize=(12.2 / 2.05, 7.5), facecolor='white')
    gs = fig.add_gridspec(nrows=2, ncols=1, left=0.045, right=0.985, top=0.925, bottom=0.085, wspace=0.05, hspace=0.23)
    panel_labels = ['a', 'b']
    for i, item in enumerate([raster_items[0], raster_items[3]]):
        row = i
        col = 0
        ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
        remove_geo_outline(ax)
        ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
        add_land_background(ax)
        raster_path = get_raster_path(item['file'])
        vmin = item['vmin']
        vmax = item['vmax']
        ticks = item['ticks']
        cmap_name = item['cmap']
        cmap = build_cmap(cmap_name)
        if item.get('scale') == 'log':
            norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        data, extent = read_raster(raster_path=raster_path, vmin=vmin, vmax=vmax)
        im = ax.imshow(data, origin='upper', extent=extent, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, interpolation='nearest', zorder=2, alpha=1)
        title = ax.set_title(f"{panel_labels[i]}  {item['title']}", loc='left', pad=7, fontsize=11, color='#111111', fontweight='normal')
        prefix_width = fig.canvas.get_renderer().get_text_width_height_descent(f'{panel_labels[i]}  ', title.get_fontproperties(), ismath=False)[0]
        ax.text(*title.get_position(), panel_labels[i], transform=title.get_transform(), ha=title.get_ha(), va=title.get_va(), fontsize=11, fontweight='bold', color='#111111', clip_on=False)
        title.set_text(item['title'])
        title.set_transform(title.get_transform() + mpl.transforms.ScaledTranslation(prefix_width / fig.dpi, 0, fig.dpi_scale_trans))
        ax.set_xticks([])
        ax.set_yticks([])
        add_north_south_annotation(ax=ax, north=item['north'], south=item['south'], xpos=0.04, ypos=0.36)
        bbox = ax.get_position()
        cbar_width = bbox.width * 0.68
        cbar_height = 0.02
        cbar_left = bbox.x0 + bbox.width * 0.16
        cbar_bottom = bbox.y0 - 0.035
        cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=ticks)
        if item.get('scale') == 'log':
            cbar.ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda value, _: f'{value:g}'))
            cbar.minorticks_off()
        cbar.outline.set_linewidth(0.5)
        cbar.outline.set_edgecolor('#555555')
        cbar.ax.tick_params(labelsize=10.5, width=0.5, length=3.5, pad=2, colors='#333333')
        cbar.set_label(item['unit_label'], fontsize=10.5, labelpad=3, color='#333333')
    plt.savefig(output_png, dpi=DPI, facecolor='white')
    plt.savefig(output_pdf, dpi=DPI, facecolor='white')
    plt.show()
    print('\n绘图完成！')
    print('PNG输出：', output_png)
    print('PDF输出：', output_pdf)

def _script_entry():
    plot_global_four_density_maps()

def _build_manuscript_figure():
    _script_entry()
    return None, None


if __name__ == "__main__":
    _configure_land_functions(sys.modules[__name__])
    _figure, _bbox = _build_manuscript_figure()
    _save_figure(_figure, _bbox, 'Extended_Data_Fig_7', _figure_options)
