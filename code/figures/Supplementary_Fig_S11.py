"""Plot Supplementary_Fig_S11 using the packaged numerical tables or supplied local inputs."""
from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_support"))
from figure_cli import prepare_figure as _prepare_figure, save_figure as _save_figure
from figure_cli import read_table as _read_table, model_tables as _model_tables
from figure_cli import configure_land_functions as _configure_land_functions

_figure_options = _prepare_figure('Supplementary_Fig_S11', 'external_inputs_required') if __name__ == "__main__" else None

from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
import os
import csv
from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from scipy.stats import gaussian_kde
MAX_PLOT_WIDTH = 20000
MAX_PLOT_HEIGHT = 10000
data_dir = Path(__file__).resolve().parent / '机动车'
output_dir = str(data_dir)
os.makedirs(output_dir, exist_ok=True)
output_png = str(output_dir('Supplementary_Fig_S11') / 'Supplementary_Fig_S11.png')
output_pdf = str(output_dir('Supplementary_Fig_S11') / 'Supplementary_Fig_S11.pdf')
MAP_EXTENT = [-180, 180, -60, 85]
DPI = 600
panel_configs = [{'file': str(data_dir / 'newnonRoadSpeedKmh.tif'), 'title': 'Non-road speed', 'cmap': 'plasma', 'vmin': 0, 'vmax': 5, 'cbar_label': 'Speed (km/h)', 'mask_le_zero': True}, {'file': str(data_dir / 'nroadSpeedKmh.tif'), 'title': 'Road speed', 'density_max': 80, 'cmap': 'plasma', 'vmin': 0, 'vmax': 60, 'cbar_label': 'Speed (km/h)', 'mask_le_zero': True}, {'file': str(data_dir / 'finalSpeedKmh.tif'), 'title': 'Final speed', 'density_max': 50, 'cmap': 'plasma', 'vmin': 0, 'vmax': 60, 'cbar_label': 'Speed (km/h)', 'mask_le_zero': True}, {'file': str(data_dir / 'finalCostMinPerM.tif'), 'title': 'Friction surface', 'density_max': 0.08, 'cmap': 'RdYlBu_r', 'vmin': 0, 'vmax': 0.05, 'cbar_label': 'Cost (min/meter)', 'mask_le_zero': True}]
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 9
mpl.rcParams['axes.titlesize'] = 11
mpl.rcParams['axes.titleweight'] = 'normal'
mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['savefig.bbox'] = 'tight'

def get_cmap_with_nan(cmap_name):
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color=(1, 1, 1, 0))
    return cmap

def read_raster(raster_path, vmin, vmax, mask_le_zero=True):
    if not os.path.exists(resource(raster_path)):
        raise FileNotFoundError(f'找不到栅格文件：{raster_path}')
    with rasterio.open(resource(raster_path)) as src:
        bounds = src.bounds
        nodata = src.nodata
        src_height = src.height
        src_width = src.width
        scale_factor = max(src_width / MAX_PLOT_WIDTH, src_height / MAX_PLOT_HEIGHT, 1)
        out_width = int(src_width / scale_factor)
        out_height = int(src_height / scale_factor)
        print(f'读取用于绘图的降采样栅格: {os.path.basename(raster_path)} | 原始尺寸: {src_width} × {src_height} | 绘图尺寸: {out_width} × {out_height}')
        data = src.read(1, out_shape=(out_height, out_width), resampling=Resampling.bilinear).astype(np.float32)
        if nodata is not None:
            data[data == nodata] = np.nan
        data[~np.isfinite(data)] = np.nan
        if mask_le_zero:
            data[data <= 0] = np.nan
        data = np.clip(data, vmin, vmax)
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

def add_panel_colorbar(fig, ax, mappable, vmin, vmax, label):
    bbox = ax.get_position()
    cb_width = bbox.width * 0.42
    cb_height = bbox.height * 0.055
    cb_left = bbox.x0 + (bbox.width - cb_width) / 2.0 + bbox.width * 0.06
    cb_bottom = bbox.y0 - bbox.height * 0.025
    cax = fig.add_axes([cb_left, cb_bottom, cb_width, cb_height])
    cbar = fig.colorbar(mappable, cax=cax, orientation='horizontal', ticks=[vmin, vmax])
    cbar.outline.set_linewidth(0.55)
    cbar.outline.set_edgecolor('#252525')
    cbar.ax.tick_params(labelsize=10, width=0.55, length=3.5, pad=1, colors='#252525')
    cbar.ax.set_xticklabels([f'{vmin:g}', f'{vmax:g}'])
    cbar.set_label(label, fontsize=11, labelpad=5, color='#252525')
    cbar.ax.xaxis.set_label_position('top')

def sample_native_values(raster_path, stride=32):
    samples = []
    with rasterio.open(resource(raster_path)) as src:
        cols = np.arange(stride // 2, src.width, stride)
        xs = src.transform.c + (cols + 0.5) * src.transform.a
        keep_x = (xs >= MAP_EXTENT[0]) & (xs <= MAP_EXTENT[1])
        for start in range(0, src.height, 128):
            stop = min(start + 128, src.height)
            rows = np.arange(start + stride // 2, stop, stride)
            ys = src.transform.f + (rows + 0.5) * src.transform.e
            rows = rows[(ys >= MAP_EXTENT[2]) & (ys <= MAP_EXTENT[3])]
            if not len(rows):
                continue
            block = src.read(1, window=Window(0, start, src.width, stop - start), masked=True)
            values = block[np.ix_(rows - start, cols[keep_x])].compressed()
            samples.append(values[np.isfinite(values) & (values > 0)])
    values = np.concatenate(samples).astype(np.float64)
    if len(values) < 3 or np.ptp(values) == 0:
        raise ValueError(f'Insufficient variable data for density: {raster_path}')
    return values

def add_map_density(fig, map_ax, values, cmap, norm, display_max=None):
    box = map_ax.get_position()
    ax = fig.add_axes([box.x0 + 0.04 * box.width, box.y0 + 0.02 * box.height, 0.23 * 0.8 * box.width, 0.18 * 1.3 * box.height])
    kde_values = values
    if len(values) > 40000:
        kde_values = np.random.default_rng(42).choice(values, 40000, replace=False)
    kde = gaussian_kde(kde_values, bw_method='scott')
    maximum = float(values.max())
    step = 10 ** np.floor(np.log10(maximum))
    if maximum / step < 2:
        step /= 5
    upper = float(np.ceil(maximum / step) * step)
    if display_max is not None:
        upper = display_max
    grid = np.linspace(float(values.min()), maximum, 400)
    density = kde(grid)
    for i in range(len(grid) - 1):
        ax.fill_between(grid[i:i + 2], density[i:i + 2], color=cmap(norm(grid[i:i + 2].mean())), linewidth=0)
    ax.plot(grid, density, color='#252525', linewidth=0.8)
    median = float(np.median(values))
    height = float(kde([median])[0])
    ax.vlines(median, 0, height, color='#252525', linewidth=0.65, linestyle=':')
    ax.set(xlim=(0, upper), ylim=(0, density.max() * 1.12))
    ax.set_xticks([0, upper], ['0', f'{upper:g}'])
    ax.set_yticks([])
    for name in ('left', 'right', 'top'):
        ax.spines[name].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.55)
    ax.tick_params(axis='x', labelsize=9.5, pad=1, length=2, width=0.5)
    ax.text(0, 1.1, 'Density', transform=ax.transAxes, fontsize=10.5, va='bottom')
    ax.text(1.03, 1.58, 'Median', transform=ax.transAxes, ha='right', va='bottom', fontsize=9.5)
    ax.annotate('', xy=(median, height), xytext=(0.82, 1.51), textcoords='axes fraction', arrowprops={'arrowstyle': '-', 'color': '#252525', 'linewidth': 0.5, 'connectionstyle': 'angle,angleA=-90,angleB=0,rad=0'}, annotation_clip=False)
    return {'sampling': 'native grid every 32 cells; equal cell weight; visible extent; finite > 0', 'n_sample': len(values), 'minimum': float(values.min()), 'maximum': maximum, 'median': median, 'kde_n': len(kde_values), 'inset_max': upper}

def plot_global_maps():
    if len(panel_configs) != 4:
        raise ValueError('panel_configs 必须正好包含 4 个变量配置，用于 2×2 绘图。')
    fig = plt.figure(figsize=(13.8, 7), facecolor='white')
    gs = fig.add_gridspec(nrows=2, ncols=2, left=0.035, right=0.99, top=0.965, bottom=0.06, wspace=0.06, hspace=0.14)
    summaries = []
    for i, cfg in enumerate(panel_configs):
        row = i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
        remove_geo_outline(ax)
        ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
        add_land_background(ax)
        raster_path = cfg['file']
        title = cfg['title']
        cmap = get_cmap_with_nan(cfg['cmap'])
        vmin = cfg['vmin']
        vmax = cfg['vmax']
        cbar_label = cfg['cbar_label']
        mask_le_zero = cfg.get('mask_le_zero', True)
        data, extent = read_raster(raster_path=raster_path, vmin=vmin, vmax=vmax, mask_le_zero=mask_le_zero)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        im = ax.imshow(data, origin='upper', extent=extent, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, interpolation='nearest', zorder=2, alpha=1)
        title_artist = ax.set_title(title, loc='left', pad=4, fontsize=12, color='#111111', fontweight='normal')
        title_artist.set_x(0.032)
        ax.text(0, 1, chr(ord('a') + i), transform=title_artist.get_transform(), ha='left', va=title_artist.get_va(), fontsize=12, fontweight='bold', color='#111111')
        ax.set_xticks([])
        ax.set_yticks([])
        add_panel_colorbar(fig=fig, ax=ax, mappable=im, vmin=vmin, vmax=vmax, label=cbar_label)
        values = sample_native_values(raster_path)
        summary = add_map_density(fig, ax, values, cmap, norm, cfg.get('density_max'))
        summary = {'panel': title, 'source': raster_path, **summary}
        summaries.append(summary)
        print('Density:', summary, flush=True)
    with open_resource(os.path.join(output_dir, 'density_sampling_summary.csv'), 'w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    fig.canvas.draw()
    plt.savefig(output_png, dpi=DPI, facecolor='white')
    plt.savefig(output_pdf, dpi=DPI, facecolor='white')
    plt.show()
    print('绘图完成！')
    print('PNG输出：', output_png)
    print('PDF输出：', output_pdf)

def _script_entry():
    plot_global_maps()

def _build_manuscript_figure():
    _script_entry()
    return None, None


if __name__ == "__main__":
    _configure_land_functions(sys.modules[__name__])
    _figure, _bbox = _build_manuscript_figure()
    _save_figure(_figure, _bbox, 'Supplementary_Fig_S11', _figure_options)
