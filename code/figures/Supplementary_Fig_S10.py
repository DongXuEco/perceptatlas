"""Plot Supplementary_Fig_S10 using the packaged numerical tables or supplied local inputs."""
from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_support"))
from figure_cli import prepare_figure as _prepare_figure, save_figure as _save_figure
from figure_cli import read_table as _read_table, model_tables as _model_tables
from figure_cli import configure_land_functions as _configure_land_functions

_figure_options = _prepare_figure('Supplementary_Fig_S10', 'analysis_tables') if __name__ == "__main__" else None

from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
import json
import platform
import sys
from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from Supplementary_Fig_S10_validation_functions import API_CSV, AXIS_LIMITS_MIN, MODE_DIRS, POI_LABEL, POI_ORDER, build_metrics, metric_block, read_api_reference, read_gee_mode
VALIDATION_DIR = Path(__file__).resolve().parent
CITY_GEOJSON = VALIDATION_DIR / 'accessibility_validation_city_sample500.geojson'
POINT_GEOJSON = VALIDATION_DIR / 'accessibility_validation_origins_sample500_merged_local.geojson'
OUT_DIR = output_dir('Supplementary_Fig_S10')
POI_COLOR = {'education': '#D83A3B', 'entertainment': '#247AB5', 'health': '#FF8113', 'living': '#2DA02D', 'production': '#9569BE', 'transport': '#818181'}
MODE_COLOR = {'walking': '#3B82D6', 'driving': '#F28817'}
MODE_DISPLAY_LABEL = {'walking': 'Non-motorized', 'driving': 'Motorized'}
B_DENSITY_CMAP = mpl.colormaps['RdYlBu_r']
FONT_SIZE = {'base': 6.3, 'axis_title': 7.0, 'axis_label': 6.5, 'tick': 6.5, 'legend': 6.0, 'annotation': 6.3, 'panel': 9.0, 'note': 6.3, 'group_title': 8.0, 'inset_title': 7.2, 'inset_tick': 6.0, 'inset_legend': 6.0}
OUT_SOURCE = OUT_DIR / 'fig_accessibility_validation_abcd_source_data.csv'
OUT_PLOT_SOURCE = OUT_DIR / 'fig_accessibility_validation_abcd_plot_window_source_data.csv'
OUT_METRICS = OUT_DIR / 'fig_accessibility_validation_abcd_metrics.csv'
OUT_AUDIT = OUT_DIR / 'fig_accessibility_validation_abcd_audit.json'
OUT_PNG = OUT_DIR / 'fig_accessibility_validation_abcd.png'
OUT_TIFF = OUT_DIR / 'fig_accessibility_validation_abcd.tiff'
OUT_PDF = OUT_DIR / 'fig_accessibility_validation_abcd.pdf'
OUT_SVG = OUT_DIR / 'fig_accessibility_validation_abcd.svg'

def set_style() -> None:
    try:
        plt.style.use(['science', 'no-latex'])
    except OSError:
        pass
    mpl.rcParams.update({'font.family': 'Arial', 'font.size': FONT_SIZE['base'], 'axes.titlesize': FONT_SIZE['axis_title'], 'axes.labelsize': FONT_SIZE['axis_label'], 'xtick.labelsize': FONT_SIZE['tick'], 'ytick.labelsize': FONT_SIZE['tick'], 'legend.fontsize': FONT_SIZE['legend'], 'axes.linewidth': 0.65, 'lines.linewidth': 0.8, 'xtick.major.width': 0.55, 'ytick.major.width': 0.55, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.top': False, 'ytick.right': False, 'xtick.minor.visible': False, 'ytick.minor.visible': False, 'axes.grid': False, 'savefig.dpi': 600, 'savefig.bbox': None, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

def clean_axes(ax: plt.Axes) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_linewidth(0.65)
    ax.tick_params(pad=1.8, top=False, right=False, which='both')

def panel_label(ax: plt.Axes, label: str, x: float=-0.12, y: float=1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha='left', va='top', fontsize=FONT_SIZE['panel'], fontweight='bold')

def panel_heading(fig: plt.Figure, x: float, y: float, label: str, title: str) -> None:
    fig.text(x - 0.025, y + 0.01, label, ha='left', va='top', fontsize=FONT_SIZE['panel'], fontweight='bold')
    fig.text(x, y + 0.006, title, ha='left', va='top', fontsize=FONT_SIZE['group_title'], fontweight='normal')

def p_label(p_value: float) -> str:
    if not np.isfinite(p_value):
        return 'P = NA'
    if p_value < 0.001:
        return 'P < 0.001'
    if p_value < 0.01:
        return 'P < 0.01'
    if p_value < 0.05:
        return 'P < 0.05'
    return f'P = {p_value:.2f}'

def build_validation_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    audits: dict[str, object] = {'script': str(Path(__file__).resolve()), 'python_executable': sys.executable, 'python_version': platform.python_implementation() + ' ' + platform.python_version(), 'api_reference_table': str(API_CSV), 'city_geojson': str(CITY_GEOJSON), 'point_geojson': str(POINT_GEOJSON), 'axis_limits_min': AXIS_LIMITS_MIN}
    gee_parts = []
    audits['gee_mode_audits'] = {}
    for mode, folder in MODE_DIRS.items():
        mode_df, mode_audit = read_gee_mode(mode, folder)
        gee_parts.append(mode_df)
        audits['gee_mode_audits'][mode] = mode_audit
    gee_long = pd.concat(gee_parts, ignore_index=True)
    api_reference, api_audit = read_api_reference()
    audits['api_audit'] = api_audit
    source = api_reference.merge(gee_long, on=['origin_id', 'mode', 'poi_type'], how='inner', suffixes=('_api', '_gee'), validate='one_to_one')
    source['residual_min'] = source['gee_time_min'] - source['api_time_min']
    source['abs_error_min'] = source['residual_min'].abs()
    source['within_axis_window'] = False
    for mode, limit in AXIS_LIMITS_MIN.items():
        mask = source['mode'].eq(mode) & source['api_time_min'].between(0, limit) & source['gee_time_min'].between(0, limit)
        source.loc[mask, 'within_axis_window'] = True
    plot_source = source[source['within_axis_window']].copy()
    metrics = build_metrics(source, plot_source)
    audits['joined_rows_clean'] = int(len(source))
    audits['joined_rows_plot_window'] = int(len(plot_source))
    audits['plot_window_counts_by_mode'] = {str(k): int(v) for k, v in plot_source['mode'].value_counts().sort_index().items()}
    audits['plot_window_counts_by_mode_poi'] = {f'{mode}|{poi}': int(count) for (mode, poi), count in plot_source.groupby(['mode', 'poi_type']).size().items()}
    return (source, plot_source, metrics, audits)

def remove_map_frame(ax: plt.Axes) -> None:
    ax.set_frame_on(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    outline = getattr(ax, 'outline_patch', None)
    if outline is not None:
        outline.set_visible(False)

def residual_density_inset(ax: plt.Axes, data: pd.DataFrame) -> dict[str, float]:
    hist_ax = ax.inset_axes([0.0, 0.15, 0.28, 0.2])
    x_min, x_max = (-100.0, 100.0)
    xs = np.linspace(x_min, x_max, 300)
    for mode in ['walking', 'driving']:
        values = data.loc[data['mode'].eq(mode), 'residual_min'].dropna()
        if len(values) > 2 and values.std() > 0:
            density = gaussian_kde(values)(xs)
            hist_ax.fill_between(xs, 0, density, color=MODE_COLOR[mode], alpha=0.18, linewidth=0)
            hist_ax.plot(xs, density, color=MODE_COLOR[mode], linewidth=1.05, label=MODE_DISPLAY_LABEL[mode])
    hist_ax.axvline(0, color='#4B5563', linewidth=0.7, linestyle=(0, (2, 2)), zorder=1)
    hist_ax.set_title('Residuals', loc='left', y=2.1, pad=1.2, fontsize=FONT_SIZE['inset_title'])
    hist_ax.set_xlabel('Residual (min)', labelpad=1.0, fontsize=FONT_SIZE['note'])
    hist_ax.set_ylabel('')
    hist_ax.set_xlim(x_min, x_max)
    hist_ax.set_yticks([])
    hist_ax.set_xticks([x_min, 0, x_max])
    hist_ax.tick_params(axis='both', labelsize=FONT_SIZE['inset_tick'], pad=1.0)
    hist_ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(-0.03, 1.75), handlelength=1.3, handletextpad=0.35, fontsize=FONT_SIZE['inset_legend'])
    hist_ax.set_facecolor((1, 1, 1, 0.86))
    clean_axes(hist_ax)
    hist_ax.spines['left'].set_visible(False)
    hist_ax.tick_params(axis='y', left=False, labelleft=False)
    return {'residual_density_xmin': x_min, 'residual_density_xmax': x_max}

def plot_map_panel(ax: plt.Axes, validation_data: pd.DataFrame) -> dict[str, int | float]:
    cities = gpd.read_file(resource(CITY_GEOJSON)).to_crs('EPSG:4326')
    points = gpd.read_file(resource(POINT_GEOJSON)).to_crs('EPSG:4326')
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor('white')
    land_path = shapereader.natural_earth('110m', 'physical', 'land')
    land_geometries = [geom for geom in shapereader.Reader(resource(land_path)).geometries() if geom.bounds[3] > -60]
    ax.add_geometries(land_geometries, crs=ccrs.PlateCarree(), facecolor='#E9EBF1', edgecolor='#BCC3CD', linewidth=0.25, zorder=0)
    ax.add_feature(cfeature.BORDERS, edgecolor='#C8CDD3', linewidth=0.18, zorder=1)
    ax.add_geometries(cities.geometry, crs=ccrs.PlateCarree(), facecolor='#D7DCE3', edgecolor='#777F89', linewidth=0.13, alpha=0.38, zorder=2)
    ax.scatter(cities['lon'], cities['lat'], s=8.5 * 0.36, facecolor='white', edgecolor='#111827', linewidth=0.42 * 0.6, alpha=0.82, transform=ccrs.PlateCarree(), label=f'Sample cities ({len(cities):,})', zorder=5)
    ax.scatter(points['origin_lon'], points['origin_lat'], s=3.8 * 0.36, facecolor='#2CA6A4', edgecolor='none', alpha=0.66, transform=ccrs.PlateCarree(), label=f'Sample points ({len(points):,})', zorder=4, rasterized=True)
    hist_audit = residual_density_inset(ax, validation_data)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], linestyle='none', marker='', color='none'))
    labels.append(f'OD pairs ({len(validation_data):,})')
    ax.legend(handles, labels, frameon=False, loc='lower left', bbox_to_anchor=(0.45, 0.005), handletextpad=0.35, handlelength=1.1, borderpad=0.1)
    remove_map_frame(ax)
    return {'sample_city_count': int(len(cities)), 'sample_point_count': int(len(points)), 'od_pair_count': int(len(validation_data)), **hist_audit}

def draw_identity_and_fit(ax: plt.Axes, data: pd.DataFrame, limit: float) -> None:
    ax.plot([0, limit], [0, limit], color='#6F7886', lw=0.8, ls=(0, (4, 3)), zorder=1)
    if len(data) >= 3 and data['api_time_min'].std() > 0 and (data['gee_time_min'].std() > 0):
        slope, intercept = np.polyfit(data['api_time_min'], data['gee_time_min'], 1)
        xs = np.array([0.0, limit])
        ax.plot(xs, intercept + slope * xs, color='black', lw=1.15, zorder=4)

def density_panel(ax: plt.Axes, data: pd.DataFrame, mode: str) -> None:
    limit = AXIS_LIMITS_MIN[mode]
    view = data[data['mode'].eq(mode)].copy()
    hb = ax.hexbin(view['api_time_min'], view['gee_time_min'], gridsize=40, extent=(0, limit, 0, limit), mincnt=2, cmap=B_DENSITY_CMAP, norm=Normalize(vmin=1, vmax=30, clip=True), linewidths=0, zorder=2)
    hb.set_clim(1, 30)
    draw_identity_and_fit(ax, view, limit)
    metric = metric_block(view, mode, 'all_plot_window', limit)
    ax.text(0.055, 0.955, MODE_DISPLAY_LABEL[mode], transform=ax.transAxes, ha='left', va='top', fontsize=FONT_SIZE['axis_title'], zorder=6)
    ax.text(0.055, 0.84, f"r={metric['pearson_r']:.2f}\nrho={metric['spearman_rho']:.2f}\n{p_label(metric['spearman_p'])}", transform=ax.transAxes, ha='left', va='top', fontsize=FONT_SIZE['annotation'], linespacing=1.4, zorder=6)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ticks = [0, 150, 300, 450, 600] if limit == 600 else [0, 50, 100, 150, 200]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Travel time (min, Google)')
    ax.set_ylabel('Travel time (min, our)')
    clean_axes(ax)
    cax = ax.inset_axes([0.665, 0.115, 0.3, 0.046])
    cb = plt.colorbar(hb, cax=cax, orientation='horizontal')
    cb.outline.set_linewidth(0.35)
    cb.set_ticks([1, 15, 30])
    cb.ax.tick_params(labelsize=FONT_SIZE['inset_tick'], length=2.8, pad=0.6)

def bin_density(x: np.ndarray, y: np.ndarray, limit: float, bins: int=40) -> np.ndarray:
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=((0, limit), (0, limit)))
    xi = np.clip(np.digitize(x, xedges) - 1, 0, bins - 1)
    yi = np.clip(np.digitize(y, yedges) - 1, 0, bins - 1)
    return hist[xi, yi]

def category_scatter_panel(ax: plt.Axes, data: pd.DataFrame, mode: str, poi: str) -> None:
    limit = AXIS_LIMITS_MIN[mode]
    view = data[data['mode'].eq(mode) & data['poi_type'].eq(poi)].copy()
    color = POI_COLOR[poi]
    if len(view):
        ax.scatter(view['api_time_min'], view['gee_time_min'], s=15.5 * 0.36, color=color, alpha=0.5, linewidth=0, rasterized=True, zorder=2)
    draw_identity_and_fit(ax, view, limit)
    metric = metric_block(view, mode, poi, limit)
    ax.text(0.045, 0.965, POI_LABEL[poi], transform=ax.transAxes, ha='left', va='top', fontsize=FONT_SIZE['axis_title'], bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.74), zorder=6)
    ax.text(0.955, 0.065, f"r={metric['pearson_r']:.2f}\nrho={metric['spearman_rho']:.2f}\n{p_label(metric['spearman_p'])}", transform=ax.transAxes, ha='right', va='bottom', fontsize=FONT_SIZE['annotation'], linespacing=1.4, bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='none', alpha=0.78), zorder=6)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ticks = [0, 300, 600] if limit == 600 else [0, 100, 200]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_aspect('equal', adjustable='box')
    clean_axes(ax)

def category_grid(fig: plt.Figure, spec: mpl.gridspec.SubplotSpec, data: pd.DataFrame, mode: str, label: str) -> None:
    sub = spec.subgridspec(2, 3, wspace=0.16, hspace=0.2)
    axes = []
    for idx, poi in enumerate(POI_ORDER):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(sub[row, col])
        category_scatter_panel(ax, data, mode, poi)
        axes.append(ax)
        if col == 0:
            ax.set_ylabel('Travel time (min, our)')
        else:
            ax.tick_params(labelleft=False)
        if row == 1:
            ax.set_xlabel('Travel time (min, Google)')
        else:
            ax.tick_params(labelbottom=False)
    panel_label(axes[0], label, x=-0.34, y=1.12)

def category_grid_manual(fig: plt.Figure, data: pd.DataFrame, mode: str, label: str, left: float, top: float, axis_w: float, axis_h: float, col_gap: float, row_gap: float) -> None:
    axes = []
    for idx, poi in enumerate(POI_ORDER):
        row = idx // 3
        col = idx % 3
        x0 = left + col * (axis_w + col_gap)
        y0 = top - (row + 1) * axis_h - row * row_gap
        ax = fig.add_axes([x0, y0, axis_w, axis_h])
        category_scatter_panel(ax, data, mode, poi)
        axes.append(ax)
        if col == 0:
            ax.set_ylabel('Travel time (min, our)')
        else:
            ax.tick_params(labelleft=False)
        if row == 0:
            ax.tick_params(labelbottom=False)
    fig.text(left + (3 * axis_w + 2 * col_gap) / 2, top - 2 * axis_h - row_gap - 0.04, 'Travel time (min, Google)', ha='center', va='top', fontsize=FONT_SIZE['axis_label'])

def make_figure(plot_source: pd.DataFrame, metrics: pd.DataFrame, *, return_figure: bool=False):
    set_style()
    fig_w, fig_h = (7.2, 5.0)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=600)
    left_margin = 0.065
    right_margin = 0.985
    top_y = 0.92
    horizontal_gap = 0.075
    vertical_gap = 0.095
    block_w = (right_margin - left_margin - horizontal_gap) / 2.0
    left_x = left_margin
    right_x = left_margin + block_w + horizontal_gap
    map_h = block_w * fig_w / (2.0 * fig_h)
    ax_a = fig.add_axes([left_x, top_y - map_h, block_w, map_h], projection=ccrs.PlateCarree())
    map_counts = plot_map_panel(ax_a, plot_source)
    panel_heading(fig, left_x, top_y + 0.026, 'a', 'Global sample cities and validation points')
    b_gap = 0.05
    b_w = (block_w - b_gap) / 2.0
    b_h = 0.9 * (block_w - 0.032) / 2.0 * fig_w / fig_h
    b_top = top_y - 0.018
    ax_b1 = fig.add_axes([right_x, b_top - b_h, b_w, b_h])
    ax_b2 = fig.add_axes([right_x + b_w + b_gap, b_top - b_h, b_w, b_h])
    density_panel(ax_b1, plot_source, 'walking')
    density_panel(ax_b2, plot_source, 'driving')
    ax_b1.set_aspect('auto')
    ax_b2.set_aspect('auto')
    ax_b2.set_ylabel('')
    panel_heading(fig, right_x, top_y + 0.026, 'b', 'Overall agreement by travel mode')
    bottom_top = top_y - max(map_h, b_h) - vertical_gap
    cat_col_gap = 0.018
    cat_row_gap = 0.02
    cat_w = (block_w - 2.0 * cat_col_gap) / 3.0
    cat_h = cat_w * fig_w / fig_h
    category_grid_manual(fig, plot_source, 'walking', 'c', left_x, bottom_top, cat_w, cat_h, cat_col_gap, cat_row_gap)
    category_grid_manual(fig, plot_source, 'driving', 'd', right_x, bottom_top, cat_w, cat_h, cat_col_gap, cat_row_gap)
    panel_heading(fig, left_x, bottom_top + 0.03, 'c', 'Non-motorized mode by POI class')
    panel_heading(fig, right_x, bottom_top + 0.03, 'd', 'Motorized mode by POI class')
    if return_figure:
        return (fig, map_counts)
    fig.savefig(OUT_PNG, dpi=600)
    fig.savefig(OUT_TIFF, dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_SVG)
    plt.close(fig)
    return map_counts

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source, plot_source, metrics, audits = build_validation_data()
    source.to_csv(OUT_SOURCE, index=False, encoding='utf-8-sig')
    plot_source.to_csv(OUT_PLOT_SOURCE, index=False, encoding='utf-8-sig')
    metrics.to_csv(OUT_METRICS, index=False, encoding='utf-8-sig')
    map_counts = make_figure(plot_source, metrics)
    audits['map_counts'] = map_counts
    audits['outputs'] = {'source_data': str(OUT_SOURCE), 'plot_window_source_data': str(OUT_PLOT_SOURCE), 'metrics': str(OUT_METRICS), 'figure_png': str(OUT_PNG), 'figure_tiff': str(OUT_TIFF), 'figure_pdf': str(OUT_PDF), 'figure_svg': str(OUT_SVG)}
    OUT_AUDIT.write_text(json.dumps(audits, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Source data: {OUT_SOURCE}')
    print(f'Plot-window source data: {OUT_PLOT_SOURCE}')
    print(f'Metrics: {OUT_METRICS}')
    print(f'Audit: {OUT_AUDIT}')
    print(f'Figure PNG: {OUT_PNG}')
    print(f'Figure TIFF: {OUT_TIFF}')
    print(f'Figure PDF: {OUT_PDF}')
    print(f'Figure SVG: {OUT_SVG}')
    print()
    print(metrics[['mode', 'poi_type', 'axis_limit_min', 'n', 'pearson_r', 'spearman_rho', 'spearman_p', 'mae_min', 'bias_gee_minus_api_min']].to_string(index=False))

def _build_manuscript_figure():
    source = _read_table("Supplementary_Fig_S10/Supplementary_Fig_S10_plot_window_source_data.csv")
    metrics = _read_table("Supplementary_Fig_S10/Supplementary_Fig_S10_metrics.csv")
    return make_figure(source, metrics, return_figure=True)[0], None


if __name__ == "__main__":
    _configure_land_functions(sys.modules[__name__])
    _figure, _bbox = _build_manuscript_figure()
    _save_figure(_figure, _bbox, 'Supplementary_Fig_S10', _figure_options)
