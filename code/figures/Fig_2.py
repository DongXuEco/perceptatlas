"""Plot Fig_2 using the packaged numerical tables or supplied local inputs."""
from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_support"))
from figure_cli import prepare_figure as _prepare_figure, save_figure as _save_figure
from figure_cli import read_table as _read_table, model_tables as _model_tables
from figure_cli import configure_land_functions as _configure_land_functions

_figure_options = _prepare_figure('Fig_2', 'analysis_tables') if __name__ == "__main__" else None

from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
import argparse
import hashlib
import json
import string
import warnings
from dataclasses import dataclass
from pathlib import Path
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.transforms import Bbox
from PIL import Image
from scipy.stats import gaussian_kde, spearmanr
ROOT = PACKAGE_ROOT
EMOTIONS = ('beauty', 'wealthy', 'safer', 'livelier', 'boring', 'depression')
EMOTION_LABELS = ('Beauty', 'Wealth', 'Safety', 'Liveliness', 'Boredom', 'Depression')
MEAN_COLUMNS = [f'mean50_{e}' for e in EMOTIONS]
SERVICES = ('education', 'entertainment', 'health', 'living', 'production', 'transport')
ACCESS_COLUMNS = [f'access_nonmotor_{s}_mean' for s in SERVICES]
STEM = 'fig_global_city_emotion_atlas_and_gradients'
WIDTH_MM, HEIGHT_MM = (183.0, 224.0)
LEFT_MM, RIGHT_MM, GUTTER_MM = (12.0, 3.0, 8.0)
COLUMN_MM = (WIDTH_MM - LEFT_MM - RIGHT_MM - 2 * GUTTER_MM) / 3
FONT = 'Arial'
MAP_TICK_SIZE = 6
MAP_LIMITS = {'eli50': (-0.3, 0.3), 'positive_mean50': (0.2, 0.7), 'negative_mean50': (0.4, 0.7), 'mean50_beauty': (0.2, 0.8), 'mean50_wealthy': (0.25, 0.75), 'mean50_safer': (0.35, 0.85), 'mean50_livelier': (0.1, 0.6), 'mean50_boring': (0.2, 0.5), 'mean50_depression': (0.2, 0.9)}
INK, MUTED, LAND = ('#20252a', '#60676f', '#e8ebef')
GRADE_COLORS = ('#B01546', '#F8874F', '#FFFFBE', '#8AD0A4', '#525FA9')
ELI_CMAP = colors.LinearSegmentedColormap.from_list('eli_diverging', ['#F27C2B', '#FFB66D', '#FFE7CC', '#DCE6F2', '#A7CFF2', '#3B82D6', '#0D47A1'])

@dataclass(frozen=True)
class Driver:
    column: str
    title: str
    unit: str
    direction: str = 'higher numerical value'
DRIVERS = (Driver('population_density_mean', 'Population density', 'people/km2'), Driver('ntl_radiance_mean', 'Night-time lights', 'source radiance unit'), Driver('green_fraction_mean', 'Green-space cover', 'fraction'), Driver('building_density_mean', 'Built-up density', 'fraction'), Driver('access_nonmotor_six_mean_min', 'Non-motorized service travel time', 'min', 'longer travel time; worse access'), Driver('lst_c_mean', 'Land-surface temperature', 'deg C'))

def setup_style() -> None:
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': [FONT, 'DejaVu Sans'], 'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 8, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.linewidth': 0.6, 'axes.edgecolor': INK, 'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'axes.spines.top': False, 'axes.spines.right': False, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'axes.unicode_minus': False})

def load_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw = pd.read_csv(resource(path))
    required = {'id_1', 'lon', 'lat', 'eli50', 'ns_group', 'population_density_min', *MEAN_COLUMNS, *ACCESS_COLUMNS, *(d.column for d in DRIVERS[:-2]), 'lst_c_mean'}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    if raw['id_1'].isna().any() or raw['id_1'].duplicated().any():
        raise ValueError('City IDs must be present and unique.')
    numeric = required - {'ns_group'}
    raw[list(numeric)] = raw[list(numeric)].apply(pd.to_numeric, errors='raise')
    complete = np.isfinite(raw[['lon', 'lat', 'eli50', *MEAN_COLUMNS]]).all(axis=1)
    atlas = raw.loc[complete].copy().sort_values('id_1')
    if not atlas[MEAN_COLUMNS].ge(0).all().all() or not atlas[MEAN_COLUMNS].le(1).all().all():
        raise ValueError('Perception scores must be in [0, 1]; do not silently rescale.')
    if not atlas['lon'].between(-180, 180).all() or not atlas['lat'].between(-60, 85).all():
        raise ValueError('A city falls outside the map extent; revise the extent before plotting.')
    atlas['positive_mean50'] = atlas[MEAN_COLUMNS[:4]].mean(axis=1, skipna=False)
    atlas['negative_mean50'] = atlas[MEAN_COLUMNS[4:]].mean(axis=1, skipna=False)
    eli = atlas['positive_mean50'] - atlas['negative_mean50']
    error = float((eli - atlas['eli50']).abs().max())
    if error > 1e-10:
        raise ValueError(f'ELI formula mismatch: {error}')
    access_valid = np.isfinite(atlas[ACCESS_COLUMNS]).all(axis=1) & atlas[ACCESS_COLUMNS].ge(0).all(axis=1)
    atlas['access_nonmotor_six_mean_min'] = atlas[ACCESS_COLUMNS].mean(axis=1, skipna=False).where(access_valid)
    flags = []
    for _, row in raw.loc[~complete].iterrows():
        flags.append({'id_1': row.id_1, 'scope': 'all panels', 'reason': 'incomplete perception or coordinates'})
    for _, row in atlas.loc[atlas.population_density_min.lt(0)].iterrows():
        flags.append({'id_1': row.id_1, 'scope': 'population panel only', 'reason': 'negative source-grid population statistic'})
    audit = {'input_sha256': hashlib.sha256(resource(path).read_bytes()).hexdigest(), 'input_rows': len(raw), 'map_cities': len(atlas), 'eli_formula_max_abs_error': error, 'map_group_counts': atlas.ns_group.value_counts().to_dict(), 'population_negative_min_all_rows': int(raw.population_density_min.lt(0).sum()), 'population_negative_min_map_rows': int(atlas.population_density_min.lt(0).sum()), 'access_complete_cities': int(access_valid.sum()), 'access_all_zero_cities': int(atlas.loc[access_valid, ACCESS_COLUMNS].eq(0).all(axis=1).sum()), 'status': 'descriptive draft; upstream spatial-support audit remains open'}
    return (atlas, pd.DataFrame(flags), audit)

def make_groups(atlas: pd.DataFrame, sample_mode: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    masks = {}
    for driver in DRIVERS:
        x = atlas[driver.column]
        valid = np.isfinite(x)
        if driver.column != 'lst_c_mean':
            valid &= x.ge(0)
        if driver.column in ('green_fraction_mean', 'building_density_mean'):
            valid &= x.le(1)
        if driver.column == 'population_density_mean':
            valid &= np.isfinite(atlas.population_density_min) & atlas.population_density_min.ge(0)
        masks[driver.column] = valid
    common = pd.DataFrame(masks).all(axis=1)
    grouped, summaries, sources = ({}, [], [])
    for driver in DRIVERS:
        valid = common if sample_mode == 'common' else masks[driver.column]
        data = atlas.loc[valid, ['id_1', 'ns_group', 'eli50', driver.column]].copy()
        if len(data) < 25:
            raise ValueError(f'Insufficient cities for quintiles: {driver.column}')
        labels, edges = pd.qcut(data[driver.column], 5, labels=False, retbins=True, duplicates='raise')
        data['quintile'] = labels.astype(int) + 1
        rho = float(spearmanr(data[driver.column], data.eli50).statistic)
        grouped[driver.column] = (data, edges, rho)
        for grade, group in data.groupby('quintile', observed=True):
            q = group.eli50.quantile([0.25, 0.5, 0.75])
            summaries.append({'driver': driver.column, 'unit': driver.unit, 'direction': driver.direction, 'quintile': int(grade), 'n': len(group), 'lower_edge': edges[grade - 1], 'upper_edge': edges[grade], 'eli_mean': group.eli50.mean(), 'eli_q25': q.loc[0.25], 'eli_median': q.loc[0.5], 'eli_q75': q.loc[0.75], 'spearman_rho_unbinned': rho})
        source = data.rename(columns={driver.column: 'driver_value'})
        source['driver'] = driver.column
        sources.append(source)
    return (grouped, pd.DataFrame(summaries), pd.concat(sources, ignore_index=True))

def get_basemap(path: Path | None) -> gpd.GeoDataFrame:
    if path is None:
        hits = sorted(resource_glob(Path(pyogrio.__file__).resolve().parent, 'naturalearth_lowres.shp'))
        if not hits:
            raise FileNotFoundError('Supply a local country polygon dataset with --basemap.')
        path = hits[0]
    world = gpd.read_file(resource(path))
    if world.crs is None:
        raise ValueError('Basemap has no CRS.')
    return world.to_crs('EPSG:4326')

def rect_mm(fig, x, y, width, height):
    return fig.add_axes([x / WIDTH_MM, y / HEIGHT_MM, width / WIDTH_MM, height / HEIGHT_MM])

def heading(fig, letter: str, title: str, x: float, y: float) -> None:
    fig.text((x - 4) / WIDTH_MM, y / HEIGHT_MM, letter, fontsize=9, weight='bold', va='bottom')
    fig.text(x / WIDTH_MM, y / HEIGHT_MM, title, fontsize=8, va='bottom')

def draw_map(fig, world, data, column, x, y, width, cmap, norm, size):
    height = width * 145 / 360
    ax = rect_mm(fig, x, y, width, height)
    world.plot(ax=ax, color=LAND, edgecolor='white', linewidth=0.12, aspect=None)
    order = np.random.default_rng(4021).permutation(len(data))
    points = data.iloc[order]
    ax.scatter(points.lon, points.lat, c=points[column], cmap=cmap, norm=norm, s=size, edgecolors='none', alpha=0.9, rasterized=True)
    ax.set(xlim=(-180, 180), ylim=(-60, 85), aspect='equal')
    ax.set_axis_off()
    return ax

def map_scale(data, column):
    lower, upper = MAP_LIMITS[column]
    if column == 'eli50':
        return colors.TwoSlopeNorm(vmin=lower, vcenter=0, vmax=upper)
    return colors.Normalize(lower, upper, clip=True)

def density_scale(data, column):
    values = data[column]
    if column == 'eli50':
        bound = max(0.05, np.ceil(values.abs().max() * 20) / 20)
        return colors.TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
    lower = np.floor(values.min() * 20) / 20
    upper = np.ceil(values.max() * 20) / 20
    if upper <= lower:
        upper = lower + 0.05
    return colors.Normalize(lower, upper)

def colorbar(fig, x, y, width, cmap, norm, ticks):
    bar_width = width * 0.45 * 0.8
    ax = rect_mm(fig, x + (width - bar_width) / 2 + 3.8, y + 5.4, bar_width, 2 * 0.8 * 0.85 * 0.8)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal', ticks=ticks, format='%.2f')
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(length=2, pad=1, labelsize=MAP_TICK_SIZE)
    return ax

def draw_map_density(fig, map_ax, data, column, cmap, norm):
    west, east, south, north = (-178, -88, -58, -25)
    covered = data.lon.between(west, east) & data.lat.between(south, north)
    if covered.any():
        raise ValueError('Density inset covers a city; choose another empty map region.')
    box = map_ax.get_position()
    ax = fig.add_axes([box.x0 + (west + 180) / 360 * box.width, box.y0 + (south + 60) / 145 * box.height - 1.2 / HEIGHT_MM, (east - west) / 360 * box.width, (north - south) / 145 * box.height])
    values = data[column].to_numpy()
    grid = np.linspace(values.min(), values.max(), 160)
    density = gaussian_kde(values, bw_method='scott')(grid)
    for i in range(len(grid) - 1):
        ax.fill_between(grid[i:i + 2], density[i:i + 2], color=cmap(norm(grid[i:i + 2].mean())), linewidth=0)
    ax.plot(grid, density, color=INK, linewidth=0.6)
    median = float(np.median(values))
    height = float(gaussian_kde(values, bw_method='scott')(median)[0])
    ax.vlines(median, 0, height, color=INK, linewidth=0.5, linestyle=':')
    if column == 'eli50':
        ax.text(1.05, 1.65, 'Median', transform=ax.transAxes, ha='right', va='bottom', fontsize=5, fontfamily='Arial', color=INK)
        ax.annotate('', xy=(median, height), xytext=(0.83, 1.58), textcoords='axes fraction', arrowprops={'arrowstyle': '-', 'color': INK, 'linewidth': 0.4, 'connectionstyle': 'angle,angleA=-90,angleB=0,rad=0'}, annotation_clip=False)
    extent = density_scale(data, column)
    ax.set(xlim=(extent.vmin, extent.vmax), ylim=(0, density.max() * 1.1))
    ticks = [extent.vmin, 0, extent.vmax] if column == 'eli50' else [extent.vmin, extent.vmax]
    ax.set_xticks(ticks, [f'{v:.2f}' if column != 'eli50' else f'{v:g}' for v in ticks])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.4)
    ax.tick_params(axis='x', labelsize=MAP_TICK_SIZE, pad=1, length=1.5, width=0.4)
    ax.text(0, 1.1, 'Density', transform=ax.transAxes, fontsize=6, va='bottom')
    return ax

def load_significance(path, source, groups):
    recorded = pd.read_csv(resource(path.parent / 'analysis_city_samples.csv'))
    keys = ['driver', 'id_1']
    pd.testing.assert_frame_equal(source.sort_values(keys).reset_index(drop=True), recorded[source.columns].sort_values(keys).reset_index(drop=True), check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
    stats = pd.read_csv(resource(path)).set_index('driver')
    if not stats.index.is_unique or set(stats.index) != set(groups) or (not stats.specification.eq('km500_shift0').all()):
        raise ValueError('Expected the six matched primary 500-km significance results.')
    for driver, (data, _, rho) in groups.items():
        row = stats.loc[driver]
        if row.n_cities != len(data) or not np.isclose(row.rho, rho, rtol=0, atol=1e-12):
            raise ValueError(f'Significance sample mismatch: {driver}')
        if not np.isfinite(row.p_boot_approx) or not 0 < row.p_boot_approx <= 1:
            raise ValueError(f'Invalid approximate P value: {driver}')
    return stats.p_boot_approx.to_dict()

def draw_raincloud(ax, data, rho, ylim, seed, xlabel, p_value=None):
    medians = data.groupby('quintile')['eli50'].median().reindex(range(1, 6))
    ax.plot(range(1, 6), medians, color='#505050', linewidth=0.7, zorder=1.9, label='Quintile medians', linestyle='--')
    for grade in range(1, 6):
        values = data.loc[data.quintile.eq(grade), 'eli50'].to_numpy()
        shade = GRADE_COLORS[grade - 1]
        q05, q25, q50, q75, q95 = np.percentile(values, [5, 25, 50, 75, 95])
        hist, edges = np.histogram(values, bins=32)
        centers = (edges[:-1] + edges[1:]) / 2
        half_width = 0.34 * hist / max(hist.max(), 1)
        for yy, hw in zip(centers, half_width):
            ax.plot([grade, grade + hw], [yy, yy], color=shade, alpha=1.0, lw=1.0, zorder=1, label='Right histogram')
            ax.plot([grade - 0.2, grade - 0.2 - hw * 0.42], [yy, yy], color=shade, alpha=1.0, lw=0.48, zorder=1, label='Left histogram')
        ax.plot([grade, grade], [q05, q95], color=INK, lw=0.55, zorder=3, label='5-95 percentile whisker')
        for side in (-0.07, 0.07):
            ax.plot([grade + side, grade + side], [q25, q75], color=INK, lw=1.6, solid_capstyle='butt', zorder=3, label='IQR side')
        for quantile in (q25, q75):
            ax.plot([grade - 0.07, grade + 0.07], [quantile, quantile], color=INK, lw=0.55, zorder=3)
        ax.plot([grade - 0.075, grade + 0.075], [q50, q50], color=INK, lw=0.75, zorder=5)
    ax.axhline(0, color='#aeb6bd', linewidth=0.5, linestyle=':', zorder=0)
    ax.set(xlim=(0.45, 5.55), ylim=ylim, ylabel='City ELI', xlabel=xlabel)
    ax.set_xticks(range(1, 6), [f'Q{i}' for i in range(1, 6)])
    ax.set_yticks([-0.4, 0, 0.4])
    ax.text(0.03, 0.97, f'$\\rho$ = {rho:+.2f}', transform=ax.transAxes, fontsize=6.5, va='top')
    if p_value is not None:
        p_label = 'P < 0.001' if p_value < 0.001 else f'P ≈ {p_value:.3f}'
        ax.text(0.43, 0.97, p_label, transform=ax.transAxes, fontsize=6.5, fontfamily='Arial', fontstyle='normal', va='top')

def align_colorbar_tick_labels(fig, density_axes, colorbar_axes):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for density_ax, bar_ax in zip(density_axes, colorbar_axes):
        target_y = density_ax.get_xticklabels()[0].get_window_extent(renderer).y1
        label_y = bar_ax.get_xticklabels()[0].get_window_extent(renderer).y1
        shift_pt = (label_y - target_y) * 72 / fig.dpi
        pad = bar_ax.xaxis.get_major_ticks()[0].get_pad() + shift_pt
        bar_ax.tick_params(axis='x', which='major', pad=pad)

def build_figure(atlas, groups, world, significance=None):
    if atlas.eli50.abs().max() > 0.6:
        raise ValueError('ELI exceeds the fixed map scale; revise map and distribution limits before plotting.')
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), dpi=180)
    xs = [LEFT_MM + i * (COLUMN_MM + GUTTER_MM) for i in range(3)]
    map_specs = [('eli50', 'Emotional livability (ELI)'), ('positive_mean50', 'Positive perception'), ('negative_mean50', 'Negative perception')]
    map_specs.extend(((f'mean50_{e}', label) for e, label in zip(EMOTIONS, EMOTION_LABELS)))
    map_axes, density_axes, colorbar_axes = ([], [], [])
    for i, (column, label) in enumerate(map_specs):
        row, col = divmod(i, 3)
        bottom = (196, 160, 124)[row]
        heading(fig, string.ascii_lowercase[i], label, xs[col], bottom + COLUMN_MM * 145 / 360 + 2.5)
        cmap = ELI_CMAP if i == 0 else plt.get_cmap('Spectral')
        if i in (1, 2):
            cmap = plt.get_cmap('Greens' if i == 1 else 'YlOrRd')
        scale = map_scale(atlas, column)
        ax = draw_map(fig, world, atlas, column, xs[col], bottom, COLUMN_MM, cmap, scale, 1.0)
        map_axes.append(ax)
        density_axes.append(draw_map_density(fig, ax, atlas, column, cmap, scale))
        ticks = np.linspace(scale.vmin, scale.vmax, 3)
        colorbar_axes.append(colorbar(fig, xs[col], bottom - 6, COLUMN_MM, cmap, scale, ticks))
    stats_axes = []
    for i, driver in enumerate(DRIVERS):
        row, col = divmod(i, 3)
        bottom = (66, 15)[row] + 4.4
        heading(fig, string.ascii_lowercase[i + 9], driver.title, xs[col], bottom + 39)
        ax = rect_mm(fig, xs[col] + col * COLUMN_MM * 0.05, bottom + 3.5, COLUMN_MM * 0.9, 35 * 0.9)
        data, _, rho = groups[driver.column]
        xlabel = 'Quintile (Q1–Q5)'
        draw_raincloud(ax, data, rho, (-0.55, 0.7), 8100 + i, xlabel, None if significance is None else significance[driver.column])
        stats_axes.append(ax)
    align_colorbar_tick_labels(fig, density_axes, colorbar_axes)
    return (fig, {'maps': map_axes, 'statistics': stats_axes, 'density_insets': density_axes, 'colorbars': colorbar_axes})

def audit_layout(fig, panels) -> dict:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    texts, outside, overlaps = ([], [], [])
    drawn_text = list(fig.texts)
    for ax in fig.axes:
        drawn_text.extend(ax.texts)
        drawn_text.extend([ax.title, ax._left_title, ax._right_title])
        if ax.axison:
            for axis, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
                drawn_text.extend([axis.label, axis.get_offset_text()])
                for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                    if min(limits) - 1e-10 <= tick.get_loc() <= max(limits) + 1e-10:
                        drawn_text.extend([tick.label1, tick.label2])
    for artist in drawn_text:
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        box = artist.get_window_extent(renderer)
        if box.width == 0 or box.height == 0:
            continue
        text = artist.get_text()
        if box.x0 < -1 or box.y0 < -1 or box.x1 > canvas.x1 + 1 or (box.y1 > canvas.y1 + 1):
            outside.append(text)
        texts.append((text, box))
    for i, (text, box) in enumerate(texts):
        for other, other_box in texts[i + 1:]:
            if box.overlaps(other_box):
                overlaps.append([text, other])
    aligned = True
    for key in ('maps', 'statistics', 'colorbars'):
        axes = panels[key]
        for start in range(0, len(axes), 3):
            boxes = [ax.get_position() for ax in axes[start:start + 3]]
            aligned &= np.ptp([b.y0 for b in boxes]) < 1e-08
            aligned &= np.ptp([b.y1 for b in boxes]) < 1e-08
        for col in range(3):
            for start in range(3, len(axes), 3):
                aligned &= abs(axes[col].get_position().x0 - axes[col + start].get_position().x0) < 1e-08
    for i in range(6):
        a, b = (panels['maps'][i].get_position(), panels['statistics'][i].get_position())
        col, row = (i % 3, i // 3)
        aligned &= abs(b.x0 - (a.x0 + col * a.width * 0.05)) < 1e-08
        aligned &= abs(b.width - a.width * 0.9) < 1e-08
        aligned &= abs(b.height - 31.5 / HEIGHT_MM) < 1e-08
        aligned &= abs(b.y1 - ((66, 15)[row] + 4.4 + 35) / HEIGHT_MM) < 1e-08
    return {'text_outside_canvas': outside, 'text_overlaps': overlaps, 'grid_aligned': bool(aligned), 'render_warnings': [str(w.message) for w in caught], 'minimum_font_pt': min((t.get_fontsize() for t in drawn_text if t.get_visible() and t.get_text().strip()))}

def bottom_crop_box(fig):
    fig.canvas.draw()
    bottom = max(0, fig.get_tightbbox(fig.canvas.get_renderer()).y0 - 2 / 25.4)
    width, height = fig.get_size_inches()
    return Bbox.from_extents(0, bottom, width, height)

def methods_text(audit, summary, sample_mode) -> str:
    lines = ['# Global urban perception atlas and driver-stratified ELI', '', '## Figure contract', 'Descriptive question: how does urban visual perception vary geographically and across urban conditions?', 'Nine city maps and six raincloud panels form a map-led quantitative composite; no causal or accuracy claim is made.', 'Python only; histogram-stripe distributions with percentile boxes and whiskers.', '', '## Caption', '**Global city-level emotional livability and its distribution across urban conditions.** a-c, City emotional livability index (ELI), positive perception and negative perception. Positive perception is the equal-weight mean of beauty, wealth, safety and liveliness; negative perception is the equal-weight mean of boredom and depression, without sign reversal. d-i, Beauty, wealth, safety, liveliness, boredom and depression scores. Each map includes a lower-left KDE of the corresponding city values, with Scott bandwidth, observed-support truncation and a dotted median. Inset vertical density scales are independently adjusted, so peak heights should not be compared between insets. All nine maps use the same cities. Each map has its own colorbar and numerical color limits; identical colors across maps do not indicate identical scores. A map and its density inset share one color normalization; inset x limits retain the full observed range independently of the color limits. j-o, ELI distributions across population density, night-time light radiance, green-space cover, built-up density, non-motorized service travel time and land-surface temperature quintiles. Q1-Q5 run from the lowest to highest numerical values for each variable; in n, longer travel times indicate worse service accessibility. Right-hand horizontal stripes show 32-bin histograms of city ELI, independently scaled to equal maximum width. Thinner left-hand stripes mirror the same counts at 42% of that width. These are binned frequencies, not individual observations or fitted density curves. Open boxes show the interquartile range, central lines the median, and thin dark-gray dashed lines connect the five group medians descriptively, without curve fitting. Whiskers span the 5th to 95th percentiles; histograms include the full observed range. Upper-left rho annotations give the descriptive Spearman correlation on unbinned values; group sample sizes are reported in the accompanying source tables. P annotations are exploratory approximate two-sided spatial-cluster bootstrap P values for zero Spearman correlation, using 500-km EPSG:6933 projected cells and 9,999 paired block resamples. They are not BH/BY-adjusted q values or tests between quintile groups. P values are displayed to three decimal places, with values below 0.001 labeled P < 0.001. Underlying values of 0.0002 reach the Monte Carlo resolution floor. This working block approximation does not establish complete control of spatial dependence; see significance/methods.md and the six-specification sensitivity results. No confidence intervals or significance stars are shown. ELI is the mean of the four positive dimensions minus the mean of boredom and depression. Higher boredom/depression means more negative perception; those maps are not reversed. Maps use Plate Carree coordinates, 180 W-180 E and 60 S-85 N.', '', '## Data and exclusions', f"Input rows: {audit['input_rows']}; complete mapped cities: {audit['map_cities']}.", 'Use mean50_* and eli50 only; legacy city-point emotion attributes are not used.', 'Population panel excludes every mapped city whose stored population_density_min is negative or missing. This is a conservative exclusion, not a correction of upstream raster errors.', f'Sampling mode: {sample_mode}. Variable-specific mode retains all eligible cities for each driver; common mode uses the intersection across all six drivers. Quintile thresholds are recomputed for that sample.', 'Five global city-equal quantile groups use pandas.qcut with right-closed intervals and the minimum included. Duplicate boundaries raise an error; ties are never split by city order. Values are not winsorized or imputed.', '', '## Composite accessibility', 'A new descriptive CITY-LEVEL summary: arithmetic mean of the six non-motor service mean travel times (education, entertainment, health, living, production and transport), equal weights 1/6, minutes. All six must be finite and non-negative; partial means are prohibited. Larger means worse access. No time cap, inversion or percentile normalization is applied. This is not the existing pixel-level capped accessibility index and is not equivalent to averaging six services on a shared raster footprint. Zeros are retained as recorded travel times, not created from missing values; their upstream meaning remains to be audited.', '', '## Open provenance limitations', 'This script does not re-extract rasters or harmonize spatial support. mean50_* was computed after average resampling onto the city-zone grid, not as a native-50-m polygon zonal statistic. Most driver inputs are urban-mask-clipped, whereas the emotion aggregation did not separately apply that mask. Green-cover clipping provenance remains incomplete. Geographic and variable-specific coverage can differ. Correlations with predictors used upstream in emotion raster production are not independent validation. Country/spatial confounding is not adjusted.', '', '## Export and reproducibility', '183 x 224 mm layout canvas; export crops only the bottom whitespace, retaining a 2-mm content margin. Arial 6-9 pt; PNG/TIFF at the requested 300 or 600 dpi; PDF/SVG retain editable text. Map city point layers are rasterized for compact vector files. Fixed map draw-order seeds affect display only; observed values are unchanged. Input SHA256 is recorded in audit.json. Per-map color limits are author-specified and recorded in audit.json; ELI is centered on zero. Values outside these limits use endpoint colors; no observations are removed or altered. Density x limits encompass observed values rounded outward to 0.05, symmetric for ELI. Color and grade positions jointly encode groups; a grayscale preview is exported.', '', '## Group boundaries and sample sizes', '', '| Variable | Q | Lower | Upper | n | Median ELI |', '|---|---:|---:|---:|---:|---:|']
    for row in summary.itertuples():
        lines.append(f'| {row.driver} | {row.quintile} | {row.lower_edge:.6g} | {row.upper_edge:.6g} | {row.n} | {row.eli_median:+.4f} |')
    return '\n'.join(lines) + '\n'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'derived_data/q3_city_driver_master.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'figure_outputs_global_city_atlas')
    parser.add_argument('--basemap', type=Path, help='Local polygon dataset; no automatic downloads.')
    parser.add_argument('--sample-mode', choices=('variable', 'common'), default='variable')
    parser.add_argument('--dpi', choices=(300, 600), type=int, default=300)
    parser.add_argument('--formats', nargs='+', choices=('png', 'pdf', 'svg', 'tiff'), default=['png', 'pdf', 'svg', 'tiff'])
    parser.add_argument('--prepare-only', action='store_true', help='Audit and export source tables without rendering.')
    parser.add_argument('--significance', type=Path, default=ROOT / 'figure_outputs_global_city_atlas/significance/spatial_bootstrap_primary_500km.csv')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atlas, exclusions, audit = load_data(args.input)
    groups, summary, source = make_groups(atlas, args.sample_mode)
    significance = load_significance(args.significance, source, groups)
    audit['significance'] = {'method': 'exploratory spatial-cluster bootstrap, 500 projected km', 'p_values': significance, 'source_sha256': hashlib.sha256(resource(args.significance).read_bytes()).hexdigest()}
    audit.update({'sample_mode': args.sample_mode, 'dpi': args.dpi, 'size_mm': [WIDTH_MM, HEIGHT_MM], 'panel_counts': {d.title: len(groups[d.column][0]) for d in DRIVERS}})
    audit['map_color_limits'] = {column: [float(map_scale(atlas, column).vmin), float(map_scale(atlas, column).vmax)] for column in ['eli50', 'positive_mean50', 'negative_mean50', *MEAN_COLUMNS]}
    audit['map_color_saturation_counts'] = {column: {'below': int(atlas[column].lt(lower).sum()), 'above': int(atlas[column].gt(upper).sum())} for column, (lower, upper) in MAP_LIMITS.items()}
    source.to_csv(args.output_dir / 'source_driver_quintiles.csv', index=False, encoding='utf-8-sig')
    atlas[['id_1', 'lon', 'lat', 'ns_group', *MEAN_COLUMNS, 'eli50', 'positive_mean50', 'negative_mean50', *ACCESS_COLUMNS, 'access_nonmotor_six_mean_min']].to_csv(args.output_dir / 'source_city_maps.csv', index=False, encoding='utf-8-sig')
    summary.to_csv(args.output_dir / 'quintile_summary.csv', index=False, encoding='utf-8-sig')
    exclusions.to_csv(args.output_dir / 'exclusions.csv', index=False, encoding='utf-8-sig')
    (args.output_dir / 'methods_and_caption.md').write_text(methods_text(audit, summary, args.sample_mode), encoding='utf-8')
    if not args.prepare_only:
        setup_style()
        fig, panels = build_figure(atlas, groups, get_basemap(args.basemap), significance)
        audit['layout'] = audit_layout(fig, panels)
        export_box = bottom_crop_box(fig)
        audit['export_size_mm'] = [export_box.width * 25.4, export_box.height * 25.4]
        audit['bottom_crop_mm'] = export_box.y0 * 25.4
        preview = args.output_dir / f'{STEM}_preview.png'
        fig.savefig(preview, dpi=180, bbox_inches=export_box, pad_inches=0)
        with Image.open(resource(preview)) as img:
            img.convert('L').save(args.output_dir / f'{STEM}_grayscale.png')
        problems = audit['layout']
        if problems['text_outside_canvas'] or problems['text_overlaps'] or (not problems['grid_aligned']) or problems['render_warnings']:
            (args.output_dir / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
            plt.close(fig)
            raise RuntimeError('Layout audit failed; inspect preview and audit.json before exporting.')
        for fmt in args.formats:
            kwargs = {'pil_kwargs': {'compression': 'tiff_lzw'}} if fmt == 'tiff' else {}
            fig.savefig(args.output_dir / f'{STEM}.{fmt}', dpi=args.dpi, bbox_inches=export_box, pad_inches=0, **kwargs)
        plt.close(fig)
    (args.output_dir / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))
    print(f'Output: {args.output_dir}')

def _build_manuscript_figure():
    atlas, exclusions, audit = load_data(PACKAGE_ROOT / "data/shared/Study_city_analysis.csv")
    groups, summary, source = make_groups(atlas, "variable")
    significance = load_significance(resource(PACKAGE_ROOT / "figure_outputs_global_city_atlas/significance/spatial_bootstrap_primary_500km.csv"), source, groups)
    setup_style()
    figure, panels = build_figure(atlas, groups, get_basemap(PACKAGE_ROOT / "data/shared/World_land_polygons.geojson"), significance)
    return figure, bottom_crop_box(figure)


if __name__ == "__main__":
    _configure_land_functions(sys.modules[__name__])
    _figure, _bbox = _build_manuscript_figure()
    _save_figure(_figure, _bbox, 'Fig_2', _figure_options)
