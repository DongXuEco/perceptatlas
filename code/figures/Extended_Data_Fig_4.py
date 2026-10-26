"""Plot Extended_Data_Fig_4 using the packaged numerical tables or supplied local inputs."""
from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_support"))
from figure_cli import prepare_figure as _prepare_figure, save_figure as _save_figure
from figure_cli import read_table as _read_table, model_tables as _model_tables
from figure_cli import configure_land_functions as _configure_land_functions

_figure_options = _prepare_figure('Extended_Data_Fig_4', 'analysis_tables') if __name__ == "__main__" else None

from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
import os
from pathlib import Path
os.environ.setdefault('WINDIR', 'external_inputs/Windows')
PROJECT_ROOT = PACKAGE_ROOT
MPLCONFIG_DIR = output_dir('Extended_Data_Fig_4') / '_mplconfig'
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(MPLCONFIG_DIR))
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
OUT_DIR = output_dir('Extended_Data_Fig_4')
SUMMARY_CSV = OUT_DIR / 'q3_driver_context_contrast_group_summary.csv'
EFFECTS_CSV = OUT_DIR / 'q3_driver_context_contrast_effects.csv'
FIG_BASENAME = OUT_DIR / 'fig_q3_driver_context_contrast_value_summary'
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'], 'font.weight': 'regular', 'axes.titleweight': 'regular', 'axes.labelweight': 'regular', 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'axes.unicode_minus': False, 'text.color': '#000000', 'axes.labelcolor': '#000000', 'xtick.color': '#000000', 'ytick.color': '#000000', 'legend.labelcolor': '#000000'})
FAMILY_COLORS = {'Green/blue': '#62C0D4', 'Env. risk': '#E5170A', 'Built env.': '#357DBD', 'Econ./pop.': '#6E789D', 'POI supply': '#F28817', 'Non-motor': '#F1B47A', 'Road network': '#F0CFA1', 'Terrain': '#909BB6', 'Location': '#8170A8', 'Other': '#C8D0C1'}
CONTRASTS = ['North_minus_South_city', 'Core_minus_Expansion_zone_paired', 'NS_modification_of_Core_minus_Expansion']
CONTRAST_LABELS = {'North_minus_South_city': 'North - South', 'Core_minus_Expansion_zone_paired': 'Core - expansion', 'NS_modification_of_Core_minus_Expansion': 'NS modification'}
CONTRAST_SHORT = {'North_minus_South_city': 'NS', 'Core_minus_Expansion_zone_paired': 'C-E', 'NS_modification_of_Core_minus_Expansion': 'Mod.'}
DIVERGE = LinearSegmentedColormap.from_list('driver_value_diverge', ['#F27C2B', '#FFEFCC', '#FFFFFF', '#DCE6F2', '#0D47A1'])
BLUE_INTENSITY = LinearSegmentedColormap.from_list('blue_intensity', ['#F5F8FC', '#DCE6F2', '#A7CFF2', '#3B82D6'])

def fmt_signed(value: float, digits: int=2) -> str:
    if pd.isna(value):
        return 'NA'
    return f'{value:+.{digits}f}'

def fmt_plain(value: float, digits: int=2) -> str:
    if pd.isna(value):
        return 'NA'
    return f'{value:.{digits}f}'

def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not resource(SUMMARY_CSV).exists():
        raise FileNotFoundError(f'Missing summary table: {SUMMARY_CSV}')
    if not resource(EFFECTS_CSV).exists():
        raise FileNotFoundError(f'Missing effects table: {EFFECTS_CSV}')
    summary = pd.read_csv(resource(SUMMARY_CSV))
    effects = pd.read_csv(resource(EFFECTS_CSV))
    drivers = summary[['driver', 'driver_short', 'driver_label', 'driver_order', 'family_short', 'family_color', 'family_order', 'unit']].drop_duplicates('driver').sort_values(['family_order', 'driver_order']).reset_index(drop=True)
    drivers['family_color'] = drivers['family_short'].map(FAMILY_COLORS).fillna(drivers['family_color'])
    return (summary, effects, drivers)

def panel_title(ax: mpl.axes.Axes, letter: str, title: str, subtitle: str | None=None) -> None:
    ax.text(0.0325, 1.02, letter, transform=ax.transAxes, ha='right', va='bottom', fontsize=12, fontweight='bold', color='#000000')
    ax.text(0.0875, 1.02, title, transform=ax.transAxes, ha='left', va='bottom', fontsize=10, color='#000000')
    if subtitle:
        ax.text(0.0875, 0.98, subtitle, transform=ax.transAxes, ha='left', va='top', fontsize=7.2, color='#000000')

def draw_header_rule(ax: mpl.axes.Axes, y: float) -> None:
    ax.plot([0.0, 1.0], [y, y], transform=ax.transAxes, color='#D7DBE2', lw=0.7, clip_on=False)

def draw_family_chip(ax: mpl.axes.Axes, x: float, y: float, color: str, size: float=0.021) -> None:
    ax.add_patch(Rectangle((x, y - size / 2), size, size, transform=ax.transAxes, facecolor=color, edgecolor=color, lw=0.0, clip_on=False))

def group_z_lookup(summary: pd.DataFrame, analysis_scale: str, grouping: str) -> pd.DataFrame:
    data = summary[(summary['analysis_scale'] == analysis_scale) & (summary['grouping'] == grouping)]
    return data.pivot_table(index='driver', columns='group', values='z_median', aggfunc='first')

def merged_contrast_table(summary: pd.DataFrame, effects: pd.DataFrame, drivers: pd.DataFrame, contrast: str, value_a: str, value_b: str, group_table: pd.DataFrame, top_n: int=10) -> pd.DataFrame:
    eff = effects[effects['contrast'] == contrast].copy()
    table = drivers.merge(eff, on=['driver', 'driver_short', 'driver_label', 'driver_order', 'family_short', 'family_color', 'family_order', 'unit'])
    table['value_a'] = table['driver'].map(group_table[value_a])
    table['value_b'] = table['driver'].map(group_table[value_b])
    table['abs_effect'] = table['effect_z_median'].abs()
    return table.sort_values('abs_effect', ascending=False).head(top_n).reset_index(drop=True)

def forest_scale(values: pd.DataFrame) -> tuple[float, float]:
    lo = float(values['effect_z_ci_low'].min())
    hi = float(values['effect_z_ci_high'].max())
    lim = max(abs(lo), abs(hi), 0.75)
    lim = float(np.ceil(lim * 2.0) / 2.0)
    return (-lim, lim)

def x_to_axes(value: float, lo: float, hi: float, x0: float, x1: float) -> float:
    return x0 + (value - lo) / (hi - lo) * (x1 - x0)

def draw_value_forest_table(ax: mpl.axes.Axes, table: pd.DataFrame, letter: str, title: str, subtitle: str, a_label: str, b_label: str, delta_label: str) -> None:
    ax.set_axis_off()
    panel_title(ax, letter, title, subtitle)
    header_y = 0.88
    draw_header_rule(ax, header_y - 0.035)
    col = {'driver': 0.058, 'a': 0.292, 'b': 0.395, 'delta': 0.522, 'forest0': 0.745, 'forest1': 0.982}
    if letter == 'c':
        col['a'] = 0.288
        col['b'] = 0.414
    ax.text(col['driver'], header_y, 'Driver', transform=ax.transAxes, fontsize=7.4, ha='left', va='center')
    ax.text(col['a'], header_y, a_label, transform=ax.transAxes, fontsize=7.4, ha='center', va='center')
    ax.text(col['b'], header_y, b_label, transform=ax.transAxes, fontsize=7.4, ha='center', va='center')
    ax.text(col['delta'], header_y, delta_label, transform=ax.transAxes, fontsize=7.4, ha='left', va='center')
    ax.text((col['forest0'] + col['forest1']) / 2, header_y, '95% CI', transform=ax.transAxes, fontsize=7.4, ha='center', va='center')
    lo, hi = forest_scale(table)
    zero_x = x_to_axes(0.0, lo, hi, col['forest0'], col['forest1'])
    ax.plot([zero_x, zero_x], [0.12, 0.84], transform=ax.transAxes, color='#000000', lw=0.65, ls=(0, (3, 2)))
    rows_y = np.linspace(0.785, 0.145, len(table))
    for idx, (row, y) in enumerate(zip(table.itertuples(), rows_y)):
        if idx % 2 == 0:
            ax.add_patch(Rectangle((0.0, y - 0.03), 1.0, 0.058, transform=ax.transAxes, facecolor='#F5F8FC', edgecolor='none', zorder=0))
        draw_family_chip(ax, 0.012, y, row.family_color, size=0.02)
        ax.text(col['driver'], y, row.driver_short, transform=ax.transAxes, fontsize=7.25, ha='left', va='center')
        ax.text(col['a'], y, fmt_plain(row.value_a), transform=ax.transAxes, fontsize=7.25, ha='center', va='center')
        ax.text(col['b'], y, fmt_plain(row.value_b), transform=ax.transAxes, fontsize=7.25, ha='center', va='center')
        ax.text(col['delta'], y, f'{fmt_signed(row.effect_z_median)} [{fmt_signed(row.effect_z_ci_low)}, {fmt_signed(row.effect_z_ci_high)}]', transform=ax.transAxes, fontsize=6.85, ha='left', va='center')
        x_l = x_to_axes(row.effect_z_ci_low, lo, hi, col['forest0'], col['forest1'])
        x_h = x_to_axes(row.effect_z_ci_high, lo, hi, col['forest0'], col['forest1'])
        x_m = x_to_axes(row.effect_z_median, lo, hi, col['forest0'], col['forest1'])
        ax.plot([x_l, x_h], [y, y], transform=ax.transAxes, color='#4B5563', lw=0.75, solid_capstyle='round')
        ax.scatter([x_m], [y], transform=ax.transAxes, s=18, facecolor=row.family_color if bool(row.ci_excludes_zero_z) else '#FFFFFF', edgecolor=row.family_color, linewidth=0.75, zorder=3)
    axis_y = 0.108
    ax.plot([col['forest0'], col['forest1']], [axis_y, axis_y], transform=ax.transAxes, color='#000000', lw=0.65, solid_capstyle='butt', clip_on=False)
    for tick in [lo, 0, hi]:
        tx = x_to_axes(tick, lo, hi, col['forest0'], col['forest1'])
        ax.plot([tx, tx], [axis_y, axis_y - 0.01], transform=ax.transAxes, color='#000000', lw=0.6)
        ax.text(tx, 0.082, fmt_plain(tick, 1), transform=ax.transAxes, ha='center', va='top', fontsize=6.7)

def build_zone_gap_table(summary: pd.DataFrame, effects: pd.DataFrame, drivers: pd.DataFrame, top_n: int=10) -> pd.DataFrame:
    context = group_z_lookup(summary, 'city_zone', 'north_south_x_core_expansion')
    gap = pd.DataFrame(index=context.index)
    gap['North gap'] = context['North_core'] - context['North_expansion']
    gap['South gap'] = context['South_core'] - context['South_expansion']
    eff = effects[effects['contrast'] == 'NS_modification_of_Core_minus_Expansion'].copy()
    table = drivers.merge(eff, on=['driver', 'driver_short', 'driver_label', 'driver_order', 'family_short', 'family_color', 'family_order', 'unit'])
    table['value_a'] = table['driver'].map(gap['North gap'])
    table['value_b'] = table['driver'].map(gap['South gap'])
    table['abs_effect'] = table['effect_z_median'].abs()
    return table.sort_values('abs_effect', ascending=False).head(top_n).reset_index(drop=True)

def draw_context_value_matrix(ax: mpl.axes.Axes, summary: pd.DataFrame, effects: pd.DataFrame, drivers: pd.DataFrame) -> None:
    ax.set_axis_off()
    panel_title(ax, 'd', 'Four-context median values', 'Cells show median robust z values for the most context-variant drivers.')
    context = group_z_lookup(summary, 'city_zone', 'north_south_x_core_expansion')
    context = context[['North_core', 'North_expansion', 'South_core', 'South_expansion']]
    rank = context.abs().max(axis=1).sort_values(ascending=False).head(12).index
    table = drivers[drivers['driver'].isin(rank)].copy()
    table['rank_value'] = table['driver'].map(context.abs().max(axis=1))
    table = table.sort_values('rank_value', ascending=False).reset_index(drop=True)
    x_driver = 0.06
    x0 = 0.375
    cell_w = 0.142
    cell_h = 0.048
    header_y = 0.875
    labels = ['North\ncore', 'North\nexp.', 'South\ncore', 'South\nexp.']
    draw_header_rule(ax, header_y - 0.037)
    ax.text(x_driver, header_y, 'Driver', transform=ax.transAxes, fontsize=7.4, ha='left', va='center')
    for j, label in enumerate(labels):
        ax.text(x0 + j * cell_w + cell_w / 2, header_y, label, transform=ax.transAxes, fontsize=7.0, ha='center', va='center')
    values = context.loc[table['driver']].to_numpy(dtype=float)
    vmax = max(1.5, np.nanquantile(np.abs(values), 0.96))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    row_y = np.linspace(0.79, 0.12, len(table))
    for i, (row, y) in enumerate(zip(table.itertuples(), row_y)):
        if i % 2 == 0:
            ax.add_patch(Rectangle((0.0, y - 0.03), 1.0, 0.058, transform=ax.transAxes, facecolor='#F5F8FC', edgecolor='none'))
        draw_family_chip(ax, 0.012, y, row.family_color, size=0.02)
        ax.text(x_driver, y, row.driver_short, transform=ax.transAxes, fontsize=7.15, ha='left', va='center')
        for j, group in enumerate(context.columns):
            val = float(context.loc[row.driver, group])
            face = DIVERGE(norm(val))
            ax.add_patch(Rectangle((x0 + j * cell_w + 0.006, y - cell_h / 2), cell_w - 0.012, cell_h, transform=ax.transAxes, facecolor=face, edgecolor='#FFFFFF', lw=0.55))
            ax.text(x0 + j * cell_w + cell_w / 2, y, fmt_signed(val), transform=ax.transAxes, fontsize=6.95, ha='center', va='center', color='#FFFFFF' if norm(val) >= 0.95 else '#000000')
    cax = ax.inset_axes([0.385, 0.04, 0.515, 0.027])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=DIVERGE, norm=norm, orientation='horizontal', ticks=[-vmax, 0, vmax])
    cb.outline.set_linewidth(0.65)
    cb.ax.tick_params(labelsize=6.6, length=2, width=0.55, pad=1)
    cb.set_label('Median driver condition (z)', fontsize=6.9, labelpad=1.0)

def draw_family_magnitude_table(ax: mpl.axes.Axes, effects: pd.DataFrame, drivers: pd.DataFrame) -> None:
    ax.set_axis_off()
    panel_title(ax, 'e', 'Family-level effect magnitudes', 'Median absolute standardized contrast across drivers within each family.')
    ordered_families = drivers[['family_short', 'family_color', 'family_order']].drop_duplicates('family_short').sort_values('family_order').reset_index(drop=True)
    spec = effects.assign(abs_effect=lambda d: d['effect_z_median'].abs()).groupby(['family_short', 'contrast'], as_index=False).agg(median_abs=('abs_effect', 'median'), q75_abs=('abs_effect', lambda x: float(np.quantile(x, 0.75))))
    vmax = max(1.0, float(spec['q75_abs'].max()))
    header_y = 0.875
    draw_header_rule(ax, header_y - 0.037)
    ax.text(0.06, header_y, 'Family', transform=ax.transAxes, fontsize=7.4, ha='left', va='center')
    xs = [0.43, 0.635, 0.84]
    for x, contrast in zip(xs, CONTRASTS):
        ax.text(x, header_y, CONTRAST_SHORT[contrast], transform=ax.transAxes, fontsize=7.4, ha='center', va='center')
    y_positions = np.linspace(0.79, 0.14, len(ordered_families))
    for i, (row, y) in enumerate(zip(ordered_families.itertuples(), y_positions)):
        if i % 2 == 0:
            ax.add_patch(Rectangle((0.0, y - 0.032), 1.0, 0.06, transform=ax.transAxes, facecolor='#F5F8FC', edgecolor='none'))
        draw_family_chip(ax, 0.012, y, row.family_color, size=0.02)
        ax.text(0.06, y, row.family_short, transform=ax.transAxes, fontsize=7.2, ha='left', va='center')
        for x, contrast in zip(xs, CONTRASTS):
            value = spec[(spec['family_short'] == row.family_short) & (spec['contrast'] == contrast)]['median_abs']
            val = float(value.iloc[0]) if len(value) else np.nan
            bar_w = 0.108 * min(val / vmax, 1.0)
            ax.add_patch(Rectangle((x - 0.055, y - 0.01), 0.11, 0.02, transform=ax.transAxes, facecolor='#EFF3F8', edgecolor='none'))
            ax.add_patch(Rectangle((x - 0.055, y - 0.01), bar_w, 0.02, transform=ax.transAxes, facecolor=row.family_color, edgecolor='none', alpha=0.85))
            ax.text(x + 0.067, y, fmt_plain(val), transform=ax.transAxes, fontsize=7.05, ha='left', va='center')
    ax.text(0.06, 0.065, 'Values are median |Delta z|.', transform=ax.transAxes, fontsize=6.8, ha='left', va='center')

def draw_significant_count_table(ax: mpl.axes.Axes, effects: pd.DataFrame, drivers: pd.DataFrame) -> None:
    ax.set_axis_off()
    panel_title(ax, 'f', 'High-confidence driver counts', 'Cells report drivers whose 95% CI excludes zero within each family.')
    family_sizes = drivers.groupby('family_short')['driver'].nunique()
    ordered_families = drivers[['family_short', 'family_color', 'family_order']].drop_duplicates('family_short').sort_values('family_order').reset_index(drop=True)
    counts = effects.groupby(['family_short', 'contrast'])['ci_excludes_zero_z'].sum().reset_index(name='count')
    header_y = 0.875
    draw_header_rule(ax, header_y - 0.037)
    ax.text(0.06, header_y, 'Family', transform=ax.transAxes, fontsize=7.4, ha='left', va='center')
    xs = [0.43, 0.635, 0.84]
    for x, contrast in zip(xs, CONTRASTS):
        ax.text(x, header_y, CONTRAST_SHORT[contrast], transform=ax.transAxes, fontsize=7.4, ha='center', va='center')
    y_positions = np.linspace(0.79, 0.14, len(ordered_families))
    for i, (row, y) in enumerate(zip(ordered_families.itertuples(), y_positions)):
        if i % 2 == 0:
            ax.add_patch(Rectangle((0.0, y - 0.032), 1.0, 0.06, transform=ax.transAxes, facecolor='#F5F8FC', edgecolor='none'))
        draw_family_chip(ax, 0.012, y, row.family_color, size=0.02)
        n = int(family_sizes[row.family_short])
        ax.text(0.06, y, f'{row.family_short} ({n})', transform=ax.transAxes, fontsize=7.2, ha='left', va='center')
        for x, contrast in zip(xs, CONTRASTS):
            sel = counts[(counts['family_short'] == row.family_short) & (counts['contrast'] == contrast)]['count']
            k = int(sel.iloc[0]) if len(sel) else 0
            frac = k / n if n else 0.0
            ax.add_patch(Rectangle((x - 0.056, y - 0.023), 0.112, 0.046, transform=ax.transAxes, facecolor=BLUE_INTENSITY(frac), edgecolor='#FFFFFF', lw=0.55))
            ax.text(x, y, f'{k}/{n}', transform=ax.transAxes, fontsize=7.15, ha='center', va='center')
    ax.text(0.06, 0.065, 'Filled cells summarize robust directional differences.', transform=ax.transAxes, fontsize=6.8, ha='left', va='center')

def build_figure() -> mpl.figure.Figure:
    summary, effects, drivers = load_inputs()
    ns = group_z_lookup(summary, 'city', 'north_south')
    ce = group_z_lookup(summary, 'city_zone', 'core_expansion')
    table_a = merged_contrast_table(summary, effects, drivers, 'North_minus_South_city', 'North', 'South', ns, top_n=10)
    table_b = merged_contrast_table(summary, effects, drivers, 'Core_minus_Expansion_zone_paired', 'core', 'expansion', ce, top_n=10)
    table_c = build_zone_gap_table(summary, effects, drivers, top_n=10)
    height_scale = 1.06
    base_height = 8.9
    panel_height_total = base_height * (0.955 - 0.07) * 3 / (3 + 2 * 0.24)
    figure_height = base_height + panel_height_total * (height_scale - 1)
    fig = plt.figure(figsize=(10.6, figure_height), facecolor='white')
    gs = fig.add_gridspec(3, 2, left=0.055, right=0.985, top=1 - base_height * (1 - 0.955) / figure_height, bottom=base_height * 0.07 / figure_height, wspace=0.13, hspace=0.24 / height_scale, height_ratios=[1.05, 1.05, 1.0])
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
    draw_value_forest_table(axes[0], table_a, 'a', 'North-South strongest contrasts', 'Top drivers ranked by |North - South| robust-z median contrast.', 'North z', 'South z', 'Delta z [95% CI]')
    draw_value_forest_table(axes[1], table_b, 'b', 'Core-expansion strongest contrasts', 'Top drivers ranked by paired |core - expansion| contrast.', 'Core z', 'Exp. z', 'Delta z [95% CI]')
    draw_value_forest_table(axes[2], table_c, 'c', 'North-South modification of zone gaps', 'Top drivers ranked by interaction in core-expansion gap.', 'North gap', 'South gap', 'Delta gap [95% CI]')
    draw_context_value_matrix(axes[3], summary, effects, drivers)
    draw_family_magnitude_table(axes[4], effects, drivers)
    draw_significant_count_table(axes[5], effects, drivers)
    return fig

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(f'{FIG_BASENAME}.{ext}', dpi=600, bbox_inches='tight', pad_inches=0.025, facecolor='white')
    fig.savefig(f'{FIG_BASENAME}.tiff', dpi=600, bbox_inches='tight', pad_inches=0.025, facecolor='white', pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    print(f'Saved: {FIG_BASENAME}.png')
    print(f'Saved: {FIG_BASENAME}.tiff')
    print(f'Saved: {FIG_BASENAME}.pdf')
    print(f'Saved: {FIG_BASENAME}.svg')

def _build_manuscript_figure():
    return build_figure(), None


if __name__ == "__main__":
    _configure_land_functions(sys.modules[__name__])
    _figure, _bbox = _build_manuscript_figure()
    _save_figure(_figure, _bbox, 'Extended_Data_Fig_4', _figure_options)
