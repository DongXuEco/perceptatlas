"""Plot Fig_5 using the packaged numerical tables or supplied local inputs."""
from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_support"))
from figure_cli import prepare_figure as _prepare_figure, save_figure as _save_figure
from figure_cli import read_table as _read_table, model_tables as _model_tables
from figure_cli import configure_land_functions as _configure_land_functions

_figure_options = _prepare_figure('Fig_5', 'analysis_tables') if __name__ == "__main__" else None

from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
import math
from copy import copy
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Circle, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.transforms import ScaledTranslation, offset_copy
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['font.weight'] = 'regular'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['axes.titleweight'] = 'regular'
plt.rcParams['axes.labelweight'] = 'regular'
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['legend.frameon'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['text.color'] = '#000000'
plt.rcParams['axes.labelcolor'] = '#000000'
plt.rcParams['xtick.color'] = '#000000'
plt.rcParams['ytick.color'] = '#000000'
plt.rcParams['legend.labelcolor'] = '#000000'
PROJECT_ROOT = PACKAGE_ROOT
RUN_FORMAL = PROJECT_ROOT / 'runs' / '20260818-q3-final-mechanism-models'
DERIVED = PROJECT_ROOT / 'derived_data'
OUT_DIR = output_dir('Fig_5')
OUT_DIR.mkdir(parents=True, exist_ok=True)
FONT_SIZES = {'panel_label': 10.0, 'panel_title': 9.0, 'section_title': 7.5, 'axis_title': 7.5, 'body': 7.0, 'table_header': 7.0, 'data': 6.5, 'tick': 6.5, 'legend': 6.5}
PALETTE = {'pale_apricot': '#FFE7CC', 'soft_orange': '#FFB66D', 'burnt_orange': '#F27C2B', 'warm_sand': '#E7C9A0', 'cool_mist': '#DCE6F2', 'light_sky': '#A7CFF2', 'cerulean': '#3B82D6', 'cobalt': '#0D47A1', 'teal': '#4D807D', 'green': '#0E5E36', 'red': '#C83F26', 'rose': '#CF7C65', 'gold': '#DDBD19', 'steel': '#7F8AA9', 'violet': '#6C5CA8', 'text': '#000000', 'muted': '#000000', 'axis': '#000000', 'grid': '#E5E7EB', 'light_grid': '#F1F3F6'}
FAMILY_COLORS = {'POI supply': '#F28817', 'Motorized accessibility': '#EBAD8A', 'Non-motorized accessibility': '#EBAD8A', 'Road network': '#C8D0C1', 'Green/blue space': '#64C4DA', 'Built environment': '#3177B8', 'Economic/population': '#586990', 'Topographic control': '#C8D0C1', 'Terrain control': '#C8D0C1', 'Location control': '#7B72A2', 'Environmental risk': '#D91809', 'Other/minor drivers': '#C8D0C1', 'Other': PALETTE['muted']}
GROUP_COLORS = {'Global': PALETTE['axis'], 'North': PALETTE['cobalt'], 'South': PALETTE['burnt_orange']}
BCD_GROUP_COLORS = {'global': '#C8D0C1', 'North': '#3B82D6', 'South': '#F28817'}
BCD_OUTCOME_COLORS = {'compound_gini50': '#F28817', 'compound_theil50': '#3B82D6', 'compound_p9010_50': '#C8D0C1'}
OUTCOME_LABELS = {'eli50': 'ELI', 'compound_gini50': 'Gini', 'compound_theil50': 'Theil', 'compound_p9010_50': 'P90-P10', 'positive_score50': 'Positive', 'negative_score50': 'Negative'}
OUTCOME_COLORS = {'eli50': PALETTE['teal'], 'compound_gini50': PALETTE['burnt_orange'], 'compound_theil50': PALETTE['red'], 'compound_p9010_50': PALETTE['violet']}
ABBR_BY_COLUMN = {'green_fraction_mean': 'Green', 'water_fraction_mean': 'Water', 'building_density_mean': 'Built dens.', 'building_height_mean': 'Height', 'ntl_log1p_mean': 'NTL', 'population_density_mean': 'Pop. dens.', 'lat': 'Latitude', 'dem_mean': 'Elev.', 'slope_mean': 'Slope', 'pm25_2020_mean': 'PM2.5', 'lst_c_mean': 'LST', 'noise_mean': 'Noise', 'poi_education_mean': 'POI edu.', 'poi_entertainment_mean': 'POI ent.', 'poi_health_mean': 'POI health', 'poi_transport_mean': 'POI transp.', 'access_nonmotor_education_mean': 'Access edu.', 'access_nonmotor_entertainment_mean': 'Access ent.', 'access_nonmotor_health_mean': 'Access health', 'access_nonmotor_living_mean': 'Access living', 'access_nonmotor_transport_mean': 'Access transp.', 'road_total_mean': 'Road total', 'road_footway_mean': 'Footway', 'road_pedestrian_mean': 'Pedestrian', 'road_cycleway_mean': 'Cycleway'}
MECHANISM_OUTCOMES = ['eli50', 'compound_gini50', 'compound_theil50', 'compound_p9010_50']
INEQUALITY_OUTCOMES = ['compound_gini50', 'compound_theil50', 'compound_p9010_50']
CITY_GROUPS = ['global', 'North', 'South']
GROUP_DISPLAY = {'global': 'Global', 'North': 'North', 'South': 'South'}
SKILL_CMAP = LinearSegmentedColormap.from_list('q3_skill', ['#FFF7EC', PALETTE['pale_apricot'], PALETTE['soft_orange'], PALETTE['burnt_orange']])
DENSITY_CMAP = LinearSegmentedColormap.from_list('q3_density', ['#F6F8FB', PALETTE['cool_mist'], PALETTE['light_sky'], PALETTE['cerulean'], PALETTE['cobalt']])

def read_csv(path: Path) -> pd.DataFrame:
    if not resource(path).exists():
        raise FileNotFoundError(path)
    return pd.read_csv(resource(path))

def load_inputs() -> dict[str, pd.DataFrame]:
    response_uncertainty_path = RUN_FORMAL / 'q3_formal_response_uncertainty_bootstrap.csv'
    oof_dir = DERIVED / 'q3_city_oof_importance'
    return {'family': read_csv(oof_dir / 'family.csv'), 'shap': read_csv(RUN_FORMAL / 'q3_formal_shap.csv'), 'permutation': read_csv(oof_dir / 'permutation.csv'), 'ale': pd.concat([read_csv(RUN_FORMAL / 'q3_formal_ale.csv'), read_csv(oof_dir / 'response_ale_additions.csv')], ignore_index=True), 'pdp': pd.concat([read_csv(RUN_FORMAL / 'q3_formal_pdp.csv'), read_csv(oof_dir / 'response_pdp_additions.csv')], ignore_index=True), 'response_uncertainty': pd.concat([read_csv(response_uncertainty_path), read_csv(oof_dir / 'response_uncertainty_additions.csv')], ignore_index=True), 'spearman': read_csv(RUN_FORMAL / 'q3_formal_spearman.csv'), 'performance': read_csv(RUN_FORMAL / 'q3_formal_rf_cv_performance.csv'), 'city_master': read_csv(DERIVED / 'q3_city_driver_master.csv')}

def short_var(column: str) -> str:
    return ABBR_BY_COLUMN.get(str(column), str(column).replace('_mean', '').replace('_', ' ')[:18])

def short_family(label: str) -> str:
    text = str(label)
    text = text.replace('Environmental risk', 'Env. risk')
    text = text.replace('Built environment', 'Built env.')
    text = text.replace('Green/blue space', 'Green/blue')
    text = text.replace('Economic/population', 'Econ./pop.')
    text = text.replace('Non-motorized accessibility', 'Non-motor access')
    text = text.replace('Motorized accessibility', 'Motor access')
    text = text.replace('Location control', 'Location')
    text = text.replace('Terrain control', 'Terrain')
    text = text.replace('Other/minor drivers', 'Other/minor')
    return text

def outcome_label(outcome: str) -> str:
    return OUTCOME_LABELS.get(outcome, outcome)

def family_color(label: str) -> str:
    return FAMILY_COLORS.get(str(label), PALETTE['muted'])

def scale01(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if not np.isfinite(mn) or not np.isfinite(mx) or math.isclose(mx, mn):
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def despine(ax: mpl.axes.Axes, tick_size: float=5.7, label_size: float=6.2, tick_width: float=0.55, tick_length: float=2.4) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=tick_size, width=tick_width, length=tick_length, color=PALETTE['axis'])
    ax.xaxis.label.set_size(label_size)
    ax.yaxis.label.set_size(label_size)

def panel_title(ax: mpl.axes.Axes, letter: str, title: str, y: float=1.105, letter_x: float=-0.1, title_x: float=0.0) -> None:
    ax.text(letter_x, y, letter, transform=ax.transAxes, ha='left', va='bottom', fontsize=FONT_SIZES['panel_label'], fontweight='bold', color=PALETTE['text'], clip_on=False)
    ax.text(title_x, y, title, transform=ax.transAxes, ha='left', va='bottom', fontsize=FONT_SIZES['panel_title'], color=PALETTE['text'], clip_on=False)

def curved_link(ax: mpl.axes.Axes, p0: tuple[float, float], p1: tuple[float, float], color: str, lw: float, alpha: float=0.6, zorder: int=1) -> None:
    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    verts = [(x0, y0), (x0 + dx * 0.45, y0), (x1 - dx * 0.45, y1), (x1, y1)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    patch = PathPatch(MplPath(verts, codes), facecolor='none', edgecolor=color, lw=lw, alpha=alpha, capstyle='round', joinstyle='round', zorder=zorder, clip_on=False)
    ax.add_patch(patch)

def add_node(ax: mpl.axes.Axes, x: float, y: float, color: str, label: str, align: str='left') -> None:
    ax.add_patch(Circle((x, y), 0.012, facecolor=color, edgecolor='white', lw=0.55, zorder=4, clip_on=False))
    dx = 0.018 if align == 'left' else -0.018
    ax.text(x + dx, y, label, ha=align, va='center', fontsize=FONT_SIZES['body'], color=PALETTE['text'], zorder=5, clip_on=False)

def draw_panel_a(ax: mpl.axes.Axes, data: dict[str, pd.DataFrame], show_title: bool=True, y_shift: float=0.0, stage_x_shift: float=0.0) -> None:
    if show_title:
        panel_title(ax, 'a', 'Outcome-to-driver mechanism flow')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fam = data['family']
    perm = data['permutation']
    base = fam[(fam['table'] == 'city') & (fam['group'] == 'global') & fam['outcome'].isin(MECHANISM_OUTCOMES)].copy()
    base['within_share'] = base.groupby('outcome')['perm_importance_mean'].transform(lambda s: s / max(s.sum(), 1e-12))
    top_links = base.sort_values(['outcome', 'perm_importance_mean'], ascending=[True, False]).groupby('outcome').head(4)
    outcome_y = dict(zip(MECHANISM_OUTCOMES, np.linspace(0.82, 0.2, len(MECHANISM_OUTCOMES)) + y_shift))
    family_order = top_links.groupby('family_label')['within_share'].mean().sort_values(ascending=False).index.tolist()
    family_y = dict(zip(family_order, np.linspace(0.86, 0.16, len(family_order)) + y_shift))
    var_base = perm[(perm['table'] == 'city') & (perm['group'] == 'global') & perm['outcome'].isin(MECHANISM_OUTCOMES)].copy()
    var_top = var_base.sort_values(['outcome', 'perm_importance_mean'], ascending=[True, False]).groupby('outcome').head(2)
    var_top = var_top.sort_values('perm_importance_mean', ascending=False).drop_duplicates('column').head(8)
    var_order = var_top['column'].tolist()
    var_y = dict(zip(var_order, np.linspace(0.84, 0.18, len(var_order)) + y_shift))
    max_share = max(top_links['within_share'].max(), 1e-09)
    for _, row in top_links.iterrows():
        out = row['outcome']
        lab = row['family_label']
        color = family_color(lab)
        width = 0.9 + 7.8 * row['within_share'] / max_share
        curved_link(ax, (0.13 + stage_x_shift, outcome_y[out]), (0.46 + stage_x_shift, family_y[lab]), color, width, 0.6, 1)
    max_var = max(var_top['perm_importance_mean'].max(), 1e-09)
    for _, row in var_top.iterrows():
        lab = row['family_label']
        col = row['column']
        if lab not in family_y or col not in var_y:
            continue
        color = family_color(lab)
        width = 0.7 + 5.8 * row['perm_importance_mean'] / max_var
        curved_link(ax, (0.49 + stage_x_shift, family_y[lab]), (0.77 + stage_x_shift, var_y[col]), color, width, 0.6, 2)
    for outcome, y in outcome_y.items():
        add_node(ax, 0.12 + stage_x_shift, y, OUTCOME_COLORS.get(outcome, PALETTE['axis']), outcome_label(outcome), 'right')
    for lab in family_order:
        add_node(ax, 0.47 + stage_x_shift, family_y[lab], family_color(lab), short_family(lab), 'left')
    for _, row in var_top.iterrows():
        col = row['column']
        add_node(ax, 0.78 + stage_x_shift, var_y[col], family_color(row['family_label']), short_var(col), 'left')
    total_share = base.groupby('family_label')['within_share'].mean().reindex(family_order).fillna(0)
    x0 = 0.925
    for lab in family_order:
        y = family_y[lab]
        w = 0.055 * total_share[lab] / max(total_share.max(), 1e-09)
        ax.add_patch(Rectangle((x0, y - 0.012), w, 0.024, facecolor=family_color(lab), edgecolor='white', lw=0.35, clip_on=False))

def add_panel_a_column_labels(fig: plt.Figure, ax_a: mpl.axes.Axes, ax_b: mpl.axes.Axes, stage_x_shift: float=0.0) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_bbox = ax_b.xaxis.label.get_window_extent(renderer=renderer)
    target_y = fig.transFigure.inverted().transform((0, 0.5 * (label_bbox.y0 + label_bbox.y1)))[1]
    label_size = ax_b.xaxis.label.get_size()
    label_color = ax_b.xaxis.label.get_color()
    for x_data, label in [(0.078, 'outcome'), (0.443, 'family'), (0.726, 'top variable'), (0.92, 'mean share')]:
        if label != 'mean share':
            x_data += stage_x_shift
        x_fig = fig.transFigure.inverted().transform(ax_a.transData.transform((x_data, 0)))[0]
        fig.text(x_fig, target_y, label, ha='left', va='center', fontsize=label_size, color=label_color)

def draw_panel_b(ax: mpl.axes.Axes, data: dict[str, pd.DataFrame], show_title: bool=True) -> None:
    if show_title:
        panel_title(ax, 'b', 'City-level predictive skill')
    perf = data['performance']
    df = perf[(perf['table'] == 'city') & perf['outcome'].isin(MECHANISM_OUTCOMES) & perf['group'].isin(CITY_GROUPS)].copy()
    rows = list(MECHANISM_OUTCOMES)
    row_y = {out: y for out, y in zip(rows, np.linspace(0.805, 0.345, len(rows)))}
    offset = {'global': 0.0, 'North': 0.028, 'South': -0.028}
    marker_style = {'global': dict(marker='o', facecolor=BCD_GROUP_COLORS['global'], edgecolor=BCD_GROUP_COLORS['global'], lw=0.0), 'North': dict(marker='o', facecolor=BCD_GROUP_COLORS['North'], edgecolor=BCD_GROUP_COLORS['North'], lw=0.0), 'South': dict(marker='o', facecolor=BCD_GROUP_COLORS['South'], edgecolor=BCD_GROUP_COLORS['South'], lw=0.0)}
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    ax.set_xlabel('Cross-validated R2 (mean ± SD)', fontsize=5.95, color=PALETTE['axis'], labelpad=1.6)
    ax.xaxis.set_label_coords(0.415, 0.025)
    table_left, table_right = (0.018, 0.985)
    forest_left, forest_right = (0.195, 0.625)
    forest_min, forest_max = (0.5, 0.9)
    axis_y = 0.245
    header_y = 0.895

    def map_x(value: float) -> float:
        value = float(np.clip(value, forest_min, forest_max))
        return forest_left + (value - forest_min) / (forest_max - forest_min) * (forest_right - forest_left)
    for idx, outcome in enumerate(rows):
        y = row_y[outcome]
        if idx % 2 == 1:
            ax.add_patch(Rectangle((table_left, y - 0.052), table_right - table_left, 0.104, transform=ax.transAxes, facecolor='#E1EEFA', alpha=0.4, edgecolor='none', zorder=0))
    header_font = dict(fontsize=5.25, fontweight='bold', color=PALETTE['text'])
    body_font = dict(fontsize=5.4, color=PALETTE['text'])
    ax.text(0.03, header_y, 'Outcome', transform=ax.transAxes, ha='left', va='bottom', **header_font)
    right_cols = {'global': 0.705, 'North': 0.845, 'South': 0.973}
    for group, x in right_cols.items():
        ax.text(x, header_y, GROUP_DISPLAY[group], transform=ax.transAxes, ha='center', va='bottom', **header_font)
    for outcome in rows:
        y = row_y[outcome]
        sub = df[df['outcome'] == outcome]
        ax.text(0.03, y, outcome_label(outcome), transform=ax.transAxes, ha='left', va='center', **body_font)
        for group, x in right_cols.items():
            g = sub[sub['group'] == group]
            if g.empty:
                continue
            r = g.iloc[0]
            ax.text(x, y, f"{r['r2_mean']:.3f}\n({r['r2_sd']:.3f})", transform=ax.transAxes, ha='center', va='center', fontsize=4.66, linespacing=1.24, color=PALETTE['text'])
    ref_x = map_x(0.8)
    ax.plot([ref_x, ref_x], [axis_y, header_y - 0.018], color=PALETTE['axis'], lw=0.66, ls=(0, (3.2, 3.2)), transform=ax.transAxes, zorder=1)
    for _, row in df.iterrows():
        y = row_y[row['outcome']] + offset[row['group']]
        style = marker_style[row['group']]
        x = map_x(row['r2_mean'])
        xerr = row['r2_sd'] / (forest_max - forest_min) * (forest_right - forest_left)
        ax.plot([x - xerr, x + xerr], [y, y], color=PALETTE['axis'], lw=0.58, transform=ax.transAxes, zorder=3)
        ax.scatter([x], [y], marker=style['marker'], s=10.57, facecolors=style['facecolor'], edgecolors=style['edgecolor'], linewidths=style['lw'] * 1.2, transform=ax.transAxes, zorder=4)
    ax.plot([forest_left, forest_right], [axis_y, axis_y], color=PALETTE['axis'], lw=0.62, transform=ax.transAxes, zorder=5)
    for tick in [0.5, 0.6, 0.7, 0.8, 0.9]:
        x = map_x(tick)
        ax.plot([x, x], [axis_y, axis_y - 0.017], color=PALETTE['axis'], lw=0.46, transform=ax.transAxes, zorder=5)
        ax.text(x, axis_y - 0.043, f'{tick:.1f}', transform=ax.transAxes, ha='center', va='top', fontsize=5.4, color=PALETTE['text'])
    legend_x = forest_left
    for group, x in zip(['global', 'North', 'South'], [legend_x + 0.014, legend_x + 0.19, legend_x + 0.36]):
        y = 0.09
        style = marker_style[group]
        ax.scatter([x], [y], marker=style['marker'], s=8.05, facecolors=style['facecolor'], edgecolors=style['edgecolor'], linewidths=style['lw'] * 1.2, transform=ax.transAxes, zorder=5)
        ax.text(x + 0.02, y, GROUP_DISPLAY[group], transform=ax.transAxes, ha='left', va='center', fontsize=5.4, color=PALETTE['text'])

def variable_swarm_panel(ax: mpl.axes.Axes, data: dict[str, pd.DataFrame], letter: str, title: str, outcomes: list[str], groups: list[str], top_n: int=8, title_y: float=0.975, title_letter_x: float=-0.1, title_text_x: float=0.0, content_y_shift: float=0.0, value_col_positions: list[float] | None=None, label_x: float=0.018) -> None:
    panel_title(ax, letter, title, y=title_y, letter_x=title_letter_x, title_x=title_text_x)
    perm = data['permutation']
    shap = data['shap']
    base = perm[(perm['table'] == 'city') & perm['group'].isin(groups) & perm['outcome'].isin(outcomes)].copy()
    global_rank = perm[(perm['table'] == 'city') & (perm['group'] == 'global') & perm['outcome'].isin(outcomes)].groupby(['column', 'family_label'], as_index=False)['perm_importance_mean'].mean().sort_values('perm_importance_mean', ascending=False).head(top_n)
    cols = global_rank['column'].tolist()
    base = base[base['column'].isin(cols)].merge(shap[(shap['table'] == 'city') & shap['group'].isin(groups) & shap['outcome'].isin(outcomes)][['outcome', 'group', 'column', 'mean_shap']], on=['outcome', 'group', 'column'], how='left')
    if base.empty or not cols:
        ax.axis('off')
        return
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    forest_left, forest_right = (0.235, 0.635)
    axis_y = 0.06 + content_y_shift
    header_y = 0.925 + content_y_shift
    row_top, row_bottom = (0.85 + content_y_shift, 0.13 + content_y_shift)
    row_y = {col: y for col, y in zip(cols, np.linspace(row_top, row_bottom, len(cols)))}
    row_gap = (row_top - row_bottom) / max(len(cols) - 1, 1)
    row_h = min(0.074, row_gap * 0.72)
    extent = float((base['perm_importance_mean'] + base['perm_importance_sd'].fillna(0)).max())
    step = 0.2 if extent > 0.5 else 0.1
    forest_min = 0.0
    forest_max = max(step, math.ceil(extent / step) * step)
    ticks = list(np.arange(forest_min, forest_max + step * 0.25, step))
    if ticks[-1] < forest_max:
        ticks.append(forest_max)

    def map_x(value: float) -> float:
        value = float(np.clip(value, forest_min, forest_max))
        return forest_left + (value - forest_min) / (forest_max - forest_min) * (forest_right - forest_left)
    for idx, col in enumerate(cols):
        y = row_y[col]
        if idx % 2 == 1:
            ax.add_patch(Rectangle((0.0, y - row_h / 2), 1.0, row_h, transform=ax.transAxes, facecolor='#E1EEFA', alpha=0.4, edgecolor='none', zorder=0))
        ax.text(label_x, y, short_var(col), transform=ax.transAxes, ha='left', va='center', fontsize=FONT_SIZES['body'], color=PALETTE['text'])
    if len(groups) > 1:
        series_key = 'group'
        series_order = groups
        offsets = {'global': 0.0, 'North': 0.024, 'South': -0.024}

        def style_for(key: str) -> dict[str, object]:
            color = BCD_GROUP_COLORS[key]
            return dict(marker='o', facecolor=color, edgecolor=color, lw=0.0, label=GROUP_DISPLAY[key])
    else:
        series_key = 'outcome'
        series_order = outcomes
        offsets = dict(zip(series_order, np.linspace(0.024, -0.024, len(series_order))))

        def style_for(key: str) -> dict[str, object]:
            color = BCD_OUTCOME_COLORS.get(key, OUTCOME_COLORS.get(key, PALETTE['axis']))
            return dict(marker='o', facecolor=color, edgecolor=color, lw=0.0, label=outcome_label(key))
    if value_col_positions is None:
        value_col_positions = [0.72, 0.85, 0.965]
    right_cols = dict(zip(series_order, value_col_positions))
    header_font = dict(fontsize=FONT_SIZES['table_header'], fontweight='bold', color=PALETTE['text'])
    value_font = dict(fontsize=FONT_SIZES['data'], color=PALETTE['text'])
    ax.text(label_x, header_y, 'Variable', transform=ax.transAxes, ha='left', va='bottom', **header_font)
    for key, x_col in right_cols.items():
        style = style_for(key)
        label = str(style['label'])
        ax.text(x_col, header_y, label, transform=ax.transAxes, ha='center', va='bottom', **header_font)
        marker_transform = offset_copy(ax.transAxes, fig=ax.figure, x=0.0, y=-5.2, units='points')
        ax.scatter([x_col], [header_y + 0.005], marker=style['marker'], s=8.05, facecolors=style['facecolor'], edgecolors=style['edgecolor'], linewidths=float(style['lw']) * 1.2, transform=marker_transform, clip_on=False, zorder=5)
    for col in cols:
        y = row_y[col]
        for key, x_col in right_cols.items():
            sub = base[(base['column'] == col) & (base[series_key] == key)]
            if sub.empty:
                continue
            row = sub.iloc[0]
            ax.text(x_col, y, f"{row['perm_importance_mean']:.3f}\n({row['perm_importance_sd']:.3f})", transform=ax.transAxes, ha='center', va='center', linespacing=1.24, **value_font)
    for _, row in base.iterrows():
        key = row[series_key]
        if key not in series_order:
            continue
        y = row_y[row['column']] + offsets.get(key, 0.0)
        style = style_for(key)
        x = map_x(row['perm_importance_mean'])
        xerr = row['perm_importance_sd'] / (forest_max - forest_min) * (forest_right - forest_left)
        ax.plot([max(forest_left, x - xerr), min(forest_right, x + xerr)], [y, y], color=style['edgecolor'], lw=0.56, transform=ax.transAxes, zorder=3)
        ax.scatter([x], [y], marker=style['marker'], s=9.24, facecolors=style['facecolor'], edgecolors=style['edgecolor'], linewidths=float(style['lw']) * 1.2, transform=ax.transAxes, zorder=4)
    ax.plot([forest_left, forest_right], [axis_y, axis_y], color=PALETTE['axis'], lw=0.62, transform=ax.transAxes, zorder=5)
    for tick in ticks:
        x = map_x(tick)
        ax.plot([x, x], [axis_y, axis_y - 0.017], color=PALETTE['axis'], lw=0.46, transform=ax.transAxes, zorder=5)
        ax.text(x, axis_y - 0.022, f'{tick:.1f}', transform=ax.transAxes, ha='center', va='top', fontsize=FONT_SIZES['tick'], color=PALETTE['text'])
    ax.text((forest_left + forest_right) / 2, axis_y - 0.1, 'OOF permutation importance (decrease in $\\mathrm{R}^{\\mathrm{2}}$)', math_fontfamily='custom', transform=ax.transAxes, ha='center', va='top', fontsize=FONT_SIZES['axis_title'], color=PALETTE['axis'], clip_on=False)

def family_dumbbell_panel(ax: mpl.axes.Axes, data: dict[str, pd.DataFrame], letter: str, title: str, outcomes: list[str], top_n: int=8) -> None:
    panel_title(ax, letter, title)
    fam = data['family']
    df = fam[(fam['table'] == 'city') & fam['group'].isin(['North', 'South']) & fam['outcome'].isin(outcomes)].copy()
    df['share'] = df.groupby(['outcome', 'group'])['perm_importance_mean'].transform(lambda s: s / max(s.sum(), 1e-12))
    avg = df.groupby(['group', 'family_label'], as_index=False)['share'].mean()
    wide = avg.pivot(index='family_label', columns='group', values='share').fillna(0)
    wide['mean_share'] = wide[['North', 'South']].mean(axis=1)
    wide['delta'] = wide['North'] - wide['South']
    wide = wide.sort_values('mean_share', ascending=False).head(top_n).sort_values('delta')
    labels = wide.index.tolist()
    y = np.arange(len(labels))
    for i, lab in enumerate(labels):
        color = family_color(lab)
        ax.hlines(i, wide.loc[lab, 'South'] * 100, wide.loc[lab, 'North'] * 100, color=PALETTE['grid'], lw=1.6, zorder=1)
        ax.scatter(wide.loc[lab, 'South'] * 100, i, s=20, color=GROUP_COLORS['South'], edgecolor='white', lw=0.45, zorder=3)
        ax.scatter(wide.loc[lab, 'North'] * 100, i, s=20, color=GROUP_COLORS['North'], edgecolor='white', lw=0.45, zorder=3)
        ax.plot([wide.loc[lab, 'mean_share'] * 100], [i], marker='|', color=color, markersize=7, mew=1.2, zorder=4)
        ax.axhspan(i - 0.42, i + 0.42, color=PALETTE['light_grid'], alpha=0.5, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([short_family(v) for v in labels], fontsize=5.45)
    ax.set_xlabel('Family share (%)')
    xmax = max(wide[['North', 'South']].max().max() * 100 * 1.16, 12)
    ax.set_xlim(0, xmax)
    ax.grid(axis='x', color=PALETTE['grid'], lw=0.45)
    despine(ax)
    stack_ax = ax.inset_axes([0.12, 0.84, 0.78, 0.15])
    stack_ax.set_xlim(0, 1)
    stack_ax.set_ylim(0, 2)
    stack_ax.axis('off')
    for j, group in enumerate(['North', 'South']):
        shares = avg[avg['group'] == group].copy()
        shares = shares[shares['family_label'].isin(labels)].set_index('family_label')['share'].reindex(labels).fillna(0)
        total = max(shares.sum(), 1e-12)
        start = 0
        for lab, val in shares.items():
            w = val / total
            stack_ax.add_patch(Rectangle((start, 1.22 - j * 0.82), w, 0.36, facecolor=family_color(lab), edgecolor='white', lw=0.28))
            start += w
        stack_ax.text(-0.02, 1.4 - j * 0.82, group, ha='right', va='center', fontsize=4.8, color=PALETTE['muted'])
    handles = [mpl.lines.Line2D([0], [0], marker='o', color='none', markerfacecolor=GROUP_COLORS['North'], markersize=3.8, label='North'), mpl.lines.Line2D([0], [0], marker='o', color='none', markerfacecolor=GROUP_COLORS['South'], markersize=3.8, label='South')]
    ax.legend(handles=handles, loc='lower right', bbox_to_anchor=(1.02, -0.02), fontsize=5.1, ncol=2, columnspacing=0.7, handletextpad=0.2)

def family_composition_shares(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    df['family_label'] = df['family_label'].replace({'Terrain control': 'Other/minor drivers', 'Road network': 'Other/minor drivers'})
    df = df.groupby(['outcome', 'family_label'], as_index=False)['perm_importance_mean'].sum()
    df['within_outcome_share'] = df.groupby('outcome')['perm_importance_mean'].transform(lambda s: s / max(float(s.sum()), 1e-12))
    shares = df.pivot(index='outcome', columns='family_label', values='within_outcome_share').fillna(0).mean()
    assert np.isclose(shares.sum(), 1.0)
    return shares

def family_stacked_composition_panel(ax: mpl.axes.Axes, data: dict[str, pd.DataFrame], letter: str, title: str, title_letter_x: float=-0.1, title_text_x: float=0.0, title_y: float=1.105) -> None:
    panel_title(ax, letter, title, y=title_y, letter_x=title_letter_x, title_x=title_text_x)
    fam = data['family']
    row_specs = [('ELI', 'North', ['eli50']), ('ELI', 'South', ['eli50']), ('Inequality', 'North', INEQUALITY_OUTCOMES), ('Inequality', 'South', INEQUALITY_OUTCOMES)]
    rows: list[dict[str, object]] = []
    for outcome_group, group, outcomes in row_specs:
        df = fam[(fam['table'] == 'city') & (fam['group'] == group) & fam['outcome'].isin(outcomes)].copy()
        if df.empty:
            continue
        share = family_composition_shares(df)
        rows.append({'outcome_group': outcome_group, 'group': group, 'share': share})
    if not rows:
        ax.text(0.5, 0.5, 'Family output missing', ha='center', va='center', fontsize=6, color=PALETTE['muted'])
        ax.axis('off')
        return
    family_order = pd.concat([pd.Series(row['share'], name=f"{row['outcome_group']}_{row['group']}") for row in rows], axis=1).fillna(0).mean(axis=1).sort_values(ascending=False).index.tolist()
    y_positions = np.array([3.15, 2.35, 1.05, 0.25])
    row_labels = ['ELI  North', 'ELI  South', 'Ineq. North', 'Ineq. South']
    bar_h = 0.46
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.3, 3.7)
    row_segments: dict[tuple[str, str], dict[str, dict[str, float | str]]] = {}
    for y, row in zip(y_positions, rows):
        left = 0.0
        segments: dict[str, dict[str, float | str]] = {}
        for lab in family_order:
            value = float(row['share'].get(lab, 0.0)) * 100.0
            if value <= 0:
                continue
            segments[lab] = {'x0': left, 'x1': left + value, 'y': float(y), 'color': family_color(lab)}
            left += value
        row_segments[str(row['outcome_group']), str(row['group'])] = segments

    def add_pair_ribbons(upper_key: tuple[str, str], lower_key: tuple[str, str]) -> None:
        upper = row_segments.get(upper_key, {})
        lower = row_segments.get(lower_key, {})
        if not upper or not lower:
            return
        for lab in family_order:
            if lab not in upper or lab not in lower:
                continue
            u = upper[lab]
            l = lower[lab]
            ux0, ux1 = (float(u['x0']), float(u['x1']))
            lx0, lx1 = (float(l['x0']), float(l['x1']))
            if min(ux1 - ux0, lx1 - lx0) < 0.8:
                continue
            uy = float(u['y']) - bar_h / 2
            ly = float(l['y']) + bar_h / 2
            patch = mpl.patches.Polygon([(ux0, uy), (ux1, uy), (lx1, ly), (lx0, ly)], closed=True, facecolor=str(u['color']), edgecolor='none', alpha=0.13, zorder=0.7)
            ax.add_patch(patch)
    add_pair_ribbons(('ELI', 'North'), ('ELI', 'South'))
    add_pair_ribbons(('Inequality', 'North'), ('Inequality', 'South'))
    for y, row in zip(y_positions, rows):
        left = 0.0
        for lab in family_order:
            value = float(row['share'].get(lab, 0.0)) * 100.0
            if value <= 0:
                continue
            ax.barh(y, value, left=left, height=bar_h, color=family_color(lab), edgecolor='white', linewidth=0.35, zorder=2)
            if value >= 11.5:
                label_text = short_family(lab).replace('Non-motor access', 'Non-motor').replace('Econ./pop.', 'Econ.')
                if value < 16.0 and '/' in label_text:
                    label_text = label_text.replace('/', '/\n', 1)
                ax.text(left + value / 2, y, label_text, ha='center', va='center', fontsize=FONT_SIZES['data'], linespacing=0.9, color=PALETTE['text'], clip_on=True, zorder=3)
            left += value
    ax.set_yticks(y_positions)
    ax.set_yticklabels(row_labels, fontsize=FONT_SIZES['tick'], color=PALETTE['text'])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel('Family share (%)', fontsize=FONT_SIZES['axis_title'], color=PALETTE['axis'], labelpad=1.0)
    ax.tick_params(axis='x', labelsize=FONT_SIZES['tick'], width=0.5, length=2.8, pad=1.8, colors=PALETTE['text'])
    ax.tick_params(axis='y', width=0, length=0, pad=2.0)
    ax.grid(False)
    ax.text(100.0, 3.58, '100%', ha='right', va='center', fontsize=FONT_SIZES['data'], color=PALETTE['muted'])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color(PALETTE['axis'])
    ax.spines['bottom'].set_linewidth(0.62)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def percentile_reference(city: pd.DataFrame, outcome: str, column: str) -> pd.Series:
    values = pd.to_numeric(city.loc[pd.to_numeric(city[outcome], errors='coerce').notna(), column], errors='coerce')
    values = values.replace([np.inf, -np.inf], np.nan)
    return values.fillna(values.median())

def empirical_percentiles(reference: pd.Series, values: np.ndarray) -> np.ndarray:
    knots, counts = np.unique(reference.to_numpy(dtype=float), return_counts=True)
    midranks = (np.cumsum(counts) - 0.5 * counts) / counts.sum()
    return np.interp(values, knots, midranks, left=0.0, right=1.0)

def bootstrap_raw_grid(city: pd.DataFrame, outcome: str, column: str, normalized_grid: np.ndarray) -> np.ndarray:
    values = pd.to_numeric(city.loc[pd.to_numeric(city[outcome], errors='coerce').notna(), column], errors='coerce')
    values = values.replace([np.inf, -np.inf], np.nan)
    if len(values) > 900:
        values = values.iloc[np.random.default_rng(42).choice(len(values), size=900, replace=False)]
    values = values.fillna(values.median())
    edges = np.unique(np.quantile(values, np.linspace(0, 1, 21)))
    centers = (edges[:-1] + edges[1:]) / 2
    assert len(centers) == len(normalized_grid)
    assert np.allclose(scale01(centers), normalized_grid)
    return centers

def response_curve_block(ax: mpl.axes.Axes, data: dict[str, pd.DataFrame], letter: str, title: str, outcome: str, columns: list[str], right_outcome: str | None=None, right_columns: list[str] | None=None, title_y: float=1.105, title_letter_x: float=-0.1, title_text_x: float=0.0) -> None:
    panel_title(ax, letter, title, y=title_y, letter_x=title_letter_x, title_x=title_text_x)
    ax.axis('off')
    ale = data['ale']
    pdp = data['pdp']
    unc = data.get('response_uncertainty', pd.DataFrame())
    paired = right_outcome is not None and right_columns is not None
    response_height_scale = 1.0
    if paired:
        group_width = 1.0 / 2.22
        right_start = 1.0 - group_width
        inner_gap = 0.055
        sub_w = (group_width - 2 * inner_gap) / 3
        ax.text(group_width / 2 - 0.02, 1.0 - 0.045 / response_height_scale, 'Global city-level ELI responses', ha='center', va='center', fontsize=FONT_SIZES['section_title'], color=PALETTE['text'])
        ax.text(right_start + group_width / 2 + 0.02, 1.0 - 0.045 / response_height_scale, 'Global city-level Gini responses', ha='center', va='center', fontsize=FONT_SIZES['section_title'], color=PALETTE['text'])
        panel_specs = []
        for group_index, (current_outcome, selected) in enumerate([(outcome, columns), (right_outcome, right_columns)]):
            assert len(selected) == 6
            for index, col in enumerate(selected):
                row = index // 3
                start = right_start if group_index else 0.0
                pos = (start + index % 3 * (sub_w + inner_gap), 0.015 + (1 - row) * 0.49)
                panel_specs.append((current_outcome, col, group_index * 6 + index, pos, True))
        sub_h = 0.345 / response_height_scale
        ax.text(0.5, -0.11, 'Driver percentile', ha='center', va='top', fontsize=FONT_SIZES['axis_title'], color=PALETTE['text'], transform=ax.transAxes)
    else:
        positions = [(0.02, 0.53), (0.52, 0.53), (0.02, 0.08), (0.52, 0.08)]
        panel_specs = [(outcome, col, idx, positions[idx], idx in [0, 2]) for idx, col in enumerate(columns[:4])]
        sub_w, sub_h = (0.43, 0.34)
    ax.legend(handles=[mpl.lines.Line2D([], [], color=PALETTE['cobalt'], lw=1.15, label='ALE'), mpl.lines.Line2D([], [], color=PALETTE['burnt_orange'], lw=0.9, ls='--', label='Centered PDP'), mpl.patches.Patch(facecolor=PALETTE['cobalt'], alpha=0.14, label='95% bootstrap envelope')], loc='center', bbox_to_anchor=(0.5, 1.0 - 0.045 / response_height_scale), ncol=3, frameon=False, fontsize=FONT_SIZES['legend'], handlelength=1.4, columnspacing=0.7)
    for current_outcome, col, idx, pos, show_y_ticks in panel_specs:
        sub_ax = ax.inset_axes([pos[0], pos[1], sub_w, sub_h])
        a = ale[(ale['table'] == 'city') & (ale['group'] == 'global') & (ale['outcome'] == current_outcome) & (ale['column'] == col)].copy()
        p = pdp[(pdp['table'] == 'city') & (pdp['group'] == 'global') & (pdp['outcome'] == current_outcome) & (pdp['column'] == col)].copy()
        if a.empty:
            sub_ax.text(0.5, 0.5, 'missing', ha='center', va='center', fontsize=5.5, color=PALETTE['muted'])
            sub_ax.axis('off')
            continue
        reference = percentile_reference(data['city_master'], current_outcome, col)
        a['x_norm'] = empirical_percentiles(reference, a['x'].to_numpy())
        a = a.sort_values('x_norm')
        if not unc.empty:
            u = unc[(unc['table'] == 'city') & (unc['group'] == 'global') & (unc['outcome'] == current_outcome) & (unc['column'] == col) & (unc['effect_type'] == 'ale')].copy()
            if not u.empty:
                u = u.sort_values('x_norm')
                raw_grid = bootstrap_raw_grid(data['city_master'], current_outcome, col, u['x_norm'].to_numpy())
                sub_ax.fill_between(empirical_percentiles(reference, raw_grid), u['lo95'].to_numpy(dtype=float), u['hi95'].to_numpy(dtype=float), color=PALETTE['cobalt'], alpha=0.14, linewidth=0, zorder=1)
        sub_ax.plot(a['x_norm'], a['ale'], color=PALETTE['cobalt'], lw=1.15, marker='o', ms=1.75, mec='white', mew=0.22)
        sub_ax.axhline(0, color=PALETTE['axis'], lw=0.55, ls='--', alpha=0.7)
        if not p.empty:
            p['x_norm'] = empirical_percentiles(reference, p['x'].to_numpy())
            p = p.sort_values('x_norm')
            p['pdp_center'] = p['partial_dependence'] - p['partial_dependence'].mean()
            sub_ax.plot(p['x_norm'], p['pdp_center'], color=PALETTE['burnt_orange'], lw=0.9, ls='--', alpha=0.85)
        sub_ax.set_title(short_var(col), fontsize=FONT_SIZES['body'], pad=1.25, color=PALETTE['text'])
        sub_ax.set_xlim(0, 1)
        sub_ax.xaxis.set_major_locator(mpl.ticker.FixedLocator([0, 0.5, 1]) if paired else mpl.ticker.MultipleLocator(0.25))
        sub_ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1, decimals=0))
        sub_ax.grid(False)
        if idx in ([3, 4, 5, 9, 10, 11] if paired else [2, 3]):
            if not paired:
                sub_ax.set_xlabel('Driver percentile', fontsize=4.9, labelpad=0.8)
        else:
            sub_ax.set_xticklabels([])
        if show_y_ticks:
            sub_ax.set_ylabel('')
        else:
            sub_ax.set_yticklabels([])
        despine(sub_ax, tick_size=FONT_SIZES['tick'], label_size=FONT_SIZES['axis_title'], tick_width=0.5 if paired else 0.38, tick_length=2.8 if paired else 1.65)
        if paired:
            for spine in ['left', 'bottom']:
                sub_ax.spines[spine].set_color(PALETTE['axis'])
                sub_ax.spines[spine].set_linewidth(0.62)

def interaction_contour_panel(ax: mpl.axes.Axes, data: dict[str, pd.DataFrame], letter: str, title: str, title_letter_x: float=-0.1, title_text_x: float=0.0, title_y: float=1.105) -> None:
    panel_title(ax, letter, title, y=title_y, letter_x=title_letter_x, title_x=title_text_x)
    city = data['city_master'].copy()
    x_col = 'green_fraction_mean'
    y_col = 'lst_c_mean'
    z_col = 'eli50'
    df = city[[x_col, y_col, z_col]].apply(pd.to_numeric, errors='coerce').dropna()
    if len(df) < 80:
        ax.text(0.5, 0.5, 'Interaction data missing', ha='center', va='center', fontsize=5.8, color=PALETTE['muted'])
        ax.axis('off')
        return
    df['green_pct'] = df[x_col].rank(pct=True)
    df['lst_pct'] = df[y_col].rank(pct=True)
    x = df['green_pct'].to_numpy()
    y = df['lst_pct'].to_numpy()
    z = df[z_col].to_numpy()
    bins = 58
    edges = np.linspace(0, 1, bins + 1)
    count, _, _ = np.histogram2d(x, y, bins=[edges, edges])
    weighted, _, _ = np.histogram2d(x, y, bins=[edges, edges], weights=z)
    count_s = gaussian_filter(count, sigma=1.45)
    weighted_s = gaussian_filter(weighted, sigma=1.45)
    with np.errstate(divide='ignore', invalid='ignore'):
        response = weighted_s / count_s
    response = response.T
    density = count_s.T
    centers = (edges[:-1] + edges[1:]) / 2
    xx, yy = np.meshgrid(centers, centers)
    positive_density = density[np.isfinite(density) & (density > 0)]
    mask_level = np.nanpercentile(positive_density, 18) if positive_density.size else 0
    response = np.ma.array(response, mask=(density <= mask_level) | ~np.isfinite(response))
    lim = max(0.08, min(0.24, float(np.nanpercentile(np.abs(z), 92))))
    response_cmap = LinearSegmentedColormap.from_list('eli_interaction_response', [PALETTE['red'], '#F3C8B3', '#FFFFFF', '#C7DAD8', PALETTE['teal']])
    filled = ax.contourf(xx, yy, response, levels=np.linspace(-lim, lim, 13), cmap=response_cmap, norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim), extend='both', alpha=0.76, zorder=0)
    sample = df.sample(min(len(df), 3800), random_state=42)
    ax.scatter(sample['green_pct'], sample['lst_pct'], s=2.2, facecolor='#CBD5E1', edgecolor='none', alpha=0.24, zorder=1)
    if positive_density.size:
        levels = np.unique(np.nanquantile(positive_density, np.linspace(0.5, 0.965, 11)))
        if len(levels) > 2:
            line_colors = mpl.cm.Greys(np.linspace(0.42, 0.92, len(levels)))
            ax.contour(xx, yy, density, levels=levels, colors=line_colors, linewidths=np.linspace(0.34, 0.78, len(levels)), zorder=3)
    ax.plot([0, 1], [1, 0], color=PALETTE['axis'], lw=0.52, ls=(0, (4, 3)), alpha=0.72, zorder=2)
    high = df[df[z_col] >= df[z_col].quantile(0.95)]
    low = df[df[z_col] <= df[z_col].quantile(0.05)]
    for sub, color in [(low, PALETTE['red']), (high, PALETTE['teal'])]:
        pts = sub.sample(min(len(sub), 95), random_state=13)
        ax.scatter(pts['green_pct'], pts['lst_pct'], s=6.5, facecolor=color, edgecolor='white', lw=0.18, alpha=0.78, zorder=4)
    for sub, label, color, xytext in [(low, 'low ELI', PALETTE['red'], (0.18, 0.83)), (high, 'high ELI', PALETTE['teal'], (0.72, 0.18))]:
        mx = float(sub['green_pct'].median())
        my = float(sub['lst_pct'].median())
        ax.scatter([mx], [my], s=22, facecolor=color, edgecolor='white', lw=0.35, zorder=5)
        ax.annotate(label, xy=(mx, my), xytext=xytext, textcoords='data', ha='center', va='center', fontsize=5.0, color=PALETTE['text'], arrowprops=dict(arrowstyle='-', color=color, lw=0.38, shrinkA=1.5, shrinkB=1.8), zorder=6)
    cax = ax.inset_axes([0.925, 0.18, 0.02, 0.48])
    cb = plt.colorbar(filled, cax=cax, orientation='vertical')
    cb.set_ticks([-lim, 0, lim])
    cb.set_ticklabels([f'{-lim:.2f}', '0', f'{lim:.2f}'])
    cb.ax.tick_params(labelsize=4.4, width=0.32, length=1.4, pad=0.8)
    cb.outline.set_linewidth(0.32)
    cb.set_label('Smoothed ELI', fontsize=4.7, labelpad=1.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Green percentile', fontsize=5.7, color=PALETTE['axis'], labelpad=1.1)
    ax.set_ylabel('LST percentile', fontsize=5.7, color=PALETTE['axis'], labelpad=1.0)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(axis='both', labelsize=5.2, width=0.55, length=2.3, pad=1.4, colors=PALETTE['text'])
    ax.grid(False)
    ax.spines['left'].set_color(PALETTE['axis'])
    ax.spines['bottom'].set_color(PALETTE['axis'])
    ax.spines['left'].set_linewidth(0.72)
    ax.spines['bottom'].set_linewidth(0.72)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def eli_mechanism_bubble_heatmap_panel(ax: mpl.axes.Axes, data: dict[str, pd.DataFrame], letter: str, title: str, title_letter_x: float=-0.1, title_text_x: float=0.0, title_y: float=1.105) -> None:
    panel_title(ax, letter, title, y=title_y, letter_x=title_letter_x, title_x=title_text_x)
    perm = data['permutation']
    sp = data['spearman']
    groups = ['global', 'North', 'South']
    group_labels = [GROUP_DISPLAY[g] for g in groups]
    top_cols = perm[(perm['table'] == 'city') & (perm['group'] == 'global') & (perm['outcome'] == 'eli50')].sort_values('perm_importance_mean', ascending=False)['column'].head(8).tolist()
    base = perm[(perm['table'] == 'city') & (perm['outcome'] == 'eli50') & perm['group'].isin(groups) & perm['column'].isin(top_cols)][['group', 'column', 'family_label', 'perm_importance_mean']].copy()
    rho = sp[(sp['table'] == 'city') & (sp['outcome'] == 'eli50') & sp['group'].isin(groups) & sp['column'].isin(top_cols)][['group', 'column', 'spearman_rho']].copy()
    df = base.merge(rho, on=['group', 'column'], how='left')
    if df.empty:
        ax.text(0.5, 0.5, 'Driver output missing', ha='center', va='center', fontsize=5.8, color=PALETTE['muted'])
        ax.axis('off')
        return
    col_to_y = {col: len(top_cols) - 1 - i for i, col in enumerate(top_cols)}
    group_to_x = {group: i for i, group in enumerate(groups)}
    df['x'] = df['group'].map(group_to_x)
    df['y'] = df['column'].map(col_to_y)
    max_imp = max(float(df['perm_importance_mean'].max()), 1e-12)
    df['size'] = 11.0 + 78.0 * np.sqrt(df['perm_importance_mean'] / max_imp)
    ax.set_xlim(-0.92, 3.55)
    ax.set_ylim(-0.72, len(top_cols) - 0.1)
    for i, col in enumerate(top_cols):
        y = col_to_y[col]
        if i % 2 == 1:
            ax.axhspan(y - 0.4, y + 0.4, color=PALETTE['light_grid'], alpha=0.78, zorder=0)
        fam = df[df['column'] == col]['family_label'].dropna()
        fam_label = fam.iloc[0] if not fam.empty else 'Other'
        ax.add_patch(Rectangle((-0.86, y - 0.22), 0.055, 0.44, facecolor=family_color(fam_label), edgecolor='none', alpha=0.95, zorder=2))
    for x in range(len(groups)):
        ax.axvline(x, color=PALETTE['grid'], lw=0.35, zorder=0)
    rho_cmap = LinearSegmentedColormap.from_list('rho_teal_orange', [PALETTE['red'], '#F3C8B3', '#FFFFFF', '#BDD6D4', PALETTE['teal']])
    sc = ax.scatter(df['x'], df['y'], s=df['size'], c=df['spearman_rho'], cmap=rho_cmap, vmin=-0.65, vmax=0.65, edgecolor='white', linewidth=0.35, alpha=0.96, zorder=3)
    for _, row in df.iterrows():
        if float(row['perm_importance_mean']) >= max_imp * 0.3:
            ax.text(row['x'], row['y'], f"{row['spearman_rho']:+.2f}", ha='center', va='center', fontsize=3.75, color=PALETTE['text'], zorder=4)
    ax.set_yticks([col_to_y[col] for col in top_cols])
    ax.set_yticklabels([short_var(col) for col in top_cols], fontsize=5.1, color=PALETTE['text'])
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([])
    ax.tick_params(axis='x', width=0, length=0, labeltop=False, labelbottom=False)
    ax.tick_params(axis='y', width=0, length=0, pad=1.5)
    for x, label in enumerate(group_labels):
        ax.text(x, len(top_cols) - 0.34, label, ha='center', va='center', fontsize=5.45, fontweight='bold', color=PALETTE['text'], zorder=5)
    cax = ax.inset_axes([0.765, 0.17, 0.023, 0.48])
    cb = plt.colorbar(sc, cax=cax, orientation='vertical')
    cb.set_ticks([-0.6, 0, 0.6])
    cb.ax.tick_params(labelsize=4.15, width=0.32, length=1.4, pad=0.8)
    cb.outline.set_linewidth(0.32)
    cb.set_label('rho', fontsize=4.55, labelpad=0.8)
    legend_x = 3.03
    legend_y = [5.95, 4.95, 3.95]
    size_vals = [0.05, 0.2, 0.4]
    ax.text(legend_x - 0.13, 6.72, 'Importance', ha='left', va='center', fontsize=4.9, color=PALETTE['muted'])
    for y, val in zip(legend_y, size_vals):
        size = 11.0 + 78.0 * np.sqrt(min(val / max_imp, 1.0))
        ax.scatter([legend_x], [y], s=size, facecolor='#D7DEE8', edgecolor=PALETTE['axis'], lw=0.35, zorder=3)
        ax.text(legend_x + 0.22, y, f'{val:.2f}', ha='left', va='center', fontsize=4.4, color=PALETTE['muted'])
    ax.text(2.98, 0.42, 'size: RF importance\ncolor: Spearman rho', ha='left', va='bottom', fontsize=4.35, color=PALETTE['muted'], linespacing=1.15)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def joint_density_panel(ax: mpl.axes.Axes, data: dict[str, pd.DataFrame], show_title: bool=True) -> None:
    if show_title:
        panel_title(ax, 'i', 'Top-driver association cloud')
    ax.axis('off')
    city = data['city_master'].copy()
    x_col = 'lst_c_mean'
    y_col = 'eli50'
    df = city[[x_col, y_col]].dropna().copy()
    if len(df) > 6000:
        df = df.sample(6000, random_state=42)
    main = ax.inset_axes([0.12, 0.11, 0.64, 0.66])
    top = ax.inset_axes([0.12, 0.81, 0.64, 0.15], sharex=main)
    right = ax.inset_axes([0.79, 0.11, 0.16, 0.66], sharey=main)
    hb = main.hexbin(df[x_col], df[y_col], gridsize=36, mincnt=3, cmap=DENSITY_CMAP, linewidths=0.0)
    if len(df) > 3:
        slope, intercept = np.polyfit(df[x_col], df[y_col], 1)
        xs = np.linspace(df[x_col].quantile(0.01), df[x_col].quantile(0.99), 100)
        main.plot(xs, slope * xs + intercept, color=PALETTE['burnt_orange'], lw=1.15)
        r, p = pearsonr(df[x_col], df[y_col])
        main.text(0.04, 0.95, f'r = {r:.2f}\nP < 0.001' if p < 0.001 else f'r = {r:.2f}\nP = {p:.3f}', transform=main.transAxes, ha='left', va='top', fontsize=5.4, color=PALETTE['text'])
    main.set_xlabel('LST')
    main.set_ylabel('ELI')
    main.grid(color=PALETTE['grid'], lw=0.35)
    despine(main)
    top.hist(df[x_col], bins=32, color=PALETTE['light_sky'], edgecolor='white', lw=0.25)
    top.axis('off')
    right.hist(df[y_col], bins=32, orientation='horizontal', color=PALETTE['pale_apricot'], edgecolor='white', lw=0.25)
    right.axis('off')
    cax = ax.inset_axes([0.8, 0.84, 0.14, 0.032])
    cb = plt.colorbar(hb, cax=cax, orientation='horizontal')
    cb.ax.tick_params(labelsize=4.5, width=0.35, length=1.4, pad=0.5)
    cb.outline.set_linewidth(0.35)
    cb.set_label('count', fontsize=4.8, labelpad=0.8)

def concordance_panel(ax: mpl.axes.Axes, data: dict[str, pd.DataFrame]) -> None:
    panel_title(ax, 'j', 'Spearman-ML importance concordance')
    ax.axis('off')
    perm = data['permutation']
    sp = data['spearman']
    p = perm[(perm['table'] == 'city') & (perm['group'] == 'global') & perm['outcome'].isin(MECHANISM_OUTCOMES)].copy()
    p['ml_importance'] = p.groupby('outcome')['perm_importance_mean'].transform(lambda s: s / max(s.max(), 1e-12))
    s = sp[(sp['table'] == 'city') & (sp['group'] == 'global') & sp['outcome'].isin(MECHANISM_OUTCOMES)][['outcome', 'column', 'spearman_rho', 'family_label']].copy()
    df = p.merge(s, on=['outcome', 'column', 'family_label'], how='inner')
    df['abs_rho'] = df['spearman_rho'].abs()
    main = ax.inset_axes([0.13, 0.11, 0.62, 0.66])
    top = ax.inset_axes([0.13, 0.81, 0.62, 0.15], sharex=main)
    right = ax.inset_axes([0.79, 0.11, 0.16, 0.66], sharey=main)
    for outcome in MECHANISM_OUTCOMES:
        sub = df[df['outcome'] == outcome]
        main.scatter(sub['abs_rho'], sub['ml_importance'], s=15, facecolor=OUTCOME_COLORS.get(outcome, PALETTE['steel']), edgecolor='white', lw=0.35, alpha=0.86, label=outcome_label(outcome), zorder=3)
    main.plot([0, 1], [0, 1], color=PALETTE['axis'], lw=0.6, ls='--', alpha=0.65)
    if len(df) > 3:
        r, pval = spearmanr(df['abs_rho'], df['ml_importance'], nan_policy='omit')
        main.text(0.05, 0.95, f'rho = {r:.2f}', transform=main.transAxes, ha='left', va='top', fontsize=5.4, color=PALETTE['text'])
    main.set_xlim(0, max(0.75, df['abs_rho'].max() * 1.1))
    main.set_ylim(0, 1.08)
    main.set_xlabel('|Spearman rho|')
    main.set_ylabel('Normalized ML importance')
    main.grid(color=PALETTE['grid'], lw=0.35)
    despine(main)
    top.hist(df['abs_rho'], bins=18, color=PALETTE['cool_mist'], edgecolor='white', lw=0.25)
    top.axis('off')
    right.hist(df['ml_importance'], bins=18, orientation='horizontal', color=PALETTE['pale_apricot'], edgecolor='white', lw=0.25)
    right.axis('off')
    main.legend(loc='lower right', fontsize=4.8, ncol=1, handletextpad=0.2, borderpad=0.1, labelspacing=0.25)

def top_response_columns(data: dict[str, pd.DataFrame], outcome: str, n: int=6) -> list[str]:
    perm = data['permutation']
    available = set(data['ale'][(data['ale']['table'] == 'city') & (data['ale']['group'] == 'global') & (data['ale']['outcome'] == outcome)]['column'].unique())
    cols = perm[(perm['table'] == 'city') & (perm['group'] == 'global') & (perm['outcome'] == outcome)].sort_values('perm_importance_mean', ascending=False)['column'].head(n).tolist()
    assert len(cols) == n and set(cols).issubset(available)
    return cols

def build_figure() -> plt.Figure:
    data = load_inputs()
    fig = plt.figure(figsize=(7.8, 7.8), facecolor='white')
    gs = fig.add_gridspec(3, 2, left=0.065, right=0.985, top=0.95, bottom=0.06, wspace=0.22, hspace=0.24, height_ratios=[1.1, 1.65, 1.05])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, :])
    axes = [ax_a, ax_b, ax_c, ax_d, ax_e]
    positions = [ax.get_position().frozen() for ax in axes]
    base_height = fig.get_figheight()
    row_gap_extra = 0.4
    previous_flow_height_scale = 1.1 * 1.06
    flow_height_scale = previous_flow_height_scale * 1.03
    flow_stage_shift = -0.05
    middle_row_shift = 0.15
    response_height_scale = 1.2
    response_extra = positions[4].height * base_height * (response_height_scale - 1.0)
    new_height = base_height + row_gap_extra + response_extra
    fig.set_size_inches(fig.get_figwidth(), new_height)
    for index, (ax, pos) in enumerate(zip(axes, positions)):
        height = pos.height * base_height
        bottom = pos.y0 * base_height
        if index < 2:
            bottom += row_gap_extra + response_extra
            if index == 0:
                bottom -= height * (previous_flow_height_scale - 1.0)
                height *= flow_height_scale
        elif index < 4:
            bottom += response_extra + middle_row_shift
        else:
            height *= response_height_scale
        ax.set_position([pos.x0, bottom / new_height, pos.width, height / new_height])
    letter_x = -0.022 / ax_a.get_position().width
    title_y = 1.055
    draw_panel_a(ax_a, data, show_title=False, stage_x_shift=flow_stage_shift)
    flow_title_y = 1.0 + (title_y - 1.0 - (flow_height_scale - previous_flow_height_scale)) / flow_height_scale
    panel_title(ax_a, 'a', 'Outcome-to-driver mechanism flow', y=flow_title_y, letter_x=letter_x)
    family_stacked_composition_panel(ax_b, data, 'b', 'North-South family composition', title_letter_x=letter_x, title_y=title_y)
    variable_swarm_panel(ax_c, data, 'c', 'Top ELI variable contributions', ['eli50'], CITY_GROUPS, top_n=8, title_letter_x=letter_x, title_y=title_y)
    variable_swarm_panel(ax_d, data, 'd', 'Top inequality variable contributions', INEQUALITY_OUTCOMES, ['global'], top_n=8, title_letter_x=letter_x, title_y=title_y, value_col_positions=[0.705, 0.845, 0.965], label_x=0.03)
    response_curve_block(ax_e, data, 'e', 'Nonlinear response functions', 'eli50', top_response_columns(data, 'eli50', 6), right_outcome='compound_gini50', right_columns=top_response_columns(data, 'compound_gini50', 6), title_y=1.0 + (title_y - 1.0) / response_height_scale, title_letter_x=-0.022 / ax_e.get_position().width)
    for ax, letter, title, scale, shift in [(ax_a, 'a', 'Outcome-to-driver mechanism flow', 1.0, -2.0 / 25.4 / fig.get_figwidth()), (ax_c, 'c', 'Top ELI variable contributions', 1.05, 0.0), (ax_d, 'd', 'Top inequality variable contributions', 1.05, 0.0)]:
        pos = ax.get_position()
        new_width = pos.width * scale
        new_left = pos.x1 - new_width + shift
        headings = [text for text in ax.texts if text.get_text() in (letter, title)]
        heading_x = [pos.x0 + text.get_position()[0] * pos.width for text in headings]
        ax.set_position([new_left, pos.y0, new_width, pos.height])
        for text, x in zip(headings, heading_x):
            text.set_x((x - new_left) / new_width)
    add_panel_a_column_labels(fig, ax_a, ax_b, stage_x_shift=flow_stage_shift)
    heading_offset = ScaledTranslation(0, -1.0 / 25.4, fig.dpi_scale_trans)
    for ax, letter, title in [(ax_a, 'a', 'Outcome-to-driver mechanism flow'), (ax_b, 'b', 'North-South family composition'), (ax_c, 'c', 'Top ELI variable contributions'), (ax_d, 'd', 'Top inequality variable contributions')]:
        for text in list(ax.texts):
            if text.get_text() in {letter, title}:
                anchor = copy(text)
                anchor.set_alpha(0)
                ax.add_artist(anchor)
                text.set_transform(text.get_transform() + heading_offset)
    return fig

def save_outputs(fig: plt.Figure) -> None:
    basename = OUT_DIR / 'fig_q3_2_global_city_mechanisms_scidraw'
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(f'{basename}.{ext}', dpi=600, bbox_inches='tight', pad_inches=0.025)
    fig.savefig(f'{basename}.tiff', dpi=600, bbox_inches='tight', pad_inches=0.025, pil_kwargs={'compression': 'tiff_lzw'})

def main() -> None:
    fig = build_figure()
    save_outputs(fig)
    plt.close(fig)
    print(OUT_DIR / 'fig_q3_2_global_city_mechanisms_scidraw.png')

def _build_manuscript_figure():
    return build_figure(), None


if __name__ == "__main__":
    _configure_land_functions(sys.modules[__name__])
    _figure, _bbox = _build_manuscript_figure()
    _save_figure(_figure, _bbox, 'Fig_5', _figure_options)
