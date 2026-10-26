"""Plot Extended_Data_Fig_3 using the packaged numerical tables or supplied local inputs."""
from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_support"))
from figure_cli import prepare_figure as _prepare_figure, save_figure as _save_figure
from figure_cli import read_table as _read_table, model_tables as _model_tables
from figure_cli import configure_land_functions as _configure_land_functions

_figure_options = _prepare_figure('Extended_Data_Fig_3', 'analysis_tables') if __name__ == "__main__" else None

from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
import math
import textwrap
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.path import Path as MplPath
from matplotlib.patches import Circle, PathPatch, Polygon, Rectangle
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, pearsonr, spearmanr
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.weight'] = 'regular'
plt.rcParams['axes.titleweight'] = 'regular'
plt.rcParams['axes.labelweight'] = 'regular'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['legend.frameon'] = False
plt.rcParams['axes.unicode_minus'] = False
PROJECT_ROOT = PACKAGE_ROOT
RUN_SCREEN = PROJECT_ROOT / 'runs' / '20260818-q3-mechanism-modeling'
RUN_FORMAL = PROJECT_ROOT / 'runs' / '20260818-q3-final-mechanism-models'
RUN_ROBUST = PROJECT_ROOT / 'runs' / '20260818-q3-robustness-analysis'
DERIVED = PROJECT_ROOT / 'derived_data'
OUT_DIR = output_dir('Extended_Data_Fig_3')
OUT_DIR.mkdir(parents=True, exist_ok=True)
PALETTE = {'pale_apricot': '#FFE7CC', 'soft_orange': '#FFB66D', 'burnt_orange': '#F27C2B', 'warm_sand': '#E7C9A0', 'cool_mist': '#DCE6F2', 'light_sky': '#A7CFF2', 'cerulean': '#3B82D6', 'cobalt': '#0D47A1', 'teal': '#4D807D', 'green': '#0E5E36', 'red': '#C83F26', 'rose': '#CF7C65', 'gold': '#DDBD19', 'steel': '#7F8AA9', 'violet': '#6C5CA8', 'text': '#1F2937', 'muted': '#6B7280', 'axis': '#30343B', 'grid': '#E5E7EB', 'light_grid': '#F1F3F6'}
FAMILY_COLORS = {'POI supply': PALETTE['burnt_orange'], 'Motorized accessibility': PALETTE['soft_orange'], 'Non-motorized accessibility': PALETTE['gold'], 'Road network': PALETTE['warm_sand'], 'Green/blue space': PALETTE['teal'], 'Built environment': PALETTE['cobalt'], 'Economic performance': PALETTE['steel'], 'Topographic control': PALETTE['steel'], 'Terrain control': PALETTE['steel'], 'Location control': PALETTE['violet'], 'Environmental risk': PALETTE['red'], 'Other': PALETTE['muted']}
AJ_FAMILY_COLORS = {'Green/blue space': '#6BBDD7', 'Environmental risk': '#D81809', 'Built environment': '#3378B6', 'Economic performance': '#6A729B', 'Economic/population': '#6A729B', 'POI supply': '#F1881B', 'Motorized accessibility': '#F1881B', 'Non-motorized accessibility': '#F0B17B', 'Road network': '#EBC99B', 'Topographic control': '#8A94B2', 'Terrain control': '#8A94B2', 'Location control': '#8172A8', 'Impervious control': '#8A94B2', 'Other': '#C8D0C1'}
GROUP_COLORS = {'global': PALETTE['axis'], 'North': PALETTE['cobalt'], 'South': PALETTE['burnt_orange'], 'core': PALETTE['teal'], 'expansion': PALETTE['gold']}
SEQ_BLUE = LinearSegmentedColormap.from_list('q3_seq_blue', ['#F6F8FB', PALETTE['cool_mist'], PALETTE['light_sky'], PALETTE['cerulean'], PALETTE['cobalt']])
SEQ_ORANGE = LinearSegmentedColormap.from_list('q3_seq_orange', ['#FFF7EC', PALETTE['pale_apricot'], PALETTE['soft_orange'], PALETTE['burnt_orange']])
JOINT_DENSITY = LinearSegmentedColormap.from_list('q3_joint_density', ['#FFF3E5', '#DCEBF7', PALETTE['light_sky'], PALETTE['cerulean']])
DIV_CORR = LinearSegmentedColormap.from_list('q3_div_corr', [PALETTE['burnt_orange'], '#F7F2EB', '#DCE6F2', PALETTE['cobalt']])
DELTA_CMAP = LinearSegmentedColormap.from_list('q3_delta', [PALETTE['red'], '#F7F2EB', '#DCE6F2', PALETTE['cobalt']])
OUTCOME_LABELS = {'eli50': 'ELI', 'compound_gini50': 'Gini', 'compound_theil50': 'Theil', 'compound_p9010_50': 'P90-P10', 'positive_score50': 'Positive', 'negative_score50': 'Negative'}
MECHANISM_OUTCOMES = ['eli50', 'compound_gini50', 'compound_theil50', 'compound_p9010_50']
OUTCOME_COLORS = {'eli50': PALETTE['teal'], 'compound_gini50': PALETTE['burnt_orange'], 'compound_theil50': PALETTE['red'], 'compound_p9010_50': PALETTE['violet']}
GROUP_ORDER = ['global', 'North', 'South', 'core', 'expansion']
GROUP_LABELS = {'global': 'Global', 'North': 'North', 'South': 'South', 'core': 'Core', 'expansion': 'Expansion'}
CONFIG_LABELS = {'main_imputed_rerun': 'Main imputed', 'main_complete_case': 'Complete case', 'poi_total_only': 'POI total', 'drop_sparse_poi_road': 'Drop sparse', 'motorized_access': 'Motor access', 'impervious_replacement': 'Impervious'}
ABBR_BY_COLUMN = {'green_fraction_mean': 'Green', 'water_fraction_mean': 'Water', 'building_density_mean': 'Built dens.', 'building_height_mean': 'Height', 'ntl_log1p_mean': 'NTL', 'population_density_mean': 'Pop. dens.', 'lat': 'Latitude', 'dem_mean': 'Elev.', 'slope_mean': 'Slope', 'pm25_2020_mean': 'PM2.5', 'lst_c_mean': 'LST', 'noise_mean': 'Noise', 'poi_education_mean': 'POI edu.', 'poi_entertainment_mean': 'POI ent.', 'poi_health_mean': 'POI health', 'poi_transport_mean': 'POI transp.', 'access_nonmotor_education_mean': 'Access edu.', 'access_nonmotor_entertainment_mean': 'Access ent.', 'access_nonmotor_health_mean': 'Access health', 'access_nonmotor_living_mean': 'Access living', 'access_nonmotor_transport_mean': 'Access transp.', 'road_total_mean': 'Road total', 'road_footway_mean': 'Footway', 'road_pedestrian_mean': 'Pedestrian', 'road_cycleway_mean': 'Cycleway'}
MATRIX_ABBR_BY_COLUMN = {'green_fraction_mean': 'Green', 'water_fraction_mean': 'Water', 'building_density_mean': 'BD', 'building_height_mean': 'BH', 'ntl_log1p_mean': 'NTL', 'population_density_mean': 'Pop', 'lat': 'Lat', 'dem_mean': 'DEM', 'slope_mean': 'Slope', 'pm25_2020_mean': 'PM2.5', 'lst_c_mean': 'LST', 'noise_mean': 'Noise', 'poi_education_mean': 'P edu', 'poi_entertainment_mean': 'P ent', 'poi_health_mean': 'P hlth', 'poi_transport_mean': 'P trsp', 'access_nonmotor_education_mean': 'A edu', 'access_nonmotor_entertainment_mean': 'A ent', 'access_nonmotor_health_mean': 'A hlth', 'access_nonmotor_living_mean': 'A live', 'access_nonmotor_transport_mean': 'A trsp', 'road_total_mean': 'Road', 'road_footway_mean': 'Foot', 'road_pedestrian_mean': 'Ped', 'road_cycleway_mean': 'Cycle'}
FS_PANEL_LETTER = 8.8
FS_PANEL_TITLE = 7.1
FS_AXIS_LABEL = 5.9
FS_AXIS_TICK = 5.6
FS_BODY = 5.15
FS_TABLE_HEADER = 5.25
FS_TABLE_VALUE = 4.85
FS_DENSE_LABEL = 4.65
FS_MATRIX_TICK = 4.2
FS_LEGEND = 4.7
FS_ANNOT = 5.05
FS_COLORBAR_LABEL = 5.35
FS_COLORBAR_TICK = 4.9

def read_csv(path: Path) -> pd.DataFrame:
    if not resource(path).exists():
        raise FileNotFoundError(path)
    return pd.read_csv(resource(path))

def load_inputs() -> dict[str, pd.DataFrame]:
    return {'predictors': read_csv(RUN_SCREEN / 'q3_final_predictors_recommended.csv'), 'vif_city': read_csv(RUN_SCREEN / 'q3_final_exact_vif_city_r.csv'), 'vif_zone': read_csv(RUN_SCREEN / 'q3_final_exact_vif_city_zone_r.csv'), 'performance': read_csv(RUN_FORMAL / 'q3_formal_rf_cv_performance.csv'), 'predictions': read_csv(RUN_FORMAL / 'q3_formal_predictions.csv'), 'sample': read_csv(RUN_FORMAL / 'q3_formal_sample.csv'), 'permutation': read_csv(RUN_FORMAL / 'q3_formal_permutation.csv'), 'spearman': read_csv(RUN_FORMAL / 'q3_formal_spearman.csv'), 'robustness': read_csv(RUN_ROBUST / 'q3_robustness_summary.csv'), 'stability': read_csv(RUN_ROBUST / 'q3_robustness_family_top3_stability.csv'), 'city_master': read_csv(DERIVED / 'q3_city_driver_master.csv'), 'zone_master': read_csv(DERIVED / 'q3_city_zone_driver_master.csv')}

def clean_label(text: str, max_len: int=18) -> str:
    text = str(text).replace('Environmental', 'Env.').replace('Non-motorized', 'Non-motor')
    if len(text) <= max_len:
        return text
    return '\n'.join(textwrap.wrap(text, width=max_len, break_long_words=False)[:2])

def alluvial_family_label(text: str) -> str:
    labels = {'Built environment': 'Built\nenv.', 'Economic/population': 'Econ./\npop.', 'Environmental risk': 'Env. risk', 'Green/blue space': 'Green/\nblue', 'Impervious control': 'Imperv.\ncontrol', 'Location control': 'Location', 'Non-motorized accessibility': 'Non-motor', 'POI supply': 'POI supply'}
    return labels.get(str(text), clean_label(str(text), 12))

def var_label(row: pd.Series) -> str:
    column = str(row.get('column', ''))
    if column in ABBR_BY_COLUMN:
        return ABBR_BY_COLUMN[column]
    return clean_label(str(row.get('variable', column)), 14)

def matrix_label(row: pd.Series) -> str:
    column = str(row.get('column', ''))
    return MATRIX_ABBR_BY_COLUMN.get(column, var_label(row).replace('\n', ' '))

def outcome_row_label(table: str, outcome: str) -> str:
    prefix = 'City' if table == 'city' else 'Zone'
    return f'{prefix} {OUTCOME_LABELS.get(outcome, outcome)}'

def set_panel_header(ax: mpl.axes.Axes, letter: str, title: str, letter_x: float=-0.035, title_x: float=0.055, header_y: float=1.08) -> None:
    ax.text(letter_x, header_y, letter, transform=ax.transAxes, ha='left', va='bottom', fontsize=FS_PANEL_LETTER, fontweight='bold', color=PALETTE['text'], clip_on=False)
    ax.text(title_x, header_y, title, transform=ax.transAxes, ha='left', va='bottom', fontsize=FS_PANEL_TITLE, color=PALETTE['text'], clip_on=False)

def align_panel_header_gaps(fig: mpl.figure.Figure, reference: tuple[mpl.axes.Axes, str, str], targets: list[tuple[mpl.axes.Axes, str, str]]) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def header_text(ax: mpl.axes.Axes, value: str) -> mpl.text.Text:
        matches = [text for text in ax.texts if text.get_text() == value]
        if not matches:
            raise ValueError(f'Missing panel header text: {value}')
        return matches[0]
    ref_ax, ref_letter, ref_title = reference
    ref_letter_text = header_text(ref_ax, ref_letter)
    ref_title_text = header_text(ref_ax, ref_title)
    ref_gap_px = ref_title_text.get_window_extent(renderer).x0 - ref_letter_text.get_window_extent(renderer).x1
    for ax, letter, title in targets:
        letter_text = header_text(ax, letter)
        title_text = header_text(ax, title)
        letter_bbox = letter_text.get_window_extent(renderer)
        title_x, title_y = title_text.get_position()
        _, title_y_px = ax.transAxes.transform((title_x, title_y))
        new_title_x_px = letter_bbox.x1 + ref_gap_px
        new_title_x = ax.transAxes.inverted().transform((new_title_x_px, title_y_px))[0]
        title_text.set_x(new_title_x)

def clean_axis(ax: mpl.axes.Axes, grid: bool=False) -> None:
    ax.tick_params(axis='both', labelsize=FS_AXIS_TICK, length=2.2, width=0.45, colors=PALETTE['axis'], pad=1.5)
    for spine in ax.spines.values():
        spine.set_color(PALETTE['axis'])
        spine.set_linewidth(0.55)
    if grid:
        ax.grid(axis='x', color=PALETTE['grid'], lw=0.42, alpha=0.85, zorder=0)

def add_vertical_colorbar(ax: mpl.axes.Axes, mappable, label: str, ticks=None, box=(1.02, 0.18, 0.032, 0.62)) -> None:
    cax = ax.inset_axes(list(box))
    cbar = ax.figure.colorbar(mappable, cax=cax, orientation='vertical', ticks=ticks)
    cbar.outline.set_linewidth(0.45)
    cbar.ax.tick_params(labelsize=FS_COLORBAR_TICK, length=1.8, width=0.4, pad=1, colors=PALETTE['axis'])
    cbar.ax.set_ylabel(label, fontsize=FS_COLORBAR_LABEL, rotation=90, labelpad=5, color=PALETTE['text'])

def force_all_text_black(fig: mpl.figure.Figure) -> None:
    for text in fig.findobj(match=mpl.text.Text):
        text.set_color('#000000')

def bezier_patch(start, end, color, lw=1.0, alpha=0.35, control_scale=0.28, zorder=2) -> PathPatch:
    sx, sy = start
    ex, ey = end
    c1 = (sx * (1 - control_scale), sy * (1 - control_scale))
    c2 = (ex * (1 - control_scale), ey * (1 - control_scale))
    path = MplPath([start, c1, c2, end], [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    return PathPatch(path, facecolor='none', edgecolor=color, lw=lw, alpha=alpha, capstyle='round', zorder=zorder)

def polar_to_xy(theta_deg: float, r: float) -> tuple[float, float]:
    t = math.radians(theta_deg)
    return (r * math.cos(t), r * math.sin(t))

def plot_driver_chord(ax: mpl.axes.Axes, predictors: pd.DataFrame) -> None:
    set_panel_header(ax, 'a', 'Driver system', letter_x=0.074, title_x=0.138)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    df = predictors.copy()
    df['family_label'] = df['family_label'].fillna(df['family'])
    df['selection_order'] = pd.to_numeric(df['selection_order'], errors='coerce')
    family_order = df.groupby('family_label')['selection_order'].min().sort_values().index.tolist()
    df['family_label'] = pd.Categorical(df['family_label'], categories=family_order, ordered=True)
    df = df.sort_values(['family_label', 'selection_order', 'column']).reset_index(drop=True)
    families = df.groupby('family_label', observed=True).size().reindex(family_order)
    family_short = {'Green/blue space': 'Green/blue', 'Environmental risk': 'Env. risk', 'Built environment': 'Built env.', 'Economic/population': 'Econ./pop.', 'POI supply': 'POI supply', 'Non-motorized accessibility': 'Non-motor', 'Road network': 'Road', 'Terrain control': 'Terrain', 'Location control': 'Location'}
    main_y_shift = -0.25
    center = (-0.05, -0.06 + main_y_shift)
    root_r = 0.2
    hub_r = 0.42
    leaf_r = 0.68
    band_inner, band_outer = (0.5, 0.78)
    city_ring, zone_ring = (0.86, 0.91)
    ring_width = 0.026
    label_r = 1.035
    coverage_norm = Normalize(vmin=60, vmax=100)

    def xy(theta_deg: float, radius: float) -> tuple[float, float]:
        x, y = polar_to_xy(theta_deg, radius)
        return (center[0] + x, center[1] + y)

    def label_style(theta_deg: float) -> tuple[str, float]:
        theta = theta_deg % 360
        if 90 < theta < 270:
            return ('right', theta_deg + 180)
        return ('left', theta_deg)

    def annular_segment(theta1: float, theta2: float, r_inner: float, r_outer: float, **kwargs) -> None:
        n = max(6, int(abs(theta2 - theta1) / 2.8) + 2)
        outer = np.linspace(theta1, theta2, n)
        inner = np.linspace(theta2, theta1, n)
        verts = [xy(a, r_outer) for a in outer] + [xy(a, r_inner) for a in inner]
        ax.add_patch(Polygon(verts, closed=True, **kwargs))
    start_angle, end_angle = (-30, 210)
    gap = 3.4
    usable = end_angle - start_angle - gap * (len(families) - 1)
    total = int(families.sum())
    family_angles: dict[str, tuple[float, float, float, list[float]]] = {}
    cur = start_angle
    for fam, count_value in families.items():
        count = int(count_value)
        span = usable * count / total
        t1, t2 = (cur, cur + span)
        mid = (t1 + t2) / 2
        leaf_angles = [mid] if count == 1 else np.linspace(t1 + span / (count + 1), t2 - span / (count + 1), count).tolist()
        family_angles[str(fam)] = (t1, t2, mid, leaf_angles)
        color = AJ_FAMILY_COLORS.get(str(fam), FAMILY_COLORS.get(fam, PALETTE['steel']))
        annular_segment(t1, t2, band_inner, band_outer, facecolor=color, edgecolor='none', alpha=0.075, zorder=0)
        annular_segment(t1, t2, band_outer + 0.001, band_outer + 0.035, facecolor=color, edgecolor='white', lw=0.35, alpha=0.96, zorder=3)
        hx, hy = xy(mid, hub_r)
        ax.plot([center[0], hx], [center[1], hy], color=color, lw=0.55, alpha=0.44, zorder=1)
        ax.scatter(hx, hy, s=14, color=color, edgecolor='white', linewidth=0.35, zorder=4)
        cur = t2 + gap
    for fam_value, sub in df.groupby('family_label', observed=True, sort=False):
        fam = str(fam_value)
        t1, t2, mid, leaf_angles = family_angles[fam]
        color = AJ_FAMILY_COLORS.get(str(fam), FAMILY_COLORS.get(fam, PALETTE['steel']))
        hx, hy = xy(mid, hub_r)
        angle_width = max(0.85, min(1.95, (t2 - t1) / (len(sub) + 2) * 0.42))
        for angle, (_, row) in zip(leaf_angles, sub.iterrows()):
            lx, ly = xy(angle, leaf_r)
            ax.plot([hx, lx], [hy, ly], color=color, lw=0.5, alpha=0.46, zorder=1)
            ax.scatter(lx, ly, s=12, color=color, edgecolor='white', linewidth=0.34, zorder=5)
            annular_segment(angle - angle_width, angle + angle_width, city_ring - ring_width, city_ring, facecolor=SEQ_BLUE(coverage_norm(float(row['city_coverage_pct']))), edgecolor='white', lw=0.18, zorder=4)
            annular_segment(angle - angle_width, angle + angle_width, zone_ring - ring_width, zone_ring, facecolor=SEQ_BLUE(coverage_norm(float(row['zone_coverage_pct']))), edgecolor='white', lw=0.18, zorder=4)
            tx, ty = xy(angle, label_r)
            ha, rot = label_style(angle)
            ax.text(tx, ty, var_label(row), ha=ha, va='center', rotation=rot, rotation_mode='anchor', fontsize=FS_DENSE_LABEL, color=PALETTE['text'], zorder=6)
    ax.add_patch(Circle(center, root_r, facecolor='#FFFFFF', edgecolor=PALETTE['grid'], lw=0.55, zorder=7))
    ax.text(center[0], center[1] + 0.07, '25', ha='center', va='center', fontsize=6.3, color=PALETTE['text'], zorder=8)
    ax.text(center[0], center[1] - 0.06, 'final\ndrivers', ha='center', va='center', fontsize=FS_BODY, linespacing=0.88, color=PALETTE['text'], zorder=8)
    legend_cols_x = [-0.93, -0.28, 0.37]
    legend_y0 = -0.765 + main_y_shift
    legend_row_gap = 0.135
    for idx, (fam, count_value) in enumerate(families.items()):
        col = idx % 3
        row = idx // 3
        x = legend_cols_x[col]
        y = legend_y0 - row * legend_row_gap
        color = AJ_FAMILY_COLORS.get(str(fam), FAMILY_COLORS.get(fam, PALETTE['steel']))
        ax.plot([x, x + 0.055], [y, y], color=color, lw=1.45, solid_capstyle='butt', zorder=8, clip_on=False)
        ax.text(x + 0.068, y, f'{family_short.get(str(fam), str(fam))} ({int(count_value)})', ha='left', va='center', fontsize=FS_DENSE_LABEL, color=PALETTE['text'], zorder=8)
    ax.text(-0.05, -0.49 + main_y_shift, 'Coverage (%)', ha='center', va='bottom', fontsize=FS_BODY, color=PALETTE['text'], zorder=8)
    cax = ax.inset_axes([-0.35, -0.55 + main_y_shift, 0.6, 0.034], transform=ax.transData)
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=SEQ_BLUE, norm=coverage_norm, orientation='horizontal', ticks=[60, 80, 100])
    cax.tick_params(labelsize=FS_COLORBAR_TICK, length=1.5, width=0.35, pad=0.8, colors=PALETTE['text'])
    cbar.outline.set_linewidth(0.38)
    ax.text(0.5, -0.585 + main_y_shift, 'inner: city\nouter: zone', ha='left', va='center', fontsize=FS_DENSE_LABEL, linespacing=0.92, color=PALETTE['text'], zorder=8)
    ax.set_xlim(-1.62, 1.58)
    ax.set_ylim(-1.03, 1.03)

def plot_table_heatmap(ax: mpl.axes.Axes, predictors: pd.DataFrame) -> None:
    set_panel_header(ax, 'b', 'Coverage table')
    df = predictors.copy().sort_values(['family_label', 'selection_order', 'column']).reset_index(drop=True)
    y = np.arange(len(df))
    norm = Normalize(vmin=60, vmax=100)
    ax.set_xlim(-1.25, 3.65)
    ax.set_ylim(len(df) - 0.5, -0.5)
    for i, row in df.iterrows():
        fam = str(row['family_label'])
        color = FAMILY_COLORS.get(fam, PALETTE['steel'])
        ax.add_patch(Rectangle((-1.17, i - 0.41), 0.18, 0.82, facecolor=color, edgecolor='white', lw=0.35))
        ax.text(-0.92, i, var_label(row), ha='left', va='center', fontsize=4.25, color=PALETTE['text'])
        for j, col in enumerate(['city_coverage_pct', 'zone_coverage_pct']):
            val = float(row[col])
            ax.add_patch(Rectangle((j + 0.24, i - 0.41), 0.72, 0.82, facecolor=SEQ_BLUE(norm(val)), edgecolor='white', lw=0.42))
            ax.text(j + 0.6, i, f'{val:.0f}', ha='center', va='center', fontsize=4.2, color=PALETTE['text'])
        for j, col in enumerate(['passes_city_coverage', 'passes_zone_coverage']):
            passed = bool(row[col])
            x = 2.34 + j * 0.44
            ax.scatter(x, i, s=18, marker='o' if passed else 'x', color=color if passed else PALETTE['muted'], edgecolor='white', linewidth=0.35, zorder=3)
    ax.set_xticks([0.6, 1.6, 2.34, 2.78], ['City', 'Zone', 'C', 'Z'], fontsize=5.0)
    ax.text(2.56, -0.92, 'pass', ha='center', va='bottom', fontsize=4.8, color=PALETTE['muted'], clip_on=False)
    ax.set_yticks([])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=SEQ_BLUE)
    add_vertical_colorbar(ax, sm, 'Coverage (%)', ticks=[60, 80, 100], box=(0.96, 0.18, 0.035, 0.62))

def plot_sample_circular(ax: mpl.axes.Axes, sample: pd.DataFrame) -> None:
    set_panel_header(ax, 'c', 'Sample completeness')
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.02)
    ax.grid(False)
    ax.spines['polar'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    rows = [('city', 'eli50', 'City ELI'), ('city', 'compound_gini50', 'City Gini'), ('city', 'compound_theil50', 'City Theil'), ('city', 'compound_p9010_50', 'City P90'), ('city_zone', 'eli50', 'Zone ELI'), ('city_zone', 'positive_score50', 'Zone Pos.'), ('city_zone', 'negative_score50', 'Zone Neg.')]
    groups = GROUP_ORDER
    angles = np.linspace(0, 2 * np.pi, len(rows), endpoint=False)
    offsets = np.linspace(-0.085, 0.085, len(groups))
    ax.bar(np.linspace(0, 2 * np.pi, 96), [0.002] * 96, width=2 * np.pi / 96, bottom=0.54, color=PALETTE['grid'], edgecolor='none')
    ax.bar(np.linspace(0, 2 * np.pi, 96), [0.002] * 96, width=2 * np.pi / 96, bottom=0.84, color=PALETTE['grid'], edgecolor='none')
    for angle, (table, outcome, label) in zip(angles, rows):
        ax.text(angle, 0.99, label, ha='center', va='center', fontsize=4.85, color=PALETTE['text'])
        sub = sample[(sample['table'] == table) & (sample['outcome'] == outcome)]
        for off, group in zip(offsets, groups):
            hit = sub[sub['group'] == group]
            if hit.empty:
                continue
            r = hit.iloc[0]
            pct = float(r['n_complete_predictors']) / float(r['n'])
            radius = 0.45 + 0.42 * np.clip((pct - 0.55) / 0.3, 0, 1)
            theta = angle + off
            color = GROUP_COLORS[group]
            ax.plot([theta, theta], [0.43, radius], color=color, lw=0.6, alpha=0.55)
            ax.scatter(theta, radius, s=10 + 22 * pct, color=color, edgecolor='white', linewidth=0.35, zorder=4)
    legend_handles = [mpl.lines.Line2D([], [], marker='o', linestyle='none', markersize=3.6, markerfacecolor=GROUP_COLORS[g], markeredgecolor='white', label=GROUP_LABELS[g]) for g in groups]
    ax.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=4.65, columnspacing=0.8, handletextpad=0.25)

def plot_vif_forest(ax: mpl.axes.Axes, predictors: pd.DataFrame, vif_city: pd.DataFrame, vif_zone: pd.DataFrame) -> None:
    set_panel_header(ax, 'b', 'Exact VIF forest', letter_x=0.005, title_x=0.069)
    label_map = predictors.set_index('column')['variable'].to_dict()
    df = vif_city[['column', 'exact_vif']].rename(columns={'exact_vif': 'city_vif'}).merge(vif_zone[['column', 'exact_vif']].rename(columns={'exact_vif': 'zone_vif'}), on='column', how='outer')
    df['city_vif'] = pd.to_numeric(df['city_vif'], errors='coerce')
    df['zone_vif'] = pd.to_numeric(df['zone_vif'], errors='coerce')
    df['max_vif'] = df[['city_vif', 'zone_vif']].max(axis=1)
    df = df.sort_values('max_vif', ascending=False).head(14).reset_index(drop=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    table_left, table_right = (0.0, 0.99)
    label_x = 0.018
    forest_left, forest_right = (0.203, 0.732)
    right_cols = {'City': 0.805, 'Zone': 0.94}
    content_y_shift = 0.08
    header_y = 0.9 + content_y_shift
    row_top, row_bottom = (0.815 + content_y_shift, 0.185 + content_y_shift)
    axis_y = 0.11 + content_y_shift
    row_y = np.linspace(row_top, row_bottom, len(df))
    row_gap = abs(row_y[0] - row_y[1]) if len(row_y) > 1 else 0.07
    row_h = min(0.052, row_gap * 0.78)
    city_color = PALETTE['cobalt']
    zone_color = PALETTE['burnt_orange']
    forest_min, forest_max = (0.0, 10.0)

    def map_x(value: float) -> float:
        value = float(np.clip(value, forest_min, forest_max))
        return forest_left + (value - forest_min) / (forest_max - forest_min) * (forest_right - forest_left)
    header_font = dict(fontsize=FS_TABLE_HEADER, fontweight='bold', color=PALETTE['text'])
    label_font = dict(fontsize=FS_BODY, color=PALETTE['text'])
    value_font = dict(fontsize=FS_TABLE_VALUE, color=PALETTE['text'])
    ax.text(label_x, header_y, 'Variable', transform=ax.transAxes, ha='left', va='bottom', **header_font)
    for name, x in right_cols.items():
        ax.text(x, header_y, name, transform=ax.transAxes, ha='center', va='bottom', **header_font)
    ax.scatter([right_cols['City'], right_cols['Zone']], [header_y - 0.037, header_y - 0.037], s=11.0, color=[city_color, zone_color], edgecolor='white', linewidth=0.25, transform=ax.transAxes, zorder=5)
    for idx, row in df.iterrows():
        y = row_y[idx]
        if idx % 2 == 1:
            ax.add_patch(Rectangle((table_left, y - row_h / 2), table_right - table_left, row_h, transform=ax.transAxes, facecolor='#E1EEFA', alpha=0.4, edgecolor='none', zorder=0))
        label = ABBR_BY_COLUMN.get(row['column'], clean_label(label_map.get(row['column'], row['column']), 13).replace('\n', ' '))
        ax.text(label_x, y, label, transform=ax.transAxes, ha='left', va='center', **label_font)
        for col_name, value in [('City', row['city_vif']), ('Zone', row['zone_vif'])]:
            if np.isfinite(value):
                ax.text(right_cols[col_name], y, f'{value:.2f}', transform=ax.transAxes, ha='center', va='center', **value_font)
        values = [v for v in [row['city_vif'], row['zone_vif']] if np.isfinite(v)]
        if values:
            x0, x1 = (map_x(min(values)), map_x(max(values)))
            ax.plot([x0, x1], [y, y], color=PALETTE['axis'], lw=0.52, transform=ax.transAxes, zorder=2)
        if np.isfinite(row['city_vif']):
            ax.scatter([map_x(row['city_vif'])], [y], s=11.0, color=city_color, edgecolor='white', linewidth=0.25, transform=ax.transAxes, zorder=4)
        if np.isfinite(row['zone_vif']):
            ax.scatter([map_x(row['zone_vif'])], [y], s=11.0, color=zone_color, edgecolor='white', linewidth=0.25, transform=ax.transAxes, zorder=4)
    for ref, linestyle, label in [(5.0, (0, (3.0, 3.0)), 'VIF=5'), (10.0, (0, (1.2, 2.0)), 'VIF=10')]:
        x = map_x(ref)
        ax.plot([x, x], [axis_y, header_y - 0.022], color=PALETTE['axis'], lw=0.55, ls=linestyle, transform=ax.transAxes, zorder=1)
        if ref < 10:
            ax.text(x + 0.01, header_y - 0.015, label, transform=ax.transAxes, ha='left', va='top', fontsize=FS_ANNOT, color=PALETTE['muted'])
    ax.plot([forest_left, forest_right], [axis_y, axis_y], color=PALETTE['axis'], lw=0.62, transform=ax.transAxes, zorder=5)
    for tick in [0, 2.5, 5, 7.5, 10]:
        x = map_x(tick)
        ax.plot([x, x], [axis_y, axis_y - 0.017], color=PALETTE['axis'], lw=0.46, transform=ax.transAxes, zorder=5)
        ax.text(x, axis_y - 0.032, f'{tick:g}', transform=ax.transAxes, ha='center', va='top', fontsize=FS_AXIS_TICK, color=PALETTE['text'])
    ax.text((forest_left + forest_right) / 2, -0.005 + content_y_shift, 'Exact VIF', transform=ax.transAxes, ha='center', va='bottom', fontsize=FS_AXIS_LABEL, color=PALETTE['axis'], clip_on=False)

def plot_triangular_correlation(ax: mpl.axes.Axes, predictors: pd.DataFrame, city_master: pd.DataFrame, zone_master: pd.DataFrame) -> None:
    set_panel_header(ax, 'c', 'City-zone correlation matrix', letter_x=0.105, title_x=0.169, header_y=1.12)
    main_ax = ax.inset_axes([0.1, -0.05, 1.21, 1.1])
    ax.set_axis_off()
    ax = main_ax
    df = predictors.copy().sort_values(['family_label', 'selection_order', 'column'])
    cols = [c for c in df['column'] if c in city_master.columns and c in zone_master.columns]
    labels = [matrix_label(df[df['column'] == c].iloc[0]) for c in cols]
    corr_city = city_master[cols].corr(method='spearman', min_periods=50).to_numpy(float)
    corr_zone = zone_master[cols].corr(method='spearman', min_periods=50).to_numpy(float)
    n = len(cols)
    lower = np.full((n, n), np.nan)
    lower[np.tril_indices(n)] = corr_city[np.tril_indices(n)]
    im = ax.imshow(lower, cmap=DIV_CORR, vmin=-1, vmax=1, aspect='equal')
    for i in range(n):
        for j in range(i + 1, n):
            r = corr_zone[i, j]
            if np.isfinite(r):
                color = PALETTE['cobalt'] if r >= 0 else PALETTE['burnt_orange']
                ax.scatter(j, i, s=2.0 + 20 * abs(r), color=color, edgecolor='none', alpha=0.66)
    ax.set_xticks(np.arange(n), labels, fontsize=FS_MATRIX_TICK, rotation=90)
    ax.set_yticks(np.arange(n), labels, fontsize=FS_MATRIX_TICK)
    ax.tick_params(length=0, pad=0.7)
    for k in range(n + 1):
        ax.axhline(k - 0.5, color='white', lw=0.26, zorder=3)
        ax.axvline(k - 0.5, color='white', lw=0.26, zorder=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    add_vertical_colorbar(ax, im, 'Spearman r', ticks=[-1, 0, 1], box=(1.025, 0.18, 0.03, 0.62))
    ax.text(0.0, 1.015, 'lower city / upper zone', transform=ax.transAxes, ha='left', va='bottom', fontsize=FS_ANNOT, color=PALETTE['muted'], clip_on=False)

def plot_performance_forest(ax: mpl.axes.Axes, performance: pd.DataFrame, align_ratio: float=1.0) -> None:
    set_panel_header(ax, 'd', 'Predictive skill forest', letter_x=-0.035 * align_ratio + 0.04, title_x=-0.035 * align_ratio + 0.104)
    rows = [('city', 'eli50'), ('city', 'compound_gini50'), ('city', 'compound_theil50'), ('city', 'compound_p9010_50'), ('city_zone', 'eli50'), ('city_zone', 'positive_score50'), ('city_zone', 'negative_score50')]
    groups = ['global', 'North', 'South', 'core', 'expansion']
    group_display = {'global': 'Global', 'North': 'North', 'South': 'South', 'core': 'Core', 'expansion': 'Exp.'}
    group_colors = {'global': '#C8D0C1', 'North': '#3B82D6', 'South': '#F28817', 'core': PALETTE['teal'], 'expansion': PALETTE['gold']}
    row_keys = [f'{table}|{outcome}' for table, outcome in rows]
    df = performance[performance['outcome_role'].eq('primary') & performance['group'].isin(groups) & performance.apply(lambda r: (r['table'], r['outcome']) in rows, axis=1)].copy()
    df['row_key'] = df['table'].astype(str) + '|' + df['outcome'].astype(str)
    main_y_shift = 0.1
    row_y = {key: y + main_y_shift for key, y in zip(row_keys, np.linspace(0.825, 0.33, len(row_keys)))}
    offset = {'global': 0.0, 'North': 0.024, 'South': -0.024, 'core': 0.012, 'expansion': -0.012}
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    label_x = 0.018 * align_ratio
    table_left, table_right = (0.0, 0.985)
    forest_left, forest_right = (0.203 * align_ratio, 0.575)
    forest_min, forest_max = (0.5, 0.9)
    axis_y = 0.255 + main_y_shift
    header_y = 0.9 + main_y_shift
    ax.set_xlabel('Cross-validated R2 (mean ± SD)', fontsize=FS_AXIS_LABEL, color=PALETTE['axis'], labelpad=1.6)
    ax.xaxis.set_label_coords((forest_left + forest_right) / 2, 0.165 + main_y_shift)

    def map_x(value: float) -> float:
        value = float(np.clip(value, forest_min, forest_max))
        return forest_left + (value - forest_min) / (forest_max - forest_min) * (forest_right - forest_left)
    for idx, key in enumerate(row_keys):
        y = row_y[key]
        if idx % 2 == 1:
            ax.add_patch(Rectangle((table_left, y - 0.033), table_right - table_left, 0.066, transform=ax.transAxes, facecolor='#E1EEFA', alpha=0.4, edgecolor='none', zorder=0))
    header_font = dict(fontsize=FS_TABLE_HEADER, fontweight='bold', color=PALETTE['text'])
    body_font = dict(fontsize=FS_BODY, color=PALETTE['text'])
    ax.text(label_x, header_y, 'Outcome', transform=ax.transAxes, ha='left', va='bottom', **header_font)
    right_cols = {'global': 0.655, 'North': 0.735, 'South': 0.815, 'core': 0.895, 'expansion': 0.97}
    for group, x in right_cols.items():
        ax.text(x, header_y, group_display[group], transform=ax.transAxes, ha='center', va='bottom', fontsize=FS_TABLE_HEADER, fontweight='bold', color=PALETTE['text'])
        ax.scatter([x], [header_y - 0.021], marker='o', s=8.05, facecolors=group_colors[group], edgecolors=group_colors[group], linewidths=0, transform=ax.transAxes, zorder=5)
    for table, outcome in rows:
        key = f'{table}|{outcome}'
        y = row_y[key]
        sub = df[df['row_key'] == key]
        ax.text(label_x, y, outcome_row_label(table, outcome), transform=ax.transAxes, ha='left', va='center', **body_font)
        for group, x in right_cols.items():
            g = sub[sub['group'] == group]
            if g.empty:
                continue
            r = g.iloc[0]
            ax.text(x, y, f"{r['r2_mean']:.3f}\n({r['r2_sd']:.3f})", transform=ax.transAxes, ha='center', va='center', fontsize=FS_TABLE_VALUE, linespacing=1.18, color=PALETTE['text'])
    ref_x = map_x(0.8)
    ax.plot([ref_x, ref_x], [axis_y, header_y - 0.018], color=PALETTE['axis'], lw=0.66, ls=(0, (3.2, 3.2)), transform=ax.transAxes, zorder=1)
    for _, row in df.iterrows():
        y = row_y[row['row_key']] + offset[row['group']]
        x = map_x(row['r2_mean'])
        xerr = row['r2_sd'] / (forest_max - forest_min) * (forest_right - forest_left)
        ax.plot([x - xerr, x + xerr], [y, y], color=PALETTE['axis'], lw=0.58, transform=ax.transAxes, zorder=3)
        ax.scatter([x], [y], marker='o', s=10.57, facecolors=group_colors[row['group']], edgecolors=group_colors[row['group']], linewidths=0, transform=ax.transAxes, zorder=4)
    ax.plot([forest_left, forest_right], [axis_y, axis_y], color=PALETTE['axis'], lw=0.62, transform=ax.transAxes, zorder=5)
    for tick in [0.5, 0.6, 0.7, 0.8, 0.9]:
        x = map_x(tick)
        ax.plot([x, x], [axis_y, axis_y - 0.017], color=PALETTE['axis'], lw=0.46, transform=ax.transAxes, zorder=5)
        ax.text(x, axis_y - 0.043, f'{tick:.1f}', transform=ax.transAxes, ha='center', va='top', fontsize=FS_AXIS_TICK, color=PALETTE['text'])

def plot_joint_prediction(parent_ax: mpl.axes.Axes, predictions: pd.DataFrame, performance: pd.DataFrame) -> mpl.axes.Axes:
    set_panel_header(parent_ax, 'e', 'Global city-level ELI prediction', letter_x=0.105, title_x=0.169)
    parent_ax.axis('off')
    main = parent_ax.inset_axes([0.253, 0.005, 0.905, 1.03])
    cax = main.inset_axes([0.585, 0.095, 0.255, 0.037])
    sub = predictions[(predictions['table'] == 'city') & (predictions['outcome'] == 'eli50') & (predictions['group'] == 'global')].dropna(subset=['observed', 'oof_prediction'])
    x = sub['observed'].to_numpy(float)
    y = sub['oof_prediction'].to_numpy(float)
    lim = np.nanpercentile(np.r_[x, y], [0.5, 99.5])
    pad = (lim[1] - lim[0]) * 0.08
    lim = [lim[0] - pad, lim[1] + pad]
    hb = main.hexbin(x, y, gridsize=32, extent=(lim[0], lim[1], lim[0], lim[1]), mincnt=1, cmap=JOINT_DENSITY, vmin=0, vmax=50, linewidths=0, alpha=0.98, rasterized=True, zorder=1)
    main.plot(lim, lim, color='black', lw=0.72, ls=(0, (4.0, 3.0)), zorder=5)
    resid = y - x
    band = float(np.nanquantile(np.abs(resid), 0.84))
    main.plot(lim, [v + band for v in lim], color=PALETTE['cerulean'], lw=0.72, ls=(0, (4.0, 3.0)), alpha=0.7, zorder=5)
    main.plot(lim, [v - band for v in lim], color=PALETTE['burnt_orange'], lw=0.72, ls=(0, (4.0, 3.0)), alpha=0.7, zorder=5)
    line_labels = [('1:1 agreement', 'black', 0.37), ('+84% residual', PALETTE['cerulean'], 0.29), ('-84% residual', PALETTE['burnt_orange'], 0.21)]
    for label, color, y_label in line_labels:
        main.plot([0.56, 0.66], [y_label, y_label], transform=main.transAxes, color=color, lw=0.72, ls=(0, (4.0, 3.0)), alpha=0.82, clip_on=False, zorder=7)
        main.text(0.68, y_label, label, transform=main.transAxes, ha='left', va='center', fontsize=FS_DENSE_LABEL, color='black', clip_on=False)
    r, p_val = pearsonr(x, y)
    p_text = 'P < 0.001' if p_val < 0.001 else f'P = {p_val:.3f}'
    main.text(0.11, 0.9, f'r = {r:.2f}\n{p_text}', transform=main.transAxes, ha='left', va='top', fontsize=FS_ANNOT, color='black')
    main.set_xlim(lim)
    main.set_ylim(lim)
    main.set_aspect('auto')
    main.set_xlabel('Observed ELI', fontsize=FS_AXIS_LABEL, labelpad=1.1)
    main.set_ylabel('Predicted ELI', fontsize=FS_AXIS_LABEL, labelpad=1.1)
    clean_axis(main, grid=False)
    main.spines['left'].set_linewidth(0.7)
    main.spines['bottom'].set_linewidth(0.7)
    cbar = parent_ax.figure.colorbar(hb, cax=cax, orientation='horizontal', ticks=[0, 25, 50])
    cbar.outline.set_linewidth(0.42)
    cbar.ax.tick_params(labelsize=FS_COLORBAR_TICK, length=1.7, width=0.38, pad=1, colors=PALETTE['axis'])
    return main

def kde_curve(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = values[np.isfinite(values)]
    if values.size < 5 or np.nanstd(values) == 0:
        return np.zeros_like(grid)
    dens = gaussian_kde(values)(grid)
    return dens / np.nanmax(dens) if np.nanmax(dens) > 0 else dens

def plot_residual_ridgeline(ax: mpl.axes.Axes, predictions: pd.DataFrame) -> None:
    set_panel_header(ax, 'h', 'Residual ridgelines')
    specs = [('city', 'eli50', 'global', 'City ELI', PALETTE['cobalt']), ('city', 'compound_gini50', 'global', 'City Gini', PALETTE['cerulean']), ('city', 'compound_theil50', 'global', 'City Theil', PALETTE['light_sky']), ('city_zone', 'eli50', 'global', 'Zone ELI', PALETTE['burnt_orange']), ('city_zone', 'positive_score50', 'global', 'Zone Pos.', PALETTE['soft_orange']), ('city_zone', 'negative_score50', 'global', 'Zone Neg.', PALETTE['rose'])]
    vals = []
    for table, outcome, group, _, _ in specs:
        vals.append(predictions[(predictions['table'] == table) & (predictions['outcome'] == outcome) & (predictions['group'] == group)]['oof_residual'].dropna().to_numpy(float))
    allv = np.concatenate(vals)
    xlim = np.nanpercentile(allv, [1, 99])
    span = xlim[1] - xlim[0]
    grid = np.linspace(xlim[0] - 0.05 * span, xlim[1] + 0.05 * span, 240)
    for idx, ((_, _, _, label, color), v) in enumerate(zip(specs, vals)):
        y0 = len(specs) - 1 - idx
        dens = kde_curve(v, grid) * 0.66
        med = np.nanmedian(v)
        ax.fill_between(grid, y0, y0 + dens, color=color, alpha=0.46, lw=0)
        ax.plot(grid, y0 + dens, color=color, lw=0.72, ls=(0, (2.0, 1.2)))
        ax.plot([med, med], [y0, y0 + 0.55], color=color, lw=0.65)
        ax.text(grid[0], y0 + 0.11, label, ha='left', va='bottom', fontsize=5.0, color=PALETTE['text'])
    ax.axvline(0, color=PALETTE['axis'], lw=0.65, ls='--')
    ax.set_ylim(-0.25, len(specs) - 0.05)
    ax.set_yticks([])
    ax.set_xlabel('OOF residual', fontsize=5.6, labelpad=1.1)
    clean_axis(ax, grid=True)

def plot_robustness_dumbbell(parent_ax: mpl.axes.Axes, robustness: pd.DataFrame) -> None:
    set_panel_header(parent_ax, 'i', 'Robustness summary')
    parent_ax.axis('off')
    df = robustness[robustness['robustness_config'].isin(CONFIG_LABELS)].copy()
    configs = [c for c in CONFIG_LABELS if c in set(df['robustness_config'])]
    ax_bar = parent_ax.inset_axes([0.04, 0.18, 0.28, 0.7])
    ax_dum = parent_ax.inset_axes([0.42, 0.18, 0.54, 0.7])
    cats = [('Gain', df['delta_r2_vs_formal'] > 0.005, PALETTE['cobalt']), ('Stable', df['delta_r2_vs_formal'].abs() <= 0.005, '#D6DCE8'), ('Loss', df['delta_r2_vs_formal'] < -0.005, PALETTE['red'])]
    y = np.arange(len(configs))[::-1]
    for yi, cfg in zip(y, configs):
        sub = df[df['robustness_config'] == cfg]
        left = 0
        for label, mask, color in cats:
            pct = 100 * mask.loc[sub.index].mean()
            ax_bar.barh(yi, pct, left=left, color=color, edgecolor='white', lw=0.35, height=0.56, alpha=0.86)
            left += pct
    ax_bar.set_xlim(0, 100)
    ax_bar.set_yticks(y, [CONFIG_LABELS[c] for c in configs], fontsize=5.0)
    ax_bar.set_xlabel('Models (%)', fontsize=5.3, labelpad=1.0)
    ax_bar.set_title('Delta R2 class', fontsize=5.5, pad=2, color=PALETTE['muted'])
    clean_axis(ax_bar, grid=True)
    agg = df.groupby('robustness_config', as_index=False).agg(robust=('r2_mean', 'mean'), formal=('formal_r2_mean', 'mean'), delta=('delta_r2_vs_formal', 'mean'))
    agg = agg.set_index('robustness_config').loc[configs].reset_index()
    for yi, row in zip(y, agg.itertuples(index=False)):
        color = PALETTE['cobalt'] if row.delta >= 0 else PALETTE['red']
        ax_dum.plot([row.formal, row.robust], [yi, yi], color=color, lw=1.15, alpha=0.66)
        ax_dum.scatter(row.formal, yi, s=24, facecolor='white', edgecolor=PALETTE['axis'], lw=0.75, zorder=3)
        ax_dum.scatter(row.robust, yi, s=26, facecolor=color, edgecolor='white', lw=0.45, zorder=4)
        ax_dum.text(max(row.formal, row.robust) + 0.006, yi, f'{row.delta:+.3f}', ha='left', va='center', fontsize=4.8, color=PALETTE['text'])
    ax_dum.set_yticks(y, [''] * len(configs))
    ax_dum.set_xlim(0.6, 0.86)
    ax_dum.set_xlabel('Mean CV R2 across model set', fontsize=5.3, labelpad=1.0)
    ax_dum.set_title('Formal vs robust model', fontsize=5.5, pad=2, color=PALETTE['muted'])
    handles = [mpl.lines.Line2D([], [], marker='o', linestyle='none', markerfacecolor='white', markeredgecolor=PALETTE['axis'], markersize=4, label='Formal'), mpl.lines.Line2D([], [], marker='o', linestyle='none', markerfacecolor=PALETTE['cobalt'], markeredgecolor='white', markersize=4, label='Robust')]
    ax_dum.legend(handles=handles, loc='lower right', fontsize=4.8, handletextpad=0.25, borderpad=0.15)
    clean_axis(ax_dum, grid=True)

def node_spans(counts: pd.Series, y0=0.12, y1=0.9, gap=0.018) -> dict[str, tuple[float, float, float]]:
    total = float(counts.sum())
    usable = y1 - y0 - gap * (len(counts) - 1)
    spans = {}
    cur = y1
    for name, count in counts.items():
        h = usable * count / total
        spans[name] = (cur - h, cur, cur - h / 2)
        cur -= h + gap
    return spans

def plot_stability_summary(ax: mpl.axes.Axes, stability: pd.DataFrame, align_ratio: float=1.0) -> dict[str, float]:
    set_panel_header(ax, 'f', 'Top-family stability', letter_x=-0.035 * align_ratio + 0.04, title_x=-0.035 * align_ratio + 0.104)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    df = stability[stability['robustness_config'].isin(CONFIG_LABELS)].copy()
    summary = df.groupby('robustness_config').agg(same=('top1_same_as_formal', 'mean'), overlap=('top3_overlap_n', 'mean'))
    configs = [name for name in CONFIG_LABELS if name in summary.index]
    labels = {'main_imputed_rerun': 'Repeat baseline model', 'main_complete_case': 'Complete cases only', 'drop_sparse_poi_road': 'Exclude sparse predictors', 'impervious_replacement': 'Replace with impervious cover'}
    left, right = (0.04, 0.65)
    overlap_x = 0.92
    ax.text((left + right) / 2, 1.025, 'Same top family', ha='center', va='bottom', fontsize=FS_DENSE_LABEL)
    ax.text(overlap_x, 1.025, 'Shared top 3', ha='center', va='bottom', fontsize=FS_DENSE_LABEL)
    for name, y in zip(configs, np.linspace(0.88, 0.22, len(configs))):
        row = summary.loc[name]
        ax.text(left, y + 0.045, labels.get(name, CONFIG_LABELS[name]), ha='left', va='bottom', fontsize=FS_DENSE_LABEL)
        ax.plot([left, right], [y, y], color='#DCE6F2', lw=2.5, zorder=1)
        x = left + float(row['same']) * (right - left)
        ax.plot([left, x], [y, y], color=PALETTE['cobalt'], lw=1.1, zorder=2)
        ax.scatter([x], [y], s=16, color=PALETTE['cobalt'], edgecolor='white', linewidth=0.35, zorder=3)
        ax.text(right + 0.035, y, f"{row['same']:.0%}", ha='left', va='center', fontsize=FS_DENSE_LABEL)
        ax.text(overlap_x, y, f"{row['overlap']:.2f}", ha='center', va='center', fontsize=FS_DENSE_LABEL)
    axis_y = 0.08
    ax.plot([left, right], [axis_y, axis_y], color=PALETTE['axis'], lw=0.5)
    for tick in [0, 0.5, 1]:
        x = left + tick * (right - left)
        ax.plot([x, x], [axis_y, axis_y - 0.014], color=PALETTE['axis'], lw=0.45)
        ax.text(x, axis_y - 0.025, f'{tick:.0%}', ha='center', va='top', fontsize=FS_DENSE_LABEL)
    ax.text(overlap_x, axis_y - 0.025, 'of 3', ha='center', va='top', fontsize=FS_DENSE_LABEL)
    return {'summary_x': 0.5, 'stable': df['top1_same_as_formal'].mean() * 100, 'overlap': df['top3_overlap_n'].mean()}

def plot_importance_concordance(ax: mpl.axes.Axes, data: dict[str, pd.DataFrame]) -> None:
    set_panel_header(ax, 'g', 'Spearman-ML concordance', letter_x=0.0, title_x=0.064, header_y=1.08)
    ax.axis('off')
    perm = data['permutation']
    spearman = data['spearman']
    p = perm[(perm['table'] == 'city') & (perm['group'] == 'global') & perm['outcome'].isin(MECHANISM_OUTCOMES)].copy()
    p['ml_importance'] = p.groupby('outcome')['perm_importance_mean'].transform(lambda s: s / max(s.max(), 1e-12))
    s = spearman[(spearman['table'] == 'city') & (spearman['group'] == 'global') & spearman['outcome'].isin(MECHANISM_OUTCOMES)][['outcome', 'column', 'spearman_rho', 'family_label']].copy()
    df = p.merge(s, on=['outcome', 'column', 'family_label'], how='inner')
    df['abs_rho'] = df['spearman_rho'].abs()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['abs_rho', 'ml_importance'])
    main = ax.inset_axes([0.13, 0.005, 0.82, 1.03])
    for outcome in MECHANISM_OUTCOMES:
        sub = df[df['outcome'] == outcome]
        main.scatter(sub['abs_rho'], sub['ml_importance'], s=7.8, facecolor=OUTCOME_COLORS.get(outcome, PALETTE['steel']), edgecolor='white', lw=0.22, alpha=0.86, label=OUTCOME_LABELS.get(outcome, outcome), zorder=3)
    main.plot([0, 1], [0, 1], color=PALETTE['axis'], lw=0.42, ls='--', alpha=0.68, zorder=2)
    if len(df) > 3:
        rho, _ = spearmanr(df['abs_rho'], df['ml_importance'], nan_policy='omit')
        if np.isfinite(rho):
            main.text(0.05, 0.95, f'rho={rho:.2f}', transform=main.transAxes, ha='left', va='top', fontsize=FS_ANNOT, color=PALETTE['text'])
    xmax = max(0.75, float(df['abs_rho'].max()) * 1.1) if len(df) else 0.75
    main.set_xlim(0, xmax)
    main.set_ylim(0, 1.08)
    main.set_xlabel('|Spearman rho|', fontsize=FS_AXIS_LABEL, labelpad=1.1, color=PALETTE['axis'])
    main.set_ylabel('Norm. ML importance', fontsize=FS_AXIS_LABEL, labelpad=1.1, color=PALETTE['axis'])
    main.yaxis.set_label_coords(-0.148, 0.5)
    main.tick_params(axis='both', labelsize=FS_AXIS_TICK, length=2.2, width=0.45, colors=PALETTE['axis'], pad=1.5)
    for spine in main.spines.values():
        spine.set_color(PALETTE['axis'])
        spine.set_linewidth(0.7)
    main.grid(False)
    main.legend(loc='lower right', fontsize=FS_LEGEND, markerscale=0.65, handletextpad=0.14, borderpad=0.05, labelspacing=0.1, frameon=False)

def build_figure(data: dict[str, pd.DataFrame]) -> mpl.figure.Figure:
    fig = plt.figure(figsize=(7.25, 7.05))
    gs = fig.add_gridspec(3, 3, left=0.047, right=0.985, top=0.972, bottom=0.055, wspace=0.34, hspace=0.56, height_ratios=[1.15, 1.04, 1.03], width_ratios=[1.0, 1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_d = fig.add_subplot(gs[0, 2])
    ax_e = fig.add_subplot(gs[1, 0])
    ax_f = fig.add_subplot(gs[1, 1])
    ax_g = fig.add_subplot(gs[2, 0])
    ax_j = fig.add_subplot(gs[2, 1:3])
    pos_d = ax_d.get_position()
    d_width_base = pos_d.width * 1.44
    d_width = d_width_base * 1.25
    d_height = pos_d.height * 1.2
    right_group_shift = -0.04
    middle_row_shift = 0.03
    df_width_scale = 1.07
    f_x0 = pos_d.x1 - d_width_base + right_group_shift
    d_x0 = f_x0
    d_right = d_x0 + d_width
    d_width_expanded = d_width * df_width_scale
    d_x0_expanded = d_right - d_width_expanded
    ax_d.set_position([d_x0_expanded, pos_d.y1 - d_height, d_width_expanded, d_height])
    pos_f = ax_f.get_position()
    panel_b_width = 0.44552751953125
    panel_b_height = d_height * 1.15
    f_right = f_x0 + panel_b_width
    panel_f_width = panel_b_width * df_width_scale
    f_x0_expanded = f_right - panel_f_width
    ax_f.set_position([f_x0_expanded, pos_f.y1 - panel_b_height + middle_row_shift, panel_f_width, panel_b_height])
    plot_driver_chord(ax_a, data['predictors'])
    plot_vif_forest(ax_d, data['predictors'], data['vif_city'], data['vif_zone'])
    plot_triangular_correlation(ax_e, data['predictors'], data['city_master'], data['zone_master'])
    plot_performance_forest(ax_f, data['performance'], align_ratio=d_width_base / panel_b_width)
    main_g = plot_joint_prediction(ax_g, data['predictions'], data['performance'])
    stability_labels = plot_stability_summary(ax_j, data['stability'], align_ratio=d_width_base / panel_b_width)
    fig.canvas.draw()
    pos_a_final = ax_a.get_position()
    pos_e_final = ax_e.get_position()
    ax_e.set_position([pos_a_final.x0, pos_e_final.y0 + middle_row_shift, pos_e_final.width, pos_e_final.height])
    pos_e_final = ax_e.get_position()
    pos_f_final = ax_f.get_position()
    pos_g_final = ax_g.get_position()
    bottom_row_shift = 0.05
    ax_g.set_position([pos_e_final.x0, pos_g_final.y0 + bottom_row_shift, pos_e_final.width, pos_g_final.height])
    pos_g_final = ax_g.get_position()
    bottom_right = 0.985
    j_width = pos_e_final.width
    k_width = pos_g_final.width
    equal_gap = max(0.018, (bottom_right - pos_g_final.x1 - j_width - k_width) / 2)
    j_shift = 0.06
    j_x0 = pos_g_final.x1 + equal_gap + j_shift
    k_x0 = j_x0 + j_width + equal_gap
    ax_j.set_position([j_x0, pos_g_final.y0, j_width, pos_g_final.height])
    ax_k = fig.add_axes([k_x0, pos_g_final.y0, k_width, pos_g_final.height])
    plot_importance_concordance(ax_k, data)
    fig.canvas.draw()
    align_panel_header_gaps(fig, (ax_a, 'a', 'Driver system'), [(ax_d, 'b', 'Exact VIF forest'), (ax_e, 'c', 'City-zone correlation matrix'), (ax_f, 'd', 'Predictive skill forest'), (ax_g, 'e', 'Global city-level ELI prediction'), (ax_j, 'f', 'Top-family stability'), (ax_k, 'g', 'Spearman-ML concordance')])
    fig.canvas.draw()
    for source_label, target_label in [('a', 'b'), ('Driver system', 'Exact VIF forest')]:
        source = next((text for text in ax_a.texts if text.get_text() == source_label))
        target = next((text for text in ax_d.texts if text.get_text() == target_label))
        target_y = target.get_transform().transform(target.get_position())[1]
        source_x = source.get_transform().transform(source.get_position())[0]
        source.set_y(source.get_transform().inverted().transform((source_x, target_y))[1])
    renderer = fig.canvas.get_renderer()
    xlabel_bbox = main_g.xaxis.label.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
    text_y = 0.5 * (xlabel_bbox.y0 + xlabel_bbox.y1)
    j_pos_final = ax_j.get_position()
    text_style = {'fontsize': FS_AXIS_LABEL, 'color': PALETTE['axis'], 'ha': 'center', 'va': 'center'}
    summary_x = j_pos_final.x0 + stability_labels['summary_x'] * j_pos_final.width
    summary_line_gap = FS_AXIS_LABEL / 72.0 / fig.get_figheight() * 1.18
    fig.text(summary_x, text_y, f"Top-1 same {stability_labels['stable']:.0f}%", **text_style)
    fig.text(summary_x, text_y - summary_line_gap, f"Mean top-3 overlap {stability_labels['overlap']:.1f}", **text_style)
    force_all_text_black(fig)
    return fig

def save_figure(fig: mpl.figure.Figure, basename: Path) -> None:
    for ext in ['png', 'tiff', 'pdf', 'svg']:
        kwargs = {'bbox_inches': 'tight', 'facecolor': 'white'}
        if ext in {'png', 'tiff'}:
            kwargs['dpi'] = 600
        fig.savefig(basename.with_suffix(f'.{ext}'), **kwargs)

def main() -> None:
    data = load_inputs()
    fig = build_figure(data)
    out_base = OUT_DIR / 'fig_q3_1_driver_model_reliability_scidraw_upgrade'
    save_figure(fig, out_base)
    plt.close(fig)
    print(f'Saved {out_base}.png/.tiff/.pdf/.svg')

def _build_manuscript_figure():
    get = _model_tables(zone=True)
    data = load_inputs()
    for key, value in {"performance": "rf_cv_performance", "predictions": "predictions", "sample": "sample", "permutation": "permutation", "spearman": "spearman"}.items():
        data[key] = get(value)
    data["zone_master"] = _read_table("shared/Study_city_zone_analysis.csv")
    data["robustness"] = _read_table("Extended_Data_Fig_3/robustness/Extended_Data_Fig_3_robustness_summary.csv")
    data["stability"] = _read_table("Extended_Data_Fig_3/robustness/Extended_Data_Fig_3_robustness_family_top3_stability.csv")
    return build_figure(data), None


if __name__ == "__main__":
    _configure_land_functions(sys.modules[__name__])
    _figure, _bbox = _build_manuscript_figure()
    _save_figure(_figure, _bbox, 'Extended_Data_Fig_3', _figure_options)
