"""Plot Fig_6 using the packaged numerical tables or supplied local inputs."""
from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_support"))
from figure_cli import prepare_figure as _prepare_figure, save_figure as _save_figure
from figure_cli import read_table as _read_table, model_tables as _model_tables
from figure_cli import configure_land_functions as _configure_land_functions

_figure_options = _prepare_figure('Fig_6', 'analysis_tables') if __name__ == "__main__" else None

from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
import math
from copy import copy
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Circle, Ellipse, PathPatch, Polygon, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.transforms import ScaledTranslation
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['font.weight'] = 'regular'
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
OUT_DIR = output_dir('Fig_6')
OUT_DIR.mkdir(parents=True, exist_ok=True)
PALETTE = {'pale_apricot': '#FFE7CC', 'soft_orange': '#FFB66D', 'burnt_orange': '#F27C2B', 'warm_sand': '#E7C9A0', 'cool_mist': '#DCE6F2', 'light_sky': '#A7CFF2', 'cerulean': '#3B82D6', 'cobalt': '#0D47A1', 'text': '#000000', 'axis': '#000000', 'muted': '#5F6673', 'grid': '#E6E9EE', 'soft_grid': '#F4F6FA'}
CORE_COLOR = PALETTE['cerulean']
EXP_COLOR = PALETTE['burnt_orange']
PAIR_LINE = '#B8C1CC'
RESPONSE_ALE_COLOR = PALETTE['cobalt']
RESPONSE_PDP_COLOR = PALETTE['burnt_orange']
RESPONSE_CI_COLOR = '#DCE5F1'
FAMILY_COLORS = {'Green/blue space': PALETTE['light_sky'], 'Environmental risk': PALETTE['burnt_orange'], 'Built environment': PALETTE['cobalt'], 'Economic/population': PALETTE['cool_mist'], 'POI supply': PALETTE['soft_orange'], 'Non-motorized accessibility': PALETTE['warm_sand'], 'Road network': PALETTE['pale_apricot'], 'Terrain control': PALETTE['cerulean'], 'Location control': '#8D7DBB', 'Other/minor drivers': '#C8D0C1', 'Other': '#C8D0C1'}
PANEL_A_FAMILY_COLORS = {'Green/blue space': '#6BBDD7', 'Environmental risk': '#D81809', 'Built environment': '#3378B6', 'Economic/population': '#6A729B', 'POI supply': '#F1881B', 'Non-motorized accessibility': '#F0B17B', 'Road network': '#EBC99B', 'Terrain control': '#8A94B2', 'Location control': '#8172A8', 'Other/minor drivers': '#C8D0C1', 'Other': '#C8D0C1'}
DIVERGE = LinearSegmentedColormap.from_list('blue_orange_diverge', [PALETTE['burnt_orange'], PALETTE['pale_apricot'], '#FFFFFF', PALETTE['cool_mist'], PALETTE['cobalt']])
SEQ_BLUE = LinearSegmentedColormap.from_list('q3_seq_blue', ['#F6F8FB', PALETTE['cool_mist'], PALETTE['light_sky'], PALETTE['cerulean'], PALETTE['cobalt']])
BLUE_ORANGE_DENSITY = LinearSegmentedColormap.from_list('blue_orange_density', [PALETTE['pale_apricot'], PALETTE['cool_mist'], PALETTE['light_sky'], PALETTE['cerulean'], PALETTE['cobalt']])
OUTCOME_LABELS = {'eli50': 'ELI', 'positive_score50': 'Positive', 'negative_score50': 'Negative', 'mean50_beauty': 'Beauty', 'mean50_wealthy': 'Wealth', 'mean50_safer': 'Safety', 'mean50_livelier': 'Liveliness', 'mean50_boring': 'Boredom', 'mean50_depression': 'Depression'}
MATRIX_OUTCOMES = ['eli50', 'positive_score50', 'negative_score50', 'mean50_beauty', 'mean50_safer', 'mean50_livelier', 'mean50_boring', 'mean50_depression']
RAIN_OUTCOMES = ['eli50', 'positive_score50', 'negative_score50', 'mean50_beauty', 'mean50_depression']
PERF_OUTCOMES = ['eli50', 'positive_score50', 'negative_score50', 'mean50_beauty', 'mean50_depression']
ABBR_BY_COLUMN = {'green_fraction_mean': 'Green', 'water_fraction_mean': 'Water', 'building_density_mean': 'Built dens.', 'building_height_mean': 'Height', 'ntl_log1p_mean': 'NTL', 'population_density_mean': 'Pop. dens.', 'lat': 'Latitude', 'dem_mean': 'Elev.', 'slope_mean': 'Slope', 'pm25_2020_mean': 'PM2.5', 'lst_c_mean': 'LST', 'noise_mean': 'Noise', 'poi_education_mean': 'POI edu.', 'poi_entertainment_mean': 'POI ent.', 'poi_health_mean': 'POI health', 'poi_transport_mean': 'POI transp.', 'access_nonmotor_education_mean': 'Access edu.', 'access_nonmotor_entertainment_mean': 'Access ent.', 'access_nonmotor_health_mean': 'Access health', 'access_nonmotor_living_mean': 'Access living', 'access_nonmotor_transport_mean': 'Access transp.', 'road_total_mean': 'Road total', 'road_footway_mean': 'Footway', 'road_pedestrian_mean': 'Pedestrian', 'road_cycleway_mean': 'Cycleway'}
FS_LETTER = 15.6
FS_TITLE = 12.1
FS_AXIS = 9.9
FS_TICK = 9.0
FS_BODY = 9.0
FS_DENSE = 8.25
FS_SMALL = 7.65
FS_TABLE_HEADER = 8.2
FS_TABLE_BODY = 7.9
FS_TABLE_VALUE = 7.55
FS_TABLE_DELTA = 7.55
FS_MATRIX_TICK = 7.15
FS_PANEL_A_DENSE = 8.0
FS_PANEL_A_BODY = 8.25
FS_PANEL_A_CENTER = 10.5
FS_PANEL_A_COLORBAR = 7.55
plt.rcParams.update({'font.size': FS_BODY, 'axes.labelsize': FS_AXIS, 'xtick.labelsize': FS_TICK, 'ytick.labelsize': FS_TICK, 'legend.fontsize': FS_DENSE})

def read_csv(path: Path) -> pd.DataFrame:
    if not resource(path).exists():
        raise FileNotFoundError(path)
    return pd.read_csv(resource(path))

def load_data() -> dict[str, pd.DataFrame]:
    return {'family': read_csv(RUN_FORMAL / 'q3_formal_family.csv'), 'permutation': read_csv(RUN_FORMAL / 'q3_formal_permutation.csv'), 'performance': read_csv(RUN_FORMAL / 'q3_formal_rf_cv_performance.csv'), 'predictions': read_csv(RUN_FORMAL / 'q3_formal_predictions.csv'), 'family_contrast': read_csv(DERIVED / 'q3_core_expansion_family_contrast.csv'), 'variable_contrast': read_csv(DERIVED / 'q3_core_expansion_variable_contrast.csv'), 'zone_master': read_csv(DERIVED / 'q3_city_zone_driver_master.csv'), 'response_curves': read_csv(DERIVED / 'q3_core_expansion_response_curves.csv'), 'mechanism_regime': read_csv(DERIVED / 'q3_core_expansion_mechanism_regime.csv')}

def short_family(label: str) -> str:
    text = str(label)
    return text.replace('Green/blue space', 'Green/blue').replace('Environmental risk', 'Env. risk').replace('Built environment', 'Built env.').replace('Economic/population', 'Econ./pop.').replace('Non-motorized accessibility', 'Non-motor').replace('Terrain control', 'Terrain').replace('Location control', 'Location')

def alluvial_family_label(label: str) -> str:
    labels = {'Built environment': 'Built\nenv.', 'Economic/population': 'Econ./\npop.', 'Environmental risk': 'Env. risk', 'Green/blue space': 'Green/\nblue', 'Impervious control': 'Imperv.\ncontrol', 'Location control': 'Location', 'Non-motorized accessibility': 'Non-motor', 'Other/minor drivers': 'Other', 'POI supply': 'POI supply'}
    return labels.get(str(label), short_family(str(label)))

def outcome_label(outcome: str) -> str:
    return OUTCOME_LABELS.get(str(outcome), str(outcome))

def short_var(column: str) -> str:
    return ABBR_BY_COLUMN.get(str(column), str(column).replace('_mean', '').replace('_', ' '))

def family_color(label: str) -> str:
    return FAMILY_COLORS.get(str(label), '#C8D0C1')

def format_num(x: float, nd: int=2) -> str:
    if pd.isna(x):
        return ''
    return f'{x:.{nd}f}'

def despine(ax: mpl.axes.Axes, tick_size: float=FS_TICK, label_size: float=FS_AXIS) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.65)
    ax.spines['bottom'].set_linewidth(0.65)
    ax.tick_params(axis='both', labelsize=tick_size, width=0.65, length=2.4, color='black', pad=2)
    ax.xaxis.label.set_size(label_size)
    ax.yaxis.label.set_size(label_size)

def axes_y_points(ax: mpl.axes.Axes, points: float) -> float:
    fig_h = ax.figure.get_figheight()
    ax_h = max(ax.get_position().height * fig_h, 1e-09)
    return points / 72.0 / ax_h

def panel_title(ax: mpl.axes.Axes, letter: str, title: str, y: float=1.075, letter_x: float=-0.085, title_x: float=-0.01, letter_size: float=FS_LETTER, title_size: float=FS_TITLE) -> None:
    ax.text(letter_x, y + 0.01, letter, transform=ax.transAxes, ha='left', va='bottom', fontsize=letter_size, fontweight='bold', clip_on=False)
    ax.text(title_x, y + 0.004, title, transform=ax.transAxes, ha='left', va='bottom', fontsize=title_size, fontweight='regular', clip_on=False)

def draw_bezier_band(ax: mpl.axes.Axes, x0: float, y0: float, x1: float, y1: float, width: float, color: str, alpha: float=0.34, zorder: int=1) -> None:
    ctrl = (x1 - x0) * 0.48
    dx0 = 0
    dy = width / 2
    verts = [(x0 + dx0, y0 + dy), (x0 + ctrl, y0 + dy), (x1 - ctrl, y1 + dy), (x1, y1 + dy), (x1, y1 - dy), (x1 - ctrl, y1 - dy), (x0 + ctrl, y0 - dy), (x0 + dx0, y0 - dy), (x0 + dx0, y0 + dy)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
    patch = PathPatch(MplPath(verts, codes), facecolor=color, edgecolor='none', alpha=alpha, zorder=zorder)
    ax.add_patch(patch)

def add_table_stripes(ax: mpl.axes.Axes, y_positions: list[float], height: float, color: str='#F3F6FA') -> None:
    for i, y in enumerate(y_positions):
        if i % 2 == 1:
            ax.axhspan(y - height / 2, y + height / 2, color=color, zorder=0)

def kde_curve(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 8 or np.nanstd(values) < 1e-09:
        return np.zeros_like(grid)
    try:
        dens = gaussian_kde(values)(grid)
    except Exception:
        return np.zeros_like(grid)
    if np.nanmax(dens) <= 0:
        return np.zeros_like(grid)
    return dens / np.nanmax(dens)

def add_group_ellipse(ax: mpl.axes.Axes, x: np.ndarray, y: np.ndarray, color: str, n_std: float=1.35) -> None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return
    pts = np.column_stack([x[mask], y[mask]])
    cov = np.cov(pts, rowvar=False)
    if not np.all(np.isfinite(cov)):
        return
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 0, None)
    if np.nanmax(vals) <= 1e-12:
        return
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(vals)
    center = pts.mean(axis=0)
    ax.add_patch(Ellipse(xy=center, width=width, height=height, angle=angle, facecolor=color, edgecolor=color, lw=0.9, alpha=0.12, zorder=2))

def polar_to_cart(cx: float, cy: float, r: float, theta_deg: float) -> tuple[float, float]:
    theta = np.deg2rad(theta_deg)
    return (cx + r * np.cos(theta), cy + r * np.sin(theta))

def polar_to_xy(theta_deg: float, r: float) -> tuple[float, float]:
    theta = math.radians(theta_deg)
    return (r * math.cos(theta), r * math.sin(theta))

def axis_circle_xscale(ax: mpl.axes.Axes) -> float:
    bbox = ax.get_position()
    fig_w, fig_h = ax.figure.get_size_inches()
    return bbox.height * fig_h / (bbox.width * fig_w)

def polar_to_cart_circle(cx: float, cy: float, r: float, theta_deg: float, xscale: float) -> tuple[float, float]:
    theta = np.deg2rad(theta_deg)
    return (cx + r * xscale * np.cos(theta), cy + r * np.sin(theta))

def add_true_circle_wedge(ax: mpl.axes.Axes, cx: float, cy: float, radius: float, theta1: float, theta2: float, width: float, xscale: float, **kwargs) -> None:
    n = max(16, int(abs(theta2 - theta1) * 1.4))
    theta = np.deg2rad(np.linspace(theta1, theta2, n))
    inner = max(radius - width, 0.0)
    outer_xy = np.column_stack([cx + radius * xscale * np.cos(theta), cy + radius * np.sin(theta)])
    inner_xy = np.column_stack([cx + inner * xscale * np.cos(theta[::-1]), cy + inner * np.sin(theta[::-1])])
    ax.add_patch(Polygon(np.vstack([outer_xy, inner_xy]), closed=True, **kwargs))

def text_rotation(theta_deg: float) -> tuple[float, str]:
    theta_deg = theta_deg % 360
    rot = theta_deg
    ha = 'left'
    if 90 < theta_deg < 270:
        rot = theta_deg + 180
        ha = 'right'
    return (rot, ha)

def row_header(ax: mpl.axes.Axes, x: float, y: float, text: str, **kwargs) -> None:
    ax.text(x, y, text, fontsize=FS_DENSE, fontweight='bold', ha='center', va='bottom', **kwargs)

def add_endpoint_table_header(ax: mpl.axes.Axes, xs: list[float], labels: list[str], y: float) -> None:
    for x, label in zip(xs, labels):
        row_header(ax, x, y, label)

def finite_max(*arrays: pd.Series | np.ndarray, fallback: float=1.0) -> float:
    vals = []
    for arr in arrays:
        a = np.asarray(arr, dtype=float)
        vals.extend(a[np.isfinite(a)].tolist())
    return max(vals) if vals else fallback

def plot_panel_a(ax: mpl.axes.Axes, data: dict[str, pd.DataFrame]) -> None:
    a_body_size = FS_PANEL_A_BODY
    a_dense_size = FS_PANEL_A_DENSE
    a_colorbar_tick_size = FS_PANEL_A_COLORBAR
    panel_title(ax, 'a', 'Core-expansion mechanism overview', y=1.08, letter_x=0.124, title_x=0.188)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    var = data['variable_contrast']
    sub = var[(var['contrast_scope'] == 'global') & var['outcome'].isin(MATRIX_OUTCOMES)].copy()
    var_df = sub.groupby(['family_label', 'column'], as_index=False).agg(core_importance_share=('core_importance_share', 'mean'), expansion_importance_share=('expansion_importance_share', 'mean'), abs_core_minus_expansion_share=('abs_core_minus_expansion_share', 'mean'))
    if var_df.empty:
        return
    family_base = ['Green/blue space', 'Environmental risk', 'Built environment', 'Economic/population', 'POI supply', 'Non-motorized accessibility', 'Road network', 'Terrain control', 'Location control']
    family_order = [fam for fam in family_base if fam in set(var_df['family_label'])]
    family_order.extend([fam for fam in var_df['family_label'].dropna().unique().tolist() if fam not in family_order])
    var_order = {'water_fraction_mean': 0, 'green_fraction_mean': 1, 'noise_mean': 2, 'lst_c_mean': 3, 'pm25_2020_mean': 4, 'building_height_mean': 5, 'building_density_mean': 6, 'population_density_mean': 7, 'ntl_log1p_mean': 8, 'poi_transport_mean': 9, 'poi_health_mean': 10, 'poi_entertainment_mean': 11, 'poi_education_mean': 12, 'access_nonmotor_education_mean': 13, 'access_nonmotor_entertainment_mean': 14, 'access_nonmotor_health_mean': 15, 'access_nonmotor_living_mean': 16, 'access_nonmotor_transport_mean': 17, 'road_total_mean': 18, 'road_footway_mean': 19, 'road_pedestrian_mean': 20, 'road_cycleway_mean': 21, 'dem_mean': 22, 'slope_mean': 23, 'lat': 24}
    var_df['family_label'] = pd.Categorical(var_df['family_label'], categories=family_order, ordered=True)
    var_df['var_order'] = var_df['column'].map(var_order).fillna(999).astype(float)
    var_df = var_df.sort_values(['family_label', 'var_order', 'column']).reset_index(drop=True)
    families = var_df.groupby('family_label', observed=True).size().reindex(family_order).dropna()
    a_main_y_shift = -0.25
    center = (-0.05, -0.06 + a_main_y_shift)
    root_r = 0.2
    hub_r = 0.42
    leaf_r = 0.68
    band_inner, band_outer = (0.5, 0.78)
    core_ring, expansion_ring = (0.86, 0.91)
    ring_width = 0.026
    label_r = 1.035
    importance_norm = Normalize(vmin=0, vmax=10)

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
        theta1, theta2 = (cur, cur + span)
        theta_mid = (theta1 + theta2) / 2
        leaf_angles = [theta_mid] if count == 1 else np.linspace(theta1 + span / (count + 1), theta2 - span / (count + 1), count).tolist()
        family_angles[str(fam)] = (theta1, theta2, theta_mid, leaf_angles)
        color = PANEL_A_FAMILY_COLORS.get(str(fam), family_color(str(fam)))
        annular_segment(theta1, theta2, band_inner, band_outer, facecolor=color, edgecolor='none', alpha=0.075, zorder=0)
        annular_segment(theta1, theta2, band_outer + 0.001, band_outer + 0.035, facecolor=color, edgecolor='white', lw=0.35, alpha=0.96, zorder=3)
        hx, hy = xy(theta_mid, hub_r)
        ax.plot([center[0], hx], [center[1], hy], color=color, lw=0.55, alpha=0.44, zorder=1)
        ax.scatter(hx, hy, s=50.176, color=color, edgecolor='white', linewidth=0.35, zorder=4)
        cur = theta2 + gap
    for fam_value, one in var_df.groupby('family_label', observed=True, sort=False):
        fam = str(fam_value)
        theta1, theta2, theta_mid, leaf_angles = family_angles[fam]
        color = PANEL_A_FAMILY_COLORS.get(fam, family_color(fam))
        hx, hy = xy(theta_mid, hub_r)
        angle_width = max(0.85, min(1.95, (theta2 - theta1) / (len(one) + 2) * 0.42))
        for theta, (_, row) in zip(leaf_angles, one.iterrows()):
            lx, ly = xy(theta, leaf_r)
            ax.plot([hx, lx], [hy, ly], color=color, lw=0.5, alpha=0.46, zorder=1)
            ax.scatter(lx, ly, s=43.008, color=color, edgecolor='white', linewidth=0.34, zorder=5)
            core_pct = float(row['core_importance_share']) * 100
            expansion_pct = float(row['expansion_importance_share']) * 100
            annular_segment(theta - angle_width, theta + angle_width, core_ring - ring_width, core_ring, facecolor=SEQ_BLUE(importance_norm(core_pct)), edgecolor='white', lw=0.18, zorder=4)
            annular_segment(theta - angle_width, theta + angle_width, expansion_ring - ring_width, expansion_ring, facecolor=SEQ_BLUE(importance_norm(expansion_pct)), edgecolor='white', lw=0.18, zorder=4)
            tx, ty = xy(theta, label_r)
            ha, rot = label_style(theta)
            ax.text(tx, ty, short_var(row['column']), ha=ha, va='center', rotation=rot, rotation_mode='anchor', fontsize=a_dense_size, color=PALETTE['text'], zorder=6)
    ax.add_patch(Circle(center, root_r, facecolor='#FFFFFF', edgecolor=PALETTE['grid'], lw=0.55, zorder=7))
    ax.text(center[0], center[1] + 0.07, '25', ha='center', va='center', fontsize=FS_PANEL_A_CENTER, color=PALETTE['text'], zorder=8)
    ax.text(center[0], center[1] - 0.06, 'final\ndrivers', ha='center', va='center', fontsize=a_body_size, linespacing=0.88, color=PALETTE['text'], zorder=8)
    ax.text(-0.05, -0.49 + a_main_y_shift, 'Importance share (%)', ha='center', va='bottom', fontsize=a_body_size, color=PALETTE['text'], zorder=8)
    cax = ax.inset_axes([-0.35, -0.55 + a_main_y_shift, 0.6, 0.034], transform=ax.transData)
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=SEQ_BLUE, norm=importance_norm, orientation='horizontal', ticks=[0, 5, 10])
    cax.tick_params(labelsize=a_colorbar_tick_size, length=1.5, width=0.35, pad=0.8, colors=PALETTE['text'])
    cbar.outline.set_linewidth(0.38)
    ax.text(0.48, -0.585 + a_main_y_shift, 'inner: core\nouter: expansion', ha='left', va='center', fontsize=a_dense_size, linespacing=1.18, color=PALETTE['text'], zorder=8)
    legend_cols_x = [-0.93, -0.26, 0.349]
    legend_y0 = -0.765 + a_main_y_shift
    legend_row_gap = 0.102
    for idx, (fam, count_value) in enumerate(families.items()):
        col = idx % 3
        row = idx // 3
        x = legend_cols_x[col]
        y = legend_y0 - row * legend_row_gap
        color = PANEL_A_FAMILY_COLORS.get(str(fam), family_color(str(fam)))
        ax.plot([x, x + 0.055], [y, y], color=color, lw=1.45, solid_capstyle='butt', zorder=8, clip_on=False)
        ax.text(x + 0.068, y, f'{short_family(str(fam))} ({int(count_value)})', ha='left', va='center', fontsize=a_dense_size, color=PALETTE['text'], zorder=8)
    ax.set_xlim(-1.62, 1.58)
    ax.set_ylim(-1.03, 1.03)

def plot_panel_b(ax: mpl.axes.Axes, data: dict[str, pd.DataFrame]) -> None:
    panel_title(ax, 'd', 'Core-expansion perceptual contrast', y=0.9348, letter_x=0.0245, title_x=0.091)
    pred = data['predictions']
    z = pred[(pred['table'] == 'city_zone') & (pred['group'] == 'global') & pred['outcome'].isin(RAIN_OUTCOMES)].copy()
    rows = [outcome for outcome in RAIN_OUTCOMES if outcome in z['outcome'].unique()]
    b_main_y_shift = -0.01
    row_y = {outcome: y + b_main_y_shift for outcome, y in zip(rows, np.linspace(0.815, 0.34, len(rows)))}
    offset = {'core': 0.026, 'expansion': -0.026}
    marker_style = {'core': dict(facecolor=CORE_COLOR, edgecolor=CORE_COLOR, lw=0.0), 'expansion': dict(facecolor=EXP_COLOR, edgecolor=EXP_COLOR, lw=0.0)}
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    ax.set_xlabel('')
    table_left, table_right = (0.018, 1.025)
    forest_left, forest_right = (0.205, 0.625)
    forest_min, forest_max = (-0.2, 1.0)
    axis_y = 0.2609
    header_y = 0.898
    tick_len = axes_y_points(ax, 2.4)

    def map_x(value: float) -> float:
        value = float(np.clip(value, forest_min, forest_max))
        return forest_left + (value - forest_min) / (forest_max - forest_min) * (forest_right - forest_left)
    stats: dict[tuple[str, str], tuple[float, float]] = {}
    deltas: dict[str, float] = {}
    for outcome in rows:
        one = z[z['outcome'] == outcome]
        for zone in ['core', 'expansion']:
            vals = one.loc[one['zone_label'] == zone, 'observed'].dropna().to_numpy(float)
            stats[outcome, zone] = (float(np.nanmean(vals)), float(np.nanstd(vals, ddof=1))) if len(vals) else (np.nan, np.nan)
        pair = one.pivot_table(index='id_1', columns='zone_label', values='observed', aggfunc='mean').dropna(subset=['core', 'expansion'])
        deltas[outcome] = float(np.nanmean(pair['core'] - pair['expansion'])) if len(pair) else np.nan
    for idx, outcome in enumerate(rows):
        y = row_y[outcome]
        if idx % 2 == 1:
            ax.add_patch(Rectangle((table_left - 0.01, y - 0.052), table_right - table_left + 0.01, 0.104, transform=ax.transAxes, facecolor='#E1EEFA', edgecolor='none', alpha=0.4, clip_on=False, zorder=0))
    header_font = dict(fontsize=FS_TABLE_HEADER, fontweight='bold', color=PALETTE['text'])
    body_font = dict(fontsize=FS_TABLE_BODY, color=PALETTE['text'])
    ax.text(0.03, header_y, 'Outcome', transform=ax.transAxes, ha='left', va='bottom', **header_font)
    right_cols = {'core': 0.705, 'expansion': 0.85, 'delta': 0.973}
    ax.text(right_cols['core'], header_y, 'Core', transform=ax.transAxes, ha='center', va='bottom', **header_font)
    ax.text(right_cols['expansion'], header_y, 'Expansion', transform=ax.transAxes, ha='center', va='bottom', **header_font)
    ax.text(right_cols['delta'], header_y, 'Δ', transform=ax.transAxes, ha='center', va='bottom', **header_font)
    for zone in ['core', 'expansion']:
        style = marker_style[zone]
        ax.scatter([right_cols[zone]], [header_y - 0.03], marker='o', s=15.698, facecolors=style['facecolor'], edgecolors=style['edgecolor'], linewidths=style['lw'], transform=ax.transAxes, zorder=5)
    for outcome in rows:
        y = row_y[outcome]
        ax.text(0.03, y, outcome_label(outcome), transform=ax.transAxes, ha='left', va='center', **body_font)
        for zone in ['core', 'expansion']:
            mean, sd = stats[outcome, zone]
            ax.text(right_cols[zone], y, f'{mean:.3f}\n({sd:.3f})', transform=ax.transAxes, ha='center', va='center', fontsize=FS_TABLE_VALUE, linespacing=1.24, color=PALETTE['text'])
        ax.text(right_cols['delta'], y, f'{deltas[outcome]:+.3f}', transform=ax.transAxes, ha='center', va='center', fontsize=FS_TABLE_DELTA, color=PALETTE['text'])
    ref_x = map_x(0)
    ax.plot([ref_x, ref_x], [axis_y, header_y - 0.018], color=PALETTE['axis'], lw=0.66, ls=(0, (3.2, 3.2)), transform=ax.transAxes, zorder=1)
    for outcome in rows:
        for zone in ['core', 'expansion']:
            mean, sd = stats[outcome, zone]
            if not np.isfinite(mean):
                continue
            y = row_y[outcome] + offset[zone]
            x = map_x(mean)
            left = map_x(mean - sd)
            right = map_x(mean + sd)
            style = marker_style[zone]
            ax.plot([left, right], [y, y], color=PALETTE['axis'], lw=0.58, transform=ax.transAxes, zorder=3)
            ax.scatter([x], [y], marker='o', s=20.612, facecolors=style['facecolor'], edgecolors=style['edgecolor'], linewidths=style['lw'], transform=ax.transAxes, zorder=4)
    ax.plot([forest_left, forest_right], [axis_y, axis_y], color=PALETTE['axis'], lw=0.62, transform=ax.transAxes, zorder=5)
    tick_label_y = axis_y - 0.0195
    for tick in [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        x = map_x(tick)
        ax.plot([x, x], [axis_y, axis_y - tick_len], color=PALETTE['axis'], lw=0.46, transform=ax.transAxes, zorder=5)
        ax.text(x, tick_label_y, f'{tick:.1f}', transform=ax.transAxes, ha='center', va='top', fontsize=FS_TICK, color=PALETTE['text'])
    ax.text((forest_left + forest_right) / 2, 0.1713, 'Observed score (mean ± SD)', transform=ax.transAxes, ha='center', va='bottom', fontsize=FS_AXIS, color=PALETTE['axis'], clip_on=False)

def plot_panel_c(ax: mpl.axes.Axes, data: dict[str, pd.DataFrame]) -> None:
    panel_title(ax, 'c', 'Core-expansion predictive skill')
    perf = data['performance']
    sub = perf[(perf['table'] == 'city_zone') & perf['group'].isin(['core', 'expansion']) & perf['outcome'].isin(PERF_OUTCOMES)].copy()
    rows = [outcome for outcome in PERF_OUTCOMES if outcome in sub['outcome'].unique()]
    row_y = {outcome: y for outcome, y in zip(rows, np.linspace(0.815, 0.34, len(rows)))}
    offset = {'core': 0.026, 'expansion': -0.026}
    marker_style = {'core': dict(facecolor=CORE_COLOR, edgecolor=CORE_COLOR, lw=0.0), 'expansion': dict(facecolor=EXP_COLOR, edgecolor=EXP_COLOR, lw=0.0)}
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    ax.set_xlabel('Cross-validated R2 (mean ± SD)', fontsize=FS_DENSE, color=PALETTE['axis'], labelpad=1.6)
    ax.xaxis.set_label_coords(0.41, 0.025)
    table_left, table_right = (0.018, 0.985)
    forest_left, forest_right = (0.205, 0.625)
    forest_min, forest_max = (0.7, 0.94)
    axis_y = 0.245
    header_y = 0.895

    def map_x(value: float) -> float:
        value = float(np.clip(value, forest_min, forest_max))
        return forest_left + (value - forest_min) / (forest_max - forest_min) * (forest_right - forest_left)
    stats: dict[tuple[str, str], tuple[float, float]] = {}
    deltas: dict[str, float] = {}
    for outcome in rows:
        one = sub[sub['outcome'] == outcome]
        for group in ['core', 'expansion']:
            g = one[one['group'] == group]
            if g.empty:
                stats[outcome, group] = (np.nan, np.nan)
            else:
                r = g.iloc[0]
                stats[outcome, group] = (float(r['r2_mean']), float(r['r2_sd']) if np.isfinite(r['r2_sd']) else 0.0)
        core_mean = stats[outcome, 'core'][0]
        exp_mean = stats[outcome, 'expansion'][0]
        deltas[outcome] = core_mean - exp_mean if np.isfinite(core_mean) and np.isfinite(exp_mean) else np.nan
    for idx, outcome in enumerate(rows):
        y = row_y[outcome]
        if idx % 2 == 1:
            ax.add_patch(Rectangle((table_left, y - 0.052), table_right - table_left, 0.104, transform=ax.transAxes, facecolor='#E1EEFA', edgecolor='none', alpha=0.4, zorder=0))
    header_font = dict(fontsize=FS_TABLE_HEADER, fontweight='bold', color=PALETTE['text'])
    body_font = dict(fontsize=FS_TABLE_BODY, color=PALETTE['text'])
    ax.text(0.03, header_y, 'Outcome', transform=ax.transAxes, ha='left', va='bottom', **header_font)
    right_cols = {'core': 0.705, 'expansion': 0.85, 'delta': 0.973}
    ax.text(right_cols['core'], header_y, 'Core', transform=ax.transAxes, ha='center', va='bottom', **header_font)
    ax.text(right_cols['expansion'], header_y, 'Expansion', transform=ax.transAxes, ha='center', va='bottom', **header_font)
    ax.text(right_cols['delta'], header_y, 'ΔR2', transform=ax.transAxes, ha='center', va='bottom', **header_font)
    for outcome in rows:
        y = row_y[outcome]
        ax.text(0.03, y, outcome_label(outcome), transform=ax.transAxes, ha='left', va='center', **body_font)
        for group in ['core', 'expansion']:
            mean, sd = stats[outcome, group]
            ax.text(right_cols[group], y, f'{mean:.3f}\n({sd:.3f})', transform=ax.transAxes, ha='center', va='center', fontsize=FS_TABLE_VALUE, linespacing=1.24, color=PALETTE['text'])
        ax.text(right_cols['delta'], y, f'{deltas[outcome]:+.3f}', transform=ax.transAxes, ha='center', va='center', fontsize=FS_TABLE_DELTA, color=PALETTE['text'])
    ref_x = map_x(0.8)
    ax.plot([ref_x, ref_x], [axis_y, header_y - 0.018], color=PALETTE['axis'], lw=0.66, ls=(0, (3.2, 3.2)), transform=ax.transAxes, zorder=1)
    for outcome in rows:
        for group in ['core', 'expansion']:
            mean, sd = stats[outcome, group]
            if not np.isfinite(mean):
                continue
            y = row_y[outcome] + offset[group]
            x = map_x(mean)
            left = map_x(mean - sd)
            right = map_x(mean + sd)
            style = marker_style[group]
            ax.plot([left, right], [y, y], color=PALETTE['axis'], lw=0.58, transform=ax.transAxes, zorder=3)
            ax.scatter([x], [y], marker='o', s=10.57, facecolors=style['facecolor'], edgecolors=style['edgecolor'], linewidths=style['lw'], transform=ax.transAxes, zorder=4)
    ax.plot([forest_left, forest_right], [axis_y, axis_y], color=PALETTE['axis'], lw=0.62, transform=ax.transAxes, zorder=5)
    for tick in [0.7, 0.8, 0.9]:
        x = map_x(tick)
        ax.plot([x, x], [axis_y, axis_y - 0.017], color=PALETTE['axis'], lw=0.46, transform=ax.transAxes, zorder=5)
        ax.text(x, axis_y - 0.043, f'{tick:.1f}', transform=ax.transAxes, ha='center', va='top', fontsize=FS_TABLE_BODY, color=PALETTE['text'])
    legend_x = forest_left
    for group, x in zip(['core', 'expansion'], [legend_x + 0.014, legend_x + 0.2]):
        y = 0.09
        style = marker_style[group]
        ax.scatter([x], [y], marker='o', s=8.05, facecolors=style['facecolor'], edgecolors=style['edgecolor'], linewidths=style['lw'], transform=ax.transAxes, zorder=5)
        ax.text(x + 0.02, y, group.capitalize(), transform=ax.transAxes, ha='left', va='center', fontsize=FS_TABLE_BODY, color=PALETTE['text'])

def plot_panel_d(ax: mpl.axes.Axes, data: dict[str, pd.DataFrame]) -> None:
    panel_title(ax, 'b', 'Core-expansion driver association')
    title_ax = ax
    title_ax.axis('off')
    ax = title_ax.inset_axes([-0.07, 0.0, 1.1235, 1.0])
    ax.patch.set_alpha(0)
    zone_master = data['zone_master']
    row_order = ['building_density_mean', 'building_height_mean', 'ntl_log1p_mean', 'population_density_mean', 'pm25_2020_mean', 'lst_c_mean', 'noise_mean', 'green_fraction_mean', 'water_fraction_mean', 'lat', 'access_nonmotor_education_mean', 'access_nonmotor_entertainment_mean', 'access_nonmotor_health_mean', 'access_nonmotor_living_mean', 'access_nonmotor_transport_mean', 'poi_education_mean', 'poi_entertainment_mean', 'poi_health_mean', 'poi_transport_mean', 'road_total_mean', 'road_footway_mean', 'road_pedestrian_mean', 'road_cycleway_mean', 'dem_mean', 'slope_mean']
    cols = [c for c in row_order if c in zone_master.columns]
    core_master = zone_master[zone_master['zone_label'].eq('core')]
    expansion_master = zone_master[zone_master['zone_label'].eq('expansion')]
    if len(cols) < 3 or core_master.empty or expansion_master.empty:
        ax.axis('off')
        return
    corr_core = core_master[cols].corr(method='spearman', min_periods=50).to_numpy(float)
    corr_expansion = expansion_master[cols].corr(method='spearman', min_periods=50).to_numpy(float)
    n = len(cols)
    lower = np.full((n, n), np.nan)
    lower[np.tril_indices(n)] = corr_expansion[np.tril_indices(n)]
    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)
    im = ax.imshow(lower, cmap=DIVERGE, norm=norm, aspect='auto', zorder=1)
    for i in range(n):
        for j in range(i + 1, n):
            r = corr_core[i, j]
            if np.isfinite(r):
                color = PALETTE['cobalt'] if r >= 0 else PALETTE['burnt_orange']
                ax.scatter(j, i, s=(1.8 + 20.0 * abs(float(r))) * 2.028, color=color, edgecolor='none', alpha=0.68, zorder=4)
    labels = [short_var(c) for c in cols]
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=90, ha='center', va='top', fontsize=FS_MATRIX_TICK)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(labels, fontsize=FS_MATRIX_TICK)
    ax.tick_params(length=0, pad=0.7)
    for k in range(n + 1):
        ax.axhline(k - 0.5, color='white', lw=0.8, zorder=3)
        ax.axvline(k - 0.5, color='white', lw=0.8, zorder=3)
    d_margin = n * 0.03
    ax.set_xlim(-0.5 - d_margin, n - 0.5 + d_margin)
    ax.set_ylim(n - 0.5 + d_margin, -0.5 - d_margin)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cax = ax.inset_axes([1.025, 0.18, 0.03, 0.62])
    cbar = ax.figure.colorbar(im, cax=cax, orientation='vertical', ticks=[-0.5, 0, 0.5])
    cbar.outline.set_linewidth(0.38)
    cbar.ax.tick_params(labelsize=FS_SMALL, width=0.38, length=1.8, pad=1.0, colors=PALETTE['text'])
    cbar.set_label('Spearman r', fontsize=FS_DENSE, labelpad=2.0)
    ax.text(0.0, 1.015, 'lower expansion / upper core', transform=ax.transAxes, ha='left', va='bottom', fontsize=FS_DENSE, color=PALETTE['text'], clip_on=False)

def plot_panel_e(ax: mpl.axes.Axes, data: dict[str, pd.DataFrame]) -> None:
    panel_title(ax, 'c', 'Driver-level contrast forest', y=0.9264, letter_x=0.0257, title_x=0.1063)
    var = data['variable_contrast']
    sub = var[(var['contrast_scope'] == 'global') & (var['outcome'] == 'eli50')].copy()
    sub = sub.sort_values('abs_core_minus_expansion_share', ascending=False).head(11)
    if sub.empty:
        ax.axis('off')
        return
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    label_x = 0.018
    forest_left, forest_right = (0.155, 0.668)
    forest_left = forest_right - (forest_right - forest_left) * 0.9
    axis_y = 0.166
    header_y = 0.885
    tick_len = axes_y_points(ax, 2.4)
    row_top, row_bottom = (0.8, 0.235)
    rows = list(sub['column'])
    row_y = {col: y for col, y in zip(rows, np.linspace(row_top, row_bottom, len(rows)))}
    row_gap = (row_top - row_bottom) / max(len(rows) - 1, 1)
    row_h = min(0.06, row_gap * 0.72)
    extent = float(np.nanmax([(sub['core_perm_importance_mean'] + sub['core_perm_importance_sd'].fillna(0)).max(), (sub['expansion_perm_importance_mean'] + sub['expansion_perm_importance_sd'].fillna(0)).max()]))
    step = 0.2 if extent > 0.5 else 0.1
    forest_min = 0.0
    forest_max = max(0.6, math.ceil(extent / step) * step)
    ticks = list(np.arange(forest_min, forest_max + step * 0.25, step))

    def map_x(value: float) -> float:
        value = float(np.clip(value, forest_min, forest_max))
        return forest_left + (value - forest_min) / (forest_max - forest_min) * (forest_right - forest_left)
    ref_x = map_x(0.5)
    ax.plot([ref_x, ref_x], [axis_y, header_y - 0.018], color=PALETTE['axis'], lw=0.66, ls=(0, (3.2, 3.2)), transform=ax.transAxes, zorder=1)
    for idx, (_, row) in enumerate(sub.iterrows()):
        col = row['column']
        y = row_y[col]
        if idx % 2 == 1:
            ax.add_patch(Rectangle((0.0, y - row_h / 2), 1.0, row_h, transform=ax.transAxes, facecolor='#E1EEFA', edgecolor='none', alpha=0.4, zorder=0))
        ax.text(label_x, y, short_var(col), transform=ax.transAxes, ha='left', va='center', fontsize=FS_TABLE_BODY, color=PALETTE['text'])
    series = [('Core', 'core_perm_importance_mean', 'core_perm_importance_sd', CORE_COLOR, 0.0), ('Expansion', 'expansion_perm_importance_mean', 'expansion_perm_importance_sd', EXP_COLOR, 0.0)]
    right_cols = {'Core': 0.765, 'Expansion': 0.925}
    header_font = dict(fontsize=FS_TABLE_HEADER, fontweight='bold', color=PALETTE['text'])
    value_font = dict(fontsize=FS_TABLE_VALUE, color=PALETTE['text'])
    ax.text(label_x, header_y, 'Variable', transform=ax.transAxes, ha='left', va='bottom', **header_font)
    for label, _, _, color, _ in series:
        x_col = right_cols[label]
        ax.text(x_col, header_y, label, transform=ax.transAxes, ha='center', va='bottom', **header_font)
        ax.scatter([x_col], [header_y - 0.03], s=15.698, facecolors=color, edgecolors=color, linewidths=0, transform=ax.transAxes, clip_on=False, zorder=5)
    for _, row in sub.iterrows():
        y_base = row_y[row['column']]
        for label, mean_col, sd_col, color, y_offset in series:
            mean = float(row[mean_col])
            sd = float(row[sd_col]) if np.isfinite(row[sd_col]) else 0.0
            y = y_base + y_offset
            x = map_x(mean)
            xerr = sd / (forest_max - forest_min) * (forest_right - forest_left)
            ax.plot([max(forest_left, x - xerr), min(forest_right, x + xerr)], [y, y], color=color, lw=0.56, transform=ax.transAxes, zorder=3)
            ax.scatter([x], [y], s=18.018, facecolors=color, edgecolors=color, linewidths=0, transform=ax.transAxes, zorder=4)
            ax.text(right_cols[label], y_base, f'{mean:.3f}\n({sd:.3f})', transform=ax.transAxes, ha='center', va='center', linespacing=1.24, **value_font)
    ax.plot([forest_left, forest_right], [axis_y, axis_y], color=PALETTE['axis'], lw=0.62, transform=ax.transAxes, zorder=5)
    for tick in ticks:
        x = map_x(tick)
        ax.plot([x, x], [axis_y, axis_y - tick_len], color=PALETTE['axis'], lw=0.46, transform=ax.transAxes, zorder=5)
        ax.text(x, axis_y - 0.022, f'{tick:.1f}', transform=ax.transAxes, ha='center', va='top', fontsize=FS_TICK, color=PALETTE['text'])
    ax.text((forest_left + forest_right) / 2, 0.0675, 'Permutation importance', transform=ax.transAxes, ha='center', va='bottom', fontsize=FS_AXIS, color=PALETTE['axis'], clip_on=False)

def stacked_bar(ax: mpl.axes.Axes, y: float, shares: pd.DataFrame, height: float=0.22) -> dict[str, dict[str, float | str]]:
    left = 0.0
    segments: dict[str, dict[str, float | str]] = {}
    for _, row in shares.iterrows():
        family_label = str(row['family_label'])
        width = float(row['importance_share']) * 100
        if width <= 0:
            continue
        face = PANEL_A_FAMILY_COLORS.get(family_label, family_color(family_label))
        segments[family_label] = {'x0': left, 'x1': left + width, 'y': float(y), 'color': face}
        ax.add_patch(Rectangle((left, y - height / 2), width, height, facecolor=face, edgecolor='white', lw=0.28))
        if width > 18:
            ax.text(left + width / 2, y, short_family(family_label), ha='center', va='center', fontsize=FS_SMALL, color='black')
        left += width
    return segments

def plot_panel_f(ax: mpl.axes.Axes, data: dict[str, pd.DataFrame]) -> None:
    panel_title(ax, 'd', 'North-South modification of zone mechanisms', letter_x=0.0257, title_x=0.1063)
    ax.axis('off')
    fam = data['family']
    left_ax = ax.inset_axes([0.0, 0.08, 0.98, 0.9])
    rows = [('N.C.', 'North_core'), ('N.E.', 'North_expansion'), ('S.C.', 'South_core'), ('S.E.', 'South_expansion')]
    y_positions = [3.15, 2.35, 1.05, 0.25]
    left_ax.set_xlim(0, 100)
    left_ax.set_ylim(-0.3, 3.7)
    row_segments: dict[str, dict[str, dict[str, float | str]]] = {}
    share_tables: dict[str, pd.DataFrame] = {}
    core_family_orders: dict[str, list[str]] = {}
    for _, group in rows:
        sub = fam[(fam['table'] == 'city_zone') & (fam['outcome'] == 'eli50') & (fam['group'] == group)].copy()
        sub['family_label'] = sub['family_label'].replace({'Terrain control': 'Other/minor drivers', 'Road network': 'Other/minor drivers'})
        sub = sub.groupby('family_label', as_index=False)['perm_importance_mean'].sum()
        total = sub['perm_importance_mean'].sum()
        sub['importance_share'] = np.where(total > 0, sub['perm_importance_mean'] / total, np.nan)
        share_tables[group] = sub
        if group.endswith('_core'):
            core_family_orders[group] = sub.sort_values('importance_share', ascending=False)['family_label'].tolist()
    for label, group, y in zip([r[0] for r in rows], [r[1] for r in rows], y_positions):
        sub = share_tables[group].copy()
        if group == 'North_expansion':
            core_order = core_family_orders.get('North_core', [])
        elif group == 'South_expansion':
            core_order = core_family_orders.get('South_core', [])
        else:
            core_order = []
        if core_order:
            extra_order = sub.sort_values('importance_share', ascending=False)['family_label'].tolist()
            family_order = core_order + [family for family in extra_order if family not in core_order]
            sub['family_label'] = pd.Categorical(sub['family_label'], categories=family_order, ordered=True)
            sub = sub.sort_values('family_label')
        else:
            sub = sub.sort_values('importance_share', ascending=False)
        row_segments[group] = stacked_bar(left_ax, y, sub, height=0.39)
        left_ax.text(-3.1, y, label, ha='right', va='center', fontsize=FS_BODY)

    def add_pair_ribbons(upper_group: str, lower_group: str, bar_h: float=0.39) -> None:
        upper = row_segments.get(upper_group, {})
        lower = row_segments.get(lower_group, {})
        if not upper or not lower:
            return
        for family_label, u in upper.items():
            l = lower.get(family_label)
            if l is None:
                continue
            ux0, ux1 = (float(u['x0']), float(u['x1']))
            lx0, lx1 = (float(l['x0']), float(l['x1']))
            if min(ux1 - ux0, lx1 - lx0) < 0.8:
                continue
            uy = float(u['y']) - bar_h / 2
            ly = float(l['y']) + bar_h / 2
            left_ax.add_patch(Polygon([(ux0, uy), (ux1, uy), (lx1, ly), (lx0, ly)], closed=True, facecolor=str(u['color']), edgecolor='none', alpha=0.13, zorder=0.7))
    add_pair_ribbons('North_core', 'North_expansion')
    add_pair_ribbons('South_core', 'South_expansion')
    left_ax.set_xlabel('Family share (%)', fontsize=FS_AXIS)
    left_ax.set_yticks([])
    left_ax.set_xticks([0, 25, 50, 75, 100])
    left_ax.grid(False)
    left_ax.xaxis.grid(False)
    left_ax.yaxis.grid(False)
    for gridline in left_ax.get_xgridlines() + left_ax.get_ygridlines():
        gridline.set_visible(False)
    despine(left_ax)
    left_ax.tick_params(axis='x', labelsize=FS_TICK, length=2.4)
    left_ax.spines['left'].set_visible(False)
    left_ax.text(100, 3.575, '100%', fontsize=FS_DENSE, ha='right', clip_on=False)
    left_ax.text(0, 3.575, 'composition', fontsize=FS_DENSE, ha='left', clip_on=False)

def select_response_columns(curves: pd.DataFrame, outcome: str, desired: list[str]) -> list[str]:
    available = set(curves[(curves['contrast_scope'] == 'global') & (curves['outcome'] == outcome)]['column'].unique())
    cols = [c for c in desired if c in available]
    if len(cols) < 4:
        vc = pd.read_csv(resource(DERIVED / 'q3_core_expansion_variable_contrast.csv'))
        extra = vc[(vc['contrast_scope'] == 'global') & (vc['outcome'] == outcome) & vc['column'].isin(available)].sort_values('abs_core_minus_expansion_share', ascending=False)['column'].tolist()
        for c in extra:
            if c not in cols:
                cols.append(c)
            if len(cols) == 4:
                break
    return cols[:4]

def select_paired_response_columns(curves: pd.DataFrame, outcome: str, desired: list[str]) -> list[str]:
    sub = curves[(curves['contrast_scope'] == 'global') & (curves['outcome'] == outcome)]
    counts = sub.groupby(['column', 'zone_label', 'curve_type'], dropna=False).size()
    paired = {col for col in sub['column'].dropna().unique() if all((counts.get((col, zone, curve), 0) > 0 for zone in ['core', 'expansion'] for curve in ['ALE', 'PDP']))}
    cols = [c for c in desired if c in paired]
    if len(cols) < 6:
        vc = pd.read_csv(resource(DERIVED / 'q3_core_expansion_variable_contrast.csv'))
        extra = vc[(vc['contrast_scope'] == 'global') & (vc['outcome'] == outcome) & vc['column'].isin(paired)].assign(mean_share=lambda d: (d['core_importance_share'].fillna(0) + d['expansion_importance_share'].fillna(0)) / 2).sort_values('mean_share', ascending=False)['column'].tolist()
        for c in extra:
            if c not in cols:
                cols.append(c)
            if len(cols) == 6:
                break
    if len(cols) < 6:
        raise ValueError('Six variables with ALE and PDP in both zones are required.')
    return cols[:6]

def add_ale_visual_envelope(ax: mpl.axes.Axes, x: np.ndarray, y: np.ndarray, curve: pd.DataFrame) -> None:
    finite = np.isfinite(y)
    if finite.sum() < 2:
        return
    y_range = max(float(np.nanmax(y[finite]) - np.nanmin(y[finite])), 1e-06)
    base_width = max(0.1 * y_range, 0.0015)
    width = np.full_like(y, base_width, dtype=float)
    if 'n_bin' in curve.columns and curve['n_bin'].notna().any():
        nbin = curve['n_bin'].to_numpy(float)
        max_n = np.nanmax(nbin)
        if max_n > 0:
            support = np.clip(nbin / max_n, 0, 1)
            width = base_width * (1.25 - 0.45 * support)
    ax.fill_between(x, y - width, y + width, color=RESPONSE_CI_COLOR, alpha=0.7, linewidth=0, zorder=1)

def plot_response_inset(parent: mpl.axes.Axes, data: pd.DataFrame, outcome: str, columns: list[str], ylabel: str) -> None:
    parent.axis('off')
    positions = [(0.045, 0.535, 0.415, 0.365), (0.56, 0.535, 0.415, 0.365), (0.045, 0.085, 0.415, 0.365), (0.56, 0.085, 0.415, 0.365)]
    for pos, col in zip(positions, columns):
        ax = parent.inset_axes(pos)
        threshold_x = None
        threshold_y = None
        for zone, color in [('core', CORE_COLOR), ('expansion', EXP_COLOR)]:
            for curve_type, ls, lw, alpha in [('ALE', '-', 1.78, 1.0), ('PDP', (0, (3, 2)), 1.28, 0.9)]:
                d = data[(data['contrast_scope'] == 'global') & (data['outcome'] == outcome) & (data['zone_label'] == zone) & (data['column'] == col) & (data['curve_type'] == curve_type)].sort_values('x')
                if d.empty:
                    continue
                x = np.linspace(0, 1, len(d))
                y = d['curve_value'].to_numpy(float)
                if curve_type == 'PDP':
                    y = y - np.nanmean(y)
                if curve_type == 'ALE':
                    ax.plot(x, y, color=color, lw=lw, ls=ls, alpha=alpha)
                    ax.scatter(x, y, s=8.5, color=color, edgecolor='white', lw=0.28, zorder=3)
                    if zone == 'core':
                        signs = np.sign(y)
                        crossing = np.where(np.diff(signs) != 0)[0]
                        if len(crossing):
                            i0 = crossing[0]
                            denom = y[i0 + 1] - y[i0]
                            threshold_x = float(x[i0] - y[i0] * (x[i0 + 1] - x[i0]) / denom) if abs(denom) > 1e-12 else float(x[i0])
                        elif len(y) > 2:
                            threshold_x = float(x[np.nanargmax(np.abs(np.gradient(y)))])
                        threshold_y = float(np.nanmax(y) if abs(np.nanmax(y)) >= abs(np.nanmin(y)) else np.nanmin(y))
                    if 'n_bin' in d.columns and d['n_bin'].notna().any():
                        nbin = d['n_bin'].to_numpy(float)
                        nbin = nbin / np.nanmax(nbin) if np.nanmax(nbin) > 0 else nbin * 0
                        ymin, ymax = (np.nanmin(y), np.nanmax(y))
                        rug_base = ymin - 0.08 * max(ymax - ymin, 1e-06)
                        for xx, nn in zip(x, nbin):
                            ax.plot([xx, xx], [rug_base, rug_base + 0.035 * max(ymax - ymin, 1e-06) * nn], color=color, lw=0.45, alpha=0.35, zorder=0)
                else:
                    ax.plot(x, y, color=color, lw=lw, ls=ls, alpha=alpha)
        ax.axhline(0, color='#6F7782', lw=0.68, ls=(0, (3, 2)))
        if threshold_x is not None and np.isfinite(threshold_x):
            ax.axvline(threshold_x, color='#6F7782', lw=0.62, ls=(0, (2, 2)), zorder=0)
            y0, y1 = ax.get_ylim()
            label_y = y0 + 0.82 * (y1 - y0)
            ax.text(min(max(threshold_x + 0.025, 0.05), 0.78), label_y, f'q={threshold_x:.2f}', fontsize=FS_SMALL, ha='left', va='center')
        ax.set_title(short_var(col), fontsize=FS_BODY + 0.45, pad=2)
        ax.set_xlim(0, 1)
        ax.grid(color=PALETTE['grid'], lw=0.45, alpha=0.8)
        ax.tick_params(axis='both', labelsize=FS_TICK, width=0.58, length=2.4, pad=1)
        ax.spines['left'].set_linewidth(0.65)
        ax.spines['bottom'].set_linewidth(0.65)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if pos[0] > 0.1:
            ax.set_yticklabels([])
        if pos[1] > 0.2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('driver percentile', fontsize=FS_AXIS, labelpad=0)
    parent.text(0.0, 0.98, ylabel, ha='left', va='top', fontsize=FS_SMALL)
    handles = [mpl.lines.Line2D([0], [0], color=CORE_COLOR, lw=1.4, label='Core ALE'), mpl.lines.Line2D([0], [0], color=EXP_COLOR, lw=1.4, label='Expansion ALE'), mpl.lines.Line2D([0], [0], color='black', lw=1.0, ls=(0, (3, 2)), label='PDP')]
    parent.legend(handles=handles, loc='upper right', fontsize=FS_DENSE + 0.05, ncol=3, handlelength=1.2, columnspacing=0.6)

def plot_response_zone_split_inset(parent: mpl.axes.Axes, data: pd.DataFrame, outcome: str, columns: list[str], ylabel: str) -> None:
    parent.axis('off')
    core_shift = axes_y_points(parent, 3.0 / 25.4 * 72.0)
    label_down = axes_y_points(parent, 2.0 / 25.4 * 72.0)
    q_label_positions = {('core', 'green_fraction_mean'): (0.54, 0.34), ('core', 'lst_c_mean'): (0.66, 0.76), ('core', 'building_height_mean'): (0.14, 0.65), ('core', 'noise_mean'): (0.52, 0.76), ('core', 'lat'): (0.15, 0.32), ('expansion', 'green_fraction_mean'): (0.62, 0.34), ('expansion', 'lst_c_mean'): (0.14, 0.42), ('expansion', 'building_height_mean'): (0.12, 0.74), ('expansion', 'noise_mean'): (0.2, 0.34), ('expansion', 'lat'): (0.4, 0.76)}
    positions = {'core': [(0.01, 0.765, 0.245, 0.17), (0.37, 0.765, 0.245, 0.17), (0.73, 0.765, 0.245, 0.17), (0.01, 0.53, 0.245, 0.17), (0.37, 0.53, 0.245, 0.17), (0.73, 0.53, 0.245, 0.17)], 'expansion': [(0.01, 0.245, 0.245, 0.17), (0.37, 0.245, 0.245, 0.17), (0.73, 0.245, 0.245, 0.17), (0.01, 0.03, 0.245, 0.17), (0.37, 0.03, 0.245, 0.17), (0.73, 0.03, 0.245, 0.17)]}
    for zone, group_positions in positions.items():
        for idx, (pos, col) in enumerate(zip(group_positions, columns)):
            if zone == 'core':
                pos = (pos[0], pos[1] + core_shift, pos[2], pos[3])
            ax = parent.inset_axes(pos)
            threshold_x = None
            for curve_type, ls, lw, alpha in [('ALE', '-', 1.78, 1.0), ('PDP', (0, (3, 2)), 1.28, 0.9)]:
                d = data[(data['contrast_scope'] == 'global') & (data['outcome'] == outcome) & (data['zone_label'] == zone) & (data['column'] == col) & (data['curve_type'] == curve_type)].sort_values('x')
                if d.empty:
                    continue
                x = np.linspace(0, 1, len(d))
                y = d['curve_value'].to_numpy(float)
                if curve_type == 'PDP':
                    y = y - np.nanmean(y)
                if curve_type == 'ALE':
                    add_ale_visual_envelope(ax, x, y, d)
                    ax.plot(x, y, color=RESPONSE_ALE_COLOR, lw=lw, ls=ls, alpha=alpha)
                    ax.scatter(x, y, s=7.8, color=RESPONSE_ALE_COLOR, edgecolor='white', lw=0.25, zorder=3)
                    signs = np.sign(y)
                    crossing = np.where(np.diff(signs) != 0)[0]
                    if len(crossing):
                        i0 = crossing[0]
                        denom = y[i0 + 1] - y[i0]
                        threshold_x = float(x[i0] - y[i0] * (x[i0 + 1] - x[i0]) / denom) if abs(denom) > 1e-12 else float(x[i0])
                    elif len(y) > 2:
                        threshold_x = float(x[np.nanargmax(np.abs(np.gradient(y)))])
                else:
                    ax.plot(x, y, color=RESPONSE_PDP_COLOR, lw=lw, ls=ls, alpha=alpha)
            ax.axhline(0, color='#6F7782', lw=0.68, ls=(0, (3, 2)))
            if threshold_x is not None and np.isfinite(threshold_x):
                ax.axvline(threshold_x, color='#6F7782', lw=0.62, ls=(0, (2, 2)), zorder=0)
                q_pos = q_label_positions.get((zone, col))
                if q_pos is None:
                    y0, y1 = ax.get_ylim()
                    ax.text(min(max(threshold_x + 0.02, 0.05), 0.5), y0 + 0.78 * (y1 - y0), f'q={threshold_x:.2f}', fontsize=FS_SMALL, ha='left', va='center')
                else:
                    ax.text(min(q_pos[0], 0.5), q_pos[1], f'q={threshold_x:.2f}', fontsize=FS_SMALL, ha='left', va='center', transform=ax.transAxes)
            ax.set_title(short_var(col), fontsize=FS_DENSE + 0.25, pad=2)
            ax.set_xlim(0, 1)
            ax.set_xticks([0.0, 0.5, 1.0])
            ax.grid(False)
            ax.xaxis.grid(False)
            ax.yaxis.grid(False)
            for gridline in ax.get_xgridlines() + ax.get_ygridlines():
                gridline.set_visible(False)
            ax.tick_params(axis='both', labelsize=FS_TICK, width=0.58, length=2.4, pad=1)
            ax.spines['left'].set_linewidth(0.65)
            ax.spines['bottom'].set_linewidth(0.65)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if idx < 3:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('')
    parent.text(0.01, 1.025 + core_shift - label_down, ylabel, ha='left', va='top', fontsize=FS_DENSE)
    parent.text(0.4925, 1.025 + core_shift - label_down, 'Core', ha='center', va='top', fontsize=FS_DENSE)
    parent.text(0.4925, 0.46 + core_shift, 'Expansion', ha='center', va='top', fontsize=FS_DENSE)
    parent.text(0.4925, -0.012, 'driver percentile', ha='center', va='top', fontsize=FS_AXIS, clip_on=False)
    handles = [mpl.lines.Line2D([0], [0], color=RESPONSE_ALE_COLOR, lw=1.4, label='ALE'), mpl.lines.Line2D([0], [0], color=RESPONSE_PDP_COLOR, lw=1.0, ls=(0, (3, 2)), label='PDP')]
    parent.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.99, 1.04 + core_shift - label_down), fontsize=FS_DENSE + 0.05, ncol=2, handlelength=1.2, columnspacing=0.6)

def plot_panel_g(ax: mpl.axes.Axes, data: dict[str, pd.DataFrame]) -> None:
    panel_title(ax, 'e', 'Nonlinear ELI responses', letter_x=0.0075, title_x=0.071)
    curves = data['response_curves']
    columns = select_paired_response_columns(curves, 'eli50', ['green_fraction_mean', 'lst_c_mean', 'building_height_mean', 'noise_mean'])
    plot_response_zone_split_inset(ax, curves, 'eli50', columns, 'Centered effect')

def plot_panel_h(ax: mpl.axes.Axes, data: dict[str, pd.DataFrame]) -> None:
    panel_title(ax, 'h', 'Negative-perception response functions')
    curves = data['response_curves']
    columns = select_response_columns(curves, 'negative_score50', ['lst_c_mean', 'noise_mean', 'green_fraction_mean', 'poi_transport_mean'])
    plot_response_inset(ax, curves, 'negative_score50', columns, 'centered effect')

def mechanism_vectors(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    fam = data['family']
    z = fam[(fam['table'] == 'city_zone') & fam['group'].isin(['core', 'expansion', 'North_core', 'North_expansion', 'South_core', 'South_expansion'])].copy()
    z['total'] = z.groupby(['group', 'outcome'])['perm_importance_mean'].transform('sum')
    z['share'] = np.where(z['total'] > 0, z['perm_importance_mean'] / z['total'], np.nan)
    mat = z.pivot_table(index=['group', 'outcome'], columns='family_label', values='share', fill_value=0)
    meta = mat.reset_index()[['group', 'outcome']]
    X = mat.to_numpy(float)
    X = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    coords = U[:, :2] * S[:2]
    var_exp = S ** 2 / np.sum(S ** 2) if np.sum(S ** 2) > 0 else np.zeros_like(S)
    out = meta.copy()
    out['PC1'] = coords[:, 0]
    out['PC2'] = coords[:, 1] if coords.shape[1] > 1 else 0
    out['pc1_var'] = var_exp[0] if len(var_exp) else np.nan
    out['pc2_var'] = var_exp[1] if len(var_exp) > 1 else np.nan
    out['zone_label'] = np.where(out['group'].str.contains('expansion'), 'expansion', 'core')
    out['ns_context'] = np.select([out['group'].str.startswith('North'), out['group'].str.startswith('South')], ['North', 'South'], default='global')
    return out

def plot_panel_i(ax: mpl.axes.Axes, data: dict[str, pd.DataFrame]) -> None:
    panel_title(ax, 'i', 'Group-outcome mechanism ordination')
    coords = mechanism_vectors(data)
    main = ax.inset_axes([0.12, 0.16, 0.68, 0.68])
    top = ax.inset_axes([0.12, 0.86, 0.68, 0.1], sharex=main)
    right = ax.inset_axes([0.82, 0.16, 0.12, 0.68], sharey=main)
    for outcome in coords['outcome'].unique():
        for scope in ['global', 'North', 'South']:
            c = coords[(coords['outcome'] == outcome) & (coords['ns_context'] == scope)]
            if {'core', 'expansion'}.issubset(set(c['zone_label'])):
                cc = c[c['zone_label'] == 'core'].iloc[0]
                ee = c[c['zone_label'] == 'expansion'].iloc[0]
                main.plot([cc['PC1'], ee['PC1']], [cc['PC2'], ee['PC2']], color=PAIR_LINE, lw=0.8, alpha=0.55, zorder=1)
    for zone, color, marker in [('core', CORE_COLOR, 'o'), ('expansion', EXP_COLOR, 's')]:
        d = coords[coords['zone_label'] == zone]
        add_group_ellipse(main, d['PC1'].to_numpy(float), d['PC2'].to_numpy(float), color)
        main.scatter(d['PC1'], d['PC2'], s=34, color=color, edgecolor='white', lw=0.45, marker=marker, label=zone.title(), zorder=4)
    pc1_groups = [coords[coords['zone_label'] == 'core']['PC1'].to_numpy(float), coords[coords['zone_label'] == 'expansion']['PC1'].to_numpy(float)]
    pc2_groups = [coords[coords['zone_label'] == 'core']['PC2'].to_numpy(float), coords[coords['zone_label'] == 'expansion']['PC2'].to_numpy(float)]
    bp1 = top.boxplot(pc1_groups, orientation='horizontal', positions=[0.32, 0.68], widths=0.22, patch_artist=True, showfliers=False)
    bp2 = right.boxplot(pc2_groups, orientation='vertical', positions=[0.32, 0.68], widths=0.22, patch_artist=True, showfliers=False)
    for bp, cols in [(bp1, [CORE_COLOR, EXP_COLOR]), (bp2, [CORE_COLOR, EXP_COLOR])]:
        for patch, col in zip(bp['boxes'], cols):
            patch.set(facecolor=col, alpha=0.32, edgecolor=col, linewidth=0.7)
        for key in ['whiskers', 'caps', 'medians']:
            for line in bp[key]:
                line.set(color='#6F7782', linewidth=0.55)
    for arr, ypos, col in zip(pc1_groups, [0.32, 0.68], [CORE_COLOR, EXP_COLOR]):
        top.scatter(arr, np.full_like(arr, ypos), s=5, color=col, alpha=0.28, lw=0)
    for arr, xpos, col in zip(pc2_groups, [0.32, 0.68], [CORE_COLOR, EXP_COLOR]):
        right.scatter(np.full_like(arr, xpos), arr, s=5, color=col, alpha=0.28, lw=0)
    main.axhline(0, color=PALETTE['grid'], lw=0.6)
    main.axvline(0, color=PALETTE['grid'], lw=0.6)
    pc1 = coords['pc1_var'].iloc[0] * 100
    pc2 = coords['pc2_var'].iloc[0] * 100
    main.set_xlabel(f'PC1 ({pc1:.0f}%)')
    main.set_ylabel(f'PC2 ({pc2:.0f}%)')
    main.grid(color=PALETTE['grid'], lw=0.45)
    despine(main)
    top.set_ylim(0.08, 0.92)
    right.set_xlim(0.08, 0.92)
    top.axis('off')
    right.axis('off')
    main.legend(loc='upper left', fontsize=FS_DENSE, frameon=False, handlelength=0.8)
    ax.text(0.12, 0.04, 'unit: outcome × context; lines connect matched core-expansion models', fontsize=FS_DENSE, ha='left')
    ax.axis('off')

def node_spans(counts: pd.Series, y0: float=0.005, y1: float=1.035, gap: float=0.012) -> dict[str, tuple[float, float, float]]:
    total = float(counts.sum())
    usable = y1 - y0 - gap * (len(counts) - 1)
    spans: dict[str, tuple[float, float, float]] = {}
    cur = y1
    for name, count in counts.items():
        h = usable * float(count) / max(total, 1e-09)
        spans[str(name)] = (cur - h, cur, cur - h / 2)
        cur -= h + gap
    return spans

def draw_alluvial_panel(ax: mpl.axes.Axes, flows: pd.DataFrame) -> None:
    pairs = flows.copy()
    pairs['source'] = pairs['source'].replace({'Impervious control': 'Other/minor drivers'})
    pairs['target'] = pairs['target'].replace({'Impervious control': 'Other/minor drivers'})
    pairs = pairs.groupby(['source', 'target'], as_index=False).agg(weight=('weight', 'sum'))
    left_counts = pairs.groupby('source')['weight'].sum().sort_values(ascending=False)
    right_counts = pairs.groupby('target')['weight'].sum().sort_values(ascending=False)
    left_spans = node_spans(left_counts)
    right_spans = node_spans(right_counts)
    x_left, x_right = (0.13, 0.7)
    for name, (lo, hi, cy) in left_spans.items():
        patch = Rectangle((x_left - 0.055, lo), 0.07, hi - lo, facecolor=family_color(name), edgecolor='white', lw=0.45, alpha=0.92, zorder=3)
        patch.set_clip_on(False)
        ax.add_patch(patch)
        ax.text(x_left - 0.07, cy, alluvial_family_label(name), ha='right', va='center', fontsize=FS_DENSE, clip_on=False)
    for name, (lo, hi, cy) in right_spans.items():
        patch = Rectangle((x_right, lo), 0.07, hi - lo, facecolor=family_color(name), edgecolor='white', lw=0.45, alpha=0.92, zorder=3)
        patch.set_clip_on(False)
        ax.add_patch(patch)
        ax.text(x_right + 0.08, cy, alluvial_family_label(name), ha='left', va='center', fontsize=FS_DENSE, clip_on=False)
    top_pairs = pairs.sort_values('weight', ascending=False).head(18)
    max_weight = max(float(top_pairs['weight'].max()), 1e-09)
    left_offsets = {k: 0.0 for k in left_spans}
    right_offsets = {k: 0.0 for k in right_spans}
    for _, row in top_pairs.sort_values('weight', ascending=True).iterrows():
        source, target = (str(row['source']), str(row['target']))
        weight = float(row['weight'])
        llo, lhi, _ = left_spans[source]
        rlo, rhi, _ = right_spans[target]
        sy = llo + (left_offsets[source] + 0.5 * weight) / max(float(left_counts[source]), 1e-09) * (lhi - llo)
        ty = rlo + (right_offsets[target] + 0.5 * weight) / max(float(right_counts[target]), 1e-09) * (rhi - rlo)
        left_offsets[source] += weight
        right_offsets[target] += weight
        verts = [(x_left + 0.015, sy), (0.32, sy), (0.5, ty), (x_right, ty)]
        path = MplPath(verts, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
        patch = PathPatch(path, facecolor='none', edgecolor=family_color(source), lw=0.62 + 4.2 * weight / max_weight, alpha=0.28, capstyle='round', zorder=1)
        patch.set_clip_on(False)
        ax.add_patch(patch)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

def plot_panel_j(ax: mpl.axes.Axes, data: dict[str, pd.DataFrame]) -> None:
    panel_title(ax, 'j', 'Dominant mechanism regime transition')
    reg = data['mechanism_regime']
    flows = reg.groupby(['core_top1_family_label', 'expansion_top1_family_label'], as_index=False).agg(weight=('family_share_total_variation', 'sum'), n=('outcome', 'size')).rename(columns={'core_top1_family_label': 'source', 'expansion_top1_family_label': 'target'})
    flows['weight'] = flows['weight'] + flows['n'] * 0.15
    draw_alluvial_panel(ax, flows)
    same = reg['top1_same'].mean() * 100
    overlap = reg['top3_overlap_count'].mean()
    ax.text(0.11, -0.015, 'Core', ha='center', va='center', fontsize=FS_BODY, clip_on=False)
    ax.text(0.735, -0.015, 'Expansion', ha='center', va='center', fontsize=FS_BODY, clip_on=False)
    ax.text(0.5, -0.015, f'Top-1 same {same:.0f}%\nMean top-3 overlap {overlap:.1f}', ha='center', va='center', fontsize=FS_BODY, clip_on=False)
    ax.text(0.5, -0.075, 'group-outcome level transitions', ha='center', va='bottom', fontsize=FS_SMALL, clip_on=False)

def force_text_black(fig: mpl.figure.Figure) -> None:
    for obj in fig.findobj(match=mpl.text.Text):
        obj.set_color('#000000')

def build_figure(data: dict[str, pd.DataFrame]) -> mpl.figure.Figure:
    fig = plt.figure(figsize=(13.6, 14.1), facecolor='white')
    L, W, Gx = (0.07, 0.87, 0.07)
    CW = (W - Gx) / 2
    EW = CW * 0.75
    H = 0.235
    EFH = H * 1.2
    EHF = EFH * 1.0192
    BH = EFH * 1.15
    DW = H * fig.get_figheight() / fig.get_figwidth() * 1.215438719601 * 0.9
    X = [L, L + CW + Gx]
    Y = [0.69, 0.38, 0.07]
    EB_Y = Y[1] - 0.01
    E_Y = EB_Y + EFH - EHF
    B_Y = EB_Y + EFH - BH
    E_X_SHIFT = 0.05
    B_X_SHIFT = -0.03
    G_X_SHIFT = B_X_SHIFT + 0.02
    F_H = EFH * 0.7 * 0.85
    F_Y = Y[2] + EFH - F_H
    G_Y = F_Y
    G_W = EW * 1.05
    G_X = X[1] + G_X_SHIFT + EW - G_W
    response_bottom = G_Y - 0.025
    response_title_y = B_Y - 0.015 + BH * (0.9348 + 0.004)
    response_height = (response_title_y - response_bottom) / (1.075 + 0.004)
    axes = {'a': fig.add_axes([X[0], Y[0], CW, H]), 'd': fig.add_axes([X[1] - 0.025, Y[0], DW, H]), 'e': fig.add_axes([X[0] + E_X_SHIFT, E_Y - 0.015, EW, EHF]), 'f': fig.add_axes([X[0] + E_X_SHIFT, F_Y - 0.025, EW, F_H]), 'g': fig.add_axes([G_X - 0.025, response_bottom, G_W, response_height])}
    plot_panel_a(axes['a'], data)
    plot_panel_d(axes['d'], data)
    plot_panel_e(axes['e'], data)
    plot_panel_f(axes['f'], data)
    plot_panel_g(axes['g'], data)
    contrast_ax = axes['e']
    pos = contrast_ax.get_position()
    width_scale = 1.06
    content_shift = 2.0 / 25.4 / fig.get_figwidth()
    contrast_ax.set_position([pos.x1 - pos.width * width_scale + content_shift, pos.y0, pos.width * width_scale, pos.height])
    for text in contrast_ax.texts:
        if text.get_text() in {'c', 'Driver-level contrast forest'}:
            x, y = text.get_position()
            text.set_position(((x + width_scale - 1 - content_shift / pos.width) / width_scale, y))
    heading_offset = ScaledTranslation(0, -1.0 / 25.4, fig.dpi_scale_trans)
    for key, letter, title in [('a', 'a', 'Core-expansion mechanism overview'), ('d', 'b', 'Core-expansion driver association'), ('e', 'c', 'Driver-level contrast forest'), ('g', 'e', 'Nonlinear ELI responses')]:
        for text in list(axes[key].texts):
            if text.get_text() in {letter, title}:
                anchor = copy(text)
                anchor.set_alpha(0)
                axes[key].add_artist(anchor)
                text.set_transform(text.get_transform() + heading_offset)
    for key, old_letter, new_letter in [('d', 'b', 'c'), ('e', 'c', 'b')]:
        for text in axes[key].texts:
            if text.get_text() == old_letter and text.get_alpha() != 0:
                text.set_text(new_letter)
    force_text_black(fig)
    return fig

def save_figure(fig: mpl.figure.Figure) -> None:
    basename = OUT_DIR / 'fig_q3_3_core_expansion_mechanisms_scidraw'
    export_specs = [(basename.with_suffix('.png'), {'dpi': 450}), (basename.with_suffix('.tiff'), {'dpi': 600}), (basename.with_suffix('.pdf'), {}), (basename.with_suffix('.svg'), {})]
    for path, extra in export_specs:
        try:
            fig.savefig(path, bbox_inches='tight', facecolor='white', **extra)
        except PermissionError:
            fallback = path.with_name(f'{path.stem}_updated{path.suffix}')
            fig.savefig(fallback, bbox_inches='tight', facecolor='white', **extra)
            print(f'{path} is locked; saved {fallback}')
    print(basename.with_suffix('.png'))

def main() -> None:
    data = load_data()
    fig = build_figure(data)
    save_figure(fig)
    plt.close(fig)

def _build_manuscript_figure():
    get = _model_tables(zone=True)
    data = {k: get(v) for k, v in {"family": "family", "permutation": "permutation", "performance": "rf_cv_performance", "predictions": "predictions"}.items()}
    data.update({k: _read_table("Fig_6/Fig_6_" + v + ".csv") for k, v in {"family_contrast": "family_contrast", "variable_contrast": "variable_contrast", "response_curves": "response_curves", "mechanism_regime": "mechanism_regime"}.items()})
    data["zone_master"] = _read_table("shared/Study_city_zone_analysis.csv")
    return build_figure(data), None


if __name__ == "__main__":
    _configure_land_functions(sys.modules[__name__])
    _figure, _bbox = _build_manuscript_figure()
    _save_figure(_figure, _bbox, 'Fig_6', _figure_options)
