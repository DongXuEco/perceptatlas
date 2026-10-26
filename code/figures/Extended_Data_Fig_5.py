"""Plot Extended_Data_Fig_5 using the packaged numerical tables or supplied local inputs."""
from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_support"))
from figure_cli import prepare_figure as _prepare_figure, save_figure as _save_figure
from figure_cli import read_table as _read_table, model_tables as _model_tables
from figure_cli import configure_land_functions as _configure_land_functions

_figure_options = _prepare_figure('Extended_Data_Fig_5', 'analysis_tables') if __name__ == "__main__" else None

from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
import json
import math
from copy import copy
from pathlib import Path
from typing import Iterable
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon, Rectangle
import matplotlib.transforms as mtransforms
import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from PIL import Image, ImageOps
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
plt.rcParams['text.color'] = '#000000'
plt.rcParams['axes.labelcolor'] = '#000000'
plt.rcParams['axes.titlecolor'] = '#000000'
plt.rcParams['xtick.color'] = '#000000'
plt.rcParams['ytick.color'] = '#000000'
PROJECT_ROOT = PACKAGE_ROOT
OUT_DIR = output_dir('Extended_Data_Fig_5')
OUT_DIR.mkdir(parents=True, exist_ok=True)
MECHANISM_TABLE = OUT_DIR / 'q3_city_local_mechanism_map_data.csv'
URBAN_MASK = PROJECT_ROOT / '全球urbanmask' / 'global_polygon_1km.tif'
OUT_BASE = OUT_DIR / 'fig_q3_2_global_mechanism_map'
LAND_COLOR = '#E9EBF1'
SEA_COLOR = '#FFFFFF'
AXIS_COLOR = '#000000'
TEXT_COLOR = '#000000'
GRID_COLOR = '#E5E7EB'
MUTED = '#000000'
FONT_PANEL_LETTER = 9.2
FONT_PANEL_TITLE = 7.6
FONT_AXIS_LABEL = 7.0
FONT_TICK = 6.4
FONT_LEGEND = 6.2
FONT_LEGEND_TITLE = 6.5
FONT_INTERNAL_LABEL = 6.2
FONT_BIVAR_TITLE = 6.0
FONT_BIVAR_AXIS = 5.8
FONT_BIVAR_TICK = 5.5
FONT_BIVAR_CORNER = 5.1
BIVARIATE_ELI_BOUNDS = (-0.2, 0.15)
BIVARIATE_GINI_MAX = 0.1
MAP_LEFT_SHIFT_DEG = 8.0
FAMILY_COLORS = {'Green/blue space': '#62C0D4', 'Environmental risk': '#E5170A', 'Built environment': '#357DBD', 'Economic/population': '#6E789D', 'POI supply': '#F28817', 'Non-motorized accessibility': '#F1B47A', 'Road network': '#F0CFA1', 'Terrain control': '#909BB6', 'Topographic control': '#909BB6', 'Location control': '#8170A8', 'Other': '#C8D0C1', 'Other/minor drivers': '#C8D0C1'}
FAMILY_ORDER = ['Green/blue space', 'Environmental risk', 'Built environment', 'Economic/population', 'POI supply', 'Non-motorized accessibility', 'Road network', 'Terrain control', 'Topographic control', 'Location control', 'Other', 'Other/minor drivers']
FAMILY_SHORT = {'Green/blue space': 'Green/blue', 'Environmental risk': 'Env. risk', 'Built environment': 'Built env.', 'Economic/population': 'Econ./pop.', 'POI supply': 'POI supply', 'Non-motorized accessibility': 'Non-motor', 'Road network': 'Road', 'Terrain control': 'Terrain', 'Topographic control': 'Terrain', 'Location control': 'Location', 'Other': 'Other', 'Other/minor drivers': 'Other'}
LAYER_COLORS = {'ELI global': '#30343B', 'ELI North': '#0D47A1', 'ELI South': '#F27C2B', 'Gini global': '#F28817', 'Theil global': '#E5170A', 'P90-P10 global': '#8170A8'}
TRANSITION_ACCESS_FAMILIES = {'POI supply', 'Non-motorized accessibility', 'Road network'}
TRANSITION_ORDER = ['same family', 'Green/blue -> POI/access', 'Env. risk -> POI/access', 'Built env. -> other transitions', 'other transitions']
TRANSITION_COLORS = {'same family': '#30343B', 'Green/blue -> POI/access': '#62C0D4', 'Env. risk -> POI/access': '#E5170A', 'Built env. -> other transitions': '#357DBD', 'other transitions': '#C8D0C1'}
MAP_POINT_SIZE_SCALE = 1.0925
MAP_LAYERS = [('eli50', 'NorthSouth', 'Dominant predictor families: ELI', 'a'), ('compound_gini50', 'NorthSouth', 'Dominant predictor families: Gini', 'b')]
SUMMARY_LAYERS = [('eli50', 'global', 'ELI global'), ('eli50', 'North', 'ELI North'), ('eli50', 'South', 'ELI South'), ('compound_gini50', 'global', 'Gini global'), ('compound_theil50', 'global', 'Theil global'), ('compound_p9010_50', 'global', 'P90-P10 global')]
COMPOSITION_LAYERS = [('eli50', 'global', 'ELI global'), ('compound_gini50', 'global', 'Gini global'), ('compound_theil50', 'global', 'Theil global'), ('compound_p9010_50', 'global', 'P90-P10 global')]
STRENGTH_LAYERS = COMPOSITION_LAYERS
NCLASS = 10
BIVARIATE_PALETTE = np.array([[(114, 47, 118), (155, 66, 136), (180, 83, 151), (200, 106, 163), (216, 122, 175), (230, 143, 178), (236, 130, 147), (241, 118, 116), (246, 106, 84), (252, 93, 53)], [(139, 72, 150), (174, 95, 166), (199, 118, 180), (215, 135, 189), (229, 158, 197), (235, 168, 194), (239, 156, 161), (242, 145, 129), (246, 134, 96), (250, 123, 64)], [(151, 100, 175), (181, 127, 189), (201, 151, 201), (223, 169, 208), (234, 186, 213), (240, 192, 210), (242, 182, 176), (244, 173, 142), (246, 163, 108), (248, 154, 74)], [(161, 128, 194), (191, 159, 210), (214, 180, 219), (229, 195, 224), (240, 208, 229), (245, 216, 226), (245, 208, 191), (245, 200, 156), (245, 192, 120), (245, 184, 85)], [(167, 152, 214), (200, 183, 225), (221, 207, 235), (234, 216, 235), (243, 231, 241), (250, 241, 242), (248, 234, 206), (246, 228, 169), (245, 221, 132), (243, 214, 96)], [(152, 163, 221), (180, 190, 231), (205, 213, 239), (224, 226, 243), (238, 239, 246), (249, 246, 232), (250, 246, 214), (245, 240, 180), (238, 233, 147), (222, 212, 83)], [(130, 148, 212), (159, 177, 225), (191, 205, 236), (211, 219, 241), (228, 234, 244), (244, 244, 225), (242, 241, 192), (232, 233, 170), (218, 223, 134), (190, 203, 74)], [(106, 131, 204), (134, 160, 217), (165, 188, 230), (186, 205, 236), (211, 224, 241), (231, 237, 222), (227, 232, 194), (216, 223, 162), (197, 211, 126), (165, 191, 68)], [(90, 113, 193), (112, 143, 209), (139, 170, 225), (158, 190, 230), (184, 207, 236), (213, 224, 214), (204, 219, 180), (189, 208, 146), (169, 197, 112), (128, 169, 63)], [(72, 91, 179), (96, 124, 199), (115, 147, 214), (128, 167, 222), (157, 188, 229), (188, 208, 202), (177, 201, 160), (163, 191, 128), (142, 177, 97), (88, 140, 52)]], dtype=float) / 255.0
REQUIRED_COLUMNS = ['id_1', 'lon', 'lat', 'ns_group', 'outcome', 'model_group', 'dominant_family_label', 'dominant_family_share']
_LAND_GDF: gpd.GeoDataFrame | None = None
_LAND_COORDS: list[np.ndarray] | None = None

def find_world_land() -> Path:
    return PACKAGE_ROOT / 'data/shared/World_land_polygons.geojson'

def world_land() -> gpd.GeoDataFrame:
    global _LAND_GDF
    if _LAND_GDF is None:
        _LAND_GDF = gpd.read_file(resource(find_world_land())).to_crs('EPSG:4326')
    return _LAND_GDF

def polygon_exteriors(geometry) -> list[np.ndarray]:
    if geometry is None or geometry.is_empty:
        return []
    geom_type = geometry.geom_type
    if geom_type == 'Polygon':
        return [np.asarray(geometry.exterior.coords, dtype=float)]
    if geom_type == 'MultiPolygon':
        return [np.asarray(poly.exterior.coords, dtype=float) for poly in geometry.geoms if not poly.is_empty]
    if geom_type == 'GeometryCollection':
        coords: list[np.ndarray] = []
        for part in geometry.geoms:
            coords.extend(polygon_exteriors(part))
        return coords
    return []

def land_exterior_coords() -> list[np.ndarray]:
    global _LAND_COORDS
    if _LAND_COORDS is None:
        coords: list[np.ndarray] = []
        for geometry in world_land().geometry:
            coords.extend(polygon_exteriors(geometry))
        _LAND_COORDS = coords
    return _LAND_COORDS

def family_color(label: str) -> str:
    return FAMILY_COLORS.get(str(label), FAMILY_COLORS['Other'])

def family_short(label: str) -> str:
    return FAMILY_SHORT.get(str(label), str(label))

def ordered_families(labels: Iterable[str]) -> list[str]:
    present = [str(x) for x in pd.Series(list(labels)).dropna().unique()]
    ordered = [fam for fam in FAMILY_ORDER if fam in present]
    return ordered + sorted([fam for fam in present if fam not in ordered])

def point_size(share: pd.Series | np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(share, dtype=float)
    scaled = np.clip((arr - 0.18) / (0.75 - 0.18), 0.0, 1.0)
    size = (5.0 + 23.0 * scaled) * 0.36
    if np.isscalar(share):
        return float(size)
    return size

def load_mechanism_data() -> pd.DataFrame:
    if not resource(MECHANISM_TABLE).exists():
        raise FileNotFoundError(f'Missing {MECHANISM_TABLE}. Run build_q3_city_local_mechanism_map_data.py first.')
    df = pd.read_csv(resource(MECHANISM_TABLE))
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns in {MECHANISM_TABLE}: {missing}')
    df = df.copy()
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['dominant_family_share'] = pd.to_numeric(df['dominant_family_share'], errors='coerce')
    df['dominant_family_label'] = df['dominant_family_label'].fillna('Other')
    if df[['lon', 'lat']].isna().any().any():
        raise ValueError('The mechanism table contains missing longitude/latitude values.')
    if not df['lon'].between(-180, 180).all() or not df['lat'].between(-90, 90).all():
        raise ValueError('City coordinates are outside valid WGS84 lon/lat ranges.')
    duplicated = int(df.duplicated(['id_1', 'outcome', 'model_group']).sum())
    if duplicated:
        raise ValueError(f'Duplicate id_1/outcome/model_group keys found: {duplicated}')
    return df

def layer_data(df: pd.DataFrame, outcome: str, model_group: str) -> pd.DataFrame:
    sub = df[(df['outcome'] == outcome) & (df['model_group'] == model_group)].copy()
    sub = sub.dropna(subset=['lon', 'lat', 'dominant_family_share', 'dominant_family_label'])
    if sub.empty:
        raise ValueError(f'No rows for outcome={outcome}, model_group={model_group}.')
    return sub.sort_values('dominant_family_share')

def map_layer_data(df: pd.DataFrame, outcome: str, model_group: str) -> pd.DataFrame:
    if model_group != 'NorthSouth':
        return layer_data(df, outcome, model_group)
    sub = df[(df['outcome'] == outcome) & ((df['model_group'] == 'North') & (df['ns_group'] == 'North') | (df['model_group'] == 'South') & (df['ns_group'] == 'South'))].copy()
    sub = sub.dropna(subset=['lon', 'lat', 'dominant_family_share', 'dominant_family_label'])
    if sub.empty:
        raise ValueError(f'No rows for outcome={outcome}, model_group={model_group}.')
    duplicated = int(sub['id_1'].duplicated().sum())
    if duplicated:
        raise ValueError(f'NorthSouth map layer has duplicated city IDs: {duplicated}')
    return sub.sort_values('dominant_family_share')

def setup_map_axis(ax: mpl.axes.Axes, x_shift_deg: float=0.0) -> None:
    ax.set_facecolor(SEA_COLOR)
    ax.set_xlim(-180 + x_shift_deg, 180 + x_shift_deg)
    ax.set_ylim(-58, 84)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

def draw_land(ax: mpl.axes.Axes) -> None:
    patches = [MplPolygon(coords, closed=True) for coords in land_exterior_coords()]
    collection = PatchCollection(patches, facecolor=LAND_COLOR, edgecolor='#FFFFFF', linewidth=0.1, zorder=0)
    ax.add_collection(collection)

def add_panel_title(ax: mpl.axes.Axes, letter: str, title: str) -> None:
    ax.text(-0.008, 1.045, letter, transform=ax.transAxes, ha='left', va='bottom', fontsize=FONT_PANEL_LETTER, fontweight='bold', color=TEXT_COLOR, clip_on=False)
    ax.text(0.05, 1.045, title, transform=ax.transAxes, ha='left', va='bottom', fontsize=FONT_PANEL_TITLE, color=TEXT_COLOR, clip_on=False)

def force_black_text(fig: mpl.figure.Figure) -> None:
    for text in fig.findobj(lambda artist: isinstance(artist, mpl.text.Text)):
        text.set_color(TEXT_COLOR)

def draw_ns_composition_minibars(ax: mpl.axes.Axes, data: pd.DataFrame, row_labels: tuple[str, str]) -> None:
    groups = ('North', 'South')
    x0 = 0.042
    y0 = 0.012
    bar_w = 0.04477824
    bar_h = 0.506677248
    bar_gap = 0.032
    bar_specs = []
    for i, (group, label) in enumerate(zip(groups, row_labels)):
        x = x0 + i * (bar_w + bar_gap)
        sub = data[data['ns_group'] == group]
        shares = sub['dominant_family_label'].value_counts(normalize=True)
        families = ordered_families(shares.index)
        spans = {}
        bottom = y0
        for family in families:
            height = float(shares.loc[family]) * bar_h
            if height <= 0:
                continue
            spans[family] = (bottom, bottom + height)
            bottom += height
        bar_specs.append({'group': group, 'label': label, 'x': x, 'families': families, 'spans': spans})
    if len(bar_specs) == 2:
        left_spec, right_spec = bar_specs
        shared_families = ordered_families(set(left_spec['spans']) & set(right_spec['spans']))
        for family in shared_families:
            left_bottom, left_top = left_spec['spans'][family]
            right_bottom, right_top = right_spec['spans'][family]
            ax.add_patch(MplPolygon([(left_spec['x'] + bar_w, left_bottom), (right_spec['x'], right_bottom), (right_spec['x'], right_top), (left_spec['x'] + bar_w, left_top)], closed=True, transform=ax.transAxes, facecolor=family_color(family), edgecolor='none', alpha=0.16, clip_on=True, zorder=7))
    axis_x = x0 - 0.012
    tick_len = 0.006
    ax.plot([axis_x, axis_x], [y0, y0 + bar_h], transform=ax.transAxes, color=TEXT_COLOR, linewidth=0.35, clip_on=False, zorder=9)
    for tick, tick_label in [(0.0, '0'), (0.5, '50'), (1.0, '100')]:
        yy = y0 + tick * bar_h
        ax.plot([axis_x - tick_len, axis_x], [yy, yy], transform=ax.transAxes, color=TEXT_COLOR, linewidth=0.35, clip_on=False, zorder=9)
        ax.text(axis_x - tick_len - 0.006, yy, tick_label, transform=ax.transAxes, ha='right', va='center', fontsize=FONT_LEGEND - 0.6, color=TEXT_COLOR, clip_on=False, zorder=9)
    ax.text(axis_x - 0.066, y0 + bar_h / 2, 'City %', transform=ax.transAxes, ha='center', va='center', rotation=90, fontsize=FONT_LEGEND - 0.6, color=TEXT_COLOR, clip_on=False, zorder=9)
    for spec in bar_specs:
        x = spec['x']
        group = spec['group']
        label_short = 'Nor.' if group == 'North' else 'Sou.'
        ax.text(x + bar_w / 2, y0 - 0.032, label_short, transform=ax.transAxes, ha='center', va='top', fontsize=FONT_LEGEND - 0.6, color=TEXT_COLOR, clip_on=False, zorder=9)
        for family in spec['families']:
            if family not in spec['spans']:
                continue
            bottom, top = spec['spans'][family]
            ax.add_patch(Rectangle((x, bottom), bar_w, top - bottom, transform=ax.transAxes, facecolor=family_color(family), edgecolor='#FFFFFF', linewidth=0.22, clip_on=True, zorder=8))

def draw_mechanism_map(ax: mpl.axes.Axes, data: pd.DataFrame, title: str, letter: str, note: str | None=None, mini_bar_labels: tuple[str, str] | None=None, map_x_shift_deg: float=0.0) -> None:
    setup_map_axis(ax, x_shift_deg=map_x_shift_deg)
    draw_land(ax)
    colors = [family_color(v) for v in data['dominant_family_label']]
    sizes = point_size(data['dominant_family_share']) * MAP_POINT_SIZE_SCALE
    ax.scatter(data['lon'], data['lat'], s=sizes, c=colors, linewidths=0.12, edgecolors='#FFFFFF', alpha=0.82, zorder=3)
    add_panel_title(ax, letter, title)
    if mini_bar_labels is not None:
        draw_ns_composition_minibars(ax, data, mini_bar_labels)
    if note:
        ax.text(0.014, 0.02, note, transform=ax.transAxes, ha='left', va='bottom', fontsize=FONT_INTERNAL_LABEL, color=MUTED)

def robust_bivariate_bounds(values: pd.Series | np.ndarray, center_zero: bool=False) -> tuple[float, float]:
    vals = pd.to_numeric(pd.Series(values), errors='coerce').dropna().to_numpy(float)
    if vals.size == 0:
        return (0.0, 1.0)
    vmin, vmax = np.nanpercentile(vals, [2, 98])
    if math.isclose(float(vmin), float(vmax)):
        vmin, vmax = (float(np.nanmin(vals)), float(np.nanmax(vals)))
    if math.isclose(float(vmin), float(vmax)):
        vmin, vmax = (float(vmin) - 0.5, float(vmax) + 0.5)
    if center_zero and vmin < 0 < vmax:
        limit = max(abs(float(vmin)), abs(float(vmax)))
        return (-limit, limit)
    return (float(vmin), float(vmax))

def classify_fixed(values: pd.Series | np.ndarray, bounds: tuple[float, float], nclass: int=NCLASS) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    idx = np.full(arr.shape, -1, dtype=int)
    valid = np.isfinite(arr)
    if valid.sum() == 0:
        return idx
    vmin, vmax = bounds
    if vmax <= vmin:
        vmax = vmin + 1e-09
    edges = np.linspace(vmin, vmax, nclass + 1)
    clipped = np.clip(arr[valid], vmin, vmax)
    idx_valid = np.digitize(clipped, bins=edges[1:-1], right=False)
    idx[valid] = np.clip(idx_valid, 0, nclass - 1)
    return idx

def make_bivariate_rgba(x_values: pd.Series | np.ndarray, y_values: pd.Series | np.ndarray, x_bounds: tuple[float, float], y_bounds: tuple[float, float], alpha: float=0.86) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_cls = classify_fixed(x_values, x_bounds)
    y_cls = classify_fixed(y_values, y_bounds)
    rgba = np.zeros((len(x_cls), 4), dtype=float)
    valid = (x_cls >= 0) & (y_cls >= 0)
    rows = (NCLASS - 1 - y_cls[valid]).astype(int)
    cols = x_cls[valid].astype(int)
    rgba[valid, :3] = BIVARIATE_PALETTE[rows, cols]
    rgba[valid, 3] = alpha
    rgba[~valid, 3] = 0.0
    return (rgba, valid, x_cls, y_cls)

def bivariate_city_data(df: pd.DataFrame) -> pd.DataFrame:
    eli = map_layer_data(df, 'eli50', 'NorthSouth')[['id_1', 'lon', 'lat', 'ns_group', 'observed', 'dominant_family_share', 'dominant_family_label']].rename(columns={'observed': 'observed_eli', 'dominant_family_share': 'eli_family_share', 'dominant_family_label': 'eli_family'})
    gini = map_layer_data(df, 'compound_gini50', 'NorthSouth')[['id_1', 'lon', 'lat', 'ns_group', 'observed', 'dominant_family_share', 'dominant_family_label']].rename(columns={'observed': 'observed_gini', 'dominant_family_share': 'gini_family_share', 'dominant_family_label': 'gini_family'})
    merged = eli.merge(gini, on=['id_1', 'lon', 'lat', 'ns_group'], how='inner', validate='one_to_one')
    if merged.empty:
        raise ValueError('No matched city rows for the ELI-Gini bivariate map.')
    merged['mean_family_share'] = merged[['eli_family_share', 'gini_family_share']].mean(axis=1)
    return merged.sort_values('mean_family_share')

def draw_bivariate_key_inset(ax: mpl.axes.Axes, bivariate_meta: dict[str, object]) -> None:
    legend_scale = 0.85
    leg = ax.inset_axes([0.028, 0.06, 0.188 * legend_scale, 0.476 * legend_scale], zorder=8)
    leg.set_facecolor('white')
    leg.patch.set_alpha(0.96)
    for x_cls in range(NCLASS):
        for y_cls in range(NCLASS):
            color = BIVARIATE_PALETTE[NCLASS - 1 - y_cls, x_cls]
            leg.add_patch(Rectangle((x_cls, y_cls), 1, 1, facecolor=color, edgecolor='#FFFFFF', linewidth=0.16 * legend_scale))
    leg.set_xlim(0, NCLASS)
    leg.set_ylim(0, NCLASS)
    leg.set_aspect('equal', adjustable='box')
    leg.set_xticks([0, NCLASS])
    leg.set_yticks([0, NCLASS])
    leg.set_xticklabels(['Low', 'High'], fontsize=FONT_BIVAR_TICK)
    leg.set_yticklabels(['Low', 'High'], fontsize=FONT_BIVAR_TICK)
    leg.tick_params(length=1.35 * legend_scale, width=0.32 * legend_scale, pad=1.05, colors=AXIS_COLOR)
    leg.set_xlabel('Observed ELI', fontsize=FONT_BIVAR_AXIS, labelpad=1.7)
    leg.set_ylabel('Gini', fontsize=FONT_BIVAR_AXIS, labelpad=1.7)
    leg.set_title('ELI x Gini', fontsize=FONT_BIVAR_TITLE, pad=1.4, color=TEXT_COLOR)
    for spine in leg.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.36 * legend_scale)
        spine.set_color('#4A4F54')
    leg.text(0.38, 0.35, 'Co-low', ha='left', va='bottom', fontsize=FONT_BIVAR_CORNER, color=TEXT_COLOR)
    leg.text(9.45, 9.45, 'Co-high', ha='right', va='top', fontsize=FONT_BIVAR_CORNER, color=TEXT_COLOR)

def draw_bivariate_outcome_map(ax: mpl.axes.Axes, data: pd.DataFrame, letter: str='c', map_x_shift_deg: float=0.0) -> dict[str, object]:
    setup_map_axis(ax, x_shift_deg=map_x_shift_deg)
    draw_land(ax)
    x_bounds = BIVARIATE_ELI_BOUNDS
    y_min, _ = robust_bivariate_bounds(data['observed_gini'], center_zero=False)
    y_bounds = (y_min, BIVARIATE_GINI_MAX)
    rgba, valid, x_cls, y_cls = make_bivariate_rgba(data['observed_eli'], data['observed_gini'], x_bounds, y_bounds)
    sub = data.loc[valid].copy()
    sizes = point_size(sub['mean_family_share']) * MAP_POINT_SIZE_SCALE
    ax.scatter(sub['lon'], sub['lat'], s=sizes, c=rgba[valid], linewidths=0.08, edgecolors='#FFFFFF', alpha=0.86, zorder=3)
    add_panel_title(ax, letter, 'Observed ELI and Gini')
    class_counts = pd.DataFrame({'x_class': x_cls[valid], 'y_class': y_cls[valid]}).value_counts(['x_class', 'y_class']).reset_index(name='n')
    return {'n': int(valid.sum()), 'x_variable': 'observed_eli', 'y_variable': 'observed_gini', 'x_bounds': [float(x_bounds[0]), float(x_bounds[1])], 'y_bounds': [float(y_bounds[0]), float(y_bounds[1])], 'occupied_bins': int(len(class_counts)), 'max_bin_n': int(class_counts['n'].max()) if not class_counts.empty else 0}

def transition_label(eli_family: str, gini_family: str) -> str:
    if eli_family == gini_family:
        return 'same family'
    if eli_family == 'Green/blue space' and gini_family in TRANSITION_ACCESS_FAMILIES:
        return 'Green/blue -> POI/access'
    if eli_family == 'Environmental risk' and gini_family in TRANSITION_ACCESS_FAMILIES:
        return 'Env. risk -> POI/access'
    if eli_family == 'Built environment':
        return 'Built env. -> other transitions'
    return 'other transitions'

def concordance_city_data(bivariate_data: pd.DataFrame) -> pd.DataFrame:
    data = bivariate_data.copy()
    data['transition_label'] = [transition_label(eli_family, gini_family) for eli_family, gini_family in zip(data['eli_family'], data['gini_family'])]
    data['transition_order'] = data['transition_label'].map({label: i for i, label in enumerate(TRANSITION_ORDER)})
    data['transition_order'] = data['transition_order'].fillna(len(TRANSITION_ORDER)).astype(int)
    return data.sort_values(['transition_order', 'mean_family_share'])

def transition_color(label: str) -> str:
    return TRANSITION_COLORS.get(str(label), TRANSITION_COLORS['other transitions'])

def add_transition_legend(ax: mpl.axes.Axes, labels: list[str]) -> None:
    handles = [Line2D([0], [0], marker='o', linestyle='', markersize=3.9, markerfacecolor=transition_color(label), markeredgecolor='#FFFFFF', markeredgewidth=0.28, label=label) for label in labels]
    legend = ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(-0.048, -0.058), ncol=1, fontsize=FONT_LEGEND, title='ELI to Gini', title_fontsize=FONT_LEGEND_TITLE, handletextpad=0.34, labelspacing=0.23, borderpad=0.24, frameon=True)
    legend.get_frame().set_facecolor('#FFFFFF')
    legend.get_frame().set_edgecolor('none')
    legend.get_frame().set_alpha(0.78)
    title_shift_inches = 0.06 * ax.get_position().width * ax.figure.get_figwidth()
    legend.get_title().set_transform(legend.get_title().get_transform() + mtransforms.ScaledTranslation(-title_shift_inches, 0.0, ax.figure.dpi_scale_trans))

def draw_concordance_map(ax: mpl.axes.Axes, bivariate_data: pd.DataFrame, letter: str='d') -> dict[str, object]:
    data = concordance_city_data(bivariate_data)
    setup_map_axis(ax)
    draw_land(ax)
    colors = [transition_color(label) for label in data['transition_label']]
    sizes = point_size(data['mean_family_share']) * MAP_POINT_SIZE_SCALE
    ax.scatter(data['lon'], data['lat'], s=sizes, c=colors, linewidths=0.08, edgecolors='#FFFFFF', alpha=0.84, zorder=3)
    present_labels = [label for label in TRANSITION_ORDER if label in set(data['transition_label'])]
    add_transition_legend(ax, present_labels)
    add_panel_title(ax, letter, 'Dominant-family agreement: ELI and Gini')
    counts = data['transition_label'].value_counts().reindex(TRANSITION_ORDER, fill_value=0)
    return {'n': int(len(data)), 'transition_counts': {label: int(counts[label]) for label in TRANSITION_ORDER}, 'transition_shares': {label: float(counts[label] / len(data)) for label in TRANSITION_ORDER}}

def clean_stat_axis(ax: mpl.axes.Axes) -> None:
    ax.tick_params(axis='both', labelsize=FONT_TICK, length=2.4, width=0.45, colors=AXIS_COLOR)
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color(AXIS_COLOR)
    ax.grid(axis='x', color=GRID_COLOR, lw=0.35, alpha=0.82, zorder=0)

def draw_composition_panel(ax: mpl.axes.Axes, df: pd.DataFrame, letter: str='e') -> pd.DataFrame:
    rows = []
    all_labels: list[str] = []
    for outcome, group, label in COMPOSITION_LAYERS:
        sub = layer_data(df, outcome, group)
        counts = sub['dominant_family_label'].value_counts()
        total = counts.sum()
        all_labels.extend(counts.index.tolist())
        for family, count in counts.items():
            rows.append({'layer': label, 'family': family, 'count': int(count), 'share_pct': 100.0 * float(count) / float(total), 'n': int(total)})
    comp = pd.DataFrame(rows)
    families = ordered_families(all_labels)
    y_positions = np.arange(len(COMPOSITION_LAYERS))[::-1]
    bar_height = 0.66
    bar_half = bar_height / 2.0
    segments: dict[str, dict[str, tuple[float, float, float]]] = {}
    for y, (_, _, label) in zip(y_positions, COMPOSITION_LAYERS):
        left = 0.0
        segments[label] = {}
        for family in families:
            row = comp[(comp['layer'] == label) & (comp['family'] == family)]
            value = float(row['share_pct'].iloc[0]) if not row.empty else 0.0
            if value > 0:
                segments[label][family] = (left, left + value, float(y))
            left += value
    for upper, lower in zip(COMPOSITION_LAYERS[:-1], COMPOSITION_LAYERS[1:]):
        upper_label = upper[2]
        lower_label = lower[2]
        for family in families:
            if family not in segments[upper_label] or family not in segments[lower_label]:
                continue
            upper_start, upper_end, upper_y = segments[upper_label][family]
            lower_start, lower_end, lower_y = segments[lower_label][family]
            band = MplPolygon([(upper_start, upper_y - bar_half), (upper_end, upper_y - bar_half), (lower_end, lower_y + bar_half), (lower_start, lower_y + bar_half)], closed=True, facecolor=family_color(family), edgecolor='none', alpha=0.12, zorder=1)
            ax.add_patch(band)
    for y, (_, _, label) in zip(y_positions, COMPOSITION_LAYERS):
        left = 0.0
        for family in families:
            row = comp[(comp['layer'] == label) & (comp['family'] == family)]
            value = float(row['share_pct'].iloc[0]) if not row.empty else 0.0
            if value <= 0:
                continue
            ax.barh(y, value, left=left, height=bar_height, color=family_color(family), edgecolor='#FFFFFF', linewidth=0.25, zorder=2)
            if value >= 12.0:
                ax.text(left + value / 2.0, y, family_short(family), ha='center', va='center', fontsize=FONT_INTERNAL_LABEL, color=TEXT_COLOR, clip_on=True)
            left += value
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, len(COMPOSITION_LAYERS) - 0.3)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([label.replace(' global', '') for _, _, label in COMPOSITION_LAYERS], fontsize=FONT_TICK)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel('Cities assigned to each dominant family (%)', fontsize=FONT_AXIS_LABEL, labelpad=2)
    clean_stat_axis(ax)
    ax.grid(False, axis='both')
    add_panel_title(ax, letter, 'Dominant-family prevalence: global models')
    return comp

def smooth_density(hist: np.ndarray, sigma: float=1.05) -> np.ndarray:
    radius = max(1, int(math.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    padded = np.pad(hist, (radius, radius), mode='edge')
    return np.convolve(padded, kernel, mode='same')[radius:-radius]

def draw_strength_panel(ax: mpl.axes.Axes, df: pd.DataFrame, letter: str='f') -> pd.DataFrame:
    records = []
    bins = np.linspace(0.15, 0.8, 42)
    centers = (bins[:-1] + bins[1:]) / 2.0
    y_positions = np.arange(len(STRENGTH_LAYERS))[::-1]
    for y, (outcome, group, label) in zip(y_positions, STRENGTH_LAYERS):
        sub = layer_data(df, outcome, group)
        values = sub['dominant_family_share'].to_numpy(float)
        values = values[np.isfinite(values)]
        hist, _ = np.histogram(values, bins=bins, density=True)
        density = smooth_density(hist)
        if np.nanmax(density) > 0:
            density = density / np.nanmax(density) * 0.34
        color = LAYER_COLORS[label]
        ax.fill_between(centers, y, y + density, color=color, alpha=0.25, linewidth=0, zorder=2)
        ax.plot(centers, y + density, color=color, lw=0.72, zorder=3)
        q25, q50, q75 = np.nanpercentile(values, [25, 50, 75])
        ax.plot([q25, q75], [y - 0.065, y - 0.065], color=color, lw=1.15, solid_capstyle='round', zorder=4)
        ax.plot(q50, y - 0.065, marker='o', ms=2.9, color=color, mec='#FFFFFF', mew=0.35, zorder=5)
        ax.text(0.164, y + 0.5, label, ha='left', va='center', fontsize=FONT_INTERNAL_LABEL, color=TEXT_COLOR, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.72, 'pad': 0.35}, zorder=7)
        records.append({'layer': label, 'n': int(len(values)), 'q25': float(q25), 'median': float(q50), 'q75': float(q75), 'mean': float(np.nanmean(values))})
    ax.set_xlim(0.15, 0.6)
    ax.set_ylim(-0.75, len(STRENGTH_LAYERS) + 0.28)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([])
    ax.set_xticks(np.arange(0.2, 0.61, 0.1))
    ax.set_xlabel('Dominant-family share of local SHAP magnitude', fontsize=FONT_AXIS_LABEL, labelpad=2)
    clean_stat_axis(ax)
    ax.grid(False, axis='both')
    add_panel_title(ax, letter, 'Dominant-family share: global models')
    return pd.DataFrame(records)

def add_map_size_legend(fig: mpl.figure.Figure) -> None:
    ax = fig.add_axes([0.34, 0.405, 0.32, 0.025], label='shared_map_size_legend')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.text(0, 0.5, 'Dominant-family share (%)', va='center', fontsize=FONT_LEGEND)
    for share, x in zip([0.25, 0.5, 0.75], [0.5, 0.68, 0.86]):
        ax.scatter(x, 0.5, s=point_size(share) * MAP_POINT_SIZE_SCALE, facecolor='#B7BEC8', edgecolor='#FFFFFF', linewidth=0.25)
        ax.text(x + 0.025, 0.5, f'{share * 100:.0f}', va='center', fontsize=FONT_LEGEND)

def add_family_legend(fig: mpl.figure.Figure, families: list[str]) -> None:
    handles = [Line2D([0], [0], marker='o', linestyle='', markersize=4.7, markerfacecolor=family_color(family), markeredgecolor='#FFFFFF', markeredgewidth=0.35, label=family_short(family)) for family in families]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -3.0 / 25.4 / fig.get_figheight()), ncol=min(5, max(1, len(handles))), frameon=False, fontsize=FONT_LEGEND, handletextpad=0.34, columnspacing=0.92, borderaxespad=0.0)

def build_figure(df: pd.DataFrame) -> tuple[mpl.figure.Figure, pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, object]]:
    fig = plt.figure(figsize=(7.35, 6.42), facecolor='white')
    gs = fig.add_gridspec(3, 2, left=0.04, right=0.988, top=0.955, bottom=0.105, wspace=0.06, hspace=0.265, height_ratios=[1.03, 1.03, 1.18])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    for ax, (outcome, group, title, letter) in zip(axes, MAP_LAYERS):
        mini_labels = ('ELI North', 'ELI South') if letter == 'a' else ('Ineq. North', 'Ineq. South')
        map_shift = MAP_LEFT_SHIFT_DEG if letter == 'a' else 0.0
        draw_mechanism_map(ax, map_layer_data(df, outcome, group), title, letter, mini_bar_labels=mini_labels, map_x_shift_deg=map_shift)
    bivariate_data = bivariate_city_data(df)
    ax_c = fig.add_subplot(gs[1, 0])
    bivariate_meta = draw_bivariate_outcome_map(ax_c, bivariate_data, letter='c', map_x_shift_deg=MAP_LEFT_SHIFT_DEG)
    draw_bivariate_key_inset(ax_c, bivariate_meta)
    ax_d = fig.add_subplot(gs[1, 1])
    concordance_meta = draw_concordance_map(ax_d, bivariate_data, letter='d')
    ax_e = fig.add_subplot(gs[2, 0])
    ax_f = fig.add_subplot(gs[2, 1])
    comp = draw_composition_panel(ax_e, df, letter='e')
    strength = draw_strength_panel(ax_f, df, letter='f')
    row_shift = 3.0 / 25.4 / fig.get_figheight()
    for ax in (ax_e, ax_f):
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 - row_shift, pos.width, pos.height])
    for ax, labels, offset_mm in [(ax_e, ax_e.get_yticklabels(), -1.0), (ax_f, ax_f.get_xticklabels(), 2.0)]:
        for label in labels:
            anchor = copy(label)
            anchor.set_alpha(0)
            anchor.set_clip_on(False)
            anchor.set_in_layout(True)
            anchor.set_transform(label.get_transform() + mtransforms.ScaledTranslation(offset_mm / 25.4, 0, fig.dpi_scale_trans))
            ax.add_artist(anchor)
    for ax, letter, title in [(ax_e, 'e', 'Dominant-family prevalence: global models'), (ax_f, 'f', 'Dominant-family share: global models')]:
        pos = ax.get_position()
        new_width = pos.width * 0.95
        new_left = pos.x0 if ax is ax_e else pos.x1 - new_width
        new_left += (1.0 if ax is ax_e else -2.0) / 25.4 / fig.get_figwidth()
        headings = [text for text in ax.texts if text.get_text() in {letter, title}]
        heading_x = [pos.x0 + text.get_position()[0] * pos.width for text in headings]
        ax.set_position([new_left, pos.y0, new_width, pos.height])
        for text, x in zip(headings, heading_x):
            text.set_x((x - new_left) / new_width)
    map_family_labels = []
    for outcome, group, _, _ in MAP_LAYERS:
        map_family_labels.extend(map_layer_data(df, outcome, group)['dominant_family_label'].unique().tolist())
    add_family_legend(fig, ordered_families(map_family_labels))
    add_map_size_legend(fig)
    force_black_text(fig)
    return (fig, comp, strength, bivariate_meta, concordance_meta)

def save_outputs(fig: mpl.figure.Figure) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for ext in ['png', 'pdf', 'svg']:
        path = OUT_BASE.with_suffix(f'.{ext}')
        fig.savefig(path, dpi=600, bbox_inches='tight', pad_inches=0.025)
        outputs[ext] = str(path)
    tiff_path = OUT_BASE.with_suffix('.tiff')
    fig.savefig(tiff_path, dpi=600, bbox_inches='tight', pad_inches=0.025, pil_kwargs={'compression': 'tiff_lzw'})
    outputs['tiff'] = str(tiff_path)
    gray_path = OUT_BASE.with_name(f'{OUT_BASE.name}_grayscale.png')
    ImageOps.grayscale(Image.open(resource(outputs['png']))).save(gray_path)
    outputs['grayscale_png'] = str(gray_path)
    return outputs

def write_audit(df: pd.DataFrame, composition: pd.DataFrame, strength: pd.DataFrame, bivariate_meta: dict[str, object], concordance_meta: dict[str, object], outputs: dict[str, str]) -> Path:
    layer_counts = {}
    for outcome, group, label in SUMMARY_LAYERS:
        sub = layer_data(df, outcome, group)
        layer_counts[label] = {'outcome': outcome, 'model_group': group, 'n': int(len(sub)), 'dominant_families': sub['dominant_family_label'].value_counts().to_dict(), 'share_quantiles': {str(q): float(sub['dominant_family_share'].quantile(q)) for q in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]}}
    output_sizes = {}
    for key, value in outputs.items():
        path = Path(value)
        output_sizes[key] = {'path': str(path), 'bytes': int(path.stat().st_size) if resource(path).exists() else None}
    audit = {'figure_contract': {'core_conclusion': 'Different emotional-livability mechanisms dominate in different global city regions, with North-South heterogeneity and outcome-specific shifts from ELI to inequality.', 'figure_archetype': 'map matrix plus marginal statistics', 'target_output': 'Nature-style double-column figure', 'backend': 'Python/matplotlib/geopandas', 'final_size_inches': [7.35, 6.42], 'crs': 'WGS 1984 / EPSG:4326', 'shared_map_size_legend': {'panels': ['a', 'b', 'c', 'd'], 'a': 'ELI dominant-family absolute SHAP share', 'b': 'Gini dominant-family absolute SHAP share', 'c_d': 'Mean of ELI and Gini dominant-family shares', 'legend_percentages': [25, 50, 75], 'size_clipping_share_bounds': [0.18, 0.75], 'common_area_scale': MAP_POINT_SIZE_SCALE}, 'panel_map': {'a': 'North/South-specific city ELI dominant driver family', 'b': 'North/South-specific city inequality dominant driver family, represented by Gini', 'c': 'Merged bivariate city map: observed ELI x observed Gini, with point size showing mean ELI/Gini mechanism strength', 'd': 'ELI-inequality mechanism concordance / transition map', 'e': 'Dominant family composition across ELI and inequality layers', 'f': 'Distribution of dominant-family share for the same layers'}}, 'inputs': {'mechanism_table': str(MECHANISM_TABLE), 'urban_mask_policy': 'No raster is processed in this plotting script; upstream raster driver products must be masked/cropped with global_polygon_1km.tif.', 'urban_mask_exists': resource(URBAN_MASK).exists(), 'urban_mask': str(URBAN_MASK)}, 'data_diagnostics': {'rows': int(len(df)), 'columns': int(df.shape[1]), 'unique_city_ids': int(df['id_1'].nunique()), 'duplicate_id_outcome_group_keys': int(df.duplicated(['id_1', 'outcome', 'model_group']).sum()), 'missing_lonlat_rows': int(df[['lon', 'lat']].isna().any(axis=1).sum()), 'lon_range': [float(df['lon'].min()), float(df['lon'].max())], 'lat_range': [float(df['lat'].min()), float(df['lat'].max())], 'outcomes': sorted(df['outcome'].dropna().unique().tolist()), 'model_groups': sorted(df['model_group'].dropna().unique().tolist()), 'layer_counts': layer_counts}, 'derived_panel_tables': {'composition_rows': int(len(composition)), 'strength_rows': int(len(strength)), 'bivariate_panel': bivariate_meta, 'concordance_panel': concordance_meta}, 'outputs': output_sizes}
    audit_path = OUT_BASE.with_name(f'{OUT_BASE.name}_audit.json')
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    return audit_path

def main() -> None:
    df = load_mechanism_data()
    fig, composition, strength, bivariate_meta, concordance_meta = build_figure(df)
    outputs = save_outputs(fig)
    plt.close(fig)
    audit_path = write_audit(df, composition, strength, bivariate_meta, concordance_meta, outputs)
    print(outputs['png'])
    print(audit_path)

def _build_manuscript_figure():
    return build_figure(load_mechanism_data())[0], None


if __name__ == "__main__":
    _configure_land_functions(sys.modules[__name__])
    _figure, _bbox = _build_manuscript_figure()
    _save_figure(_figure, _bbox, 'Extended_Data_Fig_5', _figure_options)
