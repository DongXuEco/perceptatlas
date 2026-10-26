"""Plot Extended_Data_Fig_6 using the packaged numerical tables or supplied local inputs."""
from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_support"))
from figure_cli import prepare_figure as _prepare_figure, save_figure as _save_figure
from figure_cli import read_table as _read_table, model_tables as _model_tables
from figure_cli import configure_land_functions as _configure_land_functions

_figure_options = _prepare_figure('Extended_Data_Fig_6', 'analysis_tables') if __name__ == "__main__" else None

from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Rectangle
import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from Extended_Data_Fig_5 import BIVARIATE_PALETTE, NCLASS
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
DERIVED_DIR = PROJECT_ROOT / 'derived_data'
RUN_FORMAL = PROJECT_ROOT / 'runs' / '20260818-q3-final-mechanism-models'
OUT_DIR = output_dir('Extended_Data_Fig_6')
OUT_DIR.mkdir(parents=True, exist_ok=True)
ZONE_MASTER = DERIVED_DIR / 'q3_city_zone_driver_master.csv'
PREDICTORS_USED = RUN_FORMAL / 'q3_formal_predictors_used.csv'
URBAN_MASK = PROJECT_ROOT / '全球urbanmask' / 'global_polygon_1km.tif'
LOCAL_MAIN = OUT_DIR / 'q3_core_expansion_local_mechanism_data.csv'
LOCAL_LONG = OUT_DIR / 'q3_core_expansion_local_mechanism_family_long.csv'
TRANSITION_TABLE = OUT_DIR / 'q3_core_expansion_mechanism_transition_city.csv'
LOCAL_AUDIT = OUT_DIR / 'q3_core_expansion_local_mechanism_audit.json'
OUT_BASE = OUT_DIR / 'fig_q3_3_core_expansion_transition_map_atlas'
OUTCOME = 'eli50'
N_ESTIMATORS = 180
MIN_SAMPLES_LEAF = 5
MAX_FEATURES = 0.7
MIN_MODEL_N = 450
RANDOM_STATE = 42
LAND_COLOR = '#E9EBF1'
SEA_COLOR = '#FFFFFF'
TEXT_COLOR = '#000000'
GRID_COLOR = '#E6E9EE'
MUTED = '#000000'
FONT_PANEL_LETTER = 8.2
FONT_PANEL_TITLE = 6.8
FONT_AXIS_LABEL = 7.0
FONT_TICK = 6.4
FONT_LEGEND = 6.2
FONT_LEGEND_TITLE = 6.5
FONT_INTERNAL = 6.2
FONT_BIVAR_TITLE = 6.0
FONT_BIVAR_AXIS = 5.8
FONT_BIVAR_TICK = 5.5
FONT_BIVAR_CORNER = 5.1
MAP_POINT_SIZE_SCALE = 1.0925
MAP_LEFT_SHIFT_DEG = 8.0
Q33_BIVAR_LEGEND_SCALE = 0.799
Q33_BIVAR_TEXT_SCALE = 0.78
FAMILY_COLORS = {'Green/blue space': '#62C0D4', 'Environmental risk': '#E5170A', 'Built environment': '#357DBD', 'Economic/population': '#6E789D', 'POI supply': '#F28817', 'Non-motorized accessibility': '#F1B47A', 'Road network': '#F0CFA1', 'Terrain control': '#909BB6', 'Topographic control': '#909BB6', 'Location control': '#8170A8', 'Other': '#C8D0C1', 'Other/minor drivers': '#C8D0C1'}
FAMILY_ORDER = ['Green/blue space', 'Environmental risk', 'Built environment', 'Economic/population', 'POI supply', 'Non-motorized accessibility', 'Road network', 'Terrain control', 'Topographic control', 'Location control', 'Other', 'Other/minor drivers']
FAMILY_SHORT = {'Green/blue space': 'Green/blue', 'Environmental risk': 'Env. risk', 'Built environment': 'Built env.', 'Economic/population': 'Econ./pop.', 'POI supply': 'POI supply', 'Non-motorized accessibility': 'Non-motor', 'Road network': 'Road', 'Terrain control': 'Terrain', 'Topographic control': 'Terrain', 'Location control': 'Location', 'Other': 'Other', 'Other/minor drivers': 'Other'}
TRANSITION_ORDER = ['same family', 'Green/blue -> POI/access', 'Env. risk -> POI/access', 'Built env. -> other transitions', 'other transitions']
TRANSITION_COLORS = {'same family': '#30343B', 'Green/blue -> POI/access': '#62C0D4', 'Env. risk -> POI/access': '#E5170A', 'Built env. -> other transitions': '#357DBD', 'other transitions': '#C8D0C1'}
TRANSITION_ACCESS_FAMILIES = {'POI supply', 'Non-motorized accessibility', 'Road network'}
GROUP_SPECS = [('core', 'core_expansion', 'core'), ('expansion', 'core_expansion', 'expansion'), ('North_core', 'north_south_x_core_expansion', 'core'), ('North_expansion', 'north_south_x_core_expansion', 'expansion'), ('South_core', 'north_south_x_core_expansion', 'core'), ('South_expansion', 'north_south_x_core_expansion', 'expansion')]
_LAND_GDF: gpd.GeoDataFrame | None = None
_LAND_COORDS: list[np.ndarray] | None = None

@dataclass
class LocalModelAudit:
    model_group: str
    group_type: str
    zone_label: str
    n_rows: int
    n_complete_predictors: int
    max_abs_shap_reconstruction_error: float
    max_family_share_sum_error: float
    top_family_label: str
    top_family_mean_abs_shap: float

def patch_numpy_for_shap() -> None:
    if not hasattr(np, 'bool'):
        np.bool = np.bool_
    if not hasattr(np, 'int'):
        np.int = int

def make_model() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=N_ESTIMATORS, min_samples_leaf=MIN_SAMPLES_LEAF, max_features=MAX_FEATURES, random_state=RANDOM_STATE, n_jobs=-1)

def read_csv(path: Path) -> pd.DataFrame:
    if not resource(path).exists():
        raise FileNotFoundError(path)
    return pd.read_csv(resource(path), low_memory=False)

def coerce_design(df: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    x = df[predictors].apply(pd.to_numeric, errors='coerce')
    return x.replace([np.inf, -np.inf], np.nan)

def group_mask(df: pd.DataFrame, group: str) -> pd.Series:
    zone = df['zone_label'].astype(str)
    ns = df['ns_group'].astype(str)
    if group == 'core':
        return zone.eq('core')
    if group == 'expansion':
        return zone.eq('expansion')
    if group == 'North_core':
        return ns.eq('North') & zone.eq('core')
    if group == 'North_expansion':
        return ns.eq('North') & zone.eq('expansion')
    if group == 'South_core':
        return ns.eq('South') & zone.eq('core')
    if group == 'South_expansion':
        return ns.eq('South') & zone.eq('expansion')
    raise ValueError(f'Unknown group: {group}')

def safe_family_col(prefix: str, family: str) -> str:
    return f'{prefix}__{family}'

def family_color(label: str) -> str:
    return FAMILY_COLORS.get(str(label), FAMILY_COLORS['Other'])

def family_short(label: str) -> str:
    return FAMILY_SHORT.get(str(label), str(label))

def ordered_families(labels: Iterable[str]) -> list[str]:
    present = [str(x) for x in pd.Series(list(labels)).dropna().unique()]
    ordered = [fam for fam in FAMILY_ORDER if fam in present]
    return ordered + sorted([fam for fam in present if fam not in ordered])

def transition_type(core_family: str, expansion_family: str) -> str:
    core = str(core_family)
    expansion = str(expansion_family)
    if core == expansion:
        return 'same family'
    if core == 'Green/blue space' and expansion in TRANSITION_ACCESS_FAMILIES:
        return 'Green/blue -> POI/access'
    if core == 'Environmental risk' and expansion in TRANSITION_ACCESS_FAMILIES:
        return 'Env. risk -> POI/access'
    if core == 'Built environment':
        return 'Built env. -> other transitions'
    return 'other transitions'

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
    if geometry.geom_type == 'Polygon':
        return [np.asarray(geometry.exterior.coords, dtype=float)]
    if geometry.geom_type == 'MultiPolygon':
        return [np.asarray(poly.exterior.coords, dtype=float) for poly in geometry.geoms if not poly.is_empty]
    if geometry.geom_type == 'GeometryCollection':
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
    for coords in land_exterior_coords():
        if len(coords) < 3:
            continue
        ax.fill(coords[:, 0], coords[:, 1], facecolor=LAND_COLOR, edgecolor='#FFFFFF', linewidth=0.1, zorder=0)

def add_panel_title(ax: mpl.axes.Axes, letter: str, title: str) -> None:
    ax.text(-0.008, 1.045, letter, transform=ax.transAxes, ha='left', va='bottom', fontsize=FONT_PANEL_LETTER, fontweight='bold', color=TEXT_COLOR, clip_on=False)
    ax.text(0.05, 1.045, title, transform=ax.transAxes, ha='left', va='bottom', fontsize=FONT_PANEL_TITLE, color=TEXT_COLOR, clip_on=False)

def point_size_from_share(share: pd.Series | np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(share, dtype=float)
    scaled = np.clip((arr - 0.18) / (0.75 - 0.18), 0.0, 1.0)
    size = (5.0 + 23.0 * scaled) * 0.36 * MAP_POINT_SIZE_SCALE
    if np.isscalar(share):
        return float(size)
    return size

def point_size_from_strength(values: pd.Series | np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(values, dtype=float)
    scaled = np.clip((arr - 0.05) / (0.55 - 0.05), 0.0, 1.0)
    size = (5.0 + 23.0 * scaled) * 0.36 * MAP_POINT_SIZE_SCALE
    if np.isscalar(values):
        return float(size)
    return size

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

def load_predictors() -> pd.DataFrame:
    predictors = read_csv(PREDICTORS_USED)
    predictors = predictors.sort_values('selection_order').reset_index(drop=True)
    return predictors

def family_order_from_predictors(predictors: pd.DataFrame) -> list[str]:
    return predictors['family'].drop_duplicates().astype(str).tolist()

def build_one_local_model(zone: pd.DataFrame, predictors: pd.DataFrame, model_group: str, group_type: str, expected_zone_label: str) -> tuple[pd.DataFrame, pd.DataFrame, LocalModelAudit]:
    try:
        import shap
    except Exception as exc:
        raise RuntimeError('The local core-expansion mechanism table requires SHAP. Run this script with an environment that has shap installed, or rerun after q3_core_expansion_local_mechanism_data.csv exists.') from exc
    selected = predictors['column'].astype(str).tolist()
    sub = zone.loc[group_mask(zone, model_group)].copy()
    y_series = pd.to_numeric(sub[OUTCOME], errors='coerce')
    sub = sub.loc[y_series.notna()].copy()
    if len(sub) < MIN_MODEL_N:
        raise ValueError(f'Too few rows for {OUTCOME}/{model_group}: n={len(sub)}')
    y = pd.to_numeric(sub[OUTCOME], errors='coerce').to_numpy(dtype=float)
    x = coerce_design(sub, selected)
    n_complete = int(x.notna().all(axis=1).sum())
    pipe = make_pipeline(SimpleImputer(strategy='median'), make_model())
    pipe.fit(x, y)
    imputer = pipe.named_steps['simpleimputer']
    model = pipe.named_steps['randomforestregressor']
    x_imp = pd.DataFrame(imputer.transform(x), columns=selected, index=sub.index)
    patch_numpy_for_shap()
    explainer = shap.TreeExplainer(model)
    shap_values = np.asarray(explainer.shap_values(x_imp.to_numpy(), check_additivity=False), dtype=float)
    full_prediction = model.predict(x_imp.to_numpy())
    expected_value = float(np.ravel(explainer.expected_value)[0])
    reconstructed = expected_value + shap_values.sum(axis=1)
    reconstruction_error = np.abs(reconstructed - full_prediction)
    meta = predictors.set_index('column').to_dict('index')
    families = family_order_from_predictors(predictors)
    family_abs = {family: np.zeros(len(sub), dtype=float) for family in families}
    family_signed = {family: np.zeros(len(sub), dtype=float) for family in families}
    feature_abs = np.abs(shap_values)
    for col_idx, col in enumerate(selected):
        family = str(meta[col]['family'])
        family_abs[family] += feature_abs[:, col_idx]
        family_signed[family] += shap_values[:, col_idx]
    family_abs_frame = pd.DataFrame(family_abs, index=sub.index)
    family_signed_frame = pd.DataFrame(family_signed, index=sub.index)
    total_abs = family_abs_frame.sum(axis=1).replace(0, np.nan)
    family_share_frame = family_abs_frame.div(total_abs, axis=0).fillna(0.0)
    share_sum_error = np.abs(family_share_frame.sum(axis=1) - 1.0)
    family_labels = predictors.drop_duplicates('family').set_index('family')['family_label'].to_dict()
    dominant_family = family_abs_frame.idxmax(axis=1)
    dominant_abs = np.array([family_abs_frame.loc[idx, fam] for idx, fam in zip(family_abs_frame.index, dominant_family)])
    dominant_signed = np.array([family_signed_frame.loc[idx, fam] for idx, fam in zip(family_signed_frame.index, dominant_family)])
    dominant_share = np.array([family_share_frame.loc[idx, fam] for idx, fam in zip(family_share_frame.index, dominant_family)])
    sorted_shares = np.sort(family_share_frame.to_numpy(), axis=1)
    rank_margin = sorted_shares[:, -1] - sorted_shares[:, -2] if family_share_frame.shape[1] > 1 else sorted_shares[:, -1]
    base_cols = ['id_1', 'COUNTRY', 'NAME_1', 'lon', 'lat', 'NS', 'ns_group', 'zone_code', 'zone_label', OUTCOME, 'positive_score50', 'negative_score50']
    base_cols = [col for col in base_cols if col in sub.columns]
    out = sub[base_cols].copy()
    out['table'] = 'city_zone'
    out['outcome'] = OUTCOME
    out['outcome_role'] = 'primary'
    out['model_group'] = model_group
    out['group_type'] = group_type
    out['observed'] = y
    out['full_model_prediction'] = full_prediction
    out['full_model_residual'] = y - full_prediction
    out['shap_expected_value'] = expected_value
    out['shap_reconstructed_prediction'] = reconstructed
    out['shap_reconstruction_error'] = reconstruction_error
    out['n_predictors'] = len(selected)
    out['has_complete_predictors_zone'] = x.notna().all(axis=1).astype(int).to_numpy()
    out['n_missing_predictors_zone'] = x.isna().sum(axis=1).to_numpy()
    out['local_total_abs_contribution'] = total_abs.to_numpy()
    out['dominant_family'] = dominant_family.to_numpy()
    out['dominant_family_label'] = [family_labels.get(fam, fam) for fam in dominant_family]
    out['dominant_family_color'] = [family_color(family_labels.get(fam, fam)) for fam in dominant_family]
    out['dominant_family_abs_contribution'] = dominant_abs
    out['dominant_family_signed_contribution'] = dominant_signed
    out['dominant_family_share'] = dominant_share
    out['dominant_family_share_margin'] = rank_margin
    for family in families:
        label = family_labels.get(family, family)
        safe = safe_family_col('family_share', family)
        out[safe] = family_share_frame[family].to_numpy()
        out[safe_family_col('family_abs_contribution', family)] = family_abs_frame[family].to_numpy()
        out[safe_family_col('family_signed_contribution', family)] = family_signed_frame[family].to_numpy()
    long_parts = []
    for family in families:
        label = family_labels.get(family, family)
        long_parts.append(pd.DataFrame({'id_1': sub['id_1'].to_numpy(), 'COUNTRY': sub['COUNTRY'].to_numpy() if 'COUNTRY' in sub else '', 'NAME_1': sub['NAME_1'].to_numpy() if 'NAME_1' in sub else '', 'lon': sub['lon'].to_numpy() if 'lon' in sub else np.nan, 'lat': sub['lat'].to_numpy() if 'lat' in sub else np.nan, 'ns_group': sub['ns_group'].to_numpy() if 'ns_group' in sub else '', 'zone_label': sub['zone_label'].to_numpy() if 'zone_label' in sub else expected_zone_label, 'outcome': OUTCOME, 'model_group': model_group, 'family': family, 'family_label': label, 'family_color': family_color(label), 'local_abs_contribution': family_abs_frame[family].to_numpy(), 'local_signed_contribution': family_signed_frame[family].to_numpy(), 'local_share': family_share_frame[family].to_numpy(), 'is_dominant_family': dominant_family.to_numpy() == family}))
    family_long = pd.concat(long_parts, ignore_index=True)
    mean_by_family = family_long.groupby(['family', 'family_label'], as_index=False)['local_abs_contribution'].mean().sort_values('local_abs_contribution', ascending=False)
    top = mean_by_family.iloc[0]
    audit = LocalModelAudit(model_group=model_group, group_type=group_type, zone_label=expected_zone_label, n_rows=int(len(out)), n_complete_predictors=n_complete, max_abs_shap_reconstruction_error=float(np.nanmax(reconstruction_error)), max_family_share_sum_error=float(np.nanmax(share_sum_error)), top_family_label=str(top['family_label']), top_family_mean_abs_shap=float(top['local_abs_contribution']))
    return (out, family_long, audit)

def transition_subset(local: pd.DataFrame, model_group: str, prefix: str) -> pd.DataFrame:
    common = ['id_1', 'COUNTRY', 'NAME_1', 'lon', 'lat', 'NS', 'ns_group']
    metrics = ['zone_label', 'observed', 'full_model_prediction', 'local_total_abs_contribution', 'dominant_family', 'dominant_family_label', 'dominant_family_share', 'dominant_family_share_margin']
    family_share_cols = [col for col in local.columns if col.startswith('family_share__')]
    cols = common + metrics + family_share_cols
    sub = local[local['model_group'].astype(str).eq(model_group)][cols].copy()
    rename = {col: f'{prefix}_{col}' for col in metrics + family_share_cols}
    return sub.rename(columns=rename)

def finalize_transition_table(merged: pd.DataFrame, context: str) -> pd.DataFrame:
    out = merged[['id_1', 'COUNTRY', 'NAME_1', 'lon', 'lat', 'NS', 'ns_group', 'core_observed', 'expansion_observed', 'core_full_model_prediction', 'expansion_full_model_prediction', 'core_local_total_abs_contribution', 'expansion_local_total_abs_contribution', 'core_dominant_family', 'expansion_dominant_family', 'core_dominant_family_label', 'expansion_dominant_family_label', 'core_dominant_family_share', 'expansion_dominant_family_share', 'core_dominant_family_share_margin', 'expansion_dominant_family_share_margin']].copy()
    out['mechanism_context'] = context
    out['core_minus_expansion_eli'] = out['core_observed'] - out['expansion_observed']
    out['mean_dominant_family_share'] = out[['core_dominant_family_share', 'expansion_dominant_family_share']].mean(axis=1)
    out['dominant_share_gap'] = (out['core_dominant_family_share'] - out['expansion_dominant_family_share']).abs()
    core_share_cols = [col for col in merged.columns if col.startswith('core_family_share__')]
    l1 = np.zeros(len(merged), dtype=float)
    for core_col in core_share_cols:
        family = core_col.replace('core_family_share__', '')
        expansion_col = f'expansion_family_share__{family}'
        if expansion_col in merged.columns:
            l1 += (pd.to_numeric(merged[core_col], errors='coerce').fillna(0.0) - pd.to_numeric(merged[expansion_col], errors='coerce').fillna(0.0)).abs().to_numpy()
    out['family_share_l1_distance'] = l1
    out['family_share_total_variation'] = 0.5 * l1
    out['transition_strength'] = out['family_share_total_variation']
    out['top1_same'] = out['core_dominant_family_label'].eq(out['expansion_dominant_family_label'])
    out['dominant_family_transition'] = out['core_dominant_family_label'].map(family_short) + ' -> ' + out['expansion_dominant_family_label'].map(family_short)
    out['transition_type'] = [transition_type(core, expansion) for core, expansion in zip(out['core_dominant_family_label'], out['expansion_dominant_family_label'])]
    return out

def build_transition_tables(local: pd.DataFrame) -> pd.DataFrame:
    core_global = transition_subset(local, 'core', 'core')
    expansion_global = transition_subset(local, 'expansion', 'expansion')
    global_merged = core_global.merge(expansion_global, on=['id_1', 'COUNTRY', 'NAME_1', 'lon', 'lat', 'NS', 'ns_group'], how='inner', validate='one_to_one')
    frames = [finalize_transition_table(global_merged, 'global_zone_models')]
    for ns in ['North', 'South']:
        core = transition_subset(local, f'{ns}_core', 'core')
        expansion = transition_subset(local, f'{ns}_expansion', 'expansion')
        merged = core.merge(expansion, on=['id_1', 'COUNTRY', 'NAME_1', 'lon', 'lat', 'NS', 'ns_group'], how='inner', validate='one_to_one')
        frames.append(finalize_transition_table(merged, 'north_south_zone_models'))
    return pd.concat(frames, ignore_index=True)

def build_local_mechanism_tables(force: bool=False) -> dict[str, object]:
    if resource(LOCAL_MAIN).exists() and resource(LOCAL_LONG).exists() and resource(TRANSITION_TABLE).exists() and (not force):
        return {'rebuilt': False, 'local_main': str(LOCAL_MAIN), 'transition_table': str(TRANSITION_TABLE)}
    started = time.time()
    zone = read_csv(ZONE_MASTER)
    predictors = load_predictors()
    main_frames = []
    long_frames = []
    audits: list[LocalModelAudit] = []
    for model_group, group_type, zone_label in GROUP_SPECS:
        print(f'[core-expansion local SHAP] {OUTCOME} / {model_group}', flush=True)
        main_df, family_long, audit = build_one_local_model(zone, predictors, model_group, group_type, zone_label)
        main_frames.append(main_df)
        long_frames.append(family_long)
        audits.append(audit)
    local_main = pd.concat(main_frames, ignore_index=True)
    local_long = pd.concat(long_frames, ignore_index=True)
    transition = build_transition_tables(local_main)
    local_main.to_csv(LOCAL_MAIN, index=False, encoding='utf-8-sig')
    local_long.to_csv(LOCAL_LONG, index=False, encoding='utf-8-sig')
    transition.to_csv(TRANSITION_TABLE, index=False, encoding='utf-8-sig')
    audit = {'created_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'elapsed_seconds': time.time() - started, 'method': 'Formal city-zone random-forest refit; local SHAP values are aggregated to driver families by absolute contribution for core and expansion models.', 'inputs': {'zone_master': str(ZONE_MASTER), 'predictors_used': str(PREDICTORS_USED), 'urban_mask_policy': 'No raster is processed here; upstream raster products must be clipped or masked by global_polygon_1km.tif.', 'urban_mask': str(URBAN_MASK), 'urban_mask_exists': resource(URBAN_MASK).exists()}, 'rf_parameters': {'n_estimators': N_ESTIMATORS, 'min_samples_leaf': MIN_SAMPLES_LEAF, 'max_features': MAX_FEATURES, 'random_state': RANDOM_STATE}, 'outputs': {'local_main': str(LOCAL_MAIN), 'local_long': str(LOCAL_LONG), 'transition_table': str(TRANSITION_TABLE), 'audit': str(LOCAL_AUDIT)}, 'row_counts': {'local_main_rows': int(len(local_main)), 'local_long_rows': int(len(local_long)), 'transition_rows': int(len(transition)), 'transition_unique_city_ids': int(transition['id_1'].nunique())}, 'model_audits': [asdict(item) for item in audits]}
    LOCAL_AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    return audit | {'rebuilt': True}

def load_plot_tables(force_rebuild: bool=False) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    build_audit = build_local_mechanism_tables(force=force_rebuild)
    local = read_csv(LOCAL_MAIN)
    transition = read_csv(TRANSITION_TABLE)
    for df in [local, transition]:
        df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
        df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    required = ['id_1', 'lon', 'lat', 'ns_group', 'mechanism_context', 'core_dominant_family_label', 'expansion_dominant_family_label', 'transition_type', 'transition_strength']
    missing = [col for col in required if col not in transition.columns]
    if missing:
        raise ValueError(f'Missing required columns in {TRANSITION_TABLE}: {missing}')
    if transition[['lon', 'lat']].isna().any().any():
        raise ValueError('Transition table contains missing WGS84 lon/lat values.')
    if not transition['lon'].between(-180, 180).all() or not transition['lat'].between(-90, 90).all():
        raise ValueError('Transition table coordinates are outside WGS84 lon/lat bounds.')
    return (local, transition, build_audit)

def transition_layer(transition: pd.DataFrame, context: str) -> pd.DataFrame:
    sub = transition[transition['mechanism_context'].astype(str).eq(context)].copy()
    if sub.empty:
        raise ValueError(f'No transition rows for mechanism_context={context}.')
    return sub.sort_values('transition_strength')

def local_family_layer(local: pd.DataFrame, zone_label: str) -> pd.DataFrame:
    parts = []
    for ns, group in [('North', f'North_{zone_label}'), ('South', f'South_{zone_label}')]:
        sub = local[local['model_group'].astype(str).eq(group) & local['ns_group'].astype(str).eq(ns)].copy()
        parts.append(sub)
    data = pd.concat(parts, ignore_index=True)
    if data.empty:
        raise ValueError(f'No local family rows for zone={zone_label}.')
    return data.sort_values('dominant_family_share')

def core_expansion_bivariate_data(transition: pd.DataFrame) -> pd.DataFrame:
    sub = transition[transition['mechanism_context'].astype(str).eq('north_south_zone_models')].copy()
    if sub.empty:
        raise ValueError('No rows for the North/South-specific core-expansion bivariate ELI map.')
    sub['mean_family_share'] = sub[['core_dominant_family_share', 'expansion_dominant_family_share']].mean(axis=1)
    return sub.sort_values('mean_family_share')

def transition_category_display(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, str], dict[str, object]]:
    sub = data.copy()
    core_col = 'core_dominant_family_label'
    expansion_col = 'expansion_dominant_family_label'
    same = sub[core_col].eq(sub[expansion_col])
    counts = sub.loc[~same].groupby([core_col, expansion_col]).size().rename('n').reset_index()
    top_pairs = counts.sort_values(['n', core_col, expansion_col], ascending=[False, True, True]).head(4)
    sub['transition_display'] = 'Other changes'
    sub.loc[same, 'transition_display'] = 'Same family'
    order = ['Same family']
    colors = {'Same family': '#30343B', 'Other changes': '#C8D0C1'}
    for (_, row), color in zip(top_pairs.iterrows(), ['#357DBD', '#62C0D4', '#F28817', '#E5170A']):
        label = f'{family_short(row[core_col])} -> {family_short(row[expansion_col])}'
        mask = sub[core_col].eq(row[core_col]) & sub[expansion_col].eq(row[expansion_col])
        sub.loc[mask, 'transition_display'] = label
        order.append(label)
        colors[label] = color
    order.append('Other changes')
    display_counts = sub['transition_display'].value_counts().reindex(order, fill_value=0)
    meta = {'category_rule': 'Same family; four most frequent distinct core-to-expansion family pairs in the paired sample; all remaining changes. Arrows compare zones, not time.', 'display_transition_counts': {label: int(display_counts[label]) for label in order}, 'display_transition_shares': {label: float(display_counts[label] / len(sub)) for label in order}}
    order_map = {label: i for i, label in enumerate(order)}
    sub['transition_display_order'] = sub['transition_display'].map(order_map).fillna(len(order)).astype(int)
    sub = sub.sort_values(['transition_display_order', 'transition_strength'])
    return (sub, order, colors, meta)

def draw_bivariate_key_inset(ax: mpl.axes.Axes, bivariate_meta: dict[str, object]) -> None:
    legend_scale = Q33_BIVAR_LEGEND_SCALE
    text_scale = Q33_BIVAR_TEXT_SCALE
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
    leg.set_xticks([0, NCLASS / 2, NCLASS])
    leg.set_yticks([0, NCLASS / 2, NCLASS])
    leg.set_xticklabels([f"{bivariate_meta['x_bounds'][0]:.2f}", '0', f"{bivariate_meta['x_bounds'][1]:.2f}"], fontsize=FONT_BIVAR_TICK * text_scale)
    leg.set_yticklabels([f"{bivariate_meta['y_bounds'][0]:.2f}", '0', f"{bivariate_meta['y_bounds'][1]:.2f}"], fontsize=FONT_BIVAR_TICK * text_scale)
    leg.tick_params(length=1.35 * legend_scale, width=0.32 * legend_scale, pad=1.05, colors=TEXT_COLOR)
    leg.set_xlabel('Core ELI', fontsize=FONT_BIVAR_AXIS * text_scale, labelpad=1.7)
    leg.set_ylabel('Expansion ELI', fontsize=FONT_BIVAR_AXIS * text_scale, labelpad=1.7)
    leg.set_title('Core x expansion', fontsize=FONT_BIVAR_TITLE * text_scale, pad=1.4, color=TEXT_COLOR)
    for spine in leg.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.36 * legend_scale)
        spine.set_color('#4A4F54')
    leg.text(0.38, 0.35, 'Co-low', ha='left', va='bottom', fontsize=FONT_BIVAR_CORNER * text_scale, color=TEXT_COLOR)
    leg.text(9.45, 9.45, 'Co-high', ha='right', va='top', fontsize=FONT_BIVAR_CORNER * text_scale, color=TEXT_COLOR)

def draw_core_expansion_bivariate_map(ax: mpl.axes.Axes, data: pd.DataFrame, letter: str) -> dict[str, object]:
    setup_map_axis(ax, x_shift_deg=MAP_LEFT_SHIFT_DEG)
    draw_land(ax)
    combined = pd.concat([data['core_observed'], data['expansion_observed']], ignore_index=True)
    x_bounds = robust_bivariate_bounds(combined, center_zero=True)
    y_bounds = x_bounds
    rgba, valid, x_cls, y_cls = make_bivariate_rgba(data['core_observed'], data['expansion_observed'], x_bounds, y_bounds)
    sub = data.loc[valid].copy()
    sizes = point_size_from_share(0.4)
    ax.scatter(sub['lon'], sub['lat'], s=sizes, c=rgba[valid], linewidths=0.08, edgecolors='#FFFFFF', alpha=0.86, zorder=3, rasterized=True)
    add_panel_title(ax, letter, 'Observed ELI: core and expansion')
    class_counts = pd.DataFrame({'x_class': x_cls[valid], 'y_class': y_cls[valid]}).value_counts(['x_class', 'y_class']).rename('n').reset_index()
    return {'n_points': int(valid.sum()), 'x_variable': 'core_observed_eli50', 'y_variable': 'expansion_observed_eli50', 'x_bounds': [float(x_bounds[0]), float(x_bounds[1])], 'y_bounds': [float(y_bounds[0]), float(y_bounds[1])], 'non_empty_classes': int(len(class_counts)), 'max_class_count': int(class_counts['n'].max()) if not class_counts.empty else 0}

def draw_transition_legend(ax: mpl.axes.Axes, order: list[str] | None=None, colors: dict[str, str] | None=None) -> None:
    legend_scale = 0.84
    order = TRANSITION_ORDER if order is None else order
    colors = TRANSITION_COLORS if colors is None else colors
    handles = [Line2D([0], [0], marker='o', linestyle='', markersize=3.9 * legend_scale, markerfacecolor=colors[label], markeredgecolor='#FFFFFF', markeredgewidth=0.28 * legend_scale, label=label) for label in order]
    leg = ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(-0.048, -0.058), ncol=1, fontsize=FONT_LEGEND * legend_scale, title='Core to expansion', title_fontsize=FONT_LEGEND_TITLE * legend_scale, handletextpad=0.34, labelspacing=0.23, borderpad=0.24, frameon=False)
    leg.get_frame().set_facecolor('#F3F5F7')
    leg.get_frame().set_edgecolor('none')
    leg.get_frame().set_alpha(0.78)
    title_shift_inches = 0.06 * ax.get_position().width * ax.figure.get_figwidth() * legend_scale
    leg.get_title().set_transform(leg.get_title().get_transform() + mtransforms.ScaledTranslation(-title_shift_inches, 0.0, ax.figure.dpi_scale_trans))
    for text in leg.get_texts():
        text.set_color(TEXT_COLOR)
    leg.get_title().set_color(TEXT_COLOR)

def draw_transition_map(ax: mpl.axes.Axes, data: pd.DataFrame, letter: str, title: str, add_legend: bool=False, label_col: str='transition_type', order: list[str] | None=None, colors: dict[str, str] | None=None) -> None:
    order = TRANSITION_ORDER if order is None else order
    colors = TRANSITION_COLORS if colors is None else colors
    setup_map_axis(ax)
    draw_land(ax)
    for label in order:
        sub = data[data[label_col].astype(str).eq(label)]
        if sub.empty:
            continue
        ax.scatter(sub['lon'], sub['lat'], s=point_size_from_strength(sub['transition_strength']), c=colors[label], edgecolors='#FFFFFF', linewidths=0.2, alpha=0.86, zorder=2, rasterized=True)
    add_panel_title(ax, letter, title)
    if add_legend:
        draw_transition_legend(ax, order=order, colors=colors)

def draw_ns_composition_minibars(ax: mpl.axes.Axes, data: pd.DataFrame) -> None:
    groups = ('North', 'South')
    x0 = 0.042
    y0 = 0.012
    bar_w = 0.04477824
    bar_h = 0.506677248
    bar_gap = 0.045
    bar_specs = []
    for i, group in enumerate(groups):
        x = x0 + i * (bar_w + bar_gap)
        sub = data[data['ns_group'].astype(str).eq(group)]
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
        bar_specs.append({'group': group, 'x': x, 'families': families, 'spans': spans})
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
        label_short = 'North' if spec['group'] == 'North' else 'South'
        ax.text(x + bar_w / 2, y0 - 0.032, label_short, transform=ax.transAxes, ha='center', va='top', fontsize=FONT_LEGEND - 0.6, color=TEXT_COLOR, clip_on=False, zorder=9)
        for family in spec['families']:
            if family not in spec['spans']:
                continue
            bottom, top = spec['spans'][family]
            ax.add_patch(Rectangle((x, bottom), bar_w, top - bottom, transform=ax.transAxes, facecolor=family_color(family), edgecolor='#FFFFFF', linewidth=0.22, clip_on=True, zorder=8))

def draw_family_map(ax: mpl.axes.Axes, data: pd.DataFrame, letter: str, title: str, mini_bars: bool=False, map_x_shift_deg: float=0.0) -> None:
    setup_map_axis(ax, x_shift_deg=map_x_shift_deg)
    draw_land(ax)
    for family in ordered_families(data['dominant_family_label']):
        sub = data[data['dominant_family_label'].astype(str).eq(family)]
        if sub.empty:
            continue
        ax.scatter(sub['lon'], sub['lat'], s=point_size_from_share(sub['dominant_family_share']), c=family_color(family), edgecolors='#FFFFFF', linewidths=0.2, alpha=0.86, zorder=2, rasterized=True)
    add_panel_title(ax, letter, title)
    if mini_bars:
        draw_ns_composition_minibars(ax, data)

def draw_transition_matrix(ax: mpl.axes.Axes, data: pd.DataFrame, letter: str) -> pd.DataFrame:
    families = ordered_families(pd.concat([data['core_dominant_family_label'], data['expansion_dominant_family_label']], ignore_index=True))
    counts = data.groupby(['core_dominant_family_label', 'expansion_dominant_family_label'], as_index=False).size().rename(columns={'size': 'n'})
    total = float(counts['n'].sum())
    counts['share'] = counts['n'] / total
    x_map = {family: i for i, family in enumerate(families)}
    y_map = {family: i for i, family in enumerate(families)}
    counts['x'] = counts['expansion_dominant_family_label'].map(x_map)
    counts['y'] = counts['core_dominant_family_label'].map(y_map)
    ax.set_facecolor('white')
    ax.set_xlim(-0.5, len(families) - 0.5)
    ax.set_ylim(len(families) - 0.5, -0.5)
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels([family_short(f) for f in families], rotation=35, ha='right', fontsize=FONT_TICK)
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels([family_short(f) for f in families], fontsize=FONT_TICK)
    ax.tick_params(axis='both', length=2.4, width=0.55, colors=TEXT_COLOR, pad=1.5)
    for pos in np.arange(-0.5, len(families), 1.0):
        ax.axhline(pos, color=GRID_COLOR, lw=0.45, zorder=0)
        ax.axvline(pos, color=GRID_COLOR, lw=0.45, zorder=0)
    sizes = 50 + 1280 * counts['share'].to_numpy()
    colors = [family_color(f) for f in counts['core_dominant_family_label']]
    ax.scatter(counts['x'], counts['y'], s=sizes, c=colors, edgecolors='#FFFFFF', linewidths=0.35, alpha=0.88, zorder=2)
    for _, row in counts.iterrows():
        if row['share'] >= 0.035:
            ax.text(row['x'], row['y'], f"{row['share'] * 100:.0f}%", ha='center', va='center', fontsize=FONT_INTERNAL, color='#000000', zorder=3)
    ax.set_xlabel('Expansion dominant family', fontsize=FONT_AXIS_LABEL, labelpad=2)
    ax.set_ylabel('Core dominant family', fontsize=FONT_AXIS_LABEL, labelpad=2)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(0.65)
        ax.spines[spine].set_color(TEXT_COLOR)
    add_panel_title(ax, letter, 'Core-expansion transition matrix')
    return counts

def smooth_hist_density(values: np.ndarray, xmin: float, xmax: float, bins: int=80) -> tuple[np.ndarray, np.ndarray]:
    values = values[np.isfinite(values)]
    hist, edges = np.histogram(values, bins=bins, range=(xmin, xmax), density=True)
    kernel = np.array([1, 2, 3, 2, 1], dtype=float)
    kernel = kernel / kernel.sum()
    smoothed = np.convolve(hist, kernel, mode='same')
    centers = (edges[:-1] + edges[1:]) / 2
    if np.nanmax(smoothed) > 0:
        smoothed = smoothed / np.nanmax(smoothed)
    return (centers, smoothed)

def draw_strength_panel(ax: mpl.axes.Axes, data: pd.DataFrame, letter: str) -> pd.DataFrame:
    sub = data[data['mechanism_context'].astype(str).eq('north_south_zone_models')].copy()
    rows = [('North', '#0D47A1'), ('South', '#F27C2B'), ('All', '#30343B')]
    xmin, xmax = (0.0, 0.65)
    y_positions = np.arange(len(rows))[::-1]
    ax.set_facecolor('white')
    for y, (label, color) in zip(y_positions, rows):
        vals = sub['transition_strength'].to_numpy(float) if label == 'All' else sub.loc[sub['ns_group'].eq(label), 'transition_strength'].to_numpy(float)
        xs, dens = smooth_hist_density(vals, xmin, xmax)
        scale = 0.33
        ax.fill_between(xs, y, y + dens * scale, color=color, alpha=0.18, linewidth=0, zorder=1)
        ax.plot(xs, y + dens * scale, color=color, lw=1.05, zorder=2)
        med = float(np.nanmedian(vals))
        q25, q75 = np.nanquantile(vals, [0.25, 0.75])
        ax.plot([q25, q75], [y - 0.06, y - 0.06], color=color, lw=1.5, solid_capstyle='round', zorder=3)
        ax.scatter([med], [y - 0.06], s=13, color=color, edgecolor='#FFFFFF', linewidth=0.3, zorder=4)
        ax.text(xmin + 0.012, y + 0.1, label, ha='left', va='bottom', fontsize=FONT_AXIS_LABEL, color=TEXT_COLOR)
        ax.text(xmax - 0.012, y + 0.1, f'n={len(vals):,}', ha='right', va='bottom', fontsize=FONT_LEGEND, color=TEXT_COLOR)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.35, len(rows) - 0.25)
    ax.set_yticks([])
    ax.set_xticks(np.arange(0.0, 0.651, 0.1))
    ax.tick_params(axis='x', labelsize=FONT_TICK, colors=TEXT_COLOR, length=2.8, width=0.6, pad=2)
    ax.set_xlabel('Family-share total variation', fontsize=FONT_AXIS_LABEL, labelpad=2)
    for spine in ['left', 'right', 'top']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.65)
    ax.spines['bottom'].set_color(TEXT_COLOR)
    add_panel_title(ax, letter, 'Transition-strength distribution')
    summary = sub.groupby('ns_group')['transition_strength'].agg(n='size', median='median', q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75)).reset_index()
    return summary

def add_family_legend(fig: mpl.figure.Figure, families: list[str]) -> None:
    handles = [Line2D([0], [0], marker='o', linestyle='', markersize=4.7, markerfacecolor=family_color(family), markeredgecolor='#FFFFFF', markeredgewidth=0.3, label=family_short(family)) for family in families]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.35, 0.57), ncol=min(5, max(1, len(handles))), frameon=False, fontsize=FONT_LEGEND, handletextpad=0.34, columnspacing=0.92, borderaxespad=0.0)

def add_point_size_legend(fig: mpl.figure.Figure) -> None:
    specs = [(0.565, 'a,b: Dominant-family share (%)', [0.25, 0.5, 0.75], point_size_from_share), (0.505, 'd: Contribution dissimilarity (TV)', [0.1, 0.3, 0.5], point_size_from_strength)]
    for y, title, values, size_fn in specs:
        fig.text(0.66, y + 0.025, title, fontsize=FONT_LEGEND - 0.5, va='center')
        ax = fig.add_axes([0.66, y - 0.012, 0.29, 0.024], label=title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        for x, value in zip([0.02, 0.35, 0.68], values):
            ax.scatter(x, 0.5, s=size_fn(value), facecolor='#AEB8C6', edgecolor='#FFFFFF', linewidth=0.2, alpha=0.86, clip_on=False)
            label = f'{value * 100:.0f}' if size_fn is point_size_from_share else f'{value:.2f}'
            ax.text(x + 0.055, 0.5, label, va='center', fontsize=FONT_LEGEND - 0.5)

def force_black_text(fig: mpl.figure.Figure) -> None:
    for text in fig.findobj(lambda artist: isinstance(artist, mpl.text.Text)):
        text.set_color(TEXT_COLOR)

def build_figure(local: pd.DataFrame, transition: pd.DataFrame) -> tuple[mpl.figure.Figure, dict[str, object], dict[str, object]]:
    fig = plt.figure(figsize=(7.35, 4.42), facecolor='white')
    gs = fig.add_gridspec(2, 2, left=0.04, right=0.988, top=0.955, bottom=0.125, wspace=0.06, hspace=0.265, height_ratios=[1.03, 1.03])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    for ax in (ax_c, ax_d):
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 - 0.025, pos.width, pos.height])
    ns_transition = transition_layer(transition, 'north_south_zone_models')
    core_layer = local_family_layer(local, 'core')
    expansion_layer = local_family_layer(local, 'expansion')
    paired_ids = ns_transition['id_1']
    core_layer = core_layer.loc[core_layer['id_1'].isin(paired_ids)].copy()
    expansion_layer = expansion_layer.loc[expansion_layer['id_1'].isin(paired_ids)].copy()
    draw_family_map(ax_a, core_layer, 'a', 'Dominant predictor families: core', mini_bars=True, map_x_shift_deg=MAP_LEFT_SHIFT_DEG)
    draw_family_map(ax_b, expansion_layer, 'b', 'Dominant predictor families: expansion', mini_bars=True)
    bivariate_meta = draw_core_expansion_bivariate_map(ax_c, core_expansion_bivariate_data(transition), 'c')
    draw_bivariate_key_inset(ax_c, bivariate_meta)
    top_transition, top_order, top_colors, top_transition_meta = transition_category_display(ns_transition)
    draw_transition_map(ax_d, top_transition, 'd', 'Dominant-family changes: core to expansion', add_legend=True, label_col='transition_display', order=top_order, colors=top_colors)
    family_labels = []
    family_labels.extend(core_layer['dominant_family_label'].unique().tolist())
    family_labels.extend(expansion_layer['dominant_family_label'].unique().tolist())
    add_family_legend(fig, ordered_families(family_labels))
    add_point_size_legend(fig)
    force_black_text(fig)
    return (fig, bivariate_meta, top_transition_meta)

def save_outputs(fig: mpl.figure.Figure) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for ext in ['png', 'pdf', 'svg']:
        path = OUT_BASE.with_suffix(f'.{ext}')
        fig.savefig(path, dpi=600, bbox_inches='tight', pad_inches=0.025)
        outputs[ext] = str(path)
    tiff_path = OUT_BASE.with_suffix('.tiff')
    fig.savefig(tiff_path, dpi=600, bbox_inches='tight', pad_inches=0.025, pil_kwargs={'compression': 'tiff_lzw'})
    outputs['tiff'] = str(tiff_path)
    return outputs

def write_audit(local: pd.DataFrame, transition: pd.DataFrame, bivariate_meta: dict[str, object], top_transition_meta: dict[str, object], build_audit: dict[str, object], outputs: dict[str, str]) -> Path:
    output_sizes = {}
    for key, value in outputs.items():
        path = Path(value)
        output_sizes[key] = {'path': str(path), 'bytes': int(path.stat().st_size) if resource(path).exists() else None}
    layer_counts = {}
    for context in ['global_zone_models', 'north_south_zone_models']:
        sub = transition_layer(transition, context)
        transition_labels = pd.Series([transition_type(core, expansion) for core, expansion in zip(sub['core_dominant_family_label'], sub['expansion_dominant_family_label'])], index=sub.index)
        layer_counts[context] = {'n': int(len(sub)), 'unique_city_ids': int(sub['id_1'].nunique()), 'transition_type_counts': transition_labels.value_counts().reindex(TRANSITION_ORDER, fill_value=0).to_dict(), 'transition_strength_quantiles': {str(q): float(sub['transition_strength'].quantile(q)) for q in [0, 0.25, 0.5, 0.75, 1.0]}}
    audit = {'figure_contract': {'core_conclusion': 'Core and expansion zones often have different dominant ELI predictor families, with geographically heterogeneous model attribution rather than causal identification.', 'figure_archetype': 'two-by-two map atlas', 'target_output': 'Nature-style double-column figure', 'backend': 'Python/matplotlib/geopandas/sklearn/SHAP', 'crs': 'WGS 1984 / EPSG:4326', 'final_size_inches': [7.35, 4.42], 'panel_map': {'a': 'Core ELI dominant predictor families using North/South-specific core models, paired cities only.', 'b': 'Expansion ELI dominant predictor families using North/South-specific expansion models, paired cities only.', 'c': 'Observed core ELI by expansion ELI, same paired sample and fixed point area.', 'd': 'Same dominant family, four most frequent distinct family pairs, and other changes in paired cities.'}, 'point_size': {'a_b': 'Dominant-family share of absolute SHAP magnitude; clipped at 0.18 and 0.75.', 'c': 'Constant point area, no quantitative size encoding.', 'd': 'Total variation = 0.5 * sum(abs(core family share - expansion family share)); clipped at 0.05 and 0.55.', 'legends': 'Separate area-matched scatter legends for a/b and d.'}}, 'inputs': {'zone_master': str(ZONE_MASTER), 'predictors_used': str(PREDICTORS_USED), 'urban_mask_policy': 'This plotting script does not process rasters. All upstream raster driver products must be masked or cropped with global_polygon_1km.tif.', 'urban_mask': str(URBAN_MASK), 'urban_mask_exists': resource(URBAN_MASK).exists()}, 'build_audit': build_audit, 'data_diagnostics': {'local_rows': int(len(local)), 'local_unique_city_ids': int(local['id_1'].nunique()), 'transition_rows': int(len(transition)), 'transition_unique_city_ids': int(transition['id_1'].nunique()), 'missing_transition_lonlat': int(transition[['lon', 'lat']].isna().any(axis=1).sum()), 'lon_range': [float(transition['lon'].min()), float(transition['lon'].max())], 'lat_range': [float(transition['lat'].min()), float(transition['lat'].max())], 'layer_counts': layer_counts}, 'derived_panel_tables': {'paired_sample_by_region': transition_layer(transition, 'north_south_zone_models')['ns_group'].value_counts().to_dict(), 'bivariate_panel': bivariate_meta, 'transition_category_panel': top_transition_meta}, 'outputs': output_sizes}
    audit_path = OUT_BASE.with_name(f'{OUT_BASE.name}_audit.json')
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    return audit_path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force-rebuild-local', action='store_true', help='Refit city-zone models and rebuild local SHAP transition tables even if cached outputs exist.')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    local, transition, build_audit = load_plot_tables(force_rebuild=args.force_rebuild_local)
    fig, bivariate_meta, top_transition_meta = build_figure(local, transition)
    outputs = save_outputs(fig)
    plt.close(fig)
    audit_path = write_audit(local, transition, bivariate_meta, top_transition_meta, build_audit, outputs)
    print(outputs['png'])
    print(audit_path)

def _build_manuscript_figure():
    local = _read_table("Extended_Data_Fig_6/Extended_Data_Fig_6_local_mechanism_data.csv")
    transition = _read_table("Extended_Data_Fig_6/Extended_Data_Fig_6_mechanism_transition_city.csv")
    return build_figure(local, transition)[0], None


if __name__ == "__main__":
    _configure_land_functions(sys.modules[__name__])
    _figure, _bbox = _build_manuscript_figure()
    _save_figure(_figure, _bbox, 'Extended_Data_Fig_6', _figure_options)
