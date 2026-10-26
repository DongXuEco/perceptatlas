from __future__ import annotations
from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
import json
import platform
import re
import sys
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except Exception:
    ccrs = None
    cfeature = None
VALIDATION_DIR = Path(__file__).resolve().parent
GEE_BASE = VALIDATION_DIR / 'GEE计算可达性'
API_CSV = VALIDATION_DIR / 'accessibility_validation_route_times_full27130_google_routes.csv'
OUT_DIR = output_dir('Supplementary_Fig_S10')
MODE_DIRS = {'walking': GEE_BASE / 'walking_global_accessibility_points' / 'new', 'driving': GEE_BASE / 'driving_global_accessibility_points' / 'new'}
EXPECTED_TILE_COUNT = 357
AXIS_LIMITS_MIN = {'walking': 600.0, 'driving': 200.0}
BAND_TO_POI = {'T_education_min': 'education', 'T_entertainment_min': 'entertainment', 'T_health_min': 'health', 'T_life_min': 'living', 'T_production_min': 'production', 'T_transport_min': 'transport'}
API_POI_ALIAS = {'education': 'education', 'entertainment': 'entertainment', 'health': 'health', 'healthcare': 'health', 'health care': 'health', 'life': 'living', 'living': 'living', 'production': 'production', 'working': 'production', 'work': 'production', 'transport': 'transport'}
POI_ORDER = ['education', 'entertainment', 'health', 'living', 'production', 'transport']
POI_LABEL = {'education': 'Education', 'entertainment': 'Entertainment', 'health': 'Health', 'living': 'Living', 'production': 'Production', 'transport': 'Transport'}
MODE_LABEL = {'walking': 'Walking', 'driving': 'Driving'}
POI_COLOR = {'education': '#2F6DB3', 'entertainment': '#E9842A', 'health': '#C8403A', 'living': '#55B6C6', 'production': '#737FA5', 'transport': '#DCA55C'}
MODE_COLOR = {'walking': '#2F6DB3', 'driving': '#E9842A'}
OUT_GEE_LONG = OUT_DIR / 'fig_accessibility_validation_gee_long.csv'
OUT_SOURCE = OUT_DIR / 'fig_accessibility_validation_source_data.csv'
OUT_SOURCE_PLOT = OUT_DIR / 'fig_accessibility_validation_source_data_plot_window.csv'
OUT_METRICS = OUT_DIR / 'fig_accessibility_validation_metrics.csv'
OUT_AUDIT = OUT_DIR / 'fig_accessibility_validation_audit.json'
OUT_PNG = OUT_DIR / 'fig_accessibility_validation_map_scatter_density.png'
OUT_TIFF = OUT_DIR / 'fig_accessibility_validation_map_scatter_density.tiff'
OUT_PDF = OUT_DIR / 'fig_accessibility_validation_map_scatter_density.pdf'
OUT_SVG = OUT_DIR / 'fig_accessibility_validation_map_scatter_density.svg'

def parse_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({'true', '1', 'yes', 'y'})

def normalize_poi(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().replace(API_POI_ALIAS)

def standard_tile_files(mode: str, folder: Path) -> tuple[list[Path], list[str], list[int]]:
    pattern = re.compile(f'^{mode}_nc2025_accessibility_points_tile_(\\d{{3}})\\.csv$', re.I)
    files: list[Path] = []
    tile_ids: list[int] = []
    nonstandard: list[str] = []
    for path in sorted(resource_glob(folder, '*.csv')):
        match = pattern.match(path.name)
        if match:
            files.append(path)
            tile_ids.append(int(match.group(1)))
        else:
            nonstandard.append(path.name)
    return (files, nonstandard, tile_ids)

def read_gee_mode(mode: str, folder: Path) -> tuple[pd.DataFrame, dict]:
    files, nonstandard_files, tile_ids = standard_tile_files(mode, folder)
    if not files:
        raise FileNotFoundError(f'No standard {mode} tile CSV files found in {folder}')
    meta_cols = ['origin_id', 'sample_id', 'id_1', 'COUNTRY', 'NAME_1', 'city_index', 'origin_lon', 'origin_lat', 'lon', 'lat', 'lat_band', 'lon_band', 'geo_strat', 'tile_id', 'raw_tile_id', 'sample_scale_m', 'tile_buffer_m', 'max_distance_m']
    frames = []
    file_rows: dict[str, int] = {}
    for path in files:
        header = pd.read_csv(resource(path), nrows=0).columns.tolist()
        missing_bands = [col for col in BAND_TO_POI if col not in header]
        if missing_bands:
            raise ValueError(f'{path.name} missing expected GEE bands: {missing_bands}')
        usecols = [col for col in meta_cols if col in header] + list(BAND_TO_POI)
        df = pd.read_csv(resource(path), usecols=usecols, low_memory=False)
        df['source_file'] = path.name
        frames.append(df)
        file_rows[path.name] = int(len(df))
    wide = pd.concat(frames, ignore_index=True)
    long_df = wide.melt(id_vars=[col for col in wide.columns if col not in BAND_TO_POI], value_vars=list(BAND_TO_POI), var_name='gee_band', value_name='gee_time_min')
    long_df['origin_id'] = long_df['origin_id'].astype(str)
    long_df['mode'] = mode
    long_df['poi_type'] = long_df['gee_band'].map(BAND_TO_POI)
    long_df['gee_time_min'] = pd.to_numeric(long_df['gee_time_min'], errors='coerce')
    long_df = long_df.dropna(subset=['origin_id', 'poi_type'])
    long_df = long_df[np.isfinite(long_df['gee_time_min'])].copy()
    key_cols = ['origin_id', 'mode', 'poi_type']
    duplicate_rows = int(long_df.duplicated(key_cols).sum())
    keep_first = [col for col in meta_cols + ['gee_band'] if col in long_df.columns and col not in key_cols]
    grouped = long_df.groupby(key_cols, as_index=False)
    collapsed = grouped.agg({'gee_time_min': 'median', 'source_file': lambda values: '|'.join(sorted(set(map(str, values)))), **{col: 'first' for col in keep_first}})
    duplicate_counts = grouped.agg(gee_input_row_count=('source_file', 'size'), gee_source_file_count=('source_file', 'nunique'))
    collapsed = collapsed.merge(duplicate_counts, on=key_cols, how='left')
    expected_ids = set(range(EXPECTED_TILE_COUNT))
    observed_ids = set(tile_ids)
    audit = {'folder': str(folder), 'csv_total': len(list(resource_glob(folder, '*.csv'))), 'standard_tile_file_count': len(files), 'standard_tile_row_count_wide': int(len(wide)), 'standard_tile_unique_origins': int(wide['origin_id'].nunique()), 'long_rows_after_duplicate_collapse': int(len(collapsed)), 'duplicate_origin_mode_poi_rows_collapsed_by_median': duplicate_rows, 'missing_tile_ids': sorted(expected_ids - observed_ids), 'nonstandard_csv_files_excluded': nonstandard_files, 'first_8_tile_row_counts': list(file_rows.items())[:8], 'last_8_tile_row_counts': list(file_rows.items())[-8:]}
    return (collapsed, audit)

def read_api_reference() -> tuple[pd.DataFrame, dict]:
    if not resource(API_CSV).exists():
        raise FileNotFoundError(API_CSV)
    usecols = ['origin_id', 'mode', 'poi_type', 'api_status', 'route_success', 'clean_for_validation', 'route_time_min', 'route_distance_m', 'COUNTRY', 'NAME_1', 'id_1', 'sample_id', 'origin_lon', 'origin_lat']
    api = pd.read_csv(resource(API_CSV), usecols=lambda col: col in usecols, low_memory=False)
    api['origin_id'] = api['origin_id'].astype(str)
    api['mode'] = api['mode'].astype(str).str.strip().str.lower()
    api['poi_type'] = normalize_poi(api['poi_type'])
    api['api_time_min'] = pd.to_numeric(api['route_time_min'], errors='coerce')
    api['route_distance_m'] = pd.to_numeric(api['route_distance_m'], errors='coerce')
    clean = api['api_status'].astype(str).str.strip().str.lower().eq('ok')
    if 'route_success' in api.columns:
        clean &= parse_bool(api['route_success'])
    if 'clean_for_validation' in api.columns:
        clean &= parse_bool(api['clean_for_validation'])
    clean &= api['mode'].isin(MODE_DIRS)
    clean &= api['poi_type'].isin(POI_ORDER)
    clean &= np.isfinite(api['api_time_min'])
    clean &= api['api_time_min'].gt(0.1)
    clean &= api['api_time_min'].le(1440)
    clean &= api['route_distance_m'].isna() | api['route_distance_m'].le(500000)
    api_clean = api.loc[clean].copy()
    duplicate_rows = int(api_clean.duplicated(['origin_id', 'mode', 'poi_type']).sum())
    api_clean = api_clean.sort_values(['origin_id', 'mode', 'poi_type', 'api_time_min']).drop_duplicates(['origin_id', 'mode', 'poi_type'], keep='first').reset_index(drop=True)
    keep_cols = ['origin_id', 'mode', 'poi_type', 'api_time_min', 'route_distance_m', 'api_status', 'COUNTRY', 'NAME_1', 'id_1', 'sample_id', 'origin_lon', 'origin_lat']
    keep_cols = [col for col in keep_cols if col in api_clean.columns]
    audit = {'api_csv': str(API_CSV), 'api_rows_total': int(len(api)), 'api_rows_after_clean_filters': int(len(api_clean)), 'api_duplicate_origin_mode_poi_rows_dropped': duplicate_rows, 'api_clean_counts_by_mode': api_clean['mode'].value_counts().sort_index().to_dict(), 'api_clean_counts_by_mode_poi': {f'{mode}|{poi}': int(count) for (mode, poi), count in api_clean.groupby(['mode', 'poi_type']).size().items()}}
    return (api_clean[keep_cols], audit)

def metric_block(df: pd.DataFrame, mode: str, poi_type: str, axis_limit: float | None) -> dict:
    data = df.copy()
    if axis_limit is not None:
        data = data[data['api_time_min'].between(0, axis_limit) & data['gee_time_min'].between(0, axis_limit)].copy()
    x = data['api_time_min'].to_numpy(float)
    y = data['gee_time_min'].to_numpy(float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(len(x))
    residual = y - x
    pearson_r = pearson_p = spearman_rho = spearman_p = slope = intercept = np.nan
    if n >= 3 and np.nanstd(x) > 0 and (np.nanstd(y) > 0):
        pearson = stats.pearsonr(x, y)
        spearman = stats.spearmanr(x, y)
        pearson_r = float(pearson.statistic)
        pearson_p = float(pearson.pvalue)
        spearman_rho = float(spearman.statistic)
        spearman_p = float(spearman.pvalue)
        slope, intercept = np.polyfit(x, y, 1)
    return {'mode': mode, 'poi_type': poi_type, 'axis_limit_min': axis_limit, 'n': n, 'pearson_r': pearson_r, 'pearson_p': pearson_p, 'spearman_rho': spearman_rho, 'spearman_p': spearman_p, 'r2': pearson_r ** 2 if np.isfinite(pearson_r) else np.nan, 'rmse_min': float(np.sqrt(np.mean(residual ** 2))) if n else np.nan, 'mae_min': float(np.mean(np.abs(residual))) if n else np.nan, 'bias_gee_minus_api_min': float(np.mean(residual)) if n else np.nan, 'median_abs_error_min': float(np.median(np.abs(residual))) if n else np.nan, 'median_api_min': float(np.median(x)) if n else np.nan, 'median_gee_min': float(np.median(y)) if n else np.nan, 'slope_gee_on_api': float(slope) if np.isfinite(slope) else np.nan, 'intercept_gee_on_api': float(intercept) if np.isfinite(intercept) else np.nan}

def build_metrics(source: pd.DataFrame, plot_source: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode in ['walking', 'driving']:
        mode_all = source[source['mode'].eq(mode)]
        mode_plot = plot_source[plot_source['mode'].eq(mode)]
        rows.append(metric_block(mode_all, mode, 'all_clean_joined', None))
        rows.append(metric_block(mode_plot, mode, 'all_plot_window', AXIS_LIMITS_MIN[mode]))
        for poi in POI_ORDER:
            rows.append(metric_block(mode_plot[mode_plot['poi_type'].eq(poi)], mode, poi, AXIS_LIMITS_MIN[mode]))
    return pd.DataFrame(rows)

def set_style() -> None:
    try:
        plt.style.use(['science', 'no-latex'])
    except OSError:
        pass
    mpl.rcParams.update({'font.family': 'Arial', 'font.size': 6.4, 'axes.titlesize': 7.2, 'axes.labelsize': 6.8, 'xtick.labelsize': 5.7, 'ytick.labelsize': 5.7, 'legend.fontsize': 5.8, 'axes.linewidth': 0.55, 'lines.linewidth': 0.75, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2.2, 'ytick.major.size': 2.2, 'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.top': False, 'ytick.right': False, 'xtick.minor.visible': False, 'ytick.minor.visible': False, 'axes.grid': False, 'savefig.dpi': 600, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

def blend_cmap(name: str, color: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, ['#FFFFFF', '#EEF2F6', color])

def clean_axes(ax: plt.Axes) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_linewidth(0.55)
    ax.tick_params(pad=1.5, top=False, right=False, which='both')

def p_text(p_value: float) -> str:
    if not np.isfinite(p_value):
        return 'P=NA'
    if p_value < 0.001:
        return 'P<0.001'
    if p_value < 0.01:
        return 'P<0.01'
    if p_value < 0.05:
        return 'P<0.05'
    return f'P={p_value:.2f}'

def draw_fit_and_identity(ax: plt.Axes, data: pd.DataFrame, limit: float, color: str='black') -> None:
    ax.plot([0, limit], [0, limit], color='#717A87', lw=0.65, ls=(0, (4, 3)), zorder=1)
    if len(data) >= 3 and data['api_time_min'].std() > 0 and (data['gee_time_min'].std() > 0):
        slope, intercept = np.polyfit(data['api_time_min'], data['gee_time_min'], 1)
        xs = np.array([0.0, limit])
        ax.plot(xs, intercept + slope * xs, color=color, lw=0.95, zorder=4)

def overall_density_panel(ax: plt.Axes, data: pd.DataFrame, mode: str) -> None:
    limit = AXIS_LIMITS_MIN[mode]
    color = MODE_COLOR[mode]
    view = data[data['mode'].eq(mode)].copy()
    cmap = blend_cmap(f'{mode}_density', color)
    hb = ax.hexbin(view['api_time_min'], view['gee_time_min'], gridsize=42, extent=(0, limit, 0, limit), mincnt=1, cmap=cmap, norm=LogNorm(), linewidths=0, zorder=2)
    sample = view.sample(n=min(len(view), 1600), random_state=19) if len(view) else view
    ax.scatter(sample['api_time_min'], sample['gee_time_min'], s=2.2, color='#111827', alpha=0.16, linewidth=0, rasterized=True, zorder=3)
    draw_fit_and_identity(ax, view, limit)
    metric = metric_block(view, mode, 'all_plot_window', limit)
    ax.text(0.045, 0.955, f"n={metric['n']:,}\nr={metric['pearson_r']:.2f}, rho={metric['spearman_rho']:.2f}\nMAE={metric['mae_min']:.1f} min\nbias={metric['bias_gee_minus_api_min']:.1f} min", transform=ax.transAxes, ha='left', va='top', fontsize=6.1, linespacing=1.12, bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='#D5DAE1', linewidth=0.45, alpha=0.9))
    ax.set_title(f'{MODE_LABEL[mode]}: all POI classes', loc='left', pad=2)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ticks = [0, 150, 300, 450, 600] if limit == 600 else [0, 50, 100, 150, 200]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Google Routes API (min)')
    ax.set_ylabel('GEE accessibility estimate (min)')
    clean_axes(ax)
    cax = inset_axes(ax, width='33%', height='4.8%', loc='lower right', borderpad=1.05)
    cb = plt.colorbar(hb, cax=cax, orientation='horizontal')
    cb.outline.set_linewidth(0.35)
    cb.ax.tick_params(labelsize=4.7, length=1.5, pad=0.6)
    cb.ax.set_title('pairs per bin', fontsize=4.9, pad=1.1)

def point_density_by_bin(x: np.ndarray, y: np.ndarray, limit: float, bins: int=40) -> np.ndarray:
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=((0, limit), (0, limit)))
    xi = np.clip(np.digitize(x, xedges) - 1, 0, bins - 1)
    yi = np.clip(np.digitize(y, yedges) - 1, 0, bins - 1)
    return hist[xi, yi]

def poi_scatter_panel(ax: plt.Axes, data: pd.DataFrame, mode: str, poi: str) -> None:
    limit = AXIS_LIMITS_MIN[mode]
    color = POI_COLOR[poi]
    view = data[data['mode'].eq(mode) & data['poi_type'].eq(poi)].copy()
    cmap = blend_cmap(f'{mode}_{poi}_points', color)
    if len(view):
        x = view['api_time_min'].to_numpy(float)
        y = view['gee_time_min'].to_numpy(float)
        dens = point_density_by_bin(x, y, limit)
        order = np.argsort(dens)
        x = x[order]
        y = y[order]
        dens = dens[order]
        ax.scatter(x, y, c=dens, s=6.2, cmap=cmap, vmin=1, vmax=max(2, float(np.nanpercentile(dens, 98))), alpha=0.9, linewidth=0, rasterized=True, zorder=2)
    draw_fit_and_identity(ax, view, limit)
    metric = metric_block(view, mode, poi, limit)
    ax.text(0.055, 0.91, f"r={metric['pearson_r']:.2f}\nrho={metric['spearman_rho']:.2f}\nn={metric['n']:,}", transform=ax.transAxes, ha='left', va='top', fontsize=4.95, linespacing=1.0)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([0, limit / 2, limit])
    ax.set_yticks([0, limit / 2, limit])
    clean_axes(ax)

def residual_density_panel(ax: plt.Axes, data: pd.DataFrame) -> None:
    values = []
    for mode in ['walking', 'driving']:
        residual = data.loc[data['mode'].eq(mode), 'residual_min'].dropna().to_numpy(float)
        if len(residual):
            values.append(residual)
    if not values:
        return
    combined = np.concatenate(values)
    xlim = float(np.nanpercentile(np.abs(combined), 98))
    xlim = max(20.0, min(xlim, 260.0))
    grid = np.linspace(-xlim, xlim, 360)
    for mode in ['walking', 'driving']:
        residual = data.loc[data['mode'].eq(mode), 'residual_min'].dropna().to_numpy(float)
        residual = residual[np.abs(residual) <= xlim]
        if len(residual) < 3 or np.nanstd(residual) == 0:
            continue
        kde = stats.gaussian_kde(residual)
        density = kde(grid)
        density = density / density.max()
        ax.fill_between(grid, 0, density, color=MODE_COLOR[mode], alpha=0.22, linewidth=0)
        ax.plot(grid, density, color=MODE_COLOR[mode], lw=0.95, label=MODE_LABEL[mode])
        ax.axvline(np.median(residual), color=MODE_COLOR[mode], lw=0.75, ls=(0, (2, 2)))
    ax.axvline(0, color='#111827', lw=0.7)
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel('Residual, GEE accessibility - API (min)')
    ax.set_ylabel('Scaled density')
    ax.legend(frameon=False, loc='upper left', handlelength=1.7)
    ax.set_title('Residual density', loc='left', pad=2)
    clean_axes(ax)

def metric_heatmap_panel(ax: plt.Axes, metrics: pd.DataFrame) -> None:
    rows = []
    for mode in ['walking', 'driving']:
        vals = []
        for poi in POI_ORDER:
            row = metrics[metrics['mode'].eq(mode) & metrics['poi_type'].eq(poi)]
            vals.append(float(row['pearson_r'].iloc[0]) if len(row) else np.nan)
        rows.append(vals)
    matrix = np.array(rows, dtype=float)
    im = ax.imshow(matrix, cmap='viridis', vmin=0.65, vmax=1.0, aspect='auto')
    ax.set_xticks(range(len(POI_ORDER)))
    ax.set_xticklabels([POI_LABEL[p] for p in POI_ORDER], rotation=35, ha='right', rotation_mode='anchor')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Walking', 'Driving'])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=5.8, color='white' if value < 0.83 else '#111827')
    ax.set_title('Pearson r by POI class', loc='left', pad=2)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.45)
    cax = inset_axes(ax, width='3.8%', height='84%', loc='center right', borderpad=-2.2)
    cb = plt.colorbar(im, cax=cax, orientation='vertical')
    cb.set_label('r', fontsize=5.8, labelpad=1.5)
    cb.ax.tick_params(labelsize=5.0, length=1.5, pad=1)
    cb.outline.set_linewidth(0.35)

def add_panel_label(ax: plt.Axes, label: str, x: float=-0.16, y: float=1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha='left', va='top', fontsize=8.2, fontweight='bold')

def map_source_by_origin(data: pd.DataFrame) -> pd.DataFrame:
    lon = data['origin_lon_gee'].copy() if 'origin_lon_gee' in data.columns else data['origin_lon_api'].copy()
    lat = data['origin_lat_gee'].copy() if 'origin_lat_gee' in data.columns else data['origin_lat_api'].copy()
    if 'origin_lon_api' in data.columns:
        lon = lon.fillna(data['origin_lon_api'])
    if 'origin_lat_api' in data.columns:
        lat = lat.fillna(data['origin_lat_api'])
    tmp = data.assign(map_lon=pd.to_numeric(lon, errors='coerce'), map_lat=pd.to_numeric(lat, errors='coerce'))
    tmp = tmp[np.isfinite(tmp['map_lon']) & np.isfinite(tmp['map_lat'])].copy()
    grouped = tmp.groupby('origin_id', as_index=False).agg(map_lon=('map_lon', 'first'), map_lat=('map_lat', 'first'), median_residual_min=('residual_min', 'median'), median_abs_error_min=('abs_error_min', 'median'), matched_pairs=('residual_min', 'size')).reset_index(drop=True)
    return grouped

def validation_map_panel(ax: plt.Axes, data: pd.DataFrame) -> None:
    map_df = map_source_by_origin(data)
    if map_df.empty:
        ax.set_axis_off()
        ax.text(0.5, 0.5, 'No valid coordinates', ha='center', va='center', transform=ax.transAxes)
        return
    residual = map_df['median_residual_min'].to_numpy(float)
    vlim = float(np.nanpercentile(np.abs(residual), 97))
    vlim = max(20.0, min(vlim, 120.0))
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
    transform = None
    if ccrs is not None and hasattr(ax, 'set_global'):
        transform = ccrs.PlateCarree()
        ax.set_global()
        ax.add_feature(cfeature.OCEAN, facecolor='#FAFBFC', edgecolor='none', zorder=0)
        ax.add_feature(cfeature.LAND, facecolor='#EEF1F4', edgecolor='none', zorder=0)
        ax.add_feature(cfeature.BORDERS, edgecolor='#C8CDD3', linewidth=0.18, zorder=1)
        ax.coastlines(color='#9EA6B1', linewidth=0.28, zorder=1)
        gl = ax.gridlines(draw_labels=False, linewidth=0.25, color='#D6DCE3', alpha=0.7, linestyle='-')
        gl.xlocator = mpl.ticker.FixedLocator(np.arange(-180, 181, 60))
        gl.ylocator = mpl.ticker.FixedLocator(np.arange(-60, 91, 30))
    else:
        ax.set_facecolor('#FAFBFC')
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 85)
        ax.set_xticks(np.arange(-180, 181, 60))
        ax.set_yticks(np.arange(-60, 91, 30))
        ax.grid(color='#D6DCE3', linewidth=0.35)
        clean_axes(ax)
    scatter_kwargs = {'x': map_df['map_lon'], 'y': map_df['map_lat'], 'c': map_df['median_residual_min'], 's': np.clip(map_df['matched_pairs'].to_numpy(float) * 1.3, 10, 34), 'cmap': 'RdBu_r', 'norm': norm, 'alpha': 0.84, 'linewidth': 0.16, 'edgecolor': 'white', 'rasterized': True, 'zorder': 4}
    if transform is not None:
        scatter_kwargs['transform'] = transform
    sc = ax.scatter(**scatter_kwargs)
    ax.set_title('Global validation origins', loc='left', pad=2)
    ax.text(0.012, 0.045, f'{len(map_df):,} origins; colour shows median residual across matched mode-POI pairs', transform=ax.transAxes, ha='left', va='bottom', fontsize=5.6, bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='#D5DAE1', linewidth=0.4, alpha=0.88))
    cax = inset_axes(ax, width='31%', height='7%', loc='lower right', borderpad=0.8)
    cb = plt.colorbar(sc, cax=cax, orientation='horizontal')
    cb.set_label('median residual (min)', fontsize=5.4, labelpad=1.0)
    cb.ax.tick_params(labelsize=4.8, length=1.4, pad=0.8)
    cb.outline.set_linewidth(0.35)

def make_figure(plot_source: pd.DataFrame, metrics: pd.DataFrame) -> None:
    set_style()
    fig = plt.figure(figsize=(7.2, 9.7), dpi=600)
    gs = fig.add_gridspec(5, 6, height_ratios=[1.05, 1.58, 1.0, 1.0, 1.05], left=0.072, right=0.982, bottom=0.057, top=0.935, wspace=0.38, hspace=0.4)
    if ccrs is not None:
        ax_a = fig.add_subplot(gs[0, :], projection=ccrs.Robinson())
    else:
        ax_a = fig.add_subplot(gs[0, :])
    validation_map_panel(ax_a, plot_source)
    add_panel_label(ax_a, 'a', x=-0.035, y=1.05)
    ax_b = fig.add_subplot(gs[1, 0:3])
    ax_c = fig.add_subplot(gs[1, 3:6])
    overall_density_panel(ax_b, plot_source, 'walking')
    overall_density_panel(ax_c, plot_source, 'driving')
    add_panel_label(ax_b, 'b', x=-0.17, y=1.08)
    add_panel_label(ax_c, 'c', x=-0.17, y=1.08)
    scatter_axes = []
    for row, mode in enumerate(['walking', 'driving'], start=2):
        for col, poi in enumerate(POI_ORDER):
            ax = fig.add_subplot(gs[row, col])
            poi_scatter_panel(ax, plot_source, mode, poi)
            scatter_axes.append(ax)
            if row == 1:
                ax.set_title(POI_LABEL[poi], loc='left', pad=2)
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel('API')
            if col == 0:
                ax.set_ylabel(f'{MODE_LABEL[mode]}\nGEE')
            else:
                ax.tick_params(labelleft=False)
    add_panel_label(scatter_axes[0], 'd', x=-0.42, y=1.18)
    ax_e = fig.add_subplot(gs[4, 0:3])
    ax_f = fig.add_subplot(gs[4, 3:6])
    residual_density_panel(ax_e, plot_source)
    metric_heatmap_panel(ax_f, metrics[metrics['poi_type'].isin(POI_ORDER)])
    add_panel_label(ax_e, 'e', x=-0.17, y=1.1)
    add_panel_label(ax_f, 'f', x=-0.17, y=1.1)
    fig.text(0.072, 0.982, 'Accessibility validation against Google Routes API', ha='left', va='top', fontsize=8.5, fontweight='bold')
    fig.text(0.072, 0.962, 'Map shows validation-point coverage and median residual; density panels use matched clean Google API routes.', ha='left', va='top', fontsize=5.9)
    fig.savefig(OUT_PNG, dpi=600)
    fig.savefig(OUT_TIFF, dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_SVG)
    plt.close(fig)

def build_outputs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audits: dict[str, object] = {'script': str(Path(__file__).resolve()), 'python_executable': sys.executable, 'python_version': platform.python_implementation() + ' ' + platform.python_version(), 'axis_limits_min': AXIS_LIMITS_MIN, 'api_reference_table': str(API_CSV), 'template_archetype': {'primary': 'Python correlation/fitted-line validation plot', 'secondary': 'Python joint density / marginal-density evidence'}}
    gee_parts = []
    audits['gee_mode_audits'] = {}
    for mode, folder in MODE_DIRS.items():
        mode_df, mode_audit = read_gee_mode(mode, folder)
        gee_parts.append(mode_df)
        audits['gee_mode_audits'][mode] = mode_audit
    gee_long = pd.concat(gee_parts, ignore_index=True)
    gee_long.to_csv(OUT_GEE_LONG, index=False, encoding='utf-8-sig')
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
    source.to_csv(OUT_SOURCE, index=False, encoding='utf-8-sig')
    plot_source.to_csv(OUT_SOURCE_PLOT, index=False, encoding='utf-8-sig')
    metrics.to_csv(OUT_METRICS, index=False, encoding='utf-8-sig')
    audits['joined_rows_clean'] = int(len(source))
    audits['joined_rows_plot_window'] = int(len(plot_source))
    audits['joined_counts_by_mode'] = {str(k): int(v) for k, v in source['mode'].value_counts().sort_index().items()}
    audits['plot_window_counts_by_mode'] = {str(k): int(v) for k, v in plot_source['mode'].value_counts().sort_index().items()}
    audits['plot_window_counts_by_mode_poi'] = {f'{mode}|{poi}': int(count) for (mode, poi), count in plot_source.groupby(['mode', 'poi_type']).size().items()}
    audits['outputs'] = {'gee_long': str(OUT_GEE_LONG), 'source_data': str(OUT_SOURCE), 'source_data_plot_window': str(OUT_SOURCE_PLOT), 'metrics': str(OUT_METRICS), 'figure_png': str(OUT_PNG), 'figure_tiff': str(OUT_TIFF), 'figure_pdf': str(OUT_PDF), 'figure_svg': str(OUT_SVG)}
    OUT_AUDIT.write_text(json.dumps(audits, ensure_ascii=False, indent=2), encoding='utf-8')
    make_figure(plot_source, metrics)
    print(f'GEE long: {OUT_GEE_LONG}')
    print(f'Source data: {OUT_SOURCE}')
    print(f'Plot-window source data: {OUT_SOURCE_PLOT}')
    print(f'Metrics: {OUT_METRICS}')
    print(f'Audit: {OUT_AUDIT}')
    print(f'Figure PNG: {OUT_PNG}')
    print(f'Figure TIFF: {OUT_TIFF}')
    print(f'Figure PDF: {OUT_PDF}')
    print(f'Figure SVG: {OUT_SVG}')
    print()
    print(metrics[['mode', 'poi_type', 'axis_limit_min', 'n', 'pearson_r', 'spearman_rho', 'rmse_min', 'mae_min', 'bias_gee_minus_api_min']].to_string(index=False))
