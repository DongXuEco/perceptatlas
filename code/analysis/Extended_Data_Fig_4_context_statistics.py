from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "_support"))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "figures"))

from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = PACKAGE_ROOT
DERIVED_DIR = ROOT / 'derived_data'
RUN_DIR = ROOT / 'runs' / '20260818-q3-mechanism-modeling'
OUT_DIR = output_dir('Extended_Data_Fig_4')
CITY_MASTER = DERIVED_DIR / 'q3_city_driver_master.csv'
ZONE_MASTER = DERIVED_DIR / 'q3_city_zone_driver_master.csv'
PREDICTORS = RUN_DIR / 'q3_final_predictors_recommended.csv'
OUT_LONG = OUT_DIR / 'q3_driver_context_contrast_long.csv'
OUT_SUMMARY = OUT_DIR / 'q3_driver_context_contrast_group_summary.csv'
OUT_EFFECTS = OUT_DIR / 'q3_driver_context_contrast_effects.csv'
OUT_AUDIT = OUT_DIR / 'q3_driver_context_contrast_audit.json'
N_BOOT = 2000
BOOT_SEED = 20260823
FAMILY_SHORT = {'green_blue_space': 'Green/blue', 'environment_risk': 'Env. risk', 'built_environment': 'Built env.', 'economic_population': 'Econ./pop.', 'public_service_poi': 'POI supply', 'public_service_accessibility_nonmotor': 'Non-motor', 'public_service_road': 'Road network', 'terrain_control': 'Terrain', 'location_control': 'Location'}
FAMILY_COLORS = {'green_blue_space': '#62C0D4', 'environment_risk': '#E5170A', 'built_environment': '#357DBD', 'economic_population': '#6E789D', 'public_service_poi': '#F28817', 'public_service_accessibility_nonmotor': '#F1B47A', 'public_service_road': '#F0CFA1', 'terrain_control': '#909BB6', 'location_control': '#8170A8'}
FAMILY_ORDER = {'green_blue_space': 1, 'environment_risk': 2, 'built_environment': 3, 'economic_population': 4, 'public_service_poi': 5, 'public_service_accessibility_nonmotor': 6, 'public_service_road': 7, 'terrain_control': 8, 'location_control': 9}
DRIVER_SHORT = {'green_fraction_mean': 'Green', 'water_fraction_mean': 'Water', 'pm25_2020_mean': 'PM2.5', 'lst_c_mean': 'LST', 'noise_mean': 'Noise', 'building_density_mean': 'Built dens.', 'building_height_mean': 'Height', 'ntl_log1p_mean': 'NTL', 'population_density_mean': 'Pop. dens.', 'poi_education_mean': 'POI edu.', 'poi_entertainment_mean': 'POI ent.', 'poi_health_mean': 'POI health', 'poi_transport_mean': 'POI transp.', 'access_nonmotor_education_mean': 'Access edu.', 'access_nonmotor_entertainment_mean': 'Access ent.', 'access_nonmotor_health_mean': 'Access health', 'access_nonmotor_living_mean': 'Access living', 'access_nonmotor_transport_mean': 'Access transp.', 'road_total_mean': 'Road total', 'road_footway_mean': 'Footway', 'road_pedestrian_mean': 'Pedestrian', 'road_cycleway_mean': 'Cycleway', 'dem_mean': 'Elev.', 'slope_mean': 'Slope', 'lat': 'Latitude'}

def _nan_iqr(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors='coerce')
    if values.notna().sum() == 0:
        return np.nan
    return float(values.quantile(0.75) - values.quantile(0.25))

def _bootstrap_ci_one_sample(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (np.nan, np.nan)
    if values.size == 1:
        return (float(values[0]), float(values[0]))
    boots = np.empty(N_BOOT, dtype=float)
    for i in range(N_BOOT):
        sample = values[rng.integers(0, values.size, values.size)]
        boots[i] = np.nanmedian(sample)
    return tuple(np.nanpercentile(boots, [2.5, 97.5]).astype(float))

def _bootstrap_ci_independent(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return (np.nan, np.nan)
    boots = np.empty(N_BOOT, dtype=float)
    for i in range(N_BOOT):
        xs = x[rng.integers(0, x.size, x.size)]
        ys = y[rng.integers(0, y.size, y.size)]
        boots[i] = np.nanmedian(xs) - np.nanmedian(ys)
    return tuple(np.nanpercentile(boots, [2.5, 97.5]).astype(float))

def _robust_standardize(frame: pd.DataFrame, predictors: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = frame.copy()
    params: dict[str, dict[str, float]] = {}
    for row in predictors.itertuples(index=False):
        col = row.column
        raw = pd.to_numeric(out[col], errors='coerce')
        nonmissing = raw.dropna()
        if nonmissing.empty:
            q01 = q99 = center = scale = np.nan
            winsor = raw
        else:
            q01 = float(nonmissing.quantile(0.01))
            q99 = float(nonmissing.quantile(0.99))
            winsor = raw.clip(q01, q99)
            center = float(winsor.median())
            mad = float((winsor - center).abs().median() * 1.4826)
            iqr_scale = _nan_iqr(winsor) / 1.349
            sd_scale = float(winsor.std(ddof=0))
            scale = mad if mad > 0 else iqr_scale
            if not np.isfinite(scale) or scale <= 0:
                scale = sd_scale if sd_scale > 0 else np.nan
        z_col = f'{col}__robust_z'
        winsor_col = f'{col}__winsor'
        out[winsor_col] = winsor
        out[z_col] = (winsor - center) / scale if np.isfinite(scale) and scale > 0 else np.nan
        params[col] = {'q01': q01, 'q99': q99, 'robust_center_median': center, 'robust_scale': scale, 'n_nonmissing': int(nonmissing.size), 'coverage_pct': float(nonmissing.size / len(raw) * 100.0)}
    return (out, params)

def _make_long(frame: pd.DataFrame, predictors: pd.DataFrame, scale: str, params: dict) -> pd.DataFrame:
    id_cols = ['id_1', 'COUNTRY', 'NAME_1', 'lon', 'lat', 'NS', 'ns_group']
    if scale == 'city_zone':
        id_cols = ['id_1', 'zone_code', 'zone_label', 'COUNTRY', 'NAME_1', 'lon', 'lat', 'NS', 'ns_group']
    base = frame[id_cols].copy()
    if scale == 'city':
        base['zone_code'] = np.nan
        base['zone_label'] = 'city'
    base['analysis_scale'] = scale
    pieces = []
    for row in predictors.itertuples(index=False):
        col = row.column
        piece = base.copy()
        piece['driver'] = col
        piece['driver_prefix'] = row.prefix
        piece['driver_label'] = row.variable
        piece['driver_short'] = DRIVER_SHORT.get(col, row.variable)
        piece['family'] = row.family
        piece['family_label'] = row.family_label
        piece['family_short'] = FAMILY_SHORT.get(row.family, row.family_label)
        piece['family_color'] = FAMILY_COLORS.get(row.family, '#C8D0C1')
        piece['family_order'] = FAMILY_ORDER.get(row.family, 99)
        piece['driver_order'] = int(row.selection_order)
        piece['unit'] = row.unit
        piece['expected_direction'] = row.expected_direction
        piece['value_raw'] = pd.to_numeric(frame[col], errors='coerce')
        piece['value_winsor'] = pd.to_numeric(frame[f'{col}__winsor'], errors='coerce')
        piece['value_z'] = pd.to_numeric(frame[f'{col}__robust_z'], errors='coerce')
        piece['is_observed'] = piece['value_raw'].notna()
        piece['is_winsorized'] = piece['value_raw'].notna() & piece['value_winsor'].notna() & (piece['value_raw'] != piece['value_winsor'])
        piece['scale_driver_n'] = params[col]['n_nonmissing']
        piece['scale_driver_coverage_pct'] = params[col]['coverage_pct']
        piece['robust_center_median'] = params[col]['robust_center_median']
        piece['robust_scale'] = params[col]['robust_scale']
        piece['winsor_q01'] = params[col]['q01']
        piece['winsor_q99'] = params[col]['q99']
        if scale == 'city_zone':
            piece['context_group'] = piece['ns_group'].astype(str) + '_' + piece['zone_label'].astype(str)
        else:
            piece['context_group'] = piece['ns_group'].astype(str) + '_city'
        pieces.append(piece)
    long = pd.concat(pieces, ignore_index=True)
    sort_cols = ['analysis_scale', 'id_1', 'zone_code', 'driver_order']
    return long.sort_values(sort_cols, kind='stable')

def _summarize_group(df: pd.DataFrame, grouping: str, group_col: str) -> pd.DataFrame:
    rows = []
    for (scale, driver, group), sub in df.groupby(['analysis_scale', 'driver', group_col], dropna=False):
        obs = sub[sub['is_observed']].copy()
        if group_col == 'zone_label' and group == 'city':
            continue
        meta = sub.iloc[0]
        rows.append({'analysis_scale': scale, 'grouping': grouping, 'group': group, 'driver': driver, 'driver_short': meta['driver_short'], 'driver_label': meta['driver_label'], 'driver_order': int(meta['driver_order']), 'family': meta['family'], 'family_short': meta['family_short'], 'family_color': meta['family_color'], 'family_order': int(meta['family_order']), 'unit': meta['unit'], 'n_total': int(len(sub)), 'n_nonmissing': int(len(obs)), 'coverage_pct': float(len(obs) / len(sub) * 100.0) if len(sub) else np.nan, 'raw_mean': float(obs['value_raw'].mean()) if len(obs) else np.nan, 'raw_sd': float(obs['value_raw'].std(ddof=1)) if len(obs) > 1 else np.nan, 'raw_median': float(obs['value_raw'].median()) if len(obs) else np.nan, 'raw_iqr': _nan_iqr(obs['value_raw']) if len(obs) else np.nan, 'z_mean': float(obs['value_z'].mean()) if len(obs) else np.nan, 'z_sd': float(obs['value_z'].std(ddof=1)) if len(obs) > 1 else np.nan, 'z_median': float(obs['value_z'].median()) if len(obs) else np.nan, 'z_iqr': _nan_iqr(obs['value_z']) if len(obs) else np.nan})
    return pd.DataFrame(rows)

def _build_summary(long: pd.DataFrame) -> pd.DataFrame:
    parts = [_summarize_group(long, 'north_south', 'ns_group'), _summarize_group(long[long['analysis_scale'].eq('city_zone')], 'core_expansion', 'zone_label'), _summarize_group(long[long['analysis_scale'].eq('city_zone')], 'north_south_x_core_expansion', 'context_group')]
    summary = pd.concat(parts, ignore_index=True)
    return summary.sort_values(['grouping', 'analysis_scale', 'driver_order', 'group'], kind='stable')

def _effect_row(contrast: str, driver_meta: pd.Series, effect_raw: float, effect_raw_ci: tuple[float, float], effect_z: float, effect_z_ci: tuple[float, float], n_a: int, n_b: int, n_pairs: int | None=None) -> dict:
    return {'contrast': contrast, 'driver': driver_meta['driver'], 'driver_short': driver_meta['driver_short'], 'driver_label': driver_meta['driver_label'], 'driver_order': int(driver_meta['driver_order']), 'family': driver_meta['family'], 'family_short': driver_meta['family_short'], 'family_color': driver_meta['family_color'], 'family_order': int(driver_meta['family_order']), 'unit': driver_meta['unit'], 'effect_raw_median': effect_raw, 'effect_raw_ci_low': effect_raw_ci[0], 'effect_raw_ci_high': effect_raw_ci[1], 'effect_z_median': effect_z, 'effect_z_ci_low': effect_z_ci[0], 'effect_z_ci_high': effect_z_ci[1], 'n_a': n_a, 'n_b': n_b, 'n_pairs': np.nan if n_pairs is None else n_pairs, 'ci_excludes_zero_z': bool(np.isfinite(effect_z_ci[0]) and np.isfinite(effect_z_ci[1]) and (effect_z_ci[0] > 0 and effect_z_ci[1] > 0 or (effect_z_ci[0] < 0 and effect_z_ci[1] < 0)))}

def _build_effects(long: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    city = long[long['analysis_scale'].eq('city')]
    zone = long[long['analysis_scale'].eq('city_zone')]
    for driver, sub in city.groupby('driver', sort=False):
        meta = sub.iloc[0]
        north = sub[sub['ns_group'].eq('North')]
        south = sub[sub['ns_group'].eq('South')]
        north_raw = north['value_raw'].to_numpy(float)
        south_raw = south['value_raw'].to_numpy(float)
        north_z = north['value_z'].to_numpy(float)
        south_z = south['value_z'].to_numpy(float)
        rows.append(_effect_row('North_minus_South_city', meta, float(np.nanmedian(north_raw) - np.nanmedian(south_raw)), _bootstrap_ci_independent(north_raw, south_raw, rng), float(np.nanmedian(north_z) - np.nanmedian(south_z)), _bootstrap_ci_independent(north_z, south_z, rng), int(np.isfinite(north_raw).sum()), int(np.isfinite(south_raw).sum())))
    for driver, sub in zone.groupby('driver', sort=False):
        meta = sub.iloc[0]
        raw_pivot = sub.pivot(index='id_1', columns='zone_label', values='value_raw')
        z_pivot = sub.pivot(index='id_1', columns='zone_label', values='value_z')
        ns_lookup = sub[['id_1', 'ns_group']].drop_duplicates('id_1').set_index('id_1')['ns_group']
        raw_delta = (raw_pivot['core'] - raw_pivot['expansion']).replace([np.inf, -np.inf], np.nan)
        z_delta = (z_pivot['core'] - z_pivot['expansion']).replace([np.inf, -np.inf], np.nan)
        rows.append(_effect_row('Core_minus_Expansion_zone_paired', meta, float(np.nanmedian(raw_delta)), _bootstrap_ci_one_sample(raw_delta.to_numpy(float), rng), float(np.nanmedian(z_delta)), _bootstrap_ci_one_sample(z_delta.to_numpy(float), rng), int(raw_pivot['core'].notna().sum()), int(raw_pivot['expansion'].notna().sum()), int(raw_delta.notna().sum())))
        north_ids = ns_lookup[ns_lookup.eq('North')].index
        south_ids = ns_lookup[ns_lookup.eq('South')].index
        north_raw_delta = raw_delta.loc[raw_delta.index.intersection(north_ids)].to_numpy(float)
        south_raw_delta = raw_delta.loc[raw_delta.index.intersection(south_ids)].to_numpy(float)
        north_z_delta = z_delta.loc[z_delta.index.intersection(north_ids)].to_numpy(float)
        south_z_delta = z_delta.loc[z_delta.index.intersection(south_ids)].to_numpy(float)
        rows.append(_effect_row('NS_modification_of_Core_minus_Expansion', meta, float(np.nanmedian(north_raw_delta) - np.nanmedian(south_raw_delta)), _bootstrap_ci_independent(north_raw_delta, south_raw_delta, rng), float(np.nanmedian(north_z_delta) - np.nanmedian(south_z_delta)), _bootstrap_ci_independent(north_z_delta, south_z_delta, rng), int(np.isfinite(north_raw_delta).sum()), int(np.isfinite(south_raw_delta).sum()), int(np.isfinite(north_raw_delta).sum() + np.isfinite(south_raw_delta).sum())))
    effects = pd.DataFrame(rows)
    return effects.sort_values(['contrast', 'driver_order'], kind='stable')

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    predictors = pd.read_csv(resource(PREDICTORS))
    predictors = predictors[predictors['decision'].eq('final')].copy()
    predictors['selection_order'] = predictors['selection_order'].astype(int)
    predictors = predictors.sort_values('selection_order', kind='stable')
    usecols_city = ['id_1', 'COUNTRY', 'NAME_1', 'lon', 'lat', 'NS', 'ns_group'] + predictors['column'].tolist()
    usecols_zone = ['id_1', 'zone_code', 'zone_label', 'COUNTRY', 'NAME_1', 'lon', 'lat', 'NS', 'ns_group'] + predictors['column'].tolist()
    city = pd.read_csv(resource(CITY_MASTER), usecols=usecols_city)
    zone = pd.read_csv(resource(ZONE_MASTER), usecols=usecols_zone)
    missing_city = sorted(set(usecols_city) - set(city.columns))
    missing_zone = sorted(set(usecols_zone) - set(zone.columns))
    if missing_city or missing_zone:
        raise ValueError({'missing_city_columns': missing_city, 'missing_zone_columns': missing_zone})
    city_std, city_params = _robust_standardize(city, predictors)
    zone_std, zone_params = _robust_standardize(zone, predictors)
    long = pd.concat([_make_long(city_std, predictors, 'city', city_params), _make_long(zone_std, predictors, 'city_zone', zone_params)], ignore_index=True)
    summary = _build_summary(long)
    effects = _build_effects(long)
    long.to_csv(OUT_LONG, index=False, encoding='utf-8-sig')
    summary.to_csv(OUT_SUMMARY, index=False, encoding='utf-8-sig')
    effects.to_csv(OUT_EFFECTS, index=False, encoding='utf-8-sig')
    audit = {'inputs': {'city_master': str(CITY_MASTER), 'zone_master': str(ZONE_MASTER), 'predictors': str(PREDICTORS)}, 'outputs': {'long': str(OUT_LONG), 'summary': str(OUT_SUMMARY), 'effects': str(OUT_EFFECTS)}, 'n_city_rows': int(len(city)), 'n_zone_rows': int(len(zone)), 'n_predictors': int(len(predictors)), 'n_long_rows': int(len(long)), 'n_summary_rows': int(len(summary)), 'n_effect_rows': int(len(effects)), 'predictors': predictors[['selection_order', 'column', 'family', 'family_label', 'variable', 'unit', 'city_coverage_pct', 'zone_coverage_pct']].to_dict(orient='records'), 'standardization': {'method': 'per analysis scale and driver: 1st/99th percentile winsorization, median centering, MAD*1.4826 scaling; IQR/1.349 or SD fallback when MAD is zero', 'city_params': city_params, 'zone_params': zone_params}, 'bootstrap': {'n_boot': N_BOOT, 'seed': BOOT_SEED, 'ci': 'percentile 95% CI'}}
    OUT_AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: audit[k] for k in ['n_city_rows', 'n_zone_rows', 'n_predictors', 'n_long_rows', 'n_summary_rows', 'n_effect_rows']}, indent=2))
    print(f'Wrote: {OUT_LONG}')
    print(f'Wrote: {OUT_SUMMARY}')
    print(f'Wrote: {OUT_EFFECTS}')
    print(f'Wrote: {OUT_AUDIT}')
