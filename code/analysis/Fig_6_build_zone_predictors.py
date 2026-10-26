from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "_support"))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "figures"))

from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from Fig_2_build_city_analysis import DERIVED_DIR, DRIVER_SPECS, EMOTIONS, NEGATIVE, POSITIVE, PROJECT_ROOT, RasterSpec, find_project_file, load_city_table, write_inventory
DEFAULT_OUTPUT = DERIVED_DIR / 'q3_city_zone_driver_master.csv'
DEFAULT_INVENTORY = DERIVED_DIR / 'q3_city_zone_driver_variable_inventory.csv'
DEFAULT_COVERAGE = DERIVED_DIR / 'q3_city_zone_driver_coverage_report.csv'
DEFAULT_METADATA = DERIVED_DIR / 'q3_city_zone_driver_master_metadata.json'
ZONE_LABELS = {1: 'core', 2: 'expansion'}
RISK_SPECS = [RasterSpec('pm25_2020', 'environment_risk', 'GHAP PM2.5 annual concentration, 2020', '影响变量数据\\环境风险\\PM2.5\\processed\\GHAP_PM25_2020_1000m_global_urbanmask.tif', unit='micrograms per cubic metre', expected_direction='higher means greater air-pollution exposure'), RasterSpec('lst_c', 'environment_risk', 'MODIS land-surface temperature converted to Celsius', '影响变量数据\\环境风险\\热暴露\\全球降尺度LST\\processed\\MODIS_LST_C_global_urbanmask.tif', unit='degree Celsius', expected_direction='higher means stronger heat exposure'), RasterSpec('noise', 'environment_risk', 'Modelled environmental noise', '影响变量数据\\环境风险\\噪声\\RF_noise_stacking1.tif', unit='source raster unit', expected_direction='higher means greater noise exposure')]
CE_EMOTION_FILES = {'beauty': 'RF_stacking1_beauty_tif_reorganized.csv', 'wealthy': 'RF_wealthy_stacking1_tif_reorganized.csv', 'safer': 'RF_safer_stacking1_tif_reorganized.csv', 'livelier': 'RF_livelier_stacking1_tif_reorganized.csv', 'boring': 'RF_boring_stacking1_tif_reorganized.csv', 'depression': 'RF_depressing_stacking1_tif_reorganized.csv'}

def load_zone_emotion_means() -> pd.DataFrame:
    ce_dir = PROJECT_ROOT / '全球城市CE统计'
    pieces = []
    diffs = []
    for emotion, filename in CE_EMOTION_FILES.items():
        path = ce_dir / filename
        if not resource(path).exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(resource(path))
        required = {'id_1', 'mean_1', 'mean_2', 'diff'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f'{path} missing columns: {sorted(missing)}')
        df['id_1'] = pd.to_numeric(df['id_1'], errors='coerce').astype('Int64')
        for col in ['mean_1', 'mean_2', 'diff']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['id_1']).copy()
        df['id_1'] = df['id_1'].astype('int32')
        long = pd.concat([df[['id_1', 'mean_1']].rename(columns={'mean_1': f'mean50_{emotion}'}).assign(zone_code=1), df[['id_1', 'mean_2']].rename(columns={'mean_2': f'mean50_{emotion}'}).assign(zone_code=2)], ignore_index=True)
        pieces.append(long)
        diffs.append(df[['id_1', 'diff']].rename(columns={'diff': f'core_minus_expansion_mean50_{emotion}'}))
    out = pieces[0]
    for piece in pieces[1:]:
        out = out.merge(piece, on=['id_1', 'zone_code'], how='outer')
    for diff in diffs:
        out = out.merge(diff, on='id_1', how='left')
    out['zone_label'] = out['zone_code'].map(ZONE_LABELS)
    out['positive_score50'] = out[[f'mean50_{e}' for e in POSITIVE]].mean(axis=1)
    out['negative_score50'] = out[[f'mean50_{e}' for e in NEGATIVE]].mean(axis=1)
    out['eli50'] = out['positive_score50'] - out['negative_score50']
    out['zone_emotion_mean_count'] = out[[f'mean50_{e}' for e in EMOTIONS]].notna().sum(axis=1)
    return out

def collect_zone_values(mask_path: Path, raster_path: Path, band: int) -> tuple[np.ndarray, np.ndarray, dict]:
    key_chunks: list[np.ndarray] = []
    value_chunks: list[np.ndarray] = []
    valid_blocks = 0
    valid_pixels = 0
    with rasterio.open(resource(mask_path)) as mask_ds, rasterio.open(resource(raster_path)) as src:
        if band < 1 or band > src.count:
            raise ValueError(f'{raster_path} has {src.count} band(s); requested band {band}')
        with WarpedVRT(src, crs=mask_ds.crs, transform=mask_ds.transform, width=mask_ds.width, height=mask_ds.height, nodata=src.nodata, resampling=Resampling.average) as vrt:
            for _, window in mask_ds.block_windows(1):
                raw_zone = mask_ds.read(1, window=window)
                zone_valid = raw_zone > 0
                if not zone_valid.any():
                    continue
                vals = vrt.read(band, window=window, masked=True)
                sampled_values = vals[zone_valid]
                sampled_keys = raw_zone[zone_valid]
                if np.ma.isMaskedArray(sampled_values):
                    keep = ~sampled_values.mask
                    sampled_keys = sampled_keys[keep]
                    sampled_values = sampled_values.data[keep]
                sampled_values = np.asarray(sampled_values, dtype='float32')
                good_value = np.isfinite(sampled_values) & (sampled_values > -9990) & (sampled_values < 1e+20)
                if not good_value.any():
                    continue
                keys = sampled_keys[good_value].astype('int32', copy=False)
                city_ids = keys // 10
                zone_codes = keys % 10
                good_zone = (city_ids >= 1) & (city_ids <= 5905) & np.isin(zone_codes, [1, 2])
                if not good_zone.any():
                    continue
                key_chunks.append(keys[good_zone])
                value_chunks.append(sampled_values[good_value][good_zone])
                valid_blocks += 1
                valid_pixels += int(good_zone.sum())
    if not key_chunks:
        return (np.array([], dtype='int32'), np.array([], dtype='float32'), {'valid_blocks': 0, 'valid_pixels': 0})
    return (np.concatenate(key_chunks), np.concatenate(value_chunks), {'valid_blocks': valid_blocks, 'valid_pixels': valid_pixels})

def summarize_by_zone(zone_keys: np.ndarray, values: np.ndarray, prefix: str) -> pd.DataFrame:
    columns = ['id_1', 'zone_code', f'{prefix}_mean', f'{prefix}_sd', f'{prefix}_p10', f'{prefix}_p50', f'{prefix}_p90', f'{prefix}_p90_p10', f'{prefix}_min', f'{prefix}_max', f'{prefix}_n']
    if zone_keys.size == 0:
        return pd.DataFrame(columns=columns)
    order = np.argsort(zone_keys, kind='mergesort')
    sorted_keys = zone_keys[order]
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_keys)) + 1]
    ends = np.r_[starts[1:], sorted_keys.size]
    rows = []
    for start, end in zip(starts, ends):
        key = int(sorted_keys[start])
        city_id = key // 10
        zone_code = key % 10
        x = sorted_values[start:end].astype('float64', copy=False)
        if x.size == 0:
            continue
        p10, p50, p90 = np.nanpercentile(x, [10, 50, 90])
        rows.append({'id_1': city_id, 'zone_code': zone_code, f'{prefix}_mean': float(np.nanmean(x)), f'{prefix}_sd': float(np.nanstd(x, ddof=0)), f'{prefix}_p10': float(p10), f'{prefix}_p50': float(p50), f'{prefix}_p90': float(p90), f'{prefix}_p90_p10': float(p90 - p10), f'{prefix}_min': float(np.nanmin(x)), f'{prefix}_max': float(np.nanmax(x)), f'{prefix}_n': int(x.size)})
    return pd.DataFrame(rows, columns=columns)

def build_zone_base() -> pd.DataFrame:
    city = load_city_table()
    city_keep = ['id_1', 'COUNTRY', 'NAME_1', 'lon', 'lat', 'NS', 'ns_group']
    city_meta = city[city_keep].drop_duplicates('id_1')
    zones = pd.MultiIndex.from_product([city_meta['id_1'].to_numpy(dtype='int32'), [1, 2]], names=['id_1', 'zone_code']).to_frame(index=False)
    zones['zone_label'] = zones['zone_code'].map(ZONE_LABELS)
    return zones.merge(city_meta, on='id_1', how='left')

def build_master(output: Path, inventory_output: Path, coverage_output: Path, metadata_output: Path, max_variables: int | None=None) -> pd.DataFrame:
    mask_path = find_project_file('Global_City_C_E.tif')
    specs = [*DRIVER_SPECS, *RISK_SPECS]
    if max_variables:
        specs = specs[:max_variables]
    inventory = write_inventory(specs, inventory_output)
    missing = inventory.loc[~inventory['exists'], ['prefix', 'path']]
    if not missing.empty:
        raise FileNotFoundError('Missing driver rasters:\n' + missing.to_string(index=False))
    master = build_zone_base().merge(load_zone_emotion_means(), on=['id_1', 'zone_code', 'zone_label'], how='left')
    metadata: dict[str, object] = {'created_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'zone_mask': str(mask_path), 'zone_id_rule': 'Global_City_C_E value = city_id * 10 + zone_code; zone_code 1=core, 2=expansion', 'driver_resampling_to_zone_grid': 'average', 'output': str(output), 'emotion_source': str(PROJECT_ROOT / '全球城市CE统计'), 'zone_emotion_note': 'CE emotion files provide zone-level means only; zone-level gini/theil are not available in this table.', 'variables': {}}
    coverage_rows = []
    for i, spec in enumerate(specs, start=1):
        t0 = time.time()
        zone_keys, values, sample_meta = collect_zone_values(mask_path, spec.path, spec.band)
        summary = summarize_by_zone(zone_keys, values, spec.prefix)
        master = master.merge(summary, on=['id_1', 'zone_code'], how='left')
        elapsed = time.time() - t0
        zones_with_data = int(summary[['id_1', 'zone_code']].drop_duplicates().shape[0]) if not summary.empty else 0
        metadata['variables'][spec.prefix] = {'family': spec.family, 'variable': spec.variable, 'path': str(spec.path), 'band': spec.band, 'unit': spec.unit, 'expected_direction': spec.expected_direction, 'sampled_values': int(values.size), 'zones_with_data': zones_with_data, 'elapsed_seconds': round(elapsed, 2), **sample_meta}
        coverage_rows.append({'prefix': spec.prefix, 'family': spec.family, 'zones_with_data': zones_with_data, 'sampled_values': int(values.size), 'path': str(spec.path)})
        print(f'[{i:02d}/{len(specs):02d}] {spec.prefix}: values={values.size:,} zones={zones_with_data} elapsed={elapsed:.1f}s', flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(output, index=False, encoding='utf-8-sig')
    pd.DataFrame(coverage_rows).to_csv(coverage_output, index=False, encoding='utf-8-sig')
    metadata_output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved master: {output}')
    print(f'Saved inventory: {inventory_output}')
    print(f'Saved coverage: {coverage_output}')
    print(f'Saved metadata: {metadata_output}')
    return master

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build Q3 city-zone-level driver master table.')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--inventory-output', type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument('--coverage-output', type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument('--metadata-output', type=Path, default=DEFAULT_METADATA)
    parser.add_argument('--max-variables', type=int, default=None, help='Debug option: process only the first N variables.')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    build_master(args.output, args.inventory_output, args.coverage_output, args.metadata_output, args.max_variables)
