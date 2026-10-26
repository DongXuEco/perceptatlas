from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "_support"))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "figures"))

from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
PROJECT_ROOT = PACKAGE_ROOT
EMOTION_ROOT = Path('external_inputs/感知')
OUT_DIR = output_dir('Fig_4')
CITY_ZONE_RASTER = PROJECT_ROOT / '全球城市数据' / 'Global_City_C_E.tif'
FOUR_CLASS_RASTER = PROJECT_ROOT / '全球城市CE统计' / '综合分区.tif'
CITY_MASTER = PROJECT_ROOT / 'derived_data' / 'q3_city_driver_master.csv'
EMOTION_RASTERS = {'beauty': EMOTION_ROOT / 'Beauty_over' / 'stacking1' / 'RF_stacking1_beauty.tif', 'wealthy': EMOTION_ROOT / 'wealthy_over' / 'RF_wealthy_stacking1.tif', 'safer': EMOTION_ROOT / 'safer_over' / 'RF_safer_stacking1.tif', 'livelier': EMOTION_ROOT / 'livelier_over' / 'RF_livelier_stacking1.tif', 'boring': EMOTION_ROOT / 'boring_over' / 'RF_boring_stacking1.tif', 'depression': EMOTION_ROOT / 'depressing' / 'RF_depressing_stacking1.tif'}
POSITIVE = ['beauty', 'wealthy', 'safer', 'livelier']
NEGATIVE = ['boring', 'depression']
EMOTIONS = POSITIVE + NEGATIVE
CITY_ZONE_LABELS = {1: 'core', 2: 'expansion'}
FOUR_CLASS_LABELS = {21: ('South', 'expansion'), 22: ('South', 'core'), 61: ('North', 'expansion'), 62: ('North', 'core')}
CE_COMPATIBLE_NAMES = {'beauty': 'RF_stacking1_beauty_tif_reorganized_recomputed.csv', 'wealthy': 'RF_wealthy_stacking1_tif_reorganized_recomputed.csv', 'safer': 'RF_safer_stacking1_tif_reorganized_recomputed.csv', 'livelier': 'RF_livelier_stacking1_tif_reorganized_recomputed.csv', 'boring': 'RF_boring_stacking1_tif_reorganized_recomputed.csv', 'depression': 'RF_depressing_stacking1_tif_reorganized_recomputed.csv'}

@dataclass
class ZonalResult:
    sums: dict[str, np.ndarray]
    counts: dict[str, np.ndarray]
    valid_blocks: int
    scanned_blocks: int
    valid_zone_pixels: int
    elapsed_seconds: float
    zone_meta: dict[str, object]

def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not resource(path).exists()]
    if missing:
        raise FileNotFoundError('Missing required input files:\n' + '\n'.join(missing))

def city_meta() -> pd.DataFrame:
    cols = ['id_1', 'COUNTRY', 'NAME_1', 'lon', 'lat', 'NS', 'ns_group']
    df = pd.read_csv(resource(CITY_MASTER), usecols=cols)
    df['id_1'] = pd.to_numeric(df['id_1'], errors='coerce').astype('Int64')
    return df.dropna(subset=['id_1']).assign(id_1=lambda d: d['id_1'].astype(int))

def make_vrt(src: rasterio.DatasetReader, zone: rasterio.DatasetReader) -> WarpedVRT:
    return WarpedVRT(src, crs=zone.crs, transform=zone.transform, width=zone.width, height=zone.height, resampling=Resampling.average, src_nodata=src.nodata, nodata=np.nan)

def ensure_length(arrays: dict[str, np.ndarray], length: int) -> dict[str, np.ndarray]:
    out = {}
    for key, arr in arrays.items():
        if len(arr) >= length:
            out[key] = arr
            continue
        grown = np.zeros(length, dtype=arr.dtype)
        grown[:len(arr)] = arr
        out[key] = grown
    return out

def valid_zone_mask(values: np.ndarray, nodata: float | int | None, allowed_values: set[int] | None) -> np.ndarray:
    mask = np.isfinite(values)
    if nodata is not None and np.isfinite(float(nodata)):
        mask &= values != nodata
    if allowed_values is None:
        mask &= values > 0
    else:
        mask &= np.isin(values, list(allowed_values))
    return mask

def zonal_means(zone_raster: Path, allowed_values: set[int] | None, initial_length: int, progress_label: str, max_blocks: int | None=None) -> ZonalResult:
    t0 = time.time()
    sums = {emotion: np.zeros(initial_length, dtype=np.float64) for emotion in EMOTIONS}
    counts = {emotion: np.zeros(initial_length, dtype=np.int64) for emotion in EMOTIONS}
    valid_blocks = 0
    scanned_blocks = 0
    valid_zone_pixels = 0
    with rasterio.Env(GDAL_NUM_THREADS='ALL_CPUS', GDAL_CACHEMAX=1024):
        with rasterio.open(resource(zone_raster)) as zone:
            emotion_sources = {emotion: rasterio.open(resource(path)) for emotion, path in EMOTION_RASTERS.items()}
            try:
                vrts = {emotion: make_vrt(src, zone) for emotion, src in emotion_sources.items()}
                try:
                    zone_meta = {'path': str(zone_raster), 'crs': str(zone.crs), 'width': zone.width, 'height': zone.height, 'transform': tuple(zone.transform), 'nodata': zone.nodata, 'dtype': zone.dtypes[0], 'block_shape': zone.block_shapes[0]}
                    for _, window in zone.block_windows(1):
                        scanned_blocks += 1
                        zone_arr = zone.read(1, window=window)
                        zmask = valid_zone_mask(zone_arr, zone.nodata, allowed_values)
                        if not zmask.any():
                            if max_blocks is not None and scanned_blocks >= max_blocks:
                                break
                            continue
                        valid_blocks += 1
                        valid_zone_pixels += int(zmask.sum())
                        zone_ids = zone_arr.astype(np.int64, copy=False)
                        max_id = int(zone_ids[zmask].max())
                        if max_id + 1 > len(next(iter(sums.values()))):
                            sums = ensure_length(sums, max_id + 1)
                            counts = ensure_length(counts, max_id + 1)
                        for emotion, vrt in vrts.items():
                            data = vrt.read(1, window=window, masked=True)
                            data_values = np.asarray(data.filled(np.nan), dtype=np.float64)
                            data_mask = ~np.ma.getmaskarray(data)
                            mask = zmask & data_mask & np.isfinite(data_values)
                            if not mask.any():
                                continue
                            values = data_values[mask]
                            ids = zone_ids[mask]
                            local_sum = np.bincount(ids, weights=values, minlength=len(sums[emotion]))
                            local_count = np.bincount(ids, minlength=len(counts[emotion]))
                            sums[emotion][:len(local_sum)] += local_sum
                            counts[emotion][:len(local_count)] += local_count
                        if valid_blocks % 250 == 0:
                            print(f'[{progress_label}] valid_blocks={valid_blocks:,} scanned_blocks={scanned_blocks:,} valid_zone_pixels={valid_zone_pixels:,}', flush=True)
                        if max_blocks is not None and scanned_blocks >= max_blocks:
                            break
                finally:
                    for vrt in vrts.values():
                        vrt.close()
            finally:
                for src in emotion_sources.values():
                    src.close()
    return ZonalResult(sums=sums, counts=counts, valid_blocks=valid_blocks, scanned_blocks=scanned_blocks, valid_zone_pixels=valid_zone_pixels, elapsed_seconds=time.time() - t0, zone_meta=zone_meta)

def add_scores(df: pd.DataFrame, prefix: str='mean50_') -> pd.DataFrame:
    out = df.copy()
    out['positive_score50'] = out[[f'{prefix}{emotion}' for emotion in POSITIVE]].mean(axis=1)
    out['negative_score50'] = out[[f'{prefix}{emotion}' for emotion in NEGATIVE]].mean(axis=1)
    out['eli50'] = out['positive_score50'] - out['negative_score50']
    return out

def build_four_class_tables(result: ZonalResult) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    long_rows = []
    for zone_value, (ns_group, zone_label) in FOUR_CLASS_LABELS.items():
        row = {'zone_value': zone_value, 'ns_group': ns_group, 'zone_label': zone_label}
        for emotion in EMOTIONS:
            count = int(result.counts[emotion][zone_value]) if zone_value < len(result.counts[emotion]) else 0
            total = float(result.sums[emotion][zone_value]) if zone_value < len(result.sums[emotion]) else 0.0
            mean = total / count if count else math.nan
            row[f'mean50_{emotion}'] = mean
            row[f'count_{emotion}'] = count
            long_rows.append({'zone_value': zone_value, 'ns_group': ns_group, 'zone_label': zone_label, 'emotion': emotion, 'mean50': mean, 'count': count})
        rows.append(row)
    wide = add_scores(pd.DataFrame(rows))
    long = pd.DataFrame(long_rows)
    return (wide, long)

def build_city_zone_table(result: ZonalResult, meta: pd.DataFrame) -> pd.DataFrame:
    any_count = np.zeros(max((len(arr) for arr in result.counts.values())), dtype=bool)
    for arr in result.counts.values():
        any_count[:len(arr)] |= arr > 0
    zone_values = np.flatnonzero(any_count)
    rows = []
    for zone_value in zone_values:
        city_id = int(zone_value // 10)
        zone_code = int(zone_value % 10)
        if zone_code not in CITY_ZONE_LABELS:
            continue
        row = {'id_1': city_id, 'zone_value': int(zone_value), 'zone_code': zone_code, 'zone_label': CITY_ZONE_LABELS[zone_code]}
        counts_for_min = []
        for emotion in EMOTIONS:
            count = int(result.counts[emotion][zone_value])
            total = float(result.sums[emotion][zone_value])
            row[f'mean50_{emotion}'] = total / count if count else math.nan
            row[f'count_{emotion}'] = count
            counts_for_min.append(count)
        row['zone_emotion_mean_count'] = min(counts_for_min)
        rows.append(row)
    df = add_scores(pd.DataFrame(rows))
    df = meta.merge(df, on='id_1', how='left')
    ordered = ['id_1', 'zone_value', 'zone_code', 'zone_label', 'COUNTRY', 'NAME_1', 'lon', 'lat', 'NS', 'ns_group']
    rest = [col for col in df.columns if col not in ordered]
    df = df[ordered + rest].sort_values(['id_1', 'zone_code'], kind='stable')
    for emotion in EMOTIONS:
        pivot = df.pivot_table(index='id_1', columns='zone_label', values=f'mean50_{emotion}', aggfunc='first')
        if 'core' in pivot.columns and 'expansion' in pivot.columns:
            gap = pivot['core'] - pivot['expansion']
        else:
            gap = pd.Series(dtype=float)
        df[f'core_minus_expansion_mean50_{emotion}'] = df['id_1'].map(gap)
    return df

def build_city_wide(zone_df: pd.DataFrame) -> pd.DataFrame:
    meta_cols = ['id_1', 'COUNTRY', 'NAME_1', 'lon', 'lat', 'NS', 'ns_group']
    meta = zone_df[meta_cols].drop_duplicates('id_1')
    value_cols = [f'mean50_{emotion}' for emotion in EMOTIONS] + ['positive_score50', 'negative_score50', 'eli50']
    count_cols = [f'count_{emotion}' for emotion in EMOTIONS] + ['zone_emotion_mean_count']
    pieces = [meta.set_index('id_1')]
    for zone in ['core', 'expansion']:
        sub = zone_df[zone_df['zone_label'].eq(zone)].set_index('id_1')
        rename = {col: f'{zone}_{col}' for col in value_cols + count_cols if col in sub.columns}
        pieces.append(sub[list(rename.keys())].rename(columns=rename))
    wide = pd.concat(pieces, axis=1).reset_index()
    for emotion in EMOTIONS:
        wide[f'diff_mean50_{emotion}'] = wide[f'core_mean50_{emotion}'] - wide[f'expansion_mean50_{emotion}']
    wide['diff_positive_score50'] = wide['core_positive_score50'] - wide['expansion_positive_score50']
    wide['diff_negative_score50'] = wide['core_negative_score50'] - wide['expansion_negative_score50']
    wide['diff_eli50'] = wide['core_eli50'] - wide['expansion_eli50']
    return wide

def write_ce_compatible_files(wide: pd.DataFrame, out_dir: Path) -> None:
    ce_dir = out_dir / 'ce_reorganized_compatible'
    ce_dir.mkdir(parents=True, exist_ok=True)
    base_cols = ['id_1', 'COUNTRY', 'NAME_1', 'lon', 'lat', 'NS', 'ns_group']
    for emotion in EMOTIONS:
        out = wide[base_cols].copy()
        out['mean_1'] = wide[f'core_mean50_{emotion}']
        out['mean_2'] = wide[f'expansion_mean50_{emotion}']
        out['diff'] = wide[f'diff_mean50_{emotion}']
        out['count_1'] = wide[f'core_count_{emotion}']
        out['count_2'] = wide[f'expansion_count_{emotion}']
        out.to_csv(ce_dir / CE_COMPATIBLE_NAMES[emotion], index=False, encoding='utf-8-sig')

def write_outputs(four_result: ZonalResult, city_result: ZonalResult, four_wide: pd.DataFrame, four_long: pd.DataFrame, city_zone: pd.DataFrame, city_wide: pd.DataFrame, args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    four_wide.to_csv(OUT_DIR / 'ns_core_expansion_emotion_stats_wide.csv', index=False, encoding='utf-8-sig')
    four_long.to_csv(OUT_DIR / 'ns_core_expansion_emotion_stats_long.csv', index=False, encoding='utf-8-sig')
    city_zone.to_csv(OUT_DIR / 'city_zone_emotion_stats_recomputed.csv', index=False, encoding='utf-8-sig')
    city_wide.to_csv(OUT_DIR / 'city_core_expansion_emotion_wide_recomputed.csv', index=False, encoding='utf-8-sig')
    fig_ce = city_wide[['id_1', 'COUNTRY', 'NAME_1', 'lon', 'lat', 'NS', 'ns_group']].copy()
    for emotion in EMOTIONS:
        fig_ce[f'core_{emotion}'] = city_wide[f'core_mean50_{emotion}']
        fig_ce[f'expansion_{emotion}'] = city_wide[f'expansion_mean50_{emotion}']
        fig_ce[f'diff_{emotion}'] = city_wide[f'diff_mean50_{emotion}']
    fig_ce['core_eli'] = city_wide['core_eli50']
    fig_ce['expansion_eli'] = city_wide['expansion_eli50']
    fig_ce['eli_diff'] = city_wide['diff_eli50']
    fig_ce.to_csv(OUT_DIR / 'fig_2_4_3_core_expansion_input_recomputed.csv', index=False, encoding='utf-8-sig')
    write_ce_compatible_files(city_wide, OUT_DIR)
    audit = {'created_at': datetime.now().isoformat(timespec='seconds'), 'script': str(Path(__file__).resolve()), 'inputs': {'city_master': str(CITY_MASTER), 'city_zone_raster': str(CITY_ZONE_RASTER), 'four_class_raster': str(FOUR_CLASS_RASTER), 'emotion_rasters': {emotion: str(path) for emotion, path in EMOTION_RASTERS.items()}}, 'rules': {'city_points_policy': 'City metadata and coordinates are read unchanged from q3_city_driver_master.csv.', 'city_zone_rule': 'Global_City_C_E value = city_id * 10 + zone_code; 1=core, 2=expansion.', 'four_class_rule': {str(code): {'ns_group': ns, 'zone_label': zone} for code, (ns, zone) in FOUR_CLASS_LABELS.items()}, 'emotion_resampling': 'Emotion rasters are read through WarpedVRT on the zone raster grid with average resampling.', 'eli_formula': 'mean(beauty, wealthy, safer, livelier) - mean(boring, depression).', 'diff_formula': 'core - expansion.', 'nodata_policy': 'Zone nodata and emotion nodata/masked/non-finite values are excluded.'}, 'run_options': vars(args), 'diagnostics': {'four_class': {'valid_blocks': four_result.valid_blocks, 'scanned_blocks': four_result.scanned_blocks, 'valid_zone_pixels': four_result.valid_zone_pixels, 'elapsed_seconds': four_result.elapsed_seconds, 'zone_meta': four_result.zone_meta, 'rows_wide': int(len(four_wide))}, 'city_zone': {'valid_blocks': city_result.valid_blocks, 'scanned_blocks': city_result.scanned_blocks, 'valid_zone_pixels': city_result.valid_zone_pixels, 'elapsed_seconds': city_result.elapsed_seconds, 'zone_meta': city_result.zone_meta, 'rows_zone': int(len(city_zone)), 'rows_city_wide': int(len(city_wide)), 'paired_cities': int(city_wide[['core_eli50', 'expansion_eli50']].dropna().shape[0])}}, 'outputs': {'four_class_wide': str(OUT_DIR / 'ns_core_expansion_emotion_stats_wide.csv'), 'four_class_long': str(OUT_DIR / 'ns_core_expansion_emotion_stats_long.csv'), 'city_zone': str(OUT_DIR / 'city_zone_emotion_stats_recomputed.csv'), 'city_wide': str(OUT_DIR / 'city_core_expansion_emotion_wide_recomputed.csv'), 'figure_2_4_3_input': str(OUT_DIR / 'fig_2_4_3_core_expansion_input_recomputed.csv'), 'ce_compatible_dir': str(OUT_DIR / 'ce_reorganized_compatible')}}
    (OUT_DIR / 'recomputed_core_expansion_emotion_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-four-class', action='store_true')
    parser.add_argument('--skip-city-zone', action='store_true')
    parser.add_argument('--max-blocks', type=int, default=None, help='Debug only: stop after this many scanned blocks.')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    require_files([CITY_MASTER, *EMOTION_RASTERS.values()])
    if not args.skip_city_zone:
        require_files([CITY_ZONE_RASTER])
    if not args.skip_four_class:
        require_files([FOUR_CLASS_RASTER])
    meta = city_meta()
    max_city_zone_id = int(meta['id_1'].max()) * 10 + 3
    if args.skip_four_class:
        four_result = ZonalResult({}, {}, 0, 0, 0, 0.0, {})
        four_wide, four_long = (pd.DataFrame(), pd.DataFrame())
    else:
        print('[four-class] start', flush=True)
        four_result = zonal_means(FOUR_CLASS_RASTER, allowed_values=set(FOUR_CLASS_LABELS), initial_length=max(100, max(FOUR_CLASS_LABELS) + 1), progress_label='four-class', max_blocks=args.max_blocks)
        four_wide, four_long = build_four_class_tables(four_result)
        print(f'[four-class] done in {four_result.elapsed_seconds:.1f}s', flush=True)
    if args.skip_city_zone:
        city_result = ZonalResult({}, {}, 0, 0, 0, 0.0, {})
        city_zone, city_wide = (pd.DataFrame(), pd.DataFrame())
    else:
        print('[city-zone] start', flush=True)
        city_result = zonal_means(CITY_ZONE_RASTER, allowed_values=None, initial_length=max_city_zone_id, progress_label='city-zone', max_blocks=args.max_blocks)
        city_zone = build_city_zone_table(city_result, meta)
        city_wide = build_city_wide(city_zone)
        print(f'[city-zone] done in {city_result.elapsed_seconds:.1f}s', flush=True)
    write_outputs(four_result, city_result, four_wide, four_long, city_zone, city_wide, args)
    print(f'wrote outputs under: {OUT_DIR}', flush=True)
