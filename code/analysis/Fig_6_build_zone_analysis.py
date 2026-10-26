from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "_support"))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "figures"))

from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
PROJECT_ROOT = PACKAGE_ROOT
IN_DIR = PROJECT_ROOT / 'derived_data' / 'recomputed_core_expansion'
OLD_MASTER = PROJECT_ROOT / 'derived_data' / 'q3_city_zone_driver_master.csv'
OUT_MASTER = IN_DIR / 'q3_city_zone_driver_master_recomputed_emotion.csv'
OUT_WITH_COUNTS = IN_DIR / 'q3_city_zone_driver_master_recomputed_emotion_with_counts.csv'
OUT_AUDIT = IN_DIR / 'q3_city_zone_driver_master_recomputed_emotion_audit.json'
EMOTIONS = ['beauty', 'wealthy', 'safer', 'livelier', 'boring', 'depression']
POSITIVE = ['beauty', 'wealthy', 'safer', 'livelier']
NEGATIVE = ['boring', 'depression']
OUTCOME_COLS = [f'mean50_{emotion}' for emotion in EMOTIONS]
GAP_COLS = [f'core_minus_expansion_mean50_{emotion}' for emotion in EMOTIONS]
SCORE_COLS = ['positive_score50', 'negative_score50', 'eli50', 'zone_emotion_mean_count']
REPLACE_COLS = OUTCOME_COLS + GAP_COLS + SCORE_COLS
KEY_COLS = ['id_1', 'zone_code']
UNCHANGED_CHECK_COLS = ['id_1', 'zone_code', 'zone_label', 'COUNTRY', 'NAME_1', 'lon', 'lat', 'NS', 'ns_group']

def main() -> None:
    old = pd.read_csv(resource(OLD_MASTER))
    stats = pd.read_csv(resource(IN_DIR / 'city_zone_emotion_stats_recomputed.csv'))
    stats = stats.dropna(subset=KEY_COLS).copy()
    stats['id_1'] = pd.to_numeric(stats['id_1'], errors='coerce').astype('Int64')
    stats['zone_code'] = pd.to_numeric(stats['zone_code'], errors='coerce').astype('Int64')
    stats = stats.dropna(subset=KEY_COLS).copy()
    stats['id_1'] = stats['id_1'].astype(int)
    stats['zone_code'] = stats['zone_code'].astype(int)
    count_cols = [f'count_{emotion}' for emotion in EMOTIONS]
    merge_cols = KEY_COLS + ['zone_value'] + REPLACE_COLS + count_cols
    missing_cols = [col for col in merge_cols if col not in stats.columns]
    if missing_cols:
        raise KeyError(f'Missing columns in recomputed stats: {missing_cols}')
    merged = old.merge(stats[merge_cols], on=KEY_COLS, how='left', suffixes=('', '__recomputed'), validate='one_to_one')
    out = old.copy()
    out_counts = old.copy()
    out_counts['recomputed_zone_value'] = merged['zone_value']
    for col in REPLACE_COLS:
        out[col] = merged[f'{col}__recomputed']
        out_counts[col] = merged[f'{col}__recomputed']
    for col in count_cols:
        out_counts[f'recomputed_{col}'] = merged[col]
    out['positive_score50'] = out[[f'mean50_{emotion}' for emotion in POSITIVE]].mean(axis=1)
    out['negative_score50'] = out[[f'mean50_{emotion}' for emotion in NEGATIVE]].mean(axis=1)
    out['eli50'] = out['positive_score50'] - out['negative_score50']
    out_counts['positive_score50'] = out['positive_score50']
    out_counts['negative_score50'] = out['negative_score50']
    out_counts['eli50'] = out['eli50']
    for emotion in EMOTIONS:
        pivot = out.pivot_table(index='id_1', columns='zone_label', values=f'mean50_{emotion}', aggfunc='first')
        if {'core', 'expansion'}.issubset(set(pivot.columns)):
            gap = pivot['core'] - pivot['expansion']
        else:
            gap = pd.Series(dtype=float)
        out[f'core_minus_expansion_mean50_{emotion}'] = out['id_1'].map(gap)
        out_counts[f'core_minus_expansion_mean50_{emotion}'] = out[f'core_minus_expansion_mean50_{emotion}']
    out.to_csv(OUT_MASTER, index=False, encoding='utf-8-sig')
    out_counts.to_csv(OUT_WITH_COUNTS, index=False, encoding='utf-8-sig')
    old_check = old[UNCHANGED_CHECK_COLS].copy()
    new_check = out[UNCHANGED_CHECK_COLS].copy()
    unchanged_equal = old_check.astype(str).equals(new_check.astype(str))
    lon_diff = float((pd.to_numeric(old['lon'], errors='coerce') - pd.to_numeric(out['lon'], errors='coerce')).abs().max())
    lat_diff = float((pd.to_numeric(old['lat'], errors='coerce') - pd.to_numeric(out['lat'], errors='coerce')).abs().max())
    formula = out[[f'mean50_{emotion}' for emotion in POSITIVE]].mean(axis=1) - out[[f'mean50_{emotion}' for emotion in NEGATIVE]].mean(axis=1)
    formula_error = (formula - out['eli50']).abs()
    audit = {'created_at': datetime.now().isoformat(timespec='seconds'), 'inputs': {'old_master': str(OLD_MASTER), 'recomputed_stats': str(IN_DIR / 'city_zone_emotion_stats_recomputed.csv')}, 'outputs': {'same_schema_master': str(OUT_MASTER), 'with_counts': str(OUT_WITH_COUNTS), 'audit': str(OUT_AUDIT)}, 'policy': {'preserved': 'All existing city metadata, zone rows, and driver variables from q3_city_zone_driver_master.csv.', 'replaced': REPLACE_COLS, 'missing_recomputed_values': 'Rows without recomputed zone emotion values are kept, with recomputed outcome fields set to NaN.'}, 'diagnostics': {'old_rows': int(len(old)), 'output_rows': int(len(out)), 'recomputed_source_rows': int(len(stats)), 'rows_with_recomputed_eli': int(out['eli50'].notna().sum()), 'paired_cities_with_recomputed_eli': int(out.pivot_table(index='id_1', columns='zone_label', values='eli50', aggfunc='first').dropna(subset=['core', 'expansion']).shape[0]), 'zone_label_counts': out['zone_label'].value_counts(dropna=False).to_dict(), 'city_metadata_equal_stringwise': bool(unchanged_equal), 'max_abs_lon_diff': lon_diff, 'max_abs_lat_diff': lat_diff, 'eli_formula_max_abs_error': float(np.nanmax(formula_error.to_numpy(dtype=float))), 'non_null_by_outcome': {col: int(out[col].notna().sum()) for col in OUTCOME_COLS + ['eli50']}}}
    OUT_AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(audit['diagnostics'], ensure_ascii=False, indent=2))
