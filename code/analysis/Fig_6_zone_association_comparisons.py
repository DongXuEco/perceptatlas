from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "_support"))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "figures"))

from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
import hashlib
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
PROJECT_ROOT = PACKAGE_ROOT
RUN_DIR = PROJECT_ROOT / 'runs' / '20260818-q3-final-mechanism-models'
DERIVED_DIR = PROJECT_ROOT / 'derived_data'
INPUTS = {'family': RUN_DIR / 'q3_formal_family.csv', 'permutation': RUN_DIR / 'q3_formal_permutation.csv', 'ale': RUN_DIR / 'q3_formal_ale.csv', 'pdp': RUN_DIR / 'q3_formal_pdp.csv', 'sample': RUN_DIR / 'q3_formal_sample.csv', 'performance': RUN_DIR / 'q3_formal_rf_cv_performance.csv', 'predictors': RUN_DIR / 'q3_formal_predictors_used.csv'}
OUTPUTS = {'family_contrast': DERIVED_DIR / 'q3_core_expansion_family_contrast.csv', 'variable_contrast': DERIVED_DIR / 'q3_core_expansion_variable_contrast.csv', 'response_curves': DERIVED_DIR / 'q3_core_expansion_response_curves.csv', 'mechanism_regime': DERIVED_DIR / 'q3_core_expansion_mechanism_regime.csv', 'metadata': DERIVED_DIR / 'q3_core_expansion_derivatives_metadata.json'}
CONTRAST_PAIRS = [{'contrast_scope': 'global', 'core_group': 'core', 'expansion_group': 'expansion', 'ns_context': 'all'}, {'contrast_scope': 'North', 'core_group': 'North_core', 'expansion_group': 'North_expansion', 'ns_context': 'North'}, {'contrast_scope': 'South', 'core_group': 'South_core', 'expansion_group': 'South_expansion', 'ns_context': 'South'}]

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read_csv(name: str) -> pd.DataFrame:
    path = INPUTS[name]
    if not resource(path).exists():
        raise FileNotFoundError(path)
    return pd.read_csv(resource(path))

def assert_unique(df: pd.DataFrame, keys: list[str], label: str) -> None:
    dup = df.duplicated(keys, keep=False)
    if dup.any():
        sample = df.loc[dup, keys].head(10).to_dict('records')
        raise ValueError(f'{label} has duplicated keys {keys}; examples: {sample}')

def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return np.where(den.to_numpy(dtype=float) > 0, num / den, np.nan)

def add_rank_and_share(df: pd.DataFrame, value_col: str, rank_keys: list[str], rank_col: str='rank_within_group', share_col: str='importance_share') -> pd.DataFrame:
    out = df.copy()
    out[rank_col] = out.groupby(rank_keys, dropna=False)[value_col].rank(method='first', ascending=False).astype(int)
    totals = out.groupby(rank_keys, dropna=False)[value_col].transform('sum')
    out[share_col] = np.where(totals > 0, out[value_col] / totals, np.nan)
    return out

def sample_lookup(sample: pd.DataFrame) -> pd.DataFrame:
    cols = ['table', 'outcome', 'outcome_role', 'group', 'group_type', 'n', 'n_complete_predictors', 'n_any_predictor_missing']
    keep = [c for c in cols if c in sample.columns]
    return sample[keep].copy()

def core_expansion_pair_frame(df: pd.DataFrame, value_kind: str) -> pd.DataFrame:
    if value_kind == 'family':
        entity_keys = ['family', 'family_label']
        value_cols = ['perm_importance_mean', 'importance_share', 'rank_within_group']
        id_cols = ['table', 'outcome', 'outcome_role']
    elif value_kind == 'variable':
        entity_keys = ['column', 'family', 'family_label', 'variable']
        optional = [c for c in ['unit', 'expected_direction'] if c in df.columns]
        entity_keys = entity_keys + optional
        value_cols = ['perm_importance_mean', 'perm_importance_sd', 'importance_share', 'rank_within_group']
        id_cols = ['table', 'outcome', 'outcome_role']
    else:
        raise ValueError(value_kind)
    out_parts: list[pd.DataFrame] = []
    zdf = df[df['table'].eq('city_zone')].copy()
    for pair in CONTRAST_PAIRS:
        core_group = pair['core_group']
        expansion_group = pair['expansion_group']
        core = zdf[zdf['group'].eq(core_group)].copy()
        expansion = zdf[zdf['group'].eq(expansion_group)].copy()
        if core.empty or expansion.empty:
            raise ValueError(f'Missing pair: {core_group} / {expansion_group}')
        keys = id_cols + entity_keys
        assert_unique(core, keys, f'{value_kind}:{core_group}')
        assert_unique(expansion, keys, f'{value_kind}:{expansion_group}')
        core = core[keys + ['group', 'group_type'] + value_cols].rename(columns={'group': 'core_group', 'group_type': 'core_group_type', **{c: f'core_{c}' for c in value_cols}})
        expansion = expansion[keys + ['group', 'group_type'] + value_cols].rename(columns={'group': 'expansion_group', 'group_type': 'expansion_group_type', **{c: f'expansion_{c}' for c in value_cols}})
        merged = core.merge(expansion, on=keys, how='inner', validate='one_to_one')
        if len(merged) != min(len(core), len(expansion)):
            raise ValueError(f'Unexpected merge size for {value_kind} {core_group}/{expansion_group}: core={len(core)}, expansion={len(expansion)}, merged={len(merged)}')
        merged.insert(0, 'contrast_scope', pair['contrast_scope'])
        merged.insert(1, 'ns_context', pair['ns_context'])
        merged['core_minus_expansion_importance'] = merged['core_perm_importance_mean'] - merged['expansion_perm_importance_mean']
        merged['abs_core_minus_expansion_importance'] = merged['core_minus_expansion_importance'].abs()
        merged['core_to_expansion_importance_ratio'] = safe_ratio(merged['core_perm_importance_mean'], merged['expansion_perm_importance_mean'])
        merged['core_minus_expansion_share'] = merged['core_importance_share'] - merged['expansion_importance_share']
        merged['abs_core_minus_expansion_share'] = merged['core_minus_expansion_share'].abs()
        merged['core_to_expansion_share_ratio'] = safe_ratio(merged['core_importance_share'], merged['expansion_importance_share'])
        merged['rank_shift_core_minus_expansion'] = merged['core_rank_within_group'] - merged['expansion_rank_within_group']
        out_parts.append(merged)
    return pd.concat(out_parts, ignore_index=True)

def attach_sample_counts(contrast: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    look = sample_lookup(sample)
    core_lookup = look.rename(columns={'group': 'core_group', 'group_type': 'core_group_type', 'n': 'core_n', 'n_complete_predictors': 'core_n_complete_predictors', 'n_any_predictor_missing': 'core_n_any_predictor_missing'})
    expansion_lookup = look.rename(columns={'group': 'expansion_group', 'group_type': 'expansion_group_type', 'n': 'expansion_n', 'n_complete_predictors': 'expansion_n_complete_predictors', 'n_any_predictor_missing': 'expansion_n_any_predictor_missing'})
    keys_core = ['table', 'outcome', 'outcome_role', 'core_group', 'core_group_type']
    keys_exp = ['table', 'outcome', 'outcome_role', 'expansion_group', 'expansion_group_type']
    out = contrast.merge(core_lookup, on=keys_core, how='left', validate='many_to_one')
    out = out.merge(expansion_lookup, on=keys_exp, how='left', validate='many_to_one')
    return out

def build_family_contrast(family: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    required = {'table', 'outcome', 'outcome_role', 'group', 'group_type', 'family', 'family_label', 'perm_importance_mean'}
    missing = required.difference(family.columns)
    if missing:
        raise ValueError(f'family missing columns: {sorted(missing)}')
    ranked = add_rank_and_share(family, value_col='perm_importance_mean', rank_keys=['table', 'outcome', 'group'])
    contrast = core_expansion_pair_frame(ranked, 'family')
    contrast = attach_sample_counts(contrast, sample)
    sort_cols = ['contrast_scope', 'outcome_role', 'outcome', 'abs_core_minus_expansion_share', 'family_label']
    return contrast.sort_values(sort_cols, ascending=[True, True, True, False, True])

def build_variable_contrast(permutation: pd.DataFrame, predictors: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    required = {'table', 'outcome', 'outcome_role', 'group', 'group_type', 'column', 'family', 'family_label', 'variable', 'perm_importance_mean', 'perm_importance_sd'}
    missing = required.difference(permutation.columns)
    if missing:
        raise ValueError(f'permutation missing columns: {sorted(missing)}')
    predictor_meta = predictors[[c for c in ['column', 'unit', 'expected_direction', 'selection_order'] if c in predictors.columns]].drop_duplicates('column')
    enriched = permutation.merge(predictor_meta, on='column', how='left', validate='many_to_one')
    ranked = add_rank_and_share(enriched, value_col='perm_importance_mean', rank_keys=['table', 'outcome', 'group'])
    contrast = core_expansion_pair_frame(ranked, 'variable')
    contrast = attach_sample_counts(contrast, sample)
    sort_cols = ['contrast_scope', 'outcome_role', 'outcome', 'abs_core_minus_expansion_share', 'variable']
    return contrast.sort_values(sort_cols, ascending=[True, True, True, False, True])

def zone_context_from_group(group: str) -> tuple[str, str, str]:
    if group == 'core':
        return ('global', 'all', 'core')
    if group == 'expansion':
        return ('global', 'all', 'expansion')
    if group == 'North_core':
        return ('North', 'North', 'core')
    if group == 'North_expansion':
        return ('North', 'North', 'expansion')
    if group == 'South_core':
        return ('South', 'South', 'core')
    if group == 'South_expansion':
        return ('South', 'South', 'expansion')
    raise ValueError(f'Unsupported response-curve group: {group}')

def build_response_curves(ale: pd.DataFrame, pdp: pd.DataFrame, predictors: pd.DataFrame) -> pd.DataFrame:
    curve_groups = {'core', 'expansion', 'North_core', 'North_expansion', 'South_core', 'South_expansion'}
    predictor_cols = [c for c in ['column', 'family', 'family_label', 'variable', 'unit', 'expected_direction', 'selection_order'] if c in predictors.columns]
    predictor_meta = predictors[predictor_cols].drop_duplicates('column')
    ale_keep = ale[ale['table'].eq('city_zone') & ale['group'].isin(curve_groups)].copy()
    ale_keep['curve_type'] = 'ALE'
    ale_keep = ale_keep.rename(columns={'ale': 'curve_value'})
    if 'n_bin' not in ale_keep.columns:
        ale_keep['n_bin'] = np.nan
    pdp_keep = pdp[pdp['table'].eq('city_zone') & pdp['group'].isin(curve_groups)].copy()
    pdp_keep['curve_type'] = 'PDP'
    pdp_keep = pdp_keep.rename(columns={'partial_dependence': 'curve_value'})
    pdp_keep['n_bin'] = np.nan
    common_cols = ['table', 'outcome', 'outcome_role', 'group', 'group_type', 'column', 'x', 'curve_value', 'n_bin', 'curve_type']
    curves = pd.concat([ale_keep[common_cols], pdp_keep[common_cols]], ignore_index=True)
    contexts = curves['group'].map(zone_context_from_group)
    curves['contrast_scope'] = [x[0] for x in contexts]
    curves['ns_context'] = [x[1] for x in contexts]
    curves['zone_label'] = [x[2] for x in contexts]
    curves['paired_group'] = curves['group'].map({'core': 'expansion', 'expansion': 'core', 'North_core': 'North_expansion', 'North_expansion': 'North_core', 'South_core': 'South_expansion', 'South_expansion': 'South_core'})
    curves = curves.merge(predictor_meta, on='column', how='left', validate='many_to_one')
    curves['x_rank_within_curve'] = curves.groupby(['table', 'outcome', 'group', 'column', 'curve_type'], dropna=False)['x'].rank(method='first').astype(int)
    curves['n_points_in_curve'] = curves.groupby(['table', 'outcome', 'group', 'column', 'curve_type'], dropna=False)['x'].transform('size')
    cols = ['contrast_scope', 'ns_context', 'zone_label', 'paired_group', 'table', 'outcome', 'outcome_role', 'group', 'group_type', 'column', 'family', 'family_label', 'variable', 'unit', 'expected_direction', 'curve_type', 'x', 'curve_value', 'n_bin', 'x_rank_within_curve', 'n_points_in_curve']
    return curves[cols].sort_values(['contrast_scope', 'outcome_role', 'outcome', 'column', 'zone_label', 'curve_type', 'x'])

def build_mechanism_regime(family_contrast: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    base_keys = ['contrast_scope', 'ns_context', 'table', 'outcome', 'outcome_role', 'core_group', 'expansion_group']
    for keys, sub in family_contrast.groupby(base_keys, dropna=False):
        row = dict(zip(base_keys, keys))
        sub = sub.copy()
        core_ranked = sub.sort_values(['core_rank_within_group', 'family_label'])
        exp_ranked = sub.sort_values(['expansion_rank_within_group', 'family_label'])
        for prefix, ranked, group_col, share_col, imp_col in [('core', core_ranked, 'core_group', 'core_importance_share', 'core_perm_importance_mean'), ('expansion', exp_ranked, 'expansion_group', 'expansion_importance_share', 'expansion_perm_importance_mean')]:
            top = ranked.head(3).reset_index(drop=True)
            row[f'{prefix}_group'] = ranked[group_col].iloc[0]
            for i in range(3):
                if i < len(top):
                    row[f'{prefix}_top{i + 1}_family'] = top.loc[i, 'family']
                    row[f'{prefix}_top{i + 1}_family_label'] = top.loc[i, 'family_label']
                    row[f'{prefix}_top{i + 1}_share'] = top.loc[i, share_col]
                    row[f'{prefix}_top{i + 1}_importance'] = top.loc[i, imp_col]
                else:
                    row[f'{prefix}_top{i + 1}_family'] = np.nan
                    row[f'{prefix}_top{i + 1}_family_label'] = np.nan
                    row[f'{prefix}_top{i + 1}_share'] = np.nan
                    row[f'{prefix}_top{i + 1}_importance'] = np.nan
            row[f'{prefix}_top3_family_set'] = '|'.join(top['family'].astype(str).tolist())
            row[f'{prefix}_top3_family_label_set'] = '|'.join(top['family_label'].astype(str).tolist())
            row[f'{prefix}_top3_share_sum'] = top[share_col].sum()
        core_shares = sub.set_index('family')['core_importance_share'].fillna(0.0)
        exp_shares = sub.set_index('family')['expansion_importance_share'].fillna(0.0)
        all_families = sorted(set(core_shares.index) | set(exp_shares.index))
        core_vec = core_shares.reindex(all_families, fill_value=0.0).to_numpy(dtype=float)
        exp_vec = exp_shares.reindex(all_families, fill_value=0.0).to_numpy(dtype=float)
        l1 = float(np.abs(core_vec - exp_vec).sum())
        row['family_share_l1_distance'] = l1
        row['family_share_total_variation'] = 0.5 * l1
        denom = float(np.linalg.norm(core_vec) * np.linalg.norm(exp_vec))
        row['family_share_cosine_similarity'] = float(np.dot(core_vec, exp_vec) / denom) if denom > 0 else np.nan
        core_top3 = set(str(row['core_top3_family_set']).split('|'))
        exp_top3 = set(str(row['expansion_top3_family_set']).split('|'))
        union = core_top3 | exp_top3
        inter = core_top3 & exp_top3
        row['top1_same'] = row['core_top1_family'] == row['expansion_top1_family']
        row['top3_overlap_count'] = len(inter)
        row['top3_jaccard'] = len(inter) / len(union) if union else np.nan
        row['dominant_family_transition'] = f"{row['core_top1_family_label']} -> {row['expansion_top1_family_label']}"
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(['contrast_scope', 'outcome_role', 'outcome', 'family_share_total_variation'], ascending=[True, True, True, False])

def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding='utf-8-sig', float_format='%.15g')

def main() -> None:
    missing = [str(p) for p in INPUTS.values() if not resource(p).exists()]
    if missing:
        raise FileNotFoundError('Missing input files: ' + '; '.join(missing))
    family = read_csv('family')
    permutation = read_csv('permutation')
    ale = read_csv('ale')
    pdp = read_csv('pdp')
    sample = read_csv('sample')
    predictors = read_csv('predictors')
    family_contrast = build_family_contrast(family, sample)
    variable_contrast = build_variable_contrast(permutation, predictors, sample)
    response_curves = build_response_curves(ale, pdp, predictors)
    mechanism_regime = build_mechanism_regime(family_contrast)
    write_csv(family_contrast, OUTPUTS['family_contrast'])
    write_csv(variable_contrast, OUTPUTS['variable_contrast'])
    write_csv(response_curves, OUTPUTS['response_curves'])
    write_csv(mechanism_regime, OUTPUTS['mechanism_regime'])
    metadata = {'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'purpose': 'Core-expansion derivative tables for Figure Q3-3.', 'inputs': {k: {'path': str(v), 'sha256': file_sha256(v)} for k, v in INPUTS.items()}, 'outputs': {k: {'path': str(v), 'rows': int(pd.read_csv(resource(v)).shape[0]), 'sha256': file_sha256(v)} for k, v in OUTPUTS.items() if k != 'metadata'}, 'contrast_pairs': CONTRAST_PAIRS, 'formulas': {'importance_share': 'perm_importance_mean / sum(perm_importance_mean) within table-outcome-group.', 'core_minus_expansion_importance': 'core_perm_importance_mean - expansion_perm_importance_mean.', 'core_minus_expansion_share': 'core_importance_share - expansion_importance_share.', 'family_share_l1_distance': 'sum(abs(core_family_share - expansion_family_share)) across all families.', 'family_share_total_variation': '0.5 * family_share_l1_distance.', 'family_share_cosine_similarity': 'cosine similarity between core and expansion family-share vectors.', 'top3_jaccard': 'intersection(top3 core families, top3 expansion families) / union(top3 core families, top3 expansion families).'}, 'integrity_notes': ['No synthetic data are generated.', 'The mechanism_regime table is group-outcome level, not city-level, because available formal model outputs provide group-level importance rather than sample-level SHAP for city-zone groups.', 'Response curves preserve ALE and PDP as long-format rows because ALE and PDP use different x grids.'], 'row_counts': {'family_input': int(len(family)), 'permutation_input': int(len(permutation)), 'ale_input': int(len(ale)), 'pdp_input': int(len(pdp)), 'sample_input': int(len(sample)), 'family_contrast': int(len(family_contrast)), 'variable_contrast': int(len(variable_contrast)), 'response_curves': int(len(response_curves)), 'mechanism_regime': int(len(mechanism_regime))}}
    OUTPUTS['metadata'].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metadata['row_counts'], ensure_ascii=False, indent=2))
    print('Saved:')
    for k, v in OUTPUTS.items():
        print(f'- {k}: {v}')
