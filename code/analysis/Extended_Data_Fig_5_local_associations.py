from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "_support"))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "figures"))

from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
try:
    import shap
except Exception as exc:
    raise RuntimeError('This script requires shap. Use the D:\\Anaconda3\\envs\\pytorch Python environment.') from exc
PROJECT_ROOT = PACKAGE_ROOT
DERIVED_DIR = PROJECT_ROOT / 'derived_data'
RUN_FORMAL = PROJECT_ROOT / 'runs' / '20260818-q3-final-mechanism-models'
OUT_DIR = output_dir('Extended_Data_Fig_5')
CITY_MASTER = DERIVED_DIR / 'q3_city_driver_master.csv'
PREDICTORS_USED = RUN_FORMAL / 'q3_formal_predictors_used.csv'
FORMAL_PREDICTIONS = RUN_FORMAL / 'q3_formal_predictions.csv'
FORMAL_SAMPLE = RUN_FORMAL / 'q3_formal_sample.csv'
FORMAL_SHAP = RUN_FORMAL / 'q3_formal_shap.csv'
OUT_MAIN = OUT_DIR / 'q3_city_local_mechanism_map_data.csv'
OUT_FAMILY_LONG = OUT_DIR / 'q3_city_local_mechanism_family_long.csv'
OUT_AUDIT = OUT_DIR / 'q3_city_local_mechanism_map_data_audit.json'
CITY_Q32_OUTCOMES = ['eli50', 'compound_gini50', 'compound_theil50', 'compound_p9010_50']
CITY_Q32_GROUPS = ['global', 'North', 'South']
N_ESTIMATORS = 180
MIN_SAMPLES_LEAF = 5
MAX_FEATURES = 0.7
MIN_MODEL_N = 450
RANDOM_STATE = 42
FAMILY_COLORS = {'green_blue_space': '#62C0D4', 'environment_risk': '#E5170A', 'built_environment': '#357DBD', 'economic_population': '#6E789D', 'public_service_poi': '#F28817', 'public_service_accessibility_nonmotor': '#F1B47A', 'public_service_road': '#F0CFA1', 'terrain_control': '#909BB6', 'location_control': '#8170A8'}

@dataclass
class ModelAudit:
    outcome: str
    model_group: str
    group_type: str
    n_rows: int
    n_complete_predictors: int
    expected_sample_n: int | None
    sample_n_matches_formal: bool | None
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

def coerce_design(df: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    x = df[predictors].apply(pd.to_numeric, errors='coerce')
    return x.replace([np.inf, -np.inf], np.nan)

def group_mask(df: pd.DataFrame, group: str) -> pd.Series:
    if group == 'global':
        return pd.Series(True, index=df.index)
    return df['ns_group'].astype(str).eq(group)

def safe_family_col(prefix: str, family: str) -> str:
    return f'{prefix}__{family}'

def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    city = pd.read_csv(resource(CITY_MASTER), low_memory=False)
    predictors = pd.read_csv(resource(PREDICTORS_USED))
    predictors = predictors.sort_values('selection_order').reset_index(drop=True)
    predictions = pd.read_csv(resource(FORMAL_PREDICTIONS), low_memory=False)
    sample = pd.read_csv(resource(FORMAL_SAMPLE)) if resource(FORMAL_SAMPLE).exists() else pd.DataFrame()
    formal_shap = pd.read_csv(resource(FORMAL_SHAP)) if resource(FORMAL_SHAP).exists() else pd.DataFrame()
    return (city, predictors, predictions, sample, formal_shap)

def expected_sample_n(sample: pd.DataFrame, outcome: str, group: str) -> int | None:
    if sample.empty:
        return None
    hit = sample[sample['table'].astype(str).eq('city') & sample['outcome'].astype(str).eq(outcome) & sample['group'].astype(str).eq(group)]
    if hit.empty:
        return None
    return int(hit.iloc[0]['n'])

def oof_subset(predictions: pd.DataFrame, outcome: str, group: str) -> pd.DataFrame:
    cols = ['id_1', 'outcome', 'group', 'observed', 'oof_prediction', 'oof_residual']
    sub = predictions[predictions['table'].astype(str).eq('city') & predictions['outcome'].astype(str).eq(outcome) & predictions['group'].astype(str).eq(group)][cols].copy()
    return sub.rename(columns={'group': 'model_group', 'oof_prediction': 'formal_oof_prediction', 'oof_residual': 'formal_oof_residual'})

def family_order(predictors: pd.DataFrame) -> list[str]:
    return predictors['family'].drop_duplicates().astype(str).tolist()

def build_one_model(city: pd.DataFrame, predictors: pd.DataFrame, predictions: pd.DataFrame, sample: pd.DataFrame, formal_shap: pd.DataFrame, outcome: str, group: str) -> tuple[pd.DataFrame, pd.DataFrame, ModelAudit]:
    selected = predictors['column'].astype(str).tolist()
    group_type = 'global' if group == 'global' else 'north_south'
    sub = city.loc[group_mask(city, group)].copy()
    y_series = pd.to_numeric(sub[outcome], errors='coerce')
    sub = sub.loc[y_series.notna()].copy()
    if len(sub) < MIN_MODEL_N:
        raise ValueError(f'Too few rows for {outcome}/{group}: n={len(sub)}')
    y = pd.to_numeric(sub[outcome], errors='coerce').to_numpy(dtype=float)
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
    families = family_order(predictors)
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
    dominant_abs = family_abs_frame.lookup(family_abs_frame.index, dominant_family) if hasattr(family_abs_frame, 'lookup') else np.array([family_abs_frame.loc[idx, fam] for idx, fam in zip(family_abs_frame.index, dominant_family)])
    dominant_signed = np.array([family_signed_frame.loc[idx, fam] for idx, fam in zip(family_signed_frame.index, dominant_family)])
    dominant_share = np.array([family_share_frame.loc[idx, fam] for idx, fam in zip(family_share_frame.index, dominant_family)])
    sorted_shares = np.sort(family_share_frame.to_numpy(), axis=1)
    rank_margin = sorted_shares[:, -1] - sorted_shares[:, -2] if family_share_frame.shape[1] > 1 else sorted_shares[:, -1]
    top_feature_idx = feature_abs.argmax(axis=1)
    top_feature = np.array(selected, dtype=object)[top_feature_idx]
    top_feature_abs = feature_abs[np.arange(len(sub)), top_feature_idx]
    top_feature_signed = shap_values[np.arange(len(sub)), top_feature_idx]
    top_feature_family = [str(meta[col]['family']) for col in top_feature]
    top_feature_label = [str(meta[col]['variable']) for col in top_feature]
    top_feature_share = np.divide(top_feature_abs, total_abs.to_numpy(), out=np.zeros(len(sub)), where=np.isfinite(total_abs.to_numpy()))
    base_cols = ['id_1', 'COUNTRY', 'NAME_1', 'lon', 'lat', 'NS', 'ns_group', 'eli50', 'compound_gini50', 'compound_theil50', 'compound_p9010_50']
    base_cols = [col for col in base_cols if col in sub.columns]
    out = sub[base_cols].copy()
    out['table'] = 'city'
    out['outcome'] = outcome
    out['outcome_role'] = 'primary'
    out['model_group'] = group
    out['group_type'] = group_type
    out['observed'] = y
    out['full_model_prediction'] = full_prediction
    out['full_model_residual'] = y - full_prediction
    out['shap_expected_value'] = expected_value
    out['shap_reconstructed_prediction'] = reconstructed
    out['shap_reconstruction_error'] = reconstruction_error
    out['n_predictors'] = len(selected)
    out['has_complete_predictors_city'] = x.notna().all(axis=1).astype(int).to_numpy()
    out['n_missing_predictors_city'] = x.isna().sum(axis=1).to_numpy()
    out['local_total_abs_contribution'] = total_abs.to_numpy()
    out['dominant_family'] = dominant_family.to_numpy()
    out['dominant_family_label'] = [family_labels.get(fam, fam) for fam in dominant_family]
    out['dominant_family_color'] = [FAMILY_COLORS.get(fam, '#C8D0C1') for fam in dominant_family]
    out['dominant_family_abs_contribution'] = dominant_abs
    out['dominant_family_signed_contribution'] = dominant_signed
    out['dominant_family_share'] = dominant_share
    out['dominant_family_share_margin'] = rank_margin
    out['top_variable'] = top_feature
    out['top_variable_label'] = top_feature_label
    out['top_variable_family'] = top_feature_family
    out['top_variable_family_label'] = [family_labels.get(fam, fam) for fam in top_feature_family]
    out['top_variable_abs_contribution'] = top_feature_abs
    out['top_variable_signed_contribution'] = top_feature_signed
    out['top_variable_share'] = top_feature_share
    for family in families:
        out[safe_family_col('family_abs_contribution', family)] = family_abs_frame[family].to_numpy()
        out[safe_family_col('family_signed_contribution', family)] = family_signed_frame[family].to_numpy()
        out[safe_family_col('family_share', family)] = family_share_frame[family].to_numpy()
    out = out.merge(oof_subset(predictions, outcome, group), on=['id_1', 'outcome', 'model_group', 'observed'], how='left')
    long_parts = []
    for rank_family in families:
        long_parts.append(pd.DataFrame({'id_1': sub['id_1'].to_numpy(), 'COUNTRY': sub['COUNTRY'].to_numpy() if 'COUNTRY' in sub else '', 'NAME_1': sub['NAME_1'].to_numpy() if 'NAME_1' in sub else '', 'lon': sub['lon'].to_numpy() if 'lon' in sub else np.nan, 'lat': sub['lat'].to_numpy() if 'lat' in sub else np.nan, 'ns_group': sub['ns_group'].to_numpy() if 'ns_group' in sub else '', 'outcome': outcome, 'model_group': group, 'family': rank_family, 'family_label': family_labels.get(rank_family, rank_family), 'family_color': FAMILY_COLORS.get(rank_family, '#C8D0C1'), 'local_abs_contribution': family_abs_frame[rank_family].to_numpy(), 'local_signed_contribution': family_signed_frame[rank_family].to_numpy(), 'local_share': family_share_frame[rank_family].to_numpy(), 'is_dominant_family': dominant_family.to_numpy() == rank_family}))
    family_long = pd.concat(long_parts, ignore_index=True)
    mean_by_family = family_long.groupby(['family', 'family_label'], as_index=False)['local_abs_contribution'].mean().sort_values('local_abs_contribution', ascending=False)
    top = mean_by_family.iloc[0]
    audit = ModelAudit(outcome=outcome, model_group=group, group_type=group_type, n_rows=int(len(out)), n_complete_predictors=n_complete, expected_sample_n=expected_sample_n(sample, outcome, group), sample_n_matches_formal=None, max_abs_shap_reconstruction_error=float(np.nanmax(reconstruction_error)), max_family_share_sum_error=float(np.nanmax(share_sum_error)), top_family_label=str(top['family_label']), top_family_mean_abs_shap=float(top['local_abs_contribution']))
    audit.sample_n_matches_formal = None if audit.expected_sample_n is None else audit.expected_sample_n == audit.n_rows
    if not formal_shap.empty:
        formal = formal_shap[formal_shap['table'].astype(str).eq('city') & formal_shap['outcome'].astype(str).eq(outcome) & formal_shap['group'].astype(str).eq(group)]
        if not formal.empty:
            formal_family = formal.groupby(['family', 'family_label'], as_index=False)['mean_abs_shap'].sum().rename(columns={'mean_abs_shap': 'formal_sample_mean_abs_shap_sum'})
            compare = mean_by_family.merge(formal_family, on=['family', 'family_label'], how='left')
            compare_path = OUT_DIR / f'q3_city_local_mechanism_family_compare_{outcome}_{group}.csv'
            compare.to_csv(compare_path, index=False, encoding='utf-8-sig')
    return (out, family_long, audit)

def main() -> None:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    city, predictors, predictions, sample, formal_shap = load_inputs()
    main_frames = []
    long_frames = []
    audits: list[ModelAudit] = []
    for outcome in CITY_Q32_OUTCOMES:
        for group in CITY_Q32_GROUPS:
            print(f'[local mechanism] {outcome} / {group}', flush=True)
            main_df, family_long, audit = build_one_model(city, predictors, predictions, sample, formal_shap, outcome, group)
            main_frames.append(main_df)
            long_frames.append(family_long)
            audits.append(audit)
    main_out = pd.concat(main_frames, ignore_index=True)
    family_long_out = pd.concat(long_frames, ignore_index=True)
    main_out.to_csv(OUT_MAIN, index=False, encoding='utf-8-sig')
    family_long_out.to_csv(OUT_FAMILY_LONG, index=False, encoding='utf-8-sig')
    audit = {'created_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'elapsed_seconds': time.time() - started, 'method': 'Full-sample random-forest refit with formal Q3 parameters; local SHAP values aggregated to driver families by absolute contribution.', 'inputs': {'city_master': str(CITY_MASTER), 'predictors_used': str(PREDICTORS_USED), 'formal_predictions': str(FORMAL_PREDICTIONS), 'formal_sample': str(FORMAL_SAMPLE), 'formal_shap': str(FORMAL_SHAP)}, 'outputs': {'main': str(OUT_MAIN), 'family_long': str(OUT_FAMILY_LONG), 'audit': str(OUT_AUDIT)}, 'rf_parameters': {'n_estimators': N_ESTIMATORS, 'min_samples_leaf': MIN_SAMPLES_LEAF, 'max_features': MAX_FEATURES, 'random_state': RANDOM_STATE}, 'row_counts': {'main_rows': int(len(main_out)), 'family_long_rows': int(len(family_long_out)), 'unique_city_ids_in_main': int(main_out['id_1'].nunique()), 'outcomes': CITY_Q32_OUTCOMES, 'groups': CITY_Q32_GROUPS}, 'model_audits': [asdict(item) for item in audits]}
    OUT_AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(OUT_MAIN, flush=True)
    print(OUT_FAMILY_LONG, flush=True)
    print(OUT_AUDIT, flush=True)
