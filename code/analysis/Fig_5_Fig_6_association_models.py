from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "_support"))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "figures"))

from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
import json
import math
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
PROJECT_ROOT = PACKAGE_ROOT
DERIVED_DIR = PROJECT_ROOT / 'derived_data'
PREV_RUN_DIR = PROJECT_ROOT / 'runs' / '20260818-q3-mechanism-modeling'
RUN_DIR = PROJECT_ROOT / 'runs' / '20260818-q3-final-mechanism-models'
CITY_MASTER = DERIVED_DIR / 'q3_city_driver_master.csv'
ZONE_MASTER = DERIVED_DIR / 'q3_city_zone_driver_master.csv'
FINAL_PREDICTORS = PREV_RUN_DIR / 'q3_final_predictors_recommended.csv'
CITY_PRIMARY_OUTCOMES = ['eli50', 'compound_gini50', 'compound_theil50', 'compound_p9010_50']
ZONE_PRIMARY_OUTCOMES = ['eli50', 'positive_score50', 'negative_score50']
ZONE_SECONDARY_OUTCOMES = ['mean50_beauty', 'mean50_wealthy', 'mean50_safer', 'mean50_livelier', 'mean50_boring', 'mean50_depression']
N_ESTIMATORS = 180
MIN_SAMPLES_LEAF = 5
MAX_FEATURES = 0.7
N_SPLITS = 3
EXPLAIN_SAMPLE_GLOBAL = 900
EXPLAIN_SAMPLE_SUBGROUP = 650
PERM_REPEATS = 8
TOP_N_EFFECT_FEATURES = 8
MIN_MODEL_N = 450
RANDOM_STATE = 42
warnings.filterwarnings('ignore', message='X has feature names, but RandomForestRegressor was fitted without feature names', category=UserWarning)

@dataclass
class ModelResult:
    table: str
    outcome: str
    outcome_role: str
    group: str
    group_type: str
    n: int
    n_complete_predictors: int
    r2_mean: float
    r2_sd: float
    rmse_mean: float
    rmse_sd: float
    mae_mean: float
    mae_sd: float
    oof_r2: float
    oof_rmse: float
    oof_mae: float

def safe_name(value: str) -> str:
    return value.replace('/', '_').replace('\\', '_').replace(' ', '_').replace(':', '_')

def load_predictors() -> pd.DataFrame:
    pred = pd.read_csv(resource(FINAL_PREDICTORS))
    pred = pred.sort_values('selection_order')
    return pred

def available_outcomes(df: pd.DataFrame, outcomes: list[str]) -> list[str]:
    return [col for col in outcomes if col in df.columns]

def add_group_specs(df: pd.DataFrame, table_name: str, outcome_role: str='primary') -> list[tuple[str, str, pd.Series]]:
    specs: list[tuple[str, str, pd.Series]] = [('global', 'global', pd.Series(True, index=df.index))]
    if 'ns_group' in df.columns:
        for group in ['North', 'South']:
            specs.append((group, 'north_south', df['ns_group'].astype(str).eq(group)))
    if table_name == 'city_zone' and 'zone_label' in df.columns:
        for zone in ['core', 'expansion']:
            specs.append((zone, 'core_expansion', df['zone_label'].astype(str).eq(zone)))
        if outcome_role != 'primary':
            return [spec for spec in specs if spec[1] in {'global', 'core_expansion'}]
        if 'ns_group' in df.columns:
            for ns in ['North', 'South']:
                for zone in ['core', 'expansion']:
                    specs.append((f'{ns}_{zone}', 'north_south_x_core_expansion', df['ns_group'].astype(str).eq(ns) & df['zone_label'].astype(str).eq(zone)))
    return specs

def coerce_design(df: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    out = df[predictors].apply(pd.to_numeric, errors='coerce')
    return out.replace([np.inf, -np.inf], np.nan)

def make_model() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=N_ESTIMATORS, min_samples_leaf=MIN_SAMPLES_LEAF, max_features=MAX_FEATURES, random_state=RANDOM_STATE, n_jobs=-1)

def cv_fit_predict(x: pd.DataFrame, y: np.ndarray) -> tuple[ModelResult | None, RandomForestRegressor, pd.DataFrame, np.ndarray, np.ndarray, list[dict[str, float]]]:
    n = len(y)
    n_complete = int(x.notna().all(axis=1).sum())
    if n < MIN_MODEL_N:
        raise ValueError(f'Too few samples for formal model: n={n}')
    n_splits = min(N_SPLITS, max(2, n // MIN_MODEL_N))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = np.full(n, np.nan, dtype=float)
    fold_rows: list[dict[str, float]] = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(x), start=1):
        pipe = make_pipeline(SimpleImputer(strategy='median'), make_model())
        pipe.fit(x.iloc[train_idx], y[train_idx])
        pred = pipe.predict(x.iloc[test_idx])
        oof[test_idx] = pred
        fold_rows.append({'fold': fold, 'n_train': int(len(train_idx)), 'n_test': int(len(test_idx)), 'r2': float(r2_score(y[test_idx], pred)), 'rmse': float(math.sqrt(mean_squared_error(y[test_idx], pred))), 'mae': float(mean_absolute_error(y[test_idx], pred))})
    pipe_full = make_pipeline(SimpleImputer(strategy='median'), make_model())
    pipe_full.fit(x, y)
    model = pipe_full.named_steps['randomforestregressor']
    x_imp = pd.DataFrame(pipe_full.named_steps['simpleimputer'].transform(x), columns=x.columns, index=x.index)
    return (None, model, x_imp, y, oof, fold_rows)

def summarize_model(table: str, outcome: str, outcome_role: str, group: str, group_type: str, n_complete_predictors: int, y: np.ndarray, oof: np.ndarray, fold_rows: list[dict[str, float]]) -> ModelResult:
    folds = pd.DataFrame(fold_rows)
    ok = np.isfinite(oof)
    return ModelResult(table=table, outcome=outcome, outcome_role=outcome_role, group=group, group_type=group_type, n=int(len(y)), n_complete_predictors=int(n_complete_predictors), r2_mean=float(folds['r2'].mean()), r2_sd=float(folds['r2'].std(ddof=0)), rmse_mean=float(folds['rmse'].mean()), rmse_sd=float(folds['rmse'].std(ddof=0)), mae_mean=float(folds['mae'].mean()), mae_sd=float(folds['mae'].std(ddof=0)), oof_r2=float(r2_score(y[ok], oof[ok])), oof_rmse=float(math.sqrt(mean_squared_error(y[ok], oof[ok]))), oof_mae=float(mean_absolute_error(y[ok], oof[ok])))

def spearman_rows(df: pd.DataFrame, predictors: pd.DataFrame, outcome: str, table: str, group: str, group_type: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    y_all = pd.to_numeric(df[outcome], errors='coerce')
    for _, rec in predictors.iterrows():
        col = rec['column']
        x_all = pd.to_numeric(df[col], errors='coerce')
        ok = y_all.notna() & x_all.notna()
        if ok.sum() < 100:
            continue
        rho, p_value = manual_spearman(x_all[ok], y_all[ok])
        rows.append({'table': table, 'outcome': outcome, 'group': group, 'group_type': group_type, 'column': col, 'family': rec['family'], 'family_label': rec['family_label'], 'n_pairwise': int(ok.sum()), 'spearman_rho': float(rho), 'p_value': float(p_value)})
    return rows

def manual_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    xr = [float(v) for v in x.rank(method='average').tolist()]
    yr = [float(v) for v in y.rank(method='average').tolist()]
    n = len(xr)
    if n <= 2:
        return (np.nan, np.nan)
    x_mean = sum(xr) / n
    y_mean = sum(yr) / n
    x_centered = [v - x_mean for v in xr]
    y_centered = [v - y_mean for v in yr]
    sx = sum((v * v for v in x_centered))
    sy = sum((v * v for v in y_centered))
    denom = math.sqrt(sx * sy)
    rho = sum((a * b for a, b in zip(x_centered, y_centered))) / denom if denom > 0 else np.nan
    if not math.isfinite(rho) or abs(rho) >= 1:
        return (rho, np.nan)
    z_stat = rho * math.sqrt(max(1, n - 1))
    p_value = math.erfc(abs(z_stat) / math.sqrt(2))
    return (rho, p_value)

def sample_for_explain(x: pd.DataFrame, y: np.ndarray, group_type: str) -> tuple[pd.DataFrame, np.ndarray]:
    cap = EXPLAIN_SAMPLE_GLOBAL if group_type == 'global' else EXPLAIN_SAMPLE_SUBGROUP
    if len(x) <= cap:
        return (x, y)
    rng = np.random.default_rng(RANDOM_STATE)
    pos = rng.choice(np.arange(len(x)), size=cap, replace=False)
    return (x.iloc[pos].copy(), y[pos])

def explain_model(model: RandomForestRegressor, x_imp: pd.DataFrame, y: np.ndarray, predictors: pd.DataFrame, table: str, outcome: str, outcome_role: str, group: str, group_type: str) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    x_explain, y_explain = sample_for_explain(x_imp, y, group_type)
    meta = predictors.set_index('column').to_dict('index')
    perm = permutation_importance(model, x_explain.to_numpy(), y_explain, n_repeats=PERM_REPEATS, random_state=RANDOM_STATE, n_jobs=-1, scoring='r2')
    perm_rows = []
    for col, mean, sd in zip(x_imp.columns, perm.importances_mean, perm.importances_std):
        rec = meta[col]
        perm_rows.append({'table': table, 'outcome': outcome, 'outcome_role': outcome_role, 'group': group, 'group_type': group_type, 'column': col, 'family': rec['family'], 'family_label': rec['family_label'], 'variable': rec['variable'], 'perm_importance_mean': float(mean), 'perm_importance_sd': float(sd)})
    perm_df = pd.DataFrame(perm_rows)
    family_rows = perm_df.groupby(['table', 'outcome', 'outcome_role', 'group', 'group_type', 'family', 'family_label'], dropna=False)['perm_importance_mean'].sum().reset_index().sort_values('perm_importance_mean', ascending=False).to_dict('records')
    shap_rows = shap_importance_rows(model, x_explain, predictors, table, outcome, outcome_role, group, group_type)
    top_features = perm_df.sort_values('perm_importance_mean', ascending=False)['column'].head(TOP_N_EFFECT_FEATURES).tolist()
    pdp_rows = pdp_effect_rows(model, x_explain, top_features, table, outcome, outcome_role, group, group_type)
    ale_rows = ale_effect_rows(model, x_explain, top_features, table, outcome, outcome_role, group, group_type)
    return (perm_rows, family_rows, shap_rows, pdp_rows, ale_rows)

def shap_importance_rows(model: RandomForestRegressor, x: pd.DataFrame, predictors: pd.DataFrame, table: str, outcome: str, outcome_role: str, group: str, group_type: str) -> list[dict[str, object]]:
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(x.to_numpy(), check_additivity=False)
        values = np.asarray(values)
        meta = predictors.set_index('column').to_dict('index')
        rows = []
        for i, col in enumerate(x.columns):
            rec = meta[col]
            rows.append({'table': table, 'outcome': outcome, 'outcome_role': outcome_role, 'group': group, 'group_type': group_type, 'column': col, 'family': rec['family'], 'family_label': rec['family_label'], 'variable': rec['variable'], 'mean_abs_shap': float(np.abs(values[:, i]).mean()), 'mean_shap': float(values[:, i].mean())})
        return rows
    except Exception as exc:
        error_path = RUN_DIR / f'q3_formal_shap_error_{safe_name(table)}_{safe_name(outcome)}_{safe_name(group)}.txt'
        error_path.write_text(str(exc), encoding='utf-8')
        return []

def pdp_effect_rows(model: RandomForestRegressor, x: pd.DataFrame, features: list[str], table: str, outcome: str, outcome_role: str, group: str, group_type: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for feat in features:
        try:
            idx = x.columns.get_loc(feat)
            result = partial_dependence(model, x.to_numpy(), [idx], grid_resolution=30, kind='average')
            grid = result.get('grid_values', result.get('values'))[0]
            avg = result['average'][0]
            for xv, yv in zip(grid, avg):
                rows.append({'table': table, 'outcome': outcome, 'outcome_role': outcome_role, 'group': group, 'group_type': group_type, 'column': feat, 'x': float(xv), 'partial_dependence': float(yv)})
        except Exception as exc:
            rows.append({'table': table, 'outcome': outcome, 'outcome_role': outcome_role, 'group': group, 'group_type': group_type, 'column': feat, 'x': np.nan, 'partial_dependence': np.nan, 'error': str(exc)})
    return rows

def ale_effect_rows(model: RandomForestRegressor, x: pd.DataFrame, features: list[str], table: str, outcome: str, outcome_role: str, group: str, group_type: str, bins: int=20) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for feature in features:
        s = x[feature].to_numpy(dtype=float)
        quantiles = np.unique(np.quantile(s[np.isfinite(s)], np.linspace(0, 1, bins + 1)))
        if len(quantiles) < 4:
            continue
        bin_ids = np.clip(np.searchsorted(quantiles, s, side='right') - 1, 0, len(quantiles) - 2)
        effects = []
        centers = []
        counts = []
        for b in range(len(quantiles) - 1):
            idx = np.where(bin_ids == b)[0]
            counts.append(int(len(idx)))
            centers.append(float((quantiles[b] + quantiles[b + 1]) / 2))
            if len(idx) == 0:
                effects.append(0.0)
                continue
            x_low = x.iloc[idx].copy()
            x_high = x.iloc[idx].copy()
            x_low.loc[:, feature] = quantiles[b]
            x_high.loc[:, feature] = quantiles[b + 1]
            effects.append(float(np.mean(model.predict(x_high.to_numpy()) - model.predict(x_low.to_numpy()))))
        ale = np.cumsum(effects)
        ale = ale - np.mean(ale)
        for i, center in enumerate(centers):
            rows.append({'table': table, 'outcome': outcome, 'outcome_role': outcome_role, 'group': group, 'group_type': group_type, 'column': feature, 'x': center, 'ale': float(ale[i]), 'n_bin': counts[i]})
    return rows

def run_table_models(df: pd.DataFrame, table_name: str, predictors: pd.DataFrame, primary_outcomes: list[str], secondary_outcomes: list[str] | None=None) -> tuple[list[ModelResult], dict[str, list[dict[str, object]]]]:
    selected = predictors['column'].tolist()
    missing_cols = [col for col in selected if col not in df.columns]
    if missing_cols:
        raise ValueError(f'{table_name} missing predictors: {missing_cols}')
    outputs: dict[str, list[dict[str, object]]] = {'folds': [], 'predictions': [], 'spearman': [], 'permutation': [], 'family': [], 'shap': [], 'pdp': [], 'ale': [], 'sample': []}
    results: list[ModelResult] = []
    secondary_outcomes = secondary_outcomes or []
    outcome_plan = [(col, 'primary') for col in available_outcomes(df, primary_outcomes)]
    outcome_plan.extend(((col, 'secondary') for col in available_outcomes(df, secondary_outcomes)))
    for outcome, outcome_role in outcome_plan:
        for group, group_type, mask in add_group_specs(df, table_name, outcome_role):
            sub = df.loc[mask].copy()
            y_series = pd.to_numeric(sub[outcome], errors='coerce')
            sub = sub.loc[y_series.notna()].copy()
            if len(sub) < MIN_MODEL_N:
                continue
            y = pd.to_numeric(sub[outcome], errors='coerce').to_numpy(dtype=float)
            x = coerce_design(sub, selected)
            n_complete = int(x.notna().all(axis=1).sum())
            print(f'[{table_name}] {outcome} / {group}: n={len(sub)}', flush=True)
            _, model, x_imp, y_fit, oof, fold_rows = cv_fit_predict(x, y)
            result = summarize_model(table_name, outcome, outcome_role, group, group_type, n_complete, y_fit, oof, fold_rows)
            results.append(result)
            for row in fold_rows:
                outputs['folds'].append({'table': table_name, 'outcome': outcome, 'outcome_role': outcome_role, 'group': group, 'group_type': group_type, **row})
            id_cols = [c for c in ['id_1', 'zone_code', 'zone_label', 'COUNTRY', 'NAME_1', 'NS', 'ns_group'] if c in sub.columns]
            pred_frame = sub[id_cols].copy()
            pred_frame['table'] = table_name
            pred_frame['outcome'] = outcome
            pred_frame['outcome_role'] = outcome_role
            pred_frame['group'] = group
            pred_frame['group_type'] = group_type
            pred_frame['observed'] = y_fit
            pred_frame['oof_prediction'] = oof
            pred_frame['oof_residual'] = y_fit - oof
            outputs['predictions'].extend(pred_frame.to_dict('records'))
            outputs['sample'].append({'table': table_name, 'outcome': outcome, 'outcome_role': outcome_role, 'group': group, 'group_type': group_type, 'n': int(len(sub)), 'n_complete_predictors': n_complete, 'n_any_predictor_missing': int(len(sub) - n_complete)})
            outputs['spearman'].extend(spearman_rows(sub, predictors, outcome, table_name, group, group_type))
            perm, family, shap_rows, pdp, ale = explain_model(model, x_imp, y_fit, predictors, table_name, outcome, outcome_role, group, group_type)
            outputs['permutation'].extend(perm)
            outputs['family'].extend(family)
            outputs['shap'].extend(shap_rows)
            outputs['pdp'].extend(pdp)
            outputs['ale'].extend(ale)
    return (results, outputs)

def write_outputs(results: list[ModelResult], outputs: dict[str, list[dict[str, object]]], predictors: pd.DataFrame, elapsed: float) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(result) for result in results]).to_csv(RUN_DIR / 'q3_formal_rf_cv_performance.csv', index=False, encoding='utf-8-sig')
    for name, rows in outputs.items():
        pd.DataFrame(rows).to_csv(RUN_DIR / f'q3_formal_{name}.csv', index=False, encoding='utf-8-sig')
    predictors.to_csv(RUN_DIR / 'q3_formal_predictors_used.csv', index=False, encoding='utf-8-sig')
    metadata = {'created_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'elapsed_seconds': elapsed, 'city_master': str(CITY_MASTER), 'city_zone_master': str(ZONE_MASTER), 'predictor_file': str(FINAL_PREDICTORS), 'n_predictors': int(len(predictors)), 'rf': {'n_estimators': N_ESTIMATORS, 'min_samples_leaf': MIN_SAMPLES_LEAF, 'max_features': MAX_FEATURES, 'n_splits': N_SPLITS, 'permutation_repeats': PERM_REPEATS}}
    (RUN_DIR / 'q3_formal_model_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    write_report(results, predictors, elapsed)

def write_report(results: list[ModelResult], predictors: pd.DataFrame, elapsed: float) -> None:
    perf = pd.DataFrame([asdict(result) for result in results])
    primary_perf = perf.query("outcome_role == 'primary'").sort_values(['table', 'outcome', 'group_type', 'group'])
    family = pd.read_csv(resource(RUN_DIR / 'q3_formal_family.csv'))
    top_family = family.sort_values('perm_importance_mean', ascending=False).groupby(['table', 'outcome', 'group'], as_index=False).head(5)
    lines = ['# Q3 formal mechanism models', '', f"- Created: {time.strftime('%Y-%m-%d %H:%M:%S')}", f'- Elapsed seconds: {elapsed:.1f}', f'- Final predictors used: {len(predictors)}', f'- Model: Random forest with median imputation, {N_SPLITS}-fold CV, permutation importance, SHAP importance, PDP and ALE.', '', '## Predictors', '', predictors[['selection_order', 'column', 'family_label', 'variable']].to_markdown(index=False), '', '## Primary model performance', '', primary_perf[['table', 'outcome', 'group', 'group_type', 'n', 'n_complete_predictors', 'r2_mean', 'rmse_mean', 'mae_mean']].to_markdown(index=False, floatfmt='.3f'), '', '## Top family-level mechanisms by permutation importance', '', top_family[['table', 'outcome', 'group', 'family_label', 'perm_importance_mean']].head(80).to_markdown(index=False, floatfmt='.4f'), '', '## Output files', '', '- `q3_formal_rf_cv_performance.csv`: model performance for all outcomes and groups.', '- `q3_formal_permutation.csv`: feature-level permutation importance.', '- `q3_formal_family.csv`: family-level importance aggregated from permutation importance.', '- `q3_formal_shap.csv`: SHAP feature importance where SHAP succeeds.', '- `q3_formal_pdp.csv`: PDP curves for top features of each model.', '- `q3_formal_ale.csv`: ALE curves for top features of each model.', '- `q3_formal_spearman.csv`: univariate Spearman associations.', '- `q3_formal_predictions.csv`: out-of-fold predictions and residuals.', '- `q3_formal_sample.csv`: model sample and missing-predictor counts.']
    (RUN_DIR / 'RUN.md').write_text('\n'.join(lines), encoding='utf-8')

def combine_outputs(parts: list[dict[str, list[dict[str, object]]]]) -> dict[str, list[dict[str, object]]]:
    combined: dict[str, list[dict[str, object]]] = {}
    for part in parts:
        for key, rows in part.items():
            combined.setdefault(key, []).extend(rows)
    return combined

def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    predictors = load_predictors()
    city = pd.read_csv(resource(CITY_MASTER))
    zone = pd.read_csv(resource(ZONE_MASTER))
    city_results, city_outputs = run_table_models(city, 'city', predictors, CITY_PRIMARY_OUTCOMES, secondary_outcomes=[])
    zone_results, zone_outputs = run_table_models(zone, 'city_zone', predictors, ZONE_PRIMARY_OUTCOMES, secondary_outcomes=ZONE_SECONDARY_OUTCOMES)
    results = [*city_results, *zone_results]
    outputs = combine_outputs([city_outputs, zone_outputs])
    write_outputs(results, outputs, predictors, time.time() - started)
    print(f'Done. Outputs written to {RUN_DIR}', flush=True)
