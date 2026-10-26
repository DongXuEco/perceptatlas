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
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
PROJECT_ROOT = PACKAGE_ROOT
DERIVED_DIR = PROJECT_ROOT / 'derived_data'
SCREEN_DIR = PROJECT_ROOT / 'runs' / '20260818-q3-mechanism-modeling'
FORMAL_DIR = PROJECT_ROOT / 'runs' / '20260818-q3-final-mechanism-models'
RUN_DIR = PROJECT_ROOT / 'runs' / '20260818-q3-robustness-analysis'
CITY_MASTER = DERIVED_DIR / 'q3_city_driver_master.csv'
ZONE_MASTER = DERIVED_DIR / 'q3_city_zone_driver_master.csv'
FINAL_PREDICTORS = SCREEN_DIR / 'q3_final_predictors_recommended.csv'
RESERVED_PREDICTORS = SCREEN_DIR / 'q3_sensitivity_predictors_reserved.csv'
FORMAL_PERFORMANCE = FORMAL_DIR / 'q3_formal_rf_cv_performance.csv'
FORMAL_FAMILY = FORMAL_DIR / 'q3_formal_family.csv'
CITY_OUTCOMES = ['eli50', 'compound_gini50', 'compound_theil50', 'compound_p9010_50']
ZONE_OUTCOMES = ['eli50', 'positive_score50', 'negative_score50']
N_ESTIMATORS = 120
MIN_SAMPLES_LEAF = 5
MAX_FEATURES = 0.7
N_SPLITS = 3
PERM_REPEATS = 5
EXPLAIN_SAMPLE = 700
MIN_MODEL_N = 450
RANDOM_STATE = 42
OUTPUT_FILES = {'performance': RUN_DIR / 'q3_robustness_rf_cv_performance.csv', 'folds': RUN_DIR / 'q3_robustness_folds.csv', 'permutation': RUN_DIR / 'q3_robustness_permutation.csv', 'family': RUN_DIR / 'q3_robustness_family.csv', 'sample': RUN_DIR / 'q3_robustness_sample.csv', 'config': RUN_DIR / 'q3_robustness_config_predictors.csv'}

@dataclass
class ModelPerformance:
    robustness_config: str
    table: str
    outcome: str
    group: str
    group_type: str
    n: int
    n_complete_predictors: int
    n_predictors: int
    imputation: str
    r2_mean: float
    r2_sd: float
    rmse_mean: float
    rmse_sd: float
    mae_mean: float
    mae_sd: float

def append_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not resource(path).exists()
    df.to_csv(path, mode='a', index=False, header=header, encoding='utf-8-sig')

def reset_outputs() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    for path in OUTPUT_FILES.values():
        if resource(path).exists():
            path.unlink()
    for extra in ['q3_robustness_summary.csv', 'q3_robustness_family_top3_stability.csv', 'RUN.md']:
        path = RUN_DIR / extra
        if resource(path).exists():
            path.unlink()

def load_metadata() -> pd.DataFrame:
    final = pd.read_csv(resource(FINAL_PREDICTORS)).sort_values('selection_order')
    reserved = pd.read_csv(resource(RESERVED_PREDICTORS))
    combined = pd.concat([final, reserved], ignore_index=True)
    combined = combined.drop_duplicates('column', keep='first')
    return combined

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    city = pd.read_csv(resource(CITY_MASTER))
    zone = pd.read_csv(resource(ZONE_MASTER))
    final = pd.read_csv(resource(FINAL_PREDICTORS)).sort_values('selection_order')
    meta = load_metadata()
    return (city, zone, final, meta)

def build_configs(final: pd.DataFrame, meta: pd.DataFrame) -> list[dict[str, object]]:
    base = final['column'].tolist()
    configs = [{'name': 'main_imputed_rerun', 'description': 'Final 25 predictors with median imputation, rerun under the robustness RF settings.', 'predictors': base, 'complete_case': False}, {'name': 'main_complete_case', 'description': 'Final 25 predictors restricted to complete predictor rows.', 'predictors': base, 'complete_case': True}, {'name': 'poi_total_replacement', 'description': 'Replace four POI-category density variables with total POI density.', 'predictors': replace_many(base, remove=['poi_education_mean', 'poi_entertainment_mean', 'poi_health_mean', 'poi_transport_mean'], add=['poi_total_mean']), 'complete_case': False}, {'name': 'motorized_access_replacement', 'description': 'Replace non-motorized accessibility variables with motorized accessibility variables.', 'predictors': replace_many(base, remove=['access_nonmotor_education_mean', 'access_nonmotor_entertainment_mean', 'access_nonmotor_health_mean', 'access_nonmotor_living_mean', 'access_nonmotor_transport_mean'], add=['access_motor_education_mean', 'access_motor_entertainment_mean', 'access_motor_health_mean', 'access_motor_living_mean', 'access_motor_transport_mean']), 'complete_case': False}, {'name': 'drop_sparse_poi_road', 'description': 'Drop lower-coverage POI and road-density predictors to test sensitivity to sparse public-service layers.', 'predictors': [c for c in base if not c.startswith('poi_') and (not c.startswith('road_'))], 'complete_case': False}, {'name': 'impervious_replacement', 'description': 'Replace green-space fraction with impervious-surface fraction.', 'predictors': replace_many(base, remove=['green_fraction_mean'], add=['impervious_fraction_mean']), 'complete_case': False}]
    meta_cols = meta.set_index('column')
    config_rows = []
    for config in configs:
        for order, col in enumerate(config['predictors'], start=1):
            rec = meta_cols.loc[col].to_dict() if col in meta_cols.index else {}
            config_rows.append({'robustness_config': config['name'], 'description': config['description'], 'complete_case': config['complete_case'], 'order': order, 'column': col, 'family': rec.get('family', ''), 'family_label': rec.get('family_label', ''), 'variable': rec.get('variable', col)})
    append_rows(OUTPUT_FILES['config'], config_rows)
    return configs

def replace_many(base: list[str], remove: list[str], add: list[str]) -> list[str]:
    out = [col for col in base if col not in set(remove)]
    for col in add:
        if col not in out:
            out.append(col)
    return out

def group_specs(df: pd.DataFrame, table: str) -> list[tuple[str, str, pd.Series]]:
    specs: list[tuple[str, str, pd.Series]] = [('global', 'global', pd.Series(True, index=df.index))]
    if 'ns_group' in df.columns:
        for group in ['North', 'South']:
            specs.append((group, 'north_south', df['ns_group'].astype(str).eq(group)))
    if table == 'city_zone' and 'zone_label' in df.columns:
        for zone in ['core', 'expansion']:
            specs.append((zone, 'core_expansion', df['zone_label'].astype(str).eq(zone)))
    return specs

def make_model() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=N_ESTIMATORS, min_samples_leaf=MIN_SAMPLES_LEAF, max_features=MAX_FEATURES, random_state=RANDOM_STATE, n_jobs=-1)

def coerce_design(df: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    return df[predictors].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)

def fit_cv(x: pd.DataFrame, y: np.ndarray, complete_case: bool) -> tuple[list[dict[str, object]], RandomForestRegressor, pd.DataFrame]:
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    folds: list[dict[str, object]] = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(x), start=1):
        pipe = make_pipeline(SimpleImputer(strategy='median'), make_model())
        pipe.fit(x.iloc[train_idx], y[train_idx])
        pred = pipe.predict(x.iloc[test_idx])
        folds.append({'fold': fold, 'n_train': int(len(train_idx)), 'n_test': int(len(test_idx)), 'r2': float(r2_score(y[test_idx], pred)), 'rmse': float(math.sqrt(mean_squared_error(y[test_idx], pred))), 'mae': float(mean_absolute_error(y[test_idx], pred))})
    pipe_full = make_pipeline(SimpleImputer(strategy='median'), make_model())
    pipe_full.fit(x, y)
    model = pipe_full.named_steps['randomforestregressor']
    x_imp = pd.DataFrame(pipe_full.named_steps['simpleimputer'].transform(x), columns=x.columns, index=x.index)
    return (folds, model, x_imp)

def permutation_rows(model: RandomForestRegressor, x: pd.DataFrame, y: np.ndarray, config: str, table: str, outcome: str, group: str, group_type: str, meta: pd.DataFrame) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = np.random.default_rng(RANDOM_STATE)
    if len(x) > EXPLAIN_SAMPLE:
        pos = rng.choice(np.arange(len(x)), size=EXPLAIN_SAMPLE, replace=False)
        x_eval = x.iloc[pos]
        y_eval = y[pos]
    else:
        x_eval = x
        y_eval = y
    perm = permutation_importance(model, x_eval.to_numpy(), y_eval, n_repeats=PERM_REPEATS, random_state=RANDOM_STATE, n_jobs=-1, scoring='r2')
    meta_by_col = meta.set_index('column').to_dict('index')
    rows = []
    for col, mean, sd in zip(x.columns, perm.importances_mean, perm.importances_std):
        rec = meta_by_col.get(col, {})
        rows.append({'robustness_config': config, 'table': table, 'outcome': outcome, 'group': group, 'group_type': group_type, 'column': col, 'family': rec.get('family', ''), 'family_label': rec.get('family_label', ''), 'variable': rec.get('variable', col), 'perm_importance_mean': float(mean), 'perm_importance_sd': float(sd)})
    family = pd.DataFrame(rows).groupby(['robustness_config', 'table', 'outcome', 'group', 'group_type', 'family', 'family_label'], dropna=False)['perm_importance_mean'].sum().reset_index().sort_values('perm_importance_mean', ascending=False).to_dict('records')
    return (rows, family)

def run_one_dataset(df: pd.DataFrame, table: str, outcomes: list[str], configs: list[dict[str, object]], meta: pd.DataFrame) -> None:
    for config in configs:
        config_name = str(config['name'])
        predictors = list(config['predictors'])
        complete_case = bool(config['complete_case'])
        missing_predictors = [col for col in predictors if col not in df.columns]
        if missing_predictors:
            raise ValueError(f'{table}/{config_name} missing predictors: {missing_predictors}')
        for outcome in outcomes:
            if outcome not in df.columns:
                continue
            for group, group_type, mask in group_specs(df, table):
                sub = df.loc[mask].copy()
                sub[outcome] = pd.to_numeric(sub[outcome], errors='coerce')
                sub = sub.loc[sub[outcome].notna()].copy()
                x = coerce_design(sub, predictors)
                if complete_case:
                    complete = x.notna().all(axis=1)
                    sub = sub.loc[complete].copy()
                    x = x.loc[complete].copy()
                if len(sub) < MIN_MODEL_N:
                    continue
                y = pd.to_numeric(sub[outcome], errors='coerce').to_numpy(dtype=float)
                n_complete = int(x.notna().all(axis=1).sum())
                print(f'[{config_name}] {table} {outcome} / {group}: n={len(sub)}', flush=True)
                folds, model, x_imp = fit_cv(x, y, complete_case)
                fold_df = pd.DataFrame(folds)
                performance = ModelPerformance(robustness_config=config_name, table=table, outcome=outcome, group=group, group_type=group_type, n=int(len(sub)), n_complete_predictors=n_complete, n_predictors=int(len(predictors)), imputation='none_complete_case' if complete_case else 'median', r2_mean=float(fold_df['r2'].mean()), r2_sd=float(fold_df['r2'].std(ddof=0)), rmse_mean=float(fold_df['rmse'].mean()), rmse_sd=float(fold_df['rmse'].std(ddof=0)), mae_mean=float(fold_df['mae'].mean()), mae_sd=float(fold_df['mae'].std(ddof=0)))
                append_rows(OUTPUT_FILES['performance'], [asdict(performance)])
                append_rows(OUTPUT_FILES['folds'], [{'robustness_config': config_name, 'table': table, 'outcome': outcome, 'group': group, 'group_type': group_type, **row} for row in folds])
                append_rows(OUTPUT_FILES['sample'], [{'robustness_config': config_name, 'table': table, 'outcome': outcome, 'group': group, 'group_type': group_type, 'n': int(len(sub)), 'n_complete_predictors': n_complete, 'n_predictors': int(len(predictors)), 'complete_case': complete_case}])
                perm, family = permutation_rows(model, x_imp, y, config_name, table, outcome, group, group_type, meta)
                append_rows(OUTPUT_FILES['permutation'], perm)
                append_rows(OUTPUT_FILES['family'], family)

def build_stability_outputs() -> None:
    perf = pd.read_csv(resource(OUTPUT_FILES['performance']))
    family = pd.read_csv(resource(OUTPUT_FILES['family']))
    formal_perf = pd.read_csv(resource(FORMAL_PERFORMANCE))
    formal_family = pd.read_csv(resource(FORMAL_FAMILY))
    key_cols = ['table', 'outcome', 'group', 'group_type']
    main_perf = formal_perf[key_cols + ['r2_mean']].rename(columns={'r2_mean': 'formal_r2_mean'})
    summary = perf.merge(main_perf, on=key_cols, how='left')
    summary['delta_r2_vs_formal'] = summary['r2_mean'] - summary['formal_r2_mean']
    summary.to_csv(RUN_DIR / 'q3_robustness_summary.csv', index=False, encoding='utf-8-sig')
    formal_top = top_family_sets(formal_family, 'formal')
    robust_top = top_family_sets(family, 'robust')
    stability = robust_top.merge(formal_top, on=key_cols, how='left')
    stability['top1_same_as_formal'] = stability['top1_family_label'] == stability['formal_top1_family_label']
    stability['top3_overlap_n'] = [len(set(str(a).split(' | ')) & set(str(b).split(' | '))) for a, b in zip(stability['top3_family_labels'], stability['formal_top3_family_labels'])]
    stability.to_csv(RUN_DIR / 'q3_robustness_family_top3_stability.csv', index=False, encoding='utf-8-sig')
    write_report(summary, stability, family)

def top_family_sets(family: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    group_keys = ['table', 'outcome', 'group', 'group_type']
    if label == 'robust':
        group_keys = ['robustness_config', *group_keys]
    for keys, sub in family.groupby(group_keys, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        sub = sub.sort_values('perm_importance_mean', ascending=False)
        top3 = sub['family_label'].head(3).astype(str).tolist()
        row = dict(zip(group_keys, keys))
        row[f'{label}_top1_family_label' if label == 'formal' else 'top1_family_label'] = top3[0] if top3 else ''
        row[f'{label}_top3_family_labels' if label == 'formal' else 'top3_family_labels'] = ' | '.join(top3)
        rows.append(row)
    return pd.DataFrame(rows)

def write_report(summary: pd.DataFrame, stability: pd.DataFrame, family: pd.DataFrame) -> None:
    primary = summary.sort_values(['table', 'outcome', 'group', 'robustness_config'])
    perf_overview = primary.groupby('robustness_config', dropna=False).agg(n_models=('r2_mean', 'size'), mean_r2=('r2_mean', 'mean'), median_r2=('r2_mean', 'median'), mean_delta_r2_vs_formal=('delta_r2_vs_formal', 'mean'), min_delta_r2_vs_formal=('delta_r2_vs_formal', 'min')).reset_index()
    stable_overview = stability.groupby('robustness_config', dropna=False).agg(n_compared=('top3_overlap_n', 'size'), top1_same_rate=('top1_same_as_formal', 'mean'), mean_top3_overlap=('top3_overlap_n', 'mean')).reset_index()
    top_family = family.sort_values('perm_importance_mean', ascending=False).groupby(['robustness_config', 'table', 'outcome', 'group'], dropna=False).head(3)
    lines = ['# Q3 robustness analysis', '', f"- Created: {time.strftime('%Y-%m-%d %H:%M:%S')}", f'- RF settings: {N_ESTIMATORS} trees, min_samples_leaf={MIN_SAMPLES_LEAF}, max_features={MAX_FEATURES}, {N_SPLITS}-fold CV.', f"- Robustness configurations: {', '.join(sorted(summary['robustness_config'].unique()))}", '', '## Performance robustness overview', '', perf_overview.to_markdown(index=False, floatfmt='.3f'), '', '## Family-ranking stability relative to formal model', '', stable_overview.to_markdown(index=False, floatfmt='.3f'), '', '## Largest performance drops', '', summary.sort_values('delta_r2_vs_formal').head(30)[['robustness_config', 'table', 'outcome', 'group', 'n', 'n_predictors', 'r2_mean', 'formal_r2_mean', 'delta_r2_vs_formal']].to_markdown(index=False, floatfmt='.3f'), '', '## Top family mechanisms under robustness configurations', '', top_family.head(120)[['robustness_config', 'table', 'outcome', 'group', 'family_label', 'perm_importance_mean']].to_markdown(index=False, floatfmt='.4f'), '', '## Output files', '', '- `q3_robustness_rf_cv_performance.csv`: RF cross-validation performance for each robustness setting.', '- `q3_robustness_family.csv`: family-level permutation importance.', '- `q3_robustness_permutation.csv`: feature-level permutation importance.', '- `q3_robustness_summary.csv`: performance comparison against the formal model.', '- `q3_robustness_family_top3_stability.csv`: top-family overlap relative to formal mechanisms.', '- `q3_robustness_config_predictors.csv`: variable composition of each robustness setting.']
    (RUN_DIR / 'RUN.md').write_text('\n'.join(lines), encoding='utf-8')

def main() -> None:
    started = time.time()
    reset_outputs()
    city, zone, final, meta = load_data()
    configs = build_configs(final, meta)
    run_one_dataset(city, 'city', CITY_OUTCOMES, configs, meta)
    run_one_dataset(zone, 'city_zone', ZONE_OUTCOMES, configs, meta)
    build_stability_outputs()
    metadata = {'created_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'elapsed_seconds': time.time() - started, 'city_master': str(CITY_MASTER), 'zone_master': str(ZONE_MASTER), 'formal_performance': str(FORMAL_PERFORMANCE), 'formal_family': str(FORMAL_FAMILY), 'n_estimators': N_ESTIMATORS, 'n_splits': N_SPLITS, 'permutation_repeats': PERM_REPEATS}
    (RUN_DIR / 'q3_robustness_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Done. Outputs written to {RUN_DIR}', flush=True)
