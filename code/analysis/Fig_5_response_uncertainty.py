from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "_support"))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "figures"))

from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
PROJECT_ROOT = PACKAGE_ROOT
DERIVED_DIR = PROJECT_ROOT / 'derived_data'
PREV_RUN_DIR = PROJECT_ROOT / 'runs' / '20260818-q3-mechanism-modeling'
RUN_FORMAL = PROJECT_ROOT / 'runs' / '20260818-q3-final-mechanism-models'
CITY_MASTER = DERIVED_DIR / 'q3_city_driver_master.csv'
FINAL_PREDICTORS = PREV_RUN_DIR / 'q3_final_predictors_recommended.csv'
PERMUTATION = RUN_FORMAL / 'q3_formal_permutation.csv'
OUT_CSV = RUN_FORMAL / 'q3_formal_response_uncertainty_bootstrap.csv'
OUT_REPORT = RUN_FORMAL / 'q3_formal_response_uncertainty_bootstrap_RUN.md'
OUTCOMES = ['eli50', 'compound_gini50']
GROUP = 'global'
TABLE = 'city'
N_TOP_FEATURES = 4
N_BOOT = 30
N_ESTIMATORS = 180
MIN_SAMPLES_LEAF = 5
MAX_FEATURES = 0.7
EXPLAIN_SAMPLE = 900
ALE_BINS = 20
RANDOM_STATE = 42

def load_predictors() -> list[str]:
    predictors = pd.read_csv(resource(FINAL_PREDICTORS)).sort_values('selection_order')
    return predictors['column'].tolist()

def top_response_features(outcome: str) -> list[str]:
    perm = pd.read_csv(resource(PERMUTATION))
    sub = perm[(perm['table'] == TABLE) & (perm['group'] == GROUP) & (perm['outcome'] == outcome)].copy()
    return sub.sort_values('perm_importance_mean', ascending=False)['column'].head(N_TOP_FEATURES).tolist()

def make_model(seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=N_ESTIMATORS, min_samples_leaf=MIN_SAMPLES_LEAF, max_features=MAX_FEATURES, random_state=seed, n_jobs=-1)

def fixed_explain_sample(x: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    if len(x) <= EXPLAIN_SAMPLE:
        return (x.copy(), y.copy())
    rng = np.random.default_rng(RANDOM_STATE)
    pos = rng.choice(np.arange(len(x)), size=EXPLAIN_SAMPLE, replace=False)
    return (x.iloc[pos].copy(), y[pos].copy())

def fixed_ale_grid(x_eval_imp: pd.DataFrame, feature: str, bins: int=ALE_BINS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = x_eval_imp[feature].to_numpy(dtype=float)
    quantiles = np.unique(np.quantile(values[np.isfinite(values)], np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 4:
        raise ValueError(f'Too few unique ALE quantiles for {feature}')
    centers = (quantiles[:-1] + quantiles[1:]) / 2
    denom = centers[-1] - centers[0]
    x_norm = np.zeros_like(centers) if np.isclose(denom, 0) else (centers - centers[0]) / denom
    return (quantiles, centers, x_norm)

def ale_curve(model: RandomForestRegressor, x_eval: pd.DataFrame, feature: str, quantiles: np.ndarray) -> np.ndarray:
    values = x_eval[feature].to_numpy(dtype=float)
    bin_ids = np.clip(np.searchsorted(quantiles, values, side='right') - 1, 0, len(quantiles) - 2)
    effects: list[float] = []
    for b in range(len(quantiles) - 1):
        idx = np.where(bin_ids == b)[0]
        if len(idx) == 0:
            effects.append(0.0)
            continue
        x_low = x_eval.iloc[idx].copy()
        x_high = x_eval.iloc[idx].copy()
        x_low.loc[:, feature] = quantiles[b]
        x_high.loc[:, feature] = quantiles[b + 1]
        effects.append(float(np.mean(model.predict(x_high.to_numpy()) - model.predict(x_low.to_numpy()))))
    curve = np.cumsum(np.asarray(effects, dtype=float))
    return curve - np.mean(curve)

def summarize_feature(outcome: str, feature: str, x_norm: np.ndarray, curves: list[np.ndarray]) -> list[dict[str, object]]:
    arr = np.vstack(curves)
    lo, hi = np.nanpercentile(arr, [2.5, 97.5], axis=0)
    mean = np.nanmean(arr, axis=0)
    return [{'table': TABLE, 'outcome': outcome, 'group': GROUP, 'column': feature, 'effect_type': 'ale', 'x_norm': float(xv), 'mean': float(mv), 'lo95': float(lv), 'hi95': float(hv), 'n_boot': int(len(curves)), 'method': 'city-level bootstrap random forest; 2.5-97.5 percentile envelope'} for xv, mv, lv, hv in zip(x_norm, mean, lo, hi)]

def main() -> None:
    started = time.time()
    predictors = load_predictors()
    city = pd.read_csv(resource(CITY_MASTER))
    rows: list[dict[str, object]] = []
    report_features: dict[str, list[str]] = {}
    for outcome in OUTCOMES:
        top_features = top_response_features(outcome)
        report_features[outcome] = top_features
        y_series = pd.to_numeric(city[outcome], errors='coerce')
        ok_y = y_series.notna()
        sub = city.loc[ok_y].copy()
        y = y_series.loc[ok_y].to_numpy(dtype=float)
        x = sub[predictors].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
        x_explain_raw, _ = fixed_explain_sample(x, y)
        baseline_imp = SimpleImputer(strategy='median')
        baseline_x_eval = pd.DataFrame(baseline_imp.fit_transform(x_explain_raw), columns=x.columns)
        grids = {feature: fixed_ale_grid(baseline_x_eval, feature) for feature in top_features}
        curves = {feature: [] for feature in top_features}
        rng = np.random.default_rng(RANDOM_STATE + 1000)
        n = len(x)
        for boot in range(N_BOOT):
            seed = RANDOM_STATE + 100 + boot
            boot_idx = rng.integers(0, n, size=n)
            pipe = make_pipeline(SimpleImputer(strategy='median'), make_model(seed))
            pipe.fit(x.iloc[boot_idx], y[boot_idx])
            model = pipe.named_steps['randomforestregressor']
            imputer = pipe.named_steps['simpleimputer']
            x_eval_imp = pd.DataFrame(imputer.transform(x_explain_raw), columns=x.columns)
            for feature in top_features:
                quantiles, _, _ = grids[feature]
                curves[feature].append(ale_curve(model, x_eval_imp, feature, quantiles))
            print(f'{outcome} bootstrap {boot + 1}/{N_BOOT}', flush=True)
        for feature in top_features:
            _, _, x_norm = grids[feature]
            rows.extend(summarize_feature(outcome, feature, x_norm, curves[feature]))
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    elapsed = time.time() - started
    report = ['# Q3 ELI response uncertainty bootstrap', '', f"- Created: {time.strftime('%Y-%m-%d %H:%M:%S')}", f'- Elapsed seconds: {elapsed:.1f}', f"- Outcomes: {', '.join(OUTCOMES)}", f'- Group: {GROUP}', f'- Bootstrap repetitions: {N_BOOT}', f'- Random forest estimators per bootstrap: {N_ESTIMATORS}', f'- Explain sample size: {len(x_explain_raw)}', '- Features:', *[f"  - {outcome}: {', '.join(features)}" for outcome, features in report_features.items()], f'- Output: `{OUT_CSV.name}`', '', 'The interval is a city-level bootstrap percentile envelope for ALE curves.']
    OUT_REPORT.write_text('\n'.join(report), encoding='utf-8')
    print(OUT_CSV)
