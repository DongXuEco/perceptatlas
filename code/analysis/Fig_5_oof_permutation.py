from pathlib import Path as _Path
import sys as _sys

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "_support"))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "figures"))

from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import numpy as np
import pandas as pd
import sklearn
from joblib import parallel_backend
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
ROOT = PACKAGE_ROOT
FORMAL = ROOT / 'runs/20260818-q3-final-mechanism-models'
OUT = output_dir('Fig_5')

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = FORMAL / 'run_q3_formal_mechanism_models.py'
    spec = importlib.util.spec_from_file_location('formal_city_oof_reference', resource(source))
    formal = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = formal
    spec.loader.exec_module(formal)
    predictors = formal.load_predictors()
    columns = predictors['column'].tolist()
    city = pd.read_csv(resource(formal.CITY_MASTER))
    reference = pd.read_csv(resource(FORMAL / 'q3_formal_rf_cv_performance.csv'))
    reference_predictions = pd.read_csv(resource(FORMAL / 'q3_formal_predictions.csv'))
    records, fold_records, prediction_records, checks = ([], [], [], [])
    for outcome in formal.CITY_PRIMARY_OUTCOMES:
        for group, group_type, mask in formal.add_group_specs(city, 'city'):
            sub = city.loc[mask].copy()
            sub = sub.loc[pd.to_numeric(sub[outcome], errors='coerce').notna()]
            x = formal.coerce_design(sub, columns)
            y = pd.to_numeric(sub[outcome]).to_numpy(float)
            sst = float(np.sum((y - y.mean()) ** 2))
            oof = np.full(len(y), np.nan)
            repeat_drops = np.zeros((len(columns), formal.PERM_REPEATS))
            splits = min(formal.N_SPLITS, max(2, len(y) // formal.MIN_MODEL_N))
            cv = KFold(splits, shuffle=True, random_state=formal.RANDOM_STATE)
            print(f'{outcome} / {group}: {len(y)} cities', flush=True)
            for fold, (train, test) in enumerate(cv.split(x), 1):
                assert not np.intersect1d(train, test).size
                imputer = SimpleImputer(strategy='median')
                train_x = imputer.fit_transform(x.iloc[train])
                test_x = imputer.transform(x.iloc[test])
                assert test_x.shape[1] == len(columns)
                model = formal.make_model().set_params(n_jobs=4)
                model.fit(train_x, y[train])
                model.set_params(n_jobs=1)
                oof[test] = model.predict(test_x)
                with parallel_backend('threading', n_jobs=4):
                    perm = permutation_importance(model, test_x, y[test], scoring='r2', n_repeats=formal.PERM_REPEATS, random_state=formal.RANDOM_STATE + fold, n_jobs=4)
                fold_sst = float(np.sum((y[test] - y[test].mean()) ** 2))
                repeat_drops += perm.importances * fold_sst / sst
                fold_records.append({'outcome': outcome, 'group': group, 'fold': fold, 'n_train': len(train), 'n_test': len(test), 'r2': r2_score(y[test], oof[test]), 'train_test_overlap': 0})
                print(f'  fold {fold}/{splits}: held-out permutations complete', flush=True)
            baseline = r2_score(y, oof)
            ref = reference[(reference['table'] == 'city') & (reference['outcome'] == outcome) & (reference['group'] == group)].iloc[0]
            expected = reference_predictions[(reference_predictions['table'] == 'city') & (reference_predictions['outcome'] == outcome) & (reference_predictions['group'] == group)].set_index('id_1').loc[sub['id_1']]
            assert np.allclose(y, expected['observed'], atol=1e-12)
            assert abs(baseline - ref['oof_r2']) < 0.01
            checks.append({'outcome': outcome, 'group': group, 'n': len(y), 'oof_r2': baseline, 'historical_oof_r2': float(ref['oof_r2']), 'historical_prediction_max_abs_difference': float(np.max(np.abs(oof - expected['oof_prediction'].to_numpy()))), 'matches_formal_oof': bool(np.allclose(oof, expected['oof_prediction'], atol=1e-10, rtol=1e-08))})
            for position, (_, predictor) in enumerate(predictors.iterrows()):
                records.append({'table': 'city', 'outcome': outcome, 'outcome_role': 'primary', 'group': group, 'group_type': group_type, 'column': predictor['column'], 'family': predictor['family'], 'family_label': predictor['family_label'], 'variable': predictor['variable'], 'perm_importance_mean': repeat_drops[position].mean(), 'perm_importance_sd': repeat_drops[position].std(ddof=0), **{f'repeat_{i + 1}': value for i, value in enumerate(repeat_drops[position])}})
            prediction_records.extend(({'outcome': outcome, 'group': group, 'id_1': city_id, 'observed': obs, 'oof_prediction': pred} for city_id, obs, pred in zip(sub['id_1'], y, oof)))
    perm_df = pd.DataFrame(records)
    family_keys = ['table', 'outcome', 'outcome_role', 'group', 'group_type', 'family', 'family_label']
    family = perm_df.assign(perm_importance_mean=perm_df['perm_importance_mean'].clip(lower=0))
    family = family.groupby(family_keys, as_index=False)['perm_importance_mean'].sum()
    perm_df.to_csv(OUT / 'permutation.csv', index=False)
    family.to_csv(OUT / 'family.csv', index=False)
    pd.DataFrame(fold_records).to_csv(OUT / 'folds.csv', index=False)
    pd.DataFrame(prediction_records).to_csv(OUT / 'predictions.csv', index=False)
    inputs = [Path(formal.CITY_MASTER), Path(formal.FINAL_PREDICTORS), source, Path(__file__)]
    audit = {'method': 'pooled OOF R2 decrease; independently permute each feature within each held-out fold', 'n_repeats': formal.PERM_REPEATS, 'folds': formal.N_SPLITS, 'sd': 'SD over 8 pooled OOF permutation repetitions; not a confidence interval', 'family_weights': 'sum of nonnegative variable mean importance; no joint-family permutation', 'validation': 'random city KFold, not spatial or country-held-out validation', 'sklearn_version': sklearn.__version__, 'historical_reproduction': 'same source, row order, outcomes and RF parameters; predictions not necessarily bit-identical', 'inputs_sha256': {str(p): hashlib.sha256(resource(p).read_bytes()).hexdigest() for p in inputs}, 'models': checks}
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(f'Saved {len(records)} feature scores and historical reproduction diagnostics.', flush=True)
