from pathlib import Path as _Path
import sys as _sys

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "_support"))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "figures"))

from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
ROOT = PACKAGE_ROOT
import Fig_5_response_uncertainty as bootstrap
FORMAL = ROOT / 'runs/20260818-q3-final-mechanism-models'
OUT = output_dir('Fig_5')

def main():
    spec = importlib.util.spec_from_file_location('formal_top8_reference', resource(FORMAL / 'run_q3_formal_mechanism_models.py'))
    formal = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = formal
    spec.loader.exec_module(formal)
    permutation = pd.read_csv(resource(OUT / 'permutation.csv'))
    city = pd.read_csv(resource(formal.CITY_MASTER))
    predictors = formal.load_predictors().column.tolist()
    original = {kind: pd.read_csv(resource(FORMAL / filename)) for kind, filename in {'ale': 'q3_formal_ale.csv', 'pdp': 'q3_formal_pdp.csv', 'uncertainty': 'q3_formal_response_uncertainty_bootstrap.csv'}.items()}
    additions = {kind: [] for kind in original}
    selection = {}
    for outcome in bootstrap.OUTCOMES:
        top = permutation.query("group == 'global' and outcome == @outcome").nlargest(8, 'perm_importance_mean').column.tolist()
        selection[outcome] = top
        missing = {kind: [col for col in top if col not in set(frame.query("table == 'city' and group == 'global' and outcome == @outcome").column)] for kind, frame in original.items()}
        sub = city.loc[pd.to_numeric(city[outcome], errors='coerce').notna()]
        x = formal.coerce_design(sub, predictors)
        y = sub[outcome].to_numpy(float)
        if missing['ale'] or missing['pdp']:
            imputer = SimpleImputer(strategy='median')
            x_imp = pd.DataFrame(imputer.fit_transform(x), columns=predictors, index=x.index)
            model = formal.make_model().set_params(n_jobs=4).fit(x_imp.to_numpy(), y)
            model.set_params(n_jobs=1)
            x_eval, _ = formal.sample_for_explain(x_imp, y, 'global')
            for kind in ['ale', 'pdp']:
                if missing[kind]:
                    rows = getattr(formal, f'{kind}_effect_rows')(model, x_eval, missing[kind], 'city', outcome, 'primary', 'global', 'global')
                    assert rows and all((np.isfinite(row['x']) for row in rows))
                    additions[kind].extend(rows)
        features = missing['uncertainty']
        if not features:
            continue
        x_sample, _ = bootstrap.fixed_explain_sample(x, y)
        baseline = pd.DataFrame(SimpleImputer(strategy='median').fit_transform(x_sample), columns=predictors)
        grids = {col: bootstrap.fixed_ale_grid(baseline, col) for col in features}
        curves = {col: [] for col in features}
        rng = np.random.default_rng(bootstrap.RANDOM_STATE + 1000)
        for boot in range(bootstrap.N_BOOT):
            indices = rng.integers(0, len(x), size=len(x))
            imputer = SimpleImputer(strategy='median')
            train_x = imputer.fit_transform(x.iloc[indices])
            model = bootstrap.make_model(bootstrap.RANDOM_STATE + 100 + boot).set_params(n_jobs=4)
            model.fit(train_x, y[indices])
            model.set_params(n_jobs=1)
            sample = pd.DataFrame(imputer.transform(x_sample), columns=predictors)
            for col in features:
                curves[col].append(bootstrap.ale_curve(model, sample, col, grids[col][0]))
            print(f'{outcome}: bootstrap {boot + 1}/{bootstrap.N_BOOT}', flush=True)
        for col in features:
            additions['uncertainty'].extend(bootstrap.summarize_feature(outcome, col, grids[col][2], curves[col]))
    for kind, rows in additions.items():
        frame = pd.DataFrame(rows, columns=original[kind].columns)
        frame.to_csv(OUT / f'response_{kind}_additions.csv', index=False)
    (OUT / 'response_top8_selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    print('Saved top-eight response additions without changing the original curve files.', flush=True)
