from pathlib import Path as _Path
import sys as _sys

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "_support"))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "figures"))

from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
from pathlib import Path
import hashlib
import json
import shutil
import sys
ROOT = PACKAGE_ROOT
import Extended_Data_Fig_2_quantile_curves as base
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np
import pandas as pd
from PIL import Image, ImageChops, ImageOps
from scipy.ndimage import gaussian_filter1d
OUT = output_dir('Extended_Data_Fig_2')
OUT.mkdir(parents=True, exist_ok=True)
Q = np.linspace(0.05, 0.95, 181)
N_BOOT, SEED, BATCH = (5000, 20261023, 100)
GROUPS = ('South', 'North')
ARCHIVE = OUT / 'bootstrap_quantiles.npz'
DATA = base.design.INPUT / 'analysis_city_data.csv'

def sha(path):
    return hashlib.sha256(resource(path).read_bytes()).hexdigest()

def compute():
    streams = np.random.SeedSequence(SEED).spawn(2)
    arrays, rows = ({}, [])
    for group, stream in zip(GROUPS, streams):
        frame = base.design.data.loc[base.design.data.ns_group == group]
        columns = [metric for metric, _, _ in base.design.PANELS]
        matrix = frame[columns].to_numpy()
        n = len(matrix)
        boot = np.empty((N_BOOT, len(columns), len(Q)), dtype=np.float64)
        rng = np.random.default_rng(stream)
        for start in range(0, N_BOOT, BATCH):
            size = min(BATCH, N_BOOT - start)
            indices = rng.integers(0, n, size=(size, n))
            for j in range(len(columns)):
                boot[start:start + size, j] = np.quantile(matrix[:, j][indices], Q, axis=1, method='linear').T
        arrays[group] = boot
        bounds = np.quantile(boot, [0.025, 0.25, 0.75, 0.975], axis=0, method='linear')
        assert np.isfinite(boot).all() and np.all(np.diff(bounds, axis=0) >= 0)
        assert np.all(np.diff(bounds, axis=2) >= -1e-14)
        for j, (metric, label, _) in enumerate(base.design.PANELS):
            estimate = np.quantile(matrix[:, j], Q, method='linear')
            for k, q in enumerate(Q):
                rows.append(dict(group=group, metric=metric, label=label, n=n, quantile=q, estimate=estimate[k], ci95_low=bounds[0, j, k], ci50_low=bounds[1, j, k], ci50_high=bounds[2, j, k], ci95_high=bounds[3, j, k]))
        print(f'Computed {group}: {n} cities, {N_BOOT} resamples, nine outcomes', flush=True)
    np.savez_compressed(ARCHIVE, quantiles=Q, **arrays)
    pd.DataFrame(rows).to_csv(OUT / 'quantile_confidence_intervals.csv', index=False, encoding='utf-8-sig')
    metadata = dict(n_boot=N_BOOT, seed=SEED, batch=BATCH, resampling_unit='Independent cities within North and South', dependence='Same indices across outcomes; no country/spatial cluster adjustment', method='Pointwise two-sided percentile bootstrap, 2.5th and 97.5th percentiles', simultaneous_band=False, quantile_method='NumPy linear', data_sha256=sha(DATA))
    (OUT / 'bootstrap_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
