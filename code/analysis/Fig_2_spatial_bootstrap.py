from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "_support"))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "figures"))

from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyproj
import scipy
from scipy.stats import false_discovery_control, spearmanr
from Fig_2 import DRIVERS, ROOT, load_data, make_groups
SEED = 20260916
SPECS = [(500, 0.0), (250, 0.0), (1000, 0.0), (500, 0.5), (250, 0.5), (1000, 0.5)]

class WeightedSpearman:

    def __init__(self, x, y):
        self.orders = []
        for values in (x, y):
            values = np.asarray(values)
            order = np.argsort(values, kind='stable')
            starts = np.r_[0, np.flatnonzero(np.diff(values[order]) != 0) + 1]
            groups_sorted = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(values)]))
            groups = np.empty(len(values), dtype=int)
            groups[order] = groups_sorted
            self.orders.append((order, starts, groups))

    def __call__(self, weights):
        weights = np.atleast_2d(weights).astype(float)
        ranks = []
        for order, starts, groups in self.orders:
            tie_weights = np.add.reduceat(weights[:, order], starts, axis=1)
            midranks = np.cumsum(tie_weights, axis=1) - 0.5 * tie_weights + 0.5
            ranks.append(midranks[:, groups])
        total = weights.sum(axis=1, keepdims=True)
        centered = [r - (weights * r).sum(axis=1, keepdims=True) / total for r in ranks]
        cross = (weights * centered[0] * centered[1]).sum(axis=1)
        variance = [(weights * r * r).sum(axis=1) for r in centered]
        with np.errstate(divide='ignore', invalid='ignore'):
            return cross / np.sqrt(variance[0] * variance[1])

def block_ids(lon, lat, size_km, offset):
    transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:6933', always_xy=True)
    x, y = transformer.transform(np.asarray(lon), np.asarray(lat))
    cells = np.floor(np.column_stack([x, y]) / (size_km * 1000) - offset).astype(np.int64)
    unique, inverse = np.unique(cells, axis=0, return_inverse=True)
    return (inverse, unique)

def bootstrap_rho(x, y, blocks, repetitions, seed, batch_size=32):
    _, inverse = np.unique(blocks, return_inverse=True)
    n_blocks = int(inverse.max() + 1)
    if n_blocks < 20:
        raise ValueError('Fewer than 20 occupied blocks; inference needs a separate design review.')
    rng = np.random.default_rng(seed)
    statistic = WeightedSpearman(x, y)
    results = np.empty(repetitions)
    sizes = np.empty(repetitions, dtype=np.int64)
    probabilities = np.full(n_blocks, 1 / n_blocks)
    for start in range(0, repetitions, batch_size):
        stop = min(start + batch_size, repetitions)
        counts = rng.multinomial(n_blocks, probabilities, size=stop - start)
        weights = counts[:, inverse]
        results[start:stop] = statistic(weights)
        sizes[start:stop] = weights.sum(axis=1)
    if not np.isfinite(results).all() or np.any(np.abs(results) >= 1):
        raise ValueError('Degenerate bootstrap statistic; do not silently drop failed replicates.')
    return (results, sizes)

def bootstrap_inference(rho, replicates):
    observed = np.arctanh(rho)
    z_star = np.arctanh(replicates)
    errors = z_star - observed
    lower, upper = np.quantile(errors, [0.025, 0.975])
    ci = np.tanh([observed - upper, observed - lower])
    lower_count = int(np.count_nonzero(errors <= observed))
    upper_count = int(np.count_nonzero(errors >= observed))
    p = min(1.0, 2 * (1 + min(lower_count, upper_count)) / (len(errors) + 1))
    return {'ci95_low': ci[0], 'ci95_high': ci[1], 'p_boot_approx': p, 'tail_lower_count': lower_count, 'tail_upper_count': upper_count, 'bootstrap_rho_mean': float(np.mean(replicates)), 'bootstrap_rho_sd': float(np.std(replicates, ddof=1)), 'bootstrap_rho_bias': float(np.mean(replicates) - rho)}

def methods(repetitions):
    return f"# Analysis protocol: atlas panels j-o\n\nThis protocol is written before computation. This is exploratory inference,\nnot a registered confirmatory analysis. Existing figures are not modified.\n\n## Estimand and data\nTwo-sided H0: marginal city-level Spearman rho = 0 for each of six drivers and ELI.\nCity-equal, variable-specific complete-case samples are reused by importing\nload_data and make_groups from the plotting script. No quintile aggregation is\nused for correlations. Match IDs, driver values and ELI to the exported plotting\nsource table. Retain upstream exclusions (including negative population minima).\nDo not invert travel times, winsorize scores, or clip values to map color limits.\nThe source raster spatial-support limitations remain unresolved. These tests are\nnot independent accuracy validation when drivers were used to construct ELI.\n\n## Conventional benchmark\nscipy.stats.spearmanr, two-sided asymptotic P: assumes independent cities.\nAdjust six P values using Benjamini-Hochberg (BH), with Benjamini-Yekutieli (BY)\nalso provided because the six tests share ELI and correlated predictors.\n\n## Primary working spatial specification\nEPSG:6933 square cells, projected side length 500 km, origin (0,0).\nAll cities in an occupied cell form one cluster. Sample K occupied clusters\nuniformly with replacement K times per replicate; keep paired driver and ELI\nvalues and every city within a selected cluster, repeated with that cluster.\nEach original city has expected multiplicity 1; the estimand is not replaced\nwith a correlation of block means. Replicate city counts vary with cluster size.\nSpearman ranks are recomputed including multiplicities and ties on each replicate;\nfrequency-weighted midranks are mathematically identical to explicit replication.\nUse {repetitions:,} replicates, PCG64, deterministic seeds derived from {SEED}.\n\n## Interval and approximate test\nLet z=atanh(rho) and d_b=atanh(rho_b*)-z. The unadjusted 95% basic bootstrap\ninterval is tanh([z-quantile(d,.975), z-quantile(d,.025)]).\nThe approximate, centered-bootstrap, equal-tailed two-sided P is\nmin(1, 2*(1+min(count(d<=z), count(d>=z)))/(B+1)).\nThis inverts the corresponding basic-bootstrap error distribution for H0:rho=0.\nIt is NOT a permutation test and NOT twice the fraction of raw rho* crossing zero.\nThe +1 convention avoids P=0; resolution is 2/(B+1). It does not make an\napproximate bootstrap test exact. Confidence intervals are marginal, not FDR-adjusted.\n\nBH and BY corrections are applied across six tests separately within each\nspatial specification; BH over all 36 spatial tests is also supplied as a\nsensitivity column, not a reason to select a favorable specification.\n\n## Prespecified sensitivity checks\nProjected side lengths 250, 500, 1000 km, each with grid origin offset (0,0)\nor (L/2,L/2), for six specifications in total. The primary specification is\n500 km, unshifted, regardless of results. Report every specification and the\nrange of adjusted P values; do not select the smallest P. These scales are\npragmatic sensitivity choices, not estimated spatial correlation ranges.\n\n## Limits on interpretation\nThis is a pairs cluster bootstrap under a working assumption that occupied\nblocks are approximately exchangeable and cross-block dependence is limited.\nIt retains dependence within blocks, NOT all global spatial dependence.\nCities are irregularly distributed, cluster sizes differ, block boundaries cut\nneighbors, and broad geographic nonstationarity may violate these assumptions.\nEPSG:6933 preserves area, not distances: 500 projected km is not a uniform\n500-km geodesic radius. Grid translations test some boundary sensitivity, not\nall choices. Sampling uncertainty does not include raster/model uncertainty.\nNo covariate or country adjustment is made; an association is not a causal effect.\nNo claim of universal global-population representativeness is made.\nBefore submission, assess dependence range/calibration or a defensible spatial\nnull model; these approximate results alone do not certify valid figure stars.\n\n## References\n- Cluster resampling and its assumptions: Cameron & Miller (2015),\n  https://doi.org/10.3368/jhr.50.2.317\n- Irregular spatial sampling requires care; this script is NOT an implementation\n  of the paper's specialized valid spatial regression bootstrap:\n  Lahiri & Zhu (2006), https://arxiv.org/abs/math/0611261\n- Bootstrap error distributions and basic intervals:\n  https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html\n- BH/BY correction assumptions:\n  https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html\n"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repetitions', type=int, default=9999)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'figure_outputs_global_city_atlas/significance')
    args = parser.parse_args()
    if args.repetitions < 999:
        raise ValueError('Use at least 999 replicates; default 9999 for final exploratory results.')
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / 'methods.md').write_text(methods(args.repetitions), encoding='utf-8')
    started = time.perf_counter()
    atlas, _, source_audit = load_data(ROOT / 'derived_data/q3_city_driver_master.csv')
    groups, _, sources = make_groups(atlas, 'variable')
    plot_path = ROOT / 'figure_outputs_global_city_atlas/source_driver_quintiles.csv'
    recorded = pd.read_csv(resource(plot_path))
    keys = ['driver', 'id_1']
    pd.testing.assert_frame_equal(sources.sort_values(keys).reset_index(drop=True)[recorded.columns], recorded.sort_values(keys).reset_index(drop=True), check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
    source = sources.merge(atlas[['id_1', 'lon', 'lat']], on='id_1', how='left', validate='many_to_one')
    if not np.isfinite(source[['lon', 'lat', 'driver_value', 'eli50']]).all().all():
        raise ValueError('Nonfinite analysis values or coordinates.')
    source.to_csv(out / 'analysis_city_samples.csv', index=False, encoding='utf-8-sig')
    ordinary, spatial, assignments, draws = ([], [], [], {})
    for driver_index, driver in enumerate(DRIVERS):
        data = source.loc[source.driver.eq(driver.column)].sort_values('id_1')
        x, y = (data.driver_value.to_numpy(), data.eli50.to_numpy())
        test = spearmanr(x, y, alternative='two-sided')
        rho = float(test.statistic)
        if not np.isclose(rho, groups[driver.column][2], atol=1e-14, rtol=0):
            raise ValueError('Correlation no longer matches the plotted value.')
        base = {'panel': chr(ord('j') + driver_index), 'driver': driver.column, 'title': driver.title, 'n_cities': len(data), 'rho': rho}
        ordinary.append({**base, 'p_iid': float(test.pvalue)})
        for spec_index, (size, offset) in enumerate(SPECS):
            tick = time.perf_counter()
            blocks, cells = block_ids(data.lon, data.lat, size, offset)
            counts = np.bincount(blocks)
            seed = np.random.SeedSequence([SEED, driver_index, spec_index])
            values, sizes = bootstrap_rho(x, y, blocks, args.repetitions, seed)
            spec = f'km{size}_shift{int(offset * 100)}'
            draws[f'{driver.column}__{spec}'] = values
            membership = data[['id_1', 'driver']].copy()
            membership['specification'] = spec
            membership['block_x'] = cells[blocks, 0]
            membership['block_y'] = cells[blocks, 1]
            assignments.append(membership)
            spatial.append({**base, 'specification': spec, 'primary': spec_index == 0, 'block_size_projected_km': size, 'grid_offset_fraction': offset, 'n_blocks': len(counts), 'largest_block_n': int(counts.max()), 'largest_block_fraction': float(counts.max() / len(data)), 'block_size_concentration_neff': float(len(data) ** 2 / (counts ** 2).sum()), 'repetitions': args.repetitions, 'bootstrap_n_min': int(sizes.min()), 'bootstrap_n_max': int(sizes.max()), **bootstrap_inference(rho, values), 'elapsed_seconds': time.perf_counter() - tick})
            print(f"{base['panel']} {spec}: n={len(data)}, blocks={len(counts)}, rho={rho:+.6f}, P_approx={spatial[-1]['p_boot_approx']:.6g}, elapsed={spatial[-1]['elapsed_seconds']:.1f}s", flush=True)
    iid = pd.DataFrame(ordinary)
    for method in ('bh', 'by'):
        iid[f'q_{method}_six'] = false_discovery_control(iid.p_iid, method=method)
    results = pd.DataFrame(spatial)
    for _, group in results.groupby('specification'):
        for method in ('bh', 'by'):
            results.loc[group.index, f'q_{method}_six'] = false_discovery_control(group.p_boot_approx, method=method)
    results['q_bh_all36'] = false_discovery_control(results.p_boot_approx, method='bh')
    iid.to_csv(out / 'iid_spearman.csv', index=False, encoding='utf-8-sig')
    results.to_csv(out / 'spatial_bootstrap_all_specs.csv', index=False, encoding='utf-8-sig')
    primary = results.loc[results.primary].copy()
    primary.to_csv(out / 'spatial_bootstrap_primary_500km.csv', index=False, encoding='utf-8-sig')
    pd.concat(assignments, ignore_index=True).to_csv(out / 'spatial_block_membership.csv', index=False, encoding='utf-8-sig')
    np.savez_compressed(out / 'bootstrap_rho_replicates.npz', **draws)
    summary = results.groupby('driver', sort=False).agg(q_bh_min=('q_bh_six', 'min'), q_bh_max=('q_bh_six', 'max'), q_by_min=('q_by_six', 'min'), q_by_max=('q_by_six', 'max'), ci_low_min=('ci95_low', 'min'), ci_high_max=('ci95_high', 'max'), bh_significant_specs=('q_bh_six', lambda s: int(s.lt(0.05).sum())), by_significant_specs=('q_by_six', lambda s: int(s.lt(0.05).sum())))
    summary.to_csv(out / 'sensitivity_summary.csv', encoding='utf-8-sig')
    lines = ['# Results: exploratory significance of atlas correlations', '', 'Primary spatial specification: 500 projected km, unshifted; approximate P values.', 'Intervals are unadjusted 95% basic Fisher-z bootstrap intervals. Read methods.md before use.', '', '| Panel | Driver | n | rho | 95% CI | IID P | Spatial approx P | BH q | BY q | BH-significant specifications |', '|---|---|---:|---:|---|---:|---:|---:|---:|---:|']
    for row in primary.itertuples():
        conventional = iid.loc[iid.driver.eq(row.driver)].iloc[0]
        sens = summary.loc[row.driver]
        lines.append(f'| {row.panel} | {row.title} | {row.n_cities} | {row.rho:+.6f} | [{row.ci95_low:+.4f}, {row.ci95_high:+.4f}] | {conventional.p_iid:.6g} | {row.p_boot_approx:.6g} | {row.q_bh_six:.6g} | {row.q_by_six:.6g} | {int(sens.bh_significant_specs)}/6 |')
    lines += ['', 'A small q is not evidence of strong effect, causality, or independent ELI accuracy.', 'Stability across these grids does not establish full control of spatial dependence.', 'No stars have been added to the figure. No specification was selected by smallest P.']
    (out / 'results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    manifest = {'status': 'exploratory spatial-cluster approximation; calibration not established', 'source_audit': source_audit, 'plot_source_matches': True, 'plot_source_sha256': hashlib.sha256(resource(plot_path).read_bytes()).hexdigest(), 'analysis_script_sha256': hashlib.sha256(resource(Path(__file__)).read_bytes()).hexdigest(), 'repetitions_per_test': args.repetitions, 'seed': SEED, 'specifications': SPECS, 'p_resolution': 2 / (args.repetitions + 1), 'elapsed_seconds': time.perf_counter() - started, 'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__, 'pyproj': pyproj.__version__}}
    (out / 'audit.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(primary[['panel', 'rho', 'ci95_low', 'ci95_high', 'p_boot_approx', 'q_bh_six', 'q_by_six']].to_string(index=False), flush=True)
    print(f"Completed in {manifest['elapsed_seconds']:.1f}s", flush=True)
