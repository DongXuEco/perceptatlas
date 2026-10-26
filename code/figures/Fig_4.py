"""Plot Fig_4 using the packaged numerical tables or supplied local inputs."""
from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_support"))
from figure_cli import prepare_figure as _prepare_figure, save_figure as _save_figure
from figure_cli import read_table as _read_table, model_tables as _model_tables
from figure_cli import configure_land_functions as _configure_land_functions

_figure_options = _prepare_figure('Fig_4', 'analysis_tables') if __name__ == "__main__" else None

from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
from pathlib import Path
import sys, json, hashlib, shutil
from datetime import datetime
ROOT = PACKAGE_ROOT
import Fig_4_base as base
import Fig_4_composition_profiles as fg
import matplotlib as mpl
mpl.use('Agg')
ORIGINAL_RC = mpl.rcParams.copy()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, PowerNorm, to_rgb, LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator, PercentFormatter
from scipy.stats import gaussian_kde, rankdata, ttest_ind, ttest_rel
from scipy.ndimage import gaussian_filter
import numpy as np
import pandas as pd
import mpmath as mp
from PIL import Image, ImageOps
WORK = output_dir('Fig_4')
WORK.mkdir(parents=True, exist_ok=True)
OUT_DIR = output_dir('Fig_4')
CE_FILES = {k: v.replace('_reorganized.csv', '_reorganized_recomputed.csv') for k, v in base.CE_FILES.items()}
GROUPS = ('South', 'North')
COLORS = {'South': '#F27C2B', 'North': '#0D47A1'}
INK = '#263442'
MUTED = '#77848E'
GRID = '#E7EDF2'
DETAIL, AXIS, TITLE, LETTER = (7.6, 8.5, 10.0, 12.0)
ORDER = list(base.EMOTIONS)
AUDIT = {}
plt.rcParams.update({'font.family': 'Arial', 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': DETAIL, 'axes.labelsize': AXIS, 'axes.linewidth': 0.55, 'axes.spines.top': False, 'axes.spines.right': False, 'xtick.labelsize': DETAIL, 'ytick.labelsize': DETAIL, 'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'axes.unicode_minus': True, 'legend.frameon': False})

def sha(p):
    return hashlib.sha256(resource(Path(p)).read_bytes()).hexdigest()

def find_project_file(filename):
    p = ROOT / 'derived_data/recomputed_core_expansion/ce_reorganized_compatible' / filename
    if filename in CE_FILES.values():
        assert resource(p).exists(), str(p)
        return p
    hits = sorted(resource_glob(ROOT, '*/' + filename))
    if not hits:
        raise FileNotFoundError(filename)
    return hits[0]

def load_inputs():
    base.find_project_file = find_project_file
    base.CE_FILES = CE_FILES
    d = base.load_city_metadata().merge(base.load_ce_data(), on='id_1', how='inner')
    frozen = pd.read_csv(resource(WORK / 'original_city_data.csv'), float_precision='round_trip')
    assert d.id_1.is_unique and len(d) == len(frozen) == 5905
    for column in d.select_dtypes(include='number'):
        np.testing.assert_allclose(d[column], frozen[column], rtol=0, atol=2e-14, equal_nan=True)
    valid = d.dropna(subset=['eli_diff'])
    assert len(valid) == 4568 and valid.ns_group.value_counts().to_dict() == {'North': 3223, 'South': 1345}
    for e in ORDER:
        sign = -1 if e in base.NEGATIVE else 1
        np.testing.assert_allclose(d['benefit_' + e], sign * (d['core_' + e] - d['expansion_' + e]), rtol=0, atol=2e-14, equal_nan=True)
    AUDIT['input_hashes'] = {str(find_project_file(v).relative_to(ROOT)): sha(find_project_file(v)) for v in CE_FILES.values()}
    AUDIT['input_hashes'][str(find_project_file('global_city_point.dbf').relative_to(ROOT))] = PACKAGE_ROOT
    AUDIT['rows'] = len(d)
    AUDIT['paired_cities'] = len(valid)
    AUDIT['paired_by_group'] = valid.ns_group.value_counts().to_dict()
    return d

def clean(ax):
    ax.tick_params(direction='out', length=2.5, width=0.5, pad=2.5, labelsize=DETAIL)
    for s in ax.spines.values():
        s.set_linewidth(0.55)
        s.set_color('#34424D')
    ax.spines[['top', 'right']].set_visible(False)

def map_panel(parent, d):
    parent.set_axis_off()
    map_shift = 2 / (parent.get_position().width * parent.figure.get_figwidth() * 25.4)
    map_scale = 1.05
    map_down = 3 / (parent.get_position().height * parent.figure.get_figheight() * 25.4)
    content_down = 4 / (parent.get_position().height * parent.figure.get_figheight() * 25.4)
    ax = parent.inset_axes([-map_shift - (map_scale - 1) / 2, 0.18 - 0.82 * (map_scale - 1) / 2 - map_down, map_scale, 0.82 * map_scale])
    base.draw_land_background(ax)
    norm = Normalize(-0.2, 0.2, clip=True)
    points = ax.scatter(d.lon, d.lat, c=d.eli_diff, s=4.6, cmap=base.MAP_CMAP, norm=norm, linewidths=0, alpha=0.84, rasterized=True, zorder=2)
    ax.set(xlim=(-180, 180), ylim=(-58, 84))
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    cax = parent.inset_axes([0.38, 0.055 - content_down, 0.37, 0.025])
    cb = parent.figure.colorbar(points, cax=cax, orientation='horizontal', ticks=[-0.2, 0, 0.2])
    cb.outline.set_linewidth(0.45)
    cb.ax.tick_params(labelsize=DETAIL, length=2, width=0.4, pad=1)
    cb.ax.set_title('Core − expansion ELI', fontsize=AXIS, pad=4)
    vals = d.eli_diff.dropna().to_numpy()
    dx = parent.inset_axes([0.025, 0.045 - content_down, 0.22, 0.13])
    x = np.linspace(vals.min(), vals.max(), 220)
    density = gaussian_kde(vals)(x)
    for j in range(len(x) - 1):
        dx.fill_between(x[j:j + 2], density[j:j + 2], color=base.MAP_CMAP(norm(x[j:j + 2].mean())), lw=0)
    dx.plot(x, density, color=INK, lw=0.65)
    dx.axvline(np.median(vals), color=MUTED, ls=(0, (2, 2)), lw=0.6)
    dx.set(xlim=(np.floor(vals.min() * 20) / 20, np.ceil(vals.max() * 20) / 20), ylim=(0, density.max() * 1.12), yticks=[])
    dx.set_xticks([-0.3, 0, 0.5])
    dx.tick_params(labelsize=DETAIL - 1, length=2, pad=1)
    dx.spines[['left', 'top', 'right']].set_visible(False)
    dx.text(0, 1.08, 'City density', transform=dx.transAxes, fontsize=DETAIL)
    text_up = 2 / (parent.get_position().height * parent.figure.get_figheight() * 25.4)
    for y, positive, color, label in [(0.51 + text_up - content_down, True, '#0D47A1', 'Core > expansion'), (0.37 + text_up - content_down, False, '#F27C2B', 'Core < expansion')]:
        count = int((vals > 0).sum()) if positive else int((vals < 0).sum())
        parent.text(0.025, y, f'{label}\n{count:,} cities', transform=parent.transAxes, fontsize=8.5, color=color, va='top', linespacing=1.15)
    AUDIT['a_count_annotation_layout'] = {'x': 0.025, 'blue_top_y': 0.51 + text_up - content_down, 'orange_top_y': 0.37 + text_up - content_down, 'font_pt': 8.5, 'previous_font_pt': DETAIL - 1, 'location': 'above city-density curve', 'previous_up_mm': 2, 'cumulative_down_mm': 4, 'latest_up_mm': 1}
    AUDIT['a_map_left_shift_mm'] = 2
    AUDIT['a_map_width_height_scale'] = map_scale
    AUDIT['a_map_down_shift_mm'] = 3
    AUDIT['a_other_content_down_shift_mm'] = 4
    AUDIT['map_sign_counts'] = {'positive': int((vals > 0).sum()), 'negative': int((vals < 0).sum())}
    return ax

def quantile_gap(ax, d):
    q = np.linspace(0.05, 0.95, 181)
    groups = [d.loc[d.ns_group == g, 'eli_diff'].dropna().to_numpy() for g in GROUPS]
    vals = np.concatenate(groups)
    xmin = np.floor(vals.min() * 20) / 20
    xmax = np.ceil(vals.max() * 20) / 20
    xedges = np.linspace(xmin, xmax, 401)
    yedges = np.linspace(0, 100, 241)
    ranks = [(rankdata(v, method='average') - 0.5) / len(v) * 100 for v in groups]
    bw = len(vals) ** (-1 / 6)
    sx = np.std(vals, ddof=1) * bw / np.diff(xedges)[0]
    sy = np.std(np.concatenate(ranks), ddof=1) * bw / np.diff(yedges)[0]
    fields = []
    for v, pct in zip(groups, ranks):
        counts, _, _ = np.histogram2d(v, pct, bins=(xedges, yedges))
        assert counts.sum() == len(v)
        fields.append(gaussian_filter(counts.T, sigma=(sy, sx), mode='reflect'))
    total = sum(fields)
    rgba = np.zeros((*total.shape, 4))
    for field, g in zip(fields, GROUPS):
        rgba[:, :, :3] += field[:, :, None] * np.array(to_rgb(COLORS[g]))
    np.divide(rgba[:, :, :3], total[:, :, None], out=rgba[:, :, :3], where=total[:, :, None] > 0)
    rgba[:, :, 3] = 0.72 * PowerNorm(0.65)(total / total.max())
    ax.imshow(rgba, extent=(xmin, xmax, 0, 100), origin='lower', aspect='auto', interpolation='bilinear', zorder=0)
    ax.axvline(0, color=MUTED, ls=(0, (3, 3)), lw=0.6, zorder=1)
    for v, g, marker in zip(groups, GROUPS, ('o', 's')):
        curve = np.quantile(v, q)
        ax.plot(curve, q * 100, color=COLORS[g], lw=1.6, zorder=3)
        for p in (0.05, 0.25, 0.5, 0.75, 0.95):
            ax.plot(np.quantile(v, p), p * 100, marker=marker, ms=3.5 if p == 0.5 else 2.3, mfc=COLORS[g] if p == 0.5 else 'white', mec=COLORS[g], mew=0.65, zorder=4)
    diff, lo, hi = base.bootstrap_difference_ci(*groups, seed=441)
    test = ttest_ind(groups[1], groups[0], equal_var=False, alternative='two-sided')
    p_label = 'Welch P < 0.001' if test.pvalue < 0.001 else f'Welch P = {test.pvalue:.3f}'
    AUDIT['e_welch_test'] = {'comparison': 'North minus South city-level core-expansion ELI gap', 'alternative': 'two-sided', 't': float(test.statistic), 'df': float(test.df), 'p': float(test.pvalue), 'n_South': len(groups[0]), 'n_North': len(groups[1]), 'multiple_testing_adjustment': 'none; a single North-South gap contrast was tested'}
    ax.text(0.98, 0.025, f'Δ N−S = {diff:+.3f}\n95% CI [{lo:+.3f}, {hi:+.3f}]\n{p_label}', transform=ax.transAxes, fontsize=(DETAIL - 1) * 1.2, ha='right', va='bottom', linespacing=1.5, gid='e_gap_annotation')
    ax.set(xlim=(xmin, xmax), ylim=(3, 97), yticks=[5, 25, 50, 75, 95])
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.set_xlabel('Core − expansion ELI', fontsize=AXIS, labelpad=4)
    ax.set_ylabel('Percentile', fontsize=AXIS)
    clean(ax)
    AUDIT['north_minus_south_gap'] = {'mean': float(diff), 'ci95': [float(lo), float(hi)], 'bootstrap_seed': 441}
    AUDIT['quantile_curve_points'] = 181
    AUDIT['e_curve_style'] = {'linewidth_pt': 1.6, 'central_percentiles_emphasized': False}

def core_advantage_profiles(ax, d):
    rows = [('ELI', 'eli_diff')] + [(label, 'benefit_' + e) for e, label in base.EMOTIONS.items()]
    positions = np.arange(len(rows), dtype=float)[::-1]
    positions[0] += 0.4
    tests = []
    with mp.workdps(50):
        probabilities = []
        for label, col in rows:
            emotion = None if col == 'eli_diff' else col.removeprefix('benefit_')
            core = 'core_eli' if emotion is None else 'core_' + emotion
            expansion = 'expansion_eli' if emotion is None else 'expansion_' + emotion
            pair = d[[core, expansion]].replace([np.inf, -np.inf], np.nan).dropna()
            first, second = (expansion, core) if emotion in base.NEGATIVE else (core, expansion)
            difference = (pair[first] - pair[second]).to_numpy()
            np.testing.assert_allclose(d.loc[pair.index, col], difference, rtol=0, atol=2e-14)
            test = ttest_rel(pair[first], pair[second], alternative='two-sided')
            ci = test.confidence_interval(confidence_level=0.95)
            degrees = mp.mpf(float(test.df))
            statistic = mp.mpf(float(test.statistic))
            probability = mp.betainc(degrees / 2, mp.mpf('.5'), 0, degrees / (degrees + statistic ** 2), regularized=True)
            probabilities.append(probability)
            tests.append({'metric': label, 'n_pairs': len(pair), 'df': float(test.df), 'mean_core': float(pair[core].mean()), 'mean_expansion': float(pair[expansion].mean()), 'mean_core_minus_expansion': float((pair[core] - pair[expansion]).mean()), 'advantage_direction': first + ' minus ' + second, 'mean_advantage': float(difference.mean()), 'mean_advantage_ci95_low': float(ci.low), 'mean_advantage_ci95_high': float(ci.high), 't_advantage': float(test.statistic), 'paired_cohen_dz': float(difference.mean() / difference.std(ddof=1)), 'p_two_sided': mp.nstr(probability, 16), 'log10_p_two_sided': float(mp.log10(probability))})
        running = mp.mpf(0)
        for rank, index in enumerate(sorted(range(len(tests)), key=lambda i: probabilities[i])):
            running = max(running, min(mp.mpf(1), (len(tests) - rank) * probabilities[index]))
            tests[index].update(p_holm=mp.nstr(running, 16), log10_p_holm=float(mp.log10(running)), reject_holm_05=bool(running < mp.mpf('.05')), plot_label='< 0.001' if running < mp.mpf('.001') else f'{float(running):.3f}')
    summaries = []
    for y, (label, col), test in zip(positions, rows, tests):
        vals = d[col].dropna().to_numpy()
        q05, q25, q50, q75, q95 = np.quantile(vals, [0.05, 0.25, 0.5, 0.75, 0.95])
        mean = vals.mean()
        x = np.linspace(-0.16, 0.24, 601)
        density = gaussian_kde(vals)(x)
        relative = density / density.max()
        vertical = np.linspace(-0.36, 0.36, 101)
        field = np.exp(-0.5 * (vertical[:, None] / 0.11) ** 2) * relative[None, :] ** 0.9
        rgba = np.zeros((len(vertical), len(x), 4))
        rgba[:, :, :3] = to_rgb('#3667B2')
        rgba[:, :, 3] = 0.75 * field
        ax.imshow(rgba, extent=(-0.16, 0.24, y - 0.36, y + 0.36), origin='lower', aspect='auto', interpolation='bilinear', zorder=1)
        rail = y - 0.22
        ax.plot([q05, q95], [rail, rail], color='#9DB8DE', lw=0.65, zorder=2)
        ax.plot([q25, q75], [rail, rail], color='#0D47A1', lw=2.0, solid_capstyle='butt', zorder=3)
        ax.plot([q50, q50], [rail - 0.1, rail + 0.1], color='#0D47A1', lw=0.9, zorder=5)
        ax.plot(mean, rail, marker='D', ms=3.2, color='#3667B2', mec='#0D47A1', mew=0.45, zorder=4)
        ax.text(0.11, y, test['plot_label'], transform=ax.get_yaxis_transform(), ha='center', va='center', fontsize=7.0 * 1.08, color='black', clip_on=False, gid='b_holm_' + col)
        summaries.append({'metric': label, 'mean': float(mean), 'p05_p25_p50_p75_p95': [float(v) for v in [q05, q25, q50, q75, q95]]})
    ax.text(0.11, 6.85, 'Holm $P$', transform=ax.get_yaxis_transform(), ha='center', va='center', fontsize=7.0 * 1.08, color='black', clip_on=False, gid='b_holm_header')
    ax.axhline(5.7, color=base.PALETTE['cool_mist'], lw=0.55, zorder=0)
    ax.axvline(0, color=base.PALETTE['muted'], lw=0.6, ls='--', zorder=0)
    ax.set(xlim=(-0.16, 0.24), ylim=(-0.55, 6.95), xticks=[-0.1, 0, 0.1, 0.2], yticks=positions)
    ax.set_yticklabels([label for label, _ in rows], fontsize=7.0 * 1.08)
    ax.set_xlabel('Core advantage (score difference)', fontsize=8.0 * 1.08)
    base.clean_axis(ax)
    ax.tick_params(axis='both', labelsize=7.0 * 1.08)
    AUDIT['b_distribution_summaries'] = summaries
    AUDIT['b_density_definition'] = 'Scott-bandwidth KDE, normalized within each row and shown across the unchanged score axis; opacity encodes relative density, not confidence.'
    AUDIT['b_summary_symbols'] = 'Diamond: mean; vertical tick: median; thick rail: interquartile range; thin rail: 5th-95th percentiles.'
    AUDIT['b_p_annotation_layout'] = {'side': 'inside plot, left strip', 'x_axes_fraction': 0.11, 'font_pt': 7.0 * 1.08, 'row_alignment': 'metric-name rows'}
    AUDIT['b_paired_tests'] = {'method': 'Two-sided paired Student t-test on complete city-level core/expansion pairs', 'null': 'Mean within-city difference equals zero', 'adjustment': 'Holm across all seven panel-b comparisons', 'negative_score_direction': 'Boredom and depression: expansion minus core; other metrics: core minus expansion', 'mean_ci': 'Unadjusted 95% Student t interval for mean within-city advantage', 'dependence_assumption': 'City pairs treated as independent; no country-cluster or spatial correction', 'p_precision': 'Student t tails evaluated with 50 decimal digits; Holm adjustment computed before conversion to strings', 'results': tests}
    pd.DataFrame(tests).to_csv(WORK / 'b_paired_t_holm.csv', index=False)

def prevalence_density_ribbons(ax, d):
    shares = pd.Series({base.EMOTIONS[e]: (d['benefit_' + e].dropna() > 0).mean() for e in base.EMOTIONS}).sort_values()
    summaries = []
    replicates = {}
    x = np.linspace(0, 1, 1201)
    for row, (label, value) in enumerate(shares.items()):
        emotion = next((e for e, name in base.EMOTIONS.items() if name == label))
        boot = base.cluster_bootstrap_share(d, 'benefit_' + emotion, seed=210 + row)
        lo, q25, q75, hi = np.percentile(boot, [2.5, 25, 75, 97.5])
        replicates[emotion] = boot
        kde = gaussian_kde(boot)
        density = kde(x) + kde(-x) + kde(2 - x)
        relative = density / density.max()
        vertical = np.linspace(-0.36, 0.36, 101)
        field = np.exp(-0.5 * (vertical[:, None] / 0.11) ** 2) * relative[None, :] ** 0.9
        rgba = np.zeros((len(vertical), len(x), 4))
        rgba[:, :, :3] = to_rgb('#3667B2')
        rgba[:, :, 3] = 0.75 * field
        ax.imshow(rgba, extent=(0, 1, row - 0.36, row + 0.36), origin='lower', aspect='auto', interpolation='bilinear', zorder=1)
        rail = row - 0.22
        ax.plot([lo, hi], [rail, rail], color='#9DB8DE', lw=0.65, zorder=2)
        ax.plot([q25, q75], [rail, rail], color='#0D47A1', lw=2.0, solid_capstyle='butt', zorder=3)
        ax.plot(value, rail, marker='D', ms=3.2, color='#3667B2', mec='#0D47A1', mew=0.45, zorder=4)
        ax.text(1.27, row, f'{value * 100:.0f}% [{lo * 100:.0f}-{hi * 100:.0f}]', ha='center', va='center', fontsize=7.0 * 1.08, color='black')
        summaries.append({'emotion': emotion, 'share': float(value), 'original_95_CI': [float(lo), float(hi)], 'bootstrap_middle_50_percent': [float(q25), float(q75)], 'bootstrap_seed': 210 + row, 'bootstrap_replicates': len(boot)})
    ax.plot([0.5, 0.5], [-0.5, 5.4], color=base.PALETTE['muted'], lw=0.6, ls='--', zorder=0)
    ax.text(1.27, 5.5, 'Share [95% CI]', ha='center', va='center', fontsize=7.0 * 1.08, color='black', gid='d_share_header', transform=ax.transData + mpl.transforms.ScaledTranslation(0, 2 / 25.4, ax.figure.dpi_scale_trans))
    ax.set(xlim=(0, 1.48), ylim=(-0.55, 5.75), xticks=np.linspace(0, 1, 6), yticks=np.arange(len(shares)))
    ax.set_yticklabels(shares.index, fontsize=7.0 * 1.08)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.spines['bottom'].set_bounds(0, 1)
    ax.spines['left'].set_bounds(-0.55, 5.4)
    ax.set_xlabel('Cities with core advantage (%)', fontsize=8.0 * 1.08)
    base.clean_axis(ax)
    ax.tick_params(axis='both', labelsize=7.0 * 1.08)
    AUDIT['d_prevalence'] = summaries
    AUDIT['d_top_spacing'] = {'previous_ylim_top': 6.25, 'ylim_top': 5.75, 'left_spine_top': 5.4, 'header_y': 5.5, 'header_up_mm': 2}
    AUDIT['d_bootstrap_density_definition'] = 'Panel-b visual style applied to Scott-bandwidth KDE of original 700 country-cluster bootstrap share estimates, seed 210+row. KDE reflected at 0 and 1 and peak-normalized within each row. Thin rail: original 95% percentile CI. Thick rail: middle 50% of bootstrap estimates. Diamond: original city share. Dashed line: 50%. Density is not the distribution of city scores.'
    pd.DataFrame(replicates).to_csv(WORK / 'd_country_cluster_bootstrap.csv', index=False)

def matched_density_landscape(ax, d):
    limits = {'x': list(ax.get_xlim()), 'y': list(ax.get_ylim())}
    ticks = {'x': ax.get_xticks().tolist(), 'y': ax.get_yticks().tolist()}
    for artist in [*ax.collections, *ax.lines, *ax.patches]:
        artist.remove()
    for text in list(ax.texts):
        if text.get_gid() != 'panel_label':
            text.remove()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    paired = d[['expansion_eli', 'core_eli', 'ns_group']].replace([np.inf, -np.inf], np.nan).dropna()
    xy = paired[['expansion_eli', 'core_eli']].to_numpy().T
    kde = gaussian_kde(xy, bw_method='scott')
    x = np.linspace(*limits['x'], 240)
    y = np.linspace(*limits['y'], 240)
    xx, yy = np.meshgrid(x, y)
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    relative = density / density.max()
    cmap = LinearSegmentedColormap.from_list('paired_city_density', ['#FFFFFF', '#EDEAF3', '#CCC6DD', '#969BBE', '#547F97', '#236D73'])
    rgba = cmap(PowerNorm(0.65)(relative))
    rgba[:, :, 3] = np.minimum(relative / 0.025, 1) * 0.96
    threshold = 0.03
    ax.fill_between(x, x - threshold, x + threshold, color='#DFE4E8', alpha=0.45, lw=0, zorder=0)
    ax.imshow(rgba, origin='lower', extent=[*limits['x'], *limits['y']], aspect='auto', interpolation='bilinear', rasterized=True, zorder=1)
    levels = [0.08, 0.18, 0.32, 0.5, 0.72, 0.9]
    ax.contour(xx, yy, relative, levels=levels, colors='#516E83', linewidths=0.42, alpha=0.55, zorder=2)
    for offset in [-threshold, threshold]:
        ax.plot(x, x + offset, color='#A5AFB8', lw=0.4, alpha=0.85, zorder=3)
    ax.plot(x, x, color='#66727E', lw=0.7, ls=(0, (3, 2)), zorder=3)
    r, p = base.pearsonr(xy[0], xy[1])
    p_label = '$P$ < 0.001' if p < 0.001 else f'$P$ = {p:.3f}'
    ax.text(0.05, 0.94, f'$r$ = {r:.2f}\n{p_label}', transform=ax.transAxes, fontsize=7.0 * 1.08, color='black', ha='left', va='top', linespacing=1.15, zorder=6)
    means = {}
    for group, marker, offset in [('South', 'o', (-10, -23)), ('North', 's', (25, 19))]:
        tmp = paired.loc[paired.ns_group == group, ['expansion_eli', 'core_eli']]
        mx, my = tmp.mean().to_numpy()
        means[group] = {'n': len(tmp), 'expansion': float(mx), 'core': float(my)}
        ax.scatter(mx, my, s=22, marker=marker, color=COLORS[group], edgecolor='white', linewidth=0.65, zorder=6)
        ax.annotate(group + ' mean', xy=(mx, my), xytext=offset, textcoords='offset points', color=COLORS[group], fontsize=6.8 * 1.08, ha='right' if group == 'South' else 'left', va='center', arrowprops={'arrowstyle': '-', 'lw': 0.65, 'color': COLORS[group], 'shrinkA': 3, 'shrinkB': 4}, zorder=6)
    display_angle = np.degrees(np.arctan(ax.get_position().height * ax.figure.get_figheight() / np.ptp(limits['y']) / (ax.get_position().width * 0.95 * ax.figure.get_figwidth() / np.ptp(limits['x']))))
    ax.text(0.39, 0.47, 'Balanced band (±0.03)', rotation=display_angle, rotation_mode='anchor', fontsize=6.0 * 1.08, color='#66727E', ha='center', va='center', zorder=5)
    key = ax.inset_axes([0.67, 0.13, 0.28, 0.024])
    key.imshow(np.linspace(0, 1, 256)[None, :], aspect='auto', extent=(0, 1, 0, 1), cmap=cmap, norm=PowerNorm(0.65))
    key.set(xticks=[], yticks=[])
    for spine in key.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.35)
        spine.set_edgecolor('#66727E')
        spine.set_zorder(3)
    AUDIT['c_density_legend_frame'] = {side: spine.get_visible() for side, spine in key.spines.items()}
    key.text(0.5, 1.6, 'Relative density', transform=key.transAxes, ha='center', fontsize=6.0 * 1.08)
    for value, align in [(0, 'left'), (1, 'right')]:
        key.text(value, -0.8, str(value), transform=key.transAxes, ha=align, va='top', fontsize=5.5 * 1.08)
    ax.set_xlim(limits['x'])
    ax.set_ylim(limits['y'])
    np.testing.assert_array_equal(ax.get_xticks(), ticks['x'])
    np.testing.assert_array_equal(ax.get_yticks(), ticks['y'])
    AUDIT['c_density_landscape'] = {'paired_cities': len(paired), 'pearson_r': float(r), 'pearson_p': float(p), 'xlim': limits['x'], 'ylim': limits['y'], 'xticks': ticks['x'], 'yticks': ticks['y'], 'kde': 'Bivariate Gaussian KDE, Scott bandwidth, evaluated on a 240-by-240 grid; normalized by the displayed density peak.', 'bandwidth_factor': float(kde.factor), 'relative_contour_levels': levels, 'balanced_band_absolute_gap': threshold, 'balanced_band_definition': 'Existing descriptive category in panel f; not a confidence interval or equivalence test.', 'group_markers': 'Arithmetic means of paired cities; South circle, North square; direct labels replace median markers and dispersion ellipses.', 'group_means': means, 'equality_line': 'core ELI = expansion ELI'}

def build_figure():
    AUDIT.clear()
    d = load_inputs()
    base.GROUP_COLORS = {'South': base.PALETTE['burnt_orange'], 'North': base.PALETTE['cerulean']}
    with mpl.rc_context(ORIGINAL_RC):
        fig = base.build_figure()
        fig.set_dpi(450)
        fig.canvas.draw()
        original_bbox = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.1).frozen()
        fig.savefig(WORK / 'original_layout_reference.png', dpi=450, bbox_inches=original_bbox, facecolor='white')
    axes = dict(zip('acebgfd', fig.axes))
    original = {k: list(ax.get_position().bounds) for k, ax in axes.items()}
    figure_inches = fig.get_size_inches().copy()
    old_b = axes['b']
    bounds = old_b.get_position().bounds
    title_transform = old_b.title.get_transform()
    title_position = old_b.title.get_position()
    label = next((t for t in old_b.texts if t.get_gid() == 'panel_label'))
    label_position = label.get_position()
    label_transform = label.get_transform()
    old_b.clear()
    core_advantage_profiles(old_b, d)
    old_b.set_title('Distribution of core advantage', fontsize=9.0 * 1.08, pad=3)
    old_b.title.set_position(title_position)
    old_b.title.set_transform(title_transform)
    old_b.text(*label_position, 'b', transform=label_transform, fontsize=10.5 * 1.08, fontweight='bold', color='black', ha='left', va='bottom', gid='panel_label')
    old_b.set_position(bounds)
    old_d = axes['d']
    bounds = old_d.get_position().bounds
    title_transform = old_d.title.get_transform()
    title_position = old_d.title.get_position()
    label = next((t for t in old_d.texts if t.get_gid() == 'panel_label'))
    label_position = label.get_position()
    label_transform = label.get_transform()
    old_d.clear()
    prevalence_density_ribbons(old_d, d)
    old_d.set_title('Prevalence of core advantage', fontsize=9.0 * 1.08, pad=3)
    old_d.title.set_position(title_position)
    old_d.title.set_transform(title_transform)
    old_d.text(*label_position, 'd', transform=label_transform, fontsize=10.5 * 1.08, fontweight='bold', color='black', ha='left', va='bottom', gid='panel_label')
    old_d.set_position(bounds)
    matched_density_landscape(axes['c'], d)
    fg_data = fg.prepare(d)
    fg_height_scale = 1.2
    fg_header_anchors = {}
    for letter, draw, title in [('f', fg.draw_f, 'Composition of core–expansion gaps'), ('g', fg.draw_g, 'Core–expansion perception profiles')]:
        target = axes[letter]
        bounds = target.get_position().bounds
        title_transform = target.title.get_transform()
        title_position = target.title.get_position()
        label = next((t for t in target.texts if t.get_gid() == 'panel_label'))
        label_position = label.get_position()
        label_transform = label.get_transform()
        title_anchor = fig.transFigure.inverted().transform(title_transform.transform(title_position))
        label_anchor = fig.transFigure.inverted().transform(label_transform.transform(label_position))
        fg_header_anchors[letter] = {'title': title_anchor.tolist(), 'letter': label_anchor.tolist()}
        for child in list(target.child_axes):
            child.remove()
        target.clear()
        x, y, width, height = bounds
        target.set_position([x, y - height * (fg_height_scale - 1), width, height * fg_height_scale])
        draw(target, fg_data)
        target.set_title(title, fontsize=9.0 * 1.08, pad=3, y=float(title_anchor[1]))
        target.title.set_position(title_anchor)
        target.title.set_transform(fig.transFigure)
        target.text(*label_anchor, letter, transform=fig.transFigure, fontsize=10.5 * 1.08, fontweight='bold', color='black', ha='left', va='bottom', gid='panel_label')
    AUDIT['fg_bold'] = {'shares': fg_data['shares'].to_dict(), 'original_cumulative_boundaries': {g: a.tolist() for g, a in fg_data['boundaries'].items()}, 'scores_and_original_gap_intervals': fg_data['scores'], 'composition_bootstrap': '600 within-group city resamples; seed 170 North, 171 South', 'f_tiles': '10-by-10 percentage grids with enlarged boundary cells. Exact partial cell areas encode category shares.', 'f_exact_tile_units': fg_data['tile_audit'], 'f_boundary_detail': fg_data['f_boundary_detail'], 'f_definition_column': fg_data['f_definition_column'], 'f_category_colors': dict(zip(fg.CAT, fg.CAT_COLORS)), 'g_summary': 'Two-by-three grouped distributions on common 0-1 score axes. Each group has expansion/core columns, bounded reflected Scott KDE, city score jitter, IQR boxes, medians and 5th-95th percentile whiskers. Open squares and filled circles show expansion/core means. Gap statistics are retained in the audit output.', 'g_profile_statistics': fg_data['profile_audit'], 'g_layout': fg_data['g_layout'], 'paired_cities': fg_data['paired_cities']}
    for letter, draw, title in [('a', map_panel, 'Core–expansion emotional livability gap'), ('e', quantile_gap, 'Core–expansion gap by group')]:
        ax = axes[letter]
        bounds = ax.get_position().bounds
        for child in list(ax.child_axes):
            child.remove()
        ax.remove()
        ax = fig.add_axes(bounds)
        axes[letter] = ax
        draw(ax, d)
        ax.set_title(title, fontsize=9.0 * 1.08, pad=3, y=1.0 if letter == 'a' else None)
        ax.title.set_transform(ax.title.get_transform() + mpl.transforms.ScaledTranslation(0, 4 / 72, fig.dpi_scale_trans))
        if letter == 'e':
            ax.title.set_transform(ax.title.get_transform() + mpl.transforms.ScaledTranslation(0, 1 / 25.4, fig.dpi_scale_trans))
        base.add_panel_label(ax, letter, x=-0.02 / bounds[2], y=1 + 0.065 / (bounds[3] * figure_inches[1]))
        ax.texts[-1].set_fontsize(10.5 * 1.08)
    axes['e'].legend(handles=[Line2D([], [], color=COLORS[g], lw=1.6, label=g) for g in GROUPS], loc='upper left', fontsize=7.0 * 1.08, handlelength=1.1, labelspacing=0.25)
    reference = axes['e']
    g_axes_style = []
    for gax in axes['g'].child_axes:
        for side in ['left', 'bottom']:
            gax.spines[side].set_linewidth(reference.spines[side].get_linewidth())
            gax.spines[side].set_edgecolor(reference.spines[side].get_edgecolor())
        for which in ['x', 'y']:
            ref_axis = getattr(reference, which + 'axis')
            tick = ref_axis.get_major_ticks()[0]
            gax.tick_params(axis=which, which='major', direction='out', length=tick.tick1line.get_markersize(), width=tick.tick1line.get_markeredgewidth(), pad=tick.get_pad(), labelsize=tick.label1.get_fontsize(), color=tick.tick1line.get_color())
            getattr(gax, which + 'axis').label.set_fontproperties(ref_axis.label.get_fontproperties())
        g_axes_style.append({'spine_width': gax.spines['left'].get_linewidth(), 'tick_length': gax.yaxis.get_major_ticks()[0].tick1line.get_markersize(), 'tick_width': gax.yaxis.get_major_ticks()[0].tick1line.get_markeredgewidth(), 'tick_font': gax.yaxis.get_major_ticks()[0].label1.get_fontsize()})
    shared_y = next((t for t in axes['g'].texts if t.get_text() == 'Perception score'))
    shared_y.set_fontproperties(reference.yaxis.label.get_fontproperties())
    shared_y.set_position((-0.012, 0.5))
    AUDIT['g_axis_style_from_e'] = {'profiles': g_axes_style, 'axis_title_font_pt': reference.xaxis.label.get_fontsize(), 'shared_xlabel': None, 'shared_ylabel': 'Perception score'}
    for k in ['b', 'c', 'd', 'e']:
        ax = axes[k]
        bounds = ax.get_position().bounds
        label = next((t for t in ax.texts if t.get_gid() == 'panel_label'))
        anchor = fig.transFigure.inverted().transform(label.get_transform().transform(label.get_position()))
        ax.set_position([bounds[0], bounds[1], bounds[2] * 0.95, bounds[3]])
        label.set_transform(fig.transFigure)
        label.set_position(anchor)
    left_shift_mm = 4.0
    left_shift = left_shift_mm / (figure_inches[0] * 25.4)
    for k in ['b', 'e', 'g']:
        ax = axes[k]
        bounds = ax.get_position().bounds
        for text in [ax.title, *ax.texts]:
            if text.get_transform() is fig.transFigure:
                x, y = text.get_position()
                text.set_position((x - left_shift, y))
        ax.set_position([bounds[0] - left_shift, bounds[1], bounds[2], bounds[3]])
    for key in ['title', 'letter']:
        fg_header_anchors['g'][key][0] -= left_shift
    AUDIT['panel_left_shift_mm'] = {k: left_shift_mm for k in ['b', 'e', 'g']}
    restored = ['b', 'c', 'd']
    for k in ['b', 'c', 'd', 'e']:
        old = original[k]
        np.testing.assert_allclose(axes[k].get_position().bounds, [old[0] - (left_shift if k in ['b', 'e'] else 0), old[1], old[2] * 0.95, old[3]], rtol=0, atol=1e-14)
    for k in ['f', 'g']:
        new = axes[k].get_position().bounds
        old = original[k]
        np.testing.assert_allclose([new[0], new[2], new[1] + new[3], new[3]], [old[0] - (left_shift if k == 'g' else 0), old[2], old[1] + old[3], old[3] * fg_height_scale], rtol=0, atol=1e-14)
    AUDIT['restored_panels'] = []
    AUDIT['kept_new_styles'] = ['a', 'e']
    AUDIT['redesigned_panels'] = ['b', 'c', 'd', 'f', 'g']
    AUDIT['original_canvas_inches'] = figure_inches.tolist()
    AUDIT['axes_bounds_fraction'] = {k: list(axes[k].get_position().bounds) for k in restored + ['e', 'f', 'g']}
    AUDIT['axes_width_height_mm'] = {k: [axes[k].get_position().width * figure_inches[0] * 25.4, axes[k].get_position().height * figure_inches[1] * 25.4] for k in restored + ['e', 'f', 'g']}
    AUDIT['b_to_e_width_scale'] = 0.95
    AUDIT['b_to_e_widths_mm'] = {k: [original[k][2] * figure_inches[0] * 25.4, axes[k].get_position().width * figure_inches[0] * 25.4] for k in ['b', 'c', 'd', 'e']}
    AUDIT['e_gap_annotation_font_pt'] = (DETAIL - 1) * 1.2
    AUDIT['e_gap_annotation_linespacing'] = 1.5
    AUDIT['d_share_header'] = 'Share [95% CI]'
    AUDIT['layout_consistency'] = False
    AUDIT['fg_height_resize'] = {'scale': fg_height_scale, 'before_mm': 46.99, 'after_mm': 46.99 * fg_height_scale, 'widths_and_top_edges_unchanged': True, 'header_positions': True, 'panel_origins': True}
    AUDIT['category_interval_definition'] = 'Percentage grids display descriptive category shares; cumulative share intervals are retained in the audit output.'
    extra_height = original['f'][3] * figure_inches[1] * (fg_height_scale - 1)
    extra_height = np.ceil(extra_height * 450) / 450
    export_bbox = mpl.transforms.Bbox.from_extents(original_bbox.x0, original_bbox.y0 - extra_height, original_bbox.x1, original_bbox.y1)
    AUDIT['canvas_added_height_inches'] = float(extra_height)
    AUDIT['export_bbox_inches'] = list(export_bbox.bounds)
    fig._restored_export_bbox = export_bbox
    mpl.rcParams['axes.unicode_minus'] = ORIGINAL_RC['axes.unicode_minus']
    fig.canvas.draw()
    for k, anchors in fg_header_anchors.items():
        actual = fig.transFigure.inverted().transform(axes[k].title.get_transform().transform(axes[k].title.get_position()))
        np.testing.assert_allclose(actual, anchors['title'], rtol=0, atol=1e-14)
    AUDIT['fg_header_anchors_figure'] = fg_header_anchors
    safety = axes['g'].child_axes[2]
    legend_box = safety.get_legend().get_window_extent(fig.canvas.get_renderer())
    plot_box = safety.get_window_extent(fig.canvas.get_renderer())
    assert legend_box.x0 >= plot_box.x0 and legend_box.x1 <= plot_box.x1 and (legend_box.y0 >= plot_box.y0) and (legend_box.y1 <= plot_box.y1)
    AUDIT['g_legend_inside_safety'] = True
    AUDIT['fg_main_plot_geometry_mm'] = {k: [[child.get_window_extent().width / fig.dpi * 25.4, child.get_window_extent().height / fig.dpi * 25.4] for child in axes[k].child_axes] for k in ['f', 'g']}
    AUDIT['a_map_geometry_inches'] = list(axes['a'].child_axes[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted()).bounds)
    AUDIT['panel_offsets_mm'] = {'b_e_g_additional_left': 2, 'a_map_left': 2, 'a_count_text_up': 2, 'd_share_header_up': 2}
    AUDIT['bd_palette'] = {'density_and_diamond': '#3667B2', 'thick_rail_and_outline': '#0D47A1', 'thin_rail': '#9DB8DE'}
    AUDIT['design'] = 'Holm-adjusted P values are aligned with metric labels on the left of the plot.'
    return fig

def _build_manuscript_figure():
    global load_inputs
    data = _read_table("Fig_4/Fig_4_original_city_data.csv")
    _configure_land_functions(base)
    metadata = ["id_1", "COUNTRY", "NAME_1", "lon", "lat", "ns_group"]
    base.load_city_metadata = lambda: data[metadata].copy()
    base.load_ce_data = lambda: data.drop(columns=metadata[1:]).copy()
    def load_pairs():
        AUDIT.update(rows=len(data), paired_cities=int(data.eli_diff.notna().sum()),
                     paired_by_group=data.dropna(subset=["eli_diff"]).ns_group.value_counts().to_dict(), input_hashes={})
        return data.copy()
    load_inputs = load_pairs
    return build_figure(), None


if __name__ == "__main__":
    _configure_land_functions(sys.modules[__name__])
    _figure, _bbox = _build_manuscript_figure()
    _save_figure(_figure, _bbox, 'Fig_4', _figure_options)
