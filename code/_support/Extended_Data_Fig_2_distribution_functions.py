from __future__ import annotations
from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from scipy.stats import gaussian_kde, ttest_ind
SOUTH = '#F27C2B'
NORTH = '#3B82D6'
INK = '#252A30'
MUTED = '#000000'
PANELS = (('eli', 'Emotional livability', 'ELI'), ('positive_score', 'Positive emotions', 'Score'), ('negative_score', 'Negative emotions', 'Score'), ('beauty', 'Beauty', 'Score'), ('wealthy', 'Wealth', 'Score'), ('safer', 'Safety', 'Score'), ('livelier', 'Liveliness', 'Score'), ('boring', 'Boredom', 'Score'), ('depression', 'Depression', 'Score'))
QUANTILES = np.linspace(0.05, 0.95, 91)

def welch_table(df):
    import mpmath as mp
    rows, probabilities = ([], [])
    with mp.workdps(60):
        for i, (metric, title, _) in enumerate(PANELS):
            south = df.loc[df.ns_group == 'South', metric].to_numpy(dtype=float)
            north = df.loc[df.ns_group == 'North', metric].to_numpy(dtype=float)
            assert np.isfinite(south).all() and np.isfinite(north).all()
            result = ttest_ind(north, south, equal_var=False, alternative='two-sided')
            ci = result.confidence_interval(confidence_level=0.95)
            p = mp.mpf(float(result.pvalue))
            if p == 0:
                dof, statistic = (mp.mpf(float(result.df)), mp.mpf(float(result.statistic)))
                p = mp.betainc(dof / 2, mp.mpf('0.5'), 0, dof / (dof + statistic ** 2), regularized=True)
            probabilities.append(p)
            rows.append(dict(panel=chr(97 + i), metric=metric, label=title, n_south=len(south), n_north=len(north), mean_south=float(np.mean(south)), mean_north=float(np.mean(north)), sd_south=float(np.std(south, ddof=1)), sd_north=float(np.std(north, ddof=1)), mean_difference=float(np.mean(north) - np.mean(south)), ci_low=float(ci.low), ci_high=float(ci.high), median_difference=float(np.median(north) - np.median(south)), t=float(result.statistic), df=float(result.df), p_raw=mp.nstr(p, 16), log10_p_raw=float(mp.log10(p))))
        order = sorted(range(len(rows)), key=lambda i: probabilities[i])
        running = mp.mpf(0)
        for rank, index in enumerate(order):
            running = max(running, min(mp.mpf(1), (len(rows) - rank) * probabilities[index]))
            rows[index]['p_holm'] = mp.nstr(running, 16)
            rows[index]['log10_p_holm'] = float(mp.log10(running))
    return pd.DataFrame(rows)

def quantile_table(df):
    rows = []
    for metric, title, _ in PANELS:
        south = df.loc[df.ns_group == 'South', metric].to_numpy()
        north = df.loc[df.ns_group == 'North', metric].to_numpy()
        qs, qn = (np.quantile(south, QUANTILES), np.quantile(north, QUANTILES))
        for q, s, n in zip(QUANTILES, qs, qn):
            rows.append((metric, title, q, s, n, n - s, len(south), len(north)))
    return pd.DataFrame(rows, columns=['metric', 'label', 'quantile', 'south', 'north', 'north_minus_south', 'n_south', 'n_north'])

def _density(values, grid, bounds, bandwidth):
    kde = gaussian_kde(values, bw_method=bandwidth / np.std(values, ddof=1))
    return kde(grid) + kde(2 * bounds[0] - grid) + kde(2 * bounds[1] - grid)

def _bridge(ax, south, north, metric):
    bounds = (-1.0, 1.0) if metric == 'eli' else (0.0, 1.0)
    limits = (-0.5, 0.65) if metric == 'eli' else (0.0, 1.0)
    grid = np.linspace(*limits, 501)
    pooled = np.concatenate([south, north])
    bandwidth = np.std(pooled, ddof=1) * len(pooled) ** (-0.2)
    ds = _density(south, grid, bounds, bandwidth)
    dn = _density(north, grid, bounds, bandwidth)
    scale = 0.48 / max(ds.max(), dn.max())
    for x, density, direction, color in ((0, ds, -1, SOUTH), (1, dn, 1, NORTH)):
        edge = x + direction * scale * density
        ax.fill_betweenx(grid, x, edge, color=color, alpha=0.45, linewidth=0)
        ax.plot(edge, grid, color=color, lw=1.25)
    probabilities = np.linspace(0.05, 0.95, 19)
    qs, qn = (np.quantile(south, probabilities), np.quantile(north, probabilities))
    x = np.linspace(0, 1, 65)
    transition = x * x * (3 - 2 * x)
    rgb_s, rgb_n = (np.array(to_rgb(SOUTH)), np.array(to_rgb(NORTH)))
    for i in range(len(probabilities) - 1):
        low = qs[i] + (qn[i] - qs[i]) * transition
        high = qs[i + 1] + (qn[i + 1] - qs[i + 1]) * transition
        for j in range(len(x) - 1):
            color = rgb_s * (1 - x[j]) + rgb_n * x[j]
            ax.fill_between(x[j:j + 2], low[j:j + 2], high[j:j + 2], color=color, alpha=0.23 if i % 2 == 0 else 0.09, linewidth=0, rasterized=True)
    marks = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    south_marks, north_marks = (np.quantile(south, marks), np.quantile(north, marks))
    for p, s, n in zip(marks, south_marks, north_marks):
        curve = s + (n - s) * transition
        ax.plot(x, curve, color=INK if p == 0.5 else '#858D96', lw=1.45 if p == 0.5 else 0.55, alpha=1 if p == 0.5 else 0.65, ls='-' if 0.25 <= p <= 0.75 else (0, (2, 3)), zorder=3)
    for x0, vals, color in ((0, south_marks, SOUTH), (1, north_marks, NORTH)):
        ax.plot([x0, x0], vals[[0, -1]], color=color, lw=0.9, zorder=4)
        ax.plot([x0, x0], vals[[1, 3]], color=color, lw=4.0, solid_capstyle='round', zorder=4)
        ax.scatter(np.full(5, x0), vals, s=[8, 10, 36, 10, 8], color='white', edgecolor=color, linewidth=1.1, zorder=5)
    if metric == 'eli':
        ax.axhline(0, color='#BEC5CD', lw=0.7, ls=(0, (3, 3)), zorder=0)
    ax.set_xlim(-0.59, 1.59)
    ax.set_ylim(*limits)
    ax.set_xticks([0, 1], ['South', 'North'])
    ax.tick_params(axis='x', length=0, pad=7, labelsize=11.5)
    ax.tick_params(axis='y', length=3.5, width=0.7, pad=4, labelsize=10.5)
    ax.set_yticks(np.arange(-0.4, 0.61, 0.2) if metric == 'eli' else np.arange(0, 1.01, 0.2))
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_linewidth(0.7)

def _contrast(ax, table, limit):
    q = table['quantile'].to_numpy() * 100
    delta = table['north_minus_south'].to_numpy()
    ax.axhline(0, color='#A4ADB7', lw=0.7, ls=(0, (3, 3)))
    ax.fill_between(q, 0, delta, where=delta >= 0, interpolate=True, color=NORTH, alpha=0.18, linewidth=0)
    ax.fill_between(q, 0, delta, where=delta <= 0, interpolate=True, color=SOUTH, alpha=0.18, linewidth=0)
    ax.plot(q, delta, color=INK, lw=1.05)
    mid = table.iloc[45]
    ax.scatter([50], [mid.north_minus_south], s=17, color=INK, edgecolor='white', linewidth=0.5, zorder=4)
    ax.set_xlim(5, 95)
    ax.set_ylim(-limit, limit)
    ax.set_xticks([10, 50, 90])
    ax.set_yticks([-0.2, 0, 0.2])
    ax.set_ylabel('$\\Delta(q)$', fontsize=11.5, labelpad=3)
    ax.set_xlabel('Percentile', fontsize=10.5, labelpad=3)
    ax.tick_params(axis='both', length=3, width=0.65, pad=3, labelsize=10.5)
    ax.spines['bottom'].set_linewidth(0.65)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)

def build_distribution_figure(df):
    table = quantile_table(df)
    tests = welch_table(df).set_index('metric')
    limit = max(0.25, np.ceil(table.north_minus_south.abs().max() * 20) / 20)
    fig = plt.figure(figsize=(11.2, 11.65), dpi=300)
    gs = fig.add_gridspec(3, 3, left=0.065, right=0.98, bottom=0.065, top=0.885, wspace=0.32, hspace=0.47)
    for i, (metric, title, unit) in enumerate(PANELS):
        inner = gs[i // 3, i % 3].subgridspec(2, 1, height_ratios=[3.5, 0.85], hspace=0.42)
        main = fig.add_subplot(inner[0])
        shift = fig.add_subplot(inner[1])
        south = df.loc[df.ns_group == 'South', metric].to_numpy()
        north = df.loc[df.ns_group == 'North', metric].to_numpy()
        _bridge(main, south, north, metric)
        main.set_ylabel(unit, fontsize=11.5, labelpad=5)
        main.text(-0.12, 1.29, chr(97 + i), transform=main.transAxes, fontsize=17, weight='bold', ha='left', va='bottom')
        main.text(0, 1.29, title, transform=main.transAxes, fontsize=14, ha='left', va='bottom')
        median_delta = float(np.median(north) - np.median(south))
        main.text(1.0, 1.065, f'Median shift  {median_delta:+.3f}', transform=main.transAxes, fontsize=10.5, ha='right', color=MUTED)
        p_holm = float(tests.loc[metric, 'p_holm'])
        comparison = '< 0.001' if p_holm < 0.001 else f'= {p_holm:.3f}'
        main.text(1.0, 1.16, 'Welch $P_{\\mathrm{Holm}}$ ' + comparison, transform=main.transAxes, fontsize=9.5, ha='right', color=MUTED)
        _contrast(shift, table.loc[table.metric == metric], limit)
        for ax in (main, shift):
            pos = ax.get_position()
            ax.set_position([pos.x0 + pos.width * 0.075, pos.y0, pos.width * 0.85, pos.height])
    fig.welch_statistics = tests.reset_index()
    return fig
