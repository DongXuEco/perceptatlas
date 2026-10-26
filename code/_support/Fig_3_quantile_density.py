from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
from pathlib import Path
import hashlib
import json
import shutil
import sys
ROOT = PACKAGE_ROOT
import Fig_3_base as original
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np
import pandas as pd
from PIL import Image, ImageChops, ImageOps
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata
OUT = output_dir('Fig_3')
OUT.mkdir(parents=True, exist_ok=True)
FINAL = ROOT / 'All figures/最终图/正文图+扩展数据图/主图3-Q2.2.png'
CODE = ROOT / 'All figures/最终图/code'
GROUPS = ('South', 'North')
COLORS = {'South': '#F27C2B', 'North': '#0D47A1'}
PANELS = [('a', 'compound_gini')] + list(zip('cdefgh', ['gini_' + e for e in original.EMOTIONS]))
Q = np.linspace(0.05, 0.95, 181)
N_BOOT, SEED = (5000, 20261023)
NORM = PowerNorm(0.65, vmin=0, vmax=1)

def sha(path):
    return hashlib.sha256(resource(path).read_bytes()).hexdigest()

def panel_axes(fig):
    return {text.get_text(): ax for ax in fig.axes for text in ax.texts if text.get_text() in list('abcdefghijk')}

def compute():
    table = pd.read_csv(resource(OUT / 'city_inequality_data.csv'))
    streams = np.random.SeedSequence(SEED).spawn(2)
    arrays, rows = ({}, [])
    for group, stream in zip(GROUPS, streams):
        values = table.loc[table.ns_group == group, [metric for _, metric in PANELS]].to_numpy()
        rng = np.random.default_rng(stream)
        draws = np.empty((N_BOOT, len(PANELS), len(Q)))
        for start in range(0, N_BOOT, 100):
            index = rng.integers(0, len(values), size=(100, len(values)))
            for j in range(len(PANELS)):
                draws[start:start + 100, j] = np.quantile(values[:, j][index], Q, axis=1).T
        arrays[group] = draws
        low, high = np.quantile(draws, [0.025, 0.975], axis=0)
        for j, (letter, metric) in enumerate(PANELS):
            estimate = np.quantile(values[:, j], Q)
            assert np.all(np.diff(low[j]) >= 0) and np.all(np.diff(high[j]) >= 0)
            for k, q in enumerate(Q):
                rows.append(dict(panel=letter, metric=metric, group=group, n=len(values), percentile=q * 100, estimate=estimate[k], ci95_low=low[j, k], ci95_high=high[j, k]))
        print(f'Bootstrap: {group}, {len(values)} cities, seven Gini outcomes', flush=True)
    np.savez_compressed(OUT / 'bootstrap_quantiles.npz', **arrays)
    pd.DataFrame(rows).to_csv(OUT / 'quantile_confidence_intervals.csv', index=False, encoding='utf-8-sig')

def draw_panel(ax, table, letter, metric, bootstrap):
    label = next((text for text in ax.texts if text.get_text() == letter))
    difference_text = next((text.get_text() for text in ax.texts if 'Δ=' in text.get_text()))
    old_max = ax.get_ylim()[1]
    for artist in list(ax.lines) + list(ax.collections) + list(ax.patches) + list(ax.images):
        artist.remove()
    for text in list(ax.texts):
        if text is not label:
            text.remove()
    groups = [table.loc[table.ns_group == group, metric].to_numpy() for group in GROUPS]
    scores = np.concatenate(groups)
    ranks = [(rankdata(values) - 0.5) / len(values) * 100 for values in groups]
    rank_all = np.concatenate(ranks)
    xedges, yedges = (np.linspace(0, 1, 901), np.linspace(0, 100, 401))
    dx, dy = (np.diff(xedges)[0], np.diff(yedges)[0])
    factor = len(scores) ** (-1 / 6)
    sx, sy = (np.std(scores, ddof=1) * factor, np.std(rank_all, ddof=1) * factor)
    fields = []
    for values, rank in zip(groups, ranks):
        hist, _, _ = np.histogram2d(values, rank, bins=(xedges, yedges))
        assert int(hist.sum()) == len(values)
        fields.append(gaussian_filter(hist.T, sigma=(sy / dy, sx / dx), mode='reflect') / (len(scores) * dx * dy))
    total = sum(fields)
    assert abs(total.sum() * dx * dy - 1) < 1e-10
    relative = total / total.max()
    rgba = np.zeros((*total.shape, 4))
    for field, group in zip(fields, GROUPS):
        rgba[:, :, :3] += field[:, :, None] * np.array(to_rgb(COLORS[group]))
    np.divide(rgba[:, :, :3], total[:, :, None], out=rgba[:, :, :3], where=total[:, :, None] > 0)
    rgba[:, :, 3] = 0.82 * NORM(np.clip(relative, 0, 1))
    ax.imshow(rgba, origin='lower', extent=(0, 1, 0, 100), aspect='auto', interpolation='bilinear', zorder=0)
    j = [p[0] for p in PANELS].index(letter)
    curves = [np.quantile(values, Q) for values in groups]
    central = (Q >= 0.25 - 1e-12) & (Q <= 0.75 + 1e-12)
    for percentile in (0.05, 0.25, 0.5, 0.75, 0.95):
        endpoints = [np.interp(percentile, Q, curve) for curve in curves]
        ax.plot(endpoints, [percentile * 100] * 2, color='#8293A2' if percentile == 0.5 else '#CDD6DE', lw=0.6 if percentile == 0.5 else 0.35, zorder=1)
    for group, curve, marker in zip(GROUPS, curves, ('o', 's')):
        color = COLORS[group]
        ci = pd.read_csv(resource(PACKAGE_ROOT / 'data/Fig_3/Fig_3_quantile_confidence_intervals.csv'))
        ci = ci.loc[(ci['metric'] == metric) & (ci['group'] == group)].sort_values('percentile')
        low, high = (ci['ci95_low'].to_numpy(), ci['ci95_high'].to_numpy())
        ax.fill_betweenx(Q * 100, low, high, color=color, alpha=0.1, zorder=1.3)
        ax.plot(low, Q * 100, color=color, lw=0.3, alpha=0.75, zorder=1.5)
        ax.plot(high, Q * 100, color=color, lw=0.3, alpha=0.75, zorder=1.5)
        for x, y, width in ((curve, Q * 100, 0.85), (curve[central], Q[central] * 100, 1.5)):
            line, = ax.plot(x, y, color=color, lw=width, solid_capstyle='round', zorder=3)
            line.set_path_effects([pe.Stroke(linewidth=width + 0.65, foreground='white', alpha=0.88), pe.Normal()])
        for percentile in (0.05, 0.25, 0.5, 0.75, 0.95):
            median = percentile == 0.5
            ax.plot(np.interp(percentile, Q, curve), percentile * 100, marker=marker, ms=3.2 if median else 2.1, mfc=color if median else 'white', mec='white' if median else color, mew=0.4 if median else 0.65, zorder=5)
    ax.set(xlim=(0, old_max), ylim=(3, 97), yticks=[5, 25, 50, 75, 95])
    ax.set_xlabel('Composite Gini' if letter == 'a' else 'Gini', fontsize=6 * 1.21, labelpad=2)
    ax.set_ylabel('Percentile', fontsize=5.8 * 1.21, labelpad=3)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(axis='both', labelsize=5.5 * 1.21, length=2, width=0.45, pad=2, colors='black')
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.text(0.975 if letter == 'h' else 0.025, 0.985, difference_text, transform=ax.transAxes, ha='right' if letter == 'h' else 'left', va='top', fontsize=(4.25 if letter == 'a' else 4.45) * 1.21, color='black', linespacing=1.25, zorder=7)
    return dict(bandwidth_gini=sx, bandwidth_percentile=sy, max_gini_display=old_max, density_mass=float(total.sum() * dx * dy), preserved_mean_annotation=difference_text)
