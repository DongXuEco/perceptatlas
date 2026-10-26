from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
from pathlib import Path
import hashlib
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from Extended_Data_Fig_2_distribution_functions import PANELS, _density
ROOT = PACKAGE_ROOT
INPUT = ROOT / 'data/Extended_Data_Fig_2'
OUT = output_dir('Extended_Data_Fig_2')
OUT.mkdir(parents=True, exist_ok=True)
FORMAL = ROOT / 'All figures/最终图/正文图+扩展数据图/主图2-Q2.1.png'
SOUTH, NORTH, INK, GREY = ('#CB7849', '#356F9B', '#252D34', '#DDE2E6')
plt.rcParams.update({'font.family': 'Arial', 'font.size': 7, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.55, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'text.color': INK, 'axes.unicode_minus': True})
data = pd.read_csv(resource(INPUT / 'analysis_city_data.csv'))
stats = pd.read_csv(resource(INPUT / 'welch_statistics.csv'), dtype={'p_raw': str, 'p_holm': str})
statistics = stats.set_index('metric')
assert data.id_1.is_unique and len(data) == 5905
assert data.ns_group.value_counts().to_dict() == {'North': 3803, 'South': 2102}
for row in stats.itertuples():
    for group, expected in (('South', row.mean_south), ('North', row.mean_north)):
        np.testing.assert_allclose(data.loc[data.ns_group == group, row.metric].mean(), expected, rtol=1e-12, atol=1e-14)

def groups(metric):
    return [data.loc[data.ns_group == group, metric].to_numpy() for group in ('South', 'North')]

def domain(metric):
    return (-0.5, 0.65) if metric == 'eli' else (0, 1)

def p_label(metric):
    p = float(statistics.loc[metric, 'p_holm'])
    return '$P_{\\mathrm{Holm}}$ ' + ('< 0.001' if p < 0.001 else f'= {p:.3f}')

def legend(fig, y=0.965):
    handles = [Line2D([], [], color=SOUTH, lw=1.5, marker='o', mfc='white', ms=3, label='Global South  (n = 2,102)'), Line2D([], [], color=NORTH, lw=1.5, marker='s', mfc='white', ms=3, label='Global North  (n = 3,803)')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.52, y), ncol=2, frameon=False, fontsize=7, handlelength=2, columnspacing=3)

def rails(ax, metric, compact=False):
    south, north = groups(metric)
    lo, hi = domain(metric)
    grid = np.linspace(lo, hi, 501)
    pooled = np.r_[south, north]
    bandwidth = np.std(pooled, ddof=1) * len(pooled) ** (-0.2)
    densities = [_density(v, grid, (-1, 1) if metric == 'eli' else (0, 1), bandwidth) for v in (south, north)]
    scale = 0.41 / max((d.max() for d in densities))
    for baseline, values, density, color, marker in zip((0.67, -0.12), (south, north), densities, (SOUTH, NORTH), ('o', 's')):
        ax.fill_between(grid, baseline, baseline + density * scale, color=color, alpha=0.19, linewidth=0)
        ax.plot(grid, baseline + density * scale, color=color, lw=0.9)
        q = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
        ax.plot(q[[0, 4]], [baseline, baseline], color=color, lw=0.7)
        ax.plot(q[[1, 3]], [baseline, baseline], color=color, lw=3, solid_capstyle='butt')
        ax.plot(q[2], baseline, marker=marker, ms=3.5 if compact else 4.3, mfc='white', mec=color, mew=0.9, zorder=4)
    ax.set(xlim=(lo, hi), ylim=(-0.32, 1.18), yticks=[])
    ax.spines[['left', 'bottom']].set_visible(False)
    ax.tick_params(axis='x', length=2.2, width=0.5, pad=2, labelsize=6.4)
    ax.set_xticks([-0.4, 0, 0.4] if metric == 'eli' else [0, 0.5, 1])
    if not compact:
        ax.set_xlabel('ELI' if metric == 'eli' else 'Perception score', fontsize=6.6, labelpad=2)

def quantile_lens(ax, metric):
    south, north = groups(metric)
    q = np.linspace(0.05, 0.95, 181)
    qs, qn = [np.quantile(values, q) for values in (south, north)]
    ax.fill_betweenx(q * 100, qs, qn, color='#CBD2D8', alpha=0.35, linewidth=0)
    for values, color, marker in ((qs, SOUTH, 'o'), (qn, NORTH, 's')):
        ax.plot(values, q * 100, color=color, lw=1.2)
        for percentile in (0.25, 0.5, 0.75):
            value = np.interp(percentile, q, values)
            ax.plot(value, percentile * 100, marker=marker, ms=3.4 if percentile == 0.5 else 2.4, mfc='white', mec=color, mew=0.8, zorder=3)
    ax.axhline(50, color=GREY, lw=0.55, ls=(0, (2, 3)), zorder=0)
    ax.set(xlim=domain(metric), ylim=(3, 97), yticks=[10, 50, 90])
    ax.set_xticks([-0.4, 0, 0.4] if metric == 'eli' else [0, 0.5, 1])
    ax.set_xlabel('ELI' if metric == 'eli' else 'Perception score', fontsize=6.6, labelpad=2)
    ax.set_ylabel('Percentile', fontsize=6.6, labelpad=3)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='both', length=2.2, width=0.5, pad=2, labelsize=6.4)
    ax.tick_params(axis='y', length=0)

def contrast(ax, metric, text=True):
    row = statistics.loc[metric]
    ax.axvline(0, color='#A9B2BA', lw=0.65, ls=(0, (2, 3)), zorder=0)
    ax.plot([row.ci_low, row.ci_high], [0.38, 0.38], color=INK, lw=1.0, solid_capstyle='butt')
    ax.vlines([row.ci_low, row.ci_high], 0.28, 0.48, color=INK, lw=0.65)
    ax.plot(row.mean_difference, 0.38, 'D', color=INK, ms=1.8, zorder=3)
    ax.set(xlim=(-0.025, 0.18), ylim=(0, 1), yticks=[], xticks=[0, 0.1])
    ax.spines[['left', 'bottom']].set_visible(False)
    ax.tick_params(axis='x', labelsize=6.1, length=2, width=0.5, pad=1)
    if text:
        digits = 4 if abs(row.mean_difference) < 0.001 else 3
        ax.text(0, 1.02, '$\\Delta_{\\mathrm{mean}}$ ' + f'{row.mean_difference:+.{digits}f}', transform=ax.transAxes, ha='left', va='bottom', fontsize=6.8)
        ax.text(1, 1.02, p_label(metric), transform=ax.transAxes, ha='right', va='bottom', fontsize=6.4)

def grid_figure(style):
    fig = plt.figure(figsize=(7.2, 8.6), dpi=200)
    legend(fig)
    gs = fig.add_gridspec(3, 3, left=0.075, right=0.985, bottom=0.075, top=0.905, wspace=0.32, hspace=0.5)
    for i, (metric, title, _) in enumerate(PANELS):
        cell = gs[i // 3, i % 3].subgridspec(2, 1, height_ratios=[3.3, 0.65], hspace=0.62)
        main, effect = (fig.add_subplot(cell[0]), fig.add_subplot(cell[1]))
        main.text(-0.13, 1.13, chr(97 + i), transform=main.transAxes, fontsize=10, weight='bold', va='bottom')
        main.text(0, 1.13, title, transform=main.transAxes, fontsize=8.0, va='bottom')
        (rails if style == 'A' else quantile_lens)(main, metric)
        contrast(effect, metric)
    note = 'Profiles: density; bars: 5–95% / 25–75%; hollow markers: medians.\n' if style == 'A' else 'Curves: empirical 5th–95th percentiles; hollow markers: 25th, 50th and 75th.\n'
    fig.text(0.075, 0.019, note + 'Diamonds and intervals: mean difference (North − South), 95% Welch CI.  P: Holm across nine tests.', fontsize=6.3, va='bottom', linespacing=1.6)
    return fig

def row_figure():
    fig = plt.figure(figsize=(7.2, 7.3), dpi=200)
    legend(fig, 0.984)
    fig.text(0.225, 0.924, 'City-score distributions', fontsize=7.4)
    fig.text(0.677, 0.924, 'Mean difference [95% CI]', fontsize=7.4)
    fig.text(0.975, 0.924, '$P_{\\mathrm{Holm}}$', fontsize=7.4, ha='right')
    ytop, rowheight = (0.893, 0.088)
    for i, (metric, title, _) in enumerate(PANELS):
        y = ytop - (i + 1) * rowheight
        main = fig.add_axes([0.225, y + 0.007, 0.34, 0.067])
        rails(main, metric, compact=True)
        if i not in (0, 8):
            main.tick_params(axis='x', labelbottom=False)
        effect = fig.add_axes([0.677, y + 0.021, 0.205, 0.047])
        contrast(effect, metric, text=False)
        effect.tick_params(axis='x', labelbottom=i == 8)
        yc = y + 0.044
        fig.text(0.037, yc, chr(97 + i), weight='bold', fontsize=9, va='center')
        fig.text(0.066, yc, title.replace('Emotional livability', 'Emotional\nlivability'), fontsize=7.6, va='center', linespacing=1.4)
        p = float(statistics.loc[metric, 'p_holm'])
        fig.text(0.977, yc, '<0.001' if p < 0.001 else f'{p:.3f}', ha='right', va='center', fontsize=7)
        delta = statistics.loc[metric, 'mean_difference']
        digits = 4 if abs(delta) < 0.001 else 3
        fig.text(0.885, yc, f'{delta:+.{digits}f}', va='center', fontsize=6.4)
        if i == 2:
            fig.add_artist(Line2D([0.035, 0.98], [y - 0.004, y - 0.004], transform=fig.transFigure, color=GREY, lw=0.6))
    fig.text(0.225, 0.073, 'Score (0–1); ELI uses its own labelled axis', fontsize=6.1)
    fig.text(0.677, 0.073, 'North − South', fontsize=6.3)
    fig.text(0.037, 0.026, 'Profiles: density; bars: 5–95% / 25–75%; hollow markers: medians.\nDiamonds: mean differences; intervals: pointwise 95% Welch CI; P: Holm across nine tests.', fontsize=6.3, linespacing=1.6)
    return fig

def save(fig, name):
    fig.canvas.draw()
    for suffix in ('.png', '.pdf', '.svg'):
        fig.savefig(OUT / (name + suffix), dpi=450, facecolor='white')
    fig.savefig(OUT / (name + '_preview.png'), dpi=180, facecolor='white')
    plt.close(fig)
