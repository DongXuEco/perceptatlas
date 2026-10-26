from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
from pathlib import Path
import hashlib
import json
import shutil
import sys
ROOT = PACKAGE_ROOT
import Fig_3_quantile_density as prior
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter
original = prior.original
OUT = output_dir('Fig_3')
FINAL = prior.FINAL
COLORS = prior.COLORS
ORDER = list(original.TYPOLOGY_COLORS)
SHORT = [original.TYPOLOGY_SHORT[t] for t in ORDER]
TEAL = '#247C87'
INK = '#263746'
OUT.mkdir(exist_ok=True, parents=True)

def sha(path):
    return hashlib.sha256(resource(path).read_bytes()).hexdigest()

def fresh_current():
    table = pd.read_csv(resource(prior.OUT / 'city_inequality_data.csv'))
    fig = original.build_figure()
    fig.set_dpi(450)
    axes = prior.panel_axes(fig)
    bootstrap = None
    for letter, metric in prior.PANELS:
        prior.draw_panel(axes[letter], table, letter, metric, bootstrap)
    handles = [Line2D([], [], color=COLORS[g], marker=m, mfc='white', ms=2.3, lw=1, label=f'{g} (n={len(table.loc[table.ns_group == g]):,})') for g, m in zip(prior.GROUPS, ('o', 's'))]
    axes['a'].legend(handles=handles, loc='lower right', frameon=False, fontsize=4.7 * 1.21, handlelength=1.4, handletextpad=0.5, borderaxespad=0.3, labelspacing=0.4)
    for group, y in (('South', 0.31), ('North', 0.27)):
        cax = axes['a'].inset_axes([0.65, y, 0.32, 0.018])
        cmap = LinearSegmentedColormap.from_list(group + 'density', ['white', np.array(to_rgb(COLORS[group])) * 0.82 + 0.18])
        cbar = fig.colorbar(plt.matplotlib.cm.ScalarMappable(norm=prior.NORM, cmap=cmap), cax=cax, orientation='horizontal', ticks=[] if group == 'South' else [0, 1])
        cbar.outline.set_linewidth(0.3)
        cbar.ax.tick_params(labelsize=4.3 * 1.21, length=1, pad=1, width=0.3)
        if group == 'South':
            cbar.ax.set_title('Rel. density', fontsize=4.4 * 1.21, pad=2)
    return (fig, axes, table)

def clear_panel(ax, letter):
    label = next((t for t in ax.texts if t.get_text() == letter))
    label_spec = (label.get_position(), label.get_fontsize(), label.get_fontweight())
    title, size, transform = (ax.title.get_text(), ax.title.get_fontsize(), ax.title.get_transform())
    ax.clear()
    ax.set_title(title, fontsize=size, pad=3)
    ax.title.set_transform(transform)
    ax.text(*label_spec[0], letter, transform=ax.transAxes, fontsize=label_spec[1], fontweight=label_spec[2], ha='left', va='bottom', color='black')

def typology_data(table):
    comp = pd.crosstab(table.ns_group, table.typology).reindex(index=['North', 'South'], columns=ORDER)
    assert np.all(comp.sum(axis=1).to_numpy() == [3779, 2101])
    shares = comp.div(comp.sum(axis=1), axis=0)
    rng = np.random.default_rng(524)
    intervals, boundaries = ({}, {})
    for group in shares.index:
        vals = table.loc[table.ns_group == group, 'typology'].to_numpy()
        indices = rng.integers(0, len(vals), size=(800, len(vals)))
        draws = np.column_stack([(vals[indices] == typ).mean(axis=1) for typ in ORDER])
        intervals[group] = np.percentile(draws, [2.5, 97.5], axis=0).T
        boundaries[group] = np.percentile(np.cumsum(draws[:, :-1], axis=1), [2.5, 50, 97.5], axis=0).T
    old = original.bootstrap_typology_boundaries(table, ORDER, list(shares.index))
    assert all((np.array_equal(boundaries[g], old[g]) for g in shares.index))
    rows = [dict(group=g, category=typ, abbreviation=SHORT[j], count=int(comp.loc[g, typ]), n=int(comp.loc[g].sum()), share=float(shares.loc[g, typ]), ci95_low=float(intervals[g][j, 0]), ci95_high=float(intervals[g][j, 1])) for g in shares.index for j, typ in enumerate(ORDER)]
    return (shares, intervals, rows)

def draw_i(ax, table, fig, old_cbar_position):
    cols = ['compound_gini', 'compound_theil'] + ['gini_' + e for e in original.EMOTIONS]
    labels = ['Gini', 'Theil'] + list(original.EMOTIONS.values())
    mat = table[cols].corr().to_numpy()
    assert np.allclose(mat, mat.T) and mat.min() < 0
    rows = []
    cmap = LinearSegmentedColormap.from_list('correlation_magnitude', ['#F7FAFB', '#AED7D9', TEAL])
    for row in range(8):
        for col in range(8):
            r = float(mat[row, col])
            if row == col:
                ax.add_patch(Rectangle((col - 0.45, row - 0.45), 0.9, 0.9, color='#EFF3F5', lw=0))
                ax.plot([col - 0.22, col + 0.22], [row + 0.22, row - 0.22], color='#B7C4CB', lw=1)
            elif row < col:
                major, minor = (0.8 * np.sqrt((1 + abs(r)) / 2), 0.8 * np.sqrt((1 - abs(r)) / 2))
                ax.add_patch(Ellipse((col, row), major, minor, angle=-45 if r >= 0 else 45, facecolor=cmap(abs(r)), edgecolor=TEAL, lw=0.3, alpha=0.95))
            else:
                p = original.corr_pvalue_approx(r, len(table[cols].dropna(subset=[cols[row], cols[col]])))
                stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                ax.add_patch(Rectangle((col - 0.45, row - 0.45), 0.9, 0.9, facecolor=cmap(abs(r)), edgecolor='none', alpha=0.13))
                ax.text(col, row - 0.07, f'{r:.2f}', ha='center', va='center', fontsize=5.8, color=INK)
                if stars:
                    ax.text(col, row + 0.24, stars, ha='center', va='center', fontsize=4.6, color='#6A838B')
                rows.append(dict(variable_1=labels[row], variable_2=labels[col], pearson_r=r, n=len(table), p_approx=p, stars=stars))
    for edge in np.arange(0.5, 7.5, 1):
        ax.axhline(edge, lw=0.23, color='#E7EEF1', zorder=0)
        ax.axvline(edge, lw=0.23, color='#E7EEF1', zorder=0)
    ax.set(xlim=(-0.5, 7.5), ylim=(7.5, -0.5), xticks=range(8), yticks=range(8))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=5.8)
    ax.set_yticklabels(labels, fontsize=5.8)
    ax.tick_params(length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cax = fig.add_axes(old_cbar_position)
    cb = fig.colorbar(plt.matplotlib.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap), cax=cax, ticks=[0, 0.5, 1])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5.5, length=0, pad=2)
    cb.ax.set_title('|r|', fontsize=5.5, pad=4)
    return rows

def joint_density(table):
    xy = table[['eli', 'compound_gini']].to_numpy()
    bw = xy.std(axis=0, ddof=1) * len(xy) ** (-1 / 6)
    xedge = np.linspace(xy[:, 0].min() - 4 * bw[0], xy[:, 0].max() + 4 * bw[0], 451)
    yedge = np.linspace(0, xy[:, 1].max() + 4 * bw[1], 401)
    dx, dy = (np.diff(xedge)[0], np.diff(yedge)[0])
    xc, yc = ((xedge[1:] + xedge[:-1]) / 2, (yedge[1:] + yedge[:-1]) / 2)
    fields, levels = ({}, {})
    for group in ('South', 'North'):
        v = table.loc[table.ns_group == group, ['eli', 'compound_gini']].to_numpy()
        hist = np.histogram2d(v[:, 0], v[:, 1], bins=(xedge, yedge))[0].T
        assert hist.sum() == len(v)
        density = gaussian_filter(hist, sigma=(bw[1] / dy, bw[0] / dx), mode='reflect') / (len(v) * dx * dy)
        assert np.isclose(density.sum() * dx * dy, 1)
        ranked = np.sort(density.ravel())[::-1]
        cum = np.cumsum(ranked) * dx * dy
        thresholds = [ranked[np.searchsorted(cum, mass)] for mass in (0.5, 0.8)]
        levels[group] = dict(zip(('50', '80'), map(float, thresholds)))
        fields[group] = density
    return (xc, yc, fields, levels, bw, (dx, dy))

def draw_j(ax, table, fig):
    xc, yc, fields, levels, bw, (dx, dy) = joint_density(table)
    ax.set_axis_off()
    main = ax.inset_axes([0, 0, 0.82, 0.8])
    top = ax.inset_axes([0, 0.84, 0.82, 0.14])
    right = ax.inset_axes([0.86, 0, 0.14, 0.8])
    xlim = tuple(table.eli.quantile([0.01, 0.99]))
    ylim = tuple(table.compound_gini.quantile([0.01, 0.99]))
    total = sum((fields[g] * (table.ns_group == g).mean() for g in fields))
    rgba = np.zeros((*total.shape, 4))
    for group, field in fields.items():
        weighted = field * (table.ns_group == group).mean()
        rgba[:, :, :3] += weighted[:, :, None] * np.asarray(to_rgb(COLORS[group]))
    np.divide(rgba[:, :, :3], total[:, :, None], out=rgba[:, :, :3], where=total[:, :, None] > 0)
    rgba[:, :, 3] = 0.32 * (total / total.max()) ** 0.65
    main.imshow(rgba, origin='lower', extent=(xc[0] - dx / 2, xc[-1] + dx / 2, yc[0] - dy / 2, yc[-1] + dy / 2), aspect='auto', zorder=0)
    xm, ym = (table.eli.median(), table.compound_gini.median())
    for group in ('South', 'North'):
        field = fields[group]
        main.contour(xc, yc, field, levels=[levels[group]['80'], levels[group]['50']], colors=[COLORS[group]], linewidths=[0.65, 1.0], linestyles=['--', '-'])
        top.fill_between(xc, field.sum(axis=0) * dy, color=COLORS[group], alpha=0.12, lw=0)
        top.plot(xc, field.sum(axis=0) * dy, color=COLORS[group], lw=0.7)
        right.fill_betweenx(yc, field.sum(axis=1) * dx, color=COLORS[group], alpha=0.12, lw=0)
        right.plot(field.sum(axis=1) * dx, yc, color=COLORS[group], lw=0.7)
        med = table.loc[table.ns_group == group, ['eli', 'compound_gini']].median().to_numpy()
        main.plot(*med, marker='o' if group == 'South' else 's', ms=3.8, color=COLORS[group], mec='white', mew=0.6, zorder=6)
        offset = (-7, 8) if group == 'South' else (7, -7)
        main.annotate(group, med, xytext=offset, textcoords='offset points', color=COLORS[group], ha='right' if group == 'South' else 'left', fontsize=5.4, bbox=dict(facecolor='white', edgecolor='none', alpha=0.65, pad=0.35))
    main.axvline(xm, color='#96A4AE', ls=(0, (3, 3)), lw=0.5)
    main.axhline(ym, color='#96A4AE', ls=(0, (3, 3)), lw=0.5)
    quad = {'HM/HI': (table.eli >= xm) & (table.compound_gini >= ym), 'HM/LI': (table.eli >= xm) & (table.compound_gini < ym), 'LM/HI': (table.eli < xm) & (table.compound_gini >= ym), 'LM/LI': (table.eli < xm) & (table.compound_gini < ym)}
    for label, (x, y, ha, va) in {'LM/HI': (0.025, 0.975, 'left', 'top'), 'HM/HI': (0.975, 0.975, 'right', 'top'), 'LM/LI': (0.025, 0.025, 'left', 'bottom'), 'HM/LI': (0.975, 0.025, 'right', 'bottom')}.items():
        main.text(x, y, f'{label}\n{100 * quad[label].mean():.1f}%', transform=main.transAxes, ha=ha, va=va, color=INK, fontsize=5.0, zorder=8)
    main.set(xlim=xlim, ylim=ylim)
    main.set_xlabel('Mean ELI', fontsize=7.0, labelpad=2)
    main.set_ylabel('Composite Gini', fontsize=7.0, labelpad=2)
    main.xaxis.set_major_locator(MultipleLocator(0.2))
    main.yaxis.set_major_locator(MultipleLocator(0.04))
    main.tick_params(labelsize=6, width=0.5, length=2, pad=2)
    for sp in ('left', 'bottom'):
        main.spines[sp].set_linewidth(0.55)
    top.set_xlim(xlim)
    right.set_ylim(ylim)
    top.set_ylim(bottom=0)
    right.set_xlim(left=0)
    for margin in (top, right):
        margin.set_axis_off()
    main.legend(handles=[Line2D([], [], color='#637887', lw=0.8, label='50%'), Line2D([], [], color='#637887', lw=0.6, ls='--', label='80%')], loc='lower center', bbox_to_anchor=(0.5, 0.02), fontsize=4.7, frameon=False, handlelength=1.2, handletextpad=0.4, labelspacing=0.15, borderpad=0)
    return dict(display_xlim=list(xlim), display_ylim=list(ylim), points_outside_view=int(((table.eli < xlim[0]) | (table.eli > xlim[1]) | (table.compound_gini < ylim[0]) | (table.compound_gini > ylim[1])).sum()), bandwidth=list(bw), levels=levels, quadrant_shares={k: float(v.mean()) for k, v in quad.items()}, contour_mass='50% solid; 80% dashed; within-group density, not CI')

def draw_k(ax, shares, intervals, variant='paired'):
    if variant == 'paired':
        for j, typ in enumerate(ORDER):
            y = 3 - j
            ax.axhspan(y - 0.4, y + 0.4, color='#F4F7F9' if j % 2 == 0 else 'white', zorder=0)
            for group, offset, marker in (('South', 0.105, 'o'), ('North', -0.105, 's')):
                value = shares.loc[group, typ] * 100
                lo, hi = intervals[group][j] * 100
                yy = y + offset
                ax.plot([0, value], [yy, yy], color=COLORS[group], lw=3.4, alpha=0.16, solid_capstyle='round')
                ax.plot([lo, hi], [yy, yy], color=COLORS[group], lw=1.15, zorder=3)
                ax.plot([lo, lo, np.nan, hi, hi], [yy - 0.045, yy + 0.045, np.nan, yy - 0.045, yy + 0.045], color=COLORS[group], lw=0.55, zorder=3)
                ax.plot(value, yy, marker=marker, ms=3.5, mfc=COLORS[group], mec='white', mew=0.5, zorder=4)
                ax.text(hi + 1.0, yy, f'{value:.1f}', color=COLORS[group], va='center', fontsize=5.7)
        ax.set(yticks=[3, 2, 1, 0], yticklabels=SHORT, xlim=(0, 50), ylim=(-0.55, 3.8))
        ax.set_xlabel('Cities (%)', fontsize=7.0, labelpad=3)
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.legend(handles=[Line2D([], [], color=COLORS[g], marker=m, ms=3, lw=0.7, label=g) for g, m in (('South', 'o'), ('North', 's'))], loc='upper right', bbox_to_anchor=(1, 1.02), ncol=2, fontsize=5.2, columnspacing=0.8, handletextpad=0.4, handlelength=1.2, borderpad=0)
    elif variant == 'bars':
        for group, offset in (('South', -0.18), ('North', 0.18)):
            vals = shares.loc[group].to_numpy() * 100
            ax.bar(np.arange(4) + offset, vals, width=0.3, color=COLORS[group], alpha=0.75, lw=0, label=group)
            ci = intervals[group] * 100
            ax.vlines(np.arange(4) + offset, ci[:, 0], ci[:, 1], colors=INK, lw=0.55)
        ax.set(xticks=range(4), xticklabels=SHORT, ylim=(0, 50))
        ax.set_ylabel('Cities (%)', fontsize=7)
        ax.legend(loc='upper right', fontsize=5.5)
    else:
        catcolors = ['#0D47A1', '#75A8DC', '#F5C3A1', '#F27C2B']
        for y, group in enumerate(('North', 'South')):
            left = 0
            for j, typ in enumerate(ORDER):
                value = shares.loc[group, typ] * 100
                ax.barh(y, value, left=left, height=0.33, color=catcolors[j], edgecolor='white', lw=0.4)
                ax.text(left + value / 2, y, f'{value:.1f}', fontsize=5.7, ha='center', va='center', color='white' if j == 0 else INK)
                left += value
        for j, label in enumerate(SHORT):
            ax.text(0.125 + 0.25 * j, 0.97, label, transform=ax.transAxes, ha='center', fontsize=5.5)
        ax.set(yticks=[0, 1], yticklabels=['North', 'South'], xlim=(0, 100), ylim=(-0.45, 1.8))
        ax.set_xlabel('Cities (%)', fontsize=7)
    ax.tick_params(axis='both', labelsize=6.0, length=2, width=0.5, pad=3)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.55)
