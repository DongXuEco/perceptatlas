"""Plot Fig_3 using the packaged numerical tables or supplied local inputs."""
from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_support"))
from figure_cli import prepare_figure as _prepare_figure, save_figure as _save_figure
from figure_cli import read_table as _read_table, model_tables as _model_tables
from figure_cli import configure_land_functions as _configure_land_functions

_figure_options = _prepare_figure('Fig_3', 'analysis_tables') if __name__ == "__main__" else None

from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
from pathlib import Path
import sys, json
ROOT = PACKAGE_ROOT
import Fig_3_typology_layout as d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
OUT = output_dir('Fig_3')
OUT.mkdir(exist_ok=True, parents=True)

def xy(c, r, p):
    a = np.deg2rad(225 - 270 * np.asarray(p) / 50)
    return (c[0] + r * np.cos(a), c[1] + r * np.sin(a))

def draw(ax, shares, intervals):
    jax = next((a for a in ax.figure.axes if any((t.get_text() == 'j' for t in a.texts))))
    jax.yaxis.tick_right()
    jax.yaxis.set_label_position('right')
    jax.tick_params(axis='y', which='both', left=False, labelleft=False, right=True, labelright=True)
    jax.spines['right'].set_visible(True)
    jax.spines['right'].set_linewidth(jax.spines['left'].get_linewidth())
    jax.spines['right'].set_edgecolor(jax.spines['left'].get_edgecolor())
    jax.spines['left'].set_visible(False)
    sizes = {'title': jax.title.get_fontsize(), 'axis': jax.xaxis.label.get_fontsize(), 'tick': jax.get_xticklabels()[0].get_fontsize(), 'annotation': next((t.get_fontsize() for t in jax.texts if '\n' in t.get_text())), 'group': next((t.get_fontsize() for t in jax.texts if t.get_text() == 'South'))}
    title_transform = ax.title.get_transform()
    ax.set_title(ax.title.get_text(), y=ax.title.get_position()[1], fontsize=sizes['title'])
    ax.title.set_transform(title_transform)
    parent_height = ax.get_position().height * ax.figure.get_figheight()
    shift = 4 / 25.4 / parent_height
    legend_clearance = 2.5 / 72 / parent_height
    ax.set_axis_off()
    z = ax.inset_axes([-0.09557, -0.15323 + shift, 1.25114, 1.10473])
    z.set(xlim=(0, 2), ylim=(0, 2.1))
    z.set_aspect('equal')
    z.set_axis_off()
    for name, c in [('LM/HI', (0.5, 1.5)), ('HM/HI', (1.5, 1.5)), ('LM/LI', (0.5, 0.5)), ('HM/LI', (1.5, 0.5))]:
        j = d.SHORT.index(name)
        cat = d.ORDER[j]
        z.text(c[0], c[1] + 0.47, name, ha='center', va='center', fontsize=sizes['annotation'], color='#000000')
        for g, r, m, dy in [('South', 0.355, 'o', 0.075), ('North', 0.265, 's', -0.085)]:
            v = float(shares.loc[g, cat] * 100)
            lo, hi = intervals[g][j] * 100
            color = d.COLORS[g]
            assert 0 <= lo <= v <= hi <= 50
            z.plot(*xy(c, r, np.linspace(0, 50, 181)), color='#E9EFF3', lw=1.85, solid_capstyle='round')
            z.plot(*xy(c, r, np.linspace(0, v, 181)), color=color, lw=1.85, solid_capstyle='round')
            z.plot(*xy(c, r, np.linspace(lo, hi, 31)), color=color, lw=3, alpha=0.27, zorder=3)
            for limit in (lo, hi):
                z.plot(*xy(c, np.array([r - 0.023, r + 0.023]), limit), color=color, lw=0.5, zorder=4)
            z.plot(*xy(c, r, v), marker=m, ms=2.4, mfc=color, mec='white', mew=0.35, zorder=5)
            z.text(c[0], c[1] + dy, f'{v:.1f}', ha='center', va='center', fontsize=sizes['annotation'] * 1.2, color=color)
        for p in (0, 25, 50):
            z.plot(*xy(c, np.array([0.387, 0.408]), p), color='#B8C5CE', lw=0.4)
            if name == 'LM/HI' and p != 25:
                tx, ty = xy(c, 0.49, p)
                z.text(tx, ty, str(p), fontsize=sizes['tick'] * 1.2, ha='center', va='center', color='#000000')
    z.plot([1, 1], [0.08, 1.98], color='#E5ECF1', lw=0.45, zorder=-1)
    z.plot([0.06, 1.94], [1.08, 1.08], color='#E5ECF1', lw=0.45, zorder=-1)
    for y, t in [(1.5, 'Higher Gini'), (0.5, 'Lower Gini')]:
        z.text(-0.055, y, t, rotation=90, fontsize=sizes['axis'], ha='center', va='center', color='#000000', clip_on=False)
    for x, t in [(0.5, 'Lower ELI'), (1.5, 'Higher ELI')]:
        z.text(x, -0.07 + 4 / 25.4 / (parent_height * 1.10473) * 2.1, t, fontsize=sizes['axis'], ha='center', va='top', color='#000000', clip_on=False)
    return (ax.legend(handles=[Line2D([], [], color=d.COLORS[g], lw=1.4, marker=m, ms=2.5, label=g) for g, m in [('South', 'o'), ('North', 's')]], loc='upper center', bbox_to_anchor=(0.53, -0.25 + shift - legend_clearance), ncol=2, frameon=False, fontsize=sizes['axis'], handlelength=1.1, handletextpad=0.4, columnspacing=1, borderpad=0, borderaxespad=0), sizes)

def regularize_hexagons(ax, table):
    from matplotlib.collections import PolyCollection
    old = next((c for c in ax.collections if isinstance(c, PolyCollection)))

    def lengths(collection):
        vertices = ax.transData.transform(collection.get_paths()[0].vertices)
        return np.linalg.norm(np.diff(vertices, axis=0), axis=1)
    old_lengths = lengths(old)
    xlim, ylim = (ax.get_xlim(), ax.get_ylim())
    x = table['eli'].to_numpy()
    y = table['compound_gini'].to_numpy()
    transform = ax.transData.get_matrix()
    kx, ky = (abs(transform[0, 0]), abs(transform[1, 1]))
    nx = 34
    xmin, xmax = (float(x.min()), float(x.max()))
    sx = (xmax - xmin) * (1 + 2e-09) / nx
    sy = np.sqrt(3) * sx * kx / ky
    ny = max(1, int(np.ceil((y.max() - y.min()) / sy)))
    midpoint = float((y.min() + y.max()) / 2)
    extent = (xmin, xmax, midpoint - ny * sy / 2, midpoint + ny * sy / 2)
    cmap = old.get_cmap().copy()
    cmap.set_bad((0, 0, 0, 0))
    alpha = old.get_alpha()
    zorder = old.get_zorder() - 0.1
    old.remove()
    new = ax.hexbin(x, y, gridsize=(nx, ny), extent=extent, mincnt=1, cmap=cmap, linewidths=0, alpha=alpha, rasterized=True, zorder=zorder)
    counts = np.asarray(new.get_array(), dtype=float)
    assert int(counts.sum()) == len(table) == 5880
    threshold = d.original.MEAN_INEQUALITY_HEXBIN_MINCNT
    visible = counts >= threshold
    assert visible.any()
    new.set_array(np.ma.masked_where(~visible, counts))
    new.set_clim(float(counts[visible].min()), float(counts[visible].max()))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    edges = lengths(new)
    relative_error = float(np.ptp(edges) / edges.mean())
    assert relative_error < 1e-09, relative_error
    np.savez_compressed(OUT / 'hexagon_bins.npz', centers=new.get_offsets(), counts=counts, displayed=visible)
    return dict(previous_edge_length_ratio=float(old_lengths.max() / old_lengths.min()), new_edge_length_ratio=float(edges.max() / edges.min()), relative_edge_length_error=relative_error, edge_length_mm=float(edges.mean() / ax.figure.dpi * 25.4), gridsize=[nx, ny], extent=list(extent), total_city_count=int(counts.sum()), occupied_bins=len(counts), visible_bins=int(visible.sum()), min_count_threshold=threshold, all_cities_assigned=True, method='Rebin using a regular hexagonal grid in final display geometry; preserve original grid x-count and minimum count threshold')

def add_holm_annotations(axes, table):
    results = json.loads(resource(OUT / 'welch_holm_statistics.json').read_text(encoding='utf-8'))
    report = {}
    for result in results:
        letter = result['panel']
        ax = axes[letter]
        annotation = next((t for t in ax.texts if 'Δ=' in t.get_text()))
        old_text = annotation.get_text()
        p = result['p_holm']
        ptext = 'Holm $P$ < 0.001' if p < 0.001 else f'Holm $P$ = {p:.3f}'
        annotation.set_text(old_text + '\n' + ptext)
        annotation.set_position((0.975, 0.025))
        annotation.set_ha('right')
        annotation.set_va('bottom')
        annotation.set_linespacing(1.2)
        annotation.set_fontsize(annotation.get_fontsize() * 1.2)
        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_linewidth(ax.spines['bottom'].get_linewidth())
        ax.spines['left'].set_color('black')
        ax.tick_params(axis='y', which='major', length=2, width=0.45, direction='out', left=True, right=False, labelleft=True, labelright=False, colors='black')
        north = table.loc[table.ns_group == 'North', result['metric']].to_numpy()
        south = table.loc[table.ns_group == 'South', result['metric']].to_numpy()
        assert len(north) == result['n_north'] and len(south) == result['n_south']
        assert np.isclose(north.mean() - south.mean(), result['mean_difference'], rtol=0, atol=1e-14)
        report[letter] = dict(original_mean_annotation=old_text, holm_p=p, annotation_position=[0.975, 0.025], y_tick_length_pt=2)
    legend = axes['a'].get_legend()
    for text, group in zip(legend.get_texts(), ('South', 'North')):
        text.set_text(group)
    assert [text.get_text() for text in legend.get_texts()] == ['South', 'North']
    legend.set_loc('upper right')
    legend.set_bbox_to_anchor((0.98, 0.78), transform=axes['a'].transAxes)
    return report
import shutil
from PIL import Image, ImageOps
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import pandas as pd
stats_table = pd.read_csv(resource(d.prior.OUT / 'city_inequality_data.csv'))
results = []
for letter, metric in d.prior.PANELS:
    north = stats_table.loc[stats_table.ns_group == 'North', metric].to_numpy()
    south = stats_table.loc[stats_table.ns_group == 'South', metric].to_numpy()
    test = ttest_ind(north, south, equal_var=False, alternative='two-sided')
    ci = test.confidence_interval(0.95)
    results.append(dict(panel=letter, metric=metric, n_north=len(north), n_south=len(south), mean_north=float(north.mean()), mean_south=float(south.mean()), mean_difference=float(north.mean() - south.mean()), welch_t=float(test.statistic), welch_df=float(test.df), p_raw=float(test.pvalue), welch_ci95_low=float(ci.low), welch_ci95_high=float(ci.high)))
adjusted = multipletests([r['p_raw'] for r in results], method='holm')[1]
for result, p in zip(results, adjusted):
    result.update(p_holm=float(p), holm_family_size=7)
(OUT / 'welch_holm_statistics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
pd.DataFrame(results).to_csv(OUT / 'welch_holm_statistics.csv', index=False, encoding='utf-8-sig')
fig, axes, table = d.fresh_current()
removed_outlines = {}
for letter in 'acdefgh':
    changed = 0
    for line in axes[letter].lines:
        if line.get_path_effects():
            line.set_path_effects([])
            changed += 1
    removed_outlines[letter] = changed
assert all((count == 4 for count in removed_outlines.values())), removed_outlines
removed_curve_intervals = {}
for letter in 'acdefgh':
    ax = axes[letter]
    bands = [c for c in ax.collections if np.isclose(c.get_zorder(), 1.3)]
    boundaries = [line for line in ax.lines if np.isclose(line.get_zorder(), 1.5) and np.isclose(line.get_linewidth(), 0.3)]
    assert len(bands) == 2 and len(boundaries) == 4, (letter, len(bands), len(boundaries))
    for artist in bands + boundaries:
        artist.remove()
    removed_curve_intervals[letter] = dict(bands=len(bands), boundary_lines=len(boundaries))
holm_annotation_report = add_holm_annotations(axes, table)
d.clear_panel(axes['k'], 'k')
shares, intervals, rows = d.typology_data(table)
legend, matched_sizes = draw(axes['k'], shares, intervals)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
target = axes['j'].xaxis.label.get_window_extent(renderer).y0 + axes['j'].xaxis.label.get_window_extent(renderer).height / 2
legtext = legend.get_texts()[0].get_window_extent(renderer)
anchor = legend.get_bbox_to_anchor().transformed(axes['k'].transAxes.inverted())
new_y = anchor.y0 + (target - (legtext.y0 + legtext.height / 2)) / axes['k'].get_window_extent().height
legend.set_bbox_to_anchor((0.53, new_y), transform=axes['k'].transAxes)
fig.canvas.draw()
legtext = legend.get_texts()[0].get_window_extent(fig.canvas.get_renderer())
alignment_error = float(legtext.y0 + legtext.height / 2 - target)
assert abs(alignment_error) < 1e-06
hex_audit = regularize_hexagons(axes['j'], table)
fig.canvas.draw()

def _build_manuscript_figure():
    return fig, None


if __name__ == "__main__":
    _configure_land_functions(sys.modules[__name__])
    _figure, _bbox = _build_manuscript_figure()
    _save_figure(_figure, _bbox, 'Fig_3', _figure_options)
