"""Plot Extended_Data_Fig_2 using the packaged numerical tables or supplied local inputs."""
from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_support"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from figure_cli import prepare_figure as _prepare_figure, save_figure as _save_figure
from figure_cli import read_table as _read_table, model_tables as _model_tables
from figure_cli import configure_land_functions as _configure_land_functions

_figure_options = _prepare_figure('Extended_Data_Fig_2', 'analysis_tables') if __name__ == "__main__" else None

from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
from pathlib import Path
import json
import shutil
import sys
ROOT = PACKAGE_ROOT
import Extended_Data_Fig_2_quantile_intervals as prior
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, to_rgb
import numpy as np
from PIL import Image, ImageChops, ImageOps
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata
base = prior.base
OUT = output_dir('Extended_Data_Fig_2')
OUT.mkdir(parents=True, exist_ok=True)
CMAPS = {group: LinearSegmentedColormap.from_list(group + 'Density', ['#FFFFFF', np.asarray(to_rgb(color)) * 0.82 + 0.18]) for group, color in (('South', '#F27C2B'), ('North', '#0D47A1'))}
NORM = PowerNorm(gamma=0.65, vmin=0, vmax=1)
FIELDS, DETAILS = ({}, {})

def city_density(metric):
    groups = base.design.groups(metric)
    scores = np.concatenate(groups)
    ranks = np.concatenate([(rankdata(values, method='average') - 0.5) / len(values) * 100 for values in groups])
    bounds = (-1, 1) if metric == 'eli' else (0, 1)
    xedges, yedges = (np.linspace(*bounds, 901), np.linspace(0, 100, 401))
    counts, _, _ = np.histogram2d(scores, ranks, bins=(xedges, yedges))
    assert int(counts.sum()) == len(scores)
    dx, dy = (np.diff(xedges)[0], np.diff(yedges)[0])
    factor = len(scores) ** (-1 / 6)
    sx, sy = (np.std(scores, ddof=1) * factor, np.std(ranks, ddof=1) * factor)
    density = gaussian_filter(counts.T, sigma=(sy / dy, sx / dx), mode='reflect')
    density /= len(scores) * dx * dy
    assert abs(density.sum() * dx * dy - 1) < 1e-10
    contributions = []
    for values in groups:
        percentile = (rankdata(values, method='average') - 0.5) / len(values) * 100
        group_counts, _, _ = np.histogram2d(values, percentile, bins=(xedges, yedges))
        field = gaussian_filter(group_counts.T, sigma=(sy / dy, sx / dx), mode='reflect')
        contributions.append(field / (len(scores) * dx * dy))
    np.testing.assert_allclose(sum(contributions), density, atol=1e-13)
    DETAILS[metric] = dict(n=len(scores), bandwidth_score=sx, bandwidth_percentile=sy, grid_score_cells=900, grid_percentile_cells=400, density_peak=float(density.max()), mass=float(density.sum() * dx * dy))
    return (np.array(contributions) / density.max(), (bounds[0], bounds[1], 0, 100))

def density_lens(ax, metric):
    base.refined_lens(ax, metric)
    for area in list(ax.collections)[:2]:
        area.remove()
    contributions, extent = FIELDS[metric]
    density = contributions.sum(axis=0)
    rgba = np.zeros((*density.shape, 4), dtype=float)
    for field, group in zip(contributions, prior.GROUPS):
        rgba[:, :, :3] += field[:, :, None] * np.asarray(to_rgb(base.PALETTE[group]))
    np.divide(rgba[:, :, :3], density[:, :, None], out=rgba[:, :, :3], where=density[:, :, None] > 0)
    rgba[:, :, 3] = 0.82 * NORM(np.clip(density, 0, 1))
    ax.imshow(rgba, extent=extent, origin='lower', aspect='auto', interpolation='bilinear', zorder=0)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['left'].set_color(base.design.INK)
    ax.tick_params(axis='y', which='major', length=2.2, width=0.5, direction='out', left=True, right=False, labelleft=True)
    ax.set(xlim=base.design.domain(metric), ylim=(3, 97))

def build():
    for metric, _, _ in base.design.PANELS:
        FIELDS[metric] = city_density(metric)
    np.savez_compressed(OUT / 'city_density_fields.npz', **{metric: field[0] for metric, field in FIELDS.items()})
    base.design.SOUTH, base.design.NORTH = (base.PALETTE['South'], base.PALETTE['North'])
    base.design.quantile_lens = density_lens
    fig = base.design.grid_figure('B')
    fig.texts[-1].remove()
    fig.set_size_inches(7.2, 8.05)
    outer = fig.axes[0].get_subplotspec().get_gridspec()._subplot_spec.get_gridspec()
    outer.update(hspace=0.34)
    for ax in fig.axes:
        ax.set_position(ax.get_subplotspec().get_position(fig))
    for ax in fig.axes[::2]:
        assert len(ax.images) == 1 and len(ax.collections) == 0
        assert all((not line.get_path_effects() for line in ax.lines))
        assert ax.spines['left'].get_visible()
    for ax, (metric, _, _) in zip(fig.axes[::2], base.design.PANELS):
        for group, values in zip(prior.GROUPS, base.design.groups(metric)):
            line = next((line for line in ax.lines if len(line.get_xdata()) == len(prior.Q) and line.get_color() == base.PALETTE[group]))
            np.testing.assert_array_equal(line.get_xdata(), np.quantile(values, prior.Q))
    old_legend = fig.legends[0]
    legend_fontsize = old_legend.get_texts()[0].get_fontsize()
    handles, labels = (old_legend.legend_handles, ['Global South', 'Global North'])
    old_legend.remove()
    group_legend_y = 0.0305 - 1.0 / 25.4 / fig.get_figheight()
    group_legend = fig.legend(handles=handles, labels=labels, ncol=1, frameon=False, fontsize=legend_fontsize, handlelength=2, labelspacing=0.5, loc='center left', bbox_to_anchor=(0.5, group_legend_y), bbox_transform=fig.transFigure, borderaxespad=0)
    legend_downshift = 1.2 / 25.4 / fig.get_figheight()
    for group, y in (('South', 0.033), ('North', 0.027)):
        color_axis = fig.add_axes([0.28, y - legend_downshift, 0.13, 0.006])
        colorbar = fig.colorbar(plt.matplotlib.cm.ScalarMappable(norm=NORM, cmap=CMAPS[group]), cax=color_axis, orientation='horizontal', ticks=[] if group == 'South' else [0, 0.5, 1])
        if group == 'South':
            colorbar.ax.set_title('Relative city density', fontsize=legend_fontsize, pad=3)
        colorbar.ax.tick_params(labelsize=legend_fontsize, length=1.5, width=0.4, pad=1)
        colorbar.outline.set_linewidth(0.4)
    fig.canvas.draw()
    from matplotlib.text import Text
    font_scale = 1.155
    for text in fig.findobj(Text):
        text.set_fontsize(text.get_fontsize() * font_scale)
    legend_fontsize *= font_scale
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    density_axes = fig.axes[-2:]
    from matplotlib.transforms import Bbox
    footer_box = Bbox.union([a.get_tightbbox(renderer) for a in density_axes] + [group_legend.get_window_extent(renderer)])
    footer_shift = 0.5 - (footer_box.x0 + footer_box.x1) / (2 * fig.bbox.width)
    for a in density_axes:
        pos = a.get_position()
        a.set_position([pos.x0 + footer_shift, pos.y0, pos.width, pos.height])
    group_legend.set_bbox_to_anchor((0.5 + footer_shift, group_legend_y), transform=fig.transFigure)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    footer_box = Bbox.union([a.get_tightbbox(renderer) for a in density_axes] + [group_legend.get_window_extent(renderer)])
    footer_center = (footer_box.x0 + footer_box.x1) / (2 * fig.bbox.width)
    assert abs(footer_center - 0.5) < 1e-10
    assert group_legend.get_window_extent(renderer).y0 > 0
    assert [t.get_text() for t in group_legend.get_texts()] == ['Global South', 'Global North']
    assert group_legend.get_texts()[0].get_window_extent(renderer).y0 > group_legend.get_texts()[1].get_window_extent(renderer).y1
    assert all((np.isclose(a.get_position().height, 0.005 * 1.2) for a in density_axes))
    assert all((t.get_fontsize() == legend_fontsize for a in density_axes for t in a.get_xticklabels()))
    assert density_axes[0].title.get_fontsize() == legend_fontsize
    assert np.isclose(density_axes[0].get_position().y0, density_axes[1].get_position().y1, rtol=0, atol=1e-12)
    assert min((t.get_window_extent(renderer).y0 for t in density_axes[1].get_xticklabels())) > 0
    clipped_text = []
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text().strip():
            box = text.get_window_extent(renderer)
            if box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.width or (box.y1 > fig.bbox.height):
                clipped_text.append(text.get_text())
    assert not clipped_text, clipped_text
    return fig

def _build_manuscript_figure():
    return build(), None


if __name__ == "__main__":
    _configure_land_functions(sys.modules[__name__])
    _figure, _bbox = _build_manuscript_figure()
    _save_figure(_figure, _bbox, 'Extended_Data_Fig_2', _figure_options)
