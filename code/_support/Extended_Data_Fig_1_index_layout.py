from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
from pathlib import Path
import hashlib
import importlib.util
import io
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import Arc, Circle, Rectangle
from matplotlib.text import Text
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
from PIL import Image
ROOT = PACKAGE_ROOT
OUT = output_dir('Extended_Data_Fig_1')
SOURCE = ROOT / 'scripts' / 'plot_q1_measurement_validation.py'
STEM = '扩展图1-ELI验证'
FIELDS = ['spearman_rho_with_eli50', 'top_decile_retention', 'bottom_decile_retention']
LABELS = ['Spearman $ρ$', 'Top-decile\nretention', 'Bottom-decile\nretention']
COLORS = ['#208FBD', '#F27C2B', '#7F8AA9']
FONT_SIZES = {'panel_letter': 10, 'panel_title': 8.5, 'axis_title': 8, 'tick': 7, 'annotation': 7, 'note': 6.5}
SCHEMES = [('baseline_eli50', 'Baseline'), ('direction_adjusted_equal_six_dimensions', 'Equal six'), ('positive_dimensions_only', 'Positive only'), ('negative_dimensions_inverted_only', 'Negative only'), ('leave_one_out_beauty', 'Drop beauty'), ('leave_one_out_wealthy', 'Drop wealth'), ('leave_one_out_safer', 'Drop safety'), ('leave_one_out_livelier', 'Drop lively'), ('leave_one_out_boring', 'Drop boredom'), ('leave_one_out_depression', 'Drop depression')]
CONSTRUCTION_FIELDS = FIELDS + ['north_minus_south_mean']

def construction_values(data):
    return data.set_index('specification').loc[[key for key, _ in SCHEMES], CONSTRUCTION_FIELDS]

def rank_ring(axis, x, y, rho, gap, norm, cmap, coordinates='data', size=13):
    scale = size / 13
    center = 6.5 * scale
    linewidth = 0.9 if size < 13 else 1.2
    drawing = DrawingArea(size, size, 0, 0)
    drawing.add_artist(Circle((center, center), 5.5 * scale, fill=False, edgecolor='#DFE5EA', lw=linewidth))
    drawing.add_artist(Arc((center, center), 11 * scale, 11 * scale, theta1=90 - 360 * rho, theta2=90, edgecolor='#244E65', lw=linewidth))
    drawing.add_artist(Circle((center, center), 3.4 * scale, facecolor='white' if gap is None else cmap(norm(gap)), edgecolor='none'))
    if gap is not None:
        drawing.add_artist(plt.Line2D([4.9 * scale, 8.1 * scale], [center, center], lw=0.55, color='white'))
        if gap >= 0:
            drawing.add_artist(plt.Line2D([center, center], [4.9 * scale, 8.1 * scale], lw=0.55, color='white'))
    artist = AnnotationBbox(drawing, (x, y), xycoords=coordinates, frameon=False, pad=0, box_alignment=(0.5, 0.5), zorder=5)
    axis.add_artist(artist)
    return artist

def draw_construction_map(container, data):
    container.clear()
    container.set_xticks([])
    container.set_yticks([])
    container.set_axis_off()
    values = construction_values(data).to_numpy(dtype=float)
    norm = TwoSlopeNorm(vmin=-0.1, vcenter=0, vmax=0.1)
    cmap = LinearSegmentedColormap.from_list('construction_gap', ['#D95B35', '#F7F7F7', '#277DA8'])
    axis = container.inset_axes([0, 0.15 / 0.77, 1, 0.5 / 0.77])
    axis.set_xlim(0.42, 1.08)
    axis.set_ylim(0.57, 1.08)
    axis.set_xticks([0.5, 0.75, 1], ['0.50', '0.75', '1.00'])
    axis.set_yticks([0.6, 0.8, 1], ['0.60', '0.80', '1.00'])
    axis.set_xlabel('Top-decile retention', fontsize=8, labelpad=3)
    axis.set_ylabel('Bottom-decile retention', fontsize=8, labelpad=5)
    axis.tick_params(direction='out', length=3, width=0.7, pad=2, labelsize=7)
    axis.spines[['top', 'right']].set_visible(False)
    axis.axvline(1, color='#D4DCE3', lw=0.55, ls=(0, (2, 3)), zorder=0)
    axis.axhline(1, color='#D4DCE3', lw=0.55, ls=(0, (2, 3)), zorder=0)
    label_positions = [(1.035, 1.045, 'right'), (0.67, 1.04, 'left'), (0.48, 0.94, 'left'), (0.5, 0.59, 'left'), (0.87, 0.73, 'left'), (0.94, 0.965, 'left'), (0.65, 0.975, 'left'), (0.94, 0.81, 'left'), (0.46, 0.84, 'left'), (0.64, 0.7, 'left')]
    for row, (_, label), (tx, ty, alignment) in zip(values, SCHEMES, label_positions):
        rho, top, bottom, gap = row
        rank_ring(axis, top, bottom, rho, gap, norm, cmap, size=7.5)
        axis.annotate(label, xy=(top, bottom), xytext=(tx, ty), fontsize=7, ha=alignment, va='center', zorder=6, arrowprops={'arrowstyle': '-', 'color': '#9AA7B2', 'lw': 0.45, 'shrinkA': 3, 'shrinkB': 5})
    container.text(0, 0.985, 'Spearman $ρ$', transform=container.transAxes, fontsize=7, ha='left', va='center')
    legend_ring_y = 0.925 - 2 / 72 / (container.get_position().height * container.figure.get_figheight())
    for x, rho in [(0.025, 0.7), (0.2, 0.9), (0.375, 1.0)]:
        rank_ring(container, x, legend_ring_y, rho, None, norm, cmap, coordinates='axes fraction')
        container.text(x + 0.045, 0.925, f'{rho:.1f}', transform=container.transAxes, fontsize=7, ha='left', va='center')
    colour_axis = container.inset_axes([0.58, 0.915, 0.42, 0.022])
    colourbar = container.figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=colour_axis, orientation='horizontal', ticks=[-0.1, 0, 0.1])
    colourbar.ax.set_xticklabels(['-0.10', '0', '+0.10'])
    colourbar.ax.tick_params(labelsize=7, length=3, width=0.7, pad=2)
    colourbar.outline.set_linewidth(0.6)
    container.text(0.58, 0.985, 'North - South ELI', transform=container.transAxes, fontsize=8, ha='left', va='center')
    return axis

def draw_construction_signatures(ax, data):
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    values = construction_values(data).to_numpy(dtype=float)
    baseline_gap = values[0, 3]
    centers = [0.455, 0.615, 0.775, 0.94]
    half_width = 0.06
    rows = np.linspace(0.745, 0.105, len(SCHEMES))
    ax.text(0, 0.965, 'Baseline and alternative ELI constructions.', ha='left', va='center', fontsize=7)
    for center, label in zip(centers, ['$ρ$', 'top', 'bottom', 'N-S $Δ$']):
        ax.text(center, 0.9, label, ha='center', va='center', fontsize=7)
    for center in centers[:3]:
        left, right = (center - half_width, center + half_width)
        ax.plot([left, right], [0.819, 0.819], lw=0.55, color='#9AA7B2')
        for x, label in [(left, '0'), (right, '1')]:
            ax.plot([x, x], [0.814, 0.824], lw=0.55, color='#9AA7B2')
            ax.text(x, 0.832, label, fontsize=7, ha='center', va='bottom')
    gap_left, gap_right = (centers[3] - half_width, centers[3] + half_width)
    gap_x = lambda value: gap_left + (value + 0.1) / 0.2 * (gap_right - gap_left)
    ax.plot([gap_left, gap_right], [0.819, 0.819], lw=0.55, color='#9AA7B2')
    for value, label in [(-0.1, '-.1'), (0.1, '+.1')]:
        ax.text(gap_x(value), 0.832, label, fontsize=7, ha='center', va='bottom')
    ax.plot([gap_x(0)] * 2, [0.096, 0.79], color='#CCD3DA', lw=0.5, zorder=0)
    ax.plot([gap_x(baseline_gap)] * 2, [0.096, 0.79], color='#647582', lw=0.65, ls=(0, (1.2, 2)), zorder=0)
    for i, (y, (_, label)) in enumerate(zip(rows, SCHEMES)):
        if i == 0:
            ax.axhspan(y - 0.036, y + 0.038, color='#EFF3F7', lw=0, zorder=-1)
        ax.text(0.012, y, label, ha='left', va='center', fontsize=7)
        track_y = y + 0.02
        value_y = y - 0.019
        for j, (center, color) in enumerate(zip(centers[:3], COLORS)):
            left, right = (center - half_width, center + half_width)
            endpoint = left + (right - left) * values[i, j]
            ax.plot([left, right], [track_y] * 2, lw=0.55, color='#DFE6ED', zorder=1)
            ax.plot([endpoint, right], [track_y] * 2, lw=2.5, color=color, alpha=0.28, solid_capstyle='butt', zorder=2)
            ax.scatter(right, track_y, marker='o', s=9, facecolor='white', edgecolor=color, lw=0.65, zorder=3)
            ax.scatter(endpoint, track_y, marker='s', s=8, color=color, lw=0, zorder=4)
            ax.text(center, value_y, f'{values[i, j]:.2f}', ha='center', va='center', fontsize=7)
        current_gap_x = gap_x(values[i, 3])
        ax.plot([gap_x(baseline_gap), current_gap_x], [track_y] * 2, color='#687E8C', lw=1, zorder=2)
        ax.scatter(gap_x(baseline_gap), track_y, marker='o', s=9, facecolor='white', edgecolor='#526775', lw=0.65, zorder=3)
        ax.scatter(current_gap_x, track_y, marker='D', s=10, color='#263D4A', lw=0, zorder=4)
        ax.text(centers[3], value_y, f'{values[i, 3]:+.3f}', ha='center', va='center', fontsize=7, bbox={'facecolor': '#EFF3F7' if i == 0 else 'white', 'edgecolor': 'none', 'pad': 0.08})
    ax.text(0, 0.025, 'Top/bottom indicate decile retention; N-S Δ is North minus South mean ELI.', ha='left', va='bottom', fontsize=6.5)

def unify_typography(fig):
    for axis in fig.axes:
        axis.xaxis.label.set_fontsize(FONT_SIZES['axis_title'])
        axis.yaxis.label.set_fontsize(FONT_SIZES['axis_title'])
        axis.tick_params(axis='both', labelsize=FONT_SIZES['tick'])
        for text in axis.texts:
            role = 'note' if text.get_text().startswith('Top/bottom indicate') else 'annotation'
            text.set_fontsize(FONT_SIZES[role])
    for text in fig.texts:
        content = text.get_text()
        role = 'panel_letter' if content in {'a', 'b', 'c', 'd'} else 'panel_title'
        if content.startswith('EC QoL survey'):
            role = 'note'
            text.set_text(content.replace('shares; P', 'shares;\nP'))
        text.set_fontsize(FONT_SIZES[role])

def digest(path):
    return hashlib.sha256(resource(path).read_bytes()).hexdigest()

def load_original():
    spec = importlib.util.spec_from_file_location('original_validation', resource(SOURCE))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def base_axis(ax):
    ax.clear()
    ax.set_xlim(-0.27, 2.27)
    ax.set_ylim(0.5, 1.0)
    ax.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax.set_xticks([])
    for side in ax.spines:
        ax.spines[side].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=3, labelsize=6.5)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#E8EBEF', linewidth=0.45)

def paired_braid(ax, values, aligned_panel=False):
    base_axis(ax)
    height_inches = ax.get_position().height * ax.figure.get_figheight()
    if aligned_panel:
        ax.set_ylim(0.35, 1.12)
    median_y = 1 - 0.15 / height_inches if aligned_panel else 1 + 0.23 / height_inches
    range_y = 1 - 0.31 / height_inches if aligned_panel else 1 + 0.03 / height_inches
    label_y = -0.06 / height_inches
    ax.set_ylabel('Robustness', fontsize=FONT_SIZES['axis_title'], labelpad=5)
    if aligned_panel:
        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_bounds(0.5, 1.0)
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_position(('data', 0.5))
        ax.set_xticks([0, 1, 2], LABELS)
        ax.tick_params(axis='both', direction='out', length=3, width=0.7, pad=2, labelsize=FONT_SIZES['tick'])
        ax.set_xlabel('Robustness metric', fontsize=FONT_SIZES['axis_title'], labelpad=5)
    positions = np.arange(3)
    segments = np.stack([np.broadcast_to(positions, values.shape), values], axis=-1)
    ax.add_collection(LineCollection(segments, colors='#536778', linewidths=0.28, alpha=0.075, zorder=1))
    for j, (color, label) in enumerate(zip(COLORS, LABELS)):
        quantiles = np.quantile(values[:, j], np.linspace(0, 1, 21))
        for lo, hi in zip(quantiles[:-1], quantiles[1:]):
            ax.add_patch(Rectangle((j - 0.044, lo), 0.088, hi - lo, facecolor=color, alpha=0.75, edgecolor='white', linewidth=0.4, zorder=3))
        lo, q25, mid, q75, hi = np.quantile(values[:, j], [0.025, 0.25, 0.5, 0.75, 0.975])
        ax.plot([j + 0.075, j + 0.075], [lo, hi], color=color, lw=0.9, zorder=4)
        ax.plot([j + 0.075, j + 0.075], [q25, q75], color=color, lw=3, zorder=4)
        for endpoint in [lo, hi]:
            ax.plot([j + 0.04, j + 0.11], [endpoint, endpoint], color=color, lw=0.9, zorder=4)
        ax.scatter(j, mid, s=27, marker='D', color=color, edgecolor='white', lw=0.85, zorder=5)
        xfrac = (j + 0.27) / 2.54
        ax.text(xfrac, median_y, f'{mid:.2f}', transform=ax.transAxes, fontsize=FONT_SIZES['annotation'], color=color, ha='center', va='center')
        ax.text(xfrac, range_y, f'[{lo:.2f}, {hi:.2f}]', transform=ax.transAxes, fontsize=FONT_SIZES['annotation'], ha='center', va='bottom')
        if not aligned_panel:
            ax.text(xfrac, label_y, label, transform=ax.transAxes, fontsize=7, ha='center', va='top', linespacing=1.12)
    if not aligned_panel:
        ax.text(0, 1 + 0.48 / height_inches, '500 paired weight trials', transform=ax.transAxes, fontsize=7, ha='left', va='bottom')
        ax.text(0, 1 + 0.36 / height_inches, 'Median [central 95% range]', transform=ax.transAxes, fontsize=6.2, ha='left', va='bottom')
