from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
from pathlib import Path
import hashlib
import json
import shutil
import sys
ROOT = PACKAGE_ROOT
import Extended_Data_Fig_2_distribution_layout as design
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
OUT = output_dir('Extended_Data_Fig_2')
OUT.mkdir(parents=True, exist_ok=True)
FORMAL = design.FORMAL
CODE = ROOT / 'All figures/最终图/code'
PALETTE = {'South': '#F27C2B', 'North': '#0D47A1', 'lens': '#EEF1F3'}

def quantile_curves(fig):
    return [[(line.get_xdata().tolist(), line.get_ydata().tolist()) for line in ax.lines if len(line.get_xdata()) == 181] for ax in fig.axes[::2]]

def refined_lens(ax, metric):
    q = np.linspace(0.05, 0.95, 181)
    qs, qn = [np.quantile(values, q) for values in design.groups(metric)]
    central = (q >= 0.25 - 1e-12) & (q <= 0.75 + 1e-12)
    ax.fill_betweenx(q * 100, qs, qn, color=PALETTE['lens'], alpha=0.5, linewidth=0)
    ax.fill_betweenx(q * 100, qs, qn, where=central, color='#B5C3CF', alpha=0.22, linewidth=0)
    for percentile in (0.05, 0.25, 0.5, 0.75, 0.95):
        median = percentile == 0.5
        endpoints = [np.interp(percentile, q, values) for values in (qs, qn)]
        ax.plot(endpoints, [percentile * 100] * 2, color='#8293A2' if median else '#CDD6DE', lw=0.7 if median else 0.4, zorder=1)
    for values, color, marker in ((qs, design.SOUTH, 'o'), (qn, design.NORTH, 's')):
        ax.plot(values, q * 100, color=color, lw=0.85, solid_capstyle='round', zorder=2)
        ax.plot(values[central], q[central] * 100, color=color, lw=1.65, solid_capstyle='round', zorder=3)
        for percentile in (0.05, 0.25, 0.5, 0.75, 0.95):
            median = percentile == 0.5
            ax.plot(np.interp(percentile, q, values), percentile * 100, marker=marker, ms=3.6 if median else 2.4, mfc=color if median else 'white', mec='white' if median else color, mew=0.45 if median else 0.8, zorder=5 if median else 4)
    ax.set(xlim=design.domain(metric), ylim=(3, 97), yticks=[5, 25, 50, 75, 95])
    ax.set_xticks([-0.4, 0, 0.4] if metric == 'eli' else [0, 0.5, 1])
    ax.set_xlabel('ELI' if metric == 'eli' else 'Perception score', fontsize=6.6, labelpad=2)
    ax.set_ylabel('Percentile', fontsize=6.6, labelpad=3)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='both', length=2.2, width=0.5, pad=2, labelsize=6.4)
    ax.tick_params(axis='y', length=0)
