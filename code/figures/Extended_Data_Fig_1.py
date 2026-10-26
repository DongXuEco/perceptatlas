"""Plot Extended_Data_Fig_1 using the packaged numerical tables or supplied local inputs."""
from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_support"))
from figure_cli import prepare_figure as _prepare_figure, save_figure as _save_figure
from figure_cli import read_table as _read_table, model_tables as _model_tables
from figure_cli import configure_land_functions as _configure_land_functions

_figure_options = _prepare_figure('Extended_Data_Fig_1', 'analysis_tables') if __name__ == "__main__" else None

from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
from pathlib import Path
import hashlib
import importlib.util
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
from PIL import Image
ROOT = PACKAGE_ROOT
SOURCE = ROOT / 'source_modules/扩展图1-ELI验证.py'
OUT = output_dir('Extended_Data_Fig_1')
STEM = '扩展图1-ELI验证'

def build_figure():
    spec = importlib.util.spec_from_file_location('validation_layout', resource(SOURCE))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    original = mod.load_original()
    data = pd.read_csv(resource(original.H_TABLE))
    values = data.loc[data.scheme_type.eq('random_weight'), mod.FIELDS].to_numpy(dtype=float)
    fig = original.build_figure()
    fig.canvas.draw()
    bbox = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.02).frozen()
    bbox = Bbox.from_extents(bbox.x0 - 0.1, bbox.y0, bbox.x1, bbox.y1).frozen()
    mod.unify_typography(fig)
    container = fig.axes[7]
    mod.draw_construction_signatures(container, data)
    fig.canvas.draw()
    mod.draw_construction_map(container, data)
    fig.axes[-1].remove()
    positions = [ax.get_position() for ax in fig.axes[1:7]]
    left, right = (min((p.x0 for p in positions)), max((p.x1 for p in positions)))
    c_position = container.get_position()
    axis = fig.add_axes([left, c_position.y0, right - left, c_position.height])
    mod.paired_braid(axis, values, aligned_panel=True)
    c_heading_y = next((t.get_position()[1] for t in fig.texts if t.get_text() == 'c'))
    original.add_panel_label(fig, axis, 'd', 'Random-weight robustness', dy=c_heading_y - axis.get_position().y1)
    mod.unify_typography(fig)
    return (fig, bbox, original)

def exchange_columns(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    assert len(fig.axes) == 9
    left_axes = [fig.axes[0], fig.axes[7]]
    right_axes = fig.axes[1:7] + [fig.axes[8]]
    a_x = next((t.get_position()[0] for t in fig.texts if t.get_text() == 'a'))
    b_x = next((t.get_position()[0] for t in fig.texts if t.get_text() == 'b'))
    split = (a_x + b_x) / 2
    left_text = [t for t in fig.texts if t.get_position()[0] < split]
    right_text = [t for t in fig.texts if t not in left_text]

    def bounds(axes, texts):
        boxes = [a.get_tightbbox(renderer) for a in axes]
        boxes += [t.get_window_extent(renderer) for t in texts]
        return Bbox.union(boxes).frozen()
    left_box = bounds(left_axes, left_text)
    right_box = bounds(right_axes, right_text)
    assert left_box.x1 < right_box.x0
    rightward_px = round(right_box.x1 - left_box.x1)
    leftward_px = round(left_box.x0 - right_box.x0)
    old_positions = [a.get_position().frozen() for a in fig.axes]
    for axes, texts, shift in [(left_axes, left_text, rightward_px), (right_axes, right_text, leftward_px)]:
        dx = shift / fig.bbox.width
        for axis in axes:
            p = axis.get_position()
            axis.set_position([p.x0 + dx, p.y0, p.width, p.height])
        for text in texts:
            x, y = text.get_position()
            text.set_position((x + dx, y))
    labels = {'a': 'b', 'b': 'a', 'c': 'd', 'd': 'c'}
    changed_letters = []
    for text in fig.texts:
        if text.get_text() in labels:
            old = text.get_text()
            text.set_text(labels[old])
            changed_letters.append((old, text.get_text()))
    assert len(changed_letters) == 4
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for old, axis in zip(old_positions, fig.axes):
        p = axis.get_position()
        assert np.allclose([old.y0, old.width, old.height], [p.y0, p.width, p.height])
    new_left_box = bounds(right_axes, right_text)
    new_right_box = bounds(left_axes, left_text)
    assert new_left_box.x1 < new_right_box.x0
    return {'old_left_bounds_pixels': list(left_box.bounds), 'old_right_bounds_pixels': list(right_box.bounds), 'new_left_bounds_pixels': list(new_left_box.bounds), 'new_right_bounds_pixels': list(new_right_box.bounds), 'original_left_column_shift_pixels': rightward_px, 'original_right_column_shift_pixels': leftward_px, 'labels_old_to_new': labels, 'panel_dimensions_unchanged': True, 'previous_legend_circle_shift_down_pt': 2}

def _build_manuscript_figure():
    figure, bbox, original = build_figure()
    exchange_columns(figure)
    for text in list(figure.texts):
        if text.get_text().startswith("EC QoL survey"):
            text.remove()
    return figure, bbox


if __name__ == "__main__":
    _configure_land_functions(sys.modules[__name__])
    _figure, _bbox = _build_manuscript_figure()
    _save_figure(_figure, _bbox, 'Extended_Data_Fig_1', _figure_options)
