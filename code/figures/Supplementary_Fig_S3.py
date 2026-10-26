"""Plot Supplementary_Fig_S3 using the packaged numerical tables or supplied local inputs."""
from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_support"))
from figure_cli import prepare_figure as _prepare_figure, save_figure as _save_figure
from figure_cli import read_table as _read_table, model_tables as _model_tables
from figure_cli import configure_land_functions as _configure_land_functions

_figure_options = _prepare_figure('Supplementary_Fig_S3', 'external_inputs_required') if __name__ == "__main__" else None

from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.unicode_minus'] = False

def reproject_to_wgs84(src_path):
    dst_crs = 'EPSG:4326'
    with rasterio.open(resource(src_path)) as src:
        if src.count < 3:
            raise ValueError(f'{src_path} 波段数少于3，不能作为RGB显示。')
        if src.crs and src.crs.to_string() == dst_crs:
            r = src.read(1).astype(np.float32)
            g = src.read(2).astype(np.float32)
            b = src.read(3).astype(np.float32)
            transform = src.transform
            width = src.width
            height = src.height
        else:
            transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
            r = np.empty((height, width), dtype=np.float32)
            g = np.empty((height, width), dtype=np.float32)
            b = np.empty((height, width), dtype=np.float32)
            reproject(source=rasterio.band(src, 1), destination=r, src_transform=src.transform, src_crs=src.crs, dst_transform=transform, dst_crs=dst_crs, resampling=Resampling.bilinear)
            reproject(source=rasterio.band(src, 2), destination=g, src_transform=src.transform, src_crs=src.crs, dst_transform=transform, dst_crs=dst_crs, resampling=Resampling.bilinear)
            reproject(source=rasterio.band(src, 3), destination=b, src_transform=src.transform, src_crs=src.crs, dst_transform=transform, dst_crs=dst_crs, resampling=Resampling.bilinear)
        rgb = np.stack([r, g, b], axis=-1)
        rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
        rgb = np.clip(rgb, 0, 0.35)
        rgb = rgb / 0.35
        xmin = transform.c
        ymax = transform.f
        xmax = xmin + transform.a * width
        ymin = ymax + transform.e * height
        extent = [xmin, xmax, ymin, ymax]
        return (rgb, extent)

def plot_rgb_tif(tif1, out_png):
    rgb1, extent1 = reproject_to_wgs84(tif1)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(rgb1, extent=extent1, origin='upper')
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.tick_params(axis='both', labelsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f'图片已保存到: {out_png}')
    plt.close(fig)
script_dir = Path(__file__).resolve().parent
tif_path1 = script_dir / 'globalimage.tif'
out_png = script_dir.parent / 'All figures' / '最终图' / '支撑图' / '去云影像.png'
plot_rgb_tif(tif_path1, out_png)

def _build_manuscript_figure():
    return None, None


if __name__ == "__main__":
    _configure_land_functions(sys.modules[__name__])
    _figure, _bbox = _build_manuscript_figure()
    _save_figure(_figure, _bbox, 'Supplementary_Fig_S3', _figure_options)
