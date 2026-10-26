from __future__ import annotations
from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
import math
import struct
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Ellipse, Polygon, Rectangle
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator, PercentFormatter
import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from scipy.stats import gaussian_kde, pearsonr
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.weight'] = 'regular'
plt.rcParams['axes.titleweight'] = 'regular'
plt.rcParams['axes.labelweight'] = 'regular'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['legend.frameon'] = False
plt.rcParams['axes.unicode_minus'] = False
PROJECT_ROOT = PACKAGE_ROOT
OUT_DIR = output_dir('Fig_4')
OUT_DIR.mkdir(parents=True, exist_ok=True)
PALETTE = {'pale_apricot': '#FFE7CC', 'soft_orange': '#FFB66D', 'burnt_orange': '#F27C2B', 'warm_sand': '#E7C9A0', 'cool_mist': '#DCE6F2', 'light_sky': '#A7CFF2', 'cerulean': '#3B82D6', 'cobalt': '#0D47A1', 'text': '#1F2937', 'muted': '#6B7280', 'axis': '#30343B', 'grid': '#E5E7EB'}
GROUP_COLORS = {'South': PALETTE['burnt_orange'], 'North': PALETTE['cerulean']}
LAND_COLOR = '#E9EBF1'
_LAND_GDF = None
DIV_CMAP = LinearSegmentedColormap.from_list('blue_orange_div', [PALETTE['burnt_orange'], '#F8F2EA', PALETTE['cobalt']])
MAP_CMAP = LinearSegmentedColormap.from_list('unified_map_blue_orange', [PALETTE['burnt_orange'], PALETTE['soft_orange'], PALETTE['pale_apricot'], PALETTE['cool_mist'], PALETTE['light_sky'], PALETTE['cerulean'], PALETTE['cobalt']])
EMOTIONS = {'beauty': 'Beauty', 'wealthy': 'Wealth', 'safer': 'Safety', 'livelier': 'Liveliness', 'boring': 'Boredom', 'depression': 'Depression'}
POSITIVE = ['beauty', 'wealthy', 'safer', 'livelier']
NEGATIVE = ['boring', 'depression']
CE_FILES = {'boring': 'RF_boring_stacking1_tif_reorganized.csv', 'depression': 'RF_depressing_stacking1_tif_reorganized.csv', 'livelier': 'RF_livelier_stacking1_tif_reorganized.csv', 'safer': 'RF_safer_stacking1_tif_reorganized.csv', 'beauty': 'RF_stacking1_beauty_tif_reorganized.csv', 'wealthy': 'RF_wealthy_stacking1_tif_reorganized.csv'}

def find_project_file(filename: str) -> Path:
    hits = sorted(resource_glob(PROJECT_ROOT, f'*/{filename}'))
    if not hits:
        raise FileNotFoundError(f'Cannot find {filename} under {PROJECT_ROOT}')
    return hits[0]

def decode_dbf_text(raw: bytes) -> str:
    raw = raw.strip()
    if not raw:
        return ''
    for enc in ('utf-8', 'cp936', 'gbk', 'latin1'):
        try:
            return raw.decode(enc).strip()
        except UnicodeDecodeError:
            continue
    return raw.decode('latin1', errors='ignore').strip()

def read_dbf(path: Path) -> pd.DataFrame:
    with path.open('rb') as f:
        header = f.read(32)
        _, _, _, _, n_records, header_len, record_len = struct.unpack('<BBBBIHH20x', header)
        fields = []
        while True:
            desc = f.read(32)
            if not desc or desc[0] == 13:
                break
            name = desc[:11].split(b'\x00', 1)[0].decode('ascii', errors='ignore')
            fields.append((name, chr(desc[11]), desc[16], desc[17]))
        f.seek(header_len)
        rows = []
        for _ in range(n_records):
            record = f.read(record_len)
            if not record or record[0:1] == b'*':
                continue
            offset = 1
            row = {}
            for name, ftype, flen, _ in fields:
                raw = record[offset:offset + flen]
                offset += flen
                if ftype in ('N', 'F'):
                    text = raw.strip()
                    row[name] = np.nan if not text else float(text)
                else:
                    row[name] = decode_dbf_text(raw)
            rows.append(row)
    return pd.DataFrame(rows)

def load_city_metadata() -> pd.DataFrame:
    df = read_dbf(find_project_file('global_city_point.dbf'))
    for col in ['lon', 'lat', 'id_1', 'NS']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['lon', 'lat', 'id_1', 'NS']).copy()
    df['id_1'] = df['id_1'].astype(int)
    df['ns_group'] = np.where(df['NS'].round().astype(int) == 0, 'North', 'South')
    return df[['id_1', 'COUNTRY', 'NAME_1', 'lon', 'lat', 'ns_group']].copy()

def load_ce_data() -> pd.DataFrame:
    tables = []
    for emotion, filename in CE_FILES.items():
        p = find_project_file(filename)
        df = pd.read_csv(resource(p))
        required = {'id_1', 'mean_1', 'mean_2', 'diff'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f'{p} missing columns: {sorted(missing)}')
        df = df[['id_1', 'mean_1', 'mean_2', 'diff']].copy()
        df = df.rename(columns={'mean_1': f'core_{emotion}', 'mean_2': f'expansion_{emotion}', 'diff': f'diff_{emotion}'})
        tables.append(df)
    out = tables[0]
    for table in tables[1:]:
        out = out.merge(table, on='id_1', how='inner')
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors='coerce')
    out['id_1'] = out['id_1'].astype(int)
    for emotion in EMOTIONS:
        if emotion in POSITIVE:
            out[f'benefit_{emotion}'] = out[f'diff_{emotion}']
            out[f'core_oriented_{emotion}'] = out[f'core_{emotion}']
            out[f'exp_oriented_{emotion}'] = out[f'expansion_{emotion}']
        else:
            out[f'benefit_{emotion}'] = -out[f'diff_{emotion}']
            out[f'core_oriented_{emotion}'] = 1 - out[f'core_{emotion}']
            out[f'exp_oriented_{emotion}'] = 1 - out[f'expansion_{emotion}']
    out['core_eli'] = out[[f'core_{e}' for e in POSITIVE]].mean(axis=1) - out[[f'core_{e}' for e in NEGATIVE]].mean(axis=1)
    out['expansion_eli'] = out[[f'expansion_{e}' for e in POSITIVE]].mean(axis=1) - out[[f'expansion_{e}' for e in NEGATIVE]].mean(axis=1)
    out['eli_diff'] = out['core_eli'] - out['expansion_eli']
    return out

def add_panel_label(ax: mpl.axes.Axes, label: str, x: float=-0.08, y: float=1.06) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha='left', va='bottom', fontsize=8.5, fontweight='bold', color=PALETTE['text'], gid='panel_label')

def clean_axis(ax: mpl.axes.Axes, grid: bool=False) -> None:
    ax.tick_params(axis='both', labelsize=5.8, length=2.5, width=0.45, colors=PALETTE['axis'])
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color(PALETTE['axis'])
    ax.grid(False)

def force_black_text_except(fig: mpl.figure.Figure, skip_axes: set[mpl.axes.Axes]) -> None:
    for text in fig.findobj(match=lambda artist: isinstance(artist, Text)):
        if text.axes not in skip_axes:
            text.set_color('#000000')
    for ax in fig.axes:
        if ax in skip_axes:
            continue
        ax.tick_params(axis='both', colors='#000000')
        ax.xaxis.label.set_color('#000000')
        ax.yaxis.label.set_color('#000000')
        ax.title.set_color('#000000')

def add_covariance_ellipse(ax: mpl.axes.Axes, x: np.ndarray, y: np.ndarray, color: str) -> None:
    xy = np.column_stack([x, y])
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) < 4:
        return
    cov = np.cov(xy, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    scale = 1.52
    ellipse = Ellipse(xy=np.nanmedian(xy, axis=0), width=2 * scale * np.sqrt(vals[0]), height=2 * scale * np.sqrt(vals[1]), angle=angle, facecolor=color, edgecolor=color, lw=0.75, alpha=0.11, zorder=3)
    ax.add_patch(ellipse)
    ax.add_patch(Ellipse(xy=ellipse.center, width=ellipse.width, height=ellipse.height, angle=angle, facecolor='none', edgecolor=color, lw=0.55, alpha=0.7, zorder=4))

def find_world_land() -> Path:
    return PACKAGE_ROOT / 'data/shared/World_land_polygons.geojson'

def draw_land_background(ax: mpl.axes.Axes) -> None:
    global _LAND_GDF
    if _LAND_GDF is None:
        _LAND_GDF = gpd.read_file(resource(find_world_land())).to_crs('EPSG:4326')
    _LAND_GDF.plot(ax=ax, facecolor=LAND_COLOR, edgecolor='none', linewidth=0, zorder=0)

def add_map_colorbar(ax: mpl.axes.Axes, mappable: mpl.cm.ScalarMappable, label: str, ticks: list[float], ticklabels: list[str] | None=None, extend: str='neither') -> None:
    right_shift = 8 / 25.4 / (ax.get_position().width * ax.figure.get_figwidth())
    cax = ax.inset_axes([0.23 + right_shift, -0.085, 0.44 * 0.85, 0.039 * 1.1])
    cbar = ax.figure.colorbar(mappable, cax=cax, orientation='horizontal', ticks=ticks, extend=extend)
    cbar.outline.set_linewidth(0.45)
    cbar.ax.tick_params(labelsize=5.0 * 1.21, length=1.8, width=0.4, pad=1, colors=PALETTE['axis'])
    cbar.ax.set_title(label, fontsize=5.5 * 1.21, pad=5, color=PALETTE['text'])
    if ticklabels is not None:
        cbar.ax.set_xticklabels(ticklabels)

def bootstrap_mean_ci(values: np.ndarray, n_boot: int=800) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(11)
    boot = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(n_boot)])
    return (values.mean(), *np.percentile(boot, [2.5, 97.5]))

def bootstrap_difference_ci(a: np.ndarray, b: np.ndarray, n_boot: int=900, seed: int=23) -> tuple[float, float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(b, len(b), replace=True).mean() - rng.choice(a, len(a), replace=True).mean() for _ in range(n_boot)])
    return (b.mean() - a.mean(), *np.percentile(boot, [2.5, 97.5]))

def bootstrap_means(values: np.ndarray, n_boot: int=800, seed: int=11) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.array([])
    rng = np.random.default_rng(seed)
    return np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(n_boot)])

def smooth_density(values: np.ndarray, bins: int=90, qrange: tuple[float, float]=(0.5, 99.5)) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 6:
        return (np.array([]), np.array([]))
    lo, hi = np.percentile(values, qrange)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = (np.nanmin(values), np.nanmax(values))
    hist, edges = np.histogram(values, bins=bins, range=(lo, hi), density=True)
    kernel = np.array([1, 4, 7, 10, 7, 4, 1], dtype=float)
    kernel /= kernel.sum()
    hist = np.convolve(hist, kernel, mode='same')
    centers = (edges[:-1] + edges[1:]) / 2
    return (centers, hist)

def draw_half_violin(ax: mpl.axes.Axes, values: np.ndarray, x: float, color: str, side: int=1, width: float=0.34, alpha: float=0.25, zorder: int=1) -> None:
    yy, dens = smooth_density(values)
    if len(yy) == 0 or np.nanmax(dens) <= 0:
        return
    xx = x + side * width * dens / np.nanmax(dens)
    verts = [(x, yy[0]), *zip(xx, yy), (x, yy[-1])]
    ax.add_patch(Polygon(verts, closed=True, facecolor=color, edgecolor='none', alpha=alpha, zorder=zorder))
    ax.plot(xx, yy, color=color, lw=0.62, alpha=0.68, zorder=zorder + 1)

def draw_center_box(ax: mpl.axes.Axes, values: np.ndarray, x: float, width: float=0.15) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return
    q05, q25, q50, q75, q95 = np.percentile(values, [5, 25, 50, 75, 95])
    ax.plot([x, x], [q05, q95], color=PALETTE['axis'], lw=0.65, zorder=5)
    ax.add_patch(Rectangle((x - width / 2, q25), width, q75 - q25, facecolor='white', edgecolor=PALETTE['axis'], lw=0.65, zorder=6))
    ax.plot([x - width / 2, x + width / 2], [q50, q50], color=PALETTE['text'], lw=0.75, zorder=7)

def draw_horizontal_density(ax: mpl.axes.Axes, samples: np.ndarray, y: float, color: str, height: float=0.15, alpha: float=0.14) -> None:
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if len(samples) < 10 or np.nanmax(samples) == np.nanmin(samples):
        return
    lo, hi = np.percentile(samples, [1, 99])
    hist, edges = np.histogram(samples, bins=26, range=(lo, hi), density=True)
    mids = (edges[:-1] + edges[1:]) / 2
    if hist.max() <= 0:
        return
    band = hist / hist.max() * height
    ax.fill_between(mids, y - band, y + band, color=color, alpha=alpha, lw=0, zorder=1)

def draw_vertical_rain(ax: mpl.axes.Axes, values: np.ndarray, x: float, color: str, side: int, seed: int) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return
    rng = np.random.default_rng(seed)
    if len(values) > 420:
        values = rng.choice(values, 420, replace=False)
    jitter = rng.uniform(0.0, 0.16, len(values)) * side
    ax.scatter(x + 0.23 * side + jitter, values, s=1.8, color=color, alpha=0.34, linewidths=0, rasterized=True, zorder=1)

def cluster_bootstrap_share(df: pd.DataFrame, column: str, seed: int, n_boot: int=700) -> np.ndarray:
    groups = [g[column].dropna().to_numpy() > 0 for _, g in df.groupby('COUNTRY')]
    groups = [g for g in groups if len(g)]
    if not groups:
        return np.array([])
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        chosen = rng.integers(0, len(groups), len(groups))
        vals = np.concatenate([groups[i] for i in chosen])
        boot.append(vals.mean())
    return np.asarray(boot)

def add_map_density(ax: mpl.axes.Axes, values: np.ndarray, norm: Normalize) -> None:
    inset = ax.inset_axes([0.005, -0.085, 0.16, 0.18])
    inset.set_label('core_expansion_eli_density')
    grid = np.linspace(values.min(), values.max(), 160)
    kde = gaussian_kde(values, bw_method='scott')
    density = kde(grid)
    for i in range(len(grid) - 1):
        inset.fill_between(grid[i:i + 2], density[i:i + 2], color=MAP_CMAP(norm(grid[i:i + 2].mean())), linewidth=0)
    inset.plot(grid, density, color=PALETTE['text'], linewidth=0.6)
    median = float(np.median(values))
    median_height = float(kde(median)[0])
    inset.vlines(median, 0, median_height, color=PALETTE['text'], linewidth=0.5, linestyle=':')
    inset.text(1.0, 1.55, 'Median', transform=inset.transAxes, ha='right', va='bottom', fontsize=4.8 * 1.21)
    inset.annotate('', xy=(median, median_height), xytext=(0.82, 1.48), textcoords='axes fraction', arrowprops={'arrowstyle': '-', 'color': PALETTE['text'], 'linewidth': 0.4, 'connectionstyle': 'angle,angleA=-90,angleB=0,rad=0'}, annotation_clip=False)
    lower = np.floor(values.min() * 20) / 20
    upper = np.ceil(values.max() * 20) / 20
    inset.set(xlim=(lower, upper), ylim=(0, density.max() * 1.1))
    inset.set_xticks([lower, 0, upper], [f'{lower:.2f}', '0', f'{upper:.2f}'])
    inset.set_yticks([])
    for side in ['left', 'right', 'top']:
        inset.spines[side].set_visible(False)
    inset.spines['bottom'].set_linewidth(0.4)
    inset.tick_params(axis='x', labelsize=5.0 * 1.21, pad=1, length=1.5, width=0.4)
    inset.text(0, 1.1, 'Density', transform=inset.transAxes, fontsize=5.5 * 1.21, va='bottom')

def plot_map(ax: mpl.axes.Axes, df: pd.DataFrame) -> None:
    ax.set_facecolor('none')
    ax.patch.set_alpha(0)
    draw_land_background(ax)
    v = 0.2
    im = ax.scatter(df['lon'], df['lat'], c=df['eli_diff'], s=4.5, cmap=MAP_CMAP, norm=Normalize(vmin=-v, vmax=v, clip=True), linewidths=0, alpha=0.82, zorder=2, rasterized=True)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-58, 84)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title('Core-expansion emotional livability gap', fontsize=7.4, pad=4)
    ticks = [-v, 0.0, v]
    add_map_colorbar(ax, im, 'Core - expansion ELI', ticks=ticks, ticklabels=[f'{t:.2f}' for t in ticks])
    add_map_density(ax, df['eli_diff'].dropna().to_numpy(), im.norm)
    values = df.loc[np.isfinite(df['eli_diff']), 'eli_diff'].to_numpy()
    ax.text(0.01, 0.48, f'> 0: {np.count_nonzero(values > 0):,} cities', transform=ax.transAxes, ha='left', va='top', fontsize=6.0, linespacing=1.35, gid='city_sign_positive')
    ax.text(0.01, 0.48, f'< 0: {np.count_nonzero(values < 0):,} cities', transform=ax.transAxes + mpl.transforms.ScaledTranslation(0, -11 / 72, ax.figure.dpi_scale_trans), ha='left', va='top', fontsize=6.0, gid='city_sign_negative')
    shift = mpl.transforms.ScaledTranslation(-0.03 * ax.get_position().width * ax.figure.get_figwidth(), 0, ax.figure.dpi_scale_trans)
    for layer in ax.collections:
        if layer is im:
            layer.set_offset_transform(layer.get_offset_transform() + shift)
        else:
            layer.set_transform(layer.get_transform() + shift)
        layer.set_clip_path(None)
        layer.set_clip_box(mpl.transforms.TransformedBbox(ax.bbox, shift))

def plot_core_expansion_scatter(ax: mpl.axes.Axes, df: pd.DataFrame) -> None:
    valid = df[['expansion_eli', 'core_eli']].replace([np.inf, -np.inf], np.nan).dropna()
    ax.hexbin(df['expansion_eli'], df['core_eli'], gridsize=34, mincnt=1, vmin=2, vmax=60, cmap=LinearSegmentedColormap.from_list('ce_density', [PALETTE['cool_mist'], PALETTE['light_sky'], PALETTE['cobalt']]), linewidths=0, alpha=0.92, rasterized=True)
    lo = min(df['core_eli'].min(), df['expansion_eli'].min())
    hi = max(df['core_eli'].max(), df['expansion_eli'].max())
    ax.plot([lo, hi], [lo, hi], color=PALETTE['muted'], lw=0.7, ls='--')
    if len(valid) > 2:
        r_value, p_value = pearsonr(valid['expansion_eli'], valid['core_eli'])
        p_label = '$P$ < 0.001' if p_value < 0.001 else f'$P$ = {p_value:.3f}'
        ax.text(0.05, 0.94, f'$r$ = {r_value:.2f}\n{p_label}', transform=ax.transAxes, ha='left', va='top', fontsize=5.0, color=PALETTE['text'], linespacing=1.15, zorder=5)
    for group in ['South', 'North']:
        tmp = df[df['ns_group'] == group]
        add_covariance_ellipse(ax, tmp['expansion_eli'].to_numpy(), tmp['core_eli'].to_numpy(), GROUP_COLORS[group])
        ax.scatter(tmp['expansion_eli'].median(), tmp['core_eli'].median(), s=18, color=GROUP_COLORS[group], edgecolor='white', linewidth=0.45, zorder=4, label=group)
    ax.set_xlabel('Expansion ELI', fontsize=6.0)
    ax.set_ylabel('Core ELI', fontsize=6.0)
    ax.set_title('Matched density', fontsize=7.0, pad=3)
    ax.legend(fontsize=5.0, loc='lower right', handlelength=0.8)
    clean_axis(ax, grid=True)

def plot_distribution(ax: mpl.axes.Axes, df: pd.DataFrame) -> None:
    labels = ['South', 'North']
    data = [df.loc[df['ns_group'] == label, 'eli_diff'].dropna().to_numpy() for label in labels]
    for pos, label, vals in zip([0, 1], labels, data):
        vals = vals[np.isfinite(vals)]
        color = GROUP_COLORS[label]
        q05, q25, q50, q75, q95 = np.percentile(vals, [5, 25, 50, 75, 95])
        hist, edges = np.histogram(vals, bins=32)
        centers = (edges[:-1] + edges[1:]) / 2
        half_width = 0.34 * hist / max(hist.max(), 1)
        for yy, hw in zip(centers, half_width):
            ax.plot([pos, pos + hw], [yy, yy], color=color, alpha=0.85, lw=1.0)
        ax.plot([pos, pos], [q05, q95], color=PALETTE['axis'], alpha=0.8, lw=0.55)
        ax.plot([pos - 0.07, pos - 0.07], [q25, q75], color=PALETTE['axis'], alpha=0.8, lw=1.6, solid_capstyle='butt')
        ax.plot([pos + 0.07, pos + 0.07], [q25, q75], color=PALETTE['axis'], alpha=0.8, lw=1.6, solid_capstyle='butt')
        ax.plot([pos - 0.07, pos + 0.07], [q25, q25], color=PALETTE['axis'], alpha=0.8, lw=0.55)
        ax.plot([pos - 0.07, pos + 0.07], [q75, q75], color=PALETTE['axis'], alpha=0.8, lw=0.55)
        ax.plot([pos - 0.075, pos + 0.075], [q50, q50], color=PALETTE['text'], alpha=0.8, lw=0.75, zorder=5)
        for yy, hw in zip(centers, half_width):
            ax.plot([pos - 0.2, pos - 0.2 - hw * 0.42], [yy, yy], color=color, alpha=0.85, lw=0.48)
        n_y = min(np.nanpercentile(vals, 98.5) + 0.07, 0.605)
        n_x = pos
        if label == 'South':
            n_y -= 0.02
            n_x = 0.51
        ax.text(n_x, n_y, f'n={len(vals):,}', ha='center', va='bottom', fontsize=5.0, color=PALETTE['muted'])
    diff, lo, hi = bootstrap_difference_ci(data[0], data[1], seed=441)
    y_top = max(np.nanpercentile(np.concatenate(data), 99.0), 0.44)
    ax.plot([0, 1], [y_top + 0.07, y_top + 0.07], color=PALETTE['axis'], lw=0.45)
    ax.plot([0, 0], [y_top + 0.058, y_top + 0.07], color=PALETTE['axis'], lw=0.45)
    ax.plot([1, 1], [y_top + 0.058, y_top + 0.07], color=PALETTE['axis'], lw=0.45)
    ax.text(0.5, y_top + 0.085, f'Delta N-S = {diff:+.2f} [{lo:+.2f}, {hi:+.2f}]', ha='center', va='bottom', fontsize=4.2, color=PALETTE['muted'])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=5.8)
    ax.set_ylabel('Core - expansion ELI', fontsize=6.0)
    ax.set_title('Expansion penalty distribution', fontsize=7.0, pad=3)
    ax.set_xlim(-0.48, 1.5)
    ax.set_ylim(-0.2, 0.6)
    ax.set_yticks([-0.2, 0.0, 0.2, 0.4, 0.6])
    clean_axis(ax, grid=True)

def plot_effect_forest(ax: mpl.axes.Axes, df: pd.DataFrame) -> None:
    rows = [('ELI', 'eli_diff')] + [(label, f'benefit_{e}') for e, label in EMOTIONS.items()]
    y = np.arange(len(rows), dtype=float)[::-1]
    y[0] += 0.4
    for yi, (label, col) in zip(y, rows):
        vals = df[col].dropna().to_numpy()
        est = vals.mean()
        q05, q25, q50, q75, q95 = np.percentile(vals, [5, 25, 50, 75, 95])
        color = PALETTE['cobalt'] if est >= 0 else PALETTE['burnt_orange']
        ax.bxp([{'q1': q25, 'med': q50, 'q3': q75, 'whislo': q05, 'whishi': q95}], positions=[yi], widths=0.38, vert=False, patch_artist=True, showfliers=False, manage_ticks=False, boxprops={'facecolor': color, 'edgecolor': color, 'alpha': 0.25, 'linewidth': 0.8}, whiskerprops={'color': color, 'linewidth': 0.8}, capprops={'color': color, 'linewidth': 0.8}, medianprops={'color': PALETTE['text'], 'linewidth': 0.9})
        ax.plot(est, yi, marker='o', ms=3.4, color=color, markeredgecolor='white', markeredgewidth=0.35, zorder=4)
    ax.axhline(5.7, color=PALETTE['cool_mist'], lw=0.55, zorder=0)
    ax.axvline(0, color=PALETTE['muted'], lw=0.6, ls='--')
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=5.6)
    ax.set_xlim(-0.16, 0.24)
    ax.set_ylim(-0.55, 6.95)
    ax.set_xticks([-0.1, 0.0, 0.1, 0.2])
    ax.set_xlabel('Core advantage (score difference)', fontsize=6.0)
    ax.set_title('Distribution of core advantage', fontsize=7.0, pad=3)
    clean_axis(ax, grid=True)

def plot_heatmap(ax: mpl.axes.Axes, df: pd.DataFrame) -> None:
    paired_columns = [f'{zone}_{e}' for zone in ['core', 'expansion'] for e in EMOTIONS]
    df = df.loc[np.isfinite(df[paired_columns]).all(axis=1)]
    scores = np.asarray([[df.loc[df['ns_group'] == group, f'{zone}_{e}'].mean() for e in EMOTIONS] for group in ['South', 'North'] for zone in ['core', 'expansion']])
    mat = []
    sig = []
    for group in ['South', 'North']:
        mat.append([df.loc[df['ns_group'] == group, f'benefit_{e}'].mean() for e in EMOTIONS])
        group_sig = []
        for e in EMOTIONS:
            vals = df.loc[df['ns_group'] == group, f'benefit_{e}'].dropna().to_numpy()
            boot = bootstrap_means(vals, seed=70 + len(group_sig) + 10 * len(sig))
            lo, hi = np.percentile(boot, [2.5, 97.5])
            group_sig.append(lo > 0 or hi < 0)
        sig.append(group_sig)
    mat = np.asarray(mat)
    score_im = ax.pcolormesh(np.arange(len(EMOTIONS) + 1), np.arange(5), scores, cmap=MAP_CMAP, norm=Normalize(vmin=0, vmax=1), edgecolors='white', linewidth=0.9, shading='flat')
    im = ax.pcolormesh(np.arange(mat.shape[1] + 1), np.arange(mat.shape[0] + 1) + 4.25, mat, cmap=MAP_CMAP, norm=TwoSlopeNorm(vmin=-0.1, vcenter=0.0, vmax=0.1), edgecolors='white', linewidth=0.9, shading='flat')
    ax.set_xlim(0, mat.shape[1])
    ax.set_ylim(6.25, 0)
    ax.set_xticks(np.arange(len(EMOTIONS)) + 0.5)
    ax.set_xticklabels([EMOTIONS[e] for e in EMOTIONS], rotation=45, ha='right', fontsize=5.0)
    ax.set_yticks([0.5, 1.5, 2.5, 3.5, 4.75, 5.75])
    ax.set_yticklabels(['South core', 'South expansion', 'North core', 'North expansion', 'South gap', 'North gap'], fontsize=5.4)
    ax.tick_params(axis='both', length=2.5, width=0.45, colors=PALETTE['axis'])
    ax.set_title('Emotion scores and core-expansion gaps', fontsize=7.0, pad=3)
    ax.axhline(4.125, color=PALETTE['muted'], linewidth=0.4)
    for (i, j), val in np.ndenumerate(scores):
        ax.text(j + 0.5, i + 0.5, f'{val:.2f}', ha='center', va='center', fontsize=4.8, color=PALETTE['text'])
    for (i, j), val in np.ndenumerate(mat):
        ax.text(j + 0.5, i + 4.75, f'{val:+.2f}', ha='center', va='center', fontsize=4.8, color=PALETTE['text'])
        if sig[i][j]:
            ax.text(j + 0.83, i + 4.5, '*', ha='center', va='center', fontsize=5.4, color=PALETTE['text'])
    for spine in ax.spines.values():
        spine.set_visible(False)
    for image, y, height, ticks, label in [(score_im, 0.39, 0.58, [0, 0.5, 1], 'Score'), (im, 0.03, 0.26, [-0.1, 0, 0.1], 'Gap')]:
        axes_width = ax.get_position().width * ax.figure.get_figwidth()
        cax = ax.inset_axes([1 + 0.041 / axes_width, y, 0.074 / axes_width, height])
        cbar = ax.figure.colorbar(image, cax=cax, ticks=ticks)
        cbar.outline.set_linewidth(0.45)
        cbar.ax.tick_params(labelsize=5.0, length=2.5, width=0.45, pad=1, colors=PALETTE['axis'])
        cbar.set_label(label, fontsize=6.0, labelpad=2)

def plot_slopegraph(ax: mpl.axes.Axes, df: pd.DataFrame) -> None:
    ordered_items = list(EMOTIONS.items())
    label_items = []
    for idx, (emotion, label) in enumerate(ordered_items):
        core_vals = df[f'core_oriented_{emotion}'].dropna().to_numpy()
        exp_vals = df[f'exp_oriented_{emotion}'].dropna().to_numpy()
        core_boot = bootstrap_means(core_vals, seed=91 + idx)
        exp_boot = bootstrap_means(exp_vals, seed=111 + idx)
        core = core_vals.mean()
        exp = exp_vals.mean()
        core_lo, core_hi = np.percentile(core_boot, [2.5, 97.5])
        exp_lo, exp_hi = np.percentile(exp_boot, [2.5, 97.5])
        core_qlo, core_qhi = np.percentile(core_vals, [10, 90])
        exp_qlo, exp_qhi = np.percentile(exp_vals, [10, 90])
        delta = core - exp
        color = PALETTE['cobalt'] if delta >= 0 else PALETTE['burnt_orange']
        draw_horizontal_density(ax, core_vals, core, color, height=0.015, alpha=0.13)
        draw_horizontal_density(ax, exp_vals, exp, color, height=0.015, alpha=0.13)
        ax.fill_between([0, 1], [core_lo, exp_lo], [core_hi, exp_hi], color=color, alpha=0.1, lw=0, zorder=1)
        ax.plot([0, 0], [core_qlo, core_qhi], color=PALETTE['cool_mist'], lw=2.8, alpha=0.72, solid_capstyle='round', zorder=1)
        ax.plot([1, 1], [exp_qlo, exp_qhi], color=PALETTE['cool_mist'], lw=2.8, alpha=0.72, solid_capstyle='round', zorder=1)
        ax.plot([0, 1], [core, exp], color=color, lw=1.05, marker='o', ms=2.8, zorder=3)
        ax.plot([0, 0], [core_lo, core_hi], color=color, lw=0.7, alpha=0.8, zorder=2)
        ax.plot([1, 1], [exp_lo, exp_hi], color=color, lw=0.7, alpha=0.8, zorder=2)
        ax.plot([-0.022, 0.022], [core_lo, core_lo], color=color, lw=0.55, alpha=0.8, zorder=2)
        ax.plot([-0.022, 0.022], [core_hi, core_hi], color=color, lw=0.55, alpha=0.8, zorder=2)
        ax.plot([0.978, 1.022], [exp_lo, exp_lo], color=color, lw=0.55, alpha=0.8, zorder=2)
        ax.plot([0.978, 1.022], [exp_hi, exp_hi], color=color, lw=0.55, alpha=0.8, zorder=2)
        label_items.append((exp, label, color, delta))
    label_items = sorted(label_items, key=lambda x: x[0])
    min_gap = 0.035
    placed = []
    for y0, label, color, delta in label_items:
        y = y0 if not placed else max(y0, placed[-1][0] + min_gap)
        placed.append((y, y0, label, color, delta))
    overflow = placed[-1][0] - 0.72 if placed else 0
    if overflow > 0:
        placed = [(y - overflow, y0, label, color, delta) for y, y0, label, color, delta in placed]
    for y, y0, label, color, delta in placed:
        ax.plot([1.01, 1.08], [y0, y], color=color, lw=0.45, alpha=0.7)
        ax.text(1.1, y, label, fontsize=4.75, va='center', color=color)
        ax.plot([1.42, 1.57], [y, y], color=color, lw=0.62, alpha=0.4)
        ax.scatter(1.5, y, s=7.0 + min(abs(delta) * 85, 15), color=color, edgecolor='white', linewidth=0.25, zorder=4)
        ax.text(1.6, y, f'{delta:+.2f}', fontsize=4.15, va='center', ha='left', color=PALETTE['muted'])
    ax.text(1.5, 0.795, 'Delta', ha='center', va='bottom', fontsize=4.5, color=PALETTE['muted'])
    ax.set_xlim(-0.07, 1.82)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Core', 'Expansion'])
    ax.set_yticks(np.arange(0.2, 0.81, 0.2))
    ax.set_ylabel('Direction-adjusted score', fontsize=6.0)
    ax.set_title('Paired emotion profile', fontsize=7.0, pad=3)
    clean_axis(ax, grid=True)

def plot_waterfall(ax: mpl.axes.Axes, df: pd.DataFrame) -> None:
    rows = []
    for e in POSITIVE:
        rows.append((EMOTIONS[e], df[f'benefit_{e}'].dropna().to_numpy() / len(POSITIVE)))
    for e in NEGATIVE:
        rows.append((EMOTIONS[e], df[f'benefit_{e}'].dropna().to_numpy() / len(NEGATIVE)))
    rows = sorted(rows, key=lambda item: np.nanmean(item[1]))
    y = np.arange(len(rows))
    for row_idx, (yi, (label, vals)) in enumerate(zip(y, rows)):
        boot = bootstrap_means(vals, seed=131 + row_idx)
        est, lo, hi = (vals.mean(), *np.percentile(boot, [2.5, 97.5]))
        color = PALETTE['cobalt'] if est >= 0 else PALETTE['burnt_orange']
        draw_horizontal_density(ax, vals, yi, color, height=0.17)
        ax.plot([lo, hi], [yi, yi], color=color, lw=1.0)
        ax.plot(est, yi, marker='o', ms=3.0, color=color, zorder=3)
    ax.axvline(0, color=PALETTE['muted'], lw=0.6, ls='--')
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=5.2)
    ax.set_xlim(-0.04, 0.105)
    ax.set_xlabel('Contribution to ELI gap', fontsize=6.0)
    ax.set_title('Contribution forest', fontsize=7.0, pad=3)
    clean_axis(ax, grid=True)

def plot_typology(ax: mpl.axes.Axes, df: pd.DataFrame) -> None:
    df = df.loc[np.isfinite(df['eli_diff'])]
    thr = 0.03
    classes = np.select([df['eli_diff'] > thr, df['eli_diff'] < -thr], ['core advantage', 'expansion advantage'], default='balanced')
    tmp = df.assign(class_=classes)
    comp = pd.crosstab(tmp['ns_group'], tmp['class_'], normalize='index')
    colors = {'core advantage': PALETTE['cobalt'], 'balanced': PALETTE['warm_sand'], 'expansion advantage': PALETTE['burnt_orange']}
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(comp.index))
    row_segments = {}
    row_bands = {}
    for row_i, group in enumerate(comp.index):
        left = 0.0
        row_segments[group] = {}
        row_bands[group] = (row_i + 0.12, row_i + 0.82)
        vals_group = tmp.loc[tmp['ns_group'] == group, 'class_'].to_numpy()
        rng = np.random.default_rng(170 + row_i)
        boot_bounds = []
        for _ in range(600):
            draw = rng.choice(vals_group, len(vals_group), replace=True)
            shares = pd.Series(draw).value_counts(normalize=True)
            boot_bounds.append([shares.get('expansion advantage', 0.0), shares.get('expansion advantage', 0.0) + shares.get('balanced', 0.0)])
        boot_bounds = np.asarray(boot_bounds)
        for cls in ['expansion advantage', 'balanced', 'core advantage']:
            val = float(comp.loc[group, cls]) if cls in comp else 0.0
            row_segments[group][cls] = (left, left + val)
            ax.add_patch(Rectangle((left, row_i + 0.12), val, 0.7, facecolor=colors[cls], edgecolor='white', lw=0.45))
            if val > 0:
                short = {'expansion advantage': 'exp', 'balanced': 'bal', 'core advantage': 'core'}[cls]
                ax.text(left + val / 2, row_i + 0.47, short, ha='center', va='center', fontsize=5.0, rotation=90 if val < 0.08 else 0, gid='typology_class_label', color='white' if cls == 'core advantage' else PALETTE['text'])
            left += val
        for bound_idx in range(2):
            lo, hi = np.percentile(boot_bounds[:, bound_idx], [2.5, 97.5])
            mid = np.median(boot_bounds[:, bound_idx])
            ax.add_patch(Rectangle((lo, row_i + 0.86), hi - lo, 0.045, facecolor=PALETTE['cool_mist'], edgecolor='none', alpha=0.8))
            ax.plot([mid, mid], [row_i + 0.84, row_i + 0.93], color=PALETTE['axis'], lw=0.45)
        ax.text(-0.02, row_i + 0.47, group, ha='right', va='center', fontsize=5.6, color=PALETTE['text'])
    if set(comp.index) >= {'South', 'North'}:
        south_y0, south_y1 = row_bands['South']
        north_y0, north_y1 = row_bands['North']
        if south_y0 > north_y0:
            upper_edge = south_y0
            lower_edge = north_y1
        else:
            upper_edge = north_y0
            lower_edge = south_y1
        for cls in ['expansion advantage', 'balanced', 'core advantage']:
            s0, s1 = row_segments['South'][cls]
            n0, n1 = row_segments['North'][cls]
            ax.add_patch(Polygon([(s0, upper_edge), (s1, upper_edge), (n1, lower_edge), (n0, lower_edge)], closed=True, facecolor=colors[cls], edgecolor='none', alpha=0.11 if cls != 'core advantage' else 0.09, zorder=0))
            diff = n1 - n0 - (s1 - s0)
            if abs(diff) >= 0.025:
                mid = (s0 + s1 + n0 + n1) / 4
                ax.text(mid, (upper_edge + lower_edge) / 2, f'{diff:+.2f}', ha='center', va='center', fontsize=3.85, color=PALETTE['muted'], alpha=0.82, zorder=5)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_yticks([])
    ax.set_xlabel('Share of cities (%)', fontsize=6.0)
    ax.set_title('Gap typology mosaic', fontsize=7.0, pad=3)
    clean_axis(ax, grid=False)

def plot_rank(ax: mpl.axes.Axes, df: pd.DataFrame) -> None:
    rank = pd.concat([df.nsmallest(6, 'eli_diff'), df.nlargest(6, 'eli_diff')]).sort_values('eli_diff')
    y = np.arange(len(rank))
    colors = np.where(rank['eli_diff'] >= 0, PALETTE['cobalt'], PALETTE['burnt_orange'])
    ax.hlines(y, 0, rank['eli_diff'], color=colors, lw=1.0)
    ax.scatter(rank['eli_diff'], y, s=12, color=colors, zorder=3)
    ax.axvline(0, color=PALETTE['muted'], lw=0.6, ls='--')
    ax.set_yticks(y)
    ax.set_yticklabels([str(x)[:11] + ('...' if len(str(x)) > 11 else '') for x in rank['NAME_1']], fontsize=4.7)
    ax.set_xlabel('Core - expansion ELI', fontsize=6.0)
    ax.set_title('Extreme city gaps', fontsize=7.0, pad=3)
    clean_axis(ax, grid=True)

def plot_penalty_scatter(ax: mpl.axes.Axes, df: pd.DataFrame) -> None:
    v = np.nanpercentile(np.abs(df['eli_diff']), 98)
    ax.hexbin(df['expansion_eli'], df['eli_diff'], C=df['eli_diff'], reduce_C_function=np.mean, gridsize=32, mincnt=1, cmap=DIV_CMAP, vmin=-v, vmax=v, linewidths=0, alpha=0.92, rasterized=True)
    ax.axhline(0, color=PALETTE['muted'], lw=0.6, ls='--')
    ax.set_xlabel('Expansion ELI', fontsize=6.0)
    ax.set_ylabel('Core - expansion ELI', fontsize=6.0)
    ax.set_title('Penalty-baseline density', fontsize=7.0, pad=3)
    clean_axis(ax, grid=True)

def plot_sign_agreement(ax: mpl.axes.Axes, df: pd.DataFrame) -> None:
    vals = pd.Series({EMOTIONS[e]: (df[f'benefit_{e}'].dropna() > 0).mean() for e in EMOTIONS}).sort_values()
    y = np.arange(len(vals))
    for row_idx, (yi, (label, val)) in enumerate(zip(y, vals.items())):
        emotion = next((k for k, v in EMOTIONS.items() if v == label))
        post = cluster_bootstrap_share(df, f'benefit_{emotion}', seed=210 + row_idx)
        lo, hi = np.percentile(post, [2.5, 97.5]) if len(post) else (val, val)
        color = PALETTE['cobalt'] if val >= 0.5 else PALETTE['burnt_orange']
        ax.plot([lo, hi], [yi, yi], color=color, lw=1.2, zorder=3)
        ax.vlines([lo, hi], yi - 0.055, yi + 0.055, color=color, lw=0.7, zorder=3)
        ax.plot(val, yi, marker='o', ms=3.4, color=color, markeredgecolor='white', markeredgewidth=0.35, zorder=4)
        ax.text(1.27, yi, f'{val * 100:.0f}% [{lo * 100:.0f}-{hi * 100:.0f}]', ha='center', va='center', fontsize=4.2, color=PALETTE['muted'])
    ax.plot([0.5, 0.5], [-0.5, 5.4], color=PALETTE['muted'], lw=0.6, ls='--', zorder=0)
    ax.text(1.27, 5.8, 'Share\n[95% CI]', ha='center', va='center', fontsize=5.4)
    ax.set_yticks(y)
    ax.set_yticklabels(vals.index, fontsize=5.4)
    ax.set_xlim(0, 1.48)
    ax.set_ylim(-0.55, 6.25)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.spines['bottom'].set_bounds(0, 1)
    ax.spines['left'].set_bounds(-0.55, 6.25)
    ax.set_xlabel('Cities with core advantage (%)', fontsize=6.0)
    ax.set_title('Prevalence of core advantage', fontsize=7.0, pad=3)
    clean_axis(ax, grid=True)

def plot_ns_summary(ax: mpl.axes.Axes, df: pd.DataFrame) -> None:
    labels = ['South', 'North']
    y = np.arange(len(labels))
    for yi, group in zip(y, labels):
        vals = df.loc[df['ns_group'] == group, 'eli_diff'].dropna().to_numpy()
        est, lo, hi = bootstrap_mean_ci(vals)
        color = GROUP_COLORS[group]
        ax.plot([lo, hi], [yi, yi], color=color, lw=1.2)
        ax.plot(est, yi, marker='o', ms=4.0, color=color)
        ax.text(hi, yi + 0.1, f'n={len(vals):,}', fontsize=5.2, color=PALETTE['muted'], ha='left')
    ax.axvline(0, color=PALETTE['muted'], lw=0.6, ls='--')
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Mean core-expansion ELI gap', fontsize=6.0)
    ax.set_title('North-South gap summary', fontsize=7.0, pad=3)
    clean_axis(ax, grid=True)

def build_figure() -> plt.Figure:
    city = load_city_metadata()
    ce = load_ce_data()
    df = city.merge(ce, on='id_1', how='inner')
    width = 10.2
    left, right, wspace = (0.07, 0.95, 1.0)
    column_width = width * (right - left) / (6 + 5 * wspace)
    row_heights = [column_width * (4 + 3 * wspace) * 142 / 360, 1.85, 1.85]
    row_gap = 0.9
    bottom_margin, top_margin = (0.4, 0.35)
    height = sum(row_heights) + 2 * row_gap + bottom_margin + top_margin
    fig = plt.figure(figsize=(width, height), dpi=300)
    gs = fig.add_gridspec(3, 6, left=left, right=right, bottom=bottom_margin / height, top=1 - top_margin / height, height_ratios=row_heights, hspace=row_gap / np.mean(row_heights), wspace=wspace)
    axes = []
    specs = [(gs[0, 0:4], plot_map), (gs[1, 0:2], plot_core_expansion_scatter), (gs[1, 4:6], plot_distribution), (gs[0, 4:6], plot_effect_forest), (gs[2, 3:6], plot_heatmap), (gs[2, 0:3], plot_typology), (gs[1, 2:4], plot_sign_agreement)]
    comparison_height = gs[1, 0:2].get_position(fig).height
    for spec, func in specs:
        ax = fig.add_subplot(spec)
        if func in (plot_core_expansion_scatter, plot_distribution, plot_effect_forest, plot_sign_agreement):
            pos = ax.get_position()
            ax.set_position([pos.x0, pos.y1 - comparison_height, pos.width, comparison_height])
        if func in (plot_heatmap, plot_typology):
            pos = ax.get_position()
            panel_width = pos.width * 0.96
            panel_left = pos.x0 if func is plot_typology else pos.x1 - panel_width
            ax.set_position([panel_left, pos.y0, panel_width, pos.height])
        func(ax, df)
        axes.append(ax)
    for label, ax in zip(list('acebgfd'), axes):
        pos = ax.get_position()
        add_panel_label(ax, label, x=-0.02 / pos.width, y=1 + 0.065 / (pos.height * height))
        ax.title.set_transform(ax.title.get_transform() + mpl.transforms.ScaledTranslation(0, 4 / 72, fig.dpi_scale_trans))
    font_sizes = {'panel': 10.5, 'title': 9.0, 'axis': 8.0, 'detail': 7.0}
    font_sizes = {role: size * 1.08 for role, size in font_sizes.items()}
    for text in fig.findobj(Text):
        text.set_fontsize(font_sizes['detail'])
    for ax in axes:
        ax.title.set_fontsize(font_sizes['title'])
        ax.xaxis.label.set_fontsize(font_sizes['axis'])
        ax.yaxis.label.set_fontsize(font_sizes['axis'])
        ax.xaxis.set_major_locator(FixedLocator(ax.get_xticks()))
        ax.yaxis.set_major_locator(FixedLocator(ax.get_yticks()))
        ax.tick_params(axis='both', labelsize=font_sizes['detail'])
        for text in ax.texts:
            if text.get_gid() == 'panel_label':
                text.set_fontsize(font_sizes['panel'])
        for inset in ax.child_axes:
            inset.title.set_fontsize(font_sizes['axis'])
            inset.xaxis.label.set_fontsize(font_sizes['axis'])
            inset.yaxis.label.set_fontsize(font_sizes['axis'])
            inset.xaxis.set_major_locator(FixedLocator(inset.get_xticks()))
            inset.yaxis.set_major_locator(FixedLocator(inset.get_yticks()))
            inset.tick_params(axis='both', labelsize=font_sizes['detail'])
            for text in inset.texts:
                if text.get_text() == 'Density':
                    text.set_fontsize(font_sizes['axis'])
    force_black_text_except(fig, skip_axes=set())
    for text in axes[0].texts:
        if text.get_gid() == 'city_sign_positive':
            text.set_color('#0D47A1')
        elif text.get_gid() == 'city_sign_negative':
            text.set_color('#F27C2B')
    for text in axes[5].texts:
        if text.get_gid() == 'typology_class_label':
            text.set_fontsize(font_sizes['detail'] * 1.2)
        if text.get_text() == 'core':
            text.set_color('white')
    ax_b = axes[3]
    pos = ax_b.get_position()
    b_shift_inches = 4.0 / 25.4
    ax_b.set_position([pos.x0, pos.y0 - b_shift_inches / height, pos.width, pos.height])
    heading_offset = mpl.transforms.ScaledTranslation(0, b_shift_inches, fig.dpi_scale_trans)
    ax_b.title.set_transform(ax_b.title.get_transform() + heading_offset)
    for text in ax_b.texts:
        if text.get_gid() == 'panel_label':
            text.set_transform(text.get_transform() + heading_offset)
    row_axes = [axes[1], axes[6], axes[2]]
    row_bottom = row_axes[0].get_position().y0
    for ax in row_axes:
        pos = ax.get_position()
        ax.set_position([pos.x0, row_bottom, pos.width, comparison_height])
        ax.spines['bottom'].set_position(('axes', 0))
        ax.tick_params(axis='x', pad=3.5)
    fig.align_xlabels(row_axes)
    header_offset = mpl.transforms.ScaledTranslation(0, 1.0 / 25.4, fig.dpi_scale_trans)
    for ax in row_axes:
        ax.title.set_transform(ax.title.get_transform() + header_offset)
        for text in ax.texts:
            if text.get_gid() == 'panel_label':
                text.set_transform(text.get_transform() + header_offset)
    return fig

def save_figure(fig: plt.Figure) -> None:
    base = OUT_DIR / 'fig_2_4_3_core_expansion_inequality_upgrade'
    fig.savefig(base.with_suffix('.svg'), bbox_inches='tight')
    fig.savefig(base.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(base.with_suffix('.tiff'), dpi=600, bbox_inches='tight')
    fig.savefig(base.with_suffix('.png'), dpi=450, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {base}')
