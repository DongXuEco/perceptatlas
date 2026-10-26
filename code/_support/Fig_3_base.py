from __future__ import annotations
from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
import os
import math
import struct
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Ellipse
from matplotlib.text import Text
import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from scipy.stats import gaussian_kde
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
DERIVED_DIR = PROJECT_ROOT / 'derived_data'
OUT_DIR = output_dir('Fig_3')
OUT_DIR.mkdir(parents=True, exist_ok=True)
PALETTE = {'pale_apricot': '#FFE7CC', 'soft_orange': '#FFB66D', 'burnt_orange': '#F27C2B', 'warm_sand': '#E7C9A0', 'cool_mist': '#DCE6F2', 'light_sky': '#A7CFF2', 'cerulean': '#3B82D6', 'cobalt': '#0D47A1', 'text': '#1F2937', 'muted': '#6B7280', 'axis': '#30343B', 'grid': '#E5E7EB'}
GROUP_COLORS = {'South': PALETTE['burnt_orange'], 'North': PALETTE['cerulean']}
LAND_COLOR = '#E9EBF1'
MAP_COLORBAR_REL_THICKNESS = 0.039
MEAN_INEQUALITY_HEXBIN_MINCNT = 10
_LAND_GDF = None
SEQ_CMAP = LinearSegmentedColormap.from_list('ce_density', [PALETTE['cool_mist'], PALETTE['light_sky'], PALETTE['cobalt']])
MAP_CMAP = LinearSegmentedColormap.from_list('unified_map_blue_orange', [PALETTE['burnt_orange'], PALETTE['soft_orange'], PALETTE['pale_apricot'], PALETTE['cool_mist'], PALETTE['light_sky'], PALETTE['cerulean'], PALETTE['cobalt']])
EMOTIONS = {'beauty': 'Beauty', 'wealthy': 'Wealth', 'safer': 'Safety', 'livelier': 'Liveliness', 'boring': 'Boredom', 'depression': 'Depression'}
ALIASES = {'safety': 'safer', 'safe': 'safer', 'lively': 'livelier', 'liveliness': 'livelier', 'depressing': 'depression', 'depressed': 'depression', 'wealth': 'wealthy'}
POSITIVE = ['beauty', 'wealthy', 'safer', 'livelier']
NEGATIVE = ['boring', 'depression']
TYPOLOGY_COLORS = {'high mean / low inequality': PALETTE['cobalt'], 'high mean / high inequality': PALETTE['cerulean'], 'low mean / low inequality': PALETTE['warm_sand'], 'low mean / high inequality': PALETTE['burnt_orange']}
TYPOLOGY_SHORT = {'high mean / low inequality': 'HM/LI', 'high mean / high inequality': 'HM/HI', 'low mean / low inequality': 'LM/LI', 'low mean / high inequality': 'LM/HI'}

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
    for col in ['lon', 'lat', 'id_1', 'NS', *EMOTIONS.keys()]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['lon', 'lat', 'id_1', 'NS', *EMOTIONS.keys()]).copy()
    df['id_1'] = df['id_1'].astype(int)
    df['ns_group'] = np.where(df['NS'].round().astype(int) == 0, 'North', 'South')
    df['eli'] = df[POSITIVE].mean(axis=1) - df[NEGATIVE].mean(axis=1)
    return df[['id_1', 'COUNTRY', 'NAME_1', 'lon', 'lat', 'ns_group', 'eli']].copy()

def normalize_emotion(value: str) -> str:
    v = str(value).strip().lower()
    v = ALIASES.get(v, v)
    if v not in EMOTIONS:
        raise ValueError(f'Unknown emotion name: {value}')
    return v

def load_inequality_metrics() -> pd.DataFrame:
    path = DERIVED_DIR / 'city_emotion_inequality_50m.csv'
    if not resource(path).exists():
        raise FileNotFoundError('Missing derived_data/city_emotion_inequality_50m.csv. Create it from the 50 m emotion surface with columns: id_1, emotion, mean, gini, theil, p90_p10, n_pixels.')
    df = pd.read_csv(resource(path))
    df.columns = [c.strip().lower() for c in df.columns]
    if 'city_id' in df.columns and 'id_1' not in df.columns:
        df = df.rename(columns={'city_id': 'id_1'})
    required = {'id_1', 'emotion', 'mean', 'gini', 'theil', 'p90_p10'}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f'Missing columns in {path}: {missing}')
    df['id_1'] = pd.to_numeric(df['id_1'], errors='coerce').astype('Int64')
    df['emotion'] = df['emotion'].map(normalize_emotion)
    for col in ['mean', 'gini', 'theil', 'p90_p10', 'n_pixels']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['id_1', 'emotion', 'mean', 'gini', 'theil', 'p90_p10']).copy()
    df['id_1'] = df['id_1'].astype(int)
    return df

def add_panel_label(ax: mpl.axes.Axes, label: str, x: float=-0.08, y: float=1.06) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha='left', va='bottom', fontsize=8.5, fontweight='bold', color=PALETTE['text'])

def clean_axis(ax: mpl.axes.Axes, grid: bool=False) -> None:
    ax.tick_params(axis='both', labelsize=5.8, length=2.5, width=0.45, colors=PALETTE['axis'])
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color(PALETTE['axis'])
    ax.grid(False)

def force_black_text(fig: mpl.figure.Figure) -> None:
    for text in fig.findobj(match=lambda artist: isinstance(artist, Text)):
        text.set_color('#000000')
    for ax in fig.axes:
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

def corr_pvalue_approx(r: float, n: int) -> float:
    if n <= 3 or not np.isfinite(r) or abs(r) >= 1:
        return 0.0
    z = np.arctanh(r) * math.sqrt(n - 3)
    return math.erfc(abs(z) / math.sqrt(2))

def find_world_land() -> Path:
    return PACKAGE_ROOT / 'data/shared/World_land_polygons.geojson'

def draw_land_background(ax: mpl.axes.Axes, x_offset: float=0.0) -> None:
    global _LAND_GDF
    if _LAND_GDF is None:
        _LAND_GDF = gpd.read_file(resource(find_world_land())).to_crs('EPSG:4326')
    land = _LAND_GDF
    if x_offset:
        land = land.copy()
        land['geometry'] = land.geometry.translate(xoff=x_offset)
    land.plot(ax=ax, facecolor=LAND_COLOR, edgecolor='none', linewidth=0, zorder=0)

def add_map_colorbar(ax: mpl.axes.Axes, mappable: mpl.cm.ScalarMappable, label: str, ticks: list[float], ticklabels: list[str] | None=None, extend: str='neither') -> None:
    right_shift = 8 / 25.4 / (ax.get_position().width * ax.figure.get_figwidth())
    cax = ax.inset_axes([0.23 + right_shift, -0.085, 0.44 * 0.85, MAP_COLORBAR_REL_THICKNESS * 1.1])
    cbar = ax.figure.colorbar(mappable, cax=cax, orientation='horizontal', ticks=ticks, extend=extend)
    cbar.outline.set_linewidth(0.45)
    cbar.ax.tick_params(labelsize=5.0, length=1.8, width=0.4, pad=1, colors=PALETTE['axis'])
    cbar.ax.set_title(label, fontsize=5.5, pad=5, color=PALETTE['text'])
    if ticklabels is not None:
        cbar.ax.set_xticklabels(ticklabels)

def raincloud(ax: mpl.axes.Axes, values_by_group: dict[str, np.ndarray], ylabel: str) -> None:
    labels = ['South', 'North']
    data = [np.asarray(values_by_group[g], dtype=float) for g in labels]
    parts = ax.violinplot(data, positions=[0, 1], widths=0.72, showmeans=False, showextrema=False)
    for body, label in zip(parts['bodies'], labels):
        body.set_facecolor(GROUP_COLORS[label])
        body.set_edgecolor('none')
        body.set_alpha(0.25)
    ax.boxplot(data, positions=[0, 1], widths=0.16, patch_artist=True, showfliers=False, medianprops={'color': PALETTE['text'], 'lw': 0.7}, boxprops={'facecolor': 'white', 'edgecolor': PALETTE['axis'], 'lw': 0.55}, whiskerprops={'color': PALETTE['axis'], 'lw': 0.5}, capprops={'color': PALETTE['axis'], 'lw': 0.5})
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel, fontsize=6.0)
    clean_axis(ax, grid=True)

def raincloud_reference(ax: mpl.axes.Axes, values_by_group: dict[str, np.ndarray], ylabel: str) -> None:
    labels = ['South', 'North']
    data = [np.asarray(values_by_group[label], dtype=float) for label in labels]
    for pos, label, vals in zip([0, 1], labels, data):
        vals = vals[np.isfinite(vals)]
        q05, q25, q50, q75, q95 = np.percentile(vals, [5, 25, 50, 75, 95])
        hist, edges = np.histogram(vals, bins=32)
        centers = (edges[:-1] + edges[1:]) / 2
        half_width = 0.34 * hist / max(hist.max(), 1)
        for yy, hw in zip(centers, half_width):
            ax.plot([pos, pos + hw], [yy, yy], color=GROUP_COLORS[label], alpha=0.55, lw=1.0)
        ax.plot([pos, pos], [q05, q95], color=PALETTE['axis'], lw=0.55)
        ax.plot([pos - 0.07, pos - 0.07], [q25, q75], color=PALETTE['axis'], lw=1.6, solid_capstyle='butt')
        ax.plot([pos + 0.07, pos + 0.07], [q25, q75], color=PALETTE['axis'], lw=1.6, solid_capstyle='butt')
        ax.plot([pos - 0.07, pos + 0.07], [q25, q25], color=PALETTE['axis'], lw=0.55)
        ax.plot([pos - 0.07, pos + 0.07], [q75, q75], color=PALETTE['axis'], lw=0.55)
        ax.plot([pos - 0.075, pos + 0.075], [q50, q50], color=PALETTE['text'], lw=0.75, zorder=5)
        for yy, hw in zip(centers, half_width):
            ax.plot([pos - 0.2, pos - 0.2 - hw * 0.42], [yy, yy], color=GROUP_COLORS[label], alpha=0.78, lw=0.48)
        ax.text(pos, np.nanmax(vals), f'n={len(vals):,}', ha='center', va='bottom', fontsize=5.0, color=PALETTE['muted'])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=5.8)
    ax.set_ylabel(ylabel, fontsize=6.2)
    clean_axis(ax, grid=True)

def bootstrap_mean_difference(values_by_group: dict[str, np.ndarray], n_boot: int=1200, seed: int=608) -> tuple[float, float, float]:
    north = np.asarray(values_by_group['North'], dtype=float)
    south = np.asarray(values_by_group['South'], dtype=float)
    north = north[np.isfinite(north)]
    south = south[np.isfinite(south)]
    diff = float(np.mean(north) - np.mean(south))
    rng = np.random.default_rng(seed)
    north_draws = rng.choice(north, size=(n_boot, len(north)), replace=True).mean(axis=1)
    south_draws = rng.choice(south, size=(n_boot, len(south)), replace=True).mean(axis=1)
    lo, hi = np.percentile(north_draws - south_draws, [2.5, 97.5])
    return (diff, float(lo), float(hi))

def annotate_mean_difference(ax: mpl.axes.Axes, values_by_group: dict[str, np.ndarray], with_ci: bool=False, xy: tuple[float, float]=(0.98, 0.82), ha: str='right') -> None:
    diff, lo, hi = bootstrap_mean_difference(values_by_group)
    if with_ci:
        text = f'Δ={diff:+.3f}\n95% CI [{lo:+.3f}, {hi:+.3f}]'
        fontsize = 4.25
    else:
        text = f'Δ={diff:+.3f}'
        fontsize = 4.45
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, ha=ha, va='top', fontsize=fontsize, color=PALETTE['muted'], linespacing=1.25, zorder=10)

def make_city_table(metrics: pd.DataFrame, city: pd.DataFrame) -> pd.DataFrame:
    gini = metrics.pivot_table(index='id_1', columns='emotion', values='gini', aggfunc='mean')
    theil = metrics.pivot_table(index='id_1', columns='emotion', values='theil', aggfunc='mean')
    p9010 = metrics.pivot_table(index='id_1', columns='emotion', values='p90_p10', aggfunc='mean')
    out = city.merge(gini.add_prefix('gini_').reset_index(), on='id_1', how='inner')
    out = out.merge(theil.add_prefix('theil_').reset_index(), on='id_1', how='left')
    out = out.merge(p9010.add_prefix('p9010_').reset_index(), on='id_1', how='left')
    out['compound_gini'] = out[[f'gini_{e}' for e in EMOTIONS]].mean(axis=1)
    out['compound_theil'] = out[[f'theil_{e}' for e in EMOTIONS]].mean(axis=1)
    high_mean = out['eli'] >= out['eli'].median()
    high_ineq = out['compound_gini'] >= out['compound_gini'].median()
    out['typology'] = np.select([high_mean & ~high_ineq, high_mean & high_ineq, ~high_mean & ~high_ineq, ~high_mean & high_ineq], ['high mean / low inequality', 'high mean / high inequality', 'low mean / low inequality', 'low mean / high inequality'], default='unclassified')
    return out

def add_map_density(ax: mpl.axes.Axes, values: np.ndarray, norm: Normalize) -> None:
    inset = ax.inset_axes([0.005, -0.085, 0.16, 0.18])
    inset.set_label('composite_gini_density')
    grid = np.linspace(values.min(), values.max(), 160)
    kde = gaussian_kde(values, bw_method='scott')
    density = kde(grid)
    for i in range(len(grid) - 1):
        inset.fill_between(grid[i:i + 2], density[i:i + 2], color=MAP_CMAP(norm(grid[i:i + 2].mean())), linewidth=0)
    inset.plot(grid, density, color=PALETTE['text'], linewidth=0.6)
    median = float(np.median(values))
    median_height = float(kde(median)[0])
    inset.vlines(median, 0, median_height, color=PALETTE['text'], linewidth=0.5, linestyle=':')
    inset.text(1.0, 1.55, 'Median', transform=inset.transAxes, ha='right', va='bottom', fontsize=4.8)
    inset.annotate('', xy=(median, median_height), xytext=(0.82, 1.48), textcoords='axes fraction', arrowprops={'arrowstyle': '-', 'color': PALETTE['text'], 'linewidth': 0.4, 'connectionstyle': 'angle,angleA=-90,angleB=0,rad=0'}, annotation_clip=False)
    lower = np.floor(values.min() * 20) / 20
    upper = np.ceil(values.max() * 20) / 20
    inset.set(xlim=(lower, upper), ylim=(0, density.max() * 1.1))
    inset.set_xticks([lower, upper], [f'{lower:.2f}', f'{upper:.2f}'])
    inset.set_yticks([])
    for side in ['left', 'right', 'top']:
        inset.spines[side].set_visible(False)
    inset.spines['bottom'].set_linewidth(0.4)
    inset.tick_params(axis='x', labelsize=5.0, pad=1, length=1.5, width=0.4)
    inset.text(0, 1.1, 'Density', transform=inset.transAxes, fontsize=5.5, va='bottom')

def plot_map(ax: mpl.axes.Axes, df: pd.DataFrame) -> None:
    ax.set_facecolor('none')
    ax.patch.set_alpha(0)
    content_x_offset = -18.0
    draw_land_background(ax, x_offset=content_x_offset)
    vmin, vmax = np.nanpercentile(df['compound_gini'], [2, 98])
    im = ax.scatter(df['lon'] + content_x_offset, df['lat'], c=df['compound_gini'], s=4.5, cmap=MAP_CMAP, norm=Normalize(vmin, vmax), linewidths=0, alpha=0.82, zorder=2, rasterized=True)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-58, 84)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title('Within-city emotion inequality', fontsize=7.4, pad=4)
    ticks = [vmin, (vmin + vmax) / 2, vmax]
    add_map_colorbar(ax, im, 'Composite Gini', ticks=ticks, ticklabels=[f'{t:.2f}' for t in ticks])
    add_map_density(ax, df['compound_gini'].to_numpy(), im.norm)

def plot_mean_inequality_scatter(ax: mpl.axes.Axes, df: pd.DataFrame) -> None:
    xm, ym = (df['eli'].median(), df['compound_gini'].median())
    hb = ax.hexbin(df['eli'], df['compound_gini'], gridsize=34, mincnt=MEAN_INEQUALITY_HEXBIN_MINCNT, cmap=SEQ_CMAP, linewidths=0, alpha=0.9, rasterized=True)
    ax.axvline(xm, color=PALETTE['muted'], lw=0.6, ls='--')
    ax.axhline(ym, color=PALETTE['muted'], lw=0.6, ls='--')
    quadrants = {'HM/HI': (df['eli'] >= xm) & (df['compound_gini'] >= ym), 'HM/LI': (df['eli'] >= xm) & (df['compound_gini'] < ym), 'LM/HI': (df['eli'] < xm) & (df['compound_gini'] >= ym), 'LM/LI': (df['eli'] < xm) & (df['compound_gini'] < ym)}
    xlim = (df['eli'].quantile(0.01), df['eli'].quantile(0.99))
    ylim = (df['compound_gini'].quantile(0.01), df['compound_gini'].quantile(0.99))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    for group in ['South', 'North']:
        tmp = df[df['ns_group'] == group]
        add_covariance_ellipse(ax, tmp['eli'].to_numpy(), tmp['compound_gini'].to_numpy(), GROUP_COLORS[group])
    positions = {'HM/HI': (0.98, 0.94, 'right', 'top'), 'HM/LI': (0.98, 0.08, 'right', 'bottom'), 'LM/HI': (0.03, 0.94, 'left', 'top'), 'LM/LI': (0.03, 0.08, 'left', 'bottom')}
    for label, mask in quadrants.items():
        x, y, ha, va = positions[label]
        ax.text(x, y, f'{label}\n{100 * mask.mean():.1f}%', transform=ax.transAxes, ha=ha, va=va, fontsize=4.8, color=PALETTE['text'])
    label_offsets = {'South': (0.012, 0.003, 'left', 'bottom'), 'North': (-0.012, -0.003, 'right', 'top')}
    for group in ['South', 'North']:
        tmp = df[df['ns_group'] == group]
        mx = tmp['eli'].median()
        my = tmp['compound_gini'].median()
        ax.scatter(mx, my, s=18, color=GROUP_COLORS[group], edgecolor='white', linewidth=0.45, zorder=4)
        dx, dy, ha, va = label_offsets[group]
        ax.text(mx + dx, my + dy, group, ha=ha, va=va, fontsize=4.7, color=GROUP_COLORS[group])
    ax.set_xlabel('Mean ELI', fontsize=6.0)
    ax.set_ylabel('Composite Gini', fontsize=6.0)
    ax.set_title('Mean-inequality density', fontsize=7.0, pad=3)
    clean_axis(ax, grid=True)

def bootstrap_typology_boundaries(df: pd.DataFrame, order: list[str], group_order: list[str], n_boot: int=800) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(524)
    out = {}
    for group in group_order:
        vals = df.loc[df['ns_group'] == group, 'typology'].dropna().to_numpy()
        if len(vals) == 0:
            out[group] = np.full((len(order) - 1, 3), np.nan)
            continue
        draws = rng.integers(0, len(vals), size=(n_boot, len(vals)))
        shares = np.column_stack([(vals[draws] == typ).mean(axis=1) for typ in order])
        boundaries = np.cumsum(shares[:, :-1], axis=1)
        out[group] = np.percentile(boundaries, [2.5, 50, 97.5], axis=0).T
    return out

def plot_typology_composition(ax: mpl.axes.Axes, df: pd.DataFrame) -> None:
    order = list(TYPOLOGY_COLORS)
    comp = pd.crosstab(df['ns_group'], df['typology'], normalize='index').reindex(index=['North', 'South'], columns=order, fill_value=0).fillna(0)
    y = np.arange(len(comp.index))
    bar_height = 0.42
    starts = comp.cumsum(axis=1).shift(axis=1, fill_value=0)
    ends = comp.cumsum(axis=1)
    if {'North', 'South'}.issubset(comp.index):
        north_y = int(np.where(comp.index.to_numpy() == 'North')[0][0])
        south_y = int(np.where(comp.index.to_numpy() == 'South')[0][0])
        for typ in order:
            polygon = mpl.patches.Polygon([(starts.loc['North', typ], north_y + bar_height / 2), (ends.loc['North', typ], north_y + bar_height / 2), (ends.loc['South', typ], south_y - bar_height / 2), (starts.loc['South', typ], south_y - bar_height / 2)], closed=True, facecolor=TYPOLOGY_COLORS[typ], edgecolor='none', alpha=0.13, zorder=1)
            ax.add_patch(polygon)
    for typ in order:
        vals = comp[typ].to_numpy()
        ax.barh(y, vals, left=starts[typ].to_numpy(), height=bar_height, color=TYPOLOGY_COLORS[typ], edgecolor='white', lw=0.4, label=typ, zorder=3)
        for yy, x0, width in zip(y, starts[typ].to_numpy(), vals):
            if width >= 0.16:
                text_color = 'white' if typ in {'high mean / low inequality', 'high mean / high inequality'} else PALETTE['text']
                ax.text(x0 + width / 2, yy, TYPOLOGY_SHORT[typ], ha='center', va='center', fontsize=4.7, color=text_color, zorder=4)
    boundary_ci = bootstrap_typology_boundaries(df, order, list(comp.index))
    for yy, group in zip(y, comp.index):
        offset = 0.31 if group == 'South' else -0.31
        for lo, mid, hi in boundary_ci[group]:
            if not np.isfinite(mid):
                continue
            ax.plot([lo, hi], [yy + offset, yy + offset], color=PALETTE['muted'], lw=0.55, alpha=0.4, zorder=5)
            ax.plot([mid, mid], [yy + offset - 0.045, yy + offset + 0.045], color=PALETTE['muted'], lw=0.55, alpha=0.75, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels(comp.index)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlabel('Share of cities', fontsize=6.0)
    ax.set_title('Typology composition', fontsize=7.0, pad=3)
    clean_axis(ax, grid=False)

def plot_metric_heatmap(ax: mpl.axes.Axes, city_table: pd.DataFrame, cbar_width: float) -> None:
    cols = ['compound_gini', 'compound_theil'] + [f'gini_{e}' for e in EMOTIONS]
    mat = city_table[cols].corr().to_numpy()
    labels = ['Gini', 'Theil'] + [EMOTIONS[e] for e in EMOTIONS]
    n = len(labels)
    plot_mat = np.clip(mat, 0, 1)
    mask = np.triu(np.ones_like(plot_mat, dtype=bool), k=1)
    corr_cmap = MAP_CMAP.copy()
    corr_cmap.set_bad(color='white', alpha=0)
    norm = Normalize(vmin=0, vmax=1)
    im = ax.imshow(np.ma.masked_where(mask, plot_mat), cmap=corr_cmap, norm=norm, aspect='auto', interpolation='nearest')
    for i in range(n):
        for j in range(i + 1, n):
            r = mat[i, j]
            valid = city_table[[cols[i], cols[j]]].dropna()
            p = corr_pvalue_approx(float(r), len(valid))
            size = 6 + 26 * min(abs(float(r)), 1)
            ax.scatter(j, i, s=size, color=MAP_CMAP(norm(np.clip(r, 0, 1))), edgecolor='none', linewidth=0, alpha=0.9, zorder=4)
            if p < 0.001:
                sig = '***'
            elif p < 0.01:
                sig = '**'
            elif p < 0.05:
                sig = '*'
            else:
                sig = ''
            if sig:
                ax.text(j, i - 0.28, sig, ha='center', va='center', fontsize=4.8, color=PALETTE['text'], zorder=5)
    for i in range(1, n):
        for j in range(i):
            r = mat[i, j]
            color = 'white' if r >= 0.72 else PALETTE['text']
            ax.text(j, i, f'{r:.2f}', ha='center', va='center', fontsize=4.8, color=color, zorder=5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=4.9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=4.9)
    ax.tick_params(axis='both', length=2.5, width=0.45, colors=PALETTE['axis'])
    ax.set_title('Inequality metric concordance', fontsize=7.0, pad=3)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    for edge in np.arange(-0.5, n + 0.5, 1):
        ax.axhline(edge, color='white', lw=0.9, zorder=3)
        ax.axvline(edge, color='white', lw=0.9, zorder=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    pos = ax.get_position()
    cax = ax.figure.add_axes([pos.x1 + 0.006, pos.y0, cbar_width, pos.height])
    cbar = ax.figure.colorbar(im, cax=cax, ticks=np.arange(0, 1.01, 0.2))
    cbar.outline.set_linewidth(0.45)
    cbar.ax.tick_params(labelsize=5.0, length=2.5, width=0.45, colors=PALETTE['axis'])

def plot_city_ranking(ax: mpl.axes.Axes, df: pd.DataFrame) -> None:
    rank = df.sort_values('compound_gini').tail(11)
    y = np.arange(len(rank))
    q75, q90, q95, q99 = np.nanpercentile(df['compound_gini'], [75, 90, 95, 99])
    ax.axvspan(q75, q95, color=PALETTE['cool_mist'], alpha=0.55, lw=0)
    ax.axvspan(q95, df['compound_gini'].max(), color=PALETTE['light_sky'], alpha=0.35, lw=0)
    hist, edges = np.histogram(df['compound_gini'].dropna().to_numpy(), bins=44)
    centers = (edges[:-1] + edges[1:]) / 2
    density = hist / max(hist.max(), 1) * 0.46
    baseline = -0.88
    ax.fill_between(centers, baseline, baseline + density, color=PALETTE['cool_mist'], alpha=0.72, lw=0, zorder=1)
    ax.plot(centers, baseline + density, color=PALETTE['cerulean'], lw=0.48, alpha=0.65, zorder=2)
    for q, label in [(q90, 'p90'), (q95, 'p95'), (q99, 'p99')]:
        ax.axvline(q, color=PALETTE['muted'], lw=0.45, ls='--', alpha=0.62, zorder=2)
        ax.text(q, len(rank) - 0.38, label, rotation=90, ha='right', va='top', fontsize=4.3, color=PALETTE['muted'])
    rng = np.random.default_rng(37)
    emotion_cols = [f'gini_{e}' for e in EMOTIONS]
    for yi, (_, row) in zip(y, rank.iterrows()):
        color = GROUP_COLORS[row['ns_group']]
        vals = pd.to_numeric(row[emotion_cols], errors='coerce').to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) > 1:
            boot = rng.choice(vals, size=(600, len(vals)), replace=True).mean(axis=1)
            lo, hi = np.nanpercentile(boot, [2.5, 97.5])
            ax.plot([lo, hi], [yi, yi], color=color, lw=2.7, alpha=0.18, solid_capstyle='round', zorder=2)
            ax.plot([lo, lo], [yi - 0.06, yi + 0.06], color=color, lw=0.5, alpha=0.55, zorder=3)
            ax.plot([hi, hi], [yi - 0.06, yi + 0.06], color=color, lw=0.5, alpha=0.55, zorder=3)
        ax.hlines(yi, q75, row['compound_gini'], color=color, lw=0.9, alpha=0.85)
        ax.scatter(row['compound_gini'], yi, s=13, color=color, edgecolor='white', linewidth=0.35, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([str(x)[:17] for x in rank['NAME_1']], fontsize=5.0)
    ax.set_ylim(-1.05, len(rank) - 0.35)
    ax.set_xlabel('Composite Gini', fontsize=6.0)
    ax.set_title('High-inequality tail', fontsize=7.0, pad=3)
    clean_axis(ax, grid=True)

def build_figure() -> plt.Figure:
    metrics = load_inequality_metrics()
    city = load_city_metadata()
    table = make_city_table(metrics, city)
    width = 7.8
    left, right, wspace = (0.08, 0.94, 0.5)
    column_width = width * (right - left) / (3 + 2 * wspace)
    row_heights = [column_width * (2 + wspace) * 142 / 360, 1.55, 1.55, 1.7]
    row_gap = 0.65
    bottom_margin, top_margin = (0.4, 0.35)
    height = sum(row_heights) + 3 * row_gap + bottom_margin + top_margin
    fig = plt.figure(figsize=(width, height), dpi=300)
    gs = fig.add_gridspec(4, 3, left=left, right=right, bottom=bottom_margin / height, top=1 - top_margin / height, height_ratios=row_heights, hspace=row_gap / np.mean(row_heights), wspace=wspace)
    axes = []
    ax = fig.add_subplot(gs[0, 1:3])
    plot_map(ax, table)
    map_cbar_thickness = ax.get_position().height * MAP_COLORBAR_REL_THICKNESS
    axes.append(ax)
    ax = fig.add_subplot(gs[0, 0])
    b_values = {g: table.loc[table['ns_group'] == g, 'compound_gini'].to_numpy() for g in ['South', 'North']}
    raincloud_reference(ax, b_values, 'Composite Gini')
    ax.set_ylim(0, 0.18)
    ax.set_yticks(np.arange(0, 0.181, 0.04))
    annotate_mean_difference(ax, b_values, with_ci=True, xy=(0.5, 0.68), ha='center')
    ax.set_title('Internal inequality', fontsize=7.0, pad=3)
    axes.append(ax)
    ax = fig.add_subplot(gs[3, 1])
    plot_mean_inequality_scatter(ax, table)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
    axes.append(ax)
    ax = fig.add_subplot(gs[3, 2])
    plot_typology_composition(ax, table)
    axes.append(ax)
    ax = fig.add_subplot(gs[3, 0])
    plot_metric_heatmap(ax, table, map_cbar_thickness)
    axes.append(ax)
    emotion_specs = [gs[1, 0], gs[1, 1], gs[1, 2], gs[2, 0], gs[2, 1], gs[2, 2]]
    for spec, (emotion, label) in zip(emotion_specs, EMOTIONS.items()):
        ax = fig.add_subplot(spec)
        col = f'gini_{emotion}'
        emotion_values = {g: table.loc[table['ns_group'] == g, col].dropna().to_numpy() for g in ['South', 'North']}
        raincloud_reference(ax, emotion_values, 'Gini')
        if emotion in {'wealthy', 'depression'}:
            ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05))
        if emotion == 'beauty':
            ax.set_ylim(0, 0.18)
            ax.set_yticks(np.arange(0, 0.181, 0.04))
        if emotion == 'safer':
            ymin, ymax = ax.get_ylim()
            ax.set_yticks(np.arange(max(0.0, ymin), ymax + 0.02, 0.04))
        if emotion == 'livelier':
            ymin, ymax = ax.get_ylim()
            ax.set_yticks(np.arange(max(0.0, ymin), ymax + 0.05, 0.1))
        annotate_mean_difference(ax, emotion_values, with_ci=False, xy=(0.5, 0.9), ha='center')
        ax.set_title(label, fontsize=6.8, pad=3)
        axes.append(ax)
    for label, ax in zip(list('bajkicdefgh'), axes):
        pos = ax.get_position()
        add_panel_label(ax, label, x=-0.018 / pos.width, y=1 + 0.065 / (pos.height * height))
        ax.title.set_transform(ax.title.get_transform() + mpl.transforms.ScaledTranslation(0, 4 / 72, fig.dpi_scale_trans))
    force_black_text(fig)
    for text in axes[2].texts:
        if text.get_text() in GROUP_COLORS:
            text.set_color(GROUP_COLORS[text.get_text()])
    for text in fig.findobj(mpl.text.Text):
        text.set_fontsize(text.get_fontsize() * 1.21)
    for ax in axes[5:8]:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 - 1.5 / 25.4 / height, pos.width, pos.height])
    return fig

def save_figure(fig: plt.Figure) -> None:
    base = OUT_DIR / 'fig_2_4_2_within_city_inequality_typology_upgrade_v2'
    fig.savefig(base.with_suffix('.svg'), bbox_inches='tight')
    fig.savefig(base.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(base.with_suffix('.tiff'), dpi=600, bbox_inches='tight')
    fig.savefig(base.with_suffix('.png'), dpi=450, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {base}')
