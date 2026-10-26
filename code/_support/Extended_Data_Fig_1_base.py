from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.transforms import blended_transform_factory
ROOT = PACKAGE_ROOT
OUT_DIR = output_dir('Extended_Data_Fig_1')
F_TABLE = OUT_DIR / 'q1_f_multiscale_aggregation_stability.csv'
G_TABLE = OUT_DIR / 'q1_g_walking_satisfaction_validation.csv'
H_TABLE = OUT_DIR / 'q1_h_eli_weight_robustness.csv'
EC_QOL_DIR = ROOT / '影响变量数据' / '外部验证' / 'European_Commission_QoL_European_Cities_2023' / 'processed'
EC_QOL_UNITS = EC_QOL_DIR / 'ec_qol_eli_validation_units.csv'
EC_QOL_CORR = EC_QOL_DIR / 'ec_qol_eli_correlation_summary.csv'
FIG_BASENAME = 'fig_q1_measurement_validation'
COLORS = {'north': '#3B82D6', 'south': '#F27C2B', 'blue': '#208fbd', 'cyan': '#53a9c5', 'light_blue': '#DCEAF7', 'orange': '#F27C2B', 'red': '#C83F26', 'grey': '#7F8AA9', 'pale': '#F3F6F9', 'text': '#000000', 'axis': '#2C3238', 'grid': '#E7EBEF'}
FONT_BASE = 7.4
FONT_AXIS_LABEL = 7.8
FONT_TICK = 7.2
FONT_PANEL_LABEL = 9.8
FONT_PANEL_TITLE = 8.4
FONT_INPANEL_TITLE = 6.8
FONT_BODY = 7.0
FONT_NOTE = 6.5
FONT_SMALL = 6.2
QOL_PANEL_INDICATORS = [('green_space_satisfaction_top2', 'Green space'), ('public_transport_satisfaction_top2', 'Public transport'), ('healthcare_satisfaction_top2', 'Health care'), ('life_satisfaction_top2', 'Life satisfaction'), ('public_space_satisfaction_top2', 'Public space'), ('not_lonely_12m_bot2', 'Low loneliness')]
QOL_PANEL_COLORS = ['#D73A3B', '#2479B4', '#FF8113', '#2DA02D', '#9469BD', '#818181']

def setup_style():
    plt.rcParams.update({'font.family': 'Arial', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.weight': 'normal', 'font.size': FONT_BASE, 'axes.titlesize': 8.4, 'axes.titleweight': 'normal', 'axes.labelsize': FONT_AXIS_LABEL, 'xtick.labelsize': FONT_TICK, 'ytick.labelsize': FONT_TICK, 'axes.linewidth': 0.75, 'axes.edgecolor': COLORS['axis'], 'xtick.color': COLORS['text'], 'ytick.color': COLORS['text'], 'text.color': COLORS['text'], 'axes.labelcolor': COLORS['text'], 'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

def strip_axis(ax, keep=('left', 'bottom')):
    for side in ['top', 'right', 'left', 'bottom']:
        ax.spines[side].set_visible(side in keep)
    ax.tick_params(direction='out', length=3, width=0.7, pad=2)
    ax.grid(False)

def add_panel_label(fig, ax, label, title, dx=0.0, dy=0.012):
    bbox = ax.get_position()
    y = bbox.y1 + dy
    fig.text(bbox.x0 - 0.025 + dx, y, label, fontsize=FONT_PANEL_LABEL, fontweight='bold', ha='left', va='bottom', color='black')
    fig.text(bbox.x0 + 0.006 + dx, y + 0.001, title, fontsize=FONT_PANEL_TITLE, ha='left', va='bottom', color='black')

def add_panel_label_inside(ax, label, title):
    ax.text(0.016, 0.972, label, transform=ax.transAxes, ha='left', va='top', fontsize=FONT_PANEL_LABEL, fontweight='bold', color='black', zorder=8)
    ax.text(0.072, 0.972, title, transform=ax.transAxes, ha='left', va='top', fontsize=FONT_PANEL_TITLE, color='black', zorder=8)

def moving_quantiles(x, y, bins):
    out = []
    cats = pd.cut(x, bins=bins, include_lowest=True)
    for interval, sub_y in pd.Series(y).groupby(cats, observed=True):
        vals = sub_y.dropna().values
        if len(vals) < 8:
            continue
        mid = (interval.left + interval.right) / 2
        out.append((mid, np.nanquantile(vals, 0.25), np.nanquantile(vals, 0.5), np.nanquantile(vals, 0.75), len(vals)))
    return pd.DataFrame(out, columns=['x', 'q25', 'q50', 'q75', 'n'])

def gaussian_density(values, xmin, xmax, n=300):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    xs = np.linspace(xmin, xmax, n)
    if len(values) < 2:
        return (xs, np.zeros_like(xs))
    std = np.std(values)
    bw = 1.06 * std * len(values) ** (-1 / 5)
    bw = max(bw, (xmax - xmin) / 120)
    z = (xs[:, None] - values[None, :]) / bw
    dens = np.exp(-0.5 * z * z).sum(axis=1) / (len(values) * bw * np.sqrt(2 * np.pi))
    return (xs, dens)

def fisher_ci(r, n):
    r = np.clip(float(r), -0.999999, 0.999999)
    if n <= 3:
        return (np.nan, np.nan)
    z = np.arctanh(r)
    se = 1 / math.sqrt(n - 3)
    lo, hi = (z - 1.96 * se, z + 1.96 * se)
    return (np.tanh(lo), np.tanh(hi))

def draw_panel_f(ax, f):
    n = len(f)
    f = f.copy()
    f['rank_pct'] = 100 * (f['baseline_rank'] - 1) / max(n - 1, 1)
    f['rank_width_pct'] = 100 * f['bootstrap_rank_width95'] / n
    density_cmap = mcolors.LinearSegmentedColormap.from_list('q1_rank_density', ['#FFFFFF', '#EAF2FB', '#3B82D6'])
    hb = ax.hexbin(f['rank_pct'], f['rank_width_pct'], gridsize=(58, 24), extent=(0, 100, 0, 20), mincnt=1, linewidths=0, cmap=density_cmap, norm=mcolors.LogNorm(vmin=1, vmax=20), alpha=0.96, rasterized=True, zorder=1)
    q = moving_quantiles(f['rank_pct'], f['rank_width_pct'], np.linspace(0, 100, 31))
    ax.plot(q['x'], q['q50'], color='#F1863C', lw=2.2, ls=(0, (0.65, 1.35)), dash_capstyle='round', solid_capstyle='round', zorder=3)
    ax.text(3.8, 18.1, f"50 m -> city aggregation\nPearson r = 1.000\nSpearman ρ = 1.000\nmedian 95% width = {f['bootstrap_rank_width95'].median():.0f} ranks", ha='left', va='top', fontsize=FONT_BODY, linespacing=1.58, bbox=dict(boxstyle='round,pad=0.24', facecolor='white', edgecolor='none', linewidth=0.0, alpha=0.86))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 20)
    ax.set_xlabel('50 m city ELI rank percentile')
    ax.set_ylabel('95% rank width (%)')
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 5, 10, 15, 20])
    strip_axis(ax)
    grad = np.linspace(0, 1, 160, dtype='float32').reshape(1, -1)
    gx0, gx1 = (75.0, 97.0)
    gy0, gy1 = (16.47, 17.28)
    ax.imshow(grad, extent=(gx0, gx1, gy0, gy1), origin='lower', cmap=hb.cmap, aspect='auto', interpolation='nearest', zorder=5)
    ax.plot([gx0, gx1, gx1, gx0, gx0], [gy0, gy0, gy1, gy1, gy0], color='black', lw=0.45, zorder=6)
    ax.text((gx0 + gx1) / 2.0, gy1 + 0.18, 'point density', ha='center', va='bottom', fontsize=FONT_SMALL, color='black', zorder=6)
    tick_y0 = gy0
    tick_y1 = gy0 - 0.45
    label_y = tick_y1 - 0.16
    for tick_x, tick_label in ((gx0, 'Low'), (gx1, 'High')):
        ax.plot([tick_x, tick_x], [tick_y0, tick_y1], color='black', lw=0.42, zorder=6)
        ax.text(tick_x, label_y, tick_label, ha='center', va='top', fontsize=FONT_SMALL, color='black', zorder=6)
    line_y = label_y - 1.7
    line_x0 = gx0 - 2.0
    ax.plot([line_x0, line_x0 + 8.0], [line_y, line_y], color='#F1863C', lw=2.2, ls=(0, (0.65, 1.35)), dash_capstyle='round', solid_capstyle='round', zorder=6)
    ax.text(line_x0 + 9.3, line_y, 'moving median', ha='left', va='center', fontsize=FONT_SMALL, color='black', zorder=6)

def validation_scope(df):
    distance = pd.to_numeric(df['min_match_distance_km'], errors='coerce')
    exact = df['match_methods'].astype(str).str.contains('project_name', na=False)
    near = distance.le(150)
    missing_distance = distance.isna()
    return df[exact | near | missing_distance].copy()

def format_p_text(value):
    if not np.isfinite(value):
        return 'P=n/a'
    if value < 0.001:
        return 'P < 0.001'
    return f'P={value:.3f}'

def draw_panel_g(fig, gs, qol_units, qol_corr):
    axes = []
    qol_units = validation_scope(qol_units)
    qol_corr = qol_corr[qol_corr['scope'].eq('within_150km_or_exact')].copy()
    for i, (indicator, title) in enumerate(QOL_PANEL_INDICATORS):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        axes.append(ax)
        color = QOL_PANEL_COLORS[i]
        sub = qol_units[qol_units['indicator'].eq(indicator)].dropna(subset=['eli50', 'qol_value']).copy()
        x = sub['eli50'].astype(float).values
        y = 100 * sub['qol_value'].astype(float).values
        ax.scatter(x, y, s=10.5, color=color, alpha=0.72, edgecolor='white', linewidth=0.18, rasterized=True, zorder=3)
        if len(sub) > 2 and np.nanstd(x) > 0:
            coef = np.polyfit(x, y, 1)
            xs = np.linspace(np.nanmin(x), np.nanmax(x), 80)
            ax.plot(xs, coef[0] * xs + coef[1], color='black', lw=0.85, zorder=4)
        corr_row = qol_corr[qol_corr['indicator'].eq(indicator) & qol_corr['outcome'].eq('eli50')]
        if not corr_row.empty:
            row = corr_row.iloc[0]
            stat = f"r={row['pearson_r']:+.2f}\nrho={row['spearman_rho']:+.2f}\n{format_p_text(row['spearman_p'])}"
            ax.text(0.045, 0.985, stat, transform=ax.transAxes, ha='left', va='top', fontsize=5.9, linespacing=1.12, color=COLORS['axis'], zorder=6)
        ax.text(0.045, 0.055, title, transform=ax.transAxes, ha='left', va='bottom', fontsize=5.4, color='black', zorder=6)
        ax.set_xlim(-0.12, 0.24)
        ax.set_ylim(30, 120)
        ax.set_xticks([-0.1, 0.0, 0.1, 0.2])
        ax.set_yticks([30, 60, 90, 120])
        ax.grid(True, color='#EDF2F7', lw=0.42, zorder=0)
        if i % 3 == 0:
            ax.set_ylabel('positive response (%)', fontsize=FONT_AXIS_LABEL)
        else:
            ax.set_yticklabels([])
        if i // 3 == 1:
            ax.set_xlabel('city ELI', fontsize=FONT_AXIS_LABEL)
        else:
            ax.set_xticklabels([])
        strip_axis(ax)
    return axes

def enforce_panel_g_alignment(axes):
    fig = axes[0].figure
    fig.canvas.draw()
    pos = [ax.get_position() for ax in axes]
    left = min((p.x0 for p in pos))
    right = max((p.x1 for p in pos))
    bottom = min((p.y0 for p in pos))
    top = max((p.y1 for p in pos))
    gap_x = min(pos[1].x0 - pos[0].x1, pos[2].x0 - pos[1].x1, pos[4].x0 - pos[3].x1, pos[5].x0 - pos[4].x1)
    gap_y = min(pos[0].y0 - pos[3].y1, pos[1].y0 - pos[4].y1, pos[2].y0 - pos[5].y1)
    width = (right - left - 2 * gap_x) / 3
    height = (top - bottom - gap_y) / 2
    x0 = [left, left + width + gap_x, left + 2 * (width + gap_x)]
    y0 = [bottom + height + gap_y, bottom]
    for i, ax in enumerate(axes):
        row, col = divmod(i, 3)
        ax.set_position([x0[col], y0[row], width, height])

def add_panel_g_note(fig, axes):
    pos = [ax.get_position() for ax in axes]
    x0 = min((p.x0 for p in pos))
    y0 = min((p.y0 for p in pos))
    fig.text(x0, y0 - 0.062, 'EC QoL survey positive response shares; P values are Spearman tests.', ha='left', va='top', fontsize=FONT_SMALL, color='black')

def align_bottom_panels_to_top_columns(ax_left_top, axes_right_top, ax_left_bottom, ax_right_bottom):
    fig = ax_left_top.figure
    fig.canvas.draw()
    left_top = ax_left_top.get_position()
    right_positions = [ax.get_position() for ax in axes_right_top]
    right_x0 = min((p.x0 for p in right_positions))
    right_x1 = max((p.x1 for p in right_positions))
    left_bottom = ax_left_bottom.get_position()
    ax_left_bottom.set_position([left_top.x0, left_bottom.y0, left_top.width, left_bottom.height])
    ax_right_bottom.set_position([right_x0, left_bottom.y0, right_x1 - right_x0, left_bottom.height])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    left_tight = ax_left_bottom.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
    right_tight = ax_right_bottom.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
    bottom_extra = left_tight.y0 - right_tight.y0
    if bottom_extra > 0:
        right_pos = ax_right_bottom.get_position()
        ax_right_bottom.set_position([right_pos.x0, right_pos.y0 + bottom_extra, right_pos.width, max(right_pos.height - bottom_extra, 0.05)])

def expand_axis_downward(ax, extra=0.035):
    pos = ax.get_position()
    ax.set_position([pos.x0, max(pos.y0 - extra, 0.01), pos.width, pos.height + min(extra, pos.y0 - 0.01)])

def align_tight_bottom_to_fixed(ax_fixed, ax_to_move):
    fig = ax_fixed.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fixed_tight = ax_fixed.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
    move_tight = ax_to_move.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
    delta = fixed_tight.y0 - move_tight.y0
    pos = ax_to_move.get_position()
    ax_to_move.set_position([pos.x0, pos.y0 + delta, pos.width, pos.height])

def extend_axis_left(ax, extra=0.045):
    pos = ax.get_position()
    x1 = pos.x1
    x0 = max(pos.x0 - extra, 0.01)
    ax.set_position([x0, pos.y0, x1 - x0, pos.height])

def matrix_color(value, col_name, baseline_delta):
    if col_name in ['rho', 'top10', 'bottom10']:
        cmap = plt.get_cmap('Blues')
        v = np.clip(value, 0, 1)
        return cmap(0.16 + 0.72 * v)
    vmax = 0.13
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=baseline_delta, vmax=vmax)
    return plt.get_cmap('RdBu_r')(norm(value))

def draw_robustness_matrix(ax, h):
    rows = [('baseline_eli50', 'Baseline', COLORS['axis']), ('direction_adjusted_equal_six_dimensions', 'Equal six', COLORS['blue']), ('positive_dimensions_only', 'Positive only', COLORS['cyan']), ('negative_dimensions_inverted_only', 'Negative only', COLORS['orange']), ('leave_one_out_beauty', 'Drop beauty', COLORS['grey']), ('leave_one_out_wealthy', 'Drop wealth', COLORS['grey']), ('leave_one_out_safer', 'Drop safety', COLORS['grey']), ('leave_one_out_livelier', 'Drop lively', COLORS['grey']), ('leave_one_out_boring', 'Drop boring', COLORS['grey']), ('leave_one_out_depression', 'Drop depression', COLORS['grey'])]
    cols = [('spearman_rho_with_eli50', 'ρ', 'rho'), ('top_decile_retention', 'top', 'top10'), ('bottom_decile_retention', 'bottom', 'bottom10'), ('north_minus_south_mean', 'N-S Δ', 'ns')]
    h_indexed = h.set_index('specification')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ['top', 'right', 'left', 'bottom']:
        ax.spines[side].set_visible(False)
    ax.text(0.0, 0.965, 'Baseline and alternative ELI constructions.', ha='left', va='center', fontsize=FONT_BODY)
    col_x = [0.47, 0.61, 0.75, 0.94]
    for (_, label, _), x in zip(cols, col_x):
        ax.text(x, 0.875, label, ha='center', va='center', fontsize=FONT_BODY)
    ax.plot([0.0, 1.0], [0.815, 0.815], color='#DDE3EA', lw=0.75, clip_on=False)
    y_rows = np.linspace(0.745, 0.105, len(rows))
    row_half_height = 0.03
    for idx, (y, (spec, row_label, _color)) in enumerate(zip(y_rows, rows)):
        if spec not in h_indexed.index:
            continue
        if idx % 2 == 0:
            ax.axhspan(y - row_half_height, y + row_half_height, xmin=0.0, xmax=1.0, color='#F5F8FC', lw=0, zorder=0)
        ax.text(0.025, y, row_label, ha='left', va='center', fontsize=FONT_NOTE)
        row = h_indexed.loc[spec]
        for x, (col, _, col_key) in zip(col_x, cols):
            val = float(row[col])
            if col_key == 'ns':
                txt = f'{val:+.3f}'
            elif col_key in {'top10', 'bottom10'}:
                txt = f'{val:.2f}'
            else:
                txt = f'{val:.2f}'
            ax.text(x, y, txt, ha='center', va='center', fontsize=FONT_NOTE)
    ax.text(0.0, 0.025, 'Top/bottom indicate decile retention; N-S Δ is North minus South mean ELI.', ha='left', va='bottom', fontsize=FONT_SMALL)

def draw_random_weight_density(ax, h):
    rand = h[h['scheme_type'].eq('random_weight')].copy()
    series = [('Spearman ρ', rand['spearman_rho_with_eli50'].values, COLORS['blue'], 1.0), ('Top-decile retention', rand['top_decile_retention'].values, COLORS['orange'], 0.0), ('Bottom-decile retention', rand['bottom_decile_retention'].values, COLORS['grey'], -1.0)]
    ax.axvspan(0.9, 1.0, color='#EEF5FD', lw=0, zorder=0)
    text_transform = blended_transform_factory(ax.transAxes, ax.transData)
    for label, vals, color, y0 in series:
        xs, dens = gaussian_density(vals, 0.5, 1.0)
        dens = dens / dens.max() * 0.32 if dens.max() > 0 else dens
        ax.fill_between(xs, y0, y0 + dens, color=color, alpha=0.22, lw=0)
        ax.plot(xs, y0 + dens, color=color, lw=1.2)
        q025, q50, q975 = np.nanquantile(vals, [0.025, 0.5, 0.975])
        ax.plot([q025, q975], [y0 - 0.06, y0 - 0.06], color=color, lw=1.2, solid_capstyle='round')
        ax.scatter([q50], [y0 - 0.06], s=19, color=color, edgecolor='white', linewidth=0.45, zorder=5)
        ax.text(0.02, y0 + 0.18, label, transform=text_transform, ha='left', va='center', fontsize=FONT_BODY)
        ax.text(0.98, y0 + 0.46, f'{q50:.2f} [{q025:.2f}, {q975:.2f}]', transform=text_transform, ha='right', va='center', fontsize=FONT_SMALL, clip_on=False)
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(-1.25, 1.42)
    ax.set_xlabel('Robustness under 500 random weight schemes')
    ax.set_yticks([])
    ax.set_xticks(np.arange(0.5, 1.01, 0.1))
    ax.text(0.01, -1.18, 'point, median; line, 95% interval', transform=text_transform, ha='left', va='center', fontsize=FONT_SMALL, color='black')
    strip_axis(ax, keep=('bottom',))

def build_figure():
    setup_style()
    f = pd.read_csv(resource(F_TABLE))
    qol_units = pd.read_csv(resource(EC_QOL_UNITS))
    qol_corr = pd.read_csv(resource(EC_QOL_CORR))
    h = pd.read_csv(resource(H_TABLE))
    fig = plt.figure(figsize=(7.15, 6.25), dpi=600)
    outer = fig.add_gridspec(2, 2, height_ratios=[1.02, 1.06], width_ratios=[1.03, 0.97], left=0.065, right=0.985, bottom=0.07, top=0.955, wspace=0.25, hspace=0.4)
    ax_f = fig.add_subplot(outer[0, 0])
    draw_panel_f(ax_f, f)
    gs_g = outer[0, 1].subgridspec(2, 3, wspace=0.22, hspace=0.27)
    axes_g = draw_panel_g(fig, gs_g, qol_units, qol_corr)
    enforce_panel_g_alignment(axes_g)
    gs_h = outer[1, :].subgridspec(1, 2, width_ratios=[1.18, 0.82], wspace=0.18)
    ax_h_matrix = fig.add_subplot(gs_h[0, 0])
    draw_robustness_matrix(ax_h_matrix, h)
    ax_h_density = fig.add_subplot(gs_h[0, 1])
    draw_random_weight_density(ax_h_density, h)
    align_bottom_panels_to_top_columns(ax_f, axes_g, ax_h_matrix, ax_h_density)
    expand_axis_downward(ax_h_matrix, extra=0.035)
    align_tight_bottom_to_fixed(ax_h_matrix, ax_h_density)
    extend_axis_left(ax_h_density, extra=0.045)
    add_panel_g_note(fig, axes_g)
    add_panel_label(fig, ax_f, 'a', '50 m-to-city stability')
    add_panel_label(fig, axes_g[0], 'b', 'External validation with European quality of life')
    add_panel_label(fig, ax_h_matrix, 'c', 'ELI construction and weighting robustness')
    return fig

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    for ext in ['png', 'tiff', 'pdf', 'svg']:
        out = OUT_DIR / f'{FIG_BASENAME}.{ext}'
        kwargs = {'bbox_inches': 'tight', 'pad_inches': 0.02}
        if ext in {'png', 'tiff'}:
            kwargs['dpi'] = 600
        fig.savefig(out, **kwargs)
    plt.close(fig)
