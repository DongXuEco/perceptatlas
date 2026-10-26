from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
from matplotlib.transforms import ScaledTranslation, Affine2D
from scipy.stats import gaussian_kde
import Fig_4_base as base
COLORS = {'South': '#F27C2B', 'North': '#0D47A1'}
CAT = ['expansion advantage', 'balanced', 'core advantage']
CAT_COLORS = ['#F27C2B', '#E9EBF1', '#3667B2']
EMOTIONS = list(base.EMOTIONS)
NAMES = ['Beauty', 'Wealth', 'Safety', 'Liveliness', 'Boredom', 'Depression']
INK = '#263442'
LIGHT = '#E2EAF0'

def prepare(df):
    d = df.loc[np.isfinite(df.eli_diff)].copy()
    d['category'] = np.select([d.eli_diff > 0.03, d.eli_diff < -0.03], ['core advantage', 'expansion advantage'], default='balanced')
    shares = pd.crosstab(d.ns_group, d.category, normalize='index').reindex(index=['South', 'North'], columns=CAT)
    boot = {}
    bounds = {}
    scores = {}
    for i, g in enumerate(['North', 'South']):
        vals = d.loc[d.ns_group == g, 'category'].to_numpy()
        rng = np.random.default_rng(170 + i)
        draws = []
        for _ in range(600):
            p = pd.Series(rng.choice(vals, len(vals), replace=True)).value_counts(normalize=True)
            draws.append([p.get(c, 0) for c in CAT])
        boot[g] = np.array(draws)
        cum = np.cumsum(boot[g][:, :2], axis=1)
        bounds[g] = np.percentile(cum, [2.5, 50, 97.5], axis=0)
    for gi, g in enumerate(['South', 'North']):
        a = d.loc[d.ns_group == g]
        scores[g] = []
        for j, e in enumerate(EMOTIONS):
            v = a['benefit_' + e].dropna().to_numpy()
            post = base.bootstrap_means(v, seed=70 + j + 10 * gi)
            ci = np.percentile(post, [2.5, 97.5])
            scores[g].append({'emotion': e, 'core': float(a['core_' + e].mean()), 'expansion': float(a['expansion_' + e].mean()), 'benefit': float(v.mean()), 'ci95': ci.tolist()})
    return {'shares': shares, 'bootstrap': boot, 'boundaries': bounds, 'scores': scores, 'paired_cities': len(d), 'cities': d}

def draw_f(parent, data):
    parent.set_axis_off()
    shares = data['shares']
    area_audit = {}
    detail = {}
    main_scale = 1.08
    main_down = 2 / (parent.get_position().height * parent.figure.get_figheight() * 25.4)
    main_right = 3.5 / (parent.get_position().width * parent.figure.get_figwidth() * 25.4)
    main_transform = Affine2D().scale(main_scale).translate((1 - main_scale) * 0.5 + main_right, (1 - main_scale) * 0.64 - main_down) + parent.transAxes
    outward_shift = (2 + 1.5 / main_scale) / (parent.get_position().width * parent.figure.get_figwidth() * 25.4)
    for g, left in [('South', 0.01 - outward_shift), ('North', 0.56 + outward_shift)]:
        ax = parent.inset_axes([0.5 + (left - 0.5) * main_scale + main_right, 0.64 + (0.35 - 0.64) * main_scale - main_down, 0.43 * main_scale, 0.58 * main_scale])
        ax.set(xlim=(0, 10), ylim=(0, 10))
        ax.set_aspect('equal')
        ax.set_axis_off()
        edges = np.r_[0, np.cumsum(shares.loc[g].to_numpy() * 100)]
        area = np.zeros(3)
        for i in range(100):
            x = i % 10
            y = 9 - i // 10
            for k, color in enumerate(CAT_COLORS):
                lo = max(float(i), edges[k])
                hi = min(float(i + 1), edges[k + 1])
                if hi > lo:
                    ax.add_patch(Rectangle((x + 0.04 + (lo - i) * 0.92, y + 0.04), (hi - lo) * 0.92, 0.92, fc=color, ec='none'))
                    area[k] += hi - lo
        np.testing.assert_allclose(area, shares.loc[g].to_numpy() * 100, rtol=0, atol=1e-12)
        area_audit[g] = area.tolist()
        center = left + 0.215
        parent.text(center, 0.97, g, ha='center', fontsize=8.2, color=COLORS[g], transform=main_transform)
        assert 25 < edges[2] < 26
        ax.add_patch(Rectangle((5, 7), 1, 1, fc='none', ec=INK, lw=0.65, zorder=4))
        bottom = 0.58 if g == 'South' else 0.37
        zoom = parent.inset_axes([0.5 + (0.46 - 0.5) * main_scale + main_right, 0.64 + (bottom - 0.64) * main_scale - main_down, 0.08 * main_scale, 0.14 * main_scale])
        zoom.set(xlim=(0, 1), ylim=(0, 1))
        zoom.set_aspect('equal')
        zoom.set_axis_off()
        fractions = []
        for k, color in enumerate(CAT_COLORS):
            lo = max(25.0, edges[k])
            hi = min(26.0, edges[k + 1])
            fraction = max(0.0, hi - lo)
            fractions.append(fraction)
            if fraction > 0:
                zoom.add_patch(Rectangle((0.04 + (lo - 25) * 0.92, 0.04), fraction * 0.92, 0.92, fc=color, ec='none'))
        zoom.add_patch(Rectangle((0, 0), 1, 1, fc='none', ec=INK, lw=0.65))
        parent.text(0.5, bottom + 0.15, g, ha='center', va='bottom', fontsize=6.4, color=COLORS[g], transform=main_transform)
        start, end = (6, 0) if g == 'South' else (5, 1)
        bridge = 0.449 if g == 'South' else 0.551
        first_bend_shift_mm = -3 if g == 'South' else 3
        first_bend = bridge + first_bend_shift_mm / (parent.get_position().width * parent.figure.get_figwidth() * 25.4 * main_scale)
        ax.apply_aspect()
        zoom.apply_aspect()
        xy = main_transform.inverted().transform(ax.transData.transform((start, 7.5)))
        target = main_transform.inverted().transform(zoom.transData.transform((end, 0.5)))
        parent.plot([xy[0], first_bend, bridge, target[0]], [xy[1], xy[1], target[1], target[1]], color='black', lw=0.45, transform=main_transform, zorder=6, gid='f_midpoint_connector_' + g)
        np.testing.assert_allclose(sum(fractions), 1.0, rtol=0, atol=1e-12)
        detail[g] = {'cell_number': 26, 'cumulative_percent_range': [25, 26], 'category_fraction_in_cell': fractions, 'balanced_core_boundary_percent': float(edges[2]), 'connector_cell_endpoint': [start, 7.5], 'connector_zoom_endpoint': [end, 0.5], 'connector_color': 'black', 'first_bend_horizontal_shift_mm': first_bend_shift_mm}
    parent.text(0.5, 0.925, '1 cell = 1%', ha='center', va='top', fontsize=6.3, color=INK, transform=main_transform)
    parent.text(0.5, 0.81, 'Cell 26\nenlarged', ha='center', va='center', fontsize=6.3, color=INK, linespacing=1.15, transform=main_transform)
    footer_transform = parent.transAxes + ScaledTranslation(1.2 / 25.4, -3 / 25.4, parent.figure.dpi_scale_trans)
    for label, x, color in [('South', 0.4192, COLORS['South']), ('North', 0.5632, COLORS['North']), ('Δ N−S (pp)', 0.7252, INK)]:
        parent.text(x, 0.29, label, ha='center', va='center', fontsize=6.8, color=color, transform=footer_transform)
    parent.text(0.9412, 0.29, 'Core–expansion', ha='center', va='center', fontsize=6.8, color=INK, transform=footer_transform, gid='f_definition_header')
    definitions = ['< −0.03', '−0.03 to +0.03', '> +0.03']
    delta = (shares.loc['North'] - shares.loc['South']) * 100
    for j, (label, y) in enumerate(zip(['Expansion advantage', 'Balanced', 'Core advantage'], [0.215, 0.13, 0.045])):
        parent.plot(0.025, y, marker='s', ms=4, color=CAT_COLORS[j], transform=footer_transform, clip_on=False)
        parent.text(0.052, y, label, ha='left', va='center', fontsize=7.1, color=INK, transform=footer_transform)
        for g, x in [('South', 0.4192), ('North', 0.5632)]:
            parent.text(x, y, f'{shares.loc[g].iloc[j] * 100:.1f}%', ha='center', va='center', fontsize=8.2, color=INK, transform=footer_transform)
        parent.text(0.7252, y, f'{delta.iloc[j]:+.2f}', ha='center', va='center', fontsize=8.2, color=INK, transform=footer_transform)
        parent.text(0.9412, y, definitions[j], ha='center', va='center', fontsize=7.1, color=INK, transform=footer_transform, gid='f_definition_' + str(j))
        if j < 2:
            parent.plot([0.015, 1.0582], [y - 0.043, y - 0.043], color=LIGHT, lw=0.45, transform=footer_transform, clip_on=False)
    data['tile_audit'] = area_audit
    data['f_definition_column'] = {'header': 'Core–expansion', 'definition': 'Core ELI minus expansion ELI', 'rows': dict(zip(CAT, definitions)), 'balanced_interval_inclusive': True, 'x_parent_fraction': 0.9412, 'numeric_column_x': {'South': 0.4192, 'North': 0.5632, 'North_minus_South': 0.7252}, 'column_anchor_spacing_scale': 0.9, 'fixed_label_anchor_x': 0.052}
    data['f_boundary_detail'] = {'groups': detail, 'north_minus_south_percentage_points': delta.to_dict(), 'zoom_definition': 'The 26th percentage cell (25-26% cumulative grid position), shown with the same exact partial category fractions; not a city or a confidence interval.', 'grid_width_height_retained': False, 'main_width_height_scale': main_scale, 'main_down_shift_mm': 2, 'main_right_shift_mm': 3.5, 'matrix_outward_shift_mm_before_scale': 2 + 1.5 / main_scale, 'matrix_gap_increase_mm_before_scale': 4 + 3 / main_scale, 'latest_matrix_gap_increase_mm': 3, 'footer_right_shift_mm': 1.2, 'footer_down_shift_mm': 3}

def draw_g(parent, data):
    parent.set_axis_off()
    d = data['cities']
    records = []
    ygrid = np.linspace(0, 1, 280)
    main_scale = 1.08
    center_x, center_y = (0.5105, 0.495)
    main_transform = Affine2D().scale(main_scale).translate((1 - main_scale) * center_x, (1 - main_scale) * center_y) + parent.transAxes
    name_down_mm = 2.0
    name_down_fraction = name_down_mm / (parent.get_position().height * parent.figure.get_figheight() * 25.4)
    for j, e in enumerate(EMOTIONS):
        col = j % 3
        row = j // 3
        left = 0.065 + 0.313 * col
        bottom = 0.575 - 0.51 * row
        ax = parent.inset_axes([center_x + (left - center_x) * main_scale, center_y + (bottom - center_y) * main_scale, 0.265 * main_scale, 0.35 * main_scale])
        for gi, g in enumerate(COLORS):
            color = COLORS[g]
            group = d.loc[d.ns_group == g]
            for condition, x, marker in [('expansion', gi * 2.3, 's'), ('core', gi * 2.3 + 0.65, 'o')]:
                v = group[condition + '_' + e].dropna().to_numpy()
                assert len(v) == {'South': 1345, 'North': 3223}[g] and np.all((v >= 0) & (v <= 1))
                kde = gaussian_kde(v)
                density = kde(ygrid) + kde(-ygrid) + kde(2 - ygrid)
                density /= density.max()
                horizontal = np.linspace(-0.31, 0.31, 55)
                rgba = np.zeros((len(ygrid), len(horizontal), 4))
                rgba[:, :, :3] = to_rgb(color)
                rgba[:, :, 3] = 0.34 * density[:, None] ** 0.65 * np.exp(-0.5 * (horizontal[None, :] / 0.2) ** 2)
                ax.imshow(rgba, extent=(x - 0.31, x + 0.31, 0, 1), origin='lower', aspect='auto', interpolation='bilinear', zorder=0, rasterized=True)
                rng = np.random.default_rng(460 + 40 * j + 10 * gi + (condition == 'core'))
                jitter = rng.uniform(-0.2, 0.2, len(v))
                ax.scatter(x + jitter, v, s=0.18, color=color, alpha=0.075, linewidths=0, rasterized=True, zorder=1)
                q05, q25, q50, q75, q95 = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
                mean = float(v.mean())
                ax.plot([x, x], [q05, q95], color=color, lw=0.5, alpha=0.8, zorder=2)
                ax.add_patch(Rectangle((x - 0.18, q25), 0.36, q75 - q25, fc='white', alpha=0.58, ec=color, lw=0.65, zorder=3))
                ax.plot([x - 0.18, x + 0.18], [q50, q50], color=color, lw=0.65, zorder=4)
                ax.plot(x, mean, marker=marker, ms=3.1, mfc=color if condition == 'core' else 'white', mec=color, mew=0.75, zorder=5)
                records.append({'emotion': e, 'group': g, 'condition': condition, 'n': len(v), 'mean': mean, 'p05_p25_p50_p75_p95': [float(q) for q in [q05, q25, q50, q75, q95]], 'jitter_seed': 460 + 40 * j + 10 * gi + (condition == 'core')})
            summary = data['scores'][g][j]
            ax.plot([gi * 2.3, gi * 2.3 + 0.65], [summary['expansion'], summary['core']], color=color, lw=0.65, zorder=4)
        ax.set(xlim=(-0.48, 3.43), ylim=(0, 1), xticks=[0.325, 2.625], yticks=[0, 0.5, 1])
        ax.set_xticklabels(['South', 'North'], fontsize=5.9)
        for label, g in zip(ax.get_xticklabels(), COLORS):
            label.set_color(COLORS[g])
        ax.set_yticklabels(['0', '.5', '1'] if col == 0 else [], fontsize=5.7)
        ax.tick_params(axis='x', length=0, pad=1.7)
        ax.tick_params(axis='y', length=1.6, width=0.45, pad=1.2)
        ax.spines[['top', 'right']].set_visible(False)
        for spine in ax.spines.values():
            spine.set_color('#9DAAB4')
            spine.set_linewidth(0.45)
        parent.text(left + 0.1325, bottom + 0.385 - name_down_fraction, NAMES[j], ha='center', va='bottom', fontsize=7.4, color=INK, transform=main_transform, gid='g_emotion_name')
    parent.text(0.005, 0.5, 'Perception score', ha='center', va='center', rotation=90, fontsize=7.1, color=INK, transform=main_transform)
    parent.child_axes[2].legend(handles=[Line2D([], [], ls='none', marker='s', mfc='white', mec=INK, ms=3.3, label='Expansion'), Line2D([], [], ls='none', marker='o', color=INK, ms=3.3, label='Core')], loc='lower right', bbox_to_anchor=(0.98, 0.02), ncol=1, fontsize=6.5, handlelength=0.7, labelspacing=0.3, borderaxespad=0.2, frameon=False)
    data['profile_audit'] = records
    data['g_layout'] = {'profile_axes': len(parent.child_axes), 'gap_tracks': 0, 'previous_height_fraction': 0.35, 'height_fraction': 0.35 * main_scale, 'width_fraction': 0.265 * main_scale, 'main_width_height_scale': main_scale, 'height_increase_percent': 8, 'row_bottoms': [center_y + (y - center_y) * main_scale for y in [0.575, 0.065]], 'emotion_names_down_mm': name_down_mm * main_scale, 'condition_legend': 'inside Safety, lower right'}
