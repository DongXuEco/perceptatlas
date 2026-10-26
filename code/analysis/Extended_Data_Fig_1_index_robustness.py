from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "_support"))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "figures"))

from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
ROOT = PACKAGE_ROOT
OUT_DIR = output_dir('Extended_Data_Fig_1')
OUT_DIR.mkdir(parents=True, exist_ok=True)
CITY_50M = ROOT / 'derived_data' / 'city_emotion_inequality_50m.csv'
Q3_CITY = ROOT / 'derived_data' / 'q3_city_driver_master.csv'
Q3_ZONE = ROOT / 'derived_data' / 'q3_city_zone_driver_master.csv'
WALKING = ROOT / '影响变量数据' / '步行环境满意度.csv'
ELI_AUDIT = ROOT / 'figure_outputs_q3' / 'q2_q3_eli_formula_audit.json'
POSITIVE = ['beauty', 'wealthy', 'safer', 'livelier']
NEGATIVE = ['boring', 'depression']
MEAN50 = {'beauty': 'mean50_beauty', 'wealthy': 'mean50_wealthy', 'safer': 'mean50_safer', 'livelier': 'mean50_livelier', 'boring': 'mean50_boring', 'depression': 'mean50_depression'}

def pearson_safe(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    mask = x.notna() & y.notna()
    if mask.sum() < 3 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return (np.nan, np.nan)
    r, p = stats.pearsonr(x[mask].to_numpy(float), y[mask].to_numpy(float))
    return (float(r), float(p))

def spearman_safe(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    mask = x.notna() & y.notna()
    if mask.sum() < 3 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return (np.nan, np.nan)
    r, p = stats.spearmanr(x[mask].to_numpy(float), y[mask].to_numpy(float))
    return (float(r), float(p))

def weighted_pearson(x: pd.Series, y: pd.Series, w: pd.Series) -> float:
    mask = x.notna() & y.notna() & w.notna() & (w > 0)
    if mask.sum() < 3:
        return np.nan
    xv = x[mask].to_numpy(float)
    yv = y[mask].to_numpy(float)
    wv = w[mask].to_numpy(float)
    wx = np.average(xv, weights=wv)
    wy = np.average(yv, weights=wv)
    cov = np.average((xv - wx) * (yv - wy), weights=wv)
    vx = np.average((xv - wx) ** 2, weights=wv)
    vy = np.average((yv - wy) ** 2, weights=wv)
    if vx <= 0 or vy <= 0:
        return np.nan
    return float(cov / np.sqrt(vx * vy))

def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r = 6371.0088
    p1 = np.deg2rad(lat1)
    p2 = np.deg2rad(lat2)
    dp = np.deg2rad(lat2 - lat1)
    dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))

def rank_desc(values: np.ndarray) -> np.ndarray:
    return stats.rankdata(-values, method='average')

def retention(base: np.ndarray, alt: np.ndarray, frac: float, top: bool) -> float:
    n = len(base)
    k = max(1, int(np.ceil(n * frac)))
    if top:
        b = set(np.argsort(-base)[:k])
        a = set(np.argsort(-alt)[:k])
    else:
        b = set(np.argsort(base)[:k])
        a = set(np.argsort(alt)[:k])
    return len(b & a) / k

def build_q1_f(city_long: pd.DataFrame, city: pd.DataFrame) -> pd.DataFrame:
    mean_wide = city_long.pivot(index='id_1', columns='emotion', values='mean')
    p9010_wide = city_long.pivot(index='id_1', columns='emotion', values='p90_p10')
    n_wide = city_long.pivot(index='id_1', columns='emotion', values='n_pixels')
    source = pd.DataFrame(index=mean_wide.index)
    source['source_eli50'] = mean_wide[POSITIVE].mean(axis=1) - mean_wide[NEGATIVE].mean(axis=1)
    robust_sd = p9010_wide[POSITIVE + NEGATIVE] / 2.5631031310892007
    se = robust_sd / np.sqrt(n_wide[POSITIVE + NEGATIVE].clip(lower=1))
    source['eli50_bootstrap_se'] = np.sqrt(sum(((se[c] / 4.0) ** 2 for c in POSITIVE)) + sum(((se[c] / 2.0) ** 2 for c in NEGATIVE)))
    source['n_pixels_min'] = n_wide[POSITIVE + NEGATIVE].min(axis=1)
    source = source.reset_index()
    cols = ['id_1', 'COUNTRY', 'NAME_1', 'lon', 'lat', 'ns_group', 'eli50', 'emotion_n_pixels_min']
    merged = city[cols].merge(source, on='id_1', how='inner')
    merged = merged.dropna(subset=['eli50', 'source_eli50', 'eli50_bootstrap_se'])
    merged['source_master_diff'] = merged['source_eli50'] - merged['eli50']
    merged['source_master_abs_diff'] = merged['source_master_diff'].abs()
    n_boot = 1000
    rng = np.random.default_rng(20260824)
    score = merged['source_eli50'].to_numpy(float)
    sigma = merged['eli50_bootstrap_se'].to_numpy(float)
    sims = score[:, None] + rng.normal(0, sigma[:, None], size=(len(score), n_boot))
    ranks = np.empty(sims.shape, dtype=np.int32)
    base_rank = rank_desc(score)
    for j in range(n_boot):
        order = np.argsort(-sims[:, j])
        ranks[order, j] = np.arange(1, len(score) + 1)
    top_k = max(1, int(np.ceil(len(score) * 0.1)))
    bot_k = top_k
    merged['baseline_rank'] = base_rank
    merged['bootstrap_rank_median'] = np.median(ranks, axis=1)
    merged['bootstrap_rank_p025'] = np.quantile(ranks, 0.025, axis=1)
    merged['bootstrap_rank_p975'] = np.quantile(ranks, 0.975, axis=1)
    merged['bootstrap_rank_width95'] = merged['bootstrap_rank_p975'] - merged['bootstrap_rank_p025']
    merged['top_decile_probability'] = (ranks <= top_k).mean(axis=1)
    merged['bottom_decile_probability'] = (ranks >= len(score) - bot_k + 1).mean(axis=1)
    merged['n_bootstrap'] = n_boot
    merged['bootstrap_method'] = 'normal approximation from emotion p90-p10 and pixel counts'
    pr, pp = pearson_safe(merged['source_eli50'], merged['eli50'])
    sr, sp = spearman_safe(merged['source_eli50'], merged['eli50'])
    merged['source_master_pearson_r_global'] = pr
    merged['source_master_pearson_p_global'] = pp
    merged['source_master_spearman_rho_global'] = sr
    merged['source_master_spearman_p_global'] = sp
    merged['source_master_mae_global'] = merged['source_master_abs_diff'].mean()
    merged['source_master_max_abs_global'] = merged['source_master_abs_diff'].max()
    return merged.sort_values('baseline_rank')

def score_from_weights(df: pd.DataFrame, pos_w: np.ndarray, neg_w: np.ndarray) -> np.ndarray:
    pos = df[[MEAN50[c] for c in POSITIVE]].to_numpy(float)
    neg = df[[MEAN50[c] for c in NEGATIVE]].to_numpy(float)
    return pos @ pos_w - neg @ neg_w

def build_q1_h(city: pd.DataFrame) -> pd.DataFrame:
    needed = ['eli50', 'ns_group'] + [MEAN50[c] for c in POSITIVE + NEGATIVE]
    df = city.dropna(subset=needed).copy()
    base = df['eli50'].to_numpy(float)
    base_delta = df.loc[df['ns_group'].eq('North'), 'eli50'].mean() - df.loc[df['ns_group'].eq('South'), 'eli50'].mean()
    rows = []

    def add_row(name: str, scheme_type: str, score: np.ndarray, pos_w, neg_w, removed: str=''):
        pr, pp = stats.pearsonr(base, score)
        sr, sp = stats.spearmanr(base, score)
        ranks_base = rank_desc(base)
        ranks_alt = rank_desc(score)
        tmp = pd.DataFrame({'score': score, 'ns_group': df['ns_group'].to_numpy()})
        north = tmp.loc[tmp['ns_group'].eq('North'), 'score']
        south = tmp.loc[tmp['ns_group'].eq('South'), 'score']
        rows.append({'specification': name, 'scheme_type': scheme_type, 'removed_dimension': removed, 'n_cities': len(df), 'weight_beauty': pos_w[0], 'weight_wealthy': pos_w[1], 'weight_safer': pos_w[2], 'weight_livelier': pos_w[3], 'weight_boring': -neg_w[0], 'weight_depression': -neg_w[1], 'pearson_r_with_eli50': float(pr), 'pearson_p_with_eli50': float(pp), 'spearman_rho_with_eli50': float(sr), 'spearman_p_with_eli50': float(sp), 'rank_rmse': float(np.sqrt(np.mean((ranks_base - ranks_alt) ** 2))), 'top_decile_retention': retention(base, score, 0.1, True), 'bottom_decile_retention': retention(base, score, 0.1, False), 'north_mean': float(north.mean()), 'south_mean': float(south.mean()), 'north_minus_south_mean': float(north.mean() - south.mean()), 'north_median': float(north.median()), 'south_median': float(south.median()), 'baseline_north_minus_south_mean': float(base_delta)})
    eq_pos = np.repeat(1 / 4, 4)
    eq_neg = np.repeat(1 / 2, 2)
    add_row('baseline_eli50', 'baseline', base, eq_pos, eq_neg)
    add_row('formula_recomputed_from_50m_components', 'formula_check', score_from_weights(df, eq_pos, eq_neg), eq_pos, eq_neg)
    oriented = df[[MEAN50[c] for c in POSITIVE]].to_numpy(float)
    inverted = 1 - df[[MEAN50[c] for c in NEGATIVE]].to_numpy(float)
    equal_six = np.concatenate([oriented, inverted], axis=1).mean(axis=1)
    add_row('direction_adjusted_equal_six_dimensions', 'alternative_formula', equal_six, np.repeat(1 / 6, 4), -np.repeat(1 / 6, 2))
    add_row('positive_dimensions_only', 'alternative_formula', oriented.mean(axis=1), eq_pos, np.zeros(2))
    add_row('negative_dimensions_inverted_only', 'alternative_formula', inverted.mean(axis=1), np.zeros(4), -np.repeat(1 / 2, 2))
    for dim in POSITIVE:
        pos_dims = [d for d in POSITIVE if d != dim]
        pw = np.array([1 / len(pos_dims) if d in pos_dims else 0.0 for d in POSITIVE])
        add_row(f'leave_one_out_{dim}', 'leave_one_out', score_from_weights(df, pw, eq_neg), pw, eq_neg, dim)
    for dim in NEGATIVE:
        neg_dims = [d for d in NEGATIVE if d != dim]
        nw = np.array([1 / len(neg_dims) if d in neg_dims else 0.0 for d in NEGATIVE])
        add_row(f'leave_one_out_{dim}', 'leave_one_out', score_from_weights(df, eq_pos, nw), eq_pos, nw, dim)
    rng = np.random.default_rng(20260824)
    for i in range(500):
        pw = rng.dirichlet(np.ones(4))
        nw = rng.dirichlet(np.ones(2))
        add_row(f'random_weight_{i + 1:03d}', 'random_weight', score_from_weights(df, pw, nw), pw, nw)
    return pd.DataFrame(rows)
