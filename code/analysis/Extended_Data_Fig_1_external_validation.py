from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "_support"))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "figures"))

from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
import json
import math
import re
import tempfile
import unicodedata
import zipfile
from pathlib import Path
from typing import Any
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats
ROOT = PACKAGE_ROOT
DATA_DIR = ROOT / '影响变量数据' / '外部验证' / 'European_Commission_QoL_European_Cities_2023'
QOL_XLSX = DATA_DIR / 'quality_of_life_european_cities_2023_aggregated_data_revised_2024.xlsx'
QOL_MICRO_ZIP = DATA_DIR / 'qol_european_cities_2023_microdata_codebook.zip'
CITY_MASTER = ROOT / 'derived_data' / 'q3_city_driver_master.csv'
OUT_DIR = output_dir('Extended_Data_Fig_1')
GISCO_YEARS = [2024, 2021, 2020, 2018]
GISCO_TEMPLATE = DATA_DIR / 'URAU_LB_{year}_4326_CITIES.geojson'
COUNTRY_TO_CNTR = {'Albania': 'AL', 'Austria': 'AT', 'Belgium': 'BE', 'Bulgaria': 'BG', 'Croatia': 'HR', 'Cyprus': 'CY', 'Czech Republic': 'CZ', 'Czechia': 'CZ', 'Denmark': 'DK', 'Estonia': 'EE', 'Finland': 'FI', 'France': 'FR', 'Germany': 'DE', 'Greece': 'EL', 'Hungary': 'HU', 'Iceland': 'IS', 'Ireland': 'IE', 'Italy': 'IT', 'Latvia': 'LV', 'Lithuania': 'LT', 'Luxembourg': 'LU', 'Macedonia': 'MK', 'Malta': 'MT', 'Montenegro': 'ME', 'Netherlands': 'NL', 'Norway': 'NO', 'Poland': 'PL', 'Portugal': 'PT', 'Romania': 'RO', 'Serbia': 'RS', 'Slovakia': 'SK', 'Slovenia': 'SI', 'Spain': 'ES', 'Sweden': 'SE', 'Switzerland': 'CH', 'Turkey': 'TR', 'United Kingdom': 'UK'}
PROJECT_COUNTRY_ALIAS = {'Czech Republic': 'Czechia'}
CITY_ALIASES = {'Athina': ['Athina', 'Athens', 'Attica'], 'Beograd': ['Beograd', 'Belgrade', 'Grad Beograd'], 'Bruxelles / Brussel': ['Bruxelles', 'Brussels', 'Bruxelles/Brussel'], 'Bucureşti': ['Bucureşti', 'Bucharest', 'Bucuresti'], 'Genève': ['Genève', 'Geneva'], 'Helsinki / Helsingfors': ['Helsinki', 'Helsingfors'], 'Irakleio': ['Irakleio', 'Heraklion', 'Crete'], 'København': ['København', 'Copenhagen', 'Hovedstaden'], 'Lefkosia': ['Lefkosia', 'Nicosia'], 'Lisboa': ['Lisboa', 'Lisbon'], 'Málaga': ['Málaga', 'Malaga'], 'München': ['München', 'Munich'], 'Napoli': ['Napoli', 'Naples'], 'Oulu / Uleåborg': ['Oulu', 'Uleåborg'], 'Praha': ['Praha', 'Prague'], 'Reykjavík': ['Reykjavík', 'Reykjavik', 'Höfuðborgarsvæði'], 'Roma': ['Roma', 'Rome'], 'Torino': ['Torino', 'Turin'], 'Tirana': ['Tirana', 'Tiranë'], 'Tyneside conurbation': ['Newcastle upon Tyne', 'Tyneside'], 'Valletta': ['Valletta', 'Greater Valletta'], 'Warszawa': ['Warszawa', 'Warsaw'], 'Wien': ['Wien', 'Vienna', 'Greater Wien'], 'Zagreb': ['Zagreb', 'Grad Zagreb'], 'Zürich': ['Zürich', 'Zurich']}
SELECTED_INDICATORS = [('TB_Q2_1', '1 Top2', 'satisfied_live_city_top2', 'Satisfied to live in city'), ('TB_Q4_1', '1 Top2', 'neighbourhood_satisfaction_top2', 'Neighbourhood satisfaction'), ('TB_Q4_4', '1 Top2', 'life_satisfaction_top2', 'Life satisfaction'), ('Q3_1', '1 A good place to live', 'good_place_people_general', 'Good place to live'), ('TB_Q1a_5', '1 Top2', 'green_space_satisfaction_top2', 'Green space satisfaction'), ('TB_Q1a_6', '1 Top2', 'public_space_satisfaction_top2', 'Public space satisfaction'), ('TB_Q1a_1', '1 Top2', 'public_transport_satisfaction_top2', 'Public transport satisfaction'), ('TB_Q1a_2', '1 Top2', 'healthcare_satisfaction_top2', 'Health care satisfaction'), ('TB_Q1b_1', '1 Top2', 'air_quality_satisfaction_top2', 'Air quality satisfaction'), ('TB_Q1b_2', '1 Top2', 'noise_level_satisfaction_top2', 'Noise level satisfaction'), ('TB_Q1b_3', '1 Top2', 'cleanliness_satisfaction_top2', 'Cleanliness satisfaction'), ('TB_Q2_3', '1 Top2', 'safe_walk_city_top2', 'Safe walking at night in city'), ('TB_Q2_4', '1 Top2', 'safe_walk_neighbourhood_top2', 'Safe walking at night in neighbourhood'), ('Q7', '1 Yes', 'confidence_police_yes', 'Confidence in local police'), ('Q14', '3 Increased', 'qol_increased_5y', 'Quality of life increased over five years'), ('TB_Q16', '1 Bot2', 'not_lonely_4w_bot2', 'Low loneliness over past four weeks'), ('TB_Q17', '1 Bot2', 'not_lonely_12m_bot2', 'Low loneliness over past twelve months')]
OUTCOME_COLS = ['eli50', 'positive_score_city_table', 'negative_score_city_table', 'mean50_beauty', 'mean50_wealthy', 'mean50_safer', 'mean50_livelier', 'mean50_boring', 'mean50_depression']

def normalize_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ''
    s = str(value).lower()
    replacements = {'ł': 'l', 'ø': 'o', 'đ': 'd', 'ı': 'i', 'ß': 'ss', 'æ': 'ae', 'œ': 'oe'}
    for old, new in replacements.items():
        s = s.replace(old, new)
    s = unicodedata.normalize('NFKD', s)
    s = ''.join((ch for ch in s if not unicodedata.combining(ch)))
    s = re.sub('[^a-z0-9]+', ' ', s)
    return re.sub('\\s+', ' ', s).strip()

def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6371.0088
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))

def canonical_project_country(country: str) -> str:
    return PROJECT_COUNTRY_ALIAS.get(str(country), str(country))

def city_aliases(city: str) -> list[str]:
    aliases = [city, *CITY_ALIASES.get(city, [])]
    if '/' in str(city):
        aliases.extend((part.strip() for part in str(city).split('/')))
    return list(dict.fromkeys((alias for alias in aliases if alias)))

def extract_microdata_city_country() -> pd.DataFrame:
    dta_path = Path(tempfile.gettempdir()) / 'qol2023_microdata.dta'
    if not resource(dta_path).exists():
        with zipfile.ZipFile(QOL_MICRO_ZIP) as zf:
            dta_name = next((name for name in zf.namelist() if name.lower().endswith('.dta')))
            dta_path.write_bytes(zf.read(dta_name))
    micro = pd.read_stata(dta_path, columns=['Country_sample', 'D3_Cityrecode'], convert_categoricals=True)
    out = micro.dropna(subset=['Country_sample', 'D3_Cityrecode']).drop_duplicates().groupby('D3_Cityrecode', as_index=False)['Country_sample'].agg(lambda x: sorted(set(map(str, x)))[0])
    return out

def parse_qol_city_columns(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in range(4, raw.shape[1]):
        label = str(raw.iat[1, col])
        match = re.match('^(\\d+)\\s+(.+)$', label)
        if not match:
            continue
        rows.append({'qol_col': col, 'qol_city_code': match.group(1), 'qol_city_name': match.group(2).strip(), 'qol_n': pd.to_numeric(raw.iat[3, col], errors='coerce')})
    cities = pd.DataFrame(rows)
    countries = extract_microdata_city_country()
    cities = cities.merge(countries, left_on='qol_city_name', right_on='D3_Cityrecode', how='left')
    cities = cities.drop(columns=['D3_Cityrecode'])
    cities['project_country'] = cities['Country_sample'].map(canonical_project_country)
    cities['country_code'] = cities['Country_sample'].map(COUNTRY_TO_CNTR)
    return cities

def row_question_code(text: Any) -> str:
    if not isinstance(text, str):
        return ''
    return text.split(' ', 1)[0].strip()

def parse_qol_indicators() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(resource(QOL_XLSX), sheet_name='QoL in European cities 2023_V2', header=None)
    cities = parse_qol_city_columns(raw)
    active_code = ''
    rows = []
    selected = {(q, a): (name, label) for q, a, name, label in SELECTED_INDICATORS}
    for i in range(3, raw.shape[0]):
        q0 = raw.iat[i, 0]
        if isinstance(q0, str) and q0.strip():
            active_code = row_question_code(q0)
        answer = str(raw.iat[i, 1]).strip()
        key = (active_code, answer)
        if key not in selected:
            continue
        indicator, indicator_label = selected[key]
        for _, city in cities.iterrows():
            value = pd.to_numeric(raw.iat[i, int(city['qol_col'])], errors='coerce')
            rows.append({**city.drop(labels=['qol_col']).to_dict(), 'question_code': active_code, 'answer_label': answer, 'indicator': indicator, 'indicator_label': indicator_label, 'qol_value': value})
    indicators = pd.DataFrame(rows)
    return (cities, indicators)

def load_gisco_city_points() -> pd.DataFrame:
    frames = []
    for year in GISCO_YEARS:
        path = GISCO_TEMPLATE.with_name(GISCO_TEMPLATE.name.format(year=year))
        if not resource(path).exists():
            continue
        gdf = gpd.read_file(resource(path))
        if 'CNTR_CODE' not in gdf.columns or 'URAU_NAME' not in gdf.columns:
            continue
        frame = pd.DataFrame({'gisco_year': year, 'country_code': gdf['CNTR_CODE'].astype(str), 'URAU_CODE': gdf['URAU_CODE'].astype(str), 'URAU_NAME': gdf['URAU_NAME'].astype(str), 'gisco_lon': gdf.geometry.x, 'gisco_lat': gdf.geometry.y})
        frame['norm_urau_name'] = frame['URAU_NAME'].map(normalize_text)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).drop_duplicates(['gisco_year', 'country_code', 'URAU_CODE', 'norm_urau_name'])

def attach_gisco_coordinates(cities: pd.DataFrame) -> pd.DataFrame:
    gisco = load_gisco_city_points()
    out_rows = []
    for _, city in cities.iterrows():
        country_code = city['country_code']
        aliases = [normalize_text(x) for x in city_aliases(str(city['qol_city_name']))]
        candidates = gisco[gisco['country_code'].eq(country_code)].copy()
        best = None
        for _, cand in candidates.iterrows():
            exact = cand['norm_urau_name'] in aliases
            contains = any((alias and alias in cand['norm_urau_name'] for alias in aliases))
            score = 1.0 if exact else max([_similarity(alias, cand['norm_urau_name']) for alias in aliases] or [0.0])
            if contains and score < 0.98:
                score = 0.98
            rank = (score, -GISCO_YEARS.index(int(cand['gisco_year'])) if int(cand['gisco_year']) in GISCO_YEARS else -99)
            if best is None or rank > best[0]:
                best = (rank, cand)
        row = city.to_dict()
        if best is None or best[0][0] < 0.78:
            row.update({'gisco_match_score': np.nan if best is None else best[0][0], 'gisco_year': np.nan, 'URAU_CODE': '', 'URAU_NAME': '', 'gisco_lon': np.nan, 'gisco_lat': np.nan, 'gisco_match_status': 'missing_or_low_score'})
        else:
            cand = best[1]
            row.update({'gisco_match_score': float(best[0][0]), 'gisco_year': int(cand['gisco_year']), 'URAU_CODE': cand['URAU_CODE'], 'URAU_NAME': cand['URAU_NAME'], 'gisco_lon': float(cand['gisco_lon']), 'gisco_lat': float(cand['gisco_lat']), 'gisco_match_status': 'gisco'})
        out_rows.append(row)
    return pd.DataFrame(out_rows)

def _similarity(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def load_city_master() -> pd.DataFrame:
    cols = ['id_1', 'COUNTRY', 'NAME_1', 'lon', 'lat', 'ns_group', *OUTCOME_COLS]
    df = pd.read_csv(resource(CITY_MASTER), usecols=lambda c: c in cols)
    df['norm_name'] = df['NAME_1'].map(normalize_text)
    return df.dropna(subset=['id_1', 'COUNTRY', 'lon', 'lat', 'eli50']).copy()

def match_to_project_city(cities_with_coords: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, city in cities_with_coords.iterrows():
        project_country = str(city['project_country'])
        pool = master[master['COUNTRY'].astype(str).eq(project_country)].copy()
        aliases = [normalize_text(x) for x in city_aliases(str(city['qol_city_name']))]
        row = city.to_dict()
        if pool.empty:
            row.update({'matched_id_1': np.nan, 'matched_country': '', 'matched_name': '', 'matched_lon': np.nan, 'matched_lat': np.nan, 'match_method': 'no_project_country', 'match_distance_km': np.nan, 'name_match_score': np.nan})
            rows.append(row)
            continue
        exact = pool[pool['norm_name'].isin(aliases)]
        if len(exact) == 1:
            cand = exact.iloc[0]
            dist = haversine_km(city['gisco_lon'], city['gisco_lat'], cand['lon'], cand['lat']) if pd.notna(city['gisco_lon']) else 0.0
            row.update(_matched_fields(cand, 'project_name_exact_or_alias', dist, 1.0))
            rows.append(row)
            continue
        if pd.notna(city['gisco_lon']) and pd.notna(city['gisco_lat']):
            pool['dist'] = [haversine_km(float(city['gisco_lon']), float(city['gisco_lat']), float(lon), float(lat)) for lon, lat in zip(pool['lon'], pool['lat'])]
            cand = pool.sort_values('dist').iloc[0]
            best_name = max((_similarity(alias, cand['norm_name']) for alias in aliases), default=np.nan)
            row.update(_matched_fields(cand, 'same_country_nearest_to_qol_city', float(cand['dist']), float(best_name)))
        else:
            scores = pool['norm_name'].map(lambda x: max((_similarity(alias, x) for alias in aliases), default=0.0))
            best_idx = scores.idxmax()
            cand = pool.loc[best_idx]
            if scores.loc[best_idx] >= 0.88:
                row.update(_matched_fields(cand, 'project_name_fuzzy', np.nan, float(scores.loc[best_idx])))
            else:
                row.update({'matched_id_1': np.nan, 'matched_country': '', 'matched_name': '', 'matched_lon': np.nan, 'matched_lat': np.nan, 'match_method': 'no_usable_city_coordinate_or_name_match', 'match_distance_km': np.nan, 'name_match_score': float(scores.loc[best_idx])})
        rows.append(row)
    matches = pd.DataFrame(rows)
    return matches.merge(master.drop(columns=['norm_name']), left_on='matched_id_1', right_on='id_1', how='left')

def _matched_fields(cand: pd.Series, method: str, distance_km: float, name_score: float) -> dict[str, Any]:
    return {'matched_id_1': int(cand['id_1']), 'matched_country': cand['COUNTRY'], 'matched_name': cand['NAME_1'], 'matched_lon': float(cand['lon']), 'matched_lat': float(cand['lat']), 'match_method': method, 'match_distance_km': distance_km, 'name_match_score': name_score}

def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return np.nan
    return float(np.average(values[valid].astype(float), weights=weights[valid].astype(float)))

def make_validation_units(indicators: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    match_cols = ['qol_city_code', 'qol_city_name', 'Country_sample', 'matched_id_1', 'matched_country', 'matched_name', 'match_method', 'match_distance_km', 'gisco_match_status', 'gisco_year', 'URAU_CODE', 'URAU_NAME', 'gisco_lon', 'gisco_lat', *OUTCOME_COLS]
    merged = indicators.merge(matches[match_cols], on=['qol_city_code', 'qol_city_name', 'Country_sample'], how='left')
    matched = merged.dropna(subset=['matched_id_1', 'qol_value', 'eli50']).copy()
    matched['matched_id_1'] = matched['matched_id_1'].astype(int)
    group_cols = ['indicator', 'indicator_label', 'matched_id_1']
    rows = []
    for keys, sub in matched.groupby(group_cols, dropna=False):
        first = sub.iloc[0]
        row = {'indicator': keys[0], 'indicator_label': keys[1], 'matched_id_1': keys[2], 'qol_value': weighted_mean(sub['qol_value'], sub['qol_n']), 'qol_city_count_in_unit': int(sub['qol_city_name'].nunique()), 'qol_n_sum': float(sub['qol_n'].sum(skipna=True)), 'qol_city_names': '; '.join(sorted(sub['qol_city_name'].dropna().astype(str).unique())), 'matched_country': first['matched_country'], 'matched_name': first['matched_name'], 'min_match_distance_km': float(sub['match_distance_km'].min(skipna=True)) if sub['match_distance_km'].notna().any() else np.nan, 'max_match_distance_km': float(sub['match_distance_km'].max(skipna=True)) if sub['match_distance_km'].notna().any() else np.nan, 'match_methods': '; '.join(sorted(sub['match_method'].dropna().astype(str).unique()))}
        for col in OUTCOME_COLS:
            row[col] = first[col]
        rows.append(row)
    return (matched, pd.DataFrame(rows))

def fisher_ci(r: float, n: int) -> tuple[float, float]:
    if n <= 3 or not np.isfinite(r) or abs(r) >= 1:
        return (np.nan, np.nan)
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1 / math.sqrt(n - 3)
    lo, hi = np.tanh([z - 1.96 * se, z + 1.96 * se])
    return (float(lo), float(hi))

def bh_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    pv = p[valid]
    if len(pv) == 0:
        return out.tolist()
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adj = ranked * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    restored = np.empty_like(adj)
    restored[order] = np.clip(adj, 0, 1)
    out[valid] = restored
    return out.tolist()

def compute_correlations(units: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes = {'all_matched_units': units, 'within_150km_or_exact': units[units['min_match_distance_km'].isna() | (units['min_match_distance_km'] <= 150) | units['match_methods'].astype(str).str.contains('project_name', na=False)], 'within_100km_or_exact': units[units['min_match_distance_km'].isna() | (units['min_match_distance_km'] <= 100) | units['match_methods'].astype(str).str.contains('project_name', na=False)], 'exact_project_name_only': units[units['match_methods'].astype(str).str.contains('project_name_exact_or_alias', na=False)]}
    for scope, frame in scopes.items():
        for indicator, sub in frame.groupby('indicator'):
            label = sub['indicator_label'].iloc[0]
            for outcome in OUTCOME_COLS:
                tmp = sub[['qol_value', outcome]].dropna()
                n = len(tmp)
                if n < 4 or tmp['qol_value'].nunique() < 2 or tmp[outcome].nunique() < 2:
                    pearson_r = pearson_p = spearman_r = spearman_p = np.nan
                    p_lo = p_hi = s_lo = s_hi = np.nan
                else:
                    pearson_r, pearson_p = stats.pearsonr(tmp['qol_value'], tmp[outcome])
                    spearman_r, spearman_p = stats.spearmanr(tmp['qol_value'], tmp[outcome])
                    p_lo, p_hi = fisher_ci(float(pearson_r), n)
                    s_lo, s_hi = fisher_ci(float(spearman_r), n)
                rows.append({'scope': scope, 'indicator': indicator, 'indicator_label': label, 'outcome': outcome, 'n_units': n, 'pearson_r': pearson_r, 'pearson_p': pearson_p, 'pearson_ci_low': p_lo, 'pearson_ci_high': p_hi, 'spearman_rho': spearman_r, 'spearman_p': spearman_p, 'spearman_ci_low': s_lo, 'spearman_ci_high': s_hi})
    out = pd.DataFrame(rows)
    for (scope, outcome), idx in out.groupby(['scope', 'outcome']).groups.items():
        out.loc[idx, 'pearson_p_fdr_bh'] = bh_adjust(out.loc[idx, 'pearson_p'].tolist())
        out.loc[idx, 'spearman_p_fdr_bh'] = bh_adjust(out.loc[idx, 'spearman_p'].tolist())
    return out.sort_values(['outcome', 'scope', 'spearman_p_fdr_bh', 'indicator'])

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cities, indicators = parse_qol_indicators()
    cities_with_coords = attach_gisco_coordinates(cities)
    master = load_city_master()
    matches = match_to_project_city(cities_with_coords, master)
    matched_rows, units = make_validation_units(indicators, matches)
    corrs = compute_correlations(units)
    cities.to_csv(OUT_DIR / 'ec_qol_city_list.csv', index=False, encoding='utf-8-sig')
    indicators.to_csv(OUT_DIR / 'ec_qol_city_indicators_long.csv', index=False, encoding='utf-8-sig')
    matches.to_csv(OUT_DIR / 'ec_qol_eli_city_matches.csv', index=False, encoding='utf-8-sig')
    matched_rows.to_csv(OUT_DIR / 'ec_qol_eli_validation_city_rows.csv', index=False, encoding='utf-8-sig')
    units.to_csv(OUT_DIR / 'ec_qol_eli_validation_units.csv', index=False, encoding='utf-8-sig')
    corrs.to_csv(OUT_DIR / 'ec_qol_eli_correlation_summary.csv', index=False, encoding='utf-8-sig')
    eli_topline = corrs[(corrs['outcome'] == 'eli50') & corrs['scope'].isin(['all_matched_units', 'within_150km_or_exact'])].copy()
    eli_topline = eli_topline.sort_values(['scope', 'spearman_p_fdr_bh', 'indicator'])
    eli_topline.to_csv(OUT_DIR / 'ec_qol_eli_topline_summary.csv', index=False, encoding='utf-8-sig')
    audit = {'inputs': {'qol_xlsx': str(QOL_XLSX), 'qol_microdata_zip': str(QOL_MICRO_ZIP), 'city_master': str(CITY_MASTER), 'gisco_files': [str(GISCO_TEMPLATE.with_name(GISCO_TEMPLATE.name.format(year=y))) for y in GISCO_YEARS]}, 'selected_indicators': [{'question_code': q, 'answer': a, 'indicator': name, 'label': label} for q, a, name, label in SELECTED_INDICATORS], 'counts': {'qol_cities': int(cities['qol_city_name'].nunique()), 'qol_city_indicators_rows': int(len(indicators)), 'qol_cities_with_gisco_or_name_coordinates': int(cities_with_coords['gisco_lon'].notna().sum()), 'qol_cities_matched_to_project': int(matches['matched_id_1'].notna().sum()), 'unique_project_units_matched': int(matches['matched_id_1'].dropna().nunique()), 'validation_city_rows': int(len(matched_rows)), 'validation_unit_rows': int(len(units))}, 'match_method_counts': matches['match_method'].value_counts(dropna=False).to_dict(), 'distance_km_summary': matches['match_distance_km'].describe().to_dict(), 'unmatched_qol_cities': matches.loc[matches['matched_id_1'].isna(), ['qol_city_name', 'Country_sample', 'match_method', 'gisco_match_status']].to_dict(orient='records')}
    (OUT_DIR / 'ec_qol_eli_validation_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(audit['counts'], ensure_ascii=False, indent=2))
    print('\nELI topline, within_150km_or_exact:')
    cols = ['indicator', 'n_units', 'pearson_r', 'pearson_p_fdr_bh', 'spearman_rho', 'spearman_p_fdr_bh']
    print(eli_topline[eli_topline['scope'].eq('within_150km_or_exact')][cols].sort_values('spearman_p_fdr_bh').to_string(index=False))
