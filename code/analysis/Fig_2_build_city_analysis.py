from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "_support"))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "figures"))

from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
import argparse
import json
import struct
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
PROJECT_ROOT = PACKAGE_ROOT
DERIVED_DIR = PROJECT_ROOT / 'derived_data'
DEFAULT_OUTPUT = DERIVED_DIR / 'q3_city_driver_master.csv'
DEFAULT_INVENTORY = DERIVED_DIR / 'q3_driver_variable_inventory.csv'
DEFAULT_METADATA = DERIVED_DIR / 'q3_city_driver_master_metadata.json'
EMOTIONS = ['beauty', 'wealthy', 'safer', 'livelier', 'boring', 'depression']
POSITIVE = ['beauty', 'wealthy', 'safer', 'livelier']
NEGATIVE = ['boring', 'depression']

@dataclass(frozen=True)
class RasterSpec:
    prefix: str
    family: str
    variable: str
    rel_path: str
    band: int = 1
    unit: str = ''
    expected_direction: str = ''

    @property
    def path(self) -> Path:
        return PROJECT_ROOT / self.rel_path
DRIVER_SPECS: list[RasterSpec] = [RasterSpec('poi_total', 'public_service_poi', 'Total POI density', '影响变量数据\\公共服务\\poi数据\\全球POI密度栅格\\processed\\global_total_poi_density_global_urbanmask.tif', unit='density', expected_direction='higher means denser POI supply'), RasterSpec('poi_education', 'public_service_poi', 'Education POI density', '影响变量数据\\公共服务\\poi数据\\全球POI密度栅格\\processed\\educationImg_global_urbanmask.tif', unit='density', expected_direction='higher means denser POI supply'), RasterSpec('poi_entertainment', 'public_service_poi', 'Entertainment POI density', '影响变量数据\\公共服务\\poi数据\\全球POI密度栅格\\processed\\entertainmentImg_global_urbanmask.tif', unit='density', expected_direction='higher means denser POI supply'), RasterSpec('poi_health', 'public_service_poi', 'Health POI density', '影响变量数据\\公共服务\\poi数据\\全球POI密度栅格\\processed\\healthImg_global_urbanmask.tif', unit='density', expected_direction='higher means denser POI supply'), RasterSpec('poi_life', 'public_service_poi', 'Life-service POI density', '影响变量数据\\公共服务\\poi数据\\全球POI密度栅格\\processed\\lifeImg_global_urbanmask.tif', unit='density', expected_direction='higher means denser POI supply'), RasterSpec('poi_production', 'public_service_poi', 'Production-service POI density', '影响变量数据\\公共服务\\poi数据\\全球POI密度栅格\\processed\\productionImg_global_urbanmask.tif', unit='density', expected_direction='higher means denser POI supply'), RasterSpec('poi_transport', 'public_service_poi', 'Transport POI density', '影响变量数据\\公共服务\\poi数据\\全球POI密度栅格\\processed\\transportImg_global_urbanmask.tif', unit='density', expected_direction='higher means denser POI supply'), RasterSpec('access_motor_education', 'public_service_accessibility_motor', 'Motorized education accessibility', '影响变量数据\\公共服务\\全球可达性数据\\机动车\\processed\\education_global_urbanmask.tif', unit='travel cost/time', expected_direction='lower usually means better accessibility'), RasterSpec('access_motor_entertainment', 'public_service_accessibility_motor', 'Motorized entertainment accessibility', '影响变量数据\\公共服务\\全球可达性数据\\机动车\\processed\\entertainment_global_urbanmask.tif', unit='travel cost/time', expected_direction='lower usually means better accessibility'), RasterSpec('access_motor_health', 'public_service_accessibility_motor', 'Motorized health accessibility', '影响变量数据\\公共服务\\全球可达性数据\\机动车\\processed\\health_global_urbanmask.tif', unit='travel cost/time', expected_direction='lower usually means better accessibility'), RasterSpec('access_motor_living', 'public_service_accessibility_motor', 'Motorized living-service accessibility', '影响变量数据\\公共服务\\全球可达性数据\\机动车\\processed\\living_global_urbanmask.tif', unit='travel cost/time', expected_direction='lower usually means better accessibility'), RasterSpec('access_motor_production', 'public_service_accessibility_motor', 'Motorized production-service accessibility', '影响变量数据\\公共服务\\全球可达性数据\\机动车\\processed\\production_global_urbanmask.tif', unit='travel cost/time', expected_direction='lower usually means better accessibility'), RasterSpec('access_motor_transport', 'public_service_accessibility_motor', 'Motorized transport accessibility', '影响变量数据\\公共服务\\全球可达性数据\\机动车\\processed\\transport_global_urbanmask.tif', unit='travel cost/time', expected_direction='lower usually means better accessibility'), RasterSpec('access_nonmotor_education', 'public_service_accessibility_nonmotor', 'Non-motorized education accessibility', '影响变量数据\\公共服务\\全球可达性数据\\非机动车\\processed\\education_global_urbanmask.tif', unit='travel cost/time', expected_direction='lower usually means better accessibility'), RasterSpec('access_nonmotor_entertainment', 'public_service_accessibility_nonmotor', 'Non-motorized entertainment accessibility', '影响变量数据\\公共服务\\全球可达性数据\\非机动车\\processed\\entertainment_global_urbanmask.tif', unit='travel cost/time', expected_direction='lower usually means better accessibility'), RasterSpec('access_nonmotor_health', 'public_service_accessibility_nonmotor', 'Non-motorized health accessibility', '影响变量数据\\公共服务\\全球可达性数据\\非机动车\\processed\\health_global_urbanmask.tif', unit='travel cost/time', expected_direction='lower usually means better accessibility'), RasterSpec('access_nonmotor_living', 'public_service_accessibility_nonmotor', 'Non-motorized living-service accessibility', '影响变量数据\\公共服务\\全球可达性数据\\非机动车\\processed\\living_global_urbanmask.tif', unit='travel cost/time', expected_direction='lower usually means better accessibility'), RasterSpec('access_nonmotor_production', 'public_service_accessibility_nonmotor', 'Non-motorized production-service accessibility', '影响变量数据\\公共服务\\全球可达性数据\\非机动车\\processed\\production_global_urbanmask.tif', unit='travel cost/time', expected_direction='lower usually means better accessibility'), RasterSpec('access_nonmotor_transport', 'public_service_accessibility_nonmotor', 'Non-motorized transport accessibility', '影响变量数据\\公共服务\\全球可达性数据\\非机动车\\processed\\transport_global_urbanmask.tif', unit='travel cost/time', expected_direction='lower usually means better accessibility'), RasterSpec('road_total', 'public_service_road', 'Total road length density', '影响变量数据\\公共服务\\路网数据\\processed\\全球路网长度栅格\\processed\\global_total_road_length_global_urbanmask.tif', unit='length per grid cell', expected_direction='higher means denser road network'), RasterSpec('road_motorway', 'public_service_road', 'Motorway road length density', '影响变量数据\\公共服务\\路网数据\\processed\\全球路网长度栅格\\processed\\motorway_global_urbanmask.tif', unit='length per grid cell', expected_direction='higher means denser road network'), RasterSpec('road_primary', 'public_service_road', 'Primary road length density', '影响变量数据\\公共服务\\路网数据\\processed\\全球路网长度栅格\\processed\\primary_global_urbanmask.tif', unit='length per grid cell', expected_direction='higher means denser road network'), RasterSpec('road_secondary', 'public_service_road', 'Secondary road length density', '影响变量数据\\公共服务\\路网数据\\processed\\全球路网长度栅格\\processed\\secondary_global_urbanmask.tif', unit='length per grid cell', expected_direction='higher means denser road network'), RasterSpec('road_tertiary', 'public_service_road', 'Tertiary road length density', '影响变量数据\\公共服务\\路网数据\\processed\\全球路网长度栅格\\processed\\tertiary_global_urbanmask.tif', unit='length per grid cell', expected_direction='higher means denser road network'), RasterSpec('road_residential', 'public_service_road', 'Residential road length density', '影响变量数据\\公共服务\\路网数据\\processed\\全球路网长度栅格\\processed\\residential_global_urbanmask.tif', unit='length per grid cell', expected_direction='higher means denser road network'), RasterSpec('road_service', 'public_service_road', 'Service road length density', '影响变量数据\\公共服务\\路网数据\\processed\\全球路网长度栅格\\processed\\service_global_urbanmask.tif', unit='length per grid cell', expected_direction='higher means denser road network'), RasterSpec('road_footway', 'public_service_road', 'Footway length density', '影响变量数据\\公共服务\\路网数据\\processed\\全球路网长度栅格\\processed\\footway_global_urbanmask.tif', unit='length per grid cell', expected_direction='higher means more walkable road network'), RasterSpec('road_pedestrian', 'public_service_road', 'Pedestrian road length density', '影响变量数据\\公共服务\\路网数据\\processed\\全球路网长度栅格\\processed\\pedestrian_global_urbanmask.tif', unit='length per grid cell', expected_direction='higher means more walkable road network'), RasterSpec('road_cycleway', 'public_service_road', 'Cycleway length density', '影响变量数据\\公共服务\\路网数据\\processed\\全球路网长度栅格\\processed\\cycleway_global_urbanmask.tif', unit='length per grid cell', expected_direction='higher means more bike-friendly road network'), RasterSpec('green_fraction', 'green_blue_space', 'Green-space fraction', '影响变量数据\\绿色空间\\数据\\processed\\ngreenspace_global_urbanmask.tif', unit='fraction', expected_direction='higher means more green-space cover'), RasterSpec('water_fraction', 'green_blue_space', 'Water fraction', '影响变量数据\\绿色空间\\数据\\processed\\nwater_global_urbanmask.tif', unit='fraction', expected_direction='higher means more water cover'), RasterSpec('impervious_fraction', 'built_green_control', 'Impervious surface fraction', '影响变量数据\\绿色空间\\数据\\processed\\nimpervious_global_urbanmask.tif', unit='fraction', expected_direction='higher means more impervious cover'), RasterSpec('building_density', 'built_environment', 'Built-up surface density', '影响变量数据\\建成环境\\建筑密度\\processed\\GHSL_BUILT_DENSITY_500m_2020_global_urbanmask.tif', unit='percent or fraction from source raster', expected_direction='higher means denser built-up surface'), RasterSpec('building_height', 'built_environment', 'Building height', '影响变量数据\\建成环境\\建筑高度\\150_Height.tif', unit='m', expected_direction='higher means taller buildings'), RasterSpec('ntl_radiance', 'economic_population', 'VIIRS nighttime-light radiance', '影响变量数据\\经济表现\\processed\\NTL_POP_2020_2020_500m_global_urbanmask.tif', band=1, unit='radiance', expected_direction='higher means stronger night-time activity'), RasterSpec('ntl_log1p', 'economic_population', 'Log-transformed nighttime light', '影响变量数据\\经济表现\\processed\\NTL_POP_2020_2020_500m_global_urbanmask.tif', band=2, unit='log(radiance+1)', expected_direction='higher means stronger night-time activity'), RasterSpec('population_count', 'economic_population', 'Population count', '影响变量数据\\经济表现\\processed\\NTL_POP_2020_2020_500m_global_urbanmask.tif', band=3, unit='people per grid cell', expected_direction='higher means more population'), RasterSpec('population_density', 'economic_population', 'Population density', '影响变量数据\\经济表现\\processed\\NTL_POP_2020_2020_500m_global_urbanmask.tif', band=4, unit='people/km2', expected_direction='higher means denser population'), RasterSpec('dem', 'terrain_control', 'Elevation', '影响变量数据\\地形\\processed\\ALOS_AW3D30_DEM_500m_global_urbanmask.tif', unit='m', expected_direction='control variable'), RasterSpec('slope', 'terrain_control', 'Slope', '影响变量数据\\地形\\processed\\ALOS_AW3D30_SLOPE_500m_global_urbanmask.tif', unit='degree', expected_direction='control variable')]

def find_project_file(filename: str) -> Path:
    hits = sorted(resource_glob(PROJECT_ROOT, filename))
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

def load_city_table() -> pd.DataFrame:
    df = read_dbf(find_project_file('global_city_point.dbf'))
    numeric_cols = ['lon', 'lat', 'id_1', 'NS', *EMOTIONS]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['id_1', 'lon', 'lat', 'NS', *EMOTIONS]).copy()
    df['id_1'] = df['id_1'].astype('int32')
    df['NS'] = df['NS'].round().astype('int8')
    df['ns_group'] = np.where(df['NS'] == 0, 'North', 'South')
    df['positive_score_city_table'] = df[POSITIVE].mean(axis=1)
    df['negative_score_city_table'] = df[NEGATIVE].mean(axis=1)
    df['eli_city_table'] = df['positive_score_city_table'] - df['negative_score_city_table']
    keep = ['id_1', 'COUNTRY', 'NAME_1', 'lon', 'lat', 'NS', 'ns_group', 'eli_city_table', 'positive_score_city_table', 'negative_score_city_table', *EMOTIONS]
    return df[keep].drop_duplicates('id_1')

def normalize_emotion(value: str) -> str:
    aliases = {'beautiful': 'beauty', 'beauty': 'beauty', 'wealth': 'wealthy', 'wealthy': 'wealthy', 'safety': 'safer', 'safe': 'safer', 'safer': 'safer', 'lively': 'livelier', 'liveliness': 'livelier', 'livelier': 'livelier', 'boring': 'boring', 'depressing': 'depression', 'depressed': 'depression', 'depression': 'depression'}
    key = str(value).strip().lower()
    if key not in aliases:
        raise ValueError(f'Unknown emotion: {value}')
    return aliases[key]

def load_outcome_table() -> pd.DataFrame:
    path = DERIVED_DIR / 'city_emotion_inequality_50m.csv'
    if not resource(path).exists():
        raise FileNotFoundError(f'Missing Q2 inequality table: {path}')
    metrics = pd.read_csv(resource(path))
    metrics.columns = [c.strip().lower() for c in metrics.columns]
    if 'city_id' in metrics.columns and 'id_1' not in metrics.columns:
        metrics = metrics.rename(columns={'city_id': 'id_1'})
    required = {'id_1', 'emotion', 'mean', 'gini', 'theil', 'p90_p10'}
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f'{path} missing columns: {missing}')
    metrics['id_1'] = pd.to_numeric(metrics['id_1'], errors='coerce').astype('Int64')
    metrics['emotion'] = metrics['emotion'].map(normalize_emotion)
    for col in ['mean', 'gini', 'theil', 'p90_p10', 'n_pixels']:
        if col in metrics.columns:
            metrics[col] = pd.to_numeric(metrics[col], errors='coerce')
    metrics = metrics.dropna(subset=['id_1', 'emotion', 'mean', 'gini', 'theil', 'p90_p10']).copy()
    metrics['id_1'] = metrics['id_1'].astype('int32')
    pieces = []
    for metric, prefix in [('mean', 'mean50'), ('gini', 'gini50'), ('theil', 'theil50'), ('p90_p10', 'p9010_50')]:
        pivot = metrics.pivot_table(index='id_1', columns='emotion', values=metric, aggfunc='mean')
        pivot = pivot.rename(columns={c: f'{prefix}_{c}' for c in pivot.columns})
        pieces.append(pivot)
    out = pd.concat(pieces, axis=1).reset_index()
    out['id_1'] = out['id_1'].astype('int32')
    out['eli50'] = out[[f'mean50_{e}' for e in POSITIVE]].mean(axis=1) - out[[f'mean50_{e}' for e in NEGATIVE]].mean(axis=1)
    out['compound_gini50'] = out[[f'gini50_{e}' for e in EMOTIONS]].mean(axis=1)
    out['compound_theil50'] = out[[f'theil50_{e}' for e in EMOTIONS]].mean(axis=1)
    out['compound_p9010_50'] = out[[f'p9010_50_{e}' for e in EMOTIONS]].mean(axis=1)
    if 'n_pixels' in metrics.columns:
        n_pix = metrics.groupby('id_1', as_index=False)['n_pixels'].min().rename(columns={'n_pixels': 'emotion_n_pixels_min'})
        out = out.merge(n_pix, on='id_1', how='left')
    return out

def collect_city_values(city_mask_path: Path, raster_path: Path, band: int) -> tuple[np.ndarray, np.ndarray, dict]:
    city_chunks: list[np.ndarray] = []
    value_chunks: list[np.ndarray] = []
    valid_blocks = 0
    valid_pixels = 0
    with rasterio.open(resource(city_mask_path)) as city_ds, rasterio.open(resource(raster_path)) as src:
        if band < 1 or band > src.count:
            raise ValueError(f'{raster_path} has {src.count} band(s); requested band {band}')
        with WarpedVRT(src, crs=city_ds.crs, transform=city_ds.transform, width=city_ds.width, height=city_ds.height, nodata=src.nodata, resampling=Resampling.average) as vrt:
            for _, window in city_ds.block_windows(1):
                raw_city = city_ds.read(1, window=window)
                city_valid = raw_city > 0
                if not city_valid.any():
                    continue
                vals = vrt.read(band, window=window, masked=True)
                sampled = vals[city_valid]
                sampled_city = raw_city[city_valid]
                if np.ma.isMaskedArray(sampled):
                    value_mask = ~sampled.mask
                    sampled_values = sampled.data[value_mask]
                    sampled_city = sampled_city[value_mask]
                else:
                    sampled_values = np.asarray(sampled)
                sampled_values = np.asarray(sampled_values, dtype='float32')
                good = np.isfinite(sampled_values) & (sampled_values > -9990) & (sampled_values < 1e+20)
                if not good.any():
                    continue
                city_ids = (sampled_city[good].astype('int32', copy=False) // 10).astype('int32', copy=False)
                id_good = (city_ids >= 1) & (city_ids <= 5905)
                if not id_good.any():
                    continue
                city_chunks.append(city_ids[id_good])
                value_chunks.append(sampled_values[good][id_good])
                valid_blocks += 1
                valid_pixels += int(id_good.sum())
    if not city_chunks:
        return (np.array([], dtype='int32'), np.array([], dtype='float32'), {'valid_blocks': 0, 'valid_pixels': 0})
    return (np.concatenate(city_chunks), np.concatenate(value_chunks), {'valid_blocks': valid_blocks, 'valid_pixels': valid_pixels})

def summarize_by_city(city_ids: np.ndarray, values: np.ndarray, prefix: str) -> pd.DataFrame:
    columns = ['id_1', f'{prefix}_mean', f'{prefix}_sd', f'{prefix}_p10', f'{prefix}_p50', f'{prefix}_p90', f'{prefix}_p90_p10', f'{prefix}_min', f'{prefix}_max', f'{prefix}_n']
    if city_ids.size == 0:
        return pd.DataFrame(columns=columns)
    order = np.argsort(city_ids, kind='mergesort')
    sorted_ids = city_ids[order]
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_ids)) + 1]
    ends = np.r_[starts[1:], sorted_ids.size]
    rows = []
    for start, end in zip(starts, ends):
        city_id = int(sorted_ids[start])
        x = sorted_values[start:end].astype('float64', copy=False)
        if x.size == 0:
            continue
        p10, p50, p90 = np.nanpercentile(x, [10, 50, 90])
        rows.append({'id_1': city_id, f'{prefix}_mean': float(np.nanmean(x)), f'{prefix}_sd': float(np.nanstd(x, ddof=0)), f'{prefix}_p10': float(p10), f'{prefix}_p50': float(p50), f'{prefix}_p90': float(p90), f'{prefix}_p90_p10': float(p90 - p10), f'{prefix}_min': float(np.nanmin(x)), f'{prefix}_max': float(np.nanmax(x)), f'{prefix}_n': int(x.size)})
    return pd.DataFrame(rows, columns=columns)

def write_inventory(specs: list[RasterSpec], output: Path) -> pd.DataFrame:
    rows = []
    for spec in specs:
        row = asdict(spec)
        row['path'] = str(spec.path)
        row['exists'] = resource(spec.path).exists()
        rows.append(row)
    inventory = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(output, index=False, encoding='utf-8-sig')
    return inventory

def build_master(output: Path, inventory_output: Path, metadata_output: Path, max_variables: int | None=None) -> pd.DataFrame:
    city_mask = find_project_file('Global_City_C_E.tif')
    specs = DRIVER_SPECS[:max_variables] if max_variables else DRIVER_SPECS
    inventory = write_inventory(specs, inventory_output)
    missing = inventory.loc[~inventory['exists'], ['prefix', 'path']]
    if not missing.empty:
        raise FileNotFoundError('Missing driver rasters:\n' + missing.to_string(index=False))
    master = load_city_table().merge(load_outcome_table(), on='id_1', how='left')
    metadata: dict[str, object] = {'created_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'city_mask': str(city_mask), 'city_id_rule': 'Global_City_C_E value // 10', 'driver_resampling_to_city_grid': 'average', 'urban_mask_rule': 'drivers use previously generated *_global_urbanmask.tif products where available; building_height is sampled from the original 150_Height.tif because the existing processed height mask is all NoData; city cells are constrained by Global_City_C_E.tif', 'output': str(output), 'variables': {}}
    for i, spec in enumerate(specs, start=1):
        t0 = time.time()
        city_ids, values, sample_meta = collect_city_values(city_mask, spec.path, spec.band)
        summary = summarize_by_city(city_ids, values, spec.prefix)
        master = master.merge(summary, on='id_1', how='left')
        elapsed = time.time() - t0
        metadata['variables'][spec.prefix] = {'family': spec.family, 'variable': spec.variable, 'path': str(spec.path), 'band': spec.band, 'unit': spec.unit, 'expected_direction': spec.expected_direction, 'sampled_values': int(values.size), 'cities_with_data': int(summary['id_1'].nunique()) if not summary.empty else 0, 'elapsed_seconds': round(elapsed, 2), **sample_meta}
        print(f"[{i:02d}/{len(specs):02d}] {spec.prefix}: values={values.size:,} cities={metadata['variables'][spec.prefix]['cities_with_data']} elapsed={elapsed:.1f}s", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(output, index=False, encoding='utf-8-sig')
    metadata_output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved master: {output}')
    print(f'Saved inventory: {inventory_output}')
    print(f'Saved metadata: {metadata_output}')
    return master

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build Q3 city-level driver master table.')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--inventory-output', type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument('--metadata-output', type=Path, default=DEFAULT_METADATA)
    parser.add_argument('--max-variables', type=int, default=None, help='Debug option: process only the first N variables.')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    build_master(args.output, args.inventory_output, args.metadata_output, args.max_variables)
