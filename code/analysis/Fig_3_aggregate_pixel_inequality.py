from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "_support"))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "figures"))

from package_paths import PACKAGE_ROOT, resource, resource_glob, open_resource, output_dir
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
PROJECT_ROOT = PACKAGE_ROOT
DERIVED_DIR = PROJECT_ROOT / 'derived_data'
DEFAULT_OUTPUT = DERIVED_DIR / 'city_emotion_inequality_50m.csv'
DEFAULT_META = DERIVED_DIR / 'city_emotion_inequality_50m_metadata.json'
EMOTION_ALIASES = {'beauty': 'beauty', 'beautiful': 'beauty', 'safer': 'safer', 'safety': 'safer', 'livelier': 'livelier', 'liveliness': 'livelier', 'wealthy': 'wealthy', 'wealth': 'wealthy', 'boring': 'boring', 'depression': 'depression', 'depressing': 'depression'}

def find_project_file(name: str) -> Path:
    hits = sorted(resource_glob(PROJECT_ROOT, name))
    if not hits:
        raise FileNotFoundError(f'Cannot find {name} under {PROJECT_ROOT}')
    return hits[0]

def parse_emotion_rasters(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in items:
        if '=' not in item:
            raise ValueError(f'Use emotion=path format, got: {item}')
        emotion, path_text = item.split('=', 1)
        key = EMOTION_ALIASES.get(emotion.strip().lower())
        if key is None:
            raise ValueError(f'Unknown emotion name: {emotion}')
        path = Path(path_text)
        if not resource(path).exists():
            raise FileNotFoundError(f'Missing raster for {key}: {path}')
        out[key] = path
    required = {'beauty', 'wealthy', 'safer', 'livelier', 'boring', 'depression'}
    missing = sorted(required - set(out))
    if missing:
        raise ValueError(f'Missing emotion rasters: {missing}')
    return out

def gini_sorted(x: np.ndarray) -> float:
    if x.size < 2:
        return np.nan
    xs = np.sort(x.astype('float64', copy=False))
    total = xs.sum()
    if total <= 0:
        return 0.0
    n = xs.size
    weights = np.arange(1, n + 1, dtype='float64')
    return float(2.0 * np.dot(weights, xs) / (n * total) - (n + 1.0) / n)

def theil_index(x: np.ndarray) -> float:
    if x.size < 2:
        return np.nan
    vals = x.astype('float64', copy=False)
    mean = vals.mean()
    if mean <= 0:
        return np.nan
    ratio = vals / mean
    terms = np.zeros_like(ratio, dtype='float64')
    positive = ratio > 0
    terms[positive] = ratio[positive] * np.log(ratio[positive])
    return float(terms.mean())

def metrics_by_city(city_ids: np.ndarray, values: np.ndarray, emotion: str) -> pd.DataFrame:
    valid = np.isfinite(values) & (values > -9990) & (city_ids >= 1) & (city_ids <= 5905)
    city_ids = city_ids[valid].astype('int32', copy=False)
    values = values[valid].astype('float32', copy=False)
    if city_ids.size == 0:
        return pd.DataFrame(columns=['id_1', 'emotion', 'mean', 'gini', 'theil', 'p90_p10', 'n_pixels'])
    order = np.argsort(city_ids, kind='mergesort')
    sorted_ids = city_ids[order]
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_ids)) + 1]
    ends = np.r_[starts[1:], sorted_ids.size]
    rows = []
    for start, end in zip(starts, ends):
        city_id = int(sorted_ids[start])
        x = sorted_values[start:end]
        if x.size == 0:
            continue
        p10, p90 = np.percentile(x, [10, 90])
        rows.append({'id_1': city_id, 'emotion': emotion, 'mean': float(np.mean(x)), 'gini': gini_sorted(x), 'theil': theil_index(x), 'p90_p10': float(p90 - p10), 'n_pixels': int(x.size)})
    return pd.DataFrame(rows)

def collect_sampled_values(city_mask_path: Path, raster_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    city_chunks: list[np.ndarray] = []
    value_chunks: list[np.ndarray] = []
    valid_blocks = 0
    valid_pixels = 0
    with rasterio.open(resource(city_mask_path)) as city_ds, rasterio.open(resource(raster_path)) as src:
        with WarpedVRT(src, crs=city_ds.crs, transform=city_ds.transform, width=city_ds.width, height=city_ds.height, nodata=src.nodata, resampling=Resampling.average) as vrt:
            for _, window in city_ds.block_windows(1):
                raw_city = city_ds.read(1, window=window)
                city_valid = raw_city > 0
                if not city_valid.any():
                    continue
                vals = vrt.read(1, window=window, masked=True)
                sampled = vals[city_valid]
                if np.ma.isMaskedArray(sampled):
                    value_mask = ~sampled.mask
                    sampled_values = sampled.data[value_mask]
                    sampled_city = raw_city[city_valid][value_mask]
                else:
                    sampled_values = np.asarray(sampled)
                    sampled_city = raw_city[city_valid]
                sampled_values = np.asarray(sampled_values, dtype='float32')
                good = np.isfinite(sampled_values) & (sampled_values > -9990)
                if not good.any():
                    continue
                city_ids = (sampled_city[good].astype('int32', copy=False) // 10).astype('int32', copy=False)
                city_chunks.append(city_ids)
                value_chunks.append(sampled_values[good])
                valid_blocks += 1
                valid_pixels += int(good.sum())
    if not city_chunks:
        return (np.array([], dtype='int32'), np.array([], dtype='float32'), {'valid_blocks': 0, 'valid_pixels': 0})
    return (np.concatenate(city_chunks), np.concatenate(value_chunks), {'valid_blocks': valid_blocks, 'valid_pixels': valid_pixels})

def build_metrics(emotion_rasters: dict[str, Path], output: Path) -> pd.DataFrame:
    city_mask = find_project_file('Global_City_C_E.tif')
    frames = []
    audit = {'method': 'high_resolution_emotion_rasters_resampled_to_Global_City_C_E_city_mask_grid', 'resampling': 'average', 'city_id_rule': 'Global_City_C_E value // 10', 'city_mask': str(city_mask), 'output': str(output), 'emotions': {}}
    for emotion, raster_path in emotion_rasters.items():
        t0 = time.time()
        city_ids, values, meta = collect_sampled_values(city_mask, raster_path)
        df = metrics_by_city(city_ids, values, emotion)
        frames.append(df)
        elapsed = time.time() - t0
        audit['emotions'][emotion] = {'raster': str(raster_path), 'sampled_pixels': int(values.size), 'cities_with_data': int(df['id_1'].nunique()) if not df.empty else 0, 'elapsed_seconds': round(elapsed, 2), **meta}
        print(f"{emotion}: pixels={values.size:,} cities={audit['emotions'][emotion]['cities_with_data']} elapsed={elapsed:.1f}s", flush=True)
    out = pd.concat(frames, ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False, encoding='utf-8-sig')
    DEFAULT_META.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved metrics: {output}')
    print(f'Saved metadata: {DEFAULT_META}')
    return out

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--emotion-raster', action='append', required=True, help='emotion=path')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    emotion_rasters = parse_emotion_rasters(args.emotion_raster)
    build_metrics(emotion_rasters, args.output)
