"""Resolve analysis inputs independently of the working directory."""
from pathlib import Path
import builtins
import fnmatch
import json
import os

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = Path(os.environ.get("STUDY_EXTERNAL_INPUTS", PACKAGE_ROOT / "external_inputs"))
OUTPUT_ROOT = Path(os.environ.get("STUDY_OUTPUT_DIR", PACKAGE_ROOT / "outputs"))
_INPUTS = json.loads((PACKAGE_ROOT / "code/input_paths.json").read_text(encoding="utf-8"))


def output_dir(label):
    path = OUTPUT_ROOT / label
    path.mkdir(parents=True, exist_ok=True)
    return path


def resource(path):
    if hasattr(path, "read") or path is None:
        return path
    p = Path(path)
    if p.exists():
        return p
    key = str(p).replace("\\", "/")
    base = str(PACKAGE_ROOT).replace("\\", "/")
    if key.startswith(base + "/"):
        key = key[len(base) + 1:]
    if key in _INPUTS:
        return PACKAGE_ROOT / _INPUTS[key]
    matches = {_INPUTS[k] for k in _INPUTS if k.rsplit("/", 1)[-1] == p.name}
    if len(matches) == 1:
        return PACKAGE_ROOT / matches.pop()
    external = EXTERNAL_ROOT / p.name
    if external.exists():
        return external
    return p


def resource_glob(directory, pattern):
    directory = Path(directory)
    results = list(directory.glob(pattern)) if directory.is_dir() else []
    prefix = str(directory).replace("\\", "/")
    base = str(PACKAGE_ROOT).replace("\\", "/")
    if prefix.startswith(base):
        prefix = prefix[len(base):].lstrip("/")
    match = (prefix + "/" + str(pattern)).lstrip("/")
    for key, target in _INPUTS.items():
        if fnmatch.fnmatch(key, match):
            results.append(PACKAGE_ROOT / target)
    return list(dict.fromkeys(results))


def open_resource(path, mode="r", *args, **kwargs):
    if not any(flag in mode for flag in "wax+"):
        path = resource(path)
    return builtins.open(path, mode, *args, **kwargs)


def walking_configuration():
    path=EXTERNAL_ROOT/'walking_plot_parameters.json'
    if not path.is_file():
        raise FileNotFoundError('Supplementary Figure S8 requires walking_plot_parameters.json; original walking parameters are not included.')
    settings=json.loads(path.read_text(encoding='utf-8'))
    for key in ['raster_files','vmin','vmax']:
        if key not in settings:raise ValueError('Missing walking configuration field: '+key)
    if len(settings['raster_files'])!=6:
        raise ValueError('Six walking service categories are required.')
    for row in settings['raster_files']:
        if not all(key in row for key in ['file','title','north','south']):
            raise ValueError('Each walking category needs file, title, north and south.')
    return settings
