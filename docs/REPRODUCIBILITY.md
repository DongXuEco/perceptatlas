# Reproducing the supplied analyses and figures

## Obtain the data-inclusive package

Download [perceptatlas-research-v1.0.0.zip](https://github.com/DongXuEco/perceptatlas/releases/download/v1.0.0/perceptatlas-research-v1.0.0.zip). Extract it so that `code/` and `data/` share the same parent directory. The release includes 188 unchanged data files from the prepared study package. The ordinary repository contains code, documentation and platform screenshots; it does not contain those large numerical tables.

The release’s `SHA256SUMS.txt` checks the full ZIP. Inside the ZIP, `RELEASE_MANIFEST.csv` records the size and SHA-256 digest of each payload file. The historical `file_inventory.csv` describes the original study package, not the later documentation or screenshot additions.

## Environment and commands

Use Python 3.12 and the recorded versions in [`code/requirements.txt`](../code/requirements.txt). An isolated virtual environment is recommended. These are the study’s recorded dependency versions; this publication step did not rebuild every environment or rerun every statistical model.

```bash
python -m pip install -r code/requirements.txt
python code/figures/Fig_2.py --check-inputs
python code/figures/Fig_2.py --formats png pdf
python code/figures/Extended_Data_Fig_2.py --formats png pdf
python code/run_analysis.py --list
```

Outputs are written to `outputs/` by default. Set `STUDY_OUTPUT_DIR` to override the destination. Arial was used in the original figures; substituting fonts can change text extents. The exact VIF analysis is `code/analysis/Extended_Data_Fig_3_vif.R` and needs R with `data.table`.

Plotting uses stored numerical results. Association-model fitting, cross-validation and explanation are separate tasks, and are not the excluded perception-scoring or spatial-prediction pipeline. Full-model fitting and resampling can take substantial memory and time. Raster aggregation implementations require the user to supply the external rasters listed in the indexes.

## What the validation record means

[`reproduction_status.csv`](../reproduction_status.csv) records 12 passed entries, using only the supplied package: Figures 2–6, Extended Data Figures 1–6, and Supplementary Figure S10. This is a retained validation record, not a claim that all computations were rerun during GitHub publication.

The full index has 26 entries: 12 analysis-table entries, 13 entries requiring external inputs, and one requiring an external walking configuration. In particular:

- **Figure 1:** supporting illustration functions are supplied in `_support/`; no complete `Fig_1.py` is present. The final manuscript composite was assembled separately and uses external artwork.
- **Supplementary Figure S8:** the historical walking configuration is not supplied.
- **Raster-dependent maps:** original rasters and contextual imagery are not distributed. Numerical source tables do not replace those visual inputs.

Use [`figure_index.csv`](../figure_index.csv), [`figure_inputs.csv`](../figure_inputs.csv) and [`external_inputs_required.csv`](../external_inputs_required.csv) together. The release corrects the Figure 1 index to point to the actual supporting functions, without inventing a standalone figure script.

A stale alias to the absent `Fig_1.py` was also removed from `code/input_paths.json`. Otherwise the shared input checker would report a missing figure script as a missing data input. Analysis and plotting implementations are unchanged.

## Definitions and interpretation

ELI is the mean of beauty, wealth, safety and liveliness minus the mean of boredom and depression. Each dimension receives equal weight within its positive or negative group. Higher boredom or depression is not interpreted as better environmental quality.

Core–expansion ELI differences are core minus expansion. In the three-category composition, expansion advantage is below −0.03, approximate balance is −0.03 to +0.03, and core advantage is above +0.03. Higher positive perception scores and lower negative perception scores can both indicate a core advantage; check the figure-specific sign convention.

The manuscript analysis cohort has 5,880 cities. The website catalog has 5,905 geolocated records; valid counts vary by measure. Statistical environmental associations are not causal effects, and the predicted perception scores are not clinical or individual mental-health observations.

## Scope of public materials

Included: processed source tables, shared city and city-zone analysis inputs, validation tables, statistical and environmental-association analysis implementations, actual plotting implementations, indexes and documentation.

Excluded: raw street-view imagery, original gridded perception products, acquisition scripts, perception-scoring and spatial-prediction pipelines, model-service infrastructure, private manuscript files and credentials.
