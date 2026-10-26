Code organization

figures/ holds the actual plotting implementations under the final manuscript figure numbers. Run or edit the corresponding script directly; there is no drawing dispatcher. analysis/ contains the standalone data processing, statistical analysis, validation and association-model implementations. _support/ contains supporting plotting functions and shared path/command-line utilities. The shared figure_cli.py only handles command-line options, input checks and output export; it does not select or load a drawing module. run_analysis.py lists and runs the included analysis tasks.

The primary reproduction path starts from data/shared and the figure-specific numerical tables. Raster aggregation routines are supplied as implementations but need local rasters. Association-model fitting, cross-validation and model explanation are distinct from the excluded perception and spatial-prediction pipeline. Some full-model and resampling tasks may require substantial memory and runtime; they are never run automatically by a plotting entry point.

Figure 1 requires external illustration assets and does not exactly reproduce the final manually assembled composite. Supplementary Figure S8 has a missing historical walking configuration. Raster maps require external files. See figure_index.csv for the status of each entry.

The exact VIF implementation is supplied as analysis/Extended_Data_Fig_3_vif.R and requires R with the data.table package. Global_model tables contain the city models used in Figure 5. Zone_model tables retain the formal city and city-zone comparisons used in the model-reliability figure and Figure 6. Sample-to-model fold assignments are checked against the recorded held-out R2 for every fold.

The Figure 5 pooled out-of-fold importance results record scikit-learn 1.9.0. Saved formal-model predictions and response tables are separate inputs. Re-training with a different software version may not reproduce those older fitted values exactly. Plotting uses the supplied numerical outputs.
