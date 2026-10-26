Analysis scripts

Run python code/run_analysis.py --list to list the analysis tasks.
Run python code/run_analysis.py TASK to execute a task from any working directory.
Tasks marked [external inputs] require separately supplied local data and --allow-external.
Model fitting and resampling can take substantial time and memory. Plotting scripts use the supplied saved tables.

Task / implementation / external inputs
Fig_2_spatial_bootstrap / Fig_2_spatial_bootstrap.py / not required by the task entry
Fig_2_city_aggregation / Fig_2_build_city_analysis.py / required
Fig_3_pixel_inequality / Fig_3_aggregate_pixel_inequality.py / required
Fig_4_zone_aggregation / Fig_4_aggregate_core_expansion.py / required
Fig_5_Fig_6_association_models / Fig_5_Fig_6_association_models.py / not required by the task entry
Fig_5_oof_importance / Fig_5_oof_permutation.py / not required by the task entry
Fig_5_response_uncertainty / Fig_5_response_uncertainty.py / not required by the task entry
Fig_5_response_curves / Fig_5_response_curves.py / not required by the task entry
Fig_6_zone_comparisons / Fig_6_zone_association_comparisons.py / not required by the task entry
Fig_6_zone_table / Fig_6_build_zone_analysis.py / not required by the task entry
Fig_6_zone_predictors / Fig_6_build_zone_predictors.py / required
Extended_Data_Fig_1_robustness / Extended_Data_Fig_1_index_robustness.py / not required by the task entry
Extended_Data_Fig_1_external_validation / Extended_Data_Fig_1_external_validation.py / required
Extended_Data_Fig_3_model_robustness / Extended_Data_Fig_3_model_robustness.py / not required by the task entry
Extended_Data_Fig_4_context_statistics / Extended_Data_Fig_4_context_statistics.py / not required by the task entry
Extended_Data_Fig_5_local_associations / Extended_Data_Fig_5_local_associations.py / not required by the task entry

Additional analysis files:
Extended_Data_Fig_2_quantile_intervals.py: calculate bootstrap quantile intervals; also supplies the plotting configuration used by Extended Data Figure 2.
Study_cv_fold_assignments.py: reconstruct sample-to-fold assignments.
Supplementary_Fig_S2_validation_statistics.py: calculate LST validation metrics.
Extended_Data_Fig_3_vif.R: calculate predictor variance inflation factors (R and data.table required).

Some panel-specific statistics remain inside the plotting functions that use them.
