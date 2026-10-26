"""Run the statistical and association-model analyses supplied with the figures."""
from pathlib import Path
import argparse
import importlib
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "_support"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "figures"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "analysis"))
from package_paths import PACKAGE_ROOT, output_dir

TASKS = {
    "Fig_2_spatial_bootstrap": ("Fig_2_spatial_bootstrap", False),
    "Fig_2_city_aggregation": ("Fig_2_build_city_analysis", True),
    "Fig_3_pixel_inequality": ("Fig_3_aggregate_pixel_inequality", True),
    "Fig_4_zone_aggregation": ("Fig_4_aggregate_core_expansion", True),
    "Fig_5_Fig_6_association_models": ("Fig_5_Fig_6_association_models", False),
    "Fig_5_oof_importance": ("Fig_5_oof_permutation", False),
    "Fig_5_response_uncertainty": ("Fig_5_response_uncertainty", False),
    "Fig_5_response_curves": ("Fig_5_response_curves", False),
    "Fig_6_zone_comparisons": ("Fig_6_zone_association_comparisons", False),
    "Fig_6_zone_table": ("Fig_6_build_zone_analysis", False),
    "Fig_6_zone_predictors": ("Fig_6_build_zone_predictors", True),
    "Extended_Data_Fig_1_robustness": ("Extended_Data_Fig_1_index_robustness", False),
    "Extended_Data_Fig_1_external_validation": ("Extended_Data_Fig_1_external_validation", True),
    "Extended_Data_Fig_3_model_robustness": ("Extended_Data_Fig_3_model_robustness", False),
    "Extended_Data_Fig_4_context_statistics": ("Extended_Data_Fig_4_context_statistics", False),
    "Extended_Data_Fig_5_local_associations": ("Extended_Data_Fig_5_local_associations", False),
}


def run(task, allow_external=False):
    module, external = TASKS[task]
    if external and not allow_external:
        raise SystemExit("This aggregation needs local external inputs; supply them before using --allow-external.")
    m = importlib.import_module(module)
    for name in ["RUN_DIR", "OUT_DIR", "OUT", "OUTPUT_DIR"]:
        if hasattr(m, name):
            setattr(m, name, output_dir(task))
    if task=='Fig_6_zone_comparisons':
        names={'family':'family','permutation':'permutation','ale':'ale','pdp':'pdp','sample':'sample','performance':'rf_cv_performance','predictors':'predictors_used'}
        m.INPUTS={key:PACKAGE_ROOT/f'data/shared/zone_association_models/Zone_model_{name}.csv' for key,name in names.items()}
        m.OUTPUTS={key:output_dir(task)/('Fig_6_'+key+('.json' if key=='metadata' else '.csv')) for key in m.OUTPUTS}
    if task=='Extended_Data_Fig_3_model_robustness':
        m.FORMAL_PERFORMANCE=PACKAGE_ROOT/'data/shared/zone_association_models/Zone_model_rf_cv_performance.csv'
        m.FORMAL_FAMILY=PACKAGE_ROOT/'data/shared/zone_association_models/Zone_model_family.csv'
    if task=='Extended_Data_Fig_1_robustness':
        import pandas as pd
        city_long=pd.read_csv(PACKAGE_ROOT/'data/Fig_3/Fig_3_city_dimension_inequality.csv')
        city=pd.read_csv(PACKAGE_ROOT/'data/shared/Study_city_analysis.csv')
        city_long['id_1']=city_long['id_1'].astype(str)
        city['id_1']=city['id_1'].astype(str)
        m.build_q1_f(city_long,city).to_csv(output_dir(task)/'Extended_Data_Fig_1_rank_stability.csv',index=False)
        m.build_q1_h(city).to_csv(output_dir(task)/'Extended_Data_Fig_1_weight_robustness.csv',index=False)
        return
    if task=='Fig_2_spatial_bootstrap':
        sys.argv=[task,'--output-dir',str(output_dir(task))]
    m.main()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task",nargs="?",choices=TASKS)
    parser.add_argument("--list",action="store_true")
    parser.add_argument("--allow-external",action="store_true")
    args=parser.parse_args()
    if args.list:
        for task,(_,external) in TASKS.items():
            print(task + (" [external inputs]" if external else ""))
        return
    if args.task is None:
        parser.error("Choose a task or --list.")
    sys.argv=[args.task]
    run(args.task,args.allow_external)


if __name__=="__main__":
    main()
