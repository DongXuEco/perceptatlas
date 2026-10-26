"""Recalculate LST validation metrics from the supplied sample records."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"_support"))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"figures"))
from package_paths import PACKAGE_ROOT, output_dir
import pandas as pd
from Supplementary_Fig_S2 import metrics


def main():
    records=[]
    for name in ["cross_validation","independent_validation"]:
        data=pd.read_csv(PACKAGE_ROOT/f"data/Supplementary_Fig_S2/Supplementary_Fig_S2_{name}.csv")
        result=metrics(data)
        records.append({"validation":name,**result})
    path=output_dir("Supplementary_Fig_S2")/"Supplementary_Fig_S2_validation_metrics.csv"
    pd.DataFrame(records).to_csv(path,index=False)
    print(path)


if __name__=="__main__":
    main()
