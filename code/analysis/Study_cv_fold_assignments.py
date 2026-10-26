"""Recover recorded model folds from the documented splits and sample order."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

ROOT=Path(__file__).resolve().parents[2]


def build(folder,prefix,write_to=None):
    predictions=pd.read_csv(ROOT/f'data/shared/{folder}/{prefix}_predictions.csv')
    reference=pd.read_csv(ROOT/f'data/shared/{folder}/{prefix}_folds.csv')
    parts=[]
    keys=['table','outcome','outcome_role','group','group_type']
    for values,frame in predictions.groupby(keys,sort=False):
        frame=frame.copy().reset_index(drop=True)
        n=len(frame)
        splits=min(3,max(2,n//450))
        fold=np.zeros(n,dtype=int)
        selected=reference
        for key,value in zip(keys,values):selected=selected.loc[selected[key].eq(value)]
        for number,(_,test) in enumerate(KFold(splits,shuffle=True,random_state=42).split(frame),1):
            fold[test]=number
            recorded=selected.loc[selected['fold'].eq(number)].iloc[0]
            assert len(test)==int(recorded['n_test'])
            assert np.isclose(r2_score(frame.loc[test,'observed'],frame.loc[test,'oof_prediction']),recorded['r2'],rtol=0,atol=2e-11)
        columns=[c for c in ['id_1','zone_code','zone_label',*keys] if c in frame]
        result=frame[columns].copy()
        result['sample_order_in_model']=np.arange(n)
        result['held_out_fold']=fold
        parts.append(result)
    result=pd.concat(parts,ignore_index=True)
    target=write_to or ROOT/f'data/shared/{folder}/{prefix}_sample_fold_assignments.csv'
    result.to_csv(target,index=False,encoding='utf-8-sig')
    return result


def main():
    for folder,prefix in [('global_association_models','Global_model'),('zone_association_models','Zone_model')]:
        result=build(folder,prefix)
        print(prefix,len(result),'sample-to-model fold assignments; held-out R2 verified')
    predictions=pd.read_csv(ROOT/'data/Fig_5/oof_analysis/Fig_5_predictions.csv')
    reference=pd.read_csv(ROOT/'data/Fig_5/oof_analysis/Fig_5_folds.csv')
    parts=[]
    for (outcome,group),frame in predictions.groupby(['outcome','group'],sort=False):
        frame=frame.copy().reset_index(drop=True)
        fold=np.zeros(len(frame),dtype=int)
        for number,(_,test) in enumerate(KFold(3,shuffle=True,random_state=42).split(frame),1):
            fold[test]=number
            record=reference.loc[reference.outcome.eq(outcome)&reference.group.eq(group)&reference['fold'].eq(number)].iloc[0]
            assert len(test)==record.n_test
            assert np.isclose(r2_score(frame.loc[test,'observed'],frame.loc[test,'oof_prediction']),record.r2,rtol=0,atol=2e-11)
        record=frame[['id_1','outcome','group']].copy()
        record['sample_order_in_model']=np.arange(len(frame));record['held_out_fold']=fold
        parts.append(record)
    pd.concat(parts,ignore_index=True).to_csv(ROOT/'data/Fig_5/oof_analysis/Fig_5_sample_fold_assignments.csv',index=False,encoding='utf-8-sig')
    print('Figure 5 OOF importance models: fold assignments verified')


if __name__=='__main__':
    main()
