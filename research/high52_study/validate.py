import json
import numpy as np
import pandas as pd
from engine import OUT,prepare,run,run_account,rolling,SPECS,COLS

def main():
    f=prepare();dates=f['dates'];T=len(dates);checks={}
    # Known roundtrip: one initial equal-weight basket, one maturity, constant prices.
    t,n=15,20;p=np.full((t,n),100.);v=np.ones((t,n));m=np.ones((t,n),bool)
    weights=np.ones((1,t,n))/n;sched=np.full(t,-1,np.int64);sched[0]=0
    tax=np.full(t,.002);comm=np.full(t,.00015)
    result=run_account(p,p,v,m,weights,tax,comm,sched,np.array([5]),np.array([0]),0,t,.0005,1.,True)
    expected=1e7*(1-.00265)/(1+.00065)
    err=abs(result[0][-1,0]-expected);assert err<1e-5
    assert np.count_nonzero(result[2].sum(1))==2 and np.allclose(result[1][-1],0)
    checks['known_roundtrip_nav_error']=err
    # No same-close fill: changing next day's executable price changes shares but not prior decision.
    p2=p.copy();p2[1:]=200
    rr=run_account(p2,p2,v,m,weights,tax,comm,sched,np.array([5]),np.array([0]),0,t,.0005,1.,True)
    assert np.allclose(rr[1][0],0) and np.allclose(rr[1][1],result[1][1]/2)
    checks['next_close_cash_scaled_execution']=True
    # Missing volume cannot produce fills; queued maturity is retried.
    vv=v.copy();vv[6]=0
    r3=run_account(p,p,vv,m,weights,tax,comm,sched,np.array([5]),np.array([0]),0,t,.0005,1.,True)
    assert np.all(r3[2][6]==0) and r3[2][7].sum()<0
    checks['zero_volume_delays_exit']=True
    checks['accounts']={}
    for file in (OUT/'audit_arrays').glob('*.npz'):
        name=file.stem;z=np.load(file);q=z['positions'];dq=z['orders'];rec=z['recoveries'];d=pd.read_csv(OUT/'daily'/f'{name}.csv',index_col=0,parse_dates=True)
        prev=np.vstack([np.zeros(q.shape[1]),q[:-1]])
        err=float(np.max(abs(q-(prev+dq-rec))));assert err<1e-6
        values=np.abs(dq)*np.nan_to_num(f['p']);buy=np.where(dq>0,values,0).sum(1);sell=np.where(dq<0,values,0).sum(1)
        c=(buy+sell)*f['commission'];st=sell*f['tax'];sl=(buy+sell)*.0005
        cash=1e7+np.cumsum(sell-buy-c-st-sl+(rec*f['marks']).sum(1))
        nav=cash+(q*f['marks']).sum(1)
        naverr=float(np.max(abs(nav-d.nav.to_numpy())));assert naverr<.05
        bad=(abs(dq)>1e-9)&(~np.isfinite(f['p'])|~(f['v']>0));assert not bad.any()
        assert d.cash.min()>-.001 and q.min()>-1e-7
        schedule_name=name if name.startswith('ROLL_') else ('ROLL_HIGH_EW' if name=='MATCHED_HIGH_IV' else 'ROLL_MOM_EW' if name=='MATCHED_MOM_IV' else 'ROLL_MOM_WEIGHT')
        a=pd.read_csv(OUT/'schedules'/f'{schedule_name}.csv',parse_dates=['decision_date','latest_observed_return','train_end','validation_start'])
        assert (a.latest_observed_return<a.decision_date).all() and (a.train_end<a.validation_start).all()
        checks['accounts'][name]=dict(position_rollforward_error=err,independent_nav_max_error=naverr,no_untradable_fills=True,positive_cash=True,latest_training_return_strictly_before_signal=True)
    raw=dict(np.load(OUT/'inputs.npz'));cut=int(dates.searchsorted('2024-01-02'))
    rawcut={k:(v[:cut].copy() if v.shape and v.shape[0]==T else v.copy()) for k,v in raw.items()};fc=prepare(rawcut)
    np.testing.assert_allclose(fc['weights'],f['weights'][:,:cut],atol=0,rtol=0)
    rets=np.load(OUT/'candidate_returns.npz')['returns'];ids=[i for i,s in enumerate(SPECS) if s['fi']<=2]
    a,_=rolling(rets,ids,dates);b,_=rolling(rets[:cut],ids,dates[:cut]);assert np.array_equal(a[:cut],b)
    full=run(f,a,start='2017-01-02',record=True,stop=cut);trunc=run(fc,b,start='2017-01-02',record=True)
    np.testing.assert_allclose(full[0][:cut],trunc[0],atol=1e-9,rtol=0)
    np.testing.assert_allclose(full[2][:cut],trunc[2],atol=1e-9,rtol=0)
    checks['future_truncation_features_selection_orders_nav']=True
    from overlay import execute
    sch=np.load(OUT/'schedules/ROLL_MOM_WEIGHT.npy')
    z=execute(fc,sch[:cut]);saved=np.load(OUT/'audit_arrays/MID_HIGH_TILT.npz')
    np.testing.assert_allclose(z[2],saved['orders'][:cut],atol=1e-7,rtol=0)
    checks['midpoint_overlay_future_truncation_orders']=True
    checks['input_sha256']='See data_manifest.json'
    (OUT/'validation.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
