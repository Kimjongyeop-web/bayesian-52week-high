from pathlib import Path
import argparse,hashlib,json,time,sys
import numpy as np
import pandas as pd
from scipy.stats import t as student
from threadpoolctl import threadpool_limits
from posterior import prepare,forecast
from engine import run_kernel

OUT=Path(__file__).resolve().parent
MODELS=[(5,504),(5,1008),(20,504),(20,1008)]
COLS=['nav','net_return','exposure','turnover','commission','sale_tax','execution_cost','writeoff','cash','buy_notional','sell_notional','gross_return','n_holdings','review','planned_exposure','relative_probability_mean','planned_max_weight']
SPECS=[]
for cap in [.06,.08,.10]:
    for lam in [1.,3.]:
        SPECS.append(dict(name=f'EQ_C{int(cap*100)}_L{int(lam)}',model=0,p0=.5,strength=0.,cap=cap,lam=lam))
for mi,(h,w) in enumerate(MODELS):
    for p0 in [.5,.55,.6]:
        for strength in [.5,1.]:
            for cap in [.06,.08,.1]:
                for lam in [1.,3.]:
                    SPECS.append(dict(name=f'B_H{h}_W{w}_P{int(p0*100)}_S{strength}_C{int(cap*100)}_L{int(lam)}',model=mi,p0=p0,strength=strength,cap=cap,lam=lam))
BYNAME={s['name']:s for s in SPECS}
def log(x):print(time.strftime('%H:%M:%S'),x,flush=True)
def control(s):return f'EQ_C{int(s["cap"]*100)}_L{int(s["lam"])}'

def load():
    f=prepare();f['start']=int(f['dates'].searchsorted('2009-08-07'))
    path=OUT/'posterior_summary.npz'
    if not path.exists():
        allp={}
        for h,w in MODELS:
            prev=OUT.parent/'high52_bayes/predictions'/f'H{h}_W{w}.npz'
            pred=dict(np.load(prev)) if prev.exists() else forecast(f,h,w,save=False)
            for k in ['coef','cov','df']:allp[f'{h}_{w}_{k}']=pred[k]
        np.savez_compressed(path,**allp)
    data=np.load(path);preds={i:{k:data[f'{h}_{w}_{k}'] for k in ['coef','cov','df']} for i,(h,w) in enumerate(MODELS)}
    return f,preds

def simulate(f,preds,s,start=None,stop=None,record=False,slip=.0005,recovery=1.,external=None,interval=1):
    start=f['start'] if start is None else start;stop=len(f['dates']) if stop is None else stop
    pr=preds[s['model']];h,w=MODELS[s['model']]
    args=[f[k] for k in ['p','marks','v','m','eligible','near','rank','X','sigma','tax','comm']]
    return run_kernel(*args,pr['coef'],pr['cov'],pr['df'],h,s['p0'],s['strength'],s['cap'],s['lam'],start,stop,slip,recovery,record,external,interval)

def metrics(d,dates,lo,hi):
    ix=(dates>=lo)&(dates<=hi);x=d[ix];r=x[:,1];wealth=np.r_[1.,np.cumprod(1+r)];years=(dates[ix][-1]-dates[ix][0]).days/365.25
    return dict(cagr=wealth[-1]**(1/years)-1,total_return=wealth[-1]-1,sharpe=r.mean()/r.std(ddof=1)*np.sqrt(252) if r.std()>1e-12 else 0.,mdd=np.min(wealth/np.maximum.accumulate(wealth)-1),vol=r.std(ddof=1)*np.sqrt(252),exposure=x[:,2].mean(),annual_turnover=x[:,3].mean()*252,annual_cost_rate=np.mean(x[:,4:7].sum(1)/x[:,0])*252)

def posterior_advantage(x):
    x=np.asarray(x);x=x[np.isfinite(x)];n=len(x);k=12+n;a=3+n/2;mean=x.mean() if n else 0.;m=n*mean/k;b=.0008+.5*np.sum((x-mean)**2)+.5*12*n*mean*mean/k
    se=np.sqrt(b/a/k);q=student.cdf(m/se,2*a)
    return dict(n=n,mean_monthly=m,p_positive=q,lower10=m+student.ppf(.10,2*a)*se,upper90=m+student.ppf(.90,2*a)*se)

def monthly_log(r,dates):return pd.Series(np.log1p(r),index=dates).groupby(dates.to_period('M')).sum()

def train():
    for directory in ['daily','audit','decisions','orders','holdings']:(OUT/directory).mkdir(exist_ok=True)
    phash=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest();(OUT/'protocol_sha256.txt').write_text(phash)
    f,pr=load();stop=int(f['dates'].searchsorted('2017-01-01'));dates=f['dates'][:stop];rets=[];rows=[]
    for i,s in enumerate(SPECS):
        rr=[]
        for phase in range(5):
            d=simulate(f,pr,s,start=f['start']+phase,stop=stop)[0];rr.append(d[:stop,1])
            for per,lo,hi in [('fit','2009-08-07','2013-12-31'),('validation','2014-01-01','2016-12-31')]:rows.append(dict(name=s['name'],phase=phase,period=per,**metrics(d,f['dates'],lo,hi)))
        rets.append(rr)
        if i%10==0 or i==len(SPECS)-1:log(f'Train {i+1}/{len(SPECS)} '+s['name'])
    rets=np.array(rets);np.savez_compressed(OUT/'training_returns.npz',returns=rets);pd.DataFrame(rows).to_csv(OUT/'training_phase_metrics.csv',index=False)
    scores=[];nameidx={s['name']:i for i,s in enumerate(SPECS)}
    for i,s in enumerate(SPECS[6:],6):
        ci=nameidx[control(s)];diff=np.mean(np.log1p(rets[i])-np.log1p(rets[ci]),axis=0)
        monthly=pd.Series(diff,index=dates).groupby(dates.to_period('M')).sum()
        fit=monthly.loc['2009-08':'2013-12'];val=monthly.loc['2014-01':'2016-12'];post=posterior_advantage(val)
        vr=[r['sharpe'] for r in rows if r['name']==s['name'] and r['period']=='validation']
        scores.append(dict(**s,H=MODELS[s['model']][0],W=MODELS[s['model']][1],fit_mean_excess=fit.mean(),validation_mean_sharpe=np.mean(vr),**post))
    tab=pd.DataFrame(scores);tab.to_csv(OUT/'parameter_search.csv',index=False)
    allowed=tab[tab.fit_mean_excess>0];qualified=len(allowed)>0
    if not qualified:allowed=tab
    order=allowed.sort_values(['lower10','validation_mean_sharpe','name'],ascending=[False,False,True]);selected=BYNAME[order.iloc[0]['name']]
    matched=order[(order.cap==selected['cap'])&(order.lam==selected['lam'])]
    if len(matched)<3:matched=tab[(tab.cap==selected['cap'])&(tab.lam==selected['lam'])].sort_values(['lower10','validation_mean_sharpe','name'],ascending=[False,False,True])
    sources=matched.head(3).name.tolist()
    selection=dict(selected=selected,H=MODELS[selected['model']][0],W=MODELS[selected['model']][1],control=control(selected),ensemble_sources=sources,fit_positive_qualified=qualified,validation_probability=float(order.iloc[0].p_positive),validation_lower10_monthly=float(order.iloc[0].lower10),uses_post2016=False,protocol_sha256=phash)
    (OUT/'selection.json').write_text(json.dumps(selection,indent=2));log('FROZEN '+json.dumps(selection))

def setting_weights(returns,dates,names,baseline,startdate='2017-01-01'):
    # returns[name] has5 phases byT; pool phase differences, not independent counts.
    month={n:pd.Series(np.mean(np.log1p(returns[n])-np.log1p(returns[baseline]),axis=0),index=dates).groupby(dates.to_period('M')).sum() for n in names}
    weights=np.zeros((len(dates),len(names)+1));weights[:,0]=1.;audit=[]
    for period in dates[dates>=startdate].to_period('M').unique():
        pp=[]
        for n in names:
            x=month[n][month[n].index<period].tail(36);post=posterior_advantage(x);pp.append(post['p_positive'])
            audit.append(dict(decision_month=str(period),source=n,last_complete_month=str(x.index[-1]),**post))
        aa=np.maximum(np.array(pp)-.5,0.);g=min(.5,2*aa.mean());w=np.r_[1-g,g*aa/aa.sum() if aa.sum()>0 else np.zeros(len(names))]
        weights[dates.to_period('M')==period]=w
    return weights,pd.DataFrame(audit)

def export(name,res,f,s=None):
    d,pos,tr,plans,probs,rec=res;pd.DataFrame(d,index=f['dates'],columns=COLS).rename_axis('date').to_csv(OUT/'daily'/f'{name}.csv')
    if pos.size==0:return
    np.savez_compressed(OUT/'audit'/f'{name}.npz',positions=pos,trades=tr,plans=plans,probabilities=probs,recoveries=rec)
    ti,ji=np.where(abs(tr)>1e-9);q=tr[ti,ji];notional=abs(q)*f['p'][ti,ji]
    orders=pd.DataFrame(dict(date=f['dates'][ti],decision_date=f['dates'][ti-1],ticker=f['codes'][ji],qty=q,adjusted_price=f['p'][ti,ji],notional=notional,commission=notional*f['comm'][ti],sale_tax=notional*f['tax'][ti]*(q<0),execution_cost=notional*.0005));orders.to_csv(OUT/'orders'/f'{name}.csv.gz',index=False)
    ti,ji=np.where(((plans>1e-10)|(pos>1e-9))&(d[:,13]>0)[:,None]);held=pos[ti,ji]*f['marks'][ti,ji]/d[ti,0];target=plans[ti,ji]
    action=np.where(target<1e-10,'EXIT',np.where(held<1e-10,'BUY_NEW',np.where(target>held+1e-10,'ADD',np.where(target<held-1e-10,'REDUCE','HOLD'))))
    de=pd.DataFrame(dict(date=f['dates'][ti],ticker=f['codes'][ji],action=action,held_weight=held,target_weight=target,relative_probability=probs[ti,ji],near=f['near'][ti,ji],rank=f['rank'][ti,ji]));de.to_csv(OUT/'decisions'/f'{name}.csv.gz',index=False)
    j=np.flatnonzero(pos[-1]>1e-9);pd.DataFrame(dict(ticker=f['codes'][j],adjusted_units=pos[-1,j],weight=pos[-1,j]*f['marks'][-1,j]/d[-1,0],near=f['near'][-1,j],rank=f['rank'][-1,j])).sort_values('weight',ascending=False).to_csv(OUT/'holdings'/f'{name}.csv',index=False)
    if name=='OPTIMIZED':de[de.date==de.date.max()].to_csv(OUT/'latest_review_actions.csv',index=False)

def evaluate():
    selection=json.loads((OUT/'selection.json').read_text());assert selection['protocol_sha256']==hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()
    f,pr=load();start=int(f['dates'].searchsorted('2017-01-02'));selected=selection['selected'];equal=BYNAME[selection['control']];default=BYNAME['EQ_C8_L3'];sources=selection['ensemble_sources'];names=[selection['control']]+sources
    shadow={};primaryplans={}
    for name in names:
        rr=[]
        for phase in range(5):
            d=simulate(f,pr,BYNAME[name],start=f['start']+phase)[0];rr.append(d[:,1])
        shadow[name]=np.array(rr)
    weights,audit=setting_weights(shadow,f['dates'],sources,selection['control']);audit.to_csv(OUT/'setting_posteriors.csv',index=False)
    pd.DataFrame(weights,index=f['dates'],columns=names).rename_axis('date').to_csv(OUT/'setting_weights.csv')
    np.savez_compressed(OUT/'source_shadow_returns.npz',**shadow)
    performance=[];phases=[];allres={}
    policies={'OPTIMIZED':selected,'MATCHED_EQUAL':equal,'DEFAULT_EQUAL':default}
    for phase in range(5):
        phaseplans={}
        for name in names:
            res=simulate(f,pr,BYNAME[name],start=start+phase,record=True);phaseplans[name]=res[3]
            if phase==0 and name==selection['control']:export('MATCHED_EQUAL',res,f);allres['MATCHED_EQUAL']=res[0]
            if phase==0 and name==selected['name']:export('OPTIMIZED',res,f);allres['OPTIMIZED']=res[0]
            if name==selection['control']:phases.append(dict(name='MATCHED_EQUAL',phase=phase,**metrics(res[0],f['dates'],'2022-01-01','2026-07-31')))
            if name==selected['name']:phases.append(dict(name='OPTIMIZED',phase=phase,**metrics(res[0],f['dates'],'2022-01-01','2026-07-31')))
        mixed=np.zeros_like(phaseplans[names[0]])
        for k,n in enumerate(names):mixed+=phaseplans[n]*weights[:,k,None]
        res=simulate(f,pr,equal,start=start+phase,record=phase==0,external=mixed);phases.append(dict(name='ADAPTIVE_MIX',phase=phase,**metrics(res[0],f['dates'],'2022-01-01','2026-07-31')))
        if phase==0:export('ADAPTIVE_MIX',res,f);allres['ADAPTIVE_MIX']=res[0];np.savez_compressed(OUT/'mixed_targets.npz',targets=mixed)
        if phase==0:
            res=simulate(f,pr,default,start=start,record=True);export('DEFAULT_EQUAL',res,f);allres['DEFAULT_EQUAL']=res[0]
        log(f'Evaluation phase{phase} complete')
    # Fair date/cost comparison against previous absolute-probability strategy.
    sys.path.insert(0,str(OUT/'legacy'))
    from run import simulate as legacy_simulate
    oldpred=dict(np.load(OUT/'legacy_prediction.npz'));res=legacy_simulate(f,{(20,1008):oldpred},13,start=start)
    od=res[0];d=np.zeros((len(f['dates']),len(COLS)));d[:,:12]=od[:,:12];allres['OLD_BAYES']=d;export('OLD_BAYES',(d,np.empty((0,0)),None,None,None,None),f)
    for name,d in allres.items():
        for per,lo,hi in [('test','2017-01-02','2021-12-31'),('recent','2022-01-01','2026-07-31'),('all_test','2017-01-02','2026-07-31')]:performance.append(dict(name=name,period=per,**metrics(d,f['dates'],lo,hi)))
    pd.DataFrame(performance).to_csv(OUT/'performance.csv',index=False);pd.DataFrame(phases).to_csv(OUT/'phase_sensitivity.csv',index=False)
    sensitivity=[]
    for name,s in [('OPTIMIZED',selected),('MATCHED_EQUAL',equal)]:
        for variant,slip,recovery in [('execution13bp',.0013,1.),('execution25bp',.0025,1.),('recovery0',.0005,0.)]:
            res=simulate(f,pr,s,start=start,slip=slip,recovery=recovery);sensitivity.append(dict(name=name,variant=variant,**metrics(res[0],f['dates'],'2022-01-01','2026-07-31')))
    pd.DataFrame(sensitivity).to_csv(OUT/'sensitivity.csv',index=False)
    (OUT/'data_manifest.json').write_text(json.dumps(dict(input_sha256=hashlib.sha256((OUT/'inputs.npz').read_bytes()).hexdigest(),data_end=str(f['dates'][-1].date()),account_start=str(f['dates'][start].date()),warmup_start=str(f['dates'][f['start']].date()),candidates=144,review_phases=5),indent=2))
    log('Evaluation complete')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['train','evaluate']);args=parser.parse_args()
    with threadpool_limits(limits=1):train() if args.command=='train' else evaluate()
