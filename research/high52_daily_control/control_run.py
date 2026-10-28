from study import *
from itertools import product

MODEL=dict(name='B_H5_W1008_P55_S0.5_C10_L3',model=1,p0=.55,strength=.5,cap=.1,lam=3.)
DEFAULT=dict(name='N88_F3_C5_B10_W50',exit_near=.88,exit_rank=.25,confirm=3,cooldown=5,band_abs=.01,band_rel=.2,day_limit=.1,window_limit=.5)
RULES=[dict(name=f'N{int(near*100)}_F{confirm}_C{cool}_B{int(band*1000)}_W{int(window*100)}',exit_near=near,exit_rank=.25,confirm=confirm,cooldown=cool,band_abs=band,band_rel=.2,day_limit=.1,window_limit=window) for near,confirm,cool,band,window in product([.88,.85],[2,3],[3,5],[.005,.01],[.5,1.])]

def run(f,pr,rule,model=None,start=None,stop=None,record=False,slip=.0005,recovery=1.):
    s=MODEL if model is None else model;start=f['start'] if start is None else start;stop=len(f['dates']) if stop is None else stop;p=pr[s['model']];h,w=MODELS[s['model']]
    args=[f[k] for k in ['p','marks','v','m','eligible','near','rank','X','sigma','tax','comm']]
    rr=[rule[k] for k in ['exit_near','exit_rank','confirm','cooldown','band_abs','band_rel','day_limit','window_limit']]
    return run_kernel(*args,p['coef'],p['cov'],p['df'],h,s['p0'],s['strength'],s['cap'],s['lam'],start,stop,slip,recovery,record,None,1,*rr)

def main():
    for p in ['daily','audit','orders','decisions','holdings','budgets']:(OUT/p).mkdir(exist_ok=True)
    (OUT/'protocol_sha256.txt').write_text(hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest())
    f,pr=load();stop=int(f['dates'].searchsorted('2017-01-01'));rows=[]
    # Daily no-control benchmark, from the fully validated preceding revision.
    baseline_rule=dict(name='UNCONTROLLED',exit_near=.9,exit_rank=.2,confirm=1,cooldown=0,band_abs=0.,band_rel=0.,day_limit=1e6,window_limit=1e6)
    base=run(f,pr,baseline_rule,stop=stop)[0];bm=metrics(base,f['dates'],'2014-01-01','2016-12-31')
    for i,r in enumerate(RULES):
        d=run(f,pr,r,stop=stop)[0]
        row=dict(**r)
        for per,lo,hi in [('fit','2009-08-07','2013-12-31'),('validation','2014-01-01','2016-12-31')]:row.update({per+'_'+k:v for k,v in metrics(d,f['dates'],lo,hi).items()})
        rows.append(row)
        if i%8==0:log(f'Control fit {i+1}/32')
    tab=pd.DataFrame(rows);tab['fit_eligible']=(tab.fit_cagr>0)&(tab.fit_annual_turnover<=15)
    tab['validation_eligible']=(tab.validation_cagr>0)&(tab.validation_annual_turnover<=15)&(tab.validation_mdd>=bm['mdd']-.05)
    tab.to_csv(OUT/'control_search.csv',index=False)
    shortlist=tab[tab.fit_eligible].sort_values(['fit_sharpe','fit_annual_turnover','name'],ascending=[False,True,True]).head(5);passing=shortlist[shortlist.validation_eligible];qualified=bool(len(passing))
    if not qualified:
        passing=tab[(tab.fit_annual_turnover<=15)&(tab.validation_annual_turnover<=15)]
        if not len(passing):passing=tab
    best=passing.sort_values(['validation_sharpe','validation_annual_turnover','name'],ascending=[False,True,True]).iloc[0];chosen=next(r for r in RULES if r['name']==best['name'])
    selection=dict(selected=chosen,model=MODEL,qualified=qualified,fit_pass_count=int(tab.fit_eligible.sum()),validation_pass_count=int(tab.validation_eligible.sum()),shortlist=shortlist.name.tolist(),uses_post2016=False)
    (OUT/'selection.json').write_text(json.dumps(selection,indent=2));log('FROZEN '+json.dumps(selection))
    start=int(f['dates'].searchsorted('2017-01-02'));policies=[('UNCONTROLLED',baseline_rule,MODEL),('CONTROLLED',chosen,MODEL),('CONTROLLED_EQUAL',chosen,dict(MODEL,strength=0.)),('DEFAULT_RULES',DEFAULT,MODEL)];perf=[];activity=[];sens=[];annual=[]
    for name,r,s in policies:
        res=run(f,pr,r,s,start=start,record=True);export(name,res[:6],f);pd.DataFrame(res[6],index=f['dates'],columns=['ordinary_turnover','emergency_turnover','initial_turnover','rolling20_ordinary','ordinary_fill_scale']).rename_axis('date').to_csv(OUT/'budgets'/f'{name}.csv')
        d=res[0];ix=f['dates']>='2022-01-01';bud=res[6][ix];tr=res[2][ix]
        for per,lo,hi in [('test','2017-01-02','2021-12-31'),('recent','2022-01-01','2026-07-31'),('all_test','2017-01-02','2026-07-31')]:perf.append(dict(name=name,period=per,**metrics(d,f['dates'],lo,hi)))
        dec=pd.read_csv(OUT/'decisions'/f'{name}.csv.gz',dtype={'ticker':str});dc=dec[pd.to_datetime(dec.date)>='2022-01-01']
        activity.append(dict(name=name,days=int(ix.sum()),review_days=int(d[ix,13].sum()),fill_days=int((abs(tr)>1e-7).any(1).sum()),stock_fills=int((abs(tr)>1e-7).sum()),hold_decisions=int((dc.action=='HOLD').sum()),exit_decisions=int((dc.action=='EXIT').sum()),limit_binding_days=int((bud[:,4]<1.-1e-10).sum()),max_ordinary_day=float(bud[:,0].max()),max_ordinary20=float(bud[:,3].max()),annual_ordinary=bud[:,0].mean()*252,annual_emergency=bud[:,1].mean()*252,last_cash=float(d[-1,8]/d[-1,0])))
        for year in range(2017,2027):
            y=d[f['dates'].year==year];annual.append(dict(name=name,year=year,total_return=np.prod(1+y[:,1])-1,exposure=y[:,2].mean()))
        if name=='CONTROLLED':dec[dec.date==dec.date.max()].to_csv(OUT/'latest_close_decisions.csv',index=False)
        if name in ['CONTROLLED','CONTROLLED_EQUAL']:
            for variant,slip,recovery in [('execution13bp',.0013,1.),('execution25bp',.0025,1.),('recovery0',.0005,0.)]:
                alt=run(f,pr,r,s,start=start,slip=slip,recovery=recovery)[0];sens.append(dict(name=name,variant=variant,**metrics(alt,f['dates'],'2022-01-01','2026-07-31')))
        log(name+' evaluated')
    pd.DataFrame(perf).to_csv(OUT/'performance.csv',index=False);pd.DataFrame(activity).to_csv(OUT/'activity.csv',index=False);pd.DataFrame(sens).to_csv(OUT/'sensitivity.csv',index=False);pd.DataFrame(annual).to_csv(OUT/'annual_results.csv',index=False)
    (OUT/'data_manifest.json').write_text(json.dumps(dict(data_end=str(f['dates'][-1].date()),start=str(f['dates'][start].date()),input_sha256=hashlib.sha256((OUT/'inputs.npz').read_bytes()).hexdigest()),indent=2));log('Control study complete')

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
