from study import *
import hashlib

PREVIOUS=dict(name='B_H20_W1008_P55_S1.0_C10_L3',model=3,p0=.55,strength=1.,cap=.1,lam=3.)

def main():
    for directory in ['daily','audit','orders','decisions','holdings']:(OUT/directory).mkdir(exist_ok=True)
    (OUT/'protocol_sha256.txt').write_text(hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest())
    f,pr=load();stop=int(f['dates'].searchsorted('2017-01-01'));dates=f['dates'][:stop];returns={};rows=[]
    for i,s in enumerate(SPECS):
        d=simulate(f,pr,s,stop=stop)[0];returns[s['name']]=d[:stop,1]
        for per,lo,hi in [('fit','2009-08-07','2013-12-31'),('validation','2014-01-01','2016-12-31')]:rows.append(dict(name=s['name'],period=per,**metrics(d,f['dates'],lo,hi)))
        if i%25==0:log(f'Daily search {i+1}/150')
    pd.DataFrame(rows).to_csv(OUT/'training_metrics.csv',index=False);scores=[]
    for s in SPECS[6:]:
        diff=np.log1p(returns[s['name']])-np.log1p(returns[control(s)]);monthly=pd.Series(diff,index=dates).groupby(dates.to_period('M')).sum();fit=monthly.loc['2009-08':'2013-12'];val=monthly.loc['2014-01':'2016-12']
        sharpe=next(r['sharpe'] for r in rows if r['name']==s['name'] and r['period']=='validation')
        scores.append(dict(**s,fit_mean_excess=fit.mean(),validation_sharpe=sharpe,**posterior_advantage(val)))
    tab=pd.DataFrame(scores);tab.to_csv(OUT/'parameter_search.csv',index=False);eligible=tab[tab.fit_mean_excess>0];qualified=len(eligible)>0
    if not qualified:eligible=tab
    best=eligible.sort_values(['lower10','validation_sharpe','name'],ascending=[False,False,True]).iloc[0];selected=BYNAME[best['name']]
    selection=dict(selected=selected,H=MODELS[selected['model']][0],W=MODELS[selected['model']][1],fit_qualified=qualified,qualified_count=int((tab.fit_mean_excess>0).sum()),p_positive=float(best.p_positive),lower10=float(best.lower10),uses_post2016=False)
    (OUT/'selection.json').write_text(json.dumps(selection,indent=2));log('FROZEN '+json.dumps(selection))
    start=int(f['dates'].searchsorted('2017-01-02'));policies=[('PREVIOUS5',PREVIOUS,5),('DAILY_SAME',PREVIOUS,1),('DAILY_EQUAL',BYNAME[control(PREVIOUS)],1),('DAILY_TUNED',selected,1),('TUNED_EQUAL',BYNAME[control(selected)],1)];perf=[];annual=[];activity=[];sens=[]
    for name,s,interval in policies:
        res=simulate(f,pr,s,start=start,record=True,interval=interval);export(name,res,f);d=res[0]
        for per,lo,hi in [('test','2017-01-02','2021-12-31'),('recent','2022-01-01','2026-07-31'),('all_test','2017-01-02','2026-07-31')]:perf.append(dict(name=name,period=per,**metrics(d,f['dates'],lo,hi)))
        for year in range(2017,2027):
            ix=f['dates'].year==year;x=d[ix];annual.append(dict(name=name,year=year,total_return=np.prod(1+x[:,1])-1,exposure=x[:,2].mean()))
        ix=f['dates']>='2022-01-01';tr=res[2][ix];plans=res[3][ix];pos=res[1][ix];review=d[ix,13]>0
        act=abs(tr)>1e-7;dec=pd.read_csv(OUT/'decisions'/f'{name}.csv.gz');dec=dec[pd.to_datetime(dec.date)>='2022-01-01']
        activity.append(dict(name=name,trading_days=int(ix.sum()),review_days=int(review.sum()),days_with_fills=int(act.any(1).sum()),stock_fills=int(act.sum()),hold_decisions=int((dec.action=='HOLD').sum()),add_decisions=int((dec.action=='ADD').sum()),reduce_decisions=int((dec.action=='REDUCE').sum()),entry_decisions=int((dec.action=='BUY_NEW').sum()),exit_decisions=int((dec.action=='EXIT').sum()),last_cash_fraction=float(d[-1,8]/d[-1,0])))
        if name=='DAILY_TUNED':
            all_dec=pd.read_csv(OUT/'decisions'/f'{name}.csv.gz',dtype={'ticker':str});all_dec[all_dec.date==all_dec.date.max()].to_csv(OUT/'latest_close_decisions.csv',index=False)
        if name in ['DAILY_SAME','DAILY_TUNED','TUNED_EQUAL']:
            for variant,slip,recovery in [('execution13bp',.0013,1.),('execution25bp',.0025,1.),('recovery0',.0005,0.)]:
                alt=simulate(f,pr,s,start=start,slip=slip,recovery=recovery)[0];sens.append(dict(name=name,variant=variant,**metrics(alt,f['dates'],'2022-01-01','2026-07-31')))
        log(name+' complete')
    pd.DataFrame(perf).to_csv(OUT/'performance.csv',index=False);pd.DataFrame(annual).to_csv(OUT/'annual_results.csv',index=False);pd.DataFrame(activity).to_csv(OUT/'activity.csv',index=False);pd.DataFrame(sens).to_csv(OUT/'sensitivity.csv',index=False)
    (OUT/'data_manifest.json').write_text(json.dumps(dict(data_end=str(f['dates'][-1].date()),account_start=str(f['dates'][start].date()),input_sha256=hashlib.sha256((OUT/'inputs.npz').read_bytes()).hexdigest(),daily_cadence=1),indent=2));log('Daily rerun complete')

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
