from model import *
from run import simulate,metrics,rolling
from collections import deque
from threadpoolctl import threadpool_limits
import importlib.util

def boot(a,b,block,B=2000):
    rng=np.random.default_rng(685);n=len(a);md=[];sd=[]
    for _ in range(B):
        st=rng.integers(0,n,size=int(np.ceil(n/block)));ix=((st[:,None]+np.arange(block))%n).ravel()[:n];x=a[ix];y=b[ix]
        md.append((x.mean()-y.mean())*252);sd.append((x.mean()/x.std(ddof=1)-y.mean()/y.std(ddof=1))*np.sqrt(252))
    return dict(annual_mean_difference=(a.mean()-b.mean())*252,mean_low=np.quantile(md,.025),mean_high=np.quantile(md,.975),sharpe_difference=(a.mean()/a.std(ddof=1)-b.mean()/b.std(ddof=1))*np.sqrt(252),sharpe_low=np.quantile(sd,.025),sharpe_high=np.quantile(sd,.975))

def main():
    f=prepare();info=json.loads((OUT/'data_manifest.json').read_text());f['start']=info['start_i'];dates=f['dates'];pick=json.loads((OUT/'selection.json').read_text());i=pick['selected_i'];s=pick['selected'];preds={(h,w):dict(np.load(OUT/'predictions'/f'H{h}_W{w}.npz')) for h in [5,20] for w in [504,1008]}
    pred=preds[s['H'],s['W']];pm,pk,pw=probabilities(pred,f);y=labels(f['p'],f['marks'],f['v'],f['m'],s['H']);fee=2*(f['comm']+.0005)+f['tax'];success=y>fee[:,None]
    eligible=f['eligible']&(f['near']>=.90)&(f['rank']<=.10)&np.isfinite(y)&np.isfinite(pw)
    # Conditional realized-return probabilities, not latent-mean probabilities, are calibrated.
    calibration=[];bins=[]
    for per,lo,hi in [('development',info['common_start'],'2016-12-31'),('test','2017-01-01','2021-12-31'),('deployment','2022-01-01','2026-07-31')]:
        mask=eligible&((dates>=lo)&(dates<=hi))[:,None];a=pw[mask];b=success[mask];calibration.append(dict(period=per,observations=len(a),mean_predicted=a.mean(),realized_frequency=b.mean(),brier=np.mean((a-b)**2),brier_constant50=.25,entry_gate_pass_fraction=np.mean(pm[mask]>=s['prob'])))
        for low in np.arange(0,1,.05):
            use=mask&(pw>=low)&(pw<low+.05)
            if use.sum():bins.append(dict(period=per,bin_low=low,bin_high=low+.05,count=int(use.sum()),predicted=float(pw[use].mean()),realized=float(success[use].mean())))
    pd.DataFrame(calibration).to_csv(OUT/'probability_calibration.csv',index=False);pd.DataFrame(bins).to_csv(OUT/'probability_bins.csv',index=False)
    entries=[]
    for year in range(2009,2027):
        mask=eligible&(dates.year==year)[:,None]
        if mask.any():entries.append(dict(year=year,n=int(mask.sum()),mean_expected_return=float(pred['mu'][mask].mean()),mean_p_expected=float(pm[mask].mean()),gate_pass_fraction=float((pm[mask]>=s['prob']).mean())))
    pd.DataFrame(entries).to_csv(OUT/'entry_gate_by_year.csv',index=False)
    holds=[];actions=[];yearrows=[];performance={}
    for name in ['FIXED5','FIXED120','BUFFER','BUFFER_BAND','BAYES','BAYES_NOBAND','ROLL_BAYES','KOSPI200']:
        d=pd.read_csv(OUT/'daily'/f'{name}.csv',index_col=0,parse_dates=True);performance[name]=d
        for year in range(2017,2027):
            x=d.loc[f'{year}-01-01':f'{year}-12-31'];r=x.net_return.to_numpy();w=np.r_[1,np.cumprod(1+r)]
            yearrows.append(dict(name=name,year=year,return_value=w[-1]-1,exposure=x.exposure.mean(),cost=np.mean(x[['commission','sale_tax','execution_cost']].sum(1)/x.nav)*252))
        op=OUT/'orders'/f'{name}.csv.gz'
        if not op.exists():continue
        orders=pd.read_csv(op);queues={j:deque() for j in range(len(f['codes']))};ar=np.load(OUT/'audit'/f'{name}.npz');rec=ar['recoveries'];events=[]
        for r in orders.itertuples():events.append((int(r.execution_i),int(r.stock_i),float(r.qty),float(r.price),'TRADE'))
        for t,j in zip(*np.where(rec>1e-9)):events.append((int(t),int(j),-float(rec[t,j]),float(f['marks'][t,j]),'RECOVERY'))
        events.sort(key=lambda x:(x[0],x[4]!='RECOVERY'))
        durations=[];values=[]
        for t,j,q,price,kind in events:
            queue=queues[j]
            if q>0:queue.append([t,q])
            else:
                left=-q
                while left>1e-7 and queue:
                    origin,amount=queue[0];used=min(left,amount)
                    if dates[t]>=pd.Timestamp('2022-01-01'):
                        durations.append(t-origin);values.append(used*price)
                    left-=used;queue[0][1]-=used
                    if queue[0][1]<1e-7:queue.popleft()
        ages=np.array(durations);weights=np.array(values)
        holds.append(dict(name=name,closed_fifo_pieces=len(ages),weighted_mean_sessions=float(np.average(ages,weights=weights)) if len(ages) else np.nan,median_piece_sessions=float(np.median(ages)) if len(ages) else np.nan,share_sale_notional_held_over5=float(weights[ages>5].sum()/weights.sum()) if weights.sum() else np.nan,share_sale_notional_held_over20=float(weights[ages>20].sum()/weights.sum()) if weights.sum() else np.nan,open_positions=int((ar['positions'][-1]>1e-9).sum())))
        de=pd.read_csv(OUT/'decisions'/f'{name}.csv.gz');de=de[pd.to_datetime(de.date)>='2022-01-01']
        for action,n in de.action.value_counts().items():actions.append(dict(name=name,action=action,decisions=int(n)))
    pd.DataFrame(holds).to_csv(OUT/'holding_duration.csv',index=False);pd.DataFrame(actions).to_csv(OUT/'action_counts.csv',index=False);pd.DataFrame(yearrows).to_csv(OUT/'annual_results.csv',index=False)
    inference=[]
    for a,b in [('BUFFER_BAND','FIXED5'),('BAYES','BUFFER_BAND'),('BAYES','FIXED5'),('BAYES','BAYES_NOBAND')]:
        aa=performance[a].loc['2022-01-01':].net_return.to_numpy();bb=performance[b].loc['2022-01-01':].net_return.to_numpy()
        for block in [20,60]:inference.append(dict(treatment=a,control=b,block=block,**boot(aa,bb,block)))
    pd.DataFrame(inference).to_csv(OUT/'paired_inference.csv',index=False)
    # Calendar-phase robustness, fixed parameters; do not select best phase.
    import fixed
    fw=fixed.weights(f);phase=[]
    for offset in range(5):
        start=f['start']+offset
        for name,mode,band in [('BUFFER_BAND','buffer',True),('BAYES','bayes',True)]:
            d=simulate(f,preds,i,mode,band,start=start)[0];phase.append(dict(name=name,offset=offset,**metrics(d,dates,'2022-01-01','2026-07-31')))
        fr=fixed.run(f,5,start,fw)[0]
        if offset==0:np.testing.assert_allclose(fr[:,0],performance['FIXED5'].nav,rtol=1e-10,atol=.001)
        d=np.zeros((len(dates),22));d[:,:12]=fr[:,:12];phase.append(dict(name='FIXED5',offset=offset,**metrics(d,dates,'2022-01-01','2026-07-31')))
    pd.DataFrame(phase).to_csv(OUT/'phase_sensitivity.csv',index=False)
    # Daily posterior diagnostic on current holdings; latest scheduled review actions stay separate.
    h=pd.read_csv(OUT/'holdings/BAYES.csv',dtype={'ticker':str});lookup={str(c):j for j,c in enumerate(f['codes'])};js=[lookup[str(c).zfill(6)] for c in h.ticker]
    h['asof_date']=str(dates[-1].date());h['expected_H_return']=pred['mu'][-1,js];h['p_expected_net']=pm[-1,js];h['p_realized_net']=pw[-1,js];h['forecast_H']=s['H'];h.to_csv(OUT/'latest_portfolio.csv',index=False)
    de=pd.read_csv(OUT/'decisions/BAYES.csv.gz',dtype={'ticker':str});de[de.decision_i==de.decision_i.max()].to_csv(OUT/'latest_review_actions.csv',index=False)
    log('Diagnostics complete')

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
