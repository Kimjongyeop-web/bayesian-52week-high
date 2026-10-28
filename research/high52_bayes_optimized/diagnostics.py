from study import *
from collections import deque

def paired(a,b,block,B=2000):
    rng=np.random.default_rng(4206);n=len(a);means=[];cagrs=[]
    for _ in range(B):
        starts=rng.integers(0,n,int(np.ceil(n/block)));ix=((starts[:,None]+np.arange(block))%n).ravel()[:n];x=a[ix];y=b[ix]
        means.append(252*(x.mean()-y.mean()));cagrs.append(np.expm1(np.log1p(x).mean()*252)-np.expm1(np.log1p(y).mean()*252))
    return dict(annual_arithmetic_excess=252*(a.mean()-b.mean()),mean_low=np.quantile(means,.025),mean_high=np.quantile(means,.975),annual252_compound_difference=np.expm1(np.log1p(a).mean()*252)-np.expm1(np.log1p(b).mean()*252),compound_low=np.quantile(cagrs,.025),compound_high=np.quantile(cagrs,.975))

def main():
    f,pr=load();dates=f['dates'];accounts={n:pd.read_csv(OUT/'daily'/f'{n}.csv',index_col=0,parse_dates=True) for n in ['OPTIMIZED','MATCHED_EQUAL','DEFAULT_EQUAL','ADAPTIVE_MIX','OLD_BAYES']};rows=[]
    for a,b in [('OPTIMIZED','MATCHED_EQUAL'),('ADAPTIVE_MIX','MATCHED_EQUAL'),('OPTIMIZED','DEFAULT_EQUAL')]:
        x=accounts[a].loc['2022-01-01':].net_return.to_numpy();y=accounts[b].loc['2022-01-01':].net_return.to_numpy()
        for block in [20,60]:rows.append(dict(treatment=a,control=b,block=block,**paired(x,y,block)))
    pd.DataFrame(rows).to_csv(OUT/'paired_inference.csv',index=False)
    annual=[];holding=[];actions=[];recoveries=[]
    for name,d in accounts.items():
        for year in range(2017,2027):
            x=d.loc[f'{year}-01-01':f'{year}-12-31'];annual.append(dict(name=name,year=year,total_return=np.prod(1+x.net_return)-1,exposure=x.exposure.mean(),annual_cost_rate=np.mean(x[['commission','sale_tax','execution_cost']].sum(1)/x.nav)*252))
        ap=OUT/'audit'/f'{name}.npz'
        if not ap.exists():continue
        a=np.load(ap);tr=a['trades'];rec=a['recoveries'];queues=[deque() for _ in f['codes']];ages=[];vals=[]
        for t in range(len(dates)):
            for j in np.flatnonzero(rec[t]>0):
                value=rec[t,j]*f['marks'][t,j];recoveries.append(dict(name=name,date=str(dates[t].date()),ticker=f['codes'][j],assumed_recovery=value,nav_fraction=value/d.nav.iloc[t],last_mark=f['marks'][t,j]))
            for j in np.flatnonzero((abs(tr[t])>1e-9)|(rec[t]>0)):
                if rec[t,j]>0:queues[j].clear()
                q=tr[t,j]
                if q>1e-9:queues[j].append([t,q])
                elif q<-1e-9:
                    left=-q
                    while left>1e-7 and queues[j]:
                        origin,amount=queues[j][0];used=min(left,amount)
                        if dates[t]>=pd.Timestamp('2022-01-01'):ages.append(t-origin);vals.append(used*f['p'][t,j])
                        left-=used;queues[j][0][1]-=used
                        if queues[j][0][1]<1e-7:queues[j].popleft()
        ages=np.array(ages);vals=np.array(vals);holding.append(dict(name=name,closed_fifo_pieces=len(ages),weighted_mean_sessions=np.average(ages,weights=vals),median_piece_sessions=np.median(ages),sale_value_held_over5=vals[ages>5].sum()/vals.sum(),sale_value_held_over20=vals[ages>20].sum()/vals.sum(),open_positions=np.sum(a['positions'][-1]>1e-9)))
        de=pd.read_csv(OUT/'decisions'/f'{name}.csv.gz');de=de[pd.to_datetime(de.date)>='2022-01-01']
        for action,count in de.action.value_counts().items():actions.append(dict(name=name,action=action,count=count))
    pd.DataFrame(annual).to_csv(OUT/'annual_results.csv',index=False);pd.DataFrame(holding).to_csv(OUT/'holding_duration.csv',index=False);pd.DataFrame(actions).to_csv(OUT/'action_counts.csv',index=False);pd.DataFrame(recoveries).to_csv(OUT/'recovery_events.csv',index=False)
    phases=pd.read_csv(OUT/'phase_sensitivity.csv');pivot=phases.pivot(index='phase',columns='name',values='cagr');pivot['OPTIMIZED_minus_EQUAL']=pivot.OPTIMIZED-pivot.MATCHED_EQUAL;pivot['MIX_minus_EQUAL']=pivot.ADAPTIVE_MIX-pivot.MATCHED_EQUAL;pivot.to_csv(OUT/'phase_paired.csv')
    log('Diagnostics complete')

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
