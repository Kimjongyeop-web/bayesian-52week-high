from pathlib import Path
import json, hashlib, shutil, time
import numpy as np
import pandas as pd
from numba import njit

OUT=Path(__file__).resolve().parent
SOURCE=OUT/'portable_inputs/reference'
if (OUT/'portable_inputs/reference').exists():SOURCE=OUT/'portable_inputs/reference'
FAMILIES=['NEAR10','NEAR20','BREAK20','MOM10','MOM20','JOINT25','JOINT50','JOINT75']
HS=[5,10,20,40,60,120]
SPECS=[dict(name=f'{f}_H{h}_{w}',family=f,fi=i,H=h,weight=w,wi=j) for i,f in enumerate(FAMILIES) for h in HS for j,w in enumerate(['EW','IV60'])]
COLS=['nav','net_return','exposure','turnover','commission','sale_tax','slippage_impact','writeoff','cash','buy_notional','sell_notional','gross_return','new_cohort','H_new','n_holdings','unfilled','expired_notional','expired_weighted_age','max_weight','ledger_error']
def log(x): print(time.strftime('%H:%M:%S'),x,flush=True)

def prepare(raw=None):
    if raw is None: raw=dict(np.load(OUT/'inputs.npz',allow_pickle=False))
    f=raw.copy();dates=pd.DatetimeIndex(f['dates']);p=pd.DataFrame(np.where(f['p']>0,f['p'],np.nan));v=pd.DataFrame(f['v'])
    lp=np.log(p);r=lp.diff();sigma=r.rolling(60).std().shift(1).clip(lower=.005)
    bad=(r.abs()>.35)&(r.abs()>12*sigma);quarantine=bad.rolling(10,min_periods=1).max().fillna(0).astype(bool)
    full=p.notna().rolling(253).sum().eq(253)
    eligible=pd.DataFrame(f['m'])&v.gt(0)&p.gt(0)&full&sigma.notna()&~quarantine
    near=p/p.rolling(252).max();momentum=lp.shift(21)-lp.shift(252)
    nr=near.where(eligible).rank(axis=1,pct=True,method='average');mr=momentum.where(eligible).rank(axis=1,pct=True,method='average')
    breaks=p.gt(p.rolling(252).max().shift(1));recent=breaks.rolling(5,min_periods=1).max().fillna(0).astype(bool)
    weights=np.zeros((16,len(p),p.shape[1]),np.float32)
    for fi,fam in enumerate(FAMILIES):
        q=.1 if fam in ['NEAR10','MOM10'] else .2
        if fam.startswith('NEAR'): score=near.to_numpy();gate=(eligible&near.ge(.90)).to_numpy()
        elif fam=='BREAK20': score=(near+mr.fillna(0)*1e-7).to_numpy();gate=(eligible&recent&near.ge(.98)).to_numpy()
        elif fam.startswith('MOM'): score=momentum.to_numpy();gate=(eligible&momentum.gt(0)).to_numpy()
        else:
            a=int(fam[5:])/100;score=(a*nr+(1-a)*mr).to_numpy();gate=(eligible&momentum.gt(0)).to_numpy()
        for t in range(len(p)):
            cap=max(1,int(np.floor(eligible.iloc[t].sum()*q)));ids=np.flatnonzero(gate[t])
            ids=ids[np.argsort(-score[t,ids],kind='stable')[:cap]]
            if len(ids):
                weights[2*fi,t,ids]=1/len(ids)
                x=1/sigma.iloc[t,ids].to_numpy();weights[2*fi+1,t,ids]=x/x.sum()
    settlement=pd.DatetimeIndex(f['settlement']);tax=np.full(len(p),.003)
    for d,rate in [('2019-06-03',.0025),('2021-01-01',.0023),('2023-01-01',.002),('2024-01-01',.0018),('2025-01-01',.0015),('2026-01-01',.002)]:tax[settlement>=d]=rate
    commission=np.full(len(p),.00015);commission[(dates>='2020-09-16')&(dates<='2020-12-31')]=.000113604
    f.update(dates=dates,p=p.to_numpy(),marks=p.ffill().fillna(0).to_numpy(),weights=weights,near=near.to_numpy(),momentum=momentum.to_numpy(),eligible=eligible.to_numpy(),sigma=sigma.to_numpy(),quarantine=quarantine.to_numpy(),tax=tax,commission=commission)
    return f

@njit(cache=True)
def run_account(p,marks,v,member,weights,tax,comm,sched,his,wis,start,stop,slip=.0005,recovery=1.,record=False):
    T,N=p.shape;slots=64;lots=np.zeros((slots,N));ex=np.full(slots,-1);ent=np.full(slots,-1)
    qty=np.zeros(N);new=np.zeros(N);mature=np.zeros((slots,N));cash=1e7;prev=1e7
    new_h=0;decision_i=-1;pending_flag=0
    daily=np.zeros((T,20));daily[:,0]=1e7;daily[:,8]=1e7
    pos=np.zeros((T,N)) if record else np.zeros((1,1));orders=np.zeros((T,N)) if record else np.zeros((1,1))
    recoveries=np.zeros((T,N)) if record else np.zeros((1,1))
    for t in range(start,stop):
        before=cash+np.dot(qty,marks[t]);writeoff=0.
        for j in range(N):
            if qty[j]>0 and not np.isfinite(p[t,j]) and t>0 and not member[t-1,j]:
                value=qty[j]*marks[t,j];cash+=value*recovery;writeoff+=value*(1-recovery)
                if record:recoveries[t,j]=qty[j]
                qty[j]=0.;new[j]=0.
                for k in range(slots):lots[k,j]=0.;mature[k,j]=0.
        old=qty.copy();out=np.zeros(N);inn=new.copy();cb=comm[t]+slip;cs=cb+tax[t];unfilled=0
        for j in range(N):
            for k in range(slots):out[j]+=mature[k,j]
            if not (np.isfinite(p[t,j]) and p[t,j]>0 and np.isfinite(v[t,j]) and v[t,j]>0):
                if out[j]>0 or inn[j]>0:unfilled+=1
                out[j]=0.;inn[j]=0.
            elif not member[t,j]:inn[j]=0.
        # At execution only shrink previously decided purchase orders to funded size.
        lo=1.;hi=1.;cash_after=cash
        for j in range(N):
            netj=inn[j]-out[j]
            if netj>0:cash_after-=netj*p[t,j]*(1+cb)
            elif netj<0:cash_after-=netj*p[t,j]*(1-cs)
        if cash_after<0:
            lo=0.
            for iteration in range(45):
                scale=(lo+hi)/2;net=scale*inn-out;cash_after=cash
                for j in range(N):
                    if net[j]>0:cash_after-=net[j]*p[t,j]*(1+cb)
                    elif net[j]<0:cash_after-=net[j]*p[t,j]*(1-cs)
                if cash_after>=0:lo=scale
                else:hi=scale
        inn*=lo;net=inn-out;buys=0.;sells=0.;expired_value=0.;expired_age=0.
        for j in range(N):
            if out[j]>0:
                for k in range(slots):
                    q=mature[k,j];lots[k,j]-=q
                    if q>0:expired_value+=q*p[t,j];expired_age+=q*p[t,j]*(t-ent[k])
            if net[j]>0:buys+=net[j]*p[t,j]
            elif net[j]<0:sells-=net[j]*p[t,j]
        cash+=sells*(1-cs)-buys*(1+cb);qty+=net
        empty=-1
        for k in range(slots):
            if np.sum(lots[k])<1e-9:
                lots[k,:]=0.;ex[k]=-1;ent[k]=-1
                if empty<0:empty=k
        if np.sum(inn)>0:
            assert empty>=0
            lots[empty,:]=inn;ex[empty]=t+new_h;ent[empty]=t
        fees_c=(buys+sells)*comm[t];fees_t=sells*tax[t];fees_s=(buys+sells)*slip
        nav=cash+np.dot(qty,marks[t]);err=abs(nav-(before-fees_c-fees_t-fees_s-writeoff))
        assert cash>-.001 and nav>0 and np.min(qty)>-1e-7 and err<.01
        held=qty*marks[t]/nav
        daily[t]=np.array([nav,nav/prev-1,np.sum(held),(buys+sells)/before,fees_c,fees_t,fees_s,writeoff,cash,buys,sells,before/prev-1,pending_flag,new_h if pending_flag else 0,np.sum(qty>1e-9),unfilled,expired_value,expired_age,np.max(held),err])
        if record:pos[t]=qty;orders[t]=net
        prev=nav
        # Decision for next day's close, using only t or earlier.
        mature[:,:]=0.
        for k in range(slots):
            if ex[k]<0:continue
            for j in range(N):
                if ex[k]<=t+1 or not member[t,j]:mature[k,j]=lots[k,j]
        new[:]=0.;pending_flag=0;pol=sched[t]
        if (t-start)%5==0 and pol>=0:
            h=his[pol];wi=wis[pol];totalw=np.sum(weights[wi,t]);new_h=h;pending_flag=1
            if totalw>0:
                remaining=qty-np.sum(mature,axis=0);budget=nav*5/h
                for j in range(N):
                    if weights[wi,t,j]>0 and marks[t,j]>0:
                        value=min(budget*weights[wi,t,j],max(0.,.08*nav-remaining[j]*marks[t,j]))
                        new[j]=value/marks[t,j]
        decision_i=t
    return daily,pos,orders,recoveries

def run(f,schedule,start='2009-01-09',stop=None,slip=.0005,recovery=1.,record=False,zero=False):
    end=len(f['dates']) if stop is None else stop
    return run_account(f['p'],f['marks'],f['v'],f['m'],f['weights'],f['tax']*(not zero),f['commission']*(not zero),np.asarray(schedule,np.int64),np.array([s['H'] for s in SPECS]),np.array([2*s['fi']+s['wi'] for s in SPECS]),int(f['dates'].searchsorted(start)),end,0. if zero else slip,recovery,record)

def metrics(d,dates,lo,hi):
    ix=(dates>=lo)&(dates<=hi);x=d[ix];r=x[:,1];nav=np.r_[1,np.cumprod(1+r)]
    yrs=(dates[ix][-1]-dates[ix][0]).days/365.25
    return dict(total_return=nav[-1]-1,cagr=nav[-1]**(1/yrs)-1,sharpe=r.mean()/r.std(ddof=1)*np.sqrt(252) if r.std()>0 else 0.,mdd=np.min(nav/np.maximum.accumulate(nav)-1),vol=r.std(ddof=1)*np.sqrt(252),exposure=x[:,2].mean(),annual_turnover=x[:,3].mean()*252,cost_ratio=np.mean(x[:,4:7].sum(1)/x[:,0])*252,commission=x[:,4].sum(),sale_tax=x[:,5].sum(),slippage_impact=x[:,6].sum(),writeoff=x[:,7].sum(),weighted_lot_hold=x[:,17].sum()/max(x[:,16].sum(),1e-12),max_weight=x[:,18].max())

def rolling(returns,ids,dates,start='2017-01-01'):
    T=len(dates);sched=np.full(T,-1,np.int64);rows=[];current=-1;challenger=-99;streak=0
    y=returns[:,ids];cs=np.vstack([np.zeros(len(ids)),np.cumsum(y,axis=0)]);ss=np.vstack([np.zeros(len(ids)),np.cumsum(y*y,axis=0)])
    for t in range(int(dates.searchsorted(start)),T):
        end=t;mid=t-252;begin=mid-756
        if begin<0:continue
        tm=(cs[mid]-cs[begin])/756;vm=(cs[end]-cs[mid])/252
        tv=np.maximum((ss[mid]-ss[begin])/756-tm*tm,1e-14);vv=np.maximum((ss[end]-ss[mid])/252-vm*vm,1e-14)
        tr=tm/np.sqrt(tv)*np.sqrt(252);va=vm/np.sqrt(vv)*np.sqrt(252)
        good=np.flatnonzero((tm>0)&(vm>0)&(tr>0)&(va>0))
        shortlist=good[np.argsort(-tr[good],kind='stable')[:3]]
        best=int(shortlist[np.argmax(va[shortlist])]) if len(shortlist) else -1
        prop=ids[best] if best>=0 else -1;oldscore=va[ids.index(current)] if current in ids else -np.inf
        candscore=va[best] if best>=0 else -np.inf
        if current==-1 and prop>=0:eligible_change=True
        elif prop==-1:eligible_change=True
        else:eligible_change=prop!=current and candscore>=oldscore+.15
        if eligible_change:
            if challenger==prop:streak+=1
            else:challenger=prop;streak=1
            if streak>=5:current=prop;streak=0
        else:challenger=-99;streak=0
        sched[t]=current
        rows.append([dates[t],dates[begin],dates[mid-1],dates[mid],dates[end-1],SPECS[current]['name'] if current>=0 else 'CASH',SPECS[prop]['name'] if prop>=0 else 'CASH',candscore,oldscore,streak])
    return sched,pd.DataFrame(rows,columns=['decision_date','train_start','train_end','validation_start','latest_observed_return','selected','challenger','challenger_validation_sharpe','incumbent_validation_sharpe','streak'])

def export(name,result,f,details=False):
    d,pos,orders,recoveries=result;pd.DataFrame(d,index=f['dates'],columns=COLS).rename_axis('date').to_csv(OUT/'daily'/f'{name}.csv')
    if details:
        np.savez_compressed(OUT/'audit_arrays'/f'{name}.npz',positions=pos,orders=orders,recoveries=recoveries)
        ti,js=np.where(abs(orders)>1e-9);q=orders[ti,js];notional=np.abs(q)*f['p'][ti,js]
        pd.DataFrame(dict(execution_date=f['dates'][ti],decision_date=f['dates'][ti-1],ticker=f['codes'][js],qty=q,price=f['p'][ti,js],notional=notional,commission=notional*f['commission'][ti],sale_tax=notional*f['tax'][ti]*(q<0),slippage_impact=notional*.0005)).to_csv(OUT/'orders'/f'{name}.csv.gz',index=False)
        j=np.flatnonzero(pos[-1]>1e-9);nav=d[-1,0]
        pd.DataFrame(dict(ticker=f['codes'][j],adjusted_units=pos[-1,j],adjusted_close=f['marks'][-1,j],weight=pos[-1,j]*f['marks'][-1,j]/nav,near_52week=f['near'][-1,j],momentum_12_1=np.expm1(f['momentum'][-1,j]))).sort_values('weight',ascending=False).to_csv(OUT/'holdings'/f'{name}.csv',index=False)

def main():
    for sub in ['daily','orders','holdings','audit_arrays','schedules']:(OUT/sub).mkdir(exist_ok=True)
    if not (OUT/'inputs.npz').exists():shutil.copy2(SOURCE/'inputs_compact.npz',OUT/'inputs.npz')
    (OUT/'protocol_sha256.txt').write_text(hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest())
    f=prepare();dates=f['dates'];T=len(dates)
    (OUT/'data_manifest.json').write_text(json.dumps(dict(source=str(SOURCE/'inputs_compact.npz'),sha256=hashlib.sha256((OUT/'inputs.npz').read_bytes()).hexdigest(),dates=T,stocks=len(f['codes']),first=str(dates[0].date()),last=str(dates[-1].date()),eligible_min=int(f['eligible'][252:].sum(1).min()),eligible_max=int(f['eligible'].sum(1).max()),reconstructed_tax_max_diff=float(np.max(abs(f['tax']-np.load(OUT/'inputs.npz')['tax'])))),indent=2))
    pd.DataFrame(dict(date=dates,settlement=f['settlement'],sale_tax=f['tax'],commission=f['commission'],slippage_assumption=.0003,impact_assumption=.0002)).to_csv(OUT/'cost_schedule.csv',index=False)
    np.savez_compressed(OUT/'features.npz',weights=f['weights'],eligible=f['eligible'],near=f['near'],momentum=f['momentum'])
    ds=[];rows=[]
    for i,s in enumerate(SPECS):
        res=run(f,np.full(T,i));d=res[0];ds.append(d);export(s['name'],res,f)
        rows.append(dict(**s,**metrics(d,dates,'2009-01-09','2016-12-31')))
        if i%12==0:log(f'{i+1}/96 candidates')
    train=pd.DataFrame(rows);train.to_csv(OUT/'development_selection.csv',index=False)
    groups={'HIGH_EW':[i for i,s in enumerate(SPECS) if s['fi']<=2 and s['wi']==0], 'HIGH_WEIGHT':[i for i,s in enumerate(SPECS) if s['fi']<=2], 'MOM_EW':[i for i,s in enumerate(SPECS) if s['fi'] in [3,4] and s['wi']==0], 'MOM_WEIGHT':[i for i,s in enumerate(SPECS) if s['fi'] in [3,4]],'JOINT_WEIGHT':[i for i,s in enumerate(SPECS) if s['fi'] in [1,4,5,6,7]]}
    fixed={k:int(train.iloc[ids].sort_values(['sharpe','cagr','name'],ascending=[False,False,True]).index[0]) for k,ids in groups.items()}
    (OUT/'fixed_selection.json').write_text(json.dumps({k:SPECS[v] for k,v in fixed.items()},indent=2));log('Development selections frozen '+str({k:SPECS[v]['name'] for k,v in fixed.items()}))
    returns=np.stack([d[:,1] for d in ds],axis=1);np.savez_compressed(OUT/'candidate_returns.npz',returns=returns)
    allrows=[]
    periods={'development':('2009-01-09','2016-12-31'),'test':('2017-01-01','2021-12-31'),'deployment':('2022-01-01','2026-07-31'),'recent3':('2023-08-01','2026-07-31'),'2026YTD':('2026-01-01','2026-07-31')}
    for i,s in enumerate(SPECS):
        for per,(lo,hi) in periods.items():allrows.append(dict(**s,period=per,**metrics(ds[i],dates,lo,hi)))
    pd.DataFrame(allrows).to_csv(OUT/'all_96_results.csv',index=False)
    summaries=[];sens=[];paths={};schedules={}
    for g,ids in groups.items():
        schedule,audit=rolling(returns,ids,dates);name='ROLL_'+g;schedules[name]=schedule
        audit.to_csv(OUT/'schedules'/f'{name}.csv',index=False);np.save(OUT/'schedules'/f'{name}.npy',schedule)
        res=run(f,schedule,start='2017-01-02',record=True);export(name,res,f,True);paths[name]=res[0]
        fresh=run(f,schedule,start='2022-01-03');export('FRESH_'+g,fresh,f);paths['FRESH_'+g]=fresh[0]
        fixedres=run(f,np.full(T,fixed[g]),start='2017-01-02');export('FIXED_'+g,fixedres,f);paths['FIXED_'+g]=fixedres[0]
        for variant,sl,rec,z in [('execution13bp',.0013,1.,False),('execution25bp',.0025,1.,False),('recovery0',.0005,0.,False),('zero_cost',0.,1.,True)]:
            sr=run(f,schedule,start='2017-01-02',slip=sl,recovery=rec,zero=z)
            sens.append(dict(name=name,variant=variant,**metrics(sr[0],dates,*periods['deployment'])))
        log(name+' complete')
    for name,d in paths.items():
        for per,(lo,hi) in periods.items():
            if per!='development' and (not name.startswith('FRESH') or per not in ['test']):summaries.append(dict(name=name,period=per,**metrics(d,dates,lo,hi)))
    for name in ['KOSPI','KOSPI200']:
        z=pd.read_csv(SOURCE/f'benchmark_{name}.csv',index_col=0,parse_dates=True).reindex(dates);d=np.zeros((T,len(COLS)));d[:,0]=z.nav.ffill().fillna(1e7);d[:,1]=z.net_return.fillna(0);d[:,2]=1.
        paths[name]=d;export(name,(d,None,None,None),f)
        for per,(lo,hi) in periods.items():summaries.append(dict(name=name,period=per,**metrics(d,dates,lo,hi)))
    pd.DataFrame(summaries).to_csv(OUT/'performance.csv',index=False);pd.DataFrame(sens).to_csv(OUT/'sensitivity.csv',index=False)
    log('All accounts completed')

if __name__=='__main__':main()
