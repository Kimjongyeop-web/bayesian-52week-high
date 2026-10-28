from model import *
from solver import solve,objective
from threadpoolctl import threadpool_limits

COLS=['nav','net_return','exposure','turnover','commission','sale_tax','execution_cost','writeoff','cash','buy_notional','sell_notional','gross_return','review','n_holdings','unfilled','solver_gap','solver_iterations','min_target','max_target','target_sum','utility_gain','band_holds']

def review(f,t,u,qty,spec,pred,pm,pk,mode='bayes',band=True,allow_entry=True,slip=.0005,zero=False):
    N=len(u);W=u.copy();reason=np.full(N,'UNCHANGED',object);held=qty>1e-9
    know=np.isfinite(f['near'][t])&np.isfinite(f['rank'][t])
    exits=held&((~f['m'][t])|(know&((f['near'][t]<.90)|(f['rank'][t]>.20))))
    reason[exits]='EXIT_SIGNAL';reason[held&~f['m'][t]]='EXIT_MEMBERSHIP'
    if mode=='bayes':
        neg=held&know&np.isfinite(pk[t])&(pk[t]<.40);exits|=neg;reason[neg]='EXIT_POSTERIOR'
    W[exits]=0.;locked=held&~know&~exits
    remain=held&~exits&~locked
    entry=f['eligible'][t]&(f['near'][t]>=.90)&(f['rank'][t]<=.10)&~held
    if not allow_entry:entry[:]=False
    if mode=='bayes':entry&=np.isfinite(pm[t])&(pm[t]>=spec['prob'])
    new=np.flatnonzero(entry);new=new[np.argsort(-f['near'][t,new],kind='stable')[:max(0,20-int(held[~exits].sum()))]]
    ids=np.r_[np.flatnonzero(remain),new];meta=dict(gap=0.,iterations=0,gain=0.,band_holds=0)
    if not len(ids):return W,reason,meta
    budget=max(0.,1-W[locked].sum());old=u[ids];cb=0. if zero else f['comm'][t]+slip;cs=cb+(0. if zero else f['tax'][t])
    if mode=='bayes':
        available=np.isfinite(pred['mu'][t,ids])
        if not available.all():return W,reason,meta
        H=spec['H'];mu=pred['mu'][t,ids];B=f['factor'][t,ids];iv=f['iv'][t,ids]
        L=np.linalg.cholesky(pred['cov'][t]+np.eye(4)*1e-14)
        A=(f['X'][t,ids]*f['sigma'][t,ids,None]*np.sqrt(H))@L
        w,gap,its=solve(mu,B,iv,A,old,budget,spec['gamma']*H,cb,cs)
        if gap>1e-7:w,gap,its=solve(mu,B,iv,A,old,budget,spec['gamma']*H,cb,cs,maxiter=15000)
        assert gap<2e-6,(t,spec['name'],gap)
        original=np.minimum(old,.08)
        gain=objective(original,mu,B,iv,A,old,spec['gamma']*H,cb,cs)-objective(w,mu,B,iv,A,old,spec['gamma']*H,cb,cs)
        meta.update(gap=float(gap),iterations=int(its),gain=float(gain))
        if gain<=.00005:w=original
    else:w=np.full(len(ids),min(.08,budget/len(ids)))
    if band:
        same=(np.abs(w-old)<=np.maximum(.005,.20*w))&(old<=.08+1e-12)&(w>0)
        meta['band_holds']=int(same.sum());w[same]=old[same]
    # Band retention can exceed the budget: scale only desired increases, not incumbents.
    w=np.minimum(w,.08)
    if w.sum()>budget+1e-10:
        base=np.minimum(w,old);up=np.maximum(w-old,0);room=max(0.,budget-base.sum())
        w=base+up*min(1.,room/max(up.sum(),1e-15))
        if w.sum()>budget+1e-10:w*=budget/w.sum()
    W[ids]=w
    for k,j in enumerate(ids):
        if not held[j] and w[k]>0:reason[j]='BUY_NEW'
        elif w[k]<1e-10 and held[j]:reason[j]='EXIT_UTILITY'
        elif w[k]>old[k]+1e-10:reason[j]='ADD'
        elif w[k]<old[k]-1e-10:reason[j]='REDUCE'
        else:reason[j]='HOLD'
    if mode=='bayes':
        # Band can break convex optimum. Keep optional trade only if it still improves objective.
        actual_gain=objective(np.minimum(old,.08),mu,B,iv,A,old,spec['gamma']*H,cb,cs)-objective(w,mu,B,iv,A,old,spec['gamma']*H,cb,cs)
        if actual_gain<=.00005:
            W[ids]=np.minimum(old,.08);reason[ids]=np.where(old>.08,'REDUCE_CAP','HOLD');meta['gain']=0.
        else:meta['gain']=float(actual_gain)
    return W,reason,meta

def simulate(f,preds,spec_i=0,mode='bayes',band=True,schedule=None,start=None,stop=None,record=False,slip=.0005,recovery=1.,zero=False):
    T,N=f['p'].shape;start=f['start'] if start is None else start;stop=T if stop is None else stop
    qty=np.zeros(N);pending=np.zeros(N);cash=1e7;prev=1e7
    daily=np.zeros((T,len(COLS)));daily[:,0]=1e7;daily[:,8]=1e7
    orders=[];decisions=[];pos=np.zeros((T,N)) if record else None;recarr=np.zeros((T,N)) if record else None
    meta=dict(gap=0.,iterations=0,gain=0.,band_holds=0);lastspec=spec_i;reason=np.full(N,'INITIAL',object)
    probs={key:probabilities(p,f,slip,zero) for key,p in preds.items()}
    for t in range(start,stop):
        mark=f['marks'][t];before=cash+qty@mark;wo=0.;oldqty=qty.copy()
        rec=(qty>0)&~np.isfinite(f['p'][t])&~f['m'][t-1]
        for j in np.flatnonzero(rec):
            value=qty[j]*mark[j];cash+=value*recovery;wo+=value*(1-recovery)
            if record:recarr[t,j]=qty[j]
            qty[j]=0.;pending[j]=0.
        req=pending-qty;trad=np.isfinite(f['p'][t])&(f['p'][t]>0)&(f['v'][t]>0)
        cb=0. if zero else f['comm'][t]+slip;tx=0. if zero else f['tax'][t]
        ids=np.flatnonzero((req< -1e-9)&trad);qs=np.minimum(-req[ids],qty[ids]);sv=qs*f['p'][t,ids];sell=sv.sum();cash+=sell*(1-cb-tx);qty[ids]-=qs
        if record:
            for k,j in enumerate(ids):orders.append([t,t-1,int(j),-qs[k],f['p'][t,j],sv[k],sv[k]*(0 if zero else f['comm'][t]),sv[k]*tx,sv[k]*(0 if zero else slip),reason[j]])
        ids=np.flatnonzero((req>1e-9)&trad&f['m'][t]);total=np.dot(req[ids],f['p'][t,ids])*(1+cb);scale=min(1.,max(0.,cash)/total) if total>0 else 0.
        qs=req[ids]*scale;bv=qs*f['p'][t,ids];buy=bv.sum();cash-=buy*(1+cb);qty[ids]+=qs
        if record:
            for k,j in enumerate(ids):
                if qs[k]>1e-9:orders.append([t,t-1,int(j),qs[k],f['p'][t,j],bv[k],bv[k]*(0 if zero else f['comm'][t]),0.,bv[k]*(0 if zero else slip),reason[j]])
        failed=(abs(req)>1e-9)&~trad&~rec
        comm=(buy+sell)*(0 if zero else f['comm'][t]);tax=sell*tx;exc=(buy+sell)*(0 if zero else slip)
        nav=cash+qty@mark;assert abs(nav-before+comm+tax+exc+wo)<.02 and cash>-.001 and qty.min()>-1e-7
        u=qty*mark/nav;isreview=(t-start)%5==0
        daily[t,:]=[nav,nav/prev-1,u.sum(),(buy+sell)/before,comm,tax,exc,wo,cash,buy,sell,before/prev-1,isreview,np.sum(qty>1e-9),failed.sum(),meta['gap'],meta['iterations'],0,0,0,meta['gain'],meta['band_holds']]
        if record:pos[t]=qty
        prev=nav;old_pending=pending.copy();pending=qty.copy();pending[failed&(req<0)]=old_pending[failed&(req<0)]
        membership=(qty>0)&~f['m'][t];pending[membership]=0.;reason[membership]='EXIT_MEMBERSHIP'
        if isreview:
            selected=spec_i if schedule is None else int(schedule[t]);allow=selected>=0
            if selected>=0:lastspec=selected
            spec=SPECS[lastspec];pred=preds[(spec['H'],spec['W'])];pm,pk,pw=probs[(spec['H'],spec['W'])]
            W,reason,meta=review(f,t,u,qty,spec,pred,pm,pk,mode,band,allow,slip,zero)
            assert W.min()>=-1e-10 and W.sum()<=1+1e-8
            pending=np.divide(W*nav,mark,out=np.zeros(N),where=mark>0)
            unchanged=(abs(W-u)<1e-11);pending[unchanged]=qty[unchanged]
            daily[t,17:20]=[W.min(),W.max(),W.sum()]
            if record:
                for j in np.flatnonzero((qty>1e-9)|(pending>1e-9)):
                    decisions.append([t,int(j),spec['name'] if mode=='bayes' else mode,allow,reason[j],qty[j],pending[j],u[j],W[j],f['near'][t,j],f['rank'][t,j],pred['mu'][t,j] if mode=='bayes' else np.nan,pm[t,j] if mode=='bayes' else np.nan,pk[t,j] if mode=='bayes' else np.nan,pw[t,j] if mode=='bayes' else np.nan,meta['gap']])
    return daily,orders,decisions,pos,recarr

def metrics(d,dates,lo,hi):
    ix=(dates>=lo)&(dates<=hi);x=d[ix];r=x[:,1];w=np.r_[1,np.cumprod(1+r)];yrs=(dates[ix][-1]-dates[ix][0]).days/365.25
    return dict(cagr=w[-1]**(1/yrs)-1,total_return=w[-1]-1,sharpe=r.mean()/r.std(ddof=1)*np.sqrt(252) if r.std()>0 else 0.,mdd=np.min(w/np.maximum.accumulate(w)-1),vol=r.std(ddof=1)*np.sqrt(252),exposure=x[:,2].mean(),annual_turnover=x[:,3].mean()*252,annual_cost_rate=np.mean(x[:,4:7].sum(1)/x[:,0])*252,commission=x[:,4].sum(),sale_tax=x[:,5].sum(),execution_cost=x[:,6].sum(),writeoff=x[:,7].sum())

def export(name,result,f,record=False):
    d,o,de,pos,rec=result;pd.DataFrame(d,index=f['dates'],columns=COLS).rename_axis('date').to_csv(OUT/'daily'/f'{name}.csv')
    if record:
        x=pd.DataFrame(o,columns=['execution_i','decision_i','stock_i','qty','price','notional','commission','sale_tax','execution_cost','reason'])
        x['execution_date']=f['dates'][x.execution_i.astype(int)];x['decision_date']=f['dates'][x.decision_i.astype(int)];x['ticker']=f['codes'][x.stock_i.astype(int)];x.to_csv(OUT/'orders'/f'{name}.csv.gz',index=False)
        x=pd.DataFrame(de,columns=['decision_i','stock_i','policy','allow_entry','action','held_qty','target_qty','held_weight','target_weight','near','rank','mean_return','p_expected_net','p_expected_positive','p_realized_net','solver_gap'])
        x['date']=f['dates'][x.decision_i.astype(int)];x['ticker']=f['codes'][x.stock_i.astype(int)];x.to_csv(OUT/'decisions'/f'{name}.csv.gz',index=False)
        np.savez_compressed(OUT/'audit'/f'{name}.npz',positions=pos,recoveries=rec)
        j=np.flatnonzero(pos[-1]>1e-9);last=pd.DataFrame(dict(ticker=f['codes'][j],adjusted_units=pos[-1,j],adjusted_close=f['marks'][-1,j],weight=pos[-1,j]*f['marks'][-1,j]/d[-1,0],near=f['near'][-1,j],rank=f['rank'][-1,j]))
        last.sort_values('weight',ascending=False).to_csv(OUT/'holdings'/f'{name}.csv',index=False)

def rolling(rets,dates,start):
    n,k=rets.shape;cs=np.vstack([np.zeros(k),rets.cumsum(0)]);ss=np.vstack([np.zeros(k),(rets*rets).cumsum(0)]);sched=np.full(n,-1,int);current=-1;chall=-99;streak=0;rows=[]
    for t in range(max(start,1008),n):
        mid=t-252;lo=mid-756;tm=(cs[mid]-cs[lo])/756;vm=(cs[t]-cs[mid])/252
        tr=tm/np.sqrt(np.maximum((ss[mid]-ss[lo])/756-tm*tm,1e-14))*np.sqrt(252);va=vm/np.sqrt(np.maximum((ss[t]-ss[mid])/252-vm*vm,1e-14))*np.sqrt(252)
        ids=np.flatnonzero((tm>0)&(vm>0));top=ids[np.argsort(-tr[ids],kind='stable')[:3]];best=int(top[np.argmax(va[top])]) if len(top) else -1
        possible=(best==-1 and current!=-1) or (best>=0 and (current<0 or va[best]>va[current]+.15))
        if possible and best!=current:
            if chall==best:streak+=1
            else:chall=best;streak=1
            if streak>=5:current=best;streak=0
        else:streak=0;chall=-99
        sched[t]=current;rows.append([t,lo,mid-1,mid,t-1,current,best,streak])
    return sched,pd.DataFrame(rows,columns=['decision_i','train_start_i','train_end_i','validation_start_i','latest_return_i','selected_i','challenger_i','streak'])

def main():
    for s in ['predictions','daily','orders','decisions','audit','holdings']:(OUT/s).mkdir(exist_ok=True)
    if not (OUT/'inputs.npz').exists():shutil.copy2(OLD/'inputs.npz',OUT/'inputs.npz')
    (OUT/'protocol_sha256.txt').write_text(hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest())
    f=prepare();preds={}
    for H in [5,20]:
        for W in [504,1008]:
            preds[H,W]=forecast(f,H,W);log(f'Posterior H{H} W{W} ready')
    ready=max(np.flatnonzero(np.isfinite(p['mu']).any(1))[0] for p in preds.values());f['start']=int(max(f['dates'].searchsorted('2009-01-09'),ready))
    meta=dict(common_start=str(f['dates'][f['start']].date()),start_i=f['start'],data_end=str(f['dates'][-1].date()),input_sha256=hashlib.sha256((OUT/'inputs.npz').read_bytes()).hexdigest(),alpha_features=['intercept','52week_nearness','52week_nearness_squared','52week_nearness_rank'],no_momentum=True)
    (OUT/'data_manifest.json').write_text(json.dumps(meta,indent=2));log('Common start '+meta['common_start'])
    periods={'development':(meta['common_start'],'2016-12-31'),'test':('2017-01-01','2021-12-31'),'deployment':('2022-01-01','2026-07-31'),'2026YTD':('2026-01-01','2026-07-31')}
    train=[];accounts={};rr=[]
    for i,s in enumerate(SPECS):
        res=simulate(f,preds,i);export(s['name'],res,f);accounts[s['name']]=res[0];rr.append(res[0][:,1]);train.append(dict(i=i,**s,**metrics(res[0],f['dates'],*periods['development'])));log(s['name']+' done')
    train=pd.DataFrame(train);train.to_csv(OUT/'training_selection.csv',index=False)
    chosen=int(train.sort_values(['sharpe','cagr','name'],ascending=[False,False,True]).iloc[0]['i']);pick=SPECS[chosen]
    (OUT/'selection.json').write_text(json.dumps(dict(selected_i=chosen,selected=pick,selection_uses='development only',uses_test=False),indent=2));log('FROZEN '+pick['name'])
    for name,mode,band in [('BUFFER','buffer',False),('BUFFER_BAND','buffer',True),('BAYES','bayes',True),('BAYES_NOBAND','bayes',False)]:
        res=simulate(f,preds,chosen,mode,band,record=True);export(name,res,f,True);accounts[name]=res[0];log(name+' done')
    rets=np.stack(rr,1);np.savez_compressed(OUT/'candidate_returns.npz',returns=rets);schedule,audit=rolling(rets,f['dates'],int(f['dates'].searchsorted('2017-01-02')));audit.to_csv(OUT/'rolling_selection.csv',index=False);np.save(OUT/'rolling_schedule.npy',schedule)
    res=simulate(f,preds,chosen,schedule=schedule,start=int(f['dates'].searchsorted('2017-01-02')),record=True);export('ROLL_BAYES',res,f,True);accounts['ROLL_BAYES']=res[0]
    # Re-run original fixed horizons using the same common start and original ledger.
    import fixed
    oldweights=fixed.weights(f)
    for H in [5,120]:
        result=fixed.run(f,H,f['start'],oldweights);od=result[0];d=np.zeros((len(f['dates']),len(COLS)));d[:,:12]=od[:,:12]
        name=f'FIXED{H}';accounts[name]=d;export(name,(d,[],[],None,None),f)
    # Keep price index snapshots inside this deliverable for portability.
    for name in ['KOSPI200','KOSPI']:
        fp=OUT/f'benchmark_{name}.csv'
        if not fp.exists():shutil.copy2(OLD/'portable_inputs/reference'/f'benchmark_{name}.csv',fp)
        x=pd.read_csv(fp,index_col=0,parse_dates=True).reindex(f['dates']);d=np.zeros((len(f['dates']),len(COLS)));d[:,0]=x.nav.ffill().fillna(1e7);d[:,1]=x.net_return.fillna(0);d[:,2]=1.;accounts[name]=d;export(name,(d,[],[],None,None),f)
    rows=[]
    for name,d in accounts.items():
        for per,(lo,hi) in periods.items():
            if name=='ROLL_BAYES' and per=='development':continue
            rows.append(dict(name=name,period=per,**metrics(d,f['dates'],lo,hi)))
    pd.DataFrame(rows).to_csv(OUT/'performance.csv',index=False)
    se=[]
    for name,mode,band in [('BAYES','bayes',True),('BUFFER_BAND','buffer',True)]:
        for variant,slip,recovery,zero in [('execution13bp',.0013,1.,False),('execution25bp',.0025,1.,False),('recovery0',.0005,0.,False),('zero_cost',0.,1.,True)]:
            r=simulate(f,preds,chosen,mode,band,slip=slip,recovery=recovery,zero=zero);se.append(dict(name=name,variant=variant,**metrics(r[0],f['dates'],*periods['deployment'])))
    pd.DataFrame(se).to_csv(OUT/'sensitivity.csv',index=False)
    log('Complete')

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
