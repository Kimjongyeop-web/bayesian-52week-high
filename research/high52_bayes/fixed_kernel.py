import numpy as np
from numba import njit
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
