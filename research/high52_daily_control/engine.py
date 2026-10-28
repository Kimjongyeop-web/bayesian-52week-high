import math
import numpy as np
from numba import njit

@njit(cache=True)
def betacf(a,b,x):
    tiny=1e-300; qab=a+b;qap=a+1.;qam=a-1.;c=1.;d=1.-qab*x/qap
    if abs(d)<tiny:d=tiny
    d=1./d;h=d
    for m in range(1,201):
        m2=2*m;aa=m*(b-m)*x/((qam+m2)*(a+m2));d=1.+aa*d
        if abs(d)<tiny:d=tiny
        c=1.+aa/c
        if abs(c)<tiny:c=tiny
        d=1./d;h*=d*c;aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2));d=1.+aa*d
        if abs(d)<tiny:d=tiny
        c=1.+aa/c
        if abs(c)<tiny:c=tiny
        d=1./d;delta=d*c;h*=delta
        if abs(delta-1.)<3e-14:break
    return h

@njit(cache=True)
def tcdf(df,z):
    if z==0.:return .5
    x=df/(df+z*z);a=df/2;b=.5
    if x>=1.:return .5
    bt=math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log1p(-x))
    if x<(a+1.)/(a+b+2.):ib=bt*betacf(a,b,x)/a
    else:ib=1.-bt*betacf(b,a,1.-x)/b
    return 1.-.5*ib if z>0 else .5*ib

@njit(cache=True)
def projection(v,u,cap,budget,lam,cb,cs):
    n=len(v);w=np.zeros(n)
    if n==0 or budget<=0:return w
    budget=min(budget,n*cap)
    if budget>=n*cap-1e-12:return np.full(n,cap)
    lo=-2.;hi=2.
    for _ in range(48):
        dual=(lo+hi)*.5;total=0.
        for j in range(n):
            z=v[j]-dual;d=z-u[j]
            if d>lam*cb:z-=lam*cb
            elif d<-lam*cs:z+=lam*cs
            else:z=u[j]
            w[j]=min(cap,max(0.,z));total+=w[j]
        if total>budget:lo=dual
        else:hi=dual
    # hi is the feasible side; recompute explicitly.
    for j in range(n):
        z=v[j]-hi;d=z-u[j]
        if d>lam*cb:z-=lam*cb
        elif d<-lam*cs:z+=lam*cs
        else:z=u[j]
        w[j]=min(cap,max(0.,z))
    return w

@njit(cache=True)
def run_kernel(p,marks,vol,member,eligible,near,rank,X,sigma,tax,comm,coef,cov,df,H,p0,strength,cap,lam,start,stop,slip=.0005,recovery=1.,record=False,external=None,interval=1,exit_near=.9,exit_rank=.2,confirm=1,cooldown=0,band_abs=0.,band_rel=0.,day_limit=1e6,window_limit=1e6):
    T,N=p.shape;qty=np.zeros(N);pending=np.zeros(N);cash=1e7;prev=1e7
    daily=np.zeros((T,17));daily[:,0]=1e7;daily[:,8]=1e7
    streak=np.zeros(N,np.int64);last_exit=np.full(N,-10000,np.int64);pending_hard=np.zeros(N,np.bool_);budgetlog=np.zeros((T,5))
    pos=np.zeros((T,N)) if record else np.empty((0,0));trades=pos.copy();plans=pos.copy();probs=pos.copy();recq=pos.copy()
    for t in range(start,stop):
        before=cash+np.dot(qty,marks[t]);wo=0.;buys=0.;sells=0.;cb=comm[t]+slip;cs=cb+tax[t]
        # Same-market net trades, with explicit recovery and no untradable fills.
        req=np.zeros(N);trad=np.zeros(N,np.bool_)
        for j in range(N):
            if qty[j]>0 and not np.isfinite(p[t,j]) and not member[t-1,j]:
                value=qty[j]*marks[t,j];cash+=value*recovery;wo+=value*(1-recovery)
                if record:recq[t,j]=qty[j]
                qty[j]=0.;pending[j]=0.;last_exit[j]=t;pending_hard[j]=False
            req[j]=pending[j]-qty[j];trad[j]=np.isfinite(p[t,j]) and p[t,j]>0 and vol[t,j]>0
        desired=0.
        for j in range(N):
            if trad[j] and ((req[j]<-1e-9 and not pending_hard[j]) or (req[j]>1e-9 and member[t,j])):desired+=abs(req[j])*p[t,j]
        rawreq=req.copy()
        used=budgetlog[max(0,t-19):t,0].sum();allowance=max(0.,min(day_limit,window_limit-used))*before
        initial=t==start+1
        normal_scale=1. if initial or desired<=0 else min(1.,allowance/desired)
        normal_value=0.;hard_value=0.;initial_value=0.
        for j in range(N):
            if not pending_hard[j] or req[j]>=0:req[j]*=normal_scale
            if req[j]<-1e-9 and trad[j]:
                q=min(-req[j],qty[j]);val=q*p[t,j];cash+=val*(1-cs);qty[j]-=q;sells+=val
                if pending_hard[j]:hard_value+=val
                elif initial:initial_value+=val
                else:normal_value+=val
                if qty[j]<1e-9:last_exit[j]=t
                if record:trades[t,j]=-q
        total=0.
        for j in range(N):
            if req[j]>1e-9 and trad[j] and member[t,j]:total+=req[j]*p[t,j]*(1+cb)
        scale=min(1.,max(0.,cash)/total) if total>0 else 0.
        held_count=int(np.sum(qty>1e-9))
        for j in range(N):
            if req[j]>1e-9 and trad[j] and member[t,j]:
                if day_limit<1e5 and qty[j]<=1e-9 and held_count>=20:continue
                q=req[j]*scale;val=q*p[t,j];cash-=val*(1+cb);qty[j]+=q;buys+=val
                if q>1e-9 and qty[j]-q<=1e-9:held_count+=1
                if initial:initial_value+=val
                else:normal_value+=val
                if record:trades[t,j]=q
        budgetlog[t]=np.array([normal_value/before,hard_value/before,initial_value/before,used+normal_value/before,normal_scale])
        assert initial or (budgetlog[t,0]<=day_limit+1e-9 and budgetlog[t,3]<=window_limit+1e-9)
        nav=cash+np.dot(qty,marks[t]);costc=(buys+sells)*comm[t];costt=sells*tax[t];costs=(buys+sells)*slip
        assert cash>-.001 and abs(nav-before+costc+costt+costs+wo)<.1 and np.min(qty)>-1e-8
        u=qty*marks[t]/nav;review=(t-start)%interval==0
        daily[t,:13]=np.array([nav,nav/prev-1,u.sum(),(buys+sells)/before,costc,costt,costs,wo,cash,buys,sells,before/prev-1,float(np.sum(qty>1e-9))]);daily[t,13]=review
        if record:pos[t]=qty
        prev=nav;oldpending=pending.copy();pending=qty.copy()
        for j in range(N):
            if rawreq[j]<-1e-9 and not trad[j]:pending[j]=oldpending[j]
            if qty[j]>1e-9 and not member[t,j]:pending[j]=0.
        if not review:continue
        target=u.copy();ids=np.empty(N,np.int64);n=0;locked=0.;heldn=0;next_hard=np.zeros(N,np.bool_)
        for j in range(N):
            if qty[j]<=1e-9:target[j]=0.;continue
            known=np.isfinite(near[t,j]) and np.isfinite(rank[t,j])
            hard=not member[t,j] or (known and near[t,j]<.8)
            breach=known and (near[t,j]<exit_near or rank[t,j]>exit_rank)
            streak[j]=streak[j]+1 if breach else 0
            if hard or (breach and streak[j]>=confirm):
                target[j]=0.
                next_hard[j]=hard
            elif not known:
                locked+=u[j];heldn+=1
                # Missing ranks do not revoke an already authorized unfilled reduction.
                if pending[j]<qty[j]:target[j]=pending[j]*marks[t,j]/nav;next_hard[j]=pending_hard[j]
            else:ids[n]=j;n+=1;heldn+=1
        # Stable rank includes deterministic tie-break; lower rank is preferred.
        candidates=np.empty(N,np.int64);nk=0
        for j in range(N):
            if qty[j]<=1e-9 and eligible[t,j] and near[t,j]>=.9 and rank[t,j]<=.1 and t-last_exit[j]>=cooldown:candidates[nk]=j;nk+=1
        if nk>0:
            order=np.argsort(rank[t,candidates[:nk]])
            slots=max(0,20-int(np.sum(qty>1e-9))) if day_limit<1e5 else max(0,20-heldn)
            for k in range(min(nk,slots)):ids[n]=candidates[order[k]];n+=1
        ids=ids[:n];budget=max(0.,min(.98-locked,n*cap));v=np.zeros(n);prob=np.full(n,.5)
        if n>0:
            if strength>0 and np.isfinite(df[t]):
                a=np.zeros((n,4));avg=np.zeros(4)
                for k in range(n):
                    j=ids[k]
                    for d in range(4):a[k,d]=X[t,j,d]*sigma[t,j]*math.sqrt(H);avg[d]+=a[k,d]/n
                for k in range(n):
                    delta=a[k]-avg;mean=np.dot(delta,coef[t]);scale2=np.dot(delta,np.dot(cov[t],delta))*(df[t]-2)/df[t]
                    prob[k]=tcdf(df[t],mean/math.sqrt(max(scale2,1e-24))) if scale2>1e-22 else .5
                    d=prob[k]-.5;s=max(abs(d)-(p0-.5),0.)/(1-p0)
                    if d<0:s=-s
                    v[k]=math.exp(strength*s)
            else:v[:]=1.
            v*=budget/v.sum();w=projection(v,u[ids],cap,budget,lam,cb,cs)
            for k in range(n):
                j=ids[k];target[j]=w[k]
                if qty[j]>1e-9 and u[j]<=cap and w[k]>0 and abs(w[k]-u[j])<=max(band_abs,band_rel*u[j]):target[j]=u[j]
            base=np.minimum(target,u);up=np.maximum(target-u,0);room=max(0.,1.-base.sum())
            if up.sum()>room:target=base+up*room/up.sum()
        if external is not None:
            # Model targets are causal external plans, but own mandatory exclusions and locked positions apply.
            target=external[t].copy()
            for j in range(N):
                if not member[t,j]:target[j]=0.
            if target.sum()>1.:target/=target.sum()
        assert target.min()>-1e-10 and target.sum()<=1.+1e-8
        daily[t,14]=target.sum();daily[t,15]=prob.mean() if n else .5;daily[t,16]=target.max()
        if record:
            plans[t]=target
            for k in range(n):probs[t,ids[k]]=prob[k]
        for j in range(N):
            if abs(target[j]-u[j])<1e-11:pending[j]=qty[j]
            else:pending[j]=target[j]*nav/marks[t,j] if marks[t,j]>0 else 0.
        pending_hard=next_hard
    return daily,pos,trades,plans,probs,recq,budgetlog
