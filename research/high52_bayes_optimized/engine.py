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
def run_kernel(p,marks,vol,member,eligible,near,rank,X,sigma,tax,comm,coef,cov,df,H,p0,strength,cap,lam,start,stop,slip=.0005,recovery=1.,record=False,external=None):
    T,N=p.shape;qty=np.zeros(N);pending=np.zeros(N);cash=1e7;prev=1e7
    daily=np.zeros((T,17));daily[:,0]=1e7;daily[:,8]=1e7
    pos=np.zeros((T,N)) if record else np.empty((0,0));trades=pos.copy();plans=pos.copy();probs=pos.copy();recq=pos.copy()
    for t in range(start,stop):
        before=cash+np.dot(qty,marks[t]);wo=0.;buys=0.;sells=0.;cb=comm[t]+slip;cs=cb+tax[t]
        # Same-market net trades, with explicit recovery and no untradable fills.
        req=np.zeros(N);trad=np.zeros(N,np.bool_)
        for j in range(N):
            if qty[j]>0 and not np.isfinite(p[t,j]) and not member[t-1,j]:
                value=qty[j]*marks[t,j];cash+=value*recovery;wo+=value*(1-recovery)
                if record:recq[t,j]=qty[j]
                qty[j]=0.;pending[j]=0.
            req[j]=pending[j]-qty[j];trad[j]=np.isfinite(p[t,j]) and p[t,j]>0 and vol[t,j]>0
            if req[j]<-1e-9 and trad[j]:
                q=min(-req[j],qty[j]);val=q*p[t,j];cash+=val*(1-cs);qty[j]-=q;sells+=val
                if record:trades[t,j]=-q
        total=0.
        for j in range(N):
            if req[j]>1e-9 and trad[j] and member[t,j]:total+=req[j]*p[t,j]*(1+cb)
        scale=min(1.,max(0.,cash)/total) if total>0 else 0.
        for j in range(N):
            if req[j]>1e-9 and trad[j] and member[t,j]:
                q=req[j]*scale;val=q*p[t,j];cash-=val*(1+cb);qty[j]+=q;buys+=val
                if record:trades[t,j]=q
        nav=cash+np.dot(qty,marks[t]);costc=(buys+sells)*comm[t];costt=sells*tax[t];costs=(buys+sells)*slip
        assert cash>-.001 and abs(nav-before+costc+costt+costs+wo)<.1 and np.min(qty)>-1e-8
        u=qty*marks[t]/nav;review=(t-start)%5==0
        daily[t,:13]=np.array([nav,nav/prev-1,u.sum(),(buys+sells)/before,costc,costt,costs,wo,cash,buys,sells,before/prev-1,float(np.sum(qty>1e-9))]);daily[t,13]=review
        if record:pos[t]=qty
        prev=nav;oldpending=pending.copy();pending=qty.copy()
        for j in range(N):
            if req[j]<-1e-9 and not trad[j]:pending[j]=oldpending[j]
            if qty[j]>1e-9 and not member[t,j]:pending[j]=0.
        if not review:continue
        target=u.copy();ids=np.empty(N,np.int64);n=0;locked=0.;heldn=0
        for j in range(N):
            if qty[j]<=1e-9:target[j]=0.;continue
            known=np.isfinite(near[t,j]) and np.isfinite(rank[t,j])
            if not member[t,j] or (known and (near[t,j]<.9 or rank[t,j]>.2)):
                target[j]=0.
            elif not known:locked+=u[j];heldn+=1
            else:ids[n]=j;n+=1;heldn+=1
        # Stable rank includes deterministic tie-break; lower rank is preferred.
        candidates=np.empty(N,np.int64);nk=0
        for j in range(N):
            if qty[j]<=1e-9 and eligible[t,j] and near[t,j]>=.9 and rank[t,j]<=.1:candidates[nk]=j;nk+=1
        if nk>0:
            order=np.argsort(rank[t,candidates[:nk]])
            for k in range(min(nk,max(0,20-heldn))):ids[n]=candidates[order[k]];n+=1
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
            for k in range(n):target[ids[k]]=w[k]
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
    return daily,pos,trades,plans,probs,recq
