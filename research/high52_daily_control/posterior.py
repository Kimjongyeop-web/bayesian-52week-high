from pathlib import Path
import json,hashlib,time,shutil
import numpy as np
import pandas as pd
from scipy.special import stdtr
from numba import njit
OUT=Path(__file__).resolve().parent
OLD=OUT.parent/'high52_study'
SPECS=[dict(name=f'B_H{h}_W{w}_P{int(prob*100)}_G{g}',H=h,W=w,prob=prob,gamma=g) for h in [5,20] for w in [504,1008] for prob in [.60,.75] for g in [3.,10.]]
def log(s):print(time.strftime('%H:%M:%S'),s,flush=True)

def prepare(raw=None):
    if raw is None:raw=dict(np.load(OUT/'inputs.npz',allow_pickle=False))
    # Load ONLY explicitly allowed raw fields. Cached momentum arrays are not read.
    f={k:raw[k] for k in ['dates','codes','p','v','m','settlement','benchmark']}
    f['dates']=pd.DatetimeIndex(f['dates']);p=pd.DataFrame(np.where(f['p']>0,f['p'],np.nan));v=pd.DataFrame(f['v']);r=np.log(p).diff()
    sig=r.rolling(60).std().shift(1).clip(lower=.005)
    bad=(r.abs()>.35)&(r.abs()>12*sig);quarantine=bad.rolling(10,min_periods=1).max().fillna(0).astype(bool)
    valid=pd.DataFrame(f['m'])&p.gt(0)&v.gt(0)&p.notna().rolling(253).sum().eq(253)&sig.notna()
    eligible=valid&~quarantine;near=p/p.rolling(252).max()
    rank=near.where(valid).rank(axis=1,ascending=False,method='first',pct=True)
    z=((near-.90)/.10).clip(-5,1)
    X=np.stack([np.ones(p.shape),z,z*z,1-2*rank],axis=2)
    br=pd.Series(f['benchmark']).pct_change(fill_method=None)
    mv=br.rolling(60,min_periods=40).std().shift(1).clip(lower=.003).fillna(.015)
    beta=r.rolling(120,min_periods=60).cov(br).shift(1).div(br.rolling(120,min_periods=60).var().shift(1),axis=0).clip(-3,3).fillna(1.)
    factor=beta.mul(mv,axis=0).to_numpy();sigma=sig.fillna(.03).to_numpy()
    # Ensure positive idiosyncratic variance; all risk estimates lagged.
    iv=np.maximum(sigma*sigma-factor*factor,.000025)
    settlement=pd.DatetimeIndex(f['settlement']);tax=np.full(len(p),.003)
    for d,rate in [('2019-06-03',.0025),('2021-01-01',.0023),('2023-01-01',.002),('2024-01-01',.0018),('2025-01-01',.0015),('2026-01-01',.002)]:tax[settlement>=d]=rate
    comm=np.full(len(p),.00015);comm[(f['dates']>='2020-09-16')&(f['dates']<='2020-12-31')]=.000113604
    f.update(p=p.to_numpy(),marks=p.ffill().fillna(0).to_numpy(),near=near.to_numpy(),rank=rank.to_numpy(),X=X,eligible=eligible.to_numpy(),sigma=sigma,factor=factor,iv=iv,quarantine=quarantine.to_numpy(),tax=tax,comm=comm)
    return f

@njit(cache=True)
def labels(p,marks,v,m,H):
    T,N=p.shape;y=np.full((T,N),np.nan)
    for s in range(T-H-1):
        e=s+1;end=e+H
        for j in range(N):
            if not(np.isfinite(p[e,j]) and p[e,j]>0 and v[e,j]>0 and m[e,j]):continue
            value=marks[end,j]
            for k in range(e+1,end+1):
                if not np.isfinite(p[k,j]) and not m[k-1,j]:value=marks[k,j];break
            y[s,j]=value/p[e,j]-1
    return y

def forecast(f,H,W,save=True):
    X=f['X'];T,N,D=X.shape;scale=f['sigma']*np.sqrt(H)
    y=labels(f['p'],f['marks'],f['v'],f['m'],H)/scale
    ok=f['eligible']&(f['near']>=.80)&np.isfinite(X).all(2)&np.isfinite(y)
    nn=ok.sum(1);ww=np.divide(ok,nn[:,None]*H,out=np.zeros((T,N)),where=nn[:,None]>0)
    xx=np.zeros((T+1,D,D));xy=np.zeros((T+1,D));yy=np.zeros(T+1);mass=np.zeros(T+1)
    for lo in range(0,T,128):
        hi=min(lo+128,T);a=np.nan_to_num(X[lo:hi]);b=np.nan_to_num(y[lo:hi]);w=ww[lo:hi]
        xx[lo+1:hi+1]=np.einsum('tni,tn,tnj->tij',a,w,a);xy[lo+1:hi+1]=np.einsum('tni,tn->ti',a,w*b);yy[lo+1:hi+1]=(w*b*b).sum(1);mass[lo+1:hi+1]=w.sum(1)
    xx=xx.cumsum(0);xy=xy.cumsum(0);yy=yy.cumsum();mass=mass.cumsum();cnt=np.r_[0,(nn>0).cumsum()]
    mu=np.full((T,N),np.nan);se=mu.copy();predscale=mu.copy();df=np.full(T,np.nan);cov=np.full((T,D,D),np.nan);co=np.full((T,D),np.nan);au=[]
    prior=np.diag([16.,4.,4.,4.])
    for t in range(T):
        hi=t-H-1;lo=max(0,hi-W)
        if hi<=lo or cnt[hi]-cnt[lo]<126:continue
        V=np.linalg.inv(prior+xx[hi]-xx[lo]);q=xy[hi]-xy[lo];beta=V@q;a=3+(mass[hi]-mass[lo])/2;b=max(1e-12,2+(yy[hi]-yy[lo]-beta@q)/2)
        leverage=np.maximum(np.einsum('ni,ij,nj->n',X[t],V,X[t]),0)
        mu[t]=X[t]@beta*scale[t];se[t]=np.sqrt(b/a*leverage)*scale[t];predscale[t]=np.sqrt(b/a*(1+leverage))*scale[t]
        df[t]=2*a;cov[t]=b/(a-1)*V;co[t]=beta
        au.append([t,lo,hi-1,hi-1+H+1,int(cnt[hi]-cnt[lo]),mass[hi]-mass[lo],a,b])
    pred=dict(mu=mu,se=se,predscale=predscale,df=df,cov=cov,coef=co)
    if save:
        np.savez_compressed(OUT/'predictions'/f'H{H}_W{W}.npz',**pred)
        pd.DataFrame(au,columns=['decision_i','oldest_origin_i','latest_origin_i','latest_label_end_i','mature_days','effective_mass','a','b']).to_csv(OUT/'predictions'/f'H{H}_W{W}_audit.csv',index=False)
    return pred

def probabilities(pred,f,slip=.0005,zero=False):
    cost=np.zeros(len(f['dates'])) if zero else 2*(f['comm']+slip)+f['tax']
    pmean=stdtr(pred['df'][:,None],(pred['mu']-cost[:,None])/np.maximum(pred['se'],1e-12))
    pkeep=stdtr(pred['df'][:,None],pred['mu']/np.maximum(pred['se'],1e-12))
    pwin=stdtr(pred['df'][:,None],(pred['mu']-cost[:,None])/np.maximum(pred['predscale'],1e-12))
    return pmean,pkeep,pwin
