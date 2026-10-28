import numpy as np
from numba import njit

@njit(cache=True)
def trading(w,u,cb,cs):
    d=w-u;return np.maximum(d,0).sum()*cb+np.maximum(-d,0).sum()*cs
@njit(cache=True)
def unconstrained_prox(z,u,cap,step,cb,cs,dual):
    w=np.empty(len(z))
    for j in range(len(z)):
        x=z[j]-dual;d=x-u[j]
        if d>step*cb:y=x-step*cb
        elif d<-step*cs:y=x+step*cs
        else:y=u[j]
        w[j]=min(cap,max(0.,y))
    return w
@njit(cache=True)
def prox(z,u,cap,budget,step,cb,cs):
    w=unconstrained_prox(z,u,cap,step,cb,cs,0.)
    if w.sum()<=budget:return w
    lo=0.;hi=max(1.,np.max(z+step*cs))
    for k in range(45):
        mid=(lo+hi)/2;w=unconstrained_prox(z,u,cap,step,cb,cs,mid)
        if w.sum()>budget:lo=mid
        else:hi=mid
    return unconstrained_prox(z,u,cap,step,cb,cs,hi)
@njit(cache=True)
def gradient(w,mu,B,iv,A,gH):return gH*(iv*w+B*np.dot(B,w))+A@(A.T@w)-mu
@njit(cache=True)
def objective(w,mu,B,iv,A,u,gH,cb,cs):
    x=A.T@w;return .5*gH*(np.dot(w*w,iv)+np.dot(w,B)**2)+.5*np.dot(x,x)-np.dot(w,mu)+trading(w,u,cb,cs)
@njit(cache=True)
def gap(w,g,u,cap,budget,cb,cs):
    n=len(w);slopes=np.empty(2*n);sizes=np.empty(2*n)
    for j in range(n):
        b=min(cap,max(0.,u[j]));slopes[j]=g[j]-cs;sizes[j]=b;slopes[n+j]=g[j]+cb;sizes[n+j]=cap-b
    v=np.zeros(n);remaining=budget
    for k in np.argsort(slopes):
        if slopes[k]>=0:break
        q=min(sizes[k],remaining);v[k%n]+=q;remaining-=q
        if remaining<1e-13:break
    return max(0.,np.dot(g,w-v)+trading(w,u,cb,cs)-trading(v,u,cb,cs))
@njit(cache=True)
def solve(mu,B,iv,A,u,budget,gH,cb,cs,cap=.08,tol=1e-7,maxiter=2500):
    L=max(gH*(iv.max()+np.dot(B,B))+np.sum(A*A),1e-10)
    w=prox(u,u,cap,budget,0.,0.,0.);z=w.copy();theta=1.
    for k in range(maxiter):
        wn=prox(z-gradient(z,mu,B,iv,A,gH)/L,u,cap,budget,1/L,cb,cs)
        th=(1+np.sqrt(1+4*theta*theta))/2;z=wn+(theta-1)/th*(wn-w);w=wn;theta=th
        if k%20==19 or k==maxiter-1:
            err=gap(w,gradient(w,mu,B,iv,A,gH),u,cap,budget,cb,cs)
            if err<=tol:return w,err,k+1
    return w,err,maxiter
