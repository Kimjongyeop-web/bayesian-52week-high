from model import *
from run import simulate,review,COLS,rolling
from solver import solve,objective
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

def main():
    checks={};rng=np.random.default_rng(54);sol=[]
    for k in range(12):
        n=8;mu=rng.normal(.005,.008,n);B=rng.normal(.02,.005,n);iv=np.full(n,.0004);A=rng.normal(0,.005,(n,4));u=rng.uniform(0,.08,n);budget=.6;gH=15.;cb=.00065;cs=.00265
        w,g,it=solve(mu,B,iv,A,u,budget,gH,cb,cs,tol=1e-9,maxiter=10000)
        # Independent convex epigraph solver: variables(w,buy,sell) with w-u=buy-sell.
        def fun(z):
            x=z[:n];return .5*gH*(np.dot(x*x,iv)+(x@B)**2)+.5*np.sum((A.T@x)**2)-mu@x+cb*z[n:2*n].sum()+cs*z[2*n:].sum()
        init=np.r_[w,np.maximum(w-u,0),np.maximum(u-w,0)]
        res=minimize(fun,init,method='SLSQP',bounds=[(0,.08)]*n+[(0,None)]*(2*n),constraints=[{'type':'eq','fun':lambda z:z[:n]-u-z[n:2*n]+z[2*n:]},{'type':'ineq','fun':lambda z:budget-z[:n].sum()}],options={'ftol':1e-12,'maxiter':2000})
        error=abs(fun(res.x)-objective(w,mu,B,iv,A,u,gH,cb,cs));assert res.success and error<2e-7
        sol.append(dict(gap=float(g),independent_objective_error=float(error)))
    checks['solver_vs_slsqp']=sol
    f=prepare();f['start']=json.loads((OUT/'data_manifest.json').read_text())['start_i'];dates=f['dates'];T=len(dates)
    preds={(h,w):dict(np.load(OUT/'predictions'/f'H{h}_W{w}.npz')) for h in [5,20] for w in [504,1008]}
    pick=json.loads((OUT/'selection.json').read_text());chosen=pick['selected_i'];spec=pick['selected']
    checks['label_dates']={}
    for H,W in preds:
        x=pd.read_csv(OUT/'predictions'/f'H{H}_W{W}_audit.csv');assert (x.latest_label_end_i<x.decision_i).all()
        checks['label_dates'][f'H{H}_W{W}']=True
    checks['accounts']={}
    for ap in (OUT/'audit').glob('*.npz'):
        name=ap.stem;ar=np.load(ap);q=ar['positions'];rec=ar['recoveries'];d=pd.read_csv(OUT/'daily'/f'{name}.csv',index_col=0)
        o=pd.read_csv(OUT/'orders'/f'{name}.csv.gz');dq=np.zeros_like(q)
        np.add.at(dq,(o.execution_i.to_numpy(int),o.stock_i.to_numpy(int)),o.qty.to_numpy())
        prev=np.vstack([np.zeros(q.shape[1]),q[:-1]]);err=float(np.max(abs(q-prev-dq+rec)));assert err<1e-7
        vals=np.abs(dq)*np.nan_to_num(f['p']);buy=np.where(dq>0,vals,0).sum(1);sell=np.where(dq<0,vals,0).sum(1);c=(buy+sell)*f['comm'];st=sell*f['tax'];exc=(buy+sell)*.0005
        cash=1e7+np.cumsum(sell-buy-c-st-exc+(rec*f['marks']).sum(1));nav=cash+(q*f['marks']).sum(1);naverr=float(np.max(abs(nav-d.nav.to_numpy())));assert naverr<.1
        assert not ((abs(dq)>1e-9)&(~np.isfinite(f['p'])|~(f['v']>0))).any()
        assert cash.min()>-.01 and q.min()>-1e-7 and (o.decision_i<o.execution_i).all()
        de=pd.read_csv(OUT/'decisions'/f'{name}.csv.gz');assert de.solver_gap.max()<2e-6
        checks['accounts'][name]=dict(quantity_error=err,nav_error=naverr,no_untradable_fills=True,nonnegative_cash=True,no_same_close_fills=True)
    raw=dict(np.load(OUT/'inputs.npz'));cut=int(dates.searchsorted('2024-01-02'))
    rawcut={k:(v[:cut].copy() if v.shape and v.shape[0]==T else v.copy()) for k,v in raw.items()};fc=prepare(rawcut);fc['start']=f['start']
    np.testing.assert_allclose(fc['X'],f['X'][:cut],atol=0,rtol=0,equal_nan=True)
    # Remove all cached alpha/model arrays: forecast features must remain identical.
    minimal={k:raw[k] for k in ['dates','codes','p','v','m','settlement','benchmark']};minimalf=prepare(minimal)
    np.testing.assert_allclose(minimalf['X'],f['X'],atol=0,rtol=0,equal_nan=True)
    checks['no_cached_momentum_dependency']=True
    pc=forecast(fc,spec['H'],spec['W'],save=False)
    for key in ['mu','se','predscale','df','cov','coef']:np.testing.assert_allclose(pc[key],preds[spec['H'],spec['W']][key][:cut],atol=1e-10,rtol=1e-10,equal_nan=True)
    res=simulate(fc,{(spec['H'],spec['W']):pc},chosen,record=True)
    ar=np.load(OUT/'audit/BAYES.npz');np.testing.assert_allclose(res[3],ar['positions'][:cut],atol=1e-7,rtol=1e-10)
    saved=pd.read_csv(OUT/'daily/BAYES.csv',index_col=0);np.testing.assert_allclose(res[0][:,0],saved.nav[:cut],atol=.01,rtol=1e-10)
    checks['future_truncation_features_posterior_positions_nav']=True
    rr=np.load(OUT/'candidate_returns.npz')['returns'];a,_=rolling(rr,dates,int(dates.searchsorted('2017-01-02')));b,_=rolling(rr[:cut],dates[:cut],int(dates.searchsorted('2017-01-02')));assert np.array_equal(a[:cut],b)
    checks['future_truncation_rolling_selector']=True
    # Mature high-confidence data are not mistaken for realized-profit certainty.
    pm,pk,pw=probabilities(preds[spec['H'],spec['W']],f);assert np.nanmin(pw)>=0 and np.nanmax(pw)<=1
    checks['posterior_predictive_probabilities_bounded']=True
    (OUT/'validation.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
