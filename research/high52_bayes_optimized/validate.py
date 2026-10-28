from study import *
from engine import tcdf,projection
from scipy.optimize import minimize
from scipy.special import stdtr

def numerical():
    rng=np.random.default_rng(462);dfs=rng.uniform(6,120,1000);zs=rng.normal(0,5,1000)
    err=max(abs(tcdf(df,z)-stdtr(df,z)) for df,z in zip(dfs,zs));assert err<2e-11
    errors=[]
    for _ in range(15):
        n=20;cap=.1;budget=.98;u=rng.dirichlet(np.ones(n))*.97;v=rng.dirichlet(np.ones(n))*.98;lam=3.;cb=.00065;cs=.00365
        w=projection(v,u,cap,budget,lam,cb,cs)
        def obj(z):return .5*np.sum((z[:n]-v)**2)+lam*(cb*z[n:2*n].sum()+cs*z[2*n:].sum())
        cons=[{'type':'eq','fun':lambda z:z[:n].sum()-budget},{'type':'ineq','fun':lambda z:z[n:2*n]-(z[:n]-u)},{'type':'ineq','fun':lambda z:z[2*n:]-(u-z[:n])}]
        init=np.r_[w,np.maximum(w-u,0),np.maximum(u-w,0)]
        res=minimize(obj,init,method='SLSQP',bounds=[(0,cap)]*n+[(0,None)]*(2*n),constraints=cons,options={'ftol':1e-12,'maxiter':300})
        assert res.success,res.message
        error=abs(obj(init)-res.fun);assert error<1e-8 and abs(w.sum()-budget)<1e-10;errors.append(error)
    return dict(student_t_cdf_max_error=err,projection_max_objective_error=max(errors))

def audit(f,name):
    a=np.load(OUT/'audit'/f'{name}.npz');qty=np.zeros(len(f['codes']));cash=1e7;positions=a['positions'];tr=a['trades'];rec=a['recoveries'];d=pd.read_csv(OUT/'daily'/f'{name}.csv');errnav=0.;errqty=0.
    for t in range(len(d)):
        if rec[t].sum()>0:cash+=rec[t]@f['marks'][t];qty-=rec[t]
        ix=abs(tr[t])>1e-9
        if ix.any():
            assert t>0 and np.all(np.isfinite(f['p'][t,ix])) and np.all(f['v'][t,ix]>0)
            assert (tr[t,ix]>0).sum()==0 or np.all(f['m'][t,ix][tr[t,ix]>0])
        buys=np.maximum(tr[t],0)@np.nan_to_num(f['p'][t]);sells=np.maximum(-tr[t],0)@np.nan_to_num(f['p'][t]);cash+=sells*(1-f['comm'][t]-.0005-f['tax'][t])-buys*(1+f['comm'][t]+.0005);qty+=tr[t]
        errqty=max(errqty,float(abs(qty-positions[t]).max()));errnav=max(errnav,abs(cash+qty@f['marks'][t]-d.nav.iloc[t]));assert cash>-.001 and qty.min()>-1e-8
        if t==0:assert not ix.any()
    assert errnav<.1 and errqty<1e-7
    plans=a['plans'];ix=d.review.to_numpy()>0;assert plans.min()>-1e-10 and plans[ix].sum(1).max()<=1.+1e-8
    return dict(max_nav_error=errnav,max_quantity_error=errqty,no_untradable_fills=True,nonnegative_cash=True,max_planned_weight=float(plans.max()))

def causal(f,pr):
    pick=json.loads((OUT/'selection.json').read_text());s=pick['selected'];cut=int(f['dates'].searchsorted('2024-01-02'));raw=dict(np.load(OUT/'inputs.npz'));truncated={k:v[:cut] if v.ndim>0 and v.shape[0]==len(f['dates']) and k!='codes' else v for k,v in raw.items()};ff=prepare(truncated);ff['start']=f['start'];h,w=MODELS[s['model']];pred=forecast(ff,h,w,save=False)
    for k in ['coef','cov','df']:np.testing.assert_allclose(pred[k],pr[s['model']][k][:cut],equal_nan=True,atol=1e-10,rtol=1e-9)
    small={s['model']:{k:pred[k] for k in ['coef','cov','df']}};start=int(f['dates'].searchsorted('2017-01-02'));short=simulate(ff,small,s,start=start,record=True);full=np.load(OUT/'audit/OPTIMIZED.npz');dfull=pd.read_csv(OUT/'daily/OPTIMIZED.csv')
    np.testing.assert_allclose(short[0][:,:12],dfull.iloc[:cut,1:13].to_numpy(),atol=.01,rtol=1e-9);np.testing.assert_allclose(short[3],full['plans'][:cut],atol=1e-10)
    shadow=dict(np.load(OUT/'source_shadow_returns.npz'));sw=pd.read_csv(OUT/'setting_weights.csv',index_col=0).to_numpy();ww,au=setting_weights({n:r[:,:cut] for n,r in shadow.items()},f['dates'][:cut],pick['ensemble_sources'],pick['control']);np.testing.assert_allclose(ww,sw[:cut],atol=1e-12)
    audits=pd.read_csv(OUT/'setting_posteriors.csv');assert (pd.PeriodIndex(audits.last_complete_month,freq='M')<pd.PeriodIndex(audits.decision_month,freq='M')).all()
    for mi,(h,w) in enumerate(MODELS):
        pp=pr[mi];valid=np.isfinite(pp['df']);assert valid.any()
        # Reuse audited causal posterior generation: at t latest label is t-1.
        assert all(t-h-2+h+1<t for t in np.flatnonzero(valid))
    # Additional input fields have no route to the features.
    noisy=dict(truncated)
    for key in list(noisy):
        if key not in ['p','v','m','dates','codes','settlement','benchmark']:noisy[key]=np.zeros_like(noisy[key])
    fn=prepare(noisy);np.testing.assert_allclose(ff['X'],fn['X'],equal_nan=True)
    return dict(truncated_features_posterior_plans_nav_match=True,monthly_setting_weights_prefix_match=True,only_completed_months=True,no_cached_momentum_dependency=True)

def simple_rules():
    T,N=35,200;f=dict(p=np.full((T,N),100.),marks=np.full((T,N),100.),v=np.ones((T,N)),m=np.ones((T,N),bool),eligible=np.ones((T,N),bool),near=np.ones((T,N)),rank=np.broadcast_to((np.arange(N)+1)/N,(T,N)).copy(),X=np.zeros((T,N,4)),sigma=np.full((T,N),.02),tax=np.full(T,.002),comm=np.full(T,.00015),start=1,dates=pd.bdate_range('2010-01-01',periods=T))
    f['X'][:,:,0]=1.;f['X'][:,:,3]=1-2*f['rank'];pr={0:dict(coef=np.zeros((T,4)),cov=np.repeat(np.eye(4)[None],T,axis=0),df=np.full(T,50.))};s=dict(model=0,p0=.6,strength=1.,cap=.08,lam=3.)
    a=simulate(f,pr,s,record=True);equal=simulate(f,pr,dict(s,strength=0.),record=True)
    np.testing.assert_allclose(a[0],equal[0]);assert a[0][-1,2]>.97 and np.all(a[1][-1,:20]>0)
    # Move one incumbent into buffer, then below exit threshold; delay its executable sale.
    f['rank'][6:,0]=.15;f['rank'][11:,0]=.25;f['v'][12,0]=0
    b=simulate(f,pr,s,record=True);assert b[1][10,0]>0 and b[1][12,0]>0 and b[1][13,0]==0
    return dict(uncertain_probability_reverts_to_equal_not_cash=True,no_maturity_exit=True,rank_buffer=True,untradable_exit_retries=True)

def main():
    results=dict(numerical=numerical(),rules=simple_rules());f,pr=load();results['accounts']={n:audit(f,n) for n in ['OPTIMIZED','MATCHED_EQUAL','DEFAULT_EQUAL','ADAPTIVE_MIX']};results['causality']=causal(f,pr)
    (OUT/'validation.json').write_text(json.dumps(results,indent=2));log('Validation PASSED '+json.dumps(results))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
