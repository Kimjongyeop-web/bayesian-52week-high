from daily_run import *

def audit(f,name):
    a=np.load(OUT/'audit'/f'{name}.npz');tr=a['trades'];rec=a['recoveries'];qty=np.zeros(len(f['codes']));cash=1e7;d=pd.read_csv(OUT/'daily'/f'{name}.csv');err=0.;qe=0.;start=int(f['dates'].searchsorted('2017-01-02'))
    for t in range(len(d)):
        cash+=rec[t]@f['marks'][t];qty-=rec[t];ix=abs(tr[t])>1e-9
        assert not ix.any() or (t>start and np.all(f['v'][t,ix]>0) and np.all(np.isfinite(f['p'][t,ix])))
        buy=np.maximum(tr[t],0)@np.nan_to_num(f['p'][t]);sell=np.maximum(-tr[t],0)@np.nan_to_num(f['p'][t]);cash+=sell*(1-f['comm'][t]-.0005-f['tax'][t])-buy*(1+f['comm'][t]+.0005);qty+=tr[t]
        assert cash>-.001 and qty.min()>-1e-8
        qe=max(qe,abs(qty-a['positions'][t]).max());err=max(err,abs(cash+qty@f['marks'][t]-d.nav.iloc[t]))
    assert err<.1 and qe<1e-7
    if name!='PREVIOUS5':assert (d.review.iloc[start:]==1).all()
    return dict(max_nav_error=float(err),max_quantity_error=float(qe),review_days=int(d.review.sum()),every_day_review=name!='PREVIOUS5',no_untradable_fills=True,nonnegative_cash=True)

def main():
    f,pr=load();start=int(f['dates'].searchsorted('2017-01-02'));result={}
    names=['PREVIOUS5','DAILY_SAME','DAILY_EQUAL','DAILY_TUNED','TUNED_EQUAL'];result['accounts']={n:audit(f,n) for n in names};log('Account audits passed')
    previous=OUT.parent/'high52_bayes_optimized/daily/OPTIMIZED.csv'
    if previous.exists():
        np.testing.assert_allclose(pd.read_csv(previous).iloc[:,1:].to_numpy(),pd.read_csv(OUT/'daily/PREVIOUS5.csv').iloc[:,1:].to_numpy(),atol=.001,rtol=1e-9);result['previous5_reproduced']=True
    selection=json.loads((OUT/'selection.json').read_text());s=selection['selected'];cut=int(f['dates'].searchsorted('2024-01-02'));raw=dict(np.load(OUT/'inputs.npz'));small={k:v[:cut] if v.ndim and v.shape[0]==len(f['dates']) and k!='codes' else v for k,v in raw.items()};ff=prepare(small);ff['start']=f['start'];h,w=MODELS[s['model']];pred=forecast(ff,h,w,save=False)
    for k in ['coef','cov','df']:np.testing.assert_allclose(pred[k],pr[s['model']][k][:cut],equal_nan=True,atol=1e-10,rtol=1e-9)
    short=simulate(ff,{s['model']:pred},s,start=start,record=True);full=np.load(OUT/'audit/DAILY_TUNED.npz');np.testing.assert_allclose(short[3],full['plans'][:cut],atol=1e-10);np.testing.assert_allclose(short[0],pd.read_csv(OUT/'daily/DAILY_TUNED.csv').iloc[:cut,1:].to_numpy(),atol=.001,rtol=1e-9);result['future_truncated_daily_posterior_plans_nav_match']=True
    # Explicitly test a signal change between old scheduled review dates.
    T,N=20,200;sf=dict(p=np.full((T,N),100.),marks=np.full((T,N),100.),v=np.ones((T,N)),m=np.ones((T,N),bool),eligible=np.ones((T,N),bool),near=np.ones((T,N)),rank=np.broadcast_to((np.arange(N)+1)/N,(T,N)).copy(),X=np.zeros((T,N,4)),sigma=np.full((T,N),.02),tax=np.full(T,.002),comm=np.full(T,.00015),start=1,dates=pd.bdate_range('2010-01-01',periods=T));sf['X'][:,:,0]=1;sp={0:dict(coef=np.zeros((T,4)),cov=np.repeat(np.eye(4)[None],T,axis=0),df=np.full(T,50.))};spec=dict(model=0,p0=.55,strength=1.,cap=.1,lam=3.)
    def smallrun(interval=1):
        args=[sf[k] for k in ['p','marks','v','m','eligible','near','rank','X','sigma','tax','comm']]
        return run_kernel.py_func(*args,sp[0]['coef'],sp[0]['cov'],sp[0]['df'],5,.55,1.,.1,3.,1,T,.0005,1.,True,None,interval)
    stable=smallrun();assert np.all(stable[0][1:,13]==1) and np.all(stable[1][-1,:20]>0)
    assert np.all(stable[4][1:,:20]==.5)
    sf['rank'][3:,0]=.25
    daily=smallrun();weekly=smallrun(5)
    assert daily[3][3,0]==0 and daily[1][3,0]>0 and daily[1][4,0]==0 and weekly[1][4,0]>0
    sf['v'][4,0]=0;sf['rank'][4,0]=np.nan
    blocked=smallrun()
    assert blocked[1][4,0]>0 and blocked[1][5,0]==0
    result['midcycle_exit_next_close']=True;result['constant_probability_holds_existing_names']=True;result['blocked_fill_not_fabricated']=True
    (OUT/'validation.json').write_text(json.dumps(result,indent=2));log('Daily validation PASSED '+json.dumps(result))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
