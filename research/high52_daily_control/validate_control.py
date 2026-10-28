from control_run import *

def main():
    f,pr=load();start=int(f['dates'].searchsorted('2017-01-02'));checks={}
    for name in ['UNCONTROLLED','CONTROLLED','CONTROLLED_EQUAL','DEFAULT_RULES']:
        a=dict(np.load(OUT/'audit'/f'{name}.npz'));d=pd.read_csv(OUT/'daily'/f'{name}.csv');b=pd.read_csv(OUT/'budgets'/f'{name}.csv');cash=1e7;qty=np.zeros(len(f['codes']));err=0.
        for t in range(len(d)):
            tr=a['trades'][t];rec=a['recoveries'][t];cash+=rec@f['marks'][t];qty-=rec;ix=abs(tr)>1e-9
            assert not ix.any() or (t>start and np.all(f['v'][t,ix]>0) and np.all(np.isfinite(f['p'][t,ix])))
            buy=np.maximum(tr,0)@np.nan_to_num(f['p'][t]);sell=np.maximum(-tr,0)@np.nan_to_num(f['p'][t]);cash+=sell*(1-f['comm'][t]-.0005-f['tax'][t])-buy*(1+f['comm'][t]+.0005);qty+=tr
            assert cash>-.001 and qty.min()>-1e-8
            np.testing.assert_allclose(qty,a['positions'][t],atol=1e-7)
            err=max(err,abs(cash+qty@f['marks'][t]-d.nav.iloc[t]))
        assert err<.1 and (d.review.iloc[start:]==1).all()
        np.testing.assert_allclose(b.ordinary_turnover+b.emergency_turnover+b.initial_turnover,d.turnover,atol=1e-10)
        np.testing.assert_allclose(b.ordinary_turnover.rolling(20,min_periods=1).sum(),b.rolling20_ordinary,atol=1e-10)
        if name!='UNCONTROLLED':
            limit=.5 if name=='DEFAULT_RULES' else 1.
            assert b.ordinary_turnover.max()<=.1+1e-9 and b.rolling20_ordinary.max()<=limit+1e-9 and d.n_holdings.max()<=20
        checks[name]=dict(max_nav_error=err,daily_review=True,account_reconciled=True,budget_reconciled=True)
    np.testing.assert_allclose(pd.read_csv(OUT/'daily/UNCONTROLLED.csv').iloc[:,1:],pd.read_csv(OUT.parent/'high52_daily/daily/DAILY_TUNED.csv').iloc[:,1:],atol=.001,rtol=1e-9)
    checks['previous_daily_baseline_reproduced']=True
    rule=json.loads((OUT/'selection.json').read_text())['selected'];cut=int(f['dates'].searchsorted('2024-01-02'))
    short=run(f,pr,rule,start=start,stop=cut,record=True);full=np.load(OUT/'audit/CONTROLLED.npz')
    np.testing.assert_allclose(short[3][:cut],full['plans'][:cut],atol=1e-10)
    np.testing.assert_allclose(short[0][:cut],pd.read_csv(OUT/'daily/CONTROLLED.csv').iloc[:cut,1:],atol=.001,rtol=1e-9)
    checks['stop_at_2023_prefix_matches']=True
    checks['posterior_causality']='Unchanged posterior; raw future-data truncation test in ../high52_daily/validation.json'
    T,N=20,200;p=np.full((T,N),100.);near=np.ones((T,N));rank=np.broadcast_to((np.arange(N)+1)/N,(T,N)).copy();x=np.zeros((T,N,4));x[:,:,0]=1
    def synthetic():
        return run_kernel.py_func(p,p,np.ones_like(p),np.ones_like(p,dtype=bool),np.ones_like(p,dtype=bool),near,rank,x,np.full_like(p,.02),np.full(T,.002),np.full(T,.00015),np.zeros((T,4)),np.repeat(np.eye(4)[None],T,0),np.full(T,50.),5,.55,.5,.1,3.,1,T,.0005,1.,True,None,1,.88,.25,2,3,.005,.2,.1,1.)
    near[3:6,0]=.87;r=synthetic()
    assert r[3][3,0]>0 and r[3][4,0]==0 and r[1][4,0]>0 and r[1][5,0]<1e-8
    assert np.all(r[1][5:9,0]<1e-8) and r[1][9,0]>0
    near[3:,1:11]=.7;r=synthetic();assert r[6][4,1]>.1 and np.max(r[6][:,0])<=.1+1e-9
    checks['confirmation_cooldown_next_close_and_emergency_exception']=True
    (OUT/'validation.json').write_text(json.dumps(checks,indent=2));log('CONTROL VALIDATION PASSED')

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
