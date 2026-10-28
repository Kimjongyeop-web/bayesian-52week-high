from model import *
from run import simulate

def main():
    T,N=28,200;prices=np.full((T,N),100.);ranks=np.broadcast_to(np.arange(1,N+1)/N,(T,N)).copy();dates=pd.bdate_range('2010-01-01',periods=T)
    f=dict(dates=dates,codes=np.array([str(j).zfill(6) for j in range(N)]),p=prices.copy(),marks=prices.copy(),v=np.ones((T,N)),m=np.ones((T,N),bool),near=np.ones((T,N)),rank=ranks,eligible=np.ones((T,N),bool),sigma=np.full((T,N),.02),X=np.ones((T,N,4)),factor=np.full((T,N),.01),iv=np.full((T,N),.0001),tax=np.full(T,.002),comm=np.full(T,.00015),start=1)
    pred=dict(mu=np.full((T,N),.01),se=np.full((T,N),.01),predscale=np.full((T,N),.05),df=np.full(T,100.),cov=np.repeat((np.eye(4)*.0001)[None],T,axis=0))
    preds={(5,504):pred}
    base=simulate(f,preds,0,mode='buffer',band=True,record=True);orders=pd.DataFrame(base[1])
    assert len(orders)==20 and np.all(orders[0]==2)
    assert np.all(base[3][-1,:20]>0)
    # Rank11-20% is hold zone, although it does not qualify for a new purchase.
    f['rank'][6:,5]=.15
    hold=simulate(f,preds,0,mode='buffer',band=True,record=True);assert hold[3][10,5]>0
    # Crossing20% at a review creates a sell. Zero volume postpones, not fabricates a fill.
    f['rank'][11:,5]=.25;f['v'][12,5]=0.
    ex=simulate(f,preds,0,mode='buffer',band=True,record=True)
    assert ex[3][12,5]>0 and ex[3][13,5]<1e-7
    assert any(x[0]==13 and x[2]==5 and x[3]<0 for x in ex[1])
    # An overweight survivor is reduced; no liquidation just because five days passed.
    f['p'][6:,0]=200.;f['marks'][6:,0]=200.
    part=simulate(f,preds,0,mode='buffer',band=True,record=True)
    assert 0<part[3][7,0]<part[3][6,0]
    out=dict(no_mandatory_5day_exit=True,rank_buffer_preserves_holdings=True,untradable_exit_retries=True,overweight_is_partial_reduction=True)
    (OUT/'rule_tests.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))

if __name__=='__main__':main()
