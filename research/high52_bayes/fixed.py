from model import *
from fixed_kernel import run_account

def weights(f):
    T,N=f['p'].shape;w=np.zeros((1,T,N),np.float32)
    for t in range(T):
        cap=max(1,int(np.floor(f['eligible'][t].sum()*.1)));ids=np.flatnonzero(f['eligible'][t]&(f['near'][t]>=.90))
        ids=ids[np.argsort(-f['near'][t,ids],kind='stable')[:cap]]
        if len(ids):w[0,t,ids]=1/len(ids)
    return w

def run(f,H,start,prepared=None):
    w=weights(f) if prepared is None else prepared
    return run_account(f['p'],f['marks'],f['v'],f['m'],w,f['tax'],f['comm'],np.zeros(len(f['dates']),np.int64),np.array([H]),np.array([0]),start,len(f['dates']))
