from study import OUT,MODELS,log
from posterior import prepare,forecast
import numpy as np
from threadpoolctl import threadpool_limits

def main():
    f=prepare();data={}
    for h,w in MODELS:
        pred=forecast(f,h,w,save=False)
        for key in ['coef','cov','df']:data[f'{h}_{w}_{key}']=pred[key]
        if (h,w)==(20,1008):np.savez_compressed(OUT/'legacy_prediction.npz',**pred)
        log(f'Rebuilt causal posterior H{h}/W{w}')
    np.savez_compressed(OUT/'posterior_summary.npz',**data)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
