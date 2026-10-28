from engine import *

def main():
    f=prepare();idx=int(f['dates'].searchsorted('2017-01-02'));picks=json.loads((OUT/'fixed_selection.json').read_text());rows=[]
    for name in ['HIGH_WEIGHT','JOINT_WEIGHT','MOM_WEIGHT']:
        s=next(i for i,x in enumerate(SPECS) if x['name']==picks[name]['name'])
        for offset in range(5):
            start=str(f['dates'][idx+offset].date());d=run(f,np.full(len(f['dates']),s),start=start)[0]
            rows.append(dict(name='FIXED_'+name,offset=offset,start=start,**metrics(d,f['dates'],'2022-01-01','2026-07-31')))
    pd.DataFrame(rows).to_csv(OUT/'entry_phase_sensitivity.csv',index=False)
    print('Entry phase robustness completed')
if __name__=='__main__':main()
