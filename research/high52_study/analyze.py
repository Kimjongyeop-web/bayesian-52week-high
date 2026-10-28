from engine import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
import html,zipfile

PERIODS={'test':('2017-01-01','2021-12-31'),'deployment':('2022-01-01','2026-07-31'),'recent3':('2023-08-01','2026-07-31'),'2026YTD':('2026-01-01','2026-07-31')}
LABELS={'ROLL_HIGH_EW':'신고가 단독 · 매일 선택 · 동일비중','ROLL_HIGH_WEIGHT':'신고가 단독 · 매일 선택 · 비중 선택','ROLL_MOM_EW':'모멘텀 단독 · 매일 선택 · 동일비중','ROLL_MOM_WEIGHT':'모멘텀 단독 · 매일 선택 · 비중 선택','ROLL_JOINT_WEIGHT':'모멘텀+신고가 · 매일 선택','FIXED_HIGH_EW':'신고가 고정 · 120일 · 동일비중','FIXED_HIGH_WEIGHT':'신고가 고정 · 120일 · 변동성 역수','FIXED_MOM_WEIGHT':'모멘텀 고정 · 40일 · 변동성 역수','FIXED_JOINT_WEIGHT':'결합 고정 · 60일 · 변동성 역수','MID_HIGH_TILT':'모멘텀 보유 중간에 신고가로 비중조절','MATCHED_HIGH_IV':'신고가 동일 선택경로 + 변동성 역수','MATCHED_MOM_IV':'모멘텀 동일 선택경로 + 변동성 역수','KOSPI':'KOSPI 가격지수','KOSPI200':'KOSPI200 가격지수','OLD_MOM':'이전 분석 · 모멘텀','OLD_JOINT':'이전 분석 · 모멘텀+단기반전'}

def bootstrap(a,b,block,B=2000,seed=625):
    rng=np.random.default_rng(seed);n=len(a);means=[];sharpes=[];cagrs=[]
    for _ in range(B):
        starts=rng.integers(0,n,size=int(np.ceil(n/block)));ix=((starts[:,None]+np.arange(block))%n).ravel()[:n]
        x=a[ix];y=b[ix];means.append((x.mean()-y.mean())*252)
        sharpes.append((x.mean()/x.std(ddof=1)-y.mean()/y.std(ddof=1))*np.sqrt(252))
        cagrs.append(np.expm1(np.log1p(x).mean()*252)-np.expm1(np.log1p(y).mean()*252))
    return dict(annual_mean_diff=(a.mean()-b.mean())*252,mean_ci_low=np.quantile(means,.025),mean_ci_high=np.quantile(means,.975),sharpe_diff=(a.mean()/a.std(ddof=1)-b.mean()/b.std(ddof=1))*np.sqrt(252),sharpe_ci_low=np.quantile(sharpes,.025),sharpe_ci_high=np.quantile(sharpes,.975),cagr_ci_low=np.quantile(cagrs,.025),cagr_ci_high=np.quantile(cagrs,.975))

def main():
    f=prepare();dates=f['dates'];T=len(dates);paths={};summary=[]
    for fp in (OUT/'daily').glob('*.csv'):
        if fp.stem.startswith(('ROLL_','FIXED_','FRESH_','MID_')) or fp.stem in ['KOSPI','KOSPI200']:
            paths[fp.stem]=pd.read_csv(fp,index_col=0,parse_dates=True)
    for g in ['HIGH','MOM']:
        schedule=np.load(OUT/'schedules'/f'ROLL_{g}_EW.npy');adjusted=schedule.copy()
        for i in np.unique(schedule[schedule>=0]):adjusted[schedule==i]=int(i)+1
        name=f'MATCHED_{g}_IV';res=run(f,adjusted,start='2017-01-02',record=True);export(name,res,f,True)
        paths[name]=pd.DataFrame(res[0],index=dates,columns=COLS)
    oldroot=SOURCE.parent
    for name,fn in [('OLD_MOM','DEPLOY_M_H5_W756_G5'),('OLD_JOINT','DEPLOY_J_H5_W756_G20')]:
        old=pd.read_csv(oldroot/'daily'/f'{fn}.csv',index_col=0,parse_dates=True).reindex(dates)
        d=pd.DataFrame(0.,index=dates,columns=COLS);d.nav=old.nav.ffill().fillna(1e7);d.net_return=old.net_return.fillna(0);d.exposure=old.exposure.fillna(0)
        for col in ['turnover','commission','sale_tax','cash','writeoff']:
            if col in old:d[col]=old[col].fillna(0)
        d.slippage_impact=old.slippage.fillna(0)+old.impact.fillna(0);paths[name]=d
        d.to_csv(OUT/'daily'/f'{name}.csv')
    for name,d in paths.items():
        for per,(lo,hi) in PERIODS.items():
            if name.startswith('FRESH') and per=='test':continue
            summary.append(dict(name=name,period=per,**metrics(d.to_numpy(),dates,lo,hi)))
    perf=pd.DataFrame(summary);perf.to_csv(OUT/'final_performance.csv',index=False)
    years=[]
    for name,d in paths.items():
        for year in range(2017,2027):
            if name.startswith('FRESH') and year<2022:continue
            x=d.loc[f'{year}-01-01':f'{year}-12-31'];r=x.net_return.to_numpy();w=np.r_[1,np.cumprod(1+r)]
            years.append(dict(name=name,year=year,total_return=w[-1]-1,mdd=np.min(w/np.maximum.accumulate(w)-1),exposure=x.exposure.mean()))
    pd.DataFrame(years).to_csv(OUT/'annual_results.csv',index=False)
    comparisons=[('ROLL_HIGH_WEIGHT','ROLL_MOM_WEIGHT'),('ROLL_JOINT_WEIGHT','ROLL_MOM_WEIGHT'),('MID_HIGH_TILT','ROLL_MOM_WEIGHT'),('MATCHED_HIGH_IV','ROLL_HIGH_EW'),('MATCHED_MOM_IV','ROLL_MOM_EW'),('FIXED_HIGH_WEIGHT','KOSPI200'),('FIXED_JOINT_WEIGHT','FIXED_MOM_WEIGHT'),('ROLL_HIGH_WEIGHT','KOSPI200')]
    infer=[]
    for a,b in comparisons:
        aa=paths[a].loc['2022-01-01':].net_return.to_numpy();bb=paths[b].loc['2022-01-01':].net_return.to_numpy()
        for block in [20,60]:infer.append(dict(treatment=a,control=b,block=block,**bootstrap(aa,bb,block)))
    pd.DataFrame(infer).to_csv(OUT/'paired_bootstrap.csv',index=False)
    selections=[]
    for fp in (OUT/'schedules').glob('*.csv'):
        z=pd.read_csv(fp,parse_dates=['decision_date']);z=z[z.decision_date>='2022-01-01']
        selected=z.selected;hold=selected.str.extract(r'_H(\d+)_')[0].fillna('CASH')
        for h,count in hold.value_counts().items():selections.append(dict(name=fp.stem,hold=h,days=int(count),fraction=float(count/len(z))))
    pd.DataFrame(selections).to_csv(OUT/'rolling_horizon_frequency.csv',index=False)
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2,'font.size':10})
    chosen=['KOSPI200','FIXED_HIGH_WEIGHT','FIXED_JOINT_WEIGHT','ROLL_HIGH_WEIGHT','ROLL_MOM_WEIGHT','ROLL_JOINT_WEIGHT','MID_HIGH_TILT']
    fig,ax=plt.subplots(2,1,figsize=(12,9),sharex=True,gridspec_kw={'height_ratios':[2.1,1]})
    for name in chosen:
        x=paths[name].loc['2022-01-01':];nav=(1+x.net_return).cumprod();ln=ax[0].plot(x.index,(nav-1)*100,label=LABELS[name],lw=2 if name.startswith('FIXED') else 1.5)[0]
        ax[1].plot(x.index,(nav/np.maximum.accumulate(np.r_[1,nav])[1:]-1)*100,color=ln.get_color(),lw=1.2)
    ax[0].set_title('2022.01–2026.07 | 비용 차감 백테스트 누적수익률',loc='left',fontsize=16,fontweight='bold');ax[0].set_ylabel('누적수익률 (%)');ax[1].set_ylabel('고점 대비 낙폭 (%)');ax[0].legend(fontsize=9,ncol=2,loc='upper left')
    fig.text(.07,.015,'2022년 직전 NAV를 0%로 정규화. 기존 보유분 승계. 지수는 비용·배당 없는 가격지수. 실제 계좌 성적이 아님.',fontsize=9,color='#52606d');fig.tight_layout(rect=[0,.035,1,1]);fig.savefig(OUT/'net_returns.png');plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(12,4.7))
    allres=pd.read_csv(OUT/'all_96_results.csv');colors=['#2257b5','#13876e','#d3871c']
    for fam,col in zip(['NEAR10','BREAK20','MOM20'],colors):
        for k,period in enumerate(['development','deployment']):
            z=allres[(allres.family==fam)&(allres.weight=='IV60')&(allres.period==period)].sort_values('H')
            ax[k].plot(z.H,z.cagr*100,marker='o',label=fam,color=col)
    ax[0].set_title('학습 2009–2016 | 사전 선택 근거');ax[1].set_title('후속 2022–2026.07 | 모든 기간 공개')
    for a in ax:a.set_xlabel('묶음별 보유기간 (거래일)');a.set_ylabel('연복리수익률 (%)');a.set_xticks(HS);a.legend()
    fig.tight_layout();fig.savefig(OUT/'horizon_comparison.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,3.8))
    for name in ['ROLL_HIGH_WEIGHT','ROLL_MOM_WEIGHT','ROLL_JOINT_WEIGHT']:
        x=paths[name].loc['2022-01-01':];ax.plot(x.index,x.exposure*100,label=LABELS[name],lw=1)
    ax.set_ylim(0,110);ax.set_ylabel('주식 비중 (%)');ax.set_title('매일 평가하되, 조건 불충족 시 새 매수 중단 · 기존 묶음은 만기까지 유지',loc='left');ax.legend(ncol=3,fontsize=9);fig.tight_layout();fig.savefig(OUT/'equity_exposure.png');plt.close(fig)
    (OUT/'analysis_snapshot.json').write_text(json.dumps({'labels':LABELS,'rows':perf[perf.period=='deployment'].to_dict('records'),'latest_selected':{p.stem:pd.read_csv(p).iloc[-1]['selected'] for p in (OUT/'schedules').glob('*.csv')}},ensure_ascii=False,indent=2),encoding='utf8')
    log('Statistics and charts complete')

if __name__=='__main__':main()
