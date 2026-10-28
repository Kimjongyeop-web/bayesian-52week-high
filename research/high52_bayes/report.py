from pathlib import Path
import base64, html, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

OUT=Path(__file__).resolve().parent
TITLE='신고가_베이지안_보유전략_보고서.html'
NAMES={'FIXED5':'기존 5일 만기','FIXED120':'기존 120일 만기','BUFFER':'선별 보유·균등비중','BUFFER_BAND':'선별 보유 + 거래 억제','BAYES':'베이지안 + 거래 억제','BAYES_NOBAND':'베이지안·거래 억제 없음','ROLL_BAYES':'베이지안 설정 롤링','KOSPI200':'KOSPI200 가격지수'}
COLORS={'FIXED5':'#798ca1','BUFFER_BAND':'#177a72','BAYES':'#a85629','KOSPI200':'#244e8f'}
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#34485a','text.color':'#24394b','axes.edgecolor':'#c8d2da','grid.alpha':.2,'figure.facecolor':'white','savefig.facecolor':'white'})

def table(x):
    return '<div class="table-wrap">'+x.to_html(index=False,escape=True,border=0,classes='data')+'</div>'

def pct(x):return f'{x*100:,.2f}%'
def frame(file):return pd.read_csv(OUT/file)
def embed(file,alt):
    return f'<img src="data:image/png;base64,{base64.b64encode((OUT/file).read_bytes()).decode()}" alt="{alt}">'

def performance(per,keys):
    x=frame('performance.csv');x=x[x.period.eq(per)].set_index('name').loc[keys].reset_index()
    y=pd.DataFrame({'전략':x.name.map(NAMES),'누적수익':x.total_return.map(pct),'연복리':x.cagr.map(pct),'최대낙폭':x.mdd.map(pct),'Sharpe':x.sharpe.map(lambda v:f'{v:.3f}'),'평균 주식비중':x.exposure.map(pct),'연간 매매액/NAV':x.annual_turnover.map(lambda v:f'{v:.2f}배'),'연환산 비용률':x.annual_cost_rate.map(pct)})
    return table(y)

def figures():
    series={n:pd.read_csv(OUT/'daily'/f'{n}.csv',index_col=0,parse_dates=True).loc['2022-01-01':] for n in COLORS}
    fig,ax=plt.subplots(2,1,figsize=(11,7.7),sharex=True,gridspec_kw={'height_ratios':[2,1]},layout='constrained')
    for n,d in series.items():
        nav=(1+d.net_return).cumprod();dd=nav/np.maximum.accumulate(np.r_[1,nav])[1:]-1
        ax[0].plot(d.index,nav,label=NAMES[n],color=COLORS[n],lw=1.6);ax[1].plot(d.index,dd,color=COLORS[n],lw=1.1)
    ax[0].set_title('같은 평가기간, 다른 투자 노출',loc='left',fontweight='bold',fontsize=15,pad=15)
    ax[0].set_ylabel('2021년 말 자산 = 1');ax[0].legend(loc='upper left',frameon=False,ncol=2)
    ax[1].set_ylabel('최대점 대비 하락');ax[1].yaxis.set_major_formatter(PercentFormatter(1));[a.grid(True) for a in ax]
    fig.savefig(OUT/'performance.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(2,1,figsize=(11,5.7),sharex=True,layout='constrained')
    for n in ['BUFFER_BAND','BAYES']:
        d=series[n];ax[0].plot(d.index,d.exposure,color=COLORS[n],label=NAMES[n],lw=1)
    ax[0].set_title('낮은 낙폭에는 현금 대기가 크게 기여',loc='left',fontweight='bold',fontsize=15,pad=15)
    ax[0].set_ylabel('실제 주식비중');ax[0].set_ylim(-.02,1.05);ax[0].yaxis.set_major_formatter(PercentFormatter(1));ax[0].legend(frameon=False,ncol=2)
    for n in ['FIXED5','BUFFER_BAND','BAYES']:
        d=series[n];cost=d[['commission','sale_tax','execution_cost']].sum(axis=1)/d.nav
        ax[1].plot(d.index,cost.rolling(252,min_periods=60).mean()*252,color=COLORS[n],label=NAMES[n],lw=1.3)
    ax[1].set_ylabel('최근 252일 비용률\n연환산');ax[1].yaxis.set_major_formatter(PercentFormatter(1));ax[1].legend(frameon=False,ncol=3);[a.grid(True) for a in ax]
    fig.savefig(OUT/'exposure_cost.png',dpi=150);plt.close(fig)
    ph=frame('phase_sensitivity.csv');fig,ax=plt.subplots(figsize=(10.5,3.8),layout='constrained')
    for n in ['FIXED5','BUFFER_BAND','BAYES']:
        z=ph[ph.name.eq(n)];ax.plot(z.offset,z.cagr,marker='o',label=NAMES[n],color=COLORS[n],lw=1.6)
    ax.axhline(0,color='#b5bdc6',lw=.8);ax.set_xticks(range(5));ax.set_xlabel('공통 시작일에서 늦춘 거래일 수 (최적 날짜를 선택하지 않음)');ax.set_ylabel('2022–2026.07 연복리');ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(frameon=False,ncol=3);ax.grid(True)
    ax.set_title('시작일을 1–4거래일 바꾸면 수익률도 크게 달라진다',loc='left',fontweight='bold',fontsize=14,pad=14)
    fig.savefig(OUT/'phase_sensitivity.png',dpi=150);plt.close(fig)

def main():
    figures()
    keys=['FIXED5','BUFFER','BUFFER_BAND','BAYES','BAYES_NOBAND','ROLL_BAYES','FIXED120','KOSPI200']
    cal=frame('probability_calibration.csv');cal['period']=cal.period.map({'development':'학습','test':'테스트','deployment':'최근 후속'})
    calt=pd.DataFrame({'구간':cal.period,'평가 후보 수':cal.observations,'예측 실현수익 확률':cal.mean_predicted.map(pct),'실제 비용초과 빈도':cal.realized_frequency.map(pct),'Brier (작을수록 좋음)':cal.brier.map(lambda v:f'{v:.4f}'),'신규 진입 확률문턱 통과':cal.entry_gate_pass_fraction.map(pct)})
    h=frame('holding_duration.csv');ht=pd.DataFrame({'전략':h.name.map(NAMES),'완료분 평균 보유일':h.weighted_mean_sessions.map(lambda v:f'{v:.2f}'),'FIFO 조각 중앙값':h.median_piece_sessions,'매도금액 중 5일 초과':h.share_sale_notional_held_over5.map(pct),'20일 초과':h.share_sale_notional_held_over20.map(pct)})
    ph=frame('phase_sensitivity.csv');pt=[]
    for n in ['FIXED5','BUFFER_BAND','BAYES']:
        z=ph[ph.name.eq(n)];pt.append({'전략':NAMES[n],'연복리 최저':pct(z.cagr.min()),'중앙값':pct(z.cagr.median()),'최고':pct(z.cagr.max()),'비용률 최저':pct(z.annual_cost_rate.min()),'최고 비용률':pct(z.annual_cost_rate.max())})
    sens=frame('sensitivity.csv');st=pd.DataFrame({'전략':sens.name.map(NAMES),'가정':sens.variant.map({'execution13bp':'체결비용 편도 13bp','execution25bp':'체결비용 편도 25bp','recovery0':'가격 소실·편출 시 회수 0%','zero_cost':'수수료·세금·체결비용 0'}),'연복리':sens.cagr.map(pct),'최대낙폭':sens.mdd.map(pct)})
    hold=pd.read_csv(OUT/'latest_portfolio.csv',dtype={'ticker':str});hp=pd.DataFrame({'종목코드':hold.ticker,'실제 비중':hold.weight.map(pct),'52주 고점 대비':hold.near.map(pct),'예상 20일 수익률':hold.expected_H_return.map(pct),'기대수익 비용초과 확률':hold.p_expected_net.map(pct),'실현수익 비용초과 확률':hold.p_realized_net.map(pct)})
    actions=pd.read_csv(OUT/'latest_review_actions.csv',dtype={'ticker':str});at=pd.DataFrame({'종목코드':actions.ticker,'판단':actions.action.map({'HOLD':'보유','EXIT_SIGNAL':'청산: 신고가 조건 이탈','BUY_NEW':'신규매수'}),'변경 전 비중':actions.held_weight.map(pct),'목표비중':actions.target_weight.map(pct)})
    candidates=frame('performance.csv');train=frame('training_selection.csv').set_index('name');cg=candidates[(candidates.period.eq('deployment'))&candidates.name.str.startswith('B_H')].copy();cg['train']=cg.name.map(train.sharpe)
    cgt=pd.DataFrame({'설정 H / W / 확률 / 위험회피':cg.name,'학습 Sharpe':cg.train.map(lambda v:f'{v:.3f}'),'최근 연복리':cg.cagr.map(pct),'최근 주식비중':cg.exposure.map(pct)})
    annual=frame('annual_results.csv');annual=annual[annual.name.isin(['FIXED5','BUFFER_BAND','BAYES','KOSPI200'])].pivot(index='year',columns='name',values='return_value').rename(columns=NAMES).reset_index();annual['year']=annual.year.map(lambda y:str(y)+(' (7월까지)' if y==2026 else ''));annual=annual.rename(columns={'year':'연도'})
    for c in annual.columns[1:]:annual[c]=annual[c].map(pct)
    src=json.loads((OUT/'sources.json').read_text(encoding='utf-8-sig'));sources=''.join(f'<li><a href="{html.escape(u,quote=True)}">{html.escape(t)}</a><br><small>{html.escape(d)}</small></li>' for t,u,d in src)
    css='''*{box-sizing:border-box}body{margin:0;background:#edf1f3;color:#203748;font-family:"Malgun Gothic",sans-serif;line-height:1.85}main{max-width:1150px;margin:30px auto;background:white;padding:48px 56px;border-radius:14px}h1{font-size:35px;line-height:1.35;margin:16px 0 20px}h2{font-size:23px;border-top:1px solid #d9e1e6;padding-top:27px;margin-top:42px}h3{font-size:18px}a{color:#176d74}p{margin:13px 0}.eyebrow{letter-spacing:.10em;font-size:12px;color:#177a72;font-weight:bold}.lead{font-size:18px}.box{padding:17px 22px;background:#eaf6f3;border-left:4px solid #177a72;border-radius:4px;margin:18px 0}.warning{background:#fff4e9;border-color:#a85629}.note,small{font-size:13px;color:#536b7d}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:12.5px;margin:14px 0;line-height:1.6}td,th{padding:10px;border-bottom:1px solid #dce4e9;text-align:right;white-space:nowrap}td:first-child,th:first-child{text-align:left}th{background:#edf4f6}img{width:100%;height:auto;display:block;margin:24px 0}.eq{font-family:"Cambria Math","Malgun Gothic",serif;background:#f4f7f9;border-radius:6px;padding:16px 20px;overflow:auto;line-height:1.9;font-size:17px}code{font-family:Consolas,monospace;font-size:13px;background:#f0f4f6;padding:2px 4px}li{margin:7px 0}nav{display:flex;gap:16px;flex-wrap:wrap;font-size:13px}details{margin:18px 0}summary{cursor:pointer;font-weight:bold}@media(max-width:720px){main{margin:0;padding:24px 18px}h1{font-size:28px}}@media print{body{background:white}main{max-width:none;margin:0;padding:8mm}h2{break-after:avoid}img,.box{break-inside:avoid}}'''
    doc=f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>52주 신고가 · 베이지안 선별 보유 검증</title><style>{css}</style></head><body><main>
<div class="eyebrow">52-WEEK HIGH · BAYESIAN RETURN MODEL · 2026.10.06</div>
<h1>만기가 아니라 신호로 보유하고,<br>확률과 비용으로 비중을 정한다</h1>
<p class="lead">모멘텀 신호를 제외하고, <b>52주 신고가 접근도만으로 신규매수·보유·추가매수·일부매도·청산을 나누는 계좌</b>를 구현했다. 베이지안 확률에 따른 비중 조절까지 적용했지만, 이번 검증은 <b>비용 절감은 확인, 베이지안 수익률 개선은 미확인</b>이라는 결과다.</p>
<div class="box warning"><b>이 구현을 바로 실전 주력 전략으로 채택할 근거는 부족하다.</b> 최근 구간에서 선별 보유와 거래 억제는 연환산 비용률을 6.31%에서 3.41%로 줄였다. 반면 학습에서 선택된 베이지안 계좌는 연복리 1.79%, 평균 주식비중 9.24%였다. 낮은 낙폭을 종목선택 능력으로 해석해서는 안 된다.</div>
<p class="note">기존 자료 그대로: 2008.01.02–2026.07.31, 과거 KOSPI200 종목 407개. 공통 계좌 시작 2009.08.07. 2026년 8–10월은 검증하지 않았다. 비용 차감 역사적 시뮬레이션이며 실시간 주문·예약 실행은 없다.</p>
<nav><a href="#result">성과</a><a href="#rules">매매 규칙</a><a href="#math">베이지안 정의</a><a href="#validation">검증</a><a href="#holdings">마지막 보유분</a><a href="#files">파일·재현</a></nav>
<h2 id="result">1. 수익률보다 먼저 구분해야 할 두 가지 효과</h2>
<p><b>2022.01–2026.07, 같은 시작 위상과 동일한 가격·수수료·세금 가정</b>에서 비교했다. 2022년에 계좌를 다시 만들지 않고 이전 보유분을 이어받아 해당 구간의 자산을 1로 정규화했다.</p>
{performance('deployment',keys)}
<p>선별 보유 + 거래 억제는 기존 5일 전략 대비 비용률이 <b>약 46%</b>, 매매금액이 약 47% 줄었다. 연복리는 16.04%에서 15.88%로 비슷했지만 개선되지는 않았다. 거래 억제가 없는 선별 보유도 15.56%였으며, 두 선별 보유 방식 모두 기존 만기 방식보다 비용이 낮았다.</p>
<p>베이지안 계좌는 거래 억제 유무에 따라 연복리 1.79%와 1.84%였다. 이 모형에는 거래 억제를 추가한다고 수익률이 개선되지 않았다. 매일 설정까지 바꾸는 롤링 계좌는 0.22%여서, 설정을 자주 재선택하는 방식도 우월하지 않았다.</p>
<p class="note">연환산 비용률 = 일별 (수수료 + 매도세 + 체결비용)/당일 NAV 평균 ×252. 연간 매매액은 매수+매도 양방향 합계다. 비용률 차이는 곧바로 연복리 차이가 아니다. Sharpe는 무위험수익률 0. KOSPI200은 배당·거래비용 없는 가격지수 참고치로, 실제 ETF의 순수익률이 아니다.</p>
{embed('performance.png','2022년 이후 비용 차감 누적성과와 낙폭')}
{embed('exposure_cost.png','주식 투자비중과 비용률')}
<p class="note">기존 만기 계좌는 이전 후보 수 산정·만기별 자금 배분을 보존했다. 신규 계좌는 격리 전 적격집합의 순위로 진입을 판단하므로, 기존 5일과 선별 보유의 차이가 보유 규칙 하나에만 기인한다고 단정할 수 없다. 거래 억제 유무 비교는 같은 진입·비중 모형을 사용했다.</p>
<h3>먼저 거친 2017–2021 테스트</h3>
{performance('test',['FIXED5','BUFFER_BAND','BAYES','ROLL_BAYES','FIXED120','KOSPI200'])}
<p>학습 2009.08–2016에서 16개 베이지안 후보의 비용 차감 Sharpe로 <b>H=20, W=1008, 신규 확률문턱 60%, 위험회피 γ=10</b>을 선택했다. 이후 하이퍼파라미터는 고정했다. 테스트에서 베이지안 연복리는 1.96%, 선별 보유 + 거래 억제는 3.79%였다. 최근 구간만의 우연한 부진으로 보기 어렵다.</p>
<details><summary>연도별 수익률 펼치기</summary>{table(annual)}<p class="note">2026년은 7월까지의 누적수익률이며 연환산 수익률이 아니다.</p></details>
<h3>5일 주기의 시작일에 민감하다</h3>
{table(pd.DataFrame(pt))}
{embed('phase_sensitivity.png','5일 거래 주기 시작일에 따른 수익률 민감도')}
<p>계좌 시작일을 0–4거래일 이동해 다섯 위상을 모두 확인했다. 선별 보유 + 거래 억제의 최근 연복리가 <b>5.59–18.61%</b>로 달라졌다. 비용 절감은 다섯 경우 모두 유지됐지만, 특정 시작일의 수익률을 일반적인 기대수익으로 사용하기 어렵다. 표의 중앙값도 다섯 계좌를 결합한 수익률이 아니다.</p>
<p class="note">이전 보고서의 5일 전략 숫자와 직접 같지 않은 이유: 이번에는 모든 모형의 초기 학습 준비가 끝난 2009.08.07에 같은 현금으로 시작하고, 2017·2022에도 계좌를 이어간다. 이에 따라 5일 진입 위상과 기존 보유분도 달라졌다. 가장 좋은 위상을 골라 대표값으로 쓰지 않았다.</p>
<h2 id="rules">2. 이제 무엇을 팔고 무엇을 남기는가</h2>
<p>이전 코드도 <b>같은 날 같은 종목의 매도·재매수는 상계하고 순주문에 비용을 부과</b>했다. 따라서 동일 종목 왕복비용을 이중으로 계산한 오류를 고친 것은 아니다. 이번 변경은 만기와 새 순위 때문에 생기던 불필요한 교체를 줄이는 운용 규칙이다.</p>
<div class="eq">A<sub>i,t</sub> = P<sub>i,t</sub> / max(P<sub>i,t−251</sub>, …, P<sub>i,t</sub>)<br>q<sub>i,t</sub> = 신고가 접근도 내림차순 순위 / 당시 데이터 적격 종목 수</div>
<p>P는 수정종가다. 52주는 <b>252거래일 종가 최고치</b>이며 장중 신고가 돌파를 뜻하지 않는다. A=0.95는 그 고점보다 5% 아래라는 뜻이다. 동률은 종목코드 순으로 정한다.</p>
<table><thead><tr><th>상황</th><th>판단과 처리</th></tr></thead><tbody>
<tr><td>신규매수</td><td>A≥0.90, 접근도 상위 10%, 베이지안 기대수익 비용초과 확률≥60%, 최적 비중&gt;0</td></tr>
<tr><td>계속 보유</td><td>A≥0.90, 상위 20% 이내, 기대수익 양수 확률≥40%; 비중 조정이 불필요하면 수량 유지</td></tr>
<tr><td>일부매도</td><td>비용·위험·불확실성을 반영한 목표비중이 줄고, 거래 억제 범위를 넘는 경우</td></tr>
<tr><td>추가매수</td><td>보유 적격 종목의 목표비중이 증가하고 비용을 감안한 효용 개선이 충분한 경우</td></tr>
<tr><td>전체청산</td><td>A&lt;0.90, 상위 20% 밖, 기대수익 양수 확률&lt;40%, 또는 최적 비중이 0</td></tr>
<tr><td>지수 편출</td><td>매일 확인하고 다음 거래 가능 종가에 청산</td></tr>
<tr><td>5일·20일·120일 경과</td><td>신규 전략에는 날짜만을 이유로 한 청산이 없음</td></tr>
</tbody></table>
<p><b>예:</b> 상위 8%에 들어 매수한 뒤 13%로 내려가도 보유한다. 23%가 되면 청산 후보가 된다. 현재 비중 6%, 목표 5.5%라면 차이 0.5%p는 거래 억제 범위 1.1%p 이내여서 유지한다. 목표가 3%로 내려가면 일부만 매도한다.</p>
<p>사후분포는 <b>매 거래일 갱신</b>하고, 일반 매매 판단은 <b>5거래일마다</b> 한다. 일중 손절 주문은 없다. t일 종가 후 판단을 t+1 종가에 실행하며, 다음 날 현금이 부족하면 매수만 비례 축소한다. 0거래량·결측 가격에는 체결하지 않고 청산 주문은 다시 시도한다. 결측 순위만으로 강제 매도하지 않는다.</p>
<p>목표는 최대 20종목, 종목별 8%, 주식비중 합계 100% 이하, 나머지는 현금(수익률 0)이다. 비중은 가격변화로 검토일 사이에 8%를 넘을 수 있다. 신규 진입을 막는 롤링 상태라도 기존 보유 종목의 관리는 계속한다.</p>
<h3>5일은 검토 간격, 20일은 예측 기간, 보유기간은 결과</h3>
{table(ht)}
<p>선택된 베이지안 모형은 20거래일 앞을 예측하지만, 최근 매도 완료분의 평균 보유기간은 <b>13.33거래일</b>이었다. 선별 보유 + 거래 억제는 <b>20.63거래일</b>이었다. 이는 최적 보유기간을 13일 또는 21일로 찾아냈다는 뜻이 아니다. 신호가 유지되면 더 오래 보유할 수 있다.</p>
<p class="note">2022년 이후 매도된 수량을 FIFO로 매수분과 연결해 매도금액으로 가중한 기간이다. 일부매도는 여러 조각으로 나뉘며, 아직 보유 중인 물량은 제외한 우측 절단 통계다. 중앙값은 조각 개수 기준으로 가중평균과 기준이 다르다.</p>
<p>최근 베이지안 검토 기록은 보유 265건, 신규매수 209건, 일부축소 54건, 추가매수 21건, 신고가 조건 청산 193건, 효용상 비중 0 청산 11건이었다. 이 설정에서는 확률 40% 미만을 직접 이유로 청산한 사례는 없었다. 이 숫자는 체결 건수가 아니라 검토 시점 판단 수다.</p>
<h2 id="math">3. 적용한 베이지안 확률 모형</h2>
<p>구현은 <b>이동 학습창을 사용하는 정규–역감마 베이지안 회귀</b>다. 종목의 미래수익률을 확률변수로 두고, 관측이 쌓이면 계수와 불확실성을 갱신한다. 계수에 별도의 확률보행을 가정한 상태공간 모형이나 연속시간 확산과정은 구현하지 않았다. 다음은 실제 코드의 정의다.</p>
<h3>① 신고가 정보만 설명변수로 사용</h3>
<div class="eq">z<sub>i,t</sub> = clip((A<sub>i,t</sub>−0.90)/0.10, −5, 1)<br>x<sub>i,t</sub> = [1, z<sub>i,t</sub>, z²<sub>i,t</sub>, 1−2q<sub>i,t</sub>]<br>y<sub>i,s</sub> = (P<sub>i,s+H+1</sub>/P<sub>i,s+1</sub>−1) / (σ<sub>i,s</sub>√H)</div>
<p>σ는 s일 전에 관측한 60일 일별 로그수익률의 표준편차이며 하한은 하루 0.5%다. 방향성 과거수익률, 12−1개월 모멘텀, 추세 필터는 없다. 과거수익률은 위험 추정과 학습의 정답에만 사용한다. 학습 표본은 당시 적격 종목 중 A≥0.80이다. 목표 y는 다음 종가부터 H일 뒤의 평가가격 수익률이며, 거래정지 끝점에서 반드시 그 가격으로 매도 가능하다는 뜻은 아니다.</p>
<h3>② 사전분포와 시간 순서를 지킨 갱신</h3>
<div class="eq">y | β, s² ∼ N(x′β, s²)<br>β | s² ∼ N(0, s²Λ₀⁻¹), &nbsp; Λ₀ = diag(16,4,4,4)<br>s² ∼ InvGamma(a₀=3, b₀=2)<br>ω<sub>i,s</sub> = 1/(N<sub>s</sub>H)<br>Λ = Λ₀ + Σωxx′, &nbsp; V=Λ⁻¹, &nbsp; m=VΣωxy<br>a = 3 + ½Σω, &nbsp; b = 2 + ½(Σωy² − m′Λm)</div>
<p>InvGamma는 밀도가 (s²)<sup>−a−1</sup> exp(−b/s²)에 비례하는 모수화다. 판단일 t에는 결과가 <b>t−1까지 끝난 표본만</b> 쓴다. 가장 최근 학습 시작점은 s=t−H−2이고, 그 이전 최대 W개 거래일을 사용한다. 유효 정답이 있는 날짜가 최소 126개일 때 시작한다.</p>
<p>종목 수와 예측기간으로 우도 가중치를 줄인 것은 같은 날짜 종목 간 상관과 H일 중첩으로 확신이 과해지는 것을 완화하려는 <b>가중 우도 근사</b>다. 정확한 시장 공동분포를 추정한 것은 아니며, 이 보정이 확률 보정을 보장하지 않는다. 창 밖 표본을 버리고 고정 사전분포에서 다시 계산하므로 모든 과거를 누적하는 단일 사후분포와도 다르다. 공액 회귀의 기본 갱신은 <a href="https://statproofbook.github.io/P/blr-post.html">정규–감마 베이지안 선형회귀 증명</a>을 참고했다.</p>
<h3>③ 두 확률을 혼동하지 않는다</h3>
<div class="eq">k = σ<sub>i,t</sub>√H, &nbsp; μ = kx′m<br>기대수익 θ | D ∼ t<sub>2a</sub>(μ, k√[(b/a)x′Vx])<br>실현수익 R | D ∼ t<sub>2a</sub>(μ, k√[(b/a)(1+x′Vx)])<br>p<sub>mean</sub> = Pr(θ &gt; c<sub>round</sub> | D)<br>p<sub>realized</sub> = Pr(R &gt; c<sub>round</sub> | D)</div>
<p>괄호의 두 번째 인자는 Student-t의 <b>척도</b>이며 표준편차 자체가 아니다. c<sub>round</sub>는 신호일 기준 왕복 수수료·매도세·체결비용의 1차 근사다. <b>신규 진입의 60%는 ‘기대수익이 비용보다 클 사후확률’이지 ‘이번 거래에서 돈을 벌 확률 60%’가 아니다.</b> 실제 가격 잡음까지 포함한 p<sub>realized</sub>는 대개 훨씬 50%에 가깝다. 기존 보유분의 확률 청산 문턱은 Pr(θ&gt;0|D)&lt;40%다.</p>
<h3>④ 비중은 수익·위험·불확실성·실제 추가 거래비용을 함께 고려</h3>
<div class="eq">max<sub>w</sub> &nbsp; μ′w − (γ/2)w′Σ<sub>H</sub>w − (1/2)w′Uw − C(w−u)<br>C(w−u) = c<sub>buy</sub>Σ(w−u)<sub>+</sub> + c<sub>sell</sub>Σ(u−w)<sub>+</sub><br>0≤w<sub>i</sub>≤0.08, &nbsp; Σw<sub>i</sub>≤1</div>
<p>u는 현재 비중, Σ<sub>H</sub>는 과거자료로 추정한 H일 수익률 공분산, U는 기대수익 추정의 사후 공분산이다. 위험은 시장요인과 종목 고유분산으로 나눠 추정하며 시장요인은 매수 방향을 정하는 신호가 아니다. 보유 종목을 그대로 유지하면 C=0이고, 과거에 이미 지불한 매수비용은 다시 부과하지 않는다. 신규 진입의 왕복 확률 문턱은 후보 필터이며, 계좌에서 이 비용을 중복 차감하지 않는다.</p>
<p>최적화 후 |목표−현재|≤max(0.5%p, 목표비중×20%)면 거래를 생략한다. 선택적 거래 전체의 예상 효용 개선이 NAV의 0.5bp 이하이면 변경을 취소한다. 조건 이탈 청산과 한도 초과 축소는 예외다. 거래 억제 적용 후에도 효용을 다시 확인하지만, 최종 비중이 거래 억제를 포함한 문제의 전역 최적해라는 보장은 없다. 검증한 최적화 오차는 거래 억제 전 볼록문제에 대한 것이다.</p>
<h2 id="validation">4. 확률과 운용 규칙은 실제로 검증했는가</h2>
{table(calt)}
<p>최근 후보의 평균 실현수익 비용초과 예측확률은 <b>50.42%</b>, 실제 빈도는 <b>46.63%</b>였다. Brier 점수는 0.2507로 언제나 50%라고 답하는 기준 0.2500보다 좋아지지 않았다. 따라서 <b>이번 확률 추정을 우수한 매매 예측기로 판단할 근거가 없다.</b> 신규 후보 중 기대수익 확률문턱을 통과한 비율도 12.36%에 그쳐 현금 대기가 길었다.</p>
<p class="note">평가 대상은 정답이 끝난 신고가 접근 상위 10% 후보의 일별 관측이며 실제 체결 거래만이 아니다. 기간이 겹치는 종목·날짜 관측이므로 표본 수를 독립 시행 수로 해석하거나 단순 이항 신뢰구간을 붙이면 안 된다. p<sub>mean</sub>은 관측할 수 없는 기대수익에 관한 확률이므로 실제 승패와 직접 보정하지 않았다.</p>
<p>60거래일 블록 부트스트랩 2,000회에서 선별 보유 + 거래 억제와 기존 5일 전략의 <b>연환산 산술평균 수익률 차이</b>는 −0.21%p, 탐색적 95% 구간은 −6.84~+6.20%p였다. 우위가 확인되지 않는다. 이는 CAGR 차이의 구간이 아니며, 지금까지의 반복 연구·모형 탐색에 대한 보정도 아니다.</p>
<p>검증 코드는 순주문의 수량·현금·NAV 독립 재구성, 다음 종가 체결, 미체결 청산 재시도, 미래 라벨 차단, 미래 데이터를 자른 뒤 신호·사후분포·보유·NAV 일치, 롤링 선택의 인과성, 캐시된 모멘텀 배열에 대한 비의존성, 별도 SLSQP 풀이와의 목적함수 일치를 확인했다. 일정 가격에서 5일 후 자동청산하지 않는지, 상위 10–20% 보유, 20% 밖 청산, 한도 초과 일부축소도 확인했다.</p>
<h3>매일 설정까지 다시 고르는 추가 실험</h3>
<p>ROLL_BAYES는 판단일 전까지 완료된 756일 학습과 252일 검증 계좌수익률로 설정을 고른다. 양쪽 평균수익이 양수인 후보 중 학습 Sharpe 상위 3개를 뽑고 검증 Sharpe로 비교한다. 새 후보의 검증 Sharpe가 현 설정보다 0.15 높다는 조건이 5일 유지될 때 바꾼다. 후보가 없으면 신규 종목 진입을 멈추고 마지막 모형으로 보유분을 관리한다. 설정 변경 자체가 전량청산을 유발하지 않는다. 최근 성과는 표에서 보듯 고정 설정보다 낫지 않았다.</p>
<details><summary>16개 후보를 모두 보기 — 최근 수익률로 다시 선택하지 않음</summary>{table(cgt)}<p>학습에서 선택된 것은 B_H20_W1008_P60_G10.0이다. 최근에 더 좋아 보이는 H5/W504/P60/G3 설정을 사후에 대표 전략으로 바꾸지 않았다. 이 설정의 최근 연복리 13.40%는 새로운 독립 검증 성과가 아니다.</p></details>
<h2>5. 비용과 데이터 가정</h2>
<p>키움 KRX 온라인 일반 수수료는 <b>매수·매도 각각 0.015%</b>, 확인된 2020.09.16–12.31 한시 인하 기간에는 0.0113604%를 사용했다. 체결비용은 슬리피지와 시장충격 합계 편도 0.05%라는 가정이다. <a href="https://www1.kiwoom.com/h/domestic/stock/VStockMainView?dummyVal=0">키움 안내</a>와 <a href="https://download.kiwoom.com/notice/notice_20200915_02_01.pdf">한시 인하 공지</a>를 기준으로 했다.</p>
<table><tr><th>양도·결제 기준 기간</th><th>매도세 합계 (농특세 포함)</th></tr><tr><td>공통 시작–2019.06.02</td><td>0.30%</td></tr><tr><td>2019.06.03–2020</td><td>0.25%</td></tr><tr><td>2021–2022</td><td>0.23%</td></tr><tr><td>2023</td><td>0.20%</td></tr><tr><td>2024</td><td>0.18%</td></tr><tr><td>2025</td><td>0.15%</td></tr><tr><td>2026</td><td>0.20%</td></tr></table>
<p>체결일과 저장된 결제일을 연결해 해당 세율을 적용했다. 2019.06.03 결제 시행은 2019.05.30 체결부터다. 2026년 비용문턱은 대략 0.015%×2+0.20%+0.05%×2=<b>0.33%</b>이고, 실제 계좌 비용은 체결 금액별로 계산한다. 개인별 양도소득세·배당세·계좌 세제는 포함하지 않았다.</p>
{table(st)}
<p>민감도 분석은 선택된 H/W/확률/위험회피를 유지하고 다시 운용했다. 특히 베이지안 모형은 비용이 커지면 확률 필터와 최적 비중도 달라진다. 따라서 동일한 주문에 비용만 더한 실험이 아니다. 한계 체결비용이 편도 13bp일 때 베이지안 연복리가 −1.32%로 바뀌어 구현의 비용 민감성도 크다.</p>
<ul><li>이전 연구에서 이미 관찰한 역사 구간이다. 이번 설계와 후보군을 계산 전에 기록했더라도 완전히 손대지 않은 최종 외부검증은 아니다.</li><li>과거 편입 이력을 사용했지만 공시 시점과 수정가격의 과거 버전까지 감사하지 않았다. 현금배당·기업행위 지급액의 반영 범위도 독립 확인하지 못해 배당 재투자 총수익률로 표시하지 않는다.</li><li>수정종가 분할 단위를 사용했다. 실제 정수 주식·호가단위·상하한가 체결 가능성·대규모 운용의 시장충격은 재현하지 못한다.</li><li>편출 후 가격 소실은 마지막 평가가격 100% 회수를 기본 가정으로 두고 0%도 비교했다. 최근 선택된 계좌의 결과가 같아도, 모든 상장폐지·기업행위가 정확히 처리됐다는 증거는 아니다.</li><li>이상 일수익률은 신규 매수·학습에서 10일 격리하며 계좌 손익을 사후 제거하지 않았다. 데이터 적격 조건과 회복 가정의 영향은 남아 있다.</li></ul>
<h2 id="holdings">6. 마지막 자료 시점의 계좌와 실제 변경 예시</h2>
<p><b>2026.07.31 기준</b> 베이지안 보유분이다. 현금은 {pct(1-hold.weight.sum())}다. 10월 현재 매수 추천이나 현재 주문 목록이 아니다. 실제 계좌 수량으로 사용할 수 없는 수정가격 기준 단위는 별도 CSV에 제공했다.</p>
{table(hp)}
<p>483650의 7월 31일 접근도는 0.834로 보유 기준보다 낮지만, 마지막 정기 판단일인 7월 29일에는 1.000이었다. 이 버전은 일반 조건을 5거래일마다 확인하므로 그 사이에는 보유가 남는다. 매일 가격 이탈 때 청산하도록 바꾸면 별도의 운용 규칙이며 이번 결과에 포함되지 않는다.</p>
<details><summary>2026.07.29 판단 → 07.30 체결한 변경 보기</summary>{table(at)}<p>033780·055550은 유지하고, 조건을 벗어난 5종목은 청산했으며, 3종목을 신규 편입했다. 전량 일괄 교체가 아니다. 표는 판단일 목표비중이고 체결가격 변화 후 실제 비중은 다를 수 있다.</p></details>
<h2 id="files">7. 사용과 재현</h2>
<p><b>현재 채택할 수 있는 결론은 ‘무조건 만기 청산 대신 선별 보유를 구현한다’까지다.</b> 베이지안 확률 비중은 구현과 검증 자료를 보관하되, 이번 설정을 실전 자금의 주력 운용으로 선택하지 않는다. 새 자료가 생겼을 때 고정된 규칙으로 모의운용을 이어가야 최근 결과를 보고 조건을 맞춘 효과와 구분할 수 있다.</p>
<p><a href="신고가_베이지안_재현자료.zip">전체 재현자료 ZIP</a> · <a href="latest_portfolio.csv">마지막 보유분 CSV</a> · <a href="latest_review_actions.csv">마지막 검토 판단 CSV</a> · <a href="performance.csv">전체 성과 CSV</a> · <a href="validation.json">검증 결과</a></p>
<p>ZIP에는 원입력 스냅샷, 코드, 사전 설계와 정정 기록, 16개 후보 결과, 일별 계좌, 실제 순주문, 판단 사유, 보유분, 사후분포와 라벨 날짜 감사 결과를 담았다. 실행 순서는 README에 있다. 모멘텀 배열이 원자료에 있더라도 모형은 지정한 원가격·거래량·편입·달력·위험 기준자료만 읽는다.</p>
<details><summary>문헌·비용 근거</summary><ol>{sources}</ol></details>
<p class="note">작성 2026.10.06 · 과거자료 검증 · 실시간 투자 적용 전 새로운 자료와 실행환경 검증 필요</p>
</main></body></html>'''
    (OUT/TITLE).write_text(doc,encoding='utf-8')
    print('Report created:',TITLE,flush=True)

if __name__=='__main__':main()
