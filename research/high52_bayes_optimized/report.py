from pathlib import Path
import base64,json,html
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
OUT=Path(__file__).resolve().parent
TITLE='신고가_베이지안_최적화_보고서.html'
NAMES={'OLD_BAYES':'이전: 절대확률 매수 제한','DEFAULT_EQUAL':'균등비중 · 8% 한도','MATCHED_EQUAL':'같은 조건 균등 · 10% 한도','OPTIMIZED':'확률 비중 조절 · 10% 한도','ADAPTIVE_MIX':'설정까지 베이지안 혼합'}
COLORS={'OLD_BAYES':'#a1aab5','MATCHED_EQUAL':'#687ea0','OPTIMIZED':'#087e79','ADAPTIVE_MIX':'#bc7434'}
def read(name):return pd.read_csv(OUT/name)
def pct(x):return f'{x*100:.2f}%'
def tab(x):return '<div class="table-wrap">'+x.to_html(index=False,border=0,escape=True)+'</div>'
def embed(name):return '<img alt="'+name+'" src="data:image/png;base64,'+base64.b64encode((OUT/name).read_bytes()).decode()+'">'
def perf(period):
    x=read('performance.csv');x=x[x.period.eq(period)].set_index('name').loc[list(NAMES)].reset_index()
    return tab(pd.DataFrame({'전략':x.name.map(NAMES),'연복리':x.cagr.map(pct),'누적수익':x.total_return.map(pct),'최대낙폭':x.mdd.map(pct),'Sharpe':x.sharpe.map(lambda v:f'{v:.3f}'),'평균 주식비중':x.exposure.map(pct),'연환산 비용률':x.annual_cost_rate.map(pct)}))

def main():
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'font.size':10,'text.color':'#203c4c','axes.labelcolor':'#203c4c','grid.alpha':.2})
    daily={n:pd.read_csv(OUT/'daily'/f'{n}.csv',index_col=0,parse_dates=True).loc['2022-01-01':] for n in COLORS}
    fig,ax=plt.subplots(2,1,figsize=(11,7),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[2,1]})
    for n,d in daily.items():
        wealth=(1+d.net_return).cumprod();ax[0].plot(d.index,wealth,label=NAMES[n],c=COLORS[n],lw=1.5);ax[1].plot(d.index,d.exposure,c=COLORS[n],lw=1.1)
    ax[0].set_title('현금 대기 해소와 추가 비중 조절 효과를 분리',loc='left',fontweight='bold',fontsize=15,pad=14);ax[0].set_ylabel('2021년 말 자산 = 1');ax[0].legend(frameon=False,ncol=2)
    ax[1].set_ylabel('실제 주식비중');ax[1].yaxis.set_major_formatter(PercentFormatter(1));[a.grid(True) for a in ax];fig.savefig(OUT/'equity_exposure.png',dpi=150);plt.close(fig)
    ph=read('phase_paired.csv');fig,ax=plt.subplots(1,2,figsize=(11,3.8),layout='constrained')
    for n in ['MATCHED_EQUAL','OPTIMIZED','ADAPTIVE_MIX']:ax[0].plot(ph.phase,ph[n],marker='o',label=NAMES[n],c=COLORS[n])
    ax[0].set_title('거래 시작일별 최근 연복리');ax[0].legend(frameon=False,fontsize=8);ax[0].yaxis.set_major_formatter(PercentFormatter(1))
    ax[1].bar(ph.phase,ph.OPTIMIZED_minus_EQUAL*100,color=COLORS['OPTIMIZED']);ax[1].set_title('같은 조건 균등 대비 추가 수익');ax[1].set_ylabel('연복리 차이 (%p)')
    for a in ax:a.set_xticks(range(5));a.set_xlabel('시작일 이동 (거래일)');a.grid(True,axis='y')
    fig.savefig(OUT/'phase_comparison.png',dpi=150);plt.close(fig)
    selection=json.loads((OUT/'selection.json').read_text());phase_table=ph.rename(columns={'phase':'시작일 이동','MATCHED_EQUAL':'균등','OPTIMIZED':'확률 비중 조절','ADAPTIVE_MIX':'설정 혼합','OPTIMIZED_minus_EQUAL':'확률 조절 − 균등','MIX_minus_EQUAL':'설정 혼합 − 균등'})
    for c in phase_table.columns[1:]:phase_table[c]=phase_table[c].map(pct)
    hold=read('holding_duration.csv');holdtable=pd.DataFrame({'전략':hold.name.map(NAMES),'완료분 평균 보유일':hold.weighted_mean_sessions.map(lambda v:f'{v:.2f}'),'조각 중앙값':hold.median_piece_sessions,'매도금액 중 5일 초과':hold.sale_value_held_over5.map(pct)})
    hp=pd.read_csv(OUT/'holdings/OPTIMIZED.csv',dtype={'ticker':str});last=pd.DataFrame({'종목코드':hp.ticker,'실제 비중':hp.weight.map(pct),'52주 고점 대비':hp.near.map(pct)})
    de=pd.read_csv(OUT/'latest_review_actions.csv',dtype={'ticker':str});de['relative_probability']=de.relative_probability.where(de.action!='EXIT',np.nan)
    det=pd.DataFrame({'종목코드':de.ticker,'판단':de.action.map({'EXIT':'조건 이탈 청산','BUY_NEW':'신규매수','ADD':'추가매수','REDUCE':'일부축소','HOLD':'유지'}),'변경 전 비중':de.held_weight.map(pct),'목표비중':de.target_weight.map(pct),'상대 기대수익 확률':de.relative_probability.map(lambda v:'—' if pd.isna(v) else pct(v))})
    search=read('parameter_search.csv').sort_values('lower10',ascending=False);top=search.head(12)
    tops=pd.DataFrame({'설정':top.name,'초기학습 월평균 초과로그수익':top.fit_mean_excess.map(pct),'내부검증 개선확률':top.p_positive.map(pct),'월평균 초과수익 하단10%':top.lower10.map(pct)})
    annual=read('annual_results.csv').pivot(index='year',columns='name',values='total_return').rename(columns=NAMES).reset_index();annual['year']=annual.year.map(lambda y:f'{y}'+(' (7월까지)' if y==2026 else ''));annual=annual.rename(columns={'year':'연도'})
    for c in annual.columns[1:]:annual[c]=annual[c].map(pct)
    sens=read('sensitivity.csv');senstab=pd.DataFrame({'전략':sens.name.map(NAMES),'가정':sens.variant.map({'execution13bp':'체결비용 편도 13bp','execution25bp':'체결비용 편도 25bp','recovery0':'가격 소실분 회수 0%'}),'연복리':sens.cagr.map(pct),'최대낙폭':sens.mdd.map(pct)})
    mix=read('setting_posteriors.csv');mix=mix[mix.decision_month==mix.decision_month.max()];mw=pd.read_csv(OUT/'setting_weights.csv').iloc[-1];mt=pd.DataFrame({'설정':mix.source,'기대 초과로그수익 양수 확률':mix.p_positive.map(pct),'목표 조합 비중':[pct(mw[n]) for n in mix.source]})
    css='''*{box-sizing:border-box}body{margin:0;background:#edf3f4;color:#233d4c;font-family:"Malgun Gothic",sans-serif;line-height:1.85}main{max-width:1150px;margin:28px auto;background:#fff;padding:46px 54px;border-radius:14px}h1{font-size:35px;line-height:1.4;margin:14px 0 18px}h2{font-size:23px;margin-top:40px;padding-top:24px;border-top:1px solid #dbe5e9}h3{font-size:18px}.eyebrow{color:#087e79;letter-spacing:.12em;font-size:12px;font-weight:bold}.lead{font-size:18px}.box{padding:17px 22px;background:#eaf7f4;border-left:4px solid #087e79;border-radius:5px;margin:18px 0}.warn{background:#fff5e8;border-color:#b77b36}.note,small{font-size:13px;color:#577081}a{color:#087570}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:12.5px;line-height:1.7;margin:15px 0}th,td{padding:10px;border-bottom:1px solid #dce5ea;text-align:right;white-space:nowrap}th{background:#edf5f6}td:first-child,th:first-child{text-align:left}img{width:100%;height:auto;display:block;margin:22px 0}.eq{padding:16px 20px;background:#f4f7f8;border-radius:5px;overflow:auto;font-family:"Cambria Math","Malgun Gothic",serif;font-size:17px;line-height:1.9}nav{display:flex;flex-wrap:wrap;gap:16px;font-size:13px}li{margin:7px 0}summary{cursor:pointer;font-weight:bold}details{margin:18px 0}code{background:#f2f5f7;padding:2px 5px;font-size:13px}@media(max-width:720px){main{margin:0;padding:22px 18px}h1{font-size:28px}}@media print{body{background:white}main{padding:8mm;margin:0;max-width:none}h2{break-after:avoid}img,.box{break-inside:avoid}}'''
    doc=f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>신고가 베이지안 확률·비중 최적화</title><style>{css}</style></head><body><main>
<div class="eyebrow">52-WEEK HIGH · PROBABILITY & ALLOCATION · 2026.10.06</div>
<h1>현금 대기를 줄이고,<br>베이지안 확률은 종목 비중에 반영</h1>
<p class="lead">모멘텀 없이 신고가 전략만 사용했다. 기존의 <b>“확률이 낮으면 매수 금지”</b>를 없애고, <b>“확신이 약하면 균등비중, 상대적으로 유리하면 비중 확대”</b>로 바꿨다. 확률 문턱·비중 강도·종목 한도·매매 억제 강도 144개 조합을 비교했다.</p>
<div class="box"><b>최근 구간 결과:</b> 확률 비중 조절의 연복리 <b>16.28%</b>, 같은 조건 균등비중 <b>14.83%</b>, 차이 <b>+1.44%p</b>. 두 계좌의 평균 주식비중은 모두 약 96.35%다. 이전 베이지안 1.79%에서 회복한 대부분은 현금 대기를 줄인 효과이며, 그 전체를 베이지안 예측 능력으로 볼 수 없다.</div>
<div class="box warn"><b>확정된 실전 최적값은 아니다.</b> 2009–2013 초기 학습에서 균등비중 대비 개선 조건을 통과한 후보가 없었다. 아래 설정은 내부검증에서 가장 유력했던 <b>비교용 후보</b>이며, 이후에도 2017–2021의 개선은 작았다. 이미 반복 연구한 데이터이므로 최근 개선을 새 외부검증의 확정적 증거로 해석하지 않는다.</div>
<p class="note">자료: 기존 407종목 스냅샷, 2008.01.02–2026.07.31. 모든 비교 계좌는 2017.01.02 동일 현금으로 시작, 2022년에 초기화하지 않음. 2026년 8–10월·실시간 주문은 포함하지 않음.</p>
<nav><a href="#settings">선택 설정</a><a href="#results">성과</a><a href="#math">수학적 정의</a><a href="#mix">설정도 베이지안 적용</a><a href="#holdings">보유·주문</a><a href="#files">파일</a></nav>
<h2 id="settings">1. 다시 설정한 확률과 종목비중</h2>
<table><tr><th>항목</th><th>이번 비교용 설정</th></tr>
<tr><td>신고가 정의</td><td>수정종가 / 최근 252거래일 최고 수정종가</td></tr>
<tr><td>신규·보유 기준</td><td>고점의 90% 이상, 신규 상위 10%, 보유 상위 20%</td></tr>
<tr><td>예측 / 학습창</td><td>향후 20거래일 / 최근 1,008개 성숙 라벨 시작일</td></tr>
<tr><td>확률 중립구간</td><td>45–55%: 상대 비중 점수 중립, 매수 차단 아님</td></tr>
<tr><td>비중 강도</td><td>지수형 기울임 계수 κ=1.0</td></tr>
<tr><td>목표 주식비중</td><td>98%, 다만 적격 종목 수·종목 한도·미체결로 낮아질 수 있음</td></tr>
<tr><td>종목별 목표 한도</td><td>10%, 최대 20종목</td></tr>
<tr><td>거래 억제</td><td>목표비중 추종 오차 + λ=3의 거래비용 벌점 최소화</td></tr>
<tr><td>갱신 / 거래 판단</td><td>사후분포 매일 갱신 / 일반 매매 5거래일마다 / 편출은 매일 확인</td></tr>
<tr><td>보유 만기</td><td>없음. 조건 이탈 시 청산, 목표 감소 시 일부축소</td></tr></table>
<p><b>55%는 성공률이나 매수 합격선이 아니다.</b> “해당 종목의 기대수익이 현재 신고가 후보들의 평균 기대수익보다 높을 확률”에 대한 중립구간 경계다. 55%보다 높으면 비중 확대 점수, 45%보다 낮으면 축소 점수를 준다. 모든 종목이 불확실하면 균등비중으로 돌아간다.</p>
<p>종목별 확률만으로 매도를 강제하지 않는다. 신규/보유 자격은 신고가 규칙으로 판단하고, 비중은 후보 전체를 함께 정규화한 뒤 비용·한도를 반영한다. 따라서 낮은 확률의 종목도 보유할 수 있고, 높은 확률이어도 이미 과대 보유 중이면 일부 매도할 수 있다.</p>
<h2 id="results">2. 비용 차감 성과와 추가 효과</h2>
<h3>최근 후속 구간: 2022.01–2026.07</h3>{perf('recent')}
<p>기존 현금 대기 문제는 해소됐다. 그러나 최대낙폭도 이전 −12.79%에서 <b>−34.21%</b>로 커졌다. 주식 투자 노출을 되살린 대가다. 비교 조건을 맞춘 균등비중의 낙폭은 −34.72%로 비슷했다.</p>
<p>원래 8% 한도를 유지한 새 균등비중 계좌도 <b>15.61%</b>였다. 이것과의 차이는 <b>+0.67%p</b>이며, 한도를 10%로 바꾸는 것 자체가 수익을 높였다는 결과는 아니다. 확률의 순수한 추가 효과를 볼 때는 같은 10% 한도·같은 비용 벌점의 14.83%와 비교해야 한다.</p>
<p class="note">비용률은 일별 (수수료+매도세+체결비용)/NAV 평균 ×252이며 연복리 수익률 차이와 같지 않다. Sharpe의 무위험수익률·현금 이자는 0. 기존 보고서의 균등전략과는 98% 목표·거래 억제 방식·계좌 초기화가 달라 직접 같은 숫자는 아니다.</p>
{embed('equity_exposure.png')}
<h3>시간 순서상 먼저 평가한 테스트: 2017–2021</h3>{perf('test')}
<p>확률 조절은 연복리 <b>3.32%</b>, 같은 조건 균등은 <b>3.25%</b>였다. 최근 구간처럼 큰 개선이 일관되게 나타난 것은 아니다. 이 구간 최대낙폭은 확률 조절 −44.17%, 균등 −43.73%로 오히려 조금 커졌다.</p>
<details><summary>2017–2026.07 전체 평가 및 연도별 수익률</summary>{perf('all_test')}{tab(annual)}<p class="note">2026년 연도별 표는 7월까지 누적수익률이며 연환산 수익률이 아니다.</p></details>
<h3>거래 시작일을 다섯 가지로 바꿔도 같은 효과인가</h3>{tab(phase_table)}
{embed('phase_comparison.png')}
<p>최근 구간에서는 다섯 시작일 모두 균등보다 높았다. 개선 폭은 <b>+0.62~+1.62%p</b>다. 다만 확률 조절 자체의 연복리는 <b>5.23~18.75%</b>로 변동이 크다. 가장 좋은 날짜를 골라 대표값으로 삼지 않았으며, 다섯 위상은 독립적인 다섯 검증표본이 아니다.</p>
<p>60일 블록 부트스트랩 2,000회에서 같은 조건 균등 대비 <b>연환산 산술평균 수익률 차이</b>의 탐색적 95% 구간은 <b>+0.12~+3.05%p</b>였다. 8% 한도 균등 대비는 <b>−0.76~+2.50%p</b>로 0을 포함했다. 144개 설정 탐색과 이전 연구의 자료 재사용을 보정하지 않은 구간이며, CAGR 차이의 신뢰구간과도 다르다.</p>
<h2>3. 무엇을 기준으로 최적화했는가</h2>
<p>H ∈ {{5,20}}, W ∈ {{504,1008}}, 중립구간 경계 p₀ ∈ {{50%,55%,60%}}, 비중 강도 κ ∈ {{0.5,1}}, 종목 한도 ∈ {{6%,8%,10%}}, 비용 벌점 λ ∈ {{1,3}}의 <b>144개</b>를 계산 전에 정했다. 같은 한도·벌점의 균등 기준 6개도 계산했다.</p>
<p>2009.08–2013을 초기 학습, 2014–2016을 내부검증으로 나누고 다섯 거래 위상에 걸친 <b>월별 균등 대비 초과로그수익</b>을 평가했다. 내부검증의 평균 초과수익을 정규–역감마 모형으로 추정하고, 그 사후분포 10% 분위수가 높은 순서로 비교했다. 월별 초과수익의 사전 평균은 0, κ₀=12, a₀=3, b₀=0.0008로, 작은 표본의 큰 성과를 0 쪽으로 줄인다.</p>
<p><b>중요한 실패 조건:</b> 초기 학습 평균 초과수익이 양수인 후보가 하나도 없었다. 프로토콜에 따라 가장 유력한 설정을 비교용으로 남겼지만, 합격 후보로 승격하지 않았다. 선택 후보의 내부검증 개선확률은 <b>75.24%</b>이고, 월평균 초과수익의 10% 하단은 <b>−0.065%</b>였다. “75% 확률로 투자에 성공”을 뜻하지 않는다.</p>
<details><summary>내부검증 상위 후보와 전체 탐색 파일</summary>{tab(tops)}<p><a href="parameter_search.csv">144개 설정 전체 결과</a> · <a href="training_phase_metrics.csv">학습·검증·위상별 지표</a> · <a href="selection.json">2017년 이후를 보기 전에 고정한 설정</a></p></details>
<h2 id="math">4. 종목비중의 수학적 정의</h2>
<p>기존 신고가 베이지안 회귀를 재사용했다. 설명변수는 x=[1,z,z²,1−2q], z=(신고가 접근도−0.90)/0.10, q는 신고가 접근 순위 비율이다. 방향성 과거수익률이나 모멘텀 필터는 없다. 정답은 다음 종가부터 H거래일의 수익률을 과거 변동성 σ√H로 나눈 값이다.</p>
<p>매일 t−1까지 결과가 완성된 라벨만 학습하고, 같은 날짜 종목 수와 중첩 기간을 반영한 가중 우도를 사용한다. 이전의 보수적인 사후분포를 더 확신하는 것처럼 바꾼 것이 아니라, 그 불확실성을 <b>현금 대신 균등비중으로 표현</b>하도록 바꿨다. 공액 회귀 갱신의 기본 근거는 <a href="https://statproofbook.github.io/P/blr-post.html">베이지안 선형회귀 사후분포</a>이며, 아래 배분 규칙은 이 연구에서 정한 운용 설계다.</p>
<div class="eq">aᵢ = σᵢ√H xᵢ, &nbsp; δᵢ = aᵢ − (1/n)Σⱼaⱼ<br>θᵢ − θ̄ = δᵢ′β<br>pᵢ = Pr(δᵢ′β &gt; 0 | D)</div>
<p>β의 정규–역감마 사후분포를 적분하면 위 차이는 Student-t 분포다. 평균은 δᵢ′m, 척도 제곱은 (b/a)δᵢ′Vδᵢ, 자유도는 2a다. 종목들이 같은 회귀계수를 공유하므로 공분산을 포함해 계산한다. 실현수익률의 승패 확률과 잠재 기대수익의 상대 우위 확률은 다르다.</p>
<div class="eq">sᵢ = sign(pᵢ−0.5) · max(|pᵢ−0.5|−(p₀−0.5), 0)/(1−p₀)<br>vᵢ = G · exp(κsᵢ) / Σⱼexp(κsⱼ)<br>G = max(0, min(0.98 − 거래 불가 보유분 비중, n × 종목 한도))</div>
<p>중립 점수는 0, 비중 강도 κ=1을 적용한다. v는 한도와 실제 거래비용을 반영하기 전 목표다. 이후 다음 문제를 푼다.</p>
<div class="eq">min<sub>w</sub> &nbsp; ½Σᵢ(wᵢ−vᵢ)² + λ[c<sub>매수</sub>Σᵢ(wᵢ−uᵢ)₊ + c<sub>매도</sub>Σᵢ(uᵢ−wᵢ)₊]<br>Σᵢwᵢ=G, &nbsp; 0≤wᵢ≤종목 한도</div>
<p>u는 현재 비중이다. 목표에 가까워지는 이득보다 거래비용 벌점이 크면 기존 수량을 남긴다. λ=3은 최적화에서 쓰는 억제 강도이며 <b>계좌 수수료를 세 배 차감한다는 뜻이 아니다.</b> 실제 비용은 순주문에 한 번만 부과한다. 이 최적화는 “가장 수익률 높은 주식비중”의 증명이 아니라, 확률로 만든 목표를 비용과 한도 안에서 추종하는 문제의 해다.</p>
<h2 id="mix">5. 설정 자체에도 베이지안을 적용한 결과</h2>
<p>선택된 한도·비용 벌점을 공유하는 상위 세 설정을 2016년 말에 고정했다. 이후 최대 36개 <b>완료된 월</b>의 균등 대비 초과로그수익으로 각 설정의 기대 개선확률을 갱신했다. 다섯 거래 위상의 차이를 먼저 평균내므로 같은 월을 다섯 독립 관측으로 세지 않는다.</p>
<div class="eq">a<sub>c</sub> = max(Pr(설정 c의 평균 초과로그수익 &gt; 0 | D)−0.5, 0)<br>g = min(0.5, 2 · mean(a<sub>c</sub>))<br>설정 c 비중 = g · a<sub>c</sub>/Σa<sub>c</sub>, &nbsp; 균등 기준 비중 = 1−g</div>
<p>최소 50%는 균등 모델 목표를 유지하고, 확신이 없으면 전부 균등으로 돌아간다. 이것도 <b>현금으로 전환하는 규칙이 아니다.</b> 비중은 월마다 갱신하되 실제 주문은 5거래일 검토 때 순매매로 실행한다. 주변우도로 계산한 정식 모형 사후확률이 아니라, 사후 개선확률을 이용한 보수적인 조합 규칙이다.</p>
<p>최근 연복리는 <b>15.11%</b>로 균등 14.83%보다는 조금 높았지만, 고정 확률 조절 16.28%보다 낮았다. 2017–2021에는 3.14%로 균등보다 낮았다. 설정을 계속 바꾸는 기능을 더할수록 좋아진다는 결과는 아니다.</p>
<details><summary>2026년 7월의 설정 조합 비중</summary>{tab(mt)}<p>나머지 <b>50%</b>는 균등 기준 모델이다. 2026년 6월까지 완료된 자료만 사용했다.</p></details>
<h2 id="holdings">6. 보유기간과 마지막 판단</h2>{tab(holdtable)}
<p>고정 확률 조절에서 최근 매도 완료분 평균 보유기간은 <b>20.08거래일</b>이었다. 20일 예측기간과 숫자가 비슷하지만 만기 때문에 파는 것이 아니다. FIFO로 매수분과 연결한 매도금액 가중 평균이며, 아직 보유 중인 수량과 회수 처리분은 제외했다.</p>
<p>최근 검토 기록은 보유 1,986건, 신규매수 895건, 추가매수 706건, 일부축소 616건, 청산 910건이었다. 판단 건수이며 체결 건수와는 다르다.</p>
<h3>2026.07.31 보유분</h3>{tab(last)}
<p><b>마지막 날의 현금은 {pct(1-hp.weight.sum())}다.</b> 최근 평균 주식비중 96.35%와 다르다. 마지막 검토일에는 자격을 만족한 종목이 5개뿐이어서 10% 한도 ×5=50%까지만 목표를 채웠다. 이번에 없앤 것은 확률 때문에 생기는 현금 대기이고, 적격 종목이 부족할 때 한도를 어겨서까지 전액 투자하지는 않는다.</p>
<p class="note">실제 비중은 검토일 사이의 가격변동으로 10%를 넘을 수 있다. 483650은 마지막 검토일 접근도가 1.0이었지만 7월 31일에는 0.834로 내려갔다. 일반 조건은 5일마다 판단하므로 그 사이 보유가 남는다. 현재 10월의 보유 추천이나 주문표가 아니다.</p>
<details><summary>2026.07.29 판단 → 다음 종가 주문</summary>{tab(det)}<p>이 날짜에는 적격 종목 5개가 모두 한도에 걸려 확률이 달라도 목표는 각각 10%다. 후보가 충분할 때 확률별 비중 차이가 나타난다.</p></details>
<h2>7. 비용·자료 민감도와 검증</h2>
<p>키움 KRX 온라인 일반요율 매수·매도 각 0.015%, 확인된 2020년 한시 인하, 결제일별 거래세 변화를 이전 연구와 동일하게 적용했다. 기본 체결비용은 편도 5bp다. <a href="https://www1.kiwoom.com/h/domestic/stock/VStockMainView?dummyVal=0">키움 현재 안내</a>와 <a href="https://download.kiwoom.com/notice/notice_20200915_02_01.pdf">2020년 한시 인하 공지</a>를 참조했다. 정확한 시기별 세율 근거는 sources.json에 함께 제공한다.</p>
{tab(senstab)}
<p>비용 가정을 바꾸면 최적화의 거래 억제도 함께 다시 계산했다. 기존 주문에 비용만 덧붙인 실험이 아니다. 편도 25bp에서 확률 조절은 11.76%, 같은 조건 균등은 10.68%였다.</p>
<div class="box warn"><b>기업행위 정산의 실제 한계가 1건 확인됐다.</b> 000060의 가격 소실 시 2023.02.21 마지막 평가가격 회수를 적용했으며 당시 계좌의 약 4.18%였다. 실제 메리츠화재는 포괄적 주식교환 대상이었으므로 이 규칙은 정확한 교환주식 입고·매도 정산이 아니다. 회수를 0으로 놓으면 확률 조절의 최근 연복리는 15.17%, 균등은 13.72%다. 두 가정이 실제 정산의 엄밀한 상·하한이라는 뜻은 아니다. <a href="https://m.meritzgroup.com/commfiles/hld/attach/2025/20250319/202503191809305730001U.pdf">회사 공시 자료</a>에서 주식교환 사실을 확인했다.</div>
<p>독립 검증: Student-t CDF는 SciPy와 최대 오차 약 2.6×10⁻¹⁴, 비중 최적화는 별도 SLSQP 문제와 목적함수 일치, 수량·현금·NAV를 체결로 재구성했다. 미래자료를 잘라 선택 모형의 사후분포·목표·NAV가 같은지, 설정 혼합이 완료된 월만 쓰는지 확인했다. 거래량 0에서 체결하지 않고 청산을 재시도하며, 확률 불확실성이 균등비중으로 돌아가는 규칙도 통과했다.</p>
<p>다만 원가격 수정 이력·현금배당·모든 기업행위와 편입 공시 시점을 전면 감사한 것은 아니다. 수정가격 분할 단위를 쓰므로 실제 정수 주식·호가단위·상하한가 주문 가능성·대규모 운용 수용량을 그대로 재현하지 못한다.</p>
<h2 id="files">8. 결과를 사용하는 방법</h2>
<p><b>비교용 설정은 H20/W1008, 중립구간 45–55%, 강도 1, 종목 한도 10%, 비용 벌점 3이다.</b> 현금 과다 문제를 해결한 구현으로 보관하고, 설정 자동 혼합은 별도 보조 실험으로 둔다. 초기 학습 합격 조건 미충족과 기업행위 정산 한계가 있으므로 새 자료·정확한 정산으로 모의운용을 이어갈 후보로 해석한다.</p>
<p><a href="최적화_재현자료.zip">코드·데이터·전체 결과 ZIP</a> · <a href="parameter_search.csv">144개 설정</a> · <a href="latest_review_actions.csv">마지막 판단</a> · <a href="holdings/OPTIMIZED.csv">마지막 보유분</a> · <a href="validation.json">검증 결과</a> · <a href="README.txt">재현 방법</a></p>
<p class="note">작성 2026.10.06 · 기존 연구를 보존한 후속 분석 · 데이터 종료 2026.07.31 · 실시간 주문 없음</p>
</main></body></html>'''
    (OUT/TITLE).write_text(doc,encoding='utf-8');print('Optimization report saved.',flush=True)

if __name__=='__main__':main()
