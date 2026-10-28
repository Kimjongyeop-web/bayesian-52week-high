from daily_run import *
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
TITLE='신고가_매일판단_수정검증보고서.html'
NAMES={'PREVIOUS5':'이전: 5일마다 판단','DAILY_SAME':'매일 판단 · 이전 설정','DAILY_EQUAL':'매일 판단 · 균등비중','DAILY_TUNED':'매일 판단 · 베이지안','TUNED_EQUAL':'매일 판단 · 같은 조건 균등'}
KEYS=['PREVIOUS5','DAILY_SAME','DAILY_TUNED','TUNED_EQUAL']
def pct(x):return f'{x*100:.2f}%'
def read(n):return pd.read_csv(OUT/n)
def table(x):return '<div class="table-wrap">'+x.to_html(index=False,border=0,escape=True)+'</div>'
def embed(n):return '<img alt="'+n+'" src="data:image/png;base64,'+base64.b64encode((OUT/n).read_bytes()).decode()+'">'
def perf(period):
    x=read('performance.csv');x=x[x.period.eq(period)].set_index('name').loc[KEYS].reset_index()
    return table(pd.DataFrame({'전략':x.name.map(NAMES),'연복리':x.cagr.map(pct),'누적수익':x.total_return.map(pct),'최대낙폭':x.mdd.map(pct),'Sharpe':x.sharpe.map(lambda v:f'{v:.3f}'),'평균 주식비중':x.exposure.map(pct),'연환산 비용률':x.annual_cost_rate.map(pct),'연간 매매액/NAV':x.annual_turnover.map(lambda v:f'{v:.2f}배')}))

def main():
    selection=json.loads((OUT/'selection.json').read_text());s=selection['selected'];p=read('performance.csv').query("period=='recent'").set_index('name');a=read('activity.csv').set_index('name');d={n:pd.read_csv(OUT/'daily'/f'{n}.csv',index_col=0,parse_dates=True).loc['2022-01-01':] for n in KEYS};colors={'PREVIOUS5':'#8797ac','DAILY_SAME':'#485f91','DAILY_TUNED':'#117f79','TUNED_EQUAL':'#b97b39'}
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'font.size':10,'grid.alpha':.2})
    fig,ax=plt.subplots(2,1,figsize=(11,7),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[2,1]})
    for n,x in d.items():
        wealth=(1+x.net_return).cumprod();ax[0].plot(x.index,wealth,label=NAMES[n],c=colors[n],lw=1.5);cost=x[['commission','sale_tax','execution_cost']].sum(1)/x.nav;ax[1].plot(x.index,cost.rolling(252,min_periods=60).mean()*252,c=colors[n],lw=1.4)
    ax[0].set_title('매일 판단으로 수정한 실제 계좌 결과',loc='left',fontweight='bold',fontsize=15,pad=15);ax[0].set_ylabel('2021년 말 자산 = 1');ax[0].legend(frameon=False);ax[1].set_ylabel('최근 252일 비용률\n연환산');ax[1].yaxis.set_major_formatter(PercentFormatter(1));[x.grid(True) for x in ax];fig.savefig(OUT/'daily_result.png',dpi=150);plt.close(fig)
    act=a.loc[KEYS].reset_index();acttable=pd.DataFrame({'전략':act.name.map(NAMES),'자료 거래일':act.trading_days,'판단한 날':act.review_days,'실제 체결일':act.days_with_fills,'종목별 순체결':act.stock_fills,'보유 유지 판단':act.hold_decisions,'신규 판단':act.entry_decisions,'청산 판단':act.exit_decisions})
    h=pd.read_csv(OUT/'holdings/DAILY_TUNED.csv',dtype={'ticker':str});ht=pd.DataFrame({'종목코드':h.ticker,'실제 보유비중':h.weight.map(pct),'52주 고점 대비':h.near.map(pct)})
    dec=pd.read_csv(OUT/'latest_close_decisions.csv',dtype={'ticker':str});dt=pd.DataFrame({'종목코드':dec.ticker,'판단':dec.action.map({'HOLD':'유지','ADD':'추가','REDUCE':'축소','EXIT':'청산','BUY_NEW':'신규'}),'현재비중':dec.held_weight.map(pct),'다음 체결 목표':dec.target_weight.map(pct),'신고가 접근도':dec.near.map(pct)})
    sens=read('sensitivity.csv');sens=sens[sens.name.isin(['DAILY_TUNED','TUNED_EQUAL'])];st=pd.DataFrame({'전략':sens.name.map(NAMES),'가정':sens.variant.map({'execution13bp':'체결비용 편도 13bp','execution25bp':'체결비용 편도 25bp','recovery0':'가격 소실분 회수 0%'}),'연복리':sens.cagr.map(pct),'최대낙폭':sens.mdd.map(pct)})
    yr=read('annual_results.csv');yr=yr[yr.name.isin(KEYS)].pivot(index='year',columns='name',values='total_return').rename(columns=NAMES).reset_index();yr['year']=yr.year.map(lambda y:f'{y}'+(' (7월까지)' if y==2026 else ''));yr=yr.rename(columns={'year':'연도'})
    for c in yr.columns[1:]:yr[c]=yr[c].map(pct)
    css='''*{box-sizing:border-box}body{margin:0;background:#eef3f4;color:#203b4c;font-family:"Malgun Gothic",sans-serif;line-height:1.85}main{max-width:1150px;margin:28px auto;background:white;padding:46px 52px;border-radius:14px}h1{font-size:35px;line-height:1.35;margin:16px 0}h2{font-size:23px;border-top:1px solid #dce6e9;margin-top:40px;padding-top:25px}h3{font-size:18px}.eyebrow{font-size:12px;letter-spacing:.12em;color:#117f79;font-weight:bold}.box{background:#eaf6f3;padding:18px 22px;border-left:4px solid #117f79;border-radius:4px;margin:20px 0}.warn{background:#fff5e8;border-color:#b97b39}.note,small{font-size:13px;color:#557183}a{color:#087670}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:12.5px;line-height:1.7;margin:16px 0}td,th{padding:10px;border-bottom:1px solid #dce5e9;text-align:right;white-space:nowrap}td:first-child,th:first-child{text-align:left}th{background:#edf5f6}img{display:block;width:100%;height:auto;margin:22px 0}summary{cursor:pointer;font-weight:bold}details{margin:18px 0}.eq{background:#f3f6f8;padding:16px 20px;overflow:auto;border-radius:5px}li{margin:7px 0}nav{display:flex;gap:16px;flex-wrap:wrap;font-size:13px}@media(max-width:720px){main{margin:0;padding:24px 18px}h1{font-size:27px}}@media print{body{background:white}main{margin:0;padding:8mm}img,.box{break-inside:avoid}}'''
    doc=f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>신고가 전략 · 매일 판단 수정 검증</title><style>{css}</style></head><body><main>
<div class="eyebrow">DAILY DECISIONS · CORRECTED IMPLEMENTATION · 2026.10.06</div>
<h1>매일 종목과 비중을 판단하도록<br>수정하고 다시 계산했습니다</h1>
<p><b>이전 구현의 잘못:</b> 확률은 매일 갱신했지만 종목 선정·청산·비중 변경은 5거래일마다 판단했다. 사용자가 요청한 매일 판단과 달랐다. 이번에는 <b>매 거래일 종가 후 전체 보유·후보 종목을 검토</b>하고, 필요한 순주문을 다음 거래일 종가에 체결한다.</p>
<div class="box"><b>수정된 최근 결과:</b> 매일 판단 베이지안의 비용 차감 연복리는 <b>{pct(p.loc['DAILY_TUNED','cagr'])}</b>, 같은 조건의 매일 판단 균등비중은 <b>{pct(p.loc['TUNED_EQUAL','cagr'])}</b>다. 이전 <b>16.28%</b>는 5일마다 판단한 전략의 결과이며, 매일 판단하는 전략의 성과로 사용하면 안 된다.</div>
<p class="note">기존 자료 그대로: 2008.01.02–2026.07.31. 계좌는 모두 2017.01.02 같은 현금으로 시작, 2022년에도 보유를 이어간다. 2026년 8–10월 자료와 실시간 주문은 포함하지 않았다.</p>
<nav><a href="#rules">매일 운용</a><a href="#results">성과</a><a href="#cost">거래와 비용</a><a href="#latest">마지막 판단</a><a href="#files">파일</a></nav>
<h2 id="rules">1. 이제 매일 무엇을 하는가</h2>
<ol><li><b>t일 장 마감:</b> 수정종가 기준 252일 신고가 접근도, 당시 편입·거래 가능 여부, 베이지안 사후분포를 갱신한다. 학습에는 t−1까지 결과가 끝난 미래수익률 라벨만 사용한다.</li><li><b>같은 날 전체 종목 판단:</b> 새 매수 후보, 계속 보유할 종목, 조건 이탈 청산 종목을 정한다. 각 후보의 상대 기대수익 확률로 목표비중을 계산한다.</li><li><b>현재 비중과 비교:</b> 매수·매도 비용을 포함한 최적화로 유지·추가·축소를 나눈다. 이미 가진 수량을 전부 팔고 다시 사는 방식이 아니다.</li><li><b>t+1일 종가:</b> 확정된 순주문만 체결한다. 거래 불가면 체결하지 않으며 미체결 매도는 유지한다. 새로 판단 가능한 정보가 생기면 다음 주문을 다시 정한다.</li></ol>
<p>5일 간격 검사와 그 시작일에 따른 매매 일정은 <b>새 전략에서 제거</b>했다. 5일 계좌는 이전 결과와 비교하기 위한 대조군으로만 남긴다. 예측기간 H는 미래수익을 예측하는 기간이지 보유 만기나 검토 주기가 아니다. 재선정된 H=5 역시 매일 판단하면서 5거래일 앞을 예측한다는 뜻이다.</p>
<table><tr><th>항목</th><th>일별 전략의 규칙</th></tr><tr><td>신규 후보</td><td>252일 종가 고점의 90% 이상, 신고가 접근도 상위 10%</td></tr><tr><td>보유 후보</td><td>고점의 90% 이상, 상위 20% 이내; 매일 확인</td></tr><tr><td>확률 역할</td><td>후보 평균보다 기대수익이 높을 확률로 비중 조절; 절대확률 매수 차단 없음</td></tr><tr><td>확률 중립구간</td><td>{pct(1-s['p0'])}–{pct(s['p0'])}; 모두 불확실하면 균등비중</td></tr><tr><td>예측 / 학습창</td><td>{selection['H']} / {selection['W']} 거래일</td></tr><tr><td>종목 한도 / 최대 종목 수</td><td>{pct(s['cap'])} / 20종목</td></tr><tr><td>비중 강도 / 비용 벌점</td><td>{s['strength']} / {s['lam']}</td></tr><tr><td>전체 목표비중</td><td>98%, 적격 종목 수·한도·미체결에 따라 낮아질 수 있음</td></tr><tr><td>보유기간</td><td>상한 없음; 날짜만으로 청산하지 않음</td></tr></table>
<h2 id="results">2. 매일 판단으로 다시 계산한 성과</h2>
<h3>2022.01–2026.07</h3>{perf('recent')}
{embed('daily_result.png')}
<p>같은 설정에서 판단 주기를 5일에서 1일로 바꾸자 연복리가 <b>{pct(p.loc['PREVIOUS5','cagr'])} → {pct(p.loc['DAILY_SAME','cagr'])}</b>로 낮아졌다. 평균 주식비중은 비슷하므로 이번 감소는 현금 대기의 재발이 아니다. 거래 시점·종목 경로가 달라지고 실제 매매비용이 늘어난 결과다. 비용률 차이만으로 수익률 감소 전체를 정확히 분해한 것은 아니다.</p>
<p>매일 판단끼리 비교하면 베이지안이 균등비중보다 연복리 <b>{(p.loc['DAILY_TUNED','cagr']-p.loc['TUNED_EQUAL','cagr'])*100:.2f}%p</b> 높았다. 이는 이 표본의 관찰 결과이며 독립적인 새 데이터에서 확인된 미래수익 보장은 아니다.</p>
<h3>2017–2021 테스트</h3>{perf('test')}
<details><summary>2017–2026.07 전체 및 연도별 결과</summary>{perf('all_test')}{table(yr)}</details>
<h2 id="cost">3. 매일 판단해도 종목별 보유는 유지하지만, 계좌 거래는 많았다</h2>
{table(acttable)}
<p>최근 {int(a.loc['DAILY_TUNED','trading_days']):,}거래일 <b>전부에서 종목·목표비중을 판단</b>했다. 보유 유지 판단도 {int(a.loc['DAILY_TUNED','hold_decisions']):,}건 있었다. 다만 계좌 전체로는 {int(a.loc['DAILY_TUNED','days_with_fills']):,}일에 체결이 발생했다. “매일 판단”과 “전 종목 매일 교체”는 다르지만, 이번 규칙에서 계좌의 거래 빈도가 낮았다고 말할 수는 없다.</p>
<p>연환산 비용률은 <b>{pct(p.loc['PREVIOUS5','annual_cost_rate'])} → {pct(p.loc['DAILY_TUNED','annual_cost_rate'])}</b>다. 종목이 고점 90% 또는 보유 순위 경계를 넘나들면 매일 청산·재진입할 수 있다. 또 98% 목표비중을 정확히 맞추는 제약 때문에 작은 현금비중 유지 거래도 생긴다. 비용 벌점은 일부 비중 거래를 억제하지만 이런 필수 조건을 무시하지 않는다.</p>
<div class="box warn"><b>요청한 매일 판단은 구현했지만, 불필요한 거래를 충분히 억제했다고 결론내리지는 않는다.</b> 이번 계산에서는 기존 진입·보유 경계와 목표비중 제약을 보존해 판단 주기의 영향을 분리했다. 거래를 더 줄이려면 판단을 다시 5일로 늦추는 것이 아니라, 매일 판단 안에서 진입·청산 경계의 완충폭과 총비중 허용범위를 별도로 검증해야 한다.</div>
<p class="note">비용률 = 일별 (수수료+매도세+체결비용)/NAV 평균 ×252. 매매액은 매수+매도 합계. 종목별 순체결은 같은 종목의 매수·매도를 상계한 뒤 집계했다. 실제 정수 주식·최소 주문금액 제약은 포함하지 않은 수정가격 분할 단위 모의계좌다.</p>
<h2>4. 설정도 일별 방식에서 다시 학습했는가</h2>
<p>기존 144개 베이지안 설정과 균등 기준 6개를 <b>일별 판단</b>으로 다시 계산했다. 2009.08–2013 초기 학습과 2014–2016 내부검증만 사용했고, 거래 시작일 다섯 가지의 성과를 섞거나 좋은 시작일을 선택하지 않았다.</p>
<p>내부검증 월별 초과로그수익의 베이지안 하단으로 선택한 비교용 설정은 <b>{s['name']}</b>다. 기존 H20·강도1에서 <b>H5·강도0.5</b>로 바뀌었다. 최근 성과는 재선정 설정이 10.45%, 기존 설정을 매일 적용한 경우가 10.92%다. 최근 구간을 보고 더 높은 쪽으로 다시 선택하지 않았다. 초기 학습에서 균등 대비 양수 개선 조건을 통과한 후보는 <b>{selection['qualified_count']}개</b>였다. 따라서 실전 합격 설정으로 표시하지 않고 비교용 후보로 남긴다.</p>
<p>비중 계산은 상대 기대수익 확률 p에 따라 중립 점수 또는 확대·축소 점수를 주고 지수형 목표비중 v를 만든다. 현재 비중 u에서 다음 문제를 풀어 실제 목표를 정한다.</p>
<div class="eq">min ½Σ(w−v)² + λ[c<sub>매수</sub>Σ(w−u)₊ + c<sub>매도</sub>Σ(u−w)₊]<br>Σw = 가능한 목표 주식비중, &nbsp; 0≤wᵢ≤종목 한도</div>
<p>확률은 잠재 기대수익의 상대 우위에 관한 모형값이지 거래 성공률이 아니다. 비용 벌점 λ는 계좌 수수료를 그 배수만큼 실제 차감한다는 뜻도 아니다. 실제 비용은 체결 금액에 한 번만 부과한다.</p>
<h2 id="latest">5. 마지막 실제 보유분과 마지막 날의 새 판단</h2>
<p><b>2026.07.31 실제 모의 보유분</b>이며 현금은 {pct(1-h.weight.sum())}다.</p>{table(ht)}
<p><b>같은 날 종가 후 다시 판단한 목표</b>는 아래와 같다. 이 목표는 다음 거래일 체결 대기 상태다. 데이터가 7월 31일에 끝나므로 이후 체결이나 수익은 만들지 않았다.</p>{table(dt)}
<p class="note">실제 보유와 다음 목표는 다를 수 있다. 오늘 종가를 본 뒤 이미 지나간 오늘 종가로 주문을 체결하지 않기 때문이다. 10월 현재 투자 추천이나 증권사 주문 목록이 아니다.</p>
<h2>6. 민감도·검증·자료 한계</h2>{table(st)}
<p>이전과 같은 키움 KRX 온라인 일반요율, 확인된 시기별 한시 인하와 결제일별 매도세를 적용했다. 기본 체결비용은 편도 5bp, 위 표는 13/25bp로 높여 거래 의사결정도 다시 계산한 결과다. 비용 근거는 <a href="sources.json">sources.json</a>에 보관했다.</p>
<p>검증 항목: 이전 5일 계좌의 숫자 재현, 모든 거래일의 판단 기록, 기존 5일 일정 사이에 발생한 이탈 신호의 다음 종가 청산, 미체결 매도 유지, 실제 주문으로 현금·수량·NAV 독립 재구성, 미래 데이터를 잘라 일별 사후분포·목표·NAV가 같은지 확인. <a href="validation.json">검증 결과</a>에 수치와 통과 여부를 기록했다.</p>
<p>기존 데이터의 기업행위 정산 한계는 남아 있다. 특히 000060 주식교환은 정확한 교환주식 입고가 아니라 가격 소실 시 마지막 평가가격 회수로 근사했다. 회수 0% 민감도는 정확한 정산을 대신하지 않는다. 배당·수정가격의 과거 버전·호가와 상하한가 체결 가능성·운용 규모별 시장충격도 완전히 검증하지 못했다. 역사 구간을 이미 반복 연구했으므로 깨끗한 신규 외부검증으로 부르지 않는다.</p>
<h2 id="files">7. 수정본 파일</h2>
<p><a href="매일판단_재현자료.zip">코드·원입력·전체 결과 ZIP</a> · <a href="performance.csv">성과</a> · <a href="activity.csv">매일 판단·실제 거래 수</a> · <a href="latest_close_decisions.csv">마지막 판단</a> · <a href="parameter_search.csv">144개 설정</a> · <a href="README.txt">재현 방법</a></p>
<p class="note">수정일 2026.10.06 · 모멘텀 없음 · 모든 거래일 종목·비중 판단 · 이전 보고서는 비교 기록으로 보존</p>
</main></body></html>'''
    (OUT/TITLE).write_text(doc,encoding='utf-8');log('Daily report saved')

if __name__=='__main__':main()
