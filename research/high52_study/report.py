from engine import *
from analyze import LABELS
import base64,html,zipfile

SOURCES=[
('이호·김은정(2013), 한국 주식시장에서 52주 고가가 투자자의 매매심리에 미치는 영향 — KCI 초록','https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART001814834','논문 초록 확인. 본 연구는 논문을 그대로 복제하지 않고 종가 기반 매수 전용 전략으로 구체화했다.'),
('키움증권 2008–2009 연차보고서','https://download.kiwoom.com/company/kfinance/FY08%20Annual%20Report.pdf','2008년 5월 0.025%에서 0.015%로 변경했다고 설명. 거래 시뮬레이션은 2009년부터 시작.'),
('키움증권 2013년 약관 수수료표','https://download.kiwoom.com/web/treaty06_20130916.pdf','HTS 온라인 거래소·코스닥 수수료 0.015%.'),
('키움증권 거래수수료 한시 인하 공지','https://download.kiwoom.com/notice/notice_20200915_02_01.pdf','2020.09.16–12.31 매매일 기준 0.015% → 0.0113604%.'),
('키움증권 2025년 수수료 안내 문서','https://download.kiwoom.com/deploy/AG003/pdf/AG003_27.pdf','KRX 코스피·코스닥 온라인/API 표준수수료 0.015%.'),
('키움증권 현재 국내주식 세금·수수료 안내','https://www1.kiwoom.com/h/domestic/stock/VStockMainView?dummyVal=0','온라인 매매수수료 0.015%, 코스피 매도세 합계 0.20%.'),
('코스콤: 2019년 거래세 인하 안내','https://www.koscom.co.kr/mobile/bbs/B0000006/view.do?delcode=0&deptId=&edate=&gubun=&ise=&isk=&menuNo=400104&nttId=27374&pageIndex=12&replyAt=&sdate=&searchBgnDe=&searchCnd=&searchEndDe=&searchWrd=&type=&useAt=&viewType=&year=','2019.05.30 체결분부터 인하, 2019.06.03 결제분부터 시행. 코스피 농특세는 별도.'),
('기획재정부: 2021년 증권거래세율 인하','https://whatsnew.moef.go.kr/mec/ots/dif/view.do?comBaseCd=DIFGODEPRT&difGovDepart1=DIFGODR001&difSer=5df48aab-ae37-437c-8e86-d6f1a9ee7823&temp=2021&temp2=HALF001','2021.01.01 양도분부터 코스피 0.08% + 농특세 0.15%.'),
('국회예산정책처: 2024 대한민국 조세','https://korea.nabo.go.kr/board/file/down.do?fid=33317987','2023·2024·2025 세율 및 코스피 농특세 0.15% 확인.'),
('재정경제부: 2026년부터 이렇게 달라집니다, 14쪽 — 상공회의소 배포본','https://jincci.korcham.net/file/dext5uploaddata/2026/%5B%EA%B8%B0%ED%9A%8D%EC%9E%AC%EC%A0%95%EB%B6%80%5D%20%E3%80%8C2026%EB%85%84%EB%B6%80%ED%84%B0%20%EC%9D%B4%EB%A0%87%EA%B2%8C%20%EB%8B%AC%EB%9D%BC%EC%A7%91%EB%8B%88%EB%8B%A4%E3%80%8D.pdf','2026.01.01 양도분부터 코스피 0.05% + 농특세 0.15%, 코스닥 0.20%.')]

def pct(x):return f'{x*100:+.2f}%'
def table(df):return '<div class="table-wrap">'+df.to_html(index=False,escape=False,border=0)+'</div>'
def perf_table(perf,names,period):
    z=perf[perf.period==period].set_index('name').loc[names].reset_index()
    z['name']=z.name.map(lambda x:LABELS.get(x,x))
    fields=['name','total_return','cagr','sharpe','mdd','exposure','annual_turnover']
    z=z[fields]
    for c in ['total_return','cagr','mdd','exposure']:z[c]=z[c].map(pct)
    z.sharpe=z.sharpe.map(lambda x:f'{x:.3f}');z.annual_turnover=z.annual_turnover.map(lambda x:f'{x:.2f}배')
    z.columns=['전략','누적','연복리','Sharpe','최대낙폭','평균 주식비중','연 매수+매도/NAV'];return table(z)
def pic(name):return '<img src="data:image/png;base64,'+base64.b64encode((OUT/name).read_bytes()).decode()+'" alt="'+html.escape(name)+'">'

def main():
    perf=pd.read_csv(OUT/'final_performance.csv');allres=pd.read_csv(OUT/'all_96_results.csv');years=pd.read_csv(OUT/'annual_results.csv');boot=pd.read_csv(OUT/'paired_bootstrap.csv');freq=pd.read_csv(OUT/'rolling_horizon_frequency.csv')
    fixed=['FIXED_HIGH_EW','FIXED_HIGH_WEIGHT','FIXED_MOM_WEIGHT','FIXED_JOINT_WEIGHT','KOSPI200'];roll=['ROLL_HIGH_EW','ROLL_HIGH_WEIGHT','ROLL_MOM_EW','ROLL_MOM_WEIGHT','ROLL_JOINT_WEIGHT','MID_HIGH_TILT','KOSPI200']
    h=allres[(allres.family=='NEAR10')&(allres.weight=='IV60')&(allres.period.isin(['development','test','deployment']))].pivot(index='H',columns='period',values='cagr').reset_index()[['H','development','test','deployment']]
    for c in h.columns[1:]:h[c]=h[c].map(pct)
    h.columns=['보유 거래일','학습 2009–2016 연복리','테스트 2017–2021 연복리','후속 2022–2026.07 연복리']
    annual=years[years.name.isin(['FIXED_HIGH_WEIGHT','FIXED_JOINT_WEIGHT','ROLL_HIGH_WEIGHT','KOSPI200'])].pivot(index='year',columns='name',values='total_return').reset_index()
    for c in annual.columns[1:]:annual[c]=annual[c].map(pct)
    annual=annual.rename(columns={**LABELS,'year':'연도'});annual['연도']=annual['연도'].astype(str).replace('2026','2026.01–07')
    infer=boot[boot.block==60].copy();infer['비교']=infer.treatment.map(LABELS)+' − '+infer.control.map(LABELS)
    infer['연평균 일수익률 차이']=infer.annual_mean_diff.map(pct);infer['95% 구간']=infer.apply(lambda r:f'{pct(r.mean_ci_low)} ~ {pct(r.mean_ci_high)}',axis=1)
    infer['Sharpe 차이']=infer.sharpe_diff.map(lambda x:f'{x:+.3f}')
    infer=infer[['비교','연평균 일수익률 차이','95% 구간','Sharpe 차이']]
    latest={p.stem:pd.read_csv(p).iloc[-1]['selected'] for p in (OUT/'schedules').glob('*.csv')}
    se=pd.read_csv(OUT/'sensitivity.csv');se=se[se.name.isin(['ROLL_HIGH_WEIGHT','ROLL_JOINT_WEIGHT','ROLL_MOM_WEIGHT'])][['name','variant','cagr','mdd']]
    se.name=se.name.map(LABELS);se.cagr=se.cagr.map(pct);se.mdd=se.mdd.map(pct);se.columns=['전략','가정','연복리','최대낙폭']
    phases=pd.read_csv(OUT/'entry_phase_sensitivity.csv').groupby('name').cagr.agg(['min','max']).reset_index();phases['name']=phases.name.map(LABELS)
    phases['min']=phases['min'].map(pct);phases['max']=phases['max'].map(pct);phases.columns=['사전 선택 고정 전략','시작일 5가지 중 최저 연복리','최고 연복리']
    validations=json.loads((OUT/'validation.json').read_text());hold=pd.read_csv(OUT/'holdings/ROLL_HIGH_WEIGHT.csv',dtype={'ticker':str}).head(15)
    for c in ['weight','near_52week','momentum_12_1']:hold[c]=hold[c].map(pct)
    hold=hold.rename(columns={'ticker':'종목코드','adjusted_units':'수정가격 기준 단위','adjusted_close':'수정종가','weight':'실제 보유비중','near_52week':'52주 고가 접근도','momentum_12_1':'12–1개월 수익률'})
    text=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>52주 신고가 전략 · 단기반전 대체 검증</title><style>
    :root{{--ink:#142b3e;--muted:#516779;--blue:#2257b5;--line:#d8e1e9}}*{{box-sizing:border-box}}body{{font-family:"Malgun Gothic",sans-serif;margin:0;color:var(--ink);background:#edf2f6;line-height:1.85}}main{{max-width:1130px;margin:32px auto;background:#fff;padding:54px 58px;border-radius:14px}}h1{{font-size:35px;line-height:1.35;margin:10px 0 18px}}h2{{font-size:23px;margin:42px 0 14px;padding-top:20px;border-top:1px solid var(--line)}}h3{{font-size:17px;margin:25px 0 9px}}p{{margin:12px 0}}.eyebrow{{color:var(--blue);font-weight:700;font-size:13px;letter-spacing:.12em}}.note{{color:var(--muted);font-size:13px}}.box{{padding:18px 24px;background:#eef5ff;border-left:4px solid var(--blue);border-radius:5px;margin:18px 0}}.caution{{background:#fff7e9;border-color:#ad7610}}.table-wrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:12.5px;line-height:1.6;margin:16px 0}}th{{background:#edf3f8;color:#294359;text-align:right}}td,th{{padding:10px 11px;border-bottom:1px solid var(--line);white-space:nowrap}}td:first-child,th:first-child{{text-align:left;white-space:normal;min-width:145px}}td{{text-align:right}}img{{display:block;width:100%;height:auto;margin:18px 0}}.eq{{background:#f5f7fa;padding:16px 22px;border-radius:6px;font-family:"Cambria Math","Malgun Gothic",serif;font-size:17px;overflow:auto}}a{{color:var(--blue)}}li{{margin:7px 0}}code{{font-family:Consolas,monospace;background:#f1f4f6;padding:2px 5px}}details{{margin:16px 0}}summary{{cursor:pointer;font-weight:bold}}@media(max-width:700px){{main{{margin:0;padding:24px 18px}}h1{{font-size:27px}}}}@media print{{body{{background:white}}main{{max-width:none;padding:10mm;margin:0}}h2{{break-after:avoid}}table,img,.box{{break-inside:avoid}}}}
    </style><main><div class="eyebrow">RESEARCH NOTE · 2026.10.06 · 기존 데이터 재사용</div><h1>52주 신고가 전략을<br>단기반전 대신 사용할 수 있을까?</h1>
    <p>판정: <b>신고가 접근 전략은 연구 후보로 남길 가치가 있지만, 지금 자료만으로 단기반전을 전면 대체하거나 시장 초과수익이 검증됐다고 말할 수 없다.</b> 수익률만 보면 고정 모멘텀이 앞섰고, 신고가와 모멘텀의 결합은 이 표본에서 낙폭과 Sharpe를 개선했다. 매일 설정을 다시 선택하는 방식은 장기간 현금 대기를 만들었으며, 고정한 규칙을 일관되게 이기지 못했다.</p>
    <div class="box"><b>검토 우선순위</b><br>신고가 단독의 출발점: 52주 종가 고점 접근도 상위 10%, 접근도 90% 이상, 묶음별 120거래일 보유.<br>모멘텀에 추가하는 출발점: 신고가 순위 75% + 모멘텀 순위 25%, 상위 20%, 60거래일 보유, 변동성 역수 비중.<br>두 설정 모두 2009–2016 학습기간에서 골랐다. 2022년 이후 성과를 보고 선택한 최적값이 아니며, 향후 최적이라는 뜻도 아니다.</div>
    <p class="note">분석대상: 당시 KOSPI200 편입 이력을 가진 과거 종목 407개. 가격 자료 2008.01.02–2026.07.31, 초기 1년은 신호 준비. 2026.08–10은 검증 범위 밖이다. 실제 계좌가 아닌 비용 차감 백테스트이며, 이 역사 구간은 이전 연구에서 이미 여러 번 관찰했다.</p>
    <h2>1. 최근 성과: 고정 규칙과 매일 선택을 구분</h2>
    <p>학습에서 고른 규칙을 고정한 계좌는 2017년 동일한 현금으로 시작한다. 2022년 이후 표는 2021년 말의 실제 시뮬레이션 보유분을 이어받고, 그 시점 NAV를 1로 정규화한 결과다. 모든 전략의 신규 진입·체결·비용 규칙은 같다. Sharpe의 무위험수익률과 현금 수익률은 0으로 놓았다.</p>
    <h3>사전에 선택한 고정 규칙 · 2022.01–2026.07</h3>{perf_table(perf,fixed,'deployment')}
    <p>고정 모멘텀 40일의 연복리는 <b>16.38%</b>, 고정 결합 60일은 <b>15.08%</b>였다. 결합 전략은 수익률이 더 높지는 않았지만 최대낙폭이 <b>−42.63% → −26.97%</b>, Sharpe가 <b>0.684 → 0.749</b>로 개선됐다. 이는 선택 방식·보유기간·비중이 함께 달라진 전략 전체의 비교이므로, 개선 전부를 신고가 신호 하나의 효과로 해석하지 않는다.</p>
    <h3>매일 롤링 선택 · 2022.01–2026.07</h3>{perf_table(perf,roll,'deployment')}
    <p>신고가 롤링 비중 선택은 연복리 <b>11.18%</b>, 모멘텀은 <b>10.69%</b>, 결합은 <b>12.30%</b>다. 결합 롤링은 수익률이 높지만 낙폭도 커졌다. 평균 주식비중이 각각 약 58%, 48%, 61%로 달라, 이 차이를 순수 종목선택 능력으로 읽으면 안 된다. KOSPI200은 배당·거래비용을 차감하지 않은 100% 투자 가격지수이므로 참고 기준이며, 투자 가능한 ETF의 순수익률은 아니다.</p>
    {pic('net_returns.png')}
    <h3>중간 테스트에서도 통했는가? · 2017–2021</h3>{perf_table(perf,['FIXED_HIGH_WEIGHT','FIXED_MOM_WEIGHT','FIXED_JOINT_WEIGHT','ROLL_HIGH_WEIGHT','ROLL_MOM_WEIGHT','ROLL_JOINT_WEIGHT','KOSPI200'],'test')}
    <div class="box caution">신고가 고정 전략의 테스트기간 연복리는 3.19%, 최대낙폭은 −54.26%였다. 롤링 신고가 전략은 같은 기간 연복리 약 0.02%에 그쳤다. <b>“최근에 돈을 벌었다”와 “여러 시장 국면에서 초과수익이 안정적으로 검증됐다”는 다른 판단이다.</b> 이번 결과는 후자를 충족하지 못한다.</div>
    <h3>연도별 수익률</h3>{table(annual)}
    <p>2026년은 7개월 누적수익률이며 연환산하지 않았다. 2025–2026 상승장의 기여가 크므로 후속 전체 수익률 하나만으로 결론내리지 않는다.</p>
    <h2>2. 보유기간: 120일을 기준점으로, 60일을 이웃 후보로</h2>
    <p>신고가 접근 상위 10%·변동성 역수 비중을 고정하고 보유기간만 바꾼 진단이다. 후보 계좌는 2009년부터 시작한 연속 계좌다. 위의 2017년 새 현금 출발 계좌와는 진입 묶음의 5일 주기가 달라 수치가 조금 다르다.</p>{table(h)}{pic('horizon_comparison.png')}
    <p>학습기간에서는 120일이 가장 좋고 60일도 근접했다. 따라서 <b>신고가 단독의 기준 보유기간은 120거래일, 비교 범위는 60–120거래일</b>로 제시한다. 이는 약 3–6개월이며, 정확한 달력 기간은 휴장일에 따라 달라진다. 5일은 학습 연복리가 약 2.02%로, 단기반전의 짧은 보유기간을 그대로 옮기는 근거가 약했다.</p>
    <p>롤링 신고가 비중 선택의 후속기간 의사결정일 중 120일 선택은 약 39.6%, 40일은 16.5%, 20일은 14.1%, 5일은 3.5%, 신규매수 중단은 26.3%였다. “평균 실제 묶음 보유기간”은 약 51.8일로, 고정 120일 계좌와 다르다. 종목 편출·체결 불가·만기 연장 때문에 실제 기간이 달라질 수 있고, 같은 종목을 새 묶음에서도 선택하면 계좌에서는 매도·재매수를 상계하여 연속 보유할 수 있다.</p>
    <h3>주간 진입일에 지나치게 의존하는가?</h3>{table(phases)}
    <p>2017년 시작점을 첫 5거래일로 각각 옮겨 같은 규칙을 재실행했다. 가장 좋은 시작일을 선택하지 않았다. 범위가 넓으면 단일 백테스트의 소수점 성과를 정밀한 기대수익률로 해석하면 안 된다.</p>
    <h2>3. 매일 학습하되 매일 갈아타지 않는 구조</h2>
    <p>비교 가능한 후보는 8개 선택 방식 × 6개 보유기간 × 2개 비중 방식, 총 96개다. 최초 학습·독립적인 기간 비교는 2009–2016 → 2017–2021 → 2022–2026.07이다. 운영용 롤링 선택은 아래와 같이 매 거래일 반복한다.</p>
    <div class="eq">판단일 t: 학습 [t−1008, t−253] 756일 → 검증 [t−252, t−1] 252일 → 판단 t → 체결 t+1</div>
    <ol><li>t−1까지 실제로 관찰된 후보 전략의 비용 차감 일수익률만 쓴다. 미완성 미래 H일 수익률은 학습에 넣지 않는다.</li><li>학습·검증 평균수익률이 모두 양수인 후보 중 학습 Sharpe 상위 3개를 추린다. 그중 검증 Sharpe가 가장 높은 후보를 도전자로 정한다.</li><li>도전자의 검증 Sharpe가 현재 규칙보다 0.15 이상 높다는 조건을 5일 연속 충족해야 변경한다. 적격 후보가 5일 연속 없으면 신규 매수를 중단한다. 기존 묶음은 즉시 청산하지 않고 만기까지 간다.</li><li>새 후보의 종목 순위·변동성·검증 성과는 매일 계산하되, 신규 묶음은 5거래일 간격으로만 만든다. 기존 묶음의 보유기간 H는 진입할 때 확정하며 모델이 바뀌어도 소급 변경하지 않는다.</li><li>다음 날이 되면 창을 한 칸 옮긴다. 과거 테스트 날짜의 수익률도 이미 관찰된 이후에는 뒤의 판단에 쓸 수 있다. 앞선 판단이나 저장된 성과를 재작성하지 않는다.</li></ol>
    <p>따라서 2022년 운용 시에는 2021년까지의 과거 테스트 자료를 사용할 수 있다. 이후 매일 최근 4년 창으로 갱신한다. 이는 미래 가격을 보고 선택한 후보 수익률들을 이어붙이는 방식이 아니라, <b>하나의 현금·보유수량 계좌를 계속 운용</b>한 결과다. 비교용으로 2022년 같은 현금에서 모두 새로 출발한 FRESH 결과도 자료에 포함했다.</p>
    <p>보유기간·선택 규칙을 매일 재평가하는 것이 반드시 성과를 높이지는 않는다. 이번 규칙은 적격성 조건으로 인해 반등 초기에 현금 상태가 길었고, 그 기회비용이 발생했다. <b>매일 재학습 자체를 우월성의 증거로 삼지 않고 고정 규칙도 계속 대조군으로 남겨야 한다.</b></p>{pic('equity_exposure.png')}
    <h2>4. 신호와 비중의 수학적 정의</h2>
    <p>P<sub>i,t</sub>는 종목 i의 t일 수정종가, U<sub>t</sub>는 당시 투자 가능한 종목 집합이다. 이 데이터에는 검증된 수정 장중고가 전체 이력이 없으므로 고가는 <b>종가의 최댓값</b>으로 정의한다.</p>
    <div class="eq">H<sub>i,t</sub><sup>52</sup> = max(P<sub>i,t−251</sub>, …, P<sub>i,t</sub>)<br>A<sub>i,t</sub> = P<sub>i,t</sub> / H<sub>i,t</sub><sup>52</sup> ∈ (0,1]</div>
    <p>A=0.97이면 최근 252거래일 최고 종가보다 3% 낮다. NEAR10은 A≥0.90인 종목을 A 내림차순으로 정렬해 U의 상위 10% 개수까지 선택하고, NEAR20은 20%다. 조건을 만족하는 수가 적으면 모두 채우려고 기준을 낮추지 않는다. 동일 점수는 종목코드 순서로 처리한다.</p>
    <div class="eq">B<sub>i,t</sub> = 1{{P<sub>i,t</sub> &gt; max(P<sub>i,t−252</sub>, …, P<sub>i,t−1</sub>)}}</div>
    <p>BREAK20은 최근 5일 중 B=1이 있었고 현재 A≥0.98인 종목을 대상으로, 전체 투자 가능 종목의 20% 개수까지 접근도순으로 뽑는다. 현재 종가를 포함한 최고가와 현재 종가를 비교하면 돌파를 잘못 정의할 수 있어, 돌파 기준 고점에서는 당일을 제외했다. 2008년 이전 전체 역사가 없어 역사적 신고가는 분석하지 않았다.</p>
    <div class="eq">M<sub>i,t</sub> = log(P<sub>i,t−21</sub> / P<sub>i,t−252</sub>)<br>S<sub>i,t</sub>(a) = (1−a)·rank<sub>Uₜ</sub>(M<sub>i,t</sub>) + a·rank<sub>Uₜ</sub>(A<sub>i,t</sub>), a∈{{0.25,0.50,0.75}}</div>
    <p>모멘텀은 최근 약 1개월을 제외한 과거 수익률이며 M&gt;0만 산다. rank는 동일 날짜 U에서의 0–1 백분위다. JOINT는 S 상위 20%를 산다. a=0.75는 <b>신호 점수에서 신고가 순위를 75% 반영</b>한다는 뜻이며 투자금 75%를 별도 신고가 계좌에 넣는 뜻이 아니다. 결합 롤링 후보에는 a=0에 해당하는 MOM20, a=1의 NEAR20도 남겨 신고가 추가가 필요 없으면 제외할 수 있게 했다. 다만 NEAR20만 A≥0.90 절대조건이 있으므로 결합식의 순수 연속 극한과 정확히 같지는 않다.</p>
    <div class="eq">동일비중: w<sub>i,t</sub> = 1 / |S<sub>t</sub>|<br>변동성 역수 비중: w<sub>i,t</sub> = σ<sub>i,t</sub><sup>−1</sup> / Σ<sub>j∈Sₜ</sub> σ<sub>j,t</sub><sup>−1</sup></div>
    <p>σ는 t−1까지 60거래일 로그수익률 표준편차이며 일 0.5% 하한을 둔다. 변동성이 2배인 종목에는 대략 절반의 비중을 배정한다. 이는 공분산 전체를 이용한 최적화나 기대수익률 최적해가 아니라, 설명 가능한 단순 위험 배분이다. 신규 추가 후 종목별 비중은 판단 시 NAV의 8% 이내로 제한하며, 이후 가격 상승으로 초과한 비중을 매일 강제로 매도하지 않는다.</p>
    <div class="eq">신규 묶음 예산 b<sub>t</sub> = NAV<sub>t</sub> × 5/H<br>주문수량 q<sub>i,t+1</sub><sup>의도</sup> = b<sub>t</sub> w<sub>i,t</sub> / P<sub>i,t</sub></div>
    <p>H=120이면 5거래일마다 약 4.17%의 예산으로 한 묶음을 만들고 약 24개 묶음을 겹쳐 운용한다. H=60이면 약 8.33%, 12개 묶음이다. 다음 날 실제 체결 가격과 비용 때문에 현금이 부족하면 매수량을 줄인다. 진입 예정 예산은 추가 납입금이 아니라 계좌 내부 자금이며 레버리지는 없다.</p>
    <h2>5. 비중조절 효과를 분리해서 확인</h2>
    <p>ROLL_HIGH_WEIGHT와 ROLL_HIGH_EW는 비중뿐 아니라 선택한 H와 선택 방식도 달라진다. 그래서 동일비중 계좌에서 골랐던 <b>날짜별 규칙·H를 그대로 유지하고 비중만 변동성 역수로 변경한</b> MATCHED 비교를 따로 계산했다.</p>{perf_table(perf,['ROLL_HIGH_EW','MATCHED_HIGH_IV','ROLL_MOM_EW','MATCHED_MOM_IV'],'deployment')}
    <p>신고가에서는 같은 선택경로의 비중만 바꾸면 연복리가 약 8.72% → 7.43%로 낮아졌다. 따라서 신고가 비중 선택 계좌의 11.18%를 “역변동성 배분만의 효과”라고 말할 수 없다. 반대로 모멘텀에서는 −0.21% → 1.57%로 개선됐다. 그래도 절대 성과는 낮았고, 전체 선택·현금 대기 구조를 함께 보아야 한다.</p>
    <h3>보유 중간에 신고가 신호를 넣는 경우</h3>
    <p>MID_HIGH_TILT는 ROLL_MOM_WEIGHT와 똑같이 종목을 사고, 각 묶음이 H의 절반에 도달했을 때 기존 종목 사이의 비중만 한 번 조정한다. 수익이 난 종목을 무조건 추가 매수하거나 외부 자금을 추가하는 방식이 아니다.</p>
    <div class="eq">중간 조정 전 묶음 금액 v<sub>i</sub>, 접근도 백분위 a<sub>i</sub><br>v′<sub>i</sub> = (Σv<sub>j</sub>) · [v<sub>i</sub>(0.5+a<sub>i</sub>)] / Σ[v<sub>j</sub>(0.5+a<sub>j</sub>)]</div>
    <p>고점에 가까운 종목에 더 큰 비중을 주되 보유 만기는 연장하지 않는다. 매수 상한 8%, 묶음별 변경금액이 계좌 NAV 0.05% 미만이면 거래하지 않는 기준이 있다. 상한·거래 최소금액·비용 때문에 주식비중이 정확히 같지는 않지만 평균 47.66% → 47.76%로 가까웠다.</p>
    <p>2022년 이후 연복리는 <b>10.69% → 11.31%</b>, Sharpe는 <b>0.597 → 0.626</b>, 최대낙폭은 <b>−27.72% → −27.62%</b>였다. 개선은 작고, 2017–2021에는 연복리 <b>−3.65% → −4.13%</b>로 악화됐다. 현재로서는 중간 조정을 기본 운용에 의무적으로 넣을 근거가 부족하다.</p>
    <h2>6. 개선의 불확실성</h2>
    <p>동일 날짜 수익률을 짝지어 60거래일 블록으로 2,000번 재표집했다. 자기상관과 두 계좌의 같은 날 움직임을 일부 보존한다. 아래 연평균 차이는 일평균수익률 차이×252이며 CAGR 차이와 다르다. 20일 블록 결과도 CSV에 남겼다.</p>{table(infer)}
    <p>신고가 추가·중간 조정의 구간은 대부분 0을 포함한다. 따라서 관찰된 수익률 차이를 안정적인 추가 수익으로 단정할 수 없다. 모멘텀의 동일 선택경로 역변동성 비교는 이 탐색적 구간에서 양수였지만, 이전까지의 모든 실험·후보 선택을 보정한 확정적 유의성 검정은 아니다. 새 데이터나 전향적 모의운용이 필요하다.</p>
    <h2>7. 시기별 수수료와 매도세</h2>
    <p>수수료는 <b>키움 KRX 온라인 일반요율</b>을 적용했다. 증권사 수수료와 세금은 다르다. 수수료는 매수·매도 양쪽에, 아래 세금은 매도금액에만 부과한다. 개인별 평생우대·신규가입 이벤트·NXT 요율은 포함하지 않았다.</p>
    <table><tr><th>양도·결제 기준 기간</th><th>코스피 거래세</th><th>농특세</th><th>합계 매도세</th></tr><tr><td>2009–2019.06.02</td><td>0.15%</td><td>0.15%</td><td>0.30%</td></tr><tr><td>2019.06.03–2020.12.31</td><td>0.10%</td><td>0.15%</td><td>0.25%</td></tr><tr><td>2021–2022</td><td>0.08%</td><td>0.15%</td><td>0.23%</td></tr><tr><td>2023</td><td>0.05%</td><td>0.15%</td><td>0.20%</td></tr><tr><td>2024</td><td>0.03%</td><td>0.15%</td><td>0.18%</td></tr><tr><td>2025</td><td>0.00%</td><td>0.15%</td><td>0.15%</td></tr><tr><td>2026</td><td>0.05%</td><td>0.15%</td><td>0.20%</td></tr></table>
    <p>표의 2019.06.03 시행은 2019.05.30 체결분부터 해당한다. 각 체결일에 저장된 실제 거래일 달력의 결제일을 대응해 과세했다. 재구성한 세율은 기존 세율 배열과 전 날짜에서 일치했다.</p>
    <table><tr><th>매매일 기준</th><th>매수 수수료</th><th>매도 수수료</th><th>성격</th></tr><tr><td>2009–2020.09.15</td><td>0.015%</td><td>0.015%</td><td>일반 온라인 요율</td></tr><tr><td>2020.09.16–12.31</td><td>0.0113604%</td><td>0.0113604%</td><td>확인된 한시 인하</td></tr><tr><td>2021–2026</td><td>0.015%</td><td>0.015%</td><td>일반 온라인 요율</td></tr></table>
    <p>2008–2009 연차보고서, 2013 약관, 2020 인하 공지, 2025 문서와 현재 안내를 대조했다. 매해 모든 개별 계좌 조건과 일시 이벤트를 완전 복원한 거래명세서는 아니다. 수수료에 유관기관 비용을 별도로 다시 더하지 않았다. 일반요율 계좌를 가정한 결과다.</p>
    <div class="eq">TC<sub>t</sub> = (매수금액+매도금액)×(수수료<sub>t</sub>+슬리피지+시장충격)<br>　　　　　 + 매도금액×매도세율<sub>t</sub><br>NAV<sub>t</sub> = 현금<sub>t</sub> + Σ 보유수량<sub>i,t</sub>×평가가격<sub>i,t</sub></div>
    <p>슬리피지 3bp + 시장충격 2bp, 합계 편도 5bp는 가정이다. 1bp=0.01%p다. 2026년 왕복 비용은 가격변동이 없을 때 약 0.015%×2 + 0.20% + 0.05%×2 = <b>0.33%</b>이며, 정확한 손익은 매수·매도 당시 실제 금액을 기준으로 계산했다. 투자자의 대주주 여부에 따른 양도소득세, 배당세와 개인 계좌 세제는 따로 모델링하지 않았다.</p>
    <h3>체결비용·회수금 가정을 바꾼 결과</h3>{table(se)}
    <p>execution13bp/25bp는 수수료·세금 외에 편도 슬리피지+충격을 각각 0.13%/0.25%로 높였다. 선택 일정은 그대로 두고 계좌를 다시 계산했다. recovery0은 편출 후 가격이 사라진 보유 종목을 마지막 가격의 100%로 회수한다는 기본 가정을 0%로 바꾼 스트레스다. 기본 회수금은 확인된 기업행위 현금흐름이 아니므로 이 민감도를 함께 보아야 한다. zero_cost는 세금까지 없앤 진단용이며 실현 가능한 계좌가 아니다.</p>
    <h2>8. 단기반전 대신 도입한다면</h2>
    <p>기존 마지막 분석의 모멘텀 계좌는 같은 2022–2026.07 구간 연복리 약 13.30%, 모멘텀+반전 계좌는 1.86%였다. 새 고정 신고가·결합 전략은 후자보다 높은 수익률을 보였다. 다만 기존 계좌는 다른 베이지안 예측·최적화·위험예산을 사용하므로, 이는 <b>원인 효과를 분리한 교체 실험이 아니라 과거 연구 결과와의 참고 비교</b>다. 이전 반전 신호를 동일한 최적화기에 신고가 변수로만 바꾼 전면 재학습은 이번 단순 규칙 실험과 구별해야 한다.</p>
    {perf_table(perf,['OLD_MOM','OLD_JOINT','FIXED_HIGH_WEIGHT','FIXED_JOINT_WEIGHT'],'deployment')}
    <p><b>제안은 “반전을 없애고 신고가만으로 전액 운용”이 아니라, 모멘텀을 기준 전략으로 유지하면서 신고가를 추가하는 후보를 따로 검증하는 것이다.</b> 수익률 우선이면 이 표본의 고정 모멘텀 40일, 낙폭과 단순성도 고려하면 고정 결합 60일을 비교 기준으로 남긴다. 신고가 단독이 목적이라면 120일을 기준점으로 두되 테스트기간의 큰 낙폭을 감수할 수 있는지를 먼저 평가해야 한다.</p>
    <p>추후 새 신호는 기본 전략 → 비중 방식만 변경 → 진입 시 신고가 추가 → 보유 중간 재조정 순으로 한 요소씩 비교한다. 변경 전 규칙·비용·검증기간을 기록하고, 이미 성과를 본 기간을 다시 최종 테스트라고 부르지 않는다. 중간 추가 신호의 비용 차감 개선이 새 자료에서도 반복될 때만 편입하는 것이 합리적이다.</p>
    <h2>9. 자료·실행 한계와 검증</h2>
    <ul><li>모든 날짜의 당시 편입 이력을 사용했다. 원자료의 편입 공시 시점과 수정가격의 과거 버전까지 별도로 감사한 것은 아니다.</li><li>현재가·거래량 양수, 과거 253개 가격 완비, 60일 변동성 확보 조건을 사용했다. 이상 일수익률은 신규 후보에서 10일 격리하지만 계좌 손익을 잘라내거나 보정하지 않았다.</li><li>신호는 t일 종가 이후, 체결은 t+1 종가다. 수정가격 기준 분할 단위를 사용하므로 실제 정수 주식·호가단위·주문장 체결을 완전히 재현하지 않는다. 거래량 0·결측 가격에서는 체결하지 않지만, 거래량이 있는 상한가·하한가의 주문 가능 여부는 알 수 없다.</li><li>수정종가를 사용했지만 현금배당·기업행위 지급액의 반영 범위는 독립 확인하지 못했다. 별도 배당 재투자 총수익률로 표시하지 않는다. 원거래대금·호가 데이터가 없어 운용자금 규모에 따른 시장 수용량도 확정할 수 없다.</li><li>앞으로의 매일 갱신에는 새로운 가격·편입 이력 입력이 필요하다. 현재 파일은 2026.07.31까지의 일별 재현 결과이며 예약 실행·실시간 주문은 설정하지 않았다.</li></ul>
    <p>검증 완료: 일정한 가격에서 정확한 왕복 비용, 다음 종가 체결, 현금 부족 시 매수 축소, 거래량 0일 미체결·다음 거래일 재시도, 실제 주문으로 현금·수량·NAV 독립 재구성, 학습 마지막 날짜가 판단일보다 이전인지 확인, 미래 데이터를 잘라낸 뒤 신호·선택·주문·NAV가 같은지 확인했다. 중간 재조정에서도 미래 제거 검사를 통과했다.</p>
    <details><summary>자료 마지막 날짜의 선택 상태와 보유 상위 15개</summary><p>2026.07.31 기준이다. 오늘 주문할 목록이 아니다. 가장 최근 선택과 실제 보유분은 다를 수 있다. 기존 묶음이 만기까지 남아 있기 때문이다.</p><pre>{html.escape(json.dumps(latest,ensure_ascii=False,indent=2))}</pre>{table(hold)}</details>
    <h2>10. 원자료와 재현 파일</h2>
    <p>원자료 SHA-256: <code>1cf35548707bb6732d5a6118f488b5cf5879be24fcb6f2c25b15ba108782a637</code>. 기존 자료를 복사해 사용했고 원본은 변경하지 않았다.</p>
    <p>ZIP에 실행 코드·입력자료·사전 프로토콜·96개 후보 결과·일별 계좌·주문내역·최종 보유·검증 결과를 포함했다. 수익률은 재현 가능한 역사적 계산이며 미래의 예상수익률이 아니다.</p>
    <h3>확인한 문헌·비용 출처</h3><ol>{''.join('<li><a href="'+u+'">'+html.escape(title)+'</a><br><span class="note">'+html.escape(note)+'</span></li>' for title,u,note in SOURCES)}</ol>
    <p class="note">작성일 2026.10.06 · 시간 순서에 따른 탐색적 검증 · 라이브 거래 없음</p></main></html>'''
    (OUT/'신고가전략_검증보고서.html').write_text(text,encoding='utf8')
    (OUT/'sources.json').write_text(json.dumps(SOURCES,ensure_ascii=False,indent=2),encoding='utf8')
    (OUT/'REPRODUCE.txt').write_text('Python dependencies: numpy pandas numba scipy matplotlib. Tested with the original research Python environment.\nRun from any directory, in this order: python engine.py; python overlay.py; python analyze.py; python phase_check.py; python validate.py; python report.py.\nKeep inputs.npz, all protocol files, and portable_inputs/ alongside scripts. No network download needed. Bundled local reference benchmark and earlier strategy snapshots are selected automatically.\nPrimary return results: final_performance.csv; all96: all_96_results.csv; all costs: cost_schedule.csv; matched uncertainty: paired_bootstrap.csv.\nPrices are adjusted fractional units, not actual brokerage shares. Cash rate0, no independent dividend payout reconstruction.\n',encoding='utf8')
    print('Report written',len(text))

if __name__=='__main__':main()
