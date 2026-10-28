from pathlib import Path
import json,base64,sys,zipfile,hashlib,platform
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
names={'UNCONTROLLED':'매일 판단·제어 없음','CONTROLLED':'매일 판단·제어+베이지안','CONTROLLED_EQUAL':'동일 제어·균등비중','DEFAULT_RULES':'강한 제어·20일 50%'}
def table(df):return df.to_html(index=False,border=0,escape=True)
def main():
    p=pd.read_csv(OUT/'performance.csv');sel=json.loads((OUT/'selection.json').read_text());a=pd.read_csv(OUT/'activity.csv');val=json.loads((OUT/'validation.json').read_text())
    plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
    fig,ax=plt.subplots(figsize=(11,4.7))
    for n,label in names.items():
        d=pd.read_csv(OUT/'daily'/f'{n}.csv',parse_dates=['date']);d=d[d.date>='2017-01-02'];ax.plot(d.date,d.nav/1e7,label=label,lw=1.4)
    ax.set_ylabel('2017년 최초 원금 대비 순자산');ax.grid(alpha=.2);ax.legend(fontsize=9);fig.tight_layout();fig.savefig(OUT/'control_result.png',dpi=150);plt.close(fig)
    png=base64.b64encode((OUT/'control_result.png').read_bytes()).decode()
    def perf(period):
        x=p[p.period==period].copy();x['name']=x.name.map(names)
        for col in ['cagr','mdd','exposure','annual_cost_rate']:x[col]=x[col].map(lambda v:f'{v:.2%}')
        x['annual_turnover']=x.annual_turnover.map(lambda v:f'{v:.2f}배')
        return table(x[['name','cagr','mdd','exposure','annual_turnover','annual_cost_rate']].rename(columns={'name':'방식','cagr':'순 CAGR','mdd':'최대낙폭','exposure':'평균 주식비중','annual_turnover':'연 환산 매수+매도','annual_cost_rate':'연 환산 비용/NAV'}))
    acts=a.copy();acts['name']=acts.name.map(names)
    acts=acts[['name','days','review_days','fill_days','limit_binding_days']].rename(columns={'name':'방식','days':'거래일','review_days':'판단일','fill_days':'체결일','limit_binding_days':'일반주문 축소일'})
    latest=pd.read_csv(OUT/'latest_close_decisions.csv',dtype={'ticker':str})
    body=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>매일 판단·회전율 제어 검증 및 다음 전략</title><style>body{{max-width:1180px;margin:36px auto;padding:0 24px;font:17px/1.7 'Malgun Gothic',sans-serif;color:#192b40;background:#f8fafc}}h1,h2{{line-height:1.3}}h2{{margin-top:38px}}table{{border-collapse:collapse;width:100%;font-size:14px;background:white}}td,th{{border-bottom:1px solid #dce3ed;padding:10px;text-align:right}}td:first-child,th:first-child{{text-align:left}}.box{{background:#e8effa;border-left:5px solid #3064b0;padding:20px}}.scroll{{overflow-x:auto}}a{{color:#215ca7}}img{{width:100%}}code{{background:#e5eaf0;padding:3px}}small{{color:#52647a}}</style>
    <h1>매일 판단 + 회전율 제어</h1><p>52주 신고가 접근도 · 베이지안 비중 조절 · 모멘텀 수익률 신호 제외<br>자료 마지막 날짜: 2026-07-31 | 작성: 2026-10-06</p>
    <div class="box"><b>매 거래일 종목과 비중을 다시 계산하고, 조건을 충족한 차이만 거래합니다.</b><br>최근 구간에서 회전율과 비용은 줄었지만, 순 CAGR은 제어 전 10.45%에서 제어 후 10.31%로 소폭 낮아졌습니다. 낙폭은 완화됐습니다. 또한 2017년부터 전체 기간에서는 동일 제어의 균등비중이 베이지안보다 높은 CAGR을 기록했습니다. 베이지안의 일관된 우월성은 확인되지 않았습니다.</div>
    <h2>1. 적용 조건</h2><p>학습·내부 검증으로 선택한 규칙: <code>{sel['selected']['name']}</code>. 매일 종가로 판단하여 다음 거래일 종가에 체결하는 가정입니다. 하루 중 미리 알 수 없는 종가에 같은 날 체결하지 않습니다.</p>
    <ul><li>진입: 수정종가 / 최근 252거래일 수정종가 최고값 ≥ 90%, 접근도 순위 상위 10%. 최대 20종목, 목표 종목비중 상한 10%.</li>
    <li>일반 매도: 접근도 88% 미만 또는 순위 상위 25% 밖인 상태가 2거래일 연속 확인될 때. 일수가 찼다는 이유로 전량 청산하지 않습니다.</li>
    <li>재진입: 실제 전량 매도일로부터 3거래일이 지나야 다시 매수 대상으로 검토합니다.</li>
    <li>비중 유지 구간: 현재비중과 목표비중 차이가 max(0.5%p, 현재비중의 20%) 이하면 수량 유지. 청산 또는 상한 초과 조정은 예외입니다.</li>
    <li>일반 주문 실제 체결한도: 하루 매수+매도 합계 ≤ 거래 전 NAV의 10%. 최근 20거래일 각각의 거래/NAV 비율 합계 ≤ 100%. 한도 부족 시 주문을 비례 축소하고 다음 날 다시 판단합니다.</li>
    <li>편입대상 제외 또는 접근도 80% 미만은 긴급 매도이며 일반 한도에서 제외합니다. 최초 자금 투입도 예외입니다. 예외 거래까지 모두 실제 회전율과 비용에 포함합니다. 접근도 80%는 매입가 대비 손절률이 아닙니다.</li></ul>
    <p>처음 제시했던 20일 50% 한도도 별도로 비교했습니다. 선택 규칙은 100%입니다. 강한 제어 규칙은 매도 확인 3일·재진입 대기 5일·비중 무거래 구간 1%p로 함께 강화한 묶음이므로, 두 결과 차이를 한도 하나의 효과로 해석하면 안 됩니다. 가격 변동과 부분 체결 때문에 실제 종목비중은 목표 10%를 넘을 수 있습니다.</p>
    <h2>2. 비용 차감 성과</h2><p>최근: 2022-01-01 ~ 2026-07-31. 모든 계좌는 2017-01-02 현금 1천만 원으로 시작하며 2022년에 재설정하지 않습니다. CAGR은 일별 수익률에 연 252거래일을 적용합니다.</p><div class="scroll">{perf('recent')}</div>
    <p>초기 외부 평가: 2017 ~ 2021</p><div class="scroll">{perf('test')}</div><p>전체 외부 평가: 2017 ~ 2026-07</p><div class="scroll">{perf('all_test')}</div><img src="data:image/png;base64,{png}">
    <p>비용률은 매일 수수료·매도세금·체결비용을 NAV로 나눈 값의 연 환산 평균입니다. CAGR 차이와 동일한 개념이 아닙니다. 매수와 매도를 합친 회전율이므로 통상적인 편도 회전율과 직접 비교하지 마세요. 개선에는 비용 절감뿐 아니라 보유종목·보유기간·현금비중 변화가 함께 작용합니다.</p>
    <h2>3. 매일 판단은 유지했는가</h2>{table(acts)}<p>일반 한도가 작동해도 긴급 매도는 가능합니다. 따라서 총 일 회전율이나 총 연간 회전율의 절대 상한을 보장하는 방식은 아닙니다. 마지막 날짜의 목표 주문은 아직 다음 날 체결된 것으로 계산하지 않았습니다.</p>
    <h2>4. 확률·학습·검증의 의미</h2><p>A=P/max(P,252일), z=clip((A−0.9)/0.1,−5,1), 설명변수 X=(1,z,z²,1−2·순위비율). 과거 60일 변동성으로 표준화한 미래 5거래일 수익률을 정규–역감마 베이지안 회귀로 학습합니다. 보유기간 5일 고정 청산이나 5일마다 판단한다는 뜻이 아닙니다. 결과가 이미 관측된 표본만 사용하며 1,008거래일 창을 매일 이동합니다.</p>
    <p>pᵢ=P(종목의 조건부 기대수익률 &gt; 현재 후보 바스켓 평균의 조건부 기대수익률 | 과거 자료). 이는 실제 수익이 날 확률이 아닙니다. p가 45~55%이면 비중 신호를 0으로 두고, 그 밖에서는 s=sign(p−0.5)·max(|p−0.5|−0.05,0)/0.45, vᵢ∝exp(0.5sᵢ)로 완만하게 조정합니다. 이후 ½Σ(w−v)²+3Σ[매수비용·(w−u)₊+매도비용·(u−w)₊]를 비중·예산 제약 아래 최소화합니다. 회전율 제어는 그 뒤 실제 주문에 적용됩니다.</p>
    <p>32개 제어 묶음을 2009~2013 학습, 2014~2016 내부 검증에서 선택하고 2017년 이후로 평가했습니다. 학습 양의 CAGR·연 회전율 15배 이하, 내부 검증 양의 CAGR·15배 이하·기준 대비 MDD 악화 5%p 이내를 요구했습니다. 학습 Sharpe 상위 5개 중 내부 검증 Sharpe로 선택했습니다. 연 15배는 선택 기준일 뿐 실제 하드 한도가 아닙니다. 앞선 전략 연구에서 같은 과거 데이터를 반복해서 보았으므로, 완전히 새로운 미관측 검증으로 간주할 수 없습니다.</p>
    <h2>5. 검산과 한계</h2><p>독립적인 수량·현금·NAV 재구성, 거래가능일, 매일 판단, 실제 회전율 분해와 20일 누적 한도, 기존 일별 기준 결과 재현, 평가 종료일을 앞당긴 경로 일치, 합성자료의 매도 확인·재진입 대기·익일 체결·긴급 예외를 검증했습니다. <a href="validation.json">검증 로그</a>.</p>
    <p>과거 KOSPI200 관련 407개 종목 자료를 사용합니다. 수정종가의 소수 수량 모형이며 정수 주식·호가·실제 체결·모든 기업행사를 재현하지 않습니다. 일부 제외 후 가격 소실 종목은 마지막 가격으로 현금 회수합니다. 특히 메리츠화재 주식교환은 실제 교환으로 재현하지 못했습니다. 회수율 0% 민감도도 실제 기업행사 처리를 대체하지 못하며 일반적인 최악 시나리오를 뜻하지 않습니다.</p>
    <p>키움 온라인 일반 수수료 0.015%와 확인된 2020년 한시 인하, 시기별 매도세금, 편도 체결비용 0.05%를 반영했습니다. <a href="https://www1.kiwoom.com/h/domestic/stock/VStockMainView?dummyVal=0">키움 안내</a> · <a href="https://download.kiwoom.com/notice/notice_20200915_02_01.pdf">2020년 공지</a> · <a href="sources.json">세율 등 출처</a> · <a href="sensitivity.csv">편도 0.13%/0.25% 및 회수율 민감도</a>.</p>
    <h2>6. 모멘텀을 제외한 다음 연구 후보</h2><p><b>우선 후보는 박스권 조건부 평균회귀입니다.</b> 최근 하락률 순서대로 사는 방식에서 벗어나, 과거 자료에서 범위 내 왕복이 관측되는 상태인지 판단하고 평균 대비 이탈이 클 때만 거래합니다. 이는 넓게는 반전 전략이며, 새로운 수익원이 검증됐다는 뜻이 아닙니다. 하락 추세를 박스권으로 오인하면 손실이 커질 수 있습니다.</p>
    <p>가격의 로그 이탈 x에 x(t+1)=φx(t)+ε를 두고 베이지안으로 φ와 수익·손실 분포를 함께 추정하는 방식을 시험할 수 있습니다. 0&lt;φ&lt;1일 때 반감기 h=log(0.5)/log(φ)입니다. 평균과 계수는 당시 과거 자료만으로 추정해야 하며, 단순 이동평균 차감만으로 평균회귀가 증명되지는 않습니다. 체결 지연을 반영한 비용 후 기대손익과 하방 위험이 진입·보유·매도의 기준입니다.</p>
    <p>두 번째는 시장·업종 수익률을 제거한 잔차의 과도한 하락과 회복입니다. 업종 분류의 시점별 자료가 추가로 필요합니다. 세 번째는 관련 종목 쌍의 가격 차이 회복을 노리는 페어 거래입니다. 상대가치 전략의 연구 근거는 있으나 현재 한국 시장 수익성을 보장하지 않으며, 매도 쪽 대주 가능성과 차입비용을 별도로 반영해야 합니다. 매수 쪽만 실행하면 시장중립이 아닙니다. <a href="https://www.nber.org/papers/w7032">페어 거래 연구</a> · <a href="https://www.kiwoom.com/h/banking/credit/VCreditLoanInfo2View">키움 대주 조건</a>.</p>
    <p>단기반전이 나빴던 원인은 별도로 세전·비용 후 손익, 신호 후 첫 체결까지의 수익, 거래 횟수와 보유기간, 극단 손실 및 기업행사로 분해해야 합니다. 아직 해당 단기반전 원본을 재검산하지 않아 원인을 확정하지 않았습니다. 베이지안은 불확실성을 추정하는 도구이며 음의 기대수익 전략을 자동으로 양의 전략으로 바꾸지 않습니다.</p>
    <h2>7. 파일과 재현</h2><p><a href="performance.csv">전체 성과</a> · <a href="control_search.csv">32개 학습 비교</a> · <a href="activity.csv">거래 활동</a> · <a href="latest_close_decisions.csv">마지막 종가 판단과 목표</a> · <a href="README.txt">실행 설명</a> · <a href="회전율제어_재현자료.zip">데이터·코드·보고서 ZIP</a></p><p>마지막 판단표는 과거 자료의 모의 목표이며 현재 시점 주문 지시가 아닙니다.</p></html>'''
    (OUT/'신고가_매일판단_회전율제어_보고서.html').write_text(body,encoding='utf-8')
    (OUT/'README.txt').write_text('매일 판단·회전율 제어 연구\nPython: control_run.py → validate_control.py → report.py\n필요 패키지: numpy pandas scipy numba threadpoolctl matplotlib\n입력과 posterior_summary.npz 포함. 사후확률 재계산: posterior.py의 prepare/forecast 참고. study.py는 공통 함수 모듈이며 이전 CLI는 사용하지 마세요.\n판단 매일, 다음 종가 체결. 마지막 판단은 미체결. 일반 일10%,20일100%; 최초 투자와 긴급 매도 예외. DEFAULT_RULES는20일50%.\n성과·한계·학습선택 기준은 HTML/protocol.json 참고. 정수주식/기업행사를 완전히 재현한 실거래 결과가 아닙니다.\n',encoding='utf-8')
    (OUT/'environment.json').write_text(json.dumps(dict(python=sys.version,platform=platform.platform()),indent=2))
    files=[x for x in OUT.rglob('*') if x.is_file() and '__pycache__' not in x.parts and x.suffix!='.zip' and x.name!='manifest_sha256.json']
    (OUT/'manifest_sha256.json').write_text(json.dumps({str(x.relative_to(OUT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in files},indent=2))
    with zipfile.ZipFile(OUT/'회전율제어_재현자료.zip','w',zipfile.ZIP_DEFLATED) as z:
        for x in files+[OUT/'manifest_sha256.json']:z.write(x,x.relative_to(OUT))
    print('REPORT AND ZIP READY')
if __name__=='__main__':main()
