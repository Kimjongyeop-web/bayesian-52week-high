# Bayesian 52-Week High Research

**진행 중 | 김종엽의 52주 신고가·주가 신호 베이지안 연구 기록**

52주 최고가 대비 가격 근접도와 종목 간 순위로 후보를 구성하고 베이지안 회귀의 사후 불확실성을 비용 반영 종목비중에 연결합니다.

| 단계 | 소스 | 상태 |
|---|---|---|
| 신고가 기준 | `research/high52_study` | 과거 실험 확보 |
| 베이지안 보유 | `research/high52_bayes` | 과거 실험 확보 |
| 상대 확률·비중 최적화 | `research/high52_bayes_optimized` | 과거 실험·보고서 확보 |
| 매일 판단 | `research/high52_daily` | 과거 실험·보고서 확보 |
| 매일 판단·회전율 제어 | `research/high52_daily_control` | 후속 검증 대상 |

연구는 진행 중입니다. 과거 비교실험의 존재를 최종 모형·실전 성과 검증 완료로 해석하지 않습니다.

## 설계·해석

근접도·순위·위험 추정의 시점을 구분하고 판단 다음 종가 체결과 순주문 비용을 모델링합니다. H=5/20, W=504/1008 등의 후보와 통제 규칙은 각 protocol.json에 있습니다. 사후 확률은 실제 거래 승률과 다릅니다.
회전율 제어 과거 기록의 2017년 이후 전체 평가 CAGR은 CONTROLLED 4.88%, UNCONTROLLED 6.46%입니다. 회전율 감소를 수익률 개선으로 단정하지 않습니다. `reports/high52_daily_control/performance.csv`를 함께 제공합니다. 수정주가의 분수 단위·기업행위 근사·반복적으로 본 표본이라는 한계가 있으며 이번에는 재학습하지 않았습니다.

## 재현·제한

원본 Excel부터의 통합 재현 경로는 정리 중입니다. 코드가 요구하는 inputs.npz는 원자료에서 만든 입력 스냅샷이므로 GitHub에서 제외하고 기존 로컬 위치에 보존합니다. 필수 필드: dates, codes, p(수정주가), v(거래량), m(편입마스크), settlement(정산일), benchmark(기준지수). 기준 연구의 추가 필드·portable_inputs 의존성은 해당 소스를 확인합니다.
새 실행 폴더에 research 구조를 복사해 동일 해시 입력을 준비합니다. optimized 실행 순서는 rebuild_posterior.py → study.py train → study.py evaluate → validate.py → diagnostics.py → report.py입니다. daily와 daily_control의 검증에는 전 단계 비교결과도 필요합니다. 재학습에는 시간이 필요합니다.

과거 HTML 보고서는 보존용 기록으로 일부 상세 데이터 링크가 미포함 자료를 가리킵니다. 필요한 데이터는 원본에서 재생성해야 합니다.

[진행 중 과제](IN_PROGRESS.md) · [데이터 정책](DATA_POLICY.md)

공동 연구 참고: https://github.com/minjunxjang/bayesian-strategy-allocation/tree/main/part1_momentum
이 링크의 최신 내용은 현재 권한으로 읽지 못했습니다. 해당 최신본과 동일하거나 개인 단독 저작물이라고 주장하지 않습니다.

## 격리 실행 폴더 만들기

먼저 python prepare_run.py --run-id research_001 을 실행하면 소스 복사본과 빈 work/outputs 폴더만 생성됩니다. 연구·다운로드·삭제는 실행하지 않습니다. 이후 위 재현 순서에 맞춰 입력과 작업 디렉터리를 준비하세요.

