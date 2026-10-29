# 52주 신고가와 베이지안 비중 배분

**진행 중** — 신고가 근접도로 종목을 고르고, 베이지안 추정의 불확실성을 포트폴리오 비중에 반영하는 연구입니다.

[실험 현황](IN_PROGRESS.md) · [연구 코드](research/) · [결과 자료](reports/)

## 현재 확인한 결과

회전율을 줄인 실험에서 수익률이 함께 개선되지는 않았습니다.

| 2017년 이후 전체 평가 | CAGR |
|---|---:|
| 회전율 제어 적용 — CONTROLLED | 4.88% |
| 제어 미적용 — UNCONTROLLED | 6.46% |

출처: [회전율 제어 성과표](reports/high52_daily_control/performance.csv). CAGR은 연평균 복리수익률입니다. 반복해서 관찰한 표본의 중간 결과이며, 최종 모형의 미관찰 구간 성과는 아직 평가하지 않았습니다.

![회전율 제어 비교](reports/high52_daily_control/control_result.png)

## 연구 질문

52주 최고가에 대한 가격 근접도와 종목 간 순위로 후보군을 구성합니다. 베이지안 회귀의 사후 불확실성을 이용해 종목 비중과 현금 비중을 정하고, 거래비용을 반영해 평가합니다.

비교 대상은 보유기간, 매일 선정·청산하는 방식, 거래 횟수를 줄이는 규칙입니다. 사후 확률을 실제 거래 승률로 해석하지 않습니다.

## 실험 구성

| 단계 | 코드 | 비교 내용 |
|---|---|---|
| 기준 전략 | [high52_study](research/high52_study/) | 신고가 신호와 보유기간 |
| 베이지안 보유 | [high52_bayes](research/high52_bayes/) | 불확실성을 반영한 보유 판단 |
| 비중 최적화 | [high52_bayes_optimized](research/high52_bayes_optimized/) | 상대 확률과 종목 비중 |
| 매일 판단 | [high52_daily](research/high52_daily/) | 일별 선정·청산·재배분 |
| 회전율 제어 | [high52_daily_control](research/high52_daily_control/) | 거래 감소 규칙, 후속 검증 중 |

<details>
<summary>비중 최적화 실험 결과</summary>

![비중 최적화 실험](reports/high52_bayes_optimized/equity_exposure.png)

[성과표](reports/high52_bayes_optimized/performance.csv) · [연도별 결과](reports/high52_bayes_optimized/annual_results.csv)

</details>

## 재현에 필요한 입력

`inputs.npz`의 기본 필드는 다음과 같습니다.

| 필드 | 내용 |
|---|---|
| `dates`, `codes` | 날짜, 종목 코드 |
| `p`, `v` | 수정주가, 거래량 |
| `m` | 편입 마스크 |
| `settlement`, `benchmark` | 결제 관련 입력, 비교지수 |

원본 입력과 입력 스냅샷은 공개하지 않습니다. 기준 연구에는 추가 필드와 `portable_inputs` 의존성이 있습니다. H=5/20, W=504/1008 등 설정은 각 실험의 `protocol.json`에 기록돼 있습니다.

비중 최적화 실험의 실행 순서는 다음과 같습니다.

`rebuild_posterior.py` → `study.py train` → `study.py evaluate` → `validate.py` → `diagnostics.py` → `report.py`

daily·daily_control 검증은 앞 단계의 비교 결과도 사용합니다. 일부 HTML 보고서 링크가 가리키는 상세 데이터는 별도로 생성해야 합니다.

`python prepare_run.py --run-id research_001`은 실행 폴더를 구성합니다. 원본 Excel부터 전체 실험까지 이어지는 재현 경로는 정리 중입니다.

## 남은 검증

- 회전율 감소와 현금 비중 변화의 효과 분리
- 거래비용, 수정주가의 분수 단위, 기업행위 근사에 따른 민감도
- 아직 관찰하지 않은 후속 구간 평가
- 공동 저장소 최신본과 개인별 기여 범위 대조

공동 연구 경로: [bayesian-strategy-allocation / part1_momentum](https://github.com/minjunxjang/bayesian-strategy-allocation/tree/main/part1_momentum). 최신본과의 일치 여부와 기여 범위는 아직 확정하지 않았습니다.

[실험 현황](IN_PROGRESS.md) · [데이터 범위](DATA_POLICY.md)
