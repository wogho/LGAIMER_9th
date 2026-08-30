# 08. Gemini 작업 위임서 — 1,000점 이상 달성

> 작성일: 2026-08-16  
> 위임자: GPT/Codex 작업 세션  
> 후속 담당: Gemini  
> 제출자: 김재호  
> 팀명: 나란차  
> Phase 3 참가 여부: 아니요  
> 현재 공식 최고점: **SUB-002 · `886.2488171351`**  
> 최소 목표: **공식 리더보드 `1,000`점 이상**

## 1. 이 문서의 역할

이 문서를 Gemini의 첫 진입점이자 최신 상태의 단일 인계 기준으로 사용한다. 이전 Gemini 제안과 과거 문서의 `[다음]` 표시는 실행 시점에 이미 완료됐을 수 있다. 작업 시작 전에 반드시 이 문서와 아래 네 문서를 대조한다.

1. `01_제약과금지사항.md`
2. `05_실험로그.md`
3. `06_제출체크리스트.md`
4. `07_1000점_달성계획.md`

문서 간 충돌이 있으면 대회 원문 규칙, 이 문서, 최신 실험 산출물의 순서로 판단하고 충돌 내용을 먼저 기록한다.

## 2. 절대 규칙

- 공식 Phase 2 데이터 외 외부 데이터 사용 금지.
- OpenAI, Gemini 등 원격 모델 API를 추론 또는 학습에 사용 금지.
- 모든 학습·추론은 로컬 코드로 재현 가능해야 한다.
- test 각 행은 독립적으로 예측한다.
- 특정 test 행 예측에 다른 test 행, test 전체 분포, 평균, 빈도, 순위, 분위수, groupby, rolling, lag를 사용하지 않는다.
- test 예측 전체를 보고 확률을 재조정하지 않는다.
- 보정기와 라우팅 규칙은 공식 train의 시간 OOF에서 미리 고정한다.
- 리더보드 점수만 보고 연속 가중치나 보정계수를 탐색하지 않는다.
- 하루 최대 제출은 5회다. 제출 직전 플랫폼의 남은 횟수를 직접 확인한다.
- 외부 플랫폼 업로드는 사용자의 명시적 지시 또는 확인 후 수행한다.

## 3. 점수 현황과 목표 수치

| 항목 | 값 |
| --- | ---: |
| SUB-001 | `815.20127` |
| SUB-002 | `886.2488171351` |
| SUB-001 대비 상승 | `+71.0475471351` |
| 상대 상승률 | `+8.7153%` |
| 최초 목표 간격 해소율 | `38.4459%` |
| 최소 목표 | `1,000` |
| 남은 점수 | `113.7511828649` |
| 현재 환산 BSS | `0.008862488171351` |
| 남은 BSS 개선량 | `+0.001137511828649` |
| 기준 Brier 0.25 가정 시 추가 필요 Brier 감소 | 약 `0.000284378` |

`0.000284378`은 평가 정답 평균을 알 수 없는 상태에서 규모를 이해하기 위한 근사치다. 공식 점수 상승을 보장하는 수치가 아니다.

## 4. 현재 공식 최고 모델 SUB-002

### 4-1. 구조

- 실험 ID: `REGIME-R-001-FINAL-E2E`
- 피처: FE-001 60개
- F 행: 전체 학습 CatBoost 259 rounds 100%
- R 행: R-only LightGBM 110 trees 75% + CatBoost 259 rounds 25%
- 후처리: 전체 행에 고정 global Platt
- Platt coefficient: `1.100620991038283`
- Platt intercept: `-0.02760275170069481`
- Platt fit OOF seasons: 2022, 2023, 2024
- 외부 데이터/API: 없음
- test 분포 기반 선택: 없음

### 4-2. 제출물과 해시

- 실제 제출 ZIP: `output/candidates/submit_regime_r_candidate.zip`
- ZIP SHA-256: `62de6d960c770cca03dc3bd9a0abac4d2364ae96426d95dd4292f5cd71993aa8`
- ZIP 크기: `534,864 bytes`
- 제출 예측 SHA-256: `83c7367bf3c61c63c162d41435f90f09b6315b2cde8dc00b6d8cdad58c746e00`
- R-only LightGBM SHA-256: `78ea48ab4b909b42ae8a2f870954992b2adcdc407999fe414ebefb6a25d54144`
- CatBoost SHA-256: `da9e0da9d07bd84a48694243c3bb5a09e1696d9efaed3c76aec7270f0f39a94a`
- feature columns SHA-256: `6dc53a45eb0aa70f46416f7bb4fdde758ce603fc022bd59776f5eb6477c41608`
- ZIP manifest: `output/candidates/submit_regime_r_candidate_manifest.json`

### 4-3. 중요한 경로 구분

루트 `script.py`, 루트 `model/`, `output/submit_final_selective.zip`은 SUB-001 계열의 활성/rollback 파일이다. SUB-002는 별도 candidate 경로에서 제출됐다. 후속 작업에서 루트 활성 파일을 현재 최고 모델로 오인하지 않는다.

- SUB-002 candidate source: `candidate/regime_r_submission/`
- SUB-002 R 모델: `model/LGBM-FE001-RONLY-FINAL-2019-2024-R110/`
- SUB-002 CatBoost: `model/CAT-FE001-FINAL-2019-2024-R259/`
- SUB-002 계약·검증: `model/REGIME-R-001-FINAL-E2E/`
- SUB-002 생성기: `scripts/build_regime_r_submit_zip.py`
- SUB-002 최종 검증기: `scripts/verify_regime_final_candidate.py`
- SUB-001 rollback ZIP: `output/submit_final_selective.zip`
- SUB-001 rollback SHA-256: `22dc61a85a5f6ea26e81645b6e21eed3c59584435c0a172b98779159f2b997ff`

현재 최고 ZIP과 rollback ZIP을 덮어쓰거나 삭제하지 않는다. 새 후보는 새 실험 ID·새 디렉터리·새 ZIP 이름을 사용한다.

## 5. SUB-002 로컬 검증과 공식 결과 해석

| 시즌 | SUB-001 구조 대비 Brier 변화 | 해석 |
| --- | ---: | --- |
| 2022 | `+0.000037280` | 허용 범위 내 악화 |
| 2023 | `-0.000736780` | 큰 개선 |
| 2024 | `-0.000052155` | 작은 개선 |
| 2023·2024 평균 | `-0.000394467` | 기존 Gate A PASS |

공식 점수도 `+71.0475471351` 상승했으므로 R 전용 구조와 전역 보정 방향은 유효하다. 그러나 2023·2024 양의 Brier 개선 합계 중 `93.3892%`가 2023에서 발생했고, 최신 2024 개선은 작았다. 따라서 로컬 평균 개선량이 암시한 상승폭보다 실제 전이는 약했다.

핵심 결론:

1. SUB-002를 rollback하지 않는다. 현재 공식 최고점이다.
2. REGIME-R 방향을 버리지 않는다. 공식 평가에서 양의 전이가 확인됐다.
3. 다음 후보는 2024·최근 시즌 성능을 더 강하게 요구한다.
4. SUB-001과 SUB-002 두 점만으로 최적 가중치를 역산하지 않는다.
5. 1,000점까지는 미세 보정보다 새로운 직교 신호 또는 R 모델 다양성이 필요할 가능성이 높다.

## 6. 이미 완료했거나 폐기한 실험

아래 작업은 동일 가설·동일 설정으로 반복하지 않는다.

| 실험 | 결과 | 후속 원칙 |
| --- | --- | --- |
| `DIAG-1000-001` | R calibration gap 반복 확인 | REGIME-R로 연결 완료 |
| `CAL-SEL-OOF-001` global Platt | 2023·2024 개선 | SUB-002에 포함 |
| R-only Platt | 평균 개선 `0.000081570` | 기존 구조 그대로 반복 금지 |
| `ENS-SEED-001` | 상관 `0.991725`, 개선 `0.000020472` | 시드 추가 중단 |
| FE-002 state/form/support | 시즌 반복 개선 실패 | 동일 피처 묶음 반복 금지 |
| target aggregates | 시간 일반화 실패 | 같은 단순 집계 반복 금지 |
| `BLEND-RECENT-001` | Platt 대비 추가 `0.000002807` | 폐기 |
| season 제거 | 2024 악화 | 폐기 |
| game_type 제거 | 다수 시즌 악화 | 폐기 |
| sample-weight 0.85/0.70 | 2024 악화 | 폐기 |
| min_child_samples 150/200 | 안정적 개선 실패 | 폐기 |
| TrackMan 선수 조인 | 공식 crosswalk 부재·ID 교집합 0 | 중단 유지 |
| Isotonic | 시즌 일관성 실패 | 반복 금지 |

과거 Gemini 문서의 exact count, form, support 계열 제안은 FE-002에서 이미 선행 검증됐다. 과거 `07` 문서 9절을 새로운 TODO로 해석하지 않는다.

## 7. SUB-003 이후 성능 게이트

모든 새 후보는 SUB-002 OOF와 같은 행에서 paired 비교한다.

### 필수 기술 게이트

- 공식 train만 사용.
- expanding-window 또는 엄격한 과거→미래 OOF.
- 단일행·배치, 순열, 무관행 추가 예측 차이 `0.0`.
- 245,789행 600초·28GB·ZIP 10GB 이하.
- dtype, 피처 순서, 모델 tree 수, 파일 SHA-256 고정.
- ZIP 내부 독립 실행과 workspace E2E 예측 해시 일치.

### 개정 성능 게이트

- 2024 Brier 개선: 원칙적으로 `0.00010` 이상.
- 2022 Brier 악화: `0.00005` 이하.
- 2023 개선이 전체 가중 개선의 70% 이상을 지배하면 보류.
- 최신 시즌 calibration gap과 ECE 동시 악화 금지.
- 2022·2023·2024 최악 시즌 변화량을 반드시 보고.
- 보조 지표로 최근 시즌 가중 평균을 사용하되 가중치는 실험 전에 고정.

권장 최근 시즌 가중치는 `2022=0.20`, `2023=0.30`, `2024=0.50`이다. 다른 가중치를 쓰려면 결과 확인 전에 근거를 기록한다.

## 8. Gemini 실행 우선순위

### P0. 인계 검증 및 기준선 동결

1. 이 문서와 `01`, `05`, `06`, `07`을 읽는다.
2. SUB-002 ZIP SHA-256과 manifest를 재확인한다.
3. `model/REGIME-R-001/regime_results.json`과 E2E report를 읽는다.
4. SUB-002 OOF를 새 비교 기준으로 로더 하나에 고정한다.
5. 기존 dirty worktree와 사용자 파일을 보존한다.

완료 조건: 현재 최고점, 후보 경로, rollback 경로, OOF 행 수·해시가 하나의 새 round-2 summary에 기록된다.

### P1. `TRANSFER-DIAG-002` — CV–LB 전이 진단

목적은 LB 정답을 추정하는 것이 아니라 로컬 선택 기준의 편향을 줄이는 것이다.

- SUB-002의 raw REGIME-R와 global Platt 효과를 2022·2023·2024에서 분리한다.
- 2023 개선 편중률, 최근 시즌 가중 개선, 최악 시즌 변화를 계산한다.
- R 0.50/0.50 후보와 0.75/0.25 후보를 새 게이트로 다시 비교한다.
- 이 단계에서는 새 제출 ZIP을 만들지 않는다.

주의: R 0.50 후보는 2024 개선과 2022 안정성이 더 좋지만 2023 개선은 작다. LB 결과를 근거로 곧바로 0.50을 제출하지 말고 새 게이트와 calibration을 먼저 검증한다.

### P2. `CAL-REGIME-002` — REGIME-R 전용 시간 OOF 보정

- SUB-002 raw regime 예측을 기준으로 global Platt를 재현한다.
- 고정 후보로 Platt, 저복잡도 Beta calibration, F/R 분리 Platt를 비교한다.
- 모든 보정기는 해당 검증 시즌보다 과거 OOF만 사용한다.
- 파라미터 수, clipping, 실패 시 identity fallback을 계약에 기록한다.
- 2024 `0.00010` 개선 게이트를 못 넘으면 보정만으로 SUB-003를 만들지 않는다.

기존 R-only Platt 실패는 SUB-001 선택형 예측 기준이었다. REGIME-R raw 예측 기준의 분리 보정은 별도 가설이므로 검증 가능하지만, 과적합 위험 때문에 2~3개 고정 후보만 허용한다.

### P3. `CAT-RONLY-001` — R 전용 CatBoost 다양성

- FE-001 그대로 R 행만 학습한 CatBoost를 2022·2023·2024 expanding-window에서 검증한다.
- R-only LightGBM과의 잔차 상관·불일치 구간을 계산한다.
- 가중치는 사전 고정 후보 `0.50/0.50`, `0.75/0.25` 이하로 제한한다.
- F는 현재 CatBoost를 그대로 유지해 변경 축을 R 모델 하나로 제한한다.
- 최신 시즌 개선과 모델 다양성이 모두 확인될 때만 전체 학습으로 확장한다.

### P4. `REGIME-CONFIDENCE-001` — 표본 신뢰도 기반 고정 라우팅

- 현재 행에 이미 제공된 `asof_pitcher_n`, `asof_batter_n`, `asof_pitcher_pitchmix_n`, cold-start flag만 사용한다.
- 임곗값과 블렌드 규칙은 train OOF에서 사전 고정한다.
- test 빈도나 test 분포로 임곗값을 정하지 않는다.
- 복잡한 다단계 라우팅보다 `충분한 이력/부족한 이력` 2구간부터 시작한다.
- 최소 두 시즌 반복 개선과 최신 시즌 `0.00010` 개선이 없으면 폐기한다.

### P5. 제출 후보 결합

P2~P4 중 개별 게이트를 통과한 구성만 결합한다. 개별로 실패한 후보를 여러 개 합쳐 평균상 좋아 보이게 만들지 않는다. 결합 후에도 개정 게이트와 모든 기술 게이트를 다시 통과해야 한다.

## 9. 제출 슬롯 운영

- SUB-002는 완료됐고 현재 최고점이다.
- SUB-003: 최근 시즌 게이트를 통과한 단일 구조 가설.
- SUB-004: SUB-003와 직교하는 후보 또는 명확한 원인 분리.
- SUB-005: 최고 후보 재현 또는 비상 rollback 슬롯.
- 같은 날 제출이라면 이론상 최대 3회가 남지만 플랫폼 표시가 최종 기준이다.
- 게이트 통과 후보가 없으면 제출하지 않고 `886.2488171351`을 보존한다.

각 제출 전에 아래를 기록한다.

1. 실험 ID와 단일 가설.
2. 2022·2023·2024 paired Brier/BSS/calibration.
3. 최신 시즌 개선과 최악 시즌 악화.
4. 코드·모델·계약·ZIP SHA-256.
5. 독립성·benchmark·격리 E2E 결과.
6. 남은 제출 횟수와 rollback ZIP.

## 10. 재현 명령과 산출물 규칙

작업 환경 루트:

```text
/home/ubuntu/orca/workspaces/LG AIMer/Infra-setup
```

기본 Python은 `.venv/bin/python`을 사용한다. 새 실험은 반드시 `scripts/`에 재현 스크립트를 남기고 `model/<EXPERIMENT_ID>/`에 JSON·Markdown·OOF 예측·metadata를 저장한다.

SUB-002 점검 명령:

```bash
sha256sum output/candidates/submit_regime_r_candidate.zip
unzip -t output/candidates/submit_regime_r_candidate.zip
.venv/bin/python scripts/verify_regime_final_candidate.py
.venv/bin/python scripts/audit_checklists.py
```

새 후보는 기존 builder를 덮어쓰지 말고 별도 builder와 별도 ZIP을 만든다. 활성 루트 파일 교체는 독립 후보가 모든 게이트를 통과한 뒤에만 검토한다.

## 11. Gemini가 첫 답변에서 보고할 내용

1. 이 문서와 필수 규정 문서를 읽었다는 확인.
2. 현재 최고점 `886.2488171351`과 목표 잔여 `113.7511828649` 재확인.
3. SUB-002 실제 ZIP 해시 재확인.
4. P0 인계 검증 결과.
5. P1 `TRANSFER-DIAG-002` 실행 계획과 사전 고정 지표.
6. 규칙 또는 파일 상태에서 발견한 충돌.

## 12. 완료 정의

- 공식 리더보드 `1,000`점 이상.
- 해당 점수의 ZIP·코드·모델·계약·학습 환경·해시 보존.
- train 시간 OOF 결과와 공식 LB 결과 연결.
- 평가 행 독립성, 외부 데이터/API 금지, 누수 방지 PASS.
- `05`, `06`, `07`, 이 위임서, 최종 PPT에 달성 경로 반영.

목표 미달 상태에서 작업을 종료해야 한다면, 최고점과 rollback을 보존하고 실행 완료·실패·미실행 항목을 이 문서 하단에 추가한다.
