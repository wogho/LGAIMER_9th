# LG Aimers 전체 진행 체크리스트

> 기준일: 2026-08-16  
> 표기: `[x]` 완료 · `[ ]` 미완료/다음 작업
> 체크리스트 업데이트 시 마다, 01_제약과금지사항.md 재 검증 매번 진행
> `[x]`에는 현재 활성 상태뿐 아니라 완료·기각·보류가 근거와 함께 확정된 과거 실험도 포함한다.

## 0. 현재 종합 상태

- [x] 대회 문제·평가 방식·제출 규칙 문서화
- [x] 로컬 개발 환경과 Python 3.11 제출 검증 환경 분리
- [x] 시즌 기반 Holdout 학습·검증 파이프라인 구축
- [x] LightGBM 개발 기준 모델 `LGBM-001` 확보
- [x] P0 제출 코드 안정성 보완 완료
- [x] 동일 설정 반복 재현성 PASS
- [x] 2023 expanding-window 검증 수행
- [x] 현재 판정 확정: 개발 기준 모델 조건부 PASS
- [x] 현재 판정 확정: 시즌 이동 안정성 HOLD
- [x] 이전 판정 기록: 최종 2025 제출 모델 HOLD
- [x] 선택형 앙상블의 2022·2023·2024 시즌 이동 안정성 검증
- [x] 최종 모델 확정 및 2019~2024 전체 재학습
- [x] 현재 판정: 선택형 최종 제출 후보 기술 승인
- [x] 첫 공식 리더보드 제출 점수 `815.20127` 확인
- [x] 두 번째 공식 제출 `SUB-002` 점수 `886.2488171351` 확인
- [x] 다음 최소 목표 `1,000`점 및 현재 필요 개선폭 `113.7511828649`점 확정
- [ ] 공식 리더보드 `1,000`점 이상 달성

## 1. 문제·규정·데이터 사용 원칙

- [x] Brier Score·BSS 공식 산식 확인
- [x] 로컬 환산 점수와 공식 Public Score 구분
- [x] 평가 행 독립성 원칙 문서화
- [x] test 행 간 `groupby`, `rolling`, `shift`, 빈도·분포·순위 사용 금지 확인
- [x] test 예측 전체를 이용한 사후 보정 금지 확인
- [x] 현재 투구 이후 정보·정답·실제 구종·위치 사용 금지 확인
- [x] 외부 API 및 비공식 외부 데이터 사용 금지 확인
- [x] 공식 train 기반 사전 집계만 허용하는 원칙 확정
- [x] 생성 AI 작성 코드의 참가자 검토 책임 확인
- [x] 피처·보정·앙상블 변경에 대한 데이터 출처와 누수 여부 재검토
- [x] 최종 제출 직전 전체 추론 경로 수동 코드 리뷰

## 2. 데이터 및 기본 EDA

- [x] `train.csv`, `test.csv`, `sample_submission.csv` 확보
- [x] `trackman_history.csv` 확보
- [x] train 2019~2024 시즌 범위 확인
- [x] 평가 데이터 2025 시즌 구조 확인
- [x] 시즌별 행 수와 타깃 성공률 확인
- [x] 2019~2023 평균 성공률 `0.53158` 확인
- [x] 2024 성공률 `0.48610` 확인
- [x] 시간적 타깃 비율 하락 확인
- [x] 공식 입력 피처 47개 확정
- [x] 범주형 피처 5개 확정
- [x] train 49개 컬럼별 결측률·고유값·수치 `min/p0.1/p99.9/max` 전수 점검
- [x] 2019~2024 시즌별 16개 `asof_*` 결측률 전수 점검
- [x] 2023·2024 투수·타자·투수팀·타자팀 ID 신규 등장률 점검
- [x] 2023·2024 선택형 예측의 entity·`asof_*` cold-start 구간 Brier 점검
- [x] 메인↔TrackMan 선수 ID 교집합 0·공식 crosswalk 부재로 안전한 직접 매핑 불가 확인
- [x] 공식 crosswalk 제공 전 TrackMan 미사용·보류 결정 기록

## 3. 개발·제출 환경

- [x] 개발용 가상환경 구성
- [x] 제출 검증용 Python `3.11.15` 환경 구성
- [x] 제출 환경 pandas `2.0.3` 확인
- [x] 제출 환경 numpy `1.26.4` 확인
- [x] 제출 환경 LightGBM `4.7.0` 확인
- [x] 초기 `requirements_submit.txt`를 `lightgbm==4.7.0`으로 최소화
- [x] 최종 `requirements_submit.txt`를 LightGBM `4.7.0`·CatBoost `1.2.10`으로 확정
- [x] 평가 서버 기본 패키지의 불필요한 재설치 제거
- [x] `scripts/validate_env.py` 활성 추론 의존성 기준으로 갱신
- [x] CatBoost 전이 의존성 누락을 `pip check`로 검출하고 제출 검증 환경 보완
- [x] 최종 requirements `--no-index --dry-run` 및 `pip check` PASS
- [x] 최종 모델 확정 후 requirements 구성·의존성 재확인
- [x] 최종 제출 ZIP의 네트워크 없는 추론 E2E 확인

## 4. 학습·추론 피처 계약

- [x] 공용 학습 피처 함수 `src.features.build_features()` 구성
- [x] `row_id`, `control_success` 모델 입력 제외
- [x] 학습 피처 목록 `feature_columns.json` 저장
- [x] `script.py`에서 피처 계약 JSON 필수 로드
- [x] 누락 피처 Fail-Fast 적용
- [x] 예상하지 않은 추가 피처 Fail-Fast 적용
- [x] 입력 컬럼 순서가 달라도 학습 순서로 복원
- [x] LightGBM 모델 피처 이름·순서·개수와 JSON 대조
- [x] 50행 학습·추론 피처 값·dtype·순서 일치 확인
- [x] 최종 FE-001 파생 피처가 `src/features.py`와 활성 `script.py`에서 exact 일치함을 재검증
- [x] 최종 FE-001 60피처 계약 JSON 재생성 및 두 모델 재검증

## 5. P0 제출 코드 안정성

- [x] `script.py`를 LightGBM 네이티브 모델 단일 경로로 변경
- [x] RandomForest fallback 제거
- [x] 제출 ZIP에서 `rf.pkl` 제외
- [x] 로컬 `rf.pkl`을 과거 개발 산출물로만 분리
- [x] 전체 재학습 `rf.pkl`의 부적절한 2024 Holdout 비교 제거
- [x] 탐색 학습의 활성 모델 자동 덮어쓰기 방지
- [x] `--sync-root-model` 기본 비활성화
- [x] 최종 후보에서만 명시적 모델 동기화하도록 변경
- [x] ZIP 허용 목록을 네 파일로 고정
- [x] 중복 모델·검증 예측·실험 메타데이터 ZIP 제외
- [x] P0 검증 ZIP `submit_LGBM_001_P0_20260815_210741.zip` 생성
- [x] P0 검증 ZIP 크기 `0.32MB` 확인
- [x] P0 검증 ZIP 격리 E2E PASS
- [x] 최종 모델 ZIP `output/submit_final_selective.zip` 생성

## 6. 출력 무결성 및 행 독립성

- [x] 입력·제출 `row_id` 결측·중복 차단
- [x] 입력과 제출 양식 행 수 불일치 차단
- [x] 입력과 제출 양식 ID 집합 불일치 차단
- [x] 예측 개수 불일치 차단
- [x] placeholder 기반 누락 예측 은폐 제거
- [x] 예측 NaN·Inf 차단
- [x] 예측 범위 `[0, 1]` 검사
- [x] 출력 컬럼 `row_id`, `control_success` 고정
- [x] 출력 ID 순서 보존 확인
- [x] 추론 중 고위험 배치 연산·재학습 호출 정적 검사
- [x] singleton 예측 불변성 PASS
- [x] permutation 예측 불변성 PASS
- [x] augmentation 예측 불변성 PASS
- [x] 세 독립성 검사의 최대 오차 `0.000e+00` 확인
- [x] 최종 모델·피처·앙상블 활성화 후 5행·50행 독립성 검사 재실행

## 7. 기준 모델 및 검증 결과

- [x] 2019~2023 학습 / 2024 Holdout 고정
- [x] Train 평균 상수 모델 측정
- [x] `LGBM-001` 학습
- [x] `LGBM-001` Best iteration `99` 확인
- [x] 2024 Brier `0.248267` 확인
- [x] 2024 BSS `0.006163` 확인
- [x] 2024 로컬 환산 점수 `616.35` 확인
- [x] 2024 평균 예측 편향 `+1.05%p` 확인
- [x] `616.35`가 공식 수료선 통과 근거가 아님을 문서화
- [x] `LGBM-001-R1`, `LGBM-001-R2` 반복 학습
- [x] 원본·R1·R2 Best iteration 완전 일치
- [x] 원본·R1·R2 모델 SHA-256 완전 일치
- [x] 원본 대비 R1·R2 검증 예측 최대 차이 `0.0` 확인
- [x] 반복 재현성 PASS
- [x] 활성 제출 모델 해시 불변 확인

## 8. Expanding-window 및 시즌 안정성

- [x] 2019~2022 학습 / 2023 검증 수행
- [x] `LGBM-EW-2023` Best iteration `9` 확인
- [x] 2023 Brier `0.251103` 확인
- [x] 2023 BSS `-0.004413` 확인
- [x] 2023 평균 예측 편향 `+3.28%p` 확인
- [x] 2023 모델이 Train 평균 상수 Brier보다 `0.000463` 개선됨을 확인
- [x] 2023 Valid 평균 기반 BSS 기준선 미달 확인
- [x] 시즌 이동 안정성 HOLD 판정
- [x] 2019~2021 학습 / 2022 검증 추가
- [x] 2019~2020 학습 / 2021 검증 실행
- [x] 2021 Brier `0.245864`, BSS `0.012302` 확인
- [x] 2021 Best iteration `57`, 평균 예측 편향 `-0.74%p` 확인
- [x] 2021 10분위 ECE `0.015436`, 상위 구간 오차 `-0.069221` 확인
- [x] 2021 `F` 예측 `0.593044` / 실제 `0.703840` / 오차 `-0.110796` 확인
- [x] 2022 Brier `0.243579`, BSS `0.022415` 확인
- [x] 2022 Best iteration `90` 확인
- [x] 2022 평균 예측 편향 `+0.25%p` 확인
- [x] 2022·2023·2024 Brier·BSS 평균과 표준편차 계산
- [x] 2022·2023·2024 10분위 calibration ECE 비교
- [x] 2023 최상위 예측 구간의 `+0.111318` 과대 예측 확인
- [x] 2023 최상위 예측 10분위 선수·팀·`asof_*` 결측·범주 분포 진단
- [x] 상위 10분위의 `99.98%`가 `game_type=F`임을 확인
- [x] `game_type=F` 실제 성공률의 2022 `0.708749` → 2023 `0.472904` 급락 확인
- [x] `F` 검증군 예측 `0.585762` / 실제 `0.472904` / 오차 `+0.112859` 확인
- [x] 직전 경기 피처 결측 여부에 따른 상위 구간 오차가 유사함을 확인
- [x] 과거 미등장 타자 여부에 따른 상위 구간 오차가 유사함을 확인
- [x] 단일 선수·결측보다 `game_type=F`의 시간적 관계 반전이 주원인이라고 판정
- [x] 2021~2024 평균 Brier `0.247203`, 표준편차 `0.002796` 계산
- [x] 2021~2024 평균 BSS `0.009117`, 표준편차 `0.009731` 계산
- [x] `F` 오차 표준편차 `0.079442`가 `R`의 `0.007847`보다 약 10.1배 큼을 확인
- [x] `F` 체제 전환과 팀 ID 대리 학습이 2023 Best iteration 급락의 주요 원인임을 확인
- [x] 고정 선택형 앙상블이 2022·2023·2024 세 시간 Holdout에서 단독 모델 대비 개선됨을 확인

## 9. 최우선 다음 실험

- [x] `game_type` 제외 설정을 2021 Holdout에 적용
- [x] `game_type` 제외 설정을 2022 Holdout에 적용
- [x] `game_type` 제외 설정을 2023 Holdout에 적용
- [x] `game_type` 제외 설정을 2024 Holdout에 적용
- [x] 포함/제외 설정의 4개 시즌 평균·표준편차·최악 BSS 비교
- [x] 제외 설정이 2023만 개선하고 2021·2022·2024를 악화시킴을 확인
- [x] 제외 설정 평균 Brier `0.247439`, 평균 BSS `0.008168` 확인
- [x] 제외 설정 Brier 표준편차 `0.001985`, 최악 BSS `-0.000219` 확인
- [x] 제외 설정 `F` 오차 표준편차 `0.086046`으로 기준보다 악화됨을 확인
- [x] 팀 ID가 `game_type=F` 대리 변수로 작동함을 확인
- [x] `NO-GAMETYPE` 폐기 및 기존 `game_type` 포함 기준 유지
- [x] Gemini 18번 의견과 기존 손실 근거를 반영해 `NO-GAMETYPE-NOTEAM` 미실행 결정

- [x] `season` 피처 포함 기준 모델 재측정
- [x] `season` 피처 제외 모델 구현
- [x] 2023 Holdout에서 season 포함/제외 비교
- [x] 2024 Holdout에서 season 포함/제외 비교
- [x] season 제외가 평균 예측 편향에 미치는 영향 확인
- [x] 최근 시즌 가중치 후보 `0.85`, `0.70` 정의
- [x] 동일 가중치 후보를 2023·2024 Holdout에 공통 적용
- [x] 가중치별 Brier·BSS·평균 예측 편향 기록
- [x] 한 시즌만 개선되는 설정 제외
- [x] 두 Holdout 공통 개선 후보 없음 확인
- [x] 기존 `season` 포함·무가중 `LGBM-001` 유지 결정
- [x] FE-001 기반 선택형 앙상블을 2022·2023·2024 공통 개선 최종 기준으로 선정

## 10. 단일 행 파생 피처 실험

- [x] 볼·스트라이크 `count_state` 피처 추가
- [x] `count_state`에 투수·타자 유리 및 중립 상태 반영
- [x] `count_state`에 풀카운트 상태 반영
- [x] 득점권 주자 상황 피처 추가
- [x] 점수 차·레버리지 압박 및 후반 이닝 피처 추가
- [x] 투수·타자 좌우 매치업 피처 추가
- [x] `asof_*` 결측 개수·여부·cold-start 플래그 추가
- [x] `asof_*` 값 간 안전한 차이·비율 피처 추가
- [x] 피처 묶음별 단일 변경 실험
- [x] FE-001 전체 묶음의 2021~2024 네 Holdout 평가
- [x] 네 Holdout 모두 Brier 개선 확인
- [x] FE-001 평균 Brier `0.247013`, 평균 BSS `0.009879` 확인
- [x] 2024 Brier `0.248065`, BSS `0.006973`, 로컬 환산 `697.29` 확인
- [x] 활성 47피처와 FE-001 60피처 계약·행 독립성 모두 PASS
- [x] FE-001을 차기 개발 후보로 채택하고 활성 모델 교체는 보류
- [x] 채택·보류 결과 `05_실험로그.md` 기록
- [x] 결측·cold-start 3개 제거가 2023·2024 모두 개선됨을 확인
- [x] 상황 4개 제거는 시즌별 방향이 갈려 그룹 단위 삭제를 폐기
- [x] 핵심 매치업·폼·비율 6개 제거는 두 시즌 모두 악화되어 유지
- [x] 57피처 `FE-001-NOMISSING`을 1차 정제 후보로 선정
- [x] `FE-001-NOMISSING`을 2021 Holdout에서 재검증
- [x] `FE-001-NOMISSING`을 2022 Holdout에서 재검증
- [x] 2021·2022에서 각각 Brier가 악화되어 4개 Holdout 공통 개선 조건 미충족 확인
- [x] 57피처 `FE-001-NOMISSING` 정제안 폐기
- [x] 정규화 실험 기준을 기존 FE-001 60피처로 유지
- [x] 학습 파이프라인에 `--min-child-samples` 옵션과 양수 검증 추가
- [x] FE-001 60피처에서 `min_child_samples=150`을 2023·2024 Holdout에 적용
- [x] FE-001 60피처에서 `min_child_samples=200`을 2023·2024 Holdout에 적용
- [x] `150`이 두 Holdout 모두 악화됨을 확인하고 폐기
- [x] `200`이 2023만 개선하고 2024를 악화시킴을 확인하고 폐기
- [x] 두 Holdout 공통 개선 정규화 후보 없음 확인
- [x] FE-001의 `min_child_samples=100` 기본 설정 유지

## 11. Train-only 집계 및 TrackMan

- [x] 집계 피처 생성 코드를 행 단위 피처·추론 코드와 분리
- [x] 학습 행용 집계를 엄격한 이전 시즌 expanding 방식으로 생성
- [x] 2024 검증 집계를 2019~2023 데이터만으로 생성
- [x] 투수별 과거 성공률·표본 수 2024 Holdout 1차 실험
- [x] 타자별 과거 상대 성공률·표본 수 집계를 엔터티 일반형 모듈로 구현
- [x] 타자 집계 strict prior-season 누수 차단·fallback·lookup 검증
- [x] 타자 smoothing 50을 2023 Holdout에서 선행 검증
- [x] 2023 Brier `0.250923`, BSS `-0.003691`로 FE-001보다 악화됨을 확인
- [x] 신규·기존 타자 구간이 모두 악화되어 타자 집계 폐기
- [x] 2023 선행 게이트 실패로 타자 집계의 2024 Holdout 미실행 결정
- [x] 타자 단독 타깃 과거 집계 실험 종료
- [x] 팀·시즌·구종·조건부·TrackMan 집계 타당성 감사 수행
- [x] 메인↔TrackMan 투수·타자 ID 교집합 0 확인
- [x] TrackMan 공통 문맥 키 다중 후보율 `0.985330`으로 1:1 조인 불가 확인
- [x] 공식 crosswalk 부재로 TrackMan 선수 물리 profile 실험 보류
- [x] test에 현재 구종이 없어 현재 구종별 과거 통계 lookup 불가 확인
- [x] 팀 ID↔TrackMan 팀명 공식 crosswalk 부재로 팀 물리 집계 보류
- [x] 투수×`count_state` 2023 커버리지 `0.861737`, support 중앙값 `1187` 확인
- [x] 투수×득점권 집계는 가능하지만 count 조건부 편차보다 후순위 판정
- [x] 투수×`count_state` 조건부 성공률 편차 두 피처 구현
- [x] 첫 학습 시즌·신규 투수·미관측 투수×count_state를 `n=0`, `delta=0`으로 처리
- [x] 조건부 성공률을 같은 투수의 전체 smoothing 성공률로 수축
- [x] 동일 시즌 타깃 불변성·fallback·행 순서 결정성·lookup 재적용 검증
- [x] 투수×`count_state` 조건부 성공률 편차를 2023 Holdout에서 단일 검증
- [x] 2023 Brier `0.250933`, BSS `-0.003732`로 FE-001보다 악화됨을 확인
- [x] 조회·미조회 구간 모두 악화되고 10분위 ECE가 `0.033716`으로 악화됨을 확인
- [x] 2023 선행 게이트 실패로 조건부 count 편차의 2024 Holdout 미실행 결정
- [x] 투수×`count_state` 조건부 편차 폐기 및 활성 모델 미변경
- [x] 투수×득점권 여부 조건부 성공률 편차 두 피처 구현
- [x] 득점권을 현재 행의 2루 또는 3루 주자 존재 여부로 고정
- [x] 첫 시즌·신규 투수·미관측 투수×득점권 상태를 `n=0`, `delta=0`으로 처리
- [x] 동일 시즌 타깃 불변성·fallback·결정성·lookup 재적용 검증
- [x] 투수×득점권 여부 조건부 성공률 편차를 2023 Holdout에서 단일 검증
- [x] 2023 Brier `0.250954`, BSS `-0.003817`, ECE `0.033249` 확인
- [x] 조회·미조회 및 득점권·비득점권 구간이 모두 악화됨을 확인
- [x] 2023 선행 게이트 실패로 득점권 편차의 2024 Holdout 미실행 결정
- [x] 투수×득점권 조건부 편차 폐기 및 타깃 집계 확장 종료
- [x] 표본 수 기반 Bayesian smoothing 100 적용
- [x] 신규 투수용 train 전체 성공률 fallback 고정
- [x] 사전 집계 lookup과 집계 계약을 `model/` 실험 산출물로 저장
- [x] 집계 후보를 폐기해 최종 test 추론에 train-only lookup 자체를 포함하지 않음
- [x] TrackMan 매핑 안전성 점검 결과 공식 crosswalk 전까지 프로파일 미실행 결정
- [x] 누수·fallback·결정성·lookup 재현·62피처 모델 계약 재검증
- [x] 2024 Brier `0.248048`, BSS `0.007040`, 로컬 환산 `703.99` 확인
- [x] 10분위 ECE 악화로 단일 시즌 채택을 보류
- [x] 동일 투수 과거 집계 설정을 2023 Holdout에서 재검증
- [x] 2023 Brier `0.250905`, BSS `-0.003619`로 FE-001보다 악화됨을 확인
- [x] smoothing 100이 2024만 개선하고 2023을 악화시켜 폐기
- [x] smoothing 100의 2021·2022 확대 검증과 test lookup 통합 미실행 결정
- [x] 투수 과거 집계 smoothing 30을 2023 Holdout에서 단일 변경 검증
- [x] smoothing 30의 2023 Brier `0.250930`, BSS `-0.003720` 확인
- [x] smoothing 30이 FE-001과 smoothing 100보다 모두 악화됨을 확인하고 폐기
- [x] 2023 선행 게이트 실패로 smoothing 30의 2024 Holdout 미실행 결정
- [x] 투수 단독 타깃 과거 집계 실험 종료

## 12. Calibration·대안 모델·앙상블

- [x] FE-001 2021~2024 expanding-window OOF `993,592`행 생성
- [x] OOF 원본 모델의 60피처·하이퍼파라미터·시간 순서·저장 Brier 계약 검증
- [x] calibration 학습 데이터와 평가 데이터를 시즌 순서로 분리
- [x] 2023은 2021~2022, 2024는 2021~2023 OOF만 보정 학습에 사용
- [x] 전역 Platt scaling 실험
- [x] 전역 Isotonic regression 실험
- [x] 보정 전후 Brier·BSS·ECE·`F/R` calibration 비교
- [x] 같은 Holdout으로 보정 학습·평가한 결과 제외
- [x] 평가 보정기 파라미터를 JSON 형식으로 저장
- [x] Platt가 2024만 개선하고 2023을 악화시켜 전역 적용 폐기
- [x] Isotonic이 2023·2024 Brier 공통 개선에 실패해 폐기
- [x] `R` 행 전용 시간 OOF Platt와 `F` 무보정 조합 검증
- [x] `F` 예측 적용 전후 exact identity 검증
- [x] R-only Platt 2023 Brier `0.250718`, BSS `-0.002872` 확인
- [x] R-only Platt 2024 Brier `0.247952`, BSS `0.007424` 확인
- [x] R-only Platt가 2023·2024 Brier·BSS·ECE를 모두 개선해 게이트 PASS
- [x] 2021~2024 전체 `R` OOF `881,587`행 기반 미래 추론용 Platt JSON 계약 생성
- [x] 미래 Platt JSON에 OOF·계약 SHA-256과 적용 범위·identity 정책 저장
- [x] pickle·scikit-learn 없는 NumPy 보정 로드·적용 함수 구현
- [x] FE-001 후보 모델과 R-only Platt 결합 E2E 추론 검증
- [x] JSON source hash·60피처 계약·확률 범위·단독 행·순서 불변성 PASS
- [x] 미래 보정기를 source OOF에서 재평가하지 않고 시간 평가 결과만 근거로 유지
- [x] 보정 helper가 활성 `script.py` main 예측을 변경하지 않음을 확인
- [x] 최종 선택형 후보에서 보정 미채택을 확정하고 보정 JSON을 활성 main·ZIP에 연결하지 않음
- [x] XGBoost 3.4.0 로컬 categorical·native JSON 지원 확인
- [x] 제출 검증 환경에 XGBoost가 없고 최소 requirements에도 미포함임을 확인
- [x] XGBoost FE-001 60피처·2023 동일 Holdout 학습 파이프라인 구현
- [x] XGBoost model JSON·피처 순서·예측·Brier 재현 검증
- [x] XGBoost 2023 Brier `0.251001`, BSS `-0.004002` 확인
- [x] XGBoost가 전체·F·R·ECE 모두 LightGBM FE-001보다 악화됨을 확인
- [x] XGBoost 2023 선행 게이트 실패로 2024·제출 환경 설치 미실행 및 폐기
- [x] CatBoost 1.2.10 로컬 categorical·native CBM 지원 확인
- [x] 제출 검증 환경에 CatBoost가 없고 최소 requirements에도 미포함임을 확인
- [x] CatBoost FE-001 60피처·2023 동일 Holdout 학습 파이프라인 구현
- [x] CatBoost CBM·피처 순서·검증 예측·Brier exact 재현
- [x] CatBoost 2023 Brier `0.249979`, BSS `0.000083`로 FE-001 개선 확인
- [x] 2023 게이트 통과 후 CatBoost 2024 Holdout 확대 검증
- [x] CatBoost 2024 Brier `0.247982`, BSS `0.007306`로 FE-001 개선 확인
- [x] CatBoost가 최근 두 시즌 전체 Brier·BSS·ECE를 공통 개선해 후보 승격
- [x] 2023 `R` 소폭 악화와 최적 rounds `2/259` 불안정성 기록
- [x] 임시 Python 3.11 제출 스택 CatBoost wheel·CBM 추론 호환성 PASS
- [x] CatBoost 최종 채택 후 `requirements_submit.txt` 변경·dependency metadata·ZIP 격리 검증
- [x] LightGBM·CatBoost 고정 50:50 평균을 2023·2024에서 검증
- [x] 가중치를 사전 고정하고 검증·test 예측 기반 가중치 탐색을 하지 않음
- [x] 두 원천 모델의 시즌·60피처 계약·`row_id`·정답 일대일 일치 확인
- [x] 고정 50:50 Brier 2023 `0.250135`, 2024 `0.247976` 확인
- [x] 2023 CatBoost 대비 `0.000155` 악화로 전체 고정 50:50 채택 게이트 FAIL 및 폐기
- [x] `R` 구간 50:50이 2023·2024 모두 두 단독 모델보다 개선됨을 확인
- [x] FE-001 후보 모델의 시즌별 OOF 예측 저장
- [x] 고정 평균 앙상블 실험
- [x] 앙상블 가중치와 F/R 규칙을 검증 시즌에서 고정하고 test 기반 탐색을 하지 않음
- [x] `F=CatBoost`, `R=고정 50:50` 선택형 앙상블을 사전 선언해 2023·2024 검증
- [x] `F` CatBoost·`R` 50:50 적용 exact identity를 두 시즌 모두 확인
- [x] 선택형 Brier 2023 `0.249714`, 2024 `0.247962` 확인
- [x] 선택형이 CatBoost 대비 2023 `0.000265`, 2024 `0.000020` 개선해 후보 게이트 PASS
- [x] 2024 ECE·평균 편향 악화와 동일 Holdout 진단 파생 규칙이라는 한계 기록
- [x] 고정된 선택 규칙을 2022 추가 시간 Holdout에서 재검증
- [x] CatBoost 2019~2021 학습/2022 Holdout Brier `0.243425`, best rounds `215` 확인
- [x] CatBoost 2022 native CBM·60피처·247,472행 예측·Brier 재현 PASS
- [x] 2022 선택형 Brier `0.243381`로 CatBoost 대비 `0.000044` 개선 확인
- [x] 2022 `F=CatBoost`, `R=50:50` 구간별 우세 방향과 적용 exact identity 확인
- [x] 선택형이 2022·2023·2024 세 시즌 CatBoost Brier를 공통 개선해 일반화 게이트 PASS
- [x] 2022·2024 ECE 악화와 작은 Brier 개선 폭을 최종 채택 위험으로 기록
- [x] test 예측 분포 기반 가중치·보정 금지 재확인

## 13. 최종 모델 선정 및 전체 재학습

- [x] 최종 FE-001 60피처 목록 확정
- [x] LightGBM 100 trees·CatBoost 259 rounds와 최종 하이퍼파라미터 확정
- [x] 최종 Best iteration 결정 기준 확정
- [x] CatBoost Holdout best rounds `215/2/259`와 동일 60피처·설정 계약 확인
- [x] 2023 체제를 학습에 포함한 최신 2024 시간 Holdout을 CatBoost rounds source로 선정
- [x] CatBoost 최종 반복 수 `259` 고정 및 평균·중앙값·데이터량 배율 조정 배제
- [x] CatBoost 전체 학습 시 validation·early stopping·`use_best_model` 미사용 계약 확정
- [x] CatBoost 최종 반복 수 정책 JSON·Markdown 결정론적 생성
- [x] 정책 기반 2019~2024 CatBoost 최종 후보를 격리 학습하고 native CBM 검증
- [x] CatBoost 전체 1,475,092행·FE-001 60피처·고정 259 rounds 계약 확인
- [x] CatBoost 최종 학습에서 validation·early stopping·성능 평가·test 예측 미사용 확인
- [x] CatBoost native CBM 259개 트리·피처 순서·전체 학습행 예측 지문 재현 PASS
- [x] CatBoost 최종 후보를 `model/CAT-FE001-FINAL-2019-2024-R259/`에 격리
- [x] LightGBM FE-001 Holdout `81/11/100`과 동일 60피처·설정 계약 확인
- [x] 2023 체제를 학습에 포함한 최신 2024 시간 Holdout을 LightGBM iteration source로 선정
- [x] LightGBM 최종 반복 수 `100` 고정 및 평균·중앙값·데이터량 배율 조정 배제
- [x] LightGBM 전체 학습 시 validation·early stopping 미사용 계약 확정
- [x] LightGBM 최종 반복 수 정책 JSON·Markdown 결정론적 생성
- [x] 정책 기반 2019~2024 LightGBM FE-001 최종 후보를 격리 학습하고 native model 검증
- [x] LightGBM 전체 1,475,092행·FE-001 60피처·무가중치·100 trees 계약 확인
- [x] LightGBM 최종 학습에서 validation·early stopping·성능 평가·test 예측 미사용 확인
- [x] LightGBM native model 100개 트리·피처 순서·전체 학습행 예측 지문 재현 PASS
- [x] LightGBM 최종 후보를 `model/LGBM-FE001-FINAL-2019-2024-R100/`에 격리
- [x] 두 최종 후보와 `F=CatBoost`, `R=50:50` 규칙의 2025 test E2E 추론 계약 검증
- [x] 두 모델 metadata·native 파일·공통 FE-001 60피처 해시 검증
- [x] 실제 test `R` 5행의 50:50 exact identity 및 `F` 분기 단위 테스트 PASS
- [x] 단독 행/배치 일치·행 순열 불변성·확률 범위 PASS
- [x] sample submission row_id 집합·순서·스키마 PASS
- [x] test 예측 분포로 선택 규칙·가중치·보정을 변경하지 않음
- [x] 후보 submission·계약·지문·보고서 결정론적 재생성 PASS
- [x] 선택형 후보 전용 추론 스크립트·requirements·ZIP 화이트리스트 구성 및 Python 3.11 격리 E2E
- [x] 후보 스크립트에 FE-001 60피처·명시적 dtype·두 native 모델 해시 계약 고정
- [x] 후보 requirements에 `lightgbm==4.7.0`, `catboost==1.2.10`만 명시
- [x] `.venv-submit` Python 3.11 환경의 pandas `2.0.3`·numpy `1.26.4`·LightGBM `4.7.0`·CatBoost `1.2.10` 확인
- [x] 후보 ZIP을 스크립트·requirements·두 모델·피처 계약·앙상블 계약의 정확히 6개 파일로 제한
- [x] 입력 dtype 누락으로 발생한 개발 E2E 차이 `0.002286` 검출·수정 후 최대 차이 `0.0` 확인
- [x] Python 3.11 후보 submission SHA-256이 개발 E2E `3f575460...99fb`와 동일함을 확인
- [x] 후보 ZIP 단일행·행 순열·무관 행 추가 독립성 최대 차이 모두 `0.0`
- [x] 후보 ZIP 재생성 결정론 및 SHA-256 `22dc61a8...97ff`, 크기 `500,976` bytes 확인
- [x] 패키징 중 활성 스크립트·requirements·모델·submission 미변경 확인
- [x] 활성 전환 전 후보 ZIP·활성 파일 handover manifest 및 rollback 감사
- [x] 후보 ZIP 6개 파일과 활성 대상 6개 경로의 일대일 activation map 고정
- [x] 활성 5개 파일의 해시·크기를 결정론적 rollback ZIP에 보존
- [x] rollback exact restore 5경로와 후보 전용 제거 3경로 계약 고정
- [x] 임시 루트 activation→rollback dry-run 및 실제 활성 상태 불변 PASS
- [x] rollback ZIP SHA-256 `3ba5f768...2a08`, handover manifest SHA-256 `cdbbca6d...f23a` 결정론 확인
- [x] handover 감사 시 `01_제약과금지사항.md` 재검토 및 규정 문서 해시 고정
- [x] Gemini 53번 승인과 사용자 진행 요청 후 선택형 후보를 활성 경로에 원자적 동기화
- [x] 2022·2023·2024 시즌별 성능 안정성 승인
- [x] 별도 확률 보정 없이 고정 선택형 앙상블 채택 확정
- [x] 최종 실험 ID `ENS-CATF-LGBMCATR5050-FINAL-ACTIVE` 확정
- [x] 2019~2024 전체 데이터 최종 재학습
- [x] Holdout 모델과 최종 모델 파일·실험 ID 분리
- [x] 최종 `model.txt` 또는 네이티브 모델 저장
- [x] 최종 `feature_columns.json` 저장
- [x] 최종 구성에 별도 train-only 집계·보정 산출물이 불필요함을 확정
- [x] 최종 모델 메타데이터 저장
- [x] 승인된 handover manifest 기반 atomic activation으로 최종 후보만 활성화
- [x] 활성 스크립트·requirements·두 모델·피처·앙상블 계약 해시 기록

## 14. 최종 제출 게이트

- [x] `.venv-submit/bin/python scripts/validate_env.py` PASS
- [x] `.venv-submit/bin/python scripts/verify_features.py` PASS
- [x] `.venv-submit/bin/python scripts/verify_independence.py` PASS
- [x] 대표 50행 확대 독립성 검사 PASS·최대 차이 `0.0`
- [x] `.venv-submit/bin/python scripts/dry_run.py` PASS
- [x] `.venv-submit/bin/python scripts/dry_run.py --benchmark` PASS
- [x] 245,789행 추론 시간 `4.49`초로 10분 이내 확인
- [x] 최대 메모리 `488.4MB`로 28GB 이하 확인
- [x] 최종 requirements 최소 의존성 및 dependency metadata 확인
- [x] 최종 ZIP 정확히 6개 허용 파일 확인
- [x] 최종 ZIP `500,976` bytes로 10GB 이하 확인
- [x] 최종 ZIP 격리 E2E PASS
- [x] 최종 ZIP 내부에 `rf.pkl`·검증 CSV·중복 모델 없음 확인
- [x] 실제 업로드 대상 ZIP SHA-256 `22dc61a8...97ff` 기록
- [x] 최종 제출 후보 기술 승인
- [x] `01_제약과금지사항.md` 최종 8개 항목 전수 재검증 PASS
- [x] 최종 ZIP 12개 의도적 오류 입력 Fail-Fast 전수 PASS
- [x] `start_all`·`01`·`06` 체크리스트 item-level evidence ledger 생성·미분류 사전 항목 0 확인

## 15. 리더보드 제출 및 사후 기록

- [x] 공식 리더보드 첫 제출 `SUB-001` 완료
- [x] 첫 리더보드 점수 `815.20127` 기록
- [x] 수료 기준 `549.51` 충족 확인
- [x] `SUB-001`의 로컬 시간 검증과 리더보드 점수 연결
- [x] 제출 ID·실험 ID·모델 해시·ZIP 해시 연결
- [x] 제출 결과를 `05_실험로그.md`에 추가
- [x] 두 번째 제출 `SUB-002` 완료 및 `886.2488171351`점 기록
- [x] SUB-001 대비 `+71.0475471351`점 개선과 CV–LB 괴리 해석
- [ ] 제출 실패 시 설치 오류와 추론 오류 구분

### 15-1. 1,000점 목표 관리

- [x] 현재 점수 `886.2488171351`, 목표 `1,000`, 차이 `113.7511828649` 문서화
- [x] 현재 환산 BSS `0.008862488171351`, 잔여 BSS 개선 목표 `+0.001137511828649` 계산
- [x] `07_1000점_달성계획.md` 작성
- [x] `DIAG-1000-001` 기준선 오차 지도 생성 및 R 과대예측 반복 확인
- [x] `CAL-SEL-OOF-001` 검증: global Platt Gate A PASS, 격리 후보 유지
- [x] `ENS-SEED-001` 2-seed 선행 게이트: 상관 `0.991725`·개선 `0.000020472`로 조기 중단
- [x] `FE-002` state/form/support 검증: 시즌 반복 개선 실패로 폐기
- [x] `BLEND-RECENT-001` 검증: global Platt 대비 추가 개선 `0.000002807`로 폐기
- [x] `REGIME-R-001` 검증: 2023·2024 평균 Brier 개선 `0.000394467`, Gate A PASS
- [x] 최고 후보의 2022·2023·2024 Gate A/B 판정
- [x] 최고 후보의 행 독립성·순열 불변성·245,789행 benchmark PASS
- [x] 독립 후보 ZIP `submit_regime_r_candidate.zip` 생성·내부 실행·예측 해시 일치 확인
- [x] `SUB-002` 제출 점수 `886.2488171351`과 REGIME-R 가설 연결 기록
- [x] Gemini 위임용 현황·규정·산출물·실험 우선순위 문서 작성
- [ ] 개정 최근 시즌 게이트를 통과하는 `SUB-003` 후보 확보
- [ ] 공식 리더보드 `1,000`점 이상 확인
- [x] 최종 모델 활성화 후 기존 ZIP을 재사용하지 않고 byte-identical 최종 ZIP 재생성
- [x] 최종 코드·모델·requirements·ZIP·activation·rollback 해시로 재현성 보존

## 16. Phase 3 대비

- [ ] 학습 코드 실행 절차 정리
- [ ] 학습 환경 OS·Python·패키지 버전 기록
- [ ] Private Score 재현용 학습 코드 정리
- [ ] 최종 추론 코드와 리더보드 제출물 보존
- [ ] 사용 데이터·피처·누수 방지 근거 정리
- [ ] 모델 선택·실패 실험·성능 개선 과정 정리
- [ ] 솔루션 PPT 초안 작성
- [ ] 팀원 오프라인 참가 여부 정리
- [ ] 코드 및 PPT 제출 규격 최종 확인
