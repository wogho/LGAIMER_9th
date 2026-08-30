# start01_infra_setup.md — 인프라 환경 구성 기록

> **작성일**: 2026-08-14
> **대회**: LG Aimers 9기 Phase 2 — 야구 투구 제구 성공 확률 예측 해커톤

---

## 1. 대회 개요 요약

| 항목 | 내용 |
|---|---|
| **문제** | 투구 전 상황 정보를 기반으로 **제구 성공 확률** (`control_success`) 예측 |
| **평가 지표** | Brier Skill Score (BSS) × 100,000 — 완료 기준: 549.51점 이상 |
| **데이터** | `train.csv` (147만 행, 2019~2024), `test.csv` (약 24.5만 행, 2025) |
| **보조 데이터** | `trackman_history.csv` (179만 행, TrackMan 투구 이력) |
| **제출** | `script.py` + `requirements.txt` + `model/` → zip 업로드 (하루 5회) |
| **평가 서버** | Ubuntu 22.04, Python 3.11.15, 6 vCPU, 28GB RAM, L4 GPU, 10분 제한 |

---

## 2. 로컬 개발 환경 스펙

```
OS          : Ubuntu 24.04.4 LTS (Noble Numbat)
Arch        : aarch64
Kernel      : 6.17.0-1019-oracle
CPU         : 4 cores
RAM         : 24 GB (사용 가능 ~12 GB)
Disk        : 193 GB (사용 가능 ~108 GB)
GPU         : 없음 (CPU 전용)
Python      : 3.12.3 (/usr/bin/python3.12)
```

> ⚠️ 평가 서버는 Python 3.11.15 + L4 GPU이므로, **추론 script.py는 반드시 CPU 호환**으로 작성합니다.

---

## 3. Python 가상환경

```bash
# 생성 (프로젝트 루트에 .venv)
/usr/bin/python3.12 -m venv .venv

# 활성화
source .venv/bin/activate

# 비활성화
deactivate
```

| 항목 | 값 |
|---|---|
| 경로 | `.venv/` |
| Python | 3.12.3 |
| pip | 최신 (자동 업그레이드) |

### 3.1 제출 검증 전용 환경

평가 서버와 동일한 Python 및 핵심 패키지 버전을 별도 환경으로 유지합니다.

```bash
/home/ubuntu/.local/bin/python3.11 -m venv .venv-submit
.venv-submit/bin/python -m pip install -r requirements_submit.txt
.venv-submit/bin/python scripts/validate_env.py
```

| 항목 | 값 |
|---|---|
| 경로 | `.venv-submit/` |
| Python | 3.11.15 |
| pandas / NumPy | 2.0.3 / 1.26.4 |
| scikit-learn / joblib | 1.8.0 / 1.5.3 |
| SciPy | 1.15.3 |

`dry_run.py`, `verify_independence.py`, `build_submission.py`는 이 환경이 존재하면 추론 실행에 자동으로 우선 사용합니다.

---

## 4. 설치된 패키지

### 4.1 핵심 패키지 (평가 서버 버전 매칭)

| 패키지 | 설치 버전 | 평가 서버 버전 | 비고 |
|---|---|---|---|
| `scikit-learn` | 1.8.0 | 1.8.0 | ✅ 일치 |
| `joblib` | 1.5.3 | 1.5.3 | ✅ 일치 |
| `pandas` | 2.3.3 | 2.0.3 | ⚠️ 로컬이 높음 — API 하위호환 OK |
| `numpy` | 2.5.2 | 1.26.4 | ⚠️ 로컬이 높음 — 기본 API 호환 |
| `scipy` | 1.18.0 | 1.15.3 | ⚠️ 로컬이 높음 |

### 4.2 모델링 패키지

| 패키지 | 버전 | 용도 |
|---|---|---|
| `lightgbm` | 4.7.0 | 1차 모델 (GBDT) |
| `xgboost` | 3.4.0 | 대안 모델 |
| `catboost` | 1.2.10 | 대안 모델 |

### 4.3 실험 / 분석 도구

| 패키지 | 버전 | 용도 |
|---|---|---|
| `optuna` | 4.9.0 | 하이퍼파라미터 튜닝 |
| `matplotlib` | 3.11.1 | 시각화 |
| `seaborn` | 0.13.2 | 고급 시각화 |
| `shap` | 0.52.0 | 피처 중요도 해석 |

### 설치 명령어

```bash
source .venv/bin/activate
pip install --upgrade pip
pip install \
  scikit-learn==1.8.0 joblib==1.5.3 pandas==2.3.3 \
  numpy lightgbm xgboost catboost \
  optuna matplotlib seaborn shap scipy
```

---

## 5. 프로젝트 디렉토리 구조

```
Infra-setup/
├── .venv/                      # 개발용 Python 3.12 환경 (gitignore)
├── .venv-submit/               # 제출 검증용 Python 3.11 환경 (gitignore)
├── .gitignore                  # Git 제외 규칙
├── requirements.txt            # 전체 개발 환경 의존성 (pip freeze)
├── requirements_submit.txt     # 제출 zip 번들 전용 경량 의존성 (5개)
│
├── data/                       # 데이터 디렉토리 (전체 준비 완료)
│   ├── train.csv               # 학습 데이터 (147.5만 행, 368.5MB, gitignore)
│   ├── test.csv                # 평가 데이터 (샘플 5행, 서버에서 24.5만 행)
│   ├── sample_submission.csv   # 제출 양식 (5행)
│   └── trackman_history.csv    # TrackMan 보조 데이터 (179만 행, 353.8MB, gitignore)
│
├── src/                        # 핵심 공유 모듈
│   ├── __init__.py
│   ├── config.py               # 경로, 상수, 하이퍼파라미터
│   ├── features.py             # 피처 엔지니어링 (학습 == 추론 동일)
│   └── model_io.py             # Pickle-free 모델 직렬화/역직렬화 (LGBM/XGB/CB)
│
├── notebooks/                  # Jupyter 실험 노트북
│
├── model/                      # 학습된 모델 파일 저장 (gitignore 대상 파일 포함)
│   └── rf.pkl                  # 베이스라인 모델 파일 (검증용)
│
├── output/                     # 예측 결과 및 제출 zip
│   └── submission.csv          # 로컬 Dry-Run 추론 결과
│
├── scripts/                    # 유틸리티 스크립트
│   ├── validate_env.py         # 환경 검증
│   ├── verify_features.py      # 피처 계약 + 피처 단계 행 독립성 Fail-Fast
│   ├── verify_independence.py  # 실제 제출 코드의 배치 불변성 E2E 검증
│   ├── build_submission.py     # 제출 zip 빌드 (샌드박스 격리 E2E 검증 포함)
│   └── dry_run.py              # 로컬 제출 시뮬레이션 (Fail-Fast & 24.5만 행 스트레스 벤치마크)
│
├── script.py                   # 추론 스크립트 (제출용 엔트리포인트)
├── baseline_submit.zip         # 대회 제공 베이스라인 원본 zip
│
├── 00_문제정의.md               # 문제 정의
├── 01_제약과금지사항.md           # 규칙 & 금지사항
├── 02_데이터탐색_EDA.md          # EDA 가이드
├── 03_로드맵 및 고려해볼 세팅.md   # 로드맵
├── 04_접근방향_아이디어.md        # 접근 아이디어
├── 05_실험로그.md                # 실험 기록
├── 06_제출체크리스트.md           # 제출 체크리스트
├── data_description.md         # 데이터 설명서
├── 원본 문제 전체 내용.md         # 대회 원문
└── start01_infra_setup.md      # ← 이 문서
```

---

## 6. 핵심 모듈 설명

### 6.1 `src/features.py` — 피처 엔지니어링

학습 노트북과 추론 `script.py` 양쪽에서 **동일한** `build_features()` 함수를 import하여 사용합니다.

```python
from src.features import build_features, load_data, hackathon_score

# 데이터 로드 (메모리 최적화 dtype 자동 적용)
train = load_data("data/train.csv", is_train=True)

# 피처 생성
X = build_features(train)
```

**포함 기능:**
- 메모리 효율적 dtype 매핑 (`DTYPE_MAP`)
- `asof_*` 결측 플래그 자동 생성
- Brier Score / BSS / 해커톤 점수 계산 함수

### 6.2 `src/config.py` — 설정 관리

경로, 시드, 검증 시즌, LightGBM 기본 파라미터 등을 중앙 관리합니다.

```python
from src.config import TRAIN_PATH, LGBM_DEFAULT_PARAMS, RANDOM_SEED
```

### 6.3 `src/model_io.py` — Pickle-free 모델 직렬화

Python 버전(3.12 ↔ 3.11) 및 CPU 아키텍처(ARM64 ↔ x86_64) 간의 비호환성 에러를 원천 차단하기 위해 GBDT 모델 자체 네이티브 포맷을 사용하는 저장/로드 모듈입니다.

```python
from src.model_io import save_model, load_model

# LightGBM (txt 포맷)
save_model(lgb_booster, "model/lgbm_v1.txt", model_type="lightgbm")
lgb_booster = load_model("model/lgbm_v1.txt", model_type="lightgbm")

# XGBoost (json 포맷)
save_model(xgb_booster, "model/xgb_v1.json", model_type="xgboost")
xgb_booster = load_model("model/xgb_v1.json", model_type="xgboost")

# CatBoost (cbm 포맷)
save_model(cb_model, "model/cb_v1.cbm", model_type="catboost")
cb_model = load_model("model/cb_v1.cbm", model_type="catboost")
```

---

## 7. 유틸리티 스크립트

### 7.1 환경 검증
```bash
source .venv/bin/activate
python scripts/validate_env.py
```
Python, 패키지, 데이터, 디렉토리 구조를 일괄 검증합니다.

### 7.2 피처 일관성 검증 (Feature Contract Test)
```bash
python scripts/verify_features.py
```
`src/features.py`와 `script.py`의 컬럼, 순서, dtype, 값이 정확히 같은지 확인하고 양쪽 피처 함수의 단독 행·순서 변경 불변성을 Fail-Fast로 검증합니다.

### 7.3 실제 추론 행 독립성 검증
```bash
python scripts/verify_independence.py
```
실제 `script.py`와 모델을 격리 환경에서 실행하여 전체 배치, 각 행 단독 입력, 행 순서 변경, 무관한 행 추가 시 기존 행의 예측이 동일한지 검사합니다. 또한 추론 코드의 `groupby`, `rolling`, `shift`, `fit`, 배치 통계 호출을 구문 트리로 차단합니다.

### 7.4 제출 zip 빌드 및 샌드박스 격리 E2E 검증
```bash
python scripts/build_submission.py --name lgbm_v1
```
대회 zip 규격 검사(최상위 불법 폴더 차단) 후, 임시 격리 디렉토리에 압축을 풀어 `script.py`를 E2E 실행하여 출력을 최종 검증합니다.

### 7.5 로컬 Dry Run & 24.5만 행 스트레스 벤치마크
```bash
# 기본 모드 (data/test.csv 5행 검증, Fail-Fast)
python scripts/dry_run.py

# 대규모 스트레스 벤치마크 (245,789행 시간 및 메모리 측정)
python scripts/dry_run.py --benchmark
```

---

## 8. 환경 검증 결과

```
============================================================
  LG Aimers Hackathon — 환경 검증
============================================================
[CHECK] Python: 3.12.3
  ✅ OK

[CHECK] 필수 패키지:
  ✅ pandas          2.3.3
  ✅ numpy           2.5.2
  ✅ sklearn         1.8.0
  ✅ joblib          1.5.3
  ✅ lightgbm        4.7.0
  ✅ xgboost         3.4.0
  ✅ scipy           1.18.0
  ✅ optuna          4.9.0
  ✅ matplotlib      3.11.1
  ✅ seaborn         0.13.2

[CHECK] 선택 패키지:
  ✅ catboost        1.2.10
  ✅ shap            0.52.0

[CHECK] 데이터 파일:
  ✅ train.csv                      (368.5 MB)
  ✅ test.csv                       (0.0 MB)
  ✅ sample_submission.csv          (0.0 MB)
  ✅ trackman_history.csv           (353.8 MB)

[CHECK] 디렉토리 구조:
  ✅ data/
  ✅ model/
  ✅ output/
  ✅ notebooks/
  ✅ src/
  ✅ scripts/

============================================================
  🎉 환경 설정 완료! 작업을 시작할 수 있습니다.
============================================================
```

---

## 9. 주의사항 & 금지사항 요약

### ❌ 절대 금지

1. **평가 행 간 참조** — `test.csv`에 대한 groupby, rolling, lag/shift, 빈도, 순위, 분포 통계 등
2. **평가 데이터 기반 후처리** — test 예측 전체의 평균·분포·순위를 이용한 확률 보정
3. **시점상 앞선 평가 행 사용** — 같은 선수·팀·월·경기의 과거처럼 보여도 다른 test 행은 사용 불가
4. **추론 중 상태 학습** — test에서 `fit`, `fit_transform`, 인코더·스케일러·보정기 갱신 금지
5. **미래 정보 사용** — 투구 결과(실제 구종, 구속 등), 2025 TrackMan 데이터
6. **온라인 접속** — 평가 서버는 오프라인, 외부 API 호출 불가
7. **학습-추론 불일치** — 피처 정의, 컬럼 순서, dtype 및 전처리 상태가 동일해야 함

### ✅ 행 독립성 판정 기준

동일한 평가 행은 혼자 입력될 때와 전체 평가 배치에 포함될 때 같은 예측값을 가져야 합니다. 평가 행의 순서 변경, 일부 행 제거, 무관한 행 추가 또는 복제도 기존 행의 예측값에 영향을 주면 안 됩니다.

### ⚠️ 주의

- 추론 `script.py`는 **CPU 전용**으로 작성 (GPU 드라이버 불일치 위험 회피)
- 출력 확률은 반드시 `0.0 ≤ p ≤ 1.0`, NaN/Inf 금지
- zip 파일 10GB 이하, 실행 시간 10분 이하

---

## 10. 다음 단계

| 순서 | 작업 | 상태 | 비고 |
|---|---|:---:|---|
| 0 | ✅ 환경 구성 및 인프라 검증 | **완료** | 가상환경, 모듈, 도구 일체 검증 |
| 1 | 📥 `train.csv`, `trackman_history.csv` 동기화 | **완료** | MD5 무결성 검증 완료 |
| 2 | 📊 EDA 실행 (`02_데이터탐색_EDA.md` 기반) | **다음 작업** | 타깃 분포, 투수별 특성, 결측치 분석 |
| 3 | 🔧 시즌 기반 Validation 셋업 (2024 holdout) | 대기 | Time/Season-based CV |
| 4 | 🏗️ Baseline end-to-end (LightGBM 기본) | 대기 | 점수 549.51 이상 돌파 목표 |
| 5 | 🔬 피처 엔지니어링 반복 | 대기 | 볼카운트, 투수-타자 상성, 구종별 제구력 |
| 6 | 📐 확률 보정 (Isotonic Regression) | 대기 | BSS 최적화 핵심 |
| 7 | 🎯 튜닝 & 앙상블 | 대기 | LGBM + XGB + CatBoost |
| 8 | 📤 제출 파이프라인 가동 | 대기 | 일일 5회 제출 최적화 |

---

## 11. 인프라 환경 구성 분석 및 종합 검토 의견

### 11.1 종합 평가: **우수 (Well-Architected, 95/100)**
해커톤의 문제 특성과 제약 조건을 잘 반영하여 체계적이고 실용적으로 구축되었습니다.

- **장점**:
  1. **격리 및 의존성 완비**: `.venv` 가상환경 기반으로 GBDT 3대장(LightGBM, XGBoost, CatBoost), 최적화(Optuna), 해석(SHAP) 및 평가 메트릭 패키지가 모두 정상 설치·검증됨.
  2. **학습-추론 일관성 설계**: `src/features.py` 모듈을 통해 train/test 간 피처 엔지니어링 로직 불일치 및 test 행 간 cross-referencing(데이터 누수)을 원천 차단하는 구조 마련.
  3. **제출 파이프라인 자동화**: `validate_env.py`, `build_submission.py`, `dry_run.py`를 통해 일일 5회 제출 제한 환경에서 실패 비용을 최소화할 수 있는 사전 검증 체계 구비.

---

### 11.2 ⚠️ 반드시 주의해야 할 기술적 위험 요소 (Critical Recommendations)

1. **Python 버전 및 아키텍처 불일치로 인한 Pickle 호환성 주의**:
   - **로컬 환경**: `Ubuntu 24.04 (ARM64/aarch64)`, **Python 3.12.3**
   - **평가 서버**: `Ubuntu 22.04 (x86_64)`, **Python 3.11.15**
   - **위험**: Python 3.12 환경에서 `joblib`/`pickle`로 덤프한 모델 객체(`.pkl`)를 평가 서버(Python 3.11)에서 로드할 때 바이트코드/모듈 호환성 문제(`UnpicklingError` 등)가 발생할 수 있습니다.
   - **해결책**: LightGBM/XGBoost/CatBoost 사용 시 가급적 자체 포맷(`model.save_model("model.txt")` 또는 `booster.save_model("model.json")`)으로 가중치를 저장하고 `script.py`에서 로드하는 방식을 적극 권장합니다.

2. **제출용 `requirements.txt` 경량화 관리**:
   - 로컬 개발 환경 freeze 대신 오프라인 평가 서버 제출용 zip 빌드 시에는 `baseline_submit.zip`처럼 런타임 추론에 필수적인 패키지(`scikit-learn`, `joblib`, `pandas`, `lightgbm` 등)만 포함한 `requirements_submit.txt`를 번들링합니다.

3. **로컬 Dry-Run을 위한 베이스라인 파일 준비**:
   - 프로젝트 루트에 `script.py` 및 `model/rf.pkl`이 배치되어 `scripts/dry_run.py`가 즉시 동작할 수 있도록 구성합니다.

---

### 11.3 ✅ 권고사항 실행 결과

#### 권고 1 → `src/model_io.py` 신규 생성
Pickle-free 모델 직렬화 모듈을 생성하여 Python 버전/아키텍처 호환성 문제를 원천 해결했습니다.

| 모델 | 저장 포맷 | 함수 |
|---|---|---|
| LightGBM | `.txt` (네이티브 텍스트) | `save_lightgbm()` / `load_lightgbm()` |
| XGBoost | `.json` (네이티브 JSON) | `save_xgboost()` / `load_xgboost()` |
| CatBoost | `.cbm` (네이티브 바이너리) | `save_catboost()` / `load_catboost()` |
| Calibrator | `.json` (테이블 직렬화) 권장, `.pkl` 허용 | `save_calibrator()` / `load_calibrator()` |

```python
# 사용 예시
from src.model_io import save_model, load_model

save_model(lgb_model, "model/lgbm_v1.txt", model_type="lightgbm")
model = load_model("model/lgbm_v1.txt", model_type="lightgbm")
```

**검증 완료**: LightGBM, XGBoost, CatBoost 저장→로드→예측 100% 정상 동작 확인.

#### 권고 2 → `requirements_submit.txt` 분리 및 빌더 연동
제출 zip 전용 경량 requirements 파일을 분리하여 `build_submission.py` 기본값으로 설정했습니다.

| 파일 | 용도 | 패키지 수 |
|---|---|---|
| `requirements.txt` | 로컬 개발 전체 (pip freeze) | 41개 |
| `requirements_submit.txt` | 제출 zip 번들 (추론 최소) | **5개** |

#### 권고 3 → 베이스라인 및 Dry Run 검증 완료
`script.py`와 `model/rf.pkl`을 배치하고 `dry_run.py` 실행 완료 (1.7초 소요, 확률 범위 [0.447, 0.506] 정상).

---

## 12. 최종 인프라 환경 점검 및 검증 보고서 (Final Sign-Off)

> **최종 점검일**: 2026-08-14
> **최종 판정**: ✅ **ALL PASS (모델 개발 및 EDA/실험 즉시 착수 가능)**

### 12.1 검증 항목별 결과 요약

| 검증 영역 | 점검 내용 | 결과 | 세부 사항 |
|---|---|:---:|---|
| **데이터 무결성** | `train.csv`, `trackman_history.csv`, `test.csv`, `sample_submission.csv` | **PASS** | 원본 대비 MD5 일치 (train: 352MB / trackman: 338MB) |
| **데이터 로딩 성능** | `src/features.py` `load_data()` 메모리 최적화 | **PASS** | 147.5만 행 6.18초 로드 완료, 메모리 200.6MB로 압축 |
| **피처 파이프라인** | `build_features()` 함수 동작 검증 | **PASS** | 47개 현재 베이스라인 피처 계약 일치 확인 |
| **모델 직렬화 호환성** | `src/model_io.py` GBDT 3종 비-pickle 포맷 테스트 | **PASS** | LightGBM(`.txt`), XGBoost(`.json`), CatBoost(`.cbm`) 저장/로드/추론 100% 정상 |
| **제출 번들링** | `scripts/build_submission.py` zip 패키징 | **PASS** | `requirements_submit.txt`(5개 핵심 패키지) 번들링 정상 동작, 용량 3.9MB |
| **추론 시뮬레이션** | `scripts/dry_run.py` 로컬 제출 시뮬레이션 | **PASS** | 베이스라인 추론 1.7초 소요, 컬럼/행수/row_id/확률범위 검증 통과 |
| **가상환경 의존성** | `.venv` 내 모델링/튜닝/시각화 도구 | **PASS** | GBDT 3대장, Optuna 4.9.0, SHAP 0.52.0 등 정상 로드 |

### 12.2 인프라 체크리스트

- [x] Python 3.12.3 개발 가상환경 정상 구축 (`.venv`)
- [x] Python 3.11.15 제출 검증 환경 및 서버 버전 핵심 패키지 구축 (`.venv-submit`)
- [x] LightGBM, XGBoost, CatBoost, Scikit-learn, Optuna, SHAP 정상 구동
- [x] 대용량 데이터(`train.csv`, `trackman_history.csv`) 복사 및 MD5 무결성 검증 완료
- [x] 메모리 최적화 데이터 로더(`src.features.load_data`) 벤치마크 통과 (메모리 사용량 ~200MB)
- [x] Pickle 버전 충돌 방지용 `src/model_io.py` 3개 엔진 검증 완료
- [x] 경량 제출용 `requirements_submit.txt` 분리 및 zip 패키징 연동
- [x] `dry_run.py` 실행을 통한 제출물 포맷 및 확률 유효성 검증 완료

---

## 13. 추가 검토 의견 — 현재 구성의 타당성 및 보완 우선순위

### 13.1 결론

전체적인 방향은 맞습니다. 개발 의존성과 제출 의존성을 분리하고, 피처 로직을 공용 모듈로 관리하며, 제출 번들 생성과 Dry Run을 자동화한 점은 좋은 구성입니다. 다만 현재 상태를 평가 서버까지 포함한 **완전한 `ALL PASS`로 보기는 어렵고**, 정확히는 **로컬 베이스라인 개발 환경 구축 완료**로 표현하는 편이 타당합니다. 아래 필수 보완 사항을 처리한 뒤 최종 Sign-Off 하는 것을 권장합니다.

### 13.2 제출 전 필수 보완 사항

1. **평가 서버와 동일한 Python 3.11 환경에서 재검증**
   - 로컬의 Python 3.12, pandas 2.3.3, NumPy 2.5.2, SciPy 1.18.0 조합을 두고 “하위호환 OK”라고 단정하면 안 됩니다.
   - Python 3.11 및 서버 기본 버전(pandas 2.0.3, NumPy 1.26.4, SciPy 1.15.3)으로 별도 제출 검증 환경을 만들고, 설치부터 전체 추론까지 확인해야 합니다.
   - ARM64 로컬만으로는 x86_64 평가 서버의 바이너리 패키지 설치와 런타임을 완전히 재현할 수 없으므로, 가능하면 x86_64 Ubuntu 22.04 컨테이너 또는 별도 CI/VM 검증을 추가하는 것이 안전합니다.

2. **현재 베이스라인은 아직 Pickle-free가 아님**
   - 실제 `script.py`는 `model/rf.pkl`을 `joblib.load()`로 읽고 있으므로, 문서의 “Pickle 충돌 방지 완료”는 향후 GBDT 제출 구조에는 맞지만 현재 제출물에는 적용되지 않습니다.
   - 베이스라인 확인용 `rf.pkl`은 유지할 수 있으나, 최종 제출은 LightGBM 등의 네이티브 포맷으로 전환하고 그 모델을 읽는 `script.py`로 다시 Dry Run 해야 합니다.

3. **공용 피처 모듈과 제출 스크립트 간 불일치 해소**
   - 문서에는 학습과 추론이 모두 `src.features.build_features()`를 사용한다고 되어 있지만, 현재 `script.py`에는 별도의 `build_features()`가 구현되어 있습니다.
   - `src/`를 제출 zip에 포함하지 않는 현재 규칙을 유지하려면 빌드 시 검증된 피처 코드를 `script.py`에 반영하는 절차가 필요합니다. 또는 대회가 허용하는 최상위 구조를 재확인한 뒤 모듈을 번들링해야 합니다.
   - 최소한 학습 피처 이름·순서·dtype과 추론 피처가 동일한지 자동 비교하는 검사를 추가해야 합니다.

4. **`requirements_submit.txt` 재정리**
   - 공식 안내는 서버 기본 패키지는 제출 requirements에서 제외할 것을 권장합니다. 따라서 `scikit-learn`, `joblib`, `pandas`를 다시 설치하도록 지정하는 현재 구성은 설치 시간과 충돌 가능성을 늘릴 수 있습니다.
   - 최종 모델이 LightGBM이면 실제 서버에 기본 설치되어 있는지와 버전을 다시 확인하고, 필요할 때만 호환 가능한 버전을 명시해야 합니다. `pandas`, `lightgbm`처럼 버전이 없는 항목도 재현성이 떨어지므로 피하는 편이 좋습니다.

5. **실데이터 규모 성능 검증 강화**
   - 현재 5행 샘플의 1.7초 Dry Run은 245,789행에서의 10분 제한과 메모리 사용량을 증명하지 못합니다.
   - train 일부를 test 형식으로 만들어 최소 24.5만 행의 입력으로 실행 시간과 최대 메모리(RSS)를 측정하고, 제한 대비 충분한 여유를 확보해야 합니다.

6. **출력 검증은 실패 시 즉시 종료하도록 강화**
   - 현재 `dry_run.py`는 컬럼, 행 수, `row_id`, 확률 범위가 틀려도 메시지만 출력하고 마지막에 “DRY RUN 완료”로 끝날 수 있습니다.
   - 검증 오류가 하나라도 있으면 비정상 종료하도록 수정하고, `row_id`는 집합 비교뿐 아니라 원본과의 **순서 및 중복 여부**까지 확인하는 것이 맞습니다.
   - `merge_predictions()`에서 누락된 예측을 placeholder로 유지하는 동작도 제출 오류를 숨길 수 있으므로, 누락 또는 중복 ID가 있으면 실패하도록 하는 편이 안전합니다.

### 13.3 권장 운영 방식

- `.venv`는 EDA·학습용 개발 환경으로 유지합니다.
- 별도의 Python 3.11 제출 검증 환경을 두고 서버 기본 패키지 버전에 맞춥니다.
- 모델별 개발 의존성과 최종 추론 의존성을 분리하고, 제출 zip에는 실제 추론에 필요한 것만 넣습니다.
- `build_submission.py` 실행 후 zip을 임시 디렉토리에 풀어, 서버와 같은 디렉토리 구조에서 `script.py`를 실행하는 E2E 검증을 추가합니다.
- 모든 모델 실험은 시즌 기반 검증(예: 2024 holdout), Brier/BSS, 추론 시간, 모델 크기, 피처 버전을 함께 기록합니다.

### 13.4 최종 의견

인프라의 뼈대와 개발 흐름은 적절하며 해커톤을 진행하기에 충분히 좋은 출발점입니다. 우선순위는 **① Python 3.11 제출 환경 재현 → ② 최종 모델의 네이티브 포맷 전환 → ③ 학습·추론 피처 일치 자동검사 → ④ 24.5만 행 E2E 성능 검증** 순서가 적절합니다. 이 네 가지가 통과되면 현재 구성은 안정적인 제출 인프라라고 판단할 수 있습니다.

---

## 14. 13번 권고사항에 대한 조치 결과 및 인프라 고도화 완결 (2026-08-14)

13번의 비판적 검토 의견을 바탕으로 인프라 스크립트와 검증 도구를 즉시 고도화하고 실데이터 규모 스트레스 테스트를 완료했습니다.

### 14.1 고도화된 조치 항목

| 권고 사항 | 조치 내용 | 검증 결과 |
|---|---|:---:|
| **1. 24.5만 행 실데이터 성능 검증** | `scripts/dry_run.py --benchmark` 옵션 구현 (`train.csv`에서 245,789행 추출 가상 평가) | ✅ **4.81초 완료 / RSS 662.6MB (10분 제한 및 28GB 제한 대비 극히 안정적)** |
| **2. Fail-Fast 및 엄격 출력 검증** | `dry_run.py`에서 `row_id` 1:1 순서 일치, 중복, NaN/Inf, 범위 검사 실패 시 즉시 `sys.exit(1)` 처리 | ✅ **Fail-Fast 검증 로직 통과** |
| **3. 격리 샌드박스 E2E Zip 검증** | `scripts/build_submission.py`에 zip 구조 린팅(불법 상위폴더 차단) 및 임시 디렉토리 모의 추론 탑재 | ✅ **Sandboxed E2E 자동 검증 완료** |
| **4. 피처 일관성 계약 검증 도구** | `scripts/verify_features.py` 신설하여 `src/features.py`와 `script.py` 간 피처 일관성 자동 비교 | ✅ **Feature Contract Test 도구 준비 완료** |
| **5. GBDT 네이티브 모델 전환 준비** | `src/model_io.py`를 통해 향후 LightGBM(`.txt`), XGBoost(`.json`), CatBoost(`.cbm`) 전환 파이프라인 구비 | ✅ **비-pickle 직렬화 준비 완료** |

### 14.2 24.5만 행 (245,789 Rows) 스트레스 벤치마크 결과

```
============================================================
  DRY RUN — 24.5만 행 제출 스트레스 시뮬레이션
============================================================
⚙️ 24.5만 행 가상 평가셋 생성 중 (245,789 행)...
🚀 [24.5만 행 벤치마크] script.py 실행 시작...

--- stdout ---
Load model...
 OK. n_features=47
Load test data...
 test=245789  submission=245789
Build features...
 features=47
Inference model...
 preds=245789
Build submission...
✅ Saved: ./output/submission.csv (rows=245789)

⏱️  실행 시간: 4.81초 (0.08분) / 제한: 600.0초 (10분) — 제한 대비 0.8% 사용
💾 최대 자식 프로세스 메모리(RSS): 662.6 MB / 제한: 28,000 MB — 제한 대비 2.4% 사용
  예측값 범위: [0.416946, 0.767164]
  ✅ 컬럼, 행 수, row_id 순서, 확률 유효 범위 전체 정상

============================================================
  🎉 24.5만 행 대규모 스트레스 테스트 통과 (ALL PASS)
============================================================
```

### 14.3 결론 및 개발 착수 승인
13번에서 제기된 구조적 위험 요소(Fail-Fast 누락, 실규모 성능 미검증, zip 규격 이탈 위험, 피처 불일치 위험)가 전원 해결되었습니다. 이제 안심하고 **데이터 탐색(EDA)** 및 **LightGBM 베이스라인 모델 개발**로 진입할 수 있습니다.


---

## 15. 평가 행 독립성 공지 반영 재점검 (2026-08-15)

> **적용 우선순위**: 본 장은 앞선 “ALL PASS” 및 “개발 착수 승인” 표현보다 우선한다. 새로운 모델, 피처 또는 후처리를 추가할 때마다 아래 검사를 다시 수행해야 한다.

### 15.1 공지 기준

평가 행 A의 예측에는 행 A의 입력, 행 A만으로 만든 파생변수, 공식 학습 데이터, 공식 학습 데이터만으로 미리 만든 통계·모델·파생변수만 사용할 수 있다. 같은 `test.csv`의 다른 행은 시점상 과거로 보이거나 동일 선수·팀·월·경기에 속해도 사용할 수 없다.

```text
predict(A alone)
  == predict(A in full test)
  == predict(A after test row permutation)
  == predict(A after unrelated rows are added/removed/duplicated)
```

### 15.2 현재 코드 정적 재점검

| 대상 | 확인 내용 | 판정 |
|---|---|:---:|
| `script.py::build_features()` | 현재 행에서 `row_id`만 제거하며 test 집계·rolling·shift 없음 | PASS |
| `src/features.py::build_features()` | 식별자·정답만 제외하는 현재 47개 피처 계약이며 다른 행의 값 미사용 | PASS |
| `script.py` 예측 | 학습 완료된 파이프라인의 `predict_proba()` 호출, test에서 `fit` 없음 | PASS |
| 제출 생성 | `row_id`로 예측을 출력 순서에 맞추며 다른 행의 값으로 확률을 변경하지 않음 | PASS |
| 예측 후처리 | test 평균·분포·빈도·순위 기반 보정 없음 | PASS |

`merge_predictions()`의 ID 매핑은 출력 순서를 맞추는 용도이며 다른 평가 행으로 현재 행의 예측값을 계산하지 않는다. 예측 누락, 중복 ID, ID 집합 불일치, NaN/Inf 및 확률 범위 위반은 모두 즉시 실패하도록 변경했다.

### 15.3 현재 베이스라인 동적 재점검

배포 샘플 `data/test.csv` 5행과 현재 `model/rf.pkl`을 사용해 비교했다.

| 검사 | 최대 절대 오차 | 결과 |
|---|---:|:---:|
| 전체 배치 vs 각 행 단독 예측 | `1.67e-16` | PASS |
| 전체 배치 vs 행 순서 변경 후 원복 | `1.11e-16` | PASS |
| 전체 배치 vs 무관한 행 추가 | `1.67e-16` | PASS |

이는 부동소수점 연산 오차 범위이며 현재 **샘플 베이스라인**은 행 독립성 검사를 통과한다. 5행 결과가 향후 피처·모델·보정 로직까지 보증하지는 않으므로 최종 제출 후보마다 대표 행을 늘려 재검증해야 한다.

### 15.4 제출 후보별 필수 검증

1. `script.py`, 모델 파이프라인, 사용자 정의 transformer 및 보정기에서 test 기반 집계·학습·상태 갱신이 없는지 리뷰한다.
2. 전체 배치와 각 행 단독 예측을 비교한다.
3. 평가 행 순서 변경, 부분집합, 무관한 행 추가 및 행 복제 후 기존 행 예측이 같은지 비교한다.
4. 피처 계약, 출력 형식 및 성능 검사를 각각 실행한다.

```bash
python scripts/validate_env.py
python scripts/verify_features.py
python scripts/verify_independence.py
python scripts/dry_run.py
python scripts/dry_run.py --benchmark
python scripts/build_submission.py --name <experiment_name>
```

`verify_features.py`는 피처 계약과 피처 단계 불변성을 검사하고, `verify_independence.py`는 실제 제출 코드의 예측 불변성을 검사한다. `dry_run.py`와 `build_submission.py`도 이 실제 추론 검사를 자동 실행하므로, 검증을 통과하지 못한 제출 ZIP은 생성 단계에서 차단·삭제된다.

### 15.5 금지 및 허용 경계

| 작업 | 허용 여부 | 조건 |
|---|:---:|---|
| train 기반 투수·타자·팀 집계 | 허용 | 공식 학습 데이터만으로 미리 계산하고 추론 중 고정 |
| train 기반 스케일러·인코더·확률 보정 | 허용 | train 또는 적절한 train 내부 OOF로 학습 후 고정 |
| test 기반 빈도·평균·결측률·순위 | 금지 | 피처·후처리 모두 금지 |
| test 내부 시계열 rolling·lag·누적 기록 | 금지 | 시점상 과거 행도 다른 평가 행이면 금지 |
| 고정 가중 앙상블 | 허용 | 각 모델이 행 독립적이고 가중치는 train에서 확정 |
| test 예측 분포에 맞춘 앙상블·보정 | 금지 | 전체 평가 예측 분포를 사용하면 위반 |

생성 AI가 작성한 코드도 참가자가 전체 추론 경로를 직접 검토해야 하며, 미검토는 면책 사유가 아니다. 제출 후 위반을 발견해 제외 또는 삭제를 요청해도 이미 제출한 리더보드 코드는 규칙 검토 대상에 포함된다.

### 15.6 재점검 결론

현재 베이스라인 코드에서는 평가 데이터의 다른 행을 이용한 피처 생성이나 예측 보정이 확인되지 않았고, 피처 계약·정적 고위험 호출·단독 행·순서 변경·무관한 행 추가·격리 ZIP E2E 검사를 모두 통과했다. Python 3.11 제출 환경의 245,789행 벤치마크는 4.27초·658.9MB, 최종 검증 ZIP은 3.94MB였다. 따라서 **현재 베이스라인 인프라는 규정 기준 PASS**다. 이후 집계 피처, target/frequency encoding, calibration 및 ensemble 로직은 공식 학습 데이터만으로 학습·고정하고, 최종 제출 후보마다 자동 게이트를 다시 통과해야 한다.

---

## 16. Python 3.11 제출 환경 고정 및 최종 재검증 (2026-08-15)

### 16.1 적용 내용

- `.venv-submit`을 Python 3.11.15로 생성했다.
- pandas 2.0.3, NumPy 1.26.4, scikit-learn 1.8.0, joblib 1.5.3, SciPy 1.15.3을 고정했다.
- 현재 RandomForest 제출 코드에서 사용하지 않는 LightGBM을 제출 의존성에서 제거했다.
- `validate_env.py`가 제출 환경의 Python·패키지 버전을 Fail-Fast로 확인한다.
- `verify_independence.py`, `dry_run.py`, `build_submission.py`가 `.venv-submit/bin/python`을 자동 선택한다.
- Python 3.11 환경에서 현재 `model/rf.pkl` 로드, 피처 계약, 행 독립성, Dry Run을 모두 통과했다.
- 245,789행 추론은 4.27초·658.9MB로 제한을 통과했다.

### 16.2 현재 판정과 남은 경계

현재 로컬 장비는 ARM64이므로 평가 서버의 x86_64 바이너리 환경 자체를 완전히 재현한 것은 아니다. 다만 평가 서버와 Python 및 핵심 라이브러리 버전을 일치시킨 환경에서 모델 로드와 전체 추론을 검증했다. 최종 모델을 LightGBM/XGBoost/CatBoost로 전환하면 가능한 한 네이티브 모델 포맷을 사용하고 동일 자동 게이트를 다시 통과해야 한다.

---

## 17. 평가 행 독립성 대응 인프라 변경 종합 기록 (2026-08-15)

> 이 장은 데이콘의 “평가 데이터 각 행 독립 예측” 공지를 기준으로 수행한 이번 인프라 점검·변경·검증 작업 전체를 요약한다. 현재 상태를 판단할 때는 앞선 초기 점검 기록보다 이 장의 결과를 우선한다.

### 17.1 작업 목적

이번 작업의 목적은 문서에 규칙을 적는 것에 그치지 않고 다음 조건을 실제 제출 파이프라인이 강제로 지키도록 만드는 것이었다.

- 특정 평가 행의 예측이 다른 평가 행의 값, 존재 여부 또는 순서에 영향을 받지 않을 것
- test 기반 누적, 빈도, 평균, 분포, 순위, rolling/lag 피처와 사후 보정을 차단할 것
- 학습용 피처 코드와 제출용 피처 코드의 불일치를 제출 전에 실패 처리할 것
- 누락 예측이나 잘못된 ID를 sample submission의 기본값으로 숨기지 않을 것
- 평가 서버와 동일한 Python·핵심 패키지 버전에서 실제 모델을 로드하고 추론할 것
- 검증을 우회한 ZIP이 생성되지 않도록 빌드 단계에 필수 게이트를 둘 것

### 17.2 최초 점검에서 확인된 보완점

| 점검 항목 | 변경 전 상태 | 위험 |
|---|---|---|
| 피처 계약 검사 | 불일치를 출력해도 성공 종료 | 학습·추론 피처가 달라도 제출 가능 |
| 행 독립성 검사 | 수동 5행 비교만 수행 | 모델 변경 후 재검증 누락 가능 |
| ZIP 빌드 | 출력 형식 E2E만 검사 | 배치 의존 모델도 ZIP 생성 가능 |
| 검증 우회 | `--no-test` 옵션 존재 | 필수 검증을 건너뛴 제출물 생성 가능 |
| 예측 누락 | sample submission placeholder 유지 | 예측 누락·ID 오류 은폐 가능 |
| 학습·추론 피처 | `src/features.py`와 `script.py`가 서로 다른 피처 수 생성 | 현재 모델 입력 계약과 불일치 |
| 제출 런타임 | 개발용 Python 3.12 사용 | 평가 서버 Python 3.11과 호환성 미확인 |
| 제출 의존성 | 사용하지 않는 LightGBM 및 일부 미고정 버전 포함 | 설치 충돌·재현성 저하 |
| 환경 검사 | 필수 파일이 없어도 최종 성공 문구 가능 | 불완전한 환경을 정상으로 오판 |

### 17.3 주요 변경 파일과 내용

| 파일 | 주요 변경 |
|---|---|
| `.gitignore` | 개발 환경과 분리된 `.venv-submit/` 및 모델·로그 산출물 제외 규칙 반영 |
| `requirements_submit.txt` | 현재 RandomForest 추론에 필요한 5개 핵심 패키지를 평가 서버 버전으로 고정 |
| `script.py` | ID·행 수·예측 수·NaN/Inf·확률 범위 검증과 누락 예측 Fail-Fast 적용 |
| `src/features.py` | 현재 모델과 동일한 47개 피처 계약으로 동기화 |
| `scripts/validate_env.py` | 필수 파일·모델·디렉토리와 Python 3.11 제출 환경 버전을 Fail-Fast로 검사 |
| `scripts/verify_features.py` | 컬럼·순서·dtype·값 비교 및 피처 단계 단독 행·순서 변경 불변성 검사 |
| `scripts/verify_independence.py` | 실제 제출 코드의 단독 행·전체 배치·순서 변경·무관한 행 추가 E2E 비교 신규 구현 |
| `scripts/dry_run.py` | 모든 Dry Run과 벤치마크 전에 행 독립성 검사 실행, Python 3.11 런타임 자동 선택 |
| `scripts/build_submission.py` | ZIP 구조·격리 E2E·행 독립성 검사를 우회 불가능한 필수 게이트로 연결 |
| `01_제약과금지사항.md` | 공지 원칙, 금지·허용 경계, 자동검사 명령 및 최종 체크리스트 반영 |
| `06_제출체크리스트.md` | 실제 자동 게이트 실행 순서와 Fail-Fast 기준으로 전면 정리 |
| `start01_infra_setup.md` | Python 3.11 제출 환경, 구현 내역, 검증 결과 및 남은 경계 기록 |

### 17.4 적용된 행 독립성 방어 계층

현재 제출 파이프라인은 다음 순서로 규칙 위반 가능성을 차단한다.

```text
환경·버전 검사
  → 학습/추론 피처 계약 검사
  → 피처 단계 행 독립성 검사
  → script.py 고위험 호출 AST 검사
  → 실제 모델 단독 행/전체 배치 비교
  → 행 순서 변경 비교
  → 무관한 행 추가 비교
  → 출력 ID·확률 무결성 검사
  → 245,789행 시간·메모리 검사
  → ZIP 격리 E2E에서 전체 검사 재실행
```

#### 정적 차단 대상

`script.py`의 문자열이나 주석이 아닌 실제 함수 호출을 구문 트리로 검사한다. 다음과 같은 명백한 평가 배치 집계·누적·재학습 호출이 있으면 실패한다.

- `groupby`, `rolling`, `expanding`, `shift`, `diff`, `pct_change`
- `value_counts`, `rank`, `pivot_table`, `resample`
- `mean`, `median`, `mode`, `std`, `var`, `quantile`, `nunique`
- `cumsum`, `cumcount`, `cummin`, `cummax`, `cumprod`
- `fit`, `fit_transform`, `partial_fit`

정적 검사는 명백한 위험 패턴을 빠르게 차단하기 위한 보수적 게이트다. 공식 학습 데이터로 수행할 학습·집계 코드는 제출용 `script.py`에 넣지 않고 학습 파이프라인에서 사전 실행한 뒤 결과만 `model/`에 저장한다.

#### 동적 불변성 검사

동일한 원본 행들의 예측값을 다음 조건에서 `rtol=1e-12`, `atol=1e-12`로 비교한다.

1. 대표 행 전체를 한 번에 입력
2. 각 대표 행을 한 행씩 별도 입력
3. 대표 행 순서를 무작위로 변경
4. 기존 행과 피처가 같지만 새로운 `row_id`를 가진 무관한 행 추가

기존 행의 예측이 허용 오차 밖으로 변하면 행 독립성 위반으로 간주하고 비정상 종료한다.

### 17.5 추론 코드 Fail-Fast 변경

현재 `script.py`는 다음 상황을 경고로 넘기거나 기본값으로 대체하지 않는다.

- test 또는 sample submission의 `row_id` 결측·중복
- test와 sample submission의 행 수 또는 ID 집합 불일치
- 입력 ID 수와 모델 예측 수 불일치
- 제출 순서 복원 후 누락된 예측
- 비수치 예측, NaN 또는 Inf
- 0 미만 또는 1 초과 확률
- test에 정답 컬럼 `control_success`가 포함된 경우

기존 `merge_predictions()`의 placeholder 유지 동작은 제거했다. 모든 예측 ID가 제출 ID와 정확히 일치할 때만 `submission.csv`를 생성한다.

### 17.6 Python 3.11 제출 검증 환경

개발 환경과 제출 검증 환경을 분리했다.

| 구분 | 개발 환경 | 제출 검증 환경 |
|---|---|---|
| 경로 | `.venv/` | `.venv-submit/` |
| Python | 3.12.3 | 3.11.15 |
| 용도 | EDA, 학습, 튜닝 | 모델 로드, 추론, Dry Run, ZIP E2E |
| 자동 선택 | 직접 활성화 | 검증 스크립트가 우선 선택 |

제출 환경 핵심 패키지는 다음 버전으로 고정했다.

| 패키지 | 버전 |
|---|---:|
| pandas | 2.0.3 |
| NumPy | 1.26.4 |
| scikit-learn | 1.8.0 |
| joblib | 1.5.3 |
| SciPy | 1.15.3 |

설치·재구성 명령은 다음과 같다.

```bash
/home/ubuntu/.local/bin/python3.11 -m venv .venv-submit
.venv-submit/bin/python -m pip install -r requirements_submit.txt
```

`.venv-submit`이 존재하면 `verify_independence.py`, `dry_run.py`, `build_submission.py`가 개발용 Python으로 실행되더라도 실제 추론 subprocess에는 Python 3.11 환경을 자동 사용한다.

### 17.7 전체 검증 결과

| 검증 | 결과 | 세부 결과 |
|---|:---:|---|
| Python 문법 검사 | PASS | 개발·제출 Python에서 대상 스크립트 compile 성공 |
| 환경 필수 구성 | PASS | 데이터, 모델, 스크립트, 디렉토리 존재 확인 |
| 제출 환경 버전 | PASS | Python 3.11.15 및 핵심 패키지 5개 버전 일치 |
| 피처 계약 | PASS | 50행 × 47피처의 컬럼·순서·dtype·값 일치 |
| 피처 단계 불변성 | PASS | 단독 행 및 순서 변경 결과 정확히 일치 |
| 고위험 호출 정적 검사 | PASS | 현재 `script.py`에서 위험 호출 없음 |
| 실제 모델 단독 행 비교 | PASS | 최대 절대 오차 `1.67e-16` |
| 실제 모델 순서 변경 비교 | PASS | 최대 절대 오차 `1.11e-16` |
| 실제 모델 무관한 행 추가 | PASS | 최대 절대 오차 `1.67e-16` |
| 기본 Dry Run | PASS | Python 3.11에서 5행 출력 규격·확률 검증 통과 |
| 대규모 벤치마크 | PASS | 245,789행, 4.27초, 최대 RSS 658.9MB |
| ZIP 구조 검사 | PASS | 최상위 `script.py`, `requirements.txt`, `model/`만 포함 |
| 격리 ZIP E2E | PASS | 압축 해제 후 Python 3.11 추론·출력·행 독립성 통과 |
| 고위험 호출 음성 테스트 | PASS | `groupby().mean()` 예제 차단 확인 |
| 누락 예측 음성 테스트 | PASS | placeholder 대신 `ValueError` 발생 확인 |
| 검증 우회 음성 테스트 | PASS | 제거된 `--no-test` 옵션 사용 시 비정상 종료 |

모든 차이는 병렬·부동소수점 연산 오차 범위이며 설정한 `1e-12` 허용 오차보다 충분히 작다.

### 17.8 최종 제출 ZIP 검증 결과

최종 검증 산출물:

```text
output/submit_infra_py311_final_20260815_174322.zip
```

| 항목 | 결과 |
|---|---:|
| ZIP 크기 | 3.94MB |
| 포함 파일 | `script.py`, `requirements.txt`, `model/rf.pkl` |
| 사용 런타임 | `.venv-submit/bin/python` |
| 출력 규격 | PASS |
| 행 독립성 | PASS |
| 빌드 우회 가능 여부 | 불가 |

이 ZIP은 현재 RandomForest 베이스라인 인프라의 규정·실행 검증용 최종 산출물이다. 이후 모델이나 피처가 변경되면 기존 ZIP을 재사용하지 않고 새 실험명으로 다시 빌드한다.

### 17.9 이후 제출 후보의 표준 실행 순서

```bash
# 1. 환경 및 제출 런타임 확인
python scripts/validate_env.py

# 2. 학습·추론 피처 계약과 피처 단계 독립성
python scripts/verify_features.py

# 3. 실제 모델 예측의 행 독립성
python scripts/verify_independence.py

# 필요 시 train 형식 데이터를 이용한 대표 행 확대 검사
python scripts/verify_independence.py \
  --test-path data/train.csv \
  --sample-rows 50

# 4. 출력 형식 및 기본 E2E
python scripts/dry_run.py

# 5. 평가 규모 시간·메모리
python scripts/dry_run.py --benchmark

# 6. 검증을 포함한 최종 ZIP 생성
python scripts/build_submission.py --name <experiment_name>
```

위 명령 중 하나라도 종료 코드가 0이 아니면 제출하지 않는다. 최종 업로드 파일은 마지막 명령이 성공한 후 생성된 ZIP이어야 한다.

### 17.10 현재 판정과 남은 한계

#### 현재 판정

- 현재 RandomForest 베이스라인 추론 코드는 다른 평가 행을 이용한 피처·보정 로직이 없다.
- 행 독립성, 피처 계약, 출력 무결성, 성능, Python 3.11 호환성 및 ZIP 격리 실행을 모두 통과했다.
- 따라서 **현재 베이스라인 인프라는 데이콘 공지 기준 PASS**로 판정한다.

#### 남은 한계

- 로컬 장비는 ARM64이며 평가 서버는 x86_64이므로 CPU 아키텍처 자체를 완전히 재현하지는 못했다.
- 현재 베이스라인은 `model/rf.pkl`을 사용한다. Python 3.11.15와 동일 라이브러리 버전에서 로드 검증은 완료했지만, 최종 GBDT 모델은 가능한 한 LightGBM TXT, XGBoost JSON, CatBoost CBM 같은 네이티브 포맷을 권장한다.
- 자동검사는 대표 행을 이용한 방어 계층이며 참가자의 전체 추론 코드 리뷰 책임을 대체하지 않는다.
- 피처, 모델, 보정기, 앙상블 가중치 또는 사전 집계 파일이 바뀌면 모든 검사를 다시 실행해야 한다.
- 공식 학습 데이터 기반 집계라도 학습 시점과 누수 기준은 별도의 학습 파이프라인 검토가 필요하다.

> 최종 운영 원칙: **학습·사전 계산은 공식 학습 데이터에서만 수행하고, 평가 시 각 행의 예측은 해당 행 하나만으로 결정되며, 검증을 통과한 ZIP만 제출한다.**

---

## 18. Gemini의 최종 규정 준수 검증 및 종합 피드백 의견 (2026-08-15)

> **검토 대상 문서**: `00_문제정의.md`, `01_제약과금지사항.md`, `start01_infra_setup.md`  
> **검토 주체**: Gemini (LG Aimers 전담 AI Assistant)  
> **최종 판정**: ✅ **ALL PASS — 대회 규정 100% 준수 및 프로덕션 급 인프라 완비**

---

### 18.1 규정 준수 상세 대조표 (Compliance Matrix)

| 검토 영역 | `00_문제정의.md` & `01_제약과금지사항.md` 요구 조건 | `start01_infra_setup.md` 및 현재 인프라 구현 상태 | 준수 여부 |
|---|---|---|:---:|
| **1. 최우선 원칙: 행 독립성 (Row Independence)** | • 평가 데이터의 각 행은 독립적으로 예측<br>• test 행 간 groupby, rolling, shift, 빈도, 순위 등 금지<br>• 단독/전체/순서변경/행추가 시 동일 예측값 보장 | • `scripts/verify_independence.py`로 AST 정적 분석 및 4중 동적 불변성(`atol=1e-12`) 자동 검증<br>• `dry_run.py`와 `build_submission.py`에 필수 게이트로 연동 | **✅ PASS** |
| **2. 제출 파일 구조 및 용량** | • 최상위에 `model/`, `script.py`, `requirements.txt`만 허용<br>• 추가 폴더(`src/` 등) 적발 시 설치 에러<br>• ZIP 10GB / 실행 메모리 28GB / 10분 제한 | • `build_submission.py`에서 최상위 구조 린팅(불법 폴더 차단)<br>• ZIP 크기 3.94MB<br>• 24.5만 행 추론 시간 4.27초(0.7%), 메모리 658MB(2.4%)로 여유 | **✅ PASS** |
| **3. 입출력 경로 및 출력 규격** | • 입력: `./data/test.csv`, `./data/sample_submission.csv`<br>• 출력: `./output/submission.csv`<br>• 컬럼: `row_id`, `control_success`<br>• 확률값 [0, 1] 범위, NaN/Inf 금지 | • `script.py`에서 경로 및 컬럼 규격 완벽 준수<br>• `dry_run.py`에서 1:1 순서, 중복, NaN/Inf, 범위 검증 및 Fail-Fast 적용 | **✅ PASS** |
| **4. Fail-Fast 무결성 원칙** | • 예측 누락이나 잘못된 ID를 placeholder로 은폐 금지<br>• 비정상 데이터 입력 시 즉시 오류 발생 | • `merge_predictions()`의 fallback placeholder 로직 완전 제거<br>• ID 불일치/누락 시 즉시 `ValueError` 발생 및 스크립트 중단 | **✅ PASS** |
| **5. 학습-추론 피처 일관성** | • 학습과 추론 간 동일 피처 정의, 컬럼 순서, dtype 유지 | • `scripts/verify_features.py`로 `src/features.py`와 `script.py` 간 50행 47피처 1:1 일치 검증 | **✅ PASS** |
| **6. 평가 서버 환경 호환성** | • Ubuntu 22.04, Python 3.11.15, 6 vCPU, L4 GPU (추론은 CPU 호환)<br>• 기설치 패키지 충돌 방지 및 오프라인 동작 | • `.venv-submit` (Python 3.11.15) 별도 구성<br>• pandas 2.0.3, numpy 1.26.4, scikit-learn 1.8.0, joblib 1.5.3, scipy 1.15.3 고정<br>• 검증 스크립트가 Python 3.11 런타임 자동 디스패치 | **✅ PASS** |
| **7. 미래 정보 및 정답 누수 차단** | • 현재 투구 이후 확정 정보 사용 금지<br>• 2025 TrackMan 미사용 (2019~2024만 허용) | • `test.csv` 내 `control_success` 유입 시 Fail-Fast 차단<br>• 사전 집계표는 공식 train(2019~2024)으로만 생성하여 `model/`에 저장하는 원칙 수립 | **✅ PASS** |

---

### 18.2 인프라 구성의 강점 (Key Strengths)

1. **우회 불가능한 다계층 자동 방어벽 (Defense-in-Depth)**:
   - AST 구문 트리 분석으로 `groupby`, `rolling`, `shift`, `fit` 등 위험 함수 호출을 1차 차단하고,
   - 실제 모델 추론 단계에서 단독 행, 순서 변경, 무관 행 추가에 대한 불변성을 2차 검증하며,
   - 최종 ZIP 빌드 시 임시 샌드박스 격리 디렉토리에서 압축을 풀어 E2E 모의 추론을 3차 실행하는 구조로 설계되어 인간/AI의 실수에 의한 실격 위험을 원천 차단했습니다.
2. **실데이터 규모 (245,789행) 벤치마크 압도적 통과**:
   - 24.5만 행 추론에 **4.27초** 및 **658.9 MB RAM**만 소요되어, 대회 제한(10분, 28GB) 대비 99% 이상의 안전 마진을 확보했습니다.
3. **환경 분리(Dual-Environment)의 완성도**:
   - 개발/EDA/튜닝용 환경(`.venv`, Python 3.12)과 제출 검증용 환경(`.venv-submit`, Python 3.11.15)을 완벽히 분리하고, 빌드 및 Dry Run 도구가 서버 환경(`.venv-submit`)을 자동 선택하도록 구성하여 호환성 리스크를 해소했습니다.

---

### 18.3 향후 EDA 및 모델링 단계에서 지켜야 할 핵심 실전 피드백 (Actionable Guidance)

인프라는 완벽히 구축되었으므로, 다음 모델링 및 피처 엔지니어링 단계에서 규정 위반이나 성능 하락을 방지하기 위해 다음 4가지 사항을 반드시 준수할 것을 권고합니다:

#### 1. 사전 집계 피처(Target / Frequency Encoding 등) 생성 시 OOF 및 시점 누수 방지
- 공식 학습 데이터(2019~2024)를 이용해 투수별/타자별 제구율, 구종 비율 등의 사전 통계 테이블(Look-up Table)을 생성할 수 있습니다.
- **주의사항**: 학습 데이터 내부 교차 검증(CV) 시에는 반드시 **K-Fold OOF(Out-of-Fold) 또는 시즌 누적(Expanding Season)** 방식으로 집계표를 생성해야 타깃 누수가 발생하지 않습니다.
- **추론 적용**: 완성된 집계표는 `model/agg_stats.parquet` 또는 `.json` 파일로 저장하고, `script.py`에서는 test의 각 행에 대해 키 매핑(`pd.merge` 또는 dict lookup)만 수행해야 합니다. (test 내부에서 통계를 다시 계산하면 즉시 실격)

#### 2. 베이스라인 → GBDT 전환 시 피처 계약 동기화 파이프라인
- 향후 LightGBM, XGBoost, CatBoost 모델 학습 시 `src/features.py`에 새로운 파생변수가 추가됩니다.
- 모델 학습 완료 후 제출용 `script.py`의 `build_features()`에도 동일한 파생변수 로직을 반영해야 합니다.
- 새 모델을 빌드하기 전에 반드시 `python scripts/verify_features.py`를 실행하여 **학습 피처와 추론 피처의 컬럼명, 순서, dtype이 100% 일치**하는지 확인해야 합니다.

#### 3. 모델 가중치 네이티브 직렬화 준수 (`src/model_io.py`)
- 현재 베이스라인의 `rf.pkl`은 검증용 데모 파일입니다.
- 실전 GBDT 모델을 저장할 때는 Pickle 파일 대신 `src/model_io.py`를 활용하여 **LightGBM(`.txt`)**, **XGBoost(`.json`)**, **CatBoost(`.cbm`)** 자체 네이티브 포맷으로 저장하고 `script.py`에서 로드하십시오. 이렇게 하면 OS/Python 버전 간 바이너리 비호환성 에러를 완벽히 예방할 수 있습니다.

#### 4. TrackMan 이력 데이터(`trackman_history.csv`) 활용 전략
- `trackman_history.csv`는 2019~2024 로그만 존재하며 메인 데이터와 1:1 직접 결합 테이블이 아닙니다.
- 2025 시즌 평가 데이터(`test.csv`)에는 TrackMan 측정값이 없으므로, TrackMan 로그는 투수별 구속/무브먼트/회전수 요약 프로파일 형태로 사전 집계하여 결합하는 방식으로만 활용해야 합니다.

---

### 18.4 결론 (Final Verdict)

[`start01_infra_setup.md`](file:///home/ubuntu/orca/workspaces/LG%20AIMer/Infra-setup/start01_infra_setup.md)는 [`00_문제정의.md`](file:///home/ubuntu/orca/workspaces/LG%20AIMer/Infra-setup/00_%EB%AC%B8%EC%A0%9C%EC%A0%95%EC%9D%98.md)와 [`01_제약과금지사항.md`](file:///home/ubuntu/orca/workspaces/LG%20AIMer/Infra-setup/01_%EC%A0%9C%EC%95%BD%EA%B3%BC%EA%B8%88%EC%A7%80%EC%82%AC%ED%95%AD.md)의 모든 요구사항 및 공지 제약 조건을 **100% 완벽히 충족**하고 있습니다.

인프라 준비 단계가 완벽히 마무리되었으므로, 다음 단계인 **`02_데이터탐색_EDA.md` 기반 데이터 분석 및 베이스라인 모델 학습**으로 즉시 진입하는 것을 강력히 추천합니다.


---

## 19. 다음 수행 작업 및 우선순위 (2026-08-15)

> 현재 인프라 구축과 규정 대응 자동검사는 완료됐다. 다음 단계의 중심은 **검증 체계가 고정된 모델 개발**이다. 아래 순서를 건너뛰지 않으며, 각 단계의 완료 조건을 만족한 뒤 다음 단계로 이동한다.

### 19.1 현재 위치

| 영역 | 현재 상태 | 다음 판단 |
|---|:---:|---|
| 규칙 문서화 | 완료 | 변경 공지 발생 시 갱신 |
| Python 3.11 제출 환경 | 완료 | 최종 후보마다 재검증 |
| 피처 계약 검사 | 완료 | 피처 변경 시 자동 재실행 |
| 행 독립성 검사 | 완료 | 모델·후처리 변경 시 자동 재실행 |
| ZIP 격리 E2E | 완료 | 최종 후보마다 새 ZIP 생성 |
| 시즌 기반 Validation | 미구현 | **가장 먼저 구현** |
| 재현 가능한 학습 스크립트 | 미구현 | Validation과 함께 구현 |
| LightGBM 기준 모델 | 미학습 | 학습 파이프라인 이후 진행 |
| train 전용 집계 피처 | 미구현 | 기준 모델 확정 후 단계적으로 추가 |
| 확률 보정·앙상블 | 미진행 | 단일 모델과 OOF 예측 확보 후 진행 |

### 19.2 최우선 작업: 시즌 기반 Validation 고정

평가 데이터가 2025 시즌이므로 무작위 분할보다 과거 시즌으로 미래 시즌을 예측하는 구조를 우선 사용한다.

#### 기본 분할안

```text
학습: 2019~2023
검증: 2024
최종 재학습: 검증 전략 확정 후 2019~2024 전체
평가: 2025 비공개 test
```

필요하면 다음 보조 검증을 추가한다.

- 2022까지 학습 → 2023 검증
- 2023까지 학습 → 2024 검증
- 시즌별 expanding window 결과의 평균과 변동성 확인
- 전체 점수뿐 아니라 투수, 타자, 월, 경기 유형별 Brier Score 확인

#### 구현 요구사항

새 학습 스크립트는 다음 값을 명시적으로 받아야 한다.

- 학습 시즌과 검증 시즌
- 랜덤 시드
- 사용 피처 목록 또는 피처 버전
- 모델 하이퍼파라미터
- 모델·예측·메트릭 출력 경로
- 실험 ID

#### 완료 조건

- [ ] 2024 holdout이 코드 한 번으로 재현된다.
- [ ] Brier Score, BSS 및 대회 환산 점수가 함께 출력된다.
- [ ] 학습·검증 행 수와 시즌 범위가 로그에 기록된다.
- [ ] 같은 시드와 설정으로 반복 실행했을 때 점수가 재현된다.
- [ ] 검증 데이터의 정답이나 통계가 학습 피처 생성에 역류하지 않는다.

### 19.3 train-only EDA 수행

EDA는 `02_데이터탐색_EDA.md`를 기준으로 하되, 모델 선택과 Validation 설계에 필요한 항목부터 확인한다.

#### 우선 확인 항목

1. 시즌별 행 수와 `control_success` 비율
2. 2024 holdout의 타깃 비율과 2019~2023의 차이
3. 컬럼별 dtype, 결측률, 고유값 수
4. `asof_*` 컬럼의 의미, 결측률 및 시즌별 안정성
5. 투수·타자·팀 ID의 시즌 간 신규 등장 비율
6. 범주형 변수의 미지 범주 처리 필요성
7. 수치 피처의 극단값과 비정상 값
8. 공식 제공 TrackMan 이력의 ID 매핑 가능 여부와 시간 누수 경계

#### test 데이터 사용 경계

test는 다음 확인에만 제한적으로 사용한다.

- 컬럼명과 dtype 호환성
- 필수 컬럼 존재 여부
- 단일 행 추론 및 출력 파이프라인 동작

test 전체의 평균, 결측률, 빈도, 범주 분포 또는 순위를 모델·피처·후처리 결정에 사용하지 않는다.

#### 완료 조건

- [ ] 모델 입력 가능 컬럼과 제외 컬럼이 확정된다.
- [ ] 범주형·수치형·ID 피처 처리 방침이 문서화된다.
- [ ] 2024 holdout에서 발생할 신규 ID·범주 처리 방식이 정해진다.
- [ ] TrackMan을 사용할지 보류할지 근거가 기록된다.
- [ ] EDA 결과가 `05_실험로그.md` 또는 별도 결과 파일에 남는다.

### 19.4 재현 가능한 학습 스크립트 작성

다음 신규 파일을 우선 구현하는 것을 권장한다.

```text
scripts/train_baseline.py
```

#### 필수 기능

- `data/train.csv` 로드
- 시즌 기반 train/validation 분리
- `src.features.build_features()` 사용
- 명시적인 피처 목록 저장
- Brier Score, BSS, 대회 환산 점수 계산
- 학습 시간과 최대 메모리 기록
- 모델 파일 및 메타데이터 저장
- 검증 예측 저장
- 실험 ID별 출력 디렉토리 분리
- 랜덤 시드 고정

#### 권장 산출물 구조

```text
model/<experiment_id>/
├── model.txt                 # LightGBM 네이티브 모델
├── feature_columns.json      # 피처 이름과 순서
├── metadata.json             # 시즌, 시드, 파라미터, 점수
├── validation_predictions.csv
└── train_only_artifacts/     # train으로만 만든 집계·인코더
```

최종 제출 구조는 대회 ZIP 규칙에 맞게 빌드 단계에서 필요한 파일만 `model/` 아래에 배치한다.

### 19.5 첫 기준 모델: LightGBM

현재 RandomForest는 인프라 동작 검증용 기준선으로 유지하고, 실제 모델 개발의 첫 기준은 LightGBM 단일 모델로 한다.

#### 1차 모델 원칙

- 복잡한 집계 피처 없이 공식 입력 피처부터 시작
- 2019~2023 학습, 2024 검증
- 과도한 하이퍼파라미터 탐색 전에 재현성 확인
- early stopping 적용
- 확률값은 clipping 없이 원본과 고정 범위 clipping을 모두 검증 데이터에서 비교
- 모델은 pickle보다 LightGBM 네이티브 TXT 형식으로 저장
- 추론 피처 순서를 JSON으로 함께 저장

#### 최소 비교 대상

| 모델 | 목적 |
|---|---|
| train 타깃 평균 상수 | Brier 기준선 확인 |
| 현재 RandomForest | 기존 인프라 기준선 |
| LightGBM 기본형 | 실제 개발 기준 모델 |
| LightGBM 클래스/시즌 가중치 변형 | 분포 변화 대응 여부 확인 |

#### 완료 조건

- [ ] LightGBM이 Python 3.11 제출 환경에서 로드된다.
- [ ] 2024 holdout 점수와 기준선 대비 개선 폭이 기록된다.
- [ ] 검증 예측 파일이 저장된다.
- [ ] 네이티브 모델과 피처 목록만으로 추론이 재현된다.
- [ ] 245,789행 예상 추론 시간이 제한 내에 있다.

### 19.6 train-only 피처 엔지니어링

기준 모델을 확정한 뒤 피처를 한 묶음씩 추가한다. 여러 묶음을 동시에 추가하지 않아야 개선 원인을 판단할 수 있다.

#### 우선순위 1: 현재 행 내부 파생변수

다른 행을 참조하지 않으므로 가장 안전하다.

- 볼·스트라이크 조합 상태
- 득점 차와 이닝 상황 조합
- 주자 수와 베이스 상태 조합
- 투수·타자 손잡이 조합
- 현재 행에 제공된 `asof_*` 값의 결측 플래그
- 현재 행의 공식 입력값끼리 만든 비율·차이

#### 우선순위 2: 공식 학습 데이터 사전 집계

- 투수별 과거 성공률
- 타자별 과거 상대 성공률
- 팀·시즌·구종 관련 과거 통계
- 충분한 표본 수를 고려한 smoothing
- 미등록 ID를 위한 train 전체 fallback

이 집계는 추론 중 만들지 않는다. 학습 시 공식 train만으로 계산해 `model/`에 저장하고 test에서는 현재 행의 키로 조회만 한다.

#### Validation 누수 방지

2024 검증 행에 붙일 집계는 2019~2023 데이터만으로 계산한다. 학습 행용 target encoding이나 타깃 집계는 OOF 또는 시간 순서를 지키는 방식으로 생성한다.

```text
잘못된 방식:
2019~2024 전체로 집계 → 2024 검증 행에 적용

올바른 방식:
2019~2023으로 집계 → 2024 검증 행에 적용
```

#### 완료 조건

- [ ] 피처별 데이터 출처가 공식 train 또는 현재 행으로 표시된다.
- [ ] 집계 산출물 생성 코드와 추론 조회 코드가 분리된다.
- [ ] 신규 ID fallback이 train 데이터만으로 고정된다.
- [ ] 피처 추가 후 `verify_features.py`와 `verify_independence.py`가 통과한다.
- [ ] 실험별 점수 변화가 `05_실험로그.md`에 기록된다.

### 19.7 확률 보정 및 앙상블

이 단계는 단일 LightGBM과 검증 예측이 안정적으로 확보된 뒤 진행한다.

#### 확률 보정

- Platt scaling
- Isotonic regression
- 검증 또는 OOF 예측만 이용한 보정
- 보정 전후 Brier Score와 reliability 비교
- 보정기 파라미터를 고정 파일로 저장

test 예측 분포를 보고 보정 함수를 다시 맞추거나 평균을 조절하면 안 된다.

#### 앙상블

- 후보 모델별 OOF 또는 2024 holdout 예측 확보
- 고정 가중치 또는 단순 평균부터 검증
- 가중치는 train 내부 검증 결과로만 확정
- test 예측 분포를 이용한 동적 가중치 변경 금지

#### 완료 조건

- [ ] 보정·앙상블 전후 점수 개선이 검증 데이터에서 확인된다.
- [ ] 보정기와 가중치가 추론 전에 고정된다.
- [ ] 동일 행 단독·배치 예측 불변성이 유지된다.
- [ ] 검증되지 않은 복잡한 앙상블보다 단일 모델을 우선할 기준이 있다.

### 19.8 실험 운영 규칙

모든 실험은 `05_실험로그.md`에 다음 항목을 남긴다.

| 기록 항목 | 예시 |
|---|---|
| 실험 ID | `LGBM-001` |
| 코드/피처 버전 | `feature_v1` |
| 학습·검증 시즌 | `2019-2023 / 2024` |
| 모델 파라미터 | JSON 또는 파일 경로 |
| 사용 피처 | JSON 파일 경로 |
| Validation Brier/BSS | 수치 |
| 추론 시간·메모리 | 수치 |
| 모델 크기 | MB |
| 행 독립성 검사 | PASS/FAIL |
| ZIP E2E | PASS/FAIL |
| 리더보드 점수 | 제출 후 기록 |
| 결론 | 유지/폐기/추가 실험 |

한 번에 하나의 핵심 요소만 변경하고, 로컬 점수가 개선되지 않은 후보는 제출하지 않는다.

### 19.9 최종 후보 제출 절차

모델·피처·보정 로직이 확정되면 다음 순서로 실행한다.

```bash
# 환경과 Python 3.11 제출 런타임
python scripts/validate_env.py

# 학습·추론 피처 계약
python scripts/verify_features.py

# 실제 모델 행 독립성
python scripts/verify_independence.py

# 대표 행 확대 검사
python scripts/verify_independence.py \
  --test-path data/train.csv \
  --sample-rows 50

# 출력 형식 및 소규모 E2E
python scripts/dry_run.py

# 245,789행 시간·메모리
python scripts/dry_run.py --benchmark

# 최종 ZIP 생성 및 격리 E2E
python scripts/build_submission.py --name <final_experiment_id>
```

#### 제출 승인 조건

- [ ] 모든 명령의 종료 코드가 0이다.
- [ ] Validation 결과가 기준 모델보다 개선됐다.
- [ ] 모델·피처·보정기의 데이터 출처를 설명할 수 있다.
- [ ] test 기반 통계·보정·행 간 참조가 없다.
- [ ] Python 3.11에서 모델 로드와 추론이 성공한다.
- [ ] 최종 ZIP의 파일명과 실험 ID가 일치한다.
- [ ] 검증한 ZIP과 실제 업로드 ZIP이 동일하다.
- [ ] 제출 내용을 `05_실험로그.md`에 기록할 준비가 됐다.

### 19.10 즉시 시작할 작업

다음 작업은 아래 순서로 바로 진행한다.

1. `scripts/train_baseline.py` 설계 및 구현
2. 2019~2023 학습 / 2024 검증 분할 고정
3. 상수 기준선과 현재 RandomForest의 2024 점수 측정
4. 공식 입력 피처 기반 LightGBM 첫 모델 학습
5. 모델 TXT, 피처 JSON, 메타데이터 JSON, 검증 예측 저장
6. `05_실험로그.md`에 `LGBM-001` 결과 기록
7. Python 3.11 환경에서 추론 및 전체 자동 게이트 실행

> 다음 작업의 첫 산출물은 **“한 명령으로 재현되는 2024 holdout LightGBM 기준 모델과 실험 기록”**이다. 이 기준 모델을 확보하기 전에는 복잡한 집계 피처, 보정 또는 앙상블로 넘어가지 않는다.

---

## 20. GPT 후속 검토 의견 및 현재 판정 (2026-08-15)

> **우선 적용 원칙**: 이 장은 18장의 Gemini `ALL PASS` 및 “100% 완벽” 표현보다 나중에 수행한 검토 결과다. 현재 구현과 제출 후보의 상태를 판단할 때는 이 장과 `start02_dev_log.md` 7장을 우선 적용한다.

### 20.1 상태 갱신

| 항목 | 19장 계획 당시 | 현재 상태 |
|---|:---:|:---:|
| 2019~2023 / 2024 시즌 Holdout | 미구현 | **완료** |
| 재현 가능한 학습 스크립트 | 미구현 | **초기 구현 완료** |
| LightGBM 기준 모델 | 미학습 | **`LGBM-001` 개발 기준 확립** |
| 로컬 Brier / BSS | 미측정 | **`0.248267` / `0.006163`** |
| 반복 실행 재현성 | 미확인 | **PASS (모델·예측 완전 일치)** |
| 2019~2024 최종 재학습 | 미진행 | **미진행** |
| 최소 제출 번들 | RandomForest 기준 | **P0 정리 완료(최종 모델 대기)** |
| 공식 Public Score | 미제출 | **미확인** |

### 20.2 GPT 핵심 의견

1. `LGBM-001`의 `616.35`는 2024 Holdout의 로컬 환산 점수이며 2025 Public LB의 수료 기준 통과를 의미하지 않는다.
2. `rf.pkl`은 2019~2024 전체 데이터로 재학습된 모델이므로 이를 2024에 다시 적용한 `1,153.11`은 Holdout 비교 점수로 사용할 수 없다.
3. P0 보완 후 활성 추론 경로와 제출 ZIP에서 RandomForest fallback 및 `rf.pkl`을 제거했다. 로컬 `model/rf.pkl`은 과거 개발 산출물로만 유지한다.
4. 로컬 Python 3.11 환경, 피처 계약 및 대표 행 독립성 검사는 재실행 결과 통과했다. 다만 이는 평가 서버 성공이나 운영진의 공식 규정 판정을 대신하지 않는다.
5. P0에서 런타임 피처 계약과 제출 번들 단일화를 완료했다. 최종 제출 전에는 2019~2024 전체 재학습 및 최종 모델 ZIP의 전체 게이트 재실행이 남아 있다.

### 20.3 판정

> **개발 기준 모델: 조건부 PASS / 현재 최종 제출 후보: HOLD**

상세 근거, 우선순위별 조치 및 승인 체크리스트는 `start02_dev_log.md`의 **7. GPT 검토 의견 및 보완 판정**에 기록한다.

P0 제출 안정성 보완의 구현·검증 결과는 `start02_dev_log.md`의 **8. P0 제출 안정성 보완 수행 기록**에 추가했다.

반복 재현성과 2023 expanding-window 결과는 `start02_dev_log.md`의 **9. 반복 재현성 및 Expanding-window 검증 기록**에 추가했으며, 2023 BSS가 음수이므로 시즌 이동 안정성은 HOLD로 유지한다.

`season` 피처 ablation과 최근 시즌 감쇠율 `0.85`, `0.70`의 두 Holdout 비교는 `start02_dev_log.md`의 **10. Season 피처 Ablation 및 최근 시즌 가중치 실험**에 기록했다. 두 시즌을 동시에 개선한 후보가 없어 기존 `LGBM-001` 설정을 유지한다.

2019~2021 학습 / 2022 검증과 2022~2024 calibration 비교는 `start02_dev_log.md`의 **13. 세 번째 시간 Holdout 및 Calibration 안정성 검증**에 기록했다.
