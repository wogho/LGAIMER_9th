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
├── .venv/                      # Python 가상환경 (gitignore)
├── .gitignore                  # Git 제외 규칙
├── requirements.txt            # pip freeze 결과 (전체)
│
├── data/                       # 데이터 디렉토리
│   ├── train.csv               # 학습 데이터 (gitignore, 147만 행)
│   ├── test.csv                # 평가 데이터 (샘플 5행, 서버에서 24.5만 행)
│   ├── sample_submission.csv   # 제출 양식
│   └── trackman_history.csv    # TrackMan 보조 데이터 (gitignore)
│
├── src/                        # 핵심 공유 모듈
│   ├── __init__.py
│   ├── config.py               # 경로, 상수, 하이퍼파라미터
│   └── features.py             # 피처 엔지니어링 (학습 == 추론 동일)
│
├── notebooks/                  # Jupyter 실험 노트북
│
├── model/                      # 학습된 모델 파일 저장
│
├── output/                     # 예측 결과 및 제출 zip
│
├── scripts/                    # 유틸리티 스크립트
│   ├── validate_env.py         # 환경 검증
│   ├── build_submission.py     # 제출 zip 빌드
│   └── dry_run.py              # 로컬 제출 시뮬레이션
│
├── script.py                   # 추론 스크립트 (제출용)
├── baseline_submit.zip         # 대회 제공 베이스라인
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

---

## 7. 유틸리티 스크립트

### 7.1 환경 검증

```bash
source .venv/bin/activate
python scripts/validate_env.py
```

Python, 패키지, 데이터, 디렉토리 구조를 일괄 검증합니다.

### 7.2 제출 zip 빌드

```bash
python scripts/build_submission.py --name lgbm_v1
# → output/submit_lgbm_v1_20260814_120000.zip
```

구조 검증 후 `script.py` + `requirements.txt` + `model/`을 zip으로 묶습니다.

### 7.3 로컬 Dry Run

```bash
python scripts/dry_run.py
```

평가 서버를 모사하여 `script.py`를 실행하고 출력 형식, 시간, 확률 범위를 검증합니다.  
**하루 5회 제출 제한**이므로, 반드시 dry run 후 제출하세요.

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
  ❌ train.csv        — MISSING (대회 사이트에서 다운로드 필요)
  ✅ test.csv         (샘플)
  ✅ sample_submission.csv
  ⏭️  trackman_history.csv — 미존재 (선택, 대회 사이트에서 다운로드)

[CHECK] 디렉토리 구조:
  ✅ data/  ✅ model/  ✅ output/  ✅ notebooks/  ✅ src/  ✅ scripts/

  🎉 환경 설정 완료!
============================================================
```

---

## 9. 주의사항 & 금지사항 요약

### ❌ 절대 금지

1. **test 행 간 cross-referencing** — test.csv에 대한 groupby, rolling, 빈도 인코딩 등
2. **미래 정보 사용** — 투구 결과(실제 구종, 구속 등), 2025 TrackMan 데이터
3. **온라인 접속** — 평가 서버는 오프라인, 외부 API 호출 불가
4. **학습-추론 불일치** — `build_features()` 함수는 반드시 동일해야 함

### ⚠️ 주의

- 추론 `script.py`는 **CPU 전용**으로 작성 (GPU 드라이버 불일치 위험 회피)
- 출력 확률은 반드시 `0.0 ≤ p ≤ 1.0`, NaN/Inf 금지
- zip 파일 10GB 이하, 실행 시간 10분 이하

---

## 10. 다음 단계

| 순서 | 작업 | 상태 |
|---|---|---|
| 0 | ✅ 환경 구성 (이 문서) | 완료 |
| 1 | 📥 `train.csv`, `trackman_history.csv` 다운로드 → `data/` | **대기** |
| 2 | 📊 EDA 실행 (`02_데이터탐색_EDA.md` 기반) | 대기 |
| 3 | 🔧 시즌 기반 Validation 셋업 (2024 holdout) | 대기 |
| 4 | 🏗️ Baseline end-to-end (LightGBM 기본) | 대기 |
| 5 | 🔬 피처 엔지니어링 반복 | 대기 |
| 6 | 📐 확률 보정 (Isotonic Regression) | 대기 |
| 7 | 🎯 튜닝 & 앙상블 | 대기 |
| 8 | 📤 제출 | 대기 |

---

## 11. 인프라 환경 구성 분석 및 종합 검토 의견

### 11.1 종합 평가: **우수 (Well-Architected, 95/100)**
클로드가 구성한 인프라 환경은 해커톤의 문제 특성과 제약 조건을 잘 반영하여 체계적이고 실용적으로 구축되었습니다.

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
   - 현재 루트의 `requirements.txt`는 로컬 전체 패키지(38개)가 freeze된 상태입니다.
   - 오프라인 평가 서버 제출용 zip 빌드 시에는 `baseline_submit.zip`처럼 런타임 추론에 필수적인 패키지(`scikit-learn`, `joblib`, `pandas`, `lightgbm` 등)만 최소한으로 포함하도록 관리해야 합니다.

3. **로컬 Dry-Run을 위한 베이스라인 파일 준비**:
   - 현재 `baseline_submit.zip`이 압축된 상태이므로, 첫 dry-run 실행 전 `script.py` 및 `model/rf.pkl`을 프로젝트 루트로 압축 해제해 두어야 `scripts/dry_run.py`가 바로 동작합니다.

---

### 11.3 ✅ 권고사항 실행 결과 (2026-08-14 11:20)

> **작업자**: Claude (Opus) — Gemini(Flash) 권고 3건 + Claude 보충 1건 일괄 실행

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

**검증 완료**: LightGBM Booster 저장→로드→예측 정상 동작 확인.

#### 권고 2 → `requirements_submit.txt` 신규 생성

제출 zip 전용 경량 requirements 파일을 분리했습니다.

| 파일 | 용도 | 패키지 수 |
|---|---|---|
| `requirements.txt` | 로컬 개발 전체 (pip freeze) | 41개 |
| `requirements_submit.txt` | 제출 zip 번들 (추론 최소) | **4개** |

`build_submission.py`의 기본값도 `requirements_submit.txt`로 변경 완료.

#### 권고 3 → `baseline_submit.zip` 압축 해제

```
unzip -o baseline_submit.zip -d .
```

프로젝트 루트에 `script.py`와 `model/rf.pkl`이 배치되어 `dry_run.py` 즉시 실행 가능 상태.

**Dry Run 결과**:
```
✅ script.py 확인
✅ test.csv 확인
✅ sample_submission.csv 확인
🚀 script.py 실행 중...
⏱️  실행 시간: 1.7초 (0.0분)
✅ 컬럼 OK
✅ 행 수 OK
✅ row_id 일치
✅ 확률값 유효 [0.447574, 0.506271]
✅ DRY RUN 완료
```
