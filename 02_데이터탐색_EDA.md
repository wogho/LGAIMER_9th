
```markdown
# 02. 데이터 탐색 (EDA) 🔍

> 목적: 데이터를 "느낌"이 아니라 "근거"로 이해한다.
> 이 EDA의 결과가 → 검증 설계(02)·피처 전략(04)의 근거가 된다.
> 원칙: EDA에서도 **test로 통계 만들지 않는다** (train 위주로 탐색).

---

## 0. 준비 & 메모리 절약 로드

```python
import pandas as pd
import numpy as np

# 먼저 소량만 읽어 컬럼/타입 파악
train_head = pd.read_csv("./data/train.csv", nrows=1000)
print(train_head.shape)
print(train_head.dtypes)
print(train_head.columns.tolist())
```

```python
# dtype 최적화 로드 (28GB 여유 확보)
def reduce_mem(df):
    for c in df.columns:
        t = df[c].dtype
        if t == "float64":
            df[c] = df[c].astype("float32")
        elif t == "int64":
            cmin, cmax = df[c].min(), df[c].max()
            if cmin >= -128 and cmax <= 127:
                df[c] = df[c].astype("int8")
            elif cmin >= -32768 and cmax <= 32767:
                df[c] = df[c].astype("int16")
            else:
                df[c] = df[c].astype("int32")
    return df

train = reduce_mem(pd.read_csv("./data/train.csv"))
test  = reduce_mem(pd.read_csv("./data/test.csv"))
print(train.shape, test.shape)  # (1470000, 49) (245000, 48)
```

> 💡 `train`엔 `control_success` 있고 `test`엔 없다 → 컬럼 1개 차이.

---

## 1. 기본 점검 (제일 먼저) ✅

```python
# 1-1. 컬럼 차집합 확인 (train에만 있는 컬럼 = target)
print("train - test :", set(train.columns) - set(test.columns))
print("test - train :", set(test.columns) - set(train.columns))
```
**볼 것:** `train - test == {'control_success'}` 이어야 정상.
`test - train`이 비어있어야 함(추가 컬럼 있으면 피처 설계에 반영).

```python
# 1-2. 결측치 비율
miss = train.isna().mean().sort_values(ascending=False)
print(miss[miss > 0])
```
**볼 것:** 결측 많은 컬럼 = cold-start(신인·데이터 부족) 신호.
→ 결측 자체가 정보일 수 있음(플래그 피처 후보).

```python
# 1-3. 타입 분류
num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = train.select_dtypes(exclude=[np.number]).columns.tolist()
print("수치형:", len(num_cols), "범주형:", len(cat_cols))
print("범주형 컬럼:", cat_cols)
```

---

## 2. 타겟 분포 (control_success) 🎯

```python
vc = train["control_success"].value_counts(dropna=False)
print(vc)
print("성공 비율:", train["control_success"].mean())
```
**볼 것:**
- 클래스 비율(불균형 여부). 예: 성공률 0.6~0.7이면 약한 불균형.
- **Brier 계열 평가 → 기준선(전체 평균확률)이 중요**.
  이 평균값이 곧 "상수 예측 베이스라인"의 기준.

```python
# 상수 베이스라인의 Brier score (감 잡기용)
p = train["control_success"].mean()
brier_const = np.mean((train["control_success"] - p) ** 2)
print("상수예측 Brier:", brier_const)
```
> 이 값을 이겨야 의미가 있다. (BSS의 기준선 감각)

---

## 3. 시즌 분포 — 검증 설계의 핵심 근거 📅

> 대회 구조: 학습(2019~2024) → 평가(2025). **미래 예측**.

```python
# season 관련 컬럼명은 실제 데이터에 맞게 조정
print(train["season"].value_counts().sort_index())
if "season" in test.columns:
    print(test["season"].value_counts().sort_index())
```
**볼 것:**
- train 시즌 범위, test 시즌(2025 단일?) 확인.
- **train↔test 시즌 gap** → 랜덤 KFold 대신 **시즌 기반 split** 근거 확정.
- 시즌별 타겟 평균 추세 → drift 여부.

```python
# 시즌별 타겟 평균 (분포 이동 점검)
print(train.groupby("season")["control_success"].agg(["mean", "count"]))
```
> 시즌별 성공률이 흔들리면 → 최근 시즌 가중/검증 반영 고려.

---

## 4. asof_* 결측 분석 — cold-start 규모 🧊

> `asof_*` = 투구 시점 기준 과거 집계값(공식 제공 피처).
> 결측 = 이력 부족(신인/첫 등장) → **fallback 전략 필요**.

```python
asof_cols = [c for c in train.columns if c.startswith("asof_")]
print("asof 컬럼 수:", len(asof_cols))

asof_miss = train[asof_cols].isna().mean().sort_values(ascending=False)
print(asof_miss)
```
**볼 것:**
- asof 결측률이 test에서도 비슷한가? (분포 일치 확인)
```python
if all(c in test.columns for c in asof_cols):
    print((train[asof_cols].isna().mean() - test[asof_cols].isna().mean()))
```
- 결측 처리 방향:
  - 전체 평균/시즌 평균으로 대체 + **결측 플래그 피처**
  - 사전집계 fallback (train 통계) → `model/`에 저장

---

## 5. ID 매핑 검증 — trackman 활용 전략 결정 ⭐ (가장 중요)

> trackman을 조인할 수 있는지에 따라 **피처 전략 전체가 갈린다.**

```python
trackman = pd.read_csv("./data/trackman_history.csv")
print(trackman.shape)         # (1790000, 30)
print(trackman.columns.tolist())
```

### 5-1. ID 체계 비교

```python
# 메인 데이터의 pitcher id 후보
main_ids = train["pitcher_id"].unique()

# trackman의 pitcher id 후보 (컬럼명 실제 확인 필요)
tm_ids = trackman["pitcher_trackman_id"].unique()

print("main pitcher 수:", len(main_ids))
print("trackman pitcher 수:", len(tm_ids))

# 교집합 확인 (핵심!)
inter = set(main_ids) & set(tm_ids)
print("교집합 크기:", len(inter))
print("교집합 비율(main 기준):", len(inter) / len(main_ids))
```
**판단 기준:**
| 교집합 비율 | 해석 | 전략 |
| --- | --- | --- |
| 높음(~1.0) | **같은 ID 체계** | pitcher 단위 직접 조인 가능 ✅ |
| 낮음/0 | **다른 체계** | 직접 조인 불가 → 유형/구종군 단위 집계로만 활용 |

> ⚠️ ID가 문자열/숫자 타입 불일치로 교집합이 0처럼 보일 수 있음 → dtype 맞춰 재확인.

```python
# 값 샘플 눈으로 비교 (형식 확인)
print("main 샘플:", main_ids[:5])
print("tm   샘플:", tm_ids[:5])
```

### 5-2. trackman 시즌 범위 확인 (누수 방지)

```python
print(trackman["season"].value_counts().sort_index())
```
**볼 것:**
- 2019~2024만 있어야 정상 (2025 없음 확인).
- **현재 투구 이후 정보가 아님**을 보장 → asof 방식으로만 요약.

---

## 6. 수치 피처 빠른 스캔

```python
# 이상치·범위 감각
print(train[num_cols].describe().T[["min", "mean", "max", "std"]])
```
**볼 것:**
- 말도 안 되는 범위(음수 구속 등) → 오류/센티넬값(-999 등) 탐지.
- 상수 컬럼(std=0) → 제거 후보.

```python
# 상수/거의 상수 컬럼 탐지
nunique = train.nunique()
print("유니크 1개(상수):", nunique[nunique <= 1].index.tolist())
```

---

## 7. 누수(leakage) 의심 컬럼 재점검 🚨

> 01번 문서의 금지 정보와 대조.

```python
# 이름으로 의심되는 컬럼 훑기
suspect_kw = ["result", "actual", "outcome", "post", "final",
              "location", "pitch_result", "measured"]
for kw in suspect_kw:
    hit = [c for c in train.columns if kw in c.lower()]
    if hit:
        print(kw, "→", hit)
```
**볼 것:**
- 현재 투구의 **결과/실측/판정**을 담은 컬럼이 test에도 있는지.
- test에 없고 train에만 있다면 → **학습에도 쓰면 안 됨**(추론 시 부재).

```python
# train에만 있는 피처(타겟 제외) = 추론 시 못 쓰는 컬럼 → 사용 금지 후보
train_only = (set(train.columns) - set(test.columns)) - {"control_success"}
print("train에만 존재(주의):", train_only)
```

---

## 8. EDA 결과 → 다음 단계 연결

| EDA 발견 | 영향 문서 |
| --- | --- |
| 시즌 gap 존재 | → 검증: **시즌 기반 split** (03 로드맵 [2]) |
| asof 결측 규모 | → 피처: 결측 플래그 + fallback (04) |
| ID 매핑 가능 여부 | → trackman 전략 확정 (04) |
| 누수 의심 컬럼 | → 사용 금지 목록 확정 (01) |
| 타겟 성공률 p | → 캘리브레이션 기준값 (05) |

---

## 9. EDA 체크리스트

```
□ train/test 컬럼 차이 = control_success 뿐인지
□ 타겟 성공률 p, 상수예측 Brier 기록
□ season 분포 (train 범위 / test gap) 확인
□ 시즌별 타겟 평균 drift 여부
□ asof_ 결측률 (train vs test 유사한지)
□ trackman ID 교집합 비율 → 조인 가능 여부 결론
□ trackman 시즌 = 2019~2024 (2025 없음) 확인
□ 상수/센티넬(-999)/이상치 컬럼 탐지
□ 누수 의심(train-only, result/actual 등) 목록화
```

---

## 한 줄 요약

> **"시즌 gap과 ID 매핑, 이 두 가지가 검증 설계와 피처 전략의 뼈대를 결정한다."**
> EDA는 예쁜 그래프가 아니라 **의사결정 근거를 남기는 작업**이다.
```

---

## 작성 완료 ✅

**이 문서의 설계 포인트:**

- 🔍 **모든 코드에 "볼 것"을 명시** — 코드를 돌리는 것보다 *무엇을 보고 무슨 결정을 할지*가 핵심
- ⭐ **5번 ID 매핑 검증을 가장 강조** — trackman 조인 가능 여부가 피처 전략 전체를 가름
  - dtype 불일치로 교집합이 0처럼 보이는 함정까지 경고
- 📅 **3번 시즌 분포 → 검증 설계 근거** — 랜덤 KFold가 아닌 시즌 split을 써야 하는 이유를 데이터로 확인
- 🚨 **7번 누수 재점검** — `train`에만 있는 컬럼은 추론 시 부재 → 자동 탐지 코드로 사고 방지
- 🔗 **8번 결과→문서 연결표** — EDA 발견이 어느 문서의 어떤 결정으로 이어지는지 매핑

---

## ⚠️ 실행 시 주의사항

문서 내 **컬럼명은 예시**입니다. 실제 데이터에 맞게 조정하세요:

- `"season"` → 실제 시즌 컬럼명 (없을 수도, `game_date`에서 추출해야 할 수도)
- `"pitcher_id"`, `"pitcher_trackman_id"` → 실제 ID 컬럼명
- `"asof_"` 접두사 → 실제 피처 네이밍 규칙

> 💡 **가장 먼저 할 일**: `train_head = pd.read_csv(..., nrows=1000)` 로
> **실제 컬럼명을 눈으로 확인**한 뒤, 위 코드의 컬럼명을 치환하세요.

작성 완료되었습니다! 필요하시면 다음 문서나 코드 실행 결과 해석을 도와드리겠습니다 😊