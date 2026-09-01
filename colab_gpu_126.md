# [보존본] REF4-126 GPU 로컬·Colab 혼합 실행 가이드

> 이 문서는 2026-09-01 완료된 GPU 검증 실행의 역사적 계약이다. CPU 후속 실행에는 사용하지 않는다. 현재 실행 계약은 `colab.md`를 따른다.

이 문서는 Gemini가 `REF4-JM-R-RESIDUAL-STRICT-GPU-126`을 Google Colab T4에서 실행하고, 결과를 증거와 함께 로컬 담당자에게 돌려주기 위한 **고정 실행 계약**이다. 이 문서에 없는 모델 변경, 하이퍼파라미터 탐색, 제출 ZIP 생성은 허용하지 않는다.

## 1. 이번 작업의 범위

126은 113A를 대체하는 새 전체 모델이 아니다. 감사된 strict 113A OOF를 고정 anchor로 두고, Regular(`game_type=R`) 행에만 얕은 CatBoost residual을 추가할 가치가 있는지 strict-forward로 검증하는 단일 가설이다.

- 실험 ID: `REF4-JM-R-RESIDUAL-STRICT-GPU-126`
- 기준 예측: `p113a_strict`
- 변경 대상: Regular 행만
- Futures 행: `p126 == p113a_strict`의 **bitwise exact identity**
- 모델 seed: `17, 42, 777`
- residual scale 후보: `0, 0.025, 0.05, 0.075`
- residual clip: `±0.12`
- 최종 확률 clip: `[0.005, 0.995]`
- 이번 단계: 검증 전용
- 금지: `test.csv` 읽기, production fit, 제출 ZIP 생성, 113A 수정

시간 분할은 다음과 같이 고정한다.

| 평가 시즌 | residual 학습에 허용되는 시즌 | scale 선택에 허용되는 완료 fold |
|---|---:|---:|
| 2022 | 없음(워밍업, 보정 0) | 없음 |
| 2023 | 2022 | 없음이므로 보정 0 |
| 2024 | 2022~2023 | 2023만 |
| 향후 배포 권고치 | 해당 없음 | 완료된 2023~2024 correction fold |

무작위 K-fold, 동시즌 label 사용, validation label 기반 scale 선택은 허용하지 않는다. 126에서 `2022/2023`의 최종 보정이 0인 것은 오류가 아니라 strict scale 선택 규칙의 결과다.

## 2. 역할 분담

혼합 실행은 아래 세 층으로 분리한다.

1. **로컬 제어면**: 코드 작성, 입력 고정, SHA-256 manifest 생성, 결과 회수, 독립 감사
2. **Google Drive 전달면**: 불변 입력, checkpoint, 로그, 결과 artifact 보관
3. **Colab T4 계산면(Gemini 담당)**: 고정 notebook을 순서대로 실행하고 artifact를 생성

Gemini는 계산 운영자이자 증거 수집자다. 결과 수치를 손으로 옮기거나, 실패를 고치기 위해 모델 구조·seed·scale·feature를 임의 변경해서는 안 된다.

## 3. 고정된 Drive 위치와 파일

- 사용자 공유 폴더: <https://drive.google.com/drive/folders/1ClBWXk-Hn1X1eh2et_MO2g7oIc8vjksx?usp=sharing>
- 126 직접 폴더: <https://drive.google.com/drive/folders/13Tje6IQ5FCPIdwFtftKJwwgmtpRGVo3d>
- Colab 기본 경로: `/content/drive/MyDrive/LG aimer/REF4_126`

공유 폴더가 `MyDrive`에서 보이지 않으면 Google Drive 웹에서 공유된 `LG aimer` 폴더를 **내 드라이브에 바로가기 추가**한다. 같은 이름의 새 폴더를 만들지 않는다. 바로가기 이름이나 위치가 다르면 notebook의 `DRIVE_ROOT_TEXT`만 실제 mount 경로에 맞춘다.

정식 구조는 다음과 같다.

```text
REF4_126/
├── REF4_126_T4.ipynb
├── SHA256SUMS
├── RUNBOOK.md
├── code/
│   └── REF4_126_CODE.zip
├── data/
│   ├── train.csv
│   └── trackman_history.csv
├── anchor/
│   └── strict_113A/
│       ├── oof_predictions.csv
│       ├── audit_contract.json
│       ├── preflight_report.json
│       ├── result.json
│       └── validation_report.json
├── checkpoints/                 # 실행 중 생성
├── logs/                        # 실행 중 생성
└── results/                     # 실행 완료 시 생성
```

`trackman_history.csv`는 bundle 무결성 대상이지만 현재 126 runner는 사용하지 않는다. 존재한다는 이유로 TrackMan feature를 추가하지 않는다. `reference/`와 `SHA256SUMS.base`가 보이더라도 참고 자료일 뿐 이번 실행 입력이 아니다. 정식 입력 목록은 루트의 `SHA256SUMS`만 따른다.

고정 anchor의 핵심 사실은 다음과 같다.

- 파일: `anchor/strict_113A/oof_predictions.csv`
- SHA-256: `560e1ca40a21f0b9b296f612e6764e50eaa2a6f62b08561b86dd9d1803c23aa6`
- 전체 746,504행
- 시즌별 행 수: 2022=`247,472`, 2023=`245,525`, 2024=`253,507`
- 기존 감사: 17/17, mismatch 0

단, 이 anchor는 현재 가능한 가장 정직한 비교군인 `strict p109C + fold 재학습 113A EB`이다. production 113A 전체 파이프라인의 완전한 fold parity라고 과장하면 안 된다. 이 한계는 최종 보고서의 남은 위험에 반드시 기록한다.

## 4. 실행 전 필수 점검

Gemini는 다음 순서로 점검한다. 하나라도 실패하면 상태는 `BLOCKED`이며 학습을 시작하지 않는다.

1. Colab에서 `런타임 > 런타임 유형 변경 > T4 GPU`를 선택한다.
2. `REF4_126_T4.ipynb`를 연다.
3. 동시에 같은 `results/` 또는 `checkpoints/`에 쓰는 다른 notebook이 없는지 확인한다.
4. notebook 설정을 아래와 같이 유지한다.

```text
DRIVE_ROOT_TEXT=/content/drive/MyDrive/LG aimer/REF4_126
TARGETS=2022,2023,2024
REQUIRE_T4=True
INSTALL_REQUIREMENTS=True
RESUME=True
```

5. notebook의 manifest 검증 셀이 모든 항목을 `OK`로 출력하는지 확인한다.
6. GPU 셀이 정확히 한 대의 T4를 확인하는지 확인한다.
7. 코드 ZIP이 manifest에 묶여 있고 안전하게 `/content/ref4_126_runtime`에 풀리는지 확인한다.
8. CatBoost `1.2.10`, CUDA 사용 가능 상태를 확인한다.

정식 manifest는 `SHA256SUMS.base`가 아니라 루트의 `SHA256SUMS`이며 정확히 8개 항목을 담는다. 현재 고정 code ZIP SHA-256은 `39d2176a1eb75ee5c14fe93562023b6c8dd19dfcdf5c089f59af2a480d8af891`, ZIP 내부 runner SHA-256은 `a08d76b6ba55d757010c86d167eeda65e0d76aa93d06303b81db212f3a5b7def`이다. `logs/last_launch.json`의 runner hash도 후자와 같아야 한다.

이미 `results/result.json`이 존재하면 새 실행으로 덮어쓰지 않는다. 현재 결과인지 오래된 결과인지 로컬 담당자가 확인하고 보존 경로를 정할 때까지 `BLOCKED`로 보고한다. 반면 중단된 실행의 `checkpoints/`만 존재하는 경우에는 아래 재개 감사 후 이어갈 수 있다.

manifest mismatch가 나면 파일이나 `SHA256SUMS`를 현장에서 고쳐 맞추지 않는다. 실제 값, 기대 값, 파일 경로를 기록하고 `BLOCKED`로 중단한다.

### 4.1 필수 pre-launch guard

현재 notebook은 기존 `result.json` 차단과 resume checkpoint의 실제 SHA/모델 load 검사를 자동 수행하지 않는다. 따라서 dependency/CUDA 확인 셀까지 실행한 뒤, 정식 launch 셀 **직전**에 아래 셀을 한 번 실행한다. 실패하면 파일을 삭제하거나 우회하지 말고 `BLOCKED`로 보고한다.

```python
# Run after notebook dependency validation and before the training launch cell.
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

EXPECTED_RUNNER_SHA = 'a08d76b6ba55d757010c86d167eeda65e0d76aa93d06303b81db212f3a5b7def'
EXPECTED_TRAIN_SHA = 'd2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff'
EXPECTED_ANCHOR_SHA = '560e1ca40a21f0b9b296f612e6764e50eaa2a6f62b08561b86dd9d1803c23aa6'
EXPECTED_ROWS = {2022: 247472, 2023: 245525, 2024: 253507}

def guard_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def guard_row_fingerprint(values: pd.Series) -> str:
    hashes = pd.util.hash_pandas_object(values.astype(str), index=False).to_numpy(np.uint64)
    return hashlib.sha256(hashes.tobytes()).hexdigest()

if (RESULT_DIR / 'result.json').exists():
    raise RuntimeError(
        'Existing results/result.json would be overwritten. Stop and ask the local owner '
        'to archive it or assign a fresh run root.'
    )

assert guard_sha256(RUNNER) == EXPECTED_RUNNER_SHA
assert guard_sha256(DATA_DIR / 'train.csv') == EXPECTED_TRAIN_SHA
assert guard_sha256(ANCHOR_OOF) == EXPECTED_ANCHOR_SHA

resumable = []
for year in [2022, 2023, 2024]:
    fold_dir = CHECKPOINT_DIR / str(year)
    npz_path = fold_dir / 'predictions.npz'
    meta_path = fold_dir / 'metadata.json'
    if npz_path.exists() != meta_path.exists():
        raise RuntimeError(f'Incomplete checkpoint pair: {fold_dir}')
    if npz_path.exists():
        resumable.append(year)

if resumable:
    previous_config_path = RESULT_DIR / 'run_config.json'
    if not previous_config_path.is_file():
        raise RuntimeError('Checkpoint exists but prior results/run_config.json is missing.')
    previous = json.loads(previous_config_path.read_text(encoding='utf-8'))
    assert previous['runner_sha256'] == EXPECTED_RUNNER_SHA
    assert previous['train_sha256'] == EXPECTED_TRAIN_SHA
    assert previous['anchor_provenance']['method'] == 'load_audited_117a_oof'
    assert previous['anchor_provenance']['sha256'] == EXPECTED_ANCHOR_SHA
    assert previous['requested_device'] == 'gpu' and previous['cpu_fallback'] is False
    assert previous['smoke'] is False and previous['max_rows_per_year'] is None
    assert int(previous['iterations']) == 256 and int(previous['depth']) == 6
    assert float(previous['learning_rate']) == 0.025
    assert previous['seeds'] == [17, 42, 777]
    assert previous['scales'] == [0.0, 0.025, 0.05, 0.075]

    train_order = pd.read_csv(
        DATA_DIR / 'train.csv', usecols=['row_id', 'season'], low_memory=False
    )
    train_order = train_order.loc[train_order['season'].isin([2022, 2023, 2024])]
    for year in resumable:
        fold_dir = CHECKPOINT_DIR / str(year)
        npz_path = fold_dir / 'predictions.npz'
        meta = json.loads((fold_dir / 'metadata.json').read_text(encoding='utf-8'))
        expected_order = train_order.loc[train_order['season'].eq(year), 'row_id'].reset_index(drop=True)
        assert len(expected_order) == EXPECTED_ROWS[year]
        assert meta['prediction_sha256'] == guard_sha256(npz_path)
        assert meta['config_fingerprint'] == previous['config_fingerprint']
        assert meta['row_fingerprint'] == guard_row_fingerprint(expected_order)
        with np.load(npz_path) as archive:
            correction = archive['raw_correction']
        assert len(correction) == EXPECTED_ROWS[year]
        assert np.isfinite(correction).all()
        if year == 2022:
            assert meta['warmup'] is True and meta['device_used'] == 'NONE'
            assert np.array_equal(correction, np.zeros_like(correction))
        else:
            expected_fit_years = [2022] if year == 2023 else [2022, 2023]
            assert meta['fit_years'] == expected_fit_years
            assert meta['validation_labels_used_in_fit'] is False
            assert str(meta['device_used']).upper() == 'GPU'
            model_dir = fold_dir / 'models'
            for seed in [17, 42, 777]:
                model_path = model_dir / f'r_residual_seed{seed}.cbm'
                assert model_path.is_file()
                CatBoostRegressor().load_model(str(model_path))

print({'prelaunch_guard': 'PASS', 'resumable_folds': resumable})
```

정상 최초 실행이면 `resumable_folds`는 빈 목록이다. 중단 후 재개라면 이 목록에 완성된 fold만 나타나야 한다. smoke나 다른 device에서 만든 checkpoint는 정식 run과 섞지 않는다.

## 5. 정식 실행

가장 안전한 방법은 notebook의 1~7번 실행 셀을 위에서 아래로 한 번씩 실행하는 것이다. 내부적으로 다음 명령이 실행되어야 한다.

```bash
python scripts/run_ref4_jm_r_residual_strict_gpu_126.py \
  --data-dir "/content/drive/MyDrive/LG aimer/REF4_126/data" \
  --anchor-oof "/content/drive/MyDrive/LG aimer/REF4_126/anchor/strict_113A/oof_predictions.csv" \
  --output-dir "/content/drive/MyDrive/LG aimer/REF4_126/results" \
  --checkpoint-dir "/content/drive/MyDrive/LG aimer/REF4_126/checkpoints" \
  --targets 2022,2023,2024 \
  --device gpu \
  --no-cpu-fallback \
  --resume
```

다음 옵션은 정식 실행에서 추가하면 안 된다.

- `--smoke`
- `--max-rows-per-year`
- `--device cpu` 또는 `--cpu-fallback`
- seed, scale, iteration, depth, learning-rate 변경

실행 명령, GPU 정보, 검증된 입력 hash, 로그 경로는 `logs/last_launch.json`과 타임스탬프 로그에 자동 저장된다. 화면에 보인 수치를 별도 문서에 손으로 복사해 공식 결과로 사용하지 않는다.

## 6. 연결 종료와 재개

Colab 연결이 끊기면 다음과 같이 처리한다.

1. 새 T4 runtime에 다시 연결한다.
2. 같은 Drive 경로를 mount한다.
3. `RESUME=True`를 유지한다.
4. notebook을 처음부터 순서대로 다시 실행한다.
5. runner가 checkpoint fingerprint와 행 순서를 검증한 뒤 완료 fold만 재사용하게 둔다.

금지 사항:

- checkpoint를 임의 삭제하거나 이름을 바꾸고 정상 재개라고 보고하기
- 서로 다른 코드·입력·device로 만든 checkpoint를 강제 재사용하기
- 같은 output 경로에 두 프로세스를 동시에 실행하기
- 오류를 피하려고 CPU fallback을 켜기

checkpoint hash/fingerprint 불일치, 손상, GPU 오류는 `BLOCKED`다. 수정 실험을 새로 시작하려면 로컬 담당자의 승인을 받고 별도 output 디렉터리와 새 manifest를 사용한다.

주의할 점은 runner의 resume loader가 metadata에 기록된 `prediction_sha256`을 실제 파일과 다시 비교하지 않는다는 것이다. 또한 이전 실행에서 만들어진 CPU fallback checkpoint가 남아 있을 수 있다. 재개 전에는 아래 조건을 별도로 확인한다.

- `predictions.npz`를 오류 없이 열 수 있고 `raw_correction`이 finite이며 해당 시즌 행 수와 일치
- `metadata.json`의 `prediction_sha256`과 실제 NPZ SHA-256 일치
- `row_fingerprint`와 현재 시즌별 `row_id` 순서 일치
- `config_fingerprint`와 현재 실행 계약 일치
- 2023·2024 metadata의 `device_used == "GPU"`
- 2023·2024에 각각 `r_residual_seed17.cbm`, `r_residual_seed42.cbm`, `r_residual_seed777.cbm`이 있고 CatBoost에서 load 가능
- 2022는 warmup이므로 모델이 없는 것이 정상

`last_launch.status == "RUNNING"`인 채 VM이 종료됐다면 성공도 실패도 아니다. 상태를 `INCOMPLETE`로 기록하고 위 checkpoint 검사를 통과한 뒤에만 재개한다.

## 7. 필수 산출물

정상 종료라면 최소한 아래 파일이 있어야 한다.

```text
logs/last_launch.json
logs/ref4_126_<UTC timestamp>.log
results/run_config.json
results/oof_predictions.csv
results/strict_metrics.csv
results/strict_scale_search.csv
results/result.json
results/colab_recalculation.json   # 8절 실행 후 생성
checkpoints/<fold별 prediction, metadata, model 파일>
```

`result.json`의 의미를 혼동하지 않는다.

- `status == "COMPLETE"`: 프로그램이 끝까지 실행됐다는 뜻일 뿐이다.
- `candidate_status == "PERFORMANCE_GATE_PASS"`: runner 내부 성능 gate 통과다.
- `candidate_status == "NO_PROMOTION"`: 정상 실험 결과로서 113A 유지다.
- `PERFORMANCE_GATE_PASS_UNAUDITED`: runner의 `candidate_status` 값이 아니라, `colab_recalculation.json`의 `candidate_decision`이다. runner gate와 Colab 재계산은 통과했지만 로컬 독립 감사 전인 상태다.
- 최종 `AUDIT_VERIFIED`: 별도 로컬 validator가 artifact를 회수해 독립 검증한 뒤에만 사용할 수 있다.

따라서 notebook 마지막 셀이 `COMPLETE`를 출력해도 곧바로 “126 승격” 또는 “제출 가능”이라고 보고하면 안 된다.

## 8. Gemini의 Colab 재계산 감사

학습 프로세스가 종료된 뒤, 같은 Colab에서 아래 감사 코드를 **별도 셀**로 실행한다. 이 코드는 runner가 기록한 표를 그대로 믿지 않고 OOF CSV에서 핵심 수치, bootstrap, checkpoint를 다시 계산한다. 다만 같은 실행 환경에서 수행하는 1차 재계산이므로 독립 로컬 감사의 대체물이 아니다.

```python
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path('/content/drive/MyDrive/LG aimer/REF4_126')
OUT = ROOT / 'results'
CHECKPOINTS = ROOT / 'checkpoints'
LOGS = ROOT / 'logs'

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

result = json.loads((OUT / 'result.json').read_text(encoding='utf-8'))
config = json.loads((OUT / 'run_config.json').read_text(encoding='utf-8'))
launch = json.loads((LOGS / 'last_launch.json').read_text(encoding='utf-8'))
pred = pd.read_csv(OUT / 'oof_predictions.csv', low_memory=False)
anchor_path = ROOT / 'anchor' / 'strict_113A' / 'oof_predictions.csv'
anchor = pd.read_csv(
    anchor_path,
    usecols=['row_id', 'season', 'target', 'p113a_strict'],
    low_memory=False,
)
reported = pd.read_csv(OUT / 'strict_metrics.csv')
scale_search = pd.read_csv(OUT / 'strict_scale_search.csv')

expected_counts = {2022: 247472, 2023: 245525, 2024: 253507}
assert result['experiment_id'] == 'REF4-JM-R-RESIDUAL-STRICT-GPU-126'
assert result['status'] == 'COMPLETE'
assert launch['status'] == 'COMPLETE' and int(launch['returncode']) == 0
assert launch['runner_sha256'] == 'a08d76b6ba55d757010c86d167eeda65e0d76aa93d06303b81db212f3a5b7def'
assert result['production_fit'] is False
assert result['zip_created'] is False
assert result['strict_rules']['test_read'] is False
assert config['test_read'] is False
assert config['runner_sha256'] == launch['runner_sha256']
assert config['requested_device'] == 'gpu' and config['cpu_fallback'] is False
assert config['smoke'] is False and config['max_rows_per_year'] is None
assert int(config['iterations']) == 256 and int(config['depth']) == 6
assert float(config['learning_rate']) == 0.025
assert config['seeds'] == [17, 42, 777]
assert config['scales'] == [0.0, 0.025, 0.05, 0.075]
assert int(config['bootstrap_repeats']) == 10000
assert len(launch['verified_inputs']) == 8
assert result['anchor_provenance']['method'] == 'load_audited_117a_oof'
assert result['anchor_provenance']['sha256'] == '560e1ca40a21f0b9b296f612e6764e50eaa2a6f62b08561b86dd9d1803c23aa6'
assert sha256(anchor_path) == result['anchor_provenance']['sha256']
assert pred['row_id'].astype(str).is_unique
assert pred.groupby('season').size().to_dict() == expected_counts
assert len(pred) == sum(expected_counts.values()) == result['rows']
assert set(pred['target'].dropna().unique()).issubset({0, 1})
assert np.array_equal(pred['row_id'].astype(str), anchor['row_id'].astype(str))
assert np.array_equal(pred['season'].to_numpy(int), anchor['season'].to_numpy(int))
assert np.array_equal(pred['target'].to_numpy(float), anchor['target'].to_numpy(float))
assert np.array_equal(pred['p113a_strict'].to_numpy(float), anchor['p113a_strict'].to_numpy(float))

for col in ['target', 'p113a_strict', 'raw_r_correction', 'selected_scale', 'p126']:
    assert np.isfinite(pred[col].to_numpy(float)).all(), col
assert pred['p113a_strict'].between(0, 1).all()
assert pred['p126'].between(0, 1).all()

f = pred['game_type'].eq('F')
assert np.array_equal(pred.loc[f, 'raw_r_correction'].to_numpy(float), np.zeros(int(f.sum())))
assert np.array_equal(
    pred.loc[f, 'p126'].to_numpy(float),
    pred.loc[f, 'p113a_strict'].to_numpy(float),
)
assert np.array_equal(
    pred.loc[pred['season'].isin([2022, 2023]), 'p126'].to_numpy(float),
    pred.loc[pred['season'].isin([2022, 2023]), 'p113a_strict'].to_numpy(float),
)

def brier(y, p):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))

rows = []
for year in [2022, 2023, 2024]:
    year_mask = pred['season'].eq(year)
    for group in ['ALL', 'R', 'F']:
        mask = year_mask if group == 'ALL' else year_mask & pred['game_type'].eq(group)
        y = pred.loc[mask, 'target'].to_numpy(float)
        p0 = pred.loc[mask, 'p113a_strict'].to_numpy(float)
        p1 = pred.loc[mask, 'p126'].to_numpy(float)
        rows.append({
            'year': year,
            'group': group,
            'rows': int(mask.sum()),
            'anchor_brier_recalc': brier(y, p0),
            'candidate_brier_recalc': brier(y, p1),
            'delta_brier_recalc': brier(y, p1) - brier(y, p0),
        })

recalc = pd.DataFrame(rows)
joined = reported.merge(recalc, on=['year', 'group'], validate='one_to_one')
assert len(joined) == 9
assert np.array_equal(joined['rows_x'].to_numpy(int), joined['rows_y'].to_numpy(int))
for left, right in [
    ('anchor_brier', 'anchor_brier_recalc'),
    ('candidate_brier', 'candidate_brier_recalc'),
    ('delta_brier', 'delta_brier_recalc'),
]:
    np.testing.assert_allclose(joined[left], joined[right], rtol=0, atol=5e-15)

overall = recalc.loc[recalc['group'].eq('ALL')].set_index('year')['delta_brier_recalc']
for year in [2022, 2023, 2024]:
    np.testing.assert_allclose(
        overall.loc[year], float(result['delta_brier_by_year'][str(year)]),
        rtol=0, atol=5e-15,
    )
weighted = 0.2 * overall.loc[2022] + 0.3 * overall.loc[2023] + 0.5 * overall.loc[2024]
np.testing.assert_allclose(weighted, result['time_weighted_delta_brier'], rtol=0, atol=5e-15)

assert float(result['applied_scales']['2022']) == 0.0
assert float(result['applied_scales']['2023']) == 0.0
assert float(result['applied_scales']['2024']) in {0.0, 0.025, 0.05, 0.075}
assert set(scale_search['scale'].astype(float).unique()).issubset({0.0, 0.025, 0.05, 0.075})
search_2024 = scale_search.loc[scale_search['applied_to'].astype(str).eq('2024')].copy()
selection_text = {
    value.replace('.0', '')
    for value in search_2024['selection_years'].dropna().astype(str)
}
assert selection_text == {'2023'}
expected_scale_2024 = float(search_2024.sort_values(['objective', 'scale']).iloc[0]['scale'])
assert float(result['applied_scales']['2024']) == expected_scale_2024

def row_fingerprint(values):
    hashes = pd.util.hash_pandas_object(values.astype(str), index=False).to_numpy(np.uint64)
    return hashlib.sha256(hashes.tobytes()).hexdigest()

from catboost import CatBoostRegressor
for year in [2022, 2023, 2024]:
    fold_dir = CHECKPOINTS / str(year)
    npz_path = fold_dir / 'predictions.npz'
    meta_path = fold_dir / 'metadata.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    assert meta['config_fingerprint'] == result['config_fingerprint'] == config['config_fingerprint']
    assert meta['prediction_sha256'] == sha256(npz_path)
    year_rows = pred.loc[pred['season'].eq(year)].reset_index(drop=True)
    assert meta['row_fingerprint'] == row_fingerprint(year_rows['row_id'])
    with np.load(npz_path) as archive:
        correction = archive['raw_correction']
    assert len(correction) == expected_counts[year]
    assert np.isfinite(correction).all()
    if year == 2022:
        assert meta['warmup'] is True and meta['device_used'] == 'NONE'
        assert np.array_equal(correction, np.zeros_like(correction))
    else:
        assert str(meta['device_used']).upper() == 'GPU'
        assert meta['validation_labels_used_in_fit'] is False
        model_dir = fold_dir / 'models'
        expected_models = [model_dir / f'r_residual_seed{seed}.cbm' for seed in [17, 42, 777]]
        assert all(path.is_file() for path in expected_models)
        for model_path in expected_models:
            CatBoostRegressor().load_model(str(model_path))

for year in ['2023', '2024']:
    meta = result['devices'][year]
    assert str(meta['device_used']).upper() == 'GPU', (year, meta['device_used'])
    assert meta['validation_labels_used_in_fit'] is False
assert result['devices']['2023']['fit_years'] == [2022]
assert result['devices']['2024']['fit_years'] == [2022, 2023]
assert result['strict_rules']['target_year_labels_used_in_fit'] is False
assert result['strict_rules']['scale_selected_from_strictly_prior_correction_folds'] is True

# 2024 전체 행(F의 0 contribution 포함)을 pitcher cluster로 다시 bootstrap한다.
latest = pred.loc[pred['season'].eq(2024)].reset_index(drop=True)
row_delta = (
    (latest['p126'].to_numpy(float) - latest['target'].to_numpy(float)) ** 2
    - (latest['p113a_strict'].to_numpy(float) - latest['target'].to_numpy(float)) ** 2
)
grouped = pd.DataFrame({
    'pitcher': latest['pitcher_id'].astype(str),
    'delta': row_delta,
}).groupby('pitcher', sort=False)['delta'].agg(['sum', 'size'])
sums = grouped['sum'].to_numpy(float)
sizes = grouped['size'].to_numpy(float)
repeats = int(result['pitcher_cluster_bootstrap_2024']['repeats'])
rng = np.random.default_rng(1262024)
boot_values = np.empty(repeats, dtype=float)
for start in range(0, repeats, 64):
    count = min(64, repeats - start)
    sample = rng.integers(0, len(grouped), size=(count, len(grouped)))
    boot_values[start:start + count] = sums[sample].sum(axis=1) / sizes[sample].sum(axis=1)
bootstrap_recalc = {
    'repeats': repeats,
    'pitcher_clusters': int(len(grouped)),
    'delta_brier': float(row_delta.mean()),
    'bootstrap_mean_delta_brier': float(boot_values.mean()),
    'ci_low': float(np.quantile(boot_values, 0.025)),
    'ci_high': float(np.quantile(boot_values, 0.975)),
    'improvement_probability': float(np.mean(boot_values < 0.0)),
}
bootstrap = result['pitcher_cluster_bootstrap_2024']
for key, value in bootstrap_recalc.items():
    if isinstance(value, int):
        assert int(bootstrap[key]) == value
    else:
        np.testing.assert_allclose(float(bootstrap[key]), value, rtol=0, atol=5e-15)

# Runner 내부 gate를 값에서 다시 판정한다.
internal_gates = {
    'f_exact_identity': True,
    'latest_delta_brier_negative': overall.loc[2024] < 0.0,
    'time_weighted_delta_negative': weighted < 0.0,
    'latest_bootstrap_ci_high_below_zero': float(bootstrap['ci_high']) < 0.0,
    'nonzero_strict_scale_selected': float(result['applied_scales']['2024']) > 0.0,
}
assert internal_gates == result['gate_results']
assert bool(all(internal_gates.values())) == bool(result['performance_gate_pass'])

# 대폭 개선을 주장하기 위한 외부 승격 gate. 내부 PASS보다 엄격하다.
external_promotion = {
    '2024_delta_at_most_minus_1e_4': overall.loc[2024] <= -0.00010,
    'no_season_worse_than_plus_5e_5': overall.max() <= 0.00005,
    'time_weighted_improvement': weighted < 0.0,
    'pitcher_bootstrap_ci_high_below_zero': float(bootstrap['ci_high']) < 0.0,
    'f_exact_identity': True,
    'gpu_only': all(str(result['devices'][y]['device_used']).upper() == 'GPU' for y in ['2023', '2024']),
}

candidate_decision = (
    'PERFORMANCE_GATE_PASS_UNAUDITED'
    if all(external_promotion.values()) else 'NO_PROMOTION'
)
artifact_paths = [
    OUT / 'run_config.json', OUT / 'result.json', OUT / 'oof_predictions.csv',
    OUT / 'strict_metrics.csv', OUT / 'strict_scale_search.csv',
    LOGS / 'last_launch.json', Path(launch['log']),
] + sorted(path for path in CHECKPOINTS.rglob('*') if path.is_file())
assert all(path.is_file() for path in artifact_paths)
artifact_inventory = [
    {
        'path': str(path.relative_to(ROOT)),
        'bytes': path.stat().st_size,
        'sha256': sha256(path),
    }
    for path in artifact_paths
]
audit = {
    'audit_status': 'PASS',
    'candidate_decision': candidate_decision,
    'local_independent_audit_required': True,
    'result_sha256': sha256(OUT / 'result.json'),
    'oof_sha256': sha256(OUT / 'oof_predictions.csv'),
    'metrics_sha256': sha256(OUT / 'strict_metrics.csv'),
    'scale_search_sha256': sha256(OUT / 'strict_scale_search.csv'),
    'recalculated_overall_delta_brier': {str(k): float(v) for k, v in overall.items()},
    'recalculated_time_weighted_delta_brier': float(weighted),
    'internal_gates': internal_gates,
    'external_promotion_gates': external_promotion,
    'bootstrap_2024_recalculated': bootstrap_recalc,
    'artifact_inventory': artifact_inventory,
}
(OUT / 'colab_recalculation.json').write_text(
    json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
)
print(json.dumps(audit, ensure_ascii=False, indent=2))
```

이 감사 셀이 assertion 없이 끝난 것만으로 모델이 자동 승격되는 것은 아니다. 가능한 최고 결론은 `PERFORMANCE_GATE_PASS_UNAUDITED`다. 내부 gate는 통과했지만 외부 승격 gate가 실패하면 결론은 `NO_PROMOTION`이며 113A를 유지한다. 현재 bundle에는 별도 126 validator, audit manifest, attestation이 없으므로 Gemini가 `AUDIT_VERIFIED`라고 선언하는 것은 금지한다.

## 9. 판정 상태

Gemini는 감사 진행 상태와 후보 결정을 분리한다.

- 감사 진행 상태: `PASS`, `FAIL`, `INCOMPLETE`, `BLOCKED` 중 하나
- 후보 결정: `PERFORMANCE_GATE_PASS_UNAUDITED` 또는 `NO_PROMOTION`
- `AUDIT_VERIFIED`는 로컬 독립 validator 완료 전 사용 금지

`FAIL`은 실행은 끝났지만 무결성/재계산 assertion이 실패한 경우다. 성능 gate만 미달하고 무결성은 정상이라면 감사 상태는 `PASS`, 후보 결정은 `NO_PROMOTION`이다. `제출 가능`, `대폭 개선` 같은 표현을 근거 없이 쓰거나 로컬 성능을 DACON 예상 점수로 임의 환산하지 않는다.

## 10. Gemini 최종 보고 형식

최종 응답은 다음 순서로 작성한다. 모든 수치와 hash는 생성된 JSON/CSV에서 프로그램으로 읽은 값만 사용한다.

1. **상태**: 감사 진행 상태와 후보 결정을 각각 표시
2. **실행 사실**: GPU 이름, 명령, return code, runtime, resume 사용 여부
3. **데이터 무결성**: 검증 파일 수, 입력 SHA 검증 결과, OOF 전체/시즌별 행 수
4. **strict 증거**: 2023/2024 fit years, validation label 미사용, scale selection years
5. **재계산 결과**: 시즌별 `delta_brier`, 시간가중 delta, 2024 전체행 pitcher-cluster bootstrap CI
6. **gate 표**: runner 내부 gate와 외부 승격 gate를 분리
7. **artifact**: 각 결과 파일 경로, byte 수, SHA-256
8. **변경 사항**: Colab이 새로 만든 파일만 열거
9. **미검증 항목과 남은 위험**: 로컬 독립 감사 미완료 및 anchor parity 한계를 항상 명시
10. **권고**: `113A 유지` 또는 `로컬 독립 감사 대상으로 126 전달`

Gemini는 제출 ZIP을 만들거나 126을 곧바로 제출하라고 권고하지 않는다. `PERFORMANCE_GATE_PASS_UNAUDITED`인 경우 로컬 담당자가 Drive artifact를 회수해 독립 검증한 뒤 production 작업을 별도로 승인한다.

남은 재현성 위험도 숨기지 않는다. CatBoost GPU 학습은 floating-point reduction 순서 때문에 bit-exact 결정성을 보장하지 않는다. 현재 bundle은 CatBoost `1.2.10`만 고정하고, `run_config.json`에는 Python/NumPy/Pandas 버전만 기록한다. PyTorch/CUDA 버전은 notebook 셀 출력에만 보이므로 별도 artifact가 없으면 보존됐다고 주장하지 않는다. 따라서 동일 seed라는 이유만으로 다른 runtime의 예측이 bitwise 동일하다고 주장하지 않는다.

## 11. 로컬 결과 회수

이 단계는 Gemini가 Colab에서 실행하지 않는다. 로컬 담당자가 현재 구성된 rclone remote를 이용해 결과를 **비어 있는 타임스탬프 디렉터리**로 복사한다. 아래 `<UTC_TIMESTAMP>`는 실제 회수 시각으로 바꾼다.

```bash
rclone copy \
  "lgaimer126_gdrive:REF4_126/results" \
  "colab/126/retrieved/<UTC_TIMESTAMP>/results" \
  --checksum --immutable --check-first --transfers 2 --checkers 4 --progress

rclone copy \
  "lgaimer126_gdrive:REF4_126/logs" \
  "colab/126/retrieved/<UTC_TIMESTAMP>/logs" \
  --checksum --immutable --check-first --transfers 2 --checkers 4 --progress

rclone copy \
  "lgaimer126_gdrive:REF4_126/checkpoints" \
  "colab/126/retrieved/<UTC_TIMESTAMP>/checkpoints" \
  --checksum --immutable --check-first --transfers 2 --checkers 4 --progress

rclone check \
  "lgaimer126_gdrive:REF4_126/results" \
  "colab/126/retrieved/<UTC_TIMESTAMP>/results" \
  --checksum

rclone check \
  "lgaimer126_gdrive:REF4_126/checkpoints" \
  "colab/126/retrieved/<UTC_TIMESTAMP>/checkpoints" \
  --checksum
```

회수 후 `colab_recalculation.json`의 `artifact_inventory`와 로컬 파일의 byte 수/SHA-256을 다시 대조한다. `sync`, `move`, `delete`, `purge`는 사용하지 않는다. Google API `403 rateLimitExceeded`가 일시 발생하면 파일이나 인증을 바꾸지 말고 잠시 뒤 동일한 비파괴 명령을 재시도한다. rclone token이나 OAuth secret은 notebook, Drive, 로그, 저장소에 기록하지 않는다.

## 12. 즉시 중단 조건

다음 중 하나라도 발생하면 수정 실험을 계속하지 말고 증거를 보존한 채 보고한다.

- manifest SHA-256 불일치
- T4가 아닌 runtime 또는 CUDA 미인식
- CPU fallback 발생
- train/anchor row_id 중복, 시즌별 행 수 불일치
- target 비이진 또는 예측값 non-finite/[0,1] 범위 이탈
- F prediction exact identity 실패
- validation 시즌 label이 해당 fold 학습에 포함됨
- 2024 scale이 2024 결과를 보고 선택됨
- 두 writer가 동일 checkpoint/output을 사용함
- 결과 JSON과 OOF 재계산 수치 불일치
- `test.csv` 접근 또는 제출 ZIP 생성 흔적

## 13. Gemini에게 전달할 시작 지시문

아래 문장을 그대로 Gemini에게 전달한다.

```text
워크스페이스 루트의 colab.md를 최우선 실행 계약으로 읽고 REF4-JM-R-RESIDUAL-STRICT-GPU-126만 수행하세요. Google Drive의 REF4_126_T4.ipynb를 T4 runtime에서 셀 순서대로 실행하되, dependency/CUDA 확인 뒤 학습 launch 전에 colab.md 4.1절 pre-launch guard를 반드시 별도 셀로 실행하세요. 입력 SHA-256을 모두 검증하고 --device gpu --no-cpu-fallback --resume 조건을 유지하세요. 모델·feature·seed·scale·분할을 변경하거나 test.csv/제출 ZIP을 만들지 마세요. 종료 후 colab.md 8절의 Colab 재계산 감사를 실행하고, COMPLETE와 PERFORMANCE_GATE_PASS_UNAUDITED 및 NO_PROMOTION을 구분하세요. 로컬 독립 validator 전에는 AUDIT_VERIFIED를 선언하지 마세요. 수치를 손으로 복사하지 말고 result.json, CSV, colab_recalculation.json에서 프로그램으로 읽어 정해진 보고 형식으로만 결과를 제출하세요. 문제가 생기면 임의 우회하지 말고 INCOMPLETE 또는 BLOCKED로 증거와 함께 보고하세요.
```
