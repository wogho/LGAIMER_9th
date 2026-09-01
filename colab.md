# LG AIMer Colab L4 실행·감사 계약

이 문서는 앞으로의 고비용 학습과 strict-forward OOF 생성을 Google Colab **NVIDIA L4**에서 수행하기 위한 운영 계약이다. Gemini가 실행을 맡더라도 모델 선택, 결과 판정, 제출 승격은 이 문서의 증거 규칙을 따라야 한다.

기존 126 T4 실행 계약과 결과는 `colab_gpu_126.md` 및 Drive의 기존 `REF4_126` 경로에 보존한다. L4 작업은 그 경로와 checkpoint를 덮어쓰거나 재사용하지 않는다.

## 1. 규정 판단과 운영 방침

- 공개 대회 규정에는 GPU 사용 금지 조항이 없다.
- 공식 평가 환경에는 CPU 6 vCPU와 함께 NVIDIA L4, CUDA 12.8이 명시되어 있다.
- 따라서 L4 사용은 규정 회피가 아니라 공식 환경과 정합적인 학습·검증 수단이다.
- 다만 제출 ZIP은 공식 시간·메모리·네트워크·행 독립성 제한을 별도로 통과해야 한다. Colab에서 빠르게 학습됐다는 사실은 제출 가능성을 보장하지 않는다.
- 앞으로 GPU가 유효한 CatBoost/FiLM/다중 seed·다중 fold 작업은 L4를 우선 사용한다. ExtraTrees처럼 GPU를 사용하지 않는 작업은 L4 시간을 소비하지 않는다.

공식 근거:

- 평가 환경·시간 제한: <https://www.dacon.io/competitions/official/236743/overview/evaluation>
- 데이터·외부 통신·행 독립성 규칙: <https://www.dacon.io/competitions/official/236743/overview/rules>

## 2. Strict-Forward OOF 운영 원칙

1. `smoke_gpu`로 경로·패키지·GPU·메모리 오류만 확인한다.
2. `validated_full_gpu`로 첫 strict F-regime 연구 baseline을 만든다.
3. shared 채널까지 완전히 fold별 재학습할 가치가 확인된 경우에만 `six_seed_full_gpu`를 실행한다.
4. 연도별 checkpoint를 저장하고 동일 profile·동일 설정에서만 재개한다.
5. `p_model_only`와 global shift가 포함된 `p_deployment`를 분리해 평가한다.
6. 구조·실행 전략·checkpoint 설계를 자체 구현하며, 후보는 현재 최고 기준 모델과 production-faithful strict OOF 위에서 독립 검증한다.

## 3. L4 작업 식별과 Drive 분리

각 새 실험은 실행 전에 하나의 불변 `EXPERIMENT_ID`를 부여한다. 예:

```text
REF4-<HYPOTHESIS>-L4-<NUMBER>
```

Drive 구조는 실험별로 분리한다.

```text
/content/drive/MyDrive/LG aimer/L4_EXPERIMENTS/<EXPERIMENT_ID>/
├── input/
│   ├── data/
│   ├── anchor/
│   └── code/
├── checkpoints/
├── logs/
├── results/
├── manifest/
│   ├── SHA256SUMS.input
│   └── run_contract.json
└── audit/
```

금지 사항:

- 기존 `/content/drive/MyDrive/LG aimer/REF4_126`에 새 결과 저장
- T4 checkpoint를 L4 실행에서 재사용
- 다른 실험의 `results/` 또는 `checkpoints/`를 같은 경로에 혼합
- `sync`, `move`, `delete`, `purge`로 원격 원본 정리
- 결과를 본 뒤 기존 파일을 같은 이름으로 덮어쓰기

## 4. L4 런타임 필수 확인

모든 실행은 첫 셀에서 다음을 기록한다.

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python --version
python - <<'PY'
import os, platform
print('platform=', platform.platform())
print('cpu_count=', os.cpu_count())
try:
    import catboost
    print('catboost=', catboost.__version__)
except Exception as exc:
    print('catboost_error=', repr(exc))
try:
    import torch
    print('torch=', torch.__version__)
    print('cuda_available=', torch.cuda.is_available())
    print('cuda_device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
except Exception as exc:
    print('torch_error=', repr(exc))
PY
```

필수 조건:

- GPU 이름에 `L4`가 포함되어야 한다.
- `torch.cuda.is_available()`가 `True`여야 한다.
- CatBoost GPU 1회 최소 fit smoke가 성공해야 한다.
- 실제 런타임 버전과 설치 명령을 `logs/environment.txt`에 저장해야 한다.
- L4가 아닌 런타임이면 full 학습을 시작하지 않고 `BLOCKED_WRONG_ACCELERATOR`로 종료한다.
- CPU fallback은 금지한다. GPU 실패를 CPU 성공으로 숨기지 않는다.

## 5. 실행 전 불변 계약

`manifest/run_contract.json`에는 적어도 다음을 결과 확인 전에 기록한다.

```json
{
  "experiment_id": "REF4-<HYPOTHESIS>-L4-<NUMBER>",
  "device_required": "NVIDIA L4",
  "profile": "validated_full_gpu",
  "target_years": [2022, 2023, 2024],
  "seeds": [],
  "model_config": {},
  "selection_years": [],
  "final_holdout_year": 2024,
  "test_read": false,
  "production_fit": false,
  "zip_created": false,
  "promotion_gates": {}
}
```

다음 파일을 `SHA256SUMS.input`에 고정한다.

- 학습 데이터와 필요한 TrackMan 파일
- frozen anchor OOF와 provenance 파일
- runner와 notebook
- code ZIP
- feature/config 파일

SHA 항목 수, 실제 파일 수, 검증 성공 수가 정확히 일치하기 전에는 실행하지 않는다. 해시는 보고서에 손으로 복사하지 않고 명령 결과를 저장한다.

## 6. 기본 실행 순서

### 단계 A — L4 smoke

- 목표 시즌 한 개만 사용한다.
- seed·iteration·epoch를 축소한다.
- 데이터 로드, 특징 생성, GPU fit, checkpoint 저장, resume, 결과 schema만 검사한다.
- smoke 수치는 성능 비교나 후보 승격에 사용하지 않는다.
- smoke 출력은 `results_smoke/`, checkpoint는 `checkpoints_smoke/`로 분리한다.

### 단계 B — 첫 정식 검증

1Physics-Trajectory Baseline 계열을 직접 재현하는 경우 첫 정식 profile은 `validated_full_gpu`다.

- 2022: 2022보다 과거만 학습
- 2023: 2023보다 과거만 학습
- 2024: 2024보다 과거만 학습
- target-season label을 fit, feature lookup, calibration, gate 선택에 사용하지 않는다.
- 검증 시즌별 전체/R/F 지표와 pooled 지표를 함께 저장한다.
- global shift 포함/제외 예측을 별도 열로 저장한다.

### 단계 C — 완전 strict 재학습

`six_seed_full_gpu`는 자동 후속 단계가 아니다. 아래가 모두 참일 때만 실행한다.

- `validated_full_gpu`가 사전 고정 gate를 통과했다.
- shared OOF 재사용이 결론을 바꿀 실질적 가능성이 있다.
- 남은 대회 시간과 L4 세션 시간이 충분하다.
- 예상 산출물과 checkpoint 용량이 Drive 여유 공간 안에 있다.

실행한다면 새 profile 경로를 사용하고 validated checkpoint를 섞지 않는다.

## 7. 엄격한 모델 선택 규칙

- 2024 결과를 보고 feature, seed, scale, blend weight, clipping 값을 바꾸지 않는다.
- 후보가 여러 개라면 후보 집합과 개수를 실행 전에 고정한다.
- 설정 선택은 2023 이전 정보와 2023 strict fold까지만 사용한다.
- 선택된 하나의 설정만 2024에 적용해 최종 OOT 성능을 계산한다.
- 2024를 여러 번 보고 가장 좋은 후보를 고르면 2024는 더 이상 holdout이 아니다.
- `p_deployment`의 global shift가 OOF를 악화시키더라도 model-only 성능과 혼합해 보고하지 않는다.
- 기존 최고 기준 모델은 byte/hash로 고정하며, 후보 실패 시 그 모델로 exact fallback한다.

기본 승격 gate는 실험별 계약에서 수치까지 고정하되 최소한 다음을 포함한다.

- 2024 Brier 개선
- 시간 가중 Brier 개선
- worst-season 악화 한도
- R/F 중 한쪽의 큰 손실 금지
- pitcher-cluster bootstrap의 2024 CI 상한 < 0
- F 동결 실험이면 F bit-exact identity
- 행 단독·배치·열 순열 예측 일치
- 공식 전체 행 환산 추론 시간 < 600초
- 메모리 < 28GB

## 8. checkpoint와 재개

연도별 checkpoint에는 다음을 저장한다.

- validation row ID/order fingerprint
- fit row ID, target, feature fingerprint
- target year와 허용 학습 시즌
- model config fingerprint
- device가 실제 `GPU`였다는 기록과 GPU 이름
- prediction NPZ SHA-256
- 각 모델 파일의 path, size, SHA-256
- runner SHA-256

재개 전에 NPZ를 실제로 열어 finite·shape·SHA를 재검산하고 모든 모델을 load한다. metadata에 적힌 SHA를 신뢰만 해서는 안 된다. 손상 checkpoint는 삭제하지 말고 `<year>.corrupt.<UTC>`로 격리한다.

동일 checkpoint 경로에 writer를 두 개 실행하지 않는다. 설정, runner, target year, 입력 hash, L4/CPU 장치가 하나라도 바뀌면 새 실행으로 취급한다.

## 9. 완료 판정과 회수

프로세스 종료 성공과 성능 승격을 구분한다.

- `COMPLETE`: 명령이 정상 종료되고 필수 파일이 생성됨
- `PERFORMANCE_GATE_PASS`: 사전 성능 gate 통과
- `PERFORMANCE_GATE_PASS_UNAUDITED`: 성능은 통과했으나 독립 감사 전
- `AUDIT_VERIFIED_VALIDATION_ONLY`: 독립 검증까지 통과했으나 production/ZIP 미구축
- `PRODUCTION_VERIFIED`: production fit과 전체 제출 감사를 통과
- `NO_PROMOTION`: gate 실패, 기준 모델 유지
- `BLOCKED_*`: 입력·장치·환경·무결성 문제로 실행 중단

`COMPLETE`만으로 후보를 제출하지 않는다. 결과 회수 시 다음을 모두 가져온다.

- 실행 command와 전체 timestamp log
- environment 정보
- run contract와 입력 manifest
- config/result/metrics/OOF/scale-search 파일
- checkpoint metadata·predictions·model 파일
- output 전체 SHA inventory

로컬에서는 fresh timestamp 디렉터리로 `rclone copy`한 뒤 `rclone check --checksum --one-way`를 실행한다. JSON의 자기보고 수치를 복사하지 않고 OOF와 원천 데이터에서 Brier/BSS/bootstrap/gate를 독립 재계산한다.

## 10. production 전환

strict OOF가 통과해도 바로 ZIP을 만들지 않는다.

1. strict anchor와 실제 production anchor의 feature·prediction geometry를 비교한다.
2. 호환성 gate를 결과 전에 고정한다.
3. 통과한 strict OOF 행만으로 최종 residual/expert를 학습한다.
4. 기준 ZIP을 byte-copy하고 최소 변경만 적용한다.
5. 미해당·unknown·F 동결 행은 기준 모델과 exact fallback한다.
6. 단일행, unrelated-row, 열 순열, 반복 실행 불변성을 검증한다.
7. 공식 전체 테스트 행 규모에서 L4 추론 시간과 CPU/RAM을 함께 측정한다.
8. 독립 validator, manifest, attestation을 생성한 뒤에만 제출 후보로 승격한다.

## 11. 다음 작업 지시문

Gemini에게는 다음 내용을 그대로 전달한다.

> 다음 실험은 Colab NVIDIA L4 전용으로 구현한다. 공개 규정상 GPU는 금지되지 않으며 공식 평가 환경에도 L4가 명시되어 있다. 먼저 단일 가설, 후보 수, 2023 선택 규칙, 2024 최종 holdout gate를 `run_contract.json`에 고정한다. 기존 `REF4_126` T4 결과와 checkpoint는 읽기 전용으로 보존하고 새 `L4_EXPERIMENTS/<EXPERIMENT_ID>` 경로를 사용한다. L4 확인, 입력 SHA 100% 일치, smoke 완료 뒤에만 정식 실행한다. CPU fallback, test 읽기, 결과 후 튜닝, production fit, ZIP 생성은 검증 단계에서 금지한다. 종료 후 모든 원천 artifact를 회수해 로컬 독립 감사 전까지 최대 상태를 `PERFORMANCE_GATE_PASS_UNAUDITED`로 보고한다.

## 12. 현재 상태

- Colab L4: 구매 완료(사용자 확인)
- 기존 126 T4 결과: 보존, 별도 감사 기록 유지
- 차기 실험: `REF4-ANCHOR-INVARIANT-R-RESIDUAL-L4-127`
- 단일 변경: 126 residual expert 입력에서 `base_prediction` 제거
- 결합: frozen 113A와 bounded auxiliary probability의 볼록결합
- 선택: 2023 strict correction fold만 사용
- 최종 검증: 2024를 한 번만 평가
- runner: `scripts/run_ref4_anchor_invariant_r_residual_l4_127.py`
- notebook: `colab/REF4_127_L4.ipynb`
- Drive root: `/content/drive/MyDrive/LG aimer/L4_EXPERIMENTS/REF4-ANCHOR-INVARIANT-R-RESIDUAL-L4-127`
- 따라서 현재 상태: `READY_FOR_L4_UPLOAD_AND_SMOKE`

입력 manifest 5/5 검증과 L4 smoke가 끝나기 전에는 full run을 시작하지 않는다.
