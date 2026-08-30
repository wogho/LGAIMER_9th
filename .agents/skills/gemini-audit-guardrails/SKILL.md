---
name: gemini-audit-guardrails
description: Gemini 작업에서 검증 누락, 수동 수치 복사, 후보 수 불일치, 조잡한 PASS 판정을 방지하는 증거 기반 실행·감사·보고 규칙.
---

# Gemini 전용 작업 스킬 — 증거 기반 실행·검증·보고

이 문서는 `/home/ubuntu/orca/workspaces/LG AIMer/Infra-setup`에서 Gemini가 작업할 때 반드시 적용한다. 목적은 검증 누락, 수동 수치 복사, 후보 수 불일치, 조잡한 PASS 판정을 재발시키지 않는 것이다.

## 1. 최우선 규칙

1. **자기보고를 증거로 사용하지 않는다.** JSON의 `PASS`, Markdown의 `ALL PASS`, 이전 에이전트의 설명은 검증 대상일 뿐 근거가 아니다.
2. **구현되지 않은 검사는 PASS가 아니다.** 결과 객체·assertion·원천 파일 재계산이 없으면 상태는 반드시 `INCOMPLETE`다.
3. **실험보다 검증이 먼저다.** 감사가 `AUDIT_VERIFIED`가 아니면 신규 학습, 가중치/threshold 탐색, robustness, ZIP 생성, test 추론, 제출을 모두 중단한다.
4. **수동 입력을 금지한다.** 후보 수, 점수, gate, 파일 수, 해시는 코드가 산출하고 문서는 결과 객체에서 읽어야 한다.
5. **모호한 성공 보고를 금지한다.** 실패·미검증·부분 검증을 PASS처럼 표현하지 않는다.

## 2. 작업 시작 전 체크

작업을 시작하기 전에 다음을 실제로 수행하고 로그에 명령과 결과를 남긴다.

- 현재 작업 디렉터리와 Git 변경 상태 확인
- `start02_dev_log.md`, `01_제약과금지사항.md`, `08_Gemini_작업위임서.md`의 최신 지시 확인
- 목표 점수와 기준 제출물(SUB-001/SUB-002) 및 rollback ZIP 확인
- 새 작업의 실험 ID, 단일 가설, 변경 파일 목록, 성공/중단 기준을 먼저 선언
- 기존 파일을 덮어쓰지 않고 새 실험 디렉터리와 새 ZIP을 사용

사전 조건 하나라도 확인할 수 없으면 작업을 시작하지 말고 `BLOCKED`로 보고한다.

## 3. 원천 데이터 독립 검증 계약

검증기는 보고서의 상태 문자열을 읽지 말고 원천 파일을 다시 읽어 아래를 계산한다.

- 모든 입력 CSV의 행 수와 `row_id` 유일성
- 보조 CSV를 포함한 exact `row_id` 집합 일치
- merge 전후 행 수 보존 및 paired 정렬
- 모든 target의 finite·binary 여부
- 모든 원천·중간·최종 prediction의 finite 및 `[0,1]` 범위
- 모든 후보 가중치 합, threshold 범위
- calibration 학습 시즌 `<` 검증 시즌 여부
- 모델 metadata, 스크립트, 입력, 출력 JSON/Markdown, 보존 ZIP의 SHA-256

각 검사는 `checked`, 실제 수치, 실패 목록을 결과 JSON에 기록한다. 빈 `{}`나 요약 문자열만으로 PASS를 만들지 않는다.

## 4. 후보·표·gate 검증

중첩 구조는 재귀적으로 leaf candidate로 펼친다. 다음 필드를 모든 leaf에 대해 JSON과 Markdown 양쪽에서 비교한다.

`candidate_name`, `candidate_status`, 2022/2023/2024 delta, 2024 BSS, local CV proxy, 시간가중 delta, worst-season delta, gate 결과

CAL 등 중첩 후보는 `base_candidate/calibrator` 복합 ID를 사용한다. 누락·이름 충돌·표시 정밀도 불일치·부호 불일치가 하나라도 있으면 전체 정합성은 `AUDIT_FAIL_REPORT`다.

gate는 반올림값이 아니라 full precision으로 독립 계산한다.

```text
expected_gate = (
    delta_2024_vs_sub002 <= -0.000100000000
    and worst_season_delta_vs_sub002 <= 0.000050000000
)
```

`expected_gate`와 기록값을 후보별로 저장하고, `gate_checks_count == actual_leaf_count`를 assertion으로 강제한다. 후보 수를 사람이 세어 입력하지 않는다.

## 5. 감사 산출물 생성 순서

순환 해시를 만들지 않도록 순서를 고정한다.

1. 실행기에서 코드·입력·metadata·출력·ZIP을 수집해 `audit_manifest.json` 생성
2. 독립 검증기가 manifest와 원천 파일을 읽어 `validation_report.json` 생성
3. manifest/report/validator SHA-256과 모든 count·PASS/FAIL 수를 `audit_attestation.json`에 기록
4. 문서에는 attestation에서 읽은 값만 기록
5. 생성 직후 별도 명령으로 해시와 count를 재검증

attestation 자체는 순환 해시를 피하기 위해 manifest에 넣지 않을 수 있으나, manifest SHA·report SHA·validator SHA를 반드시 상호 검증한다.

## 6. 실패 상태와 즉시 중단

- `AUDIT_FAIL_HASH`: 파일 해시 불일치
- `AUDIT_FAIL_DATA`: 원천 배열·row_id·target·prediction 실패
- `AUDIT_FAIL_REPORT`: JSON↔Markdown 누락·수치·이름·상태 불일치
- `AUDIT_FAIL_COUNT`: leaf/gate/checked count 불일치
- `AUDIT_FAIL_PROVENANCE`: 환경·manifest·attestation 불일치
- `AUDIT_INCOMPLETE`: 검증 코드 또는 증거가 일부라도 없음
- `BLOCKED`: 필요한 파일·권한·규칙 확인이 불가능함

위 상태에서는 후보를 제출 승인으로 승격하지 않는다. 원인과 재현 명령을 기록하고 사용자 지시를 기다린다.

## 7. 보고서 형식

모든 답변은 다음 순서로 짧고 검증 가능하게 작성한다.

1. 상태: `PASS`, `FAIL`, `INCOMPLETE`, `BLOCKED` 중 하나
2. 실제 확인 수치: 파일 수, leaf 수, checked 수, mismatch 수, gate 수
3. 실행한 명령과 핵심 출력
4. 변경 파일과 SHA-256
5. 미검증 항목과 남은 위험
6. 다음 작업 또는 중단 사유

`AUDIT_VERIFIED`는 아래 조건을 모두 만족할 때만 사용할 수 있다.

- 모든 필수 검사가 코드로 구현됨
- 원천 데이터 재계산 PASS
- 모든 leaf의 모든 필드 비교 PASS
- hash/count/gate/attestation 상호 검증 PASS
- 미검증 항목 0건

## 8. 재발 방지 규칙

- 검증기를 수정한 뒤에는 반드시 validator 자체의 SHA-256을 갱신하고 재실행한다.
- 이전 결과를 복사해 새 실험 결과처럼 보고하지 않는다.
- 점수 향상이 없거나 gate 실패면 조기 중단하고, 실패 원인을 분석하지 않은 채 새 가중치·threshold를 추가하지 않는다.
- 사용자가 “검증”을 요청하면 파일을 실제로 읽고 독립 명령을 실행한 뒤 답한다. 추정·기억·자기보고만으로 결론을 내리지 않는다.
- 규칙 위반 가능성, 외부 데이터/API 사용, 평가 데이터 행 간 정보 사용이 의심되면 즉시 중단하고 보고한다.

> **최종 경고:** 한 필드라도 검사하지 않았거나, 한 후보라도 세지 않았거나, 한 숫자라도 수동 입력했으면 `PASS`가 아니라 `INCOMPLETE`다. 증거가 완전할 때만 `AUDIT_VERIFIED`라고 보고하라.
