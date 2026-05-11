# Phase 1b v2.2.5 Runbook

## 0. 최종 기준 파일

최종 사람이 확인할 shortlist:

    data/review/phase1b/v2_2_5/shortlist_global_dedupe/index.html

최종 CSV:

    data/review/phase1b/v2_2_5/shortlist_global_dedupe/recommendation_shortlist_v2_2_5_global_dedupe.csv

최종 모델:

    models/phase1b_smoke/classifier_smoke_model_v2_2_5.joblib

최종 score snapshot:

    data/retrieval/phase1b/v2_2_5/candidate_score_snapshot_v2_2_5.jsonl

최종 feature snapshot:

    data/feature_snapshots/v2_2_5/phase1b_duplicate_canonicalized/

최종 training snapshot:

    data/review/phase1b/v2_2_5/training_snapshot_classifier_v2_2_5.jsonl
    data/review/phase1b/v2_2_5/training_snapshot_ranker_v2_2_5.jsonl

최종 duplicate canonical map:

    data/ontology/duplicate_canonical_map_v2_2_5.jsonl

---

## 1. v2.2.5 현재 상태

v2.2.5의 목적은 모델 성능 향상이 아니라 데이터 정규화와 review batch 정리다.

해결한 문제:

1. exact duplicate image가 서로 다른 경로/source group 때문에 다른 feature를 갖는 문제
2. 같은 duplicate group이 shortlist에 반복 노출되는 문제
3. 사람이 campaign_id만 보고 판단하기 어려운 문제
4. coverage-gap campaign을 일반 모델 품질 평가에 섞는 문제

현재 최종 shortlist 결과:

    selected_rows: 32
    active_campaigns: 4
    excluded_campaign: phase1b_indoor_gallery_winter_art
    excluded_rows: 50
    global_duplicate_group_repeats: 0

bucket 구성:

    model_top_shortlist: 20
    prompt_conflict_audit: 4
    clip_model_disagreement_audit: 4
    dinov2_model_disagreement_audit: 4

---

## 2. Non-claims

다음은 주장하지 않는다.

    production model quality를 주장하지 않는다.
    diagnostic score를 calibrated probability로 해석하지 않는다.
    diagnostic score를 pass/fail threshold로 사용하지 않는다.
    campaign 간 score를 직접 비교하지 않는다.
    duplicate canonicalization을 이미지 품질 판단으로 사용하지 않는다.
    global duplicate suppression은 review batch 다양성 확보용이지 품질 기준이 아니다.

---

## 3. 전체 재실행 순서

### 3.1 기본 validation

    cd /Users/ksc/Desktop/cap/reranker

    python scripts/validate_phase0_specs.py
    python scripts/validate_phase1b_specs.py

성공 기준:

    ALL PHASE 0 SPEC CHECKS PASSED
    ALL PHASE 1B SPEC CHECKS PASSED

---

### 3.2 v2.2.2 DINOv2 anchor feature 생성

    cd /Users/ksc/Desktop/cap/reranker

    python scripts/build_dinov2_anchor_features_from_db.py \
      --out-dir data/feature_snapshots/v2_2_2/phase1b_dinov2_anchor

    python scripts/audit_active_features_from_db.py \
      --jsonl-glob 'data/feature_snapshots/v2_2_2/phase1b_dinov2_anchor/*.jsonl' \
      --out audit/phase_1b/active_feature_audit_v2_2_2_dinov2_anchor_jsonl.json

성공 기준:

    hard_error_count: 0

---

### 3.3 v2.2.2 candidate score snapshot 생성

    cd /Users/ksc/Desktop/cap/reranker

    python scripts/score_phase1b_candidates_v2_2_2_from_jsonl.py

출력:

    data/retrieval/phase1b/v2_2_2/candidate_score_snapshot_v2_2_2.jsonl
    audit/phase_1b/candidate_score_snapshot_v2_2_2_top_by_campaign.csv

---

### 3.4 triage 25개 생성

    cd /Users/ksc/Desktop/cap/reranker

    python scripts/build_review_queue_v2_2_2_triage_sample.py

    open data/review/phase1b/v2_2_2/triage/review_queue_v2_2_2_triage_25.html

출력:

    data/review/phase1b/v2_2_2/triage/review_queue_v2_2_2_triage_25.csv
    data/review/phase1b/v2_2_2/triage/review_queue_v2_2_2_triage_25.labeled_template.csv
    data/review/phase1b/v2_2_2/triage/review_queue_v2_2_2_triage_25.html

---

### 3.5 triage 25개 라벨 반영

확정 라벨 파일:

    data/review/phase1b/v2_2_2/triage/review_queue_v2_2_2_triage_25.labeled.csv

출력:

    data/review/phase1b/v2_2_2/triage/review_events_v2_2_2_triage_25.jsonl
    data/review/phase1b/v2_2_2/triage/training_snapshot_classifier_v2_2_2_triage_25.jsonl
    data/review/phase1b/v2_2_2/triage/training_snapshot_ranker_v2_2_2_triage_25.jsonl
    audit/phase_1b/triage_25_ingest_audit_v2_2_2.json

현재 triage 결과:

    accept: 7
    acceptable: 6
    best: 3
    reject: 9
    positive total: 16
    reject total: 9

---

## 4. v2.2.3 단계

### 4.1 campaign brief / review policy

핵심 파일:

    configs/campaign_review_briefs_v1.yaml
    configs/review_queue_policy_v2_2_3.yaml

핵심 정책:

    phase1b_indoor_gallery_winter_art = coverage_gap_diagnostic
    model_high_clip_negative_high = prompt_conflict_audit
    campaign brief는 review HTML/CSV에 필수 포함

---

### 4.2 v2.2.3 training snapshot 병합

    cd /Users/ksc/Desktop/cap/reranker

    python scripts/build_phase1b_training_snapshots_v2_2_3.py

출력:

    data/review/phase1b/v2_2_3/training_snapshot_classifier_v2_2_3.jsonl
    data/review/phase1b/v2_2_3/training_snapshot_ranker_v2_2_3.jsonl
    audit/phase_1b/training_snapshot_v2_2_3_merge_audit.json

성공 기준 예시:

    classifier_rows: 131
    ranker_rows: 131
    v2_2_2_base: 106
    v2_2_2_triage_25: 25

---

### 4.3 v2.2.3 diagnostic classifier 학습

    cd /Users/ksc/Desktop/cap/reranker

    python scripts/train_phase1b_classifier_smoke_v2_2_3_from_jsonl.py

출력:

    models/phase1b_smoke/classifier_smoke_model_v2_2_3.joblib
    audit/phase_1b/phase_1b_classifier_smoke_v2_2_3_jsonl.json

현재 v2.2.3 진단 결과:

    rows_all: 131
    rows_used_for_training: 126
    rows_excluded: 5
    OOF balanced accuracy: 0.5903890160183066
    OOF ROC-AUC: 0.5329265191965421
    OOF average precision: 0.5411487993917143

---

### 4.4 v2.2.3 review HTML/CSV 생성

    cd /Users/ksc/Desktop/cap/reranker

    python scripts/build_review_queue_v2_2_3_contact_sheets.py

    open data/review/phase1b/v2_2_3/html/index.html

출력:

    data/review/phase1b/v2_2_3/html/index.html
    data/review/phase1b/v2_2_3/html/review_queue_v2_2_3_enriched.csv
    data/review/phase1b/v2_2_3/html/review_queue_v2_2_3_enriched.labeled_template.csv

---

## 5. v2.2.5 duplicate canonicalization

### 5.1 duplicate canonicalization audit

    cd /Users/ksc/Desktop/cap/reranker

    python scripts/build_duplicate_canonicalization_v2_2_5.py

    open audit/phase_1b/duplicate_canonicalization_report_v2_2_5.md

출력:

    data/ontology/duplicate_canonical_map_v2_2_5.jsonl
    audit/phase_1b/duplicate_feature_conflict_audit_v2_2_5.json
    audit/phase_1b/duplicate_canonicalization_report_v2_2_5.md

현재 결과:

    duplicate_group_count: 24
    canonical_map_rows: 48
    metadata_conflict_group_count: 24
    feature_conflict_count: 185

주요 충돌 feature:

    path_has_architecture: 45
    path_has_garden: 45
    dinov2_campaign_margin: 38
    dinov2_campaign_neg_nn_sim: 26
    dinov2_campaign_pos_nn_sim: 12
    dinov2_family_margin: 19

---

### 5.2 canonicalized feature snapshot 생성

    cd /Users/ksc/Desktop/cap/reranker

    python scripts/build_pair_feature_snapshots_v2_2_5_duplicate_canonicalized.py

    open audit/phase_1b/feature_snapshot_canonicalization_report_v2_2_5.md

출력:

    data/feature_snapshots/v2_2_5/phase1b_duplicate_canonicalized/
    audit/phase_1b/feature_snapshot_canonicalization_audit_v2_2_5.json
    audit/phase_1b/feature_snapshot_canonicalization_report_v2_2_5.md

현재 결과:

    input_rows: 250
    output_rows: 250
    exact_duplicate_canonicalized: 91
    not_in_exact_duplicate_canonical_map: 159
    path_derived_changed_rows: 91

---

### 5.3 v2.2.5 training snapshot relink

    cd /Users/ksc/Desktop/cap/reranker

    python scripts/build_phase1b_training_snapshots_v2_2_5.py

출력:

    data/review/phase1b/v2_2_5/training_snapshot_classifier_v2_2_5.jsonl
    data/review/phase1b/v2_2_5/training_snapshot_ranker_v2_2_5.jsonl
    audit/phase_1b/training_snapshot_v2_2_5_relink_audit.json

성공 기준:

    classifier_missing: 0
    ranker_missing: 0
    classifier_rows: 131
    ranker_rows: 131

---

### 5.4 v2.2.5 diagnostic classifier 학습

    cd /Users/ksc/Desktop/cap/reranker

    python scripts/train_phase1b_classifier_smoke_v2_2_5_from_jsonl.py

출력:

    models/phase1b_smoke/classifier_smoke_model_v2_2_5.joblib
    audit/phase_1b/phase_1b_classifier_smoke_v2_2_5_jsonl.json

현재 결과:

    rows_all: 131
    rows_used_for_training: 126
    rows_excluded: 5
    OOF balanced accuracy: 0.5614035087719298
    OOF ROC-AUC: 0.5339435545385202
    OOF average precision: 0.5481169013633469

해석:

    v2.2.5는 성능 상승 단계가 아니라 duplicate canonicalization 단계다.
    balanced accuracy는 v2.2.3보다 낮아졌지만, ROC-AUC와 AP는 거의 유지 또는 소폭 개선되었다.
    path-derived duplicate leakage를 줄인 상태에서도 ranking signal은 유지되었다.

---

### 5.5 v2.2.5 candidate score snapshot 생성

    cd /Users/ksc/Desktop/cap/reranker

    python scripts/score_phase1b_candidates_v2_2_5_from_jsonl.py

출력:

    data/retrieval/phase1b/v2_2_5/candidate_score_snapshot_v2_2_5.jsonl
    audit/phase_1b/candidate_score_snapshot_v2_2_5_summary.json
    audit/phase_1b/candidate_score_snapshot_v2_2_5_top_by_campaign.csv

성공 기준:

    rows: 250
    campaign_count: 5
    model_id: classifier_smoke_v2_2_5

주의:

    diagnostic_accept_score는 calibrated probability가 아니다.
    campaign_score_percentile_desc는 campaign 내부 rank 표시다.

---

## 6. v2.2.5 shortlist 생성

### 6.1 campaign 내부 dedupe shortlist

    cd /Users/ksc/Desktop/cap/reranker

    python scripts/build_phase1b_recommendation_shortlist_v2_2_5.py

    open data/review/phase1b/v2_2_5/shortlist/index.html

출력:

    data/review/phase1b/v2_2_5/shortlist/index.html
    data/review/phase1b/v2_2_5/shortlist/recommendation_shortlist_v2_2_5.csv

성공 기준:

    selected_rows: 32
    active_campaigns: 4
    excluded_campaign_counts:
      phase1b_indoor_gallery_winter_art: 50
    duplicate groups within same campaign: 0

---

### 6.2 global dedupe shortlist

최종 사람이 볼 파일은 이 버전이다.

    cd /Users/ksc/Desktop/cap/reranker

    python scripts/build_phase1b_recommendation_shortlist_v2_2_5_global_dedupe.py

    open data/review/phase1b/v2_2_5/shortlist_global_dedupe/index.html

출력:

    data/review/phase1b/v2_2_5/shortlist_global_dedupe/index.html
    data/review/phase1b/v2_2_5/shortlist_global_dedupe/recommendation_shortlist_v2_2_5_global_dedupe.csv
    data/review/phase1b/v2_2_5/shortlist_global_dedupe/recommendation_shortlist_v2_2_5_global_dedupe.labeled_template.csv

현재 결과:

    selected_rows: 32
    active_campaigns: 4
    excluded_campaign_counts:
      phase1b_indoor_gallery_winter_art: 50
    global_duplicate_group_repeats: 0

---

## 7. Closure reports

주요 closure report:

    audit/phase_1b/v2_2_3_closure_report.md
    audit/phase_1b/v2_2_4_shortlist_report.md
    audit/phase_1b/v2_2_5_closure_report.md

열기:

    open audit/phase_1b/v2_2_5_closure_report.md

---

## 8. 다음 단계 후보

### 8.1 사유원 이미지 pool 확장

현재 가장 큰 병목은 raw image pool이다.

특히 다음 campaign은 coverage-gap으로 유지한다.

    phase1b_indoor_gallery_winter_art

필요한 이미지:

    실내 갤러리
    겨울
    전시/예술 공간
    차분한 실내 공간감

### 8.2 review batch usability 확인

최종 shortlist:

    data/review/phase1b/v2_2_5/shortlist_global_dedupe/index.html

확인 질문:

    1. model_top_shortlist가 실제로 쓸 만한가?
    2. prompt_conflict_audit은 좋은 후보인가, 애매한 후보인가?
    3. clip/dinov2 disagreement 후보는 audit용으로 남기는 게 맞는가?
    4. global dedupe 때문에 campaign별 후보 품질이 과하게 떨어지지는 않았는가?

### 8.3 다음 라벨링 원칙

다음에도 150개 전체 라벨링은 하지 않는다.

필요하면 소량만 본다.

    model_top_shortlist: campaign별 3~5개
    prompt_conflict_audit: 전체 4~8개
    clip/dinov2 disagreement: 전체 4~8개

---

## 9. 최종 결정

v2.2.5는 완료 상태로 본다.

앞으로 사람이 확인할 shortlist는 다음 버전을 기준으로 한다.

    data/review/phase1b/v2_2_5/shortlist_global_dedupe/index.html

phase1b_indoor_gallery_winter_art는 coverage-gap campaign으로 계속 분리한다.

diagnostic score는 계속 diagnostic-only이며, calibrated threshold나 production quality claim으로 사용하지 않는다.
