# v2.2.5 Closure Report

## 상태

v2.2.5는 exact duplicate image가 서로 다른 경로/source group으로 존재하면서 feature와 model score를 흔드는 문제를 정리한 단계다.

이 단계는 모델 성능 향상 단계가 아니라 duplicate canonicalization과 review batch diversity 확보 단계다.

## 완료한 작업

1. exact duplicate group별 canonical image_id 지정
2. duplicate canonical map 생성
3. metadata conflict audit 생성
4. path-derived feature canonicalization
5. v2.2.5 feature snapshot 생성
6. v2.2.5 training snapshot relink
7. v2.2.5 diagnostic classifier 재학습
8. v2.2.5 candidate score snapshot 생성
9. v2.2.5 shortlist 생성
10. global duplicate dedupe shortlist 생성

## 주요 산출물

- `data/ontology/duplicate_canonical_map_v2_2_5.jsonl`
- `audit/phase_1b/duplicate_feature_conflict_audit_v2_2_5.json`
- `data/feature_snapshots/v2_2_5/phase1b_duplicate_canonicalized/`
- `data/review/phase1b/v2_2_5/training_snapshot_classifier_v2_2_5.jsonl`
- `models/phase1b_smoke/classifier_smoke_model_v2_2_5.joblib`
- `data/retrieval/phase1b/v2_2_5/candidate_score_snapshot_v2_2_5.jsonl`
- `data/review/phase1b/v2_2_5/shortlist_global_dedupe/`

## Duplicate canonicalization 결과

- duplicate groups: 24
- canonical map rows: 48
- metadata conflict groups: 24
- feature conflict rows: 185

충돌 feature에는 `path_has_architecture`, `path_has_garden`, `dinov2_campaign_margin`, `dinov2_campaign_pos_nn_sim`, `dinov2_campaign_neg_nn_sim`, `dinov2_family_margin`이 포함되었다.

## Feature snapshot canonicalization 결과

- input rows: 250
- output rows: 250
- exact duplicate canonicalized rows: 91
- non-duplicate rows: 159
- path-derived changed rows: 91

## Training relink 결과

- classifier rows: 131
- ranker rows: 131
- classifier missing: 0
- ranker missing: 0

## Diagnostic classifier 결과

- rows used for training: 126
- rows excluded: 5
- OOF balanced accuracy: 0.5614035087719298
- OOF ROC-AUC: 0.5339435545385202
- OOF average precision: 0.5481169013633469

해석:

v2.2.5 canonicalization이 모델 성능을 크게 올린 것은 아니다.  
그러나 path-derived duplicate leakage를 줄인 상태에서도 ranking signal은 유지되었다.

## Global dedupe shortlist 결과

- selected rows: 32
- active campaigns: 4
- excluded campaign: `phase1b_indoor_gallery_winter_art`
- excluded rows: 50
- global duplicate group repeats: 0

Bucket 구성:

- model_top_shortlist: 20
- prompt_conflict_audit: 4
- clip_model_disagreement_audit: 4
- dinov2_model_disagreement_audit: 4

## 결정

1. 앞으로 사람이 볼 shortlist는 `shortlist_global_dedupe` 버전을 기준으로 한다.
2. same-campaign duplicate suppression뿐 아니라 review-batch global duplicate suppression도 유지한다.
3. `phase1b_indoor_gallery_winter_art`는 coverage-gap campaign으로 계속 제외한다.
4. diagnostic score는 최종 품질 점수나 pass/fail threshold가 아니다.
5. duplicate canonicalization은 품질 판단이 아니라 데이터 정규화 단계다.

## Non-claims

- production model quality를 주장하지 않는다.
- diagnostic score를 calibrated probability로 해석하지 않는다.
- campaign 간 score를 직접 비교하지 않는다.
- duplicate canonicalization을 accept/reject threshold로 사용하지 않는다.
- global duplicate suppression은 review batch 다양성 확보용이지 품질 기준이 아니다.
