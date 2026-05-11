# Phase 1a 요약 — v2.2.1 End-to-End Dry Run

## 메타데이터

spec_version: v2.2.1
phase: phase_1a
phase_name: small_gallery_embedding_index_dry_run
exit_status: pass_with_diagnostic_warnings
score_status: diagnostic_only
created_from: audit/phase_1a/phase_1a_exit_report.json
related_freeze_doc: docs/design_history/v2_2_1_freeze.md

## 한 줄 결론

Phase 1a는 사용 가능한 206개 raw image gallery에 대해 v2.2.1 image reranker pipeline이 처음부터 끝까지 실제로 동작함을 검증했다.

다만 이 결과는 **모델 성능 검증이 아니다.** 아직 production quality, calibrated threshold, ranker generalization, visual critic 성능, 최종 PPTX/Canva export 품질을 주장하면 안 된다.

현재 모든 score는 다음 상태다.

score_status = diagnostic_only

## Phase 1a의 목적

Phase 1a의 목적은 embedding-first reranker 구조가 실제 데이터에서 end-to-end로 연결되는지 확인하는 것이다.

검증한 흐름은 다음과 같다.

raw image manifest
→ CLIP / DINOv2 image embeddings
→ duplicate groups / DINOv2 clusters / region safety maps
→ CLIP retrieval
→ PairFeatureSnapshot
→ cold_start review queue
→ human review labels
→ ReviewEvent
→ classifier/ranker TrainingSnapshot
→ Phase 1a exit report

즉 Phase 1a는 “좋은 모델을 만들었는가”를 보는 단계가 아니라, “나중에 좋은 모델을 만들 수 있는 데이터 파이프라인이 실제로 도는가”를 보는 단계다.

## 산출물 count

raw_gallery_count = 206

CLIP embedding count = 206
CLIP embedding dim = 512
CLIP index count = 206

DINOv2 embedding count = 206
DINOv2 embedding dim = 768
DINOv2 index count = 206

duplicate_group_rows = 206
duplicate_group_count = 182
exact_duplicate_rows = 48

DINOv2 cluster rows = 206
cluster_coarse_count = 20
cluster_mid_count = 100
cluster_fine_count = 182

region_safety_count = 206
regions_per_image = 16

retrieval_candidate_count = 50
pair_feature_snapshot_count = 50

review_queue_row_count = 30
review_event_count = 30

training_classifier_rows = 30
training_ranker_rows = 30

## Review queue 결과

Phase 1a cold-start review queue는 30개 row로 생성되었고, 30개 모두 서로 다른 duplicate group을 가졌다.

review_queue rows = 30
unique review_queue duplicate ids = 30

Bucket 구성은 다음과 같다.

clip_high_model_low = 12
cluster_diversity = 8
layout_safe_coverage = 6
uncertainty = 3
random_coverage = 1

초기 critique 이후 skipped bucket audit도 추가했다.

dinov2_anchor_high_model_low
→ cold_start에서 campaign-positive anchor가 없으므로 skip

model_high_clip_negative_high
→ cold_start에서 diagnostic_model_score가 없으므로 skip

critic_high_risk_reranker_high
→ Phase 1a에서 critic이 없으므로 skip

classifier_ranker_disagreement
→ Phase 1a에서 ranker가 없으므로 skip

이는 spec 위반이 아니다. Phase 1a는 cold_start dry run이므로, 해당 bucket들은 필요한 신호가 생긴 뒤 활성화된다.

## Human review label 요약

ReviewEvent label 분포:

accept = 10
acceptable = 7
reject = 13

Classifier snapshot mapping:

reject → 0
acceptable → 1
accept → 1

Classifier label 분포:

0 = 13
1 = 17

Ranker snapshot mapping:

reject → 0
acceptable → 1
accept → 2
best → 3

Ranker label 분포:

0 = 13
1 = 7
2 = 10

Issue tag 분포:

semantic_mismatch = 12
too_busy_background = 1

## Bucket별 관찰

### 1. clip_high_model_low

rows: 12
accept / acceptable: 11
reject: 1

이 bucket은 의도대로 동작했다.

CLIP semantic retrieval이 sample campaign인 “여름 정원 산책”에 대해 상당히 높은 비율의 usable candidate를 가져왔다.

이 결과는 CLIP을 text-image semantic retrieval에 쓰고, DINOv2를 image-image visual similarity에 쓰는 v2.2.1 역할 분리가 타당하다는 초기 증거다.

### 2. cluster_diversity

rows: 8
reject-heavy

cluster_diversity는 accept rate를 극대화하기 위한 bucket이 아니다. 목적은 under-reviewed DINOv2 visual region을 노출하는 것이다.

이번 라운드에서는 reject가 많았지만, 이 자체가 정상이다. 이 bucket은 blind spot을 줄이는 역할을 한다.

### 3. layout_safe_coverage

rows: 6
mostly semantic_mismatch rejects

layout-safe하다고 해서 campaign-relevant한 것은 아니다.

즉 region safety는 accept/reject 기준이 아니라 feature와 sampling signal로만 사용해야 한다. 이 판단은 calibration_policy_v1과 일치한다.

### 4. uncertainty

rows: 3
acceptable / reject 혼합

uncertainty bucket은 현재 score distribution의 middle band에서 후보를 뽑는 diagnostic sampling rule이다.

이는 calibrated uncertainty threshold가 아니다.

### 5. random_coverage

rows: 1
acceptable

random coverage는 낮은 예산으로 blind spot을 확인하는 용도로 유지할 가치가 있다.

## CLIP score distribution audit

초기 Phase 1a critique 이후, CLIP retrieval script에 전체 206개 score distribution audit을 추가했다.

전체 206개 candidate pool:

all_count = 206
clip_positive_max_sim_min = 0.2037400752
clip_positive_max_sim_max = 0.2906503081
clip_positive_max_sim_mean = 0.2454800457
clip_positive_max_sim_std = 0.0179398395

clip_negative_max_sim_min = 0.1914626360
clip_negative_max_sim_max = 0.2873419225
clip_negative_max_sim_mean = 0.2341272682
clip_negative_max_sim_std = 0.0196224898

clip_margin_min = -0.0414145589
clip_margin_max = 0.0695060939
clip_margin_mean = 0.0113528054
clip_margin_std = 0.0252036862

Top-50 retrieval candidates:

top_k = 50
top_k_clip_margin_min = 0.0290045291
top_k_clip_margin_max = 0.0695060939
top_k_clip_margin_mean = 0.0461218916
top_k_clip_margin_std = 0.0119169485

top_k_clip_positive_max_sim_min = 0.2260169983
top_k_clip_positive_max_sim_max = 0.2903265655
top_k_clip_positive_max_sim_mean = 0.2621235549
top_k_clip_positive_max_sim_std = 0.0150149334

해석:

Top-50 후보는 전체 206개 pool보다 명확히 높은 margin 영역에 있다. 그러나 absolute score range는 좁다. 따라서 CLIP score는 계속 diagnostic_only로 유지한다.

즉 이 score로 final accept/reject threshold를 만들면 안 된다.

## Duplicate group 관찰

Phase 1a에서 발견된 exact duplicate:

exact_duplicate_rows = 48
duplicate_group_count = 182

이는 size 2 duplicate group이 24개 있고, singleton group이 158개 있다는 뜻이다.

Review queue에서는 duplicate group suppression이 제대로 동작했다.

review_queue rows = 30
unique review_queue duplicate ids = 30

이 결과는 작은 206개 pool에서도 duplicate suppression이 필수임을 보여준다.

## Phase 1a가 증명한 것

Phase 1a는 다음을 증명했다.

1. v2.2.1 schemas/configs가 실제 코드로 구현 가능하다.
2. raw image asset을 stable manifest row로 변환할 수 있다.
3. CLIP / DINOv2 embedding을 추출하고 index로 관리할 수 있다.
4. exact duplicate group과 DINOv2 cluster를 생성할 수 있다.
5. region safety map을 diagnostic feature로 생성할 수 있다.
6. CLIP retrieval이 campaign-relevant candidate를 가져온다.
7. PairFeatureSnapshot row를 생성하고 schema validation할 수 있다.
8. cold_start bucketed review queue를 생성할 수 있다.
9. 사람이 채운 label을 ReviewEvent로 변환할 수 있다.
10. classifier/ranker TrainingSnapshot을 분리 생성할 수 있다.

## Phase 1a가 증명하지 않은 것

Phase 1a는 다음을 증명하지 않는다.

1. production reranker quality
2. new campaign에 대한 generalization
3. calibrated acceptance threshold
4. LGBMRanker validity
5. Visual Critic performance
6. layout-aware final poster quality
7. PPTX / Canva export fidelity
8. multi-annotator reliability

따라서 Phase 1a 이후에도 모든 model score와 heuristic score는 다음 상태다.

diagnostic_only

## Phase 1b 권고

### 1. Campaign 수를 늘린다

Phase 1b에서는 같은 pipeline을 여러 campaign payload에 대해 반복한다.

최소 목표:

held-out validation 전 최소 campaign 수: 5
진지한 reranker 성능 주장 전 campaign 수: 5개 초과 권장

### 2. Bucketed review queue를 유지한다

top-N 정렬로 돌아가면 안 된다.

계속 유지할 bucket:

clip_high_model_low
cluster_diversity
layout_safe_coverage
uncertainty
random_coverage

나중에 신호가 생기면 활성화할 bucket:

dinov2_anchor_high_model_low
model_high_clip_negative_high
critic_high_risk_reranker_high
classifier_ranker_disagreement

### 3. 첫 positive가 쌓이면 DINOv2 anchor feature를 활성화한다

campaign 또는 campaign family에 accept image가 생기면 다음 feature를 활성화한다.

dinov2_campaign_pos_nn_sim
dinov2_campaign_neg_nn_sim
dinov2_campaign_margin
dinov2_campaign_anchor_missing = 0

### 4. Second annotator audit을 추가한다

Phase 1a는 human_001 한 명의 label만 사용했다.

Phase 1b에서는 일부 overlap set을 두 번째 annotator가 다시 보고 disagreement를 측정한다.

특히 다음 구분에서 disagreement를 봐야 한다.

acceptable vs accept
semantic_mismatch vs mood_mismatch
too_busy_background vs text_region_conflict

### 5. Preview renderer v1을 빨리 붙인다

Phase 1a에서 layout-related issue tag는 거의 나오지 않았다.

too_busy_background = 1

이는 preview rendering이 없으면 critic 학습용 label이 느리게 쌓인다는 뜻이다.

따라서 box-overlay preview renderer v1을 Phase 1b에서 같이 붙이는 것이 좋다.

### 6. Multiple layout specs를 추가한다

Phase 1a에서는 layout이 하나뿐이었다.

layout_top_left_bottom_left

Phase 1b에서는 최소 2~3개의 layout spec을 같은 image에 적용해, image label이 text placement에 따라 달라지는지 측정해야 한다.

### 7. Exact duplicate suppression을 계속 유지한다

SHA256 exact duplicate grouping이 이미 많은 중복을 잡았다.

Review queue는 명시적 audit/relabeling 목적이 아니라면 duplicate group 단위로 계속 suppression해야 한다.

## 최종 결정

Phase 1a exit_status = pass_with_diagnostic_warnings

Phase 1b로 진행한다.

단, 아직 model quality를 주장하지 않는다. Held-out campaign validation이 생기기 전까지 모든 score는 diagnostic_only로 유지한다.

