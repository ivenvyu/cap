
# Phase 1b Scope — Multi-Campaign Label Expansion

## 메타데이터

spec_version: v2.2.1  
phase: phase_1b  
phase_name: multi_campaign_label_expansion  
depends_on: phase_1a  
phase_1a_exit_status: pass_with_diagnostic_warnings  
score_status: diagnostic_only  
related_docs:
- docs/design_history/v2_2_1_freeze.md
- docs/design_history/phase_1a_summary.md
- audit/phase_1a/phase_1a_exit_report.json

## 한 줄 결론

Phase 1b의 목적은 Phase 1a에서 검증된 end-to-end pipeline을 여러 campaign에 반복 적용하여, 학습과 held-out campaign validation이 가능한 human-reviewed dataset을 만들기 시작하는 것이다.

Phase 1b의 목표는 **모델 성능 주장**이 아니다.  
Phase 1b의 목표는 **campaign coverage와 human label 수를 늘리는 것**이다.

## Phase 1a에서 확인된 것

Phase 1a는 다음 흐름이 실제로 동작함을 확인했다.

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

Phase 1a 산출물 요약:

raw_gallery_count = 206  
CLIP embedding = 206 × 512  
DINOv2 embedding = 206 × 768  
duplicate_group_count = 182  
exact_duplicate_rows = 48  
DINOv2 clusters = coarse 20 / mid 100 / fine 182  
region_safety = 206 rows × 16 regions  
retrieval candidates = 50  
PairFeatureSnapshot = 50  
review queue = 30  
ReviewEvent = 30  
classifier TrainingSnapshot = 30  
ranker TrainingSnapshot = 30  

Phase 1a의 결론:

exit_status = pass_with_diagnostic_warnings  
score_status = diagnostic_only  

따라서 Phase 1b는 “파이프라인이 도는가?”를 다시 확인하는 단계가 아니다.  
이제는 “같은 파이프라인을 여러 campaign에 반복 적용할 수 있는가?”를 확인하는 단계다.

## Phase 1b의 핵심 목표

Phase 1b의 핵심 목표는 다음과 같다.

1. campaign 수를 늘린다.
2. 각 campaign에 대해 CLIP retrieval, duplicate suppression, review queue를 반복 생성한다.
3. 사람이 라벨링한 ReviewEvent를 누적한다.
4. classifier/ranker TrainingSnapshot을 campaign across로 누적한다.
5. held-out campaign split이 가능한 최소 데이터 구조를 만든다.
6. DINOv2 positive anchor feature를 활성화할 수 있는 조건을 만든다.
7. multiple layout spec과 preview renderer v1의 필요성을 실험적으로 확인한다.

## Phase 1b의 비목표

Phase 1b에서는 다음을 하지 않는다.

1. production reranker quality 주장
2. calibrated accept/reject threshold 주장
3. LGBMRanker 성능 주장
4. final layout/poster quality 주장
5. Visual Critic 학습 완료 주장
6. support explanation 도입
7. top-N score sorting으로 review queue 단순화
8. unvalidated threshold 기반 자동 reject

모든 model score와 heuristic score는 계속 다음 상태를 유지한다.

score_status = diagnostic_only

## Campaign 목표

Phase 1b의 최소 campaign 목표:

minimum_campaigns = 5

5개는 성능 주장에 충분한 수가 아니다.  
다만 held-out campaign split을 처음으로 형식화할 수 있는 최소 수준이다.

권장 목표:

preferred_campaigns > 5

가능하면 5개보다 많은 campaign을 사용한다. campaign이 많을수록 다음 판단이 가능해진다.

1. 특정 campaign에만 맞는 retrieval bias인지 확인
2. gallery / arbor / course 계열의 label 분포 비교
3. accept / acceptable / reject 기준의 campaign 간 안정성 확인
4. held-out campaign split 준비
5. ranker group 수 확보

## Campaign payload 구성

각 campaign payload는 최소한 다음 필드를 포함한다.

campaign_id  
campaign_version  
campaign_text  
purpose_type  
space_type  
season  
mood_tags  
place_name  
target_channel  
brand_tone  

Phase 1b에서 우선 포함할 campaign 유형:

1. summer garden walking program
2. autumn garden walking program
3. architecture-focused exhibition or visit program
4. flower or botanical seasonal campaign
5. calm premium brand/background campaign

중요한 것은 campaign끼리 충분히 달라야 한다는 점이다.  
모두 “여름 정원 산책”과 비슷하면 held-out campaign validation의 의미가 약해진다.

## Phase 1b pipeline

각 campaign에 대해 다음 순서를 반복한다.

1. campaign payload 작성
2. CLIP prompt generation
3. CLIP retrieval candidates 생성
4. duplicate group suppression 적용
5. PairFeatureSnapshot 생성
6. cold_start 또는 anchor-aware review queue 생성
7. HTML/contact sheet로 이미지 확인
8. human label 입력
9. ReviewEvent ingest
10. TrainingSnapshot 생성
11. campaign-level audit summary 생성
12. global cumulative label summary 업데이트

## Review queue 정책

Phase 1b에서도 bucketed review queue를 유지한다.

계속 사용할 bucket:

clip_high_model_low  
cluster_diversity  
layout_safe_coverage  
uncertainty  
random_coverage  

조건부로 활성화할 bucket:

dinov2_anchor_high_model_low  
model_high_clip_negative_high  
critic_high_risk_reranker_high  
classifier_ranker_disagreement  

Phase 1b 초반에는 trained model과 critic이 없으므로 다음 bucket은 계속 skip될 수 있다.

model_high_clip_negative_high  
critic_high_risk_reranker_high  
classifier_ranker_disagreement  

이 경우 skip 이유를 audit log에 명시한다.

## DINOv2 anchor activation

Phase 1a에서는 campaign positive가 없었으므로 다음 값이 cold-start 상태였다.

dinov2_campaign_pos_count = 0  
dinov2_campaign_anchor_missing = 1  

Phase 1b에서는 각 campaign 또는 campaign family에 accepted image가 생기면 DINOv2 positive anchor를 활성화한다.

활성화 조건:

1. 같은 campaign에서 accept 또는 acceptable 이미지가 존재한다.
2. 또는 같은 campaign family에서 accept 이미지가 충분히 존재한다.
3. 해당 anchor set의 provenance가 명확하다.
4. duplicate group leakage가 제거되어 있다.

활성화 후 사용할 feature:

dinov2_campaign_pos_nn_sim  
dinov2_campaign_neg_nn_sim  
dinov2_campaign_margin  
dinov2_campaign_pos_count  
dinov2_campaign_neg_count  
dinov2_campaign_anchor_missing = 0  

단, 이 feature도 Phase 1b에서는 diagnostic_only다.

## Duplicate group 정책

Phase 1a에서 exact duplicate가 이미 확인되었다.

exact_duplicate_rows = 48  
duplicate_group_count = 182  

따라서 Phase 1b review queue에서도 duplicate group suppression을 유지한다.

원칙:

1. 같은 review queue 안에서는 같은 duplicate_group_id를 반복 노출하지 않는다.
2. campaign 간에는 필요하면 같은 duplicate group이 다시 나올 수 있으나 audit에 기록한다.
3. 학습/검증 split에서는 duplicate group leakage를 막는다.
4. later parent_image_id 정책이 생기면 duplicate_group_id와 함께 사용한다.

## Layout policy

Phase 1a에서는 하나의 layout만 사용했다.

layout_top_left_bottom_left

Phase 1b에서는 최소 2개 이상의 layout spec을 도입할 수 있다.

후보 layout:

layout_top_left_bottom_left  
layout_top_center_bottom_center  
layout_left_center_right_info  

목적은 같은 image가 layout에 따라 label이 바뀌는지 확인하는 것이다.

단, layout spec을 늘리면 review burden이 증가한다.  
따라서 모든 image에 모든 layout을 적용하지 말고, 일부 campaign 또는 일부 candidate에만 적용한다.

권장 방식:

1. campaign당 기본 layout 1개 사용
2. 일부 high-CLIP 후보에 layout variant 1개 추가
3. 동일 image + 다른 layout의 decision 차이를 audit
4. layout-related issue_tags가 실제로 늘어나는지 확인

## Preview renderer v1

Phase 1a에서는 preview renderer 없이 라벨링했다.  
그 결과 layout-related issue tag는 거의 나오지 않았다.

too_busy_background = 1

이는 preview가 없으면 layout/critic 관련 label이 느리게 쌓인다는 신호다.

Phase 1b에서는 box-overlay 수준의 preview renderer v1 도입을 검토한다.

preview renderer v1의 최소 요구사항:

1. raw image 표시
2. title box rectangle 표시
3. info box rectangle 표시
4. optional role box rectangle 표시
5. translucent overlay 표시
6. 실제 Korean text rendering은 아직 제외 가능
7. final gradient, font fallback, kerning은 제외 가능

목적은 final design rendering이 아니라, human reviewer가 text_region_conflict와 too_busy_background를 더 일관되게 판단할 수 있게 하는 것이다.

## Support explanation defer 결정

Phase 1b에서는 candidate-level support explanation을 도입하지 않는다.

이론적으로 support explanation은 적절한 개념이다.  
특정 campaign-image-layout candidate가 semantic, visual, duplicate, layout, metadata axis에서 어떻게 지지되거나 약한지 설명하는 diagnostic audit object로 사용할 수 있다.

그러나 Phase 1b 시점에서는 다음 조건이 부족하다.

1. stable empirical baseline이 없다.
2. accepted image 수가 적다.
3. campaign 수가 적다.
4. critic score가 없다.
5. reviewer는 이미 이미지를 보고 충분히 판단할 수 있다.
6. 구현 비용 대비 label expansion에 주는 이익이 작다.

따라서 Phase 1b에서는 support explanation을 명시적으로 defer한다.

Support explanation은 다음 시점에 재검토한다.

Phase 3:
- multi-campaign label data가 충분히 쌓인 뒤
- empirical baseline을 정의할 수 있을 때
- candidate fragility를 percentile/reference 기준으로 말할 수 있을 때

Phase 4:
- critic score가 생긴 뒤
- layout_region_fragility와 issue_tags의 bridge를 검증할 수 있을 때
- support signal과 human reject reason의 대응을 empirical bridge hypothesis로 측정할 수 있을 때

결정:

support_explanation_status = deferred_from_phase_1b

## Human review policy

Phase 1b에서도 decision label은 기존 기준을 유지한다.

0 = reject  
1 = acceptable  
2 = accept  

나중에 best label이 필요하면 추가할 수 있지만, Phase 1b의 기본 review queue에서는 0/1/2를 유지한다.

가능하면 issue_tags는 closed vocabulary를 사용한다.

semantic_mismatch  
season_mismatch  
brand_tone_mismatch  
mood_mismatch  
text_region_conflict  
low_contrast  
too_busy_background  
visual_hierarchy_weak  
duplicate_or_too_similar  
already_used_in_recent_campaign  
low_resolution  
poor_composition  

reject row에는 issue_tags를 비워두지 않는다.

acceptable과 accept row는 issue_tags가 비어 있어도 된다.  
단, weak accept 이유를 추적하고 싶으면 notes를 사용한다.

## Multi-annotator audit

Phase 1b에서는 모든 row에 두 번째 annotator를 붙이지 않는다.  
비용이 크기 때문이다.

대신 일부 overlap set을 만든다.

권장 overlap set:

1. uncertainty bucket 후보
2. acceptable vs accept 경계 후보
3. semantic_mismatch로 reject된 후보 중 CLIP margin이 높은 후보
4. layout_safe_coverage에서 reject된 후보
5. 같은 image의 다른 layout variant 후보

목적:

1. label noise 측정
2. acceptable/accept 기준 안정성 확인
3. issue_tags 일관성 확인
4. ranker relevance grade의 신뢰도 확인

## TrainingSnapshot policy

Phase 1b에서도 classifier와 ranker snapshot을 분리한다.

Classifier mapping:

reject → 0  
acceptable → 1  
accept → 1  
best → 1  

Ranker mapping:

reject → 0  
acceptable → 1  
accept → 2  
best → 3  

중요:

1. classifier는 binary usability 학습용이다.
2. ranker는 campaign 내부 relative relevance 학습용이다.
3. campaign이 충분히 쌓이기 전까지 ranker metric을 성능 주장으로 사용하지 않는다.
4. held-out campaign split 전까지 모든 결과는 diagnostic_only다.

## Held-out campaign split 준비

Phase 1b의 중요한 산출물은 held-out campaign split 가능성이다.

조건:

1. campaign 수가 최소 5개 이상
2. campaign별 label row가 충분히 있음
3. duplicate group leakage가 통제됨
4. 같은 duplicate_group_id가 train/test에 동시에 들어가지 않도록 관리
5. 같은 campaign family leakage를 별도 audit

Phase 1b 말기에 다음 split을 준비한다.

train_campaigns  
validation_campaigns  
heldout_campaigns  

하지만 Phase 1b 초반에는 아직 split metric을 주장하지 않는다.

## Phase 1b 성공 기준

Phase 1b는 다음 조건을 만족하면 성공으로 본다.

1. 최소 5개 campaign에 대해 retrieval → review → ReviewEvent → TrainingSnapshot이 반복 실행됨
2. campaign별 review queue가 duplicate group suppression을 유지함
3. cumulative ReviewEvent가 schema validation을 통과함
4. cumulative TrainingSnapshot이 classifier/ranker로 분리 생성됨
5. campaign별 label 분포가 audit됨
6. bucket별 accept/reject 분포가 audit됨
7. held-out campaign split 설계가 가능해짐
8. 모든 score가 diagnostic_only로 유지됨
9. support explanation을 도입하지 않은 이유가 문서화됨

## Phase 1b 실패 기준

다음 중 하나가 발생하면 Phase 1b는 실패 또는 재설계가 필요하다.

1. campaign별 pipeline이 반복 실행되지 않음
2. review queue가 특정 category 또는 duplicate group에 과도하게 쏠림
3. CLIP retrieval이 campaign 의미를 거의 반영하지 못함
4. arbor/flower 또는 gallery/architecture 같은 특정 계열이 계속 오분류됨
5. issue_tags가 지나치게 semantic_mismatch 하나로만 쏠림
6. layout-related label이 preview 없이 계속 수집되지 않음
7. TrainingSnapshot schema가 campaign 누적에서 깨짐
8. held-out campaign split을 만들 수 없을 정도로 campaign diversity가 부족함

## Phase 1b 산출물

예상 산출물:

campaign payloads  
campaign-level prompt sets  
campaign-level CLIP retrieval candidates  
campaign-level PairFeatureSnapshots  
campaign-level review queues  
labeled review CSVs  
ReviewEvent JSONL 누적본  
classifier TrainingSnapshot 누적본  
ranker TrainingSnapshot 누적본  
campaign-level audit reports  
phase_1b_exit_report.json  

생성 artifact는 git에 넣지 않는다.  
script, config, schema, design history 문서만 git에 넣는다.

## 다음 구현 순서

Phase 1b 시작 순서:

1. campaign payload 4개 이상 추가
2. existing Phase 1a scripts를 campaign-id parameter 기반으로 일반화
3. campaign loop runner 작성
4. campaign별 retrieval 실행
5. campaign별 PairFeatureSnapshot 생성
6. campaign별 review queue 생성
7. HTML/contact sheet 생성
8. human label 입력
9. ReviewEvent 누적 ingest
10. TrainingSnapshot 누적 생성
11. campaign-level audit report 생성
12. Phase 1b 중간 점검

## 최종 결정

Phase 1b로 진행한다.

단, Phase 1b의 중심은 support explanation이 아니라 multi-campaign label expansion이다.

모든 score는 held-out campaign validation 전까지 diagnostic_only로 유지한다.

Support explanation은 Phase 3 또는 Phase 4에서 baseline, critic signal, 충분한 label data가 생긴 뒤 재검토한다.
## Phase 1b 구현 전 보강 결정

### 1. Campaign diversity 보강

Phase 1b의 최소 단위는 campaign 5개가 아니라, 실질적으로는 **campaign family diversity**다.

다음 두 campaign은 서로 다른 campaign이지만 같은 family로 본다.

summer garden walking program  
autumn garden walking program  

둘 다 purpose_type이 walking_program이고 space_type이 garden이므로, CLIP retrieval 관점에서는 positive prompt가 매우 유사하다. 따라서 이 둘만으로는 held-out campaign validation의 독립성이 약하다.

Phase 1b에서는 최소한 다음 family 구성을 목표로 한다.

garden_walking_family  
architecture_family  
botanical_or_flower_family  
indoor_gallery_or_winter_family  
brand_background_family  

특히 하나는 기존 negative prompt와 의미가 뒤집히는 campaign이어야 한다.

예:

indoor gallery/art program  
또는 winter landscape program  

이유는 이전 campaign에서 negative였던 의미 영역이 다른 campaign에서는 positive가 될 수 있어야, retrieval과 reranker가 단순히 한 방향으로만 bias되지 않았는지 확인할 수 있기 때문이다.

Phase 1b 최소 기준을 다음처럼 수정한다.

minimum_campaigns = 5  
minimum_campaign_families = 4  

campaign 수만 채우는 것은 충분하지 않다. campaign family diversity가 부족하면 held-out campaign validation은 형식적으로만 가능하고 실질적 의미는 약하다.

### 2. Campaign loop runner scope

Phase 1b에서는 full workflow engine을 만들지 않는다.

허용되는 runner 수준:

shell_script_runner = allowed  
single_python_pipeline_runner = optional_later  
snakemake_or_workflow_engine = deferred  

Phase 1b의 campaign 수는 작으므로 shell script 기반 순차 실행으로 충분하다.

예상 구조:

for campaign in examples/campaigns/phase1b/*.json; do  
  build_clip_retrieval_candidates  
  build_pair_feature_snapshots  
  build_phase1b_review_queue  
done  

단, campaign별 산출물과 cumulative 산출물은 모두 유지한다.

campaign별 파일:

data/review/review_events_phase1b_v1__{campaign_id}.jsonl  
data/review/training_snapshot_phase1b_classifier_v1__{campaign_id}.jsonl  
data/review/training_snapshot_phase1b_ranker_v1__{campaign_id}.jsonl  

누적 파일:

data/review/review_events_phase1b_v1.jsonl  
data/review/training_snapshot_phase1b_classifier_v1.jsonl  
data/review/training_snapshot_phase1b_ranker_v1.jsonl  

이유:

1. campaign별 재실행이 가능해야 한다.
2. 특정 campaign의 label correction이 전체 누적본을 직접 깨뜨리면 안 된다.
3. cumulative file은 concat/rebuild artifact로 취급한다.
4. campaign별 파일이 source of truth가 된다.

### 3. DINOv2 anchor activation timing

Phase 1b 첫 라운드는 모든 campaign을 cold_start로 돌린다.

phase_1b_anchor_policy:

first_round = cold_start_all_campaigns  
optional_second_round = anchor_aware_with_first_round_labels  
required = false  

이유:

Phase 1b의 핵심 목표는 anchor 정밀화가 아니라 campaign coverage와 human label 확보다.

Sequential anchor-aware 방식은 다음 장점이 있다.

campaign 1 label  
→ campaign 2에서 campaign 1의 accepted image를 family anchor로 사용 가능  

하지만 이 방식은 사람이 campaign 1을 먼저 라벨링해야 campaign 2 queue를 만들 수 있으므로 운영 속도를 늦춘다.

따라서 Phase 1b 첫 라운드는 다음 방식으로 한다.

campaign 1~5 retrieval  
→ campaign 1~5 review queue 생성  
→ 사람이 한 번에 라벨링  
→ ReviewEvent 누적 ingest  
→ TrainingSnapshot 누적 생성  

이후 시간이 있으면 optional second round로 anchor-aware queue를 생성한다.

optional second round에서 활성화할 feature:

dinov2_campaign_pos_nn_sim  
dinov2_campaign_neg_nn_sim  
dinov2_campaign_margin  
dinov2_campaign_pos_count  
dinov2_campaign_neg_count  
dinov2_campaign_anchor_missing = 0  

단, second round는 Phase 1b 성공 필수 조건이 아니다. 시간이 부족하면 Phase 2로 넘긴다.

### 4. Issue tag concentration failure criterion 보정

Phase 1a에서는 reject issue_tags가 semantic_mismatch에 강하게 쏠렸다.

semantic_mismatch = 12  
too_busy_background = 1  

이 현상은 preview renderer가 없는 상태에서는 자연스러울 수 있다. 사람이 실제 text box overlay를 보지 못하면 text_region_conflict, low_contrast, visual_hierarchy_weak 같은 layout-related issue_tags를 일관되게 쓰기 어렵다.

따라서 Phase 1b의 failure criterion을 다음처럼 분리한다.

#### 4a. Preview renderer v1이 없는 경우

issue_tags가 semantic_mismatch에 쏠리는 것은 즉시 실패가 아니다.

이 경우 status는 다음과 같다.

issue_tag_concentration_status = preview_renderer_trigger  

의미:

semantic_mismatch 쏠림은 review vocabulary 실패라기보다 preview renderer 도입 필요 신호로 본다.

#### 4b. Preview renderer v1이 있는 경우

preview renderer v1이 있는데도 issue_tags가 계속 semantic_mismatch에만 쏠리면 실패 또는 재설계 신호다.

가능한 원인:

1. reviewer UI가 layout issue를 충분히 보이게 하지 못함
2. issue_tags vocabulary가 reviewer에게 명확하지 않음
3. layout box overlay가 실제 text conflict를 잘 드러내지 못함
4. review instruction이 semantic 판단에만 치우침

이 경우 다음을 재검토한다.

review UI  
preview renderer  
issue_tags instruction  
layout-related examples  
critic_preview_renderer_v1 config  

수정된 실패 기준:

preview_renderer_v1_absent_and_semantic_mismatch_concentrated  
→ failure가 아니라 preview_renderer_trigger

preview_renderer_v1_present_and_semantic_mismatch_concentrated  
→ failure_or_review_tool_redesign_required

## Phase 1b 구현 전 보강 결정

### 1. Campaign diversity 보강

Phase 1b의 최소 단위는 campaign 5개가 아니라, 실질적으로는 **campaign family diversity**다.

다음 두 campaign은 서로 다른 campaign이지만 같은 family로 본다.

summer garden walking program  
autumn garden walking program  

둘 다 purpose_type이 walking_program이고 space_type이 garden이므로, CLIP retrieval 관점에서는 positive prompt가 매우 유사하다. 따라서 이 둘만으로는 held-out campaign validation의 독립성이 약하다.

Phase 1b에서는 최소한 다음 family 구성을 목표로 한다.

garden_walking_family  
architecture_family  
botanical_or_flower_family  
indoor_gallery_or_winter_family  
brand_background_family  

특히 하나는 기존 negative prompt와 의미가 뒤집히는 campaign이어야 한다.

예:

indoor gallery/art program  
또는 winter landscape program  

이유는 이전 campaign에서 negative였던 의미 영역이 다른 campaign에서는 positive가 될 수 있어야, retrieval과 reranker가 단순히 한 방향으로만 bias되지 않았는지 확인할 수 있기 때문이다.

Phase 1b 최소 기준을 다음처럼 수정한다.

minimum_campaigns = 5  
minimum_campaign_families = 4  

campaign 수만 채우는 것은 충분하지 않다. campaign family diversity가 부족하면 held-out campaign validation은 형식적으로만 가능하고 실질적 의미는 약하다.

### 2. Campaign loop runner scope

Phase 1b에서는 full workflow engine을 만들지 않는다.

허용되는 runner 수준:

shell_script_runner = allowed  
single_python_pipeline_runner = optional_later  
snakemake_or_workflow_engine = deferred  

Phase 1b의 campaign 수는 작으므로 shell script 기반 순차 실행으로 충분하다.

예상 구조:

for campaign in examples/campaigns/phase1b/*.json; do  
  build_clip_retrieval_candidates  
  build_pair_feature_snapshots  
  build_phase1b_review_queue  
done  

단, campaign별 산출물과 cumulative 산출물은 모두 유지한다.

campaign별 파일:

data/review/review_events_phase1b_v1__{campaign_id}.jsonl  
data/review/training_snapshot_phase1b_classifier_v1__{campaign_id}.jsonl  
data/review/training_snapshot_phase1b_ranker_v1__{campaign_id}.jsonl  

누적 파일:

data/review/review_events_phase1b_v1.jsonl  
data/review/training_snapshot_phase1b_classifier_v1.jsonl  
data/review/training_snapshot_phase1b_ranker_v1.jsonl  

이유:

1. campaign별 재실행이 가능해야 한다.
2. 특정 campaign의 label correction이 전체 누적본을 직접 깨뜨리면 안 된다.
3. cumulative file은 concat/rebuild artifact로 취급한다.
4. campaign별 파일이 source of truth가 된다.

### 3. DINOv2 anchor activation timing

Phase 1b 첫 라운드는 모든 campaign을 cold_start로 돌린다.

phase_1b_anchor_policy:

first_round = cold_start_all_campaigns  
optional_second_round = anchor_aware_with_first_round_labels  
required = false  

이유:

Phase 1b의 핵심 목표는 anchor 정밀화가 아니라 campaign coverage와 human label 확보다.

Sequential anchor-aware 방식은 다음 장점이 있다.

campaign 1 label  
→ campaign 2에서 campaign 1의 accepted image를 family anchor로 사용 가능  

하지만 이 방식은 사람이 campaign 1을 먼저 라벨링해야 campaign 2 queue를 만들 수 있으므로 운영 속도를 늦춘다.

따라서 Phase 1b 첫 라운드는 다음 방식으로 한다.

campaign 1~5 retrieval  
→ campaign 1~5 review queue 생성  
→ 사람이 한 번에 라벨링  
→ ReviewEvent 누적 ingest  
→ TrainingSnapshot 누적 생성  

이후 시간이 있으면 optional second round로 anchor-aware queue를 생성한다.

optional second round에서 활성화할 feature:

dinov2_campaign_pos_nn_sim  
dinov2_campaign_neg_nn_sim  
dinov2_campaign_margin  
dinov2_campaign_pos_count  
dinov2_campaign_neg_count  
dinov2_campaign_anchor_missing = 0  

단, second round는 Phase 1b 성공 필수 조건이 아니다. 시간이 부족하면 Phase 2로 넘긴다.

### 4. Issue tag concentration failure criterion 보정

Phase 1a에서는 reject issue_tags가 semantic_mismatch에 강하게 쏠렸다.

semantic_mismatch = 12  
too_busy_background = 1  

이 현상은 preview renderer가 없는 상태에서는 자연스러울 수 있다. 사람이 실제 text box overlay를 보지 못하면 text_region_conflict, low_contrast, visual_hierarchy_weak 같은 layout-related issue_tags를 일관되게 쓰기 어렵다.

따라서 Phase 1b의 failure criterion을 다음처럼 분리한다.

#### 4a. Preview renderer v1이 없는 경우

issue_tags가 semantic_mismatch에 쏠리는 것은 즉시 실패가 아니다.

이 경우 status는 다음과 같다.

issue_tag_concentration_status = preview_renderer_trigger  

의미:

semantic_mismatch 쏠림은 review vocabulary 실패라기보다 preview renderer 도입 필요 신호로 본다.

#### 4b. Preview renderer v1이 있는 경우

preview renderer v1이 있는데도 issue_tags가 계속 semantic_mismatch에만 쏠리면 실패 또는 재설계 신호다.

가능한 원인:

1. reviewer UI가 layout issue를 충분히 보이게 하지 못함
2. issue_tags vocabulary가 reviewer에게 명확하지 않음
3. layout box overlay가 실제 text conflict를 잘 드러내지 못함
4. review instruction이 semantic 판단에만 치우침

이 경우 다음을 재검토한다.

review UI  
preview renderer  
issue_tags instruction  
layout-related examples  
critic_preview_renderer_v1 config  

수정된 실패 기준:

preview_renderer_v1_absent_and_semantic_mismatch_concentrated  
→ failure가 아니라 preview_renderer_trigger

preview_renderer_v1_present_and_semantic_mismatch_concentrated  
→ failure_or_review_tool_redesign_required
