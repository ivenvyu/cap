# Phase 1b Classifier Smoke Training Summary

## 메타데이터

spec_version: v2.2.1  
phase: phase_1b  
summary_name: phase_1b_classifier_smoke_summary  
related_filter_policy: configs/phase1b_training_filter_v1.yaml  
related_claim_support_policy: configs/diagnostic_claim_support_v1.yaml  
related_smoke_report: audit/phase_1b/phase_1b_classifier_smoke_report.json  
score_status: diagnostic_only  
threshold_status: no_calibrated_threshold  
candidate_support_explanation_status: deferred_from_phase_1b  
claim_level_diagnostic_support_status: active  

## 한 줄 결론

Phase 1b filtered training set을 사용한 classifier smoke training은 성공했다.

다만 이 결과는 **모델 성능 검증이 아니다.**  
이번 smoke training은 다음이 실제로 동작하는지 확인하기 위한 진단이다.

1. filtered TrainingSnapshot 로딩
2. PairFeatureSnapshot join
3. numeric feature matrix 생성
4. leave-one-campaign-out loop 실행
5. classifier 학습/예측 루프 실행
6. model serialization
7. diagnostic report 생성

따라서 이 결과로 production quality, calibrated threshold, automatic accept/reject를 주장하지 않는다.

## 입력 데이터

Smoke training에는 Phase 1b filtered set을 사용했다.

제외 campaign:

phase1b_indoor_gallery_winter_art

제외 이유:

이 campaign은 30개 review 후보 중 29개가 reject되었다.  
이 결과는 모델 품질 실패가 아니라 raw image pool의 indoor / winter / gallery coverage gap evidence로 해석한다.

따라서 해당 campaign은 audit에는 보존하되, classifier/ranker smoke training 및 evaluation claim에서는 제외했다.

## Dataset summary

사용된 campaign:

phase1b_architecture_exhibition_visit  
phase1b_autumn_garden_walk  
phase1b_botanical_spring_program  
phase1b_summer_garden_walk  

Rows:

total rows = 120

Campaign별 row 수:

phase1b_architecture_exhibition_visit = 30  
phase1b_autumn_garden_walk = 30  
phase1b_botanical_spring_program = 30  
phase1b_summer_garden_walk = 30  

Classifier label 분포:

0 = 70  
1 = 50  

해석:

negative가 positive보다 많지만, smoke training을 실행하기에는 충분하다.  
다만 production model training 또는 threshold calibration을 주장하기에는 데이터가 매우 작다.

## Feature matrix

생성된 numeric feature 수:

feature_count = 33

사용 feature:

brightness  
campaign_is_garden  
campaign_is_summer  
campaign_is_walking_program  
clip_margin  
clip_negative_max_sim  
clip_negative_mean_sim  
clip_positive_max_sim  
clip_positive_mean_sim  
clip_rank_percentile  
contrast  
dinov2_campaign_anchor_missing  
dinov2_campaign_neg_count  
dinov2_campaign_pos_count  
dinov2_cluster_review_count_coarse  
dinov2_cluster_review_count_fine  
dinov2_cluster_review_count_mid  
dinov2_duplicate_group_seen  
dinov2_family_anchor_missing  
dinov2_family_support_count  
edge_density  
image_category_course  
image_category_flower  
image_category_gallery  
image_category_tree  
image_season_unknown  
info_region_safe_score  
path_has_architecture  
path_has_garden  
required_region_safe_mean  
required_region_safe_min  
saturation  
title_region_safe_score  

해석:

FeatureSnapshot에서 classifier input matrix까지 연결되는 feature plumbing은 정상적으로 동작했다.

## Model

사용 모델:

sklearn LogisticRegression  
class_weight = balanced  
solver = lbfgs  
calibration = uncalibrated  

이 모델은 smoke training용이다.  
최종 reranker 모델도 아니고, calibrated classifier도 아니다.

## Leave-one-campaign-out diagnostic result

Out-of-fold 전체 결과:

accuracy = 0.5917  
balanced_accuracy = 0.5786  
roc_auc = 0.5977  
average_precision = 0.4792  

Confusion matrix:

labels = [0, 1]

[[46, 24],  
 [25, 25]]

Per-class result:

class 0:
  precision = 0.6479
  recall = 0.6571
  f1 = 0.6525
  support = 70

class 1:
  precision = 0.5102
  recall = 0.5000
  f1 = 0.5051
  support = 50

해석:

classifier smoke loop는 정상적으로 동작했다.  
하지만 metric 수준은 낮고, threshold도 calibrated되지 않았다.  
이 결과는 feature plumbing 검증용으로만 사용한다.

## Fold별 결과

### phase1b_architecture_exhibition_visit held-out

test_n = 30  
labels = {'0': 10, '1': 20}  
balanced_accuracy = 0.3500  
roc_auc = 0.2600  

해석:

architecture campaign은 다른 campaign에서 학습한 신호와 label 기준이 잘 맞지 않는다.  
이 fold는 사실상 반대로 작동하는 구간이므로, campaign shift가 실제로 존재한다는 강한 진단 신호다.

이 결과를 모델 실패로 단정하지 않는다.  
오히려 architecture campaign family에 대해 별도 feature, prompt, image pool coverage, review policy를 점검해야 한다는 신호로 본다.

### phase1b_autumn_garden_walk held-out

test_n = 30  
labels = {'0': 21, '1': 9}  
balanced_accuracy = 0.6349  
roc_auc = 0.6190  

해석:

autumn garden campaign은 season_mismatch가 많이 발생한 campaign이다.  
현재 classifier는 일부 season/semantic signal을 포착하지만, 충분히 안정적이라고 보기 어렵다.

### phase1b_botanical_spring_program held-out

test_n = 30  
labels = {'0': 20, '1': 10}  
balanced_accuracy = 0.6000  
roc_auc = 0.6150  

해석:

botanical campaign은 season_mismatch와 semantic_mismatch가 함께 나타났다.  
사진만 보고 봄/여름/초가을을 구분하기 어려운 경우가 있으므로, season_mismatch annotation rule을 계속 명확히 유지해야 한다.

현재 원칙:

Season mismatch는 시각적으로 명확하거나, 식물 개화 시기상 명확할 때만 hard reject 사유로 사용한다.  
봄/여름/초가을 구별이 애매한 경우에는 season만으로 reject하지 않고 acceptable로 둘 수 있다.

### phase1b_summer_garden_walk held-out

test_n = 30  
labels = {'0': 19, '1': 11}  
balanced_accuracy = 0.6220  
roc_auc = 0.7129  

해석:

summer garden campaign은 4개 fold 중 가장 양호하다.  
하지만 표본 수가 30개뿐이므로 metric을 성능 주장으로 해석하지 않는다.

## Coefficient diagnostic

Top absolute coefficients:

clip_positive_mean_sim = 0.9374  
clip_negative_max_sim = 0.6274  
clip_negative_mean_sim = -0.5340  
edge_density = 0.4840  
clip_positive_max_sim = 0.4796  
image_category_flower = -0.4575  
saturation = -0.3782  
image_category_tree = 0.3281  
path_has_architecture = 0.3251  
image_category_gallery = 0.2951  

주의:

이 coefficient들은 final explanation이 아니다.  
특히 clip_negative_max_sim의 positive coefficient는 negative prompt 유사도가 높을수록 accept 가능성이 커진 것처럼 보일 수 있다.

이는 다음 원인 때문에 생길 수 있다.

1. 작은 데이터 크기
2. campaign 간 confounding
3. prompt set 구성 문제
4. positive/negative prompt가 campaign family별로 다르게 작동
5. raw image pool coverage bias

따라서 coefficient는 feature sanity check 정도로만 본다.  
candidate-level support explanation으로 사용하지 않는다.

## Diagnostic claim support

Phase 1b에서는 candidate-level support explanation을 도입하지 않는다.

candidate-level support explanation은 특정 campaign-image-layout 후보가 어떤 axis에서 지지되는지 설명하는 artifact다. 이 설명은 empirical baseline, critic signal, 충분한 label data가 없으면 axis 이름 나열에 가까워질 위험이 있다. 따라서 candidate-level support explanation은 Phase 3~4로 defer한다.

다만 Phase 1b에서는 claim-level diagnostic support를 사용한다.

claim-level diagnostic support는 특정 후보의 accept/reject를 설명하지 않는다. 대신 다음과 같은 프로젝트 판단이 어떤 evidence에 의해 지지되고, 어떤 추가 관측이 그 판단을 강화하거나 약화시키는지 기록한다.

1. indoor/winter campaign을 smoke training에서 제외하는 판단
2. architecture fold를 campaign shift diagnostic으로 해석하는 판단
3. classifier smoke metric을 production quality claim으로 사용하지 않는 판단
4. preview renderer v1을 다음 unblocked task로 보는 판단

이 정책은 `configs/diagnostic_claim_support_v1.yaml`에 정의되어 있고, `scripts/validate_phase1b_specs.py`로 검증한다.

특히 `would_strengthen` / `would_weaken` entry에서 다음 status를 쓰는 경우 `next_action`이 필수다.

diagnostic_trigger_only_not_final_threshold  
diagnostic_trend_audit_only  
diagnostic_ablation_audit_only  
diagnostic_integrity_check_only  

반대로 `diagnostic_distribution_audit_only`는 record-only monitor signal로 취급하며, `default_next_action`을 통해 다음 audit에서 비교하도록 한다.

이 구분의 목적은 claim-level support를 단순 문서 주석이 아니라 운영 지침으로 만들기 위함이다.

중요한 제한:

1. diagnostic trigger는 final threshold가 아니다.
2. claim-level support는 candidate-level explanation이 아니다.
3. feature coefficient를 explanation으로 사용하지 않는다.
4. claim-level support는 automatic accept/reject rule이 아니다.
5. 모든 score와 metric은 diagnostic_only다.

## Smoke training이 증명한 것

이번 smoke training은 다음을 증명했다.

1. filtered TrainingSnapshot을 읽을 수 있다.
2. PairFeatureSnapshot과 feature_snapshot_id 기준으로 join할 수 있다.
3. 33개 numeric feature matrix를 만들 수 있다.
4. classifier label 0/1을 정상적으로 구성할 수 있다.
5. leave-one-campaign-out diagnostic loop가 돈다.
6. out-of-fold prediction을 만들 수 있다.
7. diagnostic metric report를 만들 수 있다.
8. model artifact를 저장할 수 있다.

## Smoke training이 증명하지 않은 것

이번 smoke training은 다음을 증명하지 않는다.

1. production reranker quality
2. calibrated accept/reject threshold
3. automatic accept/reject 가능성
4. ranker generalization
5. visual critic 성능
6. layout-aware poster quality
7. candidate-level support explanation의 유효성
8. raw pool expansion 이후의 성능

## 주요 진단

### 1. Architecture campaign shift

architecture_exhibition_visit fold가 가장 약했다.

balanced_accuracy = 0.35  
roc_auc = 0.26  

이는 architecture family가 garden/botanical campaign과 다른 decision boundary를 가질 수 있음을 보여준다.

후속 작업:

architecture campaign을 더 늘린다.  
architecture-specific prompt를 점검한다.  
path_has_architecture, image_category_gallery 같은 metadata feature가 실제로 도움이 되는지 재검토한다.  
건축 이미지와 정원 이미지가 duplicate path로 같이 등장하는 문제를 계속 audit한다.

### 2. Season axis는 강하지만 noise 위험이 있다

autumn / botanical / summer campaign에서 season_mismatch가 많이 나왔다.

다만 계절 판단은 이미지상 불확실한 경우가 많다.  
따라서 season_mismatch는 명확한 경우에만 hard reject로 사용한다.

### 3. Layout signal은 아직 부족하다

Phase 1b first round 전체에서 layout-related issue tag는 거의 없었다.

text_region_conflict = 1

이는 layout 문제가 없다는 뜻이 아니다.  
preview renderer가 없어서 reviewer가 text box conflict를 안정적으로 판단하기 어려웠다는 뜻일 가능성이 크다.

따라서 다음 unblocked task는 preview renderer v1이다.

### 4. Indoor/winter는 계속 coverage-gap diagnostic으로 유지한다

indoor_gallery_winter_art는 classifier smoke training에서 제외했다.

이 campaign은 삭제하지 않는다.  
raw pool expansion이 가능해진 뒤 재활성화한다.

## 다음 작업

현재 raw image pool expansion은 새 이미지가 없어 blocked 상태다.

따라서 다음 작업은 preview renderer v1이다.

목표:

1. raw image 위에 title/info box rectangle 표시
2. layout_top_left_bottom_left 우선 지원
3. translucent box fill 표시
4. reviewer가 text_region_conflict, low_contrast, too_busy_background를 더 쉽게 판단하게 함
5. Korean text rendering, final typography, gradient, kerning은 제외

Preview renderer v1은 final poster renderer가 아니다.  
reviewer audit용 box overlay placeholder다.

## 최종 결정

classifier smoke training = pass  
model quality claim = no  
threshold claim = no  
automatic accept/reject = no  

candidate-level support explanation = deferred  
claim-level diagnostic support = active  

다음 단계:

preview_renderer_v1
