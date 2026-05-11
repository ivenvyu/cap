— 전체 설계 개요


purpose: |
  6개월 뒤 본인 또는 다른 사람이 "왜 이렇게 결정했지?"를 reverse-engineer
  하지 않고 이 문서 한 장으로 답을 찾을 수 있게 한다.
```

---

## 1. 한 문장 요약

**10,000장 raw image 중에서 campaign에 맞는 이미지를 추천하는 시스템을, 사람이 metadata를 채우지 않고 pair decision만 주는 방식으로 만든다.**

기존 접근(metadata-first: season/category를 사람이 채움)이 10,000장 규모에서 불가능했기 때문에, embedding-first(CLIP/DINOv2)로 후보를 만들고 사람은 최종 판단만 하는 구조로 바꿨다.

---

## 2. 직관 — 왜 이런 구조가 나왔나

### 출발점의 문제

```text
사유원 마케팅팀:
  - 5만장 visual asset 보유
  - 매번 홍보물 만들 때마다 사람이 전수 탐색
  - season/category metadata 없음
  - "안개 낀 풍설기천년" 같은 검색이 불가능

Capstone team:
  - reranker_v1_1_antileak이 Recall@1=1.0 (leakage 또는 암기)
  - 81개 완성 포스터로 학습했는데 추천 대상은 raw 사진
  - cap_project-main과 cap_project_template 두 repo의 데이터 불연결
```

### 4개의 큰 결정

| 결정 | 이전 방식 | 새 방식 | 왜 |
|---|---|---|---|
| **무엇을 학습?** | 한 모델이 다 함 | 4개 모델로 분리 | 81개 데이터로 end-to-end는 불가능 |
| **사람의 역할?** | metadata 채우기 | pair decision만 | 10,000장 metadata는 절대 안 끝남 |
| **후보 생성?** | 태그 매칭 | CLIP retrieval + DINOv2 expansion | 태그가 없으니 embedding으로 |
| **품질 평가?** | top-N score | 7-bucket discovery sampling | top-N만 보면 confirmation bias |

---

## 3. 4개 모델 — 역할 분리

```text
┌──────────────────────────────────────────────────────────────┐
│  A. Template Planner    (이미 작동: 79% 정확도)              │
│     brief → layout_type / text_region / overlay_type         │
├──────────────────────────────────────────────────────────────┤
│  B. BBox Layout         (지금 fine-tune 안 함, retrieval만)  │
│     layout_family → reference bank에서 좌표 가져옴           │
├──────────────────────────────────────────────────────────────┤
│  C. Image Reranker      (← v2.2.1의 주제)                   │
│     campaign × raw_image → fit_score                         │
├──────────────────────────────────────────────────────────────┤
│  D. Visual Critic       (Phase 4에서 분리)                   │
│     image + layout → text_region_conflict_score 등           │
└──────────────────────────────────────────────────────────────┘
```

v2.2.1이 다루는 건 **C (Image Reranker)** 이고, 그 안에 D (Critic)가 부분적으로 통합된다.

---

## 4. C (Reranker)의 내부 구조 — 4개 컴포넌트

```text
                    ┌─────────────────────────────────────┐
                    │ campaign metadata + layout spec     │
                    └────────────────┬────────────────────┘
                                     │
                    ┌────────────────▼────────────────────┐
                    │ prompt_template_bank_v1             │
                    │ (한국어 → 영어 visual descriptor)    │
                    └────────────────┬────────────────────┘
                                     │
   ┌─────────────────────────────────┼──────────────────────────────┐
   │                                 │                              │
   ▼                                 ▼                              ▼
┌──────────┐                  ┌─────────────┐               ┌────────────┐
│  CLIP    │                  │  DINOv2     │               │  Layout    │
│ (의미)   │                  │ (시각 유사) │               │  Safety    │
└────┬─────┘                  └──────┬──────┘               └──────┬─────┘
     │                               │                             │
     │ campaign-text와                │ accepted 이미지와            │ 텍스트 얹을
     │ 의미적으로 맞는가?              │ 시각적으로 가까운가?         │ 공간 있는가?
     │                               │                             │
     └─────────────┬─────────────────┴─────────────────────────────┘
                   │
                   ▼
        ┌─────────────────────────┐         ┌──────────────────┐
        │  LightGBM Reranker      │ ◄────── │  Visual Critic   │
        │  (feature combiner)     │         │  (Phase 4부터)   │
        └────────────┬────────────┘         └──────────────────┘
                     │
                     ▼
        ┌─────────────────────────┐
        │  7-bucket Review Queue  │
        │  (discovery sampling)   │
        └────────────┬────────────┘
                     │
                     ▼
        ┌─────────────────────────┐
        │  사람: pair decision    │
        │  (reject/.../best)      │
        └────────────┬────────────┘
                     │
                     ▼
              ReviewEvent log
                     │
                     ▼
        ┌─────────────────────────┐
        │  Training Snapshot      │
        │  (classifier / ranker)  │
        └─────────────────────────┘
                     │
                     └──── retrain ────┐
                                       │
                                       ▼
                              (Reranker 다시 학습)
```

---

## 5. CLIP과 DINOv2의 역할 분리 — 가장 중요한 분리

| | **CLIP** | **DINOv2** |
|---|---|---|
| 무엇을? | text ↔ image 의미 매칭 | image ↔ image 시각 유사도 |
| 입력 | 영어 visual descriptor | 이미지만 |
| 강점 | semantic alignment | fine-grained visual similarity |
| 약점 | 한국어 약함, 시각 디테일 약함 | text 이해 불가 |
| 역할 | **1차 검색 (10,000 → top-N)** | **2차 확장 + cluster + duplicate** |
| 예시 입력 | "calm summer garden walking path" | accept된 이미지 주변 후보 확장 |

**핵심 통찰**: 둘이 같은 자리에 쓰면 안 된다. CLIP은 retrieval 단계, DINOv2는 expansion/diversity 단계.

### 한국어 처리

CLIP은 한국어가 약하므로 **prompt_template_bank**가 한국어 metadata를 영어 visual descriptor로 변환한다.

```yaml
# 예시
campaign:
  season: summer
  purpose_type: walking_program
  space_type: garden
  mood_tags: [calm_elegant]

→ prompt_template_bank_v1:
  positive:
    - "calm summer garden walking path"
    - "green trees and peaceful botanical garden"
    - "elegant nature trail poster background"
  negative:
    - "indoor gallery exhibition"
    - "performance stage"
    - "winter landscape"
```

사유원 고유 공간명("풍설기천년", "오랑오랑")은 텍스트로 변환할 수 없으므로 **image_anchor_query** 모드로 처리한다 — 그 공간의 대표 이미지를 CLIP query로 사용해 image-image similarity로 retrieval.

---

## 6. DINOv2 anchor 설계 — global centroid의 함정

### 잘못된 방식 (v2.0 초기 설계)

```text
all_positive_images의 평균 임베딩을 global centroid로
```

→ campaign이 늘어나면 "여름 정원"과 "겨울 건축물" positive가 평균돼서 의미 없는 점이 됨.

### v2.1에서 수정된 방식

```text
1. campaign-local anchor
   - dinov2_campaign_pos_nn_sim
   - dinov2_campaign_pos_count
   - dinov2_campaign_anchor_missing  ← cold-start 처리

2. campaign-family anchor (similarity feature vector)
   - same_purpose_type, same_space_type, same_season, mood_overlap
   - dinov2_family_pos_nn_sim
   - dinov2_family_support_count

3. hierarchical cluster
   - dinov2_cluster_id_coarse (K=20)
   - dinov2_cluster_id_mid (K=100)
   - dinov2_cluster_id_fine (K=500)
   - cluster_positive_rate + cluster_review_count

4. duplicate group
   - augmentation manifest의 parent_image_id
   - DINOv2 near-duplicate (empirical percentile threshold)
```

---

## 7. 7-bucket Review Queue — confirmation bias 차단의 핵심

### Cold-start (label < 200, held-out campaigns < 5)

```text
LightGBM 안 믿음. CLIP/DINO/cluster 중심 discovery.

┌──────────────────────────────────────────────┐
│ Bucket A: CLIP high / model low      primary │  ← missed positive
│ Bucket B: DINO anchor high / model low primary│
│ Bucket F: Cluster diversity        secondary │  ← coverage
│ Bucket E: Layout-safe coverage     secondary │
│ Bucket  : LightGBM top              limited │
│ Bucket  : Uncertainty               limited │
│ Bucket G: Random                    minimal │
└──────────────────────────────────────────────┘
```

### Trained (label ≥ 200, held-out campaigns ≥ 5)

```text
LightGBM 점수 + 모델 간 disagreement 중심.

┌──────────────────────────────────────────────┐
│ Bucket  : LightGBM top              primary  │
│ Bucket D: Critic high / Reranker high primary│  ← cross-loop bias 차단
│ Bucket  : Uncertainty             secondary  │
│ Bucket A,B: model_disagreement    secondary  │
│ Bucket F: Cluster diversity         limited  │
│ Bucket G: Random                    minimal  │
└──────────────────────────────────────────────┘
```

### Critic unavailable fallback (Phase 4 이전)

Bucket D를 비워두지 않고 다음으로 대체:

```text
1순위: clip_high / model_low
2순위: dinov2_anchor_high / model_low
3순위: model_high / clip_negative_high
4순위: classifier_ranker_disagreement
```

**핵심 통찰**: top-N 정렬은 confirmation bias loop를 만든다. CLIP-high/model-low 같은 disagreement bucket이 매 review마다 "모델이 놓쳤을 가능성"을 강제로 노출한다.

---

## 8. Visual Critic — cross-loop bias 차단

### Critic의 task

```text
이 raw image + layout + text가 시각적으로 사용 가능한가?
```

출력 score:
- `critic_text_region_conflict_score`
- `critic_low_contrast_score`
- `critic_too_busy_background_score`
- `critic_visual_hierarchy_risk_score`

### 왜 reranker queue에서 학습하면 안 되는가

```text
Reranker가 layout_safety 낮은 후보를 queue에 적게 넣음
→ 사람이 text_region_conflict 라벨을 적게 만남
→ Critic 학습 데이터에 hard case 부족
→ Critic이 약한 영역에서만 학습됨
→ 그 critic score가 다시 reranker feature로 들어감
→ Reranker가 그 영역 후보를 더 적게 queue에 넣음

이건 single-loop confirmation bias가 아니라
reranker ↔ critic cross-loop confirmation bias.
```

### 해결: Critic 전용 sampling pool

```yaml
critic_training_pool:
  layout_risk_sampling:    # 의미는 맞는데 텍스트 얹기 어려운 케이스 강제 노출
  contrast_risk_sampling:  # 배경-텍스트 색상 충돌 케이스
  busy_background_sampling: # edge_density 높은 케이스
  preview_rendering_pool:  # 실제 layout overlay된 preview를 사람에게 노출
```

### Preview renderer 단계 분리

v1.4에서 한글 폰트 깨짐, PPTX 박스 겹침 등으로 실패한 영역이라 두 단계로 나눔:

```text
critic_preview_renderer_v1: box overlay placeholder만 (영문 "TITLE", "DATE")
  → "이 위치에 텍스트 들어가면 충돌"만 안정적으로 라벨링

critic_preview_renderer_v2: 한글 reflow + gradient overlay (prerequisite: v1 안정)
```

---

## 9. 6 Phase Implementation 순서

```text
Phase 0: Schema 작성          ◄── 코드 전 단계 (지금)
         12개 config/schema 파일 + design log

Phase 1a: 150장 dry run       end-to-end pipeline 검증 (성능 X)
Phase 1b: 1,000~5,000장       실제 cold-start 운영
Phase 1c: 10,000+             production scale (GPU 예산 필요)

Phase 2: prompt bank + retrieval
Phase 3: cold_start review queue 운영, label 모음
Phase 4: Critic 분리 (전용 sampling pool + preview renderer)
Phase 5: Ranker 전환 (LGBMClassifier → LGBMRanker)
Phase 6: held-out test campaign 검증, production
```

**핵심 통찰**: Phase 0가 가장 큰 변화. 이전 라운드에선 코드부터 짰다가 schema migration 비용이 폭발했다. 이번엔 schema를 먼저 박는다.

### Phase 0 → Phase 1a 전환 의존성

```text
issue_tags_v1.yaml
  └→ review_event.schema.json (enum import)
       └→ pair_feature_snapshot.schema.json (외래키)
            └→ training_snapshot.schema.json (입력)
                 └→ pair_feature_snapshot_storage_v1.yaml (저장 정책)
                      └→ training_snapshot_aggregation_v1.yaml (집계 규칙)
                           └→ review_queue_policy_v2_2.yaml (stage trigger)
                                └→ calibration_policy_v1.yaml (threshold)
                                     └→ critic_preview_renderer_v1.yaml
                                          └→ prompt_template_bank_v1.yaml
                                               └→ prompt_template_bank_governance_v1.yaml
                                                    └→ phase_1a_exit_criteria.yaml (모든 선행 참조)
                                                         └→ docs/design_history/v2_2_1_freeze.md
```

---

## 10. 5라운드의 진화 궤적

```text
v1 (initial)
  4-task 분리 (Template/BBox/Recommender/Critic)
  fine-tuning 전략 — 외부 데이터 + 사유원 데이터
  ↓
v2.0
  metadata-first → embedding-first 전환
  CLIP + DINOv2 + LightGBM + human pair decision
  ↓
v2.1
  confirmation bias 차단 (cold_start vs trained bucket)
  global centroid → campaign/family/cluster anchor
  Visual Critic 분리
  ↓
v2.2
  Phase 0 = "코드 전 schema 작성" 단계 분리
  critic unavailable fallback
  calibration 절차 (target/metric/trigger/fallback)
  preview renderer v1/v2 분리
  Phase 1을 1a/1b/1c로 분할
  ↓
v2.2.1 ◄── FROZEN
  pair_feature_snapshot 저장 정책
  phase_1a_exit_criteria
  training_snapshot_aggregation (multi-annotator, overwrite)
  prompt_template_bank governance + image_anchor_query
```

각 라운드가 바로 직전 라운드의 약점을 메웠다. v2.2 → v2.2.1은 architecture가 아니라 정책 보강만이라서, 이건 정상적인 수렴 신호다.

---

## 11. 12개 spec 파일 — 무엇을 만들 것인가

```text
configs/ (9개)
  ├── issue_tags_v1.yaml
  │     사람이 reject할 때 선택할 수 있는 닫힌 vocabulary
  │     critic_trainable boolean으로 학습 대상 명시
  │     → critic 학습 가능 tag(layout) vs reranker negative tag 구분
  │
  ├── review_queue_policy_v2_2.yaml
  │     cold_start / trained stage별 7-bucket allocation
  │     critic unavailable fallback 정책
  │
  ├── calibration_policy_v1.yaml
  │     layout_safety / duplicate / uncertainty의 threshold 결정 절차
  │     uncertainty는 metric calibration 대상 아님 (percentile band only)
  │
  ├── prompt_template_bank_v1.yaml
  │     한국어 campaign metadata → 영어 CLIP prompt 매핑
  │
  ├── prompt_template_bank_governance_v1.yaml
  │     prompt bank 운영 (ownership, versioning, fallback)
  │     image_anchor_query 모드 (한국어 고유 공간명 처리)
  │
  ├── critic_preview_renderer_v1.yaml
  │     critic 학습용 preview 렌더링
  │     v1: box placeholder, v2: 한글 reflow (prerequisite)
  │
  ├── pair_feature_snapshot_storage_v1.yaml
  │     feature snapshot 저장 path/granularity/immutability
  │     write_once + indefinite retention
  │
  ├── phase_1a_exit_criteria.yaml
  │     150장 dry run을 언제 "끝났다"고 선언할 것인가
  │     verification 방법 + auto_check boolean
  │
  └── training_snapshot_aggregation_v1.yaml
        ReviewEvent 여러 개를 어떻게 training label로 집계?
        primary_key: (pair_id, layout_spec_id, preview_renderer_version)
        multi_annotator: majority_vote_with_tie_break
        critic_snapshot: exclude_if_layout_label_disagreement

schemas/ (3개)
  ├── review_event.schema.json
  │     feature_snapshot_id (외래키)
  │     model_score_at_review (review 시점 모델 점수 보존)
  │
  ├── pair_feature_snapshot.schema.json
  │     v2.1의 모든 feature (CLIP/DINO/layout/critic/weak metadata)
  │
  └── training_snapshot.schema.json
        classifier 또는 ranker 학습용 최종 label

docs/design_history/ (1개)
  └── v2_2_1_freeze.md  (왜 이렇게 결정했는지 기록)
```

각 파일 상단에는 공통 metadata:

```yaml
metadata:
  spec_version: v2.2.1
  freeze_date: 2026-05-10
  decision_log: docs/design_history/v2_2_1_freeze.md
  status: frozen
  superseded_by: null
```

---

## 12. ReviewEvent schema — 가장 중요한 외래키

```json
{
  "review_event_id": "rev_20260510_001234",
  "timestamp": "2026-05-10T14:23:11Z",
  "annotator_id": "human_001",

  "pair_id": "campaign_001__raw_000123",
  "campaign_id": "campaign_001",
  "image_id": "raw_000123",
  "duplicate_group_id": "dup_grp_0042",

  "review_context": {
    "queue_version": "review_queue_v3",
    "queue_stage": "cold_start",
    "source_bucket": "clip_high_model_low",
    "preview_renderer_version": null,
    "layout_spec_id": "layout_001"
  },

  "decision": {
    "label": "reject",
    "issue_tags": ["text_region_conflict", "season_mismatch"],
    "preference_rank": null,
    "notes": "텍스트 영역과 겨울 느낌이 문제"
  },

  "feature_snapshot_id": "feat_v2_2_1__campaign_001__raw_000123",

  "model_score_at_review": {
    "lightgbm_classifier_v0": 0.71,
    "lightgbm_ranker_v0": null,
    "critic_v0": null,
    "clip_positive_max_sim": 0.34,
    "dinov2_campaign_pos_nn_sim": null
  }
}
```

**왜 feature_snapshot_id가 핵심인가**: feature schema가 v2.2 → v2.3로 바뀌어도 과거 ReviewEvent를 그대로 활용 가능. 새 snapshot_version으로 분기되고 과거 snapshot은 immutable 보존.

**왜 model_score_at_review가 핵심인가**: "어느 모델이 이 pair를 어느 score로 보고 사람이 review했는가"를 보존해야 model_disagreement가 시간이 지나도 분석 가능.

---

## 13. Pair Feature Schema (v2.1)

```json
{
  "pair_id": "campaign_001__raw_000123",
  "campaign_id": "campaign_001",
  "image_id": "raw_000123",
  "features": {
    "clip_positive_max_sim": 0.34,
    "clip_positive_mean_sim": 0.30,
    "clip_negative_max_sim": 0.12,
    "clip_negative_mean_sim": 0.09,
    "clip_margin": 0.22,
    "clip_rank_percentile": 0.91,

    "dinov2_campaign_pos_nn_sim": 0.82,
    "dinov2_campaign_neg_nn_sim": 0.37,
    "dinov2_campaign_margin": 0.45,
    "dinov2_campaign_pos_count": 4,
    "dinov2_campaign_anchor_missing": 0.0,

    "dinov2_family_pos_nn_sim": 0.76,
    "dinov2_family_neg_nn_sim": 0.41,
    "dinov2_family_margin": 0.35,
    "dinov2_family_support_count": 18,
    "dinov2_family_anchor_missing": 0.0,

    "dinov2_cluster_positive_rate_coarse": 0.41,
    "dinov2_cluster_positive_rate_mid": 0.63,
    "dinov2_cluster_positive_rate_fine": 0.50,
    "dinov2_cluster_review_count_mid": 12,
    "dinov2_duplicate_group_seen": 0.0,

    "required_region_safe_mean": 0.74,
    "required_region_safe_min": 0.61,
    "title_region_safe_score": 0.78,
    "info_region_safe_score": 0.70,
    "edge_density": 0.09,
    "contrast": 0.23,
    "brightness": 0.48,

    "critic_text_region_conflict_score": 0.18,
    "critic_low_contrast_score": 0.24,
    "critic_too_busy_background_score": 0.31,

    "image_season_unknown": 1.0,
    "image_category_gallery": 1.0,
    "path_has_architecture": 0.0,
    "path_has_garden": 1.0,

    "campaign_is_summer": 1.0,
    "campaign_is_walking_program": 1.0,
    "campaign_is_garden": 1.0
  }
}
```

핵심 feature 우선순위:
1. CLIP semantic match
2. DINOv2 visual anchor / cluster
3. critic / layout safety
4. weak metadata flags (보조)

---

## 14. Training Snapshot — Classifier vs Ranker 분리

### Classifier snapshot (cold-start)

```text
reject = 0
acceptable = 1
accept = 1
best = 1
```

용도: diagnostic reranker, review queue helper.

### Ranker snapshot (Phase 5+)

```text
reject = 0
acceptable = 1
accept = 2
best = 3

group = campaign_id  ← 같은 campaign 내부 ranking만 학습
cross-campaign score 직접 비교 금지
```

두 snapshot을 별도 파일로 보존:
```
training_label_snapshot_classifier_vN.jsonl
training_label_snapshot_ranker_vN.jsonl
```

### Multi-annotator 집계 규칙

```yaml
training_snapshot_aggregation_v1:
  primary_key:
    - pair_id
    - layout_spec_id
    - preview_renderer_version
    # 이 셋이 모두 같아야 같은 review context

  same_annotator_overwrite:
    method: most_recent

  multi_annotator_aggregation:
    method: majority_vote_with_tie_break
    tie_break: most_recent
    log_disagreement: true

  critic_snapshot:
    exclude_if_layout_label_disagreement: true
    # critic은 noisy label에 치명적이므로 disagreement 있으면 제외
```

---

## 15. Evaluation policy

### Split

```text
train campaigns
validation campaigns
test campaigns
```

같은 campaign 내부 random split 금지 (`campaign_id`가 leakage 채널).

### Leakage 차단

```text
parent_image_id        같으면 train/test 분리 금지
duplicate_group_id     같으면 train/test 분리 금지
augmentation_source    같으면 train/test 분리 금지
```

### Metric

```text
Recall@K
NDCG@K
MRR
pairwise win rate
hard negative win rate
coverage diversity
duplicate exposure rate
critic conflict rate
```

**campaign 1개에서 metric 1.0은 성능이 아니다 — in-sample diagnostic only.**

(이건 reranker_v1_1_antileak이 빠진 함정이다.)

---

## 16. 핵심 통찰 7개 — 외워두면 좋은 것

```text
1. metadata-first는 10,000장에서 절대 안 끝난다.
   사람은 metadata가 아니라 pair decision만 준다.

2. CLIP과 DINOv2는 같은 자리에 쓰지 않는다.
   CLIP은 retrieval, DINOv2는 expansion.

3. 한국어 campaign text를 CLIP에 직접 넣지 않는다.
   metadata-driven 영어 prompt template으로 변환.
   고유 공간명("풍설기천년")은 image-anchor query.

4. global positive centroid는 함정이다.
   campaign/family/cluster-conditional anchor로.

5. top-N 정렬은 confirmation bias loop를 만든다.
   discovery-oriented bucket sampling이 답.

6. Critic은 reranker queue에서 학습하면 cross-loop bias.
   전용 sampling pool + preview renderer로 분리.

7. Schema를 먼저 박지 않으면 Phase 5에서 migration 비용 폭발.
   ReviewEvent.feature_snapshot_id + write_once snapshot 저장.
```

---

## 17. 다음 단계

### 지금 (Phase 0)

12개 spec 파일 + 1개 design log 작성. 순서:

```text
1. issue_tags_v1.yaml
2. review_event.schema.json
3. pair_feature_snapshot.schema.json
4. training_snapshot.schema.json
5. pair_feature_snapshot_storage_v1.yaml
6. training_snapshot_aggregation_v1.yaml
7. review_queue_policy_v2_2.yaml
8. calibration_policy_v1.yaml
9. critic_preview_renderer_v1.yaml
10. prompt_template_bank_v1.yaml
11. prompt_template_bank_governance_v1.yaml
12. phase_1a_exit_criteria.yaml
13. docs/design_history/v2_2_1_freeze.md
```

### 그 다음 (Phase 1a)

- raw_gallery 경로 복구 (cap_project-main → cap_project_template)
- reranker_v1_1_antileak 폐기
- toy active-learning loop를 smoke_test/로 격리
- 150장으로 CLIP/DINOv2 embedding extraction
- duplicate_group_id, cluster_id, region_safety_map 생성
- pair_feature_snapshot v2.2 생성 dry run
- cold_start review queue end-to-end 동작 확인

### Critique 재개 시점

**Phase 1a end-to-end가 한 번 돌고 실제 데이터/로그가 나온 뒤에만**.
종이 위 라운드는 v2.2.1로 종료.

---

## 부록 — 폐기된 것들

```text
✗ reranker_v1_1_antileak (Recall@1=1.0 — leakage 또는 암기, 신뢰 불가)
✗ golden_layout.py 직접 좌표 생성 방식 (외부 layout bank로 대체)
✗ end-to-end VLM fine-tuning (81개 데이터로 불가능)
✗ metadata 수동 라벨링 기반 reranker (10,000장에서 비현실적)
✗ 단일 K cluster (hierarchical로 대체)
✗ global positive centroid (campaign/family/cluster anchor로 대체)
✗ top-N review queue 정렬 (7-bucket discovery sampling으로 대체)
✗ "Classifier에서 Ranker로 자연스럽게 이행" (snapshot 분리)
✗ 고정 threshold (calibration_policy로 대체)
```
