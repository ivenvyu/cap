# Ontology DB Scaffold Scope — Phase 1b

## 메타데이터

spec_version: v2.2.1  
phase: phase_1b  
status: active  
db_role: operational_source_of_truth  
score_status: diagnostic_only  
ontology_status: scaffold_ready_not_complete  

## 한 줄 결론

현재 목표는 완성된 이미지 태깅 온톨로지를 만드는 것이 아니다.

현재 목표는 샘플 데이터 206장을 DB source of truth로 정규화하고, 향후 수천~수만 장으로 확장될 때 계층적 이미지 태깅 온톨로지를 누적 구축할 수 있는 DB scaffold와 labeling loop를 만드는 것이다.

즉 현재 성과는 다음이다.

1. ontology-ready DB scaffold
2. provenance-preserving import structure
3. seed tag vocabulary
4. DINOv2 cluster 기반 labeling queue
5. future cluster/image tag assertion infrastructure

현재 DB는 완전한 ontology coverage를 주장하지 않는다.

## 프로젝트 요구와의 연결

프로젝트 요구는 다음을 포함한다.

- 계층적 다중 관점 이미지 태깅 온톨로지
- 공간, 시간, 피사체, 무드, 용도 등 다중 축 metadata
- Top-down 기획만이 아니라 이미지 임베딩 실험과 전체 이미지 분포 분석 기반의 metadata 체계 구축
- 향후 대규모 이미지 풀에서도 검색 최적화 가능한 구조

현재 Phase 1b는 이 요구를 완성하지 않는다.

대신 다음을 가능하게 하는 기반을 구축한다.

```text
sample image artifacts
→ DB source of truth
→ DINOv2 cluster distribution
→ cluster-level labeling queue
→ cluster_tag_assertions
→ propagated image_tag_assertions
→ future ontology-based retrieval
```

## 현재 데이터 규모의 한계

현재 이미지 풀은 206장이다. 이 규모만으로는 완전한 계층적 이미지 태깅 온톨로지를 만들 수 없다.

특히 다음 축들은 아직 sparse하거나 비어 있다.

- season / time / weather
- mood
- usage
- visitor / sculpture / object-level subject
- design affordance

따라서 현재 단계에서 다음을 주장하지 않는다.

1. 완성된 ontology
2. 충분한 tag coverage
3. calibrated tag propagation
4. production search quality
5. final RAG retrieval quality

## 현재 DB에 들어간 핵심 구조

현재 DB는 다음 주요 table을 가진다.

### Core asset layer

- images
- campaigns

### Computed visual artifact layer

- image_embeddings
- image_duplicates
- image_clusters
- image_regions
- retrieval_candidates
- pair_features

### Review and label layer

- review_events
- training_snapshots
- training_sets
- training_set_items

### Ontology seed layer

- tag_axes
- tag_values
- cluster_label_queue
- cluster_tag_assertions
- image_tag_assertions

### Provenance layer

- db_builds
- artifact_sources
- import_batches

## 현재 검증된 DB 수치

DB foundation 기준 검증값:

```text
images = 206
image_embeddings = 412
image_duplicates = 206
image_clusters = 206
image_regions = 3296

campaigns = 6
retrieval_candidates = 300
pair_features = 300
review_events = 180

training_snapshots = 360
training_sets = 2
training_set_items = 240

tag_axes = 7
tag_values = 52
```

Phase 1b filtered training set:

```text
phase1b_filtered_classifier_v1:
  rows = 120
  labels = {0: 70, 1: 50}

phase1b_filtered_ranker_v1:
  rows = 120
  labels = {0: 70, 1: 24, 2: 26}
```

Cluster label queue:

```text
cluster_level = coarse
cluster_label_queue rows = 20
representative images = 93
representative_method = dinov2_centroid_medoid_plus_greedy_diversity
score_status = diagnostic_only
```

## Seed ontology axes

현재 seed tag vocabulary는 다음 축을 가진다.

1. space_axis
2. temporal_axis
3. weather_light_axis
4. subject_axis
5. mood_axis
6. usage_axis
7. design_affordance_axis

이 vocabulary는 최종 ontology가 아니다. 향후 human cluster labels, image-level corrections, search failure analysis에 따라 확장될 수 있는 seed vocabulary다.

## Cluster labeling loop

현재 구축된 cluster labeling loop는 다음과 같다.

```text
DINOv2 image embeddings
→ image_clusters
→ coarse cluster 20개
→ cluster별 representative images 선택
→ cluster_label_queue 생성
→ 사람이 cluster-level tag label 작성
→ cluster_tag_assertions 저장
→ image_tag_assertions로 전파
```

대표 이미지 선택 방식:

```text
dinov2_centroid_medoid_plus_greedy_diversity
```

의미:

1. cluster centroid에 가까운 대표 이미지 1장 선택
2. 나머지는 cluster 내부 다양성을 보존하도록 선택
3. 사람이 cluster 의미를 빠르게 파악할 수 있도록 구성

이 방식은 tag 품질을 보장하지 않는다. 단지 human ontology labeling 비용을 줄이기 위한 diagnostic queue 생성 방식이다.

## 왜 cluster-level labeling인가

이미지 206장을 모두 하나씩 태깅할 수도 있지만, 향후 수천~수만 장으로 확장되면 image-level full labeling은 비효율적이다.

따라서 현재 전략은 다음이다.

```text
cluster-level label first
→ propagated image-level tag assertions
→ uncertain cluster / search failure / edge case만 image-level correction
```

이 방식은 프로젝트 요구의 “이미지 임베딩을 통한 실험과 전체 이미지 분포 분석 기반 ontology 구축”에 직접 대응한다.

## Assertion 설계

향후 tag는 단순 컬럼 값이 아니라 assertion으로 저장한다.

### cluster_tag_assertions

cluster 단위로 붙은 tag.

예:

```text
cluster_id = coarse_0007
tag_id = mood_axis:quiet
label_source = human_cluster_label
confidence_status = diagnostic_cluster_label
```

### image_tag_assertions

image 단위로 붙은 tag.

가능한 source:

```text
human_image_label
human_cluster_label_propagation
folder_derived
model_suggested_unverified
search_failure_correction
```

이렇게 해야 나중에 “사람이 직접 붙인 태그”와 “cluster에서 전파된 태그”를 구분할 수 있다.

## 현재 non-claims

현재 DB scaffold는 다음을 주장하지 않는다.

1. ontology tag coverage가 충분하다고 주장하지 않는다.
2. cluster label이 모든 cluster member에 정확히 적용된다고 주장하지 않는다.
3. propagated tag를 calibrated confidence로 해석하지 않는다.
4. search quality를 production 수준으로 주장하지 않는다.
5. design quality 또는 generated poster quality를 주장하지 않는다.
6. candidate-level support explanation을 생성하지 않는다.

## 다음 단계

다음 단계는 ontology tag completion이 아니라, ontology tagging loop 검증이다.

우선 coarse cluster 20개 중 일부 또는 전체에 대해 사람이 cluster-level tag를 작성한다.

입력 파일:

```text
data/review/ontology/cluster_label_queue_v1/cluster_label_queue__coarse.csv
```

검토 HTML:

```text
data/review/ontology/cluster_label_queue_v1/cluster_label_queue__coarse.html
```

작성할 컬럼:

```text
space_axis_tags
temporal_axis_tags
weather_light_axis_tags
subject_axis_tags
mood_axis_tags
usage_axis_tags
design_affordance_axis_tags
confidence_status
notes
```

이후 필요한 스크립트:

1. ingest_cluster_tag_labels_to_db.py
2. propagate_cluster_tags_to_images.py
3. validate_ontology_tag_assertions.py
4. export_ontology_search_index_from_db.py

## 최종 정리

현재 DB는 완성된 ontology가 아니다.

현재 DB는 다음을 가능하게 하는 source-of-truth scaffold다.

```text
sample artifacts
→ validated DB
→ embedding clusters
→ cluster label queue
→ tag assertions
→ ontology search
→ future RAG retrieval
```

Phase 1b에서의 올바른 성공 기준은 완전한 태그 coverage가 아니라, 향후 확장 가능한 ontology tagging loop가 DB 위에서 재현 가능하게 작동하는 것이다.
