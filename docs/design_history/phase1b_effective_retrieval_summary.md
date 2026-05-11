# Phase 1b Effective Retrieval 요약

## 목적

Phase 1b의 목적은 완성된 온톨로지를 주장하는 것이 아니라, 현재 샘플 이미지와 검토 결과를 DB source of truth로 묶고, 이후 더 큰 이미지 풀에서도 확장 가능한 검색 후보 생성 구조를 만드는 것이다.

이번 단계에서 핵심적으로 다룬 문제는 다음과 같다.

1. review 결과와 training snapshot이 파일 단위로 흩어져 있었다.
2. image, campaign, retrieval candidate, feature, review label, training item 사이의 관계를 한 곳에서 확인하기 어려웠다.
3. 계절 campaign에서 계절과 맞지 않는 꽃 이미지가 후보로 섞였다.
4. classifier smoke metric과 retrieval 후보 품질 평가가 섞여 해석될 위험이 있었다.
5. ontology tag, plant bloom prior, retrieval eligibility를 검색/RAG용 index로 연결하는 재현 가능한 경로가 필요했다.

## DB source of truth 전환

Phase 1b에서는 다음 artifact들을 DB로 적재했다.

- images
- image embeddings
- duplicate groups
- DINOv2 clusters
- image regions
- campaigns
- retrieval candidates
- pair features
- review events
- training snapshots
- training sets
- training set items
- ontology tag axes and values

DB는 다음 역할을 한다.

- 운영 기준 source of truth
- review label과 retrieval candidate의 join 기준
- training snapshot과 feature snapshot의 연결 기준
- ontology tag propagation의 저장소
- effective retrieval view의 기준
- search index export의 입력

최종 DB 검증에서 foreign key check는 통과했다.

## Ontology scaffold 범위

현재 온톨로지는 완성된 coverage를 주장하지 않는다.

현재 구축한 것은 다음 흐름이다.

1. DINOv2 cluster 기반 cluster label queue 생성
2. 사람이 검토한 coarse cluster label CSV ingest
3. cluster tag assertion 저장
4. image tag assertion으로 propagation
5. image-level ontology tag를 search index로 export

현재 image tag assertion은 coarse cluster 기반 전파 결과다.  
따라서 image 하나하나에 대해 완전한 수동 ontology label이 붙었다고 해석하면 안 된다.

현재 ontology tag는 다음 축을 중심으로 생성되었다.

- space_axis
- subject_axis
- mood_axis

Phase 2에서는 더 많은 이미지와 더 정교한 label policy가 들어온 뒤 axis coverage를 확장해야 한다.

## 계절 문제 처리 방향

초기에는 계절을 image temporal_axis label로 처리할 수 있는지 검토했다.

하지만 현재 데이터에서는 봄과 여름을 이미지 하나하나에서 안정적으로 구분하기 어렵다. 또한 bare tree 같은 약한 winter cue는 홍보 이미지 검색에서 반드시 겨울 이미지로 살릴 필요가 없다.

따라서 Phase 1b에서는 계절을 image temporal label로 강하게 쓰지 않고, campaign-specific retrieval cue와 domain rule로 처리했다.

정리하면 다음과 같다.

- snow, autumn foliage처럼 강한 visual cue는 retrieval cue로 사용한다.
- spring/summer의 일반적인 green vegetation은 image-level season label로 쓰지 않는다.
- known flower의 개화시기 mismatch는 retrieval 후보에서 제외한다.
- 이 처리는 campaign-specific filter이며 이미지 자체를 reject하는 것이 아니다.

## 꽃 개화시기 제외 규칙

꽃 개화시기 제외 규칙은 다음과 같다.

1. campaign에 요청 계절이 있다.
2. 이미지 subject_name이 DB에 등록된 꽃이다.
3. 그 꽃의 개화시기에 campaign 요청 계절이 없다.
4. 그러면 해당 campaign-image pair를 effective retrieval 후보에서 제외한다.

예시:

- 봄 campaign + 산수국 -> 제외
- 봄 campaign + 참나리 -> 제외
- 여름 campaign + 진달래 -> 제외
- 가을 campaign + 수련 -> 제외
- 가을 campaign + 참나리 -> 제외

이 규칙은 classifier 성능을 높이기 위한 feature가 아니다.  
명백히 계절과 맞지 않는 꽃 후보를 DB 지식으로 사전에 제거하는 retrieval pre-filter다.

## Effective retrieval 결과

원본 retrieval 후보와 effective retrieval 후보의 차이는 다음과 같다.

- original retrieval candidates: 300
- effective retrieval candidates: 286
- flower-season excluded candidates: 14

campaign별 effective 후보 수는 다음과 같다.

- phase1a_summer_garden_walk: 49
- phase1b_architecture_exhibition_visit: 50
- phase1b_autumn_garden_walk: 43
- phase1b_botanical_spring_program: 45
- phase1b_indoor_gallery_winter_art: 50
- phase1b_summer_garden_walk: 49

합계는 286이다.

## Review label impact

꽃 개화시기 제외 규칙이 실제 review label 기준으로 무엇을 제거했는지 확인했다.

제외된 후보 14개의 label impact는 다음과 같다.

- excluded reviewed accept: 0
- excluded reviewed acceptable: 0
- excluded reviewed reject: 9
- excluded unlabeled: 5

즉 이 규칙은 사람이 좋다고 판단한 후보를 제거하지 않았다.  
반대로 사람이 reject한 후보 9개를 제거했다.

원본 labeled 후보:

- accept: 37
- acceptable: 31
- reject: 112

effective labeled 후보:

- accept: 37
- acceptable: 31
- reject: 103

accept와 acceptable은 유지되었고 reject만 줄었다.  
따라서 이 규칙은 retrieval pre-filter로 유지한다.

## Effective training view

flower-season exclusion을 training 분석에도 반영하기 위해 effective training view를 만들었다.

- v_effective_training_set_items_v1
- v_effective_training_snapshots_v1
- v_training_set_items_excluded_by_flower_season_v1
- v_training_snapshots_excluded_by_flower_season_v1

현재 effective training 기준은 다음과 같다.

Classifier:

- effective classifier rows: 112
- label 0: 62
- label 1: 50

Ranker:

- effective ranker rows: 112
- label 0: 62
- label 1: 24
- label 2: 26

원래 Phase 1b filtered training set은 120 rows였다.  
flower-season exclusion을 적용하면 reject 쪽 일부가 제거되어 112 rows가 된다.

## Classifier smoke metric 해석

flower-season exclusion 이후 classifier smoke metric은 낮아질 수 있다.  
이것은 규칙이 실패했다는 뜻이 아니다.

이유는 다음과 같다.

적용 전에는 training set 안에 “계절과 맞지 않는 꽃”이라는 비교적 쉬운 reject가 포함되어 있었다.  
적용 후에는 그런 쉬운 reject가 DB rule로 제거된다.  
따라서 classifier가 보는 문제는 더 어려워진다.

즉 flower-season exclusion은 classifier에게 더 많은 정보를 주는 feature가 아니라, classifier가 학습할 필요 없는 domain mismatch를 사전에 제거하는 pre-filter다.

따라서 이 규칙의 평가는 classifier smoke metric이 아니라 retrieval 후보 품질 기준으로 해야 한다.

현재 기준에서 중요한 사실은 다음이다.

- reviewed positive 제거: 0
- reviewed reject 제거: 9

이 기준에서 flower-season exclusion은 유효하다.

## Effective search index

DB의 정보를 검색/RAG용 JSONL index로 export했다.

검색 index에는 다음 정보가 들어간다.

- image 기본 정보
- ontology tags
- ontology tag assertions
- plant bloom prior
- effective campaigns
- excluded campaigns
- campaign visual cue scores
- retrieval policy metadata

현재 search index 결과는 다음과 같다.

- search index images: 206
- images with ontology tags: 202
- images with plant bloom prior: 28
- images with effective campaigns: 112
- images with excluded campaigns: 9

여기서 images with effective campaigns는 unique image 수다.  
campaign-image pair 수는 effective retrieval candidates 286개다.

## Query smoke test

effective search index에 대해 query smoke test를 수행했다.

확인한 조건은 다음이다.

- excluded campaign-image pair가 effective_campaigns에 동시에 남아 있지 않아야 한다.
- flower-season exclusion이 search index에 반영되어야 한다.
- campaign별 excluded subject가 예상과 맞아야 한다.

결과:

- query smoke passed: true
- excluded/effective same-campaign conflict: 0

campaign별 exclusion은 다음과 같다.

봄 식물 프로그램:

- excluded: 맥문동, 산수국, 수련, 참나리

여름 정원 산책:

- excluded: 진달래

가을 정원 산책:

- excluded: 산딸기, 산수국, 수련, 진달래, 참나리

겨울 실내 전시:

- excluded: 0

## End-to-end runner

Phase 1b effective retrieval 상태를 다시 만들기 위해 runner를 추가했다.

실행 명령:

    python scripts/run_phase1b_effective_retrieval_pipeline.py \
      --db data/db/cap_reranker_ontology.db

runner는 다음을 순서대로 수행한다.

1. DB 생성
2. DB 검증
3. cluster label queue 생성
4. reviewed cluster label ingest
5. cluster tag propagation
6. ontology tag assertion 검증
7. plant bloom prior ingest
8. seasonal cue policy ingest
9. CLIP cue score 계산
10. botanical bloom prior scoring
11. flower-season exclusion 생성
12. retrieval candidate filter 적용
13. effective retrieval view 생성
14. effective training view 생성
15. effective search index export
16. query smoke test
17. 최종 DB 검증

이 runner를 통해 현재 effective retrieval 상태는 DB 기준으로 재현 가능하다.

## Effective result numbers export

흩어진 audit 결과를 하나의 JSON으로 모으는 export script를 추가했다.

실행 명령:

    python scripts/export_phase1b_effective_result_numbers_from_db.py \
      --db data/db/cap_reranker_ontology.db

출력:

- audit/phase_1b/phase_1b_effective_result_numbers_db.json

핵심 고정값은 다음과 같다.

- original retrieval candidates: 300
- effective retrieval candidates: 286
- flower-season excluded candidates: 14
- excluded reviewed positives: 0
- excluded reviewed rejects: 9
- effective classifier rows: 112
- effective ranker rows: 112
- search index images: 206
- query smoke passed: true

## 현재 한계

현재 Phase 1b에는 다음 한계가 있다.

1. ontology coverage가 완전하지 않다.
2. coarse cluster 기반 tag propagation이므로 image-level 정밀 label로 보면 안 된다.
3. subject_name이 없는 이미지가 많다.
4. known flower에 대해서만 개화시기 exclusion이 적용된다.
5. 나무의 계절성은 강하게 제외하지 않는다.
6. CLIP cue score는 calibrated threshold가 아니다.
7. classifier smoke metric은 quality claim으로 사용할 수 없다.
8. 현재 결과는 샘플 데이터 기준이다.

## Phase 2로 넘길 일

Phase 2에서는 다음을 진행해야 한다.

1. 이미지 풀 확장
2. subject_name coverage 개선
3. image-level ontology label 검토 확대
4. spring/summer 구분이 필요한 경우 별도 cue policy 설계
5. search index 기반 retrieval smoke test를 실제 query ranking으로 확장
6. campaign별 retrieval 후보 평가 UI 개선
7. ontology tag와 visual cue score를 함께 쓰는 ranking policy 설계
8. classifier/ranker는 domain rule로 제거되지 않은 어려운 후보에 집중하도록 재정의

## 결론

Phase 1b에서는 DB source of truth 기반 effective retrieval scaffold를 만들었다.

가장 중요한 결론은 다음이다.

- flower-season exclusion은 classifier 성능 개선용이 아니다.
- flower-season exclusion은 retrieval pre-filter로 유효하다.
- 이 규칙은 reviewed positive 후보를 제거하지 않았다.
- 이 규칙은 reviewed reject 후보 9개를 제거했다.
- effective retrieval candidates는 300개에서 286개로 줄었다.
- effective search index와 query smoke test가 이 상태를 일관되게 반영한다.
- 전체 상태는 runner로 재현 가능하다.

따라서 Phase 1b effective retrieval pipeline은 다음 단계로 넘길 수 있는 상태다.
