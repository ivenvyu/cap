# Phase 1b 종료 보고서

## 결론

Phase 1b는 종료 가능한 상태다.

이번 단계의 목표는 완성된 이미지 검색 시스템이나 완전한 온톨로지를 만드는 것이 아니었다. 목표는 샘플 이미지, campaign, retrieval 후보, feature, review label, training snapshot을 DB 기준으로 연결하고, 이후 확장 가능한 검색 후보 생성 구조를 만드는 것이었다.

현재 그 목표는 충족되었다.

최종 상태는 다음과 같다.

- DB source of truth 구축 완료
- ontology scaffold 구축 완료
- cluster 기반 tag propagation 완료
- 사유원 식물 개화시기 prior ingest 완료
- 계절 visual cue policy 및 CLIP cue score scaffold 완료
- 꽃 개화시기 기반 retrieval pre-filter 완료
- effective retrieval view 완료
- effective training view 완료
- search/RAG용 effective index export 완료
- query smoke test 완료
- end-to-end runner 완료
- effective result numbers export 완료
- runbook 및 design summary 문서화 완료

## Phase 1b에서 해결한 문제

Phase 1b 이전에는 다음 문제가 있었다.

1. review 결과와 training snapshot이 파일 단위로 흩어져 있었다.
2. retrieval candidate와 review label을 안정적으로 join하기 어려웠다.
3. image, campaign, feature, label, training item 사이의 관계를 한 곳에서 검증하기 어려웠다.
4. 계절 campaign에서 계절과 맞지 않는 꽃 이미지가 후보에 섞였다.
5. ontology tag와 domain knowledge를 retrieval 후보 생성에 연결하는 구조가 없었다.
6. classifier smoke metric과 retrieval 후보 품질 평가가 섞여 해석될 위험이 있었다.

Phase 1b에서는 이 문제를 DB-first 구조로 정리했다.

## DB source of truth

현재 DB는 운영 기준 source of truth 역할을 한다.

DB에 적재된 주요 객체는 다음과 같다.

- images
- image embeddings
- image duplicates
- DINOv2 clusters
- image regions
- campaigns
- retrieval candidates
- pair features
- review events
- training snapshots
- training sets
- training set items
- ontology tag axes
- ontology tag values
- cluster label queue
- cluster tag assertions
- image tag assertions
- visual cues
- campaign visual cue requirements
- campaign image cue scores
- plant entities
- plant names
- plant bloom priors
- flower season exclusions
- retrieval candidate filter decisions

최종 DB 검증에서 foreign key check는 통과했다.

## Ontology scaffold의 범위

현재 온톨로지는 완성된 coverage를 주장하지 않는다.

현재 구축된 것은 다음 수준이다.

1. DINOv2 coarse cluster 기준으로 label queue 생성
2. 검토된 cluster label CSV ingest
3. cluster tag assertion 저장
4. image tag assertion으로 propagation
5. search index에 ontology tag 반영

현재 image tag는 cluster 기반 전파 결과다. 따라서 image 하나하나에 완전한 수동 label이 붙었다고 해석하면 안 된다.

현재 유효하게 사용한 주요 축은 다음이다.

- space_axis
- subject_axis
- mood_axis

Phase 2에서는 image-level 검토 확대와 axis coverage 개선이 필요하다.

## 계절 처리 결정

Phase 1b에서는 계절을 image temporal_axis label로 강하게 쓰지 않기로 결정했다.

이유는 다음과 같다.

- 봄과 여름은 개별 이미지에서 안정적으로 구분하기 어렵다.
- bare tree 같은 약한 winter cue는 홍보 이미지 검색에서 반드시 winter로 살릴 필요가 없다.
- 반대로 snow, autumn foliage, flower bloom처럼 강한 cue는 retrieval cue로 쓰는 것이 적절하다.
- known flower의 개화시기 mismatch는 DB 지식으로 직접 제거할 수 있다.

따라서 계절은 다음 방식으로 처리했다.

- 강한 계절 visual cue는 retrieval cue score로 저장한다.
- spring/summer 일반 식생은 image-level season label로 쓰지 않는다.
- known flower 개화시기 mismatch는 campaign-specific retrieval pre-filter로 제거한다.

## 꽃 개화시기 기반 pre-filter

꽃 개화시기 제외 규칙은 다음과 같다.

1. campaign에 요청 계절이 있다.
2. 이미지 subject_name이 DB에 등록된 꽃이다.
3. 그 꽃의 개화시기에 campaign 요청 계절이 없다.
4. 그러면 해당 campaign-image pair를 effective retrieval 후보에서 제외한다.

이 규칙은 이미지 자체를 reject하는 규칙이 아니다. 특정 campaign 후보에서만 제외하는 campaign-specific rule이다.

예를 들어 산수국은 봄 campaign에서는 제외될 수 있지만, 여름 campaign에서는 제외되지 않는다.

## Effective retrieval 결과

원본 retrieval 후보는 300개였다.

꽃 개화시기 기반 pre-filter를 적용한 뒤 effective retrieval 후보는 286개가 되었다.

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

flower-season exclusion이 제거한 후보를 review label과 대조했다.

결과는 다음과 같다.

- excluded reviewed accept: 0
- excluded reviewed acceptable: 0
- excluded reviewed reject: 9
- excluded unlabeled: 5

즉 사람이 좋다고 판단한 후보는 제거하지 않았다. 반대로 사람이 reject한 후보 9개를 제거했다.

원본 labeled 후보는 다음과 같다.

- accept: 37
- acceptable: 31
- reject: 112

effective labeled 후보는 다음과 같다.

- accept: 37
- acceptable: 31
- reject: 103

accept와 acceptable은 유지되었고 reject만 줄었다.

따라서 꽃 개화시기 기반 pre-filter는 retrieval 후보 품질 기준에서 유효하다.

## Classifier smoke metric 해석

flower-season exclusion은 classifier 성능을 높이기 위한 feature가 아니다.

이 규칙을 적용하면 classifier smoke metric이 낮아질 수 있다. 이것은 실패가 아니다.

이유는 다음과 같다.

적용 전에는 training set 안에 "계절과 맞지 않는 꽃"이라는 비교적 쉬운 reject가 포함되어 있었다. 적용 후에는 그런 쉬운 reject가 DB rule로 제거된다. 따라서 classifier는 더 어려운 reject만 보게 된다.

따라서 이 규칙은 classifier metric이 아니라 retrieval 후보 품질로 평가해야 한다.

현재 평가 기준에서 중요한 사실은 다음이다.

- reviewed positive 제거: 0
- reviewed reject 제거: 9

이 기준에서 규칙은 통과했다.

## Effective training view

flower-season exclusion을 training 분석에도 반영하기 위해 effective training view를 만들었다.

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

이 값은 원본 Phase 1b filtered training set 120 rows에서 flower-season mismatch reject가 제거된 결과다.

## Search index와 query smoke

DB 정보를 search/RAG용 JSONL index로 export했다.

search index에는 다음 정보가 포함된다.

- image 기본 정보
- ontology tags
- ontology tag assertions
- plant bloom prior
- effective campaigns
- excluded campaigns
- campaign visual cue scores
- retrieval policy metadata

현재 search index 기준값은 다음이다.

- search index images: 206
- images with ontology tags: 202
- images with plant bloom prior: 28
- images with effective campaigns: 112
- images with excluded campaigns: 9

query smoke test에서는 excluded campaign-image pair가 effective_campaigns에 동시에 남아 있지 않은지 확인했다.

결과는 다음과 같다.

- query smoke passed: true
- excluded/effective same-campaign conflict: 0

따라서 search index export는 effective retrieval rule을 일관되게 반영한다.

## End-to-end 재현성

Phase 1b effective retrieval 상태는 runner로 재현 가능하다.

실행 명령은 다음이다.

    conda activate cap_embed
    cd /Users/ksc/Desktop/cap/reranker
    python scripts/run_phase1b_effective_retrieval_pipeline.py \
      --db data/db/cap_reranker_ontology.db

runner는 다음을 수행한다.

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

## 결과 숫자 export

핵심 결과 숫자는 다음 script로 다시 export할 수 있다.

    python scripts/export_phase1b_effective_result_numbers_from_db.py \
      --db data/db/cap_reranker_ontology.db

출력 파일은 다음이다.

- audit/phase_1b/phase_1b_effective_result_numbers_db.json

현재 고정 기준값은 다음이다.

- original retrieval candidates: 300
- effective retrieval candidates: 286
- flower-season excluded candidates: 14
- excluded reviewed positives: 0
- excluded reviewed rejects: 9
- effective classifier rows: 112
- effective ranker rows: 112
- search index images: 206
- query smoke passed: true

## 프로젝트 요구사항과의 연결

프로젝트 요구사항은 다음 방향을 포함한다.

- 계층적 다중 관점 이미지 태깅 온톨로지
- 공간, 시간, 피사체, 무드, 용도 등 다중 축 metadata
- Top-down 기획만이 아니라 이미지 임베딩 실험과 이미지 분포 분석 기반 metadata 체계
- 향후 대규모 이미지 풀에서도 검색 최적화 가능한 구조

Phase 1b는 이 요구를 완성하지 않는다.

대신 다음 기반을 구축했다.

- DB source of truth
- DINOv2 cluster 기반 label queue
- cluster tag assertion
- propagated image tag assertion
- domain knowledge 기반 plant bloom prior
- campaign-specific retrieval pre-filter
- effective retrieval view
- effective search index
- query smoke validation
- end-to-end runner

따라서 Phase 1b는 최종 기능 완성이 아니라 Phase 2 검색 고도화를 위한 scaffold 완료 단계로 해석해야 한다.

## 현재 한계

현재 한계는 다음과 같다.

1. ontology coverage가 완전하지 않다.
2. image-level 수동 ontology label이 충분하지 않다.
3. coarse cluster propagation은 정밀 label이 아니다.
4. subject_name이 없는 이미지가 많다.
5. 꽃 개화시기 rule은 known flower에만 적용된다.
6. 나무 계절성은 강하게 제외하지 않는다.
7. CLIP cue score는 calibrated threshold가 아니다.
8. classifier smoke metric은 품질 claim으로 사용할 수 없다.
9. 현재 결과는 샘플 데이터 기준이다.

## Phase 2로 넘길 일

Phase 2에서는 다음을 진행해야 한다.

1. 이미지 풀 확장
2. subject_name coverage 개선
3. image-level ontology label 검토 확대
4. search index 기반 실제 query ranking 평가
5. campaign별 retrieval 후보 review UI 개선
6. ontology tag와 visual cue score를 함께 쓰는 ranking policy 설계
7. domain rule로 제거되지 않은 어려운 후보에 classifier/ranker를 집중하도록 task 재정의
8. 현재 scaffold를 대규모 이미지 풀에 적용했을 때의 성능과 운영 비용 검증

## 종료 판단

Phase 1b는 다음 조건을 만족한다.

- DB source of truth가 구축되었다.
- effective retrieval 후보 생성이 가능하다.
- flower-season exclusion이 reviewed positive를 제거하지 않는 것이 확인되었다.
- effective search index가 export된다.
- query smoke test가 통과한다.
- end-to-end runner로 재현 가능하다.
- 핵심 결과 숫자가 export된다.
- 운영 문서와 요약 문서가 작성되었다.

따라서 Phase 1b는 종료하고 Phase 2로 넘어갈 수 있다.
