# Phase 1b 효과적 검색 후보 생성 실행 문서

## 목적

이 문서는 Phase 1b 검색 후보 생성 과정을 처음부터 다시 실행하는 방법을 정리한다.

현재 목표는 완전한 온톨로지를 완성하는 것이 아니다. 현재 샘플 데이터와 검토 결과를 DB에 모으고, 이후 더 큰 이미지 풀에서 확장 가능한 검색 후보 생성 구조를 만드는 것이다.

이 실행 흐름은 다음을 재현한다.

1. 이미지, campaign, retrieval 후보, feature, review 결과를 DB에 적재한다.
2. 검토된 cluster tag를 DB에 넣고 image tag로 전파한다.
3. 사유원 식물 개화시기 지식을 DB에 넣는다.
4. 계절 visual cue score를 계산한다.
5. 계절과 맞지 않는 known flower 후보를 제거한다.
6. effective retrieval view와 effective training view를 만든다.
7. 검색/RAG용 JSONL index를 export한다.
8. query smoke test로 index 일관성을 확인한다.
9. 핵심 결과 숫자를 audit JSON으로 요약한다.

## 실행 환경

CLIP cue scoring 단계가 있으므로 cap_embed 환경에서 실행한다.

명령:

    conda activate cap_embed
    cd /Users/ksc/Desktop/cap/reranker

## 전체 pipeline 실행

아래 명령 하나로 DB를 다시 만들고 effective retrieval 상태를 재생성한다.

    python scripts/run_phase1b_effective_retrieval_pipeline.py \
      --db data/db/cap_reranker_ontology.db

이 명령은 내부적으로 다음 단계를 순서대로 실행한다.

1. build_ontology_db.py
2. validate_ontology_db.py
3. build_cluster_label_queue_from_db.py
4. ingest_cluster_tag_labels_to_db.py
5. propagate_cluster_tags_to_images.py
6. validate_ontology_tag_assertions.py
7. ingest_sayuwon_plant_bloom_priors_to_db.py
8. ingest_seasonal_prompt_cue_policy_to_db.py
9. score_campaign_visual_cues_from_clip.py
10. score_botanical_bloom_season_priors_from_db.py
11. build_flower_season_exclusions_from_db.py
12. apply_flower_season_exclusions_to_retrieval_candidates.py
13. build_effective_retrieval_views_from_db.py
14. build_effective_training_views_from_db.py
15. export_ontology_effective_search_index_from_db.py
16. query_smoke_ontology_effective_search_index.py
17. validate_ontology_db.py

## 주요 산출물

전체 pipeline 실행 후 다음 파일이 생성된다.

- DB: data/db/cap_reranker_ontology.db
- 검색 index: data/search/ontology_effective_search_index_v1.jsonl
- query smoke report: audit/ontology/ontology_effective_search_index_query_smoke_v1.summary.json
- pipeline summary: audit/ontology/phase1b_effective_retrieval_pipeline_run_v1.summary.json

이 파일들은 재생성 가능한 산출물이므로 기본적으로 git에 커밋하지 않는다.

## 핵심 결과 숫자 export

전체 pipeline 실행 후 핵심 결과 숫자를 하나의 JSON으로 모으려면 다음 명령을 실행한다.

    python scripts/export_phase1b_effective_result_numbers_from_db.py \
      --db data/db/cap_reranker_ontology.db

출력 파일은 다음이다.

- audit/phase_1b/phase_1b_effective_result_numbers_db.json

이 파일도 재생성 가능한 audit 산출물이다.

## 현재 기준 핵심 숫자

현재 Phase 1b effective retrieval 기준값은 다음이다.

Retrieval 후보:

- 원본 retrieval 후보: 300
- effective retrieval 후보: 286
- 꽃 개화시기 규칙으로 제외된 후보: 14

Review label 기준 영향:

- 제외된 reviewed accept 후보: 0
- 제외된 reviewed acceptable 후보: 0
- 제외된 reviewed reject 후보: 9
- 제외된 unlabeled 후보: 5

즉 꽃 개화시기 제외 규칙은 사람이 좋다고 판단한 후보를 제거하지 않았고, 사람이 reject한 후보 9개를 제거했다.

Effective training set:

- effective classifier rows: 112
- classifier label 0: 62
- classifier label 1: 50
- effective ranker rows: 112
- ranker label 0: 62
- ranker label 1: 24
- ranker label 2: 26

Search index:

- search index images: 206
- images with effective campaigns: 112
- images with exclusions: 9
- query smoke consistency check: true
- excluded/effective same-campaign conflict: 0

## 꽃 개화시기 제외 규칙

이 규칙은 classifier 성능을 높이기 위한 feature가 아니다. retrieval 후보를 만들기 전에 명백히 계절과 맞지 않는 꽃 이미지를 제거하는 DB 기반 pre-filter다.

규칙은 다음과 같다.

1. campaign에 요청 계절이 있다.
2. 이미지의 subject_name이 DB에 등록된 꽃이다.
3. 그 꽃의 개화시기에 campaign 요청 계절이 없다.
4. 그러면 해당 campaign-image pair를 effective retrieval 후보에서 제외한다.

예시:

- 봄 campaign + 산수국 -> 제외
- 봄 campaign + 참나리 -> 제외
- 여름 campaign + 진달래 -> 제외
- 가을 campaign + 수련 -> 제외

이 규칙은 이미지 자체를 reject하는 것이 아니다. 특정 campaign 후보에서만 제외하는 campaign-specific rule이다.

## Classifier smoke metric과의 관계

꽃 개화시기 제외 규칙을 적용하면 classifier smoke metric이 낮아질 수 있다. 그것은 규칙이 실패했다는 뜻이 아니다.

이 규칙을 적용하기 전에는 training set 안에 "계절과 안 맞는 꽃"이라는 쉬운 reject가 포함되어 있었다. 규칙 적용 후에는 그런 쉬운 reject가 DB rule로 제거된다. 따라서 classifier가 보는 문제는 더 어려워진다.

이 규칙은 classifier metric으로 평가하지 않는다. 평가 기준은 retrieval 후보 품질이다.

현재 기준에서 중요한 결과는 다음이다.

- reviewed positive 제거: 0
- reviewed reject 제거: 9

따라서 꽃 개화시기 제외 규칙은 retrieval pre-filter로 유지한다.

## Downstream에서 사용할 DB view

후속 retrieval 후보 처리는 원본 table이 아니라 다음 view를 사용한다.

- v_effective_retrieval_candidates_v1
- v_effective_pair_features_v1

후속 training 분석은 다음 view를 사용한다.

- v_effective_training_set_items_v1
- v_effective_training_snapshots_v1

제외된 후보를 확인할 때는 다음 view를 사용한다.

- v_effective_retrieval_candidates_excluded_v1
- v_training_set_items_excluded_by_flower_season_v1
- v_training_snapshots_excluded_by_flower_season_v1

## 재현성 확인 방법

전체 pipeline 실행 후 다음 명령으로 핵심 숫자를 다시 export한다.

    python scripts/export_phase1b_effective_result_numbers_from_db.py \
      --db data/db/cap_reranker_ontology.db

그다음 다음 명령으로 핵심 값을 확인한다.

    python - <<'PY'
    import json
    from pathlib import Path

    p = Path("audit/phase_1b/phase_1b_effective_result_numbers_db.json")
    r = json.loads(p.read_text(encoding="utf-8"))

    print("effective retrieval:", r["retrieval_effective"]["effective_retrieval_candidates"])
    print("excluded retrieval:", r["retrieval_effective"]["excluded_retrieval_candidates"])
    print("label impact:", r["retrieval_label_impact"]["safety_check"])
    print("classifier rows:", r["effective_training"]["classifier_rows"])
    print("ranker rows:", r["effective_training"]["ranker_rows"])
    print("query smoke:", r["query_smoke"])
    PY

정상 기준은 다음이다.

- effective retrieval candidates = 286
- excluded retrieval candidates = 14
- excluded accept/acceptable = 0
- excluded reject = 9
- classifier rows = 112
- ranker rows = 112
- query smoke consistency check = true

## 커밋 대상과 비대상

커밋 대상:

- scripts/
- configs/
- docs/
- annotations/ontology/*.reviewed*.csv

커밋하지 않는 대상:

- reranker/data/
- 재생성 가능한 audit JSON
- .DS_Store

현재 .gitignore는 reranker/data/를 무시한다. audit 산출물은 현재 작업에서는 재생성 가능한 결과물로 취급한다.

## 관련 문서

- docs/design_history/flower_season_exclusion_retrieval_filter.md
- docs/design_history/ontology_db_scaffold_scope.md

## 결론

Phase 1b effective retrieval pipeline은 DB source of truth에서 재현 가능하다.

현재 기준에서 꽃 개화시기 제외 규칙은 reviewed positive 후보를 제거하지 않고 reviewed reject 후보만 제거했다. 따라서 이 규칙은 retrieval pre-filter로 유지한다.
