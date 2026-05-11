# DB Source-of-Truth Transition

## 메타데이터

spec_version: v2.2.1  
phase: phase_1b  
status: active  
db_role: operational_source_of_truth  
score_status: diagnostic_only  

## 결정

이 프로젝트의 기준 데이터는 SQLite ontology DB로 승격한다.

기존 JSONL/CSV/NPY/HTML 산출물은 다음 중 하나로만 취급한다.

1. DB import source
2. DB export/cache
3. audit artifact
4. generated preview artifact

Downstream script는 최종적으로 DB에서 읽어야 한다.

## 원칙

DB가 source of truth라는 말은 `.db` 파일만 만든다는 뜻이 아니다.

다음이 함께 있어야 한다.

1. schema
2. import provenance
3. import batch 기록
4. artifact source hash
5. validator
6. DB-first reader
7. export script
8. migration policy

## 파일 artifact의 역할

다음 파일들은 더 이상 authoritative source가 아니다.

- raw_image_manifest jsonl
- clip/dinov2 index csv
- duplicate/cluster/region jsonl
- retrieval jsonl
- pair_feature_snapshot jsonl
- review_event jsonl
- training_snapshot jsonl/csv
- review queue csv

이들은 DB를 채우기 위한 import source 또는 DB에서 다시 뽑는 export/cache다.

## DB에 들어가야 하는 주요 layer

### Core image ontology

- images
- campaigns
- tag_axes
- tag_values
- image_tag_assertions
- cluster_tag_assertions

### Computed visual artifacts

- image_embeddings
- image_duplicates
- image_clusters
- image_regions
- retrieval_candidates
- pair_features

### Human review and labels

- review_events
- training_snapshots

### Provenance

- db_builds
- artifact_sources
- import_batches

## 현재 전환 범위

이번 foundation 단계에서는 다음을 DB에 넣는다.

- images
- campaigns
- image_embeddings
- image_duplicates
- image_clusters
- image_regions
- retrieval_candidates
- pair_features
- review_events
- training_snapshots
- ontology tag seed tables

## Non-claims

이 전환은 model quality를 주장하지 않는다.  
이 전환은 calibrated threshold를 만들지 않는다.  
이 전환은 candidate-level support explanation을 만들지 않는다.  
이 전환은 design quality를 주장하지 않는다.

## 다음 단계

1. build_ontology_db.py로 기존 artifact를 DB에 import한다.
2. validate_ontology_db.py로 row count, FK, join integrity를 검증한다.
3. classifier smoke, audit report, claim support report를 DB-first로 전환한다.
4. cluster 기반 ontology tag labeling loop를 DB 위에 구축한다.
