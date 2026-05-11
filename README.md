# CAP Image Reranker
## 홍보물 이미지 자동 추천 시스템

---

## 프로젝트 목적

홍보물 제작 시 수만 장의 사진 중에서 적합한 이미지를 수작업으로 탐색하는 병목을 해소하기 위한 프로젝트이다.

홍보물 주제(계절, 프로그램 종류, 공간, 분위기)를 입력하면 raw image DB에서 후보 이미지를 자동으로 순위화하고, 담당자는 상위 소수 후보에 대한 최종 수락/거절 판단만 한다.

---

## 시스템 구조

```
Campaign 정보 (계절/프로그램/공간/분위기)
        │
        ▼
  Prompt 변환 (한국어 → 영어 시각 설명어)
        │
        ├──→ CLIP Retrieval       (1차: 의미 매칭)
        ├──→ DINOv2 Anchor        (2차: 시각 유사 확장)
        └──→ Ontology Filter      (식물 개화 시기 불일치 자동 제거)
                                          │
                                          ▼
                              LightGBM Reranker (통합 순위화)
                                          │
                                          ▼
                              Review Queue → 담당자 판단 → 학습 데이터 누적
```

---

## 실행 환경
```bash

conda env create -f environment.yaml
conda activate cap_embed
```

**주요 의존성:**
- `torch`, `transformers`: CLIP/DINOv2 임베딩 추출 시 필요 (GPU 권장)
- `scikit-learn`, `joblib`: classifier 학습
- `jsonschema`: schema 검증
- `tqdm`: 진행 표시

> 임베딩 추출(torch 필요) 외 나머지 스크립트는 CPU로 실행 가능하다.

---

## 스크립트 실행 가이드

### 검증 스크립트 (즉시 실행 가능)
```bash

# Phase 0 schema cross-file consistency 검증 (16개 항목)
python scripts/validate_phase0_specs.py

# Phase 1b 운영 정책 검증
python scripts/validate_phase1b_specs.py

# DB 무결성 검증
python scripts/validate_ontology_db.py

# 온톨로지 태그 전파 검증
python scripts/validate_ontology_tag_assertions.py
```

### Classifier 학습 (즉시 실행 가능)
```bash

# v2.2.5 진단용 classifier 학습
python scripts/train_phase1b_classifier_smoke_v2_2_5_from_jsonl.py
```

### 이미지 데이테 폴더 추가
```bash

raw 폴더를 data/raw로 넣어주세요!!!
```

### 추천 shortlist 생성
```bash
# candidate score snapshot 필요 (data/retrieval/phase1b/v2_2_5/)
python scripts/score_phase1b_candidates_v2_2_5_from_jsonl.py
python scripts/build_phase1b_recommendation_shortlist_v2_2_5_global_dedupe.py
```

### 추천 결과 생성 확인
```bash

open data/review/phase1b/v2_2_5/shortlist_global_dedupe/index.html (사파리에서 안 열리면 크롬에서 열어주세요)
```


### 전체 파이프라인 (이미지 pool 확장 시)

```bash
# 1. 임베딩 추출 (GPU 필요)
python scripts/extract_clip_image_embeddings.py --image-dir /path/to/images --out data/embeddings/
python scripts/extract_dinov2_image_embeddings.py --image-dir /path/to/images --out data/embeddings/

# 2. 기본 구조 생성
python scripts/build_raw_image_manifest.py --raw-root /path/to/images --out data/ontology/raw_image_manifest.jsonl
python scripts/build_dinov2_duplicate_and_cluster_artifacts.py
python scripts/build_region_safety_maps.py

# 3. DB 구축
python scripts/build_ontology_db.py

# 4. 온톨로지 태그 적용
python scripts/ingest_sayuwon_plant_bloom_priors_to_db.py
python scripts/propagate_cluster_tags_to_images.py

# 5. CLIP 검색 후보 생성
python scripts/build_clip_retrieval_candidates.py --campaign examples/campaigns/phase1b/01_summer_garden_walk.json

# 6. feature snapshot → classifier → shortlist
python scripts/build_pair_feature_snapshots.py
python scripts/train_phase1b_classifier_smoke_v2_2_5_from_jsonl.py
```

---

## 개발 단계 및 현황

| Phase | 목표 | 상태 |
|---|---|---|
| **Phase 0** | Schema 및 운영 규칙 고정 | ✅ 완료 |
| **Phase 1a** | 소규모(206장) dry run, pipeline 검증 | ✅ 완료 |
| **Phase 1b** | 5개 campaign, DINOv2 anchor 활성화, 온톨로지 구축 | ✅ 완료 |
| Phase 2 | 이미지 pool 확장 (10,000장 수령 예정) | 대기 중 |
| Phase 3 | Classifier → Ranker 전환, held-out validation | 예정 |
| Phase 4 | Visual Critic 분리 학습 | 예정 |

---

## Phase 1b 주요 결과

### 데이터 현황

| 항목 | 수치 |
|---|---|
| Raw image pool | 206장 |
| Campaign 수 | 5개 (4개 family) |
| 담당자 판단 데이터 | 205개 (1차 180개 + triage 25개) |
| 온톨로지 태그 | 7개 축, 52개 tag value, 1,173개 image assertion |
| 식물 개화 필터 | 19종 식물, 44건 자동 제거 |

### 버전별 진행

| 버전 | 주요 변경 | OOF ROC-AUC |
|---|---|---|
| v2.2.1 | baseline, CLIP feature만 | 0.598 |
| v2.2.2 | DINOv2 anchor feature 활성화 (cold_start 탈출) | — |
| v2.2.3 | triage-25 label 반영 | — |
| v2.2.5 | duplicate canonicalization (path leakage 제거) | 0.534 |

> v2.2.5 ROC-AUC가 v2.2.1보다 낮은 건 성능 저하가 아니라, leakage를 제거한 후 더 정직한 수치다.
> 모든 metric은 `diagnostic_only` — held-out campaign validation 전까지 성능 주장으로 사용하지 않는다.

### Campaign별 판단 결과

| Campaign | 거절 | 보통 | 수락 | 비고 |
|---|---|---|---|---|
| 건축/전시 방문 | 10 (33%) | 9 | 11 | architecture family |
| 가을 정원 산책 | 21 (70%) | 3 | 6 | season_mismatch 다수 |
| 봄 식물 프로그램 | 20 (67%) | 8 | 2 | 봄/여름 경계 모호 |
| 여름 정원 산책 | 19 (63%) | 4 | 7 | Phase 1a 기준점 |
| 실내 갤러리 겨울 | 29 (97%) | 0 | 1 | raw pool coverage gap |

---

## 온톨로지 설계

DINOv2 cluster(coarse 20개)로 이미지 분포를 먼저 분석하고, cluster별 대표 이미지를 보고 사람이 태그를 붙인 뒤 전체 이미지로 전파하는 방식을 사용했다.

### Tag Axes (7개)

| Axis | 예시 | 용도 |
|---|---|---|
| space_axis | garden / architecture / indoor | 공간 유형 |
| temporal_axis | spring / summer / autumn / winter | 계절 |
| weather_light_axis | sunny / foggy / cloudy / backlit | 날씨/조명 |
| subject_axis | flower / tree / building / visitor | 피사체 |
| mood_axis | quiet / calm / active / mysterious | 분위기 |
| usage_axis | sns / brochure / poster / proposal | 용도 |
| design_affordance_axis | text_overlay_easy / text_overlay_hard | 디자인 적합성 |

### 식물 개화 시기 필터

사유원 고유 식물 19종의 개화 시기를 DB에 저장하고, campaign 계절과 맞지 않는 꽃 이미지를 자동 제거한다.

예시:
- summer campaign → 진달래(봄 개화) 자동 제외
- autumn campaign → 산수국(여름 개화) 자동 제외

---

## 프로젝트 구조

```
reranker/
├── scripts/                 파이프라인 코드 (65개)
│   ├── extract_*.py         임베딩 추출 (torch/GPU 필요)
│   ├── build_*.py           각 단계 산출물 생성
│   ├── train_*.py           classifier 학습
│   ├── validate_*.py        schema 및 DB 검증
│   └── score_*.py           후보 점수화
│
├── configs/                 운영 정책 (17개 YAML)
│   └── domain_knowledge/    사유원 고유 공간/식물 지식
│
├── schemas/                 JSON Schema (3개)
├── docs/design_history/     설계 결정 기록 (10개 MD)
├── examples/campaigns/      Campaign payload 예시 (6개)
├── annotations/             사람이 검토한 cluster 라벨
├── audit/                   실험 결과 보고서
│
└── data/
    ├── db/                  SQLite 온톨로지 DB (28개 테이블)
    ├── embeddings/          임베딩 인덱스 CSV (벡터 .npy는 별도)
    ├── ontology/            이미지 manifest, cluster, duplicate
    └── review/              담당자 판단 데이터 (labeled CSV + JSONL)
```

---

## 방법론적 원칙

**Schema-first**: 코드 작성 전 data schema와 운영 규칙을 먼저 확정. 이후 schema 변경 시에도 과거 데이터가 불변(immutable)으로 보존된다.

**Diagnostic-only**: 소규모 데이터에서 나온 metric을 성능으로 주장하지 않는다. held-out campaign validation 전까지 모든 결과에 `score_status: diagnostic_only`를 명시한다.

**Confirmation bias 차단**: score 상위 N개 정렬 대신 7-bucket discovery sampling. CLIP-high/model-low 같이 "모델이 놓칠 가능성" 있는 후보를 review queue에 강제 포함한다.

**사람의 역할 제한**: 사람은 campaign에 이미지가 맞는지만 판단한다. metadata 입력, layout 선택, 디자인 품질 판단은 시스템이 처리한다.

---

## 현재 한계

| 항목 | 현황 |
|---|---|
| 이미지 pool | 206장 (목표 10,000장, 데이터 수령 대기 중) |
| 성능 지표 | 모두 diagnostic_only — held-out validation 미완료 |
| Layout 품질 | 디자이너/Visual QA 없음, 보수적 fallback(split panel) 적용 |
| RAG 생성 | Image Reranker 범위 외 — 별도 모듈 필요 |
