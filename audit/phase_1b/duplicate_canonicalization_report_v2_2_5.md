# v2.2.5 Duplicate Canonicalization Report

## 상태

이 보고서는 exact duplicate group 단위의 canonical image_id와 metadata/feature 충돌을 진단한다.
품질 threshold 또는 accept/reject 기준이 아니다.

## 요약

- duplicate groups: 24
- canonical map rows: 48
- metadata conflict groups: 24
- feature conflict rows: 185

## 주요 결정

1. exact duplicate group마다 deterministic canonical image_id를 지정한다.
2. 동일 파일이 여러 경로/source_group에 존재하는 경우 단일 path feature로 덮지 않는다.
3. `path_has_architecture`, `path_has_garden` 등 path-derived feature 충돌은 diagnostic warning으로 기록한다.
4. 다음 단계에서 feature snapshot 생성 시 duplicate group-level canonical metadata를 사용할 수 있게 한다.

## Non-claims

- duplicate canonicalization은 품질 threshold가 아니다.
- duplicate canonicalization은 이미지가 좋다/나쁘다를 판정하지 않는다.
- feature conflict는 모델 입력 데이터 정규화 이슈이며, production 성능 claim이 아니다.

## Metadata conflict examples

### dup_exact_004db23b5f5d

- canonical_image_id: `raw_bd98826b6e`
- members: raw_bd98826b6e, raw_a1ad23ca7a
- source_groups: 건축, 정원
- paths: gallery/건축/몽몽미방/space7_4_3.jpg, gallery/정원/몽몽미방/space7_4_3.jpg

### dup_exact_08c8d66882d2

- canonical_image_id: `raw_e4033ce404`
- members: raw_e4033ce404, raw_8883bfdaec
- source_groups: 건축, 정원
- paths: gallery/건축/몽몽미방/space7_4_2.jpg, gallery/정원/몽몽미방/space7_4_2.jpg

### dup_exact_0c34e0f8e4a7

- canonical_image_id: `raw_70729accac`
- members: raw_70729accac, raw_3b4c5ad14b
- source_groups: 건축, 정원
- paths: gallery/건축/몽몽미방/space7_4_7.jpg, gallery/정원/몽몽미방/space7_4_7.jpg

### dup_exact_25369eb1b632

- canonical_image_id: `raw_0f37eb334a`
- members: raw_0f37eb334a, raw_c3f434c77f
- source_groups: 건축, 정원
- paths: gallery/건축/몽몽미방/space7_3_3.jpg, gallery/정원/몽몽미방/space7_3_3.jpg

### dup_exact_38106b9781dd

- canonical_image_id: `raw_34da4bc897`
- members: raw_34da4bc897, raw_ea42a7592c
- source_groups: 건축, 정원
- paths: gallery/건축/몽몽미방/space7_1_1.jpg, gallery/정원/몽몽미방/space7_1_1.jpg

### dup_exact_39d154f18e8e

- canonical_image_id: `raw_19c9aa3ffa`
- members: raw_19c9aa3ffa, raw_b37dfe627b
- source_groups: 건축, 정원
- paths: gallery/건축/몽몽미방/space7_4_1.jpg, gallery/정원/몽몽미방/space7_4_1.jpg

### dup_exact_4e8c6a6984f6

- canonical_image_id: `raw_6b174f2b03`
- members: raw_6b174f2b03, raw_54a74aeb35
- source_groups: 건축, 정원
- paths: gallery/건축/유원/space5_3_1.jpg, gallery/정원/유원/space5_3_1.jpg

### dup_exact_4ffacf24cb3a

- canonical_image_id: `raw_4b3c524f29`
- members: raw_4b3c524f29, raw_cad8298a7b
- source_groups: 건축, 정원
- paths: gallery/건축/몽몽미방/space7_1_2.jpg, gallery/정원/몽몽미방/space7_1_2.jpg

### dup_exact_6bcf2fd25019

- canonical_image_id: `raw_7f5320f89f`
- members: raw_7f5320f89f, raw_e271fa38ea
- source_groups: 건축, 정원
- paths: gallery/건축/몽몽미방/space7_3_1.jpg, gallery/정원/몽몽미방/space7_3_1.jpg

### dup_exact_6d5800f28fea

- canonical_image_id: `raw_f60ec3a3b4`
- members: raw_f60ec3a3b4, raw_ab1479ebb7
- source_groups: 건축, 정원
- paths: gallery/건축/유원/space5_4_4.jpg, gallery/정원/유원/space5_4_4.jpg

## Feature conflict examples

### dup_exact_25369eb1b632 / phase1b_architecture_exhibition_visit / dinov2_campaign_margin

- feature_family: diagnostic_numeric
- values: `{"raw_0f37eb334a": ["-0.5196309686"], "raw_c3f434c77f": ["0.3020074517"]}`

### dup_exact_25369eb1b632 / phase1b_architecture_exhibition_visit / dinov2_campaign_neg_nn_sim

- feature_family: diagnostic_numeric
- values: `{"raw_0f37eb334a": ["1.000000238"], "raw_c3f434c77f": ["0.1783618182"]}`

### dup_exact_25369eb1b632 / phase1b_architecture_exhibition_visit / path_has_architecture

- feature_family: path_derived
- values: `{"raw_0f37eb334a": ["1"], "raw_c3f434c77f": ["0"]}`

### dup_exact_25369eb1b632 / phase1b_architecture_exhibition_visit / path_has_garden

- feature_family: path_derived
- values: `{"raw_0f37eb334a": ["0"], "raw_c3f434c77f": ["1"]}`

### dup_exact_38106b9781dd / phase1b_architecture_exhibition_visit / dinov2_campaign_margin

- feature_family: diagnostic_numeric
- values: `{"raw_34da4bc897": ["0.3086273968"], "raw_ea42a7592c": ["0.6243943274"]}`

### dup_exact_38106b9781dd / phase1b_architecture_exhibition_visit / dinov2_campaign_pos_nn_sim

- feature_family: diagnostic_numeric
- values: `{"raw_34da4bc897": ["0.6842331886"], "raw_ea42a7592c": ["1.000000119"]}`

### dup_exact_38106b9781dd / phase1b_architecture_exhibition_visit / dinov2_family_margin

- feature_family: diagnostic_numeric
- values: `{"raw_34da4bc897": ["-0.3198307753"], "raw_ea42a7592c": ["-0.6355977058"]}`

### dup_exact_38106b9781dd / phase1b_architecture_exhibition_visit / path_has_architecture

- feature_family: path_derived
- values: `{"raw_34da4bc897": ["1"], "raw_ea42a7592c": ["0"]}`

### dup_exact_38106b9781dd / phase1b_architecture_exhibition_visit / path_has_garden

- feature_family: path_derived
- values: `{"raw_34da4bc897": ["0"], "raw_ea42a7592c": ["1"]}`

### dup_exact_39d154f18e8e / phase1b_architecture_exhibition_visit / dinov2_family_margin

- feature_family: diagnostic_numeric
- values: `{"raw_19c9aa3ffa": ["0.04396629333"], "raw_b37dfe627b": ["0"]}`

### dup_exact_39d154f18e8e / phase1b_architecture_exhibition_visit / path_has_architecture

- feature_family: path_derived
- values: `{"raw_19c9aa3ffa": ["1"], "raw_b37dfe627b": ["0"]}`

### dup_exact_39d154f18e8e / phase1b_architecture_exhibition_visit / path_has_garden

- feature_family: path_derived
- values: `{"raw_19c9aa3ffa": ["0"], "raw_b37dfe627b": ["1"]}`

### dup_exact_6e9a8dfc5211 / phase1b_architecture_exhibition_visit / dinov2_campaign_margin

- feature_family: diagnostic_numeric
- values: `{"raw_5ed5b2d6c4": ["0.8143937141"], "raw_9d0d132603": ["0.5332327038"]}`

### dup_exact_6e9a8dfc5211 / phase1b_architecture_exhibition_visit / dinov2_campaign_pos_nn_sim

- feature_family: diagnostic_numeric
- values: `{"raw_5ed5b2d6c4": ["1.000000119"], "raw_9d0d132603": ["0.7188391089"]}`

### dup_exact_6e9a8dfc5211 / phase1b_architecture_exhibition_visit / path_has_architecture

- feature_family: path_derived
- values: `{"raw_5ed5b2d6c4": ["0"], "raw_9d0d132603": ["1"]}`

### dup_exact_6e9a8dfc5211 / phase1b_architecture_exhibition_visit / path_has_garden

- feature_family: path_derived
- values: `{"raw_5ed5b2d6c4": ["1"], "raw_9d0d132603": ["0"]}`

### dup_exact_b72fd4e92342 / phase1b_architecture_exhibition_visit / dinov2_campaign_margin

- feature_family: diagnostic_numeric
- values: `{"raw_572e8cc431": ["0.1939848661"], "raw_a820c309c8": ["0.5485260487"]}`

### dup_exact_b72fd4e92342 / phase1b_architecture_exhibition_visit / dinov2_campaign_pos_nn_sim

- feature_family: diagnostic_numeric
- values: `{"raw_572e8cc431": ["0.6454588175"], "raw_a820c309c8": ["1"]}`

### dup_exact_b72fd4e92342 / phase1b_architecture_exhibition_visit / dinov2_family_margin

- feature_family: diagnostic_numeric
- values: `{"raw_572e8cc431": ["-0.3679496348"], "raw_a820c309c8": ["-0.6957803667"]}`

### dup_exact_b72fd4e92342 / phase1b_architecture_exhibition_visit / path_has_architecture

- feature_family: path_derived
- values: `{"raw_572e8cc431": ["1"], "raw_a820c309c8": ["0"]}`
