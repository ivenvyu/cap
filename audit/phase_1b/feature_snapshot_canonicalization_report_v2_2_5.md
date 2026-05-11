# v2.2.5 Canonicalized Feature Snapshot Report

## 상태

exact duplicate group의 metadata divergence를 줄이기 위해 v2.2.5 canonicalized feature snapshot을 생성했다.

이 단계는 품질 threshold나 자동 accept/reject 규칙이 아니다.

## 요약

- input rows: 250
- output rows: 250
- exact duplicate canonicalized rows: 91
- non-duplicate rows: 159
- path-derived changed rows: 91

## Canonicalization policy

- exact duplicate group에 포함된 row에는 `canonical_image_id`를 부여한다.
- `path_has_architecture`, `path_has_garden`은 개별 파일 경로가 아니라 duplicate group-level source flag로 통일한다.
- `duplicate_has_architecture_source`, `duplicate_has_garden_source`, `duplicate_source_group_count`, `duplicate_metadata_conflict`를 feature로 추가한다.
- DINOv2 campaign/family feature는 아직 재계산하지 않고, conflict audit 결과를 보존한다.

## Non-claims

- canonicalization은 이미지 품질 판단이 아니다.
- canonicalization은 calibrated threshold가 아니다.
- 이 snapshot만으로 production 성능을 주장하지 않는다.
- DINOv2 anchor feature conflict는 다음 단계에서 group-aware recomputation 대상으로 남긴다.
