# v2.2.3 Closure Report

## 상태

v2.2.3은 v2.2.2 triage-25 결과를 반영해 training snapshot과 diagnostic classifier를 갱신한 단계다.

이 단계는 production model 성능 평가가 아니라, 다음 사항을 확인하기 위한 diagnostic update다.

- campaign review brief config 생성
- review queue policy v2.2.3 생성
- indoor_gallery_winter_art coverage-gap 분리
- triage 25개 label 중 학습 가능한 20개 반영
- diagnostic classifier 재학습

## Training snapshot 결과

- 전체 v2.2.3 classifier rows: 131
- 실제 학습 사용 rows: 126
- 제외 rows: 5

제외된 5개는 `phase1b_indoor_gallery_winter_art` coverage-gap campaign에 해당한다.

## Label 분포

- label 1: 57
- label 0: 69

## Campaign 분포

- phase1b_architecture_exhibition_visit: 32
- phase1b_autumn_garden_walk: 32
- phase1b_botanical_spring_program: 30
- phase1b_summer_garden_walk: 32

`phase1b_indoor_gallery_winter_art`는 coverage-gap diagnostic campaign으로 분리되어 일반 training에서 제외했다.

## Source snapshot 분포

- v2_2_2_base: 106
- v2_2_2_triage_25: 20

triage 25개 중 5개는 coverage-gap campaign이므로 학습에서 제외되었고, 20개만 v2.2.3 diagnostic classifier에 반영되었다.

## Diagnostic classifier 결과

- OOF balanced accuracy: 0.5903890160183066
- OOF ROC-AUC: 0.5329265191965421
- OOF average precision: 0.5411487993917143

## 해석

v2.2.3에서 balanced accuracy는 v2.2.2 대비 소폭 개선되었다.

그러나 ROC-AUC와 average precision은 여전히 낮다. 따라서 현재 모델을 production reranker나 calibrated accept/reject classifier로 해석하면 안 된다.

현재 score는 계속 `diagnostic_only`이며, threshold는 `no_calibrated_threshold` 상태다.

## 결정

1. v2.2.3 snapshot merge는 성공으로 본다.
2. indoor_gallery_winter_art는 coverage-gap campaign으로 유지한다.
3. v2.2.3 classifier는 diagnostic model로만 유지한다.
4. 추가 대량 라벨링은 하지 않는다.
5. 다음 단계는 모델 복잡도 증가가 아니라 review policy, campaign brief, prompt conflict bucket 정리다.

## Non-claims

- production model quality를 주장하지 않는다.
- campaign 간 score를 직접 비교하지 않는다.
- diagnostic score를 pass/fail threshold로 사용하지 않는다.
- triage positive rate를 최종 성능 지표로 사용하지 않는다.
- coverage-gap campaign을 일반 모델 품질 평가에 섞지 않는다.

## 추가 정성 검토 메모

사용자가 v2.2.3 review sheet를 빠르게 훑어본 결과, `phase1b_indoor_gallery_winter_art`를 제외한 active diagnostic campaign에서는 다음 경향이 관찰되었다.

- CLIP이 높게 봤지만 model이 낮게 본 이미지는 대체로 애매하거나 부적합한 경우가 많았다.
- DINOv2가 높게 봤지만 model이 낮게 본 이미지도 시각적으로는 비슷하더라도 campaign-fit은 약한 경우가 많았다.
- 반대로 model이 높게 본 이미지는 대체로 홍보물 후보로 쓸 만해 보였다.
- 겨울 실내 갤러리 전시는 여전히 적합 이미지 pool이 부족해 보였다.

해석:

이 관찰은 diagnostic model이 CLIP/DINO 단일 신호보다 campaign-fit을 더 잘 조합하고 있을 가능성을 시사한다.

단, 이는 정성적 검토 메모이며 calibrated performance claim이 아니다.  
따라서 v2.2.4에서는 model_top_diagnostic bucket의 운영상 우선순위를 높이되, diagnostic_only / no_calibrated_threshold 상태는 유지한다.

정책 반영 후보:

- `model_top_diagnostic`은 active diagnostic campaign에서 추천 shortlist의 주 bucket으로 사용한다.
- `clip_high_model_low`는 missed-positive bucket이 아니라 `clip_model_disagreement_audit` 성격으로 재해석한다.
- `dinov2_high_model_low`는 missed-positive bucket이 아니라 `dinov2_model_disagreement_audit` 성격으로 재해석한다.
- `phase1b_indoor_gallery_winter_art`는 coverage-gap campaign으로 유지한다.
