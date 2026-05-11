# 꽃 개화시기 기반 후보 제외 규칙

## 왜 이 규칙을 넣었나

Phase 1b에서 계절이 있는 campaign을 만들면서 꽃 이미지가 잘못 섞이는 문제가 있었다.

예를 들어 봄 campaign에 산수국, 수련, 참나리처럼 여름에 피는 꽃 이미지가 들어오거나, 여름 campaign에 진달래처럼 봄에 피는 꽃 이미지가 들어오는 식이다.

이 문제는 classifier가 학습해서 해결할 문제가 아니다. 이미 사유원 식물 DB에 꽃별 개화시기가 정리되어 있으므로, 계절과 맞지 않는 known flower는 retrieval 후보 단계에서 제거하는 것이 더 직접적이다.

## 적용한 규칙

규칙은 단순하다.

- campaign에 요청 계절이 있다.
- 이미지의 subject_name이 DB에 등록된 꽃이다.
- 그 꽃의 개화시기에 campaign 요청 계절이 없다.
- 그러면 해당 campaign-image pair를 후보에서 제외한다.

예시는 다음과 같다.

- 봄 campaign + 산수국 → 제외
- 봄 campaign + 참나리 → 제외
- 여름 campaign + 진달래 → 제외
- 가을 campaign + 수련 → 제외
- 가을 campaign + 참나리 → 제외

이 규칙은 이미지 자체를 reject하는 규칙이 아니다. 특정 campaign에서만 제외하는 campaign-specific filter다.

예를 들어 산수국 이미지는 봄 campaign에서는 제외되지만, 여름 campaign에서는 제외되지 않는다.

## 사용한 데이터

꽃 개화시기 정보는 다음 파일에서 가져온다.

- configs/domain_knowledge/sayuwon_entity_knowledge_v1.json

이 파일에는 사유원의 식물 이름, 식물 유형, 학명, 개화시기 등이 들어 있다.

DB에는 다음 테이블로 저장된다.

- plant_entities
- plant_names
- plant_bloom_priors

꽃 계절 불일치로 제외된 campaign-image pair는 다음 테이블에 저장된다.

- campaign_image_flower_season_exclusions

## 관측 결과

원본 retrieval 후보는 300개였다.

꽃 개화시기 규칙을 적용한 뒤 effective 후보는 286개가 되었다.

- 원본 후보: 300
- 제외 후보: 14
- 남은 후보: 286

제외된 14개를 기존 review label과 대조했다.

- 제외된 accept: 0
- 제외된 acceptable: 0
- 제외된 reject: 9
- 제외된 unlabeled: 5

즉 이 규칙은 사람이 좋다고 판단한 후보를 제거하지 않았다. 반대로 사람이 reject한 후보 9개를 제거했다.

원본 labeled 후보와 effective labeled 후보의 변화는 다음과 같다.

원본 labeled 후보:

- accept: 37
- acceptable: 31
- reject: 112

규칙 적용 후 labeled 후보:

- accept: 37
- acceptable: 31
- reject: 103

accept와 acceptable은 그대로 유지되었고, reject만 줄었다.

## 해석

이 결과는 꽃 개화시기 규칙이 retrieval pre-filter로 유효하다는 뜻이다.

이 규칙의 목적은 classifier 성능을 높이는 것이 아니다. 명백히 계절과 맞지 않는 꽃 후보를 DB 지식으로 사전에 제거하는 것이다.

따라서 이 규칙을 평가할 때 classifier smoke metric을 기준으로 삼으면 안 된다.

실제로 이 규칙을 적용하면 classifier training set에서 쉬운 reject가 빠진다. 그러면 남은 문제는 더 어려워지고 classifier smoke metric은 낮아질 수 있다. 그것은 규칙이 실패했다는 뜻이 아니다.

이번 경우에도 classifier metric보다 중요한 사실은 다음이다.

- reviewed positive 제거: 0
- reviewed reject 제거: 9

이 기준에서 규칙은 통과했다.

## downstream 사용 방식

후속 retrieval 후보 처리는 원본 retrieval_candidates가 아니라 다음 view를 사용한다.

- v_effective_retrieval_candidates_v1

후속 pair feature 처리는 다음 view를 사용한다.

- v_effective_pair_features_v1

제외된 후보를 확인할 때는 다음 view를 사용한다.

- v_effective_retrieval_candidates_excluded_v1

## 이 규칙이 하지 않는 것

이 규칙은 다음을 하지 않는다.

- 이미지의 실제 촬영 계절을 추정하지 않는다.
- spring/summer를 image temporal_axis label로 자동 저장하지 않는다.
- 꽃이 아닌 나무 subject를 강제로 제외하지 않는다.
- subject_name이 없거나 DB에 없는 식물은 제외하지 않는다.
- CLIP score를 calibrated threshold처럼 사용하지 않는다.

## 결론

꽃 개화시기 기반 제외 규칙은 retrieval 후보 품질을 개선하는 DB 기반 pre-filter다.

현재 관측에서는 reviewed accept와 acceptable 후보를 하나도 제거하지 않았고, reviewed reject 후보 9개를 제거했다.

따라서 이 규칙은 유지한다.
