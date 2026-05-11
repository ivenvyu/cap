# v2.2.2 Triage-25 판단 보고서

## 상태

v2.2.2 triage 검토 25개를 완료했다.

이 보고서는 모델의 최종 성능 평가가 아니라, review workflow와 bucket sampling이 유효한지 확인하기 위한 진단 보고서다.

## 전체 결과

- 전체 검토 수: 25개
- accept: 7개
- acceptable: 6개
- best: 3개
- reject: 9개
- positive 총합: 16개
- reject 총합: 9개

여기서 positive는 `acceptable + accept + best`로 정의한다.

## 거절 사유 요약

- semantic_mismatch: 7개
- season_mismatch: 3개
- too_busy_background: 1개
- low_resolution: 1개
- brand_tone_mismatch: 1개

## Bucket별 해석

### clip_high_model_low

- n = 5
- positive = 3
- reject = 2
- positive_rate_diagnostic = 0.6

해석:

CLIP은 높게 봤지만 diagnostic model은 낮게 본 후보 중 실제 positive가 3개 있었다.  
즉, 현재 모델이 CLIP semantic signal을 일부 놓칠 수 있다.

결정:

이 bucket은 유지한다.  
모델이 놓치는 semantic positive를 찾는 데 유용하다.

---

### dinov2_high_model_low

- n = 5
- positive = 2
- reject = 3
- positive_rate_diagnostic = 0.4

해석:

DINOv2가 시각적으로 유사하다고 판단한 후보가 항상 campaign에 적합한 것은 아니다.  
이는 DINOv2 visual similarity와 campaign semantic fit이 다르다는 것을 보여준다.

결정:

이 bucket은 유지한다.  
DINOv2 anchor가 실제로 어디서 맞고 틀리는지 확인하는 진단 bucket으로 유용하다.

---

### model_high_clip_negative_high

- n = 5
- positive = 4
- reject = 1
- positive_rate_diagnostic = 0.8

해석:

원래 의도는 “모델 점수는 높지만 negative prompt와도 가까운 hard negative 후보”를 찾는 것이었다.  
하지만 이번 triage에서는 5개 중 4개가 positive였다.

이는 현재 negative prompt signal이 너무 넓거나, campaign별 반대 의미를 충분히 잘 잡지 못하고 있을 가능성을 시사한다.

결정:

이 bucket은 당장은 유지하되, “hard negative bucket”이라고 강하게 해석하지 않는다.  
다음 버전에서는 `prompt_conflict_audit` 성격으로 재정의하는 것이 적절하다.

---

### model_top_diagnostic

- n = 5
- positive = 3
- reject = 2
- positive_rate_diagnostic = 0.6

해석:

diagnostic model의 top 후보가 항상 좋은 것은 아니다.  
현재 단계에서는 model top만으로 review queue를 구성하면 안 된다.

결정:

이 bucket은 제한적으로 유지한다.  
top-N 단독 review queue는 사용하지 않는다.

---

### random_control

- n = 5
- positive = 4
- reject = 1
- positive_rate_diagnostic = 0.8

해석:

random control에서도 positive가 4개 나왔다.  
단, 이 random은 전체 raw image random이 아니라 retrieval candidate pool 내부 random이다.

따라서 CLIP retrieval candidate pool 자체가 이미 어느 정도 좋은 후보를 포함하고 있다는 뜻이다.  
반대로 말하면, 현재 diagnostic model이 retrieval pool random보다 명확히 우월하다고 주장할 수는 없다.

결정:

random control은 소량 유지한다.  
모델이 과하게 걸러내는 후보가 있는지 확인하는 blind-spot 점검용으로 필요하다.

## Campaign별 해석

### phase1b_architecture_exhibition_visit

- positive_rate_diagnostic = 0.8

해석:

건축/전시 방문 campaign은 현재 이미지 pool 안에서 비교적 판단 가능한 후보가 있다.

결정:

다음 진단 학습에 포함 가능하다.

---

### phase1b_autumn_garden_walk

- positive_rate_diagnostic = 0.8

해석:

가을 정원 산책 campaign도 현재 이미지 pool 안에서 비교적 잘 작동한다.

결정:

다음 진단 학습에 포함 가능하다.

---

### phase1b_botanical_spring_program

- positive_rate_diagnostic = 0.8

해석:

봄 식물 프로그램 campaign도 usable하다.  
다만 `acceptable` 라벨이 일부 존재하므로, 식물 프로그램과 일반 정원 산책 이미지 사이의 경계가 다소 모호하다.

결정:

다음 진단 학습에 포함 가능하되, botanical program과 garden walk의 구분 기준을 더 명확히 해야 한다.

---

### phase1b_indoor_gallery_winter_art

- positive_rate_diagnostic = 0.2

해석:

실내 갤러리 겨울 campaign은 reject가 많다.  
이는 모델 실패라기보다 현재 raw image pool에 실내, 겨울, 갤러리 계열 이미지가 부족한 coverage gap으로 보는 것이 타당하다.

결정:

일반 모델 성능 평가에 섞지 않는다.  
coverage-gap diagnostic campaign으로 분리한다.

---

### phase1b_summer_garden_walk

- positive_rate_diagnostic = 0.6

해석:

여름 정원 산책 campaign은 usable하지만, 계절 mismatch와 정원/건축 ambiguity가 남아 있다.

결정:

다음 진단 학습에 포함 가능하다.  
다만 가을 정원, 건축 방문 campaign과 구분되는 prompt/feature가 더 필요하다.

## 최종 결정

이번 round에서는 25개 검토로 중단한다.

150개 전체 라벨링으로 확장하지 않는다.

## 다음 작업

1. triage 25개 라벨을 human-reviewed diagnostic label로 보존한다.
2. `phase1b_indoor_gallery_winter_art`는 coverage-gap campaign으로 분리한다.
3. `model_high_clip_negative_high` bucket은 hard negative bucket이 아니라 prompt-conflict audit bucket으로 재해석한다.
4. discovery bucket은 유지한다.
5. 사람이 검토하기 전에 campaign brief를 HTML/CSV에 기본 탑재한다.
6. 아직 calibrated performance claim은 하지 않는다.

## 결론

v2.2.2 triage 결과, review queue 자체는 작동한다.

다만 현재 diagnostic model score는 최종 품질 판단 기준이 아니며, campaign 간 score 비교나 pass/fail threshold로 사용하면 안 된다.

다음 단계는 추가 라벨링이 아니라 review policy와 campaign 설명 구조를 정리하는 것이다.
