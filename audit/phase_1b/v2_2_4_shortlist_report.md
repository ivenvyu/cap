
## 추가 정성 검토 메모: near-duplicate 문제

v2.2.4 shortlist를 빠르게 검토한 결과, active campaign에서 model_top_shortlist 후보는 대체로 사용 가능해 보였다. 사용자의 정성 검토 기준으로는 모델이 높게 본 후보의 약 90% 정도가 괜찮아 보였다.

다만 near-duplicate 문제가 관찰되었다.

예시:

- `raw_22d63fd86a`
- `raw_41179036b8`

두 이미지는 시각적으로 거의 같은 사진처럼 보이지만 서로 다른 image_id로 취급되었고, shortlist 내에서 서로 다른 bucket에 배정되었다.

해석:

현재 shortlist dedupe는 `image_id` 또는 기존 `duplicate_group_id`에 의존한다. 따라서 동일하거나 거의 동일한 사진이 다른 파일명, 다른 image_id, 다른 경로, 다른 압축/해상도로 들어오면 중복 억제가 충분히 작동하지 않을 수 있다.

결정:

v2.2.4 shortlist는 운영상 유용하지만, near-duplicate suppression을 보강해야 한다.

다음 수정 후보:

1. 동일 campaign 내 DINOv2 similarity 기반 near-duplicate audit 추가
2. shortlist 생성 시 `image_id + duplicate_group_id`뿐 아니라 visual-near-duplicate key로도 dedupe
3. near-duplicate suppression은 최종 품질 threshold가 아니라 shortlist 다양성 확보용 diagnostic rule로 정의
4. 중복 억제 후 model_top_shortlist를 재생성

Non-claim:

near-duplicate similarity 기준은 calibrated quality threshold가 아니다.

## 중복 억제 패치 결과

v2.2.4 shortlist builder에 feature snapshot 기반 `duplicate_group_id` 조회를 추가했다.

검증 결과:

- selected rows: 32
- same-campaign duplicate groups: 0
- `raw_22d63fd86a` / `raw_41179036b8`는 같은 exact duplicate group인 `dup_exact_c90c21a044aa`로 인식됨
- 같은 campaign 안에서는 중복 노출되지 않음

해석:

v2.2.4 shortlist의 campaign 내부 중복 억제는 정상 작동한다.

단, 같은 exact duplicate image가 서로 다른 campaign에 등장하는 것은 현재 정책상 허용된다. campaign별 shortlist를 독립적으로 생성하기 때문이다.

## 남은 이슈: duplicate group feature canonicalization

같은 exact duplicate라도 서로 다른 경로에 저장되어 있으면 path-derived feature가 달라질 수 있다.

예시:

- `raw_22d63fd86a`: `gallery/건축/유원/space5_3_2.jpg`
- `raw_41179036b8`: `gallery/정원/유원/space5_3_2.jpg`

두 이미지는 exact duplicate지만, 경로 기반 feature가 다르게 생성될 수 있다.

결정:

이 문제는 shortlist dedupe 문제가 아니라 feature canonicalization 문제로 분리한다.

v2.2.5에서 다음을 검토한다.

1. exact duplicate group마다 canonical image_id를 지정한다.
2. path-derived feature는 duplicate member별로 다르게 두지 않고 group-level로 통합하거나 multi-source flag로 변환한다.
3. `path_has_architecture`, `path_has_garden` 같은 feature는 exact duplicate에서 서로 충돌할 경우 diagnostic warning을 낸다.
4. 모델 학습/score snapshot 생성 시 canonical duplicate group feature를 사용한다.

Non-claim:

duplicate canonicalization은 품질 threshold가 아니다. 동일 파일의 metadata divergence를 줄이기 위한 데이터 정규화 단계다.
