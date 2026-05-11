# Raw Pool Expansion Scope — after Phase 1b First Round

## 메타데이터

spec_version: v2.2.1  
scope_name: raw_pool_expansion_after_phase_1b_first_round  
related_phase: phase_1b  
related_report: audit/phase_1b/phase_1b_first_round_report.json  
score_status: diagnostic_only  
support_explanation_status: deferred_from_phase_1b  

## 한 줄 결론

Phase 1b first round 결과, 현재 206장 raw image pool은 multi-campaign reranker 학습을 시작하기에는 coverage가 부족하다.

특히 `indoor_gallery_winter_art` campaign에서 30개 review 후보 중 29개가 reject되었으므로, 이는 reranker 모델 실패라기보다 **raw image pool coverage gap**으로 보는 것이 맞다.

따라서 다음 우선순위는 classifier smoke training이 아니라 **raw image pool 확장**이다.

## 현재 상태 요약

현재 Phase 1b first round 산출물:

retrieval rows = 250  
PairFeatureSnapshot rows = 250  
review queue rows = 150  
ReviewEvent rows = 150  
classifier TrainingSnapshot rows = 150  
ranker TrainingSnapshot rows = 150  

Label 분포:

reject = 99  
accept = 27  
acceptable = 24  

Issue tag 분포:

semantic_mismatch = 58  
season_mismatch = 56  
poor_composition = 3  
text_region_conflict = 1  

주요 진단:

1. indoor / winter / gallery 계열 raw image coverage가 약하다.
2. reject 사유가 semantic_mismatch와 season_mismatch에 집중되어 있다.
3. layout-related issue tag가 거의 없다.
4. classifier smoke training은 가능하지만, 현재 pool bias를 그대로 학습할 위험이 있다.
5. support explanation은 baseline과 critic signal이 없으므로 여전히 deferred 상태가 맞다.

## Raw pool 확장의 목적

Raw pool 확장의 목적은 모델 성능을 주장하기 위함이 아니다.

목적은 다음과 같다.

1. campaign family diversity를 이미지 후보 pool이 실제로 받쳐주는지 확인한다.
2. indoor / winter / gallery / architecture / botanical 계열의 retrieval coverage를 개선한다.
3. semantic_mismatch와 season_mismatch가 단순히 pool 부족 때문에 발생하는지 분리한다.
4. Phase 1b campaign 5개에 대해 retrieval 후보의 quality와 diversity를 다시 측정한다.
5. 이후 classifier smoke training이 pool bias만 학습하지 않도록 한다.

## 확장 목표

현재 raw pool:

raw_gallery_count = 206

1차 확장 목표:

target_raw_image_count_minimum = 1000

권장 확장 목표:

target_raw_image_count_preferred = 3000_to_5000

이 숫자는 quality threshold가 아니다.  
작업 규모를 정하기 위한 collection target이다.

최종 accept/reject 또는 model quality threshold로 사용하지 않는다.

## 우선 보강할 이미지 계열

Phase 1b first round 결과를 기준으로 우선순위는 다음과 같다.

### 1. indoor / gallery / winter 계열

가장 우선순위가 높다.

이유:

indoor_gallery_winter_art campaign에서 reject가 29/30으로 발생했다.  
현재 pool에는 실내 전시, 겨울, 조용한 실내 공간, gallery-like 이미지가 부족하다.

보강 대상:

indoor gallery  
interior exhibition  
winter garden  
snow / winter landscape  
quiet indoor space  
low-light gallery  
minimal architecture interior  
calm winter background  

### 2. architecture 계열

architecture_exhibition_visit campaign은 상대적으로 usable candidate가 많았다.  
다만 건축/정원 duplicate path가 많이 섞였으므로, 더 다양한 건축 이미지가 필요하다.

보강 대상:

architecture exterior  
architecture detail  
courtyard  
pavilion  
stone / wood / concrete texture  
architectural path  
door / gate / corridor  
quiet built environment  

### 3. botanical / flower 계열

botanical_spring_program에서 season_mismatch와 semantic_mismatch가 많이 나왔다.

보강 대상:

spring flower  
early spring botanical  
plant close-up  
garden plant  
native flower  
seasonal bloom  
leaf / stem / botanical detail  

주의:

봄/여름/초가을 구별이 이미지상 애매한 경우가 많으므로, raw metadata에 season을 무리하게 확정하지 않는다.  
가능하면 `season_unknown` 또는 `season_candidate` 형태로 둔다.

### 4. garden walking 계열

summer / autumn garden walking은 기존 pool에서도 어느 정도 후보가 나왔다.  
하지만 season axis가 강하게 작동했으므로 계절 다양성을 보강한다.

보강 대상:

summer garden path  
autumn garden path  
green walking path  
fallen leaves path  
forest path  
quiet trail  
garden wide shot  
seasonal garden landscape  

## 수집/저장 원칙

Raw image는 다음 구조로 넣는다.

data/raw/
  gallery/
  arbor/
  course/
  external_expansion/

권장 추가 구조:

data/raw/external_expansion/
  indoor_gallery/
  winter/
  architecture/
  botanical/
  garden_walking/

또는 기존 ontology에 맞춰 정리할 수 있으면 다음처럼 넣는다.

data/raw/gallery/건축/
data/raw/gallery/정원/
data/raw/arbor/flower/
data/raw/arbor/tree/
data/raw/course/

중요한 원칙:

1. 원본 파일은 git에 넣지 않는다.
2. symlink 또는 ignored data directory를 유지한다.
3. raw manifest로만 추적한다.
4. 파일명은 가능하면 안정적으로 유지한다.
5. 같은 이미지의 중복 저장은 허용하되 duplicate_group artifact에서 잡히게 둔다.
6. 출처/라이선스가 중요한 이미지라면 별도 metadata file에 남긴다.

## Metadata policy

Phase 1b에서는 metadata를 과신하지 않는다.

특히 season metadata는 다음처럼 취급한다.

season_known:
  when: 촬영 시점 또는 식물 개화 시기가 명확한 경우

season_candidate:
  when: 이미지상 추정은 가능하지만 확실하지 않은 경우

season_unknown:
  when: 이미지로 봄/여름/초가을 구분이 어려운 경우

Annotation rule:

Season mismatch is used only when the reviewer judges the seasonal mismatch to be visually clear or botanically clear from known bloom timing. Ambiguous spring/summer/early-autumn cases should not be rejected solely for season and may be labeled acceptable.

## 재생성해야 할 artifacts

Raw pool 확장 후 다음 artifact를 전부 다시 만든다.

1. raw image manifest
2. CLIP image embeddings
3. DINOv2 image embeddings
4. embedding indexes
5. exact duplicate groups
6. DINOv2 clusters
7. region safety maps
8. Phase 1b campaign retrieval candidates
9. PairFeatureSnapshots
10. review queues
11. contact sheets

기존 Phase 1b first round labeled CSV와 ReviewEvent는 보존한다.

새 raw pool 기반 결과는 별도 version으로 생성한다.

권장 version:

raw_image_manifest_v2_2_1_expanded_v1  
clip_image_embeddings_expanded_v1  
dinov2_image_embeddings_expanded_v1  
duplicate_groups_expanded_v1  
dinov2_clusters_expanded_v1  
region_safety_maps_expanded_v1  
review_queue_phase1b_expanded_v1  

## 비교해야 할 것

Raw pool 확장 전후로 다음을 비교한다.

### 1. Retrieval coverage

campaign별 top-50 category/source_group 분포 비교

특히 확인할 campaign:

phase1b_indoor_gallery_winter_art  
phase1b_botanical_spring_program  
phase1b_architecture_exhibition_visit  

### 2. Review queue quality

campaign별 30개 review queue에서 reject rate 비교

중요 비교:

indoor_gallery_winter_art reject rate  
botanical_spring_program season_mismatch rate  
summer/autumn garden semantic_mismatch rate  

### 3. Issue tag distribution

확장 후에도 semantic_mismatch / season_mismatch만 강하게 나오면, prompt design 또는 campaign payload 자체를 재검토한다.

layout-related tag가 여전히 거의 나오지 않으면 preview renderer v1 도입을 진행한다.

### 4. Duplicate pressure

확장 후에도 top-50 안에 duplicate group이 많이 반복되면, retrieval stage에서 duplicate suppression을 더 이르게 적용하는 방안을 검토한다.

## 성공 기준

Raw pool expansion은 model quality 기준으로 성공/실패를 판단하지 않는다.

성공 기준은 다음과 같은 diagnostic 조건이다.

1. raw image count가 최소 1000장 이상으로 증가한다.
2. indoor / winter / gallery 계열 이미지가 명시적으로 추가된다.
3. expanded manifest가 정상 생성된다.
4. CLIP / DINOv2 embedding이 전체 expanded pool에 대해 생성된다.
5. duplicate / cluster / region safety artifact가 정상 생성된다.
6. Phase 1b 5개 campaign retrieval이 expanded pool에서 다시 실행된다.
7. indoor_gallery_winter_art의 후보 다양성이 증가한다.
8. review queue duplicate suppression이 계속 유지된다.
9. 모든 score는 diagnostic_only로 유지된다.

## 실패 또는 재수집 trigger

다음 중 하나가 발생하면 raw pool 확장을 다시 설계한다.

1. indoor / winter / gallery 이미지가 충분히 추가되지 않음
2. expanded retrieval에서도 indoor_gallery_winter_art가 거의 전부 outdoor/garden 후보만 반환
3. duplicate group이 top candidates를 계속 지배
4. botanical campaign에서 식물/꽃 후보가 충분히 나오지 않음
5. architecture campaign에서 건축 이미지가 아닌 정원 이미지가 과도하게 섞임
6. source metadata가 너무 불안정해서 campaign별 분석이 어려움
7. image quality가 낮아 poor_composition reject가 크게 증가

## 금지할 해석

Raw pool expansion 후에도 다음 해석은 금지한다.

1. CLIP score가 높으므로 accept 가능하다는 주장
2. retrieval margin이 threshold를 넘었다는 주장
3. classifier smoke training 결과를 production quality로 해석
4. ranker metric을 generalization 성능으로 해석
5. issue_tag 감소를 model improvement로 단정
6. raw pool 확장만으로 support explanation baseline이 충분해졌다는 주장

모든 판단은 계속 diagnostic_only다.

## 다음 구현 순서

1. external_expansion 디렉터리 구조 생성
2. indoor / winter / gallery 우선 이미지 수집
3. architecture / botanical / garden_walking 보강
4. raw file count 확인
5. expanded raw manifest 생성
6. expanded CLIP embedding 생성
7. expanded DINOv2 embedding 생성
8. expanded duplicate/cluster/region safety artifact 생성
9. Phase 1b 5개 campaign retrieval 재실행
10. expanded retrieval audit report 생성
11. 기존 Phase 1b first round와 비교
12. 그 다음 preview renderer v1 또는 classifier smoke training 여부 결정

## 최종 결정

Phase 1b first round 이후 다음 우선순위는 raw image pool expansion이다.

Classifier smoke training은 가능하지만, 현재 pool coverage gap이 크므로 먼저 실행하지 않는다.

Support explanation은 여전히 Phase 1b에서 defer한다.

Preview renderer v1은 raw pool expansion 이후에도 layout-related issue tag가 부족하면 다음 우선순위로 진행한다.
