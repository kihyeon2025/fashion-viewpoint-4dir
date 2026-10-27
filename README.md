# 실험 코드: 쇼핑몰 의류 이미지 4방향 촬영 방향 분류

석사 학위논문 「공개 데이터셋 기반 4방향 시점 라벨 구축과 외형-자세 결합을 통한 쇼핑몰 의류 이미지 촬영 방향 분류」(남기현, 충북대학교 산업인공지능학과)의 실험 코드입니다.

- 실행 환경: NVIDIA DGX Station (Tesla V100 32GB × 4), Ubuntu 22.04.5, Python 3.11.16, PyTorch 2.14.0 + torchvision 0.29.0 (CUDA 12.6)
- 필요 패키지: `pip install -r requirements.txt` (PyTorch는 CUDA 버전에 맞게 별도 설치)

---

## 1. 폴더 구성

| 폴더 | 내용 |
|---|---|
| `code/` | 논문에 쓴 모든 실험 코드. **서버에서는 이 폴더의 파일을 한 폴더에 함께 두고 실행** (서로 import 함) |
| `figures/` | 논문·발표 그림을 만든 스크립트 (로컬 PC, matplotlib + Noto Sans CJK 글꼴 필요) |

---

## 2. 전체 실행 순서

```
[로컬 PC] ① 라벨링 → ② 전신 필터 → ③ 분할·내보내기 → 서버로 전송
[서버]    ④ 외형 모델 학습(본 실험+어블레이션) → ⑤ 심화 실험(노이즈·라벨 수)
          → ⑥ 관절 추출·자세 모델 → ⑦ 외형-자세 결합 → ⑧ 정성 결과 그림
[로컬 PC] ⑨ 논문 그림
```

### ① 4방향 라벨링 (로컬, `label_viewpoint.py`)
DeepFashion2 train 이미지를 한 장씩 보며 키보드로 라벨을 붙이는 도구 (tkinter).
키: 1 정면 / 2 후면 / 3 좌측면 / 4 우측면 / S 제외 / Z 되돌리기 / Q 종료. 매 장 자동 저장.
```
python label_viewpoint.py --raw_root "<DF2>/train" --out labels.csv --candidates fullbody_candidates.json
python label_viewpoint.py --raw_root "<DF2>/train" --out labels.csv --recheck 300   # 자기 일치도용 재라벨링
python label_viewpoint.py --raw_root "<DF2>/train" --out labels.csv --agreement     # 일치율·Cohen kappa 계산
```
결과: `labels.csv` (이미지 파일명, DF2 원래 viewpoint, 직접 라벨) → 6,000장 중 5,963장 사용, 37장 제외.

### ② 전신 1인 필터 (로컬, `filter_fullbody.py`)
Keypoint R-CNN으로 "사람 1명 + 머리 + 어깨 + 발목이 보이는" 사진만 라벨링 후보로 남김.
```
python filter_fullbody.py --raw_root "<DF2>/train" --candidates labels_candidates.json --out fullbody_candidates.json
```

### ③ 학습·검증·평가 분할 (로컬, `export_labeled.py`)
pair_id(같은 의류) 단위로 나눠 데이터 누수를 막음 → 학습 4,170 / 검증 897 / 평가 896.
```
python export_labeled.py --raw_root "<DF2>/train" --labels labels.csv --out_root vp4_dataset
```
결과: `vp4_dataset/(train|val|test)/(front|back|left|right)/`, `split_manifest.csv` (재현용 분할 기록).
쇼핑몰 평가셋 200장은 같은 구조로 `shop_test/(front|back|left|right)/` (방향별 50장).

### ④ 외형 모델 학습: 본 실험 + 어블레이션 (서버)
`run_parallel.py`가 빈 GPU를 자동으로 찾아 seed 3회 × 7개 실험 = 21회 학습을 병렬 실행하고, 끝나면 쇼핑몰 평가와 어블레이션 비교까지 실행.
```
conda activate <환경이름>   # 프로젝트 폴더에서 실행
python3 run_parallel.py --models convnext_tiny            # 주 모델
python3 run_parallel.py --models resnet50                 # 기준 모델
python3 run_parallel.py --models convnext_tiny --dry_run  # 무엇을 돌릴지만 확인
```
| 실험 이름 꼬리 | 내용 | train_viewpoint.py 옵션 |
|---|---|---|
| `4c` | 4분류 (주 과제) | `--mode 4class` |
| `3c` | 3분류 (정면/후면/측면) | `--mode 3class` |
| `2c_manual` | 2분류, 구축 라벨 | `--mode 2class --label_source manual` |
| `2c_df2` | 2분류, DF2 원래 라벨 (어블레이션 A) | `--mode 2class --label_source df2` |
| `4c_noflip` | 반전 없음 (어블레이션 B) | `--flip none` |
| `4c_noswap` | 반전, 라벨 유지 (어블레이션 B) | `--flip noswap` |
| `4c_scratch` | 사전학습 없음 (어블레이션 C) | `--no_pretrained` |

run 폴더 이름: `runs_vp/<모델태그>_<실험>_s<seed>` (예: `runs_vp/cnxt_4c_s42`). 모델 태그: `r50`=ResNet50, `cnxt`=ConvNeXt-Tiny.
각 run 폴더: `best_model.pt`, `result.json`(평가셋 지표·혼동행렬·학습 기록), `train.log`.
클래스 가중치 실험(ResNet50)은 `train_viewpoint.py ... --class_weight`로 따로 실행.
(`run_modern_cnn.sh`는 run_parallel.py 이전에 쓰던 순차 실행 버전으로, 같은 실험을 GPU 1장에서 차례로 돌림.)

평가만 다시 할 때:
```
python3 predict_viewpoint.py --runs runs_vp/cnxt_4c_s42 runs_vp/cnxt_4c_s43 runs_vp/cnxt_4c_s44 --images shop_test --out cnxt_4c_shop.csv
python3 ablation_eval.py --data_root vp4_dataset --shop_root shop_test --runs runs_vp/cnxt_*_s4[234]
```

### ⑤ 심화 실험: 라벨 노이즈 · 라벨 수 곡선 (서버)
```
python3 run_parallel.py --models convnext_tiny --suite deep      # 노이즈 8종 + 라벨 수 3종, seed 3회
python3 deep_curves.py --summary runs_vp/cnxt_ablation_summary.json --tag cnxt
```
| 실험 이름 꼬리 | 내용 |
|---|---|
| `4c_nrand10~40` | 학습 라벨의 10~40%를 다른 세 방향 중 하나로 무작위 교체 |
| `4c_nadj10~40` | 이웃 방향으로만 교체 (정면·후면 ↔ 좌·우측면) |
| `4c_frac10/25/50` | 학습 이미지의 10/25/50%만 사용 (방향별 비율 유지) |

노이즈는 학습셋에만 넣고 검증·평가는 깨끗한 라벨. 같은 seed면 같은 이미지가 바뀜(재현 가능).

### ⑥ 관절 추출과 자세 모델 K1·K2 (서버, `keypoint_baseline.py`)
Keypoint R-CNN(COCO 사전학습, 추가 학습 없음)으로 관절 17개를 추출해 캐시로 저장하고, 규칙 기반 K1과 학습 기반 K2(52차원 특징 + HistGradientBoosting)를 평가.
```
python3 keypoint_baseline.py --data_root vp4_dataset --shop_root shop_test --out kp_baseline
```
결과: `kp_baseline/kp_{train,val,test,shop}.json` (관절 캐시, 이후 재사용), `kp_baseline/kp_result.json`.

### ⑦ 외형-자세 결합 (서버, `fusion_eval.py`)
K2를 캐시로 다시 학습(수 초)하고, CNN 3개 확률과 결합. 가중치 w는 검증셋 Macro F1로 선택(결과 w=0.6). 상호보완성 표와 오라클 정확도도 출력.
```
CUDA_VISIBLE_DEVICES=0 python3 fusion_eval.py \
    --runs runs_vp/cnxt_4c_s42 runs_vp/cnxt_4c_s43 runs_vp/cnxt_4c_s44 \
    --kp_dir kp_baseline --out fusion_result.json
```

### ⑧ 정성 결과 그림 (서버, `sample_grid.py`)
최종 결합 모델의 예측으로 방향별 정답 예시, 오분류 예시, "결합으로 바로잡힌 예시" 그림과 이미지별 예측 CSV 생성.
```
CUDA_VISIBLE_DEVICES=0 python3 sample_grid.py \
    --runs runs_vp/cnxt_4c_s42 runs_vp/cnxt_4c_s43 runs_vp/cnxt_4c_s44 \
    --kp_dir kp_baseline --w 0.6 --out sample_fig
```

### ⑨ 논문 그림 (로컬, `figures/`)
서버 결과 수치를 스크립트 안에 넣어 그림을 그림 (서버에 matplotlib가 없어 로컬에서 제작).
| 스크립트 | 그림 |
|---|---|
| `fig_problem.py` | DF2 원래 라벨 vs 직접 라벨 누적 막대 (문제 정의) |
| `fig1.py` | 제안 방법 전체 구조도 (두 갈래 파이프라인) |
| `report_figs.py` | 방법별 정확도 비교, 어블레이션(ResNet50 vs ConvNeXt), 외형·자세 상호보완성, 쇼핑몰 방향별 정확도 |
| `deep_fig.py` | 라벨 노이즈 곡선, 라벨 수 곡선 |

---

## 3. 논문 결과 ↔ 코드 ↔ 결과 파일

| 논문 결과 | 코드 | 결과 파일 |
|---|---|---|
| DF2 라벨 오류 정량화 (26.3%, 82.4%) | `label_viewpoint.py` (labels.csv의 DF2 viewpoint와 직접 라벨 교차 집계) | `labels.csv` |
| 데이터 분할 4,170 / 897 / 896 | `export_labeled.py` | `vp4_dataset/split_manifest.csv` |
| 4분류·3분류 정확도, 좌우 혼동률 (ResNet50, ConvNeXt-T) | `run_parallel.py` → `train_viewpoint.py` | `runs_vp/*/result.json` |
| 쇼핑몰 200장 정확도·앙상블·방향별 | `predict_viewpoint.py` | `<tag>_4c_shop.log`, `<tag>_3c_shop.log` |
| 어블레이션 A·B·C, 재채점 | `ablation_eval.py` | `<tag>_ablation.log`, `<tag>_ablation_summary.json` |
| 라벨 노이즈·라벨 수 곡선 | `run_parallel.py --suite deep`, `deep_curves.py` | `cnxt_ablation_summary.json`, `cnxt_deep_curves.json` |
| 관절 모델 K1·K2 | `keypoint_baseline.py` | `kp_baseline/kp_result.json` |
| 외형-자세 결합 86.27% / 93.5%, 상호보완성 | `fusion_eval.py` | `fusion_result.json` |
| 분류 결과 샘플 그림 | `sample_grid.py` | `sample_fig/*.jpg`, `pred_*.csv` |

---

## 4. 응용 실험 (논문 본 결과 외)

| 코드 | 용도 |
|---|---|
| `shop_auto_sort.py` | 정답 없는 쇼핑몰 사진을 전신 필터 → 4방향 자동 분류 → 신뢰도·일치율 요약 |
| `df2_extract.py` | 라벨링에 쓰지 않은 DF2 이미지(validation 권장)를 추출 |
| `df2_compare.py` | 위 자동 분류 결과를 DF2 원래 라벨과 비교 (그림 2의 대규모 자동 버전) |

---

## 5. 주의 사항

- **관절 특징은 52차원**입니다 (정규화 좌표 34 + 관절 확신도 17 + 상자 가로세로 비율 1). 초기 문서의 "51차원"은 오기입니다.
- `models.py`에는 연구 초기에 시험한 다른 모듈(NSCM, SBSM, CBAM)도 함께 들어 있습니다. 최종 논문에서 쓰는 것은 `resnet50`(IMAGENET1K_V2)과 `convnext_tiny`(IMAGENET1K_V1)입니다.
- 이미 끝난 run(`result.json` 있음)은 `run_parallel.py`가 건너뛰므로, 중간에 끊겨도 같은 명령으로 이어서 실행됩니다. 오래 걸리는 실험은 tmux 안에서 실행하세요.
- `predict_viewpoint.py` 맨 위 설명의 "측면·후면(sideback)" 모드는 DF2 원래 라벨 시기의 사용법이며, 4분류 run을 주면 4방향으로 채점합니다.
- **데이터는 포함되어 있지 않습니다.** DeepFashion2는 [공식 저장소](https://github.com/switchablenorms/DeepFashion2)에서 이용 동의 후 받아야 하며, 라이선스상 이미지를 재배포할 수 없습니다. 쇼핑몰 평가 이미지 200장도 저작권 문제로 공개하지 않습니다.
- 학습된 모델 가중치와 실험 결과 파일(`runs_vp/`, `kp_baseline/`)은 용량 문제로 포함하지 않았습니다.
