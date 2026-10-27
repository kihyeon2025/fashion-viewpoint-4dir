#!/bin/bash
# =========================================================================
# run_modern_cnn.sh — 기존 ResNet50 실험 전체를 최신 CNN으로 동일하게 재실행
# -------------------------------------------------------------------------
# 하이퍼파라미터는 ResNet50 실험과 완전히 같게 둠 (공정 비교):
#   AdamW lr 1e-4, wd 1e-4, Cosine, batch 32, 최대 30 epoch, 조기종료(val Macro F1, patience 10)
#
# 사용법 (서버, 프로젝트 폴더에서)
#   conda activate <환경이름>
#   bash run_modern_cnn.sh                    # 기본: convnext_tiny
#   bash run_modern_cnn.sh efficientnet_v2_s  # 다른 모델
#   선택 가능: convnext_tiny / convnext_small / efficientnet_v2_s / regnety_8gf
#
# 이미 끝난 run(result.json 있음)은 건너뛰므로, 중간에 끊겨도 다시 실행하면 이어서 진행.
#
# [빠르게: GPU 여러 개로 seed 나눠 동시에 돌리기]  (nvidia-smi 로 GPU 개수 확인)
#   tmux 창 3개에서 각각:
#   CUDA_VISIBLE_DEVICES=0 SEEDS=42 SKIP_EVAL=1 bash run_modern_cnn.sh
#   CUDA_VISIBLE_DEVICES=1 SEEDS=43 SKIP_EVAL=1 bash run_modern_cnn.sh
#   CUDA_VISIBLE_DEVICES=2 SEEDS=44 SKIP_EVAL=1 bash run_modern_cnn.sh
#   셋 다 끝나면:  bash run_modern_cnn.sh   (학습은 전부 건너뛰고 평가만 실행)
# 학습 로그는 runs_vp/<run이름>/train.log 에 저장.
# =========================================================================
set -e
MODEL=${1:-convnext_tiny}
DATA=${DATA:-vp4_dataset}
SHOP=${SHOP:-shop_test}
OUT=${OUT:-runs_vp}
SEEDS=${SEEDS:-"42 43 44"}
NW=${NW:-8}            # 데이터 로딩 프로세스 수 (GPU 사용률이 낮으면 늘리기)

case $MODEL in
  convnext_tiny)     TAG=cnxt ;;
  convnext_small)    TAG=cnxs ;;
  efficientnet_v2_s) TAG=effv2s ;;
  regnety_8gf)       TAG=regy8 ;;
  *) echo "알 수 없는 모델: $MODEL"; exit 1 ;;
esac

run () {  # run <이름> <추가 옵션...>
  local name=$1; shift
  local dir=$OUT/$name
  if [ -f "$dir/result.json" ]; then echo "[건너뜀] $name (이미 완료)"; return; fi
  mkdir -p "$dir"
  echo "=============== [$(date '+%m-%d %H:%M')] $name ==============="
  python3 train_viewpoint.py --data_root $DATA --model $MODEL --num_workers $NW --out_dir "$dir" "$@" 2>&1 | tee "$dir/train.log"
}

for s in $SEEDS; do
  # ---- 본 실험 ----
  run ${TAG}_4c_s$s          --mode 4class --seed $s
  run ${TAG}_3c_s$s          --mode 3class --seed $s
  # ---- 어블레이션 A: 라벨 출처 (정면/비정면 2분류) ----
  run ${TAG}_2c_manual_s$s   --mode 2class --label_source manual --seed $s
  run ${TAG}_2c_df2_s$s      --mode 2class --label_source df2    --seed $s
  # ---- 어블레이션 B: 좌우 반전 처리 ----
  run ${TAG}_4c_noflip_s$s   --mode 4class --flip none   --seed $s
  run ${TAG}_4c_noswap_s$s   --mode 4class --flip noswap --seed $s
  # ---- 어블레이션 C: 사전학습 없이 ----
  run ${TAG}_4c_scratch_s$s  --mode 4class --no_pretrained --seed $s
done

# 여러 GPU로 seed를 나눠 돌릴 때는 마지막 프로세스만 평가하도록 SKIP_EVAL=1 사용
if [ "${SKIP_EVAL:-0}" = "1" ]; then echo "학습 완료 (평가 생략: SKIP_EVAL=1)"; exit 0; fi
for s in 42 43 44; do
  for k in 4c 3c 2c_manual 2c_df2 4c_noflip 4c_noswap 4c_scratch; do
    [ -f "$OUT/${TAG}_${k}_s$s/result.json" ] || { echo "아직 안 끝난 run: ${TAG}_${k}_s$s -> 평가는 전부 끝난 뒤 다시 실행하세요"; exit 0; }
  done
done

echo; echo "=============== 쇼핑몰 200장 평가 ==============="
python3 predict_viewpoint.py --runs $OUT/${TAG}_4c_s42 $OUT/${TAG}_4c_s43 $OUT/${TAG}_4c_s44 \
    --images $SHOP --out $OUT/${TAG}_4c_shop.csv | tee $OUT/${TAG}_4c_shop.log
python3 predict_viewpoint.py --runs $OUT/${TAG}_3c_s42 $OUT/${TAG}_3c_s43 $OUT/${TAG}_3c_s44 \
    --images $SHOP --out $OUT/${TAG}_3c_shop.csv | tee $OUT/${TAG}_3c_shop.log

echo; echo "=============== 어블레이션 통합 비교 (${MODEL}) ==============="
python3 ablation_eval.py --data_root $DATA --shop_root $SHOP \
    --runs $OUT/${TAG}_*_s4[234] --out $OUT/${TAG}_ablation_summary.json | tee $OUT/${TAG}_ablation.log

echo; echo "완료. 결과 보낼 파일:"
echo "  $OUT/${TAG}_4c_shop.log  $OUT/${TAG}_3c_shop.log  $OUT/${TAG}_ablation.log"
echo "  $OUT/${TAG}_*_s4[234]/result.json"
