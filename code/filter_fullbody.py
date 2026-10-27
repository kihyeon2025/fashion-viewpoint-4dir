r"""
filter_fullbody.py — 라벨링 후보 중 "머리~발목까지 보이는 1인 전신 사진"만 남기기
=========================================================================
사전학습된 사람 관절 인식 모델(torchvision Keypoint R-CNN, 학습 불필요)로 판별합니다.
  통과 조건
    1) 확신도 높은 사람이 정확히 1명
    2) 머리가 보임: 코/눈/귀 중 하나라도 검출 (뒷모습은 귀·뒤통수로 잡힘)
    3) 어깨와 발목이 보임 (전신)

필요: pip install torch torchvision   (GPU 없으면 CPU로 동작, 대신 느림)

사용 예시 (라벨링 도구가 만든 후보 목록을 걸러서 새 후보 목록 생성)
  python filter_fullbody.py --raw_root "C:\...\train" --candidates labels_candidates.json --out fullbody_candidates.json
  python label_viewpoint.py --raw_root "C:\...\train" --out labels.csv --candidates fullbody_candidates.json
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
from torchvision.transforms.functional import to_tensor

# COCO 관절 번호
HEAD = [0, 1, 2, 3, 4]        # 코, 왼눈, 오른눈, 왼귀, 오른귀
SHOULDERS = [5, 6]
ANKLES = [15, 16]


def judge(output, person_thr, kp_thr, head_thr):
    """모델 출력 1개(사진 1장) -> (통과 여부, 사유)"""
    keep = output["scores"] > person_thr
    n = int(keep.sum())
    if n == 0:
        return False, "사람 없음"
    if n > 1:
        return False, "여러 명"
    ks = output["keypoints_scores"][keep][0]
    head = any(ks[i] > head_thr for i in HEAD)
    shoulders = all(ks[i] > kp_thr for i in SHOULDERS)
    ankle = any(ks[i] > kp_thr for i in ANKLES)
    if not head:
        return False, "머리 잘림"
    if not shoulders:
        return False, "어깨 안 보임"
    if not ankle:
        return False, "발목 잘림(전신 아님)"
    return True, "통과"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_root", required=True, help="DeepFashion2 train 폴더 (image/ 포함)")
    ap.add_argument("--candidates", required=True, help="label_viewpoint.py가 만든 *_candidates.json")
    ap.add_argument("--out", default="fullbody_candidates.json")
    ap.add_argument("--person_thr", type=float, default=0.9, help="사람 검출 확신도 기준")
    ap.add_argument("--kp_thr", type=float, default=2.0, help="어깨·발목 관절 확신도 기준 (높을수록 엄격)")
    ap.add_argument("--head_thr", type=float, default=0.0, help="머리 관절 기준 (뒷모습을 위해 느슨하게)")
    ap.add_argument("--min_size", type=int, default=512, help="입력 크기 (작을수록 빠름)")
    ap.add_argument("--limit", type=int, default=0, help="앞에서부터 N장만 검사 (0=전부, 시험용)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[모델] Keypoint R-CNN 로드 중... (device={device}, 최초 1회 가중치 다운로드)")
    model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT,
                                      min_size=args.min_size).to(device).eval()

    with open(args.candidates, encoding="utf-8") as f:
        cands = json.load(f)
    if args.limit:
        cands = cands[:args.limit]

    # 이어하기: 이미 판정한 결과가 있으면 재사용
    cache_path = Path(args.out).with_suffix(".judged.json")
    judged = json.load(open(cache_path, encoding="utf-8")) if cache_path.exists() else {}

    img_dir = Path(args.raw_root) / "image"
    if not cands:
        raise RuntimeError(f"후보 목록이 비어 있습니다: {args.candidates}\n"
                           f"label_viewpoint.py를 올바른 --raw_root로 다시 실행해 후보를 만드세요.")
    if not img_dir.exists():
        raise FileNotFoundError(f"이미지 폴더가 없습니다: {img_dir}  (--raw_root 경로 확인)")
    t0 = time.time()
    todo = [c for c in cands if c["stem"] not in judged]
    print(f"[검사] 후보 {len(cands)}장 중 새로 검사할 {len(todo)}장")
    with torch.no_grad():
        for i, c in enumerate(todo, 1):
            img = Image.open(img_dir / f"{c['stem']}.jpg").convert("RGB")
            out = model([to_tensor(img).to(device)])[0]
            out = {k: v.cpu() for k, v in out.items()}
            ok, why = judge(out, args.person_thr, args.kp_thr, args.head_thr)
            judged[c["stem"]] = why
            if i % 50 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  {i}/{len(todo)}  ({el/i:.2f}초/장, 남은 시간 약 {el/i*(len(todo)-i)/60:.0f}분)")
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(judged, f, ensure_ascii=False)

    passed = [c for c in cands if judged.get(c["stem"]) == "통과"]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(passed, f, ensure_ascii=False)

    print(f"\n[결과] 통과 {len(passed)}/{len(cands)}장 -> {args.out}")
    print("  탈락 사유:", dict(Counter(v for s, v in judged.items() if v != "통과" and s in {c['stem'] for c in cands})))
    print("  DeepFashion2 방향별 통과율 (특정 방향만 많이 빠지면 기준 조정 필요):")
    for vp, name in [(2, "정면"), (3, "측면/후면")]:
        grp = [c for c in cands if c["df2_viewpoint"] == vp]
        if grp:
            ok = sum(judged.get(c["stem"]) == "통과" for c in grp)
            print(f"    {name}: {ok}/{len(grp)} ({100*ok/len(grp):.0f}%)")


if __name__ == "__main__":
    main()
