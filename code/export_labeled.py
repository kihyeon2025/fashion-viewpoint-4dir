r"""
export_labeled.py — 로컬 PC에서 실행
=========================================================================
labels.csv(직접 라벨)의 사진만 골라 train/val/test 폴더로 나눠 저장한다. 이 폴더를 압축해 서버로 전송.

- skip(제외) 사진은 사용하지 않음
- 같은 옷(pair_id)의 사진이 train과 test에 동시에 들어가지 않도록 pair 단위로 분할 (데이터 누수 방지)
- 전송 용량을 줄이기 위해 긴 변을 --max_side 픽셀로 축소

결과:
  <out_root>/(train|val|test)/(front|back|left|right)/<stem>.jpg
  <out_root>/split_manifest.csv   (어떤 사진이 어느 split인지 기록 — 논문 재현용)

사용 예시
  python export_labeled.py --raw_root "C:\...\deepfashion2_original_images\train" --labels labels.csv --out_root vp4_dataset
"""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

LABELS = ["front", "back", "left", "right"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_root", required=True, help="DeepFashion2 train 폴더 (annos/, image/ 포함)")
    ap.add_argument("--labels", default="labels.csv")
    ap.add_argument("--out_root", default="vp4_dataset")
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--max_side", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw_root, out_root = Path(args.raw_root), Path(args.out_root)
    with open(args.labels, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    n_skip = sum(r["label"] == "skip" for r in rows)
    rows = [r for r in rows if r["label"] in LABELS]
    print(f"[라벨] 사용 {len(rows)}장, 제외(skip) {n_skip}장")

    # pair_id 조회 (같은 옷끼리 묶기)
    for r in rows:
        with open(raw_root / "annos" / f"{r['stem']}.json", encoding="utf-8") as f:
            r["pair_id"] = json.load(f).get("pair_id", r["stem"])

    groups = defaultdict(list)
    for r in rows:
        groups[r["pair_id"]].append(r)
    keys = list(groups)
    random.Random(args.seed).shuffle(keys)

    # pair 단위로 test -> val -> train 순서로 채움
    n = len(rows)
    target = {"test": args.test_ratio * n, "val": args.val_ratio * n}
    count = {"test": 0, "val": 0, "train": 0}
    for k in keys:
        split = next((s for s in ("test", "val") if count[s] < target[s]), "train")
        for r in groups[k]:
            r["split"] = split
        count[split] += len(groups[k])

    stats = defaultdict(Counter)
    for i, r in enumerate(rows, 1):
        dst = out_root / r["split"] / r["label"] / f"{r['stem']}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        img = Image.open(raw_root / "image" / f"{r['stem']}.jpg").convert("RGB")
        img.thumbnail((args.max_side, args.max_side))
        img.save(dst, quality=92)
        stats[r["split"]][r["label"]] += 1
        if i % 500 == 0:
            print(f"  저장 중 {i}/{len(rows)}")

    with open(out_root / "split_manifest.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "label", "split", "pair_id", "source", "df2_viewpoint"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})

    print("\n[분할 결과]")
    print(f"  {'split':6s}" + "".join(f"{l:>8s}" for l in LABELS) + "    합계")
    for s in ("train", "val", "test"):
        print(f"  {s:6s}" + "".join(f"{stats[s][l]:>8d}" for l in LABELS) + f"  {sum(stats[s].values()):>6d}")
    print(f"\n완료: {out_root}  -> 이 폴더를 압축해서 서버로 전송하세요.")


if __name__ == "__main__":
    main()
