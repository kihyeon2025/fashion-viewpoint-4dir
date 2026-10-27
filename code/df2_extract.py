r"""
df2_extract.py — 라벨링에 쓰지 않은 DeepFashion2 이미지를 뽑아 자동 분류용 폴더로 준비
=========================================================================
권장: DeepFashion2 'validation' 폴더 사용
  - 우리 6,000장은 DeepFashion2 'train'에서 뽑았고, DF2의 train/validation은 서로 다른 의류(pair)로
    나뉘어 있으므로 validation을 쓰면 같은 의류가 섞일 걱정(데이터 누수)이 없습니다.
  - train 폴더를 쓸 경우 --exclude 로 라벨링한 이미지와 같은 pair_id를 전부 제외합니다.

DF2 원본 viewpoint 라벨(2=정면, 3=측면·후면)도 함께 기록 -> df2_compare.py 로
"모델 예측 vs DF2 원본 라벨" 비교표(그림 2의 대규모 자동 버전)를 만듭니다.

사용 예시 (서버)
  # 1) validation에서 정면/측면·후면 각 1,500장 무작위 추출 (이미지는 복사하지 않고 링크만)
  python3 df2_extract.py --df2_root ~/data/deepfashion2/validation --per_viewpoint 1500 --out_dir df2_val3000

  # (train 폴더를 쓸 때) 라벨링한 6,000장과 같은 pair 제외
  python3 df2_extract.py --df2_root ~/data/deepfashion2/train --exclude labels.csv --per_viewpoint 1500 --out_dir df2_tr3000

  # 2) 자동 분류 (기존 스크립트 그대로)
  python3 shop_auto_sort.py --runs runs_vp/cnxt_4c_s42 runs_vp/cnxt_4c_s43 runs_vp/cnxt_4c_s44 \
      --images df2_val3000/images --out_dir df2_val3000_result --no_copy

  # 3) DF2 원본 라벨과 비교
  python3 df2_compare.py --meta df2_val3000/meta.csv --pred df2_val3000_result/predictions.csv
"""

import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path


def image_viewpoint(data):
    """label_viewpoint.py 와 같은 규칙: 착용 아이템 viewpoint의 최빈값 (전부 1이면 1)"""
    vps = [v.get("viewpoint") for k, v in data.items() if k.startswith("item") and isinstance(v, dict)]
    vps = [v for v in vps if v is not None]
    if not vps:
        return None
    worn = [v for v in vps if v != 1]
    return Counter(worn).most_common(1)[0][0] if worn else 1


def read_exclude(paths):
    stems = set()
    for p in paths or []:
        with open(p, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                s = r.get("stem") or Path(r.get("file", "")).stem
                if s:
                    stems.add(s)
    return stems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--df2_root", required=True, help="annos/ 와 image/ 가 들어있는 DF2 폴더 (validation 권장)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--per_viewpoint", type=int, default=1500, help="정면/측면·후면 각각 뽑을 장수 (0 = 전부)")
    ap.add_argument("--sources", nargs="+", default=["shop", "user"], help="DF2 source: shop(판매자) / user(소비자)")
    ap.add_argument("--exclude", nargs="*", help="제외할 이미지 목록 csv (labels.csv, split_manifest.csv 등; stem 열)")
    ap.add_argument("--copy", action="store_true", help="링크 대신 실제 복사 (다른 PC로 옮길 때)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.df2_root).expanduser()
    annos = sorted((root / "annos").glob("*.json"))
    if not annos:
        raise FileNotFoundError(f"{root/'annos'} 에 json이 없습니다. --df2_root 확인")

    ex_stems = read_exclude(args.exclude)
    print(f"[스캔] 어노테이션 {len(annos)}개, 제외 목록 {len(ex_stems)}장")

    recs, ex_pairs = [], set()
    for i, p in enumerate(annos):
        if i and i % 20000 == 0:
            print(f"  ...{i}/{len(annos)}")
        d = json.load(open(p, encoding="utf-8"))
        r = {"stem": p.stem, "source": d.get("source"), "pair_id": d.get("pair_id"),
             "df2_viewpoint": image_viewpoint(d)}
        if p.stem in ex_stems:
            ex_pairs.add(r["pair_id"])
            continue
        recs.append(r)

    n0 = len(recs)
    recs = [r for r in recs if r["pair_id"] not in ex_pairs]
    print(f"[누수 방지] 제외 이미지와 같은 pair_id {len(ex_pairs)}개 -> {n0 - len(recs)}장 추가 제외")
    recs = [r for r in recs if r["source"] in args.sources and r["df2_viewpoint"] in (2, 3)]
    pools = defaultdict(list)
    for r in recs:
        pools[r["df2_viewpoint"]].append(r)

    rng = random.Random(args.seed)
    chosen = []
    for vp, name in [(2, "정면"), (3, "측면·후면")]:
        pool = pools[vp]
        rng.shuffle(pool)
        take = pool if args.per_viewpoint == 0 else pool[:args.per_viewpoint]
        chosen += take
        print(f"  DF2 '{name}': 후보 {len(pool)}장 중 {len(take)}장")

    out = Path(args.out_dir)
    img_out = out / "images"
    img_out.mkdir(parents=True, exist_ok=True)
    miss = 0
    for r in chosen:
        src = (root / "image" / f"{r['stem']}.jpg").resolve()
        dst = img_out / f"{r['stem']}.jpg"
        if not src.exists():
            miss += 1
            continue
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if args.copy:
            import shutil
            shutil.copy(src, dst)
        else:
            os.symlink(src, dst)
    with open(out / "meta.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "source", "pair_id", "df2_viewpoint"])
        w.writeheader()
        w.writerows(chosen)
    print(f"\n완료: {len(chosen) - miss}장 -> {img_out}  (이미지 없음 {miss}장)")
    print(f"  source 분포: {dict(Counter(r['source'] for r in chosen))}")
    print(f"  메타 정보: {out/'meta.csv'}")


if __name__ == "__main__":
    main()
