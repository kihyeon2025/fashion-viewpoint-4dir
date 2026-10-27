r"""
df2_compare.py — 모델 자동 분류 결과를 DeepFashion2 원본 viewpoint 라벨과 비교
=========================================================================
df2_extract.py 의 meta.csv + shop_auto_sort.py 의 predictions.csv 를 합쳐서
  "DF2가 '정면'이라고 한 사진을 모델은 무엇으로 분류했나" 를 비율로 보여줍니다.
  = 논문 그림 2(사람이 6,000장을 직접 본 결과)를 모델이 대규모로 재현한 버전.

사용 예시
  python3 df2_compare.py --meta df2_val3000/meta.csv --pred df2_val3000_result/predictions.csv
결과: 화면 출력 + <pred 폴더>/df2_compare.json
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

DIRS = ["front", "left", "right", "back"]
KO = {"front": "정면", "back": "후면", "left": "좌측면", "right": "우측면"}
DF2 = {"2": "DF2 '정면'", "3": "DF2 '측면·후면'"}
# 직접 라벨링 6,000장 결과 (그림 2) — 비교 기준
MANUAL = {"2": {"front": 73.3, "left": 15.1, "right": 11.2, "back": 0.4},
          "3": {"front": 8.5, "left": 31.4, "right": 24.9, "back": 35.2}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--min_conf", type=float, default=0.0, help="이 신뢰도 이상인 예측만 집계 (예: 0.9)")
    args = ap.parse_args()

    meta = {r["stem"]: r for r in csv.DictReader(open(args.meta, encoding="utf-8-sig"))}
    rows = []
    for r in csv.DictReader(open(args.pred, encoding="utf-8-sig")):
        m = meta.get(Path(r["file"]).stem)
        if m is None:
            continue
        rows.append({**m, **r})
    print(f"[입력] 예측 {len(rows)}장 (meta와 매칭)")

    res = {}
    for subset, cond in [("전신 통과", lambda r: r["fullbody"] == "통과"),
                         ("전체(필터 무시)", lambda r: r["pred"] != "")]:
        sel = [r for r in rows if cond(r) and r["pred"] and float(r["confidence"]) >= args.min_conf]
        print(f"\n== {subset}: {len(sel)}장"
              + (f" (신뢰도 >= {args.min_conf})" if args.min_conf else "") + " ==")
        print(f"  {'':16s}" + "".join(f"{KO[d]:>8s}" for d in DIRS) + "      n")
        res[subset] = {}
        for vp in ("2", "3"):
            g = [r for r in sel if str(r["df2_viewpoint"]) == vp]
            if not g:
                continue
            c = Counter(r["pred"] for r in g)
            pct = {d: 100 * c.get(d, 0) / len(g) for d in DIRS}
            res[subset][vp] = {"n": len(g), "pct": pct,
                               "by_source": {s: len([r for r in g if r["source"] == s]) for s in ("shop", "user")}}
            print(f"  {DF2[vp]:14s}" + "".join(f"{pct[d]:7.1f}%" for d in DIRS) + f"  {len(g):5d}   <- 모델")
            print(f"  {'':14s}" + "".join(f"{MANUAL[vp][d]:7.1f}%" for d in DIRS) + "          <- 직접 라벨(그림 2)")
        if "2" in res[subset]:
            side = res[subset]["2"]["pct"]["left"] + res[subset]["2"]["pct"]["right"]
            print(f"  -> DF2 '정면' 중 모델이 측면으로 본 비율: {side:.1f}%  (직접 라벨 26.3%)")

    out = Path(args.pred).parent / "df2_compare.json"
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
