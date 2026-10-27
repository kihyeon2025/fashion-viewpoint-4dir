"""
ablation_eval.py — 어블레이션 결과를 같은 기준으로 한 번에 비교 (추가 학습 없음)
=========================================================================
평가 데이터 (둘 다 직접 구축한 정답으로 채점)
  - DeepFashion2 평가셋: <data_root>/test/(front|back|left|right)
  - 쇼핑몰 평가셋:       <shop_root>/(front|back|left|right)

지표
  - 정면/비정면 정확도 : 모든 모델 공통 (어블레이션 A의 공정 비교용)
  - 3분류 정확도       : 3분류·4분류 모델 (4분류는 좌·우를 측면으로 합쳐 채점)
  - 4분류 정확도       : 4분류 모델만
  - 좌↔우 혼동률       : 4분류 모델만

run 폴더 이름의 _s42 같은 seed 꼬리를 떼서 그룹으로 묶고 평균±표준편차를 냅니다.

사용 예시
  python3 ablation_eval.py --data_root vp4_dataset --shop_root shop_test --runs runs_vp/*_s4[234]
"""

import argparse
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import transforms

from models import build_model

LABELS = ["front", "back", "left", "right"]
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def load_set(root):
    items = []
    for lab in LABELS:
        for p in sorted((Path(root) / lab).glob("*")):
            if p.suffix.lower() in EXTS:
                items.append((p, lab))
    return items


@torch.no_grad()
def predict(model, items, classes, device, bs=64):
    preds = []
    for i in range(0, len(items), bs):
        x = torch.stack([TF(ImageOps.exif_transpose(Image.open(p)).convert("RGB")) for p, _ in items[i:i + bs]])
        out = model(x.to(device))
        if isinstance(out, tuple):
            out = out[0]
        preds += [classes[k] for k in out.argmax(1).tolist()]
    return preds


def metrics(preds, gts, classes):
    n = len(gts)
    m = {"binary": sum((p == "front") == (g == "front") for p, g in zip(preds, gts)) / n}
    to3 = lambda x: "side" if x in ("left", "right") else x
    if "side" in classes or "left" in classes:
        m["3class"] = sum(to3(p) == to3(g) for p, g in zip(preds, gts)) / n
    if "left" in classes:
        m["4class"] = sum(p == g for p, g in zip(preds, gts)) / n
        side = [(p, g) for p, g in zip(preds, gts) if g in ("left", "right")]
        m["lr_conf"] = sum((g == "left" and p == "right") or (g == "right" and p == "left") for p, g in side) / max(len(side), 1)
    return m


def fmt(v):
    if not v:
        return "-"
    v = [x * 100 for x in v]
    return f"{st.mean(v):.2f}" + (f"±{st.stdev(v):.2f}" if len(v) > 1 else "")


def describe(args_):
    mode = args_.get("mode")
    parts = [mode]
    if args_.get("label_source") == "df2":
        parts.append("DF2라벨")
    if args_.get("flip", "swap") != "swap":
        parts.append(f"flip={args_['flip']}")
    if args_.get("no_pretrained"):
        parts.append("사전학습X")
    if args_.get("class_weight"):
        parts.append("가중치")
    if args_.get("noise_rate"):
        parts.append(f"노이즈 {args_.get('noise_type')} {args_['noise_rate']:.0%}")
    if args_.get("train_frac", 1.0) < 1.0:
        parts.append(f"학습 {args_['train_frac']:.0%}")
    return " / ".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--shop_root", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="ablation_summary.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sets = {"DF2": load_set(Path(args.data_root) / "test"), "쇼핑몰": load_set(args.shop_root)}
    print(f"[평가셋] DF2 test {len(sets['DF2'])}장, 쇼핑몰 {len(sets['쇼핑몰'])}장")

    groups = defaultdict(lambda: {"desc": "", "res": defaultdict(list)})
    for run in sorted(set(args.runs)):
        rp = Path(run)
        if not (rp / "result.json").exists():
            continue
        r = json.load(open(rp / "result.json", encoding="utf-8"))
        classes = r["classes"]
        model = build_model(r.get("model", "resnet50"), num_classes=len(classes), pretrained=False)
        model.load_state_dict(torch.load(rp / "best_model.pt", map_location=device))
        model.to(device).eval()
        g = re.sub(r"_s\d+$", "", rp.name)
        groups[g]["desc"] = describe(r.get("args", {}))
        for sname, items in sets.items():
            m = metrics(predict(model, items, classes, device), [lab for _, lab in items], classes)
            for k, v in m.items():
                groups[g]["res"][f"{sname}_{k}"].append(v)
        print(f"  완료: {rp.name}")

    cols = [("DF2_binary", "DF2 정면/비정면"), ("DF2_3class", "DF2 3분류"), ("DF2_4class", "DF2 4분류"),
            ("DF2_lr_conf", "DF2 좌우혼동"), ("쇼핑몰_binary", "쇼핑몰 정면/비정면"),
            ("쇼핑몰_3class", "쇼핑몰 3분류"), ("쇼핑몰_4class", "쇼핑몰 4분류"), ("쇼핑몰_lr_conf", "쇼핑몰 좌우혼동")]
    print("\n[어블레이션 결과] 단위 %, seed 평균±표준편차 (좌우혼동은 낮을수록 좋음)")
    for g, d in groups.items():
        n = len(next(iter(d["res"].values())))
        print(f"\n■ {g}  ({d['desc']}, n={n})")
        for key, name in cols:
            if d["res"].get(key):
                print(f"    {name:16s} {fmt(d['res'][key])}")
    json.dump({g: {"desc": d["desc"], **{k: v for k, v in d["res"].items()}} for g, d in groups.items()},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
