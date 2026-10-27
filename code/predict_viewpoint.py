"""
predict_viewpoint.py — 실제 쇼핑몰 사진으로 정확도 측정 / 자동 분류
=========================================================================
[모드 1] 정확도 측정 — 정답 폴더로 나눠둔 쇼핑몰 사진
  shop_test/
    front/     정면 사진
    side/      측면 사진
    back/      후면 사진
    nowear/    옷만 찍은 사진 (없으면 생략 가능)
  * 모델은 측면·후면을 구분하지 않으므로(DeepFashion2 라벨 한계) side, back은 둘 다 'sideback'을
    정답으로 채점합니다. 대신 side/back 각각이 어떻게 예측됐는지도 따로 보여줍니다.

  python3 predict_viewpoint.py --runs runs_vp/resnet50_s42 runs_vp/resnet50_s43 runs_vp/resnet50_s44 \\
      --images shop_test --out shop_test_result.csv

[모드 2] 자동 분류 — 정답 없는 사진 폴더를 예측 결과별 폴더로 복사
  python3 predict_viewpoint.py --runs runs_vp/resnet50_s42 runs_vp/resnet50_s43 runs_vp/resnet50_s44 \\
      --images shop_unlabeled --out shop_pred.csv --sort_to shop_sorted

여러 run을 주면 각 모델 정확도를 따로 보고하고, 최종 예측은 확률 평균(앙상블)으로 냅니다.
"""

import argparse
import csv
import json
import shutil
import statistics as st
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import transforms

from models import build_model

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
GT_NAMES = {"front", "back", "left", "right", "side", "sideback", "nowear"}
KO = {"front": "정면", "back": "후면", "left": "좌측면", "right": "우측면", "side": "측면",
      "sideback": "측면·후면", "nowear": "옷만"}


def map_gt(gt, classes):
    """정답 폴더명을 모델 클래스로 변환. 채점 불가(예: 4분류 모델에 'side' 폴더)면 None."""
    if gt is None:
        return None
    if gt in classes:
        return gt
    if "side" in classes and gt in ("left", "right"):
        return "side"
    if "sideback" in classes and gt in ("side", "back", "left", "right"):
        return "sideback"
    return None
TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def load_run(run, device):
    run = Path(run)
    with open(run / "result.json", encoding="utf-8") as f:
        r = json.load(f)
    model = build_model(r["model"], num_classes=len(r["classes"]), pretrained=False)
    model.load_state_dict(torch.load(run / "best_model.pt", map_location=device))
    return model.to(device).eval(), r["classes"]


def list_images(root):
    """(경로, 정답폴더명 or None) 목록. 하위 폴더 이름이 정답 규칙에 맞으면 정답으로 사용."""
    root = Path(root)
    items = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMG_EXTS:
            gt = p.parent.name.lower() if p.parent != root else None
            items.append((p, gt if gt in GT_NAMES else None))
    return items


@torch.no_grad()
def probs_for(model, paths, device, bs=32):
    out = []
    for i in range(0, len(paths), bs):
        batch = []
        for p in paths[i:i + bs]:
            img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")  # 폰 사진 회전 정보 반영
            batch.append(TF(img))
        logits = model(torch.stack(batch).to(device))
        if isinstance(logits, tuple):
            logits = logits[0]
        out.append(torch.softmax(logits, 1).cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--sort_to", default=None, help="예측 결과별 폴더로 복사할 위치 (자동 분류)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items = list_images(args.images)
    if not items:
        raise RuntimeError(f"이미지가 없습니다: {args.images}")
    paths = [p for p, _ in items]
    raw_gts = [g for _, g in items]
    print(f"[입력] 이미지 {len(items)}장 (정답 폴더: {dict(Counter(g for g in raw_gts if g))})")

    all_probs, classes = [], None
    per_run_acc = []
    for run in args.runs:
        model, cls = load_run(run, device)
        if classes is None:
            classes = cls
            gts = [map_gt(g, classes) for g in raw_gts]
            labeled = [i for i, g in enumerate(gts) if g]
            n_unscorable = sum(1 for g, m in zip(raw_gts, gts) if g and not m)
            print(f"[채점] 모델 클래스 {classes}, 채점 가능 {len(labeled)}장"
                  + (f", 채점 불가 {n_unscorable}장(모델에 없는 정답 폴더)" if n_unscorable else ""))
        assert cls == classes, "run들의 클래스 목록이 다릅니다."
        pr = probs_for(model, paths, device)
        all_probs.append(pr)
        if labeled:
            pred = pr.argmax(1).tolist()
            a = sum(classes[pred[i]] == gts[i] for i in labeled) / len(labeled)
            per_run_acc.append(a)
            print(f"  {Path(run).name}: 정확도 {a:.4f}")

    probs = torch.stack(all_probs).mean(0)
    pred = [classes[k] for k in probs.argmax(1).tolist()]
    conf = probs.max(1).values.tolist()

    if labeled:
        ens = sum(pred[i] == gts[i] for i in labeled) / len(labeled)
        sd = f"±{st.stdev(per_run_acc):.4f}" if len(per_run_acc) > 1 else ""
        print(f"\n[정확도] 모델별 평균 {st.mean(per_run_acc):.4f}{sd}  |  앙상블 {ens:.4f}")
        print("\n[정답별 예측 분포]  (행=실제 폴더, 열=모델 예측)")
        print("  " + " " * 10 + "".join(f"{KO[c]:>10s}" for c in classes) + "   정답률")
        for g in ["front", "back", "left", "right", "side", "nowear"]:
            idx = [i for i in labeled if raw_gts[i] == g]
            if not idx:
                continue
            c = Counter(pred[i] for i in idx)
            ok = sum(pred[i] == gts[i] for i in idx) / len(idx)
            print(f"  {KO[g]:>8s}  " + "".join(f"{c.get(k, 0):>10d}" for k in classes) + f"   {ok:.1%} (n={len(idx)})")
        wrong = sorted((i for i in labeled if pred[i] != gts[i]), key=lambda i: -conf[i])[:10]
        if wrong:
            print("\n[확신하며 틀린 사진 상위 10] — 직접 열어보면 원인 파악에 도움")
            for i in wrong:
                print(f"  {paths[i]}  실제={KO[raw_gts[i]]}  예측={KO[pred[i]]}  확신도={conf[i]:.2f}")

    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "true_folder", "pred", "confidence", *[f"p_{c}" for c in classes]])
        for i, p in enumerate(paths):
            w.writerow([str(p), raw_gts[i] or "", pred[i], f"{conf[i]:.4f}", *[f"{v:.4f}" for v in probs[i].tolist()]])
    print(f"\n예측 저장: {args.out}")

    if args.sort_to:
        for i, p in enumerate(paths):
            dst = Path(args.sort_to) / pred[i] / p.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(p, dst)
        print(f"자동 분류 완료: {args.sort_to}/({'|'.join(classes)})/  {dict(Counter(pred))}")


if __name__ == "__main__":
    main()
