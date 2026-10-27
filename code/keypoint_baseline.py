r"""
keypoint_baseline.py — "사람 관절 인식만으로 방향을 알 수 있지 않나?"에 대한 비교 실험
=========================================================================
사전학습된 Keypoint R-CNN(COCO 17관절, 추가 학습 없음)으로 관절을 뽑은 뒤 두 가지 방법으로 4방향 분류:

  [K1] 규칙 기반 (사람이 생각할 법한 규칙, 임계값만 학습 셋에서 최적화)
       - 어깨 폭 / 몸통 길이 < t_w        -> 측면
           측면 중 좌/우: 왼쪽(귀+어깨) 관절 확신도 > 오른쪽 이면 좌측면 (부호는 학습 셋에서 결정)
       - 아니면 정면/후면: (a) 왼쪽 어깨가 화면 오른쪽에 있으면 정면  또는  (b) 얼굴(코·눈) 확신도 > t_f 이면 정면
  [K2] 관절 좌표 + 학습 분류기 (관절 17개의 정규화 좌표·확신도·상자 비율 52차원 -> HistGradientBoosting)
       - 좌우 반전 증강도 동일하게 적용 (x 뒤집고 좌/우 관절 교환 + 좌/우측면 라벨 교환)

평가: 우리 CNN과 같은 DF2 평가셋(896장) + 쇼핑몰 200장. 사람을 못 찾은 사진은 '정면'으로 처리(최빈 클래스)하고 그 수도 보고.

사용 예시 (서버)
  python3 keypoint_baseline.py --data_root vp4_dataset --shop_root shop_test --out kp_baseline
  -> 관절 추출 결과는 kp_baseline/kp_*.json 에 저장 (다시 실행하면 재사용)
  -> 결과: 화면 + kp_baseline/kp_result.json
"""

import argparse
import itertools
import json
import statistics as st
import time
from pathlib import Path

import numpy as np

CLASSES = ["back", "front", "left", "right"]
KO = {"front": "정면", "back": "후면", "left": "좌측면", "right": "우측면"}
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
# COCO: 0 코, 1 왼눈, 2 오른눈, 3 왼귀, 4 오른귀, 5 왼어깨, 6 오른어깨, 7 왼팔꿈치, 8 오른팔꿈치,
#       9 왼손목, 10 오른손목, 11 왼엉덩이, 12 오른엉덩이, 13 왼무릎, 14 오른무릎, 15 왼발목, 16 오른발목
FLIP_PERM = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


# ---------------------------------------------------------------- 1. 관절 추출
def list_split(root):
    items = []
    for lab in ["front", "back", "left", "right"]:
        for p in sorted((Path(root) / lab).glob("*")):
            if p.suffix.lower() in EXTS:
                items.append((p, lab))
    return items


def extract(items, cache, device, min_size):
    if cache.exists():
        data = json.load(open(cache, encoding="utf-8"))
        if len(data) == len(items):
            print(f"  캐시 사용: {cache} ({len(data)}장)")
            return data
    import torch
    from PIL import Image, ImageOps
    from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
    from torchvision.transforms.functional import to_tensor
    model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT,
                                      min_size=min_size).to(device).eval()
    data, t0 = [], time.time()
    with torch.no_grad():
        for i, (p, lab) in enumerate(items, 1):
            img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
            o = model([to_tensor(img).to(device)])[0]
            rec = {"file": str(p), "label": lab, "w": img.width, "h": img.height, "person": None}
            if len(o["scores"]) and float(o["scores"][0]) > 0.5:   # 가장 확실한 사람 1명
                rec["person"] = {"score": float(o["scores"][0]),
                                 "box": o["boxes"][0].cpu().tolist(),
                                 "kp": o["keypoints"][0, :, :2].cpu().tolist(),
                                 "ks": o["keypoints_scores"][0].cpu().tolist()}
            data.append(rec)
            if i % 500 == 0 or i == len(items):
                print(f"  {i}/{len(items)} ({(time.time()-t0)/i:.2f}초/장)")
    json.dump(data, open(cache, "w", encoding="utf-8"))
    return data


# ---------------------------------------------------------------- 2. 특징
def geo(rec):
    """규칙 기반용 기하 특징"""
    P = rec["person"]
    kp, ks = np.array(P["kp"]), np.array(P["ks"])
    x0, y0, x1, y1 = P["box"]
    sh_mid, hip_mid = kp[[5, 6]].mean(0), kp[[11, 12]].mean(0)
    torso = np.linalg.norm(sh_mid - hip_mid)
    if not np.isfinite(torso) or torso < 1 or min(ks[[11, 12]]) < 0:
        torso = 0.3 * (y1 - y0)  # 엉덩이가 안 잡히면 키의 30%로 근사
    return {"w": abs(kp[5, 0] - kp[6, 0]) / max(torso, 1e-6),          # 어깨 폭 / 몸통 길이
            "order": (kp[5, 0] - kp[6, 0]) / max(torso, 1e-6),         # +: 왼어깨가 화면 오른쪽 (정면 자세)
            "face": float(max(ks[0], ks[1], ks[2])),                    # 얼굴 관절 확신도
            "lr": float(ks[3] + ks[5] - ks[4] - ks[6])}                # +: 왼쪽 몸이 더 잘 보임


def feat(rec, flip=False):
    """학습 분류기용 52차원: 박스 기준 정규화 좌표(34) + 확신도(17) + 박스 가로세로 비율(1)"""
    P = rec["person"]
    kp, ks = np.array(P["kp"], dtype=float), np.array(P["ks"], dtype=float)
    x0, y0, x1, y1 = P["box"]
    w, h = max(x1 - x0, 1), max(y1 - y0, 1)
    nx, ny = (kp[:, 0] - (x0 + x1) / 2) / h, (kp[:, 1] - (y0 + y1) / 2) / h
    if flip:
        nx, ny, ks = -nx[FLIP_PERM], ny[FLIP_PERM], ks[FLIP_PERM]
    return np.concatenate([nx, ny, np.clip(ks, -5, 20), [w / h]])


SWAP = {"left": "right", "right": "left"}


# ---------------------------------------------------------------- 3. 지표
def metrics(y, p):
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    cm = confusion_matrix(y, p, labels=CLASSES)
    L, R = CLASSES.index("left"), CLASSES.index("right")
    lr = (cm[L][R] + cm[R][L]) / max(cm[L].sum() + cm[R].sum(), 1)
    return {"acc": float(accuracy_score(y, p)), "macro_f1": float(f1_score(y, p, labels=CLASSES, average="macro")),
            "lr_conf": float(lr), "per_class": {c: float(cm[i][i] / max(cm[i].sum(), 1)) for i, c in enumerate(CLASSES)},
            "cm": cm.tolist()}


def rule_predict(g, prm):
    t_w, t_f, fb, sgn = prm
    if g is None:
        return "front"
    if g["w"] < t_w:
        return "left" if sgn * g["lr"] > 0 else "right"
    if fb == "order":
        return "front" if g["order"] > 0 else "back"
    if fb == "face":
        return "front" if g["face"] > t_f else "back"
    return "front" if (g["order"] > 0 and g["face"] > t_f) else "back"   # both


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default="vp4_dataset")
    ap.add_argument("--shop_root", default="shop_test")
    ap.add_argument("--out", default="kp_baseline")
    ap.add_argument("--min_size", type=int, default=512)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = ap.parse_args()

    from sklearn.metrics import f1_score
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        device = "cpu"

    print("[1/3] 관절 추출 (Keypoint R-CNN, 추가 학습 없음)")
    D = {}
    for name, root in [("train", Path(args.data_root) / "train"), ("val", Path(args.data_root) / "val"),
                       ("test", Path(args.data_root) / "test"), ("shop", Path(args.shop_root))]:
        items = list_split(root)
        print(f" {name}: {len(items)}장")
        D[name] = extract(items, out / f"kp_{name}.json", device, args.min_size)
    for name, d in D.items():
        miss = sum(r["person"] is None for r in d)
        print(f"  {name}: 사람 미검출 {miss}장 ({miss/len(d):.1%})")

    G = {n: [geo(r) if r["person"] else None for r in d] for n, d in D.items()}
    Y = {n: [r["label"] for r in d] for n, d in D.items()}
    result = {"missing": {n: sum(r["person"] is None for r in d) for n, d in D.items()}}

    # ---------------- K1 규칙 기반: 임계값을 train+val에서 탐색
    print("\n[2/3] K1 규칙 기반 — 임계값을 학습+검증 셋에서 탐색 (평가 셋은 보지 않음)")
    tune_g, tune_y = G["train"] + G["val"], Y["train"] + Y["val"]
    ws = [x for x in (r["w"] for r in tune_g if r) if np.isfinite(x)]
    grid_w = np.quantile(ws, np.linspace(0.05, 0.6, 23)) if ws else [0.5]
    grid_f = [-2, 0, 1, 2, 3, 5, 8]
    best = (-1, None)
    for prm in itertools.product(grid_w, grid_f, ["order", "face", "both"], [1, -1]):
        p = [rule_predict(g, prm) for g in tune_g]
        f = f1_score(tune_y, p, labels=CLASSES, average="macro")
        if f > best[0]:
            best = (f, prm)
    prm = best[1]
    print(f"  최적 규칙: 어깨폭/몸통 < {prm[0]:.3f} -> 측면, 정면/후면 기준={prm[2]}, 얼굴 임계={prm[1]}, "
          f"좌우 부호={prm[3]}  (학습셋 macro F1 {best[0]:.4f})")
    result["K1_rule"] = {"params": {"t_w": float(prm[0]), "t_f": prm[1], "front_back_cue": prm[2], "lr_sign": prm[3]},
                         **{n: metrics(Y[n], [rule_predict(g, prm) for g in G[n]]) for n in ("test", "shop")}}

    # ---------------- K2 관절 좌표 + 학습 분류기
    print("\n[3/3] K2 관절 좌표 + 학습 분류기 (HistGradientBoosting, 좌우 반전 증강)")
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xtr, ytr = [], []
    for r in D["train"]:
        if r["person"]:
            Xtr.append(feat(r)); ytr.append(r["label"])
            Xtr.append(feat(r, flip=True)); ytr.append(SWAP.get(r["label"], r["label"]))
    Xtr, ytr = np.array(Xtr), np.array(ytr)

    def pred(clf, d):
        idx = [i for i, r in enumerate(d) if r["person"]]
        p = ["front"] * len(d)
        if idx:
            for i, v in zip(idx, clf.predict(np.array([feat(d[i]) for i in idx]))):
                p[i] = v
        return p

    # 하이퍼파라미터는 val로 선택
    best = (-1, None)
    for lr_, depth in itertools.product([0.05, 0.1], [None, 6]):
        clf = HistGradientBoostingClassifier(learning_rate=lr_, max_depth=depth, max_iter=400,
                                             random_state=42).fit(Xtr, ytr)
        f = f1_score(Y["val"], pred(clf, D["val"]), labels=CLASSES, average="macro")
        if f > best[0]:
            best = (f, (lr_, depth))
    lr_, depth = best[1]
    print(f"  선택: learning_rate={lr_}, max_depth={depth} (val macro F1 {best[0]:.4f})")
    per_seed = {"test": [], "shop": []}
    for s in args.seeds:
        clf = HistGradientBoostingClassifier(learning_rate=lr_, max_depth=depth, max_iter=400,
                                             random_state=s).fit(Xtr, ytr)
        for n in per_seed:
            per_seed[n].append(metrics(Y[n], pred(clf, D[n])))
    result["K2_learned"] = {"params": {"learning_rate": lr_, "max_depth": depth}, "per_seed": per_seed}

    # ---------------- 출력
    def ms(v):
        v = [x * 100 for x in v]
        return f"{st.mean(v):.2f}" + (f"±{st.stdev(v):.2f}" if len(v) > 1 else "")

    print("\n================ 결과 (단위 %) ================")
    for n, nm in [("test", "DF2 평가셋"), ("shop", "쇼핑몰 200장")]:
        k1 = result["K1_rule"][n]
        k2 = per_seed[n]
        print(f"\n[{nm}]  (사람 미검출 {result['missing'][n]}장은 '정면'으로 처리)")
        print(f"  K1 규칙 기반   : 4분류 {k1['acc']*100:.2f}  macroF1 {k1['macro_f1']*100:.2f}  좌우혼동 {k1['lr_conf']*100:.2f}  "
              + " ".join(f"{KO[c]} {k1['per_class'][c]*100:.0f}" for c in ["front", "back", "left", "right"]))
        print(f"  K2 관절+학습   : 4분류 {ms([m['acc'] for m in k2])}  macroF1 {ms([m['macro_f1'] for m in k2])}  "
              f"좌우혼동 {ms([m['lr_conf'] for m in k2])}  "
              + " ".join(f"{KO[c]} {st.mean(m['per_class'][c] for m in k2)*100:.0f}" for c in ["front", "back", "left", "right"]))
    print("\n  (비교) ConvNeXt-T 이미지 학습: DF2 4분류 83.15±0.78, 좌우혼동 0.73 / 쇼핑몰 4분류 88.67±1.15")
    json.dump(result, open(out / "kp_result.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n저장: {out/'kp_result.json'}")


if __name__ == "__main__":
    main()
