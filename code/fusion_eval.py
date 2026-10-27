r"""
fusion_eval.py — 이미지 모델(CNN) + 관절 모델(K2) 결합 실험 (추가 학습 없음, 수 분)
=========================================================================
  - CNN : 이미 학습한 4분류 run들 (예: ConvNeXt-T seed 42/43/44)
  - K2  : keypoint_baseline.py 가 만든 관절 캐시(kp_*.json)로 관절 분류기를 다시 학습 (수 초)
  - 결합: p = w * p_CNN + (1 - w) * p_K2   (w는 검증 셋에서 선택, 평가 셋은 보지 않음 / w=0.5 결과도 함께)

보고 항목 (DF2 평가셋 896장 + 쇼핑몰 200장)
  1) 정확도: CNN 단일(seed 평균) / CNN 앙상블 / K2 / 결합(단일 CNN+K2, seed 평균) / 결합(CNN 앙상블+K2)
  2) 상호보완성: 둘 다 맞음 / CNN만 맞음 / K2만 맞음 / 둘 다 틀림, 오라클(둘 중 하나라도 맞음) 정확도
  3) 결합 모델의 방향별 정확도, 좌↔우 혼동률

사용 예시 (서버, keypoint_baseline.py 실행 후)
  CUDA_VISIBLE_DEVICES=3 python3 fusion_eval.py \
      --runs runs_vp/cnxt_4c_s42 runs_vp/cnxt_4c_s43 runs_vp/cnxt_4c_s44 \
      --kp_dir kp_baseline --out fusion_result.json
"""

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np
import torch

from keypoint_baseline import CLASSES, KO, SWAP, feat, metrics
from predict_viewpoint import load_run, probs_for


def kp_probs(clf, data):
    """사람을 못 찾은 사진은 균등 확률(결합 시 CNN 판단을 그대로 따름)"""
    P = np.full((len(data), 4), 0.25)
    idx = [i for i, r in enumerate(data) if r["person"]]
    if idx:
        pr = clf.predict_proba(np.array([feat(data[i]) for i in idx]))
        order = [list(clf.classes_).index(c) for c in CLASSES]
        P[idx] = pr[:, order]
    return P


def acc_f1(y, P):
    from sklearn.metrics import f1_score
    p = [CLASSES[k] for k in P.argmax(1)]
    return float(np.mean([a == b for a, b in zip(y, p)])), float(f1_score(y, p, labels=CLASSES, average="macro"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="CNN 4분류 run 폴더들")
    ap.add_argument("--kp_dir", default="kp_baseline")
    ap.add_argument("--out", default="fusion_result.json")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kd = Path(args.kp_dir)

    D = {n: json.load(open(kd / f"kp_{n}.json", encoding="utf-8")) for n in ("train", "val", "test", "shop")}
    Y = {n: [r["label"] for r in d] for n, d in D.items()}

    # ---------------- K2 재학습 (keypoint_baseline.py와 같은 설정)
    from sklearn.ensemble import HistGradientBoostingClassifier
    prm = {"learning_rate": 0.1, "max_depth": None}
    if (kd / "kp_result.json").exists():
        prm = json.load(open(kd / "kp_result.json", encoding="utf-8"))["K2_learned"]["params"]
    X, y = [], []
    for r in D["train"]:
        if r["person"]:
            X.append(feat(r)); y.append(r["label"])
            X.append(feat(r, flip=True)); y.append(SWAP.get(r["label"], r["label"]))
    clf = HistGradientBoostingClassifier(max_iter=400, random_state=42, **prm).fit(np.array(X), np.array(y))
    PK = {n: kp_probs(clf, D[n]) for n in ("val", "test", "shop")}
    print(f"[K2] 관절 분류기 재학습 완료 {prm}")

    # ---------------- CNN 확률 (run별)
    PC = {n: [] for n in ("val", "test", "shop")}
    for run in args.runs:
        model, cls = load_run(run, device)
        order = [cls.index(c) for c in CLASSES]
        for n in PC:
            PC[n].append(probs_for(model, [Path(r["file"]) for r in D[n]], device).numpy()[:, order])
        print(f"[CNN] {Path(run).name} 예측 완료")
        del model
    PCe = {n: np.mean(PC[n], 0) for n in PC}

    # ---------------- 결합 가중치: val에서 선택
    ws = np.round(np.linspace(0, 1, 11), 2)
    val_f1 = [acc_f1(Y["val"], w * PCe["val"] + (1 - w) * PK["val"])[1] for w in ws]
    w_best = float(ws[int(np.argmax(val_f1))])
    print(f"[결합] 검증 셋에서 선택한 CNN 가중치 w = {w_best}  (val macro F1 {max(val_f1)*100:.2f})")

    def ms(v):
        v = [x * 100 for x in v]
        return f"{st.mean(v):.2f}" + (f"±{st.stdev(v):.2f}" if len(v) > 1 else "")

    res = {"w_best": w_best, "val_f1_by_w": dict(zip(map(str, ws), val_f1))}
    for n, nm in [("test", "DF2 평가셋"), ("shop", "쇼핑몰 200장")]:
        y = Y[n]
        rows = {
            "CNN 단일": [acc_f1(y, p) for p in PC[n]],
            "CNN 앙상블": [acc_f1(y, PCe[n])],
            "K2 관절": [acc_f1(y, PK[n])],
            f"결합 단일CNN+K2 (w={w_best})": [acc_f1(y, w_best * p + (1 - w_best) * PK[n]) for p in PC[n]],
            "결합 단일CNN+K2 (w=0.5)": [acc_f1(y, 0.5 * p + 0.5 * PK[n]) for p in PC[n]],
            f"결합 CNN앙상블+K2 (w={w_best})": [acc_f1(y, w_best * PCe[n] + (1 - w_best) * PK[n])],
        }
        print(f"\n================ {nm} (단위 %) ================")
        print(f"  {'방법':32s}{'4분류 정확도':>14s}{'macro F1':>14s}")
        res[n] = {}
        for k, v in rows.items():
            print(f"  {k:32s}{ms([a for a, _ in v]):>14s}{ms([f for _, f in v]):>14s}")
            res[n][k] = {"acc": [a for a, _ in v], "f1": [f for _, f in v]}

        # 상호보완성 (CNN 앙상블 vs K2)
        pc = PCe[n].argmax(1); pk = PK[n].argmax(1); yt = np.array([CLASSES.index(c) for c in y])
        cc, kc = pc == yt, pk == yt
        comp = {"둘 다 맞음": int((cc & kc).sum()), "CNN만 맞음": int((cc & ~kc).sum()),
                "K2만 맞음": int((~cc & kc).sum()), "둘 다 틀림": int((~cc & ~kc).sum())}
        print(f"\n  [상호보완성: CNN 앙상블 vs K2]  " + ", ".join(f"{k} {v}" for k, v in comp.items()))
        print(f"  오라클(둘 중 하나라도 맞음) 정확도 {(cc | kc).mean()*100:.2f}%  "
              f"| 둘 다 틀린 사진 중 같은 답으로 틀린 비율 {((pc == pk) & ~cc & ~kc).sum() / max((~cc & ~kc).sum(), 1)*100:.1f}%")
        fused = w_best * PCe[n] + (1 - w_best) * PK[n]
        m = metrics(y, [CLASSES[k] for k in fused.argmax(1)])
        print(f"  결합(CNN앙상블+K2) 방향별: " + " ".join(f"{KO[c]} {m['per_class'][c]*100:.0f}" for c in ["front", "back", "left", "right"])
              + f" | 좌우혼동 {m['lr_conf']*100:.2f}")
        res[n]["complementarity"] = comp
        res[n]["oracle_acc"] = float((cc | kc).mean())
        res[n]["fused_detail"] = m

    json.dump(res, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
