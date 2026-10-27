r"""
sample_grid.py — 분류 결과 샘플 이미지 그림 만들기 (논문 정성 결과용, 추가 학습 없음)
=========================================================================
fusion_eval.py 와 같은 방식으로 최종 모델(CNN 앙상블 + K2 관절, w=0.6)의 예측을 다시 계산한 뒤
  ① 이미지별 예측 결과 저장           → <out>/pred_<set>.csv
  ② 방향별 정답 샘플 그림             → <out>/fig_<set>_correct.jpg   (4행: 정면/후면/좌측면/우측면)
  ③ 오분류 샘플 그림                  → <out>/fig_<set>_errors.jpg    (정답 → 예측, 확신도)
  ④ 결합 효과 샘플 그림               → <out>/fig_<set>_fusion_fix.jpg (CNN은 틀렸지만 결합 모델이 맞힌 사진)
를 DF2 평가셋(test)과 쇼핑몰 200장(shop) 각각에 대해 만듭니다.
matplotlib 없이 PIL만 사용합니다. 한글 글꼴이 없으면 영어 라벨로 그립니다.

사용 예시 (서버, 프로젝트 폴더, conda 환경 활성화 후)
  CUDA_VISIBLE_DEVICES=0 python3 sample_grid.py \
      --runs runs_vp/cnxt_4c_s42 runs_vp/cnxt_4c_s43 runs_vp/cnxt_4c_s44 \
      --kp_dir kp_baseline --w 0.6 --out sample_fig
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from keypoint_baseline import CLASSES, SWAP, feat
from predict_viewpoint import load_run, probs_for
from fusion_eval import kp_probs

ORDER = ["front", "back", "left", "right"]
KO = {"front": "정면", "back": "후면", "left": "좌측면", "right": "우측면"}
EN = {"front": "Front", "back": "Back", "left": "Left", "right": "Right"}
FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf",
]
EN_FONT = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]


def get_font(size):
    for f in FONT_CANDIDATES:
        if Path(f).exists():
            return ImageFont.truetype(f, size), KO
    for f in EN_FONT:
        if Path(f).exists():
            return ImageFont.truetype(f, size), EN
    return ImageFont.load_default(), EN


def thumb(path, w, h):
    """비율 유지 + 흰 여백으로 w×h 칸에 맞춤"""
    try:
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    except Exception:
        return Image.new("RGB", (w, h), (220, 220, 220))
    img.thumbnail((w, h))
    canvas = Image.new("RGB", (w, h), "white")
    canvas.paste(img, ((w - img.width) // 2, (h - img.height) // 2))
    return canvas


def grid(rows, title, out, cw=180, ch=260, cap=46, font_size=17):
    """rows: [(행 이름, [(파일, 캡션, 정답여부), ...]), ...]"""
    font, _ = get_font(font_size)
    tfont, _ = get_font(font_size + 5)
    lw = 110                                   # 행 이름 칸
    ncol = max(len(r[1]) for r in rows)
    W = lw + ncol * (cw + 8) + 8
    H = 50 + len(rows) * (ch + cap + 10)
    G = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(G)
    d.text((10, 12), title, fill="black", font=tfont)
    y = 50
    for name, cells in rows:
        d.text((10, y + ch // 2 - 10), name, fill="black", font=tfont)
        x = lw
        for f, caption, ok in cells:
            G.paste(thumb(f, cw, ch), (x, y))
            col = (30, 140, 60) if ok else (200, 40, 40)
            d.rectangle([x, y, x + cw - 1, y + ch - 1], outline=col, width=4)
            for k, line in enumerate(caption.split("\n")):
                d.text((x + 4, y + ch + 4 + k * (font_size + 3)), line, fill=col, font=font)
            x += cw + 8
        y += ch + cap + 10
    G.save(out, quality=92)
    print(f"  저장: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--kp_dir", default="kp_baseline")
    ap.add_argument("--w", type=float, default=0.6, help="CNN 가중치 (fusion_eval에서 val로 고른 값)")
    ap.add_argument("--n", type=int, default=6, help="방향별 정답 샘플 수")
    ap.add_argument("--n_err", type=int, default=12, help="오분류 샘플 최대 수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="sample_fig")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kd = Path(args.kp_dir)
    _, L = get_font(16)
    print(f"[글꼴] {'한글' if L is KO else '영어'} 라벨 사용")

    D = {n: json.load(open(kd / f"kp_{n}.json", encoding="utf-8")) for n in ("train", "test", "shop")}

    # K2 재학습 (fusion_eval.py와 동일)
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

    for n, nm_ko, nm_en in [("shop", "쇼핑몰 평가셋", "Shop test"), ("test", "DF2 평가셋", "DF2 test")]:
        data = D[n]
        files = [r["file"] for r in data]
        yt = [r["label"] for r in data]
        PK = kp_probs(clf, data)
        PCs = []
        for run in args.runs:
            model, cls = load_run(run, device)
            order = [cls.index(c) for c in CLASSES]
            PCs.append(probs_for(model, [Path(f) for f in files], device).numpy()[:, order])
            del model
        PC = np.mean(PCs, 0)
        PF = args.w * PC + (1 - args.w) * PK
        pf, pc, pk = PF.argmax(1), PC.argmax(1), PK.argmax(1)
        pred = [CLASSES[k] for k in pf]
        conf = PF.max(1)
        acc = np.mean([a == b for a, b in zip(yt, pred)])
        print(f"\n[{nm_ko}] {len(files)}장, 결합 정확도 {acc*100:.2f}%")

        # ① 예측 CSV
        with open(out / f"pred_{n}.csv", "w", newline="", encoding="utf-8-sig") as fo:
            wr = csv.writer(fo)
            wr.writerow(["file", "label", "pred_fused", "conf_fused", "pred_cnn", "pred_k2", "person_found",
                         *[f"p_fused_{c}" for c in CLASSES]])
            for i, f in enumerate(files):
                wr.writerow([f, yt[i], pred[i], f"{conf[i]:.4f}", CLASSES[pc[i]], CLASSES[pk[i]],
                             int(bool(data[i]["person"])), *[f"{v:.4f}" for v in PF[i]]])
        print(f"  저장: {out / f'pred_{n}.csv'}")

        rng = random.Random(args.seed)
        title = nm_ko if L is KO else nm_en

        # ② 방향별 정답 샘플
        rows = []
        for c in ORDER:
            idx = [i for i in range(len(files)) if yt[i] == c and pred[i] == c]
            pick = rng.sample(idx, min(args.n, len(idx)))
            rows.append((L[c], [(files[i], f"{L[c]} {conf[i]*100:.0f}%", True) for i in pick]))
        grid(rows, f"{title}: " + ("방향별 정답 예시 (확신도)" if L is KO else "correctly classified samples (confidence)"),
             out / f"fig_{n}_correct.jpg")

        # ③ 오분류 샘플 (정답 방향별 행)
        err = [i for i in range(len(files)) if yt[i] != pred[i]]
        rng.shuffle(err)
        err = err[:args.n_err]
        rows = []
        for c in ORDER:
            cells = [(files[i], f"{L[yt[i]]}→{L[pred[i]]}\n{conf[i]*100:.0f}%", False) for i in err if yt[i] == c]
            if cells:
                rows.append(((L[c] + " 정답") if L is KO else ("GT " + L[c]), cells))
        if rows:
            grid(rows, f"{title}: " + ("오분류 예시 (정답→예측, 확신도)" if L is KO else "misclassified (GT→pred, confidence)"),
                 out / f"fig_{n}_errors.jpg")

        # ④ 결합 효과: CNN 앙상블은 틀렸지만 결합은 맞힘
        yi = np.array([CLASSES.index(c) for c in yt])
        fix = [i for i in range(len(files)) if pc[i] != yi[i] and pf[i] == yi[i]]
        hurt = [i for i in range(len(files)) if pc[i] == yi[i] and pf[i] != yi[i]]
        print(f"  결합으로 고쳐진 사진 {len(fix)}장 / 결합으로 틀려진 사진 {len(hurt)}장")
        rng.shuffle(fix)
        if fix:
            cells = [(files[i], (f"CNN {L[CLASSES[pc[i]]]} ✗\n결합 {L[pred[i]]} ✓" if L is KO
                                 else f"CNN {L[CLASSES[pc[i]]]} x\nFused {L[pred[i]]} ok"), True) for i in fix[:args.n_err]]
            half = (len(cells) + 1) // 2
            rows = [("", cells[:half])] + ([("", cells[half:])] if cells[half:] else [])
            grid(rows, f"{title}: " + ("자세 정보 결합으로 바로잡힌 예시" if L is KO else "corrected by adding pose"),
                 out / f"fig_{n}_fusion_fix.jpg", cap=50)

    print(f"\n완료. {out}/ 폴더의 jpg와 pred_*.csv 를 보내 주세요.")


if __name__ == "__main__":
    main()
