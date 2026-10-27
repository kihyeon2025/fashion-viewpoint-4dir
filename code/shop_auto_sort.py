r"""
shop_auto_sort.py — 정답 없는 쇼핑몰 사진(예: 3,000장)을 4방향으로 자동 분류하고 결과를 요약
=========================================================================
같은 폴더에 models.py, predict_viewpoint.py, filter_fullbody.py 가 있어야 합니다.

처리 순서
  1) 전신 1인 필터 (Keypoint R-CNN, filter_fullbody.py와 같은 기준)
       -> 학습 데이터 조건(모델 1명 + 머리~발목)에 맞는 사진만 '분류 대상'
  2) 모든 사진에 4분류 앙상블 예측 (run 3개 확률 평균)
       -> 전신 통과 사진만 방향별 폴더로 복사, 탈락 사진은 _excluded/<사유>/ 로 복사
  3) 요약 (정답이 없어도 낼 수 있는 지표)
       - 전신 필터 통과율과 탈락 사유
       - 방향별 예측 분포
       - 신뢰도(앙상블 최대 확률) 분포, 기준값별 자동 처리 비율
       - 모델 3개 예측 일치율 (3개가 모두 같은 답을 낸 비율)
       - 전신 통과 vs 탈락 사진의 신뢰도 비교
       - 방향별 미리보기 그리드 (신뢰도 낮은 순 / 높은 순) -> 눈으로 빠르게 확인

사용 예시 (서버)
  conda activate <환경이름>
  python3 shop_auto_sort.py \
      --runs runs_vp/r50_4c_s42 runs_vp/r50_4c_s43 runs_vp/r50_4c_s44 \
      --images shop3000 --out_dir shop3000_result

  시험 삼아 앞 100장만:  --limit 100
  전신 필터 없이 전부 분류: --no_filter

결과물 (--out_dir 아래)
  predictions.csv        사진별: 필터 결과, 앙상블 예측, 신뢰도, 클래스별 확률, 모델별 예측
  summary.txt / .json    위 3)의 모든 수치 (논문 표에 그대로 사용)
  sorted/<front|back|left|right>/   전신 통과 사진 자동 분류 결과
  sorted/_excluded/<사유>/          전신 필터 탈락 사진
  grid_<방향>_low.jpg / _high.jpg   신뢰도 낮은/높은 사진 미리보기
  confidence_hist.png    신뢰도 분포 그림
  fullbody_judged.json   전신 판정 캐시 (중단 후 다시 실행하면 이어서 진행)
"""

import argparse
import csv
import json
import shutil
import statistics as st
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageOps

from predict_viewpoint import IMG_EXTS, KO, load_run, probs_for

THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def list_flat(root):
    root = Path(root)
    return [p for p in sorted(root.rglob("*")) if p.suffix.lower() in IMG_EXTS]


def fullbody_filter(paths, root, cache_path, args, device):
    from filter_fullbody import judge
    from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
    from torchvision.transforms.functional import to_tensor

    judged = json.load(open(cache_path, encoding="utf-8")) if cache_path.exists() else {}
    key = lambda p: str(p.relative_to(root))
    todo = [p for p in paths if key(p) not in judged]
    print(f"[1/3 전신 필터] 전체 {len(paths)}장, 새로 검사 {len(todo)}장 (device={device})")
    if todo:
        model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT,
                                          min_size=args.min_size).to(device).eval()
        t0 = time.time()
        with torch.no_grad():
            for i, p in enumerate(todo, 1):
                try:
                    img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
                    out = model([to_tensor(img).to(device)])[0]
                    out = {k: v.cpu() for k, v in out.items()}
                    _, why = judge(out, args.person_thr, args.kp_thr, args.head_thr)
                except Exception as e:  # 깨진 파일 등
                    why = "열기 실패"
                    print(f"  [경고] {p}: {e}")
                judged[key(p)] = why
                if i % 100 == 0 or i == len(todo):
                    el = time.time() - t0
                    print(f"  {i}/{len(todo)}  ({el/i:.2f}초/장, 남은 시간 약 {el/i*(len(todo)-i)/60:.0f}분)")
                    json.dump(judged, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return [judged[key(p)] for p in paths]


def make_grid(paths, labels, out_path, cols=8, cell=160):
    if not paths:
        return
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 18)), "white")
    d = ImageDraw.Draw(sheet)
    for i, (p, lab) in enumerate(zip(paths, labels)):
        try:
            im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
            im.thumbnail((cell, cell))
        except Exception:
            continue
        x, y = (i % cols) * cell, (i // cols) * (cell + 18)
        sheet.paste(im, (x + (cell - im.width) // 2, y + (cell - im.height) // 2))
        d.text((x + 3, y + cell + 2), lab, fill="black")  # 기본 글꼴: 영문/숫자만
    sheet.save(out_path, quality=90)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="4분류 run 폴더들 (result.json, best_model.pt 포함)")
    ap.add_argument("--images", required=True, help="쇼핑몰 사진 폴더 (하위 폴더 포함 전부 읽음)")
    ap.add_argument("--out_dir", default="shop_auto_result")
    ap.add_argument("--no_filter", action="store_true", help="전신 필터 생략 (모든 사진을 분류 대상으로)")
    ap.add_argument("--no_copy", action="store_true", help="방향별 폴더 복사 생략 (csv/요약만)")
    ap.add_argument("--limit", type=int, default=0, help="앞에서부터 N장만 (시험용)")
    ap.add_argument("--grid_n", type=int, default=40, help="미리보기 그리드에 넣을 장수")
    ap.add_argument("--person_thr", type=float, default=0.9)
    ap.add_argument("--kp_thr", type=float, default=2.0)
    ap.add_argument("--head_thr", type=float, default=0.0)
    ap.add_argument("--min_size", type=int, default=512)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.images)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    paths = list_flat(root)
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise RuntimeError(f"이미지가 없습니다: {root}")

    # ---------------- 1) 전신 필터 ----------------
    if args.no_filter:
        reasons = []
        for p in paths:
            try:
                with Image.open(p) as im:
                    im.verify()
                reasons.append("통과")
            except Exception:
                reasons.append("열기 실패")
                print(f"  [경고] 열 수 없는 파일: {p}")
        print("[1/3 전신 필터] 생략 (--no_filter)")
    else:
        reasons = fullbody_filter(paths, root, out / "fullbody_judged.json", args, device)
    passed = [i for i, r in enumerate(reasons) if r == "통과"]
    print(f"  통과 {len(passed)}/{len(paths)}장, 탈락 사유 {dict(Counter(r for r in reasons if r != '통과'))}")

    # 열기 실패 사진은 예측에서 제외
    ok_idx = [i for i, r in enumerate(reasons) if r != "열기 실패"]
    ok_paths = [paths[i] for i in ok_idx]

    # ---------------- 2) 앙상블 예측 ----------------
    print(f"\n[2/3 예측] {len(ok_paths)}장 x 모델 {len(args.runs)}개")
    all_probs, classes = [], None
    for run in args.runs:
        model, cls = load_run(run, device)
        classes = classes or cls
        assert cls == classes, "run들의 클래스 목록이 다릅니다 (4분류 run만 넣으세요)."
        all_probs.append(probs_for(model, ok_paths, device))
        print(f"  {Path(run).name} 완료")
        del model
    if set(classes) != {"front", "back", "left", "right"}:
        print(f"  [주의] 4분류 모델이 아닙니다: {classes}")
    stack = torch.stack(all_probs)                 # (모델, N, C)
    probs = stack.mean(0)
    ens = probs.argmax(1).tolist()
    conf = probs.max(1).values.tolist()
    each = stack.argmax(2).T.tolist()              # (N, 모델)
    agree_all = [len(set(e)) == 1 for e in each]

    rec = {}  # 원래 인덱스 -> 결과
    for j, i in enumerate(ok_idx):
        rec[i] = dict(pred=classes[ens[j]], conf=conf[j], probs=probs[j].tolist(),
                      each=[classes[k] for k in each[j]], agree=agree_all[j])

    # ---------------- 저장: csv ----------------
    with open(out / "predictions.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "fullbody", "pred", "confidence", "models_agree",
                    *[f"p_{c}" for c in classes], *[f"pred_{Path(r).name}" for r in args.runs]])
        for i, p in enumerate(paths):
            r = rec.get(i)
            if r is None:
                w.writerow([str(p), reasons[i], "", "", "", *[""] * len(classes), *[""] * len(args.runs)])
            else:
                w.writerow([str(p), reasons[i], r["pred"], f"{r['conf']:.4f}", int(r["agree"]),
                            *[f"{v:.4f}" for v in r["probs"]], *r["each"]])

    # ---------------- 저장: 방향별 폴더 ----------------
    if not args.no_copy:
        sd = out / "sorted"
        for i, p in enumerate(paths):
            name = f"{i:05d}_{p.name}"  # 하위 폴더에 같은 파일명이 있어도 덮어쓰지 않게
            if reasons[i] == "통과" and i in rec:
                dst = sd / rec[i]["pred"] / name
            else:
                dst = sd / "_excluded" / reasons[i].replace("/", "_") / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(p, dst)

    # ---------------- 3) 요약 ----------------
    P = [i for i in passed if i in rec]          # 분류 대상(전신 통과)
    X = [i for i in rec if reasons[i] != "통과"]  # 탈락했지만 예측은 한 사진
    S = {"n_total": len(paths), "n_fullbody": len(passed),
         "fullbody_rate": len(passed) / len(paths),
         "excluded_reasons": dict(Counter(r for r in reasons if r != "통과")),
         "classes": classes, "runs": [Path(r).name for r in args.runs]}

    def block(idx):
        if not idx:
            return {}
        cs = [rec[i]["conf"] for i in idx]
        b = {"n": len(idx),
             "pred_dist": {c: sum(rec[i]["pred"] == c for i in idx) for c in classes},
             "conf_mean": st.mean(cs), "conf_median": st.median(cs),
             "models_agree_rate": sum(rec[i]["agree"] for i in idx) / len(idx),
             "auto_rate_at": {str(t): sum(c >= t for c in cs) / len(idx) for t in THRESHOLDS},
             "per_class": {}}
        for c in classes:
            ci = [i for i in idx if rec[i]["pred"] == c]
            if ci:
                cc = [rec[i]["conf"] for i in ci]
                b["per_class"][c] = {"n": len(ci), "conf_mean": st.mean(cc),
                                     "models_agree_rate": sum(rec[i]["agree"] for i in ci) / len(ci),
                                     "auto_rate_at_0.9": sum(v >= 0.9 for v in cc) / len(ci)}
        return b

    S["fullbody"] = block(P)
    S["excluded_but_predicted"] = block(X)
    json.dump(S, open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    L = []
    L.append(f"[입력] 전체 {S['n_total']}장 | 모델 {', '.join(S['runs'])}")
    if args.no_filter:
        L.append(f"[전신 필터] 생략 (--no_filter), 분류 대상 {S['n_fullbody']}장  제외 {S['excluded_reasons']}")
    else:
        L.append(f"[전신 필터] 통과 {S['n_fullbody']}장 ({S['fullbody_rate']:.1%})  탈락 사유 {S['excluded_reasons']}")
    for title, b in [("분류 대상 (전신 통과)", S["fullbody"]), ("참고: 전신 탈락 사진", S["excluded_but_predicted"])]:
        if not b:
            continue
        L.append(f"\n== {title}: {b['n']}장 ==")
        L.append("  방향별 예측 분포: " + ", ".join(
            f"{KO[c]} {b['pred_dist'][c]} ({b['pred_dist'][c]/b['n']:.1%})" for c in classes))
        L.append(f"  신뢰도 평균 {b['conf_mean']:.3f}, 중앙값 {b['conf_median']:.3f}")
        L.append(f"  모델 3개 예측 일치율 {b['models_agree_rate']:.1%}")
        L.append("  신뢰도 기준별 자동 처리 비율: " + ", ".join(
            f">={t} {b['auto_rate_at'][str(t)]:.1%}" for t in THRESHOLDS))
        L.append("  방향별: " + " | ".join(
            f"{KO[c]} n={v['n']} 신뢰도 {v['conf_mean']:.3f} 일치율 {v['models_agree_rate']:.1%} "
            f">=0.9 {v['auto_rate_at_0.9']:.1%}" for c, v in b["per_class"].items()))
    txt = "\n".join(L)
    print("\n[3/3 요약]\n" + txt)
    open(out / "summary.txt", "w", encoding="utf-8").write(txt + "\n")

    # 미리보기 그리드 (전신 통과 사진, 방향별 신뢰도 낮은/높은 순)
    for c in classes:
        ci = sorted((i for i in P if rec[i]["pred"] == c), key=lambda i: rec[i]["conf"])
        for tag, sel in [("low", ci[:args.grid_n]), ("high", ci[::-1][:args.grid_n])]:
            make_grid([paths[i] for i in sel], [f"{i:05d} {rec[i]['conf']:.2f}" for i in sel],
                      out / f"grid_{c}_{tag}.jpg")

    # 신뢰도 히스토그램
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 3.5))
        bins = [i / 20 for i in range(5, 21)]
        plt.hist([rec[i]["conf"] for i in P], bins=bins, alpha=0.8, label=f"full-body (n={len(P)})")
        if X:
            plt.hist([rec[i]["conf"] for i in X], bins=bins, alpha=0.6, label=f"excluded (n={len(X)})")
        plt.xlabel("ensemble confidence"); plt.ylabel("images"); plt.legend(); plt.tight_layout()
        plt.savefig(out / "confidence_hist.png", dpi=150)
    except ImportError:
        print("  (matplotlib 없음: 히스토그램 생략)")

    print(f"\n완료: {out}/  (predictions.csv, summary.txt, sorted/, grid_*.jpg)")


if __name__ == "__main__":
    main()
