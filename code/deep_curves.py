r"""
deep_curves.py — 심화 실험(라벨 노이즈 곡선, 라벨 수 곡선) 표와 그림 만들기
=========================================================================
run_parallel.py --suite deep 이 끝나면 생기는 <tag>_ablation_summary.json 을 읽어서
  ① 노이즈 비율(0~40%) x 노이즈 종류(무작위 / 이웃방향) 에 따른 4분류 정확도·좌우혼동
  ② 학습 라벨 수(10/25/50/100%) 에 따른 4분류 정확도
를 표로 출력하고 그림(png)으로 저장합니다. 0% 노이즈 / 100% 라벨 지점은 기본 4c 결과를 사용.

사용 예시
  python3 deep_curves.py --summary runs_vp/cnxt_ablation_summary.json --tag cnxt
"""

import argparse
import json
import statistics as st
from pathlib import Path


def ms(v):
    v = [x * 100 for x in v]
    return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--tag", default="cnxt")
    ap.add_argument("--n_train", type=int, default=4170, help="전체 학습 이미지 수 (x축 표시용)")
    args = ap.parse_args()
    S = json.load(open(args.summary, encoding="utf-8"))
    base = S.get(f"{args.tag}_4c")
    if base is None:
        raise SystemExit(f"{args.tag}_4c 결과가 summary에 없습니다.")
    metrics = [("DF2_4class", "DF2 4-class acc"), ("쇼핑몰_4class", "Shop 4-class acc"),
               ("DF2_lr_conf", "DF2 L/R confusion"), ("DF2_3class", "DF2 3-class acc")]

    # ---------------- ① 노이즈 곡선
    rates = [0, 10, 20, 30, 40]
    noise = {}
    print("\n① 라벨 노이즈 비율별 성능 (%, seed 평균±표준편차)")
    print(f"  {'종류':10s}{'비율':>6s}" + "".join(f"{n:>22s}" for _, n in metrics))
    for kind, key in [("random", "nrand"), ("adjacent", "nadj")]:
        noise[kind] = {}
        for r in rates:
            g = base if r == 0 else S.get(f"{args.tag}_4c_{key}{r}")
            if not g:
                continue
            noise[kind][r] = {m: ms(g[m]) for m, _ in metrics if m in g}
            print(f"  {kind:10s}{r:>5d}%" + "".join(
                f"{noise[kind][r][m][0]:>15.2f}±{noise[kind][r][m][1]:<5.2f}" for m, _ in metrics if m in noise[kind][r]))

    # ---------------- ② 라벨 수 곡선
    fracs = [10, 25, 50, 100]
    frac = {}
    print("\n② 학습 라벨 수별 성능 (%)")
    for f in fracs:
        g = base if f == 100 else S.get(f"{args.tag}_4c_frac{f}")
        if not g:
            continue
        frac[f] = {m: ms(g[m]) for m, _ in metrics if m in g}
        print(f"  {f:>3d}% (약 {round(args.n_train*f/100):>4d}장)" + "".join(
            f"   {n}: {frac[f][m][0]:.2f}±{frac[f][m][1]:.2f}" for m, n in metrics[:2] if m in frac[f]))

    # ---------------- 그림
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 없음: 그림 생략"); return
    out = Path(args.summary).parent
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for kind, mk, lab in [("random", "o-", "random noise"), ("adjacent", "s-", "adjacent-direction noise")]:
        xs = sorted(noise.get(kind, {}))
        for i, (m, title) in enumerate([("DF2_4class", "DF2 test: 4-class accuracy (%)"),
                                        ("DF2_lr_conf", "DF2 test: left/right confusion (%)")]):
            ys = [noise[kind][x][m][0] for x in xs]
            es = [noise[kind][x][m][1] for x in xs]
            ax[i].errorbar(xs, ys, yerr=es, fmt=mk, capsize=3, label=lab)
            ax[i].set_title(title); ax[i].set_xlabel("label noise rate (%)"); ax[i].grid(alpha=.3)
    ax[0].legend()
    xs = sorted(frac)
    for m, mk, lab in [("DF2_4class", "o-", "DF2 test"), ("쇼핑몰_4class", "s-", "shop 200")]:
        ax[2].errorbar([round(args.n_train * x / 100) for x in xs], [frac[x][m][0] for x in xs],
                       yerr=[frac[x][m][1] for x in xs], fmt=mk, capsize=3, label=lab)
    ax[2].set_xscale("log"); ax[2].set_title("4-class accuracy vs. # training labels")
    ax[2].set_xlabel("training images (log)"); ax[2].grid(alpha=.3); ax[2].legend()
    plt.tight_layout()
    png = out / f"{args.tag}_deep_curves.png"
    plt.savefig(png, dpi=200)
    json.dump({"noise": noise, "frac": frac}, open(out / f"{args.tag}_deep_curves.json", "w"), indent=2)
    print(f"\n저장: {png}")


if __name__ == "__main__":
    main()
