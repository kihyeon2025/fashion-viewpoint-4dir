r"""
run_parallel.py — 비어 있는 GPU를 모두 사용해 실험을 동시에 돌리는 실행기
=========================================================================
run_modern_cnn.sh와 같은 실험(본 실험 + 어블레이션 A·B·C, seed 3회)을 만들되,
학습 21회를 '작업 대기열'에 넣고 비어 있는 GPU마다 하나씩 꺼내 동시에 실행합니다.
GPU 하나가 끝나면 바로 다음 작업을 가져가므로 GPU가 놀지 않습니다.
모든 학습이 끝나면 쇼핑몰 200장 평가 + 어블레이션 통합 비교까지 자동 실행.

사용 예시 (서버, 프로젝트 폴더에서, tmux 안에서 실행 권장)
  conda activate <환경이름>
  python3 run_parallel.py --models convnext_tiny                    # 빈 GPU 자동 감지
  python3 run_parallel.py --models efficientnet_v2_s regnety_8gf    # 여러 모델을 한 번에
  python3 run_parallel.py --models convnext_tiny --gpus 1,2,3       # GPU 직접 지정
  python3 run_parallel.py --models convnext_tiny --only 4c 3c       # 일부 실험만
  python3 run_parallel.py --models convnext_tiny --dry_run          # 무엇을 돌릴지 목록만 확인
  python3 run_parallel.py --models convnext_tiny --suite deep       # 심화: 노이즈 곡선 + 라벨 수 곡선 (24종 x seed 3)

이미 끝난 run(result.json 있음)은 건너뜁니다 -> 중간에 끊겨도 같은 명령으로 이어서 진행.
각 학습 로그: runs_vp/<run이름>/train.log
"""

import argparse
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

TAGS = {"resnet50": "r50", "convnext_tiny": "cnxt", "convnext_small": "cnxs",
        "efficientnet_v2_s": "effv2s", "regnety_8gf": "regy8"}

# 이름 꼬리 -> train_viewpoint.py 옵션 (run_modern_cnn.sh 와 동일)
EXPERIMENTS = {
    "4c":         ["--mode", "4class"],
    "3c":         ["--mode", "3class"],
    "2c_manual":  ["--mode", "2class", "--label_source", "manual"],
    "2c_df2":     ["--mode", "2class", "--label_source", "df2"],
    "4c_noflip":  ["--mode", "4class", "--flip", "none"],
    "4c_noswap":  ["--mode", "4class", "--flip", "noswap"],
    "4c_scratch": ["--mode", "4class", "--no_pretrained"],
}
# 심화 실험 ① 라벨 노이즈 곡선 (무작위 vs 이웃방향=DF2형)
for r in (10, 20, 30, 40):
    EXPERIMENTS[f"4c_nrand{r}"] = ["--mode", "4class", "--noise_type", "random", "--noise_rate", str(r / 100)]
    EXPERIMENTS[f"4c_nadj{r}"] = ["--mode", "4class", "--noise_type", "adjacent", "--noise_rate", str(r / 100)]
# 심화 실험 ② 라벨 수 대비 성능 곡선 (100%는 기본 4c 결과 사용)
for f in (10, 25, 50):
    EXPERIMENTS[f"4c_frac{f}"] = ["--mode", "4class", "--train_frac", str(f / 100)]

SUITES = {
    "base": ["4c", "3c", "2c_manual", "2c_df2", "4c_noflip", "4c_noswap", "4c_scratch"],
    "noise": [k for k in EXPERIMENTS if k.startswith("4c_nrand") or k.startswith("4c_nadj")],
    "frac": [k for k in EXPERIMENTS if k.startswith("4c_frac")],
}
SUITES["deep"] = SUITES["noise"] + SUITES["frac"]

print_lock = threading.Lock()


def log(msg):
    with print_lock:
        print(f"[{datetime.now():%m-%d %H:%M:%S}] {msg}", flush=True)


def free_gpus(max_used_mib):
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    gpus = []
    for line in out.strip().splitlines():
        idx, used = [s.strip() for s in line.split(",")]
        if int(used) <= max_used_mib:
            gpus.append(idx)
    return gpus


def worker(gpu, jobs, args, results):
    while True:
        try:
            name, opts = jobs.get_nowait()
        except queue.Empty:
            return
        run_dir = Path(args.out) / name
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "train_viewpoint.py", "--data_root", args.data, "--out_dir", str(run_dir),
               "--num_workers", str(args.num_workers), *opts]
        log(f"GPU{gpu} 시작  {name}   (남은 대기 {jobs.qsize()}개)")
        t0 = time.time()
        with open(run_dir / "train.log", "w", encoding="utf-8") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": gpu}).returncode
        mins = (time.time() - t0) / 60
        ok = rc == 0 and (run_dir / "result.json").exists()
        results[name] = ok
        log(f"GPU{gpu} {'완료' if ok else '실패'}  {name}   ({mins:.1f}분)"
            + ("" if ok else f"  -> 로그 확인: {run_dir/'train.log'}"))


def evaluate(model, args, gpu):
    tag = TAGS[model]
    out = Path(args.out)
    env = {**__import__("os").environ, "CUDA_VISIBLE_DEVICES": gpu}
    steps = []
    for mode in ("4c", "3c"):
        runs = [str(out / f"{tag}_{mode}_s{s}") for s in args.seeds]
        if all((Path(r) / "result.json").exists() for r in runs):
            steps.append((f"{tag}_{mode}_shop.log",
                          [sys.executable, "predict_viewpoint.py", "--runs", *runs, "--images", args.shop,
                           "--out", str(out / f"{tag}_{mode}_shop.csv")]))
    runs = sorted(str(p) for p in out.glob(f"{tag}_*_s*") if (p / "result.json").exists()
                  and p.name.rsplit("_s", 1)[-1] in args.seeds)
    if runs:
        steps.append((f"{tag}_ablation.log",
                      [sys.executable, "ablation_eval.py", "--data_root", args.data, "--shop_root", args.shop,
                       "--runs", *runs, "--out", str(out / f"{tag}_ablation_summary.json")]))
    for logname, cmd in steps:
        log(f"평가 {logname}")
        res = subprocess.run(cmd, capture_output=True, text=True, env=env)
        (out / logname).write_text(res.stdout + res.stderr, encoding="utf-8")
        if res.returncode != 0:
            log(f"  [오류] {out/logname} 확인")
    return [str(out / l) for l, _ in steps]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["convnext_tiny"], choices=list(TAGS))
    ap.add_argument("--suite", default="base", choices=list(SUITES),
                    help="base=본 실험+어블레이션 / noise=라벨 노이즈 곡선 / frac=라벨 수 곡선 / deep=noise+frac")
    ap.add_argument("--only", nargs="+", choices=list(EXPERIMENTS), help="개별 실험 직접 지정 (--suite 대신)")
    ap.add_argument("--seeds", nargs="+", default=["42", "43", "44"])
    ap.add_argument("--gpus", default="auto", help="'auto' = 메모리 거의 안 쓰는 GPU 전부, 또는 '0,1,2'")
    ap.add_argument("--max_used_mib", type=int, default=1000, help="auto일 때 이 이하로 쓰는 GPU를 '빈 GPU'로 봄")
    ap.add_argument("--data", default="vp4_dataset")
    ap.add_argument("--shop", default="shop_test")
    ap.add_argument("--out", default="runs_vp")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--no_eval", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    gpus = free_gpus(args.max_used_mib) if args.gpus == "auto" else [g.strip() for g in args.gpus.split(",")]
    if not gpus:
        sys.exit("빈 GPU가 없습니다. nvidia-smi 확인 후 --gpus 로 직접 지정하세요.")

    # 작업 목록: 본 실험(4c, 3c)을 먼저 넣어 핵심 결과가 가장 빨리 나오게,
    # 오래 걸리는 scratch는 맨 뒤
    exps = args.only or SUITES[args.suite]
    jobs, skipped = queue.Queue(), []
    for exp in exps:
        for model in args.models:
            for s in args.seeds:
                name = f"{TAGS[model]}_{exp}_s{s}"
                if (Path(args.out) / name / "result.json").exists():
                    skipped.append(name)
                    continue
                jobs.put((name, ["--model", model, "--seed", s, *EXPERIMENTS[exp]]))

    log(f"GPU {gpus} 사용 | 새로 학습 {jobs.qsize()}개 | 이미 완료라 건너뜀 {len(skipped)}개")
    if args.dry_run:
        for name, opts in list(jobs.queue):
            print("  ", name, " ".join(opts))
        return

    results, threads = {}, []
    t0 = time.time()
    for g in gpus[:max(1, jobs.qsize())]:
        th = threading.Thread(target=worker, args=(g, jobs, args, results))
        th.start(); threads.append(th)
        time.sleep(5)  # 동시에 사진 캐시를 읽는 부하를 조금 분산
    for th in threads:
        th.join()

    failed = [n for n, ok in results.items() if not ok]
    log(f"학습 종료: 성공 {len(results) - len(failed)} / 실패 {len(failed)}  (총 {(time.time()-t0)/60:.0f}분)")
    if failed:
        log("실패한 run: " + ", ".join(failed) + "  -> 원인 해결 후 같은 명령을 다시 실행하면 그것만 재학습")

    if not args.no_eval:
        files = []
        for model in args.models:
            files += evaluate(model, args, gpus[0])
        log("결과 파일:\n  " + "\n  ".join(files))


if __name__ == "__main__":
    main()
