r"""
label_viewpoint.py — 촬영 방향(정면/후면/좌측면/우측면) 라벨링 도구 (로컬 PC에서 실행)
=========================================================================
추가 설치 없이 Python 기본 tkinter + Pillow만 사용합니다.

[라벨링 기준 — 인물 몸 기준]
  1 정면   : 몸통이 거의 정면을 향함
  2 후면   : 등이 거의 정면을 향함
  3 좌측면 : 인물의 "왼쪽 몸"이 보임 (45도 비스듬한 경우 포함)
  4 우측면 : 인물의 "오른쪽 몸"이 보임 (45도 비스듬한 경우 포함)
  S 제외   : 여러 명 / 사람 없음(옷만) / 판단 불가
  Z 되돌리기,  Q 저장 후 종료  (매 장마다 자동 저장되므로 언제 꺼도 이어서 가능)

[사용 1] 라벨링 (처음 실행 시 후보 목록을 만들며 19만 개 어노테이션을 한 번 스캔 — 몇 분 소요)
  python label_viewpoint.py --raw_root "C:\...\deepfashion2_original_images\train" --out labels.csv

[사용 2] 신뢰도 확인용 재라벨링 (며칠 뒤, 이전 라벨을 보지 않고 100장 다시 라벨링)
  python label_viewpoint.py --raw_root "..." --out labels.csv --recheck 100

[사용 3] 일치율 계산
  python label_viewpoint.py --raw_root "..." --out labels.csv --agreement
"""

import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

from PIL import Image

KEYS = {"1": "front", "2": "back", "3": "left", "4": "right", "s": "skip"}
KO = {"front": "정면", "back": "후면", "left": "좌측면", "right": "우측면", "skip": "제외"}
DF2_VP_KO = {1: "옷만(착용X)", 2: "정면", 3: "측면/후면"}
FIELDS = ["stem", "source", "df2_viewpoint", "label", "time"]
GUIDE = ("[기준: 인물 몸 기준]  1 정면   2 후면   3 좌측면(인물 왼쪽 몸 보임)   4 우측면(인물 오른쪽 몸 보임)\n"
         "45° 비스듬 → 보이는 몸 쪽으로 좌/우   |   S 제외(여러 명·옷만·판단불가)   Z 되돌리기   Q 종료\n"
         "※ 몸통(가슴·어깨선) 방향만 판단. 팔·다리·얼굴 방향은 무시. 어깨 폭이 정면의 절반 이하로 좁아 보이면 측면")


# ------------------------------------------------------------------ 후보 목록
def image_viewpoint(data):
    """한 이미지의 대표 viewpoint (착용 아이템들의 최빈값). 전부 1(옷만)이면 1."""
    vps = [v.get("viewpoint") for k, v in data.items() if k.startswith("item") and isinstance(v, dict)]
    vps = [v for v in vps if v is not None]
    if not vps:
        return None
    worn = [v for v in vps if v != 1]
    return Counter(worn).most_common(1)[0][0] if worn else 1


def build_candidates(raw_root, cache_path, sources, per_viewpoint, seed):
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            cands = json.load(f)
        if cands:
            print(f"[후보] 캐시 사용: {cache_path} ({len(cands)}장)")
            return cands
        print(f"[후보] 캐시가 비어 있어 다시 만듭니다: {cache_path}")

    anno_dir = raw_root / "annos"
    files = sorted(anno_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"어노테이션을 찾지 못했습니다: {anno_dir}\n"
                                f"--raw_root 경로를 확인하세요 (annos/, image/ 폴더가 들어있는 train 폴더).")
    print(f"[후보] 어노테이션 {len(files)}개 스캔 중... (최초 1회)")
    pools = {2: [], 3: []}
    for i, p in enumerate(files):
        if i and i % 20000 == 0:
            print(f"  ...{i}/{len(files)}")
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("source") not in sources:
            continue
        vp = image_viewpoint(d)
        if vp in pools:
            pools[vp].append({"stem": p.stem, "source": d.get("source"), "df2_viewpoint": vp})

    rng = random.Random(seed)
    cands = []
    for vp, pool in pools.items():
        rng.shuffle(pool)
        cands += pool[:per_viewpoint]
        print(f"  DeepFashion2 viewpoint={vp}({DF2_VP_KO[vp]}): 전체 {len(pool)}장 중 {min(len(pool), per_viewpoint)}장 선택")
    rng.shuffle(cands)  # 정면/측면후면이 섞여 나오도록
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cands, f, ensure_ascii=False)
    print(f"[후보] {len(cands)}장 저장: {cache_path}")
    return cands


# ------------------------------------------------------------------ CSV
def read_rows(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def append_row(path, row):
    new = not path.exists()
    with open(path, "a", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def rewrite_rows(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def counts_text(rows):
    c = Counter(r["label"] for r in rows)
    return "   ".join(f"{KO[k]} {c.get(k, 0)}" for k in ["front", "back", "left", "right", "skip"])


# ------------------------------------------------------------------ GUI
class Labeler:
    def __init__(self, items, img_dir, out_csv, show_hint, title):
        import tkinter as tk
        from PIL import ImageTk
        self.tk, self.ImageTk = tk, ImageTk
        self.img_dir, self.out_csv, self.show_hint = img_dir, out_csv, show_hint

        done = {r["stem"] for r in read_rows(out_csv)}
        self.total = len(items)
        self.queue = [it for it in items if it["stem"] not in done]
        self.history = []  # 이번 세션에서 라벨링한 항목 (되돌리기용)

        self.root = tk.Tk()
        self.root.title(title)
        font = ("Malgun Gothic", 11)
        self.guide = tk.Label(self.root, text=GUIDE, font=font, justify="left", fg="#1c3d7a")
        self.guide.pack(padx=8, pady=(8, 2), anchor="w")
        self.status = tk.Label(self.root, text="", font=("Malgun Gothic", 12, "bold"))
        self.status.pack(padx=8, pady=2, anchor="w")
        self.hint = tk.Label(self.root, text="", font=font, fg="#888888")
        self.hint.pack(padx=8, anchor="w")
        self.canvas = tk.Label(self.root)
        self.canvas.pack(padx=8, pady=8)

        for k in KEYS:
            self.root.bind(k, lambda e, k=k: self.label(KEYS[k]))
            self.root.bind(k.upper(), lambda e, k=k: self.label(KEYS[k]))
        for k in ("z", "Z"):
            self.root.bind(k, lambda e: self.undo())
        for k in ("q", "Q", "<Escape>"):
            self.root.bind(k, lambda e: self.root.destroy())
        self.show()

    def show(self):
        rows = read_rows(self.out_csv)
        n_done = len({r["stem"] for r in rows})
        if not self.queue:
            self.status.config(text=f"완료! {n_done}/{self.total}   |   {counts_text(rows)}   (Q로 종료)")
            self.canvas.config(image="")
            self.hint.config(text="")
            return
        it = self.queue[0]
        self.status.config(text=f"진행 {n_done}/{self.total}   |   {counts_text(rows)}")
        self.hint.config(text=(f"참고) DeepFashion2 원래 라벨: {DF2_VP_KO[it['df2_viewpoint']]}   "
                               f"|  출처: {it['source']}  |  {it['stem']}.jpg") if self.show_hint else it["stem"])
        img = Image.open(self.img_dir / f"{it['stem']}.jpg").convert("RGB")
        img.thumbnail((820, 820))
        self._photo = self.ImageTk.PhotoImage(img)
        self.canvas.config(image=self._photo)

    def label(self, lab):
        if not self.queue:
            return
        it = self.queue.pop(0)
        append_row(self.out_csv, {"stem": it["stem"], "source": it["source"],
                                  "df2_viewpoint": it["df2_viewpoint"], "label": lab,
                                  "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        self.history.append(it)
        self.show()

    def undo(self):
        if not self.history:
            return
        it = self.history.pop()
        rows = read_rows(self.out_csv)
        for i in range(len(rows) - 1, -1, -1):  # 마지막으로 저장된 해당 항목 삭제
            if rows[i]["stem"] == it["stem"]:
                del rows[i]
                break
        rewrite_rows(self.out_csv, rows)
        self.queue.insert(0, it)
        self.show()

    def run(self):
        self.root.mainloop()


# ------------------------------------------------------------------ 검토 모드
class Reviewer(Labeler):
    """이미 라벨링한 사진 중 특정 라벨만 다시 보며 수정. Enter/Space=유지, 1~4/S=변경."""

    def __init__(self, out_csv, img_dir, review_labels):
        self.done_path = out_csv.with_name(out_csv.stem + "_reviewed.txt")
        reviewed = set(self.done_path.read_text(encoding="utf-8").split()) if self.done_path.exists() else set()
        rows = read_rows(out_csv)
        items = [{"stem": r["stem"], "source": r["source"], "df2_viewpoint": int(r["df2_viewpoint"]), "cur": r["label"]}
                 for r in rows if r["label"] in review_labels and r["stem"] not in reviewed]
        self.review_labels = review_labels
        self.n_changed = 0
        # 부모 초기화: 이미 라벨된 사진을 건너뛰지 않도록 가짜 CSV 경로로 queue 구성 후 교체
        super().__init__(items, img_dir, out_csv.with_name("__none__.csv"), show_hint=False, title="측면 라벨 검토")
        self.out_csv = out_csv
        self.queue = items
        self.total = len(items)
        self.guide.config(text=GUIDE.replace("Z 되돌리기", "Enter/Space 유지   Z 되돌리기")
)
        for k in ("<Return>", "<space>"):
            self.root.bind(k, lambda e: self.keep())
        self.show()

    def show(self):
        if not hasattr(self, "review_labels"):
            return
        if not self.queue:
            self.status.config(text=f"검토 완료! 변경 {self.n_changed}장 (Q로 종료)")
            self.canvas.config(image="")
            self.hint.config(text="")
            return
        it = self.queue[0]
        done = self.total - len(self.queue)
        self.status.config(text=f"검토 {done}/{self.total}   |   현재 라벨: 【 {KO[it['cur']]} 】   |   변경 {self.n_changed}장")
        self.hint.config(text=it["stem"])
        img = Image.open(self.img_dir / f"{it['stem']}.jpg").convert("RGB")
        img.thumbnail((820, 820))
        self._photo = self.ImageTk.PhotoImage(img)
        self.canvas.config(image=self._photo)

    def _set(self, stem, lab):
        rows = read_rows(self.out_csv)
        for r in rows:
            if r["stem"] == stem:
                r["label"] = lab
        rewrite_rows(self.out_csv, rows)

    def _mark(self, stem, add=True):
        done = set(self.done_path.read_text(encoding="utf-8").split()) if self.done_path.exists() else set()
        done.add(stem) if add else done.discard(stem)
        self.done_path.write_text("\n".join(sorted(done)), encoding="utf-8")

    def keep(self):
        if not self.queue:
            return
        it = self.queue.pop(0)
        self._mark(it["stem"])
        self.history.append((it, it["cur"]))
        self.show()

    def label(self, lab):
        if not self.queue:
            return
        it = self.queue.pop(0)
        old = it["cur"]
        if lab != old:
            self._set(it["stem"], lab)
            self.n_changed += 1
        self._mark(it["stem"])
        self.history.append((it, old))
        self.show()

    def undo(self):
        if not self.history:
            return
        it, old = self.history.pop()
        rows = {r["stem"]: r["label"] for r in read_rows(self.out_csv)}
        if rows.get(it["stem"]) != old:
            self._set(it["stem"], old)
            self.n_changed -= 1
        self._mark(it["stem"], add=False)
        it["cur"] = old
        self.queue.insert(0, it)
        self.show()


# ------------------------------------------------------------------ 일치율
def agreement(first_csv, recheck_csv):
    a = {r["stem"]: r["label"] for r in read_rows(first_csv)}
    b = {r["stem"]: r["label"] for r in read_rows(recheck_csv)}
    common = [s for s in b if s in a]
    if not common:
        print("비교할 항목이 없습니다. 먼저 --recheck 로 재라벨링하세요.")
        return
    same = sum(a[s] == b[s] for s in common)
    print(f"[라벨 일치율] {same}/{len(common)} = {same / len(common):.1%}")
    labs = ["front", "back", "left", "right", "skip"]
    print("  행=처음 라벨, 열=재라벨 | " + " ".join(f"{KO[l]:>5s}" for l in labs))
    for la in labs:
        row = [sum(1 for s in common if a[s] == la and b[s] == lb) for lb in labs]
        print(f"  {KO[la]:>5s} | " + " ".join(f"{v:>6d}" for v in row))
    # Cohen's kappa: 우연히 같은 답을 낼 확률을 뺀 일치도 (논문 보고용)
    n = len(common)
    pe = sum((sum(a[s] == l for s in common) / n) * (sum(b[s] == l for s in common) / n) for l in labs)
    po = same / n
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    print(f"[Cohen's kappa] {kappa:.3f}  (관찰 일치 {po:.3f}, 우연 일치 {pe:.3f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_root", required=True, help="DeepFashion2 train 폴더 (annos/, image/ 포함)")
    ap.add_argument("--out", default="labels.csv", help="라벨 저장 CSV")
    ap.add_argument("--source", choices=["shop", "user", "all"], default="all")
    ap.add_argument("--per_viewpoint", type=int, default=1500,
                    help="DeepFashion2 정면(2)/측면후면(3)에서 각각 몇 장을 후보로 뽑을지")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--show_hint", action="store_true",
                    help="DeepFashion2 원래 라벨을 화면에 표시 (기본은 숨김: 판단이 끌려가는 것 방지)")
    ap.add_argument("--candidates", default=None,
                    help="후보 목록 파일 직접 지정 (예: filter_fullbody.py 결과 fullbody_candidates.json)")
    ap.add_argument("--recheck", type=int, default=0, help="재라벨링할 장수 (신뢰도 확인용)")
    ap.add_argument("--agreement", action="store_true", help="처음 라벨과 재라벨의 일치율 계산")
    ap.add_argument("--review", default=None,
                    help="특정 라벨만 다시 보며 수정 (예: left,right). Enter=유지, 1~4/S=변경")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    out = Path(args.out)
    recheck_out = out.with_name(out.stem + "_recheck.csv")

    if args.review:
        labs = {x.strip() for x in args.review.split(",")}
        Reviewer(out, raw_root / "image", labs).run()
        return

    if args.agreement:
        agreement(out, recheck_out)
        return

    if args.recheck:
        rows = [r for r in read_rows(out) if r["label"] != "skip"]
        rng = random.Random(args.seed + 1)
        rng.shuffle(rows)
        items = [{"stem": r["stem"], "source": r["source"], "df2_viewpoint": int(r["df2_viewpoint"])}
                 for r in rows[:args.recheck]]
        print(f"[재라벨링] {len(items)}장 (처음 라벨은 보이지 않음) -> {recheck_out}")
        Labeler(items, raw_root / "image", recheck_out, show_hint=False, title="재라벨링 (신뢰도 확인)").run()
        return

    sources = {"shop", "user"} if args.source == "all" else {args.source}
    if args.candidates:
        with open(args.candidates, encoding="utf-8") as f:
            items = json.load(f)
        print(f"[후보] 지정 파일 사용: {args.candidates} ({len(items)}장). 이미 라벨링한 사진은 건너뜁니다.")
    else:
        cache = out.with_name(out.stem + "_candidates.json")
        items = build_candidates(raw_root, cache, sources, args.per_viewpoint, args.seed)
    Labeler(items, raw_root / "image", out, show_hint=args.show_hint, title="촬영 방향 라벨링").run()


if __name__ == "__main__":
    main()
