"""
train_viewpoint.py — 촬영 방향 분류 학습 (서버에서 실행)
=========================================================================
  --mode 4class : front / back / left / right
  --mode 3class : front / back / side   (left, right 를 side 로 합침)
  --mode 2class : front / nonfront      (어블레이션 A용: 정면 vs 비정면)

어블레이션 옵션
  --label_source df2 : 학습·검증 라벨로 DeepFashion2 원래 viewpoint 사용 (2class 전용, 2=정면 / 3=비정면)
                       평가(test)는 항상 직접 구축한 라벨로 채점
  --flip swap|noswap|none : 좌우 반전 + 라벨 교환(기본) / 반전하되 라벨 유지 / 반전 안 함
  --no_pretrained    : ImageNet 사전학습 없이 처음부터 학습

심화 실험 옵션 (학습 셋에만 적용, 검증·평가는 항상 깨끗한 직접 라벨)
  --noise_rate 0.2 --noise_type random|adjacent|lr : 학습 라벨의 20%를 일부러 틀리게 바꿈
        random   : 다른 3방향 중 아무거나 (무작위 노이즈)
        adjacent : 이웃 방향으로만 (정면<->좌/우측면, 후면<->좌/우측면) = DF2처럼 몸 회전이 애매해서 생기는 쏠린 노이즈
        lr       : 좌측면<->우측면만 서로 바꿈 (좌우 혼동 노이즈)
  --train_frac 0.25  : 학습 이미지의 25%만 사용 (방향별 비율 유지) -> 라벨 수 대비 성능 곡선

좌우 반전 증강 처리:
  4class: 반전하면 좌측면<->우측면이 바뀌므로 라벨도 left<->right 로 교환
  3class: 반전해도 라벨 그대로 (정면/후면/측면은 좌우 대칭)

사용 예시
  python3 train_viewpoint.py --data_root vp4_dataset --mode 3class --seed 42 --out_dir runs_vp/r50_3c_s42
  python3 train_viewpoint.py --data_root vp4_dataset --mode 4class --seed 42 --out_dir runs_vp/r50_4c_s42
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from models import build_model

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
CLASSES = {"4class": ["back", "front", "left", "right"], "3class": ["back", "front", "side"],
           "2class": ["front", "nonfront"]}
DF2_TO_2CLASS = {"2": "front", "3": "nonfront"}
SWAP = {"left": "right", "right": "left"}


ADJ = {"front": ["left", "right"], "back": ["left", "right"], "left": ["front", "back"], "right": ["front", "back"]}
ALL4 = ["front", "back", "left", "right"]


def corrupt_labels(items, rate, ntype, seed):
    """학습 라벨 일부를 일부러 틀리게 바꿈. 같은 seed면 항상 같은 사진이 바뀜 (재현 가능)."""
    rng = random.Random(seed * 1000 + 7)
    if ntype == "lr":
        cand = [i for i, (_, l) in enumerate(items) if l in ("left", "right")]
        k = round(rate * len(cand))  # lr은 측면 사진 중 rate 비율
    else:
        cand = list(range(len(items)))
        k = round(rate * len(items))
    chosen = set(rng.sample(cand, k))
    out, changes = [], {}
    for i, (p, l) in enumerate(items):
        if i in chosen:
            if ntype == "random":
                new = rng.choice([c for c in ALL4 if c != l])
            elif ntype == "adjacent":
                new = rng.choice(ADJ[l])
            else:
                new = {"left": "right", "right": "left"}[l]
            changes[f"{l}->{new}"] = changes.get(f"{l}->{new}", 0) + 1
            out.append((p, new))
        else:
            out.append((p, l))
    return out, {"n_changed": len(chosen), "n_total": len(items), "changes": changes}


def subsample(items, frac, seed):
    """방향별 비율을 유지하며 frac 만큼만 남김"""
    rng = random.Random(seed * 1000 + 11)
    by = {}
    for it in items:
        by.setdefault(it[1], []).append(it)
    out = []
    for lab, lst in sorted(by.items()):
        lst = sorted(lst, key=lambda x: str(x[0]))
        rng.shuffle(lst)
        out += lst[:max(1, round(frac * len(lst)))]
    return out


class ViewpointDataset(Dataset):
    def __init__(self, root, mode, train, flip="swap", df2_labels=None):
        """df2_labels: {stem: 'front'|'nonfront'} 주면 폴더(직접 라벨) 대신 이 라벨 사용"""
        self.mode, self.train, self.flip = mode, train, flip
        self.classes = CLASSES[mode]
        self.items = []
        for lab in ["front", "back", "left", "right"]:
            for p in sorted((Path(root) / lab).glob("*.jpg")):
                if df2_labels is not None:
                    if p.stem not in df2_labels:
                        continue
                    self.items.append((p, df2_labels[p.stem]))
                else:
                    self.items.append((p, lab))
        if train:
            self.tf = transforms.Compose([
                transforms.Resize((256, 256)), transforms.RandomCrop(224),
                transforms.ColorJitter(0.2, 0.2, 0.2), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        else:
            self.tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                          transforms.Normalize(MEAN, STD)])

    def to_class(self, lab):
        if self.mode == "3class" and lab in ("left", "right"):
            lab = "side"
        if self.mode == "2class" and lab != "front":
            lab = "nonfront"
        return self.classes.index(lab)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, lab = self.items[i]
        img = Image.open(p).convert("RGB")
        if self.train and self.flip != "none" and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if self.flip == "swap":
                lab = SWAP.get(lab, lab)  # 좌우 반전 시 좌측면<->우측면
        return self.tf(img), self.to_class(lab)


@torch.no_grad()
def evaluate(model, loader, device, n_cls, criterion=None):
    model.eval()
    P, Y, loss_sum, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        if criterion is not None:
            loss_sum += criterion(out, y).item() * x.size(0)
            n += x.size(0)
        P.append(out.argmax(1).cpu()); Y.append(y.cpu())
    P, Y = torch.cat(P).numpy(), torch.cat(Y).numpy()
    cm = confusion_matrix(Y, P, labels=list(range(n_cls)))
    return {"accuracy": float(accuracy_score(Y, P)),
            "macro_f1": float(f1_score(Y, P, average="macro", labels=list(range(n_cls)))),
            "per_class_acc": (cm.diagonal() / cm.sum(1).clip(min=1)).tolist(),
            "confusion_matrix": cm.tolist(),
            "loss": loss_sum / n if n else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--mode", choices=["2class", "3class", "4class"], required=True)
    ap.add_argument("--label_source", choices=["manual", "df2"], default="manual")
    ap.add_argument("--flip", choices=["swap", "noswap", "none"], default="swap")
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--class_weight", action="store_true", help="클래스 불균형 보정 (적은 클래스에 가중치)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--noise_rate", type=float, default=0.0, help="학습 라벨 오염 비율 (0~1)")
    ap.add_argument("--noise_type", choices=["random", "adjacent", "lr"], default="random")
    ap.add_argument("--train_frac", type=float, default=1.0, help="학습 이미지 사용 비율 (0~1)")
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.data_root)

    df2 = None
    if args.label_source == "df2":
        if args.mode != "2class":
            raise ValueError("--label_source df2 는 --mode 2class 에서만 사용할 수 있습니다 (DF2 라벨은 측면·후면 미구분).")
        import csv
        with open(root / "split_manifest.csv", encoding="utf-8-sig") as f:
            df2 = {r["stem"]: DF2_TO_2CLASS[str(r["df2_viewpoint"])] for r in csv.DictReader(f)
                   if str(r["df2_viewpoint"]) in DF2_TO_2CLASS}
    ds = {s: ViewpointDataset(root / s, args.mode, train=(s == "train"), flip=args.flip,
                              df2_labels=(df2 if s in ("train", "val") else None))  # test는 항상 직접 라벨
          for s in ("train", "val", "test")}
    extra = {}
    if args.train_frac < 1.0:
        n0 = len(ds["train"].items)
        ds["train"].items = subsample(ds["train"].items, args.train_frac, args.seed)
        extra["train_frac"] = {"before": n0, "after": len(ds["train"].items)}
        print(f"[학습 데이터 축소] {n0}장 -> {len(ds['train'].items)}장 (train_frac={args.train_frac})")
    if args.noise_rate > 0:
        if args.label_source != "manual":
            raise ValueError("--noise_rate 는 직접 라벨(--label_source manual)에서만 사용")
        ds["train"].items, info = corrupt_labels(ds["train"].items, args.noise_rate, args.noise_type, args.seed)
        extra["noise"] = info
        print(f"[라벨 노이즈] {args.noise_type} {args.noise_rate:.0%}: {info['n_changed']}/{info['n_total']}장 변경 {info['changes']}")
    classes = CLASSES[args.mode]
    counts = {s: [sum(1 for _, l in d.items if d.to_class(l) == k) for k in range(len(classes))] for s, d in ds.items()}
    print(f"[설정] mode={args.mode}, label_source={args.label_source}, flip={args.flip}, "
          f"pretrained={not args.no_pretrained}, seed={args.seed}, device={device}")
    print(f"[데이터] 클래스 {classes}")
    for s in ds:
        print(f"  {s:5s}: " + ", ".join(f"{c}={n}" for c, n in zip(classes, counts[s])))

    loaders = {s: DataLoader(d, batch_size=args.batch_size, shuffle=(s == "train"), drop_last=(s == "train"),
                             num_workers=args.num_workers, pin_memory=True) for s, d in ds.items()}

    model = build_model(args.model, num_classes=len(classes), pretrained=not args.no_pretrained).to(device)
    weight = None
    if args.class_weight:
        c = torch.tensor(counts["train"], dtype=torch.float)
        weight = (c.sum() / (len(c) * c.clamp(min=1))).to(device)
        print(f"[클래스 가중치] {dict(zip(classes, [round(w, 2) for w in weight.tolist()]))}")
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1, best_ep, bad, history = -1, -1, 0, []
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); tot, n = 0.0, 0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            loss = criterion(out, y); loss.backward(); optimizer.step()
            tot += loss.item() * x.size(0); n += x.size(0)
        scheduler.step()
        v = evaluate(model, loaders["val"], device, len(classes), criterion)
        print(f"[Epoch {ep:>3d}/{args.epochs}] train_loss={tot/n:.4f}  val_loss={v['loss']:.4f}  "
              f"val_acc={v['accuracy']:.4f}  val_macroF1={v['macro_f1']:.4f}  ({time.time()-t0:.1f}s)")
        history.append({"epoch": ep, "train_loss": tot / n, **{f"val_{k}": v[k] for k in ("loss", "accuracy", "macro_f1")}})
        if v["macro_f1"] > best_f1:
            best_f1, best_ep, bad = v["macro_f1"], ep, 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            print("  -> 최고 성능 갱신, 저장됨")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  -> {args.patience} epoch 동안 개선 없음, 조기 종료"); break

    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
    t = evaluate(model, loaders["test"], device, len(classes), criterion)
    print(f"\n[최종] best epoch={best_ep}, val macroF1={best_f1:.4f}")
    print(f"[Test] accuracy={t['accuracy']:.4f}, macro_f1={t['macro_f1']:.4f}")
    print("  클래스별 정확도: " + ", ".join(f"{c}={a:.3f}" for c, a in zip(classes, t["per_class_acc"])))
    print(f"  혼동행렬 (행=실제, 열=예측), 순서 {classes}")
    for row in t["confusion_matrix"]:
        print("   ", row)
    if args.mode == "4class":
        cm = t["confusion_matrix"]; L, R = classes.index("left"), classes.index("right")
        side_total = sum(cm[L]) + sum(cm[R])
        lr_conf = (cm[L][R] + cm[R][L]) / max(side_total, 1)
        print(f"  좌<->우 혼동률: {lr_conf:.4f}  (측면 사진 중 반대쪽으로 틀린 비율)")
        t["left_right_confusion"] = lr_conf

    json.dump({"model": args.model, "mode": args.mode, "classes": classes, "best_epoch": best_ep,
               "best_val_macro_f1": best_f1, "test": t, "counts": counts, "history": history, "args": vars(args), **extra},
              open(out_dir / "result.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n저장: {out_dir/'result.json'}, {out_dir/'best_model.pt'}")


if __name__ == "__main__":
    main()
