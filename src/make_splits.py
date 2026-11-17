"""Stratified train/val/test split of the raw dataset.

Copies images from the raw Kaggle folder into data/splits/{train,val,test}/<class>/
with a stratified 70/15/15 ratio.
"""
import random
import shutil
from collections import defaultdict

from common import CLASS_NAMES, RAW_DIR, SPLIT_DIR, TRAIN_RATIO, VAL_RATIO, SEED


def find_class_dirs():
    """Locate the directory that contains the six class sub-folders."""
    for root in [RAW_DIR, *RAW_DIR.rglob("*")]:
        if not root.is_dir():
            continue
        subdirs = {d.name.lower(): d for d in root.iterdir() if d.is_dir()}
        if all(c in subdirs for c in CLASS_NAMES):
            return root
    raise FileNotFoundError(
        f"Could not find class folders {CLASS_NAMES} under {RAW_DIR}"
    )


def make_splits():
    root = find_class_dirs()
    rng = random.Random(SEED)

    if SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)

    counts = defaultdict(lambda: defaultdict(int))
    for cls in CLASS_NAMES:
        files = sorted((root / cls).glob("*"))
        files = [f for f in files if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        rng.shuffle(files)
        n = len(files)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        subsets = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }
        for split, split_files in subsets.items():
            dest = SPLIT_DIR / split / cls
            dest.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy2(f, dest / f.name)
                counts[split][cls] += 1

    print("Split complete:")
    for split in ("train", "val", "test"):
        total = sum(counts[split].values())
        print(f"  {split:5s}: {total:5d} images  {dict(counts[split])}")


if __name__ == "__main__":
    make_splits()
