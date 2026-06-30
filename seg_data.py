"""LandCover.ai aerial imagery — download, tile, and serve as a torch Dataset.

LandCover.ai is 41 orthophotos of Poland at 0.25-0.5 m/px, hand-labelled into
five classes. Its features are exactly the ones a vegetation-monitoring
platform cares about, and the class frequencies are brutally imbalanced —
woodland and background dominate, buildings and roads are <2% of pixels — so
it doubles as a rare-class evaluation problem in the image domain.

The published train/val/test .txt files are *tile* splits — every orthophoto
contributes tiles to all three, which would leak a scene across the split. We
deliberately do NOT use them as-is: instead we assign **whole orthophotos** to
train/val/test (a true geographic holdout, so no scene the model trains on
appears at test time), and only borrow the published tile *names* to know which
512 px grid cells are valid full tiles. The k-index in each name encodes its
(row, col) on the orthophoto grid, so we reconstruct the exact crop, then resize
to a trainable tile size and cache to disk.
"""

import os
import urllib.request
import zipfile

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # the orthophotos are ~9000x9600

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "data", "landcoverai")
IMG_DIR = os.path.join(ROOT, "images")
MASK_DIR = os.path.join(ROOT, "masks")
ZIP_URL = "https://landcover.ai.linuxpolska.com/download/landcover.ai.v1.zip"
ZIP_PATH = os.path.join(ROOT, "landcover.ai.v1.zip")
SOURCE_TILE = 512  # the grid the published splits index against

CLASS_NAMES = ["background", "building", "woodland", "water", "road"]
# muted, map-like palette consistent with the tabular notebook
CLASS_COLORS = ["#E9E4D8", "#B0413E", "#5C9D7E", "#1F6F8B", "#D88C4A"]
N_CLASSES = len(CLASS_NAMES)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def download():
    os.makedirs(ROOT, exist_ok=True)
    if os.path.isdir(IMG_DIR) and len(os.listdir(IMG_DIR)) >= 41:
        return
    if not os.path.exists(ZIP_PATH):
        print(f"downloading {ZIP_URL} (~1.5 GB)")
        urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
    print("extracting")
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(ROOT)


def _read_split(name):
    with open(os.path.join(ROOT, f"{name}.txt")) as f:
        return [ln.strip() for ln in f if ln.strip()]


def _ortho_and_k(basename):
    """'M-33-20-D-c-4-2_137' -> ('M-33-20-D-c-4-2', 137)."""
    ortho, k = basename.rsplit("_", 1)
    return ortho, int(k)


def _crop_box(k, width):
    """Reconstruct the (left, upper, right, lower) box for tile k on the grid.

    split.py iterates `for y in range(0, H, 512): for x in range(0, W, 512)`,
    incrementing k each step, so n_cols = ceil(W / 512) and (row, col) = divmod.
    """
    n_cols = (width + SOURCE_TILE - 1) // SOURCE_TILE
    row, col = divmod(k, n_cols)
    x, y = col * SOURCE_TILE, row * SOURCE_TILE
    return (x, y, x + SOURCE_TILE, y + SOURCE_TILE)


def _all_tiles_by_orthophoto():
    """{orthophoto_id: [tile_name, ...]} over every published (valid) tile."""
    inv = {}
    for split in ("train", "val", "test"):
        for nm in _read_split(split):
            o, _ = _ortho_and_k(nm)
            inv.setdefault(o, []).append(nm)
    return inv


def orthophoto_split(seed=0):
    """Partition the 41 orthophotos into disjoint train/val/test groups.

    This is the geographic holdout: a whole scene lands in exactly one split,
    so test tiles come from orthophotos the model never saw during training.
    """
    orthos = sorted(_all_tiles_by_orthophoto())
    perm = list(np.random.default_rng(seed).permutation(orthos))
    n_hold = max(6, len(perm) // 7)          # ~6 each for val and test
    return {"test": set(perm[:n_hold]),
            "val":  set(perm[n_hold:2 * n_hold]),
            "train": set(perm[2 * n_hold:])}


def prepare(size=256, n_train=1200, n_val=350, n_test=350, seed=0):
    """Tile a geographic (by-orthophoto) split and cache it.

    Crops 512 px tiles from the orthophotos and resizes to `size` (bilinear for
    imagery, nearest for masks so labels stay integers). The cache directory
    encodes every parameter, so changing size / budgets / seed forces a rebuild
    rather than silently reusing a stale cache.
    """
    download()
    cache = os.path.join(ROOT, f"tiles{size}_{n_train}-{n_val}-{n_test}_s{seed}")
    done_flag = os.path.join(cache, ".done")
    if os.path.exists(done_flag):
        return cache

    inv = _all_tiles_by_orthophoto()
    groups = orthophoto_split(seed)
    rng = np.random.default_rng(seed)
    budgets = {"train": n_train, "val": n_val, "test": n_test}
    for split, budget in budgets.items():
        names = [nm for o in groups[split] for nm in inv[o]]
        names = list(rng.permutation(names))[:budget]
        out = os.path.join(cache, split)
        os.makedirs(out, exist_ok=True)
        by_ortho = {}
        for nm in names:
            o, k = _ortho_and_k(nm)
            by_ortho.setdefault(o, []).append((k, nm))
        for o, items in sorted(by_ortho.items()):
            img = Image.open(os.path.join(IMG_DIR, f"{o}.tif")).convert("RGB")
            mask = Image.open(os.path.join(MASK_DIR, f"{o}.tif"))
            W = img.size[0]
            for k, nm in items:
                box = _crop_box(k, W)
                img.crop(box).resize((size, size), Image.BILINEAR).save(
                    os.path.join(out, f"{nm}.png"))
                mask.crop(box).resize((size, size), Image.NEAREST).save(
                    os.path.join(out, f"{nm}_m.png"))
        print(f"  [{split}] {len(groups[split])} orthophotos -> {len(names)} tiles")
    open(done_flag, "w").close()
    return cache


def class_pixel_fractions(cache, split="train"):
    """Per-class pixel share over a split (for the imbalance discussion)."""
    out = os.path.join(cache, split)
    counts = np.zeros(N_CLASSES, dtype=np.int64)
    for f in os.listdir(out):
        if f.endswith("_m.png"):
            m = np.asarray(Image.open(os.path.join(out, f)))
            b = np.bincount(m.ravel(), minlength=N_CLASSES)
            counts += b[:N_CLASSES]
    return counts / counts.sum()


try:
    import torch
    from torch.utils.data import Dataset

    class LandCoverTiles(Dataset):
        """Serves (image CHW float32, mask HW int64) tile pairs.

        Light geometric augmentation (flips + 90° rotations) on the train split
        only — colour is left alone since spectral values are the signal.
        """

        def __init__(self, cache, split, augment=False):
            self.dir = os.path.join(cache, split)
            self.ids = sorted(f[:-4] for f in os.listdir(self.dir)
                              if f.endswith(".png") and not f.endswith("_m.png"))
            self.augment = augment

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, i):
            stem = self.ids[i]
            img = np.asarray(Image.open(os.path.join(self.dir, f"{stem}.png")),
                             dtype=np.float32) / 255.0
            mask = np.asarray(Image.open(os.path.join(self.dir, f"{stem}_m.png")),
                              dtype=np.int64)
            if self.augment:
                if np.random.rand() < 0.5:
                    img, mask = img[:, ::-1], mask[:, ::-1]
                if np.random.rand() < 0.5:
                    img, mask = img[::-1], mask[::-1]
                r = np.random.randint(4)
                if r:
                    img, mask = np.rot90(img, r), np.rot90(mask, r)
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            mask = np.ascontiguousarray(mask)
            return torch.from_numpy(img), torch.from_numpy(mask)

except ImportError:  # torch not installed — module still usable for prep
    pass
