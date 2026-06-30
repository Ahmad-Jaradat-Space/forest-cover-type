"""Load UCI Covertype.

The dataset is one CSV-like file with 581012 rows and 55 columns:
10 continuous, 4 binary wilderness areas, 40 binary soil types, then
the cover type label (1..7). No header row.
"""

import gzip
import os
import shutil
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "covtype.data.gz")
CSV = os.path.join(HERE, "data", "covtype.csv")

CONTINUOUS = [
    "elevation", "aspect", "slope",
    "h_dist_hydro", "v_dist_hydro", "h_dist_road",
    "hillshade_9am", "hillshade_noon", "hillshade_3pm",
    "h_dist_fire",
]
WILDERNESS = [f"wilderness_{i}" for i in range(1, 5)]
SOIL = [f"soil_{i}" for i in range(1, 41)]
COLUMNS = CONTINUOUS + WILDERNESS + SOIL + ["cover_type"]


def download():
    os.makedirs(os.path.dirname(RAW), exist_ok=True)
    if not os.path.exists(CSV):
        if not os.path.exists(RAW):
            print(f"downloading {URL}")
            urllib.request.urlretrieve(URL, RAW)
        print("decompressing")
        with gzip.open(RAW, "rb") as src, open(CSV, "wb") as dst:
            shutil.copyfileobj(src, dst)


def load(seed=0, subsample=None):
    """Return train/val/test splits, stratified, with continuous columns scaled.

    Labels are remapped to 0..6 (sklearn-friendly).
    If `subsample` is set, take a stratified subsample first — useful when
    iterating on the from-scratch implementations.
    """
    download()
    df = pd.read_csv(CSV, header=None, names=COLUMNS)

    if subsample is not None:
        # stratified subsample that preserves class proportions
        df = df.groupby("cover_type", group_keys=False).sample(
            frac=subsample / len(df), random_state=seed
        )

    X = df.drop(columns="cover_type").values.astype(np.float32)
    y = (df["cover_type"].values - 1).astype(np.int64)

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.15 / 0.85, stratify=y_tv, random_state=seed
    )

    # scale only the first 10 columns (the continuous ones); binary indicators
    # are already on the right scale
    scaler = StandardScaler().fit(X_train[:, :10])
    for arr in (X_train, X_val, X_test):
        arr[:, :10] = scaler.transform(arr[:, :10])

    return X_train, y_train, X_val, y_val, X_test, y_test


CLASS_NAMES = [
    "Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine",
    "Cottonwood/Willow", "Aspen", "Douglas-fir", "Krummholz",
]

# The four Roosevelt National Forest wilderness areas the survey covers.
# Each is a contiguous block of terrain, so they double as spatial folds:
# holding one out tests generalisation to an unseen region rather than to
# unseen-but-neighbouring patches.
WILDERNESS_NAMES = [
    "Rawah", "Neota", "Comanche Peak", "Cache la Poudre",
]


def wilderness_group(X):
    """Return the wilderness-area index (0..3) for each row.

    Columns 10:14 of the design matrix are the four one-hot wilderness
    indicators; their argmax is the area label. Survives standardisation
    because only the first 10 (continuous) columns are scaled.
    """
    return X[:, 10:14].argmax(axis=1)
