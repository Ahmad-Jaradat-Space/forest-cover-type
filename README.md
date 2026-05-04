# Forest Cover Type

I just finished Andrew Ng's Machine Learning Specialization on Coursera and wanted a place to actually use what I learned, end to end, on real data instead of the toy assignments. This is the first of two capstones I built for that.

The dataset is UCI's Covertype: 581k rows of cartographic and soil variables describing 30x30m patches of US forest, each labelled with one of seven dominant tree species. The task is to predict the cover type from the features.

In one notebook I compare:

- logistic regression I wrote from scratch in numpy
- the same in sklearn
- a softmax neural net I wrote from scratch in numpy
- the same network in Keras
- decision tree, random forest, and histogram gradient boosting
- PCA and K-means on the same features, as a sanity check on what unsupervised methods see in the data

## Running it

Tested on macOS with Python 3.12.

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

The first cell of the notebook downloads the data (~11 MB gzipped) into `data/` if it isn't there yet. The `data/` folder is gitignored.

## What's where

- `notebook.ipynb` — the whole story, runs top to bottom
- `data.py` — download, load, stratified split, scaling
- `models.py` — the from-scratch softmax regression and one-hidden-layer NN
- `plots.py` — small matplotlib helpers used by the notebook

## Notes

- The from-scratch logistic regression and NN are trained on a 60k stratified subsample, mostly so iteration stays cheap; the sklearn and Keras versions use the full training set. Their accuracies still line up, which is the point.
- I dropped XGBoost in favour of sklearn's `HistGradientBoostingClassifier` so the repo installs cleanly without a system-level OpenMP dependency.
