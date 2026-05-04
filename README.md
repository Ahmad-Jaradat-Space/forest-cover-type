# Forest Cover Type

I just finished Andrew Ng's Machine Learning Specialization on Coursera and wanted a place to actually use what I learned, end to end, on real data instead of the toy assignments. This is the first of two capstones I built for that.

The dataset is UCI's Covertype: 581k rows of cartographic and soil variables describing 30x30m patches of US forest, each labelled with one of seven dominant tree species. The question I'm chasing is not "what's the best model on Covertype" — it's how much of forest cover type is solvable from cartography alone, and where does that approach hit its ceiling.

## How the notebook is laid out

The notebook reads as a short paper with five sections, each carrying the actual narrative beats inside:

1. **Introduction** — the problem and why anyone would care.
2. **Data** — class balance, feature distributions, correlations, and the explicit modelling hypothesis these observations support.
3. **Methods** — three model families fit in increasing capacity: logistic regression (from scratch in numpy and with sklearn), a one-hidden-layer neural net (from scratch and re-implemented in Keras as a cross-check), then tree ensembles (decision tree, random forest, histogram gradient boosting).
4. **Results** — learning curves and bias/variance diagnosis, confusion structure, unsupervised cross-checks (PCA, K-means), and the final test-set comparison.
5. **Conclusion** — the answer to the original question and where this would and wouldn't deploy.

Every plot is read out loud: a one-line setup before the cell, a finding-style title on the figure itself, and a 2–4 sentence takeaway after — *what to look at, what it means, what it indicates next.*

## Running it

Tested on macOS with Python 3.12.

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

The first cell of the notebook downloads the data (~11 MB gzipped) into `data/` if it isn't there yet. The `data/` folder is gitignored. The `notebook.ipynb` in this repo is already executed, so GitHub renders all outputs and plots inline — you can read it through without running anything.

## What's where

- `notebook.ipynb` — the whole story, runs top to bottom
- `data.py` — download, load, stratified split, scaling
- `models.py` — the from-scratch softmax regression and one-hidden-layer NN
- `plots.py` — small matplotlib helpers used by the notebook

## Notes

- The from-scratch logistic regression and NN are trained on a 60k stratified subsample, mostly so iteration stays cheap; the sklearn and Keras versions use the full training set. Their accuracies still line up, which is the point.
- I dropped XGBoost in favour of sklearn's `HistGradientBoostingClassifier` so the repo installs cleanly without a system-level OpenMP dependency.
