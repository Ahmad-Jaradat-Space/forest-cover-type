"""From-scratch numpy implementations, plus a PyTorch cross-check.

Three models:
  * multinomial logistic regression (softmax) — numpy,
  * a one-hidden-layer ReLU net — numpy (manual forward/backward),
  * the same one-hidden-layer net in PyTorch (`TorchMLP`).

The numpy nets exist to prove the math is understood end to end; the PyTorch
version is the production-grade re-implementation we actually reach for, and
overlaying their loss curves is the gradient check. All use mini-batch SGD
with L2. A small `spatial_cv` helper runs grouped (leave-one-region-out)
cross-validation for any sklearn-style estimator.
"""

import numpy as np


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _one_hot(y, k):
    out = np.zeros((y.size, k), dtype=np.float32)
    out[np.arange(y.size), y] = 1.0
    return out


class SoftmaxRegression:
    """Multinomial logistic regression trained with mini-batch SGD."""

    def __init__(self, n_classes, lr=0.1, l2=1e-4, epochs=20, batch=512, seed=0):
        self.k = n_classes
        self.lr = lr
        self.l2 = l2
        self.epochs = epochs
        self.batch = batch
        self.rng = np.random.default_rng(seed)
        self.W = None
        self.b = None
        self.history = []

    def fit(self, X, y, X_val=None, y_val=None):
        n, d = X.shape
        self.W = self.rng.normal(0, 0.01, size=(d, self.k)).astype(np.float32)
        self.b = np.zeros(self.k, dtype=np.float32)
        Y = _one_hot(y, self.k)

        for epoch in range(self.epochs):
            idx = self.rng.permutation(n)
            for start in range(0, n, self.batch):
                b = idx[start:start + self.batch]
                Xb, Yb = X[b], Y[b]
                P = _softmax(Xb @ self.W + self.b)
                grad_W = Xb.T @ (P - Yb) / len(b) + self.l2 * self.W
                grad_b = (P - Yb).mean(axis=0)
                self.W -= self.lr * grad_W
                self.b -= self.lr * grad_b
            entry = {"epoch": epoch, "train_loss": self._loss(X, y)}
            if X_val is not None:
                entry["val_loss"] = self._loss(X_val, y_val)
                entry["val_acc"] = (self.predict(X_val) == y_val).mean()
            self.history.append(entry)
        return self

    def _loss(self, X, y):
        P = _softmax(X @ self.W + self.b)
        return -np.log(P[np.arange(len(y)), y] + 1e-12).mean()

    def predict_proba(self, X):
        return _softmax(X @ self.W + self.b)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


class SoftmaxNN:
    """One hidden layer, ReLU, softmax output, He init, mini-batch SGD with L2."""

    def __init__(self, n_classes, hidden=64, lr=0.05, l2=1e-4,
                 epochs=20, batch=512, seed=0):
        self.k = n_classes
        self.hidden = hidden
        self.lr = lr
        self.l2 = l2
        self.epochs = epochs
        self.batch = batch
        self.rng = np.random.default_rng(seed)
        self.history = []

    def fit(self, X, y, X_val=None, y_val=None):
        n, d = X.shape
        h = self.hidden
        # He init for ReLU
        self.W1 = self.rng.normal(0, np.sqrt(2.0 / d), size=(d, h)).astype(np.float32)
        self.b1 = np.zeros(h, dtype=np.float32)
        self.W2 = self.rng.normal(0, np.sqrt(2.0 / h), size=(h, self.k)).astype(np.float32)
        self.b2 = np.zeros(self.k, dtype=np.float32)
        Y = _one_hot(y, self.k)

        for epoch in range(self.epochs):
            idx = self.rng.permutation(n)
            for start in range(0, n, self.batch):
                b = idx[start:start + self.batch]
                Xb, Yb = X[b], Y[b]

                # forward
                Z1 = Xb @ self.W1 + self.b1
                A1 = np.maximum(Z1, 0)
                P = _softmax(A1 @ self.W2 + self.b2)

                # backward
                m = len(b)
                dZ2 = (P - Yb) / m
                dW2 = A1.T @ dZ2 + self.l2 * self.W2
                db2 = dZ2.sum(axis=0)
                dA1 = dZ2 @ self.W2.T
                dZ1 = dA1 * (Z1 > 0)
                dW1 = Xb.T @ dZ1 + self.l2 * self.W1
                db1 = dZ1.sum(axis=0)

                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2

            entry = {"epoch": epoch, "train_loss": self._loss(X, y)}
            if X_val is not None:
                entry["val_loss"] = self._loss(X_val, y_val)
                entry["val_acc"] = (self.predict(X_val) == y_val).mean()
            self.history.append(entry)
        return self

    def _forward(self, X):
        A1 = np.maximum(X @ self.W1 + self.b1, 0)
        return _softmax(A1 @ self.W2 + self.b2)

    def _loss(self, X, y):
        P = self._forward(X)
        return -np.log(P[np.arange(len(y)), y] + 1e-12).mean()

    def predict_proba(self, X):
        return self._forward(X)

    def predict(self, X):
        return self._forward(X).argmax(axis=1)


# ----------------------------------------------------------------------
# PyTorch re-implementation of the one-hidden-layer net
# ----------------------------------------------------------------------
class TorchMLP:
    """The same architecture as `SoftmaxNN`, written in PyTorch.

    Kept deliberately faithful — He-initialised Linear → ReLU → Linear,
    plain SGD with L2 (``weight_decay``), mini-batches — so that its loss
    curve sits on top of the from-scratch numpy net's. That overlay is the
    gradient check: if my hand-derived backward pass were wrong, the two
    curves would diverge. Exposes the same sklearn-flavoured surface
    (``fit`` / ``predict`` / ``predict_proba`` / ``history``) as the numpy
    models so the notebook can treat all three interchangeably.

    Runs on Apple-Silicon MPS or CUDA when available, otherwise CPU.
    """

    def __init__(self, n_classes, hidden=128, lr=0.05, l2=1e-4,
                 epochs=25, batch=512, seed=0, device=None):
        self.k = n_classes
        self.hidden = hidden
        self.lr = lr
        self.l2 = l2
        self.epochs = epochs
        self.batch = batch
        self.seed = seed
        self.device = device
        self.history = []
        self.net = None

    def _resolve_device(self):
        import torch
        if self.device is not None:
            return torch.device(self.device)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def fit(self, X, y, X_val=None, y_val=None):
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        dev = self._resolve_device()
        d = X.shape[1]

        self.net = nn.Sequential(
            nn.Linear(d, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.k),
        ).to(dev)
        # He / Kaiming init for the ReLU layer, matching the numpy net.
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

        opt = torch.optim.SGD(self.net.parameters(), lr=self.lr,
                              weight_decay=self.l2)
        loss_fn = nn.CrossEntropyLoss()

        Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
        yt = torch.as_tensor(y, dtype=torch.long, device=dev)
        if X_val is not None:
            Xvt = torch.as_tensor(X_val, dtype=torch.float32, device=dev)
            yvt = torch.as_tensor(y_val, dtype=torch.long, device=dev)

        n = len(Xt)
        gen = torch.Generator().manual_seed(self.seed)
        for epoch in range(self.epochs):
            self.net.train()
            perm = torch.randperm(n, generator=gen).to(dev)
            for start in range(0, n, self.batch):
                idx = perm[start:start + self.batch]
                opt.zero_grad()
                loss = loss_fn(self.net(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()

            self.net.eval()
            with torch.no_grad():
                entry = {"epoch": epoch,
                         "train_loss": float(loss_fn(self.net(Xt), yt))}
                if X_val is not None:
                    logits_v = self.net(Xvt)
                    entry["val_loss"] = float(loss_fn(logits_v, yvt))
                    entry["val_acc"] = float(
                        (logits_v.argmax(1) == yvt).float().mean())
                self.history.append(entry)
        return self

    def predict_proba(self, X):
        import torch
        dev = next(self.net.parameters()).device
        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.as_tensor(X, dtype=torch.float32, device=dev))
            return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


# ----------------------------------------------------------------------
# Grouped (spatial) cross-validation
# ----------------------------------------------------------------------
def spatial_cv(make_estimator, X, y, groups, scorers):
    """Leave-one-group-out cross-validation.

    For each unique value of ``groups`` we train on every *other* group and
    score on the held-out one — so no row in a test fold has a spatial
    neighbour leaking in from the training fold. ``make_estimator`` is a
    zero-arg factory returning a fresh, unfitted estimator; ``scorers`` is a
    ``{name: fn(y_true, y_pred)}`` dict. Returns a list of per-fold result
    dicts (one per held-out group).
    """
    rows = []
    for g in np.unique(groups):
        test = groups == g
        train = ~test
        est = make_estimator()
        est.fit(X[train], y[train])
        pred = est.predict(X[test])
        row = {"held_out_group": int(g), "n_test": int(test.sum())}
        for name, fn in scorers.items():
            row[name] = float(fn(y[test], pred))
        rows.append(row)
    return rows
