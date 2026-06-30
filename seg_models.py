"""A U-Net for land-cover segmentation, written in PyTorch.

The encoder is a torchvision ResNet-34 pretrained on ImageNet — transfer
learning is the right call on a few thousand tiles, and it lets the network
borrow low-level edge/texture filters instead of learning them from scratch.
The decoder is hand-built U-Net: bilinear upsample, concatenate the matching
encoder skip, two conv-BN-ReLU blocks. Encoder parameters train at 1/10th the
decoder learning rate so the pretrained filters are nudged, not clobbered.

Everything here is deliberately explicit (no segmentation library) so the
architecture and the metrics are auditable end to end.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def resolve_device(device=None):
    if device is not None:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class _DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetResNet34(nn.Module):
    """ResNet-34 encoder (ImageNet) + U-Net decoder."""

    def __init__(self, n_classes, pretrained=True):
        super().__init__()
        from torchvision.models import resnet34, ResNet34_Weights
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        m = resnet34(weights=weights)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu)  # 64,  H/2
        self.pool = m.maxpool                              #      H/4
        self.layer1, self.layer2 = m.layer1, m.layer2      # 64,128
        self.layer3, self.layer4 = m.layer3, m.layer4      # 256,512
        self.dec4 = _DecoderBlock(512, 256, 256)           # -> H/16
        self.dec3 = _DecoderBlock(256, 128, 128)           # -> H/8
        self.dec2 = _DecoderBlock(128, 64, 64)             # -> H/4
        self.dec1 = _DecoderBlock(64, 64, 64)              # -> H/2
        self.dec0 = _DecoderBlock(64, 0, 32)               # -> H
        self.head = nn.Conv2d(32, n_classes, 1)

    def encoder_parameters(self):
        for name in ("stem", "layer1", "layer2", "layer3", "layer4"):
            yield from getattr(self, name).parameters()

    def decoder_parameters(self):
        for name in ("dec4", "dec3", "dec2", "dec1", "dec0", "head"):
            yield from getattr(self, name).parameters()

    def forward(self, x):
        e0 = self.stem(x)            # 64,  H/2
        e1 = self.layer1(self.pool(e0))  # 64,  H/4
        e2 = self.layer2(e1)         # 128, H/8
        e3 = self.layer3(e2)         # 256, H/16
        e4 = self.layer4(e3)         # 512, H/32
        d = self.dec4(e4, e3)
        d = self.dec3(d, e2)
        d = self.dec2(d, e1)
        d = self.dec1(d, e0)
        d = self.dec0(d)
        return self.head(d)


# ----------------------------------------------------------------------
# Pixel-confusion metrics
# ----------------------------------------------------------------------
def confusion(n_classes):
    return np.zeros((n_classes, n_classes), dtype=np.int64)


def accumulate(conf, true, pred):
    n = conf.shape[0]
    k = n * true.reshape(-1) + pred.reshape(-1)
    conf += np.bincount(k, minlength=n * n).reshape(n, n)
    return conf


def metrics_from_confusion(conf):
    tp = np.diag(conf).astype(np.float64)
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    iou = tp / np.maximum(tp + fp + fn, 1)
    dice = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    recall = tp / np.maximum(tp + fn, 1)
    precision = tp / np.maximum(tp + fp, 1)
    freq = conf.sum(1) / conf.sum()
    return {
        "per_class_iou": iou,
        "per_class_dice": dice,
        "per_class_recall": recall,
        "per_class_precision": precision,
        "miou": float(np.nanmean(iou)),
        "pixel_acc": float(tp.sum() / conf.sum()),
        "fw_iou": float((freq * iou).sum()),
        "support_frac": freq,
    }


# ----------------------------------------------------------------------
# Train / evaluate
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, n_classes, device):
    model.eval()
    conf = confusion(n_classes)
    for xb, yb in loader:
        logits = model(xb.to(device))
        pred = logits.argmax(1).cpu().numpy()
        accumulate(conf, yb.numpy(), pred)
    return conf


def train_segmenter(model, train_loader, val_loader, n_classes, device,
                    epochs=20, lr=1e-3, class_weights=None, seed=0):
    """AdamW with a discriminative LR (encoder = lr/10), CE loss optionally
    weighted toward rare classes. Tracks train loss and val mIoU per epoch."""
    torch.manual_seed(seed)
    model.to(device)
    opt = torch.optim.AdamW([
        {"params": model.encoder_parameters(), "lr": lr * 0.1},
        {"params": model.decoder_parameters(), "lr": lr},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    w = None if class_weights is None else torch.as_tensor(
        class_weights, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    history = []
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            running += float(loss) * len(xb)
        sched.step()
        train_loss = running / len(train_loader.dataset)
        val = metrics_from_confusion(
            evaluate(model, val_loader, n_classes, device))
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_miou": val["miou"], "val_pixel_acc": val["pixel_acc"]})
        print(f"epoch {epoch:2d}  train_loss {train_loss:.3f}  "
              f"val mIoU {val['miou']:.3f}  val pixacc {val['pixel_acc']:.3f}")
    return history
