"""Evaluate a frozen EEG comparator on an input NPZ."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .external_baselines.models import build_model
from .external_baselines.masatcn_model import build_masatcn
from .train import view


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="NPZ containing x [N, 62, 10, 5]")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    with np.load(a.input, allow_pickle=False) as f:
        x = f["x"]
    if x.ndim != 4 or x.shape[1:] != (62, 10, 5):
        raise ValueError("input must have shape [N, 62, 10, 5]")
    checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    method = checkpoint["method"]
    model = (build_masatcn(channels=62, bands=5, classes=checkpoint["classes"])
             if method == "masatcn"
             else build_model(method, channels=62, bands=5, num_classes=checkpoint["classes"],
                              functional_adjacency=checkpoint["adjacency"]))
    model.load_state_dict(checkpoint["state"], strict=True)
    model.to(a.device).eval()
    mu, sd = checkpoint["normalization"]
    x = (x - mu[None, :, None, :]) / sd[None, :, None, :]
    logits = []
    with torch.no_grad():
        for i in range(0, len(x), 256):
            batch = torch.as_tensor(x[i:i + 256], dtype=torch.float32, device=a.device)
            logits.append(model(view(batch, method)).cpu().numpy())
    scores = np.concatenate(logits)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, scores=scores, prediction=scores.argmax(1))


if __name__ == "__main__":
    main()
