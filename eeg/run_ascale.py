"""Fit the source-indexed SEED A_scale readout from FIT and source-DEV windows."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from gfdp.eeg_encoder import EEGEncoder
from . import ascale, geometry


class WindowEncoder(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.encoder = EEGEncoder(62, 25, family="attn", hidden=64, layers=1,
                                  dim=64, dropout=.25, electrode_dropout=.1,
                                  pooling="flatten", heads=4)
        self.encoder.standardised.fill_(True)
        self.encoder.centre_momentum = .05
        self.head = nn.Sequential(nn.Linear(128, 128), nn.GELU(),
                                  nn.Dropout(.25), nn.Linear(128, classes))

    def embed(self, x):
        views = torch.cat([x[:, :, :5].flatten(2), x[:, :, 5:].flatten(2)], 1)
        return self.encoder(views)

    def forward(self, x):
        z = self.embed(x)
        return self.head(torch.cat([z, z], 1))


def embeddings(model, x, device, batch_size=256):
    rows = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            v = torch.as_tensor(x[i:i + batch_size], dtype=torch.float32, device=device)
            rows.append(model.embed(v).cpu().numpy())
    return np.concatenate(rows)


def read_partition(path):
    with np.load(path, allow_pickle=False) as f:
        data = {key: f[key] for key in f.files}
    required = {"x", "y", "subject", "session", "trial", "original_index"}
    if not required.issubset(data):
        raise ValueError("partition is missing required arrays")
    if data["x"].ndim != 4 or data["x"].shape[1:] != (62, 10, 5):
        raise ValueError("EEG input must have shape [N, 62, 10, 5]")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", required=True, help="FIT partition NPZ path")
    parser.add_argument("--dev", required=True, help="source-DEV partition NPZ path")
    parser.add_argument("--encoder", required=True, help="selected encoder checkpoint path")
    parser.add_argument("--out", required=True, help="new output NPZ path")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    target = Path(args.out)
    if target.exists():
        raise FileExistsError(target)
    fit, dev = read_partition(args.fit), read_partition(args.dev)
    # Statistics are fitted on FIT subjects and time samples only.
    mu = fit["x"].mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
    sd = fit["x"].std(axis=(0, 2), dtype=np.float64).astype(np.float32).clip(1e-6)
    xf = (fit["x"] - mu[None, :, None, :]) / sd[None, :, None, :]
    xd = (dev["x"] - mu[None, :, None, :]) / sd[None, :, None, :]
    checkpoint = torch.load(args.encoder, map_location="cpu", weights_only=False)
    if not all(np.array_equal(a, b) for a, b in zip((mu, sd), checkpoint["normalization"])):
        raise ValueError("checkpoint and FIT normalisation do not match")
    model = WindowEncoder(int(checkpoint["classes"]))
    model.load_state_dict(checkpoint["state"], strict=True)
    model.to(args.device)
    zf, zd = embeddings(model, xf, args.device), embeddings(model, xd, args.device)
    bank = ascale.bank(zf, xf, fit["y"], fit, checkpoint["classes"])
    geometry.provenance(bank, fit)
    state, _ = geometry.scale_state(zf, xf, bank)
    ff = geometry.features("A_scale", zf, xf, bank, state)
    fd = geometry.features("A_scale", zd, xd, bank, state)
    solution = ascale.fit_head(ff, fit["y"], fd, dev["y"])
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, **{**bank, **state, "W": solution["W"], "b": solution["b"],
                        "C": np.array(solution["C"]), "normalization_mu": mu,
                        "normalization_sd": sd})


if __name__ == "__main__":
    main()
