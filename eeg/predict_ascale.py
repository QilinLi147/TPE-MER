"""Apply a frozen SEED A_scale readout to previously unseen windows."""
import argparse
from pathlib import Path

import numpy as np
import torch

from . import ascale, geometry
from .run_ascale import WindowEncoder, embeddings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="NPZ with x [N, 62, 10, 5]")
    parser.add_argument("--encoder", required=True, help="selected encoder checkpoint")
    parser.add_argument("--readout", required=True, help="frozen A_scale readout")
    parser.add_argument("--out", required=True, help="new prediction NPZ path")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(output)
    with np.load(args.input, allow_pickle=False) as f:
        x = f["x"]
    if x.ndim != 4 or x.shape[1:] != (62, 10, 5):
        raise ValueError("input must have shape [N, 62, 10, 5]")
    with np.load(args.readout, allow_pickle=False) as f:
        saved = {k: f[k] for k in f.files}
    checkpoint = torch.load(args.encoder, map_location="cpu", weights_only=False)
    for a, b in zip((saved["normalization_mu"], saved["normalization_sd"]),
                    checkpoint["normalization"]):
        if not np.array_equal(a, b):
            raise ValueError("encoder and readout normalisation do not match")
    model = WindowEncoder(int(checkpoint["classes"]))
    model.load_state_dict(checkpoint["state"], strict=True)
    model.to(args.device)
    x = (x - saved["normalization_mu"][None, :, None, :]) / saved["normalization_sd"][None, :, None, :]
    z = embeddings(model, x, args.device)
    features = geometry.features("A_scale", z, x, saved, {"tau": saved["tau"]})
    logits = ascale.logits(features, saved)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, logits=logits, prediction=logits.argmax(axis=1))


if __name__ == "__main__":
    main()
