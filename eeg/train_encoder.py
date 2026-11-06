"""Train the source-DEV-selected SEED encoder used by A_scale."""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .run_ascale import WindowEncoder, read_partition


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit", required=True)
    p.add_argument("--dev", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--classes", type=int, default=3)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    fit, dev = read_partition(a.fit), read_partition(a.dev)
    mu = fit["x"].mean((0, 2), dtype=np.float64).astype(np.float32)
    sd = fit["x"].std((0, 2), dtype=np.float64).astype(np.float32).clip(1e-6)
    xf = (fit["x"] - mu[None, :, None, :]) / sd[None, :, None, :]
    xd = (dev["x"] - mu[None, :, None, :]) / sd[None, :, None, :]
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(2)
    model = WindowEncoder(a.classes).to(a.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    loader = DataLoader(TensorDataset(torch.from_numpy(xf), torch.from_numpy(fit["y"])),
                        batch_size=128, shuffle=True,
                        generator=torch.Generator().manual_seed(a.seed), num_workers=0)
    best, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, 41):
        model.train()
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            z = model(x.to(a.device))
            loss = torch.nn.functional.cross_entropy(z, y.to(a.device))
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        predictions = []
        with torch.no_grad():
            for i in range(0, len(xd), 256):
                x = torch.from_numpy(xd[i:i + 256]).to(a.device)
                predictions.append(model(x).argmax(1).cpu().numpy())
        accuracy = float(np.mean(np.concatenate(predictions) == dev["y"]))
        if accuracy > best:
            best, best_epoch = accuracy, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        schedule.step()
        if epoch - best_epoch >= 8:
            break
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(classes=a.classes, state=best_state, normalization=(mu, sd),
                    selected_epoch=best_epoch, source_dev_accuracy=best, seed=a.seed), out)


if __name__ == "__main__":
    main()
