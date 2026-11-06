"""Train an EEG comparator using separate FIT and source-DEV partitions."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from eeg.run_ascale import read_partition
from .external_baselines.models import build_model
from .external_baselines.masatcn_model import build_masatcn
from .optimisation import build_optimisation, loss_function

HERE = Path(__file__).resolve().parent
METHODS = ("bf_gcn_de", "dgclink", "dgcnn", "emt", "hdgnet", "masatcn", "pgcn", "rgcb")


def graph(x):
    values = x.transpose(0, 2, 1, 3).reshape(-1, 62, 5)
    result = np.empty((62, 62, 5), dtype=np.float32)
    for band in range(5):
        r = np.corrcoef(values[:, :, band], rowvar=False)
        r = np.clip(np.abs(np.nan_to_num(r, nan=0., posinf=0., neginf=0.)), 0, 1)
        np.fill_diagonal(r, 1)
        result[:, :, band] = r
    return result


def view(x, method):
    return x.transpose(1, 2) if method in ("emt", "masatcn") else x.mean(2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=METHODS, required=True)
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
    config = json.loads((HERE / "training_defaults.json").read_text())[a.method]
    config["seed"] = a.seed
    adjacency = graph(fit["x"]) if a.method == "bf_gcn_de" else None
    model = (build_masatcn(channels=62, bands=5, classes=a.classes)
             if a.method == "masatcn"
             else build_model(a.method, channels=62, bands=5, num_classes=a.classes,
                              functional_adjacency=adjacency)).to(a.device)
    if a.method == "masatcn":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                     weight_decay=config["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=config["scheduler_factor"],
            patience=config["scheduler_patience"], min_lr=config["minimum_learning_rate"])
        warmup, criterion = None, torch.nn.functional.cross_entropy
        patience = config["early_stopping_patience"]
    else:
        options = build_optimisation(a.method, model, config)
        optimizer, scheduler, warmup = options.optimizer, options.scheduler, options.warmup
        criterion = loss_function(a.method, config)
        patience = config["patience"]
    loader = DataLoader(TensorDataset(torch.from_numpy(xf), torch.from_numpy(fit["y"])),
                        batch_size=config["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(a.seed), num_workers=0)
    best, best_epoch, best_state, repeat, previous = -1., 0, None, 0, None
    for epoch in range(1, config["max_epochs"] + 1):
        if a.method == "masatcn" and best_state is not None:
            model.load_state_dict(best_state, strict=True)
        model.train()
        correct = total = 0
        for step, (x, y) in enumerate(loader):
            if len(y) == 1 and a.method != "masatcn":
                x, y = torch.cat([x, x]), torch.cat([y, y])
            x, y = x.to(a.device), y.to(a.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(view(x, a.method))
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite comparator loss")
            loss.backward()
            optimizer.step()
            if warmup and step < len(loader) - 1:
                with warmup.dampening():
                    pass
            correct += int((logits.argmax(1) == y).sum())
            total += len(y)
        model.eval()
        outputs = []
        with torch.no_grad():
            for i in range(0, len(xd), 256):
                x = torch.from_numpy(xd[i:i + 256]).to(a.device)
                outputs.append(model(view(x, a.method)).cpu())
        logits = torch.cat(outputs)
        labels = torch.from_numpy(dev["y"])
        accuracy = float((logits.argmax(1) == labels).float().mean())
        if accuracy > best:
            best, best_epoch = accuracy, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if scheduler:
            if a.method == "masatcn":
                scheduler.step(float(torch.nn.functional.cross_entropy(logits, labels)))
            elif warmup:
                with warmup.dampening():
                    scheduler.step()
            else:
                scheduler.step()
        if a.method == "hdgnet":
            if correct / total > .9999:
                repeat = repeat + 1 if previous == accuracy else 1
                previous = accuracy
            stop = repeat > 10 or accuracy > .9999
        else:
            stop = epoch - best_epoch >= patience
        if stop:
            break
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(classes=a.classes, state=best_state, normalization=(mu, sd),
                    adjacency=adjacency, selected_epoch=best_epoch, method=a.method,
                    source_dev_accuracy=best, seed=a.seed), out)


if __name__ == "__main__":
    main()
