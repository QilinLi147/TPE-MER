"""Prototype projection ("push") onto real TRAIN segments.

Each prototype k (dimension d, class c) is replaced by the embedding of the
nearest TRAIN segment whose hard label in dimension d equals c. Distances are
measured in the unit-sphere embedding space that the cluster/separation losses
shape. Afterwards the bank stores the segment's raw input and identity, and the
prototype is recomputed as E(x_src) at every forward: it *is* a real segment.
Distinct-song constraint: no two prototypes of the same dimension may come from
the same song, which spreads evidence over songs (the old bank also used one
row per song).
"""
from __future__ import annotations

from typing import Dict, Iterable

import torch

from .model import GFDPModel


@torch.no_grad()
def push_prototypes(model: GFDPModel, batches: Iterable[Dict], device="cpu") -> Dict:
    model.eval()
    bank = model.prototypes
    target = bank.embed(model.encoder).detach()             # current vectors (free mode)
    k = bank.size
    best = [[] for _ in range(k)]
    for batch in batches:
        x = batch["x"].to(device)
        z = model.encoder(x)
        dist = (2 - 2 * z @ target.t()).clamp_min(0).cpu()   # [n, K]
        hard = (batch["y"] > 0).long().cpu()
        for j in range(k):
            d = int(bank.proto_dim[j]); c = int(bank.proto_class[j])
            ok = hard[:, d] == c
            if not bool(ok.any()):
                continue
            dj = dist[:, j].masked_fill(~ok, float("inf"))
            top = torch.topk(-dj, min(32, int(ok.sum()))).indices
            for r in top.tolist():
                best[j].append((float(dj[r]), x[r].cpu().clone(), int(batch["songs"][r]),
                                int(batch["rows"][r]), float(batch["times"][r])))
            # bounded memory: keep the 64 nearest candidates seen so far
            seen, kept = set(), []
            for b in sorted(best[j], key=lambda b: b[0]):   # nearest row per song
                if b[2] not in seen:
                    seen.add(b[2]); kept.append(b)
            best[j] = kept[:64]
    chosen_x, chosen_song, chosen_row, chosen_time, chosen_dist = [], [], [], [], []
    used_songs = {0: set(), 1: set()}
    order = sorted(range(k), key=lambda j: min([b[0] for b in best[j]] or [float("inf")]))
    picks = {}
    for j in order:
        d = int(bank.proto_dim[j])
        for dist_j, xr, song, row, t in sorted(best[j], key=lambda b: b[0]):
            if song not in used_songs[d]:
                used_songs[d].add(song)
                picks[j] = (dist_j, xr, song, row, t)
                break
        if j not in picks:
            raise RuntimeError(f"no admissible real TRAIN segment for prototype {j}")
    for j in range(k):
        dist_j, xr, song, row, t = picks[j]
        chosen_x.append(xr); chosen_song.append(song); chosen_row.append(row)
        chosen_time.append(t); chosen_dist.append(dist_j)
    bank.anchor(torch.stack(chosen_x).to(bank.source_x.device),
                torch.tensor(chosen_song, device=bank.source_song.device),
                torch.tensor(chosen_row, device=bank.source_row.device),
                torch.tensor(chosen_time, device=bank.source_time.device, dtype=bank.source_time.dtype))
    return {"push_distance": chosen_dist, "mean_push_distance": sum(chosen_dist) / k,
            "prototypes": bank.metadata()}
