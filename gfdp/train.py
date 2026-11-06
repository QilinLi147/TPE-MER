"""Training schedule for GFDP-MER.

Stage A  warm-up     encoder + similarity heads + free prototypes; readout fixed
Stage B  joint       all parameters; cosine LR with restarts (Snapshot Ensembles,
                     Huang et al. ICLR 2017); ISDA ramp; stochastic depth;
                     optional EfficientTrain-style low-pass curriculum
Stage C  push        prototypes -> nearest real TRAIN segments (distinct songs)
Stage D  last layer  encoder and prototypes frozen; readout only (convex) with
                     ISDA + wrong-class L1; optional self-distillation from the
                     snapshot ensemble (Efficient KD from checkpoints, NeurIPS 2022)

Model selection uses ONLY dev songs carved from the training side of a fold.
"""
from __future__ import annotations

import copy
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch.nn import functional as F

from .data import Corpus, iterate
from .losses import (ClassCovariance, distillation, embedding_variance, isda_bce, prototype_diversity,
                     prototype_geometry, soft_targets, wrong_class_l1)
from .certify import sample_heard, song_bounds_torch
from .metrics import all_metrics, task_metrics
from .model import GFDPModel, ModelConfig
from .push import push_prototypes


@dataclass
class TrainConfig:
    lr: float = 1e-3
    lr_readout: float = 5e-3
    weight_decay: float = 0.01
    song_batch: int = 4
    epochs_warm: int = 3
    epochs_joint: int = 24
    snapshot_cycles: int = 3
    patience: int = 8
    epochs_last: int = 10
    label_sigma: float = 0.1
    isda_strength: float = 0.5
    w_cluster: float = 0.4
    w_separation: float = 0.08
    w_diversity: float = 0.1
    w_l1: float = 1e-4
    w_variance: float = 1.0
    push_after_epochs: int = 0            # joint epochs before anchoring; 0 -> end of first cycle
    cert_weight: float = 0.0              # v1.2: certification-aware loss (0 = v1.1 behaviour)
    cert_frac_min: float = 0.2            # heard fraction sampled per song for the bound
    cert_frac_max: float = 0.6
    spec_freq_mask: int = 12
    spec_time_mask: int = 6
    curriculum_fraction: float = 0.0      # 0 disables the low-pass curriculum
    curriculum_initial_pool: int = 4
    dev_fraction: float = 0.1
    select_on: str = "mean"               # mean | valence | arousal
    self_distill: bool = False
    distill_weight: float = 0.5
    grad_clip: float = 1.0
    max_batches_per_epoch: Optional[int] = None   # smoke tests only

    @classmethod
    def from_dict(cls, d: Dict) -> "TrainConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def spec_augment(x: torch.Tensor, freq: int, time_: int, generator: torch.Generator) -> torch.Tensor:
    if freq <= 0 and time_ <= 0:
        return x
    x = x.clone()
    n, bins, frames = x.shape
    for i in range(n):
        if freq > 0:
            w = int(torch.randint(0, freq + 1, (1,), generator=generator))
            f0 = int(torch.randint(0, max(1, bins - w), (1,), generator=generator))
            x[i, f0:f0 + w] = 0
        if time_ > 0:
            w = int(torch.randint(0, time_ + 1, (1,), generator=generator))
            t0 = int(torch.randint(0, max(1, frames - w), (1,), generator=generator))
            x[i, :, t0:t0 + w] = 0
    return x


def low_pass(x: torch.Tensor, pool: int) -> torch.Tensor:
    """EfficientTrain-style 'low-frequency first': blur time-frequency maps."""
    if pool <= 1:
        return x
    y = F.avg_pool2d(x[:, None], kernel_size=pool, stride=pool, ceil_mode=True)
    return F.interpolate(y, size=x.shape[1:], mode="nearest")[:, 0]


@torch.no_grad()
def predict(model: GFDPModel, corpus: Corpus, songs: List[Dict], device="cpu",
            song_batch: int = 8) -> Dict[str, np.ndarray]:
    model.eval()
    proto = model.prototypes.embed(model.encoder)
    logits, ys, ss, rows = [], [], [], []
    for b in iterate(corpus, songs, song_batch, shuffle=False, device=device):
        logits.append(model(b["x"], b["times"], b["songs"], b["song_slices"], proto_embedding=proto).cpu())
        ys.append(b["y"].cpu()); ss.append(b["songs"].cpu()); rows.append(b["rows"].cpu())
    return dict(logit=torch.cat(logits).numpy(), y=torch.cat(ys).numpy(),
                song=torch.cat(ss).numpy(), row=torch.cat(rows).numpy())


def selection_key(metrics: Dict, select_on: str, task: str = "classification"):
    dims = ("valence", "arousal") if select_on == "mean" else (select_on,)
    if task == "regression":      # v1.4: higher CCC first, then lower RMSE
        return (float(np.mean([metrics[d]["ccc"] for d in dims])),
                -float(np.mean([metrics[d]["rmse"] for d in dims])))
    keys = ("accuracy", "macro_f1", "roc_auc", "balanced_accuracy")
    return tuple(float(np.mean([metrics[d][k] for d in dims])) for k in keys)


def regression_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """v1.4 regression objective per dimension: MSE + (1 - CCC) over the batch rows
    (CCC is the usual MER regression criterion; MSE keeps the scale anchored)."""
    terms = []
    for d in range(2):
        p, t = pred[:, d], y[:, d]
        mp, mt = p.mean(), t.mean()
        vp, vt = p.var(unbiased=False), t.var(unbiased=False)
        cov = ((p - mp) * (t - mt)).mean()
        c = 2 * cov / (vp + vt + (mp - mt) ** 2 + 1e-8)
        terms.append(F.mse_loss(p, t) + (1 - c))
    return torch.stack(terms).mean()


class Trainer:
    def __init__(self, model_cfg: ModelConfig, train_cfg: TrainConfig, device="cpu",
                 log: Callable[[Dict], None] = print) -> None:
        self.mc, self.tc, self.device, self.log = model_cfg, train_cfg, device, log

    # ------------------------------------------------------------------ helpers
    def _covariances(self, model: GFDPModel) -> List[ClassCovariance]:
        h = len(self.mc.time_scales)
        m = 2 * self.mc.prototypes_per_class
        dim = h * m + (m if self.mc.glance else 0)
        return [ClassCovariance(dim, device=self.device) for _ in range(2)]

    def _loss(self, model, batch, covs, isda_lambda, teacher=None, geometry=True):
        tc = self.tc
        out = model(batch["x"], batch["times"], batch["songs"], batch["song_slices"], decompose=True)
        y = batch["y"]
        hard = (y > 0).long()
        if self.mc.task == "regression":
            # v1.4: the additive output is the V/A value; ISDA / soft labels are
            # classification-specific and not used. Prototype classes stay y > 0.
            cls = regression_loss(out["logit"], y)
            parts = {"reg": float(cls.detach())}
        else:
            target = soft_targets(y, tc.label_sigma)
            cls = 0.0
            for d in range(2):
                f = model.features(out, d)
                covs[d].update(f, hard[:, d])
                cls = cls + isda_bce(out["logit"][:, d], target[:, d], hard[:, d],
                                     model.readout_vector(d), covs[d], isda_lambda)
            cls = cls / 2
            parts = {"cls": float(cls.detach())}
        loss = cls
        if geometry:
            geo = prototype_geometry(out["distances"]["raw"], model.prototypes.proto_dim,
                                     model.prototypes.proto_class, hard, out["allowed"])
            div = prototype_diversity(model.prototypes.embed(model.encoder), model.prototypes.proto_dim)
            loss = loss + tc.w_cluster * geo["cluster"] + tc.w_separation * geo["separation"] \
                + tc.w_diversity * div
            parts.update(cluster=float(geo["cluster"].detach()), separation=float(geo["separation"].detach()),
                         diversity=float(div.detach()))
        if geometry and tc.w_variance > 0:
            var = embedding_variance(out["embedding"])
            loss = loss + tc.w_variance * var
            parts["variance"] = float(var.detach())
        if tc.cert_weight > 0 and bool(model.prototypes.anchored):
            # applied only once prototypes are real TRAIN segments (after push)
            # v1.2 certification-aware training (IBP-style): the WORST-CASE full-song
            # logit, given only a random heard subset, should already be on the
            # song-majority side. Makes early certified decisions reachable.
            cert_terms = []
            for i, sl in enumerate(batch["song_slices"]):
                n = sl.stop - sl.start
                seen = sample_heard(n, tc.cert_frac_min, tc.cert_frac_max).to(out["activation"].device)
                lo, hi = song_bounds_torch(model, out, i, sl, seen, detach_range=True)
                major = ((y[sl] > 0).float().mean(0) > 0.5).float()
                worst = torch.where(major > 0.5, lo, hi)
                cert_terms.append(F.binary_cross_entropy_with_logits(worst, major))
            cert = torch.stack(cert_terms).mean()
            loss = loss + tc.cert_weight * cert
            parts["cert"] = float(cert.detach())
        l1 = wrong_class_l1(model)
        loss = loss + tc.w_l1 * l1
        if teacher is not None:
            kd = distillation(out["logit"], teacher)
            loss = (1 - tc.distill_weight) * loss + tc.distill_weight * kd
            parts["distill"] = float(kd.detach())
        return loss, parts

    def _epoch(self, model, corpus, songs, optimizer, scheduler, covs, isda_lambda, epoch, seed,
               pool=1, teacher_fn=None, geometry=True, encoder_frozen=False):
        model.train()
        if encoder_frozen:
            # v1.1: stage D trains the readout only. The encoder must also be in
            # eval mode, otherwise dropout/drop-path stay active and the EMA
            # centre keeps moving although the encoder weights are frozen.
            model.encoder.eval()
        gen = torch.Generator().manual_seed(seed * 1000 + epoch)
        totals, count = {}, 0
        for i, batch in enumerate(iterate(corpus, songs, self.tc.song_batch, True, seed * 7919 + epoch,
                                          self.device)):
            if self.tc.max_batches_per_epoch and i >= self.tc.max_batches_per_epoch:
                break
            stop = os.environ.get("GFDP_STOP_FILE")
            if stop and os.path.exists(stop):
                raise InterruptedError(f"STOP file present: {stop}")
            x = spec_augment(batch["x"], self.tc.spec_freq_mask, self.tc.spec_time_mask, gen)
            batch["x"] = low_pass(x, pool)
            teacher = teacher_fn(batch) if teacher_fn else None
            loss, parts = self._loss(model, batch, covs, isda_lambda, teacher, geometry)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                           self.tc.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            for k, v in parts.items():
                totals[k] = totals.get(k, 0.0) + v
            count += 1
        return {k: v / max(count, 1) for k, v in totals.items()}

    def _evaluate(self, model, corpus, songs):
        pred = predict(model, corpus, songs, self.device)
        return task_metrics(pred["y"], pred["logit"], self.mc.task), pred

    # --------------------------------------------------------------------- fit
    def fit(self, corpus: Corpus, fit_songs: List[Dict], dev_songs: List[Dict], seed: int) -> Dict:
        tc = self.tc
        if self.mc.task == "regression" and (tc.cert_weight > 0 or tc.self_distill):
            raise ValueError("certification-aware training and self-distillation are classification-only")
        seed_everything(seed)
        model = GFDPModel(self.mc).to(self.device)
        model.eval()
        centre_cos = model.encoder.init_centre(
            b["x"] for b in iterate(corpus, fit_songs[:64], 8, False, device=self.device))
        history = [dict(stage="init", mean_cosine_after_centring=centre_cos)]
        t0 = time.time()

        # Stage A: warm-up (readout frozen at class-connection initialisation)
        for p in model.readout_parameters():
            p.requires_grad_(False)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=tc.lr, weight_decay=tc.weight_decay)
        covs = self._covariances(model)
        for e in range(tc.epochs_warm):
            parts = self._epoch(model, corpus, fit_songs, opt, None, covs, 0.0, e, seed)
            history.append(dict(stage="warm", epoch=e, **parts))
        for p in model.readout_parameters():
            p.requires_grad_(True)

        # Stage B: joint, cosine restarts, snapshots
        groups = [{"params": [p for n, p in model.named_parameters()
                              if p.requires_grad and n not in ("w_loc", "w_glb", "bias")],
                   "lr": tc.lr, "weight_decay": tc.weight_decay},
                  {"params": model.readout_parameters(), "lr": tc.lr_readout, "weight_decay": 0.0}]
        opt = torch.optim.AdamW(groups)
        n_batches = math.ceil(len(fit_songs) / tc.song_batch)
        if tc.max_batches_per_epoch:
            n_batches = min(n_batches, tc.max_batches_per_epoch)
        cycle_epochs = max(1, tc.epochs_joint // max(1, tc.snapshot_cycles))
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=cycle_epochs * n_batches)
        best_key, best_state, best_epoch, waited = None, None, -1, 0
        snapshots = []
        push_at = tc.push_after_epochs or cycle_epochs
        push_report = None
        for e in range(tc.epochs_joint):
            if e == push_at and push_report is None:
                # Stage C (mid-training): anchor prototypes to real TRAIN segments,
                # then keep training the encoder against the live E(x_src) so the
                # deployed model never suffers a final "push gap".
                before_push, _ = self._evaluate(model, corpus, dev_songs)   # same weights, free prototypes
                push_report = push_prototypes(model, iterate(corpus, fit_songs, 8, False,
                                                             device=self.device), self.device)
                after_push, _ = self._evaluate(model, corpus, dev_songs)    # same weights, anchored
                history.append(dict(stage="push", epoch=e, mean_push_distance=push_report["mean_push_distance"],
                                    dev_before_push_same_weights=before_push,
                                    dev_after_push_same_weights=after_push))
                best_key, best_state, waited = None, None, 0     # select anchored states only
            frac = (e + 1) / tc.epochs_joint
            pool = 1
            if tc.curriculum_fraction > 0 and frac <= tc.curriculum_fraction:
                pool = max(1, int(round(tc.curriculum_initial_pool * (1 - frac / tc.curriculum_fraction))) + 1)
            parts = self._epoch(model, corpus, fit_songs, opt, sched, covs,
                                tc.isda_strength * frac, e, seed, pool=pool)
            dev_metrics, _ = self._evaluate(model, corpus, dev_songs)
            key = selection_key(dev_metrics, tc.select_on, self.mc.task)
            history.append(dict(stage="joint", epoch=e, pool=pool, dev=dev_metrics, **parts))
            self.log({"stage": "joint", "epoch": e, "dev_key": key, **parts})
            if push_report is not None:
                if best_key is None or key > best_key:
                    best_key, best_state, best_epoch, waited = key, copy.deepcopy(model.state_dict()), e, 0
                else:
                    waited += 1
                if (e + 1) % cycle_epochs == 0:
                    snapshots.append(copy.deepcopy(model.state_dict()))
                if waited >= tc.patience and e + 1 >= 2 * cycle_epochs:
                    break
        if push_report is None:          # very short schedules: anchor at the end
            push_report = push_prototypes(model, iterate(corpus, fit_songs, 8, False,
                                                         device=self.device), self.device)
            best_state = copy.deepcopy(model.state_dict()); best_epoch = tc.epochs_joint - 1
        model.load_state_dict(best_state)
        pushed_dev, _ = self._evaluate(model, corpus, dev_songs)

        # Stage D: last-layer optimisation with the encoder and prototypes frozen
        teacher_fn = None
        if tc.self_distill and snapshots:
            teacher_fn = self._snapshot_teacher(snapshots)
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.readout_parameters():
            p.requires_grad_(True)
        if self.mc.residual:
            model.residual_scale_logit.requires_grad_(True)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=tc.lr_readout,
                                weight_decay=0.0)
        covs = self._covariances(model)
        best_last = (selection_key(pushed_dev, tc.select_on, self.mc.task), copy.deepcopy(model.state_dict()), -1)
        for e in range(tc.epochs_last):
            parts = self._epoch(model, corpus, fit_songs, opt, None, covs, tc.isda_strength, 100 + e,
                                seed, teacher_fn=teacher_fn, geometry=False, encoder_frozen=True)
            dev_metrics, _ = self._evaluate(model, corpus, dev_songs)
            key = selection_key(dev_metrics, tc.select_on, self.mc.task)
            history.append(dict(stage="last", epoch=e, dev=dev_metrics, **parts))
            if key > best_last[0]:
                best_last = (key, copy.deepcopy(model.state_dict()), e)
        model.load_state_dict(best_last[1])
        final_dev, _ = self._evaluate(model, corpus, dev_songs)
        return dict(model=model, history=history, best_joint_epoch=best_epoch,
                    best_last_epoch=best_last[2], push=push_report, dev_after_push=pushed_dev,
                    dev_final=final_dev, snapshots=len(snapshots), seconds=time.time() - t0,
                    parameters=model.parameter_count())

    def _snapshot_teacher(self, snapshots):
        teachers = []
        for state in snapshots:
            m = GFDPModel(self.mc).to(self.device)
            m.load_state_dict(state)
            m.eval()
            teachers.append(m)

        @torch.no_grad()
        def teacher(batch):
            probs = [torch.sigmoid(t(batch["x"], batch["times"], batch["songs"], batch["song_slices"]))
                     for t in teachers]
            return torch.stack(probs).mean(0)
        return teacher
