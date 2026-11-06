"""v1.5 black-box baselines trained under exactly the same protocol as GFDP-MER.

Same corpus, song folds, fit/dev split, augmentation, epochs budget, dev-only selection,
loss family (soft-label BCE for classification; MSE + (1 - CCC) for regression) and
prediction format. None of these models is interpretable or certifiable; they answer
"what does the additive real-prototype structure cost in accuracy?".

kind
  twin        the SAME dual-view encoder as GFDP-MER (A1 recipe), but the additive
              prototype readout is replaced by a black-box head on [z_t, mean_song(z)]
              -> the direct test of "interpretability costs no accuracy";
  shortchunk  Short-chunk CNN with residual blocks (Won, Ferraro, Bogdanov, Serra,
              "Evaluation of CNN-based automatic music tagging models", SMC 2020;
              public code minzwon/sota-music-tagging-models), mel view only, n_channels=128;
              per-segment embedding + the same pooled-context head;
  crnn        Convolutional recurrent network (Choi, Fazekas, Sandler, Cho, ICASSP 2017):
              4 conv blocks (64,128,128,128) + 2-layer GRU(32), mel view only. Our segments
              (0.5-1 s) are shorter than the CNN's receptive field, so the GRU runs over the
              song's sequence of segment embeddings (causal, one output per segment). Time
              pooling is clipped to the available frames.
Adaptations are documented, not hidden; layer sizes follow the published models.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .data import Corpus, iterate
from .encoder import DualViewEncoder, count_parameters
from .losses import soft_targets
from .metrics import task_metrics
from .train import TrainConfig, regression_loss, seed_everything, selection_key, spec_augment

KINDS = ("twin", "shortchunk", "crnn")


@dataclass
class BaselineConfig:
    kind: str = "twin"
    task: str = "classification"
    mel_bins: int = 128
    second_bins: int = 84
    frames: int = 44
    channels: int = 128            # twin encoder width / shortchunk n_channels
    dim: int = 128                 # twin embedding size
    dropout: float = 0.1
    drop_path: float = 0.1
    envelope: bool = True

    @classmethod
    def from_dict(cls, d: Dict) -> "BaselineConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ------------------------------------------------------------------ encoders
class Res2d(nn.Module):
    """Residual block of the short-chunk ResNet (Won et al. 2020)."""

    def __init__(self, cin: int, cout: int, stride: int = 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(cout)
        self.diff = stride != 1 or cin != cout
        if self.diff:
            self.conv3 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1)
            self.bn3 = nn.BatchNorm2d(cout)

    def forward(self, x: Tensor) -> Tensor:
        out = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
        short = self.bn3(self.conv3(x)) if self.diff else x
        return F.relu(out + short)


class ShortChunkResCNN(nn.Module):
    """[rows, mel_bins, frames] -> [rows, 4 * n_channels]."""

    def __init__(self, n_channels: int = 128, dropout: float = 0.5) -> None:
        super().__init__()
        n = n_channels
        self.spec_bn = nn.BatchNorm2d(1)
        self.layers = nn.Sequential(Res2d(1, n), Res2d(n, n), Res2d(n, 2 * n), Res2d(2 * n, 2 * n),
                                    Res2d(2 * n, 2 * n), Res2d(2 * n, 2 * n), Res2d(2 * n, 4 * n))
        self.dense = nn.Sequential(nn.Linear(4 * n, 4 * n), nn.BatchNorm1d(4 * n), nn.ReLU(), nn.Dropout(dropout))
        self.out_dim = 4 * n

    def forward(self, mel: Tensor) -> Tensor:
        h = self.layers(self.spec_bn(mel[:, None]))
        h = h.amax(dim=(2, 3))                      # global max over remaining freq/time
        return self.dense(h)


class CRNNConv(nn.Module):
    """Conv part of Choi et al. (2017): conv3x3-BN-ELU-maxpool, pools (2,2),(3,3),(4,4),(4,4)
    on frequency; time pooling is clipped to the frames left (segments are short)."""

    def __init__(self) -> None:
        super().__init__()
        self.spec_bn = nn.BatchNorm2d(1)
        chans = [(1, 64, 2), (64, 128, 3), (128, 128, 4), (128, 128, 4)]
        self.convs = nn.ModuleList(nn.Conv2d(a, b, 3, padding=1) for a, b, _ in chans)
        self.bns = nn.ModuleList(nn.BatchNorm2d(b) for _, b, _ in chans)
        self.pools = [p for _, _, p in chans]
        self.out_dim = 128

    def forward(self, mel: Tensor) -> Tensor:
        h = self.spec_bn(mel[:, None])
        for conv, bn, p in zip(self.convs, self.bns, self.pools):
            h = F.elu(bn(conv(h)))
            kf, kt = min(p, h.shape[2]), min(p, h.shape[3])
            h = F.max_pool2d(h, (kf, kt), ceil_mode=True)
        return h.amax(dim=(2, 3))


# -------------------------------------------------------------------- model
class BaselineModel(nn.Module):
    FORMAT = "gfdp_mer_baseline_v1"

    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()
        c = self.config = config
        if c.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if c.task not in ("classification", "regression"):
            raise ValueError("task must be classification or regression")
        if c.kind == "twin":
            self.encoder = DualViewEncoder(c.mel_bins, c.second_bins, c.frames, c.channels, c.dim,
                                           c.dropout, c.drop_path, c.envelope)
            width = c.dim
        elif c.kind == "shortchunk":
            self.encoder = ShortChunkResCNN(c.channels)
            width = self.encoder.out_dim
        else:
            self.encoder = CRNNConv()
            width = self.encoder.out_dim
        if c.kind == "crnn":
            self.gru = nn.GRU(width, 32, num_layers=2, batch_first=True)
            self.head = nn.Linear(32, 2)
        else:   # same black-box context head for twin and shortchunk
            self.head = nn.Sequential(nn.Linear(2 * width, 128), nn.GELU(), nn.Dropout(c.dropout),
                                      nn.Linear(128, 2))

    def embed(self, x: Tensor) -> Tensor:
        if self.config.kind == "twin":
            return self.encoder(x)
        return self.encoder(x[:, :self.config.mel_bins])

    def forward(self, x: Tensor, times: Tensor, songs: Tensor, song_slices: Sequence[slice]) -> Tensor:
        e = self.embed(x)
        out = e.new_zeros(e.shape[0], 2)
        for sl in song_slices:
            es = e[sl]
            if self.config.kind == "crnn":
                order = torch.argsort(times[sl])          # causal GRU over the song in time order
                h, _ = self.gru(es[order][None])
                res = torch.empty_like(out[sl])
                res[order] = self.head(h[0])
                out[sl] = res
            else:
                ctx = es.mean(0, keepdim=True).expand_as(es)
                out[sl] = self.head(torch.cat([es, ctx], -1))
        return out

    def head_parameters(self) -> List[nn.Parameter]:
        ps = list(self.head.parameters())
        if self.config.kind == "crnn":
            ps += list(self.gru.parameters())
        return ps

    def parameter_count(self) -> Dict[str, int]:
        return dict(total=count_parameters(self), encoder=count_parameters(self.encoder))

    def checkpoint(self, extra=None) -> Dict:
        return dict(extra or {}, format=self.FORMAT, config=self.config.__dict__,
                    state={k: v.detach().cpu() for k, v in self.state_dict().items()})


def restore_baseline(payload: Dict, device: str = "cpu") -> BaselineModel:
    if payload.get("format") != BaselineModel.FORMAT:
        raise ValueError("not a GFDP-MER baseline checkpoint")
    m = BaselineModel(BaselineConfig.from_dict(payload["config"]))
    m.load_state_dict(payload["state"], strict=True)
    return m.to(device)


@torch.no_grad()
def predict_baseline(model: BaselineModel, corpus: Corpus, songs: List[Dict], device="cpu",
                     song_batch: int = 8) -> Dict[str, np.ndarray]:
    model.eval()
    logits, ys, ss, rows = [], [], [], []
    for b in iterate(corpus, songs, song_batch, shuffle=False, device=device):
        logits.append(model(b["x"], b["times"], b["songs"], b["song_slices"]).cpu())
        ys.append(b["y"].cpu()); ss.append(b["songs"].cpu()); rows.append(b["rows"].cpu())
    return dict(logit=torch.cat(logits).numpy(), y=torch.cat(ys).numpy(),
                song=torch.cat(ss).numpy(), row=torch.cat(rows).numpy())


# ------------------------------------------------------------------ trainer
class BaselineTrainer:
    """Budget matched to GFDP-MER: (epochs_warm + epochs_joint) full-model epochs with the same
    cosine-restart schedule and dev selection, then epochs_last head-only epochs (encoder in
    eval mode), mirroring GFDP's stage D. No prototypes, no push, no ISDA, no geometry losses."""

    def __init__(self, cfg: BaselineConfig, tc: TrainConfig, device="cpu",
                 log: Callable[[Dict], None] = print) -> None:
        self.cfg, self.tc, self.device, self.log = cfg, tc, device, log

    def _loss(self, model, batch):
        out = model(batch["x"], batch["times"], batch["songs"], batch["song_slices"])
        y = batch["y"]
        if self.cfg.task == "regression":
            loss = regression_loss(out, y)
            return loss, {"reg": float(loss.detach())}
        loss = F.binary_cross_entropy_with_logits(out, soft_targets(y, self.tc.label_sigma))
        return loss, {"cls": float(loss.detach())}

    def _epoch(self, model, corpus, songs, opt, sched, epoch, seed, head_only=False):
        model.train()
        if head_only:
            model.encoder.eval()
        gen = torch.Generator().manual_seed(seed * 1000 + epoch)
        totals, count = {}, 0
        for i, batch in enumerate(iterate(corpus, songs, self.tc.song_batch, True, seed * 7919 + epoch,
                                          self.device)):
            if self.tc.max_batches_per_epoch and i >= self.tc.max_batches_per_epoch:
                break
            import os
            stop = os.environ.get("GFDP_STOP_FILE")
            if stop and os.path.exists(stop):
                raise InterruptedError(f"STOP file present: {stop}")
            batch["x"] = spec_augment(batch["x"], self.tc.spec_freq_mask, self.tc.spec_time_mask, gen)
            loss, parts = self._loss(model, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], self.tc.grad_clip)
            opt.step()
            if sched is not None:
                sched.step()
            for k, v in parts.items():
                totals[k] = totals.get(k, 0.0) + v
            count += 1
        return {k: v / max(count, 1) for k, v in totals.items()}

    def _evaluate(self, model, corpus, songs):
        pred = predict_baseline(model, corpus, songs, self.device)
        return task_metrics(pred["y"], pred["logit"], self.cfg.task), pred

    def fit(self, corpus: Corpus, fit_songs: List[Dict], dev_songs: List[Dict], seed: int) -> Dict:
        tc = self.tc
        seed_everything(seed)
        model = BaselineModel(self.cfg).to(self.device)
        history = []
        if self.cfg.kind == "twin":
            model.eval()
            cos = model.encoder.init_centre(b["x"] for b in iterate(corpus, fit_songs[:64], 8, False,
                                                                    device=self.device))
            history.append(dict(stage="init", mean_cosine_after_centring=cos))
        t0 = time.time()
        epochs = tc.epochs_warm + tc.epochs_joint
        opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
        n_batches = math.ceil(len(fit_songs) / tc.song_batch)
        if tc.max_batches_per_epoch:
            n_batches = min(n_batches, tc.max_batches_per_epoch)
        cycle_epochs = max(1, tc.epochs_joint // max(1, tc.snapshot_cycles))
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=cycle_epochs * n_batches)
        best_key, best_state, best_epoch, waited = None, None, -1, 0
        for e in range(epochs):
            parts = self._epoch(model, corpus, fit_songs, opt, sched, e, seed)
            dev, _ = self._evaluate(model, corpus, dev_songs)
            key = selection_key(dev, tc.select_on, self.cfg.task)
            history.append(dict(stage="joint", epoch=e, dev=dev, **parts))
            self.log({"stage": "joint", "epoch": e, "dev_key": key, **parts})
            if best_key is None or key > best_key:
                best_key, best_state, best_epoch, waited = key, copy.deepcopy(model.state_dict()), e, 0
            else:
                waited += 1
            if waited >= tc.patience and e + 1 >= 2 * cycle_epochs:
                break
        model.load_state_dict(best_state)
        # head-only phase (mirrors GFDP stage D)
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.head_parameters():
            p.requires_grad_(True)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=tc.lr_readout,
                                weight_decay=0.0)
        best_last = (best_key, copy.deepcopy(model.state_dict()), -1)
        for e in range(tc.epochs_last):
            parts = self._epoch(model, corpus, fit_songs, opt, None, 100 + e, seed, head_only=True)
            dev, _ = self._evaluate(model, corpus, dev_songs)
            key = selection_key(dev, tc.select_on, self.cfg.task)
            history.append(dict(stage="last", epoch=e, dev=dev, **parts))
            if key > best_last[0]:
                best_last = (key, copy.deepcopy(model.state_dict()), e)
        model.load_state_dict(best_last[1])
        final_dev, _ = self._evaluate(model, corpus, dev_songs)
        return dict(model=model, history=history, best_joint_epoch=best_epoch, best_last_epoch=best_last[2],
                    dev_final=final_dev, seconds=time.time() - t0, parameters=model.parameter_count())
