"""Data access: TRACE-MER resident cache (read-only, TRAIN-only by default),
song-grouped stratified folds, and a synthetic corpus for CPU smoke tests.

Cache layout (as used by the D2 releases):
  <cache_base>/G_<dataset>/x.npy      [N, 212*44] float32 (128 mel + 84 second view)
  <cache_base>/G_<dataset>/y.npy      [N, 2] canonical valence, arousal in [-1, 1]
  <cache_base>/G_<dataset>/times.npy  [N] seconds
  <cache_base>/G_<dataset>/tracks.json  [{track_id, split, start, end}, ...]
Rows of a song are contiguous. Hard class = label > 0 (UDDA-s2 rule).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch

CHANNELS = 212


class Corpus:
    """Songs with contiguous rows. `songs` items: dict(track_id, song_index, start, end, split)."""

    def __init__(self, x: np.ndarray, y: np.ndarray, times: np.ndarray, songs: List[Dict],
                 frames: int, identity: Optional[Dict] = None) -> None:
        self.x, self.y, self.times = x, y, times
        self.songs = songs
        self.frames = frames
        self.identity = identity or {}
        self.access = {"rows": 0}

    def song_majority(self, song: Dict) -> np.ndarray:
        yy = np.asarray(self.y[song["start"]:song["end"]])
        return (yy > 0).mean(0) > 0.5

    def read_song(self, song: Dict) -> Dict[str, np.ndarray]:
        s, e = song["start"], song["end"]
        self.access["rows"] += e - s
        x = np.array(self.x[s:e], dtype=np.float32, copy=True).reshape(e - s, CHANNELS, self.frames)
        out = dict(x=x, y=np.array(self.y[s:e], dtype=np.float32),
                   times=np.array(self.times[s:e], dtype=np.float32),
                   songs=np.full(e - s, song["song_index"], dtype=np.int64),
                   rows=np.arange(s, e, dtype=np.int64))
        if not all(np.isfinite(v).all() for v in out.values()):
            raise ValueError("non-finite values; no silent exclusion")
        return out


def load_trace_cache(cache_base: str, dataset: str, roles: Sequence[str] = ("train",)) -> Corpus:
    """Open the cache read-only. TEST and quarantined roles are refused."""
    forbidden = {"test", "quarantined"} & set(roles)
    if forbidden:
        raise PermissionError(f"roles {sorted(forbidden)} are sealed during development")
    root = Path(cache_base) / f"G_{dataset}"
    tracks = json.loads((root / "tracks.json").read_text())
    arrays = {k: np.load(root / f"{k}.npy", mmap_mode="r", allow_pickle=False) for k in ("x", "y", "times")}
    if arrays["x"].shape[1] % CHANNELS:
        raise ValueError("unexpected cache width")
    frames = arrays["x"].shape[1] // CHANNELS
    songs = [dict(t, song_index=i) for i, t in enumerate(tracks) if t["split"] in roles]
    identity = {"cache_root": str(root), "roles": list(roles), "songs": len(songs)}
    return Corpus(arrays["x"], arrays["y"], arrays["times"], songs, frames, identity)


def synthetic_corpus(n_songs: int = 24, rows_per_song: int = 20, frames: int = 44,
                     seed: int = 0, signal: float = 1.5) -> Corpus:
    """Songs whose valence/arousal depend on band energy patterns plus noise."""
    rng = np.random.default_rng(seed)
    xs, ys, ts, songs = [], [], [], []
    start = 0
    for i in range(n_songs):
        v_song, a_song = rng.uniform(-0.6, 0.6, size=2)
        t = 15.0 + 0.5 * np.arange(rows_per_song)
        drift = 0.15 * np.sin(t / 5.0 + rng.uniform(0, 6))
        v = np.clip(v_song + drift + rng.normal(0, 0.05, rows_per_song), -1, 1)
        a = np.clip(a_song - drift + rng.normal(0, 0.05, rows_per_song), -1, 1)
        x = rng.normal(0, 1, size=(rows_per_song, CHANNELS, frames)).astype(np.float32)
        x[:, 10:30] += signal * v[:, None, None]          # low mel bands track valence
        x[:, 90:110] += signal * a[:, None, None]         # high mel bands track arousal
        x[:, 150:170] += signal * a[:, None, None] * 0.5  # second view also sees arousal
        xs.append(x.reshape(rows_per_song, -1)); ys.append(np.stack([v, a], 1)); ts.append(t)
        songs.append(dict(track_id=f"syn{i:03d}", split="train", song_index=i,
                          start=start, end=start + rows_per_song))
        start += rows_per_song
    return Corpus(np.concatenate(xs), np.concatenate(ys).astype(np.float32),
                  np.concatenate(ts).astype(np.float32), songs, frames, {"synthetic": True, "seed": seed})


def song_folds(corpus: Corpus, k: int, seed: int) -> List[Dict[str, List[Dict]]]:
    """Stratified (by song-majority V/A quadrant) song-grouped K folds.
    Each fold: train songs, holdout songs. Every song is held out exactly once."""
    rng = np.random.default_rng(seed)
    quadrant = {}
    for s in corpus.songs:
        v, a = corpus.song_majority(s)
        quadrant.setdefault(int(v) + 2 * int(a), []).append(s)
    assign = {}
    offset = 0
    for q in sorted(quadrant):
        members = [quadrant[q][i] for i in rng.permutation(len(quadrant[q]))]
        for i, s in enumerate(members):
            assign[s["track_id"]] = (i + offset) % k
        offset += len(members)
    folds = []
    for f in range(k):
        hold = [s for s in corpus.songs if assign[s["track_id"]] == f]
        train = [s for s in corpus.songs if assign[s["track_id"]] != f]
        folds.append({"train": train, "holdout": hold})
    return folds


def split_dev(songs: List[Dict], fraction: float, seed: int) -> (List[Dict], List[Dict]):
    """Early-stopping dev songs carved from the training side of a fold only."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(songs))
    n_dev = max(1, int(round(fraction * len(songs))))
    dev = [songs[i] for i in sorted(order[:n_dev])]
    fit = [songs[i] for i in sorted(order[n_dev:])]
    return fit, dev


def iterate(corpus: Corpus, songs: List[Dict], song_batch: int, shuffle: bool,
            seed: int = 0, device="cpu") -> Iterator[Dict]:
    order = np.random.default_rng(seed).permutation(len(songs)) if shuffle else np.arange(len(songs))
    for lo in range(0, len(order), song_batch):
        parts = [corpus.read_song(songs[i]) for i in order[lo:lo + song_batch]]
        batch = {k: torch.from_numpy(np.concatenate([p[k] for p in parts])).to(device)
                 for k in parts[0]}
        slices, pos = [], 0
        for p in parts:
            n = len(p["y"]); slices.append(slice(pos, pos + n)); pos += n
        batch["song_slices"] = slices
        yield batch
