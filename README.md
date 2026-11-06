# Traceable Prototype Evidence for Music Emotion Recognition

This package provides the source-linked temporal evidence model, the adapted music comparators in the main results table, the SEED cross-subject A-scale readout and the EEG comparators. Music source excerpts and EEG source windows are selected from fitting data. No audio, feature caches, predictions or pretrained weights are included.

## Setup

Use Python 3.9 or later, NumPy, SciPy, scikit-learn, PyTorch and einops. For EEG HDGNet and PGCN, install a PyTorch-compatible PyTorch Geometric build. Install the common dependencies with `pip install -r requirements.txt`.

Four EEG comparators use author implementations. To fetch the exact source revisions used by the adapters, run `python -m comparators.eeg.setup_external`. The script creates `comparators/third_party/recent_baselines/` and does not overwrite existing directories. It retrieves BF-GCN, EmT, HDGNet and PGCN from their upstream repositories. Review their individual licences before use; in particular EmT's CBCR licence permits non-commercial use subject to its conditions. BF-GCN's original constructor requires CUDA. DGC-Link, DGCNN, RGCB and MASA-TCN do not require these external source trees.

## Music inputs

Set `PATH_TO_CACHE` to a directory containing `G_<dataset>/x.npy`, `y.npy`, `times.npy` and `tracks.json`. Each flattened window has 128 Mel channels and 84 aligned secondary spectral channels. Labels contain valence and arousal in [-1,1]; tracks record contiguous song boundaries and splits. The released preprocessing begins from cached views. The upstream secondary-view extraction parameters are not available consistently across corpora.

The final-model files in `configs/selected/` cover three corpora, classification and regression, and five folds. The same selected configuration is used for seeds 2025, 2031 and 2032. For example:

```sh
python run_selected.py --task memo_cls --cache-base PATH_TO_CACHE --out PATH_TO_OUTPUT --device cuda
```

Each fold writes predictions separately. To reproduce the paper's seed-level metric, pool the five held-out folds before computing that metric.

## Music comparators

Short-chunk + Res, DAMFF-style, DSAML-style and the DAMER adapter use the same cached inputs and source-validation protocol. Their configurations are in `comparators/music/configs/`; for example:

```sh
python -m comparators.music.run --config comparators/music/configs/pmemo_cls_damff.json --cache-base PATH_TO_CACHE --dataset data_PMEmo --out PATH_TO_OUTPUT --device cuda
python -m comparators.music.linear --kind linear --task classification --dataset data_PMEmo --cache-base PATH_TO_CACHE --out PATH_TO_OUTPUT
python -m comparators.music.linear --kind majority --task classification --dataset data_PMEmo --cache-base PATH_TO_CACHE --out PATH_TO_OUTPUT
```

The architectures are protocol adaptations, not complete reproductions of the original papers' preprocessing and training recipes.

## SEED EEG adaptation

Provide separate FIT, source-DEV and target NPZ files. Each FIT/DEV NPZ contains `x` of shape [N,62,10,5], integer class labels `y`, and `subject`, `session`, `trial`, and `original_index` arrays. For each target subject, FIT uses the other 12 subjects and source DEV the next two cyclic subjects. Target data are used only after the encoder and readout are frozen. The target input for prediction needs only `x`.

```sh
python -m eeg.train_encoder --fit PATH_TO_FIT.npz --dev PATH_TO_DEV.npz --seed 8636 --classes 3 --out PATH_TO_ENCODER.pt --device cuda
python -m eeg.run_ascale --fit PATH_TO_FIT.npz --dev PATH_TO_DEV.npz --encoder PATH_TO_ENCODER.pt --out PATH_TO_READOUT.npz --device cuda
python -m eeg.predict_ascale --input PATH_TO_TARGET.npz --encoder PATH_TO_ENCODER.pt --readout PATH_TO_READOUT.npz --out PATH_TO_PREDICTIONS.npz --device cuda
```

The encoder is selected using source-DEV accuracy. A-scale keeps 28 source slots per class, sets each kernel family's width to its positive FIT-distance median, then selects the readout's regularisation strength using source DEV. This task uses the restricted development pool specified in the paper.

## EEG comparators

The methods are `bf_gcn_de`, `dgclink`, `dgcnn`, `emt`, `hdgnet`, `masatcn`, `pgcn` and `rgcb`. Each takes the same separate FIT and DEV NPZ format.

```sh
python -m comparators.eeg.train --method dgcnn --fit PATH_TO_FIT.npz --dev PATH_TO_DEV.npz --seed 8636 --classes 3 --out PATH_TO_CHECKPOINT.pt --device cuda
python -m comparators.eeg.predict --input PATH_TO_TARGET.npz --checkpoint PATH_TO_CHECKPOINT.pt --out PATH_TO_PREDICTIONS.npz --device cuda
```

BF-GCN-DE uses a FIT-only DE correlation graph in place of the original PLV graph. EmT and MASA-TCN use the temporal windows; the other comparators use their time averages. The assigned seeds are 8636, 8637, 8638 and 8639. The source-DEV selection rules and method defaults are given in `comparators/eeg/training_defaults.json`.

## Reproducibility

Model and training configurations are published, but this package does not distribute the data partitions or checkpoints. Differences in data preparation, upstream packages, hardware and numerical libraries may change results. The music and EEG controls are adapted to the input and split protocols described in the paper.
