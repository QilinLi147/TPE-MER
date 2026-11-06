# TPE-MER

Research implementation of the source-linked temporal evidence model for music emotion recognition.

## Environment
Python 3.9+, PyTorch, NumPy, and scikit-learn. Provide your own dataset cache and replace PATH_TO_CACHE in the command below. Check the dataset licence before use.

## Inputs
The cache directory contains G_<dataset>/x.npy, y.npy, times.npy, and tracks.json. Each flattened window contains 128 Mel and 84 secondary spectral channels with the same number of frames. y contains valence and arousal in [-1,1]. tracks.json records contiguous song boundaries and splits. This repository does not distribute audio, feature caches, learned weights, or official test labels.

## Example
```sh
python -m gfdp.cv --config configs/example.json --cache-base PATH_TO_CACHE --dataset data_1000songs --folds 5 --seeds 2025 2031 2032 --device cuda --out PATH_TO_OUTPUT
```

The example is a baseline configuration. The selected TPE-MER configurations for the three main-text corpora, both tasks and all five folds are in configs/selected/. For each fold use --only-folds and its matching configuration, with seeds 2025, 2031 and 2032. Selection used seed 2025 source validation. Reproduction requires the same cached data and song metadata. The repository does not include the external comparator implementations or pretrained weights. The original upstream secondary-view extractor is not available across all corpora. The programme reads train-role songs and makes song-disjoint folds. See the paper for experimental settings.

To run the saved fold configurations without manual selection:

```sh
python run_selected.py --task memo_cls --cache-base PATH_TO_CACHE --out PATH_TO_OUTPUT --device cuda
```

Valid tasks are pmemo_cls, pmemo_reg, k1000_cls, k1000_reg, memo_cls and memo_reg. Runs produce fold-specific outputs; the published table pools the held-out windows across the five folds before calculating each seed's metric. Re-running from configuration does not guarantee bitwise identical GPU results.
