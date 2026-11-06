"""Run the frozen per-fold configurations for one main-text corpus and task."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASETS = {"pmemo": "data_PMEmo", "k1000": "data_1000songs", "memo": "memo"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=[
        f"{corpus}_{kind}" for corpus in DATASETS for kind in ("cls", "reg")
    ])
    parser.add_argument("--cache-base", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2025, 2031, 2032])
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    args = parser.parse_args()
    dataset = DATASETS[args.task.split("_")[0]]
    for fold in args.folds:
        if fold not in range(5):
            parser.error("folds must be integers from 0 to 4")
        config = ROOT / "configs" / "selected" / f"{args.task}_fold{fold}.json"
        assert json.loads(config.read_text())["dataset"] == dataset
        command = [
            sys.executable, "-m", "gfdp.cv", "--config", str(config),
            "--dataset", dataset, "--cache-base", args.cache_base,
            "--folds", "5", "--fold-seed", "20261010", "--only-folds", str(fold),
            "--seeds", *map(str, args.seeds), "--device", args.device,
            "--out", str(Path(args.out) / args.task / f"fold{fold}"),
        ]
        subprocess.run(command, check=True, cwd=ROOT)

if __name__ == "__main__":
    main()
