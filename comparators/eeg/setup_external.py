"""Install the author implementations required by four EEG comparators."""
import subprocess
from pathlib import Path

DESTINATION = Path(__file__).resolve().parents[1] / "third_party" / "recent_baselines"
SOURCES = {
    "BF-GCN": ("https://github.com/UESTC-BAC/BF-GCN.git", "20c24c3c1ca459cdfc4b5e2cbed8c2c8a5de8d9a"),
    "EmT": ("https://github.com/yi-ding-cs/EmT.git", "3ea05dc90a915182ab3407191c287456f912316d"),
    "HDGNet": ("https://github.com/affctivai/HDGNet.git", "a89373d3331454af59f497e39d4d6b4e3292d90a"),
    "PGCN": ("https://github.com/Jinminbox/PGCN.git", "2e1c7dfff205c43e0d6b775247494e06d7ebd7cc"),
}


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for name, (url, commit) in SOURCES.items():
        directory = DESTINATION / name
        if directory.exists():
            raise FileExistsError(f"existing external source retained: {directory}")
        subprocess.run(["git", "clone", url, str(directory)], check=True)
        subprocess.run(["git", "-C", str(directory), "checkout", "--detach", commit], check=True)
        print(f"Installed {name} at its frozen commit. Review its licence before use.")


if __name__ == "__main__":
    main()
