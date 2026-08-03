"""compare.py - overlay learning curves from multiple training runs.

Usage:
    python compare.py <history1.json> <history2.json> [...]
    python compare.py --auto  # use every *.json in the cwd
"""

import pathlib
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_PATH = "comparison.png"


def load_history(path: str) -> dict[str, list]:
    try:
        with open(path) as f:
            history = json.load(f)
    except FileNotFoundError:
        sys.exit(f"[compare] history file not found: '{path}'")
    except json.JSONDecodeError as e:
        sys.exit(f"[compare] '{path}' is not valid JSON: {e}")

    for key in ("loss", "val_loss", "acc", "val_acc"):
        if key not in history:
            sys.exit(f"[compare] '{path}' is missing required key '{key}'")

    return history


def plot_comparison(histories: dict[str, dict[str, list]]) -> pathlib.Path:
    out = pathlib.Path(OUT_PATH)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Model comparison")

    for name, history in histories.items():
        has_val = not all(np.isnan(v) for v in history["val_loss"])
        epochs = range(1, len(history["loss"]) + 1)

        line, = ax_loss.plot(epochs, history["loss"], label=f"{name} (train)")
        if has_val:
            ax_loss.plot(epochs, history["val_loss"], label=f"{name} (val)",
                         color=line.get_color(), linestyle="--")

        line, = ax_acc.plot(epochs, history["acc"], label=f"{name} (train)")
        if has_val:
            ax_acc.plot(epochs, history["val_acc"], label=f"{name} (val)",
                        color=line.get_color(), linestyle="--")

    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.legend(fontsize="small")

    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.legend(fontsize="small")

    fig.tight_layout()
    plt.savefig(out)
    return out


def main():
    paths = sys.argv[1:]
    isAuto = False
    if paths == ["--auto"]:
        paths = [str(p) for p in pathlib.Path.cwd().glob("*.json")]
        isAuto = True

    if len(paths) < 2:
        out = __doc__
        if (isAuto):
            out = f"{out}\n--auto found {len(paths)} valid *.json files"
        out = f"{out}\nneed at least 2 files for a comparison, got: {len(paths)}"
        sys.exit(out)

    histories = {}
    for path in paths:
        name = pathlib.Path(path).stem
        print(f"[compare] loading {path} as '{name}'")
        histories[name] = load_history(path)

    saved = plot_comparison(histories)
    print(f"[compare] comparison plot saved - {saved}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        sys.exit(f"[compare] unexpected error: {type(e).__name__}: {e}")