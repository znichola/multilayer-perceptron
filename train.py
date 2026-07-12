# train.py — config-driven training launcher
#
# Usage:
#     python train.py <network.(txt/py)> <train.csv> <validation.csv>
#
# The config file is interpreted as plain Python. Everything in mlp.py is
# pre-imported, plus 'input_shape' and 'output_shape' derived from the data.
# The resulting model is saved and can be loaded to rerun the inferance

import pathlib
import pickle
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import mlp
import common as cm


def build_namespace(input_shape: int, output_shape: int) -> dict:
    print(f"Building namespace with input shape: ", input_shape, "output shape:", output_shape)
    return {
        **vars(mlp),
        "input_shape": input_shape,
        "output_shape": output_shape,
    }


def load_config(path: str, namespace: dict) -> dict:
    try:
        with open(path) as f:
            source = f.read()
    except FileNotFoundError:
        sys.exit(f"[train] config file not found: '{path}'")

    try:
        exec(compile(source, path, "exec"), namespace)
    except SyntaxError as e:
        sys.exit(f"[train] syntax error in '{path}':\n  {e}")
    except Exception as e:
        sys.exit(f"[train] error in '{path}': {type(e).__name__}: {e}")

    return namespace


def validate(namespace: dict, path: str) -> mlp.Sequential:
    model = namespace.get("model")
    if not isinstance(model, mlp.Sequential):
        sys.exit(f"[train] '{path}' must assign a compiled Sequential to 'model'")
    if not hasattr(model, "loss_fn"):
        sys.exit(f"[train] 'model' must call model.compile(...) before training")

    return model


def save_model(model: mlp.Sequential, config_path: str) -> pathlib.Path:
    out = pathlib.Path(config_path).with_suffix(".pkl")
    with open(out, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out


def save_history(history: dict[str, list], config_path: str) -> pathlib.Path:
    out = pathlib.Path(config_path).with_suffix(".json")
    with open(out, "w") as f:
        json.dump(history, f)
    return out


def plot_history(history: dict[str, list], config_path: str) -> pathlib.Path:
    has_val = not all(np.isnan(v) for v in history["val_loss"])
    epochs  = range(1, len(history["loss"]) + 1)
    out     = pathlib.Path(config_path).with_suffix(".png")

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(out.stem)

    ax_loss.plot(epochs, history["loss"], label="train")
    if has_val:
        ax_loss.plot(epochs, history["val_loss"], label="val")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.legend()

    ax_acc.plot(epochs, history["acc"], label="train")
    if has_val:
        ax_acc.plot(epochs, history["val_acc"], label="val")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.legend()

    fig.tight_layout()
    plt.savefig(out)
    return out


def main():

    config_path, train_path, valid_path = sys.argv[1], sys.argv[2], sys.argv[3]

    print("[train] loading data ...")
    X_train, y_train = cm.load_and_prep_data(train_path)
    X_valid, y_valid = cm.load_and_prep_data(valid_path)
    print(f"[train] train {X_train.shape}  valid {X_valid.shape}")

    namespace = build_namespace(X_train.shape[1], y_train.shape[1])
    namespace = load_config(config_path, namespace)
    model = validate(namespace, config_path)

    print(f"[train] {config_path} - {model}")
    history = model.fit(X_train, y_train, X_valid, y_valid)

    preds = model.predict(X_valid)
    val_acc = (np.argmax(preds, axis=1) == np.argmax(y_valid, axis=1)).mean()
    bce     = mlp.binary_crossentropy(y_valid, preds)

    print("")
    print(f"[train] validation accuracy:         {val_acc:.3%}")
    print(f"[train] binary cross-entropy (eval): {bce:.4f}")

    print("")
    print(f"[train] model saved   - {save_model(model, config_path)}")
    print(f"[train] history saved - {save_history(history, config_path)}")
    print(f"[train] plot saved    - {plot_history(history, config_path)}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("Usage: python train.py <network.(txt/py)> <train.csv> <validation.csv>")
    # main()
    # exit()
    try:
        main()
    except SystemExit:
        raise  # let sys.exit() from validate/load_config pass through
    except Exception as e:
        sys.exit(f"[train] unexpected error: {type(e).__name__}: {e}")