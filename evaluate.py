# evaluate.py — load a saved model and evaluate it against a dataset
#
# Usage:
#     python evaluate.py <model.pkl> <data.csv>

import sys
import pickle
import numpy as np
import mlp
import common as cm


def main():
    model_path, data_path = sys.argv[1], sys.argv[2]

    print(f"[evaluate] loading data ...")
    X, y = cm.load_and_prep_data(data_path)
    print(f"[evaluate] {data_path} - {X.shape[0]} entries  shape {X.shape}")

    print(f"[evaluate] loading model ...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    if not isinstance(model, mlp.Sequential):
        sys.exit(f"[evaluate] '{model_path}' does not contain a valid Sequential model")
    print(f"[evaluate] {model_path} - {len(model.layers)} layers  {model.epochs} epochs  {model.batch} batch_size")

    preds = model.predict(X)

    predicted_classes = np.argmax(preds, axis=1)
    true_classes      = np.argmax(y, axis=1)

    val_acc = (predicted_classes == true_classes).mean()
    bce     = mlp.binary_crossentropy(y, preds)

    true_negatives  = ((predicted_classes == 0) & (true_classes == 0)).sum()
    true_positives  = ((predicted_classes == 1) & (true_classes == 1)).sum()
    false_positives = ((predicted_classes == 1) & (true_classes == 0)).sum()
    false_negatives = ((predicted_classes == 0) & (true_classes == 1)).sum()

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0

    actual_positives = true_positives + false_negatives
    actual_negatives = true_negatives + false_positives

    print("")
    print(f"[evaluate] accuracy:              {val_acc:.3%}")
    print(f"[evaluate] precision:             {precision:.3%}")
    print(f"[evaluate] binary cross-entropy:  {bce:.4f}")
    print("")
    print(f"[evaluate] of {actual_positives} cancer cases:  {true_positives} detected  ({true_positives/actual_positives:.1%}), {false_negatives} missed ({false_negatives/actual_positives:.1%})")
    print(f"[evaluate] of {actual_negatives} benign cases:  {true_negatives} detected  ({true_negatives/actual_negatives:.1%}), {false_positives} false positive ({false_positives/actual_negatives:.1%})")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python evaluate.py <model.pkl> <data.csv>")

    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        sys.exit(f"[evaluate] unexpected error: {type(e).__name__}: {e}")