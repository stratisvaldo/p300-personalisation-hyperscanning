# Train an SVM on extracted P300 calibration epochs
# Input .npz is expected to already contain flattened and participant-wise standardised features
# X shape should be (N_epochs, Features)
# y should be 0 for non-target and 1 for target

# cross-val (5 stratified folds) -> balanced acc
# confusion matrix, classification report on training data (printed)

'''
Example:
python Experiment_set_up/Decoder/train_svm.py `
  --input epoched_data/p300_group_epochs_tradml_flat.npz `
  --output models_svm/svm_group_calibration.joblib `
  --kernel linear `
  --C 1.0 `
  --metrics_output models_svm/svm_group_calibration_metrics.json
'''

import os
import json
import argparse
import joblib
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Path to extracted flattened epoch dataset .npz")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save trained SVM model .joblib")
    parser.add_argument("--kernel", type=str, default="linear",
                        choices=["linear", "rbf"])
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--gamma", type=str, default="scale",
                        help="Used for RBF kernel, e.g. 'scale' or 'auto'")
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--metrics_output", type=str, default=None,
                        help="Optional path to save training metrics as .json")
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)

    X = data["X"]   # (N, Features)
    y = data["y"]   # (N,)

    print("Loaded epoch dataset:", args.input)
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    if X.ndim != 2:
        raise RuntimeError(f"Expected X to have shape (N, Features), got {X.shape}")

    if len(X) != len(y):
        raise RuntimeError("X and y have different number of samples")

    X = X.astype(np.float32)
    n_features = int(X.shape[1])

    n_target = int((y == 1).sum())
    n_nontarget = int((y == 0).sum())

    print("Features per epoch:", n_features)
    print("Class counts:")
    print("  target     :", n_target)
    print("  non-target :", n_nontarget)
    print("Input is assumed to already be participant-wise standardised.")
    print("No StandardScaler is applied in training.")

    svm = SVC(
        kernel=args.kernel,
        C=args.C,
        gamma=args.gamma,
        probability=True,
        class_weight="balanced",
        random_state=42,
    )

    metrics_dict = {
        "input_file": args.input,
        "model_output": args.output,
        "kernel": args.kernel,
        "C": args.C,
        "gamma": args.gamma,
        "requested_cv_folds": int(args.cv_folds),
        "n_samples": int(len(X)),
        "n_features": n_features,
        "class_counts": {
            "target_1": n_target,
            "non_target_0": n_nontarget,
        },
        "standardisation": "participant-wise standardisation was already applied during epoch extraction",
    }

    max_possible_folds = min(n_target, n_nontarget)

    if max_possible_folds >= 2:
        cv_folds = min(args.cv_folds, max_possible_folds)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(svm, X, y, cv=cv, scoring="balanced_accuracy")

        print(f"\nCross-validated balanced accuracy ({cv_folds}-fold):")
        print("Scores:", np.round(scores, 4))
        print("Mean  :", float(scores.mean()))
        print("Std   :", float(scores.std()))

        metrics_dict["cross_validation"] = {
            "used": True,
            "scoring": "balanced_accuracy",
            "cv_folds_used": int(cv_folds),
            "scores": [float(s) for s in scores],
            "mean": float(scores.mean()),
            "std": float(scores.std()),
        }
    else:
        print("\nNot enough samples in one of the classes for cross-validation.")
        metrics_dict["cross_validation"] = {
            "used": False,
            "reason": "Not enough samples in one of the classes for cross-validation."
        }

    svm.fit(X, y)

    y_pred = svm.predict(X)
    cm = confusion_matrix(y, y_pred)
    report_dict = classification_report(y, y_pred, digits=4, output_dict=True)
    report_text = classification_report(y, y_pred, digits=4)

    print("\nTraining-set confusion matrix:")
    print(cm)
    print("\nTraining-set classification report:")
    print(report_text)

    metrics_dict["training_set"] = {
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    n_participants = int(data["n_participants"][0]) if "n_participants" in data else None
    participant_names = data["participant_names"] if "participant_names" in data else None
    features_per_participant = int(data["features_per_participant"][0]) if "features_per_participant" in data else None

    participant_scaler_means = []
    participant_scaler_scales = []

    if n_participants is not None:
        for p in range(n_participants):
            mean_key = f"scaler_mean_p{p+1}"
            scale_key = f"scaler_scale_p{p+1}"

            if mean_key not in data or scale_key not in data:
                raise RuntimeError(
                    f"Missing participant scaler stats in epoch file for participant {p+1}. "
                    f"Expected keys {mean_key} and {scale_key}."
                )

            participant_scaler_means.append(data[mean_key].astype(np.float32))
            participant_scaler_scales.append(data[scale_key].astype(np.float32))

    model_package = {
        "svm": svm,
        "n_features": n_features,
        "classes": np.unique(y),
        "n_participants": n_participants,
        "participant_names": participant_names,
        "features_per_participant": features_per_participant,
        "participant_scaler_means": participant_scaler_means,
        "participant_scaler_scales": participant_scaler_scales,
    }

    joblib.dump(model_package, args.output)
    print("\nSaved trained SVM model to:", args.output)

    if args.metrics_output is not None:
        os.makedirs(os.path.dirname(args.metrics_output) or ".", exist_ok=True)
        with open(args.metrics_output, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=2)
        print("Saved training metrics to:", args.metrics_output)


if __name__ == "__main__":
    main()