# Train an SVM on extracted P300 calibration epochs
# Input .npz is expected to already contain flattened features
# X shape should be (N_epochs, Features)
# y should be 0 for non-target and 1 for target

# cross-val (5 stratified folds) -> balanced acc 
# confusion matrix, classification report on training data (printed)
 
'''
Example:
python Experiment_set_up/Decoder/train_svm.py `
  --input epoched_data/p300_epochs_play_01_tradml.npz `
  --output models_svm/svm_calibration.joblib `
  --kernel linear `
  --C 1.0 `
  --metrics_output models_svm/svm_calibration_metrics.json
'''

import os
import json
import argparse
import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

    svm = SVC(
        kernel=args.kernel,
        C=args.C,
        gamma=args.gamma,
        probability=True,
        class_weight="balanced",
        random_state=42,
    )

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", svm),
    ])

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
    }

    # Quick CV estimate on calibration data
    max_possible_folds = min(n_target, n_nontarget)

    if max_possible_folds >= 2:
        cv_folds = min(args.cv_folds, max_possible_folds)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="balanced_accuracy")

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

    # Fit on all calibration data
    clf.fit(X, y)

    # Training-set sanity check
    y_pred = clf.predict(X)
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

    model_package = {
        "pipeline": clf,
        "n_features": n_features,
        "classes": np.unique(y),
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