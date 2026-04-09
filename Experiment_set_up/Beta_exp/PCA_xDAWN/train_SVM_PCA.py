# train_svm_pca.py
# Train grouped SVM + PCA on flattened P300 epochs
#
# Input:
# - flattened grouped epoch .npz from your current traditional ML epoching
#
# Output:
# - trained PCA+SVM model package
# - metrics json with CV and training-set results
#
# This is the easiest comparison against your current baseline SVM
# because the input format stays exactly the same.

'''
python Experiment_set_up/Beta_exp/PCA_xDAWN/train_SVM_PCA.py \
  --input Experiment_set_up/Beta_exp/Results/p300_group_epochs_tradml_flat.npz \
  --output Experiment_set_up/Beta_exp/SVM_PCA/svm_group_pca.joblib \
  --metrics_output Experiment_set_up/Beta_exp/SVM_PCA/svm_group_pca_metrics.json \
  --kernel linear \
  --C 1.0 \
  --pca_components 0.95
'''

import os
import json
import argparse
import joblib
import numpy as np

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Path to flattened grouped epoch dataset .npz")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save trained PCA+SVM model .joblib")
    parser.add_argument("--metrics_output", type=str, default=None,
                        help="Optional path to save metrics .json")

    parser.add_argument("--kernel", type=str, default="linear",
                        choices=["linear", "rbf"])
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--gamma", type=str, default="scale")

    parser.add_argument("--pca_components", type=float, default=0.95,
                        help="PCA components. Int = exact number, float = explained variance ratio")
    parser.add_argument("--cv_folds", type=int, default=5)

    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)

    if X.ndim != 2:
        raise RuntimeError(f"Expected X to have shape (N, Features), got {X.shape}")
    if len(X) != len(y):
        raise RuntimeError("X and y have different number of samples")

    print("Loaded epoch dataset:", args.input)
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    n_target = int((y == 1).sum())
    n_nontarget = int((y == 0).sum())

    print("Class counts:")
    print("  target     :", n_target)
    print("  non-target :", n_nontarget)
    print("Input is assumed to already be participant-wise standardised.")
    print("No StandardScaler is applied in training.")

    metrics = {
        "input_file": args.input,
        "model_output": args.output,
        "model_type": "PCA + SVM",
        "kernel": args.kernel,
        "C": args.C,
        "gamma": args.gamma,
        "pca_components_requested": args.pca_components,
        "requested_cv_folds": int(args.cv_folds),
        "n_samples": int(len(X)),
        "n_features_before_pca": int(X.shape[1]),
        "class_counts": {
            "target_1": n_target,
            "non_target_0": n_nontarget,
        },
        "standardisation": "participant-wise standardisation was already applied during epoch extraction",
    }

    # Cross-validation
    max_possible_folds = min(n_target, n_nontarget)

    if max_possible_folds >= 2:
        cv_folds = min(args.cv_folds, max_possible_folds)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        cv_scores = []

        for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            # Fit PCA on fold-train only
            pca_fold = PCA(n_components=args.pca_components, svd_solver="full", random_state=42)
            X_tr_pca = pca_fold.fit_transform(X_tr)
            X_va_pca = pca_fold.transform(X_va)

            clf_fold = SVC(
                kernel=args.kernel,
                C=args.C,
                gamma=args.gamma,
                probability=True,
                class_weight="balanced",
                random_state=42,
            )
            clf_fold.fit(X_tr_pca, y_tr)

            y_va_pred = clf_fold.predict(X_va_pca)
            bal_acc = balanced_accuracy_score(y_va, y_va_pred)
            cv_scores.append(float(bal_acc))

            print(f"Fold {fold_idx} balanced accuracy: {bal_acc:.4f}")

        cv_scores = np.asarray(cv_scores, dtype=np.float64)

        print(f"\nCross-validated balanced accuracy ({cv_folds}-fold):")
        print("Scores:", np.round(cv_scores, 4))
        print("Mean  :", float(cv_scores.mean()))
        print("Std   :", float(cv_scores.std()))

        metrics["cross_validation"] = {
            "used": True,
            "scoring": "balanced_accuracy",
            "cv_folds_used": int(cv_folds),
            "scores": [float(s) for s in cv_scores],
            "mean": float(cv_scores.mean()),
            "std": float(cv_scores.std()),
        }
    else:
        print("\nNot enough samples in one of the classes for cross-validation.")
        metrics["cross_validation"] = {
            "used": False,
            "reason": "Not enough samples in one of the classes for cross-validation."
        }

    # Final fit on full training data
    pca = PCA(n_components=args.pca_components, svd_solver="full", random_state=42)
    X_pca = pca.fit_transform(X)

    clf = SVC(
        kernel=args.kernel,
        C=args.C,
        gamma=args.gamma,
        probability=True,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_pca, y)

    y_pred = clf.predict(X_pca)
    cm = confusion_matrix(y, y_pred)
    report_dict = classification_report(y, y_pred, digits=4, output_dict=True)
    report_text = classification_report(y, y_pred, digits=4)

    print("\nPCA output dimensionality:", X_pca.shape[1])
    print("\nTraining-set confusion matrix:")
    print(cm)
    print("\nTraining-set classification report:")
    print(report_text)

    metrics["training_set"] = {
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "n_features_after_pca": int(X_pca.shape[1]),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    n_participants = int(data["n_participants"][0]) if "n_participants" in data else None
    participant_names = data["participant_names"] if "participant_names" in data else None
    features_per_participant = int(data["features_per_participant"][0]) if "features_per_participant" in data else None

    participant_scaler_means = []
    participant_scaler_scales = []

    if n_participants is not None:
        for p in range(n_participants):
            participant_scaler_means.append(data[f"scaler_mean_p{p+1}"].astype(np.float32))
            participant_scaler_scales.append(data[f"scaler_scale_p{p+1}"].astype(np.float32))

    model_package = {
        "model_type": "pca_svm_grouped",
        "pca": pca,
        "svm": clf,
        "n_features_before_pca": int(X.shape[1]),
        "n_features_after_pca": int(X_pca.shape[1]),
        "classes": np.unique(y),
        "n_participants": n_participants,
        "participant_names": participant_names,
        "features_per_participant": features_per_participant,
        "participant_scaler_means": participant_scaler_means,
        "participant_scaler_scales": participant_scaler_scales,
    }

    joblib.dump(model_package, args.output)
    print("\nSaved trained PCA+SVM model to:", args.output)

    if args.metrics_output is not None:
        os.makedirs(os.path.dirname(args.metrics_output) or ".", exist_ok=True)
        with open(args.metrics_output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print("Saved training metrics to:", args.metrics_output)


if __name__ == "__main__":
    main()