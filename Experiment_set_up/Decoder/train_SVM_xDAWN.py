# Train an xDAWN + SVM on extracted P300 calibration epochs
# Input .npz is expected to contain epochs in (N, P, C, T) format
# as produced by group_epochML_xdawn.py
# y should be 0 for non-target and 1 for target

# For each participant:
#   fit xDAWN spatial filters on their (N, C, T) epochs
#   apply filters -> (N, n_components, T)
#   flatten -> (N, n_components * T)
# Concatenate across participants -> (N, P * n_components * T)
# cross-val (5 stratified folds) -> balanced acc
# confusion matrix, classification report on training data (printed)

'''
Example:
python Experiment_set_up/Decoder/train_svm_xdawn.py `
  --input epoched_data/p300_group_epochs_xdawn.npz `
  --output models_svm/svm_xdawn_group_calibration.joblib `
  --kernel linear `
  --C 1.0 `
  --n_xdawn_components 4 `
  --metrics_output models_svm/svm_xdawn_group_calibration_metrics.json
'''

import os
import json
import argparse
import joblib
import numpy as np

from scipy.linalg import eigh
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix


def shuffle_trials(X, y, rng):
    """
    Randomly shuffle the order of trials (epochs).

    Input:
        X  : (N, Features)
        y  : (N,)
        rng: numpy random generator

    Returns:
        X_shuf: (N, Features)
        y_shuf: (N,)
        idx   : (N,) shuffle indices — idx[i] is the original trial index
                that ended up at position i after shuffling
    """
    idx = rng.permutation(len(X))
    return X[idx], y[idx], idx


def fit_xdawn_filters(epochs_3d, y, n_components):
    """
    Fit xDAWN spatial filters for one participant's epochs.

    xDAWN finds spatial filters W that maximise the signal-to-noise ratio
    of the target evoked response. It solves the generalised eigenvalue problem:

        Sigma_A @ w = lambda * Sigma_X @ w

    where:
        Sigma_A = average target evoked covariance  (C, C)
        Sigma_X = total data covariance             (C, C)

    The top n_components eigenvectors (largest lambda) are kept as filters.
    These are the directions in channel space where the P300 is most
    distinguishable from background noise.

    Parameters
    ----------
    epochs_3d   : ndarray, shape (N, C, T)
                  All epochs for one participant, already standardised and
                  downsampled, as produced by group_epochML_xdawn.py.
    y           : ndarray, shape (N,)  binary labels 0/1
    n_components: int  number of spatial components to retain.
                  Will be clamped to min(n_components, C).

    Returns
    -------
    filters : ndarray, shape (C, n_components)
        Spatial filter matrix.
        Apply to a batch: np.einsum('nct,ck->nkt', epochs_3d, filters) -> (N, n_comp, T)
        Apply online to a single epoch (T, C): epoch @ filters -> (T, n_comp)
    """
    N, C, T = epochs_3d.shape
    n_comp = min(n_components, C)

    if (y == 1).sum() == 0:
        raise RuntimeError("No target epochs found; cannot fit xDAWN filters.")

    # Total data covariance: reshape (N, C, T) -> (N*T, C) then X^T X
    X_2d = epochs_3d.transpose(0, 2, 1).reshape(N * T, C)  # (N*T, C)
    Sigma_X = X_2d.T @ X_2d                                 # (C, C)

    # Average target evoked response across target epochs: (C, T)
    avg_target = epochs_3d[y == 1].mean(axis=0)  # (C, T)

    # Signal covariance based on average target evoked response
    Sigma_A = avg_target @ avg_target.T  # (C, C)

    # Solve the generalised eigenvalue problem
    # eigh returns eigenvalues in ascending order; we want the top n_comp
    eigenvalues, eigenvectors = eigh(Sigma_A, Sigma_X)

    # Reverse to descending order, take top n_comp columns
    filters = eigenvectors[:, -n_comp:][:, ::-1]  # (C, n_comp)

    return filters.astype(np.float32)


def apply_xdawn_filters(epochs_3d, filters):
    """
    Apply xDAWN spatial filters to a participant's epochs.

    Parameters
    ----------
    epochs_3d : ndarray, shape (N, C, T)
    filters   : ndarray, shape (C, n_components)

    Returns
    -------
    ndarray, shape (N, n_components, T)
    """
    # Einsum: for each epoch n, each time t, project channels onto filters
    return np.einsum("nct,ck->nkt", epochs_3d, filters).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Path to xDAWN epoch dataset .npz with X shape (N, P, C, T)")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save trained xDAWN + SVM model .joblib")
    parser.add_argument("--kernel", type=str, default="linear",
                        choices=["linear", "rbf"])
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--gamma", type=str, default="scale",
                        help="Used for RBF kernel, e.g. 'scale' or 'auto'")
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics_output", type=str, default=None,
                        help="Optional path to save training metrics as .json")
    parser.add_argument(
        "--n_xdawn_components",
        type=int,
        default=4,
        help=(
            "Number of xDAWN spatial filter components to retain per participant. "
            "xDAWN is fit independently on each participant's (N, C, T) epochs "
            "before concatenation and SVM training. "
            "Will be clamped to min(n_xdawn_components, n_channels). "
            "Default: 4."
        )
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    data = np.load(args.input, allow_pickle=True)

    # X shape: (N, P, C, T) — one row per flash, participants stacked along axis 1
    X_4d = data["X"]
    y = data["y"]

    print("Loaded xDAWN epoch dataset:", args.input)
    print("X shape (N, P, C, T):", X_4d.shape)
    print("y shape:", y.shape)

    if X_4d.ndim != 4:
        raise RuntimeError(
            f"Expected X to have shape (N, P, C, T), got {X_4d.shape}. "
            f"Use group_epochML_xdawn.py to produce the correct format."
        )

    N, n_participants, n_channels, n_times = X_4d.shape

    if len(X_4d) != len(y):
        raise RuntimeError("X and y have different number of samples")

    X_4d = X_4d.astype(np.float32)

    n_target = int((y == 1).sum())
    n_nontarget = int((y == 0).sum())

    print(f"Participants         : {n_participants}")
    print(f"Channels per participant: {n_channels}")
    print(f"Time points          : {n_times}")
    print("Class counts:")
    print("  target     :", n_target)
    print("  non-target :", n_nontarget)
    print("Input is assumed to already be participant-wise standardised.")
    print("No StandardScaler is applied in training.")

    # --- xDAWN spatial filtering (per participant) ---
    # Each participant gets their own xDAWN filters because EEG spatial topography
    # varies across individuals. Filters are fit on all epochs with their labels,
    # then applied to reduce the channel dimension from C to n_components.
    # The resulting per-participant feature blocks are flattened and concatenated.

    print(f"\nFitting xDAWN with {args.n_xdawn_components} components per participant...")

    xdawn_filters_list = []
    X_parts = []

    for p in range(n_participants):
        # epochs_p: (N, C, T) — this participant's epochs across all trials
        epochs_p = X_4d[:, p, :, :]

        filters_p = fit_xdawn_filters(epochs_p, y, args.n_xdawn_components)
        xdawn_filters_list.append(filters_p)

        n_comp_actual = filters_p.shape[1]
        print(f"  Participant {p+1}: filters {filters_p.shape}  (C={n_channels} -> {n_comp_actual} components)")

        # Apply: (N, C, T) -> (N, n_comp, T)
        epochs_p_xdawn = apply_xdawn_filters(epochs_p, filters_p)

        # Flatten to (N, n_comp * T)
        X_parts.append(epochs_p_xdawn.reshape(N, -1))

    # Concatenate across participants: (N, P * n_comp * T)
    X = np.concatenate(X_parts, axis=1).astype(np.float32)

    n_comp_actual = xdawn_filters_list[0].shape[1]
    features_per_participant = n_comp_actual * n_times
    n_features = int(X.shape[1])

    print(f"\nX shape after xDAWN + flatten: {X.shape}")
    print(f"Features per participant: {features_per_participant}  ({n_comp_actual} comp * {n_times} times)")
    print(f"Total features: {n_features}")

    # Shuffle trial order before cross-validation and fitting
    X, y, shuffle_idx = shuffle_trials(X, y, rng)
    print("Trials shuffled with seed:", args.seed)

    svm = SVC(
        kernel=args.kernel,
        C=args.C,
        gamma=args.gamma,
        probability=True,
        class_weight="balanced",
        random_state=args.seed,
    )

    metrics_dict = {
        "input_file": args.input,
        "model_output": args.output,
        "kernel": args.kernel,
        "C": args.C,
        "gamma": args.gamma,
        "seed": args.seed,
        "requested_cv_folds": int(args.cv_folds),
        "n_samples": int(len(X)),
        "n_features": n_features,
        "n_participants": n_participants,
        "n_channels": n_channels,
        "n_times": n_times,
        "class_counts": {
            "target_1": n_target,
            "non_target_0": n_nontarget,
        },
        "standardisation": "participant-wise standardisation was already applied during epoch extraction",
        "xdawn": {
            "n_xdawn_components_requested": args.n_xdawn_components,
            "n_xdawn_components_actual": n_comp_actual,
            "features_per_participant": features_per_participant,
        },
        "trial_shuffle": True,
        "shuffle_idx": shuffle_idx.tolist(),
    }

    max_possible_folds = min(n_target, n_nontarget)

    if max_possible_folds >= 2:
        cv_folds = min(args.cv_folds, max_possible_folds)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=args.seed)
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

    participant_names = data["participant_names"] if "participant_names" in data else None

    participant_scaler_means = []
    participant_scaler_scales = []

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
        # xDAWN filter matrices, one per participant, each (C, n_components).
        # Online script applies these after standardisation + downsampling,
        # before flattening and concatenation.
        "n_xdawn_components": n_comp_actual,
        "xdawn_filters": xdawn_filters_list,
    }

    joblib.dump(model_package, args.output)
    print("\nSaved trained xDAWN + SVM model to:", args.output)

    if args.metrics_output is not None:
        os.makedirs(os.path.dirname(args.metrics_output) or ".", exist_ok=True)
        with open(args.metrics_output, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=2)
        print("Saved training metrics to:", args.metrics_output)


if __name__ == "__main__":
    main()

