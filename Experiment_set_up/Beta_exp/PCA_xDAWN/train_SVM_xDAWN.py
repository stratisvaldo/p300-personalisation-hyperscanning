# train_SVM_xDAWN.py
# Train grouped xDAWN + SVM directly from grouped raw calibration recording
#
# Pipeline:
# 1. load grouped raw recording
# 2. filter each participant continuous EEG
# 3. fit one StandardScaler per participant on their own filtered continuous EEG
# 4. epoch every flash_on event for all participants together
# 5. optional artifact rejection
# 6. baseline correction
# 7. apply participant-wise standardisation
# 8. downsample
# 9. fit xDAWN separately for each participant
# 10. transform epochs into xDAWN component space
# 11. concatenate all participants
# 12. flatten and train SVM
# 13. report CV and training-set results
#
# This is the more P300-specific comparison to your current plain SVM.

'''
python Experiment_set_up/Beta_exp/PCA_xDAWN/train_SVM_xDAWN.py \
  --input Experiment_set_up/Beta_exp/Results/group_calibration_recording.npz \
  --output Experiment_set_up/Beta_exp/PCA_xDAWN/models_svm_XDawn/svm_group_xdawn.joblib \
  --metrics_output Experiment_set_up/Beta_exp/PCA_xDAWN/models_svm_XDawn/svm_group_xdawn_metrics.json \
  --tmin 0.0 \
  --tmax 0.8 \
  --baseline 0.0 \
  --lowcut 0.1 \
  --highcut 20.0 \
  --filter_order 4 \
  --downsample_factor 4 \
  --artifact_n_sd 2 \
  --max_channels 10 \
  --xdawn_components 2 \
  --kernel linear \
  --C 1.0
'''
import os
import json
import argparse
import joblib
import numpy as np
import mne

from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from mne.preprocessing import Xdawn
from mne import create_info
from mne.epochs import EpochsArray


def load_recording(path):
    return np.load(path, allow_pickle=True)


def find_epoch_sample_range(eeg_timestamps, t_start, t_end):
    i0 = np.searchsorted(eeg_timestamps, t_start, side="left")
    i1 = np.searchsorted(eeg_timestamps, t_end, side="left")
    return i0, i1


def baseline_correct(epoch, baseline_samples):
    if baseline_samples <= 0:
        return epoch
    baseline = epoch[:baseline_samples].mean(axis=0, keepdims=True)
    return epoch - baseline


def resample_epoch_if_needed(epoch, target_len):
    old_len = epoch.shape[0]
    if old_len == target_len:
        return epoch

    x_old = np.linspace(0, 1, old_len)
    x_new = np.linspace(0, 1, target_len)

    out = np.zeros((target_len, epoch.shape[1]), dtype=np.float32)
    for ch in range(epoch.shape[1]):
        out[:, ch] = np.interp(x_new, x_old, epoch[:, ch])

    return out


def bandpass_filter_continuous_eeg(eeg, srate, lowcut=0.1, highcut=20.0, order=4):
    nyq = 0.5 * srate
    if lowcut <= 0:
        raise ValueError("lowcut must be > 0")
    if highcut >= nyq:
        raise ValueError(f"highcut must be below Nyquist ({nyq})")

    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="bandpass")
    eeg_filt = filtfilt(b, a, eeg, axis=0)
    return eeg_filt.astype(np.float32)


def compute_channelwise_sd_thresholds(eeg, n_sd=4.0):
    channel_stds = np.std(eeg, axis=0)
    thresholds = n_sd * channel_stds
    return thresholds.astype(np.float32), channel_stds.astype(np.float32)


def reject_artifact(epoch, channel_thresholds):
    return np.any(np.max(np.abs(epoch), axis=0) > channel_thresholds)


def downsample_epoch(epoch, factor):
    if factor <= 1:
        return epoch
    return epoch[::factor, :]


def preprocess_group_recording(
    recording_path,
    tmin,
    tmax,
    baseline,
    expected_srate,
    lowcut,
    highcut,
    filter_order,
    downsample_factor,
    artifact_n_sd,
    max_channels,
    drop_incomplete=False,
):
    data = load_recording(recording_path)

    n_participants = int(data["n_participants"][0])
    participant_names = data["participant_names"]
    participant_srates = data["participant_srates"]

    marker_timestamps = data["marker_timestamps"]
    event_names = data["event_names"]
    event_is_target = data["event_is_target"]

    participant_eeg = []
    participant_ts = []
    used_srates = []
    used_nchans = []
    participant_thresholds = []
    participant_scalers = []
    participant_scaler_means = []
    participant_scaler_scales = []

    for p in range(n_participants):
        eeg_key = f"eeg_samples_p{p+1}"
        ts_key = f"eeg_timestamps_p{p+1}"

        eeg_samples = data[eeg_key].astype(np.float32)
        eeg_timestamps = data[ts_key].astype(np.float64)

        n_use_ch = min(max_channels, eeg_samples.shape[1])
        eeg_samples = eeg_samples[:, :n_use_ch]

        eeg_srate = float(participant_srates[p])
        srate_to_use = float(expected_srate) if expected_srate is not None else eeg_srate

        eeg_samples = bandpass_filter_continuous_eeg(
            eeg_samples,
            srate=eeg_srate,
            lowcut=lowcut,
            highcut=highcut,
            order=filter_order,
        )

        ch_thresholds, _ = compute_channelwise_sd_thresholds(eeg_samples, n_sd=artifact_n_sd)

        scaler = StandardScaler()
        scaler.fit(eeg_samples)

        participant_eeg.append(eeg_samples)
        participant_ts.append(eeg_timestamps)
        used_srates.append(srate_to_use)
        used_nchans.append(n_use_ch)
        participant_thresholds.append(ch_thresholds)
        participant_scalers.append(scaler)
        participant_scaler_means.append(scaler.mean_.astype(np.float32))
        participant_scaler_scales.append(scaler.scale_.astype(np.float32))

    if len(set(used_nchans)) != 1:
        raise RuntimeError(f"Participants do not all have same number of used channels: {used_nchans}")

    if len(set([round(x, 6) for x in used_srates])) != 1:
        raise RuntimeError(f"Participants do not all have same effective srate: {used_srates}")

    n_channels = used_nchans[0]
    srate = used_srates[0]

    epoch_len_sec = tmax - tmin
    expected_len = int(round(epoch_len_sec * srate))
    baseline_samples = int(round(baseline * srate))

    X = []
    y = []

    total_flash_on = 0
    kept_epochs = 0
    dropped_epochs = 0
    rejected_artifacts = 0

    for i in range(len(event_names)):
        if str(event_names[i]) != "flash_on":
            continue

        total_flash_on += 1
        marker_time = float(marker_timestamps[i])
        label = int(event_is_target[i])

        if label not in (0, 1):
            dropped_epochs += 1
            continue

        epoch_per_participant = []
        valid_group_epoch = True
        reject_group_epoch = False

        for p in range(n_participants):
            eeg_samples = participant_eeg[p]
            eeg_timestamps = participant_ts[p]
            channel_thresholds = participant_thresholds[p]
            scaler = participant_scalers[p]

            t_start = marker_time + tmin
            t_end = marker_time + tmax
            i0, i1 = find_epoch_sample_range(eeg_timestamps, t_start, t_end)

            if i0 >= len(eeg_samples) or i1 > len(eeg_samples) or i1 <= i0:
                valid_group_epoch = False
                break

            epoch = eeg_samples[i0:i1, :]  # (T, C)

            if epoch.shape[0] != expected_len:
                if drop_incomplete:
                    valid_group_epoch = False
                    break
                epoch = resample_epoch_if_needed(epoch, expected_len)

            if reject_artifact(epoch, channel_thresholds):
                reject_group_epoch = True
                break

            if baseline_samples > 0:
                if baseline_samples >= epoch.shape[0]:
                    valid_group_epoch = False
                    break
                epoch = baseline_correct(epoch, baseline_samples)

            epoch = scaler.transform(epoch)
            epoch = downsample_epoch(epoch, downsample_factor)

            # xDAWN expects (epochs, channels, times)
            epoch_per_participant.append(epoch.T.astype(np.float32))  # (C, T)

        if not valid_group_epoch:
            dropped_epochs += 1
            continue

        if reject_group_epoch:
            rejected_artifacts += 1
            continue

        X.append(np.stack(epoch_per_participant, axis=0))  # (P, C, T)
        y.append(label)
        kept_epochs += 1

    if len(X) == 0:
        raise RuntimeError("No valid grouped epochs were extracted.")

    X = np.stack(X, axis=0).astype(np.float32)  # (N, P, C, T)
    y = np.asarray(y, dtype=np.int64)

    print("\nPreprocessed grouped training data")
    print("Loaded participants    :", n_participants)
    print("Participant names      :", participant_names)
    print("Used channels each     :", n_channels)
    print("Using srate            :", srate)
    print("Epoch window           :", tmin, "to", tmax)
    print("Bandpass               :", lowcut, "to", highcut)
    print("Artifact thresholding  :", artifact_n_sd, "SD per channel")
    print("Downsample factor      :", downsample_factor)
    print("X grouped shape        :", X.shape)
    print("Kept epochs            :", kept_epochs)
    print("Dropped epochs         :", dropped_epochs)
    print("Rejected artifacts     :", rejected_artifacts)
    print("Target count           :", int((y == 1).sum()))
    print("Non-target count       :", int((y == 0).sum()))

    sfreq_after_downsample = srate / downsample_factor

    return (
        X,
        y,
        n_participants,
        participant_names,
        participant_scaler_means,
        participant_scaler_scales,
        sfreq_after_downsample,
    )


def make_mne_epochs(X, y, sfreq, tmin=0.0):
    N, C, T = X.shape

    ch_names = [f"EEG{i+1}" for i in range(C)]
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

    events = np.column_stack([
        np.arange(N, dtype=int),
        np.zeros(N, dtype=int),
        y.astype(int) + 1
    ])

    # Only include event ids that are actually present
    unique_ids = np.unique(y.astype(int) + 1)
    id_map = {1: "non_target", 2: "target"}
    event_id = {id_map[i]: i for i in unique_ids if i in id_map}

    epochs = EpochsArray(
        X,
        info,
        events=events,
        event_id=event_id,
        tmin=tmin,
        baseline=None,
        verbose=False,
    )

    return epochs


def fit_xdawn_transform(X_group_train, y_train, X_group_eval, sfreq, tmin, n_components):
    """
    X_group_train shape: (N, P, C, T)
    X_group_eval  shape: (M, P, C, T)

    Fit one xDAWN object per participant on training only.
    """
    _, P, _, _ = X_group_train.shape

    xdawn_list = []
    transformed_train = []
    transformed_eval = []

    for p in range(P):
        Xtr_p = X_group_train[:, p, :, :]   # (N, C, T)
        Xev_p = X_group_eval[:, p, :, :]    # (M, C, T)

        epochs_tr = make_mne_epochs(Xtr_p, y_train, sfreq=sfreq, tmin=tmin)

        # dummy labels only to create eval EpochsArray
        dummy_eval_y = np.zeros(Xev_p.shape[0], dtype=int)
        epochs_ev = make_mne_epochs(Xev_p, dummy_eval_y, sfreq=sfreq, tmin=tmin)

        xd = Xdawn(n_components=n_components)
        xd.fit(epochs_tr)

        Xtr_xd = xd.transform(epochs_tr).astype(np.float32)
        Xev_xd = xd.transform(epochs_ev).astype(np.float32)

        print(f"Participant {p+1} xDAWN train shape: {Xtr_xd.shape}")

        xdawn_list.append(xd)
        transformed_train.append(Xtr_xd)
        transformed_eval.append(Xev_xd)

    Xtr = np.stack(transformed_train, axis=1)  # (N, P, Cxd, T)
    Xev = np.stack(transformed_eval, axis=1)   # (M, P, Cxd, T)

    return xdawn_list, Xtr, Xev


def flatten_grouped_epochs(X_group):
    """
    X_group shape: (N, P, C, T)
    Flatten to (N, Features)
    """
    return X_group.reshape(X_group.shape[0], -1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Path to grouped raw calibration recording .npz")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save trained xDAWN+SVM model .joblib")
    parser.add_argument("--metrics_output", type=str, default=None,
                        help="Optional path to save metrics .json")

    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=0.8)
    parser.add_argument("--baseline", type=float, default=0.0)

    parser.add_argument("--expected_srate", type=float, default=None)
    parser.add_argument("--lowcut", type=float, default=0.1)
    parser.add_argument("--highcut", type=float, default=20.0)
    parser.add_argument("--filter_order", type=int, default=4)
    parser.add_argument("--downsample_factor", type=int, default=4)
    parser.add_argument("--artifact_n_sd", type=float, default=4.0)
    parser.add_argument("--max_channels", type=int, default=8)

    parser.add_argument("--xdawn_components", type=int, default=2)
    parser.add_argument("--kernel", type=str, default="linear",
                        choices=["linear", "rbf"])
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--gamma", type=str, default="scale")
    parser.add_argument("--cv_folds", type=int, default=5)

    parser.add_argument("--drop_incomplete", action="store_true")
    args = parser.parse_args()

    (
        X_group,
        y,
        n_participants,
        participant_names,
        participant_scaler_means,
        participant_scaler_scales,
        sfreq_after_downsample,
    ) = preprocess_group_recording(
        recording_path=args.input,
        tmin=args.tmin,
        tmax=args.tmax,
        baseline=args.baseline,
        expected_srate=args.expected_srate,
        lowcut=args.lowcut,
        highcut=args.highcut,
        filter_order=args.filter_order,
        downsample_factor=args.downsample_factor,
        artifact_n_sd=args.artifact_n_sd,
        max_channels=args.max_channels,
        drop_incomplete=args.drop_incomplete,
    )

    n_target = int((y == 1).sum())
    n_nontarget = int((y == 0).sum())

    metrics = {
        "input_file": args.input,
        "model_output": args.output,
        "model_type": "xDAWN + SVM",
        "kernel": args.kernel,
        "C": args.C,
        "gamma": args.gamma,
        "xdawn_components": int(args.xdawn_components),
        "requested_cv_folds": int(args.cv_folds),
        "grouped_input_shape": list(X_group.shape),
        "n_samples": int(len(X_group)),
        "class_counts": {
            "target_1": n_target,
            "non_target_0": n_nontarget,
        },
        "standardisation": "participant-wise standardisation was fit on each participant filtered continuous EEG",
        "sfreq_after_downsample": float(sfreq_after_downsample),
    }

    max_possible_folds = min(n_target, n_nontarget)

    if max_possible_folds >= 2:
        cv_folds = min(args.cv_folds, max_possible_folds)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        cv_scores = []

        for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_group, y), start=1):
            X_tr = X_group[tr_idx]
            y_tr = y[tr_idx]
            X_va = X_group[va_idx]
            y_va = y[va_idx]

            _, X_tr_xd, X_va_xd = fit_xdawn_transform(
                X_tr,
                y_tr,
                X_va,
                sfreq=sfreq_after_downsample,
                tmin=args.tmin,
                n_components=args.xdawn_components,
            )

            X_tr_flat = flatten_grouped_epochs(X_tr_xd)
            X_va_flat = flatten_grouped_epochs(X_va_xd)

            clf_fold = SVC(
                kernel=args.kernel,
                C=args.C,
                gamma=args.gamma,
                probability=True,
                class_weight="balanced",
                random_state=42,
            )
            clf_fold.fit(X_tr_flat, y_tr)

            y_va_pred = clf_fold.predict(X_va_flat)
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

    xdawn_list, X_xd, _ = fit_xdawn_transform(
        X_group,
        y,
        X_group,
        sfreq=sfreq_after_downsample,
        tmin=args.tmin,
        n_components=args.xdawn_components,
    )

    X_flat = flatten_grouped_epochs(X_xd)

    clf = SVC(
        kernel=args.kernel,
        C=args.C,
        gamma=args.gamma,
        probability=True,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_flat, y)

    y_pred = clf.predict(X_flat)
    cm = confusion_matrix(y, y_pred)
    report_dict = classification_report(y, y_pred, digits=4, output_dict=True)
    report_text = classification_report(y, y_pred, digits=4)

    print("\nFlattened feature shape after xDAWN:", X_flat.shape)
    print("\nTraining-set confusion matrix:")
    print(cm)
    print("\nTraining-set classification report:")
    print(report_text)

    metrics["training_set"] = {
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "n_features_after_xdawn": int(X_flat.shape[1]),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    model_package = {
        "model_type": "xdawn_svm_grouped",
        "xdawn_list": xdawn_list,
        "svm": clf,
        "classes": np.unique(y),
        "n_participants": n_participants,
        "participant_names": participant_names,
        "participant_scaler_means": participant_scaler_means,
        "participant_scaler_scales": participant_scaler_scales,
        "n_features_after_xdawn": int(X_flat.shape[1]),
        "sfreq_after_downsample": float(sfreq_after_downsample),
        "tmin": float(args.tmin),
        "tmax": float(args.tmax),
        "baseline": float(args.baseline),
        "lowcut": float(args.lowcut),
        "highcut": float(args.highcut),
        "filter_order": int(args.filter_order),
        "downsample_factor": int(args.downsample_factor),
        "max_channels": int(args.max_channels),
        "xdawn_components": int(args.xdawn_components),
    }

    joblib.dump(model_package, args.output)
    print("\nSaved trained xDAWN+SVM model to:", args.output)

    if args.metrics_output is not None:
        os.makedirs(os.path.dirname(args.metrics_output) or ".", exist_ok=True)
        with open(args.metrics_output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print("Saved training metrics to:", args.metrics_output)


if __name__ == "__main__":
    main()