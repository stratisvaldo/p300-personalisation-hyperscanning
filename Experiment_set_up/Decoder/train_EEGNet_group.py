# Train EEGNet on grouped CNN epochs
# Input X must have shape (N, T, TotalChannels)
# TotalChannels = n_participants * n_chans_per_participant

'''
python train_EEGNet_group.py \
  --epochs_path epoched_data/group_epochs_cnn.npz \
  --model_out models_eegnet/eegnet_group.pkl \
  --norm_out models_eegnet/eegnet_group_norm.npz \
  --meta_out models_eegnet/eegnet_group_meta.json \
  --metrics_out models_eegnet/eegnet_group_metrics.json \
  --lr 0.0007 \
  --batch_size 128 \
  --weight_decay 1e-5 \
  --drop_prob 0.45
'''

import os
import json
import argparse
import numpy as np
import torch

from braindecode import EEGClassifier
from braindecode.models import EEGNet
from braindecode.util import set_random_seeds

from skorch.callbacks import EpochScoring, EarlyStopping
from skorch.helper import predefined_split

from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score, classification_report, confusion_matrix
from torch.utils.data import TensorDataset


def reshape_grouped_X_to_pp(X, n_participants, n_chans_per_participant):
    """
    Convert grouped EEG from:
        (N, T, C_total)
    to:
        (N, P, Cpp, T)

    where:
        P   = n_participants
        Cpp = n_chans_per_participant
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (N, T, C_total), got {X.shape}")

    N, T, C_total = X.shape
    expected = n_participants * n_chans_per_participant
    if C_total != expected:
        raise ValueError(
            f"C_total={C_total} does not match "
            f"n_participants * n_chans_per_participant = {expected}"
        )

    X_pp = X.reshape(N, T, n_participants, n_chans_per_participant)
    X_pp = np.transpose(X_pp, (0, 2, 3, 1))  # (N, P, Cpp, T)
    return X_pp.astype(np.float32)


def reshape_pp_to_grouped_X(X_pp):
    """
    Convert participant-structured EEG from:
        (N, P, Cpp, T)
    back to:
        (N, T, C_total)
    """
    if X_pp.ndim != 4:
        raise ValueError(f"Expected X_pp with shape (N, P, Cpp, T), got {X_pp.shape}")

    N, P, Cpp, T = X_pp.shape
    X = np.transpose(X_pp, (0, 3, 1, 2))   # (N, T, P, Cpp)
    X = X.reshape(N, T, P * Cpp)
    return X.astype(np.float32)


def fit_participantwise_zscore(X_pp):
    """
    Fit z-score stats separately for each participant and channel.

    Input:
        X_pp: (N, P, Cpp, T)

    Returns:
        mean_pp: (P, Cpp, 1)
        std_pp : (P, Cpp, 1)
    """
    if X_pp.ndim != 4:
        raise ValueError(f"Expected X_pp with shape (N, P, Cpp, T), got {X_pp.shape}")

    mean_pp = X_pp.mean(axis=(0, 3), keepdims=False)[:, :, None].astype(np.float32)
    std_pp = X_pp.std(axis=(0, 3), keepdims=False)[:, :, None].astype(np.float32)
    std_pp[std_pp < 1e-8] = 1.0
    return mean_pp, std_pp


def apply_participantwise_zscore(X_pp, mean_pp, std_pp):
    """
    Apply participant-wise z-score.

    X_pp   : (N, P, Cpp, T)
    mean_pp: (P, Cpp, 1)
    std_pp : (P, Cpp, 1)
    """
    if X_pp.ndim != 4:
        raise ValueError(f"Expected X_pp with shape (N, P, Cpp, T), got {X_pp.shape}")

    return ((X_pp - mean_pp[None, :, :, :]) / std_pp[None, :, :, :]).astype(np.float32)


def shuffle_trials(X, y, rng):
    """
    Randomly shuffle the order of trials (epochs).

    Input:
        X  : (N, C, T)
        y  : (N,)
        rng: numpy random generator

    Returns:
        X_shuf: (N, C, T)
        y_shuf: (N,)
        idx   : (N,) shuffle indices — idx[i] is the original trial index
                that ended up at position i after shuffling
    """
    idx = rng.permutation(len(X))
    return X[idx], y[idx], idx


def make_eegnet_clf(n_chans, n_times, n_classes, device, lr, batch_size, weight_decay, drop_prob, valid_ds):
    model = EEGNet(
        n_chans=n_chans,
        n_outputs=n_classes,
        n_times=n_times,
        final_conv_length="auto",
        drop_prob=drop_prob,
    ).to(device)

    clf = EEGClassifier(
        model,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=100,
        device=device,
        train_split=predefined_split(valid_ds),
        classes=list(range(n_classes)),
        callbacks=[
            ("valid_acc", EpochScoring(
                scoring="accuracy",
                lower_is_better=False,
                on_train=False,
                name="valid_acc",
            )),
            ("early_stop", EarlyStopping(
                monitor="valid_loss",
                patience=10,
                lower_is_better=True,
            )),
        ],
    )
    return clf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs_path", type=str, required=True)
    parser.add_argument("--model_out", type=str, required=True)
    parser.add_argument("--norm_out", type=str, required=True)
    parser.add_argument("--meta_out", type=str, required=True)
    parser.add_argument("--metrics_out", type=str, required=True)

    parser.add_argument("--valid_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2025)

    parser.add_argument("--lr", type=float, default=0.000724795606355317)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=1.3356750400520492e-05)
    parser.add_argument("--drop_prob", type=float, default=0.44539302556010774)

    args = parser.parse_args()

    set_random_seeds(seed=args.seed, cuda=torch.cuda.is_available())
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = np.load(args.epochs_path, allow_pickle=True)
    X = data["X"].astype(np.float32)   # (N, T, C_total)
    y = data["y"].astype(np.int64)

    if X.ndim != 3:
        raise RuntimeError(f"Expected X with shape (N, T, C), got {X.shape}")

    if len(X) != len(y):
        raise RuntimeError("X and y have different number of samples")

    n_participants = int(data["n_participants"][0]) if "n_participants" in data else 1
    n_chans_per_participant = int(data["n_chans_per_participant"][0]) if "n_chans_per_participant" in data else X.shape[2]
    n_total_channels = int(data["n_total_channels"][0]) if "n_total_channels" in data else X.shape[2]
    srate = float(data["srate"][0]) if "srate" in data else np.nan

    print("Loaded epoch file:", args.epochs_path)
    print("Original X shape:", X.shape)
    print("Original y shape:", y.shape)
    print("Participants:", n_participants)
    print("Channels per participant:", n_chans_per_participant)
    print("Total channels:", n_total_channels)
    print("srate:", srate)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=args.valid_size,
        random_state=args.seed,
        stratify=y,
    )

    # Reshape from grouped channels into participant blocks
    # X_*_pp shape: (N, P, Cpp, T)
    X_train_pp = reshape_grouped_X_to_pp(X_train, n_participants, n_chans_per_participant)
    X_valid_pp = reshape_grouped_X_to_pp(X_valid, n_participants, n_chans_per_participant)

    # Fit participant-wise normalization on training split only
    mean_pp, std_pp = fit_participantwise_zscore(X_train_pp)

    # Apply participant-wise normalization
    X_train_pp = apply_participantwise_zscore(X_train_pp, mean_pp, std_pp)
    X_valid_pp = apply_participantwise_zscore(X_valid_pp, mean_pp, std_pp)

    # Convert back to grouped channel format
    # (N, P, Cpp, T) -> (N, T, C_total)
    X_train = reshape_pp_to_grouped_X(X_train_pp)
    X_valid = reshape_pp_to_grouped_X(X_valid_pp)

    # EEGNet expects (N, C, T)
    X_train = np.transpose(X_train, (0, 2, 1)).astype(np.float32)
    X_valid = np.transpose(X_valid, (0, 2, 1)).astype(np.float32)

    # Shuffle trial order in training set
    X_train, y_train, train_shuffle_idx = shuffle_trials(X_train, y_train, rng)

    print("Train shape:", X_train.shape)
    print("Valid shape:", X_valid.shape)
    print("mean_pp shape:", mean_pp.shape)
    print("std_pp shape:", std_pp.shape)
    print("Class counts train:", {int(c): int((y_train == c).sum()) for c in np.unique(y_train)})
    print("Class counts valid:", {int(c): int((y_valid == c).sum()) for c in np.unique(y_valid)})

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    valid_ds = TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid))

    n_samples, n_chans, n_times = X_train.shape
    n_classes = 2

    clf = make_eegnet_clf(
        n_chans=n_chans,
        n_times=n_times,
        n_classes=n_classes,
        device=device,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        drop_prob=args.drop_prob,
        valid_ds=valid_ds,
    )

    clf.fit(train_ds, y=None)

    y_valid_pred = clf.predict(X_valid)
    valid_bacc = balanced_accuracy_score(y_valid, y_valid_pred)
    valid_acc = accuracy_score(y_valid, y_valid_pred)
    valid_cm = confusion_matrix(y_valid, y_valid_pred)
    valid_report_dict = classification_report(y_valid, y_valid_pred, digits=4, output_dict=True)
    valid_report_text = classification_report(y_valid, y_valid_pred, digits=4)

    print("\nValidation balanced accuracy:", float(valid_bacc))
    print("Validation accuracy:", float(valid_acc))
    print("\nValidation classification report:")
    print(valid_report_text)

    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.norm_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.meta_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_out) or ".", exist_ok=True)

    clf.save_params(f_params=args.model_out)

    # Save participant-wise normalization stats
    np.savez(
        args.norm_out,
        mean_pp=mean_pp,
        std_pp=std_pp,
        n_participants=np.array([n_participants], dtype=np.int64),
        n_chans_per_participant=np.array([n_chans_per_participant], dtype=np.int64),
    )

    meta = {
        "n_participants": n_participants,
        "n_chans_per_participant": n_chans_per_participant,
        "n_total_channels": n_total_channels,
        "n_times": int(n_times),
        "n_classes": n_classes,
        "srate": srate,
        "tmin": float(data["tmin"][0]) if "tmin" in data else None,
        "tmax": float(data["tmax"][0]) if "tmax" in data else None,
        "baseline": float(data["baseline"][0]) if "baseline" in data else None,
        "lowcut": float(data["lowcut"][0]) if "lowcut" in data else None,
        "highcut": float(data["highcut"][0]) if "highcut" in data else None,
        "filter_order": int(data["filter_order"][0]) if "filter_order" in data else None,
        "drop_prob": args.drop_prob,
        "normalization": "participantwise",
        "trial_shuffle": True,
    }

    with open(args.meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    metrics = {
        "epochs_path": args.epochs_path,
        "model_out": args.model_out,
        "norm_out": args.norm_out,
        "meta_out": args.meta_out,
        "seed": args.seed,
        "valid_size": args.valid_size,
        "device": device,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "drop_prob": args.drop_prob,
        "n_samples_total": int(len(X)),
        "n_train": int(len(X_train)),
        "n_valid": int(len(X_valid)),
        "n_chans": int(n_chans),
        "n_times": int(n_times),
        "n_classes": int(n_classes),
        "class_counts_train": {str(int(c)): int((y_train == c).sum()) for c in np.unique(y_train)},
        "class_counts_valid": {str(int(c)): int((y_valid == c).sum()) for c in np.unique(y_valid)},
        "trial_shuffle": True,
        "train_shuffle_idx": train_shuffle_idx.tolist(),
        "validation": {
            "balanced_accuracy": float(valid_bacc),
            "accuracy": float(valid_acc),
            "confusion_matrix": valid_cm.tolist(),
            "classification_report": valid_report_dict,
        }
    }

    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved model to:", args.model_out)
    print("Saved normalization stats to:", args.norm_out)
    print("Saved metadata to:", args.meta_out)
    print("Saved metrics to:", args.metrics_out)


if __name__ == "__main__":
    main()