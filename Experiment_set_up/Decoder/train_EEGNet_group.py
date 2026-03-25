# Train EEGNet on grouped CNN epochs
# Input X must have shape (N, T, TotalChannels)
# TotalChannels = n_participants * n_chans_per_participant

'''
python train_EEGNet_group.py \
  --epochs_path epoched_data/group_epochs_cnn.npz \
  --model_out models_eegnet/eegnet_group.pkl \
  --norm_out models_eegnet/eegnet_group_norm.npz \
  --meta_out models_eegnet/eegnet_group_meta.json
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
from sklearn.metrics import balanced_accuracy_score, accuracy_score, classification_report
from torch.utils.data import TensorDataset


def apply_channelwise_normalizer(X, mean_ch, std_ch):
    return ((X - mean_ch) / std_ch).astype(np.float32)


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

    parser.add_argument("--valid_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2025)

    parser.add_argument("--lr", type=float, default=0.000724795606355317)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=1.3356750400520492e-05)
    parser.add_argument("--drop_prob", type=float, default=0.44539302556010774)

    args = parser.parse_args()

    set_random_seeds(seed=args.seed, cuda=torch.cuda.is_available())
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

    # (N, T, C) -> (N, C, T)
    X = np.transpose(X, (0, 2, 1)).astype(np.float32)
    print("Transposed X shape for EEGNet:", X.shape)

    n_samples, n_chans, n_times = X.shape
    n_classes = 2

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=args.valid_size,
        random_state=args.seed,
        stratify=y,
    )

    # channel-wise normalization from training split only
    mean_ch = X_train.mean(axis=(0, 2), keepdims=True).astype(np.float32)   # (1, C, 1)
    std_ch = X_train.std(axis=(0, 2), keepdims=True).astype(np.float32)
    std_ch[std_ch < 1e-8] = 1.0

    X_train = apply_channelwise_normalizer(X_train, mean_ch, std_ch)
    X_valid = apply_channelwise_normalizer(X_valid, mean_ch, std_ch)

    print("Train shape:", X_train.shape)
    print("Valid shape:", X_valid.shape)
    print("mean_ch shape:", mean_ch.shape)
    print("std_ch shape:", std_ch.shape)
    print("Class counts train:", {int(c): int((y_train == c).sum()) for c in np.unique(y_train)})
    print("Class counts valid:", {int(c): int((y_valid == c).sum()) for c in np.unique(y_valid)})

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    valid_ds = TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid))

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

    print("\nValidation balanced accuracy:", float(valid_bacc))
    print("Validation accuracy:", float(valid_acc))
    print("\nValidation classification report:")
    print(classification_report(y_valid, y_valid_pred, digits=4))

    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.norm_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.meta_out) or ".", exist_ok=True)

    clf.save_params(f_params=args.model_out)
    np.savez(args.norm_out, mean_ch=mean_ch, std_ch=std_ch)

    meta = {
        "n_participants": n_participants,
        "n_chans_per_participant": n_chans_per_participant,
        "n_total_channels": n_total_channels,
        "n_times": n_times,
        "n_classes": n_classes,
        "srate": srate,
        "tmin": float(data["tmin"][0]) if "tmin" in data else None,
        "tmax": float(data["tmax"][0]) if "tmax" in data else None,
        "baseline": float(data["baseline"][0]) if "baseline" in data else None,
        "lowcut": float(data["lowcut"][0]) if "lowcut" in data else None,
        "highcut": float(data["highcut"][0]) if "highcut" in data else None,
        "filter_order": int(data["filter_order"][0]) if "filter_order" in data else None,
        "drop_prob": args.drop_prob,
    }

    with open(args.meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved model to:", args.model_out)
    print("Saved normalization stats to:", args.norm_out)
    print("Saved metadata to:", args.meta_out)


if __name__ == "__main__":
    main()