import os
import argparse
import random
import numpy as np
import torch

from braindecode import EEGClassifier
from braindecode.models import EEGNet
from braindecode.util import set_random_seeds

from skorch.callbacks import EpochScoring, EarlyStopping
from skorch.helper import predefined_split

from sklearn.metrics import balanced_accuracy_score, accuracy_score, classification_report, make_scorer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset


'''
- loads the epoch dataset from extraction file
- converts data from (N, T, C) to (N, C, T) for EEGNet
- splits data into train/valid sets
- builds EEGNet model and EEGClassifier wrapper
- initialises model weights
- trains the model 
- prints validation performance 
- saves the trained model 
'''

'''
How to run it:
Train from calibration data only
python train_eegnet_calibration.py \
  --epochs_path data/p300_epochs_play_01.npz \
  --out_path models_eegnet/eegnet_calib_only.pkl

Train and initialise from saved weights:
python train_eegnet_calibration.py \
  --epochs_path data/p300_epochs_play_01.npz \
  --out_path models_eegnet/eegnet_calib_finetuned.pkl \
  --use_init_weights \
  --init_weights_path models_eegnet/eegnet_small_hpo_23epochs_seed2025.pkl


'''
# ---------------------------------------------------------------------
# Fixed hyperparameters from HPO

lr_eeg = 0.000724795606355317
batch_size_eeg = 128
weight_decay_eeg = 1.3356750400520492e-05
drop_prob_eeg = 0.44539302556010774
patience_eeg = 10
MAX_EPOCHS = 100

save_dir = "models_eegnet"
seed = 2025
run_id = "small_hpo_23epochs"

path_eegnet = os.path.join(save_dir, f"eegnet_{run_id}_seed{seed}.pkl")


# ---------------------------------------------------------------------
# Scoring callbacks

train_bacc_cb = EpochScoring(
    scoring=make_scorer(balanced_accuracy_score),
    on_train=True,
    name="train_bacc",
    lower_is_better=False,
)

valid_bacc_cb = EpochScoring(
    scoring=make_scorer(balanced_accuracy_score),
    on_train=False,
    name="valid_bacc",
    lower_is_better=False,
)


# ---------------------------------------------------------------------
# Model builder

def make_eegnet_clf(
    n_chans,
    n_times,
    n_classes,
    device,
    lr,
    max_epochs,
    batch_size,
    weight_decay,
    patience,
    drop_prob,
    valid_ds=None,
    monitor="valid_loss",
):
    model = EEGNet(
        n_chans=n_chans,
        n_outputs=n_classes,
        n_times=n_times,
        final_conv_length="auto",
        drop_prob=drop_prob,
    ).to(device)

    callbacks = [
        ("train_bacc", train_bacc_cb),
        ("valid_bacc", valid_bacc_cb),
    ]

    train_split = None
    if valid_ds is not None:
        train_split = predefined_split(valid_ds)
        callbacks.append(
            (
                "early_stopping",
                EarlyStopping(
                    monitor=monitor,
                    patience=patience,
                    lower_is_better=True,
                    threshold=1e-4,
                    threshold_mode="rel",
                    load_best=True,
                ),
            )
        )

    clf = EEGClassifier(
        model,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        device=device,
        iterator_train__shuffle=True,
        iterator_train__drop_last=False,
        iterator_valid__drop_last=False,
        train_split=train_split,
        classes=list(range(n_classes)),
        callbacks=callbacks,
    )
    return clf


# ---------------------------------------------------------------------
# Utilities

def set_all_seeds(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    set_random_seeds(seed=seed_value, cuda=torch.cuda.is_available())


def load_epoch_data(path):
    data = np.load(path, allow_pickle=True)

    X = data["X"]   # expected shape: (N, T, C)
    y = data["y"]   # expected labels: 0 / 1

    print("Loaded epoch file:", path)
    print("Original X shape:", X.shape)
    print("Original y shape:", y.shape)

    # Convert from (N, T, C) to (N, C, T) for EEGNet
    X = np.transpose(X, (0, 2, 1)).astype(np.float32)
    y = y.astype(np.int64)

    print("Transposed X shape for EEGNet:", X.shape)

    return X, y


def make_tensor_dataset(X, y):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    return TensorDataset(X_t, y_t)


def evaluate_model(clf, X, y, split_name="valid"):
    y_pred = clf.predict(X)
    bacc = balanced_accuracy_score(y, y_pred)
    acc = accuracy_score(y, y_pred)

    print(f"\n{split_name} results:")
    print(f"  accuracy          : {acc:.4f}")
    print(f"  balanced accuracy : {bacc:.4f}")
    print("\nClassification report:")
    print(classification_report(y, y_pred, digits=4))

    return acc, bacc


# ---------------------------------------------------------------------
# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs_path", type=str, required=True,
                        help="Path to extracted epoch dataset .npz")
    parser.add_argument("--out_path", type=str, default=os.path.join(save_dir, "eegnet_calibration_trained.pkl"),
                        help="Where to save trained EEGClassifier")
    parser.add_argument("--init_weights_path", type=str, default=path_eegnet,
                        help="Optional path to existing EEGNet weights/classifier file")
    parser.add_argument("--use_init_weights", action="store_true",
                        help="Initialize model from existing saved weights")
    parser.add_argument("--valid_size", type=float, default=0.2,
                        help="Validation fraction")
    parser.add_argument("--monitor", type=str, default="valid_loss",
                        help="Early stopping monitor, e.g. valid_loss")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)

    set_all_seeds(seed)

    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    print("device:", device)

    X, y = load_epoch_data(args.epochs_path)

    n_trials, n_chans, n_times = X.shape
    n_classes = len(np.unique(y))

    print("\nDataset summary:")
    print("  n_trials :", n_trials)
    print("  n_chans  :", n_chans)
    print("  n_times  :", n_times)
    print("  n_classes:", n_classes)
    print("  class counts:", {int(c): int((y == c).sum()) for c in np.unique(y)})

    # Stratified split because P300 is imbalanced
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=args.valid_size,
        random_state=seed,
        stratify=y,
    )

    print("\nSplit summary:")
    print("  X_train:", X_train.shape)
    print("  X_valid:", X_valid.shape)
    print("  y_train:", y_train.shape)
    print("  y_valid:", y_valid.shape)

    train_ds = make_tensor_dataset(X_train, y_train)
    valid_ds = make_tensor_dataset(X_valid, y_valid)

    clf = make_eegnet_clf(
        n_chans=n_chans,
        n_times=n_times,
        n_classes=n_classes,
        device=device,
        lr=lr_eeg,
        max_epochs=MAX_EPOCHS,
        batch_size=batch_size_eeg,
        weight_decay=weight_decay_eeg,
        patience=patience_eeg,
        drop_prob=drop_prob_eeg,
        valid_ds=valid_ds,
        monitor=args.monitor,
    )

    # Initialize network parameters/shapes before optional load
    clf.initialize()

    if args.use_init_weights:
        print("\nTrying to initialize from:", args.init_weights_path)
        print("Exists:", os.path.exists(args.init_weights_path))

        if not os.path.exists(args.init_weights_path):
            raise FileNotFoundError(f"Init weights file not found: {args.init_weights_path}")

        # This assumes the file is a skorch/Braindecode saved object compatible with load_params
        clf.load_params(f_params=args.init_weights_path)
        print("Loaded initialization weights successfully.")

    print("\nStarting training...")
    clf.fit(X_train, y_train)

    # Evaluate
    evaluate_model(clf, X_train, y_train, split_name="train")
    evaluate_model(clf, X_valid, y_valid, split_name="valid")

    print("\nSaving trained model to:", args.out_path)
    clf.save_params(f_params=args.out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()