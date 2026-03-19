# Load saved calibration recording
# Find every flash_on marker
# Filter continuous EEG from 0.1 to 20 Hz
# Extract an EEG epoch around that marker
# Label each epoch as target or non-target
# Optionally reject artifacted epochs
# Downsample each epoch by a factor of 4
# Flatten each epoch to 1 feature vector
# Save a new .npz with X,y metadata

# script saves X as (N_epochs, Features)

'''
Pre-processing
1. filter 
2. epoching
3. reject bad or artifacted epochs (amplitude thresholding + per channel SD)
4. baseline correction (subtract channel-wise mean of baseline window at epoch start)
5. downsample
6. flatten to (N, Features)
7. Standardise each feature using training-set statistics (SVM does this)
Traditional ML: standardize each feature (each channel-timepoint combination in the flattened vector) independently, since SVM/LDA treats each dimension separately

python Epoch_extraction/epoch_extraction_traditional_ml.py `
  --input Receiver_script/data/calib_play_01.npz `
  --output Receiver_script/data/p300_epochs_play_01_tradml.npz `
  --tmin 0.0 `
  --tmax 0.8 `
  --baseline 0.0 `
  --lowcut 0.1 `
  --highcut 20.0 `
  --downsample_factor 4 `
  --artifact_n_sd 4
'''

import os
import argparse
import numpy as np
from scipy.signal import butter, filtfilt


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
    """
    eeg shape: (N_samples, N_channels)
    """
    channel_stds = np.std(eeg, axis=0)
    thresholds = n_sd * channel_stds
    return thresholds.astype(np.float32), channel_stds.astype(np.float32)


def reject_artifact(epoch, channel_thresholds):
    """
    epoch shape: (T, C)
    channel_thresholds shape: (C,)
    Reject if any channel exceeds its own threshold.
    """
    return np.any(np.max(np.abs(epoch), axis=0) > channel_thresholds)


def downsample_epoch(epoch, factor):
    if factor <= 1:
        return epoch
    return epoch[::factor, :]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="p300_epochs_traditional_ml.npz")
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=0.8)
    parser.add_argument("--baseline", type=float, default=0.0)
    parser.add_argument("--expected_srate", type=float, default=None)
    parser.add_argument("--drop_incomplete", action="store_true")

    parser.add_argument("--lowcut", type=float, default=0.1)
    parser.add_argument("--highcut", type=float, default=20.0)
    parser.add_argument("--filter_order", type=int, default=4)

    parser.add_argument(
        "--artifact_n_sd",
        type=float,
        default=4.0,
        help="Reject epoch if any channel exceeds N standard deviations of that channel"
    )
    parser.add_argument("--downsample_factor", type=int, default=4)

    args = parser.parse_args()

    data = load_recording(args.input)

    eeg_samples = data["eeg_samples"][:, :8].astype(np.float32)
    eeg_timestamps = data["eeg_timestamps"]
    eeg_srate = float(data["eeg_srate"][0])

    markers_raw = data["markers_raw"]
    marker_timestamps = data["marker_timestamps"]
    event_names = data["event_names"]
    event_kinds = data["event_kinds"]
    event_idxs = data["event_idxs"]
    event_is_target = data["event_is_target"]
    event_target_chars = data["event_target_chars"]
    event_seqs = data["event_seqs"]
    event_flashes = data["event_flashes"]

    srate = float(args.expected_srate) if args.expected_srate is not None else eeg_srate
    epoch_len_sec = args.tmax - args.tmin
    expected_len = int(round(epoch_len_sec * srate))
    baseline_samples = int(round(args.baseline * srate))

    print(f"Loaded EEG shape      : {eeg_samples.shape}")
    print(f"Loaded EEG srate      : {eeg_srate}")
    print(f"Using srate           : {srate}")
    print(f"Epoch window          : {args.tmin:.3f} to {args.tmax:.3f} sec")
    print(f"Expected epoch length : {expected_len} samples")
    print(f"Baseline samples      : {baseline_samples}")
    print(f"Bandpass              : {args.lowcut:.3f} to {args.highcut:.3f} Hz")
    print(f"Artifact thresholding : {args.artifact_n_sd} SD per channel")
    print(f"Downsample factor     : {args.downsample_factor}")

    eeg_samples = bandpass_filter_continuous_eeg(
        eeg_samples,
        srate=eeg_srate,
        lowcut=args.lowcut,
        highcut=args.highcut,
        order=args.filter_order,
    )

    channel_thresholds, channel_stds = compute_channelwise_sd_thresholds(
        eeg_samples,
        n_sd=args.artifact_n_sd
    )

    print(f"Channel STDs          : {channel_stds}")
    print(f"Channel thresholds    : {channel_thresholds}")

    X = []
    y = []

    meta_marker_raw = []
    meta_marker_time = []
    meta_kind = []
    meta_idx = []
    meta_target_char = []
    meta_seq = []
    meta_flash = []

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

        t_start = marker_time + args.tmin
        t_end = marker_time + args.tmax
        i0, i1 = find_epoch_sample_range(eeg_timestamps, t_start, t_end)

        if i0 >= len(eeg_samples) or i1 > len(eeg_samples) or i1 <= i0:
            dropped_epochs += 1
            continue

        epoch = eeg_samples[i0:i1, :]   # (T, C)

        if epoch.shape[0] != expected_len:
            if args.drop_incomplete:
                dropped_epochs += 1
                continue
            epoch = resample_epoch_if_needed(epoch, expected_len)

        if reject_artifact(epoch, channel_thresholds):
            rejected_artifacts += 1
            continue

        if baseline_samples > 0:
            if baseline_samples >= epoch.shape[0]:
                dropped_epochs += 1
                continue
            epoch = baseline_correct(epoch, baseline_samples)

        epoch = downsample_epoch(epoch, args.downsample_factor)
        features = epoch.reshape(-1).astype(np.float32)

        X.append(features)
        y.append(label)

        meta_marker_raw.append(str(markers_raw[i]))
        meta_marker_time.append(marker_time)
        meta_kind.append(str(event_kinds[i]) if event_kinds[i] is not None else "")
        meta_idx.append(int(event_idxs[i]))
        meta_target_char.append(str(event_target_chars[i]) if event_target_chars[i] is not None else "")
        meta_seq.append(int(event_seqs[i]))
        meta_flash.append(int(event_flashes[i]))

        kept_epochs += 1

    if len(X) == 0:
        raise RuntimeError("No valid epochs were extracted.")

    X = np.stack(X, axis=0)
    y = np.asarray(y, dtype=np.int64)

    effective_srate = srate / args.downsample_factor
    n_features = X.shape[1]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    np.savez(
        args.output,
        X=X,
        y=y,
        srate=np.array([effective_srate], dtype=np.float32),
        original_srate=np.array([srate], dtype=np.float32),
        tmin=np.array([args.tmin], dtype=np.float32),
        tmax=np.array([args.tmax], dtype=np.float32),
        baseline=np.array([args.baseline], dtype=np.float32),
        lowcut=np.array([args.lowcut], dtype=np.float32),
        highcut=np.array([args.highcut], dtype=np.float32),
        filter_order=np.array([args.filter_order], dtype=np.int32),
        artifact_n_sd=np.array([args.artifact_n_sd], dtype=np.float32),
        channel_stds=channel_stds,
        channel_thresholds=channel_thresholds,
        downsample_factor=np.array([args.downsample_factor], dtype=np.int32),
        n_features=np.array([n_features], dtype=np.int32),
        marker_raw=np.asarray(meta_marker_raw, dtype=object),
        marker_time=np.asarray(meta_marker_time, dtype=np.float64),
        kind=np.asarray(meta_kind, dtype=object),
        idx=np.asarray(meta_idx, dtype=np.int32),
        target_char=np.asarray(meta_target_char, dtype=object),
        seq=np.asarray(meta_seq, dtype=np.int32),
        flash=np.asarray(meta_flash, dtype=np.int32),
    )

    print("\nSaved traditional ML dataset to:", args.output)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print(f"Total flash_on markers: {total_flash_on}")
    print(f"Kept epochs          : {kept_epochs}")
    print(f"Dropped epochs       : {dropped_epochs}")
    print(f"Rejected artifacts   : {rejected_artifacts}")
    print(f"Target count         : {(y == 1).sum()}")
    print(f"Non-target count     : {(y == 0).sum()}")
    print(f"Effective srate      : {effective_srate}")
    print(f"Features per epoch   : {n_features}")


if __name__ == "__main__":
    main()