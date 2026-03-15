# Load saved calibration recording 
# Find every flash_on marker 
# Extract an EEG epoch around that marker 
# Label each epoch as target or non-target
# Save a new .npz with X,y metadata

# script saved X as (N_epochs, Timepoints, Channels)

'''
python extract_p300_epochs.py --input data/calib_play_01.npz --output data/p300_epochs_play_01.npz --tmin 0.0 --tmax 0.8
'''

import os
import argparse
import numpy as np


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="p300_epochs.npz")
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=0.8)
    parser.add_argument("--baseline", type=float, default=0.0)
    parser.add_argument("--expected_srate", type=float, default=None)
    parser.add_argument("--drop_incomplete", action="store_true")
    args = parser.parse_args()

    data = load_recording(args.input)

    eeg_samples = data["eeg_samples"]
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

    if args.expected_srate is not None:
        srate = float(args.expected_srate)
    else:
        srate = eeg_srate

    epoch_len_sec = args.tmax - args.tmin
    expected_len = int(round(epoch_len_sec * srate))
    baseline_samples = int(round(args.baseline * srate))

    print(f"Loaded EEG shape: {eeg_samples.shape}")
    print(f"Loaded EEG srate: {eeg_srate}")
    print(f"Using srate     : {srate}")
    print(f"Epoch window    : {args.tmin:.3f} to {args.tmax:.3f} sec")
    print(f"Expected length : {expected_len} samples")
    print(f"Baseline        : {baseline_samples} samples")

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

        epoch = eeg_samples[i0:i1, :]

        if epoch.shape[0] != expected_len:
            if args.drop_incomplete:
                dropped_epochs += 1
                continue
            epoch = resample_epoch_if_needed(epoch, expected_len)

        if baseline_samples > 0:
            if baseline_samples >= epoch.shape[0]:
                dropped_epochs += 1
                continue
            epoch = baseline_correct(epoch, baseline_samples)

        X.append(epoch.astype(np.float32))
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

    X = np.stack(X, axis=0)   # (N, T, C)
    y = np.asarray(y, dtype=np.int64)

    meta_marker_raw = np.asarray(meta_marker_raw, dtype=object)
    meta_marker_time = np.asarray(meta_marker_time, dtype=np.float64)
    meta_kind = np.asarray(meta_kind, dtype=object)
    meta_idx = np.asarray(meta_idx, dtype=np.int32)
    meta_target_char = np.asarray(meta_target_char, dtype=object)
    meta_seq = np.asarray(meta_seq, dtype=np.int32)
    meta_flash = np.asarray(meta_flash, dtype=np.int32)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    np.savez(
        args.output,
        X=X,
        y=y,
        srate=np.array([srate], dtype=np.float32),
        tmin=np.array([args.tmin], dtype=np.float32),
        tmax=np.array([args.tmax], dtype=np.float32),
        baseline=np.array([args.baseline], dtype=np.float32),
        marker_raw=meta_marker_raw,
        marker_time=meta_marker_time,
        kind=meta_kind,
        idx=meta_idx,
        target_char=meta_target_char,
        seq=meta_seq,
        flash=meta_flash,
    )

    print("\nSaved epoch dataset to:", args.output)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print(f"Total flash_on markers: {total_flash_on}")
    print(f"Kept epochs          : {kept_epochs}")
    print(f"Dropped epochs       : {dropped_epochs}")

    n_target = int((y == 1).sum())
    n_nontarget = int((y == 0).sum())

    print("\nClass counts:")
    print(f"  target     : {n_target}")
    print(f"  non-target : {n_nontarget}")


if __name__ == "__main__":
    main()