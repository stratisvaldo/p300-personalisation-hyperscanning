
''''
Example:
python Experiment_set_up/Epoch_extraction/group_epochML.py `
  --input data_calib/group_calibration_recording.npz `
  --output epoched_data/p300_group_epochs_tradml_flat.npz `
  --tmin 0.0 `
  --tmax 0.8 `
  --baseline 0.0 `
  --lowcut 0.1 `
  --highcut 20.0 `
  --filter_order 4 `
  --downsample_factor 4 `
  --artifact_n_sd 4 `
  --max_channels 8
'''

# Load saved group calibration recording
# Find every flash_on marker
# For each participant:
#   filter continuous EEG from 0.1 to 20 Hz
#   fit one StandardScaler on that participant's own filtered continuous EEG
#   extract an EEG epoch around that marker
#   optionally reject artifacted epochs
#   baseline correct
#   standardise that participant epoch using that participant's own scaler
#   downsample
# Concatenate all participant epochs for the same flash in fixed order
# Flatten to 1 feature vector
# Save a new .npz with X,y metadata and participant-wise scaler stats

'''
Pre-processing
1. load grouped calibration recording from receiver_cal_group.py
2. filter each participant continuous EEG
3. fit one StandardScaler per participant on their own filtered continuous EEG
4. for each flash_on marker, epoch every participant around the same marker
5. reject bad grouped epochs if any participant exceeds artifact threshold
6. baseline correction per participant (optional)
7. standardise each participant epoch with that participant's own scaler
8. downsample
9. flatten grouped epoch to (N, Features)
10. save X, y and metadata

This creates one trial = all participants together in one flattened vector.
Participant order stays fixed across all epochs.
'''

import os
import argparse
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler


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
    parser.add_argument("--output", type=str, default="p300_group_epochs_traditional_ml_flat.npz")

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
        help="Reject grouped epoch if any participant/channel exceeds N standard deviations of that participant channel"
    )
    parser.add_argument("--downsample_factor", type=int, default=4)
    parser.add_argument("--max_channels", type=int, default=8)

    args = parser.parse_args()

    data = load_recording(args.input)

    n_participants = int(data["n_participants"][0])
    participant_names = data["participant_names"]
    participant_srates = data["participant_srates"]
    participant_nchans = data["participant_nchans"]

    markers_raw = data["markers_raw"]
    marker_timestamps = data["marker_timestamps"]
    event_names = data["event_names"]
    event_kinds = data["event_kinds"]
    event_idxs = data["event_idxs"]
    event_is_target = data["event_is_target"]
    event_target_chars = data["event_target_chars"]
    event_seqs = data["event_seqs"]
    event_flashes = data["event_flashes"]

    participant_eeg = []
    participant_ts = []
    used_srates = []
    used_nchans = []
    participant_thresholds = []
    participant_stds = []
    participant_scalers = []
    participant_scaler_means = []
    participant_scaler_scales = []

    for p in range(n_participants):
        eeg_key = f"eeg_samples_p{p+1}"
        ts_key = f"eeg_timestamps_p{p+1}"

        eeg_samples = data[eeg_key].astype(np.float32)
        eeg_timestamps = data[ts_key].astype(np.float64)

        n_use_ch = min(args.max_channels, eeg_samples.shape[1])
        eeg_samples = eeg_samples[:, :n_use_ch]

        eeg_srate = float(participant_srates[p])
        srate_to_use = float(args.expected_srate) if args.expected_srate is not None else eeg_srate

        eeg_samples = bandpass_filter_continuous_eeg(
            eeg_samples,
            srate=eeg_srate,
            lowcut=args.lowcut,
            highcut=args.highcut,
            order=args.filter_order,
        )

        ch_thresholds, ch_stds = compute_channelwise_sd_thresholds(
            eeg_samples,
            n_sd=args.artifact_n_sd
        )

        scaler = StandardScaler()
        scaler.fit(eeg_samples)

        participant_eeg.append(eeg_samples)
        participant_ts.append(eeg_timestamps)
        used_srates.append(srate_to_use)
        used_nchans.append(n_use_ch)
        participant_thresholds.append(ch_thresholds)
        participant_stds.append(ch_stds)
        participant_scalers.append(scaler)
        participant_scaler_means.append(scaler.mean_.astype(np.float32))
        participant_scaler_scales.append(scaler.scale_.astype(np.float32))

    if len(set(used_nchans)) != 1:
        raise RuntimeError(f"Participants do not all have same number of used channels: {used_nchans}")

    if len(set([round(x, 6) for x in used_srates])) != 1:
        raise RuntimeError(f"Participants do not all have same effective srate: {used_srates}")

    n_channels = used_nchans[0]
    srate = used_srates[0]

    epoch_len_sec = args.tmax - args.tmin
    expected_len = int(round(epoch_len_sec * srate))
    baseline_samples = int(round(args.baseline * srate))

    print(f"Loaded participants    : {n_participants}")
    print(f"Participant names      : {participant_names}")
    print(f"Used channels each     : {n_channels}")
    print(f"Using srate            : {srate}")
    print(f"Epoch window           : {args.tmin:.3f} to {args.tmax:.3f} sec")
    print(f"Expected epoch length  : {expected_len} samples")
    print(f"Baseline samples       : {baseline_samples}")
    print(f"Bandpass               : {args.lowcut:.3f} to {args.highcut:.3f} Hz")
    print(f"Artifact thresholding  : {args.artifact_n_sd} SD per channel")
    print(f"Downsample factor      : {args.downsample_factor}")
    print("Participant order      : fixed input order")

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

        epoch_per_participant = []
        valid_group_epoch = True
        reject_group_epoch = False

        for p in range(n_participants):
            eeg_samples = participant_eeg[p]
            eeg_timestamps = participant_ts[p]
            channel_thresholds = participant_thresholds[p]
            scaler = participant_scalers[p]

            t_start = marker_time + args.tmin
            t_end = marker_time + args.tmax
            i0, i1 = find_epoch_sample_range(eeg_timestamps, t_start, t_end)

            if i0 >= len(eeg_samples) or i1 > len(eeg_samples) or i1 <= i0:
                valid_group_epoch = False
                break

            epoch = eeg_samples[i0:i1, :]   # (T, C)

            if epoch.shape[0] != expected_len:
                if args.drop_incomplete:
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
            epoch = downsample_epoch(epoch, args.downsample_factor)
            epoch_per_participant.append(epoch.astype(np.float32))

        if not valid_group_epoch:
            dropped_epochs += 1
            continue

        if reject_group_epoch:
            rejected_artifacts += 1
            continue

        grouped_epoch_parts = [
            epoch_per_participant[p].reshape(-1).astype(np.float32)
            for p in range(n_participants)
        ]

        features = np.concatenate(grouped_epoch_parts, axis=0).astype(np.float32)

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
        raise RuntimeError("No valid grouped epochs were extracted.")

    X = np.stack(X, axis=0)
    y = np.asarray(y, dtype=np.int64)

    effective_srate = srate / args.downsample_factor
    n_features = X.shape[1]
    features_per_participant = n_features // n_participants

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    save_dict = {
        "X": X,
        "y": y,

        "n_participants": np.array([n_participants], dtype=np.int32),
        "participant_names": np.asarray(participant_names, dtype=object),
        "participant_srates": np.asarray(participant_srates, dtype=np.float32),
        "participant_nchans": np.asarray(used_nchans, dtype=np.int32),

        "srate": np.array([effective_srate], dtype=np.float32),
        "original_srate": np.array([srate], dtype=np.float32),
        "tmin": np.array([args.tmin], dtype=np.float32),
        "tmax": np.array([args.tmax], dtype=np.float32),
        "baseline": np.array([args.baseline], dtype=np.float32),
        "lowcut": np.array([args.lowcut], dtype=np.float32),
        "highcut": np.array([args.highcut], dtype=np.float32),
        "filter_order": np.array([args.filter_order], dtype=np.int32),
        "artifact_n_sd": np.array([args.artifact_n_sd], dtype=np.float32),
        "downsample_factor": np.array([args.downsample_factor], dtype=np.int32),

        "n_features": np.array([n_features], dtype=np.int32),
        "features_per_participant": np.array([features_per_participant], dtype=np.int32),

        "marker_raw": np.asarray(meta_marker_raw, dtype=object),
        "marker_time": np.asarray(meta_marker_time, dtype=np.float64),
        "kind": np.asarray(meta_kind, dtype=object),
        "idx": np.asarray(meta_idx, dtype=np.int32),
        "target_char": np.asarray(meta_target_char, dtype=object),
        "seq": np.asarray(meta_seq, dtype=np.int32),
        "flash": np.asarray(meta_flash, dtype=np.int32),
    }

    for p in range(n_participants):
        save_dict[f"channel_stds_p{p+1}"] = participant_stds[p]
        save_dict[f"channel_thresholds_p{p+1}"] = participant_thresholds[p]
        save_dict[f"scaler_mean_p{p+1}"] = participant_scaler_means[p]
        save_dict[f"scaler_scale_p{p+1}"] = participant_scaler_scales[p]

    np.savez(args.output, **save_dict)

    print("\nSaved grouped traditional ML dataset to:", args.output)
    print("X shape      :", X.shape)
    print("y shape      :", y.shape)
    print(f"Total flash_on markers: {total_flash_on}")
    print(f"Kept epochs          : {kept_epochs}")
    print(f"Dropped epochs       : {dropped_epochs}")
    print(f"Rejected artifacts   : {rejected_artifacts}")
    print(f"Target count         : {(y == 1).sum()}")
    print(f"Non-target count     : {(y == 0).sum()}")
    print(f"Effective srate      : {effective_srate}")
    print(f"Features per epoch   : {n_features}")
    print(f"Features per participant: {features_per_participant}")


if __name__ == "__main__":
    main()