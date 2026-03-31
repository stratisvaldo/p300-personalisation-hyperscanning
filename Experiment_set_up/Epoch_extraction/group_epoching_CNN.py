# Load saved group calibration recording
# Find every flash_on marker
# For each participant:
#   filter continuous EEG from 0.1 to 20 Hz
#   extract an EEG epoch around that marker
#   optionally reject artifacted epochs
#   baseline correct
# Concatenate all participant epochs along channels
# Save a new .npz with X,y metadata
# script saves X as (N_epochs, Timepoints, TotalChannels)

# Concatenation : epoch is (T,C) per participant -> (T, P*C) for P participants and C channels each
# [ P1_ch1  P1_ch2  ... P1_ch8  P2_ch1  P2_ch2 ... P2_ch8 ... P8_ch8 ]

# each epoch randomises order of participants in the concatenation, so that model cannot learn fixed participant order
# rnd seed 42 for reproducibility

'''
python group_epoching_CNN.py \
  --input data_calib/group_calibration_recording.npz \
  --output epoched_data/group_epochs_cnn.npz \
  --tmin 0.0 \
  --tmax 0.8 \
  --baseline 0.0 \
  --lowcut 0.1 \
  --highcut 20.0 \
  --filter_order 4 \
  --artifact_n_sd 4.0 \
  --random_seed 42

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
    channel_stds = np.std(eeg, axis=0)
    channel_thresholds = n_sd * channel_stds
    return channel_thresholds.astype(np.float32), channel_stds.astype(np.float32)


def reject_artifact(epoch, channel_thresholds):
    max_per_channel = np.max(np.abs(epoch), axis=0)
    return np.any(max_per_channel > channel_thresholds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="p300_group_epochs_cnn.npz")

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
        help="Reject grouped epoch if any participant/channel exceeds N SD"
    )
    parser.add_argument("--max_channels", type=int, default=8)
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Optional seed for reproducible random participant order per epoch"
    )

    args = parser.parse_args()

    rng = np.random.default_rng(args.random_seed)

    data = load_recording(args.input)

    n_participants = int(data["n_participants"][0])
    participant_names = np.asarray(data["participant_names"], dtype=object)
    participant_srates = np.asarray(data["participant_srates"], dtype=np.float32)

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
            n_sd=args.artifact_n_sd,
        )

        participant_eeg.append(eeg_samples)
        participant_ts.append(eeg_timestamps)
        used_srates.append(srate_to_use)
        used_nchans.append(n_use_ch)
        participant_thresholds.append(ch_thresholds)
        participant_stds.append(ch_stds)

    if len(set(used_nchans)) != 1:
        raise RuntimeError(f"Participants do not all have same used channel count: {used_nchans}")

    if len(set([round(x, 6) for x in used_srates])) != 1:
        raise RuntimeError(f"Participants do not all have same effective srate: {used_srates}")

    n_chans_per_participant = int(used_nchans[0])
    srate = float(used_srates[0])

    epoch_len_sec = args.tmax - args.tmin
    expected_len = int(round(epoch_len_sec * srate))
    baseline_samples = int(round(args.baseline * srate))

    print(f"Loaded participants      : {n_participants}")
    print(f"Participant names        : {participant_names}")
    print(f"Channels per participant : {n_chans_per_participant}")
    print(f"Using srate              : {srate}")
    print(f"Expected epoch length    : {expected_len}")
    print(f"Bandpass                 : {args.lowcut} to {args.highcut} Hz")
    print(f"Artifact thresholding    : {args.artifact_n_sd} SD")
    print(f"Baseline samples         : {baseline_samples}")
    print(f"Random seed              : {args.random_seed}")

    X = []
    y = []

    meta_marker_raw = []
    meta_marker_time = []
    meta_kind = []
    meta_idx = []
    meta_target_char = []
    meta_seq = []
    meta_flash = []
    meta_participant_order = []
    meta_participant_names_order = []

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

            epoch_per_participant.append(epoch.astype(np.float32))

        if not valid_group_epoch:
            dropped_epochs += 1
            continue

        if reject_group_epoch:
            rejected_artifacts += 1
            continue

        perm = rng.permutation(n_participants)
        grouped_epoch_parts = [epoch_per_participant[p] for p in perm]

        # concatenate along channels: [(T,C), (T,C), ...] -> (T, P*C)
        grouped_epoch = np.concatenate(grouped_epoch_parts, axis=1).astype(np.float32)

        X.append(grouped_epoch)
        y.append(label)

        meta_marker_raw.append(str(markers_raw[i]))
        meta_marker_time.append(marker_time)
        meta_kind.append(str(event_kinds[i]) if event_kinds[i] is not None else "")
        meta_idx.append(int(event_idxs[i]))
        meta_target_char.append(str(event_target_chars[i]) if event_target_chars[i] is not None else "")
        meta_seq.append(int(event_seqs[i]))
        meta_flash.append(int(event_flashes[i]))
        meta_participant_order.append(perm.astype(np.int32))
        meta_participant_names_order.append(participant_names[perm])

        kept_epochs += 1

    if len(X) == 0:
        raise RuntimeError("No valid grouped CNN epochs were extracted.")

    X = np.stack(X, axis=0)   # (N, T, P*C)
    y = np.asarray(y, dtype=np.int64)

    total_channels = X.shape[2]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    np.savez(
        args.output,
        X=X,
        y=y,
        n_participants=np.array([n_participants], dtype=np.int32),
        participant_names=np.asarray(participant_names, dtype=object),
        participant_srates=np.asarray(participant_srates, dtype=np.float32),
        n_chans_per_participant=np.array([n_chans_per_participant], dtype=np.int32),
        n_total_channels=np.array([total_channels], dtype=np.int32),
        srate=np.array([srate], dtype=np.float32),
        tmin=np.array([args.tmin], dtype=np.float32),
        tmax=np.array([args.tmax], dtype=np.float32),
        baseline=np.array([args.baseline], dtype=np.float32),
        lowcut=np.array([args.lowcut], dtype=np.float32),
        highcut=np.array([args.highcut], dtype=np.float32),
        filter_order=np.array([args.filter_order], dtype=np.int32),
        artifact_n_sd=np.array([args.artifact_n_sd], dtype=np.float32),
        random_seed=np.array([-1 if args.random_seed is None else args.random_seed], dtype=np.int32),
        marker_raw=np.asarray(meta_marker_raw, dtype=object),
        marker_time=np.asarray(meta_marker_time, dtype=np.float64),
        kind=np.asarray(meta_kind, dtype=object),
        idx=np.asarray(meta_idx, dtype=np.int32),
        target_char=np.asarray(meta_target_char, dtype=object),
        seq=np.asarray(meta_seq, dtype=np.int32),
        flash=np.asarray(meta_flash, dtype=np.int32),
        participant_order_per_epoch=np.stack(meta_participant_order, axis=0),
        participant_names_per_epoch=np.asarray(meta_participant_names_order, dtype=object),
    )

    print("\nSaved grouped CNN epoch dataset to:", args.output)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print(f"Total flash_on markers: {total_flash_on}")
    print(f"Kept epochs          : {kept_epochs}")
    print(f"Dropped epochs       : {dropped_epochs}")
    print(f"Rejected artifacts   : {rejected_artifacts}")
    print(f"Target count         : {(y == 1).sum()}")
    print(f"Non-target count     : {(y == 0).sum()}")


if __name__ == "__main__":
    main()