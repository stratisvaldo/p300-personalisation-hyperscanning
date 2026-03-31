import argparse
import numpy as np
from scipy.signal import butter, filtfilt

'''
Works with:
1. old single-participant receiver format:
   eeg_samples, eeg_timestamps, eeg_srate

2. group receiver format:
   eeg_samples_p1, eeg_timestamps_p1, ...
   participant_srates, participant_nchans, participant_names

Example:
python checkSD.py \
  --input data_calib/group_calibration_recording.npz \
  --participant_idx 1 \
  --tmin 0.0 \
  --tmax 0.8 \
  --lowcut 0.1 \
  --highcut 20.0

5-15% is a reasonable starting point
'''


def load_recording(path):
    return np.load(path, allow_pickle=True)


def find_epoch_sample_range(eeg_timestamps, t_start, t_end):
    i0 = np.searchsorted(eeg_timestamps, t_start, side="left")
    i1 = np.searchsorted(eeg_timestamps, t_end, side="left")
    return i0, i1


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


def compute_channelwise_sd(eeg):
    return np.std(eeg, axis=0).astype(np.float32)


def reject_artifact(epoch, channel_thresholds):
    return np.any(np.max(np.abs(epoch), axis=0) > channel_thresholds)


def get_eeg_from_file(data, participant_idx=1, use_first_n_channels=8):
    """
    Supports:
    - single format: eeg_samples, eeg_timestamps, eeg_srate
    - group format: eeg_samples_p1, eeg_timestamps_p1, participant_srates
    """

    # old single-participant format
    if "eeg_samples" in data and "eeg_timestamps" in data and "eeg_srate" in data:
        eeg_samples = data["eeg_samples"][:, :use_first_n_channels].astype(np.float32)
        eeg_timestamps = data["eeg_timestamps"].astype(np.float64)

        eeg_srate_raw = data["eeg_srate"]
        if np.ndim(eeg_srate_raw) == 0:
            eeg_srate = float(eeg_srate_raw)
        else:
            eeg_srate = float(eeg_srate_raw[0])

        source_name = "single-participant file"
        return eeg_samples, eeg_timestamps, eeg_srate, source_name

    # group format
    eeg_key = f"eeg_samples_p{participant_idx}"
    ts_key = f"eeg_timestamps_p{participant_idx}"

    if eeg_key in data and ts_key in data:
        eeg_samples = data[eeg_key][:, :use_first_n_channels].astype(np.float32)
        eeg_timestamps = data[ts_key].astype(np.float64)

        if "participant_srates" not in data:
            raise KeyError("participant_srates not found in group recording file.")

        participant_srates = data["participant_srates"]
        eeg_srate = float(participant_srates[participant_idx - 1])

        if "participant_names" in data:
            participant_names = data["participant_names"]
            source_name = str(participant_names[participant_idx - 1])
        else:
            source_name = f"participant_{participant_idx}"

        return eeg_samples, eeg_timestamps, eeg_srate, source_name

    raise KeyError(
        "Could not find compatible EEG keys in file. "
        "Expected either single format or group format."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to raw calibration recording .npz")
    parser.add_argument("--participant_idx", type=int, default=1, help="Used only for group receiver files")
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=0.8)
    parser.add_argument("--expected_srate", type=float, default=None)
    parser.add_argument("--lowcut", type=float, default=0.1)
    parser.add_argument("--highcut", type=float, default=20.0)
    parser.add_argument("--filter_order", type=int, default=4)
    parser.add_argument("--use_first_n_channels", type=int, default=8)
    parser.add_argument("--drop_incomplete", action="store_true")
    args = parser.parse_args()

    data = load_recording(args.input)

    eeg_samples, eeg_timestamps, eeg_srate, source_name = get_eeg_from_file(
        data,
        participant_idx=args.participant_idx,
        use_first_n_channels=args.use_first_n_channels
    )

    if "marker_timestamps" not in data or "event_names" not in data or "event_is_target" not in data:
        raise KeyError("Marker/event arrays not found in file.")

    marker_timestamps = data["marker_timestamps"]
    event_names = data["event_names"]
    event_is_target = data["event_is_target"]

    srate = float(args.expected_srate) if args.expected_srate is not None else eeg_srate
    epoch_len_sec = args.tmax - args.tmin
    expected_len = int(round(epoch_len_sec * srate))

    print(f"Loaded source         : {source_name}")
    print(f"Loaded EEG shape      : {eeg_samples.shape}")
    print(f"Loaded EEG srate      : {eeg_srate}")
    print(f"Using srate           : {srate}")
    print(f"Epoch window          : {args.tmin:.3f} to {args.tmax:.3f} sec")
    print(f"Expected epoch length : {expected_len} samples")
    print(f"Bandpass              : {args.lowcut:.3f} to {args.highcut:.3f} Hz")
    print(f"Channels used         : first {args.use_first_n_channels}")

    eeg_samples = bandpass_filter_continuous_eeg(
        eeg_samples,
        srate=eeg_srate,
        lowcut=args.lowcut,
        highcut=args.highcut,
        order=args.filter_order,
    )

    channel_stds = compute_channelwise_sd(eeg_samples)

    print("\nChannel SDs from filtered continuous calibration EEG:")
    for ch, sd in enumerate(channel_stds):
        print(f"  Ch {ch:02d}: {sd:.4f}")

    epochs = []
    labels = []

    total_flash_on = 0
    dropped_epochs = 0

    for i in range(len(event_names)):
        if str(event_names[i]) != "flash_on":
            continue

        total_flash_on += 1
        label = int(event_is_target[i])

        if label not in (0, 1):
            dropped_epochs += 1
            continue

        marker_time = float(marker_timestamps[i])
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

        epochs.append(epoch.astype(np.float32))
        labels.append(label)

    if len(epochs) == 0:
        raise RuntimeError("No valid epochs were extracted.")

    epochs = np.stack(epochs, axis=0)   # (N, T, C)
    labels = np.asarray(labels, dtype=np.int64)

    print(f"\nTotal flash_on markers: {total_flash_on}")
    print(f"Usable epochs         : {len(epochs)}")
    print(f"Dropped epochs        : {dropped_epochs}")
    print(f"Target count          : {(labels == 1).sum()}")
    print(f"Non-target count      : {(labels == 0).sum()}")

    multipliers = [2, 3, 4, 5]

    print("\nRejection rates by SD multiplier:")
    print("multiplier | rejected | kept | rejected_%")

    for m in multipliers:
        channel_thresholds = m * channel_stds
        rejected_mask = np.array([reject_artifact(ep, channel_thresholds) for ep in epochs], dtype=bool)

        n_rejected = int(rejected_mask.sum())
        n_kept = int((~rejected_mask).sum())
        rejected_pct = 100.0 * n_rejected / len(epochs)

        print(f"±{m:<9} | {n_rejected:<8} | {n_kept:<4} | {rejected_pct:>9.2f}")

    print("\nRule of thumb:")
    print("A rejection rate around 5 to 15% is often a reasonable starting point.")
    print("Example: if ±4 rejects 5% and ±3 rejects 20%, ±4 is usually the safer choice.")


if __name__ == "__main__":
    main()