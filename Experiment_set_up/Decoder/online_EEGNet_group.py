# Connect to multiple Unicorn EEG streams
# Connect to marker stream from free-speller
# Load trained grouped EEGNet model
# Load saved channel-wise normalisation statistics
# For every flash_on:
#   wait until all participant streams contain the full post-flash epoch
#   preprocess each participant epoch exactly like training
#   concatenate participants along channels
#   score target probability with EEGNet
# Accumulate evidence across sequences
# Predict the row and column and convert to symbol
# Send decision back through LSL
# Optionally save online decisions for later analysis

'''
python online_EEGNet_group.py \
  --eeg_names Unicorn_P1 Unicorn_P2 Unicorn_P3 Unicorn_P4 Unicorn_P5 Unicorn_P6 Unicorn_P7 Unicorn_P8 \
  --marker_name P300Markers_test \
  --model_path models_eegnet/eegnet_group.pkl \
  --norm_path models_eegnet/eegnet_group_norm.npz \
  --meta_path models_eegnet/eegnet_group_meta.json \
  --tmin 0.0 \
  --tmax 0.8 \
  --baseline 0.0 \
  --lowcut 0.1 \
  --highcut 20.0 \
  --filter_order 4 \
  --expected_srate 250 \
  --n_chans_per_participant 8 \
  --save_decisions data_test/testing_group_eegnet_decisions.npz
'''

import os
import time
import json
import argparse
import numpy as np
import torch

from scipy.signal import butter, filtfilt
from pylsl import resolve_byprop, StreamInlet, StreamInfo, StreamOutlet, local_clock
from braindecode import EEGClassifier
from braindecode.models import EEGNet


GRID_ROWS = 6
GRID_COLS = 6

GRID_SYMBOLS = [
    ["A", "B", "C", "D", "E", "F"],
    ["G", "H", "I", "J", "K", "L"],
    ["M", "N", "O", "P", "Q", "R"],
    ["S", "T", "U", "V", "W", "X"],
    ["Y", "Z", "1", "2", "3", "4"],
    ["5", "6", "7", "8", "9", "0"],
]


def parse_marker(marker: str):
    parts = marker.strip().split("/")

    out = {
        "raw": marker,
        "event": None,
        "kind": None,
        "idx": None,
        "selection": None,
        "seq": None,
        "flash": None,
    }

    if len(parts) == 1:
        out["event"] = parts[0]
        return out

    if len(parts) == 2 and parts[0] in {"selection_start", "selection_end"}:
        out["event"] = parts[0]
        try:
            out["selection"] = int(parts[1])
        except Exception:
            out["selection"] = None
        return out

    if len(parts) == 3 and parts[0] in {"sequence_start", "sequence_end"}:
        out["event"] = parts[0]
        try:
            out["selection"] = int(parts[1])
        except Exception:
            out["selection"] = None
        try:
            out["seq"] = int(parts[2])
        except Exception:
            out["seq"] = None
        return out

    if len(parts) == 6 and parts[0] in {"flash_on", "flash_off"}:
        out["event"] = parts[0]
        out["kind"] = parts[1]

        try:
            out["idx"] = int(parts[2])
        except Exception:
            out["idx"] = None

        if parts[3].startswith("sel_"):
            try:
                out["selection"] = int(parts[3].split("_")[1])
            except Exception:
                out["selection"] = None

        if parts[4].startswith("seq_"):
            try:
                out["seq"] = int(parts[4].split("_")[1])
            except Exception:
                out["seq"] = None

        if parts[5].startswith("flash_"):
            try:
                out["flash"] = int(parts[5].split("_")[1])
            except Exception:
                out["flash"] = None

        return out

    out["event"] = parts[0]
    return out


def make_eegnet_clf(n_chans, n_times, n_classes, device, drop_prob):
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
        batch_size=1,
        max_epochs=1,
        device=device,
        train_split=None,
        classes=list(range(n_classes)),
    )
    return clf


def find_stream(prop, value, timeout=10):
    print(f"Looking for LSL stream with {prop}='{value}' ...")
    streams = resolve_byprop(prop, value, timeout=timeout)
    if not streams:
        raise RuntimeError(f"Could not find stream with {prop}='{value}'")
    return streams[0]


def make_decision_outlet():
    info = StreamInfo(
        name="P300Decisions",
        type="Markers",
        channel_count=1,
        nominal_srate=0,
        channel_format="string",
        source_id="p300-online-group-eegnet-decoder"
    )
    return StreamOutlet(info)


def send_decision(outlet, msg):
    outlet.push_sample([msg], local_clock())
    print("[DECISION]", msg)


def baseline_correct(epoch, baseline_samples):
    if baseline_samples <= 0:
        return epoch
    baseline = epoch[:baseline_samples].mean(axis=0, keepdims=True)
    return epoch - baseline


def bandpass_filter_continuous_eeg(eeg, srate, lowcut=0.1, highcut=20.0, order=4):
    nyq = 0.5 * srate
    if lowcut <= 0:
        raise ValueError("lowcut must be > 0")
    if highcut >= nyq:
        raise ValueError(f"highcut must be below Nyquist ({nyq})")

    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="bandpass")
    eeg_filt = filtfilt(b, a, eeg, axis=0)
    return eeg_filt.astype(np.float32)


def apply_channelwise_normalizer(X, mean, std):
    return ((X - mean) / std).astype(np.float32)


def symbol_from_row_col(row_idx, col_idx):
    if row_idx is None or col_idx is None:
        return None
    if not (0 <= row_idx < GRID_ROWS and 0 <= col_idx < GRID_COLS):
        return None
    return GRID_SYMBOLS[row_idx][col_idx]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eeg_names", type=str, nargs="+", required=True)
    parser.add_argument("--marker_name", type=str, default="P300Markers")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--norm_path", type=str, required=True)
    parser.add_argument("--meta_path", type=str, required=True)

    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=0.8)
    parser.add_argument("--baseline", type=float, default=0.0)

    parser.add_argument("--lowcut", type=float, default=0.1)
    parser.add_argument("--highcut", type=float, default=20.0)
    parser.add_argument("--filter_order", type=int, default=4)

    parser.add_argument("--expected_srate", type=float, default=250.0)
    parser.add_argument("--n_chans_per_participant", type=int, default=8)
    parser.add_argument("--n_classes", type=int, default=2)

    parser.add_argument("--save_decisions", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    with open(args.meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    n_participants_model = int(meta["n_participants"])
    n_chans_per_participant_model = int(meta["n_chans_per_participant"])
    n_total_channels_model = int(meta["n_total_channels"])
    n_times_model = int(meta["n_times"])
    drop_prob = float(meta["drop_prob"])

    n_participants_runtime = len(args.eeg_names)
    if n_participants_runtime != n_participants_model:
        raise RuntimeError(
            f"Model expects {n_participants_model} participants but runtime has {n_participants_runtime}"
        )

    if args.n_chans_per_participant != n_chans_per_participant_model:
        raise RuntimeError(
            f"Model expects {n_chans_per_participant_model} channels per participant "
            f"but runtime argument gives {args.n_chans_per_participant}"
        )

    n_times_runtime = int(round((args.tmax - args.tmin) * args.expected_srate))
    if n_times_runtime != n_times_model:
        raise RuntimeError(
            f"Model expects n_times={n_times_model} but runtime gives n_times={n_times_runtime}"
        )

    n_total_channels_runtime = n_participants_runtime * args.n_chans_per_participant
    if n_total_channels_runtime != n_total_channels_model:
        raise RuntimeError(
            f"Model expects total channels={n_total_channels_model} but runtime gives {n_total_channels_runtime}"
        )

    eeg_inlets = []
    eeg_time_corrections = []
    eeg_buffers_samples = []
    eeg_buffers_timestamps = []

    print("\nConnecting EEG streams...")
    for stream_name in args.eeg_names:
        eeg_stream = find_stream("name", stream_name, timeout=15)
        inlet = StreamInlet(eeg_stream, max_chunklen=64)
        eeg_inlets.append(inlet)
        eeg_buffers_samples.append([])
        eeg_buffers_timestamps.append([])

    marker_stream = find_stream("name", args.marker_name, timeout=15)
    marker_inlet = StreamInlet(marker_stream)
    decision_outlet = make_decision_outlet()

    print("\nComputing LSL time corrections...")
    for p, inlet in enumerate(eeg_inlets):
        tc = inlet.time_correction(timeout=5.0)
        eeg_time_corrections.append(tc)
        info = inlet.info()
        print(f"Participant {p+1}:")
        print(f"  name            : {info.name()}")
        print(f"  nominal_srate   : {info.nominal_srate()}")
        print(f"  channel_count   : {info.channel_count()}")
        print(f"  time_correction : {tc:.6f} s")

    marker_time_correction = marker_inlet.time_correction(timeout=5.0)
    print(f"\nMarker time correction: {marker_time_correction:.6f} s")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\nLoading model...")
    clf = make_eegnet_clf(
        n_chans=n_total_channels_model,
        n_times=n_times_model,
        n_classes=args.n_classes,
        device=device,
        drop_prob=drop_prob,
    )
    clf.initialize()
    clf.load_params(f_params=args.model_path)
    print("Loaded model from:", args.model_path)

    norm_data = np.load(args.norm_path)
    mean_ch = norm_data["mean_ch"].astype(np.float32)
    std_ch = norm_data["std_ch"].astype(np.float32)
    print("Loaded normalisation stats from:", args.norm_path)
    print("mean_ch shape:", mean_ch.shape)
    print("std_ch shape :", std_ch.shape)

    baseline_samples = int(round(args.baseline * args.expected_srate))

    current_selection = None
    row_scores = np.zeros(GRID_ROWS, dtype=np.float64)
    col_scores = np.zeros(GRID_COLS, dtype=np.float64)

    log_selection = []
    log_best_row = []
    log_best_col = []
    log_symbol = []
    log_row_scores = []
    log_col_scores = []
    log_decision_time = []

    log_flash_selection = []
    log_flash_seq = []
    log_flash_number = []
    log_flash_kind = []
    log_flash_idx = []
    log_flash_target_prob = []
    log_flash_time = []
    log_flash_row_scores_after = []
    log_flash_col_scores_after = []

    print("\nGrouped EEGNet decoder is running...\n")

    experiment_ended = False

    try:
        while True:
            # continuously collect new EEG for all participants
            for p, inlet in enumerate(eeg_inlets):
                chunk, ts = inlet.pull_chunk(timeout=0.0, max_samples=128)
                if ts:
                    for sample, t in zip(chunk, ts):
                        eeg_buffers_samples[p].append(sample[:args.n_chans_per_participant])
                        eeg_buffers_timestamps[p].append(t + eeg_time_corrections[p])

                if len(eeg_buffers_timestamps[p]) > 50000:
                    eeg_buffers_samples[p] = eeg_buffers_samples[p][-30000:]
                    eeg_buffers_timestamps[p] = eeg_buffers_timestamps[p][-30000:]

            while True:
                sample, mts = marker_inlet.pull_sample(timeout=0.0)
                if mts is None:
                    break

                marker = sample[0]
                parsed = parse_marker(marker)
                print("[MARKER]", marker)

                if parsed["event"] == "selection_start":
                    current_selection = parsed["selection"]
                    row_scores[:] = 0.0
                    col_scores[:] = 0.0
                    print(f"\nStarted selection {current_selection}\n")
                    continue

                if parsed["event"] == "flash_on":
                    if parsed["kind"] not in {"row", "col"}:
                        continue
                    if parsed["idx"] is None:
                        continue

                    corrected_mts = float(mts) + marker_time_correction
                    t_start = corrected_mts + args.tmin
                    t_end = corrected_mts + args.tmax

                    wait_counter = 0
                    max_wait_loops = 1200

                    while True:
                        all_ready = True
                        for p in range(n_participants_runtime):
                            if len(eeg_buffers_timestamps[p]) == 0 or eeg_buffers_timestamps[p][-1] < t_end:
                                all_ready = False
                                break

                        if all_ready:
                            break

                        for p, inlet in enumerate(eeg_inlets):
                            chunk2, ts2 = inlet.pull_chunk(timeout=0.005, max_samples=128)
                            if ts2:
                                for sample2, t2 in zip(chunk2, ts2):
                                    eeg_buffers_samples[p].append(sample2[:args.n_chans_per_participant])
                                    eeg_buffers_timestamps[p].append(t2 + eeg_time_corrections[p])

                        wait_counter += 1
                        if wait_counter >= max_wait_loops:
                            break

                    grouped_epoch_parts = []
                    valid_group_epoch = True

                    for p in range(n_participants_runtime):
                        eeg_ts_arr = np.asarray(eeg_buffers_timestamps[p], dtype=np.float64)

                        if len(eeg_ts_arr) == 0:
                            valid_group_epoch = False
                            break

                        i0 = np.searchsorted(eeg_ts_arr, t_start, side="left")
                        i1 = i0 + n_times_runtime

                        if args.debug:
                            print(
                                f"  [DBG P{p+1}] t_start={t_start:.4f} t_end={t_end:.4f} "
                                f"eeg_range=[{eeg_ts_arr[0]:.4f}, {eeg_ts_arr[-1]:.4f}] "
                                f"i0={i0} i1={i1} buf={len(eeg_buffers_samples[p])}"
                            )

                        if eeg_ts_arr[-1] < t_end:
                            print(f"Skipping flash: participant {p+1} EEG buffer never reached epoch end.")
                            valid_group_epoch = False
                            break

                        if i0 < 0 or i1 > len(eeg_buffers_samples[p]) or i1 <= i0:
                            print(f"Skipping flash: participant {p+1} incomplete epoch.")
                            valid_group_epoch = False
                            break

                        eeg_arr = np.asarray(eeg_buffers_samples[p], dtype=np.float32)

                        if eeg_arr.shape[0] < max(32, n_times_runtime):
                            print(f"Skipping flash: participant {p+1} not enough buffered data for filtering.")
                            valid_group_epoch = False
                            break

                        try:
                            eeg_filt = bandpass_filter_continuous_eeg(
                                eeg_arr,
                                srate=args.expected_srate,
                                lowcut=args.lowcut,
                                highcut=args.highcut,
                                order=args.filter_order,
                            )
                        except Exception as e:
                            print(f"Skipping flash: participant {p+1} filtering failed ({e})")
                            valid_group_epoch = False
                            break

                        epoch = eeg_filt[i0:i1, :]   # (T, C)

                        if epoch.shape[0] != n_times_runtime:
                            print(
                                f"Skipping flash: participant {p+1} epoch length {epoch.shape[0]} "
                                f"does not match expected runtime length {n_times_runtime}."
                            )
                            valid_group_epoch = False
                            break

                        if baseline_samples > 0:
                            if baseline_samples >= epoch.shape[0]:
                                print(f"Skipping flash: participant {p+1} baseline window too large.")
                                valid_group_epoch = False
                                break
                            epoch = baseline_correct(epoch, baseline_samples)

                        grouped_epoch_parts.append(epoch.astype(np.float32))

                    if not valid_group_epoch:
                        continue

                    # concatenate along channels: [(T,C), (T,C), ...] -> (T, P*C)
                    grouped_epoch = np.concatenate(grouped_epoch_parts, axis=1).astype(np.float32)

                    # (T, C_total) -> (1, C_total, T)
                    X = np.transpose(grouped_epoch, (1, 0))[None, :, :].astype(np.float32)

                    if X.shape[1] != n_total_channels_model or X.shape[2] != n_times_model:
                        print(
                            f"Skipping flash: runtime sample shape {X.shape} does not match "
                            f"model expectation (1, {n_total_channels_model}, {n_times_model})."
                        )
                        continue

                    X = apply_channelwise_normalizer(X, mean_ch, std_ch)

                    proba = clf.predict_proba(X)[0]
                    target_prob = float(proba[1])

                    if parsed["kind"] == "row":
                        row_scores[parsed["idx"]] += target_prob
                    elif parsed["kind"] == "col":
                        col_scores[parsed["idx"]] += target_prob

                    log_flash_selection.append(-1 if parsed["selection"] is None else parsed["selection"])
                    log_flash_seq.append(-1 if parsed["seq"] is None else parsed["seq"])
                    log_flash_number.append(-1 if parsed["flash"] is None else parsed["flash"])
                    log_flash_kind.append(parsed["kind"])
                    log_flash_idx.append(parsed["idx"])
                    log_flash_target_prob.append(target_prob)
                    log_flash_time.append(float(mts))
                    log_flash_row_scores_after.append(row_scores.copy())
                    log_flash_col_scores_after.append(col_scores.copy())

                    print(
                        f"Flash scored | kind={parsed['kind']} idx={parsed['idx']} "
                        f"target_prob={target_prob:.4f}"
                    )
                    continue

                if parsed["event"] == "selection_end":
                    best_row = int(np.argmax(row_scores))
                    best_col = int(np.argmax(col_scores))
                    predicted_symbol = symbol_from_row_col(best_row, best_col)

                    print("\nSelection complete")
                    print("Row scores:", row_scores)
                    print("Col scores:", col_scores)
                    print("Best row:", best_row)
                    print("Best col:", best_col)
                    print("Predicted symbol:", predicted_symbol)

                    if predicted_symbol is not None:
                        send_decision(decision_outlet, f"decision/{predicted_symbol}")
                    else:
                        send_decision(decision_outlet, "decision/?")

                    log_selection.append(-1 if current_selection is None else current_selection)
                    log_best_row.append(best_row)
                    log_best_col.append(best_col)
                    log_symbol.append("?" if predicted_symbol is None else predicted_symbol)
                    log_row_scores.append(row_scores.copy())
                    log_col_scores.append(col_scores.copy())
                    log_decision_time.append(float(mts))

                    current_selection = None
                    row_scores[:] = 0.0
                    col_scores[:] = 0.0
                    continue

                if parsed["event"] == "experiment_end":
                    print("\nReceived experiment_end. Stopping.")
                    experiment_ended = True
                    break

            if experiment_ended:
                break

            time.sleep(0.002)

    except KeyboardInterrupt:
        print("\nManual stop.")

    if args.save_decisions is not None:
        os.makedirs(os.path.dirname(args.save_decisions) or ".", exist_ok=True)

        np.savez(
            args.save_decisions,
            selection=np.asarray(log_selection, dtype=np.int32),
            best_row=np.asarray(log_best_row, dtype=np.int32),
            best_col=np.asarray(log_best_col, dtype=np.int32),
            symbol=np.asarray(log_symbol, dtype=object),
            row_scores=np.asarray(log_row_scores, dtype=np.float32),
            col_scores=np.asarray(log_col_scores, dtype=np.float32),
            decision_time=np.asarray(log_decision_time, dtype=np.float64),

            flash_selection=np.asarray(log_flash_selection, dtype=np.int32),
            flash_seq=np.asarray(log_flash_seq, dtype=np.int32),
            flash_number=np.asarray(log_flash_number, dtype=np.int32),
            flash_kind=np.asarray(log_flash_kind, dtype=object),
            flash_idx=np.asarray(log_flash_idx, dtype=np.int32),
            flash_target_prob=np.asarray(log_flash_target_prob, dtype=np.float32),
            flash_time=np.asarray(log_flash_time, dtype=np.float64),
            flash_row_scores_after=np.asarray(log_flash_row_scores_after, dtype=np.float32),
            flash_col_scores_after=np.asarray(log_flash_col_scores_after, dtype=np.float32),
        )

        print("\nSaved decoder decisions to:", args.save_decisions)


if __name__ == "__main__":
    main()