'''
- Connect to Unicorn EEG
- Connect to marker stream from free-speller
- Load pre-trained EEGNet model
- Compute target probability for flashes row or column
- Accumulate evidence across sequences
- Predict the row and column and convert to symbol
- Send back through LSL
- Optionally save online decisions for later analysis
- Save per-flash outputs as well

Running:
python online_free_spell_decoder.py \
  --eeg_name Unicorn \
  --marker_name P300Markers \
  --model_path models_eegnet/eegnet_calibration_trained.pkl \
  --tmin 0.0 \
  --tmax 0.8 \
  --expected_srate 250 \
  --save_decisions data/testing_play_01_decisions.npz
'''

import os
import time
import argparse
import numpy as np
import torch

from pylsl import resolve_byprop, StreamInlet, StreamInfo, StreamOutlet, local_clock
from braindecode import EEGClassifier
from braindecode.models import EEGNet


# Grid definition
# -------------------------------------------------------------

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


# Marker parsing
# -------------------------------------------------------------

def parse_marker(marker: str):
    """
    Testing marker formats expected here:

    experiment_start
    experiment_end

    selection_start/0
    selection_end/0

    sequence_start/0/0
    sequence_end/0/0

    flash_on/row/2/sel_0/seq_0/flash_3
    flash_off/col/4/sel_0/seq_0/flash_8
    """
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


# EEGNet model builder
# Must match your training script
# -------------------------------------------------------------

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


# Utilities
# -------------------------------------------------------------

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
        source_id="p300-online-decoder"
    )
    return StreamOutlet(info)


def send_decision(outlet, msg):
    outlet.push_sample([msg], local_clock())
    print("[DECISION]", msg)


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


def baseline_correct(epoch, baseline_samples):
    if baseline_samples <= 0:
        return epoch
    baseline = epoch[:baseline_samples].mean(axis=0, keepdims=True)
    return epoch - baseline


def find_epoch_sample_range(eeg_timestamps, t_start, t_end):
    i0 = np.searchsorted(eeg_timestamps, t_start, side="left")
    i1 = np.searchsorted(eeg_timestamps, t_end, side="left")
    return i0, i1


def symbol_from_row_col(row_idx, col_idx):
    if row_idx is None or col_idx is None:
        return None
    if not (0 <= row_idx < GRID_ROWS and 0 <= col_idx < GRID_COLS):
        return None
    return GRID_SYMBOLS[row_idx][col_idx]


# Main
# -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eeg_name", type=str, default="UnicornRecorderLSLStream")
    parser.add_argument("--marker_name", type=str, default="P300Markers")
    parser.add_argument("--model_path", type=str, required=True)

    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=0.8)
    parser.add_argument("--baseline", type=float, default=0.0)

    parser.add_argument("--expected_srate", type=float, default=250.0)
    parser.add_argument("--n_chans", type=int, default=8)
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--drop_prob", type=float, default=0.44539302556010774)

    parser.add_argument("--save_decisions", type=str, default=None,
                        help="Optional path to save online decoder outputs")
    args = parser.parse_args()

    # Connect streams
    eeg_stream = find_stream("name", args.eeg_name, timeout=15)
    marker_stream = find_stream("name", args.marker_name, timeout=15)

    eeg_inlet = StreamInlet(eeg_stream, max_chunklen=64)
    marker_inlet = StreamInlet(marker_stream)
    decision_outlet = make_decision_outlet()

    eeg_info = eeg_inlet.info()
    eeg_srate_nominal = eeg_info.nominal_srate()
    eeg_nchan = eeg_info.channel_count()

    print("\nConnected EEG stream:")
    print(f"  name         : {eeg_info.name()}")
    print(f"  type         : {eeg_info.type()}")
    print(f"  nominal_srate: {eeg_srate_nominal}")
    print(f"  channel_count: {eeg_nchan}")

    print("\nConnected marker stream:")
    print(f"  name         : {marker_inlet.info().name()}")
    print(f"  type         : {marker_inlet.info().type()}")

    # Build and load EEGNet
    n_times = int(round((args.tmax - args.tmin) * args.expected_srate))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\nLoading model...")
    print("device:", device)
    print("n_chans:", args.n_chans)
    print("n_times:", n_times)

    clf = make_eegnet_clf(
        n_chans=args.n_chans,
        n_times=n_times,
        n_classes=args.n_classes,
        device=device,
        drop_prob=args.drop_prob,
    )

    # Must initialize before load_params, matching your training workflow
    clf.initialize()
    clf.load_params(f_params=args.model_path)

    print("Loaded model from:", args.model_path)

    # Buffers
    eeg_samples = []
    eeg_timestamps = []

    current_selection = None
    row_scores = np.zeros(GRID_ROWS, dtype=np.float64)
    col_scores = np.zeros(GRID_COLS, dtype=np.float64)

    # Decision logs for optional saving
    log_selection = []
    log_best_row = []
    log_best_col = []
    log_symbol = []
    log_row_scores = []
    log_col_scores = []
    log_decision_time = []

    # Per-flash logs
    log_flash_selection = []
    log_flash_seq = []
    log_flash_number = []
    log_flash_kind = []
    log_flash_idx = []
    log_flash_target_prob = []
    log_flash_time = []
    log_flash_row_scores_after = []
    log_flash_col_scores_after = []

    print("\nDecoder is running...\n")

    experiment_ended = False

    try:
        while True:
            # Pull EEG chunks continuously
            chunk, ts = eeg_inlet.pull_chunk(timeout=0.0, max_samples=128)
            if ts:
                for sample in chunk:
                    eeg_samples.append(sample[:args.n_chans])
                eeg_timestamps.extend(ts)

            # Trim very old data to limit memory growth
            if len(eeg_timestamps) > 50000:
                eeg_samples = eeg_samples[-30000:]
                eeg_timestamps = eeg_timestamps[-30000:]

            # Pull markers
            while True:
                sample, mts = marker_inlet.pull_sample(timeout=0.0)
                if mts is None:
                    break

                marker = sample[0]
                parsed = parse_marker(marker)
                print("[MARKER]", marker)

                # Start of a new character selection
                if parsed["event"] == "selection_start":
                    current_selection = parsed["selection"]
                    row_scores[:] = 0.0
                    col_scores[:] = 0.0
                    print(f"\nStarted selection {current_selection}\n")
                    continue

                # Each flash_on triggers online epoch extraction and scoring
                if parsed["event"] == "flash_on":
                    if parsed["kind"] not in {"row", "col"}:
                        continue
                    if parsed["idx"] is None:
                        continue

                    t_start = float(mts) + args.tmin
                    t_end = float(mts) + args.tmax

                    eeg_ts_arr = np.asarray(eeg_timestamps, dtype=np.float64)
                    if len(eeg_ts_arr) == 0:
                        continue

                    i0, i1 = find_epoch_sample_range(eeg_ts_arr, t_start, t_end)

                    # Wait a bit if the full future epoch is not available yet
                    wait_counter = 0
                    while i1 > len(eeg_samples) and wait_counter < 200:
                        chunk2, ts2 = eeg_inlet.pull_chunk(timeout=0.005, max_samples=128)
                        if ts2:
                            for sample2 in chunk2:
                                eeg_samples.append(sample2[:args.n_chans])
                            eeg_timestamps.extend(ts2)
                            eeg_ts_arr = np.asarray(eeg_timestamps, dtype=np.float64)
                            i0, i1 = find_epoch_sample_range(eeg_ts_arr, t_start, t_end)
                        wait_counter += 1

                    if i0 >= len(eeg_samples) or i1 > len(eeg_samples) or i1 <= i0:
                        print("Skipping flash: incomplete epoch.")
                        continue

                    # Epoch shape here is (T, C)
                    epoch = np.asarray(eeg_samples[i0:i1], dtype=np.float32)

                    # Match the time length used in training
                    if epoch.shape[0] != n_times:
                        epoch = resample_epoch_if_needed(epoch, n_times)

                    # Optional baseline correction
                    baseline_samples = int(round(args.baseline * args.expected_srate))
                    if baseline_samples > 0 and baseline_samples < epoch.shape[0]:
                        epoch = baseline_correct(epoch, baseline_samples)

                    # Convert (T, C) -> (1, C, T) for EEGNet
                    X = np.transpose(epoch, (1, 0))[None, :, :].astype(np.float32)

                    # Probability that this flash is a target flash
                    proba = clf.predict_proba(X)[0]
                    target_prob = float(proba[1])

                    if parsed["kind"] == "row":
                        row_scores[parsed["idx"]] += target_prob
                    elif parsed["kind"] == "col":
                        col_scores[parsed["idx"]] += target_prob

                    # Save per-flash information
                    log_flash_selection.append(
                        -1 if parsed["selection"] is None else parsed["selection"]
                    )
                    log_flash_seq.append(
                        -1 if parsed["seq"] is None else parsed["seq"]
                    )
                    log_flash_number.append(
                        -1 if parsed["flash"] is None else parsed["flash"]
                    )
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

                # End of one character selection
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

                    # Save decision info in memory for optional output file
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

    # Optional save of decoder outputs
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