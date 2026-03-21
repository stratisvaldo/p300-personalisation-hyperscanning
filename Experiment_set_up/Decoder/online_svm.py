'''
- Connect to Unicorn EEG
- Connect to marker stream from free-speller
- Load trained SVM model
- Compute target probability for flashes row or column
- Accumulate evidence across sequences
- Predict the row and column and convert to symbol
- Send back through LSL
- Optionally save online decisions for later analysis
- Save per-flash outputs as well

Online preprocessing matches traditional ML training preprocessing except:
- no artifact thresholding online
- no resampling online

Pipeline per flash:
1. filter continuous EEG from 0.1 to 20 Hz
2. epoch from tmin to tmax around flash_on marker
3. baseline correction
4. downsample by factor 4
5. flatten to (1, Features)
6. apply saved scaler + SVM through sklearn pipeline

Running:
python online_free_spell_decoder_svm.py `
  --eeg_name Unicorn `
  --marker_name P300Markers `
  --model_path models_svm/svm_calibration.joblib `
  --tmin 0.0 `
  --tmax 0.8 `
  --baseline 0.0 `
  --lowcut 0.1 `
  --highcut 20.0 `
  --filter_order 4 `
  --expected_srate 250 `
  --downsample_factor 4 `
  --save_decisions data/testing_play_01_svm_decisions.npz
'''

import os
import time
import argparse
import joblib
import numpy as np

from scipy.signal import butter, filtfilt
from pylsl import resolve_byprop, StreamInlet, StreamInfo, StreamOutlet, local_clock


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
        source_id="p300-online-svm-decoder"
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


def find_epoch_sample_range(eeg_timestamps, t_start, t_end):
    i0 = np.searchsorted(eeg_timestamps, t_start, side="left")
    i1 = np.searchsorted(eeg_timestamps, t_end, side="left")
    return i0, i1


def bandpass_filter_continuous_eeg(eeg, srate, lowcut=0.1, highcut=20.0, order=4):
    nyq = 0.5 * srate
    if lowcut <= 0:
        raise ValueError("lowcut must be > 0")
    if highcut >= nyq:
        raise ValueError(f"highcut must be below Nyquist ({nyq})")

    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="bandpass")
    eeg_filt = filtfilt(b, a, eeg, axis=0)
    return eeg_filt.astype(np.float32)


def downsample_epoch(epoch, factor):
    if factor <= 1:
        return epoch
    return epoch[::factor, :]


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

    parser.add_argument("--lowcut", type=float, default=0.1)
    parser.add_argument("--highcut", type=float, default=20.0)
    parser.add_argument("--filter_order", type=int, default=4)

    parser.add_argument("--expected_srate", type=float, default=250.0)
    parser.add_argument("--n_chans", type=int, default=8)
    parser.add_argument("--downsample_factor", type=int, default=4)

    parser.add_argument("--save_decisions", type=str, default=None,
                        help="Optional path to save online decoder outputs")
    args = parser.parse_args()

    eeg_stream = find_stream("name", args.eeg_name, timeout=15)
    marker_stream = find_stream("name", args.marker_name, timeout=15)

    eeg_inlet = StreamInlet(eeg_stream, max_chunklen=64)
    marker_inlet = StreamInlet(marker_stream)
    decision_outlet = make_decision_outlet()

    # Get time corrections for both inlets so that EEG and marker timestamps are on the same clock
    # time correctin is a pylsl method that measures the offset between two clocks

    print("\nComputing LSL time corrections (this takes a few seconds)...")
    eeg_time_correction = eeg_inlet.time_correction(timeout=5.0)
    marker_time_correction = marker_inlet.time_correction(timeout=5.0)
    print(f"  EEG time correction   : {eeg_time_correction:.6f} s")
    print(f"  Marker time correction: {marker_time_correction:.6f} s")

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

    # Load model
    print("\nLoading SVM model...")
    model_package = joblib.load(args.model_path)
    clf = model_package["pipeline"]
    n_features_model = int(model_package["n_features"])

    n_times_runtime = int(round((args.tmax - args.tmin) * args.expected_srate))
    n_times_after_downsample = len(np.arange(0, n_times_runtime, args.downsample_factor))
    n_features_runtime = n_times_after_downsample * args.n_chans

    print("Loaded model from:", args.model_path)
    print("Model expects n_features:", n_features_model)
    print("Runtime window n_times:", n_times_runtime)
    print("Runtime n_times after downsampling:", n_times_after_downsample)
    print("Runtime n_features:", n_features_runtime)

    if n_features_runtime != n_features_model:
        raise RuntimeError(
            f"Runtime feature count ({n_features_runtime}) does not match "
            f"model n_features ({n_features_model}). "
            f"Adjust tmin/tmax/expected_srate/downsample_factor/n_chans."
        )

    eeg_samples = []
    eeg_timestamps = []

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

    print("\nDecoder is running...\n")

    experiment_ended = False
    baseline_samples = int(round(args.baseline * args.expected_srate))

    try:
        while True:
            chunk, ts = eeg_inlet.pull_chunk(timeout=0.0, max_samples=128)
            if ts:
                for sample, t in zip(chunk, ts):
                    eeg_samples.append(sample[:args.n_chans])
                    # ----------------------------------------------------------------
                    # CHANGE 2: Apply the EEG time correction when storing each
                    # timestamp. This maps EEG timestamps onto the same clock as
                    # the marker timestamps so that searchsorted finds the right
                    # indices when we look for an epoch window.
                    # ----------------------------------------------------------------
                    eeg_timestamps.append(t + eeg_time_correction)

            if len(eeg_timestamps) > 50000:
                eeg_samples = eeg_samples[-30000:]
                eeg_timestamps = eeg_timestamps[-30000:]

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

                    # ----------------------------------------------------------------
                    # CHANGE 3: Apply the marker time correction to the marker
                    # timestamp before computing t_start / t_end. This ensures the
                    # epoch window is on the same clock as the (corrected) EEG
                    # timestamps stored above.
                    # ----------------------------------------------------------------
                    corrected_mts = float(mts) + marker_time_correction
                    t_start = corrected_mts + args.tmin
                    t_end   = corrected_mts + args.tmax

                    eeg_ts_arr = np.asarray(eeg_timestamps, dtype=np.float64)
                    if len(eeg_ts_arr) == 0:
                        continue

                    i0, i1 = find_epoch_sample_range(eeg_ts_arr, t_start, t_end)

                    # ----------------------------------------------------------------
                    # CHANGE 4: Print a diagnostic line so you can immediately see
                    # whether t_end falls inside or outside the EEG buffer range.
                    # Remove this print once everything is working.
                    # ----------------------------------------------------------------
                    print(
                        f"  [DBG] t_start={t_start:.4f} t_end={t_end:.4f} "
                        f"eeg_range=[{eeg_ts_arr[0]:.4f}, {eeg_ts_arr[-1]:.4f}] "
                        f"i0={i0} i1={i1} buf={len(eeg_samples)}"
                    )

                    # ----------------------------------------------------------------
                    # CHANGE 5: Increased the wait loop from 200 to 500 iterations
                    # (2.5 s max) so that slow EEG chunk delivery does not cause
                    # epochs to be skipped unnecessarily.
                    # ----------------------------------------------------------------
                    wait_counter = 0
                    while i1 > len(eeg_samples) and wait_counter < 500:
                        chunk2, ts2 = eeg_inlet.pull_chunk(timeout=0.005, max_samples=128)
                        if ts2:
                            for sample2, t2 in zip(chunk2, ts2):
                                eeg_samples.append(sample2[:args.n_chans])
                                eeg_timestamps.append(t2 + eeg_time_correction)  # CHANGE 2 (same correction)
                            eeg_ts_arr = np.asarray(eeg_timestamps, dtype=np.float64)
                            i0, i1 = find_epoch_sample_range(eeg_ts_arr, t_start, t_end)
                        wait_counter += 1

                    if i0 >= len(eeg_samples) or i1 > len(eeg_samples) or i1 <= i0:
                        print("Skipping flash: incomplete epoch.")
                        continue

                    eeg_arr = np.asarray(eeg_samples, dtype=np.float32)

                    if eeg_arr.shape[0] < max(32, n_times_runtime):
                        print("Skipping flash: not enough buffered data for filtering.")
                        continue

                    try:
                        eeg_filt = bandpass_filter_continuous_eeg(
                            eeg_arr,
                            srate=args.expected_srate,
                            lowcut=args.lowcut,
                            highcut=args.highcut,
                            order=args.filter_order,
                        )
                    except Exception as e:
                        print(f"Skipping flash: filtering failed ({e})")
                        continue

                    epoch = eeg_filt[i0:i1, :]   # (T, C)

                    if epoch.shape[0] != n_times_runtime:
                        print(
                            f"Skipping flash: epoch length {epoch.shape[0]} "
                            f"does not match expected runtime length {n_times_runtime}."
                        )
                        continue

                    if baseline_samples > 0:
                        if baseline_samples >= epoch.shape[0]:
                            print("Skipping flash: baseline window too large.")
                            continue
                        epoch = baseline_correct(epoch, baseline_samples)

                    epoch = downsample_epoch(epoch, args.downsample_factor)

                    X = epoch.reshape(1, -1).astype(np.float32)

                    if X.shape[1] != n_features_model:
                        print(
                            f"Skipping flash: flattened feature count {X.shape[1]} "
                            f"does not match model n_features {n_features_model}."
                        )
                        continue

                    proba = clf.predict_proba(X)[0]
                    target_prob = float(proba[1])

                    if parsed["kind"] == "row":
                        row_scores[parsed["idx"]] += target_prob
                    elif parsed["kind"] == "col":
                        col_scores[parsed["idx"]] += target_prob

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
