# Connect EEG LSL stream from unicorn
# Connect to your marker LSL stream from PsychoPy
# Continuously collect both
# Save them into one .npz file at the end
# Parse your marker strings into structured event fields

''' 
How to run it:
1) start unicorn EEG streaming (LSL Unicorn)
2) Start PsychoPy experiment with LSL markers
3) Start the receiver -python3 Receiver_script/receiver_calibration.py --eeg_name Unicorn --marker_name P300Markers --output data/calib_play_01.npz --print_markers
- eeg_timestamps: one LSL timestamp per EEG sample
- markers_raw: raw marker strings 
- marker_timestamps: one LSL timestamp per marker
- event_names: parsed event type (e.g. flash_on, target_start, etc.)
'''

import time
import os
import argparse
import numpy as np
from pylsl import resolve_byprop, StreamInlet


def parse_marker(marker: str):
    """
    Supported marker formats:

    experiment_start
    experiment_end

    focus_start/A
    focus_end/A

    target_start/A
    target_end/A

    sequence_start/A/0
    sequence_end/A/0

    flash_on/row/2/target_1/char_A/seq_0/flash_3
    flash_off/col/4/target_0/char_A/seq_0/flash_8
    """
    parts = marker.strip().split("/")

    out = {
        "raw": marker,
        "event": None,
        "kind": None,
        "idx": None,
        "is_target": None,
        "target_char": None,
        "seq": None,
        "flash": None,
    }

    if len(parts) == 1:
        out["event"] = parts[0]
        return out

    if len(parts) == 2 and parts[0] in {"focus_start", "focus_end", "target_start", "target_end"}:
        out["event"] = parts[0]
        out["target_char"] = parts[1]
        return out

    if len(parts) == 3 and parts[0] in {"sequence_start", "sequence_end"}:
        out["event"] = parts[0]
        out["target_char"] = parts[1]
        try:
            out["seq"] = int(parts[2])
        except ValueError:
            out["seq"] = None
        return out

    if len(parts) == 7 and parts[0] in {"flash_on", "flash_off"}:
        out["event"] = parts[0]
        out["kind"] = parts[1]

        try:
            out["idx"] = int(parts[2])
        except ValueError:
            out["idx"] = None

        if parts[3].startswith("target_"):
            try:
                out["is_target"] = int(parts[3].split("_")[1])
            except Exception:
                out["is_target"] = None

        if parts[4].startswith("char_"):
            out["target_char"] = parts[4].split("_", 1)[1]

        if parts[5].startswith("seq_"):
            try:
                out["seq"] = int(parts[5].split("_")[1])
            except Exception:
                out["seq"] = None

        if parts[6].startswith("flash_"):
            try:
                out["flash"] = int(parts[6].split("_")[1])
            except Exception:
                out["flash"] = None

        return out

    out["event"] = parts[0]
    return out


def find_stream(prop, value, timeout=10):
    print(f"Looking for LSL stream with {prop}='{value}' ...")
    streams = resolve_byprop(prop, value, timeout=timeout)
    if not streams:
        raise RuntimeError(f"Could not find stream with {prop}='{value}'")
    return streams[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg_name", type=str, default="UnicornRecorderLSLStream")
    parser.add_argument("--marker_name", type=str, default="P300Markers")
    parser.add_argument("--output", type=str, default="calibration_recording.npz")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--print_markers", action="store_true")
    args = parser.parse_args()

    eeg_stream = find_stream("name", args.eeg_name, timeout=15)
    marker_stream = find_stream("name", args.marker_name, timeout=15)

    eeg_inlet = StreamInlet(eeg_stream, max_chunklen=64)
    marker_inlet = StreamInlet(marker_stream)

    eeg_info = eeg_inlet.info()
    eeg_name = eeg_info.name()
    eeg_type = eeg_info.type()
    eeg_srate = eeg_info.nominal_srate()
    eeg_nchan = eeg_info.channel_count()

    print("\nConnected EEG stream:")
    print(f"  name         : {eeg_name}")
    print(f"  type         : {eeg_type}")
    print(f"  srate        : {eeg_srate}")
    print(f"  channel_count: {eeg_nchan}")

    marker_info = marker_inlet.info()
    print("\nConnected marker stream:")
    print(f"  name         : {marker_info.name()}")
    print(f"  type         : {marker_info.type()}")

    eeg_samples = []
    eeg_timestamps = []

    markers_raw = []
    marker_timestamps = []
    parsed_events = []

    experiment_finished = False
    t_wall_start = time.time()

    print("\nWaiting for data... Press Ctrl+C to stop manually.\n")

    try:
        while True:
            chunk, ts = eeg_inlet.pull_chunk(timeout=0.0, max_samples=128)
            if ts:
                eeg_samples.extend(chunk)
                eeg_timestamps.extend(ts)

            while True:
                sample, mts = marker_inlet.pull_sample(timeout=0.0)
                if mts is None:
                    break

                marker = sample[0]
                parsed = parse_marker(marker)

                markers_raw.append(marker)
                marker_timestamps.append(mts)
                parsed_events.append(parsed)

                if args.print_markers:
                    print(f"[MARKER] t={mts:.6f} | {marker}")

                if parsed["event"] == "experiment_end":
                    experiment_finished = True

            if args.duration is not None and (time.time() - t_wall_start) >= args.duration:
                print("\nDuration limit reached. Stopping.")
                break

            if experiment_finished:
                print("\nReceived experiment_end marker. Stopping.")
                break

            time.sleep(0.002)

    except KeyboardInterrupt:
        print("\nManual stop.")

    eeg_samples = np.asarray(eeg_samples, dtype=np.float32)
    eeg_timestamps = np.asarray(eeg_timestamps, dtype=np.float64)
    marker_timestamps = np.asarray(marker_timestamps, dtype=np.float64)
    markers_raw = np.asarray(markers_raw, dtype=object)

    parsed_event_names = np.asarray([e["event"] for e in parsed_events], dtype=object)
    parsed_kinds = np.asarray([e["kind"] for e in parsed_events], dtype=object)
    parsed_idxs = np.asarray([(-1 if e["idx"] is None else e["idx"]) for e in parsed_events], dtype=np.int32)
    parsed_is_target = np.asarray([(-1 if e["is_target"] is None else e["is_target"]) for e in parsed_events], dtype=np.int32)
    parsed_target_chars = np.asarray([e["target_char"] for e in parsed_events], dtype=object)
    parsed_seqs = np.asarray([(-1 if e["seq"] is None else e["seq"]) for e in parsed_events], dtype=np.int32)
    parsed_flashes = np.asarray([(-1 if e["flash"] is None else e["flash"]) for e in parsed_events], dtype=np.int32)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    np.savez(
        args.output,
        eeg_samples=eeg_samples,
        eeg_timestamps=eeg_timestamps,
        eeg_srate=np.array([eeg_srate], dtype=np.float32),
        eeg_nchan=np.array([eeg_nchan], dtype=np.int32),
        eeg_name=np.array([eeg_name], dtype=object),
        markers_raw=markers_raw,
        marker_timestamps=marker_timestamps,
        event_names=parsed_event_names,
        event_kinds=parsed_kinds,
        event_idxs=parsed_idxs,
        event_is_target=parsed_is_target,
        event_target_chars=parsed_target_chars,
        event_seqs=parsed_seqs,
        event_flashes=parsed_flashes,
    )

    print("\nSaved recording to:", args.output)
    print(f"EEG samples shape: {eeg_samples.shape}")
    print(f"EEG timestamps   : {eeg_timestamps.shape}")
    print(f"Markers          : {markers_raw.shape}")

    if eeg_samples.size > 0 and eeg_timestamps.size > 1:
        duration_sec = eeg_timestamps[-1] - eeg_timestamps[0]
        approx_srate = (len(eeg_timestamps) - 1) / duration_sec if duration_sec > 0 else 0
        print(f"Approx EEG duration: {duration_sec:.2f} s")
        print(f"Approx EEG rate    : {approx_srate:.2f} Hz")

    unique_events, counts = np.unique(parsed_event_names, return_counts=True)
    print("\nEvent summary:")
    for ev, c in zip(unique_events, counts):
        print(f"  {ev}: {c}")


if __name__ == "__main__":
    main()