'''
- multiple LSL steams at once
- same parsing as original receiver 
- one shared marker stream from speller
- each participant's data saved in one array within big npz file (put together in epoching step)

Running: 
python Experiment_set_up/Receiver_script/receiver_cal_group.py `
  --eeg_names Unicorn_P01,Unicorn_P02,Unicorn_P03,Unicorn_P04,Unicorn_P05,Unicorn_P06,Unicorn_P07,Unicorn_P08 `
  --marker_name P300Markers `
  --output data_calib/group_calibration_recording.npz `
  --print_markers
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


def parse_eeg_names(eeg_names_raw: str):
    names = [x.strip() for x in eeg_names_raw.split(",") if x.strip()]
    if len(names) == 0:
        raise ValueError("No EEG stream names provided.")
    return names


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--eeg_names",
        type=str,
        required=True,
        help="Comma-separated EEG stream names, e.g. Unicorn_P01,Unicorn_P02,...,Unicorn_P08"
    )
    parser.add_argument("--marker_name", type=str, default="P300Markers")
    parser.add_argument("--output", type=str, default="group_calibration_recording.npz")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--print_markers", action="store_true")
    parser.add_argument("--max_chunklen", type=int, default=64)
    args = parser.parse_args()

    eeg_names = parse_eeg_names(args.eeg_names)

    # ---------------------------------------------------------
    # Connect all EEG streams
    # ---------------------------------------------------------
    eeg_inlets = []
    eeg_stream_info = []

    print("\nConnecting EEG streams...")
    for name in eeg_names:
        eeg_stream = find_stream("name", name, timeout=15)
        eeg_inlet = StreamInlet(eeg_stream, max_chunklen=args.max_chunklen)

        info = eeg_inlet.info()
        eeg_inlets.append(eeg_inlet)
        eeg_stream_info.append({
            "name": info.name(),
            "type": info.type(),
            "srate": info.nominal_srate(),
            "nchan": info.channel_count(),
        })

    # ---------------------------------------------------------
    # Connect marker stream
    # ---------------------------------------------------------
    marker_stream = find_stream("name", args.marker_name, timeout=15)
    marker_inlet = StreamInlet(marker_stream)

    print("\nConnected EEG streams:")
    for i, info in enumerate(eeg_stream_info):
        print(f"Participant {i+1}:")
        print(f"  name         : {info['name']}")
        print(f"  type         : {info['type']}")
        print(f"  srate        : {info['srate']}")
        print(f"  channel_count: {info['nchan']}")

    marker_info = marker_inlet.info()
    print("\nConnected marker stream:")
    print(f"  name         : {marker_info.name()}")
    print(f"  type         : {marker_info.type()}")

    # ---------------------------------------------------------
    # Buffers for each participant
    # ---------------------------------------------------------
    eeg_samples_all = [[] for _ in range(len(eeg_inlets))]
    eeg_timestamps_all = [[] for _ in range(len(eeg_inlets))]

    markers_raw = []
    marker_timestamps = []
    parsed_events = []

    experiment_finished = False
    t_wall_start = time.time()

    print("\nWaiting for data... Press Ctrl+C to stop manually.\n")

    try:
        while True:
            # Pull chunks from all EEG streams
            for p_idx, eeg_inlet in enumerate(eeg_inlets):
                chunk, ts = eeg_inlet.pull_chunk(timeout=0.0, max_samples=128)
                if ts:
                    eeg_samples_all[p_idx].extend(chunk)
                    eeg_timestamps_all[p_idx].extend(ts)

            # Pull all pending markers
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

    # ---------------------------------------------------------
    # Convert to numpy
    # ---------------------------------------------------------
    participant_samples = []
    participant_timestamps = []
    participant_srates = []
    participant_nchans = []
    participant_names = []

    for i, info in enumerate(eeg_stream_info):
        samples_np = np.asarray(eeg_samples_all[i], dtype=np.float32)
        ts_np = np.asarray(eeg_timestamps_all[i], dtype=np.float64)

        participant_samples.append(samples_np)
        participant_timestamps.append(ts_np)
        participant_srates.append(info["srate"])
        participant_nchans.append(info["nchan"])
        participant_names.append(info["name"])

    marker_timestamps = np.asarray(marker_timestamps, dtype=np.float64)
    markers_raw = np.asarray(markers_raw, dtype=object)

    parsed_event_names = np.asarray([e["event"] for e in parsed_events], dtype=object)
    parsed_kinds = np.asarray([e["kind"] for e in parsed_events], dtype=object)
    parsed_idxs = np.asarray([(-1 if e["idx"] is None else e["idx"]) for e in parsed_events], dtype=np.int32)
    parsed_is_target = np.asarray([(-1 if e["is_target"] is None else e["is_target"]) for e in parsed_events], dtype=np.int32)
    parsed_target_chars = np.asarray([e["target_char"] for e in parsed_events], dtype=object)
    parsed_seqs = np.asarray([(-1 if e["seq"] is None else e["seq"]) for e in parsed_events], dtype=np.int32)
    parsed_flashes = np.asarray([(-1 if e["flash"] is None else e["flash"]) for e in parsed_events], dtype=np.int32)

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    save_dict = {
        "n_participants": np.array([len(eeg_inlets)], dtype=np.int32),
        "participant_names": np.asarray(participant_names, dtype=object),
        "participant_srates": np.asarray(participant_srates, dtype=np.float32),
        "participant_nchans": np.asarray(participant_nchans, dtype=np.int32),

        "markers_raw": markers_raw,
        "marker_timestamps": marker_timestamps,
        "event_names": parsed_event_names,
        "event_kinds": parsed_kinds,
        "event_idxs": parsed_idxs,
        "event_is_target": parsed_is_target,
        "event_target_chars": parsed_target_chars,
        "event_seqs": parsed_seqs,
        "event_flashes": parsed_flashes,
    }

    # store each participant separately
    for i in range(len(eeg_inlets)):
        save_dict[f"eeg_samples_p{i+1}"] = participant_samples[i]
        save_dict[f"eeg_timestamps_p{i+1}"] = participant_timestamps[i]

    np.savez(args.output, **save_dict)

    # ---------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------
    print("\nSaved recording to:", args.output)

    for i in range(len(eeg_inlets)):
        print(f"\nParticipant {i+1} summary:")
        print(f"  name            : {participant_names[i]}")
        print(f"  EEG samples shape: {participant_samples[i].shape}")
        print(f"  EEG timestamps   : {participant_timestamps[i].shape}")

        if participant_samples[i].size > 0 and participant_timestamps[i].size > 1:
            duration_sec = participant_timestamps[i][-1] - participant_timestamps[i][0]
            approx_srate = (len(participant_timestamps[i]) - 1) / duration_sec if duration_sec > 0 else 0
            print(f"  Approx duration  : {duration_sec:.2f} s")
            print(f"  Approx rate      : {approx_srate:.2f} Hz")

    print(f"\nMarkers: {markers_raw.shape}")

    unique_events, counts = np.unique(parsed_event_names, return_counts=True)
    print("\nEvent summary:")
    for ev, c in zip(unique_events, counts):
        print(f"  {ev}: {c}")


if __name__ == "__main__":
    main()