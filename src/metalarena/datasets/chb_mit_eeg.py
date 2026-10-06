import warnings
from bisect import bisect_right
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from metalarena.datasets.chb_mit_eeg_files import CHB_MIT_Files, SAMPLING_FREQ, edf_sort_key


SECONDS_PER_DAY = 24 * 3600
DROP = -1

class Zone(IntEnum):
    """Where a window sits relative to the marked seizures. Precedence order."""
    ICTAL = 0            # entirely inside a marked seizure
    ICTAL_PARTIAL = 1    # overlaps a seizure boundary at either end
    POSTICTAL = 2        # starts within postictal_secs after a seizure ends
    PREICTAL = 3         # ends within preictal_secs before a seizure starts
    INTERICTAL_NEAR = 4  # within interictal_buffer_secs of a seizure, but none of the above
    INTERICTAL_FAR = 5   # beyond the buffer from every seizure


def assign_zones(
    window_starts_s: np.ndarray,
    window_length_s: float,
    seizures_s: Sequence[Tuple[float, float]],
    preictal_s: float,
    postictal_s: float,
    interictal_buffer_s: float,
) -> np.ndarray:
    """
    Zone for each window, given seizure intervals on the SAME time axis.

    All times in seconds; intervals are half-open. Precedence follows the Zone
    declaration order, so an earlier zone wins when several apply. Two of those
    orderings are decisions rather than conveniences:

      POSTICTAL before PREICTAL -- a window shortly after seizure A and shortly
      before seizure B is postictal. Postictal EEG is distinctive, so counting
      it as preictal lets a "prediction" model score by recognising the
      aftermath of the previous seizure instead of the approach of the next.

      PREICTAL before INTERICTAL_NEAR -- the preictal horizon is normally much
      shorter than the interictal buffer and sits inside it, so preictal
      windows must be claimed first or the buffer would swallow them.
    """
    n = len(window_starts_s)
    zones = np.full(n, Zone.INTERICTAL_FAR, dtype=np.int8)
    if n == 0:
        return zones
    if len(seizures_s) == 0:
        return zones

    starts = np.asarray(window_starts_s, dtype=np.float64)
    ends = starts + window_length_s

    sz = np.asarray(seizures_s, dtype=np.float64)
    sz = sz[np.argsort(sz[:, 0], kind="stable")]     # the summary order is not ours to trust
    S = sz[:, 0][:, None]
    E = sz[:, 1][:, None]

    overlaps = (starts[None, :] < E) & (ends[None, :] > S)
    fully_in = (starts[None, :] >= S) & (ends[None, :] <= E)

    any_full = fully_in.any(axis=0)
    any_overlap = overlaps.any(axis=0)

    # A window is preictal when its LAST sample falls within preictal_s of the
    # onset. Defining it on the first sample instead shifts the class boundary
    # by one whole window length -- negligible at a 30 min horizon, not at 1 min.
    pre = (((ends[None, :] <= S) & (ends[None, :] + preictal_s > S)).any(axis=0)
           if preictal_s > 0 else np.zeros(n, dtype=bool))
    post = (((starts[None, :] >= E) & (starts[None, :] < E + postictal_s)).any(axis=0)
            if postictal_s > 0 else np.zeros(n, dtype=bool))
    near = ((((ends[None, :] > S - interictal_buffer_s) & (starts[None, :] < E + interictal_buffer_s))
             ).any(axis=0) if interictal_buffer_s > 0 else np.zeros(n, dtype=bool))

    zones[near] = Zone.INTERICTAL_NEAR
    zones[pre] = Zone.PREICTAL
    zones[post] = Zone.POSTICTAL
    zones[any_overlap & ~any_full] = Zone.ICTAL_PARTIAL
    zones[any_full] = Zone.ICTAL
    return zones


@dataclass(frozen=True)
class Task:
    """
    A mapping from temporal zones to class labels, plus sensible horizons.

    `zone_to_label` maps every Zone to a class index, or to DROP to exclude
    those windows from the dataset entirely. `defaults` supplies horizons and
    window geometry appropriate to the task; anything passed explicitly to
    CHB_MIT_Windows overrides them.
    """
    name: str
    class_names: Dict[int, str]
    zone_to_label: Dict[Zone, int]
    defaults: Dict[str, float] = field(default_factory=dict)

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def label_and_keep(self, zones: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        table = np.full(len(Zone), DROP, dtype=np.int8)
        for zone, label in self.zone_to_label.items():
            table[int(zone)] = label
        labels = table[zones]
        return labels, labels != DROP

    def zones_for(self, label: int) -> List[Zone]:
        return [z for z, v in self.zone_to_label.items() if v == label]


DETECTION = Task(
    name="detection",
    # Ground truth is unambiguous: the window either falls between the marked
    # onset and offset or it does not. Nothing is excluded -- every non-seizure
    # window is a legitimate negative, including postictal ones.
    class_names={0: "non-seizure", 1: "seizure"},
    zone_to_label={
        Zone.ICTAL: 1,
        Zone.ICTAL_PARTIAL: 1,
        Zone.POSTICTAL: 0,
        Zone.PREICTAL: 0,
        Zone.INTERICTAL_NEAR: 0,
        Zone.INTERICTAL_FAR: 0,
    },
    defaults=dict(window_length_secs=4.0, hop_length_secs=1.0,
                  preictal_time_secs=0.0, postictal_time_secs=0.0,
                  interictal_buffer_secs=0.0),
)

PREDICTION = Task(
    name="prediction",
    # No clinician marks a preictal state, so this class is a hypothesis. The
    # horizon is a per-patient hyperparameter, not a fact. Ictal and postictal
    # windows are excluded: the task is to fire BEFORE onset, so training on
    # during/after is answering a different question.
    class_names={0: "interictal", 1: "preictal"},
    zone_to_label={
        Zone.ICTAL: DROP,
        Zone.ICTAL_PARTIAL: DROP,
        Zone.POSTICTAL: DROP,
        Zone.PREICTAL: 1,
        Zone.INTERICTAL_NEAR: DROP,   # the contamination buffer
        Zone.INTERICTAL_FAR: 0,
    },
    defaults=dict(window_length_secs=5.0, hop_length_secs=1.0,
                  preictal_time_secs=30 * 60.0, postictal_time_secs=30 * 60.0,
                  interictal_buffer_secs=4 * 3600.0),
)

THREE_CLASS = Task(
    name="three_class",
    # Label values match the original module constants, so existing checkpoints
    # and confusion matrices keep their meaning.
    class_names={0: "interictal", 1: "ictal", 2: "preictal"},
    zone_to_label={
        Zone.ICTAL: 1,
        Zone.ICTAL_PARTIAL: 1,
        Zone.POSTICTAL: DROP,
        Zone.PREICTAL: 2,
        Zone.INTERICTAL_NEAR: DROP,
        Zone.INTERICTAL_FAR: 0,
    },
    defaults=dict(window_length_secs=5.0, hop_length_secs=1.0,
                  preictal_time_secs=30 * 60.0, postictal_time_secs=30 * 60.0,
                  interictal_buffer_secs=0.0),
)

TASKS = {t.name: t for t in (DETECTION, PREDICTION, THREE_CLASS)}


def parse_time_string(time_str: str) -> int:
    """
    'HH:MM:SS' -> seconds. Hours are NOT reduced modulo 24, on purpose.

    CHB-MIT extends past 24 so a file's end stays greater than its start across
    midnight, then resets at the next file:

        chb16_08.edf   start 23:40:26   end 24:40:26
        chb16_09.edf   start 00:40:33   end  1:40:33
    """
    parts = [int(x) for x in time_str.strip().split(":")]
    if len(parts) != 3:
        raise ValueError(f"Expected HH:MM:SS, got {time_str!r}")
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def file_duration_seconds(start_time: Optional[str], end_time: Optional[str]) -> Optional[int]:
    """Wall-clock duration, recovering a day if the clock wrapped."""
    if not start_time or not end_time:
        return None
    delta = parse_time_string(end_time) - parse_time_string(start_time)
    return delta + SECONDS_PER_DAY if delta < 0 else delta


def build_patient_timeline(
    file_metadata: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Optional[float]], List[Tuple[float, float]], List[str]]:
    """
    Place a patient's files on one continuous seconds axis.

    Returns (offset_per_file, absolute_seizure_times, unspecified_time_files).

    Files are ordered by `edf_sort_key` and walked in that order; each time the
    wall clock steps backwards, a day is added, which is how chb16_09 (00:40)
    lands after chb16_08 (23:40) rather than 23 hours earlier.

    Files whose summary has no start time (chb24) cannot be placed relative to
    anything, so each becomes its own island at offset 0 and only sees its own
    seizures. That is a real limitation, not an approximation: without a
    timestamp there is no way to know whether the next recording began one
    minute or one day later.
    """
    offsets: Dict[str, Optional[float]] = {}
    seizures: List[Tuple[float, float]] = []
    unanchored: List[str] = []

    day = 0
    prev_clock: Optional[int] = None

    for name in sorted(file_metadata, key=edf_sort_key):
        info = file_metadata[name]
        start_time = info.get("start_time")

        if not start_time:
            offsets[name] = None
            unanchored.append(name)
            continue

        clock = parse_time_string(start_time)
        if prev_clock is not None and clock < prev_clock:
            day += 1
        prev_clock = clock

        offset = float(clock + day * SECONDS_PER_DAY)
        offsets[name] = offset
        for s, e in info.get("seizures", []):
            seizures.append((offset + float(s), offset + float(e)))

    seizures.sort()
    return offsets, seizures, unanchored


class CHB_MIT_Windows(Dataset):
    """
    Each item is `(window, label)`:

        window  torch.float32, (n_channels, window_length_secs * 256), in uV ()
        label   int in range(task.n_classes)

    `task` is "detection", "prediction", "three_class", or a custom `Task`.
    Horizons default to the task's own values and can be overridden per call.
    """

    def __init__(self,
                 task: Literal["detection", "prediction", "three_class"] | Task = "detection",
                 window_length_secs: Optional[float] = None,
                 hop_length_secs: Optional[float] = None,
                 preictal_time_secs: Optional[float] = None,
                 postictal_time_secs: Optional[float] = None,
                 interictal_buffer_secs: Optional[float] = None,
                 overlap_is_ictal: bool = False,
                 units: Literal["uV", "V"] = "uV",
                 verbose_windows: bool = False,
                 **kwargs):
        if isinstance(task, str):
            if task not in TASKS:
                raise ValueError(f"Unknown task {task!r}. Choose from {sorted(TASKS)} "
                                 f"or pass a Task instance.")
            task = TASKS[task]
        if not isinstance(task, Task):
            raise TypeError(f"task must be a str or Task, got {type(task).__name__}")

        # overlap_is_ictal is a policy edit to the task table, not a separate
        # branch in the labelling code. False means a window straddling a
        # boundary is dropped rather than relabelled: it contains seizure
        # activity, so putting it in the negative class would be a labelling
        # error, not a choice.
        if not overlap_is_ictal:
            table = dict(task.zone_to_label)
            table[Zone.ICTAL_PARTIAL] = DROP
            task = replace(task, zone_to_label=table)

        self.task = task
        self.overlap_is_ictal = overlap_is_ictal
        self.units = units

        def pick(explicit, key):
            return task.defaults[key] if explicit is None else explicit

        self.window_length_secs = pick(window_length_secs, "window_length_secs")
        self.hop_length_secs = pick(hop_length_secs, "hop_length_secs")
        self.preictal_time_secs = pick(preictal_time_secs, "preictal_time_secs")
        self.postictal_time_secs = pick(postictal_time_secs, "postictal_time_secs")
        self.interictal_buffer_secs = pick(interictal_buffer_secs, "interictal_buffer_secs")

        self.window_length_timesteps = int(round(self.window_length_secs * SAMPLING_FREQ))
        self.hop_length_timesteps = int(round(self.hop_length_secs * SAMPLING_FREQ))
    
        if self.window_length_timesteps <= 0 or self.hop_length_timesteps <= 0:
            raise ValueError("window_length_secs and hop_length_secs must be positive.")
    
        if self.preictal_time_secs > 0 and self.interictal_buffer_secs > 0 \
                and self.preictal_time_secs > self.interictal_buffer_secs:
            warnings.warn(
                f"preictal horizon ({self.preictal_time_secs / 60:.0f} min) exceeds the "
                f"interictal buffer ({self.interictal_buffer_secs / 60:.0f} min); the "
                f"buffer is then doing nothing useful.", RuntimeWarning, stacklevel=2)

        self.dataset = CHB_MIT_Files(**kwargs)
        self._load_raw_data = True

        # Look up by NAME, never by position: edf_metadata iteration order and
        # edf_file_paths order agree only by construction, and when they drifted
        # the dataset served labels from one recording with EEG from another.
        file_index = {(int(p.parent.name[3:5]), p.name): k
                      for k, p in enumerate(self.dataset.edf_file_paths)}

        self._files: List[Dict[str, Any]] = []
        self.timelines: Dict[int, Dict[str, Any]] = {}

        for patient, file_metadata in self.dataset.edf_metadata.items():
            offsets, seizures_abs, unanchored = build_patient_timeline(file_metadata)
            self.timelines[patient] = {"offsets": offsets, "seizures": seizures_abs,
                                       "unanchored": unanchored}
            if unanchored and self.interictal_buffer_secs > 0:
                warnings.warn(
                    f"chb{patient:02d}: {len(unanchored)} file(s) have no start time in the "
                    f"summary, so they cannot be placed on the patient timeline. Their "
                    f"windows are labelled against their own seizures only, and the "
                    f"{self.interictal_buffer_secs / 3600:.1f} h interictal buffer cannot be "
                    f"enforced across file boundaries for them.",
                    RuntimeWarning, stacklevel=2)

            for file_name, file_info in file_metadata.items():
                key = (patient, file_name)
                if key not in file_index:
                    raise KeyError(f"chb{patient:02d}/{file_name} is in edf_metadata but not "
                                   f"in edf_file_paths -- the two are out of sync.")

                n_samples = self._file_n_samples(patient, file_name, file_info)
                starts = self._window_starts(n_samples)

                offset = offsets[file_name]
                if offset is None:
                    # island: own seizures only, on a file-local axis
                    local = [(float(s), float(e)) for s, e in file_info["seizures"]]
                    zones = assign_zones(starts / SAMPLING_FREQ, self.window_length_secs,
                                         local, self.preictal_time_secs,
                                         self.postictal_time_secs,
                                         self.interictal_buffer_secs)
                else:
                    zones = assign_zones(offset + starts / SAMPLING_FREQ,
                                         self.window_length_secs, seizures_abs,
                                         self.preictal_time_secs, self.postictal_time_secs,
                                         self.interictal_buffer_secs)

                labels, keep = self.task.label_and_keep(zones)
                self._files.append({
                    "patient": patient,
                    "file_name": file_name,
                    "global_file_idx": file_index[key],
                    "window_starts": starts[keep],
                    "labels": labels[keep],
                    "zones": zones[keep],
                    "n_windows_before_drop": len(starts),
                })

        self._offsets = np.zeros(len(self._files) + 1, dtype=np.int64)
        for k, f in enumerate(self._files):
            self._offsets[k + 1] = self._offsets[k] + len(f["labels"])
        self._offsets_list = self._offsets.tolist()
        self.total_windows_in_ds = int(self._offsets[-1])

        if verbose_windows:
            print(self.summary())

    def _file_n_samples(self, patient: int, file_name: str, file_info: Dict[str, Any]) -> int:
        """
        Length comes from the EDF header, never the summary's wall clock.

        mne silently truncates get_data(start, stop) past the end of a file, so
        sizing windows from end_time - start_time produced short windows that
        only surfaced as a collate error much later.
        """
        n_samples = file_info.get("n_times")
        if n_samples is None:
            raise KeyError(f"chb{patient:02d}/{file_name}: no 'n_times' in metadata. "
                           f"Update CHB_MIT_Files to record it during discovery.")
        clock = file_duration_seconds(file_info.get("start_time"), file_info.get("end_time"))
        if clock is not None:
            drift = abs(clock * SAMPLING_FREQ - n_samples) / SAMPLING_FREQ
            if drift > 2.0:
                warnings.warn(
                    f"chb{patient:02d}/{file_name}: summary says {clock} s but the EDF holds "
                    f"{n_samples / SAMPLING_FREQ:.1f} s ({drift:.1f} s apart). Using the EDF.",
                    RuntimeWarning, stacklevel=3)
        return int(n_samples)

    def _window_starts(self, n_samples: int) -> np.ndarray:
        """Start samples of every full window that fits; empty if none do."""
        if n_samples < self.window_length_timesteps:
            return np.empty(0, dtype=np.int64)
        n = (n_samples - self.window_length_timesteps) // self.hop_length_timesteps + 1
        return np.arange(n, dtype=np.int64) * self.hop_length_timesteps

    def labels(self) -> np.ndarray:
        if not self._files:
            return np.empty(0, dtype=np.int8)
        return np.concatenate([f["labels"] for f in self._files])

    def zones(self) -> np.ndarray:
        if not self._files:
            return np.empty(0, dtype=np.int8)
        return np.concatenate([f["zones"] for f in self._files])

    def class_counts(self) -> Dict[str, int]:
        counts = np.bincount(self.labels(), minlength=self.task.n_classes)
        return {name: int(counts[v]) for v, name in sorted(self.task.class_names.items())}

    def zone_counts(self) -> Dict[str, int]:
        """Zone histogram BEFORE the task table drops anything."""
        out = {z.name: 0 for z in Zone}
        for f in self._files:
            for z, c in zip(*np.unique(f["zones"], return_counts=True)):
                out[Zone(int(z)).name] += int(c)
        dropped = sum(f["n_windows_before_drop"] for f in self._files) - self.total_windows_in_ds
        out["(dropped by task)"] = dropped
        return out

    def summary(self) -> str:
        lines = [f"task: {self.task.name}   {self.total_windows_in_ds} windows "
                 f"from {len(self._files)} files"]
        for name, n in self.class_counts().items():
            pct = 100 * n / max(self.total_windows_in_ds, 1)
            lines.append(f"  {name:<14} {n:>9}  ({pct:5.2f}%)")
        lines.append(f"  {'(dropped)':<14} {self.zone_counts()['(dropped by task)']:>9}")
        return "\n".join(lines)

    def file_of(self, idx: int) -> Dict[str, Any]:
        f_idx, w = self.locate(idx)
        f = self._files[f_idx]
        start = int(f["window_starts"][w])
        return {
            "patient": f["patient"],
            "file_name": f["file_name"],
            "path": self.dataset.edf_file_paths[f["global_file_idx"]],
            "start_sample": start,
            "start_sec": start / SAMPLING_FREQ,
            "label": int(f["labels"][w]),
            "zone": Zone(int(f["zones"][w])),
        }

    def disable_raw_data_loading(self) -> None:
        self._load_raw_data = False

    def enable_raw_data_loading(self) -> None:
        self._load_raw_data = True

    def locate(self, idx: int) -> Tuple[int, int]:
        n = self.total_windows_in_ds
        if idx < 0:
            idx += n
        if not 0 <= idx < n:
            raise IndexError(f"index {idx} out of range for {n} windows")
        f_idx = bisect_right(self._offsets_list, idx) - 1
        return f_idx, idx - self._offsets_list[f_idx]

    def __getitem__(self, idx: int):
        f_idx, w = self.locate(idx)
        f = self._files[f_idx]
        label = int(f["labels"][w])
        if not self._load_raw_data:
            return None, label

        start = int(f["window_starts"][w])
        stop = start + self.window_length_timesteps

        # raw[:, a:b] returns a (data, times) tuple; get_data returns the array.
        raw, _ = self.dataset[f["global_file_idx"]]
        data = raw.get_data(start=start, stop=stop)
        if not self.dataset.load_into_memory:
            raw.close()

        if data.shape[1] != self.window_length_timesteps:
            raise RuntimeError(
                f"{f['file_name']}: window {w} returned {data.shape[1]} samples, expected "
                f"{self.window_length_timesteps}.")

        # mne returns SI units, so EEG arrives at ~1e-4 V. uV puts it at ~1e2,
        # which is a scale a network can work with -- though per-channel
        # standardisation on top of this is still the right thing to do.
        if self.units == "uV":
            data = data * 1e6
        elif self.units != "V":
            raise ValueError(f"units must be 'uV' or 'V', got {self.units!r}")

        return torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32)), label

    def __len__(self) -> int:
        return self.total_windows_in_ds

    def __repr__(self) -> str:
        return (
            f"CHB_MIT_Windows({self.task.name})\n"
            f"  Windows:   {self.total_windows_in_ds} from {len(self._files)} files\n"
            f"  Geometry:  {self.window_length_secs}s window / {self.hop_length_secs}s hop\n"
            f"  Preictal:  {self.preictal_time_secs / 60:.0f} min\n"
            f"  Postictal: {self.postictal_time_secs / 60:.0f} min\n"
            f"  Buffer:    {self.interictal_buffer_secs / 3600:.2f} h\n"
            f"  Classes:   " + ", ".join(f"{k}={v}" for k, v in self.class_counts().items())
            + f"\n  Units:     {self.units}\n"
        )
