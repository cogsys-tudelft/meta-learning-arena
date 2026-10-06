import math
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import mne
from torch.utils.data import Dataset
from tqdm import tqdm

SAMPLING_FREQ = 256  # Hz

DEFAULT_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
]

SPLIT_NAMES: Tuple[str, str, str] = ("train", "val", "test")

_EDF_NAME_RE = re.compile(r"chb(\d+)([a-z]*)_(\d+)(\+?)", re.IGNORECASE)


def channel_matches(required: str, file_channel: str) -> bool:
    """
    True iff `file_channel` is `required`, optionally carrying MNE's
    duplicate-disambiguation suffix.

    MNE renames duplicate EDF labels by appending -0, -1, ... in header order.
    A channel that appears once keeps its bare name; one that appears twice
    becomes "<name>-0" and "<name>-1".  Both forms must match.  (The full
    argument for why that ordering makes first-match selection correct is in
    CHB_MIT_Files._get_raw.)

    Two things the old `re.match(required, file_ch)` got wrong:

      re.escape(required)   the channel name is now matched LITERALLY.  Before,
                            a name containing a regex metacharacter was
                            interpreted as a pattern -- "FP1.F7" would have
                            matched "FP1-F7", "C4-P4?" would have matched
                            "C4-P" , and so on.  Harmless for DEFAULT_CHANNELS,
                            but `channels=` is a public argument.

      re.fullmatch(...)     both ends are anchored.  re.match only anchors the
                            START, so "T8-P8" also matched "T8-P8QQ" or
                            "T8-P8-REF".  Now the only thing allowed after the
                            name is the "-<digits>" MNE suffix.

    So the regex reads: [literal channel name][optionally: a hyphen and one or
    more digits][end of string].
    """
    return re.fullmatch(re.escape(required) + r"(?:-\d+)?", file_channel) is not None


def edf_has_all_channels(file_channels: Sequence[str], required_channels: Sequence[str]) -> bool:
    return all(
        any(channel_matches(req, ch) for ch in file_channels)
        for req in required_channels
    )


def parse_edf_metadata(text: str) -> Dict[str, Any]:
    files: Dict[str, Any] = {}
    current_channels: Dict[int, str] = {}
    section_channels = False

    channel_pat = re.compile(r"Channel\s+(\d+):\s*(.*)")
    file_pat = re.compile(r"File Name:\s*(\S+)")
    start_pat = re.compile(r"File Start Time:\s*(.*)")
    end_pat = re.compile(r"File End Time:\s*(.*)")
    seizure_pat = re.compile(r"Seizure\s+(\d+)\s+Start Time:\s*(\d+)\s*seconds")
    single_seizure_pat = re.compile(r"Seizure Start Time:\s*(\d+)\s*seconds")
    seizure_end_pat = re.compile(r"Seizure\s+(\d+)\s+End Time:\s*(\d+)\s*seconds")
    single_seizure_end_pat = re.compile(r"Seizure End Time:\s*(\d+)\s*seconds")
    nsz_pat = re.compile(r"Number of Seizures in File:\s*(\d+)")

    current_file = None
    seizures_temp: Dict[str, List[Optional[int]]] = {}

    for line in text.splitlines():
        line = line.strip()

        if "Channels in EDF Files" in line or "Channels changed" in line:
            section_channels = True
            current_channels = {}
            continue
        if section_channels and (m := channel_pat.match(line)):
            ch, label = int(m.group(1)), m.group(2).strip()
            if label != "-":
                current_channels[ch] = label
            continue
        if section_channels and not line:
            section_channels = False
            continue

        if (m := file_pat.match(line)):
            current_file = m.group(1)
            files[current_file] = {
                "channels": dict(current_channels),
                "start_time": None,
                "end_time": None,
                "seizures": [],
            }
            seizures_temp = {}
            continue

        if current_file:
            if (m := start_pat.match(line)):
                files[current_file]["start_time"] = m.group(1)
            elif (m := end_pat.match(line)):
                files[current_file]["end_time"] = m.group(1)
            elif (m := nsz_pat.match(line)):
                files[current_file]["n_seizures_declared"] = int(m.group(1))
            elif (m := seizure_pat.match(line)):
                seizures_temp[m.group(1)] = [int(m.group(2)), None]
            elif (m := seizure_end_pat.match(line)):
                seizure_id = m.group(1)

                if int(m.group(1)) != len(seizures_temp):
                    warnings.warn(f"Seizure end time for seizure {m.group(1)} found in metadata while waiting for end time of seizure {len(seizures_temp)}. It is assumed that the seizures are listed in order and the end time is for the last seizure.")
                    seizure_id = str(len(seizures_temp))

                if seizure_id in seizures_temp:
                    seizures_temp[seizure_id][1] = int(m.group(2))
                    files[current_file]["seizures"].append(tuple(seizures_temp[seizure_id]))
            elif (m := single_seizure_pat.match(line)):
                seizures_temp["single"] = [int(m.group(1)), None]
            elif (m := single_seizure_end_pat.match(line)):
                if "single" not in seizures_temp:
                    raise ValueError(
                        f"{current_file}: 'seizure end time found in metadata on line {line} with no matching start time"
                    )
                seizures_temp["single"][1] = int(m.group(1))
                files[current_file]["seizures"].append(tuple(seizures_temp["single"]))

    # Cross-check the parsed list against the declared count
    for name, metadata in files.items():
        declared = metadata.pop("n_seizures_declared", None)
        if declared is not None and declared != len(metadata["seizures"]):
            raise ValueError(
                f"{name}: summary declares {declared} seizure(s) but "
                f"{len(metadata['seizures'])} were parsed -- the summary format has "
                f"changed or the parser missed a case."
            )

    return files


class NotEnoughSeizuresError(ValueError):
    """Raised when a patient cannot supply one seizure per active split."""

    def __init__(self, total: int, required: int, split_names: List[str],
                 patient_id: Optional[int] = None):
        self.total = total
        self.required = required
        self.split_names = split_names
        self.patient_id = patient_id
        super().__init__(self._message())

    def _message(self) -> str:
        who = f"Patient chb{self.patient_id:02d}" if self.patient_id is not None else "This patient"
        lines = [
            f"{who} has only {self.total} seizure(s) among the files that survived "
            f"channel filtering, but a {'/'.join(self.split_names)} split needs at "
            f"least {self.required} (one per split).",
            "",
            "Every CHB-MIT patient has >= 3 seizures as distributed, so this almost "
            "always means seizure-bearing files were dropped because they lack one of "
            "the requested channels (chb12, chb13 and chb15 change montage mid-record).",
            "",
            "Options:",
            f"  - exclude this patient:      patients=[... without {self.patient_id} ...]",
            "  - request fewer channels so its files survive the filter",
            "  - use a two-way split:       split_ratios=(0.8, 0.2, 0.0)",
            "  - inspect what was dropped:  CHB_MIT_Files(root, patients=[%s]).report_dropped_files()"
            % (self.patient_id if self.patient_id is not None else "N"),
        ]
        return "\n".join(lines)


def edf_sort_key(name: str) -> Tuple[int, str, int, int, str]:
    """
    Sort EDF file names into RECORDING order.

    Plain `sorted()` is wrong for two CHB-MIT naming quirks:

      chb02_16+.edf   '+' (0x2B) sorts before '.' (0x2E), so lexicographic
                      order puts the continuation file BEFORE the file it
                      continues.  The trailing flag here forces it after.

      chb17a/b/c      patient 17 is split into three sessions whose file
                      numbering restarts; the session letter must be the
                      primary key after the patient number.

    Files that do not match the pattern sort last, in name order.
    """
    m = _EDF_NAME_RE.match(name)
    if m is None:
        return (10**9, "zzzz", 10**9, 0, name)
    return (int(m.group(1)), m.group(2).lower(), int(m.group(3)), 1 if m.group(4) else 0, name)


# ALSO REMOVE TESTS THAT USE THIS FUNCTION!!!!!!!!!
# def check_chronological_order(metadata: Dict[int, Dict[str, Any]]) -> List[str]:
#     """
#     Verify `edf_sort_key` against the 'File Start Time' fields in the summaries.

#     Returns a list of human-readable problems (empty means consistent).  Clock
#     times wrap at midnight, so a step backwards of more than 12 h is treated as
#     a day rollover rather than an inversion.  Files with no recorded start time
#     (chb24) are skipped.
#     """
#     problems = []
#     for patient_id, files in metadata.items():
#         ordered = sorted(files, key=edf_sort_key)
#         prev_name, prev_abs = None, None
#         day = 0
#         for name in ordered:
#             t = files[name].get("start_time")
#             if not t:
#                 continue
#             try:
#                 h, mnt, s = (int(x) for x in re.split(r"[.:]", t.strip()))
#             except ValueError:
#                 problems.append(f"chb{patient_id:02d}/{name}: unparsable start time {t!r}")
#                 continue
#             secs = h * 3600 + mnt * 60 + s
#             if prev_abs is not None:
#                 while secs + day * 86400 < prev_abs - 12 * 3600:
#                     day += 1
#             abs_t = secs + day * 86400
#             if prev_abs is not None and abs_t < prev_abs:
#                 problems.append(
#                     f"chb{patient_id:02d}: {name} ({t}) starts before {prev_name} "
#                     f"but sorts after it"
#                 )
#             prev_name, prev_abs = name, abs_t
#     return problems


def split_seizures(total: int, train_ratio: float, val_ratio: float, test_ratio: float
                   ) -> Tuple[int, int, int]:
    """Every active split gets at least one seizure."""

    ratios = [train_ratio, val_ratio, test_ratio]
    active = [r > 0 for r in ratios]
    num_active = sum(active)

    if num_active == 0:
        raise ValueError("At least one split ratio must be greater than zero.")

    if total < num_active:
        names = [n for n, a in zip(("train", "val", "test"), active) if a]
        raise NotEnoughSeizuresError(
            total=total,
            required=num_active,
            split_names=names,
        )

    ratio_sum = sum(ratios)
    norm_ratios = [r / ratio_sum if r > 0 else 0.0 for r in ratios]
    raw = [r * total for r in norm_ratios]
    splits = [
        (max(1, int(x)) if active[i] else 0)
        for i, x in enumerate(raw)
    ]

    # Adjust to match total
    diff = total - sum(splits)
    frac = [raw[i] - int(raw[i]) if active[i] else -1.0 for i in range(3)]

    while diff != 0:
        if diff > 0:
            # give to largest fractional part among active
            idx = max(range(3), key=lambda i: frac[i])
            splits[idx] += 1
            frac[idx] = 0.0
            diff -= 1
        else:
            # remove from largest split (but keep >=1 if active)
            idx = max(
                range(3),
                key=lambda i: splits[i] if (active[i] and splits[i] > 1) else -1
            )
            splits[idx] -= 1
            diff += 1

    assert sum(splits) == total, "Final splits do not sum to total, something went wrong."

    return tuple(splits)


def choose_chronological_cuts(
    seizures_per_file: Sequence[int],
    target_seizure_counts: Tuple[int, int, int],
    file_ratios: Tuple[float, float, float],
) -> Tuple[int, int]:
    """
    Pick cut points 0 <= i <= j <= n so that

        train = files[:i]      val = files[i:j]      test = files[j:]

    minimising the total absolute deviation from the per-split seizure targets.

    Tie-breaks, in order: seizure deviation, then file-count deviation from
    `ratios`, then the earliest (i, j).  Fully deterministic.
    """
    n = len(seizures_per_file)
    active = tuple(r > 0 for r in file_ratios)

    prefix = [0]
    for s in seizures_per_file:
        prefix.append(prefix[-1] + s)

    def candidates():
        for i in range(n + 1):
            for j in range(i, n + 1):
                if not active[0] and i != 0:
                    continue
                if not active[1] and i != j:
                    continue
                if not active[2] and j != n:
                    continue
                yield i, j

    def score(i, j):
        counts = (prefix[i], prefix[j] - prefix[i], prefix[n] - prefix[j])
        sizes = (i, j - i, n - j)
        seizure_cost = sum(abs(counts[k] - target_seizure_counts[k]) for k in range(3) if active[k])
        file_cost = sum(abs(sizes[k] - file_ratios[k] * n) for k in range(3) if active[k])
        return counts, sizes, seizure_cost, file_cost

    # Preference order: every active split has >=1 seizure AND >=1 file,
    # then >=1 file only, then no constraint at all.
    for require_seizure in (True, False):
        best = None
        for i, j in candidates():
            counts, sizes, seizure_cost, file_cost = score(i, j)
            if any(active[k] and sizes[k] == 0 for k in range(3)):
                continue
            if require_seizure and any(active[k] and counts[k] == 0 for k in range(3)):
                continue
            key = (seizure_cost, file_cost, i, j)
            if best is None or key < best[0]:
                best = (key, (i, j))
        if best is not None:
            return best[1]

    # Degenerate: fewer files than active splits.  Give everything to train.
    return (n, n)


def gather_all_edf_data_entries(root: Path, patients: Sequence[int], channels: Sequence[str]):
    """
    Returns (edf_file_paths, metadata, dropped) where `dropped` maps
    patient_id -> {filename: reason} for files excluded by the channel filter.
    """
    edf_file_paths: List[Path] = []
    metadata: Dict[int, Dict[str, Any]] = {}
    dropped: Dict[int, Dict[str, str]] = {}

    for patient_id in patients:
        patient_id_str = f"chb{patient_id:02d}"
        d = root / patient_id_str

        summary = d / f"{patient_id_str}-summary.txt"
        if not summary.exists():
            raise FileNotFoundError(f"No summary file at {summary}")
        # explicit encoding: default open() uses the locale encoding, which
        # differs between Linux (utf-8) and Windows (cp1252).
        edf_files_metadata = parse_edf_metadata(summary.read_text(encoding="utf-8"))

        per_patient_files: List[Path] = []
        dropped[patient_id] = {}

        # Sort the glob: directory iteration order is filesystem dependent.
        # Match .edf case-insensitively but explicitly, so Linux (case-sensitive
        # fs) and macOS (case-insensitive fs) see the same file set.
        edfs = sorted(
            (p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".edf"),
            key=lambda p: edf_sort_key(p.name),
        )

        for f in edfs:
            info = mne.io.read_raw_edf(f, preload=False, verbose="ERROR")
            file_ch_names: List[str] = list(info.ch_names)

            if edf_has_all_channels(file_ch_names, channels):
                per_patient_files.append(f)

                # For example for patient 24, some files are not listed in the summary.txt file
                # However, we can still use these non-seizures files for training, so we 'manually' add them to the metadata here
                if f.name not in edf_files_metadata:
                    edf_files_metadata[f.name] = {
                        "channels": {i: c for i, c in enumerate(file_ch_names, start=1)},
                        "start_time": None,
                        "end_time": None,
                        "seizures": [],
                    }

                edf_files_metadata[f.name]["n_times"] = int(info.n_times)
                edf_files_metadata[f.name]["sfreq"] = float(info.info["sfreq"])
            else:
                missing = [c for c in channels
                           if not any(channel_matches(c, ch) for ch in file_ch_names)]
                dropped[patient_id][f.name] = "missing channels: " + ", ".join(missing)
                edf_files_metadata.pop(f.name, None) # pop instead of del since file might not have been in summary

        # Drop summary entries whose EDF is not on disk at all.
        on_disk = {p.name for p in per_patient_files}
        for name in list(edf_files_metadata):
            if name not in on_disk:
                dropped[patient_id].setdefault(name, "listed in summary but not found on disk")
                edf_files_metadata.pop(name)

        # sort because it's possible that edf_files_metadata is not yet sorted
        metadata[patient_id] = {
            name: edf_files_metadata[name]
            for name in sorted(edf_files_metadata, key=edf_sort_key)
        }
        edf_file_paths.extend(per_patient_files)

    return edf_file_paths, metadata, dropped


def normalize_split_ratios(split_ratios: Dict[str, float]) -> Tuple[float, float, float]:
    """
    Validate a `{"train": .., "val": .., "test": ..}` mapping and return the
    three ratios in canonical (train, val, test) order.
 
    Omitted keys mean "this split gets nothing", so a two-way split is just
 
        {"train": 0.8, "val": 0.2}
 
    rather than a tuple with a positional 0.0 on the end.
    """
    if not isinstance(split_ratios, dict):
        got = type(split_ratios).__name__
        raise TypeError(
            f"split_ratios must be a dict keyed by split name, e.g. "
            f"{{'train': 0.7, 'val': 0.15, 'test': 0.15}} -- got a {got}. "
            f"(The positional tuple form was the old API.)"
        )

    unknown = set(split_ratios) - set(SPLIT_NAMES)
    if unknown:
        raise ValueError(
            f"Unknown split name(s) in split_ratios: {sorted(unknown)}. "
            f"Valid keys are {list(SPLIT_NAMES)}."
        )

    ratios = tuple(float(split_ratios.get(name, 0.0)) for name in SPLIT_NAMES)

    if any(r < 0 for r in ratios):
        raise ValueError(f"split_ratios must be non-negative, got {split_ratios!r}")
    if not any(r > 0 for r in ratios):
        raise ValueError("At least one split ratio must be greater than zero.")
    # exact float equality rejected 42 of the 4851 integer-percent triples,
    # e.g. {"train": 0.01, "val": 0.29, "test": 0.70}.
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"split_ratios must sum to 1.0, got {sum(ratios)!r} from {split_ratios!r}")

    return ratios


class CHB_MIT_Files(Dataset):
    def __init__(
        self,
        root: str,
        patients: Optional[List[int]] = None,
        channels: Optional[List[str]] = None,
        split: Optional[Literal["train", "val", "test"]] = None,
        split_ratios: Optional[Dict[str, float]] = None,
        load_into_memory: bool = False,
        verbose: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.split_ratios = split_ratios
        self.channels = channels or list(DEFAULT_CHANNELS)
        self.patients = patients or list(range(1, 25))
        self.load_into_memory = load_into_memory
        self.verbose = verbose
 
        # Canonical (train, val, test) order, filled in below. The helper
        # functions stay positional; the dict is an interface convenience.
        self._ratios: Optional[Tuple[float, float, float]] = None
 
        if self.split is not None:
            if self.split not in SPLIT_NAMES:
                raise ValueError(
                    f"split must be one of {list(SPLIT_NAMES)}, got {self.split!r}"
                )
            if self.split_ratios is None:
                raise ValueError("split_ratios must be provided when split is specified.")
 
        if self.split_ratios is not None:
            if self.split is None:
                raise ValueError("split must be specified when split_ratios are provided.")
            self._ratios = normalize_split_ratios(self.split_ratios)
            self.split_ratios = dict(self.split_ratios)
            if self._ratios[SPLIT_NAMES.index(self.split)] == 0:
                raise ValueError(
                    f"split={self.split!r} was requested but split_ratios gives it a "
                    f"ratio of 0, so it would be empty: {self.split_ratios!r}"
                )
 
        self.edf_file_paths, self.edf_metadata, self.dropped_files = gather_all_edf_data_entries(
            self.root, self.patients, self.channels
        )

        self._pre_loaded: Optional[List[mne.io.BaseRaw]] = None

        if self.split is not None:
            self._split_files_chronologically()

        if self.load_into_memory:
            self._warn_about_memory()
            self._pre_loaded = [
                self._get_raw(p, preload=True)
                for p in tqdm(self.edf_file_paths,
                              desc=f"Pre-loading EDF {self.split or 'all'} files", unit="file")
            ]

    def _warn_about_memory(self) -> None:
        n_files = len(self.edf_file_paths)
        # CHB-MIT files are ~1 h; MNE returns float64.
        est_bytes = n_files * len(self.channels) * SAMPLING_FREQ * 3600 * 8
        est_gib = est_bytes / 2**30
        if len(self.patients) > 1 or est_gib > 4:
            warnings.warn(
                f"load_into_memory=True for {len(self.patients)} patient(s) / {n_files} "
                f"file(s) will hold roughly {est_gib:.1f} GiB of float64 EEG in RAM "
                f"(~{est_gib * 1024 / max(n_files, 1):.0f} MiB per hour-long file). "
                "Consider a single patient, or leave load_into_memory=False and let "
                "MNE read lazily.",
                ResourceWarning,
                stacklevel=3,
            )

    def _split_files_chronologically(self) -> None:
        """
        Reads seizure counts straight from self.edf_metadata instead of
        iterating the dataset (which opened every EDF through MNE purely to
        read a dict that was already in memory), then cuts each patient's
        time-ordered file list into three contiguous blocks.
        """
        per_patient: Dict[int, List[Tuple[int, str]]] = {}
        for idx, path in enumerate(self.edf_file_paths):
            patient_id = self._patient_id(path)
            per_patient.setdefault(patient_id, []).append((idx, path.name))

        chosen: Dict[str, List[int]] = {name: [] for name in SPLIT_NAMES}

        for patient_id, entries in per_patient.items():
            entries.sort(key=lambda e: edf_sort_key(e[1]))
            indices = [i for i, _ in entries]
            counts = [len(self.edf_metadata[patient_id][name]["seizures"]) for _, name in entries]
            total = sum(counts)

            try:
                target_seizure_counts = split_seizures(total, *self._ratios)
            except NotEnoughSeizuresError as exc:
                # raise a new exception with the patient_id included
                raise NotEnoughSeizuresError(
                    total=exc.total, required=exc.required, split_names=exc.split_names,
                    patient_id=patient_id,
                ) from None

            i, j = choose_chronological_cuts(counts, target_seizure_counts, self._ratios)
            blocks = dict(zip(SPLIT_NAMES, (indices[:i], indices[i:j], indices[j:])))

            if self.verbose:
                got = (sum(counts[:i]), sum(counts[i:j]), sum(counts[j:]))
                summary = "  ".join(
                    f"{name} {len(blocks[name]):3d}f/{got[k]:2d}s (target {target_seizure_counts[k]:2d})"
                    for k, name in enumerate(SPLIT_NAMES)
                )
                print(f"chb{patient_id:02d}: {len(indices):3d} files, {total:3d} seizures | {summary}")
 
            for k in chosen:
                chosen[k].extend(blocks[k])

        keep = sorted(chosen[self.split])
        self.edf_file_paths = [self.edf_file_paths[i] for i in keep]

        updated: Dict[int, Dict[str, Any]] = {}
        for path in self.edf_file_paths:
            patient_id = self._patient_id(path)
            updated.setdefault(patient_id, {})[path.name] = self.edf_metadata[patient_id][path.name]
        self.edf_metadata = updated

    def _patient_id(self, path: Path) -> int:
        return int(path.parent.name[3:5])

    def _get_raw(self, edf_path: Path, preload: bool = False) -> mne.io.BaseRaw:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")

        # --------------------------------------------------------------
        # Duplicate channels: why taking the FIRST match is correct.
        #
        # In the CHB-MIT EDFs, T8-P8 is written twice, both times with the
        # bare label "T8-P8" -- the -0/-1 suffixes are not in the file. MNE
        # adds them on read, and it does so POSITIONALLY, in header order:
        #
        #     EDF header:  ['FP1-F7', 'T8-P8', 'F7-T7', 'T8-P8', 'CZ-PZ']
        #     raw.ch_names: ['FP1-F7', 'T8-P8-0', 'F7-T7', 'T8-P8-1', 'CZ-PZ']
        #
        # (Verified on mne 1.12.1; a triple duplicate yields -0/-1/-2.) A
        # channel that appears only once keeps its bare name, which is why
        # channel_matches() has to accept both forms.
        #
        # raw.ch_names is itself in header order, so scanning it front-to-back
        # and taking the first match is equivalent to "the lowest suffix",
        # which is equivalent to "the first header slot". All three coincide;
        # the selection is deterministic and identical in every file, and no
        # suffix parsing is needed. The obvious-looking alternative -- sorting
        # the candidates by name -- would be strictly worse, since it breaks
        # once a file has ten or more copies of a channel ("-10" < "-2").
        #
        # The one thing this does NOT establish is that the two copies carry
        # the same signal. They are the same derivation recorded twice, and
        # picking the first slot is consistent but still a choice. See
        # audit_real_data.py section 3, which samples the real files and
        # reports whether the copies are bit-identical.
        # --------------------------------------------------------------
        channels_to_pick = []
        for required_ch in self.channels:
            for file_ch in raw.ch_names:
                if channel_matches(required_ch, file_ch):
                    channels_to_pick.append(file_ch)
                    break
            else:
                raise ValueError(f"{edf_path.name}: required channel {required_ch!r} not found")

        raw = raw.pick(channels_to_pick)
        # MNE only guarantees pick() preserves the requested order from ~1.5;
        # assert rather than trust it, because the model reads channels
        # positionally and a silent reorder is unrecoverable downstream.
        if raw.ch_names != channels_to_pick:
            raise RuntimeError(
                f"{edf_path.name}: mne reordered the picked channels "
                f"({raw.ch_names} != {channels_to_pick}). Upgrade mne to >= 1.5."
            )
        if raw.info["sfreq"] != SAMPLING_FREQ:
            raise ValueError(
                f"{edf_path.name}: sampling rate is {raw.info['sfreq']} Hz, "
                f"expected {SAMPLING_FREQ} Hz"
            )
        if preload:
            raw.load_data(verbose="ERROR")

        return raw

    def __getitem__(self, idx: int) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
        edf_path = self.edf_file_paths[idx]
        file_metadata = self.edf_metadata[self._patient_id(edf_path)][edf_path.name]
        raw = self._pre_loaded[idx] if self._pre_loaded is not None else self._get_raw(edf_path)
        return raw, file_metadata

    def __len__(self) -> int:
        return len(self.edf_file_paths)

    def report_dropped_files(self) -> str:
        lines = []
        for patient_id, files in sorted(self.dropped_files.items()):
            if files:
                lines.append(f"chb{patient_id:02d}:")
                lines.extend(f"    {n}: {r}" for n, r in sorted(files.items()))
        return "\n".join(lines) if lines else "No files were dropped."

    def seizure_count(self) -> int:
        return sum(len(md["seizures"]) for p in self.edf_metadata.values() for md in p.values())

    def __repr__(self) -> str:
        return (
            "CHB_MIT_Files\n"
            f"  Patients:     {self.patients}\n"
            f"  Channels:     {len(self.channels)} ({', '.join(self.channels[:4])}, ...)\n"
            f"  Split:        {self.split} {self.split_ratios or ''}\n"
            f"  EDF files:    {len(self.edf_file_paths)}\n"
            f"  Seizures:     {self.seizure_count()}\n"
            f"  In memory:    {self.load_into_memory}\n"
        )
