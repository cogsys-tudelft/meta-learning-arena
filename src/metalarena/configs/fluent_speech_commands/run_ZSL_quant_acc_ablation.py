#!/usr/bin/env python3
"""
Quantization bit-width sweep runner: fit + test.

Generates 15 total run configs derived from a baseline quant YAML:

  Sweep 1 ("linear" fixed-point family), 4 bit-widths x 3 seeds = 12 runs:
      weight_quant: Int8WeightPerTensorFixedPoint
      act_quant / in_quant: ShiftedUint8ActPerTensorFixedPoint
      bit_width in {16, 8, 6, 4} (same value applied to weight/act/in)
      bias_quant: Int16Bias, bit_width 32 (always, untouched)

  Sweep 2 ("PoT" 4-bit family), 3 seeds = 3 runs:
      weight_quant: PoT4WeightPerTensorFixedPoint, bit_width 4
      act_quant / in_quant: ShiftedUint8ActPerTensorFixedPoint, bit_width 4
      bias_quant: Int16Bias, bit_width 32 (always, untouched)

For each run: run `autolightning fit`, find the checkpoint that run produced
(via its wandb run id), then run `autolightning test` against that checkpoint.

The 15 runs are distributed round-robin across 5 "sessions" (index % 5), so
each session gets a mix of bit-widths rather than one slow bucket.

Usage:
    python run_quant_sweep.py <session_id 0-4>
    python run_quant_sweep.py dry

Run this in 5 separate tmux sessions with session_id 0, 1, 2, 3, 4.

`dry` runs only the very first config (linear_16b_seed3406) through fit and
test, using its own status log (sweep_status_dryrun.log), so you can verify
the whole pipeline -- config generation, autolightning fit, checkpoint
discovery, autolightning test -- works before launching all 5 real sessions.
"""

import copy
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

# ---- Paths: adjust if your layout differs ----
BASELINE_CONFIG = Path("./quant_zero_shot_96k_28k.yaml")
OTHER_CONFIGS = [
    "./zero_shot_data.yaml",
    "../machine_specific/project_settings.yaml",
    "../machine_specific/lug.py",
]
GENERATED_DIR = Path("./generated_configs")
LOG_DIR = Path("./sweep_logs")
CHECKPOINT_ROOT = Path(
    "/space2/ddenblanken/Experiments/meta-learning-arena/test_wandb_lightning"
)

SEEDS = [3406, 3407, 3408]
LINEAR_BITWIDTHS = [16, 8, 6, 4]
NUM_SESSIONS = 5

WANDB_URL_RE = re.compile(r"wandb\.ai/[^\s]+/runs/([A-Za-z0-9]+)")
WANDB_LOCAL_RE = re.compile(r"wandb/run-\d{8}_\d{6}-([A-Za-z0-9]+)")


def build_run_list():
    runs = []
    for bw in LINEAR_BITWIDTHS:
        for seed in SEEDS:
            runs.append(
                {
                    "name": f"linear_{bw}b_seed{seed}",
                    "family": "linear",
                    "bit_width": bw,
                    "seed": seed,
                }
            )
    for seed in SEEDS:
        runs.append(
            {
                "name": f"pot4_seed{seed}",
                "family": "pot",
                "bit_width": 4,
                "seed": seed,
            }
        )
    return runs


def make_config(baseline: dict, run: dict) -> dict:
    cfg = copy.deepcopy(baseline)
    q = cfg["model"]["init_args"]

    if run["family"] == "linear":
        q["weight_quant"]["init_args"]["class_paths"] = ["Int8WeightPerTensorFixedPoint"]
    else:  # pot
        q["weight_quant"]["init_args"]["class_paths"] = ["PoT4WeightPerTensorFixedPoint"]

    bw = run["bit_width"]
    q["weight_quant"]["init_args"]["init_args"]["bit_width"] = bw

    q["act_quant"]["init_args"]["class_paths"] = ["ShiftedUint8ActPerTensorFixedPoint"]
    q["act_quant"]["init_args"]["init_args"]["bit_width"] = bw

    q["in_quant"]["init_args"]["class_paths"] = ["ShiftedUint8ActPerTensorFixedPoint"]
    q["in_quant"]["init_args"]["init_args"]["bit_width"] = bw

    # bias_quant is intentionally left untouched: stays Int16Bias / 32-bit

    cfg["seed_everything"] = run["seed"]
    return cfg


def log_status(log_path: Path, run_name: str, status: str, extra: str = ""):
    with open(log_path, "a") as f:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{ts}\t{run_name}\t{status}\t{extra}\n")


def extract_run_id(run_log_path: Path):
    """Best-effort extraction of the wandb run id from a run's captured log."""
    text = run_log_path.read_text(errors="ignore")
    matches = WANDB_URL_RE.findall(text) or WANDB_LOCAL_RE.findall(text)
    if matches:
        return matches[-1]
    return None


def find_run_id_by_mtime(start_time: float):
    """Fallback: newest checkpoint dir created after this run started.
    NOTE: unreliable if other parallel sessions also started a run around
    the same time -- only used if log parsing fails, and flagged as such."""
    candidates = []
    if not CHECKPOINT_ROOT.exists():
        return None
    for d in CHECKPOINT_ROOT.iterdir():
        ckpt_dir = d / "checkpoints"
        if ckpt_dir.is_dir():
            try:
                mtime = ckpt_dir.stat().st_mtime
            except OSError:
                continue
            if mtime >= start_time:
                candidates.append((mtime, d.name))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def find_checkpoint(run_id: str):
    ckpt_dir = CHECKPOINT_ROOT / run_id / "checkpoints"
    if not ckpt_dir.is_dir():
        return None
    ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        return None
    if len(ckpts) == 1:
        return ckpts[0]

    # Multiple checkpoints: pick the one with the highest step number.
    def step_num(p):
        m = re.search(r"step=(\d+)", p.name)
        return int(m.group(1)) if m else -1

    ckpts.sort(key=step_num)
    return ckpts[-1]


def run_cmd(cmd, log_path: Path):
    with open(log_path, "w") as log_f:
        subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, check=True)


def process_run(run, baseline, run_logs_dir, status_log, label):
    """Generate config, fit, locate checkpoint, test. Returns nothing;
    all outcomes are recorded in status_log."""
    run_name = run["name"]
    cfg = make_config(baseline, run)
    cfg_path = GENERATED_DIR / f"{run_name}.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    fit_cmd = ["autolightning", "fit", "-c", str(cfg_path)] + sum(
        (["-c", c] for c in OTHER_CONFIGS), []
    )
    fit_log_path = run_logs_dir / f"{run_name}_fit.log"

    log_status(status_log, run_name, "FIT_STARTED", " ".join(fit_cmd))
    print(f"[{label}] Fitting {run_name} ...")
    start_time = time.time()

    try:
        run_cmd(fit_cmd, fit_log_path)
        log_status(status_log, run_name, "FIT_COMPLETED")
        print(f"[{label}] {run_name} FIT_COMPLETED")
    except subprocess.CalledProcessError as e:
        log_status(status_log, run_name, "FIT_FAILED", f"exit_code={e.returncode}, see {fit_log_path}")
        print(f"[{label}] {run_name} FIT_FAILED, skipping test, continuing")
        return
    except Exception as e:
        log_status(status_log, run_name, "FIT_FAILED", f"{type(e).__name__}: {e}")
        print(f"[{label}] {run_name} FIT_FAILED ({e}), skipping test, continuing")
        return

    # --- locate the checkpoint this fit run produced ---
    run_id = extract_run_id(fit_log_path)
    used_fallback = False
    if run_id is None:
        run_id = find_run_id_by_mtime(start_time)
        used_fallback = True

    if run_id is None:
        log_status(status_log, run_name, "TEST_SKIPPED", "could not determine wandb run id")
        print(f"[{label}] {run_name}: could not find run id, skipping test")
        return

    ckpt_path = find_checkpoint(run_id)
    if ckpt_path is None:
        log_status(
            status_log, run_name, "TEST_SKIPPED",
            f"run_id={run_id} (fallback={used_fallback}), no checkpoint found under {CHECKPOINT_ROOT / run_id / 'checkpoints'}",
        )
        print(f"[{label}] {run_name}: no checkpoint found for run_id={run_id}, skipping test")
        return

    if used_fallback:
        log_status(
            status_log, run_name, "WARNING",
            f"run_id={run_id} determined via mtime fallback, not log parsing -- verify this is correct",
        )

    test_cmd = (
        ["autolightning", "test", "-c", str(cfg_path)]
        + sum((["-c", c] for c in OTHER_CONFIGS), [])
        + ["--ckpt_path", str(ckpt_path)]
    )
    test_log_path = run_logs_dir / f"{run_name}_test.log"

    log_status(status_log, run_name, "TEST_STARTED", f"run_id={run_id} ckpt={ckpt_path}")
    print(f"[{label}] Testing {run_name} (ckpt: {ckpt_path}) ...")

    try:
        run_cmd(test_cmd, test_log_path)
        log_status(status_log, run_name, "TEST_COMPLETED")
        print(f"[{label}] {run_name} TEST_COMPLETED")
    except subprocess.CalledProcessError as e:
        log_status(status_log, run_name, "TEST_FAILED", f"exit_code={e.returncode}, see {test_log_path}")
        print(f"[{label}] {run_name} TEST_FAILED, continuing to next run")
    except Exception as e:
        log_status(status_log, run_name, "TEST_FAILED", f"{type(e).__name__}: {e}")
        print(f"[{label}] {run_name} TEST_FAILED ({e}), continuing to next run")


def main():
    valid_args = {"0", "1", "2", "3", "4", "dry"}
    if len(sys.argv) != 2 or sys.argv[1] not in valid_args:
        print("Usage: python run_quant_sweep.py <session_id 0-4 | dry>")
        sys.exit(1)
    arg = sys.argv[1]

    GENERATED_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    run_logs_dir = LOG_DIR / "run_logs"
    run_logs_dir.mkdir(exist_ok=True)

    with open(BASELINE_CONFIG) as f:
        baseline = yaml.safe_load(f)

    all_runs = build_run_list()

    if arg == "dry":
        status_log = LOG_DIR / "sweep_status_dryrun.log"
        dry_run = all_runs[0]
        print(f"[dry-run] Running single config end-to-end: {dry_run['name']}")
        process_run(dry_run, baseline, run_logs_dir, status_log, "dry-run")
        print(f"Dry run done. Status log: {status_log}")
        return

    session_id = int(arg)
    status_log = LOG_DIR / f"sweep_status_session{session_id}.log"
    my_runs = [r for i, r in enumerate(all_runs) if i % NUM_SESSIONS == session_id]
    label = f"session {session_id}"

    print(f"Session {session_id}: {len(my_runs)} runs assigned: {[r['name'] for r in my_runs]}")

    for run in my_runs:
        process_run(run, baseline, run_logs_dir, status_log, label)

    print(f"Session {session_id} done. Status log: {status_log}")


if __name__ == "__main__":
    main()
