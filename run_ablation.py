"""
Runs the KoMA-RAG ablation study (paper Table I) end to end:
  1. Seeds the shared memory DB (db/test) with a REFLECTION pass, since the
     two memory-based configs need real few-shot examples to retrieve.
  2. Runs the 4 ablation configs, each into its own ./result/<name>/ folder.
  3. Aggregates each config's per-episode CSVs into a mean +/- std table.

Usage:
  python run_ablation.py                 # seed + all 4 configs + aggregate
  python run_ablation.py --aggregate-only  # just rebuild the table from
                                            # whatever ./result/<name>/ CSVs
                                            # already exist (no runs)

Config flags are edited in-place in config.yaml before each subprocess run
(main.py reads config.yaml at import time), then the original config.yaml
is restored when the script exits. RESULT_FOLDER / SIMULATION_DURATION are
passed as env vars, which main.py reads with those exact names.
"""
import os
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(REPO_ROOT, "config.yaml")
RESULT_ROOT = os.path.join(REPO_ROOT, "result")

EPISODES_PER_CONFIG = 20
SEED_EPISODES = 10  # bump for a richer memory store; costs extra API calls

SEED_RUN = {
    "key": "seed",
    "name": "_seed_memory",
    "folder": os.path.join(RESULT_ROOT, "_seed_memory"),
    "episodes": SEED_EPISODES,
    "flags": {"USE_MEMORY": True, "REFLECTION": True, "ENABLE_MASTER": False, "ENABLE_VERIFICATION": False},
}

# Paper Table I ablation configs, in report order.
ABLATION_RUNS = [
    {
        "key": "base_koma",
        "name": "Base KoMA",
        "folder": os.path.join(RESULT_ROOT, "base_koma"),
        "flags": {"USE_MEMORY": False, "REFLECTION": False, "ENABLE_MASTER": False, "ENABLE_VERIFICATION": False},
    },
    {
        "key": "koma_master",
        "name": "KoMA + Master",
        "folder": os.path.join(RESULT_ROOT, "koma_master"),
        "flags": {"USE_MEMORY": False, "REFLECTION": False, "ENABLE_MASTER": True, "ENABLE_VERIFICATION": False},
    },
    {
        "key": "koma_verification",
        "name": "KoMA + Verification",
        "folder": os.path.join(RESULT_ROOT, "koma_verification"),
        "flags": {"USE_MEMORY": True, "REFLECTION": False, "ENABLE_MASTER": False, "ENABLE_VERIFICATION": True},
    },
    {
        "key": "koma_rag_full",
        "name": "Full KoMA-RAG",
        "folder": os.path.join(RESULT_ROOT, "koma_rag_full"),
        "flags": {"USE_MEMORY": True, "REFLECTION": False, "ENABLE_MASTER": True, "ENABLE_VERIFICATION": True},
    },
]


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def set_config_flags(updates):
    """Rewrite only the given KEY: value lines in config.yaml; preserves
    everything else (comments, ordering, unrelated keys) as-is."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    for key, value in updates.items():
        value_str = "true" if value is True else "false" if value is False else str(value)
        pattern = re.compile(rf"^({re.escape(key)}:)(.*)$", re.MULTILINE)

        def _replace(m, value_str=value_str):
            comment = ""
            if "#" in m.group(2):
                comment = "  #" + m.group(2).split("#", 1)[1]
            return f"{key}: {value_str}{comment}"

        text, n = pattern.subn(_replace, text, count=1)
        if n == 0:
            raise ValueError(f"Key {key!r} not found in {CONFIG_PATH}")

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(text)


def run_one(run_spec):
    """Run main.py as a subprocess for one config, streaming + logging output."""
    name = run_spec["name"]
    folder = run_spec["folder"]
    episodes = run_spec.get("episodes", EPISODES_PER_CONFIG)

    # Resume-safety: main.py appends to episode_summary.csv, so a naive rerun
    # over an existing folder would double-count episodes. Skip if already
    # complete; move a partial folder aside so this run starts clean.
    summary_path = os.path.join(folder, "episode_summary.csv")
    if os.path.exists(summary_path):
        existing = pd.read_csv(summary_path)
        if len(existing) >= episodes:
            log(f"'{name}' already has {len(existing)}/{episodes} episodes in {folder}; skipping.")
            return True
        stale = f"{folder}_incomplete_{time.strftime('%Y%m%d_%H%M%S')}"
        log(f"'{name}' has a partial prior run ({len(existing)}/{episodes} episodes); "
            f"moving it to {stale} and starting fresh.")
        os.rename(folder, stale)

    os.makedirs(folder, exist_ok=True)
    set_config_flags(run_spec["flags"])
    log(f"=== Starting '{name}' ({episodes} episodes) -> {folder} ===")
    log(f"    flags: {run_spec['flags']}")

    env = os.environ.copy()
    env["RESULT_FOLDER"] = folder
    env["SIMULATION_DURATION"] = str(episodes)

    driver_log_path = os.path.join(folder, "run_ablation_stdout.log")
    with open(driver_log_path, "a", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            logf.write(line)
            if "Simulation" in line and "Done" in line:
                log(f"[{name}] {line.strip()}")
        proc.wait()

    if proc.returncode != 0:
        log(f"!!! '{name}' exited with code {proc.returncode}; see {driver_log_path}")
        return False

    log(f"=== Finished '{name}' ===")
    return True


def aggregate_results(runs=ABLATION_RUNS):
    rows = []
    for run_spec in runs:
        name = run_spec["name"]
        summary_path = os.path.join(run_spec["folder"], "episode_summary.csv")
        if not os.path.exists(summary_path):
            log(f"[aggregate] no episode_summary.csv for '{name}' yet, skipping")
            continue
        df = pd.read_csv(summary_path)

        def fmt(series):
            mean = np.nanmean(series) if len(series) else float("nan")
            std = np.nanstd(series, ddof=1) if len(series) > 1 else 0.0
            if np.isnan(mean):
                return "N/A"
            return f"{mean:.3f} +/- {std:.3f}"

        collision_pct = df["collision"] * 100.0
        rows.append({
            "Config": name,
            "Reward (mean +/- std)": fmt(df["total_reward"]),
            "Factual Consistency (mean +/- std)": fmt(df["factual_consistency"]),
            "Collision Rate % (mean +/- std)": fmt(collision_pct),
            "Episodes": len(df),
        })

    if not rows:
        log("[aggregate] nothing to aggregate yet")
        return

    table = pd.DataFrame(rows)
    print("\n" + table.to_string(index=False) + "\n")

    # Written by hand (no tabulate dependency): pipe-delimited GFM table.
    cols = list(table.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")

    lines.append("")
    lines.append(
        "Note: no external baseline was run against the original PJLab-ADG/KoMA "
        "codebase this weekend. 'Base KoMA' above is an internal reproduction "
        "(this repo with all ablation flags off), not a run of the original "
        "authors' code -- call this out explicitly wherever this table is used."
    )

    md_path = os.path.join(RESULT_ROOT, "ablation_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log(f"[aggregate] wrote {md_path}")


def main():
    if "--aggregate-only" in sys.argv:
        aggregate_results()
        return

    # --only=<key> runs a single phase (seed, base_koma, koma_master,
    # koma_verification, koma_rag_full) and skips the rest -- handy for
    # environments with a hard session time limit (e.g. Kaggle): run one
    # phase, checkpoint/save, then continue with the next phase next session.
    # Thanks to the resume-safety in run_one(), phases already completed on
    # an earlier invocation are skipped automatically even without --only.
    only = None
    for arg in sys.argv[1:]:
        if arg.startswith("--only="):
            only = arg.split("=", 1)[1].strip().lower()

    all_runs = [SEED_RUN] + ABLATION_RUNS
    if only is not None and only not in {r["key"] for r in all_runs}:
        valid = ", ".join(r["key"] for r in all_runs)
        raise SystemExit(f"Unknown --only={only!r}. Valid keys: {valid}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        original_config = f.read()

    try:
        os.makedirs(RESULT_ROOT, exist_ok=True)
        for run_spec in all_runs:
            if only is not None and run_spec["key"] != only:
                continue
            if run_spec["key"] == "seed":
                log("Seeding shared memory DB (db/test) via REFLECTION pass...")
            run_one(run_spec)

        if only is None:
            log("All configs done. Aggregating...")
            aggregate_results()
        else:
            log(f"Ran phase '{only}' only. Once all phases are done, run with --aggregate-only.")
    finally:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(original_config)
        log("Restored original config.yaml.")


if __name__ == "__main__":
    main()
