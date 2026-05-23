"""Feature-group ablation across all 4 tasks × full 10-model suite.

For each (task, feature-group) pair we run `src.train` with
`--drop-prefixes <prefixes>` and `--tag ablate_<group>`, producing a leaderboard
under `outputs/reports/<task>_ablate_<group>/`. Compared to the
canonical-task baseline leaderboards (already on disk), the macro-F1 delta
per group per task tells us how much each functional group contributes.

Run as a module: `python -m scripts.run_feature_ablation`.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PY = sys.executable

# 12 functional groups. Each entry: (label, [prefix, ...]) — prefixes are passed
# verbatim to `--drop-prefixes` and matched against column-name starts.
GROUPS: list[tuple[str, list[str]]] = [
    ("ditto",         ["ditto_"]),
    ("alphamissense", ["alphamissense_"]),
    ("revel",         ["revel_"]),
    ("cadd",          ["cadd_", "cadd_exome_"]),
    ("metarnn",       ["metarnn_"]),
    ("bayesdel",      ["bayesdel_"]),
    ("chasmplus",     ["chasmplus"]),  # 68 features — biggest single group
    ("other_vep",     [
        # The residual meta-classifier / VEP-predictor families.
        "clinpred_", "mistic_", "mutpred1_", "mutpred2_", "vest_", "sift_",
        "polyphen2_", "fathmm_", "lrt_", "mutation_assessor_",
        "mutationtaster_", "dann_", "cscape_", "eve_", "esm1b_",
        "primateai_", "gmvp_", "varity_", "provean_", "metalr_", "metasvm_",
        "phdsnpg_", "genocanyon_", "funseq2_",
    ]),
    ("conservation",  ["phastcons_", "phylop_", "gerp_", "siphy_"]),
    ("population_af", ["allofus250k_", "gnomad_", "gnomad3_", "alfa_", "regeneron_"]),
    ("functional",    ["fitcons_", "ncer_"]),
    ("position",      ["hg19_pos", "original_input_pos"]),
]

TASKS = ["4class", "2class", "5class", "3class"]


def run_one(task: str, group: str, prefixes: list[str]) -> tuple[str, str, int, float]:
    log = Path(f"outputs/reports/train_{task}_ablate_{group}.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-m", "src.train",
           "--task", task,
           "--drop-prefixes", *prefixes,
           "--tag", f"ablate_{group}"]
    t0 = time.time()
    with log.open("w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    return task, group, p.returncode, dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-workers", type=int, default=6,
                    help="Parallel processes. Each launches one src.train run "
                         "covering all 10 models on one (task, group).")
    args = ap.parse_args()

    jobs = [(t, g, p) for t in TASKS for g, p in GROUPS]
    print(f"[ablation] {len(jobs)} runs queued × full 10-model suite each "
          f"({args.max_workers} parallel)")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(run_one, t, g, p) for (t, g, p) in jobs]
        for f in as_completed(futs):
            task, group, rc, dt = f.result()
            status = "OK" if rc == 0 else f"FAIL rc={rc}"
            print(f"  [{status}] {task}/{group:14s}  ({dt:>5.1f}s)")
    print(f"[ablation] all done in {(time.time()-t0)/60:.1f} min wall-clock")


if __name__ == "__main__":
    main()
