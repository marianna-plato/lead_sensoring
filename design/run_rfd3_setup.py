# Auto-generated: runs notebook cells up to and including the RFD3 submit script.
# Equivalent to running cells 3, 4, 7, 9 in Binder_design_course.ipynb.
import os, sys, re, glob, json, math, shutil, csv
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Cell 3: locate repo root ──────────────────────────────────────────────────
REPO_ROOT = Path.cwd()
if not ((REPO_ROOT / "inputs").is_dir() and (REPO_ROOT / "lib").is_dir()):
    raise RuntimeError(
        f"cwd={REPO_ROOT} is not the repo root (no inputs/ and lib/).\n"
        "Set it manually: REPO_ROOT = Path('/dtu/blackhole/09/214281/lead_sensoring/design')"
    )

LEAD_INPUTS = REPO_ROOT / "lead_inputs"
if not LEAD_INPUTS.is_dir():
    raise RuntimeError(
        f"lead_inputs/ not found under {REPO_ROOT}. "
        "Run 00_motif_scan.sh first to generate the motif inputs."
    )

STRUCTURE_CIF = (REPO_ROOT.parent / "data" / "raw" / "5gpe.cif").resolve()
CONTEXT_PDB = (LEAD_INPUTS / "rfd3_context_fixed_segments_plus_pb.pdb").resolve()

for p in [STRUCTURE_CIF, CONTEXT_PDB]:
    if not p.exists():
        print(f"[WARNING] Not found: {p}")

sys.path.insert(0, str(REPO_ROOT / "lib" / "lib"))
import jupyter_utils
from rf3_metrics import gather_rf3_metrics

print("Repo root  :", REPO_ROOT)
print("Lead inputs:", LEAD_INPUTS)
print("5GPE CIF   :", STRUCTURE_CIF)
print("Context PDB:", CONTEXT_PDB)
print("User       :", os.environ.get("USER", "?"))

# ── Cell 4: working directories ───────────────────────────────────────────────
experiment = "exp_01"
WORK = REPO_ROOT / "work" / experiment

_subdirs = ["cmds", "submit", "logs", "configs", "scores",
            "diffusion_out", "mpnn_out", "rf3_out", "best_binders"]
for d in _subdirs:
    (WORK / d).mkdir(parents=True, exist_ok=True)

cmds_dir          = str(WORK / "cmds")
submit_dir        = str(WORK / "submit")
logs_dir          = str(WORK / "logs")
configs_dir       = str(WORK / "configs")
scores_dir        = str(WORK / "scores")
diffusion_out_dir = str(WORK / "diffusion_out")
mpnn_out_dir      = str(WORK / "mpnn_out")
rf3_out_dir       = str(WORK / "rf3_out")
best_binders_dir  = str(WORK / "best_binders")

SPYTAG_LINKER = "GSGSGS"
SPYTAG_SEQ    = "AHIVMVDAYKPTK"
SPYTAG        = SPYTAG_LINKER + SPYTAG_SEQ

pd.set_option("display.max_columns", None)
print("Working dir:", WORK)
print("SpyTag     :", SPYTAG, f"({len(SPYTAG)} aa)")

# ── Cell 7: build RFD3 input JSON ─────────────────────────────────────────────
design_name = "5gpe_pb_motif_r4"
input_pdb = str(STRUCTURE_CIF)
contig = "80-140,/0,C72-84,/0,D107-128"
length = "115-175"
redesign_motif_sidechains = False
select_hotspots = {
    "C78":  "SG",
    "D113": "SG",
    "D122": "SG",
}
infer_ori_strategy = "hotspots"
is_non_loopy       = True

rfd3_json = str(Path(configs_dir) / "rfd3_input.json")

payload = {
    design_name: {
        "dialect": 2,
        "input": input_pdb,
        "contig": contig,
        "length": length,
        "redesign_motif_sidechains": redesign_motif_sidechains,
        "select_hotspots": select_hotspots,
        "infer_ori_strategy": infer_ori_strategy,
        "is_non_loopy": is_non_loopy,
    }
}
with open(rfd3_json, "w") as f:
    json.dump(payload, f, indent=2)

print("Wrote RFD3 input ->", rfd3_json)
print("Design name:", design_name)
print("Contig     :", contig)
print("Length     :", length)
print("Hotspots   :", select_hotspots)
print()
print("Pre-computed reference (should match):")
ref = LEAD_INPUTS / "rfd3_input_radius_4p0.json"
print(" ", ref)
if ref.exists():
    with open(ref) as f:
        ref_data = json.load(f)
    ref_contig = list(ref_data.values())[0]["contig"]
    print(f"  Reference contig: {ref_contig}")
    print(f"  Contig match: {ref_contig == contig}")

# ── Cell 9: write RFD3 submit script ─────────────────────────────────────────
queue        = "c27666"
job_name     = "rfd3_pb"
cores        = 4
gpu_spec     = "num=1:mode=exclusive_process"
time_limit   = "1:00"
mem          = "10GB"

CKPT_PATH = "/dtu/blackhole/00/c27666/27666_Protein_Design/weights/rfd3_latest.ckpt"

diffusion_batch_size = 1
n_batches            = 4

script_path = os.path.join(submit_dir, "rfd3.sh")

script = f"""#!/bin/sh
#BSUB -q {queue}
#BSUB -J {job_name}
#BSUB -n {cores}
#BSUB -gpu "{gpu_spec}"
#BSUB -W {time_limit}
#BSUB -R "rusage[mem={mem}]"
#BSUB -R "span[hosts=1]"
#BSUB -o {logs_dir}/%J.out
#BSUB -e {logs_dir}/%J.err

mkdir -p {logs_dir} {diffusion_out_dir}
module load cuda/12.4
source /dtu/blackhole/00/c27666/miniforge3/etc/profile.d/conda.sh
conda activate /dtu/blackhole/00/c27666/miniforge3/envs/protein-design

# Workarounds for MIG GPU slices on c27666 A100s
export DISABLE_CUEQUIVARIANCE=true           # skip cuEquivariance (crashes on MIG via pynvml)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # reduce fragmentation on ~20GB slice

rfd3 design \\
    out_dir="{diffusion_out_dir}" \\
    inputs="{rfd3_json}" \\
    ckpt_path="{CKPT_PATH}" \\
    diffusion_batch_size={diffusion_batch_size} \\
    n_batches={n_batches} \\
    low_memory_mode=True \\
    inference_sampler.step_scale=3 \\
    inference_sampler.gamma_0=0.2

echo "Completed at $(date)"
"""

with open(script_path, "w") as f:
    f.write(script)

print("Wrote", script_path)
print(f"\nThis will generate {diffusion_batch_size * n_batches} backbone(s).")
print("\nSubmit in a terminal:")
print("  bsub < " + script_path)
print("Monitor: bstat   (job disappears when done)")
print("Logs   :", logs_dir)
