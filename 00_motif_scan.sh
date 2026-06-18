#!/usr/bin/env bash
#BSUB -q c27666
#BSUB -J 00_motif_scan
#BSUB -n 1
#BSUB -W 00:10
#BSUB -R "rusage[mem=2GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/00_motif_scan/%J.out
#BSUB -e logs/00_motif_scan/%J.err

set -euo pipefail

cd "${LS_SUBCWD:-$PWD}"

# -----------------------------
# Project settings
# -----------------------------
EXPERIMENT="exp_01"

STRUCTURE="data/raw/5gpe.cif"

MOTIF_SCRIPT="scripts/1a_extract_motif_by_radius_literature_score.py"
RFD_INPUT_SCRIPT="scripts/1b_make_rfdiffusion_inputs.py"
CONTEXT_SCRIPT="scripts/1c_export_rfd3_context_structure.py"

MOTIF_OUTDIR="work/${EXPERIMENT}/motif_scan"
RFD_INPUT_OUTDIR="work/${EXPERIMENT}/rfdiffusion_inputs"

mkdir -p "${MOTIF_OUTDIR}"
mkdir -p "${RFD_INPUT_OUTDIR}"
mkdir -p logs/00_motif_scan

# -----------------------------
# Activate environment
# -----------------------------
source /dtu/projects/dbl/foundry/miniforge3/etc/profile.d/conda.sh
conda activate /dtu/projects/dbl/foundry/miniforge3/envs/rfd3

echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Structure: ${STRUCTURE}"
echo "Python: $(which python)"
echo "Experiment: ${EXPERIMENT}"

# -----------------------------
# Sanity checks
# -----------------------------
test -f "${STRUCTURE}"
test -f "${MOTIF_SCRIPT}"
test -f "${RFD_INPUT_SCRIPT}"
test -f "${CONTEXT_SCRIPT}"

python -c "import biotite, numpy; print('Environment OK: biotite + numpy')"

# -----------------------------
# 1. Select best Pb site and extract motifs by radius
# -----------------------------
python "${MOTIF_SCRIPT}" \
    --pdb "${STRUCTURE}" \
    --metal PB \
    --radii 3 4 5 6 8 \
    --donor-cutoff 3.4 \
    --gap-window 5 \
    --outdir "${MOTIF_OUTDIR}"

# -----------------------------
# 2. Build RFD3 input JSON from selected Pb motif
# -----------------------------
python "${RFD_INPUT_SCRIPT}" \
    --structure "${STRUCTURE}" \
    --selected-site "${MOTIF_OUTDIR}/selected_pb_site.json" \
    --motifs "${MOTIF_OUTDIR}/motifs_by_radius.json" \
    --radius 4 \
    --binder-length 80-140 \
    --target-window 6 \
    --design-name 5gpe_pb_motif_r4 \
    --hotspot-mode triad \
    --outdir "${RFD_INPUT_OUTDIR}"

# -----------------------------
# 3. Export reduced context PDBs for inspection/debugging
# -----------------------------
python "${CONTEXT_SCRIPT}" \
    --structure "${STRUCTURE}" \
    --selected-site "${MOTIF_OUTDIR}/selected_pb_site.json" \
    --rfd3-json "${RFD_INPUT_OUTDIR}/rfd3_input_radius_4p0.json" \
    --outdir "${RFD_INPUT_OUTDIR}"

echo "Done at $(date)"
echo "Motif outputs:"
echo "  ${MOTIF_OUTDIR}"
echo "RFD3 inputs:"
echo "  ${RFD_INPUT_OUTDIR}"