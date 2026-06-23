#!/usr/bin/env bash
#BSUB -q c27666
#BSUB -J 01_prepare_rfd3_inputs
#BSUB -n 1
#BSUB -W 00:15
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/01_prepare_rfd3_inputs/%J.out
#BSUB -e logs/01_prepare_rfd3_inputs/%J.err

set -euo pipefail

cd "$LS_SUBCWD"

EXPERIMENT="exp_02"
STRUCTURE="data/raw/5gpe.cif"
SCRIPT="scripts/1_prepare_rfd3_inputs.py"
OUTDIR="work/${EXPERIMENT}"
DESIGN_NAME="5gpe_pb_motif_r5"

CONDA_SH="/dtu/projects/dbl/foundry/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV="rfd3"

if [[ ! -f "$CONDA_SH" ]]; then
    echo "Missing conda.sh: $CONDA_SH" >&2
    exit 1
fi

source "$CONDA_SH"
conda activate "$CONDA_ENV"

if [[ ! -f "$STRUCTURE" ]]; then
    echo "Missing structure file: $STRUCTURE" >&2
    exit 1
fi

if [[ ! -f "$SCRIPT" ]]; then
    echo "Missing Python script: $SCRIPT" >&2
    exit 1
fi

mkdir -p logs/01_prepare_rfd3_inputs
mkdir -p "$OUTDIR"

python "$SCRIPT" \
    --structure "$STRUCTURE" \
    --metal PB \
    --radii 3 4 5 6 \
    --outdir "$OUTDIR" \
    --design-name "$DESIGN_NAME" \
    --design-mode monomeric \
    --target-window 4 \
    --nterm-length 10-50 \
    --linker-length 5-20 \
    --cterm-length 10-50 \
    --ori-keys scaffold_side_5A scaffold_side_8A scaffold_side_12A scaffold_side_15A \
    --open-space-directions 1024 \
    --open-space-max-distance 14.0 \
    --open-space-cone-angle 45.0 \
    --chemical-weight 0.0

echo
echo "Done."
echo "Main outputs:"
echo "  ${OUTDIR}/motif_scan/selected_pb_site.json"
echo "  ${OUTDIR}/motif_scan/pb_vectors.bild"
echo "  ${OUTDIR}/rfdiffusion_inputs/"
echo "  ${OUTDIR}/contexts/"
