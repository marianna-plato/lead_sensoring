#!/usr/bin/env bash
#BSUB -q c27666
#BSUB -J 02_prepare_rfd3_double_inputs
#BSUB -n 1
#BSUB -W 00:15
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o logs/02_prepare_rfd3_double_inputs/%J.out
#BSUB -e logs/02_prepare_rfd3_double_inputs/%J.err

set -euo pipefail

cd "$LS_SUBCWD"

EXPERIMENT="exp_doubles_01"
STRUCTURE="data/raw/5gpe.cif"
SCRIPT="scripts/2_prepare_rdf3_doubles_inputs.py"
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

mkdir -p logs/02_prepare_rfd3_double_inputs
mkdir -p "$OUTDIR"

python "$SCRIPT" \
    --structure "$STRUCTURE" \
    --metal PB \
    --radii 3 4 5 6 \
    --outdir "$OUTDIR" \
    --design-name "$DESIGN_NAME" \
    --design-mode monomeric \
    --target-window 6 \
    --nterm-length 10-50 \
    --linker-length 5-20 \
    --cterm-length 10-50 \
    --ori-distances  13 15 18 \
    --orbit-angle-deg 180 \
    --copy-distance-scale 1.0 \
    --rotation-axis-mode triad_normal_projected \
    --inter-motif-linker 5-20 \
    --open-space-directions 1024 \
    --open-space-max-distance 14.0 \
    --open-space-cone-angle 45.0 \
    --chemical-weight 0.0 \
    --include-metal

echo
echo "Done."
echo "Main outputs:"
echo "  ${OUTDIR}/motif_scan/selected_pb_site.json"
echo "  ${OUTDIR}/motif_scan/pb_vectors.bild"
echo "  ${OUTDIR}/double_motif_structures/"
echo "  ${OUTDIR}/rfdiffusion_inputs/rfd3_double_inputs.json"
echo "  ${OUTDIR}/rfdiffusion_inputs/double_motif_geometry_metadata.json"
echo "  ${OUTDIR}/rfdiffusion_inputs/per_design/"
