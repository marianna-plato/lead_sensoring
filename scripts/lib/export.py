"""Export reduced structure contexts for RFD3 design."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from structure_io import save_structure


def parse_contig_segments(contig: str) -> list[tuple[str, int, int]]:
    """Parse fixed target segments from an RFD3 contig."""
    segments = []
    for part in contig.split(","):
        part = part.strip()
        if not part or part == "/0":
            continue
        if re.fullmatch(r"\d+-\d+", part):
            continue

        match = re.fullmatch(r"([A-Za-z0-9])(-?\d+)-(-?\d+)", part)
        if not match:
            continue

        segments.append((match.group(1), int(match.group(2)), int(match.group(3))))

    return segments


def get_first_rfd3_payload(path: Path) -> dict[str, Any]:
    """Read the first design entry from an RFD3 JSON file."""
    with path.open() as handle:
        data = json.load(handle)

    if len(data) != 1:
        raise ValueError("Expected exactly one design entry in the RFD3 JSON file")

    design_name = next(iter(data))
    return data[design_name]


def residue_mask(atoms, chain: str, start: int, end: int) -> np.ndarray:
    """Return a mask for atoms from one chain and residue interval."""
    return (atoms.chain_id == chain) & (atoms.res_id >= start) & (atoms.res_id <= end) & (~atoms.hetero)


def selected_pb_mask(atoms, selected_site: dict[str, Any], distance_tolerance: float = 0.01) -> np.ndarray:
    """Select the Pb atom using its stored coordinate."""
    pb_coord = np.array(selected_site["metal_coord"], dtype=float)
    distances = np.linalg.norm(atoms.coord - pb_coord[None, :], axis=1)
    return distances <= distance_tolerance


def triad_residue_mask(atoms, selected_site: dict[str, Any]) -> np.ndarray:
    """Return a mask for full residues from the selected Cys triad."""
    mask = np.zeros(len(atoms), dtype=bool)
    for donor in selected_site["triad_donors"]:
        chain = donor["chain"]
        res_id = int(donor["res_id"])
        mask |= (atoms.chain_id == chain) & (atoms.res_id == res_id) & (~atoms.hetero)
    return mask


def export_context_structures(
    atoms,
    selected_site: dict[str, Any],
    rfd3_payload: dict[str, Any],
    outdir: Path,
) -> dict[str, str]:
    """Export fixed-segment context and motif-only PDB files."""
    outdir.mkdir(parents=True, exist_ok=True)

    contig = rfd3_payload["contig"]
    segments = parse_contig_segments(contig)
    if not segments:
        raise ValueError(f"No fixed target segments found in contig: {contig}")

    context_mask = np.zeros(len(atoms), dtype=bool)
    for chain, start, end in segments:
        context_mask |= residue_mask(atoms, chain, start, end)

    pb_mask = selected_pb_mask(atoms, selected_site)
    triad_mask = triad_residue_mask(atoms, selected_site)

    context_pdb = outdir / "rfd3_context_fixed_segments_plus_pb.pdb"
    motif_pdb = outdir / "selected_pb_cys3_motif_only.pdb"

    save_structure(context_pdb, atoms[context_mask | pb_mask])
    save_structure(motif_pdb, atoms[triad_mask | pb_mask])

    return {
        "context_pdb": str(context_pdb),
        "motif_pdb": str(motif_pdb),
    }
