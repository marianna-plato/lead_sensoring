#!/usr/bin/env python3
"""
Export a reduced structure containing only the selected RFD3 target context.

Inputs:
    - Original PDB/mmCIF structure
    - selected_pb_site.json
    - RFD3 input JSON

Outputs:
    - context PDB with fixed target segments + selected Pb
    - motif-only PDB with triad Cys residues + selected Pb

This preserves original chain IDs and residue numbering.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import biotite.structure as struc
import biotite.structure.io as bsio


def load_structure(path: Path):
    """Load first model from PDB/mmCIF."""
    atoms = bsio.load_structure(str(path))
    if isinstance(atoms, struc.AtomArrayStack):
        atoms = atoms[0]
    return atoms


def parse_contig_segments(contig: str) -> list[tuple[str, int, int]]:
    """
    Parse fixed segments from a contig like:
        80-140,/0,C72-84,/0,D107-128

    Returns:
        [("C", 72, 84), ("D", 107, 128)]
    """
    segments = []

    for part in contig.split(","):
        part = part.strip()

        if not part or part == "/0":
            continue

        # Skip binder length such as 80-140
        if re.fullmatch(r"\d+-\d+", part):
            continue

        match = re.fullmatch(r"([A-Za-z0-9])(-?\d+)-(-?\d+)", part)
        if not match:
            continue

        chain = match.group(1)
        start = int(match.group(2))
        end = int(match.group(3))
        segments.append((chain, start, end))

    return segments


def get_rfd3_design_payload(path: Path) -> dict:
    """Read the first design entry from an RFD3 JSON."""
    with path.open() as handle:
        data = json.load(handle)

    if len(data) != 1:
        raise ValueError("Expected exactly one design entry in RFD3 JSON")

    design_name = next(iter(data))
    return data[design_name]


def residue_mask(atoms, chain: str, start: int, end: int) -> np.ndarray:
    """Mask atoms from one chain and residue interval."""
    return (
        (atoms.chain_id == chain) &
        (atoms.res_id >= start) &
        (atoms.res_id <= end) &
        (~atoms.hetero)
    )


def selected_pb_mask(atoms, selected_site: dict, distance_tolerance: float = 0.01) -> np.ndarray:
    """
    Select the Pb atom using its stored coordinate.

    This is more robust than relying only on atom index after reload.
    """
    pb_coord = np.array(selected_site["metal_coord"], dtype=float)
    distances = np.linalg.norm(atoms.coord - pb_coord[None, :], axis=1)

    return distances <= distance_tolerance


def triad_residue_mask(atoms, selected_site: dict) -> np.ndarray:
    """Mask full residues for selected Cys triad."""
    mask = np.zeros(len(atoms), dtype=bool)

    for donor in selected_site["triad_donors"]:
        chain = donor["chain"]
        res_id = int(donor["res_id"])

        mask |= (
            (atoms.chain_id == chain) &
            (atoms.res_id == res_id) &
            (~atoms.hetero)
        )

    return mask


def save_structure(path: Path, atoms) -> None:
    """Save AtomArray to PDB or CIF depending on extension."""
    path.parent.mkdir(parents=True, exist_ok=True)
    bsio.save_structure(str(path), atoms)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure", required=True)
    parser.add_argument("--selected-site", required=True)
    parser.add_argument("--rfd3-json", required=True)
    parser.add_argument("--outdir", required=True)

    args = parser.parse_args()

    structure_path = Path(args.structure).resolve()
    selected_path = Path(args.selected_site).resolve()
    rfd3_json_path = Path(args.rfd3_json).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    atoms = load_structure(structure_path)

    with selected_path.open() as handle:
        selected_site = json.load(handle)

    rfd3_payload = get_rfd3_design_payload(rfd3_json_path)
    contig = rfd3_payload["contig"]
    segments = parse_contig_segments(contig)

    if not segments:
        raise ValueError(f"No fixed target segments found in contig: {contig}")

    context_mask = np.zeros(len(atoms), dtype=bool)

    for chain, start, end in segments:
        context_mask |= residue_mask(atoms, chain, start, end)

    pb_mask = selected_pb_mask(atoms, selected_site)
    triad_mask = triad_residue_mask(atoms, selected_site)

    context_with_pb = atoms[context_mask | pb_mask]
    motif_only = atoms[triad_mask | pb_mask]

    context_pdb = outdir / "rfd3_context_fixed_segments_plus_pb.pdb"
    motif_pdb = outdir / "selected_pb_cys3_motif_only.pdb"

    save_structure(context_pdb, context_with_pb)
    save_structure(motif_pdb, motif_only)

    print("Exported context structures")
    print("---------------------------")
    print(f"Contig: {contig}")
    print(f"Fixed segments: {segments}")
    print(f"Selected Pb: {selected_site['metal_label']}")
    print(f"Triad: {[d['residue'] for d in selected_site['triad_donors']]}")
    print(f"Wrote: {context_pdb}")
    print(f"Wrote: {motif_pdb}")


if __name__ == "__main__":
    main()