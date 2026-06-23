"""Structure loading, saving, labels, and residue metadata helpers."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

import numpy as np
import biotite.structure as struc
import biotite.structure.io as bsio


def load_structure(path: Path):
    """Load the first model from a PDB/mmCIF file."""
    atoms = bsio.load_structure(str(path))
    if isinstance(atoms, struc.AtomArrayStack):
        atoms = atoms[0]
    return atoms


def save_structure(path: Path, atoms) -> None:
    """Save an AtomArray to PDB or CIF based on file extension."""
    path.parent.mkdir(parents=True, exist_ok=True)
    bsio.save_structure(str(path), atoms)


def normalize(values) -> np.ndarray:
    """Return an uppercase string array."""
    return np.char.upper(values.astype(str))


def atom_label(atoms, index: int) -> str:
    """Return a compact atom label."""
    return (
        f"{atoms.chain_id[index]}"
        f"{int(atoms.res_id[index])}"
        f":{atoms.res_name[index]}"
        f":{atoms.atom_name[index]}"
    )


def residue_label(atoms, index: int) -> str:
    """Return a compact residue label such as A113."""
    return f"{atoms.chain_id[index]}{int(atoms.res_id[index])}"


def parse_mmcif_missing_residues(path: Path) -> list[dict[str, Any]]:
    """Parse _pdbx_unobs_or_zero_occ_residues from an mmCIF file."""
    if path.suffix.lower() not in {".cif", ".mmcif"}:
        return []

    lines = path.read_text(errors="replace").splitlines()
    missing = []
    i = 0

    while i < len(lines):
        if lines[i].strip() != "loop_":
            i += 1
            continue

        i += 1
        tags = []
        while i < len(lines) and lines[i].strip().startswith("_"):
            tags.append(lines[i].strip())
            i += 1

        if not tags or not any(tag.startswith("_pdbx_unobs_or_zero_occ_residues.") for tag in tags):
            continue

        table_lines = []
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            if line == "loop_" or line.startswith("_") or line.startswith("#"):
                break
            table_lines.append(line)
            i += 1

        tag_to_idx = {tag: idx for idx, tag in enumerate(tags)}
        chain_tag = next(
            (
                tag
                for tag in [
                    "_pdbx_unobs_or_zero_occ_residues.auth_asym_id",
                    "_pdbx_unobs_or_zero_occ_residues.label_asym_id",
                ]
                if tag in tag_to_idx
            ),
            None,
        )
        seq_tag = next(
            (
                tag
                for tag in [
                    "_pdbx_unobs_or_zero_occ_residues.auth_seq_id",
                    "_pdbx_unobs_or_zero_occ_residues.label_seq_id",
                ]
                if tag in tag_to_idx
            ),
            None,
        )
        resname_tag = next(
            (
                tag
                for tag in [
                    "_pdbx_unobs_or_zero_occ_residues.auth_comp_id",
                    "_pdbx_unobs_or_zero_occ_residues.label_comp_id",
                ]
                if tag in tag_to_idx
            ),
            None,
        )

        if chain_tag is None or seq_tag is None:
            continue

        for row_line in table_lines:
            try:
                parts = shlex.split(row_line)
            except ValueError:
                parts = row_line.split()

            if len(parts) < len(tags):
                continue

            try:
                seq_id = int(parts[tag_to_idx[seq_tag]])
            except ValueError:
                continue

            missing.append(
                {
                    "chain": parts[tag_to_idx[chain_tag]],
                    "res_id": seq_id,
                    "resname": parts[tag_to_idx[resname_tag]] if resname_tag else "UNK",
                }
            )

    return missing


def find_observed_numbering_gaps(atoms) -> dict[str, list[dict[str, int]]]:
    """Detect jumps in observed residue numbering."""
    gaps = {}

    for chain in sorted(set(atoms.chain_id.astype(str))):
        chain_atoms = atoms[(atoms.chain_id == chain) & (~atoms.hetero)]
        if len(chain_atoms) == 0:
            continue

        res_ids = sorted(set(int(x) for x in chain_atoms.res_id))
        chain_gaps = []

        for left, right in zip(res_ids[:-1], res_ids[1:]):
            if right - left > 1:
                chain_gaps.append(
                    {
                        "left_observed": left,
                        "right_observed": right,
                        "gap_size": right - left - 1,
                    }
                )

        gaps[chain] = chain_gaps

    return gaps


def residue_near_missing(
    chain: str,
    res_id: int,
    missing_residues: list[dict[str, Any]],
    gap_window: int,
) -> list[dict[str, Any]]:
    """Find official missing residues close in sequence to one residue."""
    return [
        miss
        for miss in missing_residues
        if miss["chain"] == chain and abs(miss["res_id"] - res_id) <= gap_window
    ]


def residue_near_observed_gap(
    chain: str,
    res_id: int,
    observed_gaps: dict[str, list[dict[str, int]]],
    gap_window: int,
) -> list[dict[str, int]]:
    """Find observed numbering gaps close in sequence to one residue."""
    nearby = []

    for gap in observed_gaps.get(chain, []):
        left = gap["left_observed"]
        right = gap["right_observed"]
        if abs(res_id - left) <= gap_window or abs(res_id - right) <= gap_window:
            nearby.append(gap)

    return nearby
