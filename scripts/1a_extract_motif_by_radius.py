#!/usr/bin/env python3
"""
Find candidate motif residues around a residue, ligand, or XYZ point
using several distance radii.

Dependencies:
    pip install biotite numpy

Examples:
    python scripts/1a_best_motif_radius.py \
        --pdb inputs/target.pdb \
        --center-res A:170 \
        --radii 3 4 5 6 8 \
        --chain A \
        --out work/exp_01/motif_scan.json

    python scripts/1a_best_motif_radius.py \
        --pdb inputs/target.pdb \
        --center-lig PB \
        --radii 3 4 5 6 \
        --chain A \
        --out work/exp_01/motif_scan.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import biotite.structure as struc
import biotite.structure.io as bsio


COORDINATING_RESIDUES = {"CYS", "HIS", "ASP", "GLU", "MET"}


def load_structure(path: Path):
    """Load the first model if the file contains multiple models."""
    atoms = bsio.load_structure(str(path))

    if isinstance(atoms, struc.AtomArrayStack):
        atoms = atoms[0]

    return atoms


def parse_center_res(spec: str) -> tuple[str, int]:
    """Parse center residue like A:170."""
    chain, res_id = spec.split(":")
    return chain, int(res_id)


def get_center_coords(atoms, center_res: str | None, center_lig: str | None, center_xyz: str | None):
    """Return coordinates defining the center."""
    n_given = sum(x is not None for x in [center_res, center_lig, center_xyz])
    if n_given != 1:
        raise ValueError("Use exactly one of --center-res, --center-lig, or --center-xyz")

    if center_res is not None:
        chain, res_id = parse_center_res(center_res)
        mask = (atoms.chain_id == chain) & (atoms.res_id == res_id)
        if not np.any(mask):
            raise ValueError(f"Center residue not found: {center_res}")
        return atoms.coord[mask], f"residue:{center_res}"

    if center_lig is not None:
        mask = atoms.res_name == center_lig
        if not np.any(mask):
            raise ValueError(f"Ligand/residue name not found: {center_lig}")
        return atoms.coord[mask], f"ligand:{center_lig}"

    xyz = np.array([float(x) for x in center_xyz.split(",")], dtype=float)
    if xyz.shape != (3,):
        raise ValueError("--center-xyz must look like x,y,z")
    return xyz.reshape(1, 3), f"xyz:{center_xyz}"


def residue_min_distance(residue, center_coords: np.ndarray) -> float:
    """Minimum atom-atom distance between residue and center."""
    diff = residue.coord[:, None, :] - center_coords[None, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    return float(np.min(distances))


def get_residue_label(residue) -> str:
    """Return residue label like A170."""
    return f"{residue.chain_id[0]}{int(residue.res_id[0])}"


def scan_radius(atoms, center_coords: np.ndarray, radius: float, chain: str | None):
    """Find residues with at least one atom within the radius."""
    if chain is not None:
        atoms = atoms[atoms.chain_id == chain]

    # Use only polymer/protein atoms as motif candidates.
    atoms = atoms[~atoms.hetero]

    hits = []

    for residue in struc.residue_iter(atoms):
        min_dist = residue_min_distance(residue, center_coords)

        if min_dist <= radius:
            label = get_residue_label(residue)
            resname = str(residue.res_name[0])

            hits.append(
                {
                    "residue": label,
                    "resname": resname,
                    "min_distance": round(min_dist, 3),
                    "is_coordination_like": resname in COORDINATING_RESIDUES,
                }
            )

    hits.sort(key=lambda x: x["min_distance"])
    return hits


def score_hits(hits: list[dict], min_residues: int, max_residues: int) -> float:
    """
    Simple heuristic:
    - Penalize motifs that are too small or too large.
    - Prefer compact motifs enriched in coordination-capable residues.
    """
    n = len(hits)

    if n < min_residues:
        return -1000 - (min_residues - n)

    if n > max_residues:
        return -100 - (n - max_residues)

    n_coord = sum(hit["is_coordination_like"] for hit in hits)
    mean_dist = np.mean([hit["min_distance"] for hit in hits])

    return 5 * n_coord - mean_dist - 0.2 * n


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pdb", required=True)
    parser.add_argument("--center-res", default=None, help="Example: A:170")
    parser.add_argument("--center-lig", default=None, help="Example: PB, ZN, CA")
    parser.add_argument("--center-xyz", default=None, help="Example: 1.0,2.0,3.0")
    parser.add_argument("--radii", nargs="+", type=float, required=True)
    parser.add_argument("--chain", default=None)
    parser.add_argument("--min-residues", type=int, default=3)
    parser.add_argument("--max-residues", type=int, default=10)
    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    pdb_path = Path(args.pdb).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    atoms = load_structure(pdb_path)
    center_coords, center_label = get_center_coords(
        atoms,
        center_res=args.center_res,
        center_lig=args.center_lig,
        center_xyz=args.center_xyz,
    )

    results = {}

    for radius in args.radii:
        hits = scan_radius(
            atoms=atoms,
            center_coords=center_coords,
            radius=radius,
            chain=args.chain,
        )

        score = score_hits(
            hits,
            min_residues=args.min_residues,
            max_residues=args.max_residues,
        )

        results[str(radius)] = {
            "radius": radius,
            "n_residues": len(hits),
            "score": round(score, 3),
            "residues": hits,
        }

    best_radius = max(results, key=lambda r: results[r]["score"])
    best = results[best_radius]

    output = {
        "pdb": str(pdb_path),
        "center": center_label,
        "chain": args.chain,
        "best_radius": float(best_radius),
        "best_motif": best,
        "all_radii": results,
    }

    with out_path.open("w") as handle:
        json.dump(output, handle, indent=2)

    print(f"Center: {center_label}")
    print(f"Best radius: {best_radius} Å")
    print(f"Residues: {best['n_residues']}")
    print()

    for hit in best["residues"]:
        print(
            f"{hit['residue']:>6} {hit['resname']:<3} "
            f"{hit['min_distance']:>5.2f} Å"
        )

    print()
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()