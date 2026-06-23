"""Open-space vector and ORI-token calculations for metal-site scaffolding.

Important distinction:
    - ligand_void_direction: geometry-only Pb(II) hemidirected void, derived from Pb-S vectors.
    - open_space_direction: steric low-density direction, derived from the full protein context.

For ORI tokens, the default scaffold-side tokens are placed opposite to the desired
open/exposed direction, because the ORI token approximates the designed scaffold COM.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from geometry import fibonacci_sphere, round_vector, unit_vector
from structure_io import normalize


def heavy_protein_atom_mask(atoms) -> np.ndarray:
    """Return a mask for non-hydrogen protein atoms."""
    atom_names = normalize(atoms.atom_name)
    elements = normalize(atoms.element)
    return (~atoms.hetero) & (elements != "H") & (~np.char.startswith(atom_names, "H"))


def score_direction_clearance(
    atoms,
    pb_coord: np.ndarray,
    direction: np.ndarray,
    max_distance: float,
    cone_angle_deg: float,
    min_distance: float = 1.0,
) -> dict[str, float]:
    """Score steric clearance in a cone starting at the Pb atom.

    Higher clearance_score is better. The score is intentionally simple and
    interpretable: it rewards long nearest-atom distance and penalizes atom
    density inside the cone.
    """
    direction = unit_vector(direction)
    mask = heavy_protein_atom_mask(atoms)
    coords = atoms.coord[mask]

    vectors = coords - pb_coord[None, :]
    distances = np.linalg.norm(vectors, axis=1)
    valid = (distances > min_distance) & (distances <= max_distance)

    if not np.any(valid):
        return {
            "n_atoms_in_cone": 0,
            "density_penalty": 0.0,
            "closest_atom_distance": float(max_distance),
            "clearance_score": float(max_distance),
        }

    vectors = vectors[valid]
    distances = distances[valid]
    projections = vectors @ direction
    forward = projections > 0.0

    if not np.any(forward):
        return {
            "n_atoms_in_cone": 0,
            "density_penalty": 0.0,
            "closest_atom_distance": float(max_distance),
            "clearance_score": float(max_distance),
        }

    vectors = vectors[forward]
    distances = distances[forward]
    projections = projections[forward]

    cos_angles = projections / distances
    cos_cutoff = np.cos(np.radians(cone_angle_deg))
    in_cone = cos_angles >= cos_cutoff

    if not np.any(in_cone):
        return {
            "n_atoms_in_cone": 0,
            "density_penalty": 0.0,
            "closest_atom_distance": float(max_distance),
            "clearance_score": float(max_distance),
        }

    cone_distances = distances[in_cone]
    density_penalty = float(np.sum(1.0 / np.maximum(cone_distances, 1.0) ** 2))
    closest_atom_distance = float(np.min(cone_distances))
    n_atoms_in_cone = int(len(cone_distances))

    clearance_score = (
        closest_atom_distance
        - 0.35 * n_atoms_in_cone
        - 4.0 * density_penalty
    )

    return {
        "n_atoms_in_cone": n_atoms_in_cone,
        "density_penalty": density_penalty,
        "closest_atom_distance": closest_atom_distance,
        "clearance_score": float(clearance_score),
    }


def rank_open_space_directions(
    atoms,
    pb_coord: np.ndarray,
    ligand_void_direction: np.ndarray,
    n_directions: int = 1024,
    max_distance: float = 14.0,
    cone_angle_deg: float = 45.0,
    chemical_weight: float = 0.0,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Rank candidate outward directions by local steric clearance.

    The ligand-void direction is only a weak optional prior. By default,
    chemical_weight=0.0, so the selected direction is decided by protein-context
    clearance, not by the hemidirected Pb-S3 geometry.
    """
    ligand_void_direction = unit_vector(ligand_void_direction)
    candidates = fibonacci_sphere(n_directions)
    rows: list[dict[str, Any]] = []

    for direction in candidates:
        direction = unit_vector(direction)
        clearance = score_direction_clearance(
            atoms=atoms,
            pb_coord=pb_coord,
            direction=direction,
            max_distance=max_distance,
            cone_angle_deg=cone_angle_deg,
        )
        alignment = float(np.dot(direction, ligand_void_direction))
        final_score = clearance["clearance_score"] + chemical_weight * alignment

        rows.append(
            {
                "direction": direction,
                "final_score": final_score,
                "alignment_to_ligand_void": alignment,
                **clearance,
            }
        )

    rows.sort(key=lambda row: row["final_score"], reverse=True)
    return rows[:top_k]


def summarize_direction(row: dict[str, Any]) -> dict[str, Any]:
    """Convert one ranked direction row into JSON-friendly data."""
    return {
        "direction": round_vector(row["direction"], ndigits=6),
        "final_score": round(float(row["final_score"]), 6),
        "clearance_score": round(float(row["clearance_score"]), 6),
        "alignment_to_ligand_void": round(float(row["alignment_to_ligand_void"]), 6),
        "n_atoms_in_cone": int(row["n_atoms_in_cone"]),
        "density_penalty": round(float(row["density_penalty"]), 6),
        "closest_atom_distance": round(float(row["closest_atom_distance"]), 3),
    }


def compute_open_space_summary(
    atoms,
    pb_coord: np.ndarray,
    ligand_void_direction: np.ndarray,
    n_directions: int = 1024,
    max_distance: float = 14.0,
    cone_angle_deg: float = 45.0,
    chemical_weight: float = 0.0,
    top_k: int = 10,
) -> dict[str, Any]:
    """Compute the steric outward direction and diagnostics."""
    ligand_void_direction = unit_vector(ligand_void_direction)

    ranked = rank_open_space_directions(
        atoms=atoms,
        pb_coord=pb_coord,
        ligand_void_direction=ligand_void_direction,
        n_directions=n_directions,
        max_distance=max_distance,
        cone_angle_deg=cone_angle_deg,
        chemical_weight=chemical_weight,
        top_k=top_k,
    )
    if not ranked:
        raise RuntimeError("No open-space directions were ranked")

    ligand_void_clearance = score_direction_clearance(
        atoms=atoms,
        pb_coord=pb_coord,
        direction=ligand_void_direction,
        max_distance=max_distance,
        cone_angle_deg=cone_angle_deg,
    )
    opposite_ligand_void_clearance = score_direction_clearance(
        atoms=atoms,
        pb_coord=pb_coord,
        direction=-ligand_void_direction,
        max_distance=max_distance,
        cone_angle_deg=cone_angle_deg,
    )

    best = ranked[0]

    return {
        "open_space_direction": round_vector(best["direction"], ndigits=6),
        "open_space_score": round(float(best["final_score"]), 6),
        "ligand_void_direction": round_vector(ligand_void_direction, ndigits=6),
        "alignment_open_to_ligand_void": round(float(best["alignment_to_ligand_void"]), 6),
        "ligand_void_clearance": summarize_direction({
            "direction": ligand_void_direction,
            "final_score": ligand_void_clearance["clearance_score"],
            "alignment_to_ligand_void": 1.0,
            **ligand_void_clearance,
        }),
        "opposite_ligand_void_clearance": summarize_direction({
            "direction": -ligand_void_direction,
            "final_score": opposite_ligand_void_clearance["clearance_score"],
            "alignment_to_ligand_void": -1.0,
            **opposite_ligand_void_clearance,
        }),
        "ranked_open_space_directions": [summarize_direction(row) for row in ranked],
        "cone_angle_deg": cone_angle_deg,
        "max_distance": max_distance,
        "n_directions": n_directions,
        "chemical_weight": chemical_weight,
    }


def build_ori_token_sweep(
    pb_coord: np.ndarray,
    open_space_direction: np.ndarray,
    distances: list[float],
) -> dict[str, Any]:
    """Build ORI-token candidates on both sides of the Pb site.

    open_side places the ORI token toward the desired exposed/open direction.
    scaffold_side places the ORI token opposite to that open direction and is
    usually the safer starting point for keeping the metal site accessible.
    """
    open_space_direction = unit_vector(open_space_direction)
    scaffold_side = {}
    open_side = {}

    for distance in distances:
        scaffold_token = pb_coord - open_space_direction * distance
        open_token = pb_coord + open_space_direction * distance
        scaffold_side[f"scaffold_side_{distance:g}A"] = round_vector(scaffold_token, ndigits=3)
        open_side[f"open_side_{distance:g}A"] = round_vector(open_token, ndigits=3)

    return {
        "scaffold_side": scaffold_side,
        "open_side": open_side,
    }


def add_open_space_and_ori_tokens(
    atoms,
    selected_site: dict[str, Any],
    ori_distances: list[float],
    n_directions: int,
    max_distance: float,
    cone_angle_deg: float,
    chemical_weight: float,
    top_k: int,
) -> dict[str, Any]:
    """Add steric open-space direction and ORI-token sweep to a selected site."""
    pb_coord = np.array(selected_site["metal_coord"], dtype=float)
    ligand_void_direction = np.array(selected_site["approx_empty_side_direction"], dtype=float)

    if ligand_void_direction.dtype == object or np.any(np.isnan(ligand_void_direction.astype(float))):
        raise ValueError("The selected site does not contain a valid ligand-void direction")

    open_space_summary = compute_open_space_summary(
        atoms=atoms,
        pb_coord=pb_coord,
        ligand_void_direction=ligand_void_direction,
        n_directions=n_directions,
        max_distance=max_distance,
        cone_angle_deg=cone_angle_deg,
        chemical_weight=chemical_weight,
        top_k=top_k,
    )
    open_space_direction = np.array(open_space_summary["open_space_direction"], dtype=float)

    enriched = dict(selected_site)
    enriched["direction_definitions"] = {
        "approx_empty_side_direction": "Ligand-void direction from Pb-S3 hemidirected geometry only. Diagnostic, not necessarily solvent-facing.",
        "open_space_direction": "Steric low-density direction from the full input structure. Use this for visual checks and ORI-token sweep.",
        "scaffold_side_ori_tokens": "ORI tokens placed opposite to open_space_direction, intended to keep the open side accessible.",
    }
    enriched["open_space_summary"] = open_space_summary
    enriched["ori_token_sweep"] = build_ori_token_sweep(
        pb_coord=pb_coord,
        open_space_direction=open_space_direction,
        distances=ori_distances,
    )
    return enriched
