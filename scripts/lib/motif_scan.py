"""Pb-centered Cys3 motif scanning and scoring."""

from __future__ import annotations

import csv
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from constants import (
    COORDINATION_RESIDUES,
    CYS_NAMES,
    SIDECHAIN_DONOR_ATOMS,
    WATER_NAMES,
)
from geometry import angle_degrees, gaussian_score, unit_vector
from structure_io import (
    atom_label,
    normalize,
    residue_label,
    residue_near_missing,
    residue_near_observed_gap,
)


def find_metal_indices(atoms, metal: str) -> list[int]:
    """Find metal atoms by element, atom name, or residue name."""
    metal = metal.upper()
    res_names = normalize(atoms.res_name)
    atom_names = normalize(atoms.atom_name)
    elements = normalize(atoms.element)
    mask = (res_names == metal) | (atom_names == metal) | (elements == metal)
    return list(np.where(mask)[0])


def find_cys_sg_indices(atoms) -> list[int]:
    """Find all Cys-like SG atoms."""
    res_names = normalize(atoms.res_name)
    atom_names = normalize(atoms.atom_name)
    mask = np.isin(res_names, list(CYS_NAMES)) & (atom_names == "SG")
    return list(np.where(mask)[0])


def is_sidechain_donor(resname: str, atom_name: str) -> bool:
    """Return True for conservative side-chain donor atoms."""
    return atom_name.upper() in SIDECHAIN_DONOR_ATOMS.get(resname.upper(), set())


def find_extra_sidechain_donors(
    atoms,
    pb_coord: np.ndarray,
    triad_atom_indices: set[int],
    strong_cutoff: float,
    weak_cutoff: float,
) -> dict[str, Any]:
    """Find side-chain donor atoms near Pb that are not part of the triad."""
    strong = []
    weak = []
    res_names = normalize(atoms.res_name)
    atom_names = normalize(atoms.atom_name)

    for i in range(len(atoms)):
        if i in triad_atom_indices or atoms.hetero[i]:
            continue

        resname = str(res_names[i])
        atom_name = str(atom_names[i])
        if not is_sidechain_donor(resname, atom_name):
            continue

        distance = float(np.linalg.norm(atoms.coord[i] - pb_coord))
        row = {
            "atom_label": atom_label(atoms, i),
            "residue": residue_label(atoms, i),
            "resname": str(atoms.res_name[i]),
            "atom": str(atoms.atom_name[i]),
            "distance_to_pb": round(distance, 3),
        }

        if distance <= strong_cutoff:
            strong.append(row)
        elif distance <= weak_cutoff:
            weak.append(row)

    strong.sort(key=lambda row: row["distance_to_pb"])
    weak.sort(key=lambda row: row["distance_to_pb"])

    return {
        "strong_extra_donors": strong,
        "weak_extra_donors": weak,
        "n_strong_extra_donors": len(strong),
        "n_weak_extra_donors": len(weak),
    }


def build_cys_sg_donors(
    atoms,
    metal_index: int,
    cys_sg_indices: list[int],
    donor_cutoff: float,
    missing_residues: list[dict[str, Any]],
    observed_gaps: dict[str, list[dict[str, int]]],
    gap_window: int,
) -> list[dict[str, Any]]:
    """Build donor records for all Cys SG atoms within a cutoff."""
    pb_coord = atoms.coord[metal_index]
    donors = []

    for sg_index in cys_sg_indices:
        sg_coord = atoms.coord[sg_index]
        distance = float(np.linalg.norm(sg_coord - pb_coord))
        if distance > donor_cutoff:
            continue

        chain = str(atoms.chain_id[sg_index])
        res_id = int(atoms.res_id[sg_index])
        vector = sg_coord - pb_coord

        official_missing_near = residue_near_missing(
            chain=chain,
            res_id=res_id,
            missing_residues=missing_residues,
            gap_window=gap_window,
        )
        observed_gaps_near = residue_near_observed_gap(
            chain=chain,
            res_id=res_id,
            observed_gaps=observed_gaps,
            gap_window=gap_window,
        )

        donors.append(
            {
                "atom_index": int(sg_index),
                "residue": residue_label(atoms, sg_index),
                "chain": chain,
                "res_id": res_id,
                "resname": str(atoms.res_name[sg_index]),
                "atom": "SG",
                "coord": [float(x) for x in sg_coord],
                "pb_s_distance": round(distance, 3),
                "vector_pb_to_s": [float(x) for x in vector],
                "official_missing_near": official_missing_near,
                "observed_gaps_near": observed_gaps_near,
                "near_missing_or_gap": bool(official_missing_near or observed_gaps_near),
            }
        )

    donors.sort(key=lambda donor: donor["pb_s_distance"])
    return donors


def compute_triad_angles(triad: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute pairwise S-Pb-S angles for a Cys3 triad."""
    angles = []
    for donor_1, donor_2 in combinations(triad, 2):
        v1 = np.array(donor_1["vector_pb_to_s"], dtype=float)
        v2 = np.array(donor_2["vector_pb_to_s"], dtype=float)
        angles.append(
            {
                "donor_1": donor_1["residue"],
                "donor_2": donor_2["residue"],
                "angle_s_pb_s": round(angle_degrees(v1, v2), 3),
            }
        )
    return angles


def compute_hemidirected_vectors(triad: list[dict[str, Any]]) -> dict[str, Any]:
    """Estimate ligand clustering around Pb using Pb-to-S unit vectors."""
    if len(triad) != 3:
        return {
            "ligand_centroid_norm": None,
            "ligand_direction_pb_to_s_cluster": [None, None, None],
            "approx_empty_side_direction": [None, None, None],
        }

    unit_vectors = np.array(
        [unit_vector(np.array(donor["vector_pb_to_s"], dtype=float)) for donor in triad]
    )
    centroid = unit_vectors.mean(axis=0)
    centroid_norm = float(np.linalg.norm(centroid))

    if centroid_norm > 0:
        ligand_direction = unit_vector(centroid)
        empty_side_direction = -ligand_direction
    else:
        ligand_direction = np.array([np.nan, np.nan, np.nan])
        empty_side_direction = np.array([np.nan, np.nan, np.nan])

    return {
        "ligand_centroid_norm": round(centroid_norm, 3),
        "ligand_direction_pb_to_s_cluster": [
            None if np.isnan(x) else round(float(x), 4) for x in ligand_direction
        ],
        "approx_empty_side_direction": [
            None if np.isnan(x) else round(float(x), 4) for x in empty_side_direction
        ],
    }


def score_pb_s_distances(
    distances: list[float],
    target_pb_s: float,
    sigma_pb_s: float,
    min_pb_s: float,
    max_pb_s: float,
) -> float:
    """Score Pb-S distances using PbrR691-like Pb-S values."""
    if len(distances) != 3:
        return 0.0
    if any(distance < min_pb_s or distance > max_pb_s for distance in distances):
        return 0.0

    per_distance = [gaussian_score(distance, target=target_pb_s, sigma=sigma_pb_s) for distance in distances]
    spread = max(distances) - min(distances)
    spread_score = gaussian_score(spread, target=0.20, sigma=0.25)
    return 0.80 * float(np.mean(per_distance)) + 0.20 * spread_score


def score_hemidirected_vectors(triad: list[dict[str, Any]]) -> float:
    """Score whether Pb-to-S vectors cluster on one side of the metal."""
    if len(triad) != 3:
        return 0.0
    unit_vectors = np.array(
        [unit_vector(np.array(donor["vector_pb_to_s"], dtype=float)) for donor in triad]
    )
    return float(np.linalg.norm(unit_vectors.mean(axis=0)))


def score_chain_topology(triad: list[dict[str, Any]]) -> float:
    """Score whether the Cys triad has a PbrR691-like interchain topology."""
    if len(triad) != 3:
        return 0.0

    chain_counts = Counter(donor["chain"] for donor in triad)
    counts = sorted(chain_counts.values(), reverse=True)
    if counts == [2, 1]:
        return 1.0
    if counts == [3]:
        return 0.4
    return 0.2


def score_extra_donor_exclusion(n_strong_extra_donors: int, n_weak_extra_donors: int) -> float:
    """Penalize extra donor atoms around Pb."""
    penalty = 0.35 * n_strong_extra_donors + 0.15 * n_weak_extra_donors
    return max(0.0, 1.0 - penalty)


def classify_geometry(
    n_triad: int,
    distances: list[float],
    ligand_centroid_norm: float | None,
    n_strong_extra_donors: int,
    min_pb_s: float,
    max_pb_s: float,
) -> str:
    """Classify local Pb geometry using conservative rules."""
    if n_triad != 3:
        return "not_valid_pb_s3"
    if any(distance < min_pb_s or distance > max_pb_s for distance in distances):
        return "distorted_pb_s3"
    if ligand_centroid_norm is None:
        return "ambiguous_pb_s3"
    if ligand_centroid_norm >= 0.45 and n_strong_extra_donors == 0:
        return "likely_hemidirected_pb_s3"
    if ligand_centroid_norm >= 0.35:
        return "ambiguous_hemidirected_pb_s3"
    return "ambiguous_pb_s3"


def score_candidate_triad(
    triad: list[dict[str, Any]],
    n_strong_extra_donors: int,
    n_weak_extra_donors: int,
    target_pb_s: float,
    sigma_pb_s: float,
    min_pb_s: float,
    max_pb_s: float,
    has_missing_or_gap: bool,
) -> dict[str, Any]:
    """Score a Pb-centered Cys3 triad."""
    if len(triad) != 3:
        return {
            "score": -1000.0,
            "distance_score": 0.0,
            "hemidirected_score": 0.0,
            "topology_score": 0.0,
            "donor_exclusion_score": 0.0,
        }

    distances = [float(donor["pb_s_distance"]) for donor in triad]
    distance_score = score_pb_s_distances(distances, target_pb_s, sigma_pb_s, min_pb_s, max_pb_s)
    hemidirected_score = score_hemidirected_vectors(triad)
    topology_score = score_chain_topology(triad)
    donor_exclusion_score = score_extra_donor_exclusion(n_strong_extra_donors, n_weak_extra_donors)

    total = (
        0.40 * distance_score
        + 0.30 * hemidirected_score
        + 0.15 * topology_score
        + 0.15 * donor_exclusion_score
    )
    score = 100.0 * total

    if has_missing_or_gap:
        score -= 1000.0
    if distance_score == 0.0:
        score -= 500.0

    return {
        "score": round(float(score), 3),
        "distance_score": round(float(distance_score), 3),
        "hemidirected_score": round(float(hemidirected_score), 3),
        "topology_score": round(float(topology_score), 3),
        "donor_exclusion_score": round(float(donor_exclusion_score), 3),
    }


def select_best_triad_for_pb(
    atoms,
    pb_coord: np.ndarray,
    donors: list[dict[str, Any]],
    target_pb_s: float,
    sigma_pb_s: float,
    min_pb_s: float,
    max_pb_s: float,
    strong_extra_donor_cutoff: float,
    weak_extra_donor_cutoff: float,
) -> dict[str, Any]:
    """Evaluate all Cys3 donor combinations and return the best triad."""
    if len(donors) < 3:
        return {
            "triad": donors[:3],
            "score_components": {
                "score": -1000.0,
                "distance_score": 0.0,
                "hemidirected_score": 0.0,
                "topology_score": 0.0,
                "donor_exclusion_score": 0.0,
            },
            "extra_sidechain_donors": {
                "strong_extra_donors": [],
                "weak_extra_donors": [],
                "n_strong_extra_donors": 0,
                "n_weak_extra_donors": 0,
            },
        }

    best: dict[str, Any] | None = None

    for triad_tuple in combinations(donors, 3):
        triad = list(triad_tuple)
        triad_atom_indices = {int(donor["atom_index"]) for donor in triad}
        extra_donors = find_extra_sidechain_donors(
            atoms=atoms,
            pb_coord=pb_coord,
            triad_atom_indices=triad_atom_indices,
            strong_cutoff=strong_extra_donor_cutoff,
            weak_cutoff=weak_extra_donor_cutoff,
        )
        has_missing_or_gap = any(donor["near_missing_or_gap"] for donor in triad)
        score_components = score_candidate_triad(
            triad=triad,
            n_strong_extra_donors=extra_donors["n_strong_extra_donors"],
            n_weak_extra_donors=extra_donors["n_weak_extra_donors"],
            target_pb_s=target_pb_s,
            sigma_pb_s=sigma_pb_s,
            min_pb_s=min_pb_s,
            max_pb_s=max_pb_s,
            has_missing_or_gap=has_missing_or_gap,
        )
        candidate = {
            "triad": triad,
            "score_components": score_components,
            "extra_sidechain_donors": extra_donors,
        }
        if best is None or candidate["score_components"]["score"] > best["score_components"]["score"]:
            best = candidate

    if best is None:
        raise RuntimeError("Unexpected failure while selecting a Cys3 triad")
    return best


def analyze_one_pb_site(
    atoms,
    metal_index: int,
    metal_number: int,
    cys_sg_indices: list[int],
    donor_cutoff: float,
    target_pb_s: float,
    sigma_pb_s: float,
    min_pb_s: float,
    max_pb_s: float,
    strong_extra_donor_cutoff: float,
    weak_extra_donor_cutoff: float,
    missing_residues: list[dict[str, Any]],
    observed_gaps: dict[str, list[dict[str, int]]],
    gap_window: int,
) -> dict[str, Any]:
    """Analyze one Pb site."""
    pb_coord = atoms.coord[metal_index]
    donors = build_cys_sg_donors(
        atoms=atoms,
        metal_index=metal_index,
        cys_sg_indices=cys_sg_indices,
        donor_cutoff=donor_cutoff,
        missing_residues=missing_residues,
        observed_gaps=observed_gaps,
        gap_window=gap_window,
    )
    best_triad_result = select_best_triad_for_pb(
        atoms=atoms,
        pb_coord=pb_coord,
        donors=donors,
        target_pb_s=target_pb_s,
        sigma_pb_s=sigma_pb_s,
        min_pb_s=min_pb_s,
        max_pb_s=max_pb_s,
        strong_extra_donor_cutoff=strong_extra_donor_cutoff,
        weak_extra_donor_cutoff=weak_extra_donor_cutoff,
    )

    triad = best_triad_result["triad"]
    score_components = best_triad_result["score_components"]
    extra_sidechain_donors = best_triad_result["extra_sidechain_donors"]
    angles = compute_triad_angles(triad)
    vector_summary = compute_hemidirected_vectors(triad)

    distances = [float(donor["pb_s_distance"]) for donor in triad]
    angle_values = [float(angle["angle_s_pb_s"]) for angle in angles]

    mean_pb_s = float(np.mean(distances)) if distances else np.nan
    max_pb_s_deviation = float(max(abs(distance - target_pb_s) for distance in distances)) if distances else np.nan
    pb_s_distance_spread = float(max(distances) - min(distances)) if distances else np.nan
    mean_angle = float(np.mean(angle_values)) if angle_values else np.nan
    angle_spread = float(max(angle_values) - min(angle_values)) if angle_values else np.nan

    chains_involved = sorted(set(donor["chain"] for donor in triad))
    has_missing_or_gap = any(donor["near_missing_or_gap"] for donor in triad)
    n_triad = len(triad)
    n_extra_cys_sg_donors = max(0, len(donors) - 3)

    geometry_call = classify_geometry(
        n_triad=n_triad,
        distances=distances,
        ligand_centroid_norm=vector_summary["ligand_centroid_norm"],
        n_strong_extra_donors=extra_sidechain_donors["n_strong_extra_donors"],
        min_pb_s=min_pb_s,
        max_pb_s=max_pb_s,
    )

    output_triad = []
    for donor in triad:
        clean = dict(donor)
        clean.pop("atom_index", None)
        output_triad.append(clean)

    output_all_donors = []
    for donor in donors:
        clean = dict(donor)
        clean.pop("atom_index", None)
        output_all_donors.append(clean)

    return {
        "site_id": f"PB_{metal_number}",
        "metal_atom_index": int(metal_index),
        "metal_label": atom_label(atoms, metal_index),
        "metal_chain": str(atoms.chain_id[metal_index]),
        "metal_res_id": int(atoms.res_id[metal_index]),
        "metal_coord": [float(x) for x in pb_coord],
        "donor_cutoff": donor_cutoff,
        "target_pb_s": target_pb_s,
        "sigma_pb_s": sigma_pb_s,
        "min_pb_s": min_pb_s,
        "max_pb_s": max_pb_s,
        "strong_extra_donor_cutoff": strong_extra_donor_cutoff,
        "weak_extra_donor_cutoff": weak_extra_donor_cutoff,
        "n_cys_sg_donors_within_cutoff": len(donors),
        "n_triad_donors_used": n_triad,
        "n_extra_cys_sg_donors": n_extra_cys_sg_donors,
        "n_strong_extra_sidechain_donors": extra_sidechain_donors["n_strong_extra_donors"],
        "n_weak_extra_sidechain_donors": extra_sidechain_donors["n_weak_extra_donors"],
        "chains_involved": chains_involved,
        "is_interface_site": len(chains_involved) >= 2,
        "triad_donors": output_triad,
        "all_cys_sg_donors_within_cutoff": output_all_donors,
        "extra_sidechain_donors": extra_sidechain_donors,
        "angles": angles,
        "mean_pb_s_distance": None if np.isnan(mean_pb_s) else round(mean_pb_s, 3),
        "max_pb_s_deviation_from_target": None if np.isnan(max_pb_s_deviation) else round(max_pb_s_deviation, 3),
        "pb_s_distance_spread": None if np.isnan(pb_s_distance_spread) else round(pb_s_distance_spread, 3),
        "mean_s_pb_s_angle": None if np.isnan(mean_angle) else round(mean_angle, 3),
        "angle_spread": None if np.isnan(angle_spread) else round(angle_spread, 3),
        "ligand_centroid_norm": vector_summary["ligand_centroid_norm"],
        "ligand_direction_pb_to_s_cluster": vector_summary["ligand_direction_pb_to_s_cluster"],
        "approx_empty_side_direction": vector_summary["approx_empty_side_direction"],
        "has_missing_or_gap_near_triad": has_missing_or_gap,
        "geometry_call": geometry_call,
        "score_components": score_components,
        "score": score_components["score"],
    }


def select_eligible_site(
    all_sites: list[dict[str, Any]],
    allow_missing_near: bool,
    allow_distorted_pb_s: bool,
    allow_strong_extra_donors: bool,
) -> dict[str, Any]:
    """Select the best eligible Pb site."""
    eligible_sites = list(all_sites)

    if not allow_missing_near:
        eligible_sites = [site for site in eligible_sites if not site["has_missing_or_gap_near_triad"]]

    eligible_sites = [site for site in eligible_sites if site["n_triad_donors_used"] == 3]

    if not allow_distorted_pb_s:
        eligible_sites = [site for site in eligible_sites if site["score_components"]["distance_score"] > 0.0]

    if not allow_strong_extra_donors:
        eligible_sites = [site for site in eligible_sites if site["n_strong_extra_sidechain_donors"] == 0]

    if not eligible_sites:
        raise RuntimeError("No eligible Pb site found after filtering")

    return max(eligible_sites, key=lambda site: site["score"])


def extract_radius_motif(atoms, center_coord: np.ndarray, radius: float) -> list[dict[str, Any]]:
    """Extract protein residues with any atom within a radius of Pb."""
    res_names = normalize(atoms.res_name)
    residue_hits: dict[tuple[str, int, str], dict[str, Any]] = {}

    for i in range(len(atoms)):
        if atoms.hetero[i] or res_names[i] in WATER_NAMES:
            continue

        distance = float(np.linalg.norm(atoms.coord[i] - center_coord))
        if distance > radius:
            continue

        key = (str(atoms.chain_id[i]), int(atoms.res_id[i]), str(atoms.res_name[i]))
        if key not in residue_hits:
            residue_hits[key] = {
                "residue": residue_label(atoms, i),
                "chain": str(atoms.chain_id[i]),
                "res_id": int(atoms.res_id[i]),
                "resname": str(atoms.res_name[i]),
                "min_distance_to_pb": round(distance, 3),
                "closest_atom": str(atoms.atom_name[i]),
                "atoms_within_radius": set(),
                "coordination_like_residue": str(atoms.res_name[i]).upper() in COORDINATION_RESIDUES,
            }

        if distance < residue_hits[key]["min_distance_to_pb"]:
            residue_hits[key]["min_distance_to_pb"] = round(distance, 3)
            residue_hits[key]["closest_atom"] = str(atoms.atom_name[i])

        residue_hits[key]["atoms_within_radius"].add(str(atoms.atom_name[i]))

    motif = []
    for row in residue_hits.values():
        row["atoms_within_radius"] = sorted(row["atoms_within_radius"])
        motif.append(row)

    motif.sort(key=lambda row: (row["min_distance_to_pb"], row["chain"], row["res_id"]))
    return motif


def build_motifs_by_radius(atoms, selected_site: dict[str, Any], radii: list[float]) -> dict[str, Any]:
    """Extract radius motifs around the selected Pb site."""
    selected_coord = np.array(selected_site["metal_coord"], dtype=float)
    motifs_by_radius = {}

    for radius in sorted(radii):
        motif = extract_radius_motif(atoms=atoms, center_coord=selected_coord, radius=radius)
        motifs_by_radius[str(radius)] = {
            "radius": radius,
            "n_residues": len(motif),
            "residues": motif,
        }

    return motifs_by_radius


def write_site_outputs(
    outdir: Path,
    all_sites: list[dict[str, Any]],
    selected_site: dict[str, Any],
    motifs_by_radius: dict[str, Any],
    missing_residues: list[dict[str, Any]],
    observed_gaps: dict[str, list[dict[str, int]]],
) -> None:
    """Write motif scan outputs."""
    outdir.mkdir(parents=True, exist_ok=True)

    with (outdir / "all_pb_sites.json").open("w") as handle:
        json.dump(
            {
                "missing_residues_from_mmcif": missing_residues,
                "observed_numbering_gaps": observed_gaps,
                "sites": all_sites,
            },
            handle,
            indent=2,
        )

    with (outdir / "selected_pb_site.json").open("w") as handle:
        json.dump(selected_site, handle, indent=2)

    with (outdir / "motifs_by_radius.json").open("w") as handle:
        json.dump(motifs_by_radius, handle, indent=2)

    write_summary_csv(outdir / "pb_site_summary.csv", all_sites)


def write_summary_csv(path: Path, all_sites: list[dict[str, Any]]) -> None:
    """Write a compact CSV summary for all Pb sites."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "site_id",
                "metal_label",
                "n_cys_sg_donors_within_cutoff",
                "n_triad_donors_used",
                "chains_involved",
                "is_interface_site",
                "has_missing_or_gap_near_triad",
                "geometry_call",
                "mean_pb_s_distance",
                "pb_s_distance_spread",
                "mean_s_pb_s_angle",
                "angle_spread",
                "ligand_centroid_norm",
                "n_strong_extra_sidechain_donors",
                "n_weak_extra_sidechain_donors",
                "distance_score",
                "hemidirected_score",
                "topology_score",
                "donor_exclusion_score",
                "score",
            ],
        )
        writer.writeheader()

        for site in all_sites:
            components = site["score_components"]
            writer.writerow(
                {
                    "site_id": site["site_id"],
                    "metal_label": site["metal_label"],
                    "n_cys_sg_donors_within_cutoff": site["n_cys_sg_donors_within_cutoff"],
                    "n_triad_donors_used": site["n_triad_donors_used"],
                    "chains_involved": ",".join(site["chains_involved"]),
                    "is_interface_site": site["is_interface_site"],
                    "has_missing_or_gap_near_triad": site["has_missing_or_gap_near_triad"],
                    "geometry_call": site["geometry_call"],
                    "mean_pb_s_distance": site["mean_pb_s_distance"],
                    "pb_s_distance_spread": site["pb_s_distance_spread"],
                    "mean_s_pb_s_angle": site["mean_s_pb_s_angle"],
                    "angle_spread": site["angle_spread"],
                    "ligand_centroid_norm": site["ligand_centroid_norm"],
                    "n_strong_extra_sidechain_donors": site["n_strong_extra_sidechain_donors"],
                    "n_weak_extra_sidechain_donors": site["n_weak_extra_sidechain_donors"],
                    "distance_score": components["distance_score"],
                    "hemidirected_score": components["hemidirected_score"],
                    "topology_score": components["topology_score"],
                    "donor_exclusion_score": components["donor_exclusion_score"],
                    "score": site["score"],
                }
            )
