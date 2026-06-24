#!/usr/bin/env python3
"""
Prepare oriented double-motif RFdiffusion3 inputs for the Pb-Cys3 motif.

This script runs a complete preparation pipeline:
1. Scan Pb sites and select the best Pb-Cys3 motif.
2. Compute open-space ORI tokens at user-defined motif-to-ORI distances.
3. Build a single-motif RFdiffusion3 payload for each ORI token.
4. Create a second rigid motif copy by rotating the original motif 180 degrees
   around the Pb-S3 ligand-void direction (blue axis) passing through the new
   double-motif ORI token.
5. The double-motif ORI token is placed exactly between the original and copied
   motif anchors.
6. The copied motif preserves the blue direction, while red/green are transformed
   as motif-relative vectors.
7. Write double-motif PDB files and RFdiffusion3 JSON inputs.

Geometry for each ORI distance d:
    center        = Pb position, or SG/fixed-atom centroid fallback
    blue          = normalize(selected_site["approx_empty_side_direction"])
    green_single  = normalize(single_scaffold_ori_token - center)
    green         = normalize(project_perpendicular(green_single, blue))
    double_ori    = center + d * green
    rotation      = R(blue, 180 degrees) around double_ori
    copied_center = double_ori + R(blue, 180) @ (center - double_ori)
    copied_coords = double_ori + R(blue, 180) @ (original_coords - double_ori)

This makes the double ORI the midpoint between motif anchors and keeps the blue
ligand-void direction unchanged in the copied motif.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_DIR = SCRIPT_DIR / "lib"
sys.path.insert(0, str(LIB_DIR))

from constants import (  # noqa: E402
    DEFAULT_MAX_PB_S,
    DEFAULT_MIN_PB_S,
    DEFAULT_SIGMA_PB_S,
    DEFAULT_STRONG_EXTRA_DONOR_CUTOFF,
    DEFAULT_TARGET_PB_S,
    DEFAULT_WEAK_EXTRA_DONOR_CUTOFF,
)
from motif_scan import (  # noqa: E402
    analyze_one_pb_site,
    build_motifs_by_radius,
    find_cys_sg_indices,
    find_metal_indices,
    select_eligible_site,
    write_site_outputs,
)
from open_space import add_open_space_and_ori_tokens  # noqa: E402
from rfd3_inputs import build_rfd3_payload, build_target_segments  # noqa: E402
from structure_io import (  # noqa: E402
    find_observed_numbering_gaps,
    load_structure,
    parse_mmcif_missing_residues,
)
from visualization import write_pb_vector_bild  # noqa: E402


SEGMENT_RE = re.compile(r"^([A-Za-z])(-?\d+)-(-?\d+)$")
LABEL_RE = re.compile(r"^([A-Za-z])(-?\d+)([A-Za-z]?)$")
RANGE_RE = re.compile(r"^(\d+)-(\d+)$")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--structure", required=True, help="Input PDB/mmCIF structure.")
    parser.add_argument("--metal", default="PB", help="Metal atom/residue/element label.")
    parser.add_argument("--radii", nargs="+", type=float, default=[3.0, 4.0, 5.0, 6.0])
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--design-name", default="5gpe_pb_motif")

    parser.add_argument("--donor-cutoff", type=float, default=3.4)
    parser.add_argument("--gap-window", type=int, default=5)
    parser.add_argument("--target-pb-s", type=float, default=DEFAULT_TARGET_PB_S)
    parser.add_argument("--sigma-pb-s", type=float, default=DEFAULT_SIGMA_PB_S)
    parser.add_argument("--min-pb-s", type=float, default=DEFAULT_MIN_PB_S)
    parser.add_argument("--max-pb-s", type=float, default=DEFAULT_MAX_PB_S)
    parser.add_argument("--strong-extra-donor-cutoff", type=float, default=DEFAULT_STRONG_EXTRA_DONOR_CUTOFF)
    parser.add_argument("--weak-extra-donor-cutoff", type=float, default=DEFAULT_WEAK_EXTRA_DONOR_CUTOFF)

    parser.add_argument("--allow-missing-near", action="store_true")
    parser.add_argument("--allow-distorted-pb-s", action="store_true")
    parser.add_argument("--allow-strong-extra-donors", action="store_true")

    parser.add_argument("--target-window", type=int, default=4)
    parser.add_argument("--design-mode", choices=["monomeric", "binder"], default="monomeric")
    parser.add_argument("--binder-length", default="80-140")
    parser.add_argument("--nterm-length", default="10-50")
    parser.add_argument("--linker-length", default="5-20")
    parser.add_argument("--cterm-length", default="10-50")
    parser.add_argument(
        "--fixed-atom-mode",
        choices=["triad_backbone_sidechain", "sg_only"],
        default="triad_backbone_sidechain",
    )

    parser.add_argument(
        "--ori-distances",
        nargs="+",
        type=float,
        default=[5.0, 10.0, 15.0],
        help="Motif-to-ORI distances to sweep, in Angstrom.",
    )
    parser.add_argument("--open-space-directions", type=int, default=1024)
    parser.add_argument("--open-space-max-distance", type=float, default=14.0)
    parser.add_argument("--open-space-cone-angle", type=float, default=45.0)
    parser.add_argument("--chemical-weight", type=float, default=0.0)
    parser.add_argument("--open-space-top-k", type=int, default=10)
    parser.add_argument("--vector-bild-length", type=float, default=12.0)

    parser.add_argument(
        "--orbit-angle-deg",
        type=float,
        default=180.0,
        help=(
            "Rotation angle (degrees) applied to the copied motif's atomic coordinates "
            "around the outward (open-space) axis. 180 gives C2 symmetry so both motifs "
            "face outward without introducing chirality."
        ),
    )
    parser.add_argument(
        "--copy-distance-scale",
        type=float,
        default=1.0,
        help="Lateral separation = ORI distance * this scale.",
    )
    parser.add_argument(
        "--rotation-axis-mode",
        choices=["triad_normal_projected", "auto_perpendicular", "x", "y", "z"],
        default="triad_normal_projected",
        help=(
            "Axis used to compute the lateral displacement direction for the copied motif. "
            "This axis is perpendicular to outward. The atomic rotation is always around "
            "the outward vector itself."
        ),
    )
    parser.add_argument(
        "--copy-chain-map",
        nargs="*",
        default=None,
        help="Chain mapping for copied motif, e.g. C:X D:Y. If omitted, X/Y/Z are used.",
    )
    parser.add_argument("--inter-motif-linker", default="5-20")
    parser.add_argument("--metal-search-cutoff", type=float, default=5.0)
    parser.add_argument("--include-metal", action="store_true")
    parser.add_argument("--min-safe-distance", type=float, default=2.0)
    parser.add_argument("--loopy", action="store_true", help="Disable is_non_loopy in the RFD3 payload.")

    return parser


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON with readable indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(data, handle, indent=2)


def write_pdb(path: Path, atoms) -> None:
    """Write an AtomArray to PDB."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pdb_file = PDBFile()
    pdb_file.set_structure(atoms)
    pdb_file.write(path)


def _bild_arrow_line(start: np.ndarray, direction: np.ndarray, length: float) -> str:
    """Return one ChimeraX BILD arrow command."""
    end = start + normalize(direction, "BILD arrow direction") * length
    return (
        f".arrow {start[0]:.3f} {start[1]:.3f} {start[2]:.3f} "
        f"{end[0]:.3f} {end[1]:.3f} {end[2]:.3f} "
        "0.25 0.75 0.75\n"
    )


def write_double_vector_bild(metadata: dict[str, Any], path: Path, length: float = 12.0) -> None:
    """Write BILD vectors for the original and copied double-motif geometry."""
    path.parent.mkdir(parents=True, exist_ok=True)

    original_center = np.asarray(metadata["anchor_center"], dtype=float)
    copied_center = np.asarray(metadata["copied_center"], dtype=float)
    double_ori = np.asarray(metadata["double_ori_token"], dtype=float)

    blue = np.asarray(metadata["blue_direction"], dtype=float)
    copied_blue = np.asarray(metadata["copied_blue_direction"], dtype=float)
    red = np.asarray(metadata["outward"], dtype=float)
    copied_red = np.asarray(metadata["copied_outward"], dtype=float)
    green = np.asarray(metadata["original_green"], dtype=float)
    copied_green = np.asarray(metadata["copied_green"], dtype=float)

    with path.open("w") as handle:
        handle.write("# Magenta sphere: double ORI token, midpoint between motif anchors\n")
        handle.write(".color magenta\n")
        handle.write(f".sphere {double_ori[0]:.3f} {double_ori[1]:.3f} {double_ori[2]:.3f} 0.70\n")

        handle.write("# Blue: ligand-void direction; copied blue must match original blue\n")
        handle.write(".color blue\n")
        handle.write(_bild_arrow_line(original_center, blue, length))
        handle.write(_bild_arrow_line(copied_center, copied_blue, length))

        handle.write("# Red: motif-relative outward direction\n")
        handle.write(".color red\n")
        handle.write(_bild_arrow_line(original_center, red, length))
        handle.write(_bild_arrow_line(copied_center, copied_red, length))

        handle.write("# Green: motif-relative scaffold-side direction, pointing toward the double ORI\n")
        handle.write(".color green\n")
        handle.write(_bild_arrow_line(original_center, green, length))
        handle.write(_bild_arrow_line(copied_center, copied_green, length))


def normalize(vector: np.ndarray, name: str = "vector") -> np.ndarray:
    """Return a normalized vector."""
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError(f"Cannot normalize near-zero {name}")
    return vector / norm


def rotation_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Build a Rodrigues rotation matrix."""
    axis = normalize(axis, "rotation axis")
    angle = np.deg2rad(angle_deg)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    one_minus_c = 1.0 - c

    return np.array(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=float,
    )


def parse_residue_label(label: str) -> tuple[str, int, str]:
    """Parse residue labels such as C78 or D113."""
    match = LABEL_RE.match(label)
    if not match:
        raise ValueError(f"Invalid residue label: {label}")
    chain_id, res_id, ins_code = match.groups()
    return chain_id, int(res_id), ins_code


def parse_segment_token(token: str) -> tuple[str, int, int] | None:
    """Parse fixed contig tokens such as C72-84."""
    match = SEGMENT_RE.match(token)
    if not match:
        return None
    chain_id, start, end = match.groups()
    return chain_id, int(start), int(end)


def is_variable_token(token: str) -> bool:
    """Return True for tokens such as 10-50 or 5-20."""
    return RANGE_RE.match(token) is not None


def split_contig(contig: str) -> list[str]:
    """Split an RFdiffusion contig string."""
    return [token.strip() for token in contig.split(",") if token.strip() and token.strip() != "/0"]


def chain_mask(atoms, chain_id: str) -> np.ndarray:
    """Return a mask for one chain ID."""
    return np.array([str(value).strip() == chain_id for value in atoms.chain_id])


def atom_name_mask(atoms, atom_names: set[str]) -> np.ndarray:
    """Return a mask for atom names."""
    return np.array([str(value).strip() in atom_names for value in atoms.atom_name])


def residue_mask(atoms, chain_id: str, res_id: int) -> np.ndarray:
    """Return a mask for one residue."""
    return chain_mask(atoms, chain_id) & (atoms.res_id == res_id)


def segment_mask(atoms, segments: list[tuple[str, int, int]]) -> np.ndarray:
    """Return a mask for all fixed motif segments."""
    mask = np.zeros(atoms.array_length(), dtype=bool)

    for chain_id, start, end in segments:
        lo = min(start, end)
        hi = max(start, end)
        mask |= chain_mask(atoms, chain_id) & (atoms.res_id >= lo) & (atoms.res_id <= hi)

    return mask


def metal_mask(atoms, metal: str) -> np.ndarray:
    """Return a mask for metal atoms using element, atom name, or residue name."""
    metal_upper = metal.upper()
    masks = []

    if hasattr(atoms, "element"):
        masks.append(np.array([str(value).strip().upper() == metal_upper for value in atoms.element]))

    if hasattr(atoms, "atom_name"):
        masks.append(np.array([str(value).strip().upper() == metal_upper for value in atoms.atom_name]))

    if hasattr(atoms, "res_name"):
        masks.append(np.array([str(value).strip().upper() == metal_upper for value in atoms.res_name]))

    if not masks:
        return np.zeros(atoms.array_length(), dtype=bool)

    out = masks[0]
    for extra_mask in masks[1:]:
        out |= extra_mask

    return out


def find_nearest_metal(atoms, center: np.ndarray, metal: str, cutoff: float) -> int | None:
    """Find the nearest metal atom to a reference center."""
    mask = metal_mask(atoms, metal)
    indices = np.where(mask)[0]

    if len(indices) == 0:
        return None

    distances = np.linalg.norm(atoms.coord[indices] - center[None, :], axis=1)
    best_local = int(np.argmin(distances))

    if float(distances[best_local]) > cutoff:
        return None

    return int(indices[best_local])


def get_fixed_atom_coords(atoms, select_fixed_atoms: dict[str, str]) -> np.ndarray:
    """Collect coordinates from fixed atoms, preferring SG atoms."""
    coords = []

    for residue_label, atom_csv in select_fixed_atoms.items():
        chain_id, res_id, _ = parse_residue_label(residue_label)
        atom_names = {name.strip() for name in atom_csv.split(",") if name.strip()}
        preferred_atoms = {"SG"} if "SG" in atom_names else atom_names
        mask = residue_mask(atoms, chain_id, res_id) & atom_name_mask(atoms, preferred_atoms)

        if not np.any(mask):
            raise ValueError(
                f"Could not find fixed atoms for {residue_label}. "
                f"Tried atoms: {sorted(preferred_atoms)}"
            )

        coords.append(atoms.coord[mask])

    return np.vstack(coords)


def get_sg_coords(atoms, select_fixed_atoms: dict[str, str]) -> np.ndarray:
    """Return SG coordinates from fixed residues."""
    coords = []

    for residue_label, atom_csv in select_fixed_atoms.items():
        atom_names = {name.strip() for name in atom_csv.split(",") if name.strip()}
        if "SG" not in atom_names:
            continue

        chain_id, res_id, _ = parse_residue_label(residue_label)
        mask = residue_mask(atoms, chain_id, res_id) & atom_name_mask(atoms, {"SG"})

        if np.any(mask):
            coords.append(atoms.coord[mask][0])

    if len(coords) < 3:
        raise ValueError("At least three SG atoms are required for the Pb-Cys3 motif.")

    return np.vstack(coords[:3])


def get_anchor_center(
    atoms,
    select_fixed_atoms: dict[str, str],
    metal: str,
    cutoff: float,
) -> tuple[np.ndarray, int | None, str]:
    """Return the motif center used as the rotation and translation anchor."""
    fixed_coords = get_fixed_atom_coords(atoms, select_fixed_atoms)
    fixed_centroid = fixed_coords.mean(axis=0)

    metal_index = find_nearest_metal(atoms, fixed_centroid, metal=metal, cutoff=cutoff)

    if metal_index is not None:
        return atoms.coord[metal_index].copy(), metal_index, "nearest_metal"

    return fixed_centroid, None, "fixed_atom_centroid"


def perpendicular_axis_from_triad(
    atoms,
    select_fixed_atoms: dict[str, str],
    outward: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Build a lateral displacement axis from the triad normal, projected perpendicular to outward."""
    sg = get_sg_coords(atoms, select_fixed_atoms)
    triad_normal = np.cross(sg[1] - sg[0], sg[2] - sg[0])
    triad_normal = normalize(triad_normal, "triad normal")

    axis = triad_normal - np.dot(triad_normal, outward) * outward

    if np.linalg.norm(axis) >= 1e-6:
        return normalize(axis, "projected triad normal"), "projected_triad_normal"

    z_axis = np.array([0.0, 0.0, 1.0])
    axis = np.cross(outward, z_axis)

    if np.linalg.norm(axis) < 1e-6:
        axis = np.cross(outward, np.array([1.0, 0.0, 0.0]))

    return normalize(axis, "fallback perpendicular axis"), "fallback_perpendicular_axis"


def get_lateral_axis(
    atoms,
    select_fixed_atoms: dict[str, str],
    outward: np.ndarray,
    axis_mode: str,
) -> tuple[np.ndarray, str]:
    """Return the lateral displacement axis for the copied motif.

    This axis is always perpendicular to outward. The atomic rotation of the
    copied motif is performed around outward itself (not this axis), so that
    both motifs end up facing the same open-space direction.
    """
    if axis_mode == "x":
        return np.array([1.0, 0.0, 0.0]), "x"

    if axis_mode == "y":
        return np.array([0.0, 1.0, 0.0]), "y"

    if axis_mode == "z":
        return np.array([0.0, 0.0, 1.0]), "z"

    if axis_mode == "auto_perpendicular":
        z_axis = np.array([0.0, 0.0, 1.0])
        axis = np.cross(outward, z_axis)
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(outward, np.array([1.0, 0.0, 0.0]))
        return normalize(axis, "auto perpendicular axis"), "auto_perpendicular"

    if axis_mode == "triad_normal_projected":
        return perpendicular_axis_from_triad(atoms, select_fixed_atoms, outward)

    raise ValueError(f"Unsupported rotation axis mode: {axis_mode}")


def parse_chain_map(values: list[str] | None, source_chains: list[str]) -> dict[str, str]:
    """Parse or create a chain map for the copied motif."""
    if values:
        chain_map = {}

        for value in values:
            if ":" not in value:
                raise ValueError(f"Invalid chain map item: {value}. Expected format C:X")

            source, target = value.split(":", 1)
            source = source.strip()
            target = target.strip()

            if len(source) != 1 or len(target) != 1:
                raise ValueError("Only single-character chain IDs are supported.")

            chain_map[source] = target

        missing = [chain for chain in source_chains if chain not in chain_map]
        if missing:
            raise ValueError(f"Missing copy-chain-map entries for chains: {missing}")

        return chain_map

    default_targets = ["X", "Y", "Z", "W", "U", "V"]

    if len(source_chains) > len(default_targets):
        raise ValueError("Too many source chains for the default chain map.")

    return {source: default_targets[i] for i, source in enumerate(source_chains)}


def rotate_copy_atoms(
    atoms,
    anchor_center: np.ndarray,
    copied_center: np.ndarray,
    rotation: np.ndarray,
    chain_map: dict[str, str],
):
    """Create a rotated and translated copy of a motif atom array."""
    copied = atoms.copy()
    centered = copied.coord - anchor_center[None, :]
    copied.coord = copied_center[None, :] + centered @ rotation.T

    for source_chain, target_chain in chain_map.items():
        mask = chain_mask(copied, source_chain)
        copied.chain_id[mask] = target_chain

    return copied


def remap_segment_token(token: str, chain_map: dict[str, str]) -> str:
    """Remap fixed contig segment chains for the copied motif."""
    parsed = parse_segment_token(token)

    if parsed is None:
        return token

    chain_id, start, end = parsed
    return f"{chain_map[chain_id]}{start}-{end}"


def remap_residue_label(label: str, chain_map: dict[str, str]) -> str:
    """Remap a fixed residue label for the copied motif."""
    chain_id, res_id, ins_code = parse_residue_label(label)
    return f"{chain_map[chain_id]}{res_id}{ins_code}"


def build_double_contig(original_contig: str, chain_map: dict[str, str], inter_motif_linker: str) -> str:
    """Build a double-motif contig by duplicating the fixed motif core."""
    tokens = split_contig(original_contig)
    fixed_positions = [i for i, token in enumerate(tokens) if parse_segment_token(token) is not None]

    if not fixed_positions:
        raise ValueError(f"No fixed motif segments found in contig: {original_contig}")

    first_fixed = fixed_positions[0]
    last_fixed = fixed_positions[-1]

    prefix = tokens[:first_fixed]
    original_core = tokens[first_fixed : last_fixed + 1]
    suffix = tokens[last_fixed + 1 :]

    copied_core = [
        remap_segment_token(token, chain_map) if parse_segment_token(token) is not None else token
        for token in original_core
    ]

    return ",".join(prefix + original_core + [inter_motif_linker] + copied_core + suffix)


def token_length_range(token: str) -> tuple[int, int]:
    """Compute the min/max length contribution of one contig token."""
    parsed_segment = parse_segment_token(token)

    if parsed_segment is not None:
        _, start, end = parsed_segment
        length = abs(end - start) + 1
        return length, length

    if is_variable_token(token):
        lo, hi = token.split("-")
        return int(lo), int(hi)

    raise ValueError(f"Cannot compute length for contig token: {token}")


def compute_contig_length(contig: str) -> str:
    """Compute RFdiffusion length range from a contig string."""
    min_total = 0
    max_total = 0

    for token in split_contig(contig):
        lo, hi = token_length_range(token)
        min_total += lo
        max_total += hi

    return f"{min_total}-{max_total}"


def build_double_fixed_atoms(select_fixed_atoms: dict[str, str], chain_map: dict[str, str]) -> dict[str, str]:
    """Duplicate select_fixed_atoms entries for the copied motif."""
    out = dict(select_fixed_atoms)

    for residue_label, atom_csv in select_fixed_atoms.items():
        copied_label = remap_residue_label(residue_label, chain_map)
        out[copied_label] = atom_csv

    return out


def min_pairwise_distance(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """Compute the minimum distance between two coordinate arrays."""
    if len(coords_a) == 0 or len(coords_b) == 0:
        return float("nan")

    min_distance = float("inf")

    for start in range(0, len(coords_a), 512):
        chunk = coords_a[start : start + 512]
        distances = np.linalg.norm(chunk[:, None, :] - coords_b[None, :, :], axis=2)
        min_distance = min(min_distance, float(np.min(distances)))

    return min_distance


def format_distance_label(distance: float) -> str:
    """Format distance labels for design names and ORI keys."""
    if abs(distance - round(distance)) < 1e-6:
        return str(int(round(distance)))
    return str(distance).replace(".", "p")


def ori_key_from_distance(distance: float) -> str:
    """Build the ORI key used by add_open_space_and_ori_tokens()."""
    return f"scaffold_side_{format_distance_label(distance)}A"


def scan_and_select_site(
    atoms,
    structure_path: Path,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Scan all Pb sites and return all sites, selected site, motifs by radius, and metadata."""
    missing_residues = parse_mmcif_missing_residues(structure_path)
    observed_gaps = find_observed_numbering_gaps(atoms)

    metal_indices = find_metal_indices(atoms, args.metal)
    if not metal_indices:
        raise ValueError(f"No metal atoms found for metal={args.metal}")

    cys_sg_indices = find_cys_sg_indices(atoms)
    if not cys_sg_indices:
        raise ValueError("No Cys-like SG atoms found in structure")

    all_sites = []
    for metal_number, metal_index in enumerate(metal_indices, start=1):
        all_sites.append(
            analyze_one_pb_site(
                atoms=atoms,
                metal_index=metal_index,
                metal_number=metal_number,
                cys_sg_indices=cys_sg_indices,
                donor_cutoff=args.donor_cutoff,
                target_pb_s=args.target_pb_s,
                sigma_pb_s=args.sigma_pb_s,
                min_pb_s=args.min_pb_s,
                max_pb_s=args.max_pb_s,
                strong_extra_donor_cutoff=args.strong_extra_donor_cutoff,
                weak_extra_donor_cutoff=args.weak_extra_donor_cutoff,
                missing_residues=missing_residues,
                observed_gaps=observed_gaps,
                gap_window=args.gap_window,
            )
        )

    try:
        selected_site = select_eligible_site(
            all_sites=all_sites,
            allow_missing_near=args.allow_missing_near,
            allow_distorted_pb_s=args.allow_distorted_pb_s,
            allow_strong_extra_donors=args.allow_strong_extra_donors,
        )
    except RuntimeError:
        diagnostic_outdir = Path(args.outdir).resolve() / "diagnostics_no_eligible_site"
        diagnostic_outdir.mkdir(parents=True, exist_ok=True)
        write_json(
            diagnostic_outdir / "all_pb_sites.json",
            {
                "missing_residues_from_mmcif": missing_residues,
                "observed_numbering_gaps": observed_gaps,
                "sites": all_sites,
            },
        )
        raise

    selected_site = add_open_space_and_ori_tokens(
        atoms=atoms,
        selected_site=selected_site,
        ori_distances=args.ori_distances,
        n_directions=args.open_space_directions,
        max_distance=args.open_space_max_distance,
        cone_angle_deg=args.open_space_cone_angle,
        chemical_weight=args.chemical_weight,
        top_k=args.open_space_top_k,
    )
    motifs_by_radius = build_motifs_by_radius(atoms=atoms, selected_site=selected_site, radii=args.radii)

    metadata = {
        "missing_residues_from_mmcif": missing_residues,
        "observed_numbering_gaps": observed_gaps,
    }
    return all_sites, selected_site, motifs_by_radius, metadata


def build_double_payload_from_single(
    atoms,
    single_payload: dict[str, Any],
    single_design_name: str,
    selected_site: dict[str, Any],
    structure_outdir: Path,
    distance: float,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Create one oriented double-motif PDB and its RFD3 payload.

    The copied motif is generated by a 180-degree rotation around the blue
    Pb-S3 ligand-void axis passing through the new double-motif ORI token.
    The double ORI token is placed exactly between the original and copied
    motif anchors.
    """
    contig = single_payload["contig"]
    select_fixed_atoms = single_payload["select_fixed_atoms"]
    ori_token = np.asarray(single_payload["ori_token"], dtype=float)

    segments = [
        parsed for parsed in (parse_segment_token(token) for token in split_contig(contig))
        if parsed is not None
    ]

    source_chains = []
    for chain_id, _, _ in segments:
        if chain_id not in source_chains:
            source_chains.append(chain_id)

    chain_map = parse_chain_map(args.copy_chain_map, source_chains)
    motif_mask = segment_mask(atoms, segments)

    if not np.any(motif_mask):
        raise ValueError(
            "No atoms matched the motif contig segments. "
            "Check that the payload contig and structure chain IDs agree."
        )

    original_motif_atoms = atoms[motif_mask].copy()

    anchor_center, metal_index, anchor_source = get_anchor_center(
        atoms=atoms,
        select_fixed_atoms=select_fixed_atoms,
        metal=args.metal,
        cutoff=args.metal_search_cutoff,
    )

    # Blue is the Pb-S3 ligand-void direction from the motif geometry.
    # This is the axis that must remain unchanged after creating the copy.
    blue_direction = normalize(
        np.asarray(selected_site["approx_empty_side_direction"], dtype=float),
        "blue ligand-void direction",
    )

    # The single-motif ORI token is a scaffold-side token. Use it only to infer
    # the green/scaffold-side direction, not as the final double-motif ORI.
    single_scaffold_direction = normalize(ori_token - anchor_center, "single scaffold-side direction")

    # The motif separation must be tangential to the blue direction. Project the
    # scaffold-side direction onto the plane perpendicular to blue.
    lateral_axis = single_scaffold_direction - np.dot(single_scaffold_direction, blue_direction) * blue_direction
    lateral_axis_source = "projected_single_scaffold_direction"

    if np.linalg.norm(lateral_axis) < 1e-6:
        lateral_axis, lateral_axis_source = get_lateral_axis(
            atoms=atoms,
            select_fixed_atoms=select_fixed_atoms,
            outward=blue_direction,
            axis_mode=args.rotation_axis_mode,
        )
        lateral_axis = lateral_axis - np.dot(lateral_axis, blue_direction) * blue_direction
        lateral_axis_source = f"fallback_{lateral_axis_source}"

    # From the original motif, green points toward the double ORI/scaffold side.
    # Red points away from that side. The copied motif receives these vectors by
    # the same 180-degree rotation used for its coordinates.
    original_green = normalize(lateral_axis, "original green/scaffold-side direction")
    original_red = -original_green

    angle_mod = float(args.orbit_angle_deg) % 360.0
    if not np.isclose(angle_mod, 180.0, atol=1e-6):
        raise ValueError(
            "The requested midpoint geometry requires --orbit-angle-deg 180. "
            "Other angles do not keep the ORI token exactly between the two motifs."
        )

    # distance is original motif anchor -> double ORI distance. Therefore the
    # final motif-anchor separation is 2 * distance * copy_distance_scale.
    copy_radius = float(distance) * float(args.copy_distance_scale)

    # New double-motif ORI: exactly halfway between the original and copied motif anchors.
    double_ori_token = anchor_center + copy_radius * original_green

    # Rotate the copied motif 180 degrees around the blue axis passing through
    # the double ORI. A vector parallel to blue remains unchanged.
    rotation = rotation_matrix(blue_direction, args.orbit_angle_deg)
    copied_center = double_ori_token + (anchor_center - double_ori_token) @ rotation.T

    copied = original_motif_atoms.copy()
    copied.coord = double_ori_token[None, :] + (copied.coord - double_ori_token[None, :]) @ rotation.T
    for source_chain, target_chain in chain_map.items():
        mask = chain_mask(copied, source_chain)
        copied.chain_id[mask] = target_chain
    copied_motif_atoms = copied

    copied_blue = normalize(rotation @ blue_direction, "copied blue direction")
    copied_green = normalize(rotation @ original_green, "copied green/scaffold-side direction")
    copied_red = normalize(rotation @ original_red, "copied red/open-space direction")

    midpoint_error = float(np.linalg.norm(((anchor_center + copied_center) / 2.0) - double_ori_token))
    blue_direction_error = float(np.linalg.norm(copied_blue - blue_direction))

    arrays_to_write = [original_motif_atoms, copied_motif_atoms]

    if args.include_metal and metal_index is not None:
        original_metal = atoms[metal_index : metal_index + 1].copy()
        copied_metal = original_metal.copy()

        copied_metal.coord = double_ori_token[None, :] + (copied_metal.coord - double_ori_token[None, :]) @ rotation.T
        copied_metal.chain_id[:] = list(chain_map.values())[-1]
        copied_metal.res_id[:] = int(original_metal.res_id[0]) + 1000

        arrays_to_write = [original_motif_atoms, original_metal, copied_motif_atoms, copied_metal]

    double_atoms = struc.concatenate(arrays_to_write)

    distance_label = format_distance_label(distance)
    angle_label = format_distance_label(args.orbit_angle_deg)
    double_design_name = f"{single_design_name}_double_ori{distance_label}A_rot{angle_label}deg"
    double_pdb = structure_outdir / f"{double_design_name}.pdb"
    write_pdb(double_pdb, double_atoms)

    double_contig = build_double_contig(
        original_contig=contig,
        chain_map=chain_map,
        inter_motif_linker=args.inter_motif_linker,
    )
    double_fixed_atoms = build_double_fixed_atoms(
        select_fixed_atoms=select_fixed_atoms,
        chain_map=chain_map,
    )

    min_distance = min_pairwise_distance(original_motif_atoms.coord, copied_motif_atoms.coord)
    warning = None
    if np.isfinite(min_distance) and min_distance < args.min_safe_distance:
        warning = (
            f"Original and copied motifs are very close: {min_distance:.3f} A. "
            "This likely creates clashes."
        )

    double_payload = dict(single_payload)
    double_payload["input"] = str(double_pdb)
    double_payload["contig"] = double_contig
    double_payload["length"] = compute_contig_length(double_contig)
    double_payload["select_fixed_atoms"] = double_fixed_atoms
    double_payload["ori_token"] = [float(x) for x in double_ori_token]
    double_payload["redesign_motif_sidechains"] = False

    metadata = {
        "source_design_name": single_design_name,
        "double_design_name": double_design_name,
        "output_structure": str(double_pdb),
        "ori_distance_A": float(distance),
        "orbit_angle_deg": float(args.orbit_angle_deg),
        "copy_distance_scale": float(args.copy_distance_scale),
        "copy_radius_A": copy_radius,
        "inter_motif_anchor_distance_A": 2.0 * copy_radius,
        "anchor_source": anchor_source,
        "anchor_center": [float(x) for x in anchor_center],
        "single_scaffold_ori_token": [float(x) for x in ori_token],
        "ori_token": [float(x) for x in double_ori_token],
        "double_ori_token": [float(x) for x in double_ori_token],
        "blue_direction": [float(x) for x in blue_direction],
        "copied_blue_direction": [float(x) for x in copied_blue],
        "outward": [float(x) for x in original_red],
        "copied_outward": [float(x) for x in copied_red],
        "original_green": [float(x) for x in original_green],
        "copied_green": [float(x) for x in copied_green],
        "lateral_axis_mode": args.rotation_axis_mode,
        "lateral_axis_source": lateral_axis_source,
        "lateral_axis": [float(x) for x in original_green],
        "atomic_rotation_axis": "blue_direction_through_double_ori_token",
        "copied_center": [float(x) for x in copied_center],
        "midpoint_error_A": midpoint_error,
        "blue_direction_error_A": blue_direction_error,
        "chain_map": chain_map,
        "original_contig": contig,
        "double_contig": double_contig,
        "original_length": single_payload.get("length"),
        "double_length": double_payload["length"],
        "min_original_copy_atom_distance_A": min_distance,
        "warning": warning,
    }

    return double_design_name, double_payload, metadata


def main() -> None:
    """Run the complete double-motif preparation pipeline."""
    parser = build_arg_parser()
    args = parser.parse_args()

    structure_path = Path(args.structure).resolve()
    outdir = Path(args.outdir).resolve()
    motif_outdir = outdir / "motif_scan"
    rfd3_outdir = outdir / "rfdiffusion_inputs"
    double_structure_outdir = outdir / "double_motif_structures"

    atoms = load_structure(structure_path)

    all_sites, selected_site, motifs_by_radius, metadata = scan_and_select_site(
        atoms=atoms,
        structure_path=structure_path,
        args=args,
    )

    write_site_outputs(
        outdir=motif_outdir,
        all_sites=all_sites,
        selected_site=selected_site,
        motifs_by_radius=motifs_by_radius,
        missing_residues=metadata["missing_residues_from_mmcif"],
        observed_gaps=metadata["observed_numbering_gaps"],
    )

    vector_bild = motif_outdir / "pb_vectors.bild"
    write_pb_vector_bild(
        selected_site=selected_site,
        path=vector_bild,
        length=args.vector_bild_length,
    )

    target_segments = build_target_segments(selected_site=selected_site, window=args.target_window)

    double_payloads: dict[str, dict[str, Any]] = {}
    double_metadata: dict[str, dict[str, Any]] = {}

    for distance in args.ori_distances:
        ori_key = ori_key_from_distance(distance)
        distance_label = format_distance_label(distance)
        single_design_name = f"{args.design_name}_ori{distance_label}A_single"

        single_container = build_rfd3_payload(
            design_name=single_design_name,
            structure_path=structure_path,
            selected_site=selected_site,
            target_segments=target_segments,
            design_mode=args.design_mode,
            ori_key=ori_key,
            fixed_atom_mode=args.fixed_atom_mode,
            binder_length=args.binder_length,
            nterm_length=args.nterm_length,
            linker_length=args.linker_length,
            cterm_length=args.cterm_length,
            is_non_loopy=not args.loopy,
        )
        single_payload = single_container[single_design_name]

        double_design_name, double_payload, one_metadata = build_double_payload_from_single(
            atoms=atoms,
            single_payload=single_payload,
            single_design_name=single_design_name,
            selected_site=selected_site,
            structure_outdir=double_structure_outdir,
            distance=distance,
            args=args,
        )

        double_vector_bild = double_structure_outdir / f"{double_design_name}_vectors.bild"
        write_double_vector_bild(
            metadata=one_metadata,
            path=double_vector_bild,
            length=args.vector_bild_length,
        )
        one_metadata["double_vector_bild"] = str(double_vector_bild)

        double_payloads[double_design_name] = double_payload
        double_metadata[double_design_name] = one_metadata

    write_json(rfd3_outdir / "rfd3_double_inputs.json", double_payloads)
    write_json(rfd3_outdir / "target_segments.json", {"target_segments": target_segments})
    write_json(rfd3_outdir / "double_motif_geometry_metadata.json", double_metadata)

    per_design_outdir = rfd3_outdir / "per_design"
    per_design_outdir.mkdir(parents=True, exist_ok=True)
    for design_name, payload in double_payloads.items():
        write_json(per_design_outdir / f"{design_name}.json", {design_name: payload})

    print("Double-motif RFD3 preparation complete")
    print("--------------------------------------")
    print(f"Selected site: {selected_site['site_id']} | {selected_site['geometry_call']}")
    print(f"Triad: {[donor['residue'] for donor in selected_site['triad_donors']]}")
    print(f"Target segments: {target_segments}")
    print(f"Design mode: {args.design_mode}")
    print(f"ORI distances: {args.ori_distances}")
    print(f"Orbit angle: {args.orbit_angle_deg} deg (rotation around outward axis)")
    print(f"Lateral axis mode: {args.rotation_axis_mode}")
    print(f"Motif outputs: {motif_outdir}")
    print(f"Vector BILD: {vector_bild}")
    print(f"Double motif structures: {double_structure_outdir}")
    print(f"RFD3 double inputs: {rfd3_outdir / 'rfd3_double_inputs.json'}")
    print(f"Per-design inputs: {per_design_outdir}")

    for design_name, one_metadata in double_metadata.items():
        print("")
        print(design_name)
        print(f"  contig:           {one_metadata['double_contig']}")
        print(f"  length:           {one_metadata['double_length']}")
        print(f"  outward:          {one_metadata['outward']}")
        print(f"  copied_outward:   {one_metadata['copied_outward']}")
        print(f"  lateral_axis:     {one_metadata['lateral_axis']}")
        print(f"  copy_radius_A:    {one_metadata['copy_radius_A']:.3f}")
        print(f"  min_distance_A:   {one_metadata['min_original_copy_atom_distance_A']:.3f}")
        if one_metadata["warning"] is not None:
            print(f"  WARNING: {one_metadata['warning']}")


if __name__ == "__main__":
    main()
