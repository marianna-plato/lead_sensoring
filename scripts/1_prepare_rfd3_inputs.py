#!/usr/bin/env python3
"""
Prepare Pb-Cys3 motif scaffolding inputs for RFdiffusion3.

This script runs one clean pipeline:
1. Scan Pb sites and select the best Pb-Cys3 motif.
2. Extract radius-based motif diagnostics.
3. Compute an open-space vector and ORI-token sweep.
4. Build RFD3 JSON inputs using explicit select_fixed_atoms and ori_token.
5. Export reduced context PDBs for inspection.

The default design mode is monomeric motif transplantation, not binder-style design.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_DIR = SCRIPT_DIR / "lib"
sys.path.insert(0, str(LIB_DIR))

from constants import (
    DEFAULT_MAX_PB_S,
    DEFAULT_MIN_PB_S,
    DEFAULT_SIGMA_PB_S,
    DEFAULT_STRONG_EXTRA_DONOR_CUTOFF,
    DEFAULT_TARGET_PB_S,
    DEFAULT_WEAK_EXTRA_DONOR_CUTOFF,
)
from export import export_context_structures
from motif_scan import (
    analyze_one_pb_site,
    build_motifs_by_radius,
    find_cys_sg_indices,
    find_metal_indices,
    select_eligible_site,
    write_site_outputs,
)
from open_space import add_open_space_and_ori_tokens
from visualization import write_pb_vector_bild
from rfd3_inputs import build_rfd3_payload, build_target_segments, write_rfd3_inputs
from structure_io import (
    find_observed_numbering_gaps,
    load_structure,
    parse_mmcif_missing_residues,
)


def parse_float_list(values: list[str]) -> list[float]:
    """Parse a list of numeric command-line values."""
    return [float(value) for value in values]


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

    parser.add_argument("--target-window", type=int, default=6)
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
    parser.add_argument("--ori-keys", nargs="+", default=["scaffold_side_5A", "scaffold_side_8A", "scaffold_side_12A"])
    parser.add_argument("--ori-distances", nargs="+", type=float, default=[5.0, 8.0, 12.0, 15.0])
    parser.add_argument("--open-space-directions", type=int, default=512)
    parser.add_argument("--open-space-max-distance", type=float, default=14.0)
    parser.add_argument("--open-space-cone-angle", type=float, default=45.0)
    parser.add_argument("--chemical-weight", type=float, default=0.0)
    parser.add_argument("--open-space-top-k", type=int, default=10)
    parser.add_argument("--vector-bild-length", type=float, default=12.0)
    parser.add_argument("--loopy", action="store_true", help="Disable is_non_loopy in the RFD3 payload.")

    return parser


def scan_and_select_site(atoms, structure_path: Path, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Scan all Pb sites and return all sites, selected site, and motifs by radius."""
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
        with (diagnostic_outdir / "all_pb_sites.json").open("w") as handle:
            json.dump(
                {
                    "missing_residues_from_mmcif": missing_residues,
                    "observed_numbering_gaps": observed_gaps,
                    "sites": all_sites,
                },
                handle,
                indent=2,
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


def main() -> None:
    """Run the complete preparation pipeline."""
    parser = build_arg_parser()
    args = parser.parse_args()

    structure_path = Path(args.structure).resolve()
    outdir = Path(args.outdir).resolve()
    motif_outdir = outdir / "motif_scan"
    rfd3_outdir = outdir / "rfdiffusion_inputs"
    context_outdir = outdir / "contexts"

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

    payloads = {}
    first_payload = None
    for ori_key in args.ori_keys:
        label = ori_key.replace("_", "-")
        payload = build_rfd3_payload(
            design_name=f"{args.design_name}_{label}",
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
        payloads[label] = payload
        if first_payload is None:
            first_payload = payload

    write_rfd3_inputs(outdir=rfd3_outdir, payloads=payloads, target_segments=target_segments)

    if first_payload is not None:
        first_design_name = next(iter(first_payload))
        exported = export_context_structures(
            atoms=atoms,
            selected_site=selected_site,
            rfd3_payload=first_payload[first_design_name],
            outdir=context_outdir,
        )
    else:
        exported = {}

    print("RFD3 preparation complete")
    print("-------------------------")
    print(f"Selected site: {selected_site['site_id']} | {selected_site['geometry_call']}")
    print(f"Triad: {[donor['residue'] for donor in selected_site['triad_donors']]}")
    print(f"Target segments: {target_segments}")
    print(f"Design mode: {args.design_mode}")
    print(f"ORI keys: {args.ori_keys}")
    print(f"Motif outputs: {motif_outdir}")
    print(f"Vector BILD: {vector_bild}")
    print(f"RFD3 inputs: {rfd3_outdir}")
    print(f"Context outputs: {context_outdir}")
    for name, path in exported.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
