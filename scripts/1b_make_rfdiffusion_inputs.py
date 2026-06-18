#!/usr/bin/env python3
"""
Build minimal RFD3 input files from the selected Pb-Cys motif.

This creates:
1. rfd3_input_radius_X.json
2. rfd3_hotspots_radius_X.json
3. rfd3_contig_radius_X.txt

Assumption:
    This prepares an RFD3 binder-style input:
    generated binder chain + fixed target windows around the selected Pb-Cys motif.

It does NOT yet do single-chain motif transplantation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_binder_length(length_range: str) -> tuple[int, int]:
    """Parse a binder length range like 80-140."""
    left, right = length_range.split("-")
    return int(left), int(right)


def merge_segments(segments: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    """Merge overlapping sequence windows per chain."""
    by_chain: dict[str, list[tuple[int, int]]] = {}

    for chain, start, end in segments:
        by_chain.setdefault(chain, []).append((start, end))

    merged = []

    for chain, ranges in by_chain.items():
        ranges = sorted(ranges)
        current_start, current_end = ranges[0]

        for start, end in ranges[1:]:
            if start <= current_end + 1:
                current_end = max(current_end, end)
            else:
                merged.append((chain, current_start, current_end))
                current_start, current_end = start, end

        merged.append((chain, current_start, current_end))

    return sorted(merged, key=lambda x: (x[0], x[1]))


def build_target_segments(selected_site: dict, window: int) -> list[tuple[str, int, int]]:
    """Build fixed target windows around the selected Cys SG triad."""
    segments = []

    for donor in selected_site["triad_donors"]:
        chain = donor["chain"]
        res_id = int(donor["res_id"])
        start = max(1, res_id - window)
        end = res_id + window
        segments.append((chain, start, end))

    return merge_segments(segments)


def segment_length(segment: tuple[str, int, int]) -> int:
    """Return inclusive residue count for one segment."""
    _, start, end = segment
    return end - start + 1


def build_contig(binder_length: str, target_segments: list[tuple[str, int, int]]) -> str:
    """Build RFD3 contig string."""
    fixed_parts = [f"{chain}{start}-{end}" for chain, start, end in target_segments]
    return binder_length + ",/0," + ",/0,".join(fixed_parts)


def build_total_length(binder_length: str, target_segments: list[tuple[str, int, int]]) -> str:
    """Build total length range: binder length + fixed target length."""
    binder_min, binder_max = parse_binder_length(binder_length)
    fixed_len = sum(segment_length(seg) for seg in target_segments)
    return f"{binder_min + fixed_len}-{binder_max + fixed_len}"


def build_triad_hotspots(selected_site: dict) -> dict[str, str]:
    """Use only the selected Cys SG triad as RFD3 hotspots."""
    hotspots = {}

    for donor in selected_site["triad_donors"]:
        hotspots[donor["residue"]] = "SG"

    return hotspots


def get_radius_key(motifs: dict, radius: float) -> str:
    """Find the matching radius key in motifs_by_radius.json."""
    wanted = str(float(radius))

    if wanted in motifs:
        return wanted

    compact = str(int(radius))
    if compact in motifs:
        return compact

    available = ", ".join(motifs.keys())
    raise KeyError(f"Radius {radius} not found. Available radii: {available}")


def build_radius_hotspots(motifs: dict, radius: float) -> dict[str, str]:
    """Use all motif residues at a given radius as hotspot hints."""
    radius_key = get_radius_key(motifs, radius)
    residues = motifs[radius_key]["residues"]

    hotspots = {}

    for row in residues:
        hotspots[row["residue"]] = ",".join(row["atoms_within_radius"])

    return hotspots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure", required=True)
    parser.add_argument("--selected-site", required=True)
    parser.add_argument("--motifs", required=True)
    parser.add_argument("--radius", type=float, default=4.0)
    parser.add_argument("--binder-length", default="80-140")
    parser.add_argument("--target-window", type=int, default=6)
    parser.add_argument("--design-name", default="pb_motif_rfd3")
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--hotspot-mode",
        choices=["triad", "radius"],
        default="triad",
        help="triad = only Cys SG donors; radius = all residues in selected motif radius.",
    )

    args = parser.parse_args()

    structure_path = Path(args.structure).resolve()
    selected_path = Path(args.selected_site).resolve()
    motifs_path = Path(args.motifs).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    with selected_path.open() as handle:
        selected_site = json.load(handle)

    with motifs_path.open() as handle:
        motifs = json.load(handle)

    target_segments = build_target_segments(
        selected_site=selected_site,
        window=args.target_window,
    )

    contig = build_contig(
        binder_length=args.binder_length,
        target_segments=target_segments,
    )

    total_length = build_total_length(
        binder_length=args.binder_length,
        target_segments=target_segments,
    )

    if args.hotspot_mode == "triad":
        hotspots = build_triad_hotspots(selected_site)
    else:
        hotspots = build_radius_hotspots(motifs, args.radius)

    rfd3_payload = {
        args.design_name: {
            "dialect": 2,
            "input": str(structure_path),
            "contig": contig,
            "length": total_length,
            "redesign_motif_sidechains": False,
            "select_hotspots": hotspots,
            "infer_ori_strategy": "hotspots",
            "is_non_loopy": True,
        }
    }

    radius_label = str(args.radius).replace(".", "p")

    input_json = outdir / f"rfd3_input_radius_{radius_label}.json"
    hotspots_json = outdir / f"rfd3_hotspots_radius_{radius_label}.json"
    contig_txt = outdir / f"rfd3_contig_radius_{radius_label}.txt"

    with input_json.open("w") as handle:
        json.dump(rfd3_payload, handle, indent=2)

    with hotspots_json.open("w") as handle:
        json.dump(hotspots, handle, indent=2)

    with contig_txt.open("w") as handle:
        handle.write(contig + "\n")

    print("RFD3 input generated")
    print("--------------------")
    print(f"Selected site: {selected_site['site_id']}")
    print(f"Geometry: {selected_site['geometry_call']}")
    print(f"Target segments: {target_segments}")
    print(f"Contig: {contig}")
    print(f"Total length: {total_length}")
    print(f"Hotspot mode: {args.hotspot_mode}")
    print(f"Wrote: {input_json}")
    print(f"Wrote: {hotspots_json}")
    print(f"Wrote: {contig_txt}")


if __name__ == "__main__":
    main()