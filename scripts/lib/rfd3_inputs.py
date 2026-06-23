"""RFD3 JSON and contig construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def parse_length_range(length_range: str) -> tuple[int, int]:
    """Parse a length range such as 80-140."""
    left, right = length_range.split("-")
    return int(left), int(right)


def add_ranges(*ranges: str) -> str:
    """Add multiple length ranges and return a combined min-max range."""
    total_min = 0
    total_max = 0
    for length_range in ranges:
        left, right = parse_length_range(length_range)
        total_min += left
        total_max += right
    return f"{total_min}-{total_max}"


def segment_length(segment: tuple[str, int, int]) -> int:
    """Return inclusive residue count for one segment."""
    _, start, end = segment
    return end - start + 1


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


def build_target_segments(selected_site: dict[str, Any], window: int) -> list[tuple[str, int, int]]:
    """Build fixed target windows around the selected Cys SG triad."""
    segments = []
    for donor in selected_site["triad_donors"]:
        chain = donor["chain"]
        res_id = int(donor["res_id"])
        start = max(1, res_id - window)
        end = res_id + window
        segments.append((chain, start, end))
    return merge_segments(segments)


def format_segment(segment: tuple[str, int, int]) -> str:
    """Format one fixed target segment for an RFD3 contig."""
    chain, start, end = segment
    return f"{chain}{start}-{end}"


def build_binder_contig(binder_length: str, target_segments: list[tuple[str, int, int]]) -> str:
    """Build a binder-style contig with chain breaks."""
    fixed_parts = [format_segment(segment) for segment in target_segments]
    return binder_length + ",/0," + ",/0,".join(fixed_parts)


def build_binder_total_length(binder_length: str, target_segments: list[tuple[str, int, int]]) -> str:
    """Build total length range for binder-style design."""
    binder_min, binder_max = parse_length_range(binder_length)
    fixed_len = sum(segment_length(segment) for segment in target_segments)
    return f"{binder_min + fixed_len}-{binder_max + fixed_len}"


def build_monomeric_contig(
    nterm_length: str,
    linker_length: str,
    cterm_length: str,
    target_segments: list[tuple[str, int, int]],
) -> str:
    """Build a single-chain motif-transplantation contig."""
    if len(target_segments) != 2:
        raise ValueError(
            "Monomeric motif transplantation expects exactly two fixed segments. "
            f"Got {len(target_segments)} segments: {target_segments}"
        )

    first, second = target_segments
    return ",".join(
        [
            nterm_length,
            format_segment(first),
            linker_length,
            format_segment(second),
            cterm_length,
        ]
    )


def build_monomeric_total_length(
    nterm_length: str,
    linker_length: str,
    cterm_length: str,
    target_segments: list[tuple[str, int, int]],
) -> str:
    """Build total length range for single-chain motif transplantation."""
    fixed_len = sum(segment_length(segment) for segment in target_segments)
    designed_range = add_ranges(nterm_length, linker_length, cterm_length)
    designed_min, designed_max = parse_length_range(designed_range)
    return f"{designed_min + fixed_len}-{designed_max + fixed_len}"


def build_select_fixed_atoms(selected_site: dict[str, Any], mode: str = "triad_backbone_sidechain") -> dict[str, str]:
    """Build fixed-atom constraints for the Pb-coordinating Cys triad."""
    fixed_atoms = {}
    for donor in selected_site["triad_donors"]:
        residue = donor["residue"]
        if mode == "sg_only":
            fixed_atoms[residue] = "SG"
        elif mode == "triad_backbone_sidechain":
            fixed_atoms[residue] = "N,CA,C,O,CB,SG"
        else:
            raise ValueError(f"Unknown fixed-atom mode: {mode}")
    return fixed_atoms


def get_ori_token(selected_site: dict[str, Any], ori_key: str) -> list[float]:
    """Read one ORI token from selected_pb_site.json."""
    sweep = selected_site.get("ori_token_sweep", {})
    for side_name in ["scaffold_side", "open_side"]:
        if ori_key in sweep.get(side_name, {}):
            return sweep[side_name][ori_key]

    available = []
    for tokens in sweep.values():
        available.extend(tokens.keys())
    raise KeyError(f"ORI key '{ori_key}' was not found. Available keys: {', '.join(available)}")


def build_rfd3_payload(
    design_name: str,
    structure_path: Path,
    selected_site: dict[str, Any],
    target_segments: list[tuple[str, int, int]],
    design_mode: str,
    ori_key: str,
    fixed_atom_mode: str,
    binder_length: str,
    nterm_length: str,
    linker_length: str,
    cterm_length: str,
    is_non_loopy: bool,
) -> dict[str, Any]:
    """Build the RFD3 JSON payload."""
    if design_mode == "monomeric":
        contig = build_monomeric_contig(
            nterm_length=nterm_length,
            linker_length=linker_length,
            cterm_length=cterm_length,
            target_segments=target_segments,
        )
        total_length = build_monomeric_total_length(
            nterm_length=nterm_length,
            linker_length=linker_length,
            cterm_length=cterm_length,
            target_segments=target_segments,
        )
    elif design_mode == "binder":
        contig = build_binder_contig(binder_length=binder_length, target_segments=target_segments)
        total_length = build_binder_total_length(binder_length=binder_length, target_segments=target_segments)
    else:
        raise ValueError(f"Unknown design mode: {design_mode}")

    return {
        design_name: {
            "dialect": 2,
            "input": str(structure_path),
            "contig": contig,
            "length": total_length,
            "redesign_motif_sidechains": False,
            "select_fixed_atoms": build_select_fixed_atoms(selected_site, mode=fixed_atom_mode),
            "ori_token": get_ori_token(selected_site, ori_key=ori_key),
            "is_non_loopy": is_non_loopy,
        }
    }


def write_rfd3_inputs(
    outdir: Path,
    payloads: dict[str, dict[str, Any]],
    target_segments: list[tuple[str, int, int]],
) -> None:
    """Write RFD3 JSON files and a target segment report."""
    outdir.mkdir(parents=True, exist_ok=True)

    for label, payload in payloads.items():
        json_path = outdir / f"rfd3_input_{label}.json"
        with json_path.open("w") as handle:
            json.dump(payload, handle, indent=2)

        design_name = next(iter(payload))
        contig = payload[design_name]["contig"]
        with (outdir / f"rfd3_contig_{label}.txt").open("w") as handle:
            handle.write(contig + "\n")

    with (outdir / "target_segments.json").open("w") as handle:
        json.dump(
            [
                {"chain": chain, "start": start, "end": end, "length": end - start + 1}
                for chain, start, end in target_segments
            ],
            handle,
            indent=2,
        )
