"""Simple BILD exports for ChimeraX visual diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from geometry import unit_vector


def _arrow_line(start: np.ndarray, direction: np.ndarray, length: float) -> str:
    """Return a ChimeraX BILD arrow command."""
    end = start + unit_vector(direction) * length
    return (
        f".arrow {start[0]:.3f} {start[1]:.3f} {start[2]:.3f} "
        f"{end[0]:.3f} {end[1]:.3f} {end[2]:.3f} "
        "0.25 0.75 0.75\n"
    )


def write_pb_vector_bild(
    selected_site: dict[str, Any],
    path: Path,
    length: float = 12.0,
) -> None:
    """Write a BILD file showing ligand void, open-space, and scaffold-side vectors."""
    path.parent.mkdir(parents=True, exist_ok=True)

    pb = np.array(selected_site["metal_coord"], dtype=float)
    ligand_void = np.array(selected_site["approx_empty_side_direction"], dtype=float)
    open_space = np.array(selected_site["open_space_summary"]["open_space_direction"], dtype=float)
    scaffold_side = -open_space

    with path.open("w") as handle:
        handle.write("# Blue: ligand-void direction from Pb-S3 geometry only\n")
        handle.write(".color blue\n")
        handle.write(_arrow_line(pb, ligand_void, length))

        handle.write("# Red: selected steric open-space direction from full protein context\n")
        handle.write(".color red\n")
        handle.write(_arrow_line(pb, open_space, length))

        handle.write("# Green: scaffold-side direction used for scaffold_side_* ORI tokens\n")
        handle.write(".color green\n")
        handle.write(_arrow_line(pb, scaffold_side, length))
