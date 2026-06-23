"""Shared constants for Pb-Cys motif analysis."""

WATER_NAMES = {"HOH", "WAT", "DOD"}
COORDINATION_RESIDUES = {"CYS", "HIS", "ASP", "GLU", "MET", "ASN", "GLN", "SER", "THR", "TYR"}
CYS_NAMES = {"CYS", "CYX", "CYM", "CSD"}

SIDECHAIN_DONOR_ATOMS = {
    "CYS": {"SG"},
    "CYX": {"SG"},
    "CYM": {"SG"},
    "CSD": {"SG"},
    "MET": {"SD"},
    "HIS": {"ND1", "NE2"},
    "HID": {"ND1", "NE2"},
    "HIE": {"ND1", "NE2"},
    "HIP": {"ND1", "NE2"},
    "ASP": {"OD1", "OD2"},
    "GLU": {"OE1", "OE2"},
    "ASN": {"OD1", "ND2"},
    "GLN": {"OE1", "NE2"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "TYR": {"OH"},
}

DEFAULT_TARGET_PB_S = 2.67
DEFAULT_SIGMA_PB_S = 0.15
DEFAULT_MIN_PB_S = 2.30
DEFAULT_MAX_PB_S = 3.20
DEFAULT_STRONG_EXTRA_DONOR_CUTOFF = 3.20
DEFAULT_WEAK_EXTRA_DONOR_CUTOFF = 4.00
