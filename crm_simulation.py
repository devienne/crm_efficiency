"""
Markov Chain CRM Simulation for Octahedral Magnetite Particles
==============================================================
Based on: Chen et al. (2025), GRL, doi:10.1029/2025GL117964

State space (data-driven, determined by scanning MERRILL barrier files):
  Active state types per grain size are detected from the transition files
  present in the energy_barriers directory.  Typical ranges for T=20 data:
    SD  : 20–77 nm      (8 states along ±⟨111⟩)
    HAV : 56–98 nm      (6 states along ±⟨100⟩)
    EAV : 71–500 nm     (8 states along ±⟨111⟩)

Transition types and source files (new naming → old fallback):
  SD ↔ SD    : *_SD_1_1_1_to_SD_1_1_-1.txt          (representative)
  SD ↔ HAV   : *_SD_1_1_1_to_HAV_1_0_0.txt          → *_size_hyst_states.txt
  HAV ↔ HAV  : *_HAV_1_0_0_to_HAV_0_1_0.txt         → *_lower_branch_rotates.txt
  HAV ↔ EAV  : *_HAV_0_0_1_to_EAV_1_1_1.txt         → *_SV_1_0_0_to_SV_1_1_1.txt
  EAV ↔ EAV  : *_EAV_1_1_1_to_EAV_1_1_-1.txt        → *_SV_1_1_1_to_SV_1_1_-1.txt
                                                       → *_lower_branch_rotates.txt

Key equations (Chen et al. 2025):
  k_ij = (1/τ₀) exp(-ΔE_ij / kT)                   (Eq. 6)  — Néel-Arrhenius rate
  p(t+Δt) = exp(A Δt) · p(t)                        (Eq. 4)  — Markov update
  A_ij = k_{j→i} (i≠j),  A_ii = -Σ_{j≠i} k_{i→j}  (Eq. 5)  — generator matrix
  m_B = Σ_i m_i · p_i                               (Eq. 2)  — net moment

Magnetic moment convention:
  Each state's moment uses the MERRILL-derived fraction from HYST_FRACS:
    m_i = HYST_FRACS[size][type] × Ms × V × n̂_i
  SD states: fraction ≈ 1.0 (fully saturated single domain)
  HAV states: fraction < 1.0 (hard-axis vortex, reduced net moment)
  EAV states: fraction < 1.0 (easy-axis vortex, small net moment)
  crm_moment_projected() returns the raw magnetic moment in Am².
  Divide by Ms × V to obtain the dimensionless M_CRM/Ms.

Growth model:
  Constant rate from size_min to size_max over total growth time T_total.
  Time step at size i: Δt_i = (size_{i+1} - size_i) × T_total / (size_max - size_min)

Assemblage average:
  run_assemblage() averages over N_ORIENTATIONS Fibonacci-sphere directions.
  Alternatively notebooks use a single [111] orientation (exact for isotropic
  assemblages via the isotropy proof, §2.9 of crm_comprehensive_notebook.ipynb).
"""

import os
import re
import glob
import numpy as np
from scipy.linalg import expm
from typing import Dict, List, Optional, Tuple

# =============================================================================
# Physical constants
# =============================================================================

K_B  = 1.380649e-23   # Boltzmann constant (J/K)
T_K  = 293.15         # Temperature: 20 °C (K)
TAU0 = 1e-9           # Attempt time τ₀ (s) — Néel 1949; Chen et al. Eq. 1
MS   = 4.80768e5      # Magnetite saturation magnetisation at 20 °C (A/m)

REMANENCE_TIME_S = 100.0   # Zero-field relaxation duration (s) used to record
                            # "true remanence" (remanence=True in
                            # run_single_orientation, and the equivalent
                            # manual SIRM protocol in the notebooks) --
                            # typical laboratory measurement timescale.

N_ORIENTATIONS = 100   # Default Fibonacci-sphere directions for run_assemblage()

_MONO_BASE = r"d:\prebiotic_crm\octahedron\mono_phase"
DATA_DIR = os.path.join(_MONO_BASE, "energy_barriers", "T20")
HYST_DIR = os.path.join(_MONO_BASE, "hyst", "T20")

# =============================================================================
# Grain sizes  (auto-detected from DATA_DIR at import time)
# =============================================================================

def _available_sizes() -> List[int]:
    """Return grain sizes that have at least one .txt barrier file."""
    result = []
    for d in glob.glob(os.path.join(DATA_DIR, "x*_y*_z*")):
        if any(f.endswith('.txt') for f in os.listdir(d)):
            result.append(int(os.path.basename(d).split("_")[0][1:]))
    return sorted(result)

SIZES: List[int] = _available_sizes()


def _scan_state_types(data_dir: str) -> Dict[int, set]:
    """Determine active micromagnetic state types per grain size by scanning barrier files."""
    result: Dict[int, set] = {}
    for dname in os.listdir(data_dir):
        if not dname.startswith('x'):
            continue
        try:
            size = int(dname.split('_')[0][1:])
        except (ValueError, IndexError):
            continue
        folder = os.path.join(data_dir, dname)
        if not os.path.isdir(folder):
            continue

        types: set = set()
        has_lower_branch = False
        for fname in os.listdir(folder):
            if not fname.endswith('.txt'):
                continue
            for st in ('SD', 'HAV', 'EAV'):
                if f'_{st}_' in fname:
                    types.add(st)
            if '_SV_' in fname:
                types.update(('HAV', 'EAV'))
            if 'size_hyst_states' in fname:
                types.update(('SD', 'HAV'))
            if 'lower_branch_rotates' in fname:
                has_lower_branch = True

        if has_lower_branch and not types.intersection({'HAV', 'EAV'}):
            types.add('EAV')

        if types:
            result[size] = types
    return result


_STATE_TYPES: Dict[int, set] = _scan_state_types(DATA_DIR)


def grain_volume(size_nm: int) -> float:
    """
    Equivalent cubic volume for an octahedral grain of edge length size_nm (nm).
    V = a³  so that a 50 nm grain has the same volume as a 50 nm cube.
    """
    return (size_nm * 1e-9) ** 3


def equiv_spherical_vol_diameter(size_nm: float) -> float:
    """Diameter of the sphere with the same volume as a size_nm cube (nm)."""
    return 2.0 * size_nm * (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0)


# =============================================================================
# State definitions
# =============================================================================

def _n(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)

# 8 SD / EAV directions  ±⟨111⟩
_SD_EAV_DIRS: List[np.ndarray] = [
    _n([ 1,  1,  1]), _n([-1,  1,  1]), _n([ 1, -1,  1]), _n([ 1,  1, -1]),
    _n([-1, -1,  1]), _n([-1,  1, -1]), _n([ 1, -1, -1]), _n([-1, -1, -1]),
]

# 6 HAV directions  ±⟨100⟩
_HAV_DIRS: List[np.ndarray] = [
    _n([ 1, 0, 0]), _n([-1, 0, 0]),
    _n([ 0, 1, 0]), _n([ 0,-1, 0]),
    _n([ 0, 0, 1]), _n([ 0, 0,-1]),
]

# Each state is a (type_string, direction_index) tuple
State = Tuple[str, int]


def active_states(size_nm) -> List[State]:
    """Return the ordered list of active micromagnetic states for this grain size.

    State types are determined from the MERRILL barrier files scanned at import
    time (_STATE_TYPES).  For sizes not directly in the scan, the nearest
    scanned size is used.
    """
    size_int = int(round(size_nm))
    if size_int in _STATE_TYPES:
        types = _STATE_TYPES[size_int]
    else:
        sizes = sorted(_STATE_TYPES.keys())
        idx = min(range(len(sizes)), key=lambda i: abs(sizes[i] - size_int))
        types = _STATE_TYPES[sizes[idx]]

    states: List[State] = []
    if 'SD' in types:
        states += [("SD",  i) for i in range(8)]
    if 'HAV' in types:
        states += [("HAV", i) for i in range(6)]
    if 'EAV' in types:
        states += [("EAV", i) for i in range(8)]
    return states


def _direction(s: State) -> np.ndarray:
    """Unit vector for state s."""
    stype, idx = s
    return _HAV_DIRS[idx] if stype == "HAV" else _SD_EAV_DIRS[idx]


def _moment(s: State, V: float, size_nm: int) -> np.ndarray:
    """
    Net magnetic moment vector (A·m²).

    Uses the MERRILL-derived moment fraction from HYST_FRACS:
      m = HYST_FRACS[size_nm][type] × Ms × V × n̂

    For SD states the fraction is ~1.0 (fully saturated).
    For HAV/EAV vortex states it is the actual reduced net moment — critical
    for correct Zeeman energies and CRM projections in the vortex regime.
    """
    stype, _ = s
    return HYST_FRACS[size_nm][stype] * MS * V * _direction(s)


# =============================================================================
# Adjacency and transition type
# =============================================================================

def _sd_sd_adj(i: int, j: int) -> bool:
    """SD↔SD: differ in exactly one sign component."""
    si, sj = np.sign(_SD_EAV_DIRS[i]), np.sign(_SD_EAV_DIRS[j])
    return int(np.sum(si != sj)) == 1


def _sd_hav_adj(sd_i: int, hav_j: int) -> bool:
    """SD[a,b,c] ↔ HAV[sign(a),0,0] / HAV[0,sign(b),0] / HAV[0,0,sign(c)]."""
    sd_sgn = np.sign(_SD_EAV_DIRS[sd_i])
    hd     = _HAV_DIRS[hav_j]
    nz     = np.flatnonzero(np.abs(hd) > 0.5)
    return len(nz) == 1 and np.sign(hd[nz[0]]) == sd_sgn[nz[0]]


def _hav_hav_adj(i: int, j: int) -> bool:
    """HAV↔HAV: all pairs except identical or antiparallel (forbids 180° flip)."""
    di, dj = _HAV_DIRS[i], _HAV_DIRS[j]
    return not (np.allclose(di, dj) or np.allclose(di, -dj))


def _hav_eav_adj(hav_i: int, eav_j: int) -> bool:
    """HAV[±1,0,0] ↔ EAV states sharing the same sign in that axis component."""
    hd = _HAV_DIRS[hav_i]
    nz = np.flatnonzero(np.abs(hd) > 0.5)[0]
    return np.sign(hd[nz]) == np.sign(_SD_EAV_DIRS[eav_j][nz])


def _eav_eav_adj(i: int, j: int) -> bool:
    """EAV↔EAV: same topology as SD↔SD (differ in exactly one sign)."""
    return _sd_sd_adj(i, j)


def get_transition(si: State, sj: State) -> Optional[Tuple[str, bool]]:
    """
    Return (transition_type, is_forward) if si and sj are adjacent, else None.

    is_forward=True  → si→sj matches the MEP file's step-1 → step-50 convention.
    is_forward=False → si→sj is the reverse.

    MEP file conventions:
      SD-SD    : SD[111]  → SD[11-1]    (symmetric barriers)
      SD-HAV   : SD[111]  → HAV[100]    forward = SD→HAV
      HAV-HAV  : HAV[100] → HAV[010]   (symmetric barriers)
      HAV-EAV  : HAV[100] → EAV[111]   forward = HAV→EAV
      EAV-EAV  : EAV[111] → EAV[11-1]  (symmetric barriers)
    """
    ti, ii = si
    tj, ij = sj

    if ti == "SD"  and tj == "SD":
        return ("SD-SD",   True)  if _sd_sd_adj(ii, ij)   else None
    if ti == "SD"  and tj == "HAV":
        return ("SD-HAV",  True)  if _sd_hav_adj(ii, ij)  else None
    if ti == "HAV" and tj == "SD":
        return ("SD-HAV",  False) if _sd_hav_adj(ij, ii)  else None
    if ti == "HAV" and tj == "HAV":
        return ("HAV-HAV", True)  if _hav_hav_adj(ii, ij) else None
    if ti == "HAV" and tj == "EAV":
        return ("HAV-EAV", True)  if _hav_eav_adj(ii, ij) else None
    if ti == "EAV" and tj == "HAV":
        return ("HAV-EAV", False) if _hav_eav_adj(ij, ii) else None
    if ti == "EAV" and tj == "EAV":
        return ("EAV-EAV", True)  if _eav_eav_adj(ii, ij) else None
    return None


# =============================================================================
# Barrier loading
# =============================================================================

def _parse_mep(path: str) -> Tuple[float, float]:
    """
    Parse a MERRILL minimum-energy-path file.
    Returns (ΔE_forward, ΔE_backward) in Joules:
      ΔE_forward  = max(E) - E[0]     barrier from step-1 to step-50 state
      ΔE_backward = max(E) - E[-1]    barrier from step-50 to step-1 state
    Both are ≥ 0; a value of 0 means barrierless (spontaneous transition).
    """
    energies = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    energies.append(float(parts[1]))
                except ValueError:
                    pass
    if not energies:
        raise ValueError(f"No energy data found in {path}")
    E    = np.asarray(energies)
    emax = E.max()
    return float(emax - E[0]), float(emax - E[-1])


def _parse_hyst_moment(path: str) -> Optional[np.ndarray]:
    """
    Return (Mx/Ms, My/Ms, Mz/Ms) from the last data row of a MERRILL .hyst file.
    Data rows are comma-separated: mu0H, m_h, m_h/Ms, Mx/Ms, My/Ms, Mz/Ms.
    """
    if not os.path.isfile(path):
        return None
    result = None
    with open(path) as fh:
        for line in fh:
            parts = line.split(',')
            if len(parts) == 6:
                try:
                    result = np.array([float(parts[3]),
                                       float(parts[4]),
                                       float(parts[5])])
                except ValueError:
                    pass
    return result


def _build_hyst_fracs(hyst_dir: str = None) -> Dict[int, Dict[str, float]]:
    """
    Build per-size moment-fraction table from MERRILL size-loop .hyst files.

    Moment fractions are extracted directly from the size loop:
      'SD'      : |M|/Ms from upper.hyst  (SD branch, high moment)
      'HAV/EAV' : |M|/Ms from lower.hyst  (vortex branch, lower moment)
    The lower-branch value is assigned to whichever vortex state types are
    active at that size (from _STATE_TYPES), with no heuristic classification.
    Defaults (1.00 / 0.84 / 0.36) fill sizes without MERRILL hyst data.
    """
    if hyst_dir is None:
        hyst_dir = HYST_DIR
    raw: Dict[int, Dict[str, float]] = {}

    for s in SIZES:
        fracs: Dict[str, float] = {}
        types = _STATE_TYPES.get(s, set())

        upper = os.path.join(hyst_dir, f"x{s}_y{s}_z{s}_upper.hyst")
        lower = os.path.join(hyst_dir, f"x{s}_y{s}_z{s}_lower.hyst")

        m_up = _parse_hyst_moment(upper)
        m_lo = _parse_hyst_moment(lower)
        mag_up = float(np.linalg.norm(m_up)) if m_up is not None else None
        mag_lo = float(np.linalg.norm(m_lo)) if m_lo is not None else None

        # Upper branch → highest-M active state; lower → lowest-M active state.
        # When SD+HAV+EAV overlap, HAV has no clean branch measurement —
        # it is filled later by interpolation from neighboring HAV-only sizes.
        if 'SD' in types and mag_up is not None:
            fracs['SD'] = mag_up
        if 'HAV' in types and 'EAV' in types:
            if 'SD' in types:
                # SD+HAV+EAV: SD=upper, EAV=lower, HAV=skip (interpolate)
                if mag_lo is not None:
                    fracs['EAV'] = mag_lo
            else:
                # HAV+EAV only: HAV=upper (higher M), EAV=lower
                if mag_up is not None:
                    fracs['HAV'] = mag_up
                if mag_lo is not None:
                    fracs['EAV'] = mag_lo
        elif 'HAV' in types:
            if mag_lo is not None:
                fracs['HAV'] = mag_lo
        elif 'EAV' in types:
            if mag_lo is not None:
                fracs['EAV'] = mag_lo

        raw[s] = fracs

    # Interpolate HAV at sizes where it was skipped (SD+HAV+EAV overlap)
    hav_known_s = sorted(s for s in raw if 'HAV' in raw[s])
    hav_missing = sorted(s for s in raw
                         if 'HAV' in _STATE_TYPES.get(s, set())
                         and 'HAV' not in raw[s])
    if hav_known_s and hav_missing:
        _hs = np.array(hav_known_s, dtype=float)
        _hv = np.array([raw[s]['HAV'] for s in hav_known_s])
        for s in hav_missing:
            raw[s]['HAV'] = float(np.interp(s, _hs, _hv))

    # Apply defaults only for state types active at each size
    for s in raw:
        types = _STATE_TYPES.get(s, {'SD', 'HAV', 'EAV'})
        if 'SD'  in types: raw[s].setdefault('SD',  1.00)
        if 'HAV' in types: raw[s].setdefault('HAV', 0.84)
        if 'EAV' in types: raw[s].setdefault('EAV', 0.36)

    return raw


HYST_FRACS: Dict[int, Dict[str, float]] = _build_hyst_fracs()


def build_hyst_fracs(T_celsius: int = 20) -> Dict[int, Dict[str, float]]:
    """
    Build HYST_FRACS for a given temperature using the corresponding hyst subfolder.
    Equivalent to the module-level HYST_FRACS for T_celsius=20.
    """
    hyst_dir = os.path.join(_MONO_BASE, "hyst", f"T{T_celsius}")
    return _build_hyst_fracs(hyst_dir)


_NEW_FILENAMES: Dict[str, str] = {
    "SD-SD":   "x{s}_y{s}_z{s}_SD_1_1_1_to_SD_1_1_-1.txt",
    "SD-HAV":  "x{s}_y{s}_z{s}_SD_1_1_1_to_HAV_1_0_0.txt",
    "HAV-HAV": "x{s}_y{s}_z{s}_HAV_1_0_0_to_HAV_0_1_0.txt",
    "HAV-EAV": "x{s}_y{s}_z{s}_HAV_0_0_1_to_EAV_1_1_1.txt",
    "EAV-EAV": "x{s}_y{s}_z{s}_EAV_1_1_1_to_EAV_1_1_-1.txt",
}

_OLD_FILENAMES: Dict[str, List[str]] = {
    "SD-SD":   ["x{s}_y{s}_z{s}_SD_1_1_1_to_SD_1_1_-1.txt"],
    "SD-HAV":  ["x{s}_y{s}_z{s}_size_hyst_states.txt"],
    "HAV-HAV": ["x{s}_y{s}_z{s}_lower_branch_rotates.txt"],
    "HAV-EAV": ["x{s}_y{s}_z{s}_SV_1_0_0_to_SV_1_1_1.txt"],
    "EAV-EAV": ["x{s}_y{s}_z{s}_SV_1_1_1_to_SV_1_1_-1.txt",
                "x{s}_y{s}_z{s}_lower_branch_rotates.txt"],
}


def load_barriers(size_nm: int, T_celsius: int = 20) -> Dict[str, Tuple[float, float]]:
    """
    Load all available energy barriers for a given grain size and temperature.

    Tries new-convention filenames first, then falls back to old naming.
    For lower_branch_rotates.txt disambiguation:
      - Treated as HAV-HAV only when size_hyst_states.txt is also present
      - Otherwise treated as EAV-EAV
    Returns dict: transition_type -> (dE_forward, dE_backward) in Joules.
    """
    data_dir = os.path.join(_MONO_BASE, "energy_barriers", f"T{T_celsius}")
    folder   = os.path.join(data_dir, f"x{size_nm}_y{size_nm}_z{size_nm}")
    barriers: Dict[str, Tuple[float, float]] = {}

    state_types = _STATE_TYPES.get(int(round(size_nm)), set())
    has_sd_hav_indicator = os.path.isfile(
        os.path.join(folder, f"x{size_nm}_y{size_nm}_z{size_nm}_size_hyst_states.txt"))

    for ttype in ("SD-SD", "SD-HAV", "HAV-HAV", "HAV-EAV", "EAV-EAV"):
        required = set(ttype.split('-'))
        if not required.issubset(state_types):
            continue

        # Try new-style filename
        fpath = os.path.join(folder, _NEW_FILENAMES[ttype].format(s=size_nm))
        if os.path.isfile(fpath):
            barriers[ttype] = _parse_mep(fpath)
            continue

        # Try old-style fallbacks
        for tmpl in _OLD_FILENAMES[ttype]:
            if 'lower_branch_rotates' in tmpl:
                if ttype == "HAV-HAV" and not has_sd_hav_indicator:
                    continue
                if ttype == "EAV-EAV" and has_sd_hav_indicator:
                    continue
            fpath = os.path.join(folder, tmpl.format(s=size_nm))
            if os.path.isfile(fpath):
                barriers[ttype] = _parse_mep(fpath)
                break

    return barriers


# =============================================================================
# Rate and generator matrix
# =============================================================================

def _rate(delta_e: float) -> float:
    """
    Néel-Arrhenius rate k = (1/τ₀) exp(-ΔE / kT).
    Returns 1/τ₀ for barrierless transitions (ΔE ≤ 0) and 0 for overflow.
    """
    if delta_e <= 0.0:
        return 1.0 / TAU0
    exp_arg = delta_e / (K_B * T_K)
    if exp_arg > 700.0:
        return 0.0
    return np.exp(-exp_arg) / TAU0


def build_generator(
    states:   List[State],
    barriers: Dict[str, Tuple[float, float]],
    B_vec:    np.ndarray,
    size_nm:  int,
) -> np.ndarray:
    """
    Build the infinitesimal generator matrix A (Chen et al. 2025, Eq. 5).

    Convention: dp/dt = A p
      A[j, i] = k_{i→j}   for j ≠ i  (rate of flow from state i into state j)
      A[i, i] = -Σ_{j≠i} k_{i→j}     (total outflow from state i)

    Barrier with Zeeman field correction (saddle-point approximation):
      ΔE_{i→j}(B) = ΔE₀ - ½ (m_j - m_i) · B

    Moments m_i use HYST_FRACS so that vortex states (HAV, EAV) contribute
    their correct reduced Zeeman energy, not the full Ms × V value.
    """
    n  = len(states)
    V  = grain_volume(size_nm)
    A  = np.zeros((n, n))
    ms = [_moment(s, V, size_nm) for s in states]

    for i, si in enumerate(states):
        for j, sj in enumerate(states):
            if i == j:
                continue
            tr = get_transition(si, sj)
            if tr is None:
                continue
            ttype, is_fwd = tr
            if ttype not in barriers:
                continue

            dE_fwd, dE_bwd = barriers[ttype]
            dE0 = dE_fwd if is_fwd else dE_bwd

            # Zeeman correction: ΔE(B) = ΔE₀ - ½(m_j - m_i)·B
            dE = dE0 - 0.5 * float(np.dot(ms[j] - ms[i], B_vec))
            dE = max(dE, 0.0)

            k_ij = _rate(dE)
            A[j, i] += k_ij
            A[i, i] -= k_ij

    return A


# =============================================================================
# Probability redistribution at regime boundaries
# =============================================================================

def redistribute(
    p_old:      np.ndarray,
    old_states: List[State],
    new_states: List[State],
) -> np.ndarray:
    """
    Map probability vector when the state space changes between size regimes.

    Same-type mapping (SD→SD, HAV→HAV, EAV→EAV) preserves direction index.
    Cross-type mapping uses adjacency rules:
      SD ↔ EAV : same ⟨111⟩ direction index
      SD ↔ HAV : _sd_hav_adj (each SD has 3 adjacent HAV)
      HAV ↔ EAV: _hav_eav_adj (each HAV shares axis sign with 4 EAV)
    Priority: same type > EAV/SD (same index) > HAV (adjacent, split).
    """
    new_types = {s[0] for s in new_states}
    p_new     = np.zeros(len(new_states))

    for i, s_old in enumerate(old_states):
        stype, idx = s_old
        prob       = p_old[i]

        if stype == "SD":
            if "SD" in new_types:
                for j, s_new in enumerate(new_states):
                    if s_new[0] == "SD" and s_new[1] == idx:
                        p_new[j] += prob; break
            elif "EAV" in new_types:
                for j, s_new in enumerate(new_states):
                    if s_new[0] == "EAV" and s_new[1] == idx:
                        p_new[j] += prob; break
            elif "HAV" in new_types:
                targets = [j for j, s_new in enumerate(new_states)
                           if s_new[0] == "HAV" and _sd_hav_adj(idx, s_new[1])]
                if targets:
                    share = prob / len(targets)
                    for j in targets:
                        p_new[j] += share

        elif stype == "HAV":
            if "HAV" in new_types:
                for j, s_new in enumerate(new_states):
                    if s_new[0] == "HAV" and s_new[1] == idx:
                        p_new[j] += prob; break
            elif "EAV" in new_types:
                hd      = _HAV_DIRS[idx]
                nz      = int(np.flatnonzero(np.abs(hd) > 0.5)[0])
                sgn     = float(np.sign(hd[nz]))
                targets = [j for j, s_new in enumerate(new_states)
                           if s_new[0] == "EAV"
                           and float(np.sign(_direction(s_new)[nz])) == sgn]
                if targets:
                    share = prob / len(targets)
                    for j in targets:
                        p_new[j] += share
            elif "SD" in new_types:
                targets = [j for j, s_new in enumerate(new_states)
                           if s_new[0] == "SD" and _sd_hav_adj(s_new[1], idx)]
                if targets:
                    share = prob / len(targets)
                    for j in targets:
                        p_new[j] += share

        elif stype == "EAV":
            if "EAV" in new_types:
                for j, s_new in enumerate(new_states):
                    if s_new[0] == "EAV" and s_new[1] == idx:
                        p_new[j] += prob; break
            elif "HAV" in new_types:
                targets = [j for j, s_new in enumerate(new_states)
                           if s_new[0] == "HAV"
                           and _hav_eav_adj(s_new[1], idx)]
                if targets:
                    share = prob / len(targets)
                    for j in targets:
                        p_new[j] += share
            elif "SD" in new_types:
                for j, s_new in enumerate(new_states):
                    if s_new[0] == "SD" and s_new[1] == idx:
                        p_new[j] += prob; break

    total = p_new.sum()
    return p_new / total if total > 0.0 else p_new


# =============================================================================
# Markov chain integration
# =============================================================================

def boltzmann_init(states: List[State], B_vec: np.ndarray, size_nm: int) -> np.ndarray:
    """
    Initial Boltzmann distribution in the applied field.
    p_i ∝ exp(+m_i · B / kT)  [Zeeman energy only; Chen et al. Eq. 3]
    Uses HYST_FRACS moments so vortex states have the correct Zeeman weight.
    """
    V   = grain_volume(size_nm)
    E_Z = np.array([-float(np.dot(_moment(s, V, size_nm), B_vec)) for s in states])
    E_Z -= E_Z.min()                    # numerical stability
    p   = np.exp(-E_Z / (K_B * T_K))
    return p / p.sum()


def _steady_state(A: np.ndarray) -> np.ndarray:
    """
    Steady-state distribution of generator A: solve A p = 0, Σp = 1.
    Used when max_rate × dt >> 1 (system has fully equilibrated).
    """
    n = len(A)
    M = A.copy()
    M[-1, :] = 1.0
    b = np.zeros(n); b[-1] = 1.0
    try:
        p_ss = np.linalg.solve(M, b)
    except np.linalg.LinAlgError:
        p_ss = np.linalg.lstsq(M, b, rcond=None)[0]
    p_ss = np.maximum(p_ss.real, 0.0)
    s    = p_ss.sum()
    return p_ss / s if s > 0 else np.ones(n) / n


_EXPM_SAFE_ARG = 700.0   # below this, exp() of any single eigenvalue*dt is
                         # exactly representable in double precision -- see
                         # markov_step docstring for why this replaced a
                         # naive "max_rate*dt > 50" cutoff.


def markov_step(p: np.ndarray, A: np.ndarray, dt: float) -> np.ndarray:
    """
    Evolve probability by one time step: p(t+Δt) = exp(A Δt) · p(t)  (Eq. 4).

    Uses scipy.linalg.expm directly whenever that is numerically safe
    (max_rate * dt <= _EXPM_SAFE_ARG). Above that, expm(A*dt) itself can
    become unreliable (e.g. an unblocked/SP-like max_rate ~1e9/s evolved
    over a geological dt ~years gives an exponent ~1e16, far past where
    scipy's own internal scaling-and-squaring stays accurate) -- handled by
    doing the scaling-and-squaring manually: evolve a short, safe
    sub-interval directly via expm, then repeatedly square that transition
    matrix (M @ M, doubling the elapsed time each round) to reach the full
    duration.

    Two earlier fallback designs were tried and rejected during testing:
    - A single generator-wide steady-state snap whenever max_rate*dt > 50
      (max_rate is only the *fastest* relaxation rate in A; for generators
      with a wide spread of timescales -- confirmed for exchange-coupled
      multi-phase generators with sparse per-edge barrier coverage, where
      eigenvalues can span >10 orders of magnitude -- the fastest mode can
      satisfy that threshold while much slower modes are still far from
      equilibrium, making the whole-system steady state wrong by orders of
      magnitude).
    - Per-eigenmode evaluation via eigendecomposition (mathematically exact,
      and immune to the above issue, but numerically unreliable for
      near-defective matrices -- confirmed for a highly symmetric mono-phase
      generator with 8 numerically-degenerate eigenvalues and an
      eigenvector-matrix condition number ~1e31, which corrupted the result
      even though the eigenvalues themselves were fine).

    Manual scaling-and-squaring avoids both failure modes: it never
    diagonalizes anything (robust for defective/degenerate spectra), and
    every individual expm call stays within the same safe argument range
    used by the direct path (robust for wide timescale spreads, since nothing
    is assumed to have converged early).
    """
    max_rate = -np.min(np.diag(A))
    if max_rate == 0.0:
        return p.copy()

    if max_rate * dt <= _EXPM_SAFE_ARG:
        M = expm(A * dt)
    else:
        n_doublings = int(np.ceil(np.log2(max_rate * dt))) + 1
        M = expm(A * dt / (2 ** n_doublings))
        for _ in range(n_doublings):
            M = M @ M
            # Each column of a transition matrix must sum to exactly 1; tiny
            # per-round floating-point drift (e.g. a stationary eigenvalue
            # landing at 1+1e-15 instead of 1) compounds geometrically over
            # dozens of doublings and can overflow if left uncorrected.
            M = np.maximum(M, 0.0)
            col_sums = M.sum(axis=0)
            M = M / np.where(col_sums > 0, col_sums, 1.0)

    p_new = np.maximum(M @ p, 0.0)
    s     = p_new.sum()
    return p_new / s if s > 0 else p.copy()


def crm_moment_projected(
    states:  List[State],
    p:       np.ndarray,
    B_hat:   np.ndarray,
    size_nm: int,
) -> float:
    """
    Net CRM moment projected along B̂, in Am²  (Chen et al. Eq. 2).

    m_CRM = Σ_i (m_i · B̂) p_i

    where m_i = HYST_FRACS[size][type] × Ms × V × n̂_i  (Am²).
    Returns the net magnetic moment; divide by Ms × V to normalise.
    """
    V = grain_volume(size_nm)
    return sum(float(np.dot(_moment(s, V, size_nm), B_hat)) * pi
               for s, pi in zip(states, p))


# =============================================================================
# CRM simulation — single orientation
# =============================================================================

def run_single_orientation_vector(
    B_vec:        np.ndarray,
    sizes:        List[int],
    growth_time:  float,
    all_barriers: Dict[int, Dict[str, Tuple[float, float]]],
) -> np.ndarray:
    """
    Simulate CRM acquisition for one grain orientation; return the full 3D
    expected moment vector at each size step.

    Analogous to Bellon et al. (2025) Eq. 2: M̃_Nn = Σ_i p_i · M̃_i,
    the thermally-averaged remanence vector in the crystal frame.

    Returns
    -------
    vectors : ndarray, shape (len(sizes), 3)
        Expected 3D moment vector (Am²) at each size step.
    """
    d_total = sizes[-1] - sizes[0]

    states = active_states(sizes[0])
    p      = boltzmann_init(states, B_vec, sizes[0])

    vectors = np.zeros((len(sizes), 3))

    for k, size in enumerate(sizes):
        new_states = active_states(size)
        if new_states != states:
            p      = redistribute(p, states, new_states)
            states = new_states

        V   = grain_volume(size)
        vec = np.zeros(3)
        for i, (st, pi) in enumerate(zip(states, p)):
            vec += _moment(st, V, size) * pi
        vectors[k] = vec

        if k < len(sizes) - 1:
            delta_d = sizes[k + 1] - size
        else:
            delta_d = sizes[k] - sizes[k - 1]
        dt = delta_d * growth_time / d_total
        A  = build_generator(states, all_barriers[size], B_vec, size)
        p  = markov_step(p, A, dt)

    return vectors


def run_single_orientation_final_probs(
    B_vec:        np.ndarray,
    sizes:        List[int],
    growth_time:  float,
    all_barriers: Dict[int, Dict[str, Tuple[float, float]]],
) -> Tuple[List['State'], np.ndarray]:
    """
    Run the Markov chain and return (states, p) at the final size step
    — the probability distribution used to compute vectors[-1] in
    run_single_orientation_vector.  Used to compute E[m_par²] and E[m_perp²]
    for the correct CRM angular-misfit statistics.
    """
    d_total = sizes[-1] - sizes[0]
    states  = active_states(sizes[0])
    p       = boltzmann_init(states, B_vec, sizes[0])

    for k, size in enumerate(sizes):
        new_states = active_states(size)
        if new_states != states:
            p      = redistribute(p, states, new_states)
            states = new_states

        if k == len(sizes) - 1:
            break                          # return p/states BEFORE final step

        delta_d = sizes[k + 1] - size
        dt      = delta_d * growth_time / d_total
        A       = build_generator(states, all_barriers[size], B_vec, size)
        p       = markov_step(p, A, dt)

    return states, p


def run_single_orientation(
    B_vec:          np.ndarray,
    sizes:          List[int],
    growth_time:    float,
    all_barriers:   Dict[int, Dict[str, Tuple[float, float]]],
    remanence:      bool = False,
    remanence_time: float = REMANENCE_TIME_S,
) -> np.ndarray:
    """
    Simulate CRM acquisition for one grain orientation.

    Parameters
    ----------
    B_vec          : applied field vector (T); its direction encodes grain orientation
    sizes          : ordered list of grain sizes (nm), smallest → largest
    growth_time    : total growth time from sizes[0] to sizes[-1] (seconds)
    all_barriers   : pre-loaded barrier dict {size_nm: barriers_dict}
    remanence      : if True, record the moment after a zero-field relaxation step
                      (`remanence_time` seconds) instead of the in-field moment.
                      Blocked grains (tau >> remanence_time) are unaffected; SP
                      grains (tau << remanence_time) relax to zero.  This gives the
                      true remanence rather than the in-field magnetisation.
    remanence_time : zero-field relaxation duration (s) when remanence=True;
                      defaults to REMANENCE_TIME_S (100 s, a typical laboratory
                      measurement timescale).

    Returns
    -------
    crm_vs_size : m_CRM (Am²) at each size step (shape: len(sizes),)
    """
    d_total   = sizes[-1] - sizes[0]
    B_hat     = B_vec / np.linalg.norm(B_vec)
    B_zero    = np.zeros(3)

    states = active_states(sizes[0])
    p      = boltzmann_init(states, B_vec, sizes[0])

    crm_vs_size = np.zeros(len(sizes))

    for k, size in enumerate(sizes):
        new_states = active_states(size)
        if new_states != states:
            p      = redistribute(p, states, new_states)
            states = new_states

        if k < len(sizes) - 1:
            delta_d = sizes[k + 1] - size
        else:
            delta_d = sizes[k] - sizes[k - 1]   # reuse last step for final size
        dt = delta_d * growth_time / d_total
        A  = build_generator(states, all_barriers[size], B_vec, size)
        p  = markov_step(p, A, dt)

        if remanence:
            A0    = build_generator(states, all_barriers[size], B_zero, size)
            p_rec = markov_step(p, A0, remanence_time)
        else:
            p_rec = p
        crm_vs_size[k] = crm_moment_projected(states, p_rec, B_hat, size)

    return crm_vs_size


# =============================================================================
# CRM simulation — random assemblage (Fibonacci sphere of orientations)
# =============================================================================

def _fibonacci_sphere(n: int) -> np.ndarray:
    """n uniformly distributed unit vectors via the Fibonacci spiral (shape: n×3)."""
    phi   = (1.0 + np.sqrt(5.0)) / 2.0
    i     = np.arange(n, dtype=float)
    theta = 2.0 * np.pi * i / phi
    phi_  = np.arccos(1.0 - 2.0 * (i + 0.5) / n)
    return np.column_stack([
        np.sin(phi_) * np.cos(theta),
        np.sin(phi_) * np.sin(theta),
        np.cos(phi_),
    ])


def run_assemblage(
    B_magnitude:    float,
    sizes:          List[int],
    growth_time:    float,
    n_orientations: int = N_ORIENTATIONS,
    remanence:      bool = False,
) -> np.ndarray:
    """
    Simulate CRM for an isotropic assemblage by averaging over random orientations.

    Barriers are loaded once and reused across all n_orientations directions.
    Returns the mean M_CRM/Ms as a function of grain size (shape: len(sizes),).

    Note: for SD-only assemblages the single-orientation [111] shortcut
    (isotropy proof) is exact and much faster than this function.
    """
    all_barriers = {s: load_barriers(s) for s in sizes}
    directions   = _fibonacci_sphere(n_orientations)
    all_crm      = np.zeros((n_orientations, len(sizes)))

    for k, d in enumerate(directions):
        all_crm[k, :] = run_single_orientation(
            B_magnitude * d, sizes, growth_time, all_barriers,
            remanence=remanence)

    return all_crm.mean(axis=0)


# =============================================================================
# CRM assemblage efficiency: blocking model, acquisition sampling,
# misfit statistics
# =============================================================================
#
# The functions below support the *assemblage directional fidelity* problem:
# given an assemblage of N grains growing through SD → HAV → EAV, how well
# does the net CRM direction track the true field direction, and how many
# grains are needed for a reliable paleofield reconstruction?  This is a
# distinct question from the acquisition-magnitude machinery above (which
# asks how much CRM one grain acquires); see crm_efficiency_final*.ipynb.

Blocking = Dict[str, Dict[str, int]]
XiTable  = Dict[str, Dict[str, Dict[str, float]]]


def compute_blocking(
    growth_times: Dict[str, float],
    all_barriers: Dict[int, Dict[str, Tuple[float, float]]],
    sizes:        Optional[List[int]] = None,
    B_mag:        float = 30e-6,
) -> Tuple[Blocking, XiTable]:
    """
    Blocking sizes and Langevin parameters for each growth-time model.

    For each growth time, finds the first grain size at which each of the
    five critical transitions blocks (relaxation time crosses the local
    growth time step Δt = Δsize × growth_time / (sizes[-1] - sizes[0])):
      'SD'        : SP → SD intra-state blocking (SD-SD τ ≥ Δt)
      'HAV'       : SD → HAV transition onset (SD-HAV τ_fwd ≤ Δt)
      'HAV_block' : HAV intra-state blocking (HAV-HAV τ ≥ Δt)
      'EAV'       : HAV → EAV transition onset (HAV-EAV τ_fwd ≤ Δt)
      'EAV_block' : EAV intra-state blocking (HAV-EAV τ_rev ≥ Δt)

    Parameters
    ----------
    growth_times : {label: total growth time (s)}
    all_barriers  : pre-loaded {size_nm: barriers_dict}, e.g. {s: load_barriers(s) for s in sizes}
    sizes         : ordered grain sizes (nm); defaults to module-level SIZES
    B_mag         : reference field magnitude (T) for the ξ table

    Returns
    -------
    blocking : {label: {transition_name: blocking_size_nm}}  (missing keys if
               that transition never blocks/unblocks within the size range)
    xi_table : {label: {'SD'|'HAV'|'EAV': {'s', 'xi', 'm', 'f'}}}
               Langevin parameter ξ = m·B_mag/kT, moment m (Am²) and moment
               fraction f = HYST_FRACS[s][type] at each state's blocking size.
    """
    if sizes is None:
        sizes = SIZES
    d_total = float(sizes[-1] - sizes[0])
    steps   = np.diff(np.array(sizes, dtype=float))
    kT      = K_B * T_K

    blocking: Blocking = {}
    for label, T_growth in growth_times.items():
        dt_k = steps * T_growth / d_total
        b: Dict[str, int] = {}

        for k in range(len(sizes) - 1):
            s = sizes[k]
            if 'SD-SD' not in all_barriers[s]:
                continue
            tau = TAU0 * np.exp(min(all_barriers[s]['SD-SD'][0] / kT, 700))
            if tau >= dt_k[k]:
                b['SD'] = s; break

        for k in range(len(sizes) - 1):
            s = sizes[k]
            if 'SD-HAV' not in all_barriers[s]:
                continue
            tau = TAU0 * np.exp(min(all_barriers[s]['SD-HAV'][0] / kT, 700))
            if tau <= dt_k[k]:
                b['HAV'] = s; break

        for k in range(len(sizes) - 1):
            s = sizes[k]
            if 'HAV-HAV' not in all_barriers[s]:
                continue
            tau = TAU0 * np.exp(min(all_barriers[s]['HAV-HAV'][0] / kT, 700))
            if tau >= dt_k[k]:
                b['HAV_block'] = s; break

        for k in range(len(sizes) - 1):
            s = sizes[k]
            if 'HAV-EAV' not in all_barriers[s]:
                continue
            tau = TAU0 * np.exp(min(all_barriers[s]['HAV-EAV'][0] / kT, 700))
            if tau <= dt_k[k]:
                b['EAV'] = s; break

        for k in range(len(sizes) - 1):
            s = sizes[k]
            if 'HAV-EAV' not in all_barriers[s]:
                continue
            tau = TAU0 * np.exp(min(all_barriers[s]['HAV-EAV'][1] / kT, 700))
            if tau >= dt_k[k]:
                b['EAV_block'] = s; break

        blocking[label] = b

    xi_table: XiTable = {}
    _defaults = {'SD': 1.00, 'HAV': 0.84, 'EAV': 0.36}
    for label in growth_times:
        b = blocking[label]
        xi_row: Dict[str, Dict[str, float]] = {}
        for stype in ('SD', 'HAV', 'EAV'):
            s = b.get(stype)
            if s is None:
                continue
            V = grain_volume(s)
            f = HYST_FRACS[s].get(stype, _defaults[stype])
            m = f * MS * V
            xi = m * B_mag / kT
            xi_row[stype] = {'s': float(s), 'xi': xi, 'm': m, 'f': f}
        xi_table[label] = xi_row

    return blocking, xi_table


def orientation_dot_products(
    n_orient: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Precompute Fibonacci-sphere field-hat orientations and their dot products
    with each state family's easy-axis directions, plus the SD→HAV and
    HAV→EAV adjacency tables used by sample_cascade_step.

    Returns
    -------
    b_hats    : (n_orient, 3) field-hat directions (crystal frame)
    d_dot_b   : {'SD': (n_orient,8), 'HAV': (n_orient,6), 'EAV': (n_orient,8)}
                cos(theta) between each orientation and each state's easy axis
    adjacency : {'SD_TO_HAV': (8,3), 'HAV_TO_EAV': (6,4)} allowed next-state
                indices for each previous state, for sample_cascade_step
    """
    b_hats = _fibonacci_sphere(n_orient)
    D_SD  = np.array(_SD_EAV_DIRS)
    D_HAV = np.array(_HAV_DIRS)
    D_EAV = D_SD

    d_dot_b = {
        'SD':  b_hats @ D_SD.T,
        'HAV': b_hats @ D_HAV.T,
        'EAV': b_hats @ D_EAV.T,
    }
    adjacency = {
        'SD_TO_HAV':  np.array([[j for j in range(6) if _sd_hav_adj(i, j)]
                                 for i in range(8)]),
        'HAV_TO_EAV': np.array([[j for j in range(8) if _hav_eav_adj(i, j)]
                                 for i in range(6)]),
    }
    return b_hats, d_dot_b, adjacency


def softmax_sample(xi_nk: np.ndarray, rng: np.random.Generator, n: int) -> np.ndarray:
    """One Boltzmann-sampled state index per row of xi_nk (shape (n, n_states))."""
    xi_nk = xi_nk - xi_nk.max(axis=1, keepdims=True)
    p     = np.exp(xi_nk)
    p    /= p.sum(axis=1, keepdims=True)
    cum   = p.cumsum(axis=1)
    u     = rng.random(size=(n, 1))
    return np.clip((u > cum).sum(axis=1), 0, p.shape[1] - 1)


def sample_categorical(p: np.ndarray, rng: np.random.Generator, n: int) -> np.ndarray:
    """
    One state index sampled per grain from an **arbitrary** already-computed
    probability vector p (shape (n_states,)) — the same n grains all draw
    from this one distribution, e.g. the exact final Markov-chain state
    probability for a single crystal orientation from
    run_single_orientation_final_probs. Unlike softmax_sample, this does not
    assume a Boltzmann/softmax form; p can come from any source.
    """
    cum = np.cumsum(p)
    u   = rng.random(size=n)
    return np.clip(np.searchsorted(cum, u, side='right'), 0, len(p) - 1)


def sample_from_probabilities(P: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    One state index sampled per row of P (shape (n, n_states)), where each
    row is an **already-normalized probability vector** (not logits) —
    e.g. a pool of grains each carrying its own orientation's exact
    Markov-chain final-state distribution, gathered via `P_orient[n_pool]`.

    Complements sample_categorical (one shared p for all draws) and
    softmax_sample (logits, not probabilities).
    """
    n   = P.shape[0]
    cum = P.cumsum(axis=1)
    u   = rng.random(size=(n, 1))
    return np.clip((u > cum).sum(axis=1), 0, P.shape[1] - 1)


def sample_direct_state(
    xi:           float,
    d_dot_b_pool: np.ndarray,
    rng:          np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fresh-equilibrium Boltzmann sample among all states of one family
    (SD, HAV, or EAV direct model).

    Parameters
    ----------
    xi           : Langevin parameter for this state family (scalar)
    d_dot_b_pool : (N, n_states) cos(theta) between each grain's sampled
                   orientation and each state's easy-axis direction
    rng          : np.random.Generator

    Returns
    -------
    k, cos : (N,) sampled state index and cos(theta) for that state
    """
    n   = d_dot_b_pool.shape[0]
    k   = softmax_sample(xi * d_dot_b_pool, rng, n)
    cos = d_dot_b_pool[np.arange(n), k]
    return k, cos


def sample_cascade_step(
    prev_k:       np.ndarray,
    adjacency:    np.ndarray,
    xi:           float,
    d_dot_b_pool: np.ndarray,
    rng:          np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    One adjacency-restricted cascade step (e.g. SD→HAV or HAV→EAV): sample
    the next domain state from only the states adjacent to each grain's
    previous state, preserving growth history instead of re-equilibrating
    freely among the full next-family state set.

    Parameters
    ----------
    prev_k       : (N,) previous-stage sampled state index per grain
    adjacency    : (n_prev_states, n_adjacent) allowed next-state indices for
                   each previous state (e.g. orientation_dot_products()'s
                   'SD_TO_HAV' or 'HAV_TO_EAV')
    xi           : Langevin parameter of the next state family (scalar)
    d_dot_b_pool : (N, n_states) cos(theta) for the next family's full state set
    rng          : np.random.Generator

    Returns
    -------
    k, cos : (N,) sampled next-state index and cos(theta) for that state
    """
    n   = d_dot_b_pool.shape[0]
    adj = adjacency[prev_k]                                       # (N, n_adjacent)
    xd  = xi * np.take_along_axis(d_dot_b_pool, adj, axis=1)       # (N, n_adjacent)
    loc = softmax_sample(xd, rng, n)
    k   = adj[np.arange(n), loc]
    cos = d_dot_b_pool[np.arange(n), k]
    return k, cos


def moment_components(m: float, cos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Parallel/perpendicular moment components (Am²) given cos(theta) to B_hat."""
    return m * cos, m * np.sqrt(np.maximum(1.0 - cos ** 2, 0.0))


def boltzmann_p(xi: float, d_dot_b: np.ndarray) -> np.ndarray:
    """
    Deterministic per-orientation Boltzmann probability over a state
    family's states — exact population weights, no sampling involved.

    Parameters
    ----------
    xi      : Langevin parameter for this state family (scalar)
    d_dot_b : (N_ORIENT, n_states) cos(theta) between each orientation and
              each state's easy-axis direction

    Returns
    -------
    p : (N_ORIENT, n_states) Boltzmann probability per orientation
    """
    xdb = xi * d_dot_b
    xdb = xdb - xdb.max(axis=1, keepdims=True)
    p = np.exp(xdb)
    p /= p.sum(axis=1, keepdims=True)
    return p


def boltzmann_efficiency(p: np.ndarray, d_dot_b: np.ndarray) -> float:
    """
    Exact mean grain efficiency ⟨cos theta⟩ = ⟨mu_par⟩/m, Boltzmann-averaged
    over all orientations and states — no Monte Carlo sampling, so this is
    perfectly smooth in B (unlike a discrete state draw's sample mean).
    """
    return float((p * d_dot_b).sum(axis=1).mean())


def boltzmann_cascade_step(
    p_prev:       np.ndarray,
    adjacency:    np.ndarray,
    xi_next:      float,
    d_dot_b_next: np.ndarray,
) -> np.ndarray:
    """
    Exact (deterministic) marginal probability over the next state family,
    given the previous family's marginal probability p_prev and the
    adjacency-restricted Boltzmann weights for the next family — the
    population-level analog of sample_cascade_step, with no sampling.

    Parameters
    ----------
    p_prev       : (N_ORIENT, n_prev_states) previous family's marginal
                   probability per orientation
    adjacency    : (n_prev_states, n_adjacent) allowed next-state indices
                   for each previous state
    xi_next      : Langevin parameter of the next state family (scalar)
    d_dot_b_next : (N_ORIENT, n_next_states) cos(theta) for the next
                   family's full state set

    Returns
    -------
    p_next : (N_ORIENT, n_next_states) marginal probability, summed over
             all previous-state paths
    """
    n_orient, n_prev = p_prev.shape
    n_next = d_dot_b_next.shape[1]
    p_next = np.zeros((n_orient, n_next))
    for k in range(n_prev):
        adj = adjacency[k]
        xdb = xi_next * d_dot_b_next[:, adj]
        xdb = xdb - xdb.max(axis=1, keepdims=True)
        p_loc = np.exp(xdb)
        p_loc /= p_loc.sum(axis=1, keepdims=True)
        for j_loc, j in enumerate(adj):
            p_next[:, j] += p_prev[:, k] * p_loc[:, j_loc]
    return p_next


def sample_assemblage_moments(
    M_par_pool:  np.ndarray,
    M_perp_pool: np.ndarray,
    N_values:    List[int],
    K_trials:    int,
    rng:         np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Direct Monte Carlo assemblage moment components — no CLT/Gaussian
    shortcut. For every N in N_values (however large), draws K_trials
    N-grain assemblages **with replacement** from the pool, assigns each
    grain's perpendicular component a random azimuth, and sums.

    Parameters
    ----------
    M_par_pool, M_perp_pool : (N_POOL,) per-grain parallel/perpendicular
                               moment components (Am²), e.g. from
                               grain_pools[label][model]
    N_values : assemblage sizes to evaluate
    K_trials : independent assemblage draws per N
    rng      : np.random.Generator

    Returns
    -------
    M_par_sum, M_perp_x, M_perp_y : (len(N_values), K_trials) summed
        parallel and transverse (lab-frame x/y) moment components (Am²)

    Notes
    -----
    Internally batches over K_trials when N is large enough that a single
    (K_trials, N) index array would be too big to hold in memory (e.g.
    N=1e7 at K_trials=600 would need ~48 GB for `idx` alone) — this is a
    memory-layout optimization only, not a statistical shortcut: the
    result is identical to the unbatched computation, still literal
    per-grain Monte Carlo sampling with replacement, no CLT/Gaussian
    approximation at any N.
    """
    n_pool    = M_par_pool.shape[0]
    n_N       = len(N_values)
    M_par_sum = np.empty((n_N, K_trials))
    M_perp_x  = np.empty((n_N, K_trials))
    M_perp_y  = np.empty((n_N, K_trials))

    MAX_ELEMENTS = 20_000_000   # cap on (batch_K, N) temp-array size

    for i, N in enumerate(N_values):
        batch = max(1, min(K_trials, MAX_ELEMENTS // max(N, 1)))
        for k0 in range(0, K_trials, batch):
            kb   = min(batch, K_trials - k0)
            idx  = rng.integers(0, n_pool, size=(kb, N))
            phis = rng.uniform(0.0, 2.0 * np.pi, size=(kb, N))
            M_par_sum[i, k0:k0 + kb] = M_par_pool[idx].sum(axis=1)
            M_perp_x[i, k0:k0 + kb]  = (M_perp_pool[idx] * np.cos(phis)).sum(axis=1)
            M_perp_y[i, k0:k0 + kb]  = (M_perp_pool[idx] * np.sin(phis)).sum(axis=1)

    return M_par_sum, M_perp_x, M_perp_y


def angular_misfit(
    M_par_sum: np.ndarray, M_perp_x: np.ndarray, M_perp_y: np.ndarray,
) -> np.ndarray:
    """Angular misfit (degrees) between the net assemblage moment and B_hat."""
    M_tot = np.sqrt(M_par_sum ** 2 + M_perp_x ** 2 + M_perp_y ** 2)
    return np.degrees(np.arccos(np.clip(M_par_sum / M_tot, -1.0, 1.0)))


def moment_magnitude(
    M_par_sum: np.ndarray, M_perp_x: np.ndarray, M_perp_y: np.ndarray,
) -> np.ndarray:
    """Net assemblage moment magnitude (Am²)."""
    return np.sqrt(M_par_sum ** 2 + M_perp_x ** 2 + M_perp_y ** 2)


def find_log_intersection(
    x: np.ndarray, y: np.ndarray, y_target: float,
) -> Optional[float]:
    """
    First x (interpolated in log10 x) where y(x) crosses y_target.

    Returns None if y never crosses y_target within the given range.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[valid], y[valid]
    if x.size < 2:
        return None

    order = np.argsort(x)
    x, y  = x[order], y[order]
    diff  = y - y_target

    exact = np.where(np.isclose(diff, 0.0))[0]
    if exact.size:
        return float(x[exact[0]])

    crossing = np.where(diff[:-1] * diff[1:] < 0)[0]
    if crossing.size == 0:
        return None

    i = crossing[0]
    log_x1, log_x2 = np.log10(x[i]), np.log10(x[i + 1])
    frac = (y_target - y[i]) / (y[i + 1] - y[i])
    return float(10 ** (log_x1 + frac * (log_x2 - log_x1)))


def critical_N(
    M_par_pool:  np.ndarray,
    M_perp_pool: np.ndarray,
    N_grid:      List[int],
    K_trials:    int,
    rng:         np.random.Generator,
    threshold_deg: float = 10.0,
    *,
    extend:          bool = False,
    extend_max_N:    int = 10_000_000,
    extend_factor:   float = 3.0,
    extend_K_trials: Optional[List[Tuple[float, int]]] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Critical assemblage size(s) where the direct-MC angular-misfit curve
    crosses threshold_deg, for both the 95th-percentile and mean statistics
    — computed from the **same** Monte Carlo draws (one sampling pass, no
    CLT/closed-form shortcut for either statistic).

    Parameters
    ----------
    M_par_pool, M_perp_pool : (N_POOL,) per-grain moment components (Am²)
                               at the field/model of interest
    N_grid        : assemblage sizes bracketing the expected crossing
    K_trials      : independent assemblage draws per N
    rng           : np.random.Generator
    threshold_deg : misfit threshold defining the "critical" assemblage size
    extend        : if True and `N_grid` alone doesn't bracket a crossing
                     for N_p95 and/or N_mean (curve still above
                     threshold_deg at max(N_grid)), keep evaluating single,
                     geometrically-larger N values (×`extend_factor` each
                     step) beyond max(N_grid) until both are found or
                     `extend_max_N` is reached. Still exact per-grain Monte
                     Carlo at every N tried — no CLT/Gaussian shortcut, at
                     any N, ever. `K_trials` is tapered down at larger N
                     (see `extend_K_trials`) since assemblage-to-assemblage
                     variance shrinks as N grows, so a stable percentile
                     estimate needs fewer independent draws — this keeps
                     the escalation cheap even out to N=1e7 (see
                     `sample_assemblage_moments`'s docstring for how large
                     N is made memory-safe).
    extend_max_N  : hard cap on how far N is allowed to escalate.
    extend_factor : geometric step multiplier for each escalation point.
    extend_K_trials : list of (N_upper_bound, K) pairs, checked in order —
                     the K_trials used for an escalation point at size N is
                     the K paired with the first `N_upper_bound >= N`.
                     Defaults to a taper from `K_trials` down to a floor of
                     50: N<=1e5 -> K_trials//2 (>=200), N<=1e6 -> K_trials//6
                     (>=100), N>1e6 -> K_trials//12 (>=50).

    Returns
    -------
    N_p95, N_mean : critical N for the P95 and mean misfit curves (None if
                    the curve never crosses threshold_deg within N_grid,
                    or — with extend=True — within [N_grid, extend_max_N])
    """
    M_par_sum, Mx, My = sample_assemblage_moments(
        M_par_pool, M_perp_pool, N_grid, K_trials, rng)
    misfits = angular_misfit(M_par_sum, Mx, My)      # (len(N_grid), K_trials)

    Ns    = list(N_grid)
    p95s  = list(np.percentile(misfits, 95, axis=1))
    means = list(misfits.mean(axis=1))

    N_p95  = find_log_intersection(Ns, p95s,  threshold_deg)
    N_mean = find_log_intersection(Ns, means, threshold_deg)

    if extend and (N_p95 is None or N_mean is None):
        if extend_K_trials is None:
            extend_K_trials = [
                (1e5, max(K_trials // 2,  200)),
                (1e6, max(K_trials // 6,  100)),
                (float('inf'), max(K_trials // 12, 50)),
            ]
        N_next = max(Ns) * extend_factor
        while (N_p95 is None or N_mean is None) and N_next <= extend_max_N:
            N_i = max(1, int(round(N_next)))
            K_i = next(k for n_max, k in extend_K_trials if N_i <= n_max)

            Mp_i, Mx_i, My_i = sample_assemblage_moments(
                M_par_pool, M_perp_pool, [N_i], K_i, rng)
            mis_i = angular_misfit(Mp_i, Mx_i, My_i)[0]

            Ns.append(N_i)
            p95s.append(float(np.percentile(mis_i, 95)))
            means.append(float(mis_i.mean()))

            if N_p95 is None:
                N_p95 = find_log_intersection(Ns, p95s, threshold_deg)
            if N_mean is None:
                N_mean = find_log_intersection(Ns, means, threshold_deg)

            N_next *= extend_factor

    return N_p95, N_mean


def lab_frame_basis(b_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane perpendicular to b_hat."""
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(b_hat, ref)) > 0.99:
        ref = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(b_hat, ref); e1 /= np.linalg.norm(e1)
    e2 = np.cross(b_hat, e1);  e2 /= np.linalg.norm(e2)
    return e1, e2


def ea_project(
    dirs: np.ndarray, b_hat: np.ndarray, e1: np.ndarray, e2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Lambert equal-area (Schmidt net) projection of unit vectors `dirs`
    about pole b_hat, with in-plane axes (e1, e2).

    Returns
    -------
    x, y  : projected coordinates (|.| <= sqrt(2))
    upper : bool mask, True where dirs lie on the b_hat-side hemisphere
    """
    dirs = np.atleast_2d(dirs)
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    mz = dirs @ b_hat
    mx = dirs @ e1
    my = dirs @ e2
    upper = mz >= 0
    mz_p  = np.where(upper, mz, -mz)
    mx_p  = np.where(upper, mx, -mx)
    my_p  = np.where(upper, my, -my)
    r_plane = np.sqrt(np.maximum(1.0 - mz_p ** 2, 0.0))
    r_ea    = np.sqrt(np.maximum(2.0 * (1.0 - mz_p), 0.0))
    scale   = np.divide(r_ea, r_plane, out=np.zeros_like(r_ea), where=r_plane > 1e-12)
    return mx_p * scale, my_p * scale, upper


def shade_field_band(
    ax,
    x0: float,
    x1: float,
    fade_frac:  float = 0.06,
    n:          int = 220,
    color:      str = "0.5",
    core_alpha: float = 0.12,
    edge_alpha: float = 0.0,
    zorder:     int = 0,
    orientation: str = "x",
) -> None:
    """
    Shade the band [x0, x1] on a **log-scaled** axis: a roughly
    constant/solid core across most of the band, with a short smoothstep
    fade from edge_alpha to core_alpha right at each of the two borders
    (not a hard-edged rectangle).

    orientation='x' (default): shades a vertical band spanning [x0, x1] in
        x (log-x axis), full height -- the original behavior.
    orientation='y': shades a horizontal band spanning [x0, x1] in y
        (log-y axis), full width -- fades at the top/bottom borders
        instead of left/right.

    Rendered as n abutting axvspan/axhspan strips, log-uniformly spaced
    (not linearly spaced) so the fade width looks visually symmetric on
    the log-scaled axis — a linearly-spaced fade would look much wider
    near x0 than near x1 once warped through the log transform.
    """
    x0, x1 = sorted((float(x0), float(x1)))
    if x0 <= 0.0 or x1 <= x0:
        return

    u_edges = np.linspace(np.log10(x0), np.log10(x1), n + 1)
    u_mid   = 0.5 * (u_edges[:-1] + u_edges[1:])
    edges   = 10 ** u_edges
    fade_u  = fade_frac * (u_edges[-1] - u_edges[0])

    def _smoothstep(t):
        t = np.clip(t, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    if fade_u > 0:
        t_left  = (u_mid - u_edges[0])  / fade_u
        t_right = (u_edges[-1] - u_mid) / fade_u
        alpha = edge_alpha + (core_alpha - edge_alpha) * np.minimum(
            _smoothstep(t_left), _smoothstep(t_right))
    else:
        alpha = np.full(n, core_alpha)

    span_fn = ax.axhspan if orientation == "y" else ax.axvspan
    for i in range(n):
        span_fn(edges[i], edges[i + 1], color=color, alpha=float(alpha[i]),
                zorder=zorder, lw=0)


# ==== Fine-grid interpolation: polynomial barriers + linear moment fractions ====
# Ports 2_crm_SV_final.ipynb's §3 ("Generator Matrix and Spline Interpolation")
# so any notebook can integrate the master equation on a 1 nm grid instead of
# the raw, unevenly-spaced MERRILL knot sizes (as coarse as 10 nm apart above
# 140 nm) -- the same treatment used for the acquisition-magnitude notebooks.

# Single shared default for the barrier-polynomial fit degree. Notebooks pass
# this explicitly (poly_deg=crm.BARRIER_POLY_DEG) rather than relying on the
# function defaults below, so every notebook using the same value is visible
# in each notebook's own source instead of hidden behind an unpassed default.
BARRIER_POLY_DEG = 5

_BARRIER_TRANSITION_REQ: Dict[str, set] = {
    'SD-SD':   {'SD'},
    'SD-HAV':  {'SD', 'HAV'},
    'HAV-HAV': {'HAV'},
    'HAV-EAV': {'HAV', 'EAV'},
    'EAV-EAV': {'EAV'},
}


def fit_barrier_polynomials(
    sizes:        List[int],
    all_barriers: Dict[int, Dict[str, Tuple[float, float]]],
    poly_deg:     int = BARRIER_POLY_DEG,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.poly1d], Dict[str, np.poly1d]]:
    """
    Fit degree-`poly_deg` polynomials to each transition's (forward, backward)
    energy barrier vs grain size, from the MERRILL knot values in
    `all_barriers`. Ports 2_crm_SV_final.ipynb's knot-collection + polyfit cell.

    Returns
    -------
    knots : {'s': {transition: ndarray}, 'fwd': {...}, 'bwd': {...}}
    poly_fwd, poly_bwd : {transition: np.poly1d}, only for transitions with
        >= 3 knots (too few points are left unfitted, same as the notebook).
    """
    mk_s   = {k: [] for k in _BARRIER_TRANSITION_REQ}
    mk_fwd = {k: [] for k in _BARRIER_TRANSITION_REQ}
    mk_bwd = {k: [] for k in _BARRIER_TRANSITION_REQ}

    for s in sizes:
        b  = all_barriers[s]
        at = {st[0] for st in active_states(s)}
        for tr, req in _BARRIER_TRANSITION_REQ.items():
            if tr in b and req.issubset(at):
                mk_s[tr].append(s)
                mk_fwd[tr].append(b[tr][0])
                mk_bwd[tr].append(b[tr][1])

    for tr in mk_s:
        mk_s[tr]   = np.array(mk_s[tr], dtype=float)
        mk_fwd[tr] = np.array(mk_fwd[tr], dtype=float)
        mk_bwd[tr] = np.array(mk_bwd[tr], dtype=float)

    poly_fwd: Dict[str, np.poly1d] = {}
    poly_bwd: Dict[str, np.poly1d] = {}
    for tr in _BARRIER_TRANSITION_REQ:
        n = len(mk_s[tr])
        if n < 3:
            continue
        deg = min(poly_deg, n - 1)
        poly_fwd[tr] = np.poly1d(np.polyfit(mk_s[tr], mk_fwd[tr], deg))
        poly_bwd[tr] = np.poly1d(np.polyfit(mk_s[tr], mk_bwd[tr], deg))

    knots = {'s': mk_s, 'fwd': mk_fwd, 'bwd': mk_bwd}
    return knots, poly_fwd, poly_bwd


def interp_barriers_at(
    size_nm:  int,
    knots:    Dict[str, Dict[str, np.ndarray]],
    poly_fwd: Dict[str, np.poly1d],
    poly_bwd: Dict[str, np.poly1d],
) -> Dict[str, Tuple[float, float]]:
    """
    Evaluate the domain-restricted polynomial barrier fits at an arbitrary
    grain size (not necessarily a MERRILL knot). Ports
    2_crm_SV_final.ipynb's `interp_barriers`. Barriers are clipped to >= 0
    and only returned for transitions whose fit domain covers `size_nm` and
    whose state types are active at that size.
    """
    mk_s = knots['s']
    at = {st[0] for st in active_states(size_nm)}
    b: Dict[str, Tuple[float, float]] = {}

    if 'SD' in at and 'SD-SD' in poly_fwd:
        s_max = float(mk_s['SD-SD'][-1])
        if size_nm <= s_max:
            b['SD-SD'] = (max(0.0, float(poly_fwd['SD-SD'](size_nm))),
                          max(0.0, float(poly_bwd['SD-SD'](size_nm))))
    if {'SD', 'HAV'}.issubset(at) and 'SD-HAV' in poly_fwd:
        s_lo, s_hi = float(mk_s['SD-HAV'][0]), float(mk_s['SD-HAV'][-1])
        if s_lo <= size_nm <= s_hi:
            b['SD-HAV'] = (max(0.0, float(poly_fwd['SD-HAV'](size_nm))),
                           max(0.0, float(poly_bwd['SD-HAV'](size_nm))))
    if 'HAV' in at and 'HAV-HAV' in poly_fwd:
        s_lo, s_hi = float(mk_s['HAV-HAV'][0]), float(mk_s['HAV-HAV'][-1])
        if s_lo <= size_nm <= s_hi:
            b['HAV-HAV'] = (max(0.0, float(poly_fwd['HAV-HAV'](size_nm))),
                            max(0.0, float(poly_bwd['HAV-HAV'](size_nm))))
    if {'HAV', 'EAV'}.issubset(at) and 'HAV-EAV' in poly_fwd:
        s_lo, s_hi = float(mk_s['HAV-EAV'][0]), float(mk_s['HAV-EAV'][-1])
        if s_lo <= size_nm <= s_hi:
            b['HAV-EAV'] = (max(0.0, float(poly_fwd['HAV-EAV'](size_nm))),
                            max(0.0, float(poly_bwd['HAV-EAV'](size_nm))))
    if 'EAV' in at:
        if 'EAV-EAV' in poly_fwd:
            s_min_ee = float(mk_s['EAV-EAV'][0])
            if size_nm >= s_min_ee:
                b['EAV-EAV'] = (max(0.0, float(poly_fwd['EAV-EAV'](size_nm))),
                                max(0.0, float(poly_bwd['EAV-EAV'](size_nm))))
            else:
                b['EAV-EAV'] = (0.0, 0.0)
        else:
            b['EAV-EAV'] = (0.0, 0.0)
    return b


def load_size_loop_moments(
    T_celsius: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Scan raw MERRILL size-loop .hyst files (isotropic x=y=z grains, upper and
    lower branches) directly from disk, independent of `SIZES` -- the knot
    set for moment-fraction interpolation need not match the barrier knot
    set. Ports 2_crm_SV_final.ipynb's size-loop-scanning cell.

    Returns
    -------
    sl_upper_s, sl_upper_m, sl_lower_s, sl_lower_m : ndarray
        Sizes (nm) and |M|/Ms for the upper and lower branches, sorted by size.
    """
    hyst_dir = os.path.join(_MONO_BASE, 'hyst', f'T{T_celsius}')
    sl_upper: Dict[int, float] = {}
    sl_lower: Dict[int, float] = {}

    for fname in sorted(os.listdir(hyst_dir)):
        if not fname.endswith('.hyst'):
            continue
        size = int(fname.split('_')[0][1:])
        branch = fname.rsplit('_', 1)[-1].replace('.hyst', '')
        m = _parse_hyst_moment(os.path.join(hyst_dir, fname))
        if m is None:
            continue
        mag = float(np.linalg.norm(m))
        if branch == 'upper':
            sl_upper[size] = mag
        elif branch == 'lower':
            sl_lower[size] = mag

    sl_upper_s = np.array(sorted(sl_upper), dtype=float)
    sl_upper_m = np.array([sl_upper[int(s)] for s in sl_upper_s])
    sl_lower_s = np.array(sorted(sl_lower), dtype=float)
    sl_lower_m = np.array([sl_lower[int(s)] for s in sl_lower_s])
    return sl_upper_s, sl_upper_m, sl_lower_s, sl_lower_m


def interp_fracs_at(
    size_nm:    int,
    sl_upper_s: np.ndarray, sl_upper_m: np.ndarray,
    sl_lower_s: np.ndarray, sl_lower_m: np.ndarray,
) -> Dict[str, float]:
    """
    Linearly-interpolated moment fractions at an arbitrary grain size, from
    raw size-loop data (see `load_size_loop_moments`). Ports
    2_crm_SV_final.ipynb's `interp_fracs`: upper branch -> SD (or HAV where
    HAV is the higher-moment state); lower branch -> the vortex state(s).
    """
    at = {st[0] for st in active_states(size_nm)}
    fracs: Dict[str, float] = {}
    if 'SD' in at:
        fracs['SD'] = float(np.clip(np.interp(size_nm, sl_upper_s, sl_upper_m), 0.0, 1.0))
    if 'HAV' in at and 'EAV' in at:
        if 'SD' in at:
            fracs['HAV'] = float(np.clip(np.interp(size_nm, sl_lower_s, sl_lower_m), 0.0, 1.0))
            fracs['EAV'] = float(np.clip(np.interp(size_nm, sl_lower_s, sl_lower_m), 0.0, 1.0))
        else:
            fracs['HAV'] = float(np.clip(np.interp(size_nm, sl_upper_s, sl_upper_m), 0.0, 1.0))
            fracs['EAV'] = float(np.clip(np.interp(size_nm, sl_lower_s, sl_lower_m), 0.0, 1.0))
    elif 'HAV' in at:
        fracs['HAV'] = float(np.clip(np.interp(size_nm, sl_lower_s, sl_lower_m), 0.0, 1.0))
    elif 'EAV' in at:
        fracs['EAV'] = float(np.clip(np.interp(size_nm, sl_lower_s, sl_lower_m), 0.0, 1.0))
    return fracs


def build_fine_grid(
    sizes:        List[int],
    all_barriers: Dict[int, Dict[str, Tuple[float, float]]],
    step_nm:      int = 1,
    poly_deg:     int = BARRIER_POLY_DEG,
    T_celsius:    int = 20,
) -> Dict[str, object]:
    """
    High-level orchestration: fit barrier polynomials + load raw size-loop
    moments, then build a `step_nm`-spaced fine grid spanning
    [sizes[0], sizes[-1]] with interpolated barriers, plus the extra
    HYST_FRACS entries (for sizes not already in the module-level HYST_FRACS)
    needed to run the Markov chain on it. Ports 2_crm_SV_final.ipynb's §3
    wholesale, generalized to any `sizes`/`all_barriers`.

    Returns a dict with:
      'fine_sizes'    : list[int]
      'fine_barriers' : {size: barriers_dict}, one entry per fine_sizes
      'fine_fracs'    : {size: fracs_dict}, only for sizes not already in
                        HYST_FRACS -- merge into HYST_FRACS before running
                        the Markov chain on fine_sizes
      'knots', 'poly_fwd', 'poly_bwd' : from fit_barrier_polynomials, for
                        diagnostic plotting
      'sl_upper_s', 'sl_upper_m', 'sl_lower_s', 'sl_lower_m' : raw size-loop
                        moment data, for diagnostic plotting
    """
    knots, poly_fwd, poly_bwd = fit_barrier_polynomials(sizes, all_barriers, poly_deg)
    sl_upper_s, sl_upper_m, sl_lower_s, sl_lower_m = load_size_loop_moments(T_celsius)

    fine_sizes = list(range(sizes[0], sizes[-1] + 1, step_nm))
    fine_barriers = {s: interp_barriers_at(s, knots, poly_fwd, poly_bwd) for s in fine_sizes}
    fine_fracs = {
        s: interp_fracs_at(s, sl_upper_s, sl_upper_m, sl_lower_s, sl_lower_m)
        for s in fine_sizes if s not in HYST_FRACS
    }

    return {
        'fine_sizes':    fine_sizes,
        'fine_barriers': fine_barriers,
        'fine_fracs':    fine_fracs,
        'knots':         knots,
        'poly_fwd':      poly_fwd,
        'poly_bwd':      poly_bwd,
        'sl_upper_s': sl_upper_s, 'sl_upper_m': sl_upper_m,
        'sl_lower_s': sl_lower_s, 'sl_lower_m': sl_lower_m,
    }


# =============================================================================
# Multi-phase (exchange-coupled) barriers: remagnetization by spin-polarised
# molecules. Covers the full 20-200 nm range (SD, HAV, and EAV), unlike the
# SD-only multi-phase dataset used in 1_crm_SD_final.ipynb's Sec. 7-9.
#
# Symmetry is broken by exchange coupling (it favours [1,1,1]), so unlike the
# mono-phase data -- which shares one representative barrier per transition
# TYPE across all symmetry-equivalent edges -- every individual adjacency
# edge generally needs its own file. New-style files are per-edge
# (explicit direction pair); old-style (lower_branch_rotates/
# size_hyst_states, no per-pair direction) are a single shared fallback
# value per transition type, same convention as load_barriers().
# =============================================================================

_MULTI_BASE = r"d:\prebiotic_crm\octahedron\multi_phase\energy_barriers"

_MULTI_OLD_NAMES: Dict[str, str] = {
    'SD-HAV':  'size_hyst_states',
    'HAV-HAV': 'lower_branch_rotates',
    'EAV-EAV': 'lower_branch_rotates',
}


def _multi_state_label(s: State) -> str:
    """Integer direction label matching multi-phase filenames, e.g. 'SD_-1_-1_1'."""
    stype, _ = s
    d = _direction(s)
    scale = 1.0 if stype == 'HAV' else np.sqrt(3.0)
    ints = np.round(d * scale).astype(int)
    return f"{stype}_{ints[0]}_{ints[1]}_{ints[2]}"


def _parse_mep_neb_multi(path: str) -> Tuple[float, float]:
    """
    Domain-restricted NEB barrier parser for multi-phase (exchange-coupled)
    MEP files (50-point profiles): saddle = max(E[15:35]), local minima =
    min(E[0:15]) / min(E[35:50]) -- avoids picking up spurious endpoint
    wiggles the way a plain max(E)-E[0] parse (_parse_mep) would. Returns
    (dE_forward, dE_backward) in Joules.
    """
    energies = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    energies.append(float(parts[1]))
                except ValueError:
                    pass
    E = np.asarray(energies)
    saddle = E[15:35].max()
    return float(saddle - E[0:15].min()), float(saddle - E[35:50].min())


_MULTI_FNAME_RE = re.compile(
    r'^x\d+_y\d+_z\d+_(?P<t1>[A-Z]+)_(?P<d1>[\-\d_]+?)_to_(?P<t2>[A-Z]+)_(?P<d2>[\-\d_]+?)_'
    r'(?P<ev>[\d.eE+-]+)eV\.txt$')
_MULTI_OLD_RE = re.compile(
    r'^x\d+_y\d+_z\d+_(?P<name>lower_branch_rotates|size_hyst_states)_'
    r'(?P<ev>[\d.eE+-]+)eV\.txt$')


def _scan_multi_dir(size_nm: int) -> Tuple[Dict[Tuple[str, str, str, str], List[Tuple[float, str]]],
                                            Dict[str, List[Tuple[float, str]]]]:
    """
    One directory listing + regex pass, cached per size: returns
    (new_style, old_style) where new_style maps
    (type1, dir1, type2, dir2) -> sorted [(ev, filepath), ...] and old_style
    maps name -> sorted [(ev, filepath), ...]. Used to snap to the nearest
    *available* exchange energy per file group, since different edges/sizes
    have different (and not always overlapping) exchange-energy grids --
    confirmed by direct inspection (e.g. a 10-point grid at old-style-only
    sizes vs. a 15-point grid where per-edge new-style files exist).
    """
    folder = os.path.join(_MULTI_BASE, f"x{size_nm}_y{size_nm}_z{size_nm}")
    new_style: Dict[Tuple[str, str, str, str], List[Tuple[float, str]]] = {}
    old_style: Dict[str, List[Tuple[float, str]]] = {}
    if not os.path.isdir(folder):
        return new_style, old_style
    for fname in os.listdir(folder):
        if not fname.endswith('.txt') or '_finalPath' in fname or '_initialPath' in fname:
            continue
        m = _MULTI_FNAME_RE.match(fname)
        if m:
            key = (m.group('t1'), m.group('d1'), m.group('t2'), m.group('d2'))
            new_style.setdefault(key, []).append((float(m.group('ev')), os.path.join(folder, fname)))
            continue
        m2 = _MULTI_OLD_RE.match(fname)
        if m2:
            old_style.setdefault(m2.group('name'), []).append(
                (float(m2.group('ev')), os.path.join(folder, fname)))
    for d in (new_style, old_style):
        for k in d:
            d[k].sort(key=lambda t: t[0])
    return new_style, old_style


_multi_dir_cache: Dict[int, Tuple[dict, dict]] = {}


def _nearest_ev_path(entries: List[Tuple[float, str]], exchange_ev: float) -> str:
    return min(entries, key=lambda t: abs(t[0] - exchange_ev))[1]


def load_multi_pair_barriers(
    size_nm:     int,
    exchange_ev: float,
    states:      List[State],
) -> Dict[Tuple[int, int], Tuple[float, float]]:
    """
    Per-edge exchange-coupled barriers for every valid adjacency edge among
    `states`, at `size_nm`, nearest available to `exchange_ev`. New-style
    (explicit direction pair) files take priority; old-style (a single
    shared value per transition type) is used only for edges with no
    new-style file for either direction. Edges with neither are simply
    omitted (that specific transition doesn't happen for this grain -- same
    "if ttype not in barriers: skip" behaviour as build_generator's
    mono-phase case).

    Different edges (and old-style-only sizes) can have different, non-
    overlapping exchange-energy grids -- confirmed by direct inspection --
    so this snaps to the nearest available point *per file group* rather
    than requiring an exact `exchange_ev` match, which would silently drop
    coverage at sizes/edges whose grid doesn't include that exact value.

    Returns {(i, j): (dE_fwd, dE_bwd)} keyed by index into `states`, i < j,
    values in Joules. dE_fwd is i->j, dE_bwd is j->i.
    """
    if size_nm not in _multi_dir_cache:
        _multi_dir_cache[size_nm] = _scan_multi_dir(size_nm)
    new_style, old_style = _multi_dir_cache[size_nm]

    out: Dict[Tuple[int, int], Tuple[float, float]] = {}
    old_barrier_cache: Dict[str, Tuple[float, float]] = {}

    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            si, sj = states[i], states[j]
            tr = get_transition(si, sj)
            if tr is None:
                continue
            ttype, _ = tr

            lbl_i, lbl_j = _multi_state_label(si), _multi_state_label(sj)
            key_ij = tuple(lbl_i.split('_', 1) + lbl_j.split('_', 1))
            key_ji = tuple(lbl_j.split('_', 1) + lbl_i.split('_', 1))

            if key_ij in new_style:
                out[(i, j)] = _parse_mep_neb_multi(_nearest_ev_path(new_style[key_ij], exchange_ev))
                continue
            if key_ji in new_style:
                bwd, fwd = _parse_mep_neb_multi(_nearest_ev_path(new_style[key_ji], exchange_ev))
                out[(i, j)] = (fwd, bwd)
                continue

            old_name = _MULTI_OLD_NAMES.get(ttype)
            if old_name is None or old_name not in old_style:
                continue
            if old_name not in old_barrier_cache:
                old_barrier_cache[old_name] = _parse_mep_neb_multi(
                    _nearest_ev_path(old_style[old_name], exchange_ev))
            out[(i, j)] = old_barrier_cache[old_name]

    return out


def build_generator_multi(
    states:    List[State],
    pair_bars: Dict[Tuple[int, int], Tuple[float, float]],
    B_vec:     np.ndarray,
    size_nm:   int,
) -> np.ndarray:
    """
    Generator matrix A = Q^T from per-EDGE exchange-coupled barriers, as
    returned by load_multi_pair_barriers -- generalizes build_generator
    (which looks up one shared barrier per transition TYPE) to barriers that
    can differ edge-by-edge, since exchange coupling breaks the symmetry
    that lets mono-phase data share one representative value. Same Zeeman
    correction convention: dE(B) = dE0 - 0.5*(m_j-m_i).B.
    """
    n  = len(states)
    V  = grain_volume(size_nm)
    ms = [_moment(s, V, size_nm) for s in states]
    A  = np.zeros((n, n))
    for (i, j), (dE_fwd, dE_bwd) in pair_bars.items():
        for src, dst, dE0 in [(i, j, dE_fwd), (j, i, dE_bwd)]:
            dE = dE0 - 0.5 * float(np.dot(ms[dst] - ms[src], B_vec))
            dE = max(dE, 0.0)
            k  = _rate(dE)
            A[dst, src] += k
            A[src, src] -= k
    return A


# =============================================================================
# CRM remagnetization: stage-sequence engine
# =============================================================================
#
# Generalizes the isolated -> exchange-coupled -> isolated post-growth
# pipeline (originally hand-written per stage in 4_CRM_remag.ipynb) into an
# arbitrary sequence of stages, each independently isolated or exchange-
# coupled, with its own duration, field, and (for exchange stages) exchange
# energy. This is what makes repeated exposure cycles and per-stage field/
# energy control a matter of building a longer `stage_specs` list rather than
# writing a new loop per scenario. See 5_CRM_remag_paramspace.ipynb.

def _log_time_points(t_total: float, n: int) -> np.ndarray:
    """n log-spaced points from t_total/1e4 to t_total (avoids t=0 in logspace)."""
    return np.logspace(np.log10(t_total / 1e4), np.log10(t_total), n)


def moment_vector(states: List[State], p: np.ndarray, size_nm: int) -> np.ndarray:
    """Full 3-vector net moment (Am2) for probability vector p over states."""
    V  = grain_volume(size_nm)
    ms = np.array([_moment(s, V, size_nm) for s in states])
    return (p[:, None] * ms).sum(axis=0)


def run_stage(
    p:        np.ndarray,
    states:   List[State],
    size_nm:  int,
    A:        np.ndarray,
    duration: float,
    B_hat:    np.ndarray,
    n_t:      int = 25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evolve p through fixed generator A for `duration` seconds, recording the
    B_hat-projected moment at n_t log-spaced checkpoints within the stage.

    Returns (p_final, t_local, m_traj) where t_local is seconds elapsed
    since the start of this stage (shape (n_t,)) and m_traj is the
    corresponding projected moment (Am2, shape (n_t,)).

    n_t=1 is a special case: a single step spanning the *entire* duration
    (not `_log_time_points`' t_total/1e4 first point) -- since A is constant
    within a stage, this gives the exact same final p as any finer
    subdivision (expm(A*dt1) @ expm(A*dt2) == expm(A*(dt1+dt2))), just
    without an intermediate trajectory. Endpoint-only sweeps should use
    n_t=1 for a large speedup with zero accuracy loss on the final state.
    """
    if n_t == 1:
        t_local = np.array([duration])
    else:
        t_local = _log_time_points(duration, n_t)
    dt = np.diff(np.concatenate([[0.0], t_local]))
    m_traj  = np.zeros(n_t)
    for ti in range(n_t):
        p = markov_step(p, A, dt[ti])
        m_traj[ti] = crm_moment_projected(states, p, B_hat, size_nm)
    return p, t_local, m_traj


def run_stage_sequence(
    p0:           np.ndarray,
    states:       List[State],
    size_nm:      int,
    stage_specs:  List[Dict],
    B_hat:        np.ndarray,
    all_barriers: Dict[int, Dict[str, Tuple[float, float]]],
    n_t:          int = 25,
) -> List[Dict]:
    """
    Chain `run_stage` calls over an arbitrary sequence of stages.

    Each entry of `stage_specs` is a dict:
      {'kind': 'isolated', 'duration': seconds, 'B_vec': array(3,)}
      {'kind': 'exchange', 'duration': seconds, 'B_vec': array(3,),
       'exchange_ev': float}

    The generator is rebuilt once per stage (barriers are constant within a
    stage, only the sequence of stages changes them) -- 'isolated' uses
    `build_generator` with `all_barriers[size_nm]`; 'exchange' uses
    `load_multi_pair_barriers`/`build_generator_multi` at that stage's
    `exchange_ev`.

    Returns a list (one entry per stage) of
      {'p': p_end, 'vec': moment_vector(...), 't_local':..., 'm_traj':...}
    """
    p = p0.copy()
    out = []
    for spec in stage_specs:
        B_vec = spec['B_vec']
        if spec['kind'] == 'isolated':
            A = build_generator(states, all_barriers[size_nm], B_vec, size_nm)
        elif spec['kind'] == 'exchange':
            pair_bars = load_multi_pair_barriers(size_nm, spec['exchange_ev'], states)
            A = build_generator_multi(states, pair_bars, B_vec, size_nm)
        else:
            raise ValueError(f"unknown stage kind: {spec['kind']!r}")

        p, t_local, m_traj = run_stage(p, states, size_nm, A, spec['duration'], B_hat, n_t)
        out.append({'p': p, 'vec': moment_vector(states, p, size_nm),
                     't_local': t_local, 'm_traj': m_traj})
    return out


def flip_angle_deg(v0: np.ndarray, v1: np.ndarray) -> float:
    """Angle (degrees) between two moment vectors. NaN if either is ~zero."""
    n0, n1 = np.linalg.norm(v0), np.linalg.norm(v1)
    if n0 < 1e-30 or n1 < 1e-30:
        return float('nan')
    c = np.dot(v0, v1) / (n0 * n1)
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def intensity_ratio(v0: np.ndarray, v1: np.ndarray) -> float:
    """|v1| / |v0| -- final-over-initial moment magnitude. NaN if v0 ~ zero."""
    n0 = np.linalg.norm(v0)
    return float(np.linalg.norm(v1) / n0) if n0 > 1e-30 else float('nan')
