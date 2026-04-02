from __future__ import annotations

"""
LTspice / PyLTSpice search engine for hardware-neuron phase portraits and bistability.

What this script does
---------------------
1) Starts from an existing LTspice .net file (recommended) or a .asc-generated .net file.
2) Creates many circuit variants by changing:
   - membrane capacitance Cv
   - slow-variable capacitance Cu
   - input current amplitude(s)
   - optional feedback / shaping topology
   - optional bias source values
3) For each variant it runs two simulations:
   - a phase-portrait simulation (forced U and V nodes, comparator/reset removed)
   - a transient bistability test (activate pulse -> coast -> deactivate pulse)
4) Scores each configuration and writes:
   - CSV summary
   - per-run PNG plots
   - the generated .net files

Important limitation
--------------------
The AO6608 models you uploaded are .MODEL VDMOS definitions, while the ZVN3310A and
ZVP3310A uploads are .SUBCKT definitions. That means direct one-line swapping of the core
M-transistors in your current Wijekoon-style netlist is not universally compatible. This
script therefore uses the AO6608 core as the default search baseline, and treats topology
changes, capacitances, currents, and bias voltages as the main search axes.

If you later want, a second version can generate alternate netlist templates that use X...
subcircuit instantiations for ZVN/ZVP-based cores.
"""

from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
import csv
import math
import re
import shutil
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from PyLTSpice import LTspice, RawRead, SimRunner


# -----------------------------------------------------------------------------
# User configuration
# -----------------------------------------------------------------------------

BASE_NETLIST = Path(r"C:\Users\abias\Downloads\a06608workingtests\a19working92625\pyltspice1\a6608test22utest1.cir")
OUTPUT_DIR = Path(r"C:\Users\abias\Downloads\a06608workingtests\a19working92625\pyltspice1\phase_portrait_search")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# If LTspice is already discoverable by PyLTSpice, leave this as LTspice.
SIMULATOR = LTspice

# Node names for the state variables.
NODE_V = "n009"  # membrane-potential node V
NODE_U = "n005"  # slow-variable node U
VDD_NODE = "n002"

# Component references in the current design.
CV_REF = "Cv"
CU_REF = "Cu"

# Reset / comparator / external-input references to remove in phase-portrait mode.
# Keep this list editable because exact instance names can differ between schematic exports.
PHASE_REMOVE_REFS = [
    "M5", "M8", "M9", "M10", "M11", "M12", "M13", "M14",
    "V4", "I4", "XU1", "X§U1"
]

# References to remove in transient mode when replacing the original input stage
# with a controlled test pulse.
TRANSIENT_INPUT_REMOVE_REFS = ["V4", "I4", "XU1", "X§U1"]

# Default analysis windows and state-space ranges.
PHASE_V_RANGE = (0.0, 3.3, 0.10)   # start, stop, step for forced V
PHASE_U_RANGE = (0.0, 2.5, 0.10)   # start, stop, step for forced U
TRANSIENT_TSTOP = 0.120            # seconds
MAX_PARALLEL = 1                   # use 1 first for robustness; increase later if desired

# Search space.
SEARCH_SPACE = {
    "Cv": [0.10e-6, 0.22e-6, 0.47e-6],
    "Cu": [1.0e-6, 2.2e-6, 10e-6, 22e-6],
    "I_bias": [0.00e-6, 0.05e-6, 0.10e-6],
    # First pulse: include weaker "turn-on" amplitudes.
    "I_activate": [0.05e-6, 0.10e-6, 0.20e-6, 0.50e-6, 1.00e-6],
    # Second pulse: allow either polarity and larger amplitude because
    # some candidates may switch OFF with a larger pulse of the same sign.
    "I_deactivate": [-1.00e-6, -0.50e-6, -0.20e-6, 0.20e-6, 0.50e-6, 1.00e-6],
    "vdd": [3.3],
    # Approximate local-ground / negative-rail exploration by shifting all
    # ground-referenced bias sources and the forced phase-plane voltages.
    # NOTE: this is a bias-offset approximation, not a full netlist-wide ground remap.
    "vss_offset": [0.0, -0.30, -0.60],
    "vc": [0.0, 0.1, 0.2],           # only applies if a Vc source exists and is kept
    "vth": [0.8, 1.0, 1.2],          # only applies if a Vth source exists and is kept
    "vbias": [0.6, 0.8, 1.0],        # only applies if a Vbias source exists and is kept
    "topology": [
        "base",
        "cap_feedback",
        "res_feedback",
        "rc_input",
        "rlc_input",
    ],
}


# Iteration order for the Cartesian search.
# Python's itertools.product advances the rightmost iterable fastest, so put the
# parameters you want to vary first in short test runs on the RIGHT side.
SEARCH_KEY_ORDER = [
    "vdd",
    "vss_offset",
    "vc",
    "vth",
    "vbias",
    "topology",
    "I_bias",
    "Cu",
    "Cv",
    "I_activate",
    "I_deactivate",
]

# Optional: randomize candidate order after generation for broader early coverage.
RANDOMIZE_CANDIDATES = True
RANDOMIZE_SEED = 1

# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    Cv: float
    Cu: float
    I_bias: float
    I_activate: float
    I_deactivate: float
    vdd: float
    vss_offset: float
    vc: float
    vth: float
    vbias: float
    topology: str


@dataclass
class RunResult:
    name: str
    candidate: Candidate
    phase_ok: bool = False
    transient_ok: bool = False
    fixed_point_count: int = 0
    stable_point_count: int = 0
    evoked_spikes: int = 0
    coast_spikes: int = 0
    off_pulse_spikes: int = 0
    final_spikes: int = 0
    spike_threshold: float = 0.0
    phase_score: float = 0.0
    bistability_score: float = 0.0
    total_score: float = 0.0
    notes: str = ""
    phase_netlist: str = ""
    trans_netlist: str = ""
    phase_raw: str = ""
    trans_raw: str = ""


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def eng(x: float) -> str:
    """Engineering-format string that LTspice accepts well."""
    ax = abs(x)
    if ax == 0:
        return "0"
    prefixes = [
        (1e-12, "p"),
        (1e-9, "n"),
        (1e-6, "u"),
        (1e-3, "m"),
        (1e0, ""),
        (1e3, "k"),
        (1e6, "meg"),
    ]
    chosen_scale, chosen_prefix = 1.0, ""
    for scale, prefix in prefixes:
        if ax < scale * 1000:
            chosen_scale, chosen_prefix = scale, prefix
            break
    value = x / chosen_scale
    if abs(value - round(value)) < 1e-12:
        return f"{int(round(value))}{chosen_prefix}"
    return f"{value:.6g}{chosen_prefix}"


def sanitize_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def split_before_end(netlist_text: str) -> tuple[str, str]:
    m = re.search(r"(?im)^\.end\b.*$", netlist_text)
    if not m:
        return netlist_text.rstrip() + "\n", ".end\n"
    return netlist_text[:m.start()], netlist_text[m.start():]


def filter_netlist_lines(text: str, remove_refs: Sequence[str], remove_analyses: bool) -> str:
    refs = {r.upper() for r in remove_refs}
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            out.append(line)
            continue

        token = stripped.split()[0]
        token_upper = token.upper()

        # Remove specific component references.
        if token_upper in refs:
            continue

        # Remove old analysis directives so the generated variant has exactly one analysis.
        if remove_analyses and token.startswith('.'):
            directive = token_upper
            if directive in {
                ".TRAN", ".AC", ".DC", ".TF", ".NOISE", ".STEP", ".OP",
                ".MEAS", ".FOUR", ".BACKANNO", ".SAVE", ".PROBE", ".PLOT"
            }:
                continue

        out.append(line)
    return "\n".join(out) + "\n"


def replace_component_value_line(text: str, ref: str, new_value: str) -> str:
    pattern = re.compile(rf"(?im)^({re.escape(ref)}\s+\S+\s+\S+\s+)(\S+)(.*)$")
    def repl(m: re.Match[str]) -> str:
        return f"{m.group(1)}{new_value}{m.group(3)}"
    new_text, n = pattern.subn(repl, text, count=1)
    return new_text if n else text


def replace_source_dc_value(text: str, ref: str, new_value: float) -> str:
    pattern = re.compile(rf"(?im)^({re.escape(ref)}\s+\S+\s+\S+\s+)(.*)$")
    def repl(m: re.Match[str]) -> str:
        tail = m.group(2).strip()
        if tail.upper().startswith("DC "):
            return f"{m.group(1)}DC {eng(new_value)}"
        return f"{m.group(1)}{eng(new_value)}"
    new_text, n = pattern.subn(repl, text, count=1)
    return new_text if n else text


def local_maxima_count(t: np.ndarray, x: np.ndarray, threshold: float, t0: float, t1: float) -> int:
    mask = (t >= t0) & (t <= t1)
    if mask.sum() < 3:
        return 0
    y = x[mask]
    yy_prev = y[:-2]
    yy_mid = y[1:-1]
    yy_next = y[2:]
    peaks = (yy_mid > yy_prev) & (yy_mid >= yy_next) & (yy_mid > threshold)
    return int(np.count_nonzero(peaks))


def read_raw_header(raw_path: Path, max_bytes: int = 16384) -> str:
    with raw_path.open("rb") as f:
        blob = f.read(max_bytes)
    text = blob.decode("utf-8", errors="ignore")
    if "Binary:" in text:
        text = text.split("Binary:", 1)[0]
    return text


def classify_raw_header(raw_path: Path) -> tuple[str, bool, list[str]]:
    hdr = read_raw_header(raw_path)
    m = re.search(r"(?im)^Plotname:\s*(.+)$", hdr)
    plotname = m.group(1).strip() if m else ""
    has_time = bool(re.search(r"(?im)^\s*0\s+time\s+time\s*$", hdr))
    vars_found = re.findall(r"(?im)^\s*\d+\s+(\S+)\s+\S+\s*$", hdr)
    return plotname, has_time, vars_found


def resolve_preferred_raw(raw_path: Path, want_plotname: str) -> Path:
    """
    LTspice may emit sibling RAW files like:
      run_xxx_trans.raw
      run_xxx_trans.op.raw
    Choose the sibling whose header matches the desired plot.
    """
    folder = raw_path.parent
    stem = raw_path.stem
    stem = stem[:-3] if stem.endswith(".op") else stem
    candidates = []
    # Prefer same-stem siblings first
    candidates.extend(sorted(folder.glob(stem + "*.raw")))
    # Include the originally returned path even if glob missed it
    if raw_path not in candidates:
        candidates.insert(0, raw_path)

    # First pass: exact plotname match
    for p in candidates:
        if not p.exists():
            continue
        try:
            plotname, has_time, _ = classify_raw_header(p)
        except Exception:
            continue
        if plotname == want_plotname:
            if want_plotname != "Transient Analysis" or has_time:
                return p

    # Second pass: prefer any RAW with time for transient, otherwise original
    if want_plotname == "Transient Analysis":
        for p in candidates:
            if not p.exists():
                continue
            try:
                _, has_time, _ = classify_raw_header(p)
            except Exception:
                continue
            if has_time:
                return p

    return raw_path


# -----------------------------------------------------------------------------
# Netlist synthesis
# -----------------------------------------------------------------------------

def _with_vss_offset(value: float, cand: Candidate, is_supply: bool = False) -> float:
    # Approximate a negative local ground by shifting ground-referenced bias sources.
    # Example: if local ground is -0.3 V and desired Vth is +0.8 V above that local
    # ground, then the absolute source value is about +0.5 V versus the simulator 0 node.
    return value + cand.vss_offset

def add_optional_bias_overrides(text: str, cand: Candidate) -> str:
    # These edits are best-effort. If the source is absent in the base netlist, they are skipped.
    text2 = text
    overrides = (
        ("Vdd", _with_vss_offset(cand.vdd, cand, is_supply=True)),
        ("VDD", _with_vss_offset(cand.vdd, cand, is_supply=True)),
        ("Vc", _with_vss_offset(cand.vc, cand)),
        ("VTH", _with_vss_offset(cand.vth, cand)),
        ("Vth", _with_vss_offset(cand.vth, cand)),
        ("Vbias", _with_vss_offset(cand.vbias, cand)),
        ("VBIAS", _with_vss_offset(cand.vbias, cand)),
        ("Vd", _with_vss_offset(1.7, cand)),
        ("VD", _with_vss_offset(1.7, cand)),
    )
    for ref, value in overrides:
        newer = replace_source_dc_value(text2, ref, value)
        text2 = newer
    return text2


def append_topology_lines(cand: Candidate, mode: str) -> list[str]:
    lines: list[str] = []

    # Common input node for shaped-input topologies.
    if cand.topology == "base":
        if mode == "phase":
            lines.append(f"IIN 0 {NODE_V} DC {eng(cand.I_bias)}")
        else:
            lines.append(
                "IIN 0 {node} PWL(0 0 {t1} 0 {t2} {ia} {t3} 0 {t4} 0 {t5} {idc} {t6} 0)".format(
                    node=NODE_V,
                    t1=0.010, t2=0.010001, ia=eng(cand.I_activate),
                    t3=0.020, t4=0.060, t5=0.060001, idc=eng(cand.I_deactivate), t6=0.070
                )
            )

    elif cand.topology == "cap_feedback":
        lines.append(f"CFB {NODE_V} N003 100n")
        if mode == "phase":
            lines.append(f"IIN 0 {NODE_V} DC {eng(cand.I_bias)}")
        else:
            lines.append(
                "IIN 0 {node} PWL(0 0 0.010 0 0.010001 {ia} 0.020 0 0.060 0 0.060001 {idc} 0.070 0)".format(
                    node=NODE_V, ia=eng(cand.I_activate), idc=eng(cand.I_deactivate)
                )
            )

    elif cand.topology == "res_feedback":
        lines.append(f"RFB {NODE_V} N003 100k")
        if mode == "phase":
            lines.append(f"IIN 0 {NODE_V} DC {eng(cand.I_bias)}")
        else:
            lines.append(
                "IIN 0 {node} PWL(0 0 0.010 0 0.010001 {ia} 0.020 0 0.060 0 0.060001 {idc} 0.070 0)".format(
                    node=NODE_V, ia=eng(cand.I_activate), idc=eng(cand.I_deactivate)
                )
            )

    elif cand.topology == "rc_input":
        lines += [
            f"RINJ NIN {NODE_V} 10k",
            f"CINJ NIN 0 100n",
        ]
        if mode == "phase":
            lines.append(f"IIN 0 NIN DC {eng(cand.I_bias)}")
        else:
            lines.append(
                "IIN 0 NIN PWL(0 0 0.010 0 0.010001 {ia} 0.020 0 0.060 0 0.060001 {idc} 0.070 0)".format(
                    ia=eng(cand.I_activate), idc=eng(cand.I_deactivate)
                )
            )

    elif cand.topology == "rlc_input":
        lines += [
            f"LINJ NDRV NIN 10m",
            f"RINJ NIN {NODE_V} 3.3k",
            f"CINJ NIN 0 100n",
            f"RDRV NDRV 0 100",
        ]
        if mode == "phase":
            lines.append(f"IIN 0 NDRV DC {eng(cand.I_bias)}")
        else:
            lines.append(
                "IIN 0 NDRV PWL(0 0 0.010 0 0.010001 {ia} 0.020 0 0.060 0 0.060001 {idc} 0.070 0)".format(
                    ia=eng(cand.I_activate), idc=eng(cand.I_deactivate)
                )
            )
    else:
        raise ValueError(f"Unknown topology: {cand.topology}")

    return lines


def build_phase_netlist(base_text: str, cand: Candidate) -> str:
    text = filter_netlist_lines(base_text, PHASE_REMOVE_REFS + [CV_REF, CU_REF], remove_analyses=True)
    text = replace_component_value_line(text, CV_REF, eng(cand.Cv))
    text = replace_component_value_line(text, CU_REF, eng(cand.Cu))
    text = add_optional_bias_overrides(text, cand)

    before_end, end_block = split_before_end(text)
    vv0 = PHASE_V_RANGE[0] + cand.vss_offset
    vv1 = PHASE_V_RANGE[1] + cand.vss_offset
    uu0 = PHASE_U_RANGE[0] + cand.vss_offset
    uu1 = PHASE_U_RANGE[1] + cand.vss_offset

    extra = [
        f"V_VSTATE {NODE_V} 0 {{VV}}",
        f"V_USTATE {NODE_U} 0 {{UU}}",
        f".param VV={vv0}",
        f".param UU={uu0}",
        *append_topology_lines(cand, mode="phase"),
        f".step param UU {uu0} {uu1} {PHASE_U_RANGE[2]}",
        f".dc V_VSTATE {vv0} {vv1} {PHASE_V_RANGE[2]}",
        ".op",
        ".save V(*) I(V_VSTATE) I(V_USTATE)",
    ]
    return before_end.rstrip() + "\n\n" + "\n".join(extra) + "\n\n" + end_block



def build_transient_netlist(base_text: str, cand: Candidate) -> str:
    text = filter_netlist_lines(base_text, TRANSIENT_INPUT_REMOVE_REFS, remove_analyses=True)
    text = replace_component_value_line(text, CV_REF, eng(cand.Cv))
    text = replace_component_value_line(text, CU_REF, eng(cand.Cu))
    text = add_optional_bias_overrides(text, cand)

    before_end, end_block = split_before_end(text)
    extra = [
        *append_topology_lines(cand, mode="transient"),
        f".tran 0 {TRANSIENT_TSTOP} 0 10u",
        f".save V({NODE_V}) V({NODE_U}) I(*)",
    ]
    return before_end.rstrip() + "\n\n" + "\n".join(extra) + "\n\n" + end_block


# -----------------------------------------------------------------------------
# Simulation wrappers
# -----------------------------------------------------------------------------

class SearchEngine:
    def __init__(self, base_netlist: Path, output_dir: Path):
        self.base_netlist = base_netlist
        self.output_dir = output_dir
        self.base_text = read_text(base_netlist)
        self.runner = SimRunner(output_folder=str(output_dir / "runs"), simulator=SIMULATOR, parallel_sims=MAX_PARALLEL, verbose=False)

    def run_netlist_text(self, netlist_text: str, out_name: str) -> tuple[Path, Path, Path | None]:
        netlist_path = self.output_dir / "netlists" / f"{out_name}.cir"
        write_text(netlist_path, netlist_text)
        raw_file, log_file = self.runner.run_now(str(netlist_path), run_filename=netlist_path.name)
        if raw_file is None:
            raise RuntimeError(f"LTspice simulation failed or produced no RAW file for {netlist_path}. LOG: {log_file}")
        return netlist_path, Path(raw_file), (Path(log_file) if log_file is not None else None)

    def run_phase(self, cand: Candidate, name: str) -> tuple[dict, Path, Path]:
        net = build_phase_netlist(self.base_text, cand)
        net_path, raw_path, log_path = self.run_netlist_text(net, name + "_phase")
        raw_path = resolve_preferred_raw(raw_path, "DC transfer characteristic")
        data = analyze_phase(raw_path, cand, plot_prefix=self.output_dir / "plots" / (name + "_phase"))
        data["netlist_path"] = str(net_path)
        data["raw_path"] = str(raw_path)
        data["log_path"] = str(log_path) if log_path is not None else ""
        return data, net_path, raw_path

    def run_transient(self, cand: Candidate, name: str) -> tuple[dict, Path, Path]:
        net = build_transient_netlist(self.base_text, cand)
        net_path, raw_path, log_path = self.run_netlist_text(net, name + "_trans")
        raw_path = resolve_preferred_raw(raw_path, "Transient Analysis")
        data = analyze_transient(raw_path, plot_prefix=self.output_dir / "plots" / (name + "_trans"))
        data["netlist_path"] = str(net_path)
        data["raw_path"] = str(raw_path)
        data["log_path"] = str(log_path) if log_path is not None else ""
        return data, net_path, raw_path


# -----------------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------------


def trace_or_none(raw: RawRead, name: str):
    try:
        return raw.get_trace(name)
    except Exception:
        return None


def analyze_phase(raw_path: Path, cand: Candidate, plot_prefix: Path) -> dict:
    raw = RawRead(str(raw_path))
    
    # RawRead exposes get_trace_names(), get_trace(), get_wave(trace, step), and get_steps().
    # For stepped simulations, get_steps() uses the companion .log file to map the .STEP values.
    steps = raw.get_steps()
    v_tr = raw.get_trace(f"V({NODE_V})")
    u_tr = raw.get_trace(f"V({NODE_U})")
    iv_tr = raw.get_trace("I(V_VSTATE)")
    iu_tr = raw.get_trace("I(V_USTATE)")

    all_v = []
    all_u = []
    all_dv = []
    all_du = []

    for step in steps:
        v = np.asarray(v_tr.get_wave(step), dtype=float)
        u = np.asarray(u_tr.get_wave(step), dtype=float)
        iv = np.asarray(iv_tr.get_wave(step), dtype=float)
        iu = np.asarray(iu_tr.get_wave(step), dtype=float)
        dv = -iv / cand.Cv
        du = -iu / cand.Cu
        all_v.append(v)
        all_u.append(u)
        all_dv.append(dv)
        all_du.append(du)

    V = np.vstack(all_v)
    U = np.vstack(all_u)
    dV = np.vstack(all_dv)
    dU = np.vstack(all_du)

    mag = np.sqrt(dV * dV + dU * dU)
    eps = 1e-30
    dVn = dV / (mag + eps)
    dUn = dU / (mag + eps)

    # Approximate fixed points: simultaneously small derivatives.
    dv_thr = np.nanpercentile(np.abs(dV), 7)
    du_thr = np.nanpercentile(np.abs(dU), 7)
    fp_mask = (np.abs(dV) <= dv_thr) & (np.abs(dU) <= du_thr)

    fp_indices = np.argwhere(fp_mask)
    fixed_points = []
    taken = np.zeros(len(fp_indices), dtype=bool)
    for i, (r, c) in enumerate(fp_indices):
        if taken[i]:
            continue
        group = [(r, c)]
        taken[i] = True
        for j in range(i + 1, len(fp_indices)):
            if taken[j]:
                continue
            rr, cc = fp_indices[j]
            if abs(rr - r) <= 1 and abs(cc - c) <= 1:
                group.append((rr, cc))
                taken[j] = True
        gr = np.mean([g[0] for g in group])
        gc = np.mean([g[1] for g in group])
        fixed_points.append((gr, gc))

    # Jacobian via finite differences on the grid.
    dVdU, dVdV = np.gradient(dV)  # row, col
    dUdU, dUdV = np.gradient(dU)

    stable_count = 0
    for gr, gc in fixed_points:
        r = int(round(gr))
        c = int(round(gc))
        J = np.array([[dVdV[r, c], dVdU[r, c]], [dUdV[r, c], dUdU[r, c]]], dtype=float)
        eig = np.linalg.eigvals(J)
        if np.all(np.real(eig) < 0):
            stable_count += 1

    phase_score = float(2.0 * stable_count + 0.5 * len(fixed_points))

    # Save quiver + approximate nullclines.
    plot_prefix.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.quiver(V, U, dVn, dUn, angles='xy', scale_units='xy', scale=18)
    plt.scatter(V[fp_mask], U[fp_mask], s=8)
    plt.xlabel(f"V = membrane voltage at {NODE_V}")
    plt.ylabel(f"U = slow variable at {NODE_U}")
    plt.title("Phase portrait")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(plot_prefix) + "_quiver.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(V, U, np.log10(mag + eps), shading='auto')
    plt.colorbar(label="log10(|dX/dt|)")
    plt.xlabel(f"V = membrane voltage at {NODE_V}")
    plt.ylabel(f"U = slow variable at {NODE_U}")
    plt.title("Phase portrait vector-field magnitude")
    plt.tight_layout()
    plt.savefig(str(plot_prefix) + "_logmag.png", dpi=180)
    plt.close()

    return {
        "ok": True,
        "fixed_point_count": len(fixed_points),
        "stable_point_count": stable_count,
        "phase_score": phase_score,
        "plot_quiver": str(plot_prefix) + "_quiver.png",
        "plot_logmag": str(plot_prefix) + "_logmag.png",
    }


def _find_trace_name_case_insensitive(names, target):
    target_l = target.lower()
    for name in names:
        if name.lower() == target_l:
            return name
    return None


def _pick_plot_with_axis_and_trace(raw: RawRead, trace_name: str):
    """
    Return (plot, actual_trace_name) for the first plot that has an axis
    and contains trace_name, case-insensitively.
    """
    # First try default plot
    try:
        names = raw.get_trace_names()
        _ = raw.get_axis()
        actual = _find_trace_name_case_insensitive(names, trace_name)
        if actual is not None:
            return raw, actual
    except Exception:
        pass

    # Then try all plots
    if hasattr(raw, "plots"):
        for plot in raw.plots:
            try:
                names = plot.get_trace_names()
                _ = plot.get_axis()
                actual = _find_trace_name_case_insensitive(names, trace_name)
                if actual is not None:
                    return plot, actual
            except Exception:
                continue

    raise RuntimeError(f"No plot with axis and trace {trace_name} was found in RAW file.")

def analyze_transient(raw_path: Path, plot_prefix: Path) -> dict:
    raw_path = resolve_preferred_raw(raw_path, "Transient Analysis")
    raw = RawRead(str(raw_path))

    names = list(raw.get_trace_names())
    v_trace_name = _find_trace_name_case_insensitive(names, f"V({NODE_V})")
    u_trace_name = _find_trace_name_case_insensitive(names, f"V({NODE_U})")
    time_trace_name = _find_trace_name_case_insensitive(names, "time")

    if v_trace_name is None:
        plot, v_trace_name = _pick_plot_with_axis_and_trace(raw, f"V({NODE_V})")
        names = list(plot.get_trace_names())
        u_trace_name = _find_trace_name_case_insensitive(names, f"V({NODE_U})")
        time_trace_name = _find_trace_name_case_insensitive(names, "time")
    else:
        plot = raw

    try:
        t = np.asarray(plot.get_axis(), dtype=float)
    except Exception:
        if time_trace_name is None:
            raise RuntimeError(
                f"{raw_path.name}: no axis and no 'time' trace found; traces={names[:12]}"
            )
        t = np.asarray(plot.get_trace(time_trace_name).get_wave(), dtype=float)

    v = np.asarray(plot.get_trace(v_trace_name).get_wave(), dtype=float)

    u = None
    if u_trace_name is not None:
        u = np.asarray(plot.get_trace(u_trace_name).get_wave(), dtype=float)

    # Window layout:
    # 0.010-0.020 s : first ("turn-on") pulse
    # 0.022-0.058 s : coast window
    # 0.060-0.070 s : second ("turn-off" / perturbation) pulse
    # 0.075-0.115 s : final window
    baseline_mask = (t >= 0.0) & (t <= 0.010)
    v_base = float(np.median(v[baseline_mask])) if np.any(baseline_mask) else float(np.median(v))
    v_hi = float(np.max(v))
    v_lo = float(np.min(v))

    # Adaptive spike threshold on the membrane node.
    threshold = max(v_base + 0.25 * (v_hi - v_base), v_base + 0.15)

    evoked_spikes = local_maxima_count(t, v, threshold, 0.010, 0.020)
    coast_spikes = local_maxima_count(t, v, threshold, 0.022, 0.058)
    off_pulse_spikes = local_maxima_count(t, v, threshold, 0.060, 0.070)
    final_spikes = local_maxima_count(t, v, threshold, 0.075, 0.115)
    baseline_spikes = local_maxima_count(t, v, threshold, 0.000, 0.010)

    latches_on = coast_spikes >= 3
    turns_off = final_spikes <= 1

    # Richer bistability-oriented score:
    # - reward evoked spiking and especially coast spiking
    # - reward successful shutoff after the second pulse, regardless of pulse polarity
    # - penalize spontaneous baseline spiking and failure to shut off
    bistability_score = 0.0
    if baseline_spikes > 0:
        bistability_score -= 3.0

    if evoked_spikes >= 1:
        bistability_score += 1.0
    if coast_spikes >= 6:
        bistability_score += 8.0
    elif coast_spikes >= 3:
        bistability_score += 6.0
    elif coast_spikes >= 1:
        bistability_score += 2.0

    if latches_on and turns_off:
        bistability_score += 8.0
    elif latches_on and not turns_off:
        bistability_score -= 2.0
    elif (not latches_on) and turns_off:
        bistability_score += 0.5

    # Small bonus when second pulse is stronger in magnitude than first:
    # this captures the "small pulse turns on, larger pulse turns off" pattern.
    if abs(v_hi - v_base) > 0 and turns_off and latches_on:
        bistability_score += 1.0

    plot_prefix.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.plot(t, v, label='V')
    if u is not None:
        plt.plot(t, u, label='U')
    plt.axvspan(0.010, 0.020, alpha=0.15, label='on pulse')
    plt.axvspan(0.060, 0.070, alpha=0.15, label='off/perturb pulse')
    plt.axhline(threshold, linestyle='--', label='spike threshold')
    plt.xlabel("t = time (s)")
    plt.ylabel("voltage (V)")
    plt.title("Transient bistability test")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(plot_prefix) + "_transient.png", dpi=180)
    plt.close()

    return {
        "ok": True,
        "evoked_spikes": evoked_spikes,
        "coast_spikes": coast_spikes,
        "off_pulse_spikes": off_pulse_spikes,
        "final_spikes": final_spikes,
        "spike_threshold": threshold,
        "bistability_score": bistability_score,
        "plot_transient": str(plot_prefix) + "_transient.png",
    }

# -----------------------------------------------------------------------------
# Search loop
# -----------------------------------------------------------------------------


def candidate_iterator() -> Iterable[Candidate]:
    key_order = [k for k in SEARCH_KEY_ORDER if k in SEARCH_SPACE]
    missing = [k for k in SEARCH_SPACE.keys() if k not in key_order]
    key_order.extend(missing)

    candidates = [
        Candidate(**dict(zip(key_order, values)))
        for values in product(*(SEARCH_SPACE[k] for k in key_order))
    ]

    if RANDOMIZE_CANDIDATES:
        import random
        rng = random.Random(RANDOMIZE_SEED)
        rng.shuffle(candidates)

    yield from candidates



def run_search(limit: int | None = None) -> list[RunResult]:
    engine = SearchEngine(BASE_NETLIST, OUTPUT_DIR)
    results: list[RunResult] = []

    for idx, cand in enumerate(candidate_iterator(), start=1):
        if limit is not None and idx > limit:
            break

        name = sanitize_name(
            f"run_{idx:04d}_{cand.topology}_Cv_{eng(cand.Cv)}_Cu_{eng(cand.Cu)}_Vss_{eng(cand.vss_offset)}_Ib_{eng(cand.I_bias)}_Ia_{eng(cand.I_activate)}_Id_{eng(cand.I_deactivate)}"
        )
        result = RunResult(name=name, candidate=cand)

        try:
            phase_data, phase_net, phase_raw = engine.run_phase(cand, name)
            result.phase_ok = bool(phase_data["ok"])
            result.fixed_point_count = int(phase_data["fixed_point_count"])
            result.stable_point_count = int(phase_data["stable_point_count"])
            result.phase_score = float(phase_data["phase_score"])
            result.phase_netlist = str(phase_net)
            result.phase_raw = str(phase_raw)
        except Exception as e:
            result.notes += f"phase failed: {e}; "

        try:
            trans_data, trans_net, trans_raw = engine.run_transient(cand, name)
            result.transient_ok = bool(trans_data["ok"])
            result.evoked_spikes = int(trans_data["evoked_spikes"])
            result.coast_spikes = int(trans_data["coast_spikes"])
            result.off_pulse_spikes = int(trans_data["off_pulse_spikes"])
            result.final_spikes = int(trans_data["final_spikes"])
            result.spike_threshold = float(trans_data["spike_threshold"])
            result.bistability_score = float(trans_data["bistability_score"])
            result.trans_netlist = str(trans_net)
            result.trans_raw = str(trans_raw)
        except Exception as e:
            result.notes += f"transient failed: {e}; "

        result.total_score = result.phase_score + result.bistability_score
        results.append(result)
        print(f"[{idx}] {name}: total_score={result.total_score:.2f} evoked={result.evoked_spikes} coast={result.coast_spikes} off={result.off_pulse_spikes} final={result.final_spikes}")

    results.sort(key=lambda r: r.total_score, reverse=True)
    write_results_csv(results, OUTPUT_DIR / "results_summary.csv")
    return results



def write_results_csv(results: Sequence[RunResult], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "rank", "name", "topology", "Cv_F", "Cu_F", "I_bias_A", "I_activate_A", "I_deactivate_A",
            "vdd_V", "vss_offset_V", "vc_V", "vth_V", "vbias_V",
            "phase_ok", "transient_ok", "fixed_point_count", "stable_point_count",
            "evoked_spikes", "coast_spikes", "off_pulse_spikes", "final_spikes", "spike_threshold_V",
            "phase_score", "bistability_score", "total_score",
            "phase_netlist", "trans_netlist", "phase_raw", "trans_raw", "notes",
        ])
        for rank, r in enumerate(results, start=1):
            c = r.candidate
            w.writerow([
                rank, r.name, c.topology, c.Cv, c.Cu, c.I_bias, c.I_activate, c.I_deactivate,
                c.vdd, c.vss_offset, c.vc, c.vth, c.vbias,
                int(r.phase_ok), int(r.transient_ok), r.fixed_point_count, r.stable_point_count,
                r.evoked_spikes, r.coast_spikes, r.off_pulse_spikes, r.final_spikes, r.spike_threshold,
                r.phase_score, r.bistability_score, r.total_score,
                r.phase_netlist, r.trans_netlist, r.phase_raw, r.trans_raw, r.notes,
            ])


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    if not BASE_NETLIST.exists():
        raise FileNotFoundError(f"Base netlist not found: {BASE_NETLIST}")

    # Start with a small limit first; remove the limit once the first few runs behave correctly.
    # Example: results = run_search(limit=8)
    print(f"SEARCH_KEY_ORDER={SEARCH_KEY_ORDER}")
    results = run_search(limit=8)

    print("\nTop results:")
    for r in results[:5]:
        print(
            f"{r.name}: total={r.total_score:.2f}, fixed={r.fixed_point_count}, stable={r.stable_point_count}, "
            f"evoked={r.evoked_spikes}, coast={r.coast_spikes}, off_pulse={r.off_pulse_spikes}, final={r.final_spikes}, "
            f"thr={r.spike_threshold:.3f}, notes={r.notes}"
        )

    print(f"\nWrote summary to: {OUTPUT_DIR / 'results_summary.csv'}")
