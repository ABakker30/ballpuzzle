# tools/solve.py — FCC tetra-spheres solver runner (CLI)
# Runs the core search engine and writes ONLY the final solution
# as:
#   1) JSON (world coordinates per piece) — format matching example `solution_0026.world.json`
#   2) TXT  (world FCC layer printout with piece letters)
#
# Usage (PowerShell):
#   cd "C:\\Grasshopper\\Projects\\Projects\\2025\\ball puzzle\\solver"
#   $env:PYTHONPATH = (Get-Location)
#   python tools\\solve.py `
#     --container "data\\containers\\Roof.json" `
#     --pieces "data\\pieces\\pieces.py" `
#     --out "out\\Roof_run2" `
#     --present auto `
#     --status-interval 5 `
#     --max-seconds 0 `
#     --max-attempts 0 `
#     --branch-cap-open 18 `
#     --branch-cap-tight 8 `
#     --deg2-corridor 1 `
#     --roulette least-tried `
#     --tt-max 6000000 `
#     --tt-trim-keep 5000000
#
# Notes:
# - We do NOT write per-depth snapshots anymore.
# - We do NOT depend on Streamlit.
# - Advanced knobs are applied to the engine post-construction when available (hasattr checks).
# - Status line is printed every --status-interval seconds with a rolling attempts/sec.
# - The rate shown is computed from the engine's .attempts counter delta over the interval.

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import hashlib
from collections import defaultdict
import math
import importlib.util

# --- Robust import of core.solver_engine ---------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from core.solver_engine import SolverEngine
except Exception as e:
    print("[ERR] Could not import core.solver_engine.SolverEngine:", e)
    raise

# --- FCC helpers ---------------------------------------------------------------------------------
def fcc_point(i:int, j:int, k:int, r:float):
    """Map FCC lattice (i,j,k) -> world xyz where spheres of radius r just touch."""
    s = math.sqrt(2.0) * float(r)
    return (s * (j + k), s * (i + k), s * (i + j))

# --- Container JSON loader -----------------------------------------------------------------------
def load_container_json(path:str):
    """
    Expected container JSON (minimal):
    {
      "version": 1,
      "name": "Roof",
      "lattice": "fcc",
      "r": 1.0,
      "cells": [[i,j,k], ...]
    }
    Optional: "presentation" (ignored by solver; passed through in final JSON)
    Returns: (valid_set, r, name, presentation_dict_or_None)
    """
    with open(path, "r") as f:
        obj = json.load(f)
    if obj.get("lattice", "fcc").lower() != "fcc":
        raise ValueError("Only 'fcc' lattice is supported")

    name = obj.get("name") or os.path.splitext(os.path.basename(path))[0]
    r = float(obj.get("r", 1.0))
    cells = obj.get("cells") or obj.get("container") or obj.get("valid_cells")
    if not cells:
        raise ValueError("Container JSON missing 'cells' array")

    valid_set = set()
    for t in cells:
        if not (isinstance(t, (list, tuple)) and len(t) == 3):
            continue
        i, j, k = int(t[0]), int(t[1]), int(t[2])
        valid_set.add((i, j, k))

    pres = obj.get("presentation")  # optional, carried through to JSON if present
    return valid_set, r, name, pres

# --- Pieces loader (.py with PIECES dict) --------------------------------------------------------
def load_pieces(py_path:str):
    py_path = os.path.normpath(py_path)
    if not os.path.isfile(py_path):
        raise IOError("File not found: {}".format(py_path))
    modname = "__pieces__"
    if modname in sys.modules:
        del sys.modules[modname]
    spec = importlib.util.spec_from_file_location(modname, py_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    if not hasattr(mod, "PIECES"):
        raise ValueError("pieces module missing PIECES dict")
    # Normalize to {str -> tuple[tuple[(dx,dy,dz),...], ...]}
    out = {}
    for k, oris in mod.PIECES.items():
        key = str(k)
        norm_oris = []
        for ori in oris:
            norm_oris.append(tuple((int(a), int(b), int(c)) for (a,b,c) in ori))
        out[key] = tuple(norm_oris)
    return out

# --- World layer printer (letters) ---------------------------------------------------------------
def format_world_layers(valid_set:set[tuple[int,int,int]], letter_at:dict, empty_char:str="."):
    """
    World view: u=x=j+k (cols), v=y=i+k (rows), w=z=i+j (layers)
    Only prints cells in valid_set; fills with piece letter or empty_char.
    """
    # Bounds in (u,v,w)
    umin = vmin = wmin = +10**9
    umax = vmax = wmax = -10**9
    uvw_to_ijk = {}
    for (i,j,k) in valid_set:
        u = j + k
        v = i + k
        w = i + j
        if u < umin: umin = u
        if u > umax: umax = u
        if v < vmin: vmin = v
        if v > vmax: vmax = v
        if w < wmin: wmin = w
        if w > wmax: wmax = w
        uvw_to_ijk[(u,v,w)] = (i,j,k)

    lines = []
    lines.append("[SOLUTION — world FCC view]")
    lines.append("Legend: rows=y (i+k: {}..{}), cols=x (j+k: {}..{}), layers=z (i+j: {}..{})"
                 .format(vmin, vmax, umin, umax, wmin, wmax))
    lines.append("")
    for w in range(wmin, wmax + 1):
        lines.append("Layer z=i+j={}:".format(w))
        for v in range(vmin, vmax + 1):
            row = []
            for u in range(umin, umax + 1):
                ijk = uvw_to_ijk.get((u, v, w))
                if ijk is None:
                    row.append(" ")
                else:
                    row.append(letter_at.get(ijk, empty_char))
            lines.append(" ".join(row).rstrip())
        lines.append("")
    return "\n".join(lines)

# --- Final JSON writer (match example structure) -------------------------------------------------
def container_hash(valid_set:set[tuple[int,int,int]]):
    s = json.dumps(sorted(list(valid_set)), separators=(",",":"))
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def placements_to_json(eng, r:float, container_name:str, presentation):
    """
    Build JSON object matching the example:
    {
      "version": 1,
      "container_name": "...",
      "lattice": "fcc",
      "r": 1.0,
      "container_hash": "...",
      "pieces": [
        {"id":"A","world_centers":[[x,y,z], ...]},
        ...
      ],
      "presentation": {...}  # optional passthrough if provided
    }
    """
    piece_cells = defaultdict(list)
    for pl in getattr(eng, "placements", []):
        pid = str(pl.get("piece") or pl.get("id"))
        if "cells" in pl:
            ijks = [tuple(map(int, c)) for c in pl["cells"]]
        elif "cells_idx" in pl and hasattr(eng, "idx2cell"):
            ijks = [tuple(map(int, eng.idx2cell[idx])) for idx in pl["cells_idx"]]
        else:
            continue
        for (i,j,k) in ijks:
            piece_cells[pid].append((i,j,k))

    pieces_out = []
    for pid in sorted(piece_cells.keys()):
        centers = [fcc_point(i,j,k,r) for (i,j,k) in piece_cells[pid]]
        pieces_out.append({"id": pid, "world_centers": centers})

    obj = {
        "version": 1,
        "container_name": str(container_name),
        "lattice": "fcc",
        "r": float(r),
        "container_hash": container_hash(getattr(eng, "valid_set", set())),
        "pieces": pieces_out,
    }
    if presentation is not None:
        obj["presentation"] = presentation
    return obj

# --- Status printing -----------------------------------------------------------------------------
def _fmt_int(n:int): 
    try:
        return "{:,}".format(int(n))
    except Exception:
        return str(n)

def print_status(eng, t0, last_attempts, last_time):
    now = time.time()
    elapsed = now - t0
    attempts = getattr(eng, "attempts", 0)
    d_attempts = max(0, attempts - last_attempts)
    dt = max(1e-6, now - last_time)
    rate = d_attempts / dt

    placed = len(getattr(eng, "placements", []))
    total = len(getattr(eng, "order", ()))
    best_ever = getattr(eng, "best_depth_ever", None)
    best = best_ever if isinstance(best_ever, int) and best_ever >= placed else placed

    print(f"[rev13.2] Placed {placed} / {total} pieces ({(100.0*placed/total if total else 0.0):.1f}%) | "
          f"Best depth (ever): {best} | Rate: {_fmt_int(rate)} attempts/s | elapsed: {elapsed:.1f}s")

    tt_size = len(getattr(eng, "TT", {})) if hasattr(eng, "TT") else 0
    tt_hits = getattr(eng, "tt_hits", 0)
    tt_prunes = getattr(eng, "tt_prunes", 0)
    forced = getattr(eng, "forced_singletons", 0)
    in_corridor = getattr(eng, "in_corridor", False)
    deg2 = getattr(eng, "deg2_corridor", False)
    cap = getattr(eng, "branch_cap_cur", None)
    roulette = getattr(eng, "roulette_cur", "none")

    anchor_seen = getattr(eng, "anchor_seen", set())
    transitions = getattr(eng, "transitions", {})
    top_trans = "n/a"
    if transitions:
        (a,b), cnt = max(transitions.items(), key=lambda kv: kv[1])
        top_trans = f"{a}→{b}: {cnt}"

    print(f"   anchors: {len(anchor_seen)} unique | transitions: {len(transitions)} unique (top {top_trans}) | "
          f"TT: size={_fmt_int(tt_size)} hits={_fmt_int(tt_hits)} prunes={_fmt_int(tt_prunes)} | forced-singletons={_fmt_int(forced)}")
    print(f"   search mode: {roulette} | branch cap: {cap} (corridor={in_corridor} deg2={deg2})")

    def head_hist(d:dict, label:str, limit:int=12):
        if not d:
            return
        keys = sorted(d.keys())
        parts = []
        for k in keys[:limit]:
            parts.append(f"{k}: {_fmt_int(d[k])}")
        print(f"   {label}: " + ", ".join(parts))

    head_hist(getattr(eng, "stat_choices_hist", {}), "choices@depth histogram (post-cap)")
    head_hist(getattr(eng, "stat_exposure_hist", {}), "exposure buckets (low=better)")
    head_hist(getattr(eng, "stat_boundary_exposure_hist", {}), "boundary-exposure (low=better)")
    head_hist(getattr(eng, "stat_leaf_hist", {}), "leaf-empties (lower=better)")

    if hasattr(eng, "stat_anchor_deg_hist"):
        deg_parts = []
        for k in sorted(eng.stat_anchor_deg_hist.keys()):
            deg_parts.append(f"{k}: {_fmt_int(eng.stat_anchor_deg_hist[k])}")
        if deg_parts:
            print("   anchor empty-neighbor degree: " + " | ".join(deg_parts))

    if hasattr(eng, "stat_fallback_piece") and eng.stat_fallback_piece:
        top_fb = sorted(eng.stat_fallback_piece.items(), key=lambda kv: -kv[1])[:8]
        head = ", ".join(f"{k}: {_fmt_int(v)}" for k, v in top_fb)
        print("   pieces needing fallback origins most: " + head)

    return attempts, now

# --- Main ----------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", required=True, help="Path to container JSON")
    ap.add_argument("--pieces", required=True, help="Path to pieces.py")
    ap.add_argument("--out", required=True, help="Output folder (created if not exists)")
    ap.add_argument("--present", choices=["auto","none"], default="auto", help="(compat only; not used)")
    ap.add_argument("--status-interval", type=float, default=5.0, help="Seconds between status prints (0 disables)")
    ap.add_argument("--max-seconds", type=float, default=0.0, help="Stop after this many seconds (0 = no limit)")
    ap.add_argument("--max-attempts", type=int, default=0, help="Stop after this many attempts (0 = no limit)")
    # optional tuning knobs (applied if engine exposes attributes)
    ap.add_argument("--branch-cap-open", type=int, default=None)
    ap.add_argument("--branch-cap-tight", type=int, default=None)
    ap.add_argument("--deg2-corridor", type=int, choices=[0,1], default=None)
    ap.add_argument("--roulette", choices=["least-tried","none"], default=None)
    ap.add_argument("--tt-max", type=int, default=None)
    ap.add_argument("--tt-trim-keep", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    valid_set, r, container_name, presentation = load_container_json(args.container)
    pieces = load_pieces(args.pieces)

    # Build engine
    eng = SolverEngine(pieces=pieces, valid_set=valid_set)

    # Apply knobs if available
    applied = []
    if args.branch_cap_open is not None and hasattr(eng, "BRANCH_CAP_OPEN"):
        eng.BRANCH_CAP_OPEN = int(args.branch_cap_open); applied.append(f"branch_cap_open={eng.BRANCH_CAP_OPEN}")
    if args.branch_cap_tight is not None and hasattr(eng, "BRANCH_CAP_TIGHT"):
        eng.BRANCH_CAP_TIGHT = int(args.branch_cap_tight); applied.append(f"branch_cap_tight={eng.BRANCH_CAP_TIGHT}")
    if args.deg2_corridor is not None and hasattr(eng, "deg2_corridor"):
        eng.deg2_corridor = bool(args.deg2_corridor); applied.append(f"deg2_corridor={eng.deg2_corridor}")
    if args.roulette is not None and hasattr(eng, "ROULETTE_MODE"):
        eng.ROULETTE_MODE = args.roulette; applied.append(f"roulette={eng.ROULETTE_MODE}")
    if args.tt_max is not None and hasattr(eng, "TT_MAX"):
        eng.TT_MAX = int(args.tt_max); applied.append(f"tt_max={eng.TT_MAX}")
    if args.tt_trim_keep is not None and hasattr(eng, "TT_TRIM_KEEP"):
        eng.TT_TRIM_KEEP = int(args.tt_trim_keep); applied.append(f"tt_trim_keep={eng.TT_TRIM_KEEP}")
    if applied:
        print("[tuning] " + ", ".join(applied))

    # Run loop
    start = time.time()
    last_status = start
    last_attempts = getattr(eng, "attempts", 0)
    last_time = start

    solved = False
    while True:
        progressed, solved = eng.step_once()
        if solved:
            break
        if args.max_seconds and (time.time() - start) >= args.max_seconds:
            break
        if args.max_attempts and getattr(eng, "attempts", 0) >= args.max_attempts:
            break
        if args.status_interval > 0.0:
            now = time.time()
            if now - last_status >= args.status_interval:
                last_attempts, last_time = print_status(eng, start, last_attempts, last_time)
                last_status = now

    # Final status and write
    print_status(eng, start, last_attempts, last_time)

    # Build letter map and write world-layers TXT
    letter_at = {}
    for pl in getattr(eng, "placements", []):
        pid = str(pl.get("piece") or pl.get("id"))
        if "cells" in pl:
            ijks = [tuple(map(int, c)) for c in pl["cells"]]
        elif "cells_idx" in pl and hasattr(eng, "idx2cell"):
            ijks = [tuple(map(int, eng.idx2cell[idx])) for idx in pl["cells_idx"]]
        else:
            ijks = []
        for c in ijks:
            letter_at[c] = pid

    txt = format_world_layers(valid_set, letter_at, empty_char=".")
    txt_path = os.path.join(args.out, f"{container_name}_solution.world_layers.txt")
    with open(txt_path, "w") as f:
        f.write(txt)

    # JSON (match example)
    sol_obj = placements_to_json(eng, r=r, container_name=container_name, presentation=presentation)
    json_path = os.path.join(args.out, f"{container_name}_solution.world.json")
    with open(json_path, "w") as f:
        json.dump(sol_obj, f, indent=2)

    print("[done]")
    print("  " + json_path)
    print("  " + txt_path)

if __name__ == "__main__":
    main()
