# tools/solve.py
# Standalone FCC tetra-spheres solver driver (console) — rev13.2
# - Periodic status printing (default 5s)
# - Snapshots on improved depth and final solved state
# - Robust status printing (deg2 corridor flag fallback)
# - "Best depth (ever)" tracked correctly (monotonic during a run)

import os
import sys
import json
import time
import argparse
import importlib.util

# -----------------------------------------------------------------------------
# Resolve project root for imports
# -----------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import the core engine
from core.solver_engine import SolverEngine  # noqa: E402


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------
def load_container_json(path):
    """
    Load a container JSON produced by our GH utility or hand-authored.
    Expected keys:
      - "cells": list of [i, j, k] FCC lattice integer triplets
      - "r": (optional) sphere radius
      - optionally "presentation" (ignored by this CLI)
    Returns: (valid_set, bbox_tuple, r_or_None)
    """
    with open(path, "r") as f:
        data = json.load(f)

    # Accept either {"cells": [...]} or {"container": {"cells": [...]}}
    obj = data
    if "cells" not in obj and "container" in obj and isinstance(obj["container"], dict):
        obj = obj["container"]

    cells_raw = obj.get("cells", [])
    valid_set = set()
    imin = jmin = kmin = +10**9
    imax = jmax = kmax = -10**9
    for c in cells_raw:
        i, j, k = int(c[0]), int(c[1]), int(c[2])
        valid_set.add((i, j, k))
        if i < imin: imin = i
        if i > imax: imax = i
        if j < jmin: jmin = j
        if j > jmax: jmax = j
        if k < kmin: kmin = k
        if k > kmax: kmax = k

    if not valid_set:
        bbox = (0, -1, 0, -1, 0, -1)
    else:
        bbox = (imin, imax, jmin, jmax, kmin, kmax)

    r = obj.get("r", None)
    return valid_set, bbox, r


def load_pieces(py_path):
    """
    Dynamically load PIECES dict from a Python file.
    The module must define PIECES = { "A": (( (di,dj,dz), ... ), ...), ... }
    """
    if not os.path.isfile(py_path):
        raise IOError("File not found: {}".format(py_path))
    name = "__pieces__"
    if name in sys.modules:
        del sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, py_path)
    if spec is None or spec.loader is None:
        raise ImportError("Invalid module spec for: {}".format(py_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "PIECES"):
        raise ImportError("Module has no PIECES dict: {}".format(py_path))

    pieces = {}
    for key, oris in mod.PIECES.items():
        norm_oris = []
        for ori in oris:
            norm_oris.append(tuple(tuple(int(c) for c in triplet) for triplet in ori))
        pieces[str(key)] = tuple(norm_oris)
    return pieces


# -----------------------------------------------------------------------------
# World view helpers
# -----------------------------------------------------------------------------
def ijk_to_world(i, j, k):
    """FCC world indices: u=x=j+k, v=y=i+k, w=z=i+j"""
    return (j + k, i + k, i + j)


def compute_world_bounds(valid_set):
    """Compute world (u,v,w) bounds from a set of ijk cells."""
    umin = vmin = wmin = +10**9
    umax = vmax = wmax = -10**9
    for (i, j, k) in valid_set:
        u, v, w = ijk_to_world(i, j, k)
        if u < umin: umin = u
        if u > umax: umax = u
        if v < vmin: vmin = v
        if v > vmax: vmax = v
        if w < wmin: wmin = w
        if w > wmax: wmax = w
    if umin > umax:  # empty
        return (0, -1, 0, -1, 0, -1)
    return (umin, umax, vmin, vmax, wmin, wmax)


def write_snapshot(out_dir, snap_idx, meta, eng, r):
    """
    Write a .world.json snapshot of current placements.
    Schema:
      {
        "meta": {...},
        "r": <number or null>,
        "depth": <int>,
        "solved": <bool>,
        "order": [...],
        "cells_by_piece": {
           "A": [[u,v,w], ...],
           ...
        }
      }
    """
    os.makedirs(os.path.join(out_dir, "solutions"), exist_ok=True)
    fname = f"solution_{snap_idx:04d}.world.json"
    fpath = os.path.join(out_dir, "solutions", fname)

    # Build per-piece world cells from engine placements (using cells_idx -> idx2cell)
    by_piece = {}
    for pl in eng.placements:
        piece = pl["piece"]
        world_cells = []
        for ii in pl["cells_idx"]:
            i, j, k = eng.idx2cell[ii]
            world_cells.append(list(ijk_to_world(i, j, k)))
        by_piece.setdefault(piece, []).extend(world_cells)

    payload = {
        "meta": meta,
        "r": r,
        "depth": len(eng.placements),
        "solved": bool(getattr(eng, "solved", False)),
        "order": list(eng.order),
        "cells_by_piece": by_piece,
    }
    with open(fpath, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[write] depth={len(eng.placements):02d} file={fname}")


# -----------------------------------------------------------------------------
# Status printing (robust to engine attribute differences)
# -----------------------------------------------------------------------------
def _top_transition_str(eng):
    trans = getattr(eng, "transitions", {})
    if not trans:
        return "n/a"
    (a, b), cnt = max(trans.items(), key=lambda kv: kv[1])
    return f"{a}→{b}: {cnt}"


def print_status(eng, start_t):
    now = time.time()
    elapsed = now - start_t

    # Rolling attempts/s based on last print
    last_t = getattr(eng, "_last_print_t", start_t)
    last_a = getattr(eng, "_last_print_attempts", 0)
    dt = max(1e-6, now - last_t)
    da = max(0, getattr(eng, "attempts", 0) - last_a)
    rate = int(da / dt)
    eng._last_print_t = now
    eng._last_print_attempts = getattr(eng, "attempts", 0)

    best_ever = getattr(eng, "best_ever", 0)  # FIX: never fall back to current depth

    print(f"[rev13.2] Placed {len(eng.placements)} / {len(eng.order)} pieces "
          f"({100.0*len(eng.placements)/len(eng.order):.1f}%) | Best depth (ever): {best_ever} "
          f"| Rate: {rate:,} attempts/s | elapsed: {elapsed:.1f}s")

    TT = getattr(eng, "TT", {})
    tt_hits = getattr(eng, "tt_hits", 0)
    tt_prunes = getattr(eng, "tt_prunes", 0)
    forced = getattr(eng, "forced_singletons", 0)
    anchors = getattr(eng, "anchor_seen", set())
    trans = getattr(eng, "transitions", {})

    print(f"   anchors: {len(anchors)} unique | transitions: {len(trans)} unique "
          f"(top {_top_transition_str(eng)}) | TT: size={len(TT)} hits={tt_hits} prunes={tt_prunes} | forced-singletons={forced}")

    # Robust deg2 corridor flag lookup (avoid AttributeError)
    deg2_flag = getattr(eng, "deg2_corridor", None)
    if deg2_flag is None:
        deg2_flag = getattr(eng, "corridor_deg2", None)
    if deg2_flag is None:
        deg2_flag = getattr(eng, "CORRIDOR_DEG2", False)

    print(f"   search mode: {getattr(eng, 'roulette_cur', 'n/a')} | branch cap: {getattr(eng, 'branch_cap_cur', 'n/a')} "
          f"(corridor={getattr(eng, 'in_corridor', False)} deg2={deg2_flag})")

    # Optional tuning histograms (print only if present)
    ch = getattr(eng, "stat_choices_hist", None)
    if ch:
        parts = [f"{k}: {ch[k]}" for k in sorted(ch.keys())]
        print("   choices@depth histogram (post-cap): " + ", ".join(parts))

    expo = getattr(eng, "stat_exposure_hist", None)
    if expo:
        keys = sorted(expo.keys())
        show = [f"{k}: {expo[k]:,}" for k in keys[:12]]
        print("   exposure buckets (low=better): " + ", ".join(show))

    bexpo = getattr(eng, "stat_boundary_exposure_hist", None)
    if bexpo:
        keys = sorted(bexpo.keys())
        show = [f"{k}: {bexpo[k]:,}" for k in keys[:12]]
        print("   boundary-exposure (low=better): " + ", ".join(show))

    leaf = getattr(eng, "stat_leaf_hist", None)
    if leaf:
        keys = sorted(leaf.keys())
        show = [f"{k}: {leaf[k]:,}" for k in keys[:12]]
        print("   leaf-empties (lower=better): " + ", ".join(show))

    adeg = getattr(eng, "stat_anchor_deg_hist", None)
    if adeg:
        parts = [f"{k}: {adeg[k]}" for k in sorted(adeg.keys())]
        print("   anchor empty-neighbor degree: " + " | ".join(parts))

    fb = getattr(eng, "stat_fallback_piece", None)
    if fb:
        top_fb = sorted(fb.items(), key=lambda kv: -kv[1])
        head = ", ".join(f"{k}: {v}" for k, v in top_fb[:8])
        print("   pieces needing fallback origins most: " + head)


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="FCC tetra-spheres solver (console) — rev13.2")
    ap.add_argument("--container", required=True, help="Path to container JSON")
    ap.add_argument("--pieces", required=True, help="Path to pieces.py")
    ap.add_argument("--out", required=True, help="Output directory (will create solutions/)")
    ap.add_argument("--present", choices=["auto", "none"], default="auto",
                    help="Presentation hint (kept for compatibility; not applied here)")
    ap.add_argument("--status-interval", type=float, default=5.0,
                    help="Seconds between status prints (0 to disable)")
    ap.add_argument("--snap", choices=["depth", "solved", "none"], default="depth",
                    help="When to write snapshots")
    ap.add_argument("--max-seconds", type=float, default=0.0, help="Stop after N seconds (0=unbounded)")
    ap.add_argument("--max-attempts", type=int, default=0, help="Stop after N attempts (0=unbounded)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load inputs
    valid_set, bbox, r_in = load_container_json(args.container)
    pieces = load_pieces(args.pieces)

    # Init engine with just what it expects
    eng = SolverEngine(pieces=pieces, valid_set=valid_set)

    # Track "best depth ever" MONOTONIC within this run
    eng.best_ever = 0

    # Meta for snapshots
    meta = {
        "container": os.path.abspath(args.container),
        "pieces": os.path.abspath(args.pieces),
        "bbox": bbox,
        "present": args.present,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    start = time.time()
    last_status = start
    snap_idx = 1
    best_written_depth = -1

    # Prime rate baseline
    eng._last_print_t = start
    eng._last_print_attempts = 0

    # Main solve loop
    while True:
        progressed, solved = eng.step_once()

        # Update best depth ever (monotonic)
        if progressed:
            depth = len(eng.placements)
            if depth > eng.best_ever:
                eng.best_ever = depth

        # Time/attempt caps
        if args.max_seconds > 0 and (time.time() - start) >= args.max_seconds:
            print("[halt] max seconds reached")
            break
        if args.max_attempts > 0 and getattr(eng, "attempts", 0) >= args.max_attempts:
            print("[halt] max attempts reached")
            break

        # Snapshot on improved depth (if enabled)
        if progressed and args.snap == "depth":
            depth = len(eng.placements)
            if depth > best_written_depth:
                write_snapshot(args.out, snap_idx, meta, eng, r_in)
                snap_idx += 1
                best_written_depth = depth

        # Periodic status
        if args.status_interval > 0:
            now = time.time()
            if now - last_status >= args.status_interval:
                print_status(eng, start)
                last_status = now

        if solved:
            print("[solved]")
            if args.snap != "none":
                write_snapshot(args.out, snap_idx, meta, eng, r_in)
            break

    # Final status line (in case we stopped without solved)
    if not getattr(eng, "solved", False):
        print_status(eng, start)


if __name__ == "__main__":
    main()
