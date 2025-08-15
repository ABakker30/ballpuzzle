# tools/solve.py
# Standalone FCC tetra-spheres solver driver (rev13.2)
# - Writes world JSON snapshots + world-view layer printouts
# - Periodic status with correct attempts/sec (monotonic window)
# - CLI tunables for branch caps, roulette, TT limits, etc.

from __future__ import annotations

import os, sys, json, time, argparse
from typing import Dict, Tuple, List, Set

# ---------------------------------------------------------------------
# Import path fix: allow `from core...` when launching from tools/
# ---------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from core.solver_engine import SolverEngine
except Exception as e:
    print("[ERR] Could not import core.solver_engine.SolverEngine:", e)
    raise

try:
    from core.io_pieces import load_pieces
except Exception:
    # Fallback: local loader compatible with PIECES dict in a .py file
    import importlib.util
    def _local_load_pieces(py_path: str) -> Dict[str, tuple]:
        p = py_path.strip()
        if not p.lower().endswith(".py"):
            raise IOError("Pieces path must end with .py")
        if not os.path.isfile(p):
            raise IOError("File not found: {}".format(p))
        name = "__pieces__"
        if name in sys.modules:
            del sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, p)
        if spec is None or spec.loader is None:
            raise ImportError("Invalid module spec for: {}".format(p))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "PIECES"):
            raise ImportError("Module has no PIECES dict: {}".format(p))
        pieces = {}
        for key, oris in mod.PIECES.items():
            norm_oris = []
            for ori in oris:
                norm_oris.append(tuple(tuple(int(c) for c in triplet) for triplet in ori))
            pieces[str(key)] = tuple(norm_oris)
        return pieces
    load_pieces = _local_load_pieces


# ---------------------------------------------------------------------
# Container JSON loader
#   Expects:
#     {
#       "r": <float> or "radius": <float>,
#       "cells": [[i,j,k], ...],
#       "presentation": { ... }   // optional, not passed to engine; kept in snapshot meta
#     }
# ---------------------------------------------------------------------
def load_container_json(path: str) -> Tuple[Set[Tuple[int,int,int]], Tuple[int,int,int,int,int,int], float, dict]:
    with open(path, "r") as f:
        data = json.load(f)

    cells = data.get("cells") or data.get("lattice") or []
    if not isinstance(cells, list) or not cells:
        raise ValueError("Container JSON missing non-empty 'cells' array")

    S: Set[Tuple[int,int,int]] = set()
    imin = jmin = kmin = +10**9
    imax = jmax = kmax = -10**9
    for c in cells:
        if not (isinstance(c, (list, tuple)) and len(c) == 3):
            continue
        i, j, k = int(c[0]), int(c[1]), int(c[2])
        S.add((i, j, k))
        if i < imin: imin = i
        if i > imax: imax = i
        if j < jmin: jmin = j
        if j > jmax: jmax = j
        if k < kmin: kmin = k
        if k > kmax: kmax = k

    if not S:
        raise ValueError("Container cells parsed to empty set")

    r = float(data.get("r", data.get("radius", 1.0)))
    presentation = data.get("presentation", {})
    return S, (imin, imax, jmin, jmax, kmin, kmax), r, presentation


# ---------------------------------------------------------------------
# World-view layer formatter (letters by piece)
# World axes (u,v,w) where:
#   u = j + k  (x index)
#   v = i + k  (y index)
#   w = i + j  (z index / layer)
# ---------------------------------------------------------------------
def format_world_layers(valid_set: Set[Tuple[int,int,int]],
                        placements: List[dict],
                        empty_char: str = ".") -> str:
    # bounds + mapping
    umin = vmin = wmin = +10**9
    umax = vmax = wmax = -10**9
    uvw_to_ijk: Dict[Tuple[int,int,int], Tuple[int,int,int]] = {}
    for (i, j, k) in valid_set:
        u = j + k
        v = i + k
        w = i + j
        if u < umin: umin = u
        if u > umax: umax = u
        if v < vmin: vmin = v
        if v > vmax: vmax = v
        if w < wmin: wmin = w
        if w > wmax: wmax = w
        uvw_to_ijk[(u, v, w)] = (i, j, k)

    # letter map from placements
    letter_at: Dict[Tuple[int,int,int], str] = {}
    for pl in placements:
        piece = str(pl["piece"])
        for c in pl["cells"]:
            letter_at[tuple(c)] = piece

    lines = []
    lines.append("[SOLUTION — world view]")
    lines.append(f"Legend: rows=y (i+k: {vmin}..{vmax}), cols=x (j+k: {umin}..{umax}), layers=z (i+j: {wmin}..{wmax})")
    lines.append("")
    for w in range(wmin, wmax + 1):
        lines.append(f"Layer z=i+j={w}:")
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


# ---------------------------------------------------------------------
# Snapshot writer (world.json + layout.txt)
# ---------------------------------------------------------------------
def ensure_dir(p: str) -> None:
    if not os.path.isdir(p):
        os.makedirs(p)

def _idx_to_cell(eng: SolverEngine, idx: int):
    if hasattr(eng, "idx2cell"):
        return eng.idx2cell[idx]
    if hasattr(eng, "index_to_cell"):
        return eng.index_to_cell(idx)  # type: ignore
    raise AttributeError("SolverEngine missing idx2cell mapping")

def _norm_cells_for_pl(eng: SolverEngine, pl: dict):
    if "cells" in pl:
        return [tuple(map(int, c)) for c in pl["cells"]]
    if "cells_idx" in pl:
        return [tuple(map(int, _idx_to_cell(eng, ii))) for ii in pl["cells_idx"]]
    return []

def write_snapshot(out_root: str,
                   snapshot_index: int,
                   meta: dict,
                   eng: SolverEngine,
                   valid_set: Set[Tuple[int,int,int]],
                   write_layout: bool = True) -> None:
    sol_dir = os.path.join(out_root, "solutions")
    ensure_dir(sol_dir)

    seq = snapshot_index + 1
    base = f"solution_{seq:04d}"

    # placements normalized to cells
    pl_out = []
    for pl in eng.placements:
        cells = _norm_cells_for_pl(eng, pl)
        pl_out.append({"piece": pl.get("piece"), "cells": cells})

    world_json = {"meta": meta, "r": meta.get("r"), "placements": pl_out}

    world_path = os.path.join(sol_dir, f"{base}.world.json")
    with open(world_path, "w") as f:
        json.dump(world_json, f, indent=2)
    print(f"[write] depth={eng.cursor:02d} file={os.path.basename(world_path)}")

    if write_layout:
        txt_path = os.path.join(sol_dir, f"{base}.layout.txt")
        txt = format_world_layers(valid_set, pl_out, empty_char=".")
        with open(txt_path, "w") as f:
            f.write(txt)


# ---------------------------------------------------------------------
# Status printing (correct attempts/sec via monotonic window)
# ---------------------------------------------------------------------
def _top_transition_str(eng: SolverEngine) -> str:
    trans = getattr(eng, "transitions", {})
    if not trans:
        return "n/a"
    try:
        (a, b), cnt = max(trans.items(), key=lambda kv: kv[1])
        return f"{a}→{b}: {cnt}"
    except Exception:
        return "n/a"

def print_status(eng: SolverEngine, start_wall: float, meter: dict) -> None:
    now_mono = time.monotonic()
    da = eng.attempts - meter["last_attempts"]
    dt = max(1e-9, now_mono - meter["last_t"])
    rate = da / dt
    elapsed = time.time() - start_wall

    best_ever = getattr(eng, "best_depth_ever", eng.cursor)

    print(f"[rev13.2] Placed {eng.cursor} / {len(eng.order)} pieces "
          f"({(100.0*eng.cursor/len(eng.order)):.1f}%) | "
          f"Best depth (ever): {best_ever} | "
          f"Rate: {rate:,.0f} attempts/s | elapsed: {elapsed:.1f}s")

    tt_hits = getattr(eng, "tt_hits", 0)
    tt_prunes = getattr(eng, "tt_prunes", 0)
    forced_singletons = getattr(eng, "forced_singletons", 0)
    deg2 = getattr(eng, "deg2_corridor", False)

    print(f"   anchors: {len(getattr(eng, 'anchor_seen', []))} unique | "
          f"transitions: {len(getattr(eng, 'transitions', {}))} unique "
          f"(top {_top_transition_str(eng)}) | "
          f"TT: size={len(getattr(eng, 'TT', {}))} hits={tt_hits} prunes={tt_prunes} | "
          f"forced-singletons={forced_singletons}")
    print(f"   search mode: {getattr(eng, 'roulette_mode', 'none')} | "
          f"branch cap: {getattr(eng, 'branch_cap_cur', '-') } "
          f"(corridor={getattr(eng, 'in_corridor', False)} deg2={deg2})")

    if getattr(eng, "stat_choices_hist", None):
        parts = ", ".join(f"{k}: {v}" for k, v in sorted(eng.stat_choices_hist.items()))
        print("   choices@depth histogram (post-cap): " + parts)

    if getattr(eng, "stat_exposure_hist", None):
        keys = sorted(eng.stat_exposure_hist)[:12]
        print("   exposure buckets (low=better): " +
              ", ".join(f"{k}: {eng.stat_exposure_hist[k]:,}" for k in keys))

    if getattr(eng, "stat_boundary_exposure_hist", None):
        keys = sorted(eng.stat_boundary_exposure_hist)[:12]
        print("   boundary-exposure (low=better): " +
              ", ".join(f"{k}: {eng.stat_boundary_exposure_hist[k]:,}" for k in keys))

    if getattr(eng, "stat_leaf_hist", None):
        keys = sorted(eng.stat_leaf_hist)[:12]
        print("   leaf-empties (lower=better): " +
              ", ".join(f"{k}: {eng.stat_leaf_hist[k]:,}" for k in keys))

    if getattr(eng, "stat_anchor_deg_hist", None):
        parts = " | ".join(f"{k}: {v}" for k, v in sorted(eng.stat_anchor_deg_hist.items()))
        print("   anchor empty-neighbor degree: " + parts)

    if getattr(eng, "stat_fallback_piece", None):
        top_fb = sorted(eng.stat_fallback_piece.items(), key=lambda kv: -kv[1])[:8]
        print("   pieces needing fallback origins most: " +
              ", ".join(f"{k}: {v}" for k, v in top_fb))

    meter["last_t"] = now_mono
    meter["last_attempts"] = eng.attempts


# ---------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--container", required=True, help="Path to container JSON")
    p.add_argument("--pieces", required=True, help="Path to pieces.py (PIECES dict)")
    p.add_argument("--out", required=True, help="Output folder (will create solutions/ inside)")
    p.add_argument("--present", choices=["auto", "none"], default="auto",
                   help="Write presentation block from container JSON into snapshots (default auto)")
    p.add_argument("--status-interval", type=float, default=5.0,
                   help="Seconds between status prints (0 to disable)")
    p.add_argument("--snap", choices=["depth", "solved", "none"], default="depth",
                   help="When to write snapshots (default depth)")

    # Tunables / runtime
    p.add_argument("--branch-cap-open", type=int, default=None, help="Open region branch cap")
    p.add_argument("--branch-cap-tight", type=int, default=None, help="Corridor branch cap")
    p.add_argument("--deg2-corridor", type=int, default=None, help="Enable corridor on degree-2 empties (0/1)")
    p.add_argument("--roulette", choices=["least-tried", "none"], default=None, help="Roulette mode")
    p.add_argument("--tt-max", type=int, default=None, help="Transposition table max entries")
    p.add_argument("--tt-trim-keep", type=int, default=None, help="TT keep entries on trim")
    p.add_argument("--stagnation-seconds", type=float, default=None, help="Warn if no depth gain for N seconds")
    p.add_argument("--restart-limit", type=int, default=None, help="(Reserved) max restarts on stagnation")
    return p


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = build_argparser().parse_args()

    out_root = os.path.abspath(args.out)
    ensure_dir(out_root)
    ensure_dir(os.path.join(out_root, "solutions"))

    valid_set, bbox, r_in, presentation = load_container_json(args.container)
    if args.present == "none":
        presentation = {}

    pieces = load_pieces(args.pieces)

    # --- Engine: do NOT pass presentation; your SolverEngine doesn't accept it ---
    eng = SolverEngine(pieces=pieces, valid_set=valid_set)

    # Optional display-only best depth
    eng.best_depth_ever = getattr(eng, "best_depth_ever", 0)

    # Apply tunables if engine exposes them
    def maybe_set(obj, name, value):
        if value is None:
            return
        if hasattr(obj, name):
            setattr(obj, name, value)

    maybe_set(eng, "branch_cap_open", args.branch_cap_open)
    maybe_set(eng, "branch_cap_tight", args.branch_cap_tight)
    if args.deg2_corridor is not None:
        maybe_set(eng, "deg2_corridor", bool(args.deg2_corridor))
    if args.roulette is not None:
        maybe_set(eng, "roulette_mode", args.roulette)
    maybe_set(eng, "TT_MAX", args.tt_max)
    maybe_set(eng, "TT_TRIM_KEEP", args.tt_trim_keep)
    maybe_set(eng, "stagnation_seconds", args.stagnation_seconds)
    maybe_set(eng, "restart_limit", args.restart_limit)

    # Meta base (written into snapshots)
    meta_base = {
        "container_bbox": list(bbox),
        "r": r_in,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "present": presentation,
        "tunables": {
            "branch_cap_open": getattr(eng, "branch_cap_open", None),
            "branch_cap_tight": getattr(eng, "branch_cap_tight", None),
            "deg2_corridor": getattr(eng, "deg2_corridor", False),
            "roulette": getattr(eng, "roulette_mode", "none"),
            "TT_MAX": getattr(eng, "TT_MAX", None),
            "TT_TRIM_KEEP": getattr(eng, "TT_TRIM_KEEP", None),
        }
    }

    # Status meter baseline (monotonic)
    status_meter = {"last_t": time.monotonic(), "last_attempts": 0}
    last_status_t = time.monotonic()

    # Stagnation tracking (optional warn)
    stagnant_since = time.monotonic()

    # Snapshot control
    snap_mode = args.snap
    last_depth_written = -1
    snap_idx = 0

    # Main loop
    start_wall = time.time()
    solved_printed = False

    while True:
        progressed, solved = eng.step_once()

        if eng.cursor > eng.best_depth_ever:
            eng.best_depth_ever = eng.cursor
            stagnant_since = time.monotonic()

        if snap_mode == "depth" and eng.cursor > last_depth_written:
            meta = dict(meta_base)
            meta.update({"depth": eng.cursor, "solved": bool(solved)})
            write_snapshot(out_root, snap_idx, meta, eng, valid_set, write_layout=True)
            snap_idx += 1
            last_depth_written = eng.cursor

        if solved:
            if snap_mode in ("depth", "solved") and last_depth_written != eng.cursor:
                meta = dict(meta_base)
                meta.update({"depth": eng.cursor, "solved": True})
                write_snapshot(out_root, snap_idx, meta, eng, valid_set, write_layout=True)
                snap_idx += 1
                last_depth_written = eng.cursor
            if not solved_printed:
                print("[solved]")
                solved_printed = True
            break

        if args.status_interval and args.status_interval > 0.0:
            now = time.monotonic()
            if now - last_status_t >= args.status_interval:
                print_status(eng, start_wall, status_meter)
                last_status_t = now

        if args.stagnation_seconds and args.stagnation_seconds > 0:
            if time.monotonic() - stagnant_since >= args.stagnation_seconds:
                print(f"[warn] No depth improvement for {args.stagnation_seconds:.0f}s "
                      f"(best depth so far: {eng.best_depth_ever})")
                stagnant_since = time.monotonic()

    # Final status with accurate instantaneous rate
    print_status(eng, start_wall, status_meter)


if __name__ == "__main__":
    main()
