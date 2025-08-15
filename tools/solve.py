#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Solver runner with:
 - status.json heartbeat every --status-interval seconds
 - control.json protocol: {"action": "pause"|"resume"|"restart", <optional tuning keys>}
 - final outputs only: <name>_solution.world.json and <name>_solution.world_layers.txt

Assumes core.solver_engine.SolverEngine is available on PYTHONPATH and has:
  - .placements (list of dicts with "piece", "cells" (list[(i,j,k)]) or "cells_idx")
  - .order (tuple/list of piece keys)
  - .cursor (current depth)
  - .TT (dict) .tt_hits .tt_prunes
  - .forced_singletons
  - .in_corridor .deg2_corridor (bools)
  - .branch_cap_cur, .roulette_cur
  - .stat_* dicts: stat_fallback_piece, stat_choices_hist,
                   stat_exposure_hist, stat_boundary_exposure_hist, stat_leaf_hist
  - .step_once() -> (progressed: bool, solved: bool)

Container JSON expected shape:
{
  "name": "firstshape.py",
  "r": 0.5,
  "cells": [[i,j,k], ...],
  "presentation": { ... }  # optional passthrough
}

Pieces module path points to Python file that exposes PIECES dict:
  { "A": [ [(dx,dy,dz),...], [(dx,dy,dz),...], ... ], ... }
"""

from __future__ import annotations
import argparse, json, sys, time, os
from pathlib import Path

# ----------------- imports from project -----------------
try:
    from core.io_pieces import load_pieces
except Exception:
    print("[ERR] Could not import core.io_pieces.load_pieces; add solver root to PYTHONPATH.", file=sys.stderr)
    raise

try:
    from core.solver_engine import SolverEngine
except Exception:
    print("[ERR] Could not import core.solver_engine.SolverEngine", file=sys.stderr)
    raise
# --------------------------------------------------------


# ------------------ Container IO ------------------
def load_container_json(path: str):
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    name = data.get("name") or p.stem
    r = float(data.get("r", 0.5))
    cells = data.get("cells") or []
    valid_set = set((int(i), int(j), int(k)) for (i, j, k) in cells)
    pres = data.get("presentation")  # optional
    return name, valid_set, r, pres


# ------------------ Pretty helpers ------------------
def _to_plain_int_map(d):
    """Make defaultdicts JSON-friendly and robust (keys -> str, values -> int)."""
    out = {}
    try:
        for k, v in (d or {}).items():
            out[str(k)] = int(v)
    except Exception:
        pass
    return out


def make_status_dict(eng, start_time):
    placed = len(getattr(eng, "placements", []))
    total = len(getattr(eng, "order", []))
    elapsed = max(0.0, time.time() - float(start_time))

    # prefer engine rate if it has one; else derive conservative 0
    rate = getattr(eng, "rate", None)
    if rate is None:
        rate = 0.0

    return {
        "placed": placed,
        "total": total,
        "best_depth_ever": int(getattr(eng, "best_depth_ever", max(placed, 0))),
        "rate": float(rate),
        "elapsed": float(elapsed),

        "branch_cap_cur": int(getattr(eng, "branch_cap_cur", 0)),
        "roulette_cur": str(getattr(eng, "roulette_cur", "n/a")),
        "in_corridor": bool(getattr(eng, "in_corridor", False)),
        "deg2_corridor": bool(getattr(eng, "deg2_corridor", False)),

        "tt_size": int(len(getattr(eng, "TT", {}) or {})),
        "tt_hits": int(getattr(eng, "tt_hits", 0)),
        "tt_prunes": int(getattr(eng, "tt_prunes", 0)),
        "forced_singletons": int(getattr(eng, "forced_singletons", 0)),

        # histograms/dists
        "stat_fallback_piece": _to_plain_int_map(getattr(eng, "stat_fallback_piece", {})),
        "stat_choices_hist": _to_plain_int_map(getattr(eng, "stat_choices_hist", {})),
        "stat_exposure_hist": _to_plain_int_map(getattr(eng, "stat_exposure_hist", {})),
        "stat_boundary_exposure_hist": _to_plain_int_map(getattr(eng, "stat_boundary_exposure_hist", {})),
        "stat_leaf_hist": _to_plain_int_map(getattr(eng, "stat_leaf_hist", {})),
    }


def write_status_json(run_out_dir: Path, status: dict):
    run_out_dir.mkdir(parents=True, exist_ok=True)
    (run_out_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")


# ------------------ World layers formatting ------------------
def format_world_layers(valid_set, placements, empty_char="."):
    """
    World-view matrix: layers by z = (i+j), rows by y = (i+k), cols by x = (j+k).
    Writes letters where placed, else '.' for container cells; space for non-container.
    """
    letter_at = {}
    for pl in placements:
        piece = pl["piece"]
        for c in pl.get("cells", []):
            letter_at[tuple(c)] = piece

    # bounds in world (u, v, w)
    umin = vmin = wmin = +10**9
    umax = vmax = wmax = -10**9
    uvw_to_ijk = {}
    for (i, j, k) in valid_set:
        u = j + k  # x index
        v = i + k  # y index
        w = i + j  # z index
        umin = min(umin, u); umax = max(umax, u)
        vmin = min(vmin, v); vmax = max(vmax, v)
        wmin = min(wmin, w); wmax = max(wmax, w)
        uvw_to_ijk[(u, v, w)] = (i, j, k)

    lines = []
    lines.append("[SOLUTION — world view]")
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


# ------------------ Final writers ------------------
def write_final_json(out_dir: Path, shape_name: str, r: float, presentation, placements):
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "world_solution_v1",
        "shape": shape_name,
        "r": r,
        "presentation": presentation or None,
        "placements": [
            {"piece": pl["piece"], "cells": [list(map(int, c)) for c in pl["cells"]]}
            for pl in placements
        ]
    }
    (out_dir / f"{shape_name}_solution.world.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_final_layers_txt(out_dir: Path, shape_name: str, valid_set, placements):
    txt = format_world_layers(valid_set, placements)
    (out_dir / f"{shape_name}_solution.world_layers.txt").write_text(txt, encoding="utf-8")


# ------------------ Control (pause/resume/restart, tuning) ------------------
def apply_tuning_if_present(eng, ctrl: dict):
    # All keys optional; silently ignore missing attributes.
    def set_if(attr, value):
        if hasattr(eng, attr) and value is not None:
            setattr(eng, attr, value)

    # Branch caps
    if "branch_cap_open" in ctrl:
        set_if("branch_cap_open", int(ctrl["branch_cap_open"]))
    if "branch_cap_tight" in ctrl:
        set_if("branch_cap_tight", int(ctrl["branch_cap_tight"]))

    # Deg2 corridor
    if "deg2_corridor" in ctrl:
        set_if("deg2_corridor", bool(ctrl["deg2_corridor"]))

    # Roulette
    if "roulette" in ctrl:
        set_if("roulette_cur", str(ctrl["roulette"]))

    # TT bounds
    if "tt_max" in ctrl:
        set_if("TT_MAX", int(ctrl["tt_max"]))
    if "tt_trim_keep" in ctrl:
        set_if("TT_TRIM_KEEP", int(ctrl["tt_trim_keep"]))


def read_and_clear_control(run_out_dir: Path) -> dict | None:
    p = run_out_dir / "control.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        # malformed; rename for inspection
        p.rename(run_out_dir / "control.malformed.json")
        return None
    # delete after reading so actions are one-shot
    try:
        p.unlink()
    except Exception:
        pass
    return data or {}


# ------------------ Main ------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", required=True, help="Path to container JSON.")
    ap.add_argument("--pieces", required=True, help="Path to pieces .py file exposing PIECES dict.")
    ap.add_argument("--out", required=True, help="Output run folder (status.json, final files).")
    ap.add_argument("--present", choices=["auto", "none"], default="auto", help="Pass through presentation frame if present.")
    ap.add_argument("--status-interval", type=int, default=5, help="Seconds between status.json writes (0 disables).")
    ap.add_argument("--max-seconds", type=int, default=0, help="Stop after N seconds (0 = unlimited).")
    ap.add_argument("--max-attempts", type=int, default=0, help="Stop after N attempts (0 = unlimited).")
    return ap.parse_args()


def main():
    args = parse_args()
    run_out_dir = Path(args.out)

    shape_name, valid_set, r_in, presentation = load_container_json(args.container)
    if args.present == "none":
        presentation = None

    pieces = load_pieces(args.pieces)
    eng = SolverEngine(pieces=pieces, valid_set=valid_set)

    # book-keeping
    start = time.time()
    last_status_ts = 0.0
    paused = False

    print("[info] Writing ONLY final files at end of run (no intermediate snapshots).")

    solved = False
    while True:
        now = time.time()

        # limits
        if args.max_seconds and (now - start) >= args.max_seconds:
            break
        if args.max_attempts and getattr(eng, "attempts", 0) >= args.max_attempts:
            break

        # control polling
        ctrl = read_and_clear_control(run_out_dir)
        if ctrl:
            action = str(ctrl.get("action", "")).lower().strip()
            if action == "pause":
                paused = True
            elif action == "resume":
                paused = False
            elif action == "restart":
                # write a status marker and exit cleanly
                status = make_status_dict(eng, start)
                status["info"] = "restarting (requested by UI)"
                write_status_json(run_out_dir, status)
                return
            # live tuning (optional)
            apply_tuning_if_present(eng, ctrl)

        # work / pause
        if not paused and not solved:
            progressed, solved = eng.step_once()
        else:
            # Sleep a hair when paused to avoid hot spin
            time.sleep(0.01)

        # periodic status
        if args.status_interval:
            if (now - last_status_ts) >= args.status_interval:
                status = make_status_dict(eng, start)
                write_status_json(run_out_dir, status)
                last_status_ts = now

        # solved? break and write final files
        if solved:
            break

    # final snapshot + files
    final_status = make_status_dict(eng, start)
    final_status["info"] = "solved" if solved else "stopped"
    write_status_json(run_out_dir, final_status)

    # Only final files:
    placements = []
    for pl in getattr(eng, "placements", []):
        # ensure shape: {"piece": "A", "cells":[[i,j,k],...]}
        if "cells" in pl:
            cells = [list(map(int, c)) for c in pl["cells"]]
        elif "cells_idx" in pl and hasattr(eng, "idx2cell"):
            cells = [list(map(int, eng.idx2cell[ii])) for ii in pl["cells_idx"]]
        else:
            continue
        placements.append({"piece": pl["piece"], "cells": cells})

    write_final_json(run_out_dir, shape_name, r_in, presentation, placements)
    write_final_layers_txt(run_out_dir, shape_name, valid_set, placements)


if __name__ == "__main__":
    main()
