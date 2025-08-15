#!/usr/bin/env python3
# tools/write_world_layers.py
#
# Render all world-view FCC layers (z = i+j) with piece letters from a *.world.json.
# Works best with solutions written by tools/solve.py (which include integer "cells_ijk").
# If "cells_ijk" is missing (older files), we heuristically recover (i,j,k) by inverting
# the stored presentation frame; we try both scaled and unscaled variants and pick the one
# that matches the container best.

import argparse, json, os, math, sys
from collections import defaultdict

# --- local imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add repo root
from core.io_container import load_lattice_json

def invert_frame(pt, R, t):
    """Return p_iso = R^T * (p_world - t)."""
    x = pt[0] - t[0]; y = pt[1] - t[1]; z = pt[2] - t[2]
    return (
        R[0][0]*x + R[1][0]*y + R[2][0]*z,
        R[0][1]*x + R[1][1]*y + R[2][1]*z,
        R[0][2]*x + R[1][2]*y + R[2][2]*z,
    )

def uvw_to_ijk(u, v, w):
    """Given (u, v, w) = (j+k, i+k, i+j), recover integer (i, j, k)."""
    i = round((v + w - u) / 2.0)
    j = round((u + w - v) / 2.0)
    k = round((u + v - w) / 2.0)
    return i, j, k

def infer_cells_from_world(pieces_world, frame, r, valid_set):
    """
    pieces_world: list of (piece_id, [ [x,y,z], ... ])
    Try to reconstruct lattice (i,j,k) for every world point, by inverting the stored frame.
    We try two hypotheses:
       H1 (scaled):   iso = R^T (world - t) = s * [u,v,w]
       H2 (unscaled): iso = R^T (world - t) =     [u,v,w]
    where s = sqrt(2)*r.
    Pick the hypothesis that yields the most points that land inside valid_set.
    """
    R = frame["R"]; t = frame["t"]
    s = math.sqrt(2.0) * float(r)

    def run_variant(divide_by_s):
        cells_by_piece = []
        hit_in_container = 0
        all_cells = []
        for pid, world_pts in pieces_world:
            this_piece = []
            for p in world_pts:
                q = invert_frame(p, R, t)
                if divide_by_s:
                    u, v, w = (q[0]/s, q[1]/s, q[2]/s)
                else:
                    u, v, w = q
                i, j, k = uvw_to_ijk(u, v, w)
                this_piece.append((i, j, k))
                all_cells.append((i, j, k))
                if (i, j, k) in valid_set:
                    hit_in_container += 1
            cells_by_piece.append((pid, this_piece))
        return hit_in_container, len(set(all_cells)), cells_by_piece

    h1_hits, h1_unique, cells1 = run_variant(divide_by_s=True)
    h2_hits, h2_unique, cells2 = run_variant(divide_by_s=False)

    # Score: prioritize more hits in container, then more unique cells
    if (h1_hits, h1_unique) >= (h2_hits, h2_unique):
        best = cells1; variant = "scaled"
        score = (h1_hits, h1_unique)
    else:
        best = cells2; variant = "unscaled"
        score = (h2_hits, h2_unique)

    return best, variant, score

def render_layers(valid_set, letter_at, empty_char="."):
    """
    Return text with all world-view layers (z = i+j). We print only container cells.
    Rows are y = (i+k); columns are x = (j+k); layers are z = (i+j).
    """
    if not valid_set:
        return "[Empty container]"

    # Build bounds on (u, v, w)
    umin = vmin = wmin = 10**9
    umax = vmax = wmax = -10**9
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

    # Index container cells by (u,v,w) for quick lookup
    uvw_to_ijk = { (j+k, i+k, i+j): (i,j,k) for (i,j,k) in valid_set }

    lines = []
    lines.append("[SOLUTION — world view]")
    lines.append("Legend: rows=v (i+k: {}..{}), cols=u (j+k: {}..{}), layers=w (i+j: {}..{})"
                 .format(vmin, vmax, umin, umax, wmin, wmax))
    lines.append("")

    # Emit each layer
    for w in range(wmin, wmax + 1):
        lines.append("Layer w=i+j={}:".format(w))
        for v in range(vmin, vmax + 1):
            row = []
            for u in range(umin, umax + 1):
                ijk = uvw_to_ijk.get((u, v, w))
                if ijk is None:
                    row.append(" ")
                else:
                    row.append(letter_at.get(ijk, empty_char))
            # rstrip to trim trailing spaces
            lines.append(" ".join(row).rstrip())
        lines.append("")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser(description="Write a world-view layers text for a *.world.json solution.")
    ap.add_argument("--container", required=True, help="Path to container Lattice JSON (for bounds).")
    ap.add_argument("--in", dest="in_path", required=True, help="Input *.world.json")
    ap.add_argument("--out", required=True, help="Output .txt path")
    args = ap.parse_args()

    # Load container
    valid_set, _bbox, _r = load_lattice_json(args.container)
    valid_set = set(valid_set)

    # Load solution
    with open(args.in_path, "r") as f:
        sol = json.load(f)

    r = sol.get("r", 0.5)
    frame = sol.get("presentation", {}).get("frame", {"R": [[1,0,0],[0,1,0],[0,0,1]], "t": [0,0,0]})

    # Build letter map
    letter_at = {}

    # Preferred: use integer "cells_ijk" from the solution (exact, no inference)
    if all(("cells_ijk" in p) for p in sol.get("pieces", [])):
        for p in sol["pieces"]:
            pid = p["id"]
            for (i, j, k) in p["cells_ijk"]:
                letter_at[(i, j, k)] = pid
        used = "cells_ijk"
    else:
        # Fallback: infer from world positions
        pieces_world = [(p["id"], p["world_centers"]) for p in sol.get("pieces", [])]
        cells_by_piece, variant, score = infer_cells_from_world(pieces_world, frame, r, valid_set)
        for pid, cells in cells_by_piece:
            for (i, j, k) in cells:
                letter_at[(i, j, k)] = pid
        used = "frame-inverse ({}, hits={}, unique={})".format(variant, score[0], score[1])

    txt = render_layers(valid_set, letter_at, empty_char=".")

    # Write
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(txt)
    print("Wrote world layers using {} → {}".format(used, args.out))

if __name__ == "__main__":
    main()
