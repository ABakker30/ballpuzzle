# core/io_container.py
import json, hashlib
from typing import Dict, List, Tuple, Set

Cell = Tuple[int, int, int]

def _sha1_cells(cells: Set[Cell]) -> str:
    # stable hash: sorted cells
    s = "\n".join("{},{},{}".format(i,j,k) for (i,j,k) in sorted(cells))
    return "sha1:" + hashlib.sha1(s.encode("utf-8")).hexdigest()

def load_lattice_json(path: str) -> Dict:
    with open(path, "r") as f:
        obj = json.load(f)
    if obj.get("lattice") != "FCC":
        raise ValueError("Only FCC lattice is supported")
    raw = obj.get("cells")
    if not isinstance(raw, list) or not raw:
        raise ValueError("container.json missing non-empty 'cells'")
    cells: Set[Cell] = set()
    for t in raw:
        if not (isinstance(t, (list, tuple)) and len(t) == 3):
            continue
        i,j,k = int(t[0]), int(t[1]), int(t[2])
        cells.add((i,j,k))
    if not cells:
        raise ValueError("No valid cells parsed")
    r = float(obj.get("r", 0.5))
    meta = obj.get("meta", {})
    cmin, cmax = bounds(cells)
    h = _sha1_cells(cells)
    return {
        "lattice": "FCC",
        "cells": cells,
        "r": r,
        "meta": meta,
        "hash": h,
        "bounds": {"imin":cmin[0],"imax":cmax[0],"jmin":cmin[1],"jmax":cmax[1],"kmin":cmin[2],"kmax":cmax[2]},
    }

def bounds(cells: Set[Cell]) -> Tuple[Cell, Cell]:
    imin=jmin=kmin=10**9; imax=jmax=kmax=-10**9
    for (i,j,k) in cells:
        imin=min(imin,i); imax=max(imax,i)
        jmin=min(jmin,j); jmax=max(jmax,j)
        kmin=min(kmin,k); kmax=max(kmax,k)
    return (imin,jmin,kmin), (imax,jmax,kmax)

def largest_face_info(cells: Set[Cell]) -> Dict:
    """Count cells per candidate layer for square (i|j|k const) and triangular (t=i+j+k const)."""
    from collections import defaultdict
    ci=cj=ck=defaultdict(int); ct=defaultdict(int)
    for (i,j,k) in cells:
        ci[i]+=1; cj[j]+=1; ck[k]+=1; ct[i+j+k]+=1
    best_i = max(ci.items(), key=lambda kv: kv[1]) if ci else (None,0)
    best_j = max(cj.items(), key=lambda kv: kv[1]) if cj else (None,0)
    best_k = max(ck.items(), key=lambda kv: kv[1]) if ck else (None,0)
    best_t = max(ct.items(), key=lambda kv: kv[1]) if ct else (None,0)
    best_square = max([("i",)+best_i, ("j",)+best_j, ("k",)+best_k], key=lambda t:(t[2] if t[1] is not None else -1))
    return {
        "square": {"axis": best_square[0], "value": best_square[1], "count": best_square[2]},
        "triangular": {"axis": "t", "value": best_t[0], "count": best_t[1]},
    }
