# core/presenter.py
import math
from typing import Dict, List, Tuple, Set

Cell = Tuple[int,int,int]
Vec3 = Tuple[float,float,float]
Mat3 = Tuple[Tuple[float,float,float], Tuple[float,float,float], Tuple[float,float,float]]

def s_from_r(r: float) -> float:
    return math.sqrt(2.0) * float(r)

def world_from_ijk(i:int,j:int,k:int,r:float) -> Vec3:
    s = s_from_r(r)
    return (s*(j+k), s*(i+k), s*(i+j))

def _mul_W(di:int,dj:int,dk:int,r:float) -> Vec3:
    """World vector for lattice step (di,dj,dk). W = s*[[0,1,1],[1,0,1],[1,1,0]]"""
    s = s_from_r(r)
    return (s*(dj+dk), s*(di+dk), s*(di+dj))

def _norm(v: Vec3) -> float:
    return math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])

def _unit(v: Vec3) -> Vec3:
    n = _norm(v) or 1.0
    return (v[0]/n, v[1]/n, v[2]/n)

def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

def _dot(a: Vec3, b: Vec3) -> float:
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]

def compute_presentation(cells: Set[Cell], r: float, mode: str="auto", square_normal: str=None) -> Dict:
    """
    Decide bottom layer & build an orthonormal frame (ex,ey,ez) so the largest face lies on XY.
    Returns dict with mode, normal (lattice), layer_value, and frame R (columns ex,ey,ez) and t.
    """
    # Tally layers
    from collections import defaultdict
    ci=cj=ck=defaultdict(int); ct=defaultdict(int)
    for (i,j,k) in cells:
        ci[i]+=1; cj[j]+=1; ck[k]+=1; ct[i+j+k]+=1
    best_i = max(ci.items(), key=lambda kv: kv[1]) if ci else (None,0)
    best_j = max(cj.items(), key=lambda kv: kv[1]) if cj else (None,0)
    best_k = max(ck.items(), key=lambda kv: kv[1]) if ck else (None,0)
    best_t = max(ct.items(), key=lambda kv: kv[1]) if ct else (None,0)
    best_square = max([("i",)+best_i, ("j",)+best_j, ("k",)+best_k], key=lambda t:(t[2] if t[1] is not None else -1))

    chosen_mode = mode
    if chosen_mode == "auto":
        chosen_mode = "triangular" if (best_t[1] >= best_square[2]) else "square"

    if chosen_mode == "square":
        axis = square_normal or best_square[0]
        val  = {"i":best_i,"j":best_j,"k":best_k}[axis][0]
        # lattice basis for plane axis=const
        if axis == "i":
            e1=(0,1,0); e2=(0,0,1); n=(1,0,0)
        elif axis == "j":
            e1=(1,0,0); e2=(0,0,1); n=(0,1,0)
        else: # "k"
            e1=(1,0,0); e2=(0,1,0); n=(0,0,1)
    else:
        # triangular: {111} planes, t=i+j+k const, basis 60°
        axis = "t"; val = best_t[0]
        e1=(1,-1,0); e2=(1,0,-1); n=(1,1,1)

    # Map basis to world
    U = _mul_W(*e1, r=r)
    V = _mul_W(*e2, r=r)
    N = _cross(U, V)
    ez = _unit(N)
    ex = _unit(U)
    ey = _unit(_cross(ez, ex))
    R: Mat3 = ((ex[0], ey[0], ez[0]),
               (ex[1], ey[1], ez[1]),
               (ex[2], ey[2], ez[2]))  # columns ex,ey,ez

    # Translation t: put chosen layer at z'=0
    # Find min z' across cells in that layer
    def to_rotated(p: Vec3) -> Vec3:
        return (_dot(p, ex), _dot(p, ey), _dot(p, ez))
    zs = []
    for (i,j,k) in cells:
        in_layer = ((axis=="i" and i==val) or
                    (axis=="j" and j==val) or
                    (axis=="k" and k==val) or
                    (axis=="t" and (i+j+k)==val))
        if in_layer:
            p = world_from_ijk(i,j,k,r)
            zs.append(to_rotated(p)[2])
    z0 = min(zs) if zs else 0.0
    t = (0.0, 0.0, -z0)

    return {
        "mode": chosen_mode,
        "normal": {"square_axis": axis if chosen_mode=="square" else None,
                   "triangular": (axis=="t")},
        "layer_value": val,
        "frame": {"R": ((R[0][0],R[0][1],R[0][2]),
                        (R[1][0],R[1][1],R[1][2]),
                        (R[2][0],R[2][1],R[2][2])),
                  "t": t}
    }

def apply_presentation(points: List[Vec3], frame: Dict) -> List[Vec3]:
    ex = (frame["R"][0][0], frame["R"][1][0], frame["R"][2][0])
    ey = (frame["R"][0][1], frame["R"][1][1], frame["R"][2][1])
    ez = (frame["R"][0][2], frame["R"][1][2], frame["R"][2][2])
    tx,ty,tz = frame["t"]
    out = []
    for p in points:
        out.append((_dot(p,ex)+tx, _dot(p,ey)+ty, _dot(p,ez)+tz))
    return out

def render_print_layout(piece_cells: Dict[str, List[Cell]], mode: str, square_axis: str=None) -> str:
    """
    Build a simple grid text from lattice coords using integer in-plane basis.
    - square i=const → grid (j,k), j=const → (i,k), k=const → (i,j)
    - triangular t=const → grid (u=i-j, v=i-k)
    """
    # Build index maps
    mp = {}  # (u,v) -> letter
    if mode == "square":
        axis = square_axis or "i"
        for letter, lst in piece_cells.items():
            for (i,j,k) in lst:
                if axis == "i": u,v = j,k
                elif axis == "j": u,v = i,k
                else: u,v = i,j
                mp[(u,v)] = letter
    else:
        for letter, lst in piece_cells.items():
            for (i,j,k) in lst:
                u = i - j
                v = i - k
                mp[(u,v)] = letter

    if not mp:
        return "[empty layout]"

    us = [uv[0] for uv in mp.keys()]
    vs = [uv[1] for uv in mp.keys()]
    umin, umax = min(us), max(us)
    vmin, vmax = min(vs), max(vs)

    lines = []
    header = "[SOLUTION — {} view]\n".format("square" if mode=="square" else "triangular")
    lines.append(header)
    lines.append("Legend: rows=v ({}..{}), cols=u ({}..{})\n".format(vmin, vmax, umin, umax))
    for v in range(vmin, vmax+1):
        row = []
        for u in range(umin, umax+1):
            row.append(mp.get((u,v), "."))
        lines.append(" ".join(row))
    return "\n".join(lines)
