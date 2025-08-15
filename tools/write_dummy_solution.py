# Import os before modifying sys.path
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json, argparse, itertools, time
from typing import Dict, List, Tuple
from core.io_container import load_lattice_json
from core.io_pieces import load_pieces
from core.presenter import world_from_ijk, compute_presentation, apply_presentation, render_print_layout

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", required=True)
    ap.add_argument("--pieces", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--present", choices=["auto","square","triangular"], default="auto")
    ap.add_argument("--square-normal", choices=["i","j","k"], default=None)
    args = ap.parse_args()

    cont = load_lattice_json(args.container)
    pieces = load_pieces(args.pieces)
    os.makedirs(os.path.join(args.out, "solutions"), exist_ok=True)

    # Fake placement for testing: assign 4 consecutive cells to each piece id cyclically
    cells = sorted(list(cont["cells"]))
    piece_ids = list(sorted(pieces.keys()))
    groups = [cells[i:i+4] for i in range(0, len(cells), 4)]
    by_piece: Dict[str, List[Tuple[int,int,int]]] = {pid: [] for pid in piece_ids}
    it = itertools.cycle(piece_ids)
    for g in groups:
        if len(g) < 4: break
        pid = next(it)
        by_piece[pid].extend(g)

    # Presentation (auto/square/triangular) to put largest face on XY
    pres = compute_presentation(set(cells), cont["r"], mode=args.present, square_normal=args.square_normal)

    # Build world-coord payload grouped by piece
    pieces_world = []
    for pid in sorted(by_piece.keys()):
        pts = [world_from_ijk(i,j,k, cont["r"]) for (i,j,k) in by_piece[pid]]
        pts = apply_presentation(pts, pres["frame"])
        pieces_world.append({"id": pid, "world_centers": pts})

    payload = {
        "version": 1,
        "lattice": "FCC",
        "r": cont["r"],
        "container_hash": cont["hash"],
        "container_name": cont.get("meta",{}).get("name"),
        "presentation": {
            "mode": pres["mode"],
            "normal": pres["normal"],
            "layer_index": pres["layer_value"],
            "frame": pres["frame"]
        },
        "pieces": pieces_world
    }

    ts = time.strftime("%Y%m%d-%H%M%S")
    sol_dir = os.path.join(args.out, "solutions")
    sol_name = "solution_0001.world.json"
    lay_name = "solution_0001.layout.txt"

    with open(os.path.join(sol_dir, sol_name), "w") as f:
        json.dump(payload, f, indent=2)

    layout_txt = render_print_layout(by_piece, pres["mode"], pres["normal"].get("square_axis"))
    with open(os.path.join(sol_dir, lay_name), "w") as f:
        f.write(layout_txt)

    print("Wrote:")
    print(" ", os.path.join(sol_dir, sol_name))
    print(" ", os.path.join(sol_dir, lay_name))

if __name__ == "__main__":
    main()
