# tools/write_dummy_solution.py
import os, json, argparse, itertools, time
from typing import Dict, List, Tuple
from core.io_container import load_lattice_json, largest_face_info
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

    # Build a fake placement: take cells in stable order, assign 4 at a time to each piece id cyclically.
    cells = sorted(list(cont["cells"]))
    piece_ids = list(sorted(pieces.keys()))
    groups = [cells[i:i+4] for i in range(0, len(cells), 4)]
    by_piece: Dict[str, List[Tuple[int,int,int]]] = {pid: [] for pid in piece_ids}
    it = itertools.cycle(piece_ids)
    for g in groups:
        if len(g) < 4: break
        pid = next(it)
        by_piece[pid].extend(g)

    # Compute presentation (orientation to XY)
    pres = compute_presentation(set(cells), cont["r"], mode=args.present, square_normal=args.square_normal)

    # Build JSON payload
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
        "presentation": { "mode": pres["mode"], "normal": pres["normal"], "layer_index": pres["layer_value"], "frame": pres["frame"] },
        "pieces": pieces_world
    }

    # Save
    ts = time.strftime("%Y%m%d-%H%M%S")
    sol_name = f"solution_0001.world.json"
    lay_name = f"solution_0001.layout.txt"
    sol_path = os.path.join(args.out, "solutions", sol_name)
    lay_path = os.path.join(args.out, "solutions", lay_name)
    with open(sol_path, "w") as f: json.dump(payload, f, indent=2)

    # Print layout uses lattice coordinates grouped by piece
    layout_txt = render_print_layout(by_piece, pres["mode"], pres["normal"].get("square_axis"))
    with open(lay_path, "w") as f: f.write(layout_txt)

    print("Wrote:")
    print(" ", sol_path)
    print(" ", lay_path)

if __name__ == "__main__":
    main()
