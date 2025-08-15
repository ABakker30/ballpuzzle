# core/io_pieces.py
import importlib.util, sys, os
from typing import Dict, List, Tuple

Orientation = Tuple[Tuple[int,int,int], ...]
Pieces = Dict[str, Tuple[Orientation, ...]]

def load_pieces(py_path: str) -> Pieces:
    if not py_path.lower().endswith(".py"):
        raise ValueError("pieces path must end with .py")
    if not os.path.isfile(py_path):
        raise IOError("File not found: {}".format(py_path))
    modname = "__pieces__"
    if modname in sys.modules:
        del sys.modules[modname]
    spec = importlib.util.spec_from_file_location(modname, py_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    if not hasattr(mod, "PIECES"):
        raise ValueError("pieces module missing PIECES dict")
    out: Pieces = {}
    for key, oris in mod.PIECES.items():
        key = str(key)
        norm = []
        for ori in oris:
            triples = tuple((int(dx),int(dy),int(dz)) for (dx,dy,dz) in ori)
            if len(triples) != 4:
                raise ValueError("piece {} orientation not 4 cells".format(key))
            norm.append(triples)
        out[key] = tuple(norm)
    if not out:
        raise ValueError("no pieces parsed")
    return out
