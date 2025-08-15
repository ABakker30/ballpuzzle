# core/solver_engine.py
# FCC tetra-spheres solver engine — rev13.2 (standalone port from GH)
# Features: bitmask + precomputed fits, local pruning, dynamic branch-cap,
# forced-singletons, exposure+boundary+leaf heuristic, least-tried roulette
# with corridor lockout, bounded TT, deg2-corridor toggle.
#
# This file intentionally does NOT depend on Rhino/GH. It operates purely on
# integer FCC lattice coordinates (i, j, k).

from __future__ import annotations
import math
import random
import time
from collections import deque, defaultdict
from typing import Dict, Tuple, List, Set, Optional

# --------------------------
# Tunables (defaults; can be tweaked by caller after construction)
# --------------------------
DEFAULT_BRANCH_CAP_OPEN   = 18            # open regions
DEFAULT_BRANCH_CAP_TIGHT  = 10            # degree-1 corridors
DEFAULT_ROULETTE_MODE     = "least-tried" # "least-tried" or "none"
DEFAULT_RNG_SEED          = 1337
DEFAULT_TT_MAX            = 1_200_000
DEFAULT_TT_TRIM_KEEP      = 800_000

# Heuristic weights (rev13.2)
DEFAULT_EXPOSURE_WEIGHT          = 1.0
DEFAULT_BOUNDARY_EXPOSURE_WEIGHT = 0.8
DEFAULT_LEAF_WEIGHT              = 0.8

# Corridor gating (rev13.2)
DEFAULT_DEG2_CORRIDOR = False  # keep degree-2 corridors OFF unless explicitly enabled


# FCC adjacency (12-neighbor)
_NEIGH = (
    (1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1),
    (1,-1,0),(-1,1,0),(1,0,-1),(-1,0,1),(0,1,-1),(0,-1,1)
)

# Preferred piece order bias (rev13.2)
_ORDER_PREF = (
    "A","C","E","G","I","J","H","F","D","B","Y",
    "X","W","L","K","V","U","T",
    "N","M",
    "S","R","Q","P","O"
)


class SolverEngine:
    """
    Stateless inputs:
      - pieces: dict[str, tuple[tuple[(dx,dy,dz), ...], ...]]  (orientations per piece)
      - valid_set: set[(i,j,k)]                                (container cells)

    Maintains search state and stats during run.
    """

    # --------------------------
    # Construction
    # --------------------------
    def __init__(self,
                 pieces: Dict[str, Tuple[Tuple[Tuple[int,int,int], ...], ...]],
                 valid_set: Set[Tuple[int,int,int]]):
        # Inputs
        self.pieces = self._normalize_pieces(pieces)
        self.valid_set = set(valid_set)

        # Tunables (mutable; caller may update after construct)
        self.RNG_SEED          = DEFAULT_RNG_SEED
        self.BRANCH_CAP_OPEN   = DEFAULT_BRANCH_CAP_OPEN
        self.BRANCH_CAP_TIGHT  = DEFAULT_BRANCH_CAP_TIGHT
        self.ROULETTE_MODE     = DEFAULT_ROULETTE_MODE
        self.TT_MAX            = DEFAULT_TT_MAX
        self.TT_TRIM_KEEP      = DEFAULT_TT_TRIM_KEEP
        self.EXPOSURE_WEIGHT          = DEFAULT_EXPOSURE_WEIGHT
        self.BOUNDARY_EXPOSURE_WEIGHT = DEFAULT_BOUNDARY_EXPOSURE_WEIGHT
        self.LEAF_WEIGHT              = DEFAULT_LEAF_WEIGHT
        self.deg2_corridor     = DEFAULT_DEG2_CORRIDOR

        # Grid
        (self.idx2cell,
         self.cell2idx,
         self.neighbors,
         self.is_boundary) = self._build_grid(self.valid_set)

        # Precompute
        self.fits = self._precompute_fits(self.pieces, self.valid_set, self.cell2idx)

        # Order
        self.order = self._pick_order(self.pieces)

        # TT init
        random.seed(self.RNG_SEED)
        N = len(self.idx2cell)
        self.occ_keys, self.depth_keys = self._init_zobrist(N, len(self.order))
        self.TT: Dict[int, int] = {}

        # State
        self.cursor = 0
        self.occ_bits = 0
        self.placements: List[Dict] = []     # each: {"piece", "origin_idx", "ori_idx", "mask", "cells_idx"}
        self.frontier: List[deque] = []      # per-depth deque of choices
        self.solved = False
        self.dirty = False

        # Perf counters
        self.attempts = 0
        self._t0 = time.time()

        # Search bookkeeping / stats
        self.try_counts = defaultdict(int)     # (piece, origin_idx, ori_idx) -> tries
        self.anchor_seen: Set[int] = set()
        self.transitions = defaultdict(int)    # (prev_anchor, cur_anchor) -> count
        self.last_anchor: Optional[int] = None

        # Histograms / counters
        self.stat_pruned_isolated = 0
        self.stat_pruned_cavity   = 0  # reserved
        self.stat_considered      = 0

        self.stat_exposure_hist = defaultdict(int)
        self.stat_boundary_exposure_hist = defaultdict(int)
        self.stat_leaf_hist = defaultdict(int)
        self.stat_choices_hist = defaultdict(int)
        self.stat_anchor_deg_hist = defaultdict(int)
        self.stat_fallback_piece = defaultdict(int)

        # forced-singletons
        self.forced_singletons = 0

        # TT stats
        self.tt_hits   = 0
        self.tt_prunes = 0

        # Runtime toggles (updated per-depth)
        self.branch_cap_cur = self.BRANCH_CAP_OPEN
        self.roulette_cur   = self.ROULETTE_MODE
        self.in_corridor    = False

        # Best depth ever (for UX; maintained externally in some setups too)
        self.best_depth_ever = 0

    # --------------------------
    # Public helpers
    # --------------------------
    def placed_count(self) -> int:
        return len(self.placements)

    def total_pieces(self) -> int:
        return len(self.order)

    def elapsed_seconds(self) -> float:
        return time.time() - self._t0

    # --------------------------
    # Grid & fits
    # --------------------------
    def _build_grid(self, valid_set: Set[Tuple[int,int,int]]):
        idx2cell = list(sorted(valid_set))
        cell2idx = {c: i for i, c in enumerate(idx2cell)}
        neighbors: List[Tuple[int, ...]] = []
        is_boundary: List[bool] = []
        for (i,j,k) in idx2cell:
            lst = []
            for di,dj,dk in _NEIGH:
                c = (i+di, j+dj, k+dk)
                if c in cell2idx:
                    lst.append(cell2idx[c])
            neighbors.append(tuple(lst))
            on_boundary = False
            for di,dj,dk in _NEIGH:
                c = (i+di, j+dj, k+dk)
                if c not in cell2idx:
                    on_boundary = True
                    break
            is_boundary.append(on_boundary)
        return tuple(idx2cell), cell2idx, tuple(neighbors), tuple(is_boundary)

    def _precompute_fits(self, pieces, valid_set, cell2idx):
        fits = {}
        for key, oris in pieces.items():
            per_origin = {}
            for (oi,oj,ok), oidx in ((c, cell2idx[c]) for c in valid_set if c in cell2idx):
                lst = []
                for ori_idx, ori in enumerate(oris):
                    ok_all = True
                    idxs = []
                    for (dx,dy,dz) in ori:
                        c = (oi+dx, oj+dy, ok+dz)
                        idx = cell2idx.get(c)
                        if idx is None:
                            ok_all = False
                            break
                        idxs.append(idx)
                    if ok_all:
                        mask = 0
                        for ii in idxs:
                            mask |= (1 << ii)
                        lst.append((ori_idx, mask, tuple(idxs)))
                if lst:
                    per_origin[oidx] = tuple(lst)
            fits[key] = per_origin
        return fits

    # --------------------------
    # Pieces & order
    # --------------------------
    def _normalize_pieces(self, pieces_in):
        pieces = {}
        for k, oris in pieces_in.items():
            norm_oris = []
            for ori in oris:
                norm_oris.append(tuple((int(a), int(b), int(c)) for (a,b,c) in ori))
            pieces[str(k)] = tuple(norm_oris)
        return pieces

    def _pick_order(self, pieces):
        keys = set(pieces.keys())
        ordered = [k for k in _ORDER_PREF if k in keys]
        remaining = sorted(keys.difference(_ORDER_PREF))
        return tuple(ordered + remaining)

    # --------------------------
    # Bit helpers
    # --------------------------
    @staticmethod
    def _is_occupied(bitset: int, idx: int) -> int:
        return (bitset >> idx) & 1

    # --------------------------
    # Anchor select
    # --------------------------
    def _neighbor_degree(self, idx: int, occ_bits: int) -> int:
        d = 0
        for n in self.neighbors[idx]:
            if not self._is_occupied(occ_bits, n):
                d += 1
        return d

    def _select_anchor(self, N: int, occ_bits: int) -> Tuple[Optional[int], Optional[int]]:
        best = -1
        best_deg = 10**9
        for idx in range(N):
            if not self._is_occupied(occ_bits, idx):
                deg = self._neighbor_degree(idx, occ_bits)
                if deg < best_deg or (deg == best_deg and (best < 0 or idx < best)):
                    best = idx
                    best_deg = deg
        return (None, None) if best < 0 else (best, best_deg)

    # --------------------------
    # Zobrist / TT
    # --------------------------
    def _init_zobrist(self, N: int, depth_cap: int):
        random.seed(self.RNG_SEED ^ 0x9E3779B97F4A7C15)
        occ_keys = [random.getrandbits(64) for _ in range(N)]
        depth_keys = [random.getrandbits(64) for _ in range(depth_cap+1)]
        return occ_keys, depth_keys

    def _tt_hash(self, occ_bits: int, cursor: int) -> int:
        h = 0
        x = occ_bits
        idx = 0
        while x:
            if x & 1:
                h ^= self.occ_keys[idx]
            idx += 1
            x >>= 1
        if cursor < len(self.depth_keys):
            h ^= self.depth_keys[cursor]
        else:
            h ^= (cursor * 11400714819323198485) & ((1<<64)-1)
        return h

    def _tt_should_prune(self) -> bool:
        if self.TT is None:
            return False
        h = self._tt_hash(self.occ_bits, self.cursor)
        prev_best = self.TT.get(h)
        if prev_best is not None and prev_best >= self.cursor:
            self.tt_hits += 1
            self.tt_prunes += 1
            return True
        return False

    def _tt_record(self) -> None:
        if self.TT is None:
            return
        h = self._tt_hash(self.occ_bits, self.cursor)
        prev = self.TT.get(h)
        if (prev is None) or (self.cursor > prev):
            self.TT[h] = self.cursor
        if len(self.TT) > self.TT_MAX:
            to_drop = len(self.TT) - self.TT_TRIM_KEEP
            for _ in range(to_drop):
                try:
                    self.TT.pop(next(iter(self.TT)))
                except StopIteration:
                    break

    # --------------------------
    # Pruning helpers
    # --------------------------
    def _creates_isolated_empty(self, occ_after: int, touched_idxs: Tuple[int, ...]) -> bool:
        neighbors = self.neighbors
        to_check = set()
        for t in touched_idxs:
            to_check.add(t)
            for n in neighbors[t]:
                to_check.add(n)
        for x in to_check:
            if ((occ_after >> x) & 1) != 0:
                continue  # filled
            has_empty_neighbor = False
            for n in neighbors[x]:
                if ((occ_after >> n) & 1) == 0:
                    has_empty_neighbor = True
                    break
            if not has_empty_neighbor:
                return True
        return False

    def _exposure_counts_after(self, occ_after: int, newly_filled_idxs: Tuple[int, ...]) -> Tuple[int,int]:
        neighbors = self.neighbors
        is_boundary = self.is_boundary
        seen = set()
        expo = 0
        bexpo = 0
        for u in newly_filled_idxs:
            for v in neighbors[u]:
                if ((occ_after >> v) & 1) == 0 and v not in seen:
                    seen.add(v)
                    expo += 1
                    if is_boundary[v]:
                        bexpo += 1
        return expo, bexpo

    def _leaf_empties_after(self, occ_after: int, newly_filled_idxs: Tuple[int, ...]) -> int:
        neighbors = self.neighbors
        cand = set()
        for u in newly_filled_idxs:
            for v in neighbors[u]:
                if ((occ_after >> v) & 1) == 0:
                    cand.add(v)
        leafs = 0
        for v in cand:
            empty_neighbors = 0
            for w in neighbors[v]:
                if ((occ_after >> w) & 1) == 0:
                    empty_neighbors += 1
                    if empty_neighbors >= 2:
                        break
            if empty_neighbors == 1:
                leafs += 1
        return leafs

    # --------------------------
    # Build choices (ranking, cap, roulette)
    # --------------------------
    def _build_choices_bits(self, piece_key: str) -> List[Tuple[int,int,int,Tuple[int,...]]]:
        occ = self.occ_bits
        fits_map = self.fits[piece_key]
        neighbors = self.neighbors
        idx2cell = self.idx2cell
        N = len(idx2cell)
        choices = []

        anchor, a_deg = self._select_anchor(N, occ)
        if anchor is not None:
            self.anchor_seen.add(anchor)
            if self.last_anchor is not None:
                self.transitions[(self.last_anchor, anchor)] += 1
            self.last_anchor = anchor
            self.stat_anchor_deg_hist[a_deg] += 1

        # corridor mode / roulette mode switch
        in_corridor = False
        if anchor is not None:
            if a_deg == 1:
                in_corridor = True
            elif a_deg == 2 and self.deg2_corridor:
                in_corridor = True

        self.in_corridor   = bool(in_corridor)
        self.branch_cap_cur = self.BRANCH_CAP_TIGHT if in_corridor else self.BRANCH_CAP_OPEN
        self.roulette_cur   = "none" if in_corridor else self.ROULETTE_MODE

        anchor_neighbor_set = set(neighbors[anchor]) if anchor is not None else set()

        def consider(origin_idx, ori_idx, mask, cells_idx):
            occ_after = occ | mask
            self.stat_considered += 1
            if self._creates_isolated_empty(occ_after, cells_idx):
                self.stat_pruned_isolated += 1
                return

            e, be = self._exposure_counts_after(occ_after, cells_idx)
            l     = self._leaf_empties_after(occ_after, cells_idx)
            self.stat_exposure_hist[e] += 1
            self.stat_boundary_exposure_hist[be] += 1
            self.stat_leaf_hist[l] += 1
            score_expo = (self.EXPOSURE_WEIGHT * e) + (self.BOUNDARY_EXPOSURE_WEIGHT * be) + (self.LEAF_WEIGHT * l)

            # distance / anchor tie-break
            if anchor is None:
                dist_score = 0
            else:
                if anchor in cells_idx:
                    dist_score = -10
                elif any((ci in anchor_neighbor_set) for ci in cells_idx):
                    dist_score = -5
                else:
                    ai, aj, ak = idx2cell[anchor]
                    oi, oj, ok = idx2cell[origin_idx]
                    dist_score = abs(ai-oi) + abs(aj-oj) + abs(ak-ok)

            choices.append((score_expo, dist_score, origin_idx, ori_idx, mask, cells_idx))

        # Phase 1: try covering the anchor, if possible
        if anchor is not None:
            afits = fits_map.get(anchor)
            if afits:
                for (ori_idx, mask, cells_idx) in afits:
                    if (occ & mask) == 0:
                        consider(anchor, ori_idx, mask, cells_idx)

        # Fallback: any origin (kept tight cap & no roulette in corridor)
        if not choices:
            self.stat_fallback_piece[piece_key] += 1
            for idx in range(N):
                if self._is_occupied(occ, idx):
                    continue
                pfits = fits_map.get(idx)
                if not pfits:
                    continue
                for (ori_idx, mask, cells_idx) in pfits:
                    if (occ & mask) == 0:
                        consider(idx, ori_idx, mask, cells_idx)

        return self._rank_and_cap(piece_key, choices)

    def _rank_and_cap(self, piece_key, choices):
        if not choices:
            self.stat_choices_hist[0] += 1
            return []

        tc = self.try_counts
        deco = []
        for score_expo, dist_score, origin_idx, ori_idx, mask, cells_idx in choices:
            key = (piece_key, origin_idx, ori_idx)
            deco.append((score_expo, dist_score, tc[key], origin_idx, ori_idx, mask, cells_idx))

        deco.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))

        k = self.branch_cap_cur if (self.branch_cap_cur and self.branch_cap_cur > 0) else len(deco)
        top = list(deco[:k])

        if self.roulette_cur == "least-tried":
            grouped = defaultdict(list)
            for item in top:
                grouped[(item[0], item[2])].append(item)
            ordered = []
            rnd = random.Random(self.RNG_SEED ^ 0xC0FFEE ^ len(self.placements))
            for key in sorted(grouped.keys()):
                bucket = grouped[key]
                rnd.shuffle(bucket)
                ordered.extend(bucket)
            deco = ordered
        else:
            deco = top

        self.stat_choices_hist[len(deco)] += 1
        out = [(origin_idx, ori_idx, mask, cells_idx) for _,_,_, origin_idx, ori_idx, mask, cells_idx in deco]
        return out

    # --------------------------
    # Apply / remove
    # --------------------------
    def _apply_place(self, piece_key, origin_idx, ori_idx, mask, cells_idx):
        self.occ_bits |= mask
        self.placements.append({
            "piece": piece_key,
            "origin_idx": origin_idx,
            "ori_idx": ori_idx,
            "mask": mask,
            "cells_idx": tuple(cells_idx),
        })
        self.try_counts[(piece_key, origin_idx, ori_idx)] += 1

    def _remove_last(self):
        if not self.placements:
            return None
        pl = self.placements.pop()
        self.occ_bits &= ~pl["mask"]
        return pl

    # --------------------------
    # Frontier build (DEFENSIVE)
    # --------------------------
    def _build_frontier_for_depth(self, cursor: int) -> None:
        """
        Build the deque of choices for the current depth if needed.
        Defensive: if cursor is at/after end of order, do nothing.
        """
        if cursor >= len(self.order):
            return
        piece_key = self.order[cursor]
        choices = self._build_choices_bits(piece_key)
        self.frontier.append(deque(choices))

    # --------------------------
    # One search step (+ forced-singletons)
    # --------------------------
    def step_once(self):
        """
        Returns (progressed: bool, solved: bool)
        """
        if self.dirty or self.solved:
            return False, self.solved

        self.attempts += 1

        # Defensive clamp & solved check
        if self.cursor < 0:
            self.cursor = 0
        n_pieces = len(self.order)
        if self.cursor >= n_pieces:
            self.solved = True
            # Update best depth if needed
            if self.placed_count() > self.best_depth_ever:
                self.best_depth_ever = self.placed_count()
            return True, True

        # TT prune?
        if self._tt_should_prune():
            # Backtrack immediately
            if self.cursor == 0:
                return False, False
            if len(self.frontier) > self.cursor:
                self.frontier.pop()
            self.cursor -= 1
            self._remove_last()
            return True, False

        # Build frontier if needed (defensive)
        if self.cursor >= len(self.order):
            self.solved = True
            if self.placed_count() > self.best_depth_ever:
                self.best_depth_ever = self.placed_count()
            return True, True
        if len(self.frontier) <= self.cursor:
            self._build_frontier_for_depth(self.cursor)

        progressed = False

        while True:
            if self.cursor >= len(self.order):
                self.solved = True
                if self.placed_count() > self.best_depth_ever:
                    self.best_depth_ever = self.placed_count()
                return True, True

            d = self.frontier[self.cursor]
            if not d:
                # backtrack
                if self.cursor == 0:
                    # update best depth ever even on failure forward
                    if self.placed_count() > self.best_depth_ever:
                        self.best_depth_ever = self.placed_count()
                    return progressed, False
                if len(self.frontier) > self.cursor:
                    self.frontier.pop()
                self.cursor -= 1
                self._remove_last()
                progressed = True
                # record backtrack position in TT
                self._tt_record()
                break

            if len(d) == 1:
                origin_idx, ori_idx, mask, cells_idx = d.popleft()
                piece_key = self.order[self.cursor]
                self._apply_place(piece_key, origin_idx, ori_idx, mask, cells_idx)
                self.cursor += 1
                self.forced_singletons += 1
                if len(self.frontier) <= self.cursor:
                    self._build_frontier_for_depth(self.cursor)
                progressed = True
                continue
            else:
                origin_idx, ori_idx, mask, cells_idx = d.popleft()
                piece_key = self.order[self.cursor]
                self._apply_place(piece_key, origin_idx, ori_idx, mask, cells_idx)
                self.cursor += 1
                progressed = True
                break

        # Update best depth ever whenever we move forward
        if self.placed_count() > self.best_depth_ever:
            self.best_depth_ever = self.placed_count()

        return progressed, False
