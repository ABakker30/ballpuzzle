# solver/core/solver_engine.py
# FCC tetra-spheres solver engine — rev13.2+mod4
# (bitmask + precomputed fits, local pruning, dynamic branch-cap, forced-singletons,
#  exposure+boundary+leaf heuristic, least-tried roulette with corridor lockout,
#  bounded TT, optional deg2-corridor; ADDED: local component mod-4 cavity prune)

import math, random, time
from collections import deque, defaultdict

# --------------------------
# Tunables (rev13.2 + tweaks)
# --------------------------
BRANCH_CAP_OPEN   = 18            # open regions
BRANCH_CAP_TIGHT  = 12            # was 10; slightly wider in corridors to reduce overcommit
ROULETTE_MODE     = "least-tried" # "least-tried" or "none" (auto-disabled in corridor)
RNG_SEED_DEFAULT  = 1337          # fixed seed for reproducibility

# Larger TT so long runs breathe better
TT_MAX            = 3_000_000     # was 1_200_000
TT_TRIM_KEEP      = 2_000_000     # was 800_000

# Heuristic weights (slightly rebalanced for shells)
EXPOSURE_WEIGHT           = 1.0
BOUNDARY_EXPOSURE_WEIGHT  = 0.5   # was 0.8
LEAF_WEIGHT               = 1.3   # was 0.8

# Corridor gating
DEG2_CORRIDOR_DEFAULT     = False # keep degree-2 corridors OFF by default

# New prune
COMPONENT_MOD4_PRUNE      = True  # prune any empty component with size % 4 != 0

# FCC adjacency (12-neighbor)
_NEIGH = (
    (1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1),
    (1,-1,0),(-1,1,0),(1,0,-1),(-1,0,1),(0,1,-1),(0,-1,1)
)

# Preferred piece order (kept as in rev13.2)
_ORDER_PREF = (
    "A","C","E","G","I","J","H","F","D","B","Y",
    # 3D shell struts/corners & connectors
    "X","W","L","K","V","U","T",
    "N","M",
    "S","R","Q","P","O"
)

# --------------------------
# Helpers
# --------------------------
def popcount(x):
    return x.bit_count() if hasattr(int, "bit_count") else bin(x).count("1")

def _pick_order(pieces):
    keys = set(pieces.keys())
    ordered = [k for k in _ORDER_PREF if k in keys]
    remaining = sorted(keys.difference(_ORDER_PREF))
    return tuple(ordered + remaining)

# --------------------------
# Engine
# --------------------------
class SolverEngine(object):
    """
    Standalone solver engine (no Rhino/GH deps). Mirrored from rev13.2 with:
      - Optional component-size %4 prune (local to touched empties)
      - Slightly wider tight branch cap
      - Larger bounded TT
      - Heuristic reweighting for shell-heavy containers
    """

    def __init__(self,
                 valid_set,
                 pieces,
                 *,
                 rng_seed=RNG_SEED_DEFAULT,
                 branch_cap_open=BRANCH_CAP_OPEN,
                 branch_cap_tight=BRANCH_CAP_TIGHT,
                 roulette_mode=ROULETTE_MODE,
                 tt_max=TT_MAX,
                 tt_trim_keep=TT_TRIM_KEEP,
                 exposure_weight=EXPOSURE_WEIGHT,
                 boundary_exposure_weight=BOUNDARY_EXPOSURE_WEIGHT,
                 leaf_weight=LEAF_WEIGHT,
                 deg2_corridor=DEG2_CORRIDOR_DEFAULT,
                 component_mod4_prune=COMPONENT_MOD4_PRUNE):
        """
        valid_set: set of (i,j,k) ints for container cells
        pieces: dict[str] -> tuple[orientation], each orientation = tuple[(di,dj,dz)...] length 4
        """

        self.valid_set = set(valid_set)
        self.idx2cell, self.cell2idx, self.neighbors, self.is_boundary = self._build_grid(self.valid_set)
        self.pieces = pieces
        self.order = _pick_order(pieces)
        self.fits = self._precompute_fits(pieces, self.valid_set, self.cell2idx)

        # Config
        self.RNG_SEED = int(rng_seed)
        self.BRANCH_CAP_OPEN = int(branch_cap_open)
        self.BRANCH_CAP_TIGHT = int(branch_cap_tight)
        self.ROULETTE_MODE = str(roulette_mode)
        self.TT_MAX = int(tt_max)
        self.TT_TRIM_KEEP = int(tt_trim_keep)
        self.EXPOSURE_WEIGHT = float(exposure_weight)
        self.BOUNDARY_EXPOSURE_WEIGHT = float(boundary_exposure_weight)
        self.LEAF_WEIGHT = float(leaf_weight)
        self.deg2_corridor = bool(deg2_corridor)
        self.COMPONENT_MOD4_PRUNE = bool(component_mod4_prune)

        # State
        self.cursor = 0
        self.occ_bits = 0
        self.placements = []   # list of dicts (piece, origin_idx, ori_idx, mask, cells_idx, cells)
        self.frontier = []     # list[deque] per depth
        self.solved = False
        self.dirty = False

        # Stats
        self.attempts = 0
        self.try_counts = defaultdict(int)     # (piece, origin_idx, ori_idx) -> tries
        self.anchor_seen = set()
        self.transitions = defaultdict(int)    # (prev_anchor, cur_anchor) -> count
        self.last_anchor = None

        # prune/score stats
        self.stat_pruned_isolated = 0
        self.stat_pruned_cavity = 0   # reserved
        self.stat_pruned_mod4 = 0     # NEW
        self.stat_considered = 0
        self.stat_exposure_hist = defaultdict(int)
        self.stat_boundary_exposure_hist = defaultdict(int)
        self.stat_leaf_hist = defaultdict(int)
        self.stat_choices_hist = defaultdict(int)
        self.stat_anchor_deg_hist = defaultdict(int)
        self.stat_fallback_piece = defaultdict(int)

        # forced-singletons
        self.forced_singletons = 0

        # TT
        N = len(self.idx2cell)
        self.occ_keys, self.depth_keys = self._init_zobrist(N, len(self.pieces))
        self.TT = {}
        self.tt_hits = 0
        self.tt_prunes = 0

        # runtime toggles
        self.branch_cap_cur = self.BRANCH_CAP_OPEN
        self.roulette_cur = self.ROULETTE_MODE
        self.in_corridor = False

    # ---------- Grid & fits ----------
    def _build_grid(self, valid_set):
        idx2cell = list(sorted(valid_set))
        cell2idx = {c: i for i, c in enumerate(idx2cell)}
        neighbors = []
        is_boundary = []
        for (i,j,k) in idx2cell:
            lst = []
            for di,dj,dz in _NEIGH:
                c = (i+di, j+dj, k+dz)
                if c in cell2idx:
                    lst.append(cell2idx[c])
            neighbors.append(tuple(lst))
            # boundary if ANY FCC neighbor falls outside
            on_boundary = False
            for di,dj,dz in _NEIGH:
                if (i+di, j+dj, k+dz) not in cell2idx:
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

    # ---------- Zobrist / TT ----------
    def _init_zobrist(self, N, depth_cap):
        rnd = random.Random(self.RNG_SEED ^ 0x9E3779B97F4A7C15)
        occ_keys = [rnd.getrandbits(64) for _ in range(N)]
        depth_keys = [rnd.getrandbits(64) for _ in range(depth_cap+1)]
        return occ_keys, depth_keys

    def _tt_hash(self, occ_bits, cursor):
        h = 0
        x = occ_bits
        idx = 0
        while x:
            if (x & 1) != 0:
                h ^= self.occ_keys[idx]
            idx += 1
            x >>= 1
        if cursor < len(self.depth_keys):
            h ^= self.depth_keys[cursor]
        else:
            h ^= (cursor * 11400714819323198485) & ((1<<64)-1)
        return h

    def _tt_should_prune(self):
        if self.TT is None:
            return False
        h = self._tt_hash(self.occ_bits, self.cursor)
        prev_best = self.TT.get(h)
        if prev_best is not None and prev_best >= self.cursor:
            self.tt_hits += 1
            self.tt_prunes += 1
            return True
        return False

    def _tt_record(self):
        if self.TT is None:
            return
        h = self._tt_hash(self.occ_bits, self.cursor)
        prev = self.TT.get(h)
        if (prev is None) or (self.cursor > prev):
            self.TT[h] = self.cursor
        # bounded: age/trim if too large
        if len(self.TT) > self.TT_MAX:
            to_drop = len(self.TT) - self.TT_TRIM_KEEP
            for _ in range(to_drop):
                try:
                    self.TT.pop(next(iter(self.TT)))
                except StopIteration:
                    break

    # ---------- Degrees / anchor ----------
    def _is_occupied_bit(self, bitset, idx):
        return ((bitset >> idx) & 1) != 0

    def _neighbor_degree(self, idx, occ_bits):
        d = 0
        for n in self.neighbors[idx]:
            if ((occ_bits >> n) & 1) == 0:
                d += 1
        return d

    def _select_anchor(self):
        N = len(self.idx2cell)
        occ = self.occ_bits
        best = -1
        best_deg = 10**9
        for idx in range(N):
            if ((occ >> idx) & 1) != 0:
                continue
            deg = self._neighbor_degree(idx, occ)
            if deg < best_deg or (deg == best_deg and (best < 0 or idx < best)):
                best = idx
                best_deg = deg
        return (None, None) if best < 0 else (best, best_deg)

    # ---------- Prunes / scoring ----------
    def _creates_isolated_empty(self, occ_after, touched_idxs):
        """No empty cell with zero empty neighbors (local check)"""
        neighbors = self.neighbors
        to_check = set()
        for t in touched_idxs:
            to_check.add(t)
            for n in neighbors[t]:
                to_check.add(n)
        for x in to_check:
            if ((occ_after >> x) & 1) != 0:
                continue  # filled
            # empty: must have at least one empty neighbor
            for n in neighbors[x]:
                if ((occ_after >> n) & 1) == 0:
                    break
            else:
                return True
        return False

    def _components_mod4_ok(self, occ_after, touched_idxs):
        """NEW: local connected-empty components must have size % 4 == 0."""
        if not self.COMPONENT_MOD4_PRUNE:
            return True
        neighbors = self.neighbors
        def is_empty(idx):
            return ((occ_after >> idx) & 1) == 0
        # Only explore components reachable from empties adjacent to touched
        frontier = set()
        for u in touched_idxs:
            for v in neighbors[u]:
                if is_empty(v):
                    frontier.add(v)
        seen = set()
        for start in frontier:
            if start in seen or not is_empty(start):
                continue
            # BFS this empty component
            stack = [start]
            seen.add(start)
            count = 0
            while stack:
                x = stack.pop()
                count += 1
                for w in neighbors[x]:
                    if w not in seen and is_empty(w):
                        seen.add(w)
                        stack.append(w)
            if (count & 3) != 0:  # not divisible by 4
                return False
        return True

    def _exposure_counts_after(self, occ_after, newly_filled_idxs):
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

    def _leaf_empties_after(self, occ_after, newly_filled_idxs):
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

    # ---------- Choice building / ranking ----------
    def _build_choices_for_piece(self, piece_key):
        occ = self.occ_bits
        fits_map = self.fits[piece_key]
        N = len(self.idx2cell)
        choices = []

        anchor, a_deg = self._select_anchor()
        if anchor is not None:
            self.anchor_seen.add(anchor)
            if self.last_anchor is not None:
                self.transitions[(self.last_anchor, anchor)] += 1
            self.last_anchor = anchor
            self.stat_anchor_deg_hist[a_deg] += 1

        # dynamic cap / roulette based on anchor deg
        in_corridor = False
        if anchor is not None:
            if a_deg == 1:
                in_corridor = True
            elif a_deg == 2 and self.deg2_corridor:
                in_corridor = True
        self.in_corridor = bool(in_corridor)
        self.branch_cap_cur = self.BRANCH_CAP_TIGHT if in_corridor else self.BRANCH_CAP_OPEN
        self.roulette_cur = "none" if in_corridor else self.ROULETTE_MODE

        anchor_neighbor_set = set(self.neighbors[anchor]) if anchor is not None else set()

        def consider(origin_idx, ori_idx, mask, cells_idx):
            occ_after = occ | mask
            self.stat_considered += 1

            # Hard prunes
            if self._creates_isolated_empty(occ_after, cells_idx):
                self.stat_pruned_isolated += 1
                return
            if not self._components_mod4_ok(occ_after, cells_idx):
                self.stat_pruned_mod4 += 1
                return

            # Scores (lower is better)
            e, be = self._exposure_counts_after(occ_after, cells_idx)
            l = self._leaf_empties_after(occ_after, cells_idx)
            self.stat_exposure_hist[e] += 1
            self.stat_boundary_exposure_hist[be] += 1
            self.stat_leaf_hist[l] += 1
            score_expo = (self.EXPOSURE_WEIGHT * e) + (self.BOUNDARY_EXPOSURE_WEIGHT * be) + (self.LEAF_WEIGHT * l)

            # Tie-break: cover anchor >> touch anchor >> distance to anchor origin
            if anchor is None:
                dist_score = 0
            else:
                if anchor in cells_idx:
                    dist_score = -10
                elif any((ci in anchor_neighbor_set) for ci in cells_idx):
                    dist_score = -5
                else:
                    ai, aj, ak = self.idx2cell[anchor]
                    oi, oj, ok = self.idx2cell[origin_idx]
                    dist_score = abs(ai-oi) + abs(aj-oj) + abs(ak-ok)

            choices.append((score_expo, dist_score, origin_idx, ori_idx, mask, cells_idx))

        # Phase 1: anchor-covering fits
        if anchor is not None:
            afits = fits_map.get(anchor)
            if afits:
                for (ori_idx, mask, cells_idx) in afits:
                    if (occ & mask) == 0:
                        consider(anchor, ori_idx, mask, cells_idx)

        # Fallback if none added
        if not choices:
            self.stat_fallback_piece[piece_key] += 1
            for idx in range(N):
                if ((occ >> idx) & 1) != 0:
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

        deco.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))  # stable

        k = self.branch_cap_cur if self.branch_cap_cur and self.branch_cap_cur > 0 else len(deco)
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

    # ---------- Apply / remove ----------
    def _apply_place(self, piece_key, origin_idx, ori_idx, mask, cells_idx):
        self.occ_bits |= mask
        cells_xyz = tuple(self.idx2cell[ii] for ii in cells_idx)
        self.placements.append({
            "piece": piece_key,
            "origin_idx": origin_idx,
            "ori_idx": ori_idx,
            "mask": mask,
            "cells_idx": tuple(cells_idx),
            "cells": cells_xyz,
        })
        self.try_counts[(piece_key, origin_idx, ori_idx)] += 1

    def _remove_last(self):
        if not self.placements:
            return None
        pl = self.placements.pop()
        self.occ_bits &= ~pl["mask"]
        return pl

    # ---------- Frontier ----------
    def _build_frontier_for_depth(self, cursor):
        piece_key = self.order[cursor]
        choices = self._build_choices_for_piece(piece_key)
        self.frontier.append(deque(choices))

    # ---------- One step ----------
    def step_once(self):
        if self.dirty or self.solved:
            return False, self.solved
        self.attempts += 1

        # solved?
        if self.cursor >= len(self.order):
            self.solved = True
            return True, True

        # TT prune?
        if self._tt_should_prune():
            # backtrack immediately
            if self.cursor == 0:
                return False, False
            if len(self.frontier) > self.cursor:
                self.frontier.pop()
            self.cursor -= 1
            self._remove_last()
            return True, False

        # build frontier if needed
        if len(self.frontier) <= self.cursor:
            self._build_frontier_for_depth(self.cursor)

        progressed = False
        while True:
            if self.cursor >= len(self.order):
                self.solved = True
                return True, True

            d = self.frontier[self.cursor]
            if not d:
                # backtrack
                if self.cursor == 0:
                    return progressed, False
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

        return progressed, False

    # ---------- Introspection ----------
    def best_depth(self):
        """Max depth reached this run (current cursor or any prior). Use externally to track global best."""
        # In-engine, the best-so-far isn't tracked; the CLI tracks it. We expose current depth here.
        return self.cursor

    def placed_count(self):
        return len(self.placements)

    def total_pieces(self):
        return len(self.order)
