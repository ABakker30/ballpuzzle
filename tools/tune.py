# tools/tune.py
# Offline autotuner (random search + early stopping) for the FCC solver.
# It spawns tools/solve.py with different --set overrides, runs each for a fixed
# time budget, reads metrics.jsonl, scores the run, and reports the best configs.

import os, sys, json, time, random, argparse, subprocess, shutil

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
SOLVE_PY = os.path.join(THIS_DIR, "solve.py")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def read_last_metrics(metrics_path):
    if not os.path.exists(metrics_path):
        return None
    last = None
    with open(metrics_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                last = rec
            except Exception:
                pass
    return last

def score_run(m):
    """
    Simple scalar score from the last metrics record.
    Higher is better.
    """
    if not m:
        return -1e9
    depth = m.get("depth", 0)
    attempts = max(1, m.get("attempts", 1))
    rate = m.get("rate", 0.0)
    tt = m.get("tt", {})
    tt_size = tt.get("size", 0)
    fallback = m.get("fallback", {})
    fb_sum = sum(fallback.values()) if isinstance(fallback, dict) else 0

    # crude progress per attempts
    prog_per_100k = (depth / attempts) * 100000.0

    # penalize giant TT relative to attempts; penalize heavy fallback pressure
    score = 10.0*depth + 1.0*prog_per_100k - 0.1*(tt_size / attempts * 100000.0) - 0.1*(fb_sum / attempts * 100000.0)
    # slight tie-breaker by rate
    score += 0.01*rate
    return score

def sample_params(rng, move_rsq_earlier):
    # Parameter ranges
    BRANCH_CAP_TIGHT = rng.choice([8,10,12,14])
    BRANCH_CAP_OPEN  = rng.choice([16,18,20,22])
    EXPOSURE_WEIGHT  = round(rng.uniform(0.9, 1.2), 2)
    BOUNDARY_EXPOSURE_WEIGHT = round(rng.uniform(0.8, 1.2), 2)
    LEAF_WEIGHT = 0.8  # keep your rev13.2 default
    ROULETTE_MODE = rng.choice(["none","least-tried"])
    CORRIDOR_DEG2 = False  # keep off

    params = {
        "BRANCH_CAP_TIGHT": BRANCH_CAP_TIGHT,
        "BRANCH_CAP_OPEN": BRANCH_CAP_OPEN,
        "EXPOSURE_WEIGHT": EXPOSURE_WEIGHT,
        "BOUNDARY_EXPOSURE_WEIGHT": BOUNDARY_EXPOSURE_WEIGHT,
        "LEAF_WEIGHT": LEAF_WEIGHT,
        "ROULETTE_MODE": ROULETTE_MODE,
        "CORRIDOR_DEG2": CORRIDOR_DEG2
    }

    if move_rsq_earlier:
        params["_ORDER_PREF"] = [
            "A","C","E","G","I","J","H","F","D","B","Y",
            "X","W","L","K","R","S","Q",  # <-- bubble earlier
            "V","U","T",
            "N","M",
            "P","O"
        ]
    # else: engine default order
    return params

def params_to_set_args(params):
    args = []
    for k, v in params.items():
        if isinstance(v, (list, tuple, dict)):
            args.extend(["--set", f"{k}={json.dumps(v)}"])
        else:
            args.extend(["--set", f"{k}={v}"])
    return args

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", required=True, help="Path to lattice JSON")
    ap.add_argument("--pieces", required=True, help="Path to pieces.py")
    ap.add_argument("--trials", type=int, default=12, help="Number of random configs")
    ap.add_argument("--budget-seconds", type=float, default=120.0, help="Per-trial time budget")
    ap.add_argument("--seed", type=int, default=1337, help="Random seed")
    ap.add_argument("--outroot", default=None, help="Root dir to store trials (default: out/tune_<container_name>_<ts>)")
    ap.add_argument("--present", choices=["auto","none"], default="auto")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    container_name = os.path.splitext(os.path.basename(args.container))[0]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    outroot = args.outroot or os.path.join(ROOT_DIR, "out", f"tune_{container_name}_{timestamp}")
    ensure_dir(outroot)

    print(f"[tune] container={args.container}")
    print(f"[tune] pieces={args.pieces}")
    print(f"[tune] trials={args.trials} budget={args.budget_seconds:.0f}s seed={args.seed}")
    print(f"[tune] outroot={outroot}")

    results = []
    for t in range(1, args.trials+1):
        trial_dir = os.path.join(outroot, f"trial_{t:02d}")
        ensure_dir(trial_dir)

        move_rsq_earlier = (t % 2 == 0)  # half trials with RSQ earlier
        params = sample_params(rng, move_rsq_earlier)
        set_args = params_to_set_args(params)

        cmd = [
            sys.executable, SOLVE_PY,
            "--container", args.container,
            "--pieces", args.pieces,
            "--out", trial_dir,
            "--present", args.present,
            "--snap", "none",
            "--status-interval", "5",
            "--max-seconds", str(args.budget_seconds),
        ] + set_args

        print(f"[tune] trial {t:02d} launching…")
        # Let the child stream its progress; we don't parse stdout
        subprocess.run(cmd, cwd=ROOT_DIR)

        mpath = os.path.join(trial_dir, "metrics.jsonl")
        last = read_last_metrics(mpath)
        sc = score_run(last)
        results.append({
            "trial": t,
            "score": sc,
            "metrics": last,
            "params": params,
            "dir": trial_dir
        })
        depth = last.get("depth", 0) if last else 0
        attempts = last.get("attempts", 0) if last else 0
        print(f"[tune] trial {t:02d} depth={depth} attempts={attempts} score={sc:.2f}")

    # Rank and report
    results.sort(key=lambda x: x["score"], reverse=True)
    print("\n[tune] Top 3 configurations:")
    for i, r in enumerate(results[:3], 1):
        d = r["metrics"]["depth"] if r["metrics"] else 0
        a = r["metrics"]["attempts"] if r["metrics"] else 0
        print(f"{i}) score={r['score']:.2f} depth={d} attempts={a} dir={os.path.relpath(r['dir'], ROOT_DIR)}")
        print("   params:", json.dumps(r["params"], indent=2))

    # Save the best params.json next to outroot for future runs
    best = results[0]
    best_params_path = os.path.join(outroot, "best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(best["params"], f, indent=2)
    print(f"\n[tune] Best params written to: {os.path.relpath(best_params_path, ROOT_DIR)}")

    # (Optional) also write/merge a global profiles file keyed by container name
    profiles_path = os.path.join(ROOT_DIR, "data", "tuning", "profiles.json")
    os.makedirs(os.path.dirname(profiles_path), exist_ok=True)
    profiles = {}
    if os.path.exists(profiles_path):
        try:
            with open(profiles_path, "r") as f:
                profiles = json.load(f)
        except Exception:
            profiles = {}
    profiles[container_name] = best["params"]
    with open(profiles_path, "w") as f:
        json.dump(profiles, f, indent=2)
    print(f"[tune] Updated profile for '{container_name}' in {os.path.relpath(profiles_path, ROOT_DIR)}")

if __name__ == "__main__":
    main()
