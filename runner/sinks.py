# runner/sinks.py
import os, json, time

class JsonlSink:
    def __init__(self, run_dir: str):
        self.path = os.path.join(run_dir, "runlog.jsonl")
        self._f = open(self.path, "a", buffering=1)

    def handle(self, ev: dict):
        ev = dict(ev)
        ev.setdefault("t_wall", time.time())
        self._f.write(json.dumps(ev) + "\n")

class StdoutSink:
    def handle(self, ev: dict):
        t = ev.get("type")
        if t == "heartbeat":
            print("HB depth={} rate={:.0f}/s tt={}".format(ev.get("depth"), ev.get("rate",0), ev.get("tt_size",0)))
        elif t == "new_solution":
            print("Solution #{} in {} ms".format(ev.get("index"), ev.get("ms_to_solve")))

class SolutionsSink:
    def __init__(self, run_dir: str):
        self.sol_dir = os.path.join(run_dir, "solutions")
        os.makedirs(self.sol_dir, exist_ok=True)

    def save_world_json(self, filename: str, payload: dict):
        path = os.path.join(self.sol_dir, filename)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        return path

    def save_layout_txt(self, filename: str, text: str):
        path = os.path.join(self.sol_dir, filename)
        with open(path, "w") as f:
            f.write(text)
        return path
