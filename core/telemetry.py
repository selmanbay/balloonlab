import json, os, time

class TelemetryWriter:
    def __init__(self, out_dir=None):
        self.f = None
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            self.f = open(os.path.join(out_dir, "telemetry.log"), "a", encoding="utf-8")

    def write(self, obj: dict):
        if not self.f: return
        obj = dict(obj)
        obj.setdefault("ts", time.time())
        self.f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self):
        if self.f:
            self.f.close()
            self.f = None
