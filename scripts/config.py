"""Load config.local.yaml (or $SP_CONFIG); env vars override top-level keys (e.g. COMFY_URL)."""
import os, yaml
def load(path=None):
    p = path or os.environ.get("SP_CONFIG") or os.path.join(os.path.dirname(__file__), "..", "config.local.yaml")
    c = yaml.safe_load(open(p)) if os.path.exists(p) else {}
    for k in ("comfy_url", "comfy_dir", "times_csv", "judge_url"):
        if os.environ.get(k.upper()): c[k] = os.environ[k.upper()]
    c.setdefault("comfy_url", "http://127.0.0.1:8188"); c.setdefault("models", {}); c.setdefault("character", {})
    return c
