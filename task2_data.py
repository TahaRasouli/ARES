import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import random


TASK2_KEYS = ["TA", "CC", "LR", "GA"]


def load_task2(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Task2 file not found: {p}")

    if p.suffix == ".jsonl":
        out = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    if p.suffix == ".json":
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # allow either a list or a single dict
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
        raise ValueError("JSON must be a list[dict] or a dict")

    raise ValueError("Unsupported format. Use .json or .jsonl")


def normalize_task2_record(r: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(r)

    # prompt normalization
    if "prompt" not in out:
        if "Topic" in out:
            out["prompt"] = out["Topic"]
        elif "topic" in out:
            out["prompt"] = out["topic"]
        else:
            out["prompt"] = ""

    # essay normalization
    if "essay" not in out:
        if "content" in out:
            out["essay"] = out["content"]
        elif "full_text" in out:
            out["essay"] = out["full_text"]
        else:
            out["essay"] = ""

    return out


def _clean_band(x):
    """IELTS bands must be in [0, 9]. Return None if invalid."""
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if v < 0.0 or v > 9.0:
        return None
    return v


def clean_and_filter_task2(records: List[Dict[str, Any]], require_all_four: bool = True) -> List[Dict[str, Any]]:
    cleaned = []
    for r in records:
        r = normalize_task2_record(r)

        # clean labels
        for k in TASK2_KEYS:
            r[k] = _clean_band(r.get(k, None))

        if require_all_four:
            if not all(r.get(k) is not None for k in TASK2_KEYS):
                continue

        # keep only valid prompt/essay
        if not isinstance(r.get("prompt", ""), str):
            r["prompt"] = str(r.get("prompt", ""))
        if not isinstance(r.get("essay", ""), str):
            r["essay"] = str(r.get("essay", ""))

        cleaned.append(r)

    return cleaned


def split_train_val(records: List[Dict[str, Any]], val_ratio: float = 0.15, seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    idxs = list(range(len(records)))
    rng.shuffle(idxs)

    n_val = max(1, int(len(records) * val_ratio))
    val_set = set(idxs[:n_val])

    train = [records[i] for i in range(len(records)) if i not in val_set]
    val = [records[i] for i in range(len(records)) if i in val_set]
    return train, val


def debug_label_ranges(records: List[Dict[str, Any]], n: int = 2000) -> None:
    """Optional diagnostic utility."""
    from collections import defaultdict
    vals = defaultdict(list)
    for r in records[:n]:
        for k in TASK2_KEYS:
            if r.get(k) is not None:
                vals[k].append(float(r[k]))
    print("\n[DEBUG] Task2 label ranges:")
    for k in TASK2_KEYS:
        v = vals.get(k, [])
        if v:
            print(f"  {k}: min={min(v):.2f}, max={max(v):.2f}, mean={sum(v)/len(v):.2f}, n={len(v)}")
        else:
            print(f"  {k}: no valid labels found")
