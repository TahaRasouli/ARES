import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import random

TASK2_KEYS = ["TA", "CC", "LR", "GA"]

# IELTS Task 2 bands are usually in 0.5 increments (optional constraint)
VALID_HALF_BANDS = {x / 2 for x in range(0, 19)}  # 0.0 ... 9.0


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
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
        raise ValueError("JSON must be list[dict] or dict")

    raise ValueError("Unsupported format. Use .json or .jsonl")


def normalize_task2_record(r: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(r)

    if "prompt" not in out:
        if "Topic" in out:
            out["prompt"] = out["Topic"]
        elif "topic" in out:
            out["prompt"] = out["topic"]
        else:
            out["prompt"] = ""

    if "essay" not in out:
        if "content" in out:
            out["essay"] = out["content"]
        elif "full_text" in out:
            out["essay"] = out["full_text"]
        else:
            out["essay"] = ""

    # make sure strings
    out["prompt"] = "" if out["prompt"] is None else str(out["prompt"])
    out["essay"] = "" if out["essay"] is None else str(out["essay"])
    return out


def _clean_band(x):
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if v < 0.0 or v > 9.0:
        return None
    return v


def clean_and_filter_task2(
    records: List[Dict[str, Any]],
    require_all_four: bool = True,
    enforce_half_bands: bool = False,
    max_gap: float = 4.0,
) -> List[Dict[str, Any]]:
    """
    enforce_half_bands: drop samples whose labels are not multiples of 0.5
    max_gap: drop samples with extreme inconsistency across criteria (noise heuristic)
    """
    cleaned = []
    for r in records:
        r = normalize_task2_record(r)

        for k in TASK2_KEYS:
            r[k] = _clean_band(r.get(k, None))

        if require_all_four and not all(r.get(k) is not None for k in TASK2_KEYS):
            continue

        if enforce_half_bands:
            ok = True
            for k in TASK2_KEYS:
                if r[k] not in VALID_HALF_BANDS:
                    ok = False
                    break
            if not ok:
                continue

        # simple noise filter: drop extreme inconsistencies
        vals = [r[k] for k in TASK2_KEYS]
        if max(vals) - min(vals) > max_gap:
            continue

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
