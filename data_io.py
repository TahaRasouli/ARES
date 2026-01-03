import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd


def _to_float(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_json(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in JSON file: {path}")
    return data


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    records = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_eclipse_csv(path: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(path)

    required_cols = ["prompt", "full_text"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {path}")

    # ECLIPSE analytic targets (if present)
    target_cols = [
        "Overall", "Cohesion", "Syntax", "Vocabulary",
        "Phraseology", "Grammar", "Conventions"
    ]
    missing_targets = [c for c in target_cols if c not in df.columns]
    if missing_targets:
        raise ValueError(
            f"ECLIPSE CSV missing target columns {missing_targets}. "
            f"Available columns: {list(df.columns)}"
        )

    records: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        rec = {
            "task_type": "eclipse",
            "prompt": str(r["prompt"]) if not pd.isna(r["prompt"]) else "",
            "essay": str(r["full_text"]) if not pd.isna(r["full_text"]) else "",
        }
        for t in target_cols:
            rec[t] = _to_float(r[t])
        records.append(rec)

    return records
