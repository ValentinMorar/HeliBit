"""
HeliBit-AI Dataset Loader Module
Dynamically loads multi-domain training and validation datasets from 'dataset.json'.
"""

import os
import json
from typing import List, Tuple, Dict, Any

DATASET_FILE = "dataset.json"


def load_dataset_from_json(filepath: str = DATASET_FILE) -> Dict[str, List[Tuple[str, str]]]:
    """
    Loads training and validation dataset pairs dynamically from a JSON file.
    Returns dict: {'training': [(input, target), ...], 'validation': [(input, target), ...]}
    """
    abs_path = os.path.abspath(filepath)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Dataset file '{filepath}' not found at path: {abs_path}")

    with open(abs_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    training_pairs = []
    for item in data.get("training", []):
        inp = item.get("input", "").strip()
        tgt = item.get("target", "").strip()
        if inp and tgt:
            training_pairs.append((inp, tgt))

    validation_pairs = []
    for item in data.get("validation", []):
        inp = item.get("input", "").strip()
        tgt = item.get("target", "").strip()
        if inp and tgt:
            validation_pairs.append((inp, tgt))

    return {
        "training": training_pairs,
        "validation": validation_pairs,
    }


# Convenience exports
_data = load_dataset_from_json()
TRAINING_DATASET: List[Tuple[str, str]] = _data["training"]
VALIDATION_DATASET: List[Tuple[str, str]] = _data["validation"]
