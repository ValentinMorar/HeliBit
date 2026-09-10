"""
HeliBit-AI Dataset Loader Module
Loads training, compositional generalization, and adversarial test splits
from 'dataset.json' for HeliBit-AI v2.
"""

import os
import json
from typing import List, Tuple, Dict, Any

DATASET_FILE = "dataset.json"


def load_dataset_from_json(filepath: str = DATASET_FILE) -> Dict[str, List[Tuple[str, str]]]:
    """
    Loads training, compositional_test, and adversarial_test dataset splits from JSON.
    Returns dict:
      {
        'training': [(input, target), ...],
        'compositional_test': [(input, target), ...],
        'adversarial_test': [(input, target), ...],
        'validation': [(input, target), ...]
      }
    """
    abs_path = os.path.abspath(filepath)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Dataset file '{filepath}' not found at path: {abs_path}")

    with open(abs_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def extract_pairs(key: str) -> List[Tuple[str, str]]:
        pairs = []
        for item in data.get(key, []):
            inp = item.get("input", "").strip()
            tgt = item.get("target", "").strip()
            if inp and tgt:
                pairs.append((inp, tgt))
        return pairs

    training_pairs = extract_pairs("training")
    compositional_pairs = extract_pairs("compositional_test")
    adversarial_pairs = extract_pairs("adversarial_test")
    # For backward compatibility: validation contains compositional + adversarial
    validation_pairs = extract_pairs("validation") or (compositional_pairs + adversarial_pairs)

    return {
        "training": training_pairs,
        "compositional_test": compositional_pairs,
        "adversarial_test": adversarial_pairs,
        "validation": validation_pairs,
    }


# Convenience exports
_data = load_dataset_from_json()
TRAINING_DATASET: List[Tuple[str, str]] = _data["training"]
COMPOSITIONAL_TEST_DATASET: List[Tuple[str, str]] = _data["compositional_test"]
ADVERSARIAL_TEST_DATASET: List[Tuple[str, str]] = _data["adversarial_test"]
VALIDATION_DATASET: List[Tuple[str, str]] = _data["validation"]
