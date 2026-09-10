"""
HeliBit-AI v2 Training Script
Trains HeliBitEngine on 'dataset.json', evaluates seen and unseen splits,
and saves trained topological checkpoint.
"""

import sys
import io
import json
import os

# Ensure UTF-8 output encoding for Windows command line compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from helibit import HeliBitEngine, HeliBitTrainer
from dataset import (
    TRAINING_DATASET,
    COMPOSITIONAL_TEST_DATASET,
    ADVERSARIAL_TEST_DATASET,
)


def print_banner():
    banner = """
==============================================================================
                HELIBIT-AI v2: NEUROMORPHIC COMPOSITIONAL TRAINING             
       Zero-FPU Arithmetic | Structured Bitboard Roles | Hebbian Dynamics     
==============================================================================
"""
    print(banner)


def run_training(epochs: int = 5):
    print_banner()

    engine = HeliBitEngine()
    engine.reset()
    trainer = HeliBitTrainer(engine)

    print(f"Loaded Training Dataset          : {len(TRAINING_DATASET)} samples")
    print(f"Loaded Compositional Test Dataset: {len(COMPOSITIONAL_TEST_DATASET)} samples")
    print(f"Loaded Adversarial Test Dataset  : {len(ADVERSARIAL_TEST_DATASET)} samples\n")

    print(f"--- STARTING TRAINING ({epochs} EPOCHS) ---")
    print(f"{'Epoch':<8} | {'Accuracy (%)':<14} | {'Mean Resonance':<16} | {'Mean Hamming Dh':<18} | {'Elapsed (s)'}")
    print("-" * 75)

    for epoch in range(1, epochs + 1):
        stats = trainer.train_epoch(TRAINING_DATASET)
        print(
            f"Epoch {epoch:<3} | {stats['accuracy']:<14.2f} | "
            f"{stats['mean_resonance_force']:<16.4f} | {stats['mean_hamming_distance']:<18.4f} | "
            f"{stats['elapsed_sec']:<8.4f}"
        )

    checkpoint_file = "helibit_model.json"
    saved_path = trainer.save_checkpoint(checkpoint_file)
    print(f"\nSaved trained topological checkpoint to: '{saved_path}'")

    # Evaluate Compositional Generalization
    print("\n--- EVALUATING UNSEEN COMPOSITIONAL GENERALIZATION ---")
    print(f"{'Input Expression':<28} | {'Expected':<10} | {'Predicted':<10} | {'Status'}")
    print("-" * 65)

    comp_correct = 0
    for text, expected in COMPOSITIONAL_TEST_DATASET:
        res = engine.predict(text)
        status = "PASS" if res.predicted_target == expected else "FAIL"
        if status == "PASS":
            comp_correct += 1
        print(f"{text:<28} | {expected:<10} | {res.predicted_target:<10} | {status}")

    comp_acc = (comp_correct / len(COMPOSITIONAL_TEST_DATASET)) * 100.0
    print("-" * 65)
    print(f"Compositional Generalization Accuracy: {comp_acc:.2f}% ({comp_correct}/{len(COMPOSITIONAL_TEST_DATASET)})")

    # Evaluate Adversarial Rejection
    print("\n--- EVALUATING ADVERSARIAL REJECTION (ABSTENTION GUARD) ---")
    print(f"{'Input Query':<28} | {'Expected':<10} | {'Predicted':<10} | {'Status'}")
    print("-" * 65)

    adv_correct = 0
    for text, expected in ADVERSARIAL_TEST_DATASET:
        res = engine.predict(text)
        status = "PASS" if res.predicted_target == expected else "FAIL"
        if status == "PASS":
            adv_correct += 1
        print(f"{text:<28} | {expected:<10} | {res.predicted_target:<10} | {status}")

    adv_acc = (adv_correct / len(ADVERSARIAL_TEST_DATASET)) * 100.0
    print("-" * 65)
    print(f"Adversarial Rejection Accuracy: {adv_acc:.2f}% ({adv_correct}/{len(ADVERSARIAL_TEST_DATASET)})")

    # Checkpoint Reload Verification
    print("\n--- CHECKPOINT RELOAD VERIFICATION ---")
    new_engine = HeliBitEngine()
    new_trainer = HeliBitTrainer(new_engine)
    new_trainer.load_checkpoint(checkpoint_file)
    print(f"Successfully reloaded checkpoint with {len(new_engine.token_states)} token states and {len(new_engine.role_prototypes)} role prototypes.")
    print("\n==============================================================================")
    print("                    TRAINING & VALIDATION COMPLETE                            ")
    print("==============================================================================\n")


if __name__ == "__main__":
    run_training(epochs=5)
