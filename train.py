"""
HeliBit-AI Universal Training Script
Trains HeliBitEngine on 'dataset.json', evaluates accuracy, and saves trained topological checkpoint.
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
from dataset import load_dataset_from_json


def print_banner():
    banner = """
==============================================================================
               HELIBIT-AI: UNIVERSAL LOCAL HEBBIAN TRAINING                  
   No-Backpropagation Local Dynamical Adaptation & Checkpoint Persistence     
==============================================================================
"""
    print(banner)


def run_training(epochs: int = 15):
    print_banner()
    
    # Reload fresh dataset from dataset.json
    dataset_data = load_dataset_from_json("dataset.json")
    training_data = dataset_data["training"]
    validation_data = dataset_data["validation"]

    engine = HeliBitEngine()
    engine.reset()
    trainer = HeliBitTrainer(engine)
    
    print(f"Loaded Training Dataset   : {len(training_data)} samples")
    print(f"Loaded Validation Dataset : {len(validation_data)} samples\n")
    
    print(f"--- STARTING LOCAL HEBBIAN TRAINING ({epochs} EPOCHS) ---")
    print(f"{'Epoch':<8} | {'Accuracy (%)':<14} | {'Mean Resonance':<16} | {'Mean Hamming Dh':<18} | {'Elapsed (s)'}")
    print("-" * 75)
    
    for epoch in range(1, epochs + 1):
        stats = trainer.train_epoch(training_data)
        print(
            f"Epoch {epoch:<3} | {stats['accuracy']:<14.2f} | "
            f"{stats['mean_resonance_force']:<16.4f} | {stats['mean_hamming_distance']:<18.4f} | "
            f"{stats['elapsed_sec']:<8.4f}"
        )
        
    checkpoint_file = "helibit_model.json"
    saved_path = trainer.save_checkpoint(checkpoint_file)
    print(f"\nSaved trained topological checkpoint to: '{saved_path}'")
    
    # Validation Evaluation
    if validation_data:
        print("\n--- EVALUATING VALIDATION DATASET (UNSEEN PROMPTS FROM dataset.json) ---")
        print(f"{'Input Prompt':<35} | {'Target':<22} | {'Predicted':<22} | {'Status'}")
        print("-" * 90)
        
        val_correct = 0
        candidate_targets = list(engine.known_targets)
        for text, target in validation_data:
            res = engine.predict(text, candidate_targets=candidate_targets)
            status = "PASS" if res.predicted_target == target else "FAIL"
            if status == "PASS":
                val_correct += 1
            print(f"{text:<35} | {target:<22} | {res.predicted_target:<22} | {status}")
            
        val_acc = (val_correct / len(validation_data)) * 100.0
        print("-" * 90)
        print(f"Validation Accuracy: {val_acc:.2f}% ({val_correct}/{len(validation_data)})")
    
    # Reload Verification
    print("\n--- CHECKPOINT RELOAD VERIFICATION ---")
    new_engine = HeliBitEngine()
    new_trainer = HeliBitTrainer(new_engine)
    new_trainer.load_checkpoint(checkpoint_file)
    print(f"Successfully reloaded checkpoint with {len(new_engine.token_states)} token states and {len(new_engine.known_targets)} target classes.")
    print("\n==============================================================================")
    print("                    TRAINING & VALIDATION COMPLETE                            ")
    print("==============================================================================\n")


if __name__ == "__main__":
    run_training(epochs=15)
