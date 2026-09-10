"""
HeliBit-AI: Compositional Generalization & Adversarial Benchmark
Evaluates the v2 neuro-symbolic engine across 3 separate rigor splits:
  (a) Exact-match on Seen Training Data
  (b) Compositional Generalization on Unseen Valid Combinations
  (c) Explicit Abstention on Adversarial / Malformed Inputs ('UNKNOWN')
"""

import sys
import time
from typing import List, Tuple
from helibit.engine import HeliBitEngine
from helibit.trainer import HeliBitTrainer
from dataset import TRAINING_DATASET, COMPOSITIONAL_TEST_DATASET, ADVERSARIAL_TEST_DATASET


def evaluate_split(
    engine: HeliBitEngine,
    dataset: List[Tuple[str, str]],
    split_name: str,
    verbose: bool = True,
) -> Tuple[float, float, int, int]:
    """
    Evaluates engine on a given dataset split.
    Returns (accuracy_pct, avg_latency_us, correct_count, total_count).
    """
    total = len(dataset)
    if total == 0:
        return (0.0, 0.0, 0, 0)

    correct = 0
    total_latency_us = 0.0

    if verbose:
        print(f"\n======================================================================")
        print(f" EVALUATING SPLIT: {split_name} ({total} samples)")
        print(f"======================================================================")

    for text, expected in dataset:
        res = engine.predict(text)
        total_latency_us += res.latency_us
        is_correct = res.predicted_target == expected
        if is_correct:
            correct += 1

        status = "PASS" if is_correct else "FAIL"
        if verbose:
            print(
                f"  [{status}] Input: '{text:<24}' | Expected: {expected:<8} | "
                f"Predicted: {res.predicted_target:<8} | Conf: {res.confidence:.2f} | Latency: {res.latency_us:.1f}us"
            )

    accuracy = (correct / total) * 100.0
    avg_latency = total_latency_us / total

    return (accuracy, avg_latency, correct, total)


def run_benchmark() -> bool:
    print("Initializing HeliBit-AI v2 Neuromorphic Engine...")
    engine = HeliBitEngine()
    trainer = HeliBitTrainer(engine)

    print(f"Running Hebbian training across {len(TRAINING_DATASET)} training samples (3 epochs)...")
    trainer.train(TRAINING_DATASET, epochs=3)

    # Save trained checkpoint
    checkpoint_path = trainer.save_checkpoint("helibit_model.json")
    print(f"Model checkpoint persisted to: {checkpoint_path}")

    # Evaluate 3 distinct splits
    train_acc, train_lat, train_c, train_n = evaluate_split(
        engine, TRAINING_DATASET, "Split (a): Training Set (Seen Combinations)"
    )

    comp_acc, comp_lat, comp_c, comp_n = evaluate_split(
        engine, COMPOSITIONAL_TEST_DATASET, "Split (b): Compositional Generalization (Unseen Valid Combinations)"
    )

    adv_acc, adv_lat, adv_c, adv_n = evaluate_split(
        engine, ADVERSARIAL_TEST_DATASET, "Split (c): Adversarial Rejection (Invalid / Malformed Inputs)"
    )

    # Overall Summary Report
    print("\n" + "=" * 70)
    print(" HELIBIT-AI v2 BENCHMARK REPORT (HELIBIT_ARCHITECTURE_V2.md §8)")
    print("=" * 70)
    print(f"  (a) Seen Training Exact-Match:            {train_acc:6.2f}% ({train_c}/{train_n}) [avg latency: {train_lat:5.1f}us]")
    print(f"  (b) Unseen Compositional Generalization:  {comp_acc:6.2f}% ({comp_c}/{comp_n}) [avg latency: {comp_lat:5.1f}us]")
    print(f"  (c) Adversarial / Invalid Abstention:    {adv_acc:6.2f}% ({adv_c}/{adv_n}) [avg latency: {adv_lat:5.1f}us]")
    print("=" * 70)

    # Acceptance criteria: all three must achieve 100%
    all_passed = (train_acc == 100.0) and (comp_acc == 100.0) and (adv_acc == 100.0)
    if all_passed:
        print("\n [SUCCESS] ALL 3 BENCHMARK SPLITS PASSED WITH ZERO ERRORS.")
    else:
        print("\n [FAILURE] Some benchmark tests failed.")

    return all_passed


if __name__ == "__main__":
    success = run_benchmark()
    sys.exit(0 if success else 1)
