"""
run_pipeline.py — Master pipeline runner.

Executes all 7 steps in sequence with clear progress reporting.
Each step can also be run independently.

Usage:
  python run_pipeline.py              # run all steps
  python run_pipeline.py --from 3    # resume from step 3
  python run_pipeline.py --only 5    # run only step 5
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path


STEPS = {
    1: ("build_graph",           "Download OSM road network"),
    2: ("map_data_to_graph",     "Enrich graph with spatial datasets"),
    3: ("feature_engineering",   "Engineer graph features"),
    4: ("signal_intelligence",   "Synthetic signal delay estimation"),
    5: ("train_congestion_model","Train ML congestion model (weak supervision)"),
    6: ("predict_congestion",    "Apply ML predictions to graph"),
    7: ("bottleneck_analysis",   "Bottleneck, hotspot & intervention analysis"),
    8: ("emergency_routing",     "Emergency routing analysis"),
    9: ("urban_optimization",    "City-level urban optimization"),
}


def run_step(step_num: int):
    module_name, description = STEPS[step_num]
    print(f"\n{'='*65}")
    print(f"STEP {step_num}: {description.upper()}")
    print(f"{'='*65}")
    t0 = time.time()
    try:
        import importlib
        mod = importlib.import_module(module_name)
        mod.main()
        elapsed = time.time() - t0
        print(f"\n  ✓ Step {step_num} completed in {elapsed:.1f}s")
        return True
    except Exception as e:
        print(f"\n  ✗ Step {step_num} FAILED: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Bangalore Traffic Pipeline")
    parser.add_argument("--from", dest="from_step", type=int, default=1,
                        help="Start from this step (default: 1)")
    parser.add_argument("--only", dest="only_step", type=int, default=None,
                        help="Run only this step")
    args = parser.parse_args()

    if args.only_step:
        steps_to_run = [args.only_step]
    else:
        steps_to_run = list(range(args.from_step, max(STEPS.keys()) + 1))

    print("\n" + "="*65)
    print("BANGALORE TRAFFIC CONGESTION MODELING PIPELINE")
    print("Graph-Based Urban Traffic Analysis using OSM + ML")
    print("="*65)
    print(f"\nSteps to run: {steps_to_run}")
    print("\n⚠ DATA TRANSPARENCY NOTICE:")
    print("  • Real data: OSM network, BTP accidents, violations, enforcement")
    print("  • Proxy data: Area-level traffic density (Bangalore dataset)")
    print("  • Synthetic: Signal delays (topology-estimated), temporal features")
    print("  • Pseudo-labels: Congestion targets (weak supervision)")

    results = {}
    total_start = time.time()

    for step in steps_to_run:
        if step not in STEPS:
            print(f"  ⚠ Step {step} not defined — skipping.")
            continue
        success = run_step(step)
        results[step] = success
        if not success and step <= 6:
            print(f"\n  Pipeline halted at step {step} (critical failure).")
            break

    total_elapsed = time.time() - total_start
    print(f"\n{'='*65}")
    print(f"PIPELINE COMPLETE — Total time: {total_elapsed:.1f}s")
    print(f"{'='*65}")
    for step, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} Step {step}: {STEPS[step][1]}")


if __name__ == "__main__":
    main()
