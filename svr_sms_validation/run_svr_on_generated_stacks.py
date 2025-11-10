#!/usr/bin/env python3
"""
Run SVR reconstruction on all generated SMS stacks.
"""

import os
import random
import subprocess
import sys
from pathlib import Path
from itertools import product

# Get the svr_cli.py path
SVR_CLI_PATH = "/home/ajoshi/Projects/svr_gpu/svr_cli.py"

if not os.path.exists(SVR_CLI_PATH):
    SVR_CLI_PATH = "/project2/ajoshi_1183/Projects/svr_gpu/svr_cli.py"

# Configuration
MOTION_LEVELS = ["none", "mild", "moderate", "severe"]
MB_FACTORS = [1, 2, 3]
NUM_STACKS_OPTIONS = [3, 6, 9, 12]
DEFAULT_PERMUTATIONS = 3
PERMUTATIONS_PER_STACK_COUNT = {
    3: 3,
    6: 3,
    9: 3,
    12: 3,
}
RANDOM_SEED = 1337


BASEPATH = "/home/ajoshi/project2_ajoshi_1183"
PYTHON_CMD = ["python"]

if not os.path.exists(BASEPATH):
    BASEPATH = "/project2/ajoshi_1183"
    PYTHON_CMD = ["sbatch", "python3gpu.job"]
GENERATED_STACKS_DIR = os.path.join(BASEPATH, "data/sms_sim_stacks_generated")
OUTPUT_DIR = os.path.join(BASEPATH, "data/svr_reconstructions")


def main():
    generated_dir = Path(GENERATED_STACKS_DIR)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    rng = random.Random(RANDOM_SEED)

    # Find all ground truth subdirectories
    gt_subdirs = sorted([d for d in generated_dir.iterdir() if d.is_dir()])

    print("SVR Reconstruction Configuration")
    print("=" * 80)
    print(f"Generated stacks directory: {GENERATED_STACKS_DIR}")
    print(f"Ground truth subjects: {len(gt_subdirs)}")
    for gt_dir in gt_subdirs:
        print(f"  - {gt_dir.name}")
    print(f"Motion levels: {MOTION_LEVELS}")
    print(f"MB factors: {MB_FACTORS}")
    print(f"Stack counts: {NUM_STACKS_OPTIONS}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80 + "\n")

    results = []

    for gt_idx, gt_subdir in enumerate(gt_subdirs, 1):
        gt_name = gt_subdir.name
        print(f"\n[{gt_idx}/{len(gt_subdirs)}] Processing {gt_name}")

        for motion_level, mb_factor in product(MOTION_LEVELS, MB_FACTORS):
            # All 12 stacks are in the same directory, so fetch them once
            stack_dir = gt_subdir / f"motion_{motion_level}_mb{mb_factor}_stacks12"

            if not stack_dir.exists():
                continue

            all_stack_files = sorted(stack_dir.glob("sim_stack_*.nii.gz"))
            if not all_stack_files or len(all_stack_files) < 12:
                continue

            # Now iterate over different num_stacks counts
            for num_stacks in NUM_STACKS_OPTIONS:
                perm_count = PERMUTATIONS_PER_STACK_COUNT.get(
                    num_stacks, DEFAULT_PERMUTATIONS
                )
                for perm_idx in range(1, perm_count + 1):
                    selected_files = sorted(rng.sample(all_stack_files, num_stacks))
                    print(
                        f"  {motion_level:10s} MB={mb_factor} n={num_stacks:2d} perm={perm_idx:02d}",
                        end="  ",
                        flush=True,
                    )

                    output_subdir = (
                        Path(OUTPUT_DIR)
                        / gt_name
                        / f"motion_{motion_level}_mb{mb_factor}_n{num_stacks}"
                        / f"perm_{perm_idx:02d}"
                    )
                    output_subdir.mkdir(parents=True, exist_ok=True)
                    output_file = output_subdir / "svr_recon.nii.gz"

                    cmd = [
                        *PYTHON_CMD,
                        SVR_CLI_PATH,
                        "--input-stacks",
                        *[str(f) for f in selected_files],
                        "--output",
                        str(output_file),
                        "--segmentation",
                        "threshold",
                        "--segmentation-threshold",
                        "100",
                        "--device",
                        "0",
                    ]
                    # make cmd as a single string for printing
                    cmd = " ".join(f'"{c}"' if " " in c else c for c in cmd)
                    #print(cmd_str)

                    if os.path.exists(output_file):
                        print("SKIP (already exists)")
                        results.append(
                            {
                                "status": "skipped",
                                "motion": motion_level,
                                "mb": mb_factor,
                                "n": num_stacks,
                                "perm": perm_idx,
                                "output": str(output_file),
                            }
                        )
                        continue

                    print("RUNNING...\n")
                    print(cmd)
                    print("\n...\n")

                    #subprocess.run(cmd, check=True, shell=True)
                    results.append(
                        {
                            "status": "success",
                            "motion": motion_level,
                            "mb": mb_factor,
                            "n": num_stacks,
                            "perm": perm_idx,
                            "output": str(output_file),
                        }
                    )
                    print("OK")

    print("\n" + "=" * 80)
    print("SVR Reconstruction Summary")
    print("=" * 80)
    print(f"Total runs: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
