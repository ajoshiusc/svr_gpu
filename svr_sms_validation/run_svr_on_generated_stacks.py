#!/usr/bin/env python3
"""
Run SVR reconstruction on all generated SMS stacks.
"""

import sys
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent))
from svr_cli import main as svr_main

# Configuration
MOTION_LEVELS = ['none', 'mild', 'moderate', 'severe']
MB_FACTORS = [1, 2, 3]
NUM_STACKS_OPTIONS = [12]

GENERATED_STACKS_DIR = "/home/ajoshi/Projects/svr_gpu/test_data/sms_stacks_generated"
OUTPUT_DIR = "/home/ajoshi/Projects/svr_gpu/test_data/svr_reconstructions"


def main():
    generated_dir = Path(GENERATED_STACKS_DIR)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Find all ground truth subdirectories
    gt_subdirs = sorted([d for d in generated_dir.iterdir() if d.is_dir()])
    
    print("SVR Reconstruction Configuration")
    print("="*80)
    print(f"Generated stacks directory: {GENERATED_STACKS_DIR}")
    print(f"Ground truth subjects: {len(gt_subdirs)}")
    for gt_dir in gt_subdirs:
        print(f"  - {gt_dir.name}")
    print(f"Motion levels: {MOTION_LEVELS}")
    print(f"MB factors: {MB_FACTORS}")
    print(f"Stack counts: {NUM_STACKS_OPTIONS}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*80 + "\n")
    
    results = []
    
    for gt_idx, gt_subdir in enumerate(gt_subdirs, 1):
        gt_name = gt_subdir.name
        print(f"\n[{gt_idx}/{len(gt_subdirs)}] Processing {gt_name}")
        
        for motion_level, mb_factor, num_stacks in product(MOTION_LEVELS, MB_FACTORS, NUM_STACKS_OPTIONS):
            stack_dir = gt_subdir / f"motion_{motion_level}_mb{mb_factor}_stacks{num_stacks}"
            
            if not stack_dir.exists():
                print(f"  {motion_level:10s} MB={mb_factor} n={num_stacks:2d}  SKIPPED (dir not found)")
                continue
            
            stack_files = sorted(stack_dir.glob("sim_stack_*.nii.gz"))
            if not stack_files:
                print(f"  {motion_level:10s} MB={mb_factor} n={num_stacks:2d}  SKIPPED (no stacks found)")
                continue
            
            print(f"  {motion_level:10s} MB={mb_factor} n={num_stacks:2d}  ", end=" ", flush=True)
            
            # Create output subdirectory
            output_subdir = Path(OUTPUT_DIR) / gt_name / f"motion_{motion_level}_mb{mb_factor}_stacks{num_stacks}"
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_subdir / "svr_recon.nii.gz"
            
            # Run SVR
            try:
                sys.argv = [
                    'svr_cli.py',
                    '--input-stacks', *[str(f) for f in stack_files],
                    '--output', str(output_file),
                    '--segmentation', 'threshold',
                    '--segmentation-threshold', '100'
                ]
                svr_main()
                results.append({'status': 'success'})
                print("OK")
            except Exception as e:
                results.append({'status': 'failed', 'error': str(e)[:100]})
                print(f"FAILED ({str(e)[:50]})")
    
    print("\n" + "="*80)
    print("SVR Reconstruction Summary")
    print("="*80)
    print(f"Total runs: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
