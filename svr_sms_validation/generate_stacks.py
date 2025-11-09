#!/usr/bin/env python3
"""
Generate simulated stacks with various motion levels and SMS configurations.
"""

import sys
from pathlib import Path
from itertools import product

# Import the generate_simulated_stacks function
sys.path.insert(0, str(Path(__file__).parent.parent))
from simstack_scripts.simulate_stacks import generate_simulated_stacks

# Configuration
MOTION_LEVELS = {
    'none': {'max_rot_deg': 0.0, 'max_trans_mm': 0.0, 'max_disp': 0.0},
    'mild': {'max_rot_deg': 1.0, 'max_trans_mm': 0.5, 'max_disp': 2.0},
    'moderate': {'max_rot_deg': 3.0, 'max_trans_mm': 1.0, 'max_disp': 5.0},
    'severe': {'max_rot_deg': 5.0, 'max_trans_mm': 2.0, 'max_disp': 10.0}
}

MB_FACTORS = [1, 2, 3]
NUM_STACKS_OPTIONS = [12]
SLICE_THICKNESS = 3.0
INPLANE_RESOLUTION = 0.8

GROUND_TRUTH_DIR = "/home/ajoshi/Projects/svr_gpu/test_data/ground_truths"
OUTPUT_DIR = "/home/ajoshi/Projects/svr_gpu/test_data/sms_stacks_generated2"


def main():
    # Find ground truth files
    ground_truth_files = sorted(Path(GROUND_TRUTH_DIR).glob("*.nii.gz"))
    if not ground_truth_files:
        ground_truth_files = sorted(Path(GROUND_TRUTH_DIR).glob("*.nii"))
    
    if not ground_truth_files:
        print(f"ERROR: No NIfTI files found in {GROUND_TRUTH_DIR}")
        sys.exit(1)
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print("Stack Generation Configuration")
    print("="*80)
    print(f"Ground truth files: {len(ground_truth_files)}")
    for gt_file in ground_truth_files:
        print(f"  - {gt_file.name}")
    print(f"Motion levels: {list(MOTION_LEVELS.keys())}")
    print(f"MB factors: {MB_FACTORS}")
    print(f"Stack counts: {NUM_STACKS_OPTIONS}")
    print(f"Slice thickness: {SLICE_THICKNESS} mm")
    print(f"In-plane resolution: {INPLANE_RESOLUTION} mm")
    print(f"Output directory: {OUTPUT_DIR}")
    
    print("="*80 + "\n")
    
    results = []
    
    for gt_idx, gt_path in enumerate(ground_truth_files, 1):
        gt_name = gt_path.stem.replace('.nii', '')
        print(f"\n[{gt_idx}/{len(ground_truth_files)}] Processing {gt_path.name}")
        
        for motion_level, mb_factor, num_stacks in product(MOTION_LEVELS.keys(), MB_FACTORS, NUM_STACKS_OPTIONS):
            motion_params = MOTION_LEVELS[motion_level]
            
            print(f"  {motion_level:10s} MB={mb_factor} n={num_stacks:2d}  ", end=" ")
            
            # Create output dir with parameter-based naming
            output_subdir = Path(OUTPUT_DIR) / gt_name / f"motion_{motion_level}_mb{mb_factor}_stacks{num_stacks}"
            
            # Generate stacks
            generate_simulated_stacks(
                mri_path=str(gt_path),
                out_dir=str(output_subdir),
                n_stacks=num_stacks,
                slices_per_stack=40,
                mb_factor=mb_factor,
                max_rot_deg=motion_params['max_rot_deg'],
                max_trans_mm=motion_params['max_trans_mm'],
                max_disp=motion_params['max_disp'],
                slice_thickness=SLICE_THICKNESS,
                inplane_res=INPLANE_RESOLUTION,
                noise_std=0.02,
                enable_nonlinear=False
            )
            
            # Verify
            stack_files = list(output_subdir.glob("sim_stack_*.nii.gz"))
            status = 'success' if len(stack_files) == num_stacks else 'incomplete'
            results.append({'status': status, 'num_stacks': len(stack_files)})
            print(f"{status.upper()} ({len(stack_files)}/{num_stacks})")
    
    print("\n" + "="*80)
    print("Generation Summary")
    print("="*80)
    print(f"Total sets: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Incomplete: {sum(1 for r in results if r['status'] == 'incomplete')}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
