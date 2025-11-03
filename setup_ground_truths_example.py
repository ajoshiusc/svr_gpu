#!/usr/bin/env python3
"""
Example script to set up ground truth folder for SMS validation.

This script demonstrates how to organize ground truth NIfTI files 
for the comprehensive SMS validation notebook.

Usage:
    python setup_ground_truths_example.py

Expected folder structure:
    test_data/ground_truths/
        brain_01.nii.gz
        brain_02.nii.gz
        brain_03.nii.gz
        ...
"""

import shutil
from pathlib import Path

# Configuration
GROUND_TRUTH_DIR = Path("test_data/ground_truths")
EXAMPLE_SOURCE = Path("test_data/39/sim_stacks_sms/svr_output.nii.gz")

def setup_ground_truths():
    """Set up example ground truth folder structure."""
    
    # Create ground truth directory
    GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {GROUND_TRUTH_DIR}")
    
    # Example: Copy an existing file as a sample
    # In practice, you would copy your actual ground truth volumes here
    if EXAMPLE_SOURCE.exists():
        dest = GROUND_TRUTH_DIR / "brain_example_01.nii.gz"
        if not dest.exists():
            shutil.copy(EXAMPLE_SOURCE, dest)
            print(f"Copied example file to: {dest}")
        else:
            print(f"Example file already exists: {dest}")
    else:
        print(f"Warning: Example source not found: {EXAMPLE_SOURCE}")
    
    print("\n" + "="*80)
    print("INSTRUCTIONS FOR ADDING YOUR GROUND TRUTH FILES")
    print("="*80)
    print(f"\n1. Place your ground truth NIfTI files in: {GROUND_TRUTH_DIR}")
    print("   - Supported formats: .nii.gz, .nii")
    print("   - Use descriptive filenames (e.g., subject_01.nii.gz)")
    print("\n2. Requirements for ground truth volumes:")
    print("   - High-quality 3D brain volumes")
    print("   - Isotropic or near-isotropic resolution recommended")
    print("   - Sufficient SNR for realistic simulation")
    print("   - Consistent orientation (preferably)")
    print("\n3. The validation notebook will:")
    print("   - Automatically detect all NIfTI files in the folder")
    print("   - Run experiments on each ground truth independently")
    print("   - Compute statistics across all ground truths")
    print("   - Compare results between different volumes")
    print("\n4. Example commands to add your files:")
    print(f"   cp /path/to/your/brain_01.nii.gz {GROUND_TRUTH_DIR}/")
    print(f"   cp /path/to/your/brain_02.nii.gz {GROUND_TRUTH_DIR}/")
    print("\n5. Recommended: Start with 2-3 ground truths for initial validation")
    print("   - More ground truths = better statistics but longer runtime")
    print("   - Each ground truth runs 64 experiments (4 motions × 4 MB × 4 stacks)")
    print("\n" + "="*80)

if __name__ == "__main__":
    setup_ground_truths()
    
    # List current ground truth files
    gt_files = list(GROUND_TRUTH_DIR.glob("*.nii.gz")) + list(GROUND_TRUTH_DIR.glob("*.nii"))
    
    print(f"\nCurrent ground truth files ({len(gt_files)}):")
    if gt_files:
        for i, f in enumerate(sorted(gt_files), 1):
            print(f"  {i}. {f.name}")
    else:
        print("  (none found)")
    
    print(f"\nReady to run validation notebook!")
    print(f"The notebook will process {len(gt_files)} ground truth volume(s).")
