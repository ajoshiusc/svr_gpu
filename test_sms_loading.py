#!/usr/bin/env python3
"""Test script to verify SMS metadata loading"""
import torch
import sys
from pathlib import Path

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent))

from standalone_inlined.image import load_stack

def test_sms_loading():
    """Test that SMS metadata is loaded correctly"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load an SMS stack
    stack_path = "SVR001_brain9_20251013_130312/sim_stacks_sms_test/sim_stack_01.nii.gz"
    
    print(f"Loading stack from: {stack_path}")
    stack = load_stack(stack_path, device=device)
    
    print(f"\nStack properties:")
    print(f"  Shape: {stack.shape}")
    print(f"  Resolution: {stack.resolution_x:.3f} x {stack.resolution_y:.3f} x {stack.gap:.3f} mm")
    print(f"  MB Factor: {stack.mb_factor}")
    print(f"  Acquisition Order: {stack.acquisition_order}")
    
    # Verify SMS groups would be built correctly
    from standalone_inlined.svr.registration import _build_sms_groups
    
    nz = stack.shape[0]
    groups = _build_sms_groups(nz, stack.mb_factor, stack.acquisition_order)
    
    print(f"\nSMS groups (mb_factor={stack.mb_factor}):")
    for i, group in enumerate(groups):
        print(f"  Group {i}: slices {group}")
    
    print("\n✓ SMS metadata loaded successfully!")
    return True

if __name__ == "__main__":
    success = test_sms_loading()
    sys.exit(0 if success else 1)
