import os
import nibabel as nib
import numpy as np
import torch

from standalone_inlined.svr.registration import _build_sms_groups

# Helper to load stack and extract SMS metadata
def load_stack_with_sms_metadata(stack_path):
    img = nib.load(stack_path)
    data = img.get_fdata()
    affine = img.affine
    # Try to load SMS metadata from JSON sidecar
    json_path = stack_path.replace('.nii.gz', '.json').replace('.nii', '.json')
    mb_factor = 1
    acquisition_order = None
    if os.path.exists(json_path):
        import json
        with open(json_path, 'r') as f:
            metadata = json.load(f)
            mb_factor = metadata.get('mb_factor', 1)
            acquisition_order = metadata.get('acquisition_order', None)
    return data, affine, mb_factor, acquisition_order

# Validate SVR registration for SMS stacks
def validate_sms_registration(stack_paths, svr_transform_path):
    # Load SVR transforms (assume saved as numpy array or torch tensor)
    transforms = np.load(svr_transform_path)
    # For each stack, check SMS group transforms
    for stack_path in stack_paths:
        data, affine, mb_factor, acq_order = load_stack_with_sms_metadata(stack_path)
        nz = data.shape[2]
        print(f"Stack: {stack_path}, mb_factor: {mb_factor}, acquisition_order: {acq_order}")
        if mb_factor > 1:
            # Build SMS groups using registration helper for consistency
            groups = _build_sms_groups(nz, mb_factor, acq_order)
            for group in groups:
                if len(group) > 1:
                    group_transforms = transforms[group]
                    # Check if all transforms in group are (almost) equal
                    diffs = np.max(np.abs(group_transforms - group_transforms.mean(axis=0)), axis=0)
                    print(f"SMS group {group}: max transform diff = {diffs}")
                    if np.all(diffs < 1e-3):
                        print("PASS: All SMS group transforms are equal (as expected)")
                    else:
                        print("FAIL: SMS group transforms differ (unexpected)")
        else:
            print("Non-SMS stack: no group validation needed.")

if __name__ == "__main__":
    # Usage: validate SVR transforms for SMS stacks
    # The transforms will be saved by SVR in the output directory
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python validate_svr_sms.py <svr_output_dir> <stack1.nii.gz> [stack2.nii.gz ...]")
        print("Example: python validate_svr_sms.py test_data/39/sim_stacks_sms test_data/39/sim_stacks_sms/sim_stack_*.nii.gz")
        sys.exit(1)
    
    svr_output_dir = sys.argv[1]
    stack_paths = sys.argv[2:]
    
    # Look for transform files in the SVR output directory
    # Check for final SVR transforms first (after SMS averaging)
    import glob
    svr_final_path = os.path.join(svr_output_dir, "svr/transforms_svr_final.npy")
    
    if os.path.exists(svr_final_path):
        print(f"Found final SVR transforms: {svr_final_path}")
        print("These transforms include SMS averaging applied during SVR reconstruction.\n")
        
        # Load all transforms (concatenated from all stacks)
        all_transforms = np.load(svr_final_path)
        print(f"Total transforms shape: {all_transforms.shape}")
        
        # Process each stack and extract its transforms
        slice_offset = 0
        for stack_idx, stack_path in enumerate(stack_paths):
            data, affine, mb_factor, acq_order = load_stack_with_sms_metadata(stack_path)
            nz = data.shape[2]
            
            # Extract transforms for this stack
            stack_transforms = all_transforms[slice_offset:slice_offset + nz]
            slice_offset += nz
            
            print(f"\nStack {stack_idx}: {stack_path}, mb_factor: {mb_factor}, acquisition_order: {acq_order}")
            print(f"  Number of slices: {nz}, transforms shape: {stack_transforms.shape}")
            
            if mb_factor > 1:
                groups = _build_sms_groups(nz, mb_factor, acq_order)
                for group in groups:
                    if len(group) > 1:
                        group_transforms = stack_transforms[group]
                        # Check if all transforms in group are (almost) equal
                        diffs = np.max(np.abs(group_transforms - group_transforms.mean(axis=0)), axis=0)
                        print(f"  SMS group {group}: max transform diff = {diffs}")
                        if np.all(diffs < 1e-3):
                            print("  ✓ PASS: All SMS group transforms are equal (as expected)")
                        else:
                            print("  ✗ FAIL: SMS group transforms differ (unexpected)")
            else:
                print("  Non-SMS stack: no group validation needed.")
    else:
        # Fallback to SVoRT transforms (before SMS averaging)
        print("Final SVR transforms not found. Checking SVoRT transforms (before SMS averaging)...")
        transform_files = glob.glob(os.path.join(svr_output_dir, "SVR*/out/tmp/svort/transforms_out_*.npy"))
        
        if not transform_files:
            # Try svort subdirectory
            transform_files = glob.glob(os.path.join(svr_output_dir, "svort/transforms_out_*.npy"))
        
        if not transform_files:
            # Try transforms directly in the directory
            transform_files = glob.glob(os.path.join(svr_output_dir, "transforms_out_*.npy"))
        
        if not transform_files:
            print(f"No transform files found in {svr_output_dir}")
            print("Looking for transforms in current directory...")
            transform_files = glob.glob("transforms_out_*.npy")
        
        if transform_files:
            print(f"Found {len(transform_files)} SVoRT transform files (before SMS averaging)")
            for i, tf in enumerate(sorted(transform_files)):
                print(f"\nValidating stack {i} with transforms from {tf}")
                if i < len(stack_paths):
                    validate_sms_registration([stack_paths[i]], tf)
        else:
            print("ERROR: No transform files found. Please ensure SVR has completed and saved transforms.")
