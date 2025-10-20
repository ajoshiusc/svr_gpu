#!/usr/bin/env python3
import sys
import nibabel as nib
import numpy as np

if len(sys.argv) < 2:
    print("Usage: check_stack_nonzero.py stack1.nii.gz [stack2.nii.gz ...]")
    sys.exit(1)

for stack_path in sys.argv[1:]:
    img = nib.load(stack_path)
    data = img.get_fdata()
    print(f"\nStack: {stack_path}")
    print(f"  Shape: {data.shape}")
    nz_per_slice = [(i, np.count_nonzero(data[..., i])) for i in range(data.shape[-1])]
    for i, nz in nz_per_slice:
        print(f"    Slice {i:2d}: {nz} nonzero voxels")
    print(f"  Total nonzero voxels: {np.count_nonzero(data)}")
