#!/usr/bin/env python3
"""
Reorient NIfTI stacks so that slices are always in the XY plane and stacked along Z axis.

This script uses SimpleITK to match the exact logic from rotate2makeslice_z.py

This script:
1. Detects which axis has the slice thickness (largest voxel size)
2. Reorients the volume so slices are stacked along Z (3rd dimension)
3. Saves reoriented stacks to output directory

Usage:
    python reorient_stacks.py --input stack1.nii.gz stack2.nii.gz --output-dir output/reoriented/
"""

import argparse
import os
import numpy as np
import SimpleITK as sitk
from pathlib import Path


def reorient_to_axial(input_path: str, output_path: str, verbose: bool = True):
    """
    Reorient a NIfTI volume so slices are in XY plane and stacked along Z.
    Uses SimpleITK to match the exact logic from rotate2makeslice_z.py
    
    Args:
        input_path: Path to input NIfTI file
        output_path: Path to save reoriented NIfTI file
        verbose: Print information about the reorientation
    """
    # Load the image using SimpleITK
    img = sitk.ReadImage(input_path)
    
    # Get original spacing (resolutions)
    spacing = img.GetSpacing()
    
    if verbose:
        print(f"\nProcessing: {os.path.basename(input_path)}")
        print(f"  Original spacing: {spacing}")
    
    # Detect slice-stacking axis (dimension with largest voxel size)
    # This matches: sliceaxis = np.argmax(img.GetSpacing())
    slice_axis = np.argmax(spacing)
    
    if verbose:
        print(f"  Slice axis detected: {slice_axis} ({['X','Y','Z'][slice_axis]}, thickness={spacing[slice_axis]:.3f}mm)")
    
    # Reorient using PermuteAxes to match rotate2makeslice_z.py exactly:
    # if sliceaxis == 0: img2 = sitk.PermuteAxes(img, [1,2,0])
    # if sliceaxis == 1: img2 = sitk.PermuteAxes(img, [2,0,1])
    # if sliceaxis == 2: img2 = img
    
    if slice_axis == 0:  # X is slice direction (sagittal)
        img2 = sitk.PermuteAxes(img, [1, 2, 0])
        if verbose:
            print(f"  Reorienting: PermuteAxes([1,2,0])")
    elif slice_axis == 1:  # Y is slice direction (coronal)
        img2 = sitk.PermuteAxes(img, [2, 0, 1])
        if verbose:
            print(f"  Reorienting: PermuteAxes([2,0,1])")
    else:  # slice_axis == 2, Z is already slice direction (axial)
        img2 = img
        if verbose:
            print(f"  Already in axial orientation, no reorientation needed")
    
    if verbose:
        print(f"  Reoriented spacing: {img2.GetSpacing()}")
    
    # Save the reoriented image
    sitk.WriteImage(img2, output_path)
    
    if verbose:
        print(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Reorient NIfTI stacks so slices are in XY plane and stacked along Z axis'
    )
    parser.add_argument('--input', '--input-stacks', dest='input_stacks', required=True, nargs='+',
                        help='Input NIfTI files to reorient')
    parser.add_argument('--output-dir', default='output/reoriented',
                        help='Directory to save reoriented files (default: output/reoriented)')
    parser.add_argument('--prefix', default='r',
                        help='Prefix for reoriented filenames (default: "r")')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reorienting {len(args.input_stacks)} stacks...")
    print(f"Output directory: {output_dir}")
    
    # Process each input file
    reoriented_files = []
    for input_path in args.input_stacks:
        # Generate output filename
        input_filename = os.path.basename(input_path)
        output_filename = f"{args.prefix}{input_filename}"
        output_path = output_dir / output_filename
        
        # Reorient and save
        reorient_to_axial(input_path, str(output_path), verbose=not args.quiet)
        reoriented_files.append(str(output_path))
    
    print(f"\n✓ Reoriented {len(reoriented_files)} files")
    print(f"\nReoriented files:")
    for f in reoriented_files:
        print(f"  {f}")
    
    print(f"\nYou can now run SVR with these reoriented stacks:")
    print(f"  python svr_cli.py --input-stacks {' '.join(reoriented_files)} --output output/result.nii.gz --segmentation twai")


if __name__ == '__main__':
    main()
