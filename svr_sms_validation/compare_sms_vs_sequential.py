#!/usr/bin/env python3
"""
Compare SVR reconstruction quality between SMS and sequential acquisitions.

This script:
1. Simulates SMS stacks from a ground truth volume
2. Simulates sequential (non-SMS) stacks from the same ground truth
3. Runs SVR on both sets of stacks
4. Compares reconstruction quality metrics (PSNR, SSIM, etc.) against ground truth
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np
import nibabel as nib
from pathlib import Path


def compute_psnr(img1, img2, data_range=None):
    """Compute Peak Signal-to-Noise Ratio between two images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    if data_range is None:
        data_range = max(img1.max(), img2.max()) - min(img1.min(), img2.min())
    return 20 * np.log10(data_range / np.sqrt(mse))


def compute_ssim(img1, img2, data_range=None):
    """Compute Structural Similarity Index between two images."""
    try:
        from skimage.metrics import structural_similarity
        if data_range is None:
            data_range = max(img1.max(), img2.max()) - min(img1.min(), img2.min())
        return structural_similarity(img1, img2, data_range=data_range)
    except ImportError:
        print("Warning: skimage not available, SSIM not computed")
        return None


def compute_nrmse(img1, img2):
    """Compute Normalized Root Mean Squared Error."""
    rmse = np.sqrt(np.mean((img1 - img2) ** 2))
    data_range = max(img1.max(), img2.max()) - min(img1.min(), img2.min())
    return rmse / data_range if data_range > 0 else 0


def simulate_stacks(ground_truth_path, output_dir, num_stacks, mb_factor, motion_level="moderate"):
    """Simulate stacks from ground truth volume."""
    print(f"Simulating {num_stacks} stacks with mb_factor={mb_factor}...")
    
    # Map motion level to parameters
    motion_params = {
        "none": {"max_rot_deg": 0.0, "max_trans_mm": 0.0, "max_disp": 0.0},
        "mild": {"max_rot_deg": 1.0, "max_trans_mm": 0.5, "max_disp": 2.0},
        "moderate": {"max_rot_deg": 3.0, "max_trans_mm": 1.0, "max_disp": 5.0},
        "severe": {"max_rot_deg": 5.0, "max_trans_mm": 2.0, "max_disp": 10.0}
    }
    params = motion_params.get(motion_level, motion_params["moderate"])
    
    cmd = [
        sys.executable,
        "simstack_scripts/simulate_stacks.py",
        ground_truth_path,
        output_dir,
        "--n-stacks", str(num_stacks),
        "--mb-factor", str(mb_factor),
        "--max-rot-deg", str(params["max_rot_deg"]),
        "--max-trans-mm", str(params["max_trans_mm"]),
        "--max-disp", str(params["max_disp"]),
        "--slice-thickness", "2.5",
        "--inplane-res", "1.0",
        "--noise-std", "0.02",
        "--disable-nonlinear"  # Disable nonlinear deformations
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 minute timeout
    if result.returncode != 0:
        print(f"Error simulating stacks: {result.stderr}")
        sys.exit(1)
    
    print(result.stdout)
    return result.returncode == 0


def run_svr(input_stacks, output_path, output_resolution=2.0, n_iter=2, temp_dir=None):
    """Run SVR reconstruction on a set of stacks."""
    print(f"Running SVR on {len(input_stacks)} stacks...")
    print(f"Output: {output_path}")
    
    env = os.environ.copy()
    if temp_dir:
        env['SVR_TEMP_DIR'] = temp_dir
        os.makedirs(temp_dir, exist_ok=True)
    
    cmd = [
        sys.executable,
        "svr_cli.py",
        "--input-stacks"
    ] + input_stacks + [
        "--output", output_path,
        "--output-resolution", str(output_resolution),
        "--segmentation", "otsu",
        "--n-iter", str(n_iter)
    ]
    
    print(f"Running: {' '.join(cmd[:5])} ... (full command logged)")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)  # 30 minute timeout
    if result.returncode != 0:
        print(f"Error running SVR: {result.stderr}")
        return False
    
    print(result.stdout)
    return result.returncode == 0


def compare_reconstructions(ground_truth_path, sms_recon_path, seq_recon_path, mask_path=None):
    """Compare SMS and sequential reconstructions against ground truth."""
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    # Load images
    gt_img = nib.load(ground_truth_path)
    gt_data = gt_img.get_fdata()
    gt_affine = gt_img.affine
    
    sms_img = nib.load(sms_recon_path)
    sms_data = sms_img.get_fdata()
    sms_affine = sms_img.affine
    
    seq_img = nib.load(seq_recon_path)
    seq_data = seq_img.get_fdata()
    seq_affine = seq_img.affine
    
    # Resample SMS and sequential to ground truth space if shapes differ
    if sms_data.shape != gt_data.shape:
        print(f"Resampling SMS reconstruction from {sms_data.shape} to ground truth shape {gt_data.shape}...")
        from scipy.ndimage import affine_transform
        
        # Compute transformation from SMS to GT space
        gt_to_sms = np.linalg.inv(sms_affine) @ gt_affine
        
        # Resample SMS to GT space
        sms_data_resampled = affine_transform(
            sms_data,
            np.linalg.inv(gt_to_sms[:3, :3]),
            offset=np.linalg.inv(gt_to_sms[:3, :3]) @ gt_to_sms[:3, 3],
            output_shape=gt_data.shape,
            order=1
        )
        sms_data = sms_data_resampled
    
    if seq_data.shape != gt_data.shape:
        print(f"Resampling sequential reconstruction from {seq_data.shape} to ground truth shape {gt_data.shape}...")
        from scipy.ndimage import affine_transform
        
        # Compute transformation from seq to GT space
        gt_to_seq = np.linalg.inv(seq_affine) @ gt_affine
        
        # Resample seq to GT space
        seq_data_resampled = affine_transform(
            seq_data,
            np.linalg.inv(gt_to_seq[:3, :3]),
            offset=np.linalg.inv(gt_to_seq[:3, :3]) @ gt_to_seq[:3, 3],
            output_shape=gt_data.shape,
            order=1
        )
        seq_data = seq_data_resampled
    
    # Optional masking
    mask = None
    if mask_path and os.path.exists(mask_path):
        mask_img = nib.load(mask_path)
        mask = mask_img.get_fdata() > 0
        print(f"Using mask from: {mask_path}")
    
    # Normalize images to same intensity range
    def normalize(data, mask=None):
        if mask is not None:
            data_masked = data[mask]
            p1, p99 = np.percentile(data_masked, [1, 99])
        else:
            p1, p99 = np.percentile(data, [1, 99])
        return np.clip((data - p1) / (p99 - p1), 0, 1)
    
    gt_norm = normalize(gt_data, mask)
    sms_norm = normalize(sms_data, mask)
    seq_norm = normalize(seq_data, mask)
    
    # Apply mask if available
    if mask is not None:
        gt_eval = gt_norm[mask]
        sms_eval = sms_norm[mask]
        seq_eval = seq_norm[mask]
        data_range = 1.0
    else:
        gt_eval = gt_norm
        sms_eval = sms_norm
        seq_eval = seq_norm
        data_range = 1.0
    
    # Compute metrics
    print(f"\nGround Truth: {ground_truth_path}")
    print(f"Ground truth shape: {gt_data.shape}, voxel size: {gt_img.header.get_zooms()[:3]}")
    
    print(f"\n--- SMS Reconstruction ---")
    print(f"File: {sms_recon_path}")
    print(f"Shape: {sms_data.shape}, voxel size: {sms_img.header.get_zooms()[:3]}")
    sms_psnr = compute_psnr(gt_eval, sms_eval, data_range=data_range)
    sms_nrmse = compute_nrmse(gt_eval, sms_eval)
    sms_ssim = compute_ssim(gt_norm, sms_norm, data_range=data_range) if mask is None else None
    print(f"PSNR vs GT: {sms_psnr:.2f} dB")
    print(f"NRMSE vs GT: {sms_nrmse:.4f}")
    if sms_ssim is not None:
        print(f"SSIM vs GT: {sms_ssim:.4f}")
    
    print(f"\n--- Sequential Reconstruction ---")
    print(f"File: {seq_recon_path}")
    print(f"Shape: {seq_data.shape}, voxel size: {seq_img.header.get_zooms()[:3]}")
    seq_psnr = compute_psnr(gt_eval, seq_eval, data_range=data_range)
    seq_nrmse = compute_nrmse(gt_eval, seq_eval)
    seq_ssim = compute_ssim(gt_norm, seq_norm, data_range=data_range) if mask is None else None
    print(f"PSNR vs GT: {seq_psnr:.2f} dB")
    print(f"NRMSE vs GT: {seq_nrmse:.4f}")
    if seq_ssim is not None:
        print(f"SSIM vs GT: {seq_ssim:.4f}")
    
    # Compare SMS vs Sequential
    print(f"\n--- Comparison: SMS vs Sequential ---")
    psnr_diff = sms_psnr - seq_psnr
    nrmse_diff = seq_nrmse - sms_nrmse  # Lower is better, so positive means SMS is better
    print(f"PSNR difference (SMS - Seq): {psnr_diff:+.2f} dB")
    print(f"NRMSE difference (Seq - SMS): {nrmse_diff:+.4f}")
    
    if sms_ssim is not None and seq_ssim is not None:
        ssim_diff = sms_ssim - seq_ssim
        print(f"SSIM difference (SMS - Seq): {ssim_diff:+.4f}")
    
    # Direct comparison between SMS and Sequential
    sms_vs_seq_psnr = compute_psnr(sms_eval, seq_eval, data_range=data_range)
    sms_vs_seq_nrmse = compute_nrmse(sms_eval, seq_eval)
    print(f"\nDirect comparison (SMS vs Sequential):")
    print(f"PSNR: {sms_vs_seq_psnr:.2f} dB")
    print(f"NRMSE: {sms_vs_seq_nrmse:.4f}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    if abs(psnr_diff) < 0.5 and abs(nrmse_diff) < 0.01:
        print("✓ SMS and sequential reconstructions have similar quality")
        print("  This validates that SMS averaging preserves reconstruction accuracy")
    elif psnr_diff > 0.5:
        print("✓ SMS reconstruction is BETTER than sequential")
        print(f"  Improvement: {psnr_diff:.2f} dB PSNR, {nrmse_diff:.4f} NRMSE")
    else:
        print("⚠ Sequential reconstruction is slightly better than SMS")
        print(f"  Difference: {psnr_diff:.2f} dB PSNR, {nrmse_diff:.4f} NRMSE")
        print("  This may indicate an issue with SMS averaging or stack quality")
    
    print("="*80 + "\n")
    
    return {
        'sms_psnr': sms_psnr,
        'sms_nrmse': sms_nrmse,
        'sms_ssim': sms_ssim,
        'seq_psnr': seq_psnr,
        'seq_nrmse': seq_nrmse,
        'seq_ssim': seq_ssim,
        'psnr_diff': psnr_diff,
        'nrmse_diff': nrmse_diff
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compare SVR reconstruction quality: SMS vs Sequential acquisition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare SMS (mb=2) vs sequential from existing ground truth
  python compare_sms_vs_sequential.py \\
      --ground-truth test_data/39/SVR001_brain9_20251013_130312/out/tmp/svr_output.nii.gz \\
      --output-dir test_data/39/comparison \\
      --num-stacks 3 \\
      --mb-factor 2
  
  # Use pre-simulated stacks
  python compare_sms_vs_sequential.py \\
      --ground-truth ground_truth.nii.gz \\
      --sms-stacks sms_dir/sim_stack_*.nii.gz \\
      --sequential-stacks seq_dir/sim_stack_*.nii.gz \\
      --output-dir comparison
        """
    )
    
    parser.add_argument('--ground-truth', required=True,
                        help='Path to ground truth volume (high-resolution reference)')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for results')
    parser.add_argument('--num-stacks', type=int, default=6,
                        help='Number of stacks to simulate (default: 6)')
    parser.add_argument('--mb-factor', type=int, default=2,
                        help='Multiband factor for SMS stacks (default: 2)')
    parser.add_argument('--motion-level', default='moderate',
                        choices=['none', 'mild', 'moderate', 'severe'],
                        help='Motion level for simulation (default: moderate)')
    parser.add_argument('--output-resolution', type=float, default=2.0,
                        help='SVR output resolution in mm (default: 2.0)')
    parser.add_argument('--n-iter', type=int, default=2,
                        help='Number of SVR iterations (default: 2)')
    parser.add_argument('--sms-stacks', nargs='+',
                        help='Pre-simulated SMS stack paths (skip SMS simulation)')
    parser.add_argument('--sequential-stacks', nargs='+',
                        help='Pre-simulated sequential stack paths (skip sequential simulation)')
    parser.add_argument('--skip-simulation', action='store_true',
                        help='Skip simulation, use existing stacks')
    parser.add_argument('--skip-svr', action='store_true',
                        help='Skip SVR, use existing reconstructions')
    
    args = parser.parse_args()
    
    # Setup directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sms_dir = output_dir / 'sms_stacks'
    seq_dir = output_dir / 'sequential_stacks'
    
    # Simulate or use existing SMS stacks
    if args.sms_stacks:
        sms_stack_files = args.sms_stacks
        print(f"Using pre-simulated SMS stacks: {len(sms_stack_files)} files")
    elif not args.skip_simulation:
        sms_dir.mkdir(exist_ok=True)
        simulate_stacks(
            args.ground_truth,
            str(sms_dir),
            args.num_stacks,
            mb_factor=args.mb_factor,
            motion_level=args.motion_level
        )
        sms_stack_files = sorted(sms_dir.glob('sim_stack_*.nii.gz'))
    else:
        sms_stack_files = sorted(sms_dir.glob('sim_stack_*.nii.gz'))
    
    # Simulate or use existing sequential stacks
    if args.sequential_stacks:
        seq_stack_files = args.sequential_stacks
        print(f"Using pre-simulated sequential stacks: {len(seq_stack_files)} files")
    elif not args.skip_simulation:
        seq_dir.mkdir(exist_ok=True)
        simulate_stacks(
            args.ground_truth,
            str(seq_dir),
            args.num_stacks,
            mb_factor=1,  # Sequential: mb_factor=1
            motion_level=args.motion_level
        )
        seq_stack_files = sorted(seq_dir.glob('sim_stack_*.nii.gz'))
    else:
        seq_stack_files = sorted(seq_dir.glob('sim_stack_*.nii.gz'))
    
    print(f"\nSMS stacks: {[str(f) for f in sms_stack_files]}")
    print(f"Sequential stacks: {[str(f) for f in seq_stack_files]}")
    
    # Run SVR on SMS stacks
    sms_recon_path = output_dir / 'svr_sms_reconstruction.nii.gz'
    if not args.skip_svr or not sms_recon_path.exists():
        sms_temp_dir = output_dir / 'sms_svr_temp'
        success = run_svr(
            [str(f) for f in sms_stack_files],
            str(sms_recon_path),
            output_resolution=args.output_resolution,
            n_iter=args.n_iter,
            temp_dir=str(sms_temp_dir)
        )
        if not success:
            print("SMS SVR failed")
            return 1
    else:
        print(f"Using existing SMS reconstruction: {sms_recon_path}")
    
    # Run SVR on sequential stacks
    seq_recon_path = output_dir / 'svr_sequential_reconstruction.nii.gz'
    if not args.skip_svr or not seq_recon_path.exists():
        seq_temp_dir = output_dir / 'sequential_svr_temp'
        success = run_svr(
            [str(f) for f in seq_stack_files],
            str(seq_recon_path),
            output_resolution=args.output_resolution,
            n_iter=args.n_iter,
            temp_dir=str(seq_temp_dir)
        )
        if not success:
            print("Sequential SVR failed")
            return 1
    else:
        print(f"Using existing sequential reconstruction: {seq_recon_path}")
    
    # Compare reconstructions
    results = compare_reconstructions(
        args.ground_truth,
        str(sms_recon_path),
        str(seq_recon_path)
    )
    
    # Save results to JSON
    results_json = output_dir / 'comparison_results.json'
    with open(results_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_json}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
