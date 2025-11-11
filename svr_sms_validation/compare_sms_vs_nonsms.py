#!/usr/bin/env python3
"""
Compare SMS vs non-SMS SVR reconstructions.

This script analyzes reconstruction quality across different:
- Motion levels (none, mild, moderate, severe)
- MB factors (1=non-SMS, 2, 3=SMS)
- Stack counts (3, 6, 9, 12)

Generates:
- Quality metrics (PSNR, SSIM, NCC)
- Statistical tables
- Comparative plots
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from dataclasses import dataclass
import json

# Try to import optional dependencies
try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: scikit-image not found. Install with: pip install scikit-image")

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False
    print("Warning: SimpleITK not found. Install with: pip install SimpleITK")
    print("Coregistration will be disabled.")


@dataclass
class ReconResult:
    """Store reconstruction metadata and paths."""
    gt_name: str
    motion_level: str
    mb_factor: int
    n_stacks: int
    permutation: int
    recon_path: Path
    gt_path: Path
    exists: bool = False


def find_ground_truth(gt_name: str, gt_base_dir: Path) -> Optional[Path]:
    """Find ground truth file for a given subject name."""
    candidates = list(gt_base_dir.glob(f"{gt_name}.nii.gz"))
    if not candidates:
        candidates = list(gt_base_dir.glob(f"{gt_name}.nii"))
    return candidates[0] if candidates else None


def load_and_normalize(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load NIfTI image and normalize to 0-1 range. Returns (data, affine)."""
    try:
        img = nib.load(path)
        data = img.get_fdata().astype(np.float32)
        affine = img.affine
        # Also return normalized version for registration
        if data.max() > data.min():
            norm_data = (data - data.min()) / (data.max() - data.min())
        else:
            norm_data = data.copy()
        return data, affine, norm_data
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def rigid_coregister_sitk(moving: np.ndarray, fixed: np.ndarray, 
                          moving_affine: np.ndarray, fixed_affine: np.ndarray,
                          moving_path: Optional[Path] = None,
                          moving_orig: Optional[np.ndarray] = None,
                          fixed_orig: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Rigidly coregister moving image to fixed image using SimpleITK.
    Uses normalized images for registration, original scale for output.
    If _reg.nii.gz exists, loads cached result instead of recomputing.
    """
    if not HAS_SITK:
        return moving
    
    # Check if coregistered file already exists
    if moving_path is not None:
        coreg_path = Path(str(moving_path).replace('.nii.gz', '_reg.nii.gz'))
        if coreg_path.exists():
            img = nib.load(coreg_path)
            return img.get_fdata().astype(np.float32)
    
    # Convert numpy arrays to SimpleITK images
    # Use normalized images ONLY for registration metric computation
    moving_img_norm = sitk.GetImageFromArray(moving.astype(np.float32))
    fixed_img_norm = sitk.GetImageFromArray(fixed.astype(np.float32))
    # For resampling and output, use original scale images
    if moving_orig is not None:
        moving_img = sitk.GetImageFromArray(moving_orig.astype(np.float32))
    else:
        moving_img = moving_img_norm
    if fixed_orig is not None:
        fixed_img = sitk.GetImageFromArray(fixed_orig.astype(np.float32))
    else:
        fixed_img = fixed_img_norm
    
    # Set spacing from affine matrices (diagonal elements give voxel sizes)
    moving_spacing = np.abs(np.diag(moving_affine[:3, :3]))
    fixed_spacing = np.abs(np.diag(fixed_affine[:3, :3]))
    moving_img.SetSpacing(moving_spacing.tolist())
    fixed_img.SetSpacing(fixed_spacing.tolist())
    moving_img_norm.SetSpacing(moving_spacing.tolist())
    fixed_img_norm.SetSpacing(fixed_spacing.tolist())
    # Set origin from affine matrices
    moving_img.SetOrigin(moving_affine[:3, 3].tolist())
    fixed_img.SetOrigin(fixed_affine[:3, 3].tolist())
    moving_img_norm.SetOrigin(moving_affine[:3, 3].tolist())
    fixed_img_norm.SetOrigin(fixed_affine[:3, 3].tolist())
    
    # Initialize registration (use normalized images for metric computation only)
    registration_method = sitk.ImageRegistrationMethod()
    # Similarity metric: Normalized Correlation (better for similar intensity distributions)
    registration_method.SetMetricAsCorrelation()
    registration_method.SetMetricSamplingStrategy(registration_method.NONE)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=200,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=1000,
        costFunctionConvergenceFactor=1e7
    )
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[8, 4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[4, 2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_img_norm,
        moving_img_norm,
        sitk.VersorRigid3DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS
    )
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    
    # Execute registration on normalized images
    final_transform = registration_method.Execute(fixed_img_norm, moving_img_norm)
    
    # Resample moving image with the final transformation
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(final_transform)
    coregistered_img = resampler.Execute(moving_img)
    coregistered = sitk.GetArrayFromImage(coregistered_img)
    
    # Save coregistered image as NIfTI file
    if moving_path is not None:
        coreg_path = Path(str(moving_path).replace('.nii.gz', '_reg.nii.gz'))
        coreg_nifti = nib.Nifti1Image(coregistered, fixed_affine)
        nib.save(coreg_nifti, coreg_path)
    
    return coregistered


def calculate_metrics(recon: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Calculate quality metrics between reconstruction and ground truth."""
    metrics = {}
    
    # Ensure same shape (crop/pad if needed)
    min_shape = tuple(min(r, g) for r, g in zip(recon.shape, gt.shape))
    recon_crop = recon[:min_shape[0], :min_shape[1], :min_shape[2]]
    gt_crop = gt[:min_shape[0], :min_shape[1], :min_shape[2]]
    
    # Normalized Cross-Correlation
    recon_flat = recon_crop.flatten()
    gt_flat = gt_crop.flatten()
    recon_mean = recon_flat.mean()
    gt_mean = gt_flat.mean()
    numerator = np.sum((recon_flat - recon_mean) * (gt_flat - gt_mean))
    denominator = np.sqrt(np.sum((recon_flat - recon_mean)**2) * np.sum((gt_flat - gt_mean)**2))
    metrics['ncc'] = float(numerator / (denominator + 1e-10))
    
    # MSE and RMSE
    mse = np.mean((recon_crop - gt_crop)**2)
    metrics['mse'] = float(mse)
    metrics['rmse'] = float(np.sqrt(mse))
    
    if HAS_SKIMAGE:
        # PSNR
        metrics['psnr'] = float(psnr(gt_crop, recon_crop, data_range=1.0))
        
        # SSIM
        metrics['ssim'] = float(ssim(gt_crop, recon_crop, data_range=1.0))
    else:
        metrics['psnr'] = np.nan
        metrics['ssim'] = np.nan
    
    return metrics


def scan_reconstructions(output_dir: Path, gt_dir: Path) -> List[ReconResult]:
    """Scan output directory for all reconstructions."""
    results = []
    
    for gt_subdir in sorted(output_dir.iterdir()):
        if not gt_subdir.is_dir():
            continue
        
        gt_name = gt_subdir.name
        gt_path = find_ground_truth(gt_name, gt_dir)
        
        if gt_path is None:
            print(f"Warning: No ground truth found for {gt_name}")
            continue
        
        # Scan for all motion/mb/n combinations
        for param_dir in sorted(gt_subdir.iterdir()):
            if not param_dir.is_dir():
                continue
            
            # Parse directory name: motion_{level}_mb{factor}_n{stacks}
            parts = param_dir.name.split('_')
            if len(parts) < 4:
                continue
            
            try:
                motion_level = parts[1]  # after "motion_"
                mb_factor = int(parts[2].replace('mb', ''))
                n_stacks = int(parts[3].replace('n', ''))
            except (ValueError, IndexError):
                continue
            
            # Scan for permutations
            for perm_dir in sorted(param_dir.iterdir()):
                if not perm_dir.is_dir():
                    continue
                
                # Parse permutation: perm_{num}
                if not perm_dir.name.startswith('perm_'):
                    continue
                
                try:
                    perm_num = int(perm_dir.name.replace('perm_', ''))
                except ValueError:
                    continue
                
                recon_path = perm_dir / "svr_recon.nii.gz"
                
                results.append(ReconResult(
                    gt_name=gt_name,
                    motion_level=motion_level,
                    mb_factor=mb_factor,
                    n_stacks=n_stacks,
                    permutation=perm_num,
                    recon_path=recon_path,
                    gt_path=gt_path,
                    exists=recon_path.exists()
                ))
    
    return results


def compute_all_metrics(results: List[ReconResult], enable_coregistration: bool = True,
                       ) -> pd.DataFrame:
    """Compute metrics for all reconstructions."""
    records = []
    
    total = len([r for r in results if r.exists])
    print(f"\nComputing metrics for {total} reconstructions...")
    if enable_coregistration and HAS_SITK:
        print("  Coregistration: ENABLED (SimpleITK rigid alignment to ground truth)")
    else:
        print("  Coregistration: DISABLED")
    
    for idx, result in enumerate(results, 1):
        if not result.exists:
            continue
        
        if idx % 10 == 0:
            print(f"  Progress: {idx}/{total}")
        
        recon_data = load_and_normalize(result.recon_path)
        gt_data = load_and_normalize(result.gt_path)
        
        if recon_data is None or gt_data is None:
            continue
        
        recon_orig, recon_affine, recon_norm = recon_data
        gt_orig, gt_affine, gt_norm = gt_data
        
        # Coregister reconstruction to ground truth
        if enable_coregistration and HAS_SITK:
            # Use normalized images for registration, but resample original scale
            recon = rigid_coregister_sitk(recon_norm, gt_norm, recon_affine, gt_affine,
                                         moving_path=result.recon_path,
                                         moving_orig=recon_orig, fixed_orig=gt_orig)
            gt = gt_orig
        else:
            recon = recon_orig
            gt = gt_orig
        
        metrics = calculate_metrics(recon, gt)
        
        records.append({
            'gt_name': result.gt_name,
            'motion_level': result.motion_level,
            'mb_factor': result.mb_factor,
            'is_sms': result.mb_factor > 1,
            'n_stacks': result.n_stacks,
            'permutation': result.permutation,
            **metrics
        })
    
    return pd.DataFrame(records)


def create_summary_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Create summary tables with statistics."""
    tables = {}
    
    # Overall SMS vs non-SMS
    sms_summary = df.groupby('is_sms').agg({
        'ncc': ['mean', 'std', 'count'],
        'rmse': ['mean', 'std'],
        'psnr': ['mean', 'std'],
        'ssim': ['mean', 'std']
    }).round(4)
    tables['sms_vs_nonsms'] = sms_summary
    
    # By MB factor (main comparison)
    mb_summary = df.groupby('mb_factor').agg({
        'ncc': ['mean', 'std', 'count'],
        'rmse': ['mean', 'std'],
        'psnr': ['mean', 'std'],
        'ssim': ['mean', 'std']
    }).round(4)
    tables['mb_factor'] = mb_summary
    
    # By motion level and MB factor
    motion_mb_summary = df.groupby(['motion_level', 'mb_factor']).agg({
        'ncc': ['mean', 'std', 'count'],
        'rmse': ['mean', 'std'],
        'psnr': ['mean', 'std'],
        'ssim': ['mean', 'std']
    }).round(4)
    tables['motion_mb_factor'] = motion_mb_summary
    
    # By number of stacks and MB factor
    stacks_mb_summary = df.groupby(['n_stacks', 'mb_factor']).agg({
        'ncc': ['mean', 'std', 'count'],
        'rmse': ['mean', 'std'],
        'psnr': ['mean', 'std'],
        'ssim': ['mean', 'std']
    }).round(4)
    tables['stacks_mb_factor'] = stacks_mb_summary
    
    # By motion level and SMS (binary)
    motion_summary = df.groupby(['motion_level', 'is_sms']).agg({
        'ncc': ['mean', 'std'],
        'rmse': ['mean', 'std'],
        'psnr': ['mean', 'std'],
        'ssim': ['mean', 'std']
    }).round(4)
    tables['motion_sms'] = motion_summary
    
    # By number of stacks and SMS (binary)
    stacks_summary = df.groupby(['n_stacks', 'is_sms']).agg({
        'ncc': ['mean', 'std'],
        'rmse': ['mean', 'std'],
        'psnr': ['mean', 'std'],
        'ssim': ['mean', 'std']
    }).round(4)
    tables['stacks_sms'] = stacks_summary
    
    # Statistical tests - pairwise between MB factors
    mb_factors = sorted(df['mb_factor'].unique())
    if len(mb_factors) > 1:
        pairwise_tests = {}
        for i, mb1 in enumerate(mb_factors):
            for mb2 in mb_factors[i+1:]:
                data1 = df[df['mb_factor'] == mb1]
                data2 = df[df['mb_factor'] == mb2]
                
                pair_name = f'MB{mb1}_vs_MB{mb2}'
                pairwise_tests[pair_name] = {}
                
                for metric in ['ncc', 'rmse', 'psnr', 'ssim']:
                    if not df[metric].isna().all():
                        t_stat, p_value = ttest_ind(
                            data1[metric].dropna(),
                            data2[metric].dropna()
                        )
                        pairwise_tests[pair_name][metric] = {
                            't_stat': t_stat, 
                            'p_value': p_value,
                            'mean_diff': data2[metric].mean() - data1[metric].mean()
                        }
        
        # Flatten for table
        test_records = []
        for pair, metrics in pairwise_tests.items():
            for metric, stats in metrics.items():
                test_records.append({
                    'comparison': pair,
                    'metric': metric,
                    't_statistic': stats['t_stat'],
                    'p_value': stats['p_value'],
                    'mean_difference': stats['mean_diff']
                })
        
        if test_records:
            tables['mb_factor_pairwise_tests'] = pd.DataFrame(test_records).round(6)
    
    # Statistical tests - SMS vs non-SMS
    sms_data = df[df['is_sms'] == True]
    nonsms_data = df[df['is_sms'] == False]
    
    if len(sms_data) > 0 and len(nonsms_data) > 0:
        stats_tests = {}
        for metric in ['ncc', 'rmse', 'psnr', 'ssim']:
            if not df[metric].isna().all():
                t_stat, p_value = ttest_ind(
                    sms_data[metric].dropna(),
                    nonsms_data[metric].dropna()
                )
                stats_tests[metric] = {'t_statistic': t_stat, 'p_value': p_value}
        
        tables['sms_statistical_tests'] = pd.DataFrame(stats_tests).T.round(6)
    
    return tables


def create_plots(df: pd.DataFrame, output_dir: Path):
    """Create comparative plots."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Box plots: SMS vs non-SMS for each metric
    metrics = ['ncc', 'rmse', 'psnr', 'ssim']
    metric_names = ['NCC (↑)', 'RMSE (↓)', 'PSNR (↑)', 'SSIM (↑)']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        if df[metric].isna().all():
            continue
        
        ax = axes[idx]
        df_plot = df[['is_sms', metric]].copy()
        df_plot['Type'] = df_plot['is_sms'].map({True: 'SMS (MB≥2)', False: 'Non-SMS (MB=1)'})
        
        sns.boxplot(data=df_plot, x='Type', y=metric, ax=ax, palette='Set2')
        ax.set_title(f'{name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel(name, fontsize=12)
        
        # Add sample sizes
        for i, is_sms in enumerate([False, True]):
            n = len(df_plot[df_plot['is_sms'] == is_sms])
            ax.text(i, ax.get_ylim()[0], f'n={n}', ha='center', va='top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'sms_vs_nonsms_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Line plot: Performance vs number of stacks (by MB factor)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        if df[metric].isna().all():
            continue
        
        ax = axes[idx]
        
        # Aggregate by n_stacks and mb_factor
        grouped = df.groupby(['n_stacks', 'mb_factor'])[metric].agg(['mean', 'std']).reset_index()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        markers = ['o', 's', '^']
        
        for mb, color, marker in zip(sorted(df['mb_factor'].unique()), colors, markers):
            data = grouped[grouped['mb_factor'] == mb]
            ax.errorbar(data['n_stacks'], data['mean'], yerr=data['std'], 
                       label=f'MB={mb}', marker=marker, markersize=8, capsize=5, 
                       linewidth=2, color=color)
        
        ax.set_title(f'{name} vs Number of Stacks', fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Stacks', fontsize=12)
        ax.set_ylabel(name, fontsize=12)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'performance_vs_stacks_by_mb.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Heatmap: Motion level vs MB factor (NCC)
    if not df['ncc'].isna().all():
        pivot = df.pivot_table(values='ncc', index='motion_level', 
                              columns='mb_factor', aggfunc='mean')
        
        # Sort motion levels
        motion_order = ['none', 'mild', 'moderate', 'severe']
        pivot = pivot.reindex([m for m in motion_order if m in pivot.index])
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Mean NCC'}, linewidths=0.5)
        plt.title('Reconstruction Quality (NCC) by Motion Level and MB Factor', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('MB Factor', fontsize=12)
        plt.ylabel('Motion Level', fontsize=12)
        plt.tight_layout()
        plt.savefig(plots_dir / 'ncc_heatmap_motion_mb.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Violin plots: Distribution by MB factor
    if not df['ncc'].isna().all():
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=df, x='mb_factor', y='ncc', palette='muted')
        plt.title('NCC Distribution by MB Factor', fontsize=14, fontweight='bold')
        plt.xlabel('MB Factor', fontsize=12)
        plt.ylabel('NCC', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(plots_dir / 'ncc_violin_by_mb.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Line plot: Performance vs MB factor by motion level
    if not df['ncc'].isna().all():
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        motion_order = ['none', 'mild', 'moderate', 'severe']
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            if df[metric].isna().all():
                continue
            
            ax = axes[idx]
            
            # Aggregate by motion level and MB factor
            grouped = df.groupby(['motion_level', 'mb_factor'])[metric].agg(['mean', 'std']).reset_index()
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            markers = ['o', 's', '^', 'D']
            
            for motion, color, marker in zip(motion_order, colors, markers):
                if motion in grouped['motion_level'].values:
                    data = grouped[grouped['motion_level'] == motion]
                    ax.errorbar(data['mb_factor'], data['mean'], yerr=data['std'], 
                               label=motion.title(), marker=marker, markersize=8, capsize=5, 
                               linewidth=2, color=color)
            
            ax.set_title(f'{name} vs MB Factor', fontsize=14, fontweight='bold')
            ax.set_xlabel('MB Factor', fontsize=12)
            ax.set_ylabel(name, fontsize=12)
            ax.set_xticks(sorted(df['mb_factor'].unique()))
            ax.legend(fontsize=10, loc='best', title='Motion')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'performance_vs_mb_by_motion.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6. Bar plot: Mean NCC by MB factor
    if not df['ncc'].isna().all():
        fig, ax = plt.subplots(figsize=(10, 6))
        
        mb_summary = df.groupby('mb_factor')['ncc'].agg(['mean', 'std']).reset_index()
        
        bars = ax.bar(mb_summary['mb_factor'], mb_summary['mean'], 
                     yerr=mb_summary['std'], capsize=10, alpha=0.7, 
                     color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        
        # Add value labels on bars
        for bar, mean_val in zip(bars, mb_summary['mean']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean_val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('MB Factor', fontsize=14, fontweight='bold')
        ax.set_ylabel('Mean NCC', fontsize=14, fontweight='bold')
        ax.set_title('Reconstruction Quality (NCC) by MB Factor', fontsize=16, fontweight='bold')
        ax.set_xticks(sorted(df['mb_factor'].unique()))
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'ncc_bar_by_mb.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\nPlots saved to: {plots_dir}")


def save_summary_report(tables: Dict[str, pd.DataFrame], df: pd.DataFrame, output_dir: Path):
    """Save comprehensive text report."""
    report_path = output_dir / "comparison_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SMS vs Non-SMS SVR Reconstruction Comparison\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("METHODOLOGY\n")
        f.write("-" * 80 + "\n")
        f.write("1. Each SVR reconstruction was rigidly coregistered to its ground truth\n")
        f.write("   using SimpleITK with Normalized Correlation metric (primary).\n")
        f.write("2. Registration uses multi-resolution pyramid (4 levels) and VersorRigid3D\n")
        f.write("   transform with LBFGSB optimizer for robust convergence.\n")
        f.write("3. Quality check: If NCC < 0.5, automatic retry with Mutual Information metric.\n")
        f.write("4. Coregistered images saved as *_reg.nii.gz alongside original files.\n")
        f.write("5. Coregistration results reused if *_reg.nii.gz files exist.\n")
        f.write("6. Metrics computed on coregistered images to ensure fair comparison.\n")
        f.write("7. Rigid registration allows translation and rotation only (no scaling).\n")
        f.write("\n\n")
        
        # Overall summary
        f.write("OVERALL SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total reconstructions analyzed: {len(df)}\n")
        f.write(f"  Non-SMS (MB=1): {len(df[df['mb_factor'] == 1])}\n")
        f.write(f"  SMS (MB≥2): {len(df[df['mb_factor'] > 1])}\n")
        f.write(f"    MB=2: {len(df[df['mb_factor'] == 2])}\n")
        f.write(f"    MB=3: {len(df[df['mb_factor'] == 3])}\n")
        f.write(f"\nGround truth subjects: {df['gt_name'].nunique()}\n")
        f.write(f"Motion levels: {sorted(df['motion_level'].unique())}\n")
        f.write(f"Stack counts: {sorted(df['n_stacks'].unique())}\n")
        f.write("\n\n")
        
        # Tables
        for name, table in tables.items():
            f.write(f"\n{name.upper().replace('_', ' ')}\n")
            f.write("-" * 80 + "\n")
            f.write(table.to_string())
            f.write("\n\n")
        
        # Key findings
        f.write("\nKEY FINDINGS\n")
        f.write("-" * 80 + "\n")
        
        # Performance by MB factor
        f.write("1. Performance by MB Factor (NCC):\n")
        for mb in sorted(df['mb_factor'].unique()):
            mb_ncc = df[df['mb_factor'] == mb]['ncc'].mean()
            mb_std = df[df['mb_factor'] == mb]['ncc'].std()
            mb_count = len(df[df['mb_factor'] == mb])
            f.write(f"   - MB={mb}: {mb_ncc:.4f} ± {mb_std:.4f} (n={mb_count})\n")
        
        # Compare MB factors
        mb_factors = sorted(df['mb_factor'].unique())
        if len(mb_factors) > 1:
            f.write(f"\n2. Pairwise MB Factor Comparisons (NCC improvement):\n")
            for i, mb1 in enumerate(mb_factors):
                for mb2 in mb_factors[i+1:]:
                    ncc1 = df[df['mb_factor'] == mb1]['ncc'].mean()
                    ncc2 = df[df['mb_factor'] == mb2]['ncc'].mean()
                    diff = ncc2 - ncc1
                    pct = (diff / ncc1 * 100) if ncc1 > 0 else 0
                    f.write(f"   - MB{mb2} vs MB{mb1}: {diff:+.4f} ({pct:+.1f}%)\n")
        
        # Overall SMS vs non-SMS
        sms_ncc = df[df['is_sms'] == True]['ncc'].mean()
        nonsms_ncc = df[df['is_sms'] == False]['ncc'].mean()
        ncc_diff = sms_ncc - nonsms_ncc
        
        f.write(f"\n3. Overall SMS vs Non-SMS (binary comparison):\n")
        f.write(f"   - SMS (MB≥2): {sms_ncc:.4f}\n")
        f.write(f"   - Non-SMS (MB=1): {nonsms_ncc:.4f}\n")
        f.write(f"   - Difference: {ncc_diff:+.4f} ({ncc_diff/nonsms_ncc*100:+.1f}%)\n")
        
        if HAS_SKIMAGE:
            sms_psnr = df[df['is_sms'] == True]['psnr'].mean()
            nonsms_psnr = df[df['is_sms'] == False]['psnr'].mean()
            psnr_diff = sms_psnr - nonsms_psnr
            
            f.write(f"\n4. PSNR comparison:\n")
            f.write(f"   - SMS: {sms_psnr:.2f} dB\n")
            f.write(f"   - Non-SMS: {nonsms_psnr:.2f} dB\n")
            f.write(f"   - Difference: {psnr_diff:+.2f} dB\n")
        
        # Best configuration
        best_config = df.loc[df['ncc'].idxmax()]
        f.write(f"\n5. Best overall reconstruction (highest NCC={best_config['ncc']:.4f}):\n")
        f.write(f"   - Subject: {best_config['gt_name']}\n")
        f.write(f"   - MB factor: {best_config['mb_factor']}\n")
        f.write(f"   - Motion: {best_config['motion_level']}\n")
        f.write(f"   - Stacks: {best_config['n_stacks']}\n")
        f.write(f"   - Permutation: {best_config['permutation']}\n")
        
        # Statistical significance
        if 'mb_factor_pairwise_tests' in tables:
            f.write("\n6. Statistical significance (pairwise MB factor tests):\n")
            test_df = tables['mb_factor_pairwise_tests']
            for _, row in test_df.iterrows():
                if row['metric'] == 'ncc':
                    sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else "ns"
                    f.write(f"   - {row['comparison']}: p={row['p_value']:.6f} {sig}, ")
                    f.write(f"Δ={row['mean_difference']:+.4f}\n")
        
        if 'sms_statistical_tests' in tables:
            f.write("\n7. Statistical significance (SMS vs Non-SMS binary):\n")
            for metric, row in tables['sms_statistical_tests'].iterrows():
                sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else "ns"
                f.write(f"   - {metric.upper()}: p={row['p_value']:.6f} {sig}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"\nReport saved to: {report_path}")


def main():
    # Configuration
    BASEPATH = "/home/ajoshi/project2_ajoshi_1183"
    if not os.path.exists(BASEPATH):
        BASEPATH = "/project2/ajoshi_1183"
    
    OUTPUT_DIR = Path(BASEPATH) / "data" / "svr_reconstructions"
    GT_DIR = Path(BASEPATH) / "data" / "ground_truths"
    RESULTS_DIR = Path(BASEPATH) / "data" / "sms_comparison_results"
    
    print("SMS vs Non-SMS SVR Comparison")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Ground truth directory: {GT_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    
    if not OUTPUT_DIR.exists():
        print(f"\nError: Output directory not found: {OUTPUT_DIR}")
        sys.exit(1)
    
    if not GT_DIR.exists():
        print(f"\nError: Ground truth directory not found: {GT_DIR}")
        sys.exit(1)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Scan for reconstructions
    print("\nScanning for reconstructions...")
    results = scan_reconstructions(OUTPUT_DIR, GT_DIR)
    print(f"Found {len(results)} reconstruction entries")
    print(f"  Existing files: {sum(1 for r in results if r.exists)}")
    print(f"  Missing files: {sum(1 for r in results if not r.exists)}")
    
    if not any(r.exists for r in results):
        print("\nError: No reconstruction files found!")
        sys.exit(1)
    
    # Compute metrics (with coregistration enabled by default)
    df = compute_all_metrics(results, enable_coregistration=True)
    
    if len(df) == 0:
        print("\nError: No metrics could be computed!")
        sys.exit(1)
    
    print(f"\nSuccessfully computed metrics for {len(df)} reconstructions")
    
    # Save raw data
    csv_path = RESULTS_DIR / "metrics_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"Raw metrics saved to: {csv_path}")
    
    # Create summary tables
    print("\nGenerating summary tables...")
    tables = create_summary_tables(df)
    
    # Save tables
    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(exist_ok=True)
    for name, table in tables.items():
        table.to_csv(tables_dir / f"{name}.csv")
    print(f"Tables saved to: {tables_dir}")
    
    # Create plots
    print("\nGenerating plots...")
    create_plots(df, RESULTS_DIR)
    
    # Save report
    print("\nGenerating summary report...")
    save_summary_report(tables, df, RESULTS_DIR)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"All results saved to: {RESULTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
