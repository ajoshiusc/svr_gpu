import logging
from typing import List, Optional, Tuple, cast
import torch
import numpy as np
from .registration import SliceToVolumeRegistration
from .outlier import EM, global_ncc_exclusion, local_ssim_exclusion
from .reconstruction import (
    psf_reconstruction,
    srr_update,
    srr_update_quantile,
    simulate_slices,
    slices_scale,
    simulated_error,
)
from ..types import DeviceType, PathType
from ..psf import get_PSF
import os
import nibabel as nib
from ..image import Volume, Slice, load_volume, load_mask, Stack, RigidTransform
from ..inr.data import PointDataset
import inspect
import torch.nn.functional as F

try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


def _to_numpy(data: torch.Tensor) -> np.ndarray:
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def _save_nifti(data: torch.Tensor, path: str, affine: Optional[np.ndarray] = None) -> None:
    try:
        arr = _to_numpy(data)
        if arr.ndim == 4 and arr.shape[1] == 1:
            arr = np.moveaxis(arr, 1, -1)
        if affine is None:
            affine = np.eye(4)
        nib.save(nib.Nifti1Image(arr, affine), path)
        logging.info("Saved intermediate %s", path)
    except Exception:
        logging.debug("Failed to save intermediate %s", path, exc_info=True)


def _save_numpy(data: torch.Tensor, path: str) -> None:
    try:
        np.save(path, _to_numpy(data))
        logging.info("Saved intermediate %s", path)
    except Exception:
        logging.debug("Failed to save intermediate %s", path, exc_info=True)


def _save_volume_png(data: torch.Tensor, path: str, title: str = "", cmap: str = "gray") -> None:
    """Save a 3-plane orthogonal screenshot of a 3D volume as a PNG."""
    if not _HAS_MATPLOTLIB:
        return
    try:
        arr = _to_numpy(data)
        # Squeeze to 3D
        while arr.ndim > 3:
            arr = arr.squeeze(0) if arr.shape[0] == 1 else arr.squeeze(-1)
        if arr.ndim != 3:
            return
        D, H, W = arr.shape
        mid_d, mid_h, mid_w = D // 2, H // 2, W // 2
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(arr[mid_d, :, :], cmap=cmap, origin='lower', aspect='equal')
        axes[0].set_title(f'Axial (z={mid_d})')
        axes[0].axis('off')
        axes[1].imshow(arr[:, mid_h, :], cmap=cmap, origin='lower', aspect='equal')
        axes[1].set_title(f'Coronal (y={mid_h})')
        axes[1].axis('off')
        axes[2].imshow(arr[:, :, mid_w], cmap=cmap, origin='lower', aspect='equal')
        axes[2].set_title(f'Sagittal (x={mid_w})')
        axes[2].axis('off')
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logging.info("Saved PNG screenshot %s", path)
    except Exception:
        logging.debug("Failed to save PNG screenshot %s", path, exc_info=True)


def _save_stack_png(data: torch.Tensor, path: str, title: str = "", cmap: str = "gray", max_slices: int = 16) -> None:
    """Save a montage of slices from a 4D stack (N,1,H,W) as a PNG."""
    if not _HAS_MATPLOTLIB:
        return
    try:
        arr = _to_numpy(data)
        if arr.ndim == 4 and arr.shape[1] == 1:
            arr = arr[:, 0, :, :]  # (N, H, W)
        if arr.ndim != 3:
            return
        N = arr.shape[0]
        # Sample evenly spaced slices if too many
        if N > max_slices:
            indices = np.linspace(0, N - 1, max_slices, dtype=int)
            arr = arr[indices]
            N = max_slices
        ncols = min(4, N)
        nrows = int(np.ceil(N / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        if nrows == 1 and ncols == 1:
            axes = np.array([axes])
        axes = np.atleast_2d(axes)
        for idx in range(nrows * ncols):
            r, c = divmod(idx, ncols)
            ax = axes[r, c]
            if idx < N:
                ax.imshow(arr[idx], cmap=cmap, origin='lower', aspect='equal')
                ax.set_title(f'Slice {idx}', fontsize=9)
            ax.axis('off')
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        logging.info("Saved PNG screenshot %s", path)
    except Exception:
        logging.debug("Failed to save PNG screenshot %s", path, exc_info=True)


def _save_summary_png(
    volume_data: torch.Tensor,
    mask: torch.Tensor,
    path: str,
    uncertainty_data: Optional[torch.Tensor] = None,
    coverage_count: Optional[torch.Tensor] = None,
    coverage_weighted: Optional[torch.Tensor] = None,
) -> None:
    """Save 3-plane orthogonal views as a multi-row PNG.

    Rows (in order, each optional except volume):
      - Row 0: SVR reconstructed volume (always shown)
      - Row 1: Uncertainty map (only when uncertainty_data is provided)
      - Row 2: Coverage stack-count map (only when coverage_count is provided)
    """
    if not _HAS_MATPLOTLIB:
        return
    try:
        def _sq3d(a):
            while a.ndim > 3:
                a = a.squeeze(0) if a.shape[0] == 1 else a.squeeze(-1)
            return a

        vol = _sq3d(_to_numpy(volume_data))
        msk = _sq3d(_to_numpy(mask.float())) if mask is not None else np.ones_like(vol)

        has_unc = uncertainty_data is not None
        has_cov = coverage_count is not None

        if has_unc:
            unc = _sq3d(_to_numpy(uncertainty_data))
        if has_cov:
            cnt = _sq3d(_to_numpy(coverage_count))

        if vol.ndim != 3:
            return

        D, H, W = vol.shape
        mid_d, mid_h, mid_w = D // 2, H // 2, W // 2
        plane_titles = [f'Axial (z={mid_d})', f'Coronal (y={mid_h})', f'Sagittal (x={mid_w})']

        # Build list of rows: (label, slices_3, cmap, vmin, vmax, cbar_label)
        rows = []

        # Row: volume
        slices_vol = [vol[mid_d], vol[:, mid_h, :], vol[:, :, mid_w]]
        rows.append(('SVR Volume', slices_vol, 'gray', None, None, None))

        # Row: uncertainty
        if has_unc:
            unc_masked = unc * msk
            vmax_unc = float(np.percentile(unc_masked[unc_masked > 0], 99)) if (unc_masked > 0).any() else 1.0
            slices_unc = []
            slices_msk = [msk[mid_d], msk[:, mid_h, :], msk[:, :, mid_w]]
            for i, s in enumerate([unc[mid_d], unc[:, mid_h, :], unc[:, :, mid_w]]):
                s = s.copy()
                s[slices_msk[i] < 0.5] = 0
                slices_unc.append(s)
            rows.append(('Uncertainty (Q₀.₉ − Q₀.₁)', slices_unc, 'hot', 0, vmax_unc, 'Uncertainty'))

        # Row: coverage
        if has_cov:
            cnt_masked = cnt * msk
            vmax_cnt = float(cnt_masked.max()) if cnt_masked.max() > 0 else 1
            slices_cnt = [cnt[mid_d], cnt[:, mid_h, :], cnt[:, :, mid_w]]
            rows.append(('Stack Count', slices_cnt, 'viridis', 0, vmax_cnt, 'Number of Stacks'))

        nrows = len(rows)
        fig, axes = plt.subplots(nrows, 3, figsize=(18, 6 * nrows))
        if nrows == 1:
            axes = axes[np.newaxis, :]  # ensure 2D

        for ri, (label, slices_data, cmap, vmin, vmax, cbar_label) in enumerate(rows):
            im = None
            for col in range(3):
                im = axes[ri, col].imshow(
                    slices_data[col], cmap=cmap, origin='lower', aspect='equal',
                    vmin=vmin, vmax=vmax,
                )
                axes[ri, col].set_title(
                    f'{label} — {plane_titles[col]}' if ri > 0 else plane_titles[col],
                    fontsize=11,
                )
                axes[ri, col].axis('off')
            axes[ri, 0].set_ylabel(label, fontsize=12, labelpad=10)
            if cbar_label is not None and im is not None:
                cbar = fig.colorbar(im, ax=axes[ri, :].tolist(), fraction=0.02, pad=0.02)
                cbar.set_label(cbar_label, fontsize=11)

        title_parts = ['SVR Summary']
        if has_unc:
            title_parts.append('Uncertainty')
        if has_cov:
            title_parts.append('Coverage')
        fig.suptitle(' | '.join(title_parts), fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logging.info("Saved summary PNG %s", path)
    except Exception:
        logging.debug("Failed to save summary PNG %s", path, exc_info=True)


# Keep old name as alias for backward compatibility
def _save_uncertainty_overlay_png(
    volume_data: torch.Tensor,
    uncertainty_data: torch.Tensor,
    mask: torch.Tensor,
    path: str,
    coverage_count: Optional[torch.Tensor] = None,
    coverage_weighted: Optional[torch.Tensor] = None,
) -> None:
    """Backward-compatible wrapper around _save_summary_png."""
    _save_summary_png(
        volume_data, mask, path,
        uncertainty_data=uncertainty_data,
        coverage_count=coverage_count,
        coverage_weighted=coverage_weighted,
    )


def _save_coverage_png(
    coverage_count: torch.Tensor,
    coverage_weighted: torch.Tensor,
    mask: torch.Tensor,
    path: str,
) -> None:
    """Save 3-plane views: stack count (top) and weighted coverage (bottom)."""
    if not _HAS_MATPLOTLIB:
        return
    try:
        cnt = _to_numpy(coverage_count)
        wgt = _to_numpy(coverage_weighted)
        msk = _to_numpy(mask.float()) if mask is not None else np.ones_like(cnt)

        for a in [cnt, wgt, msk]:
            while a.ndim > 3:
                a = a.squeeze(0) if a.shape[0] == 1 else a.squeeze(-1)
        while cnt.ndim > 3:
            cnt = cnt.squeeze(0) if cnt.shape[0] == 1 else cnt.squeeze(-1)
        while wgt.ndim > 3:
            wgt = wgt.squeeze(0) if wgt.shape[0] == 1 else wgt.squeeze(-1)
        while msk.ndim > 3:
            msk = msk.squeeze(0) if msk.shape[0] == 1 else msk.squeeze(-1)

        if cnt.ndim != 3:
            return

        D, H, W = cnt.shape
        mid_d, mid_h, mid_w = D // 2, H // 2, W // 2

        cnt_masked = cnt * msk
        wgt_masked = wgt * msk
        vmax_cnt = cnt_masked.max() if cnt_masked.max() > 0 else 1
        vmax_wgt = np.percentile(wgt_masked[wgt_masked > 0], 99) if (wgt_masked > 0).any() else 1.0

        slices_cnt = [cnt[mid_d], cnt[:, mid_h, :], cnt[:, :, mid_w]]
        slices_wgt = [wgt[mid_d], wgt[:, mid_h, :], wgt[:, :, mid_w]]
        titles = [f'Axial (z={mid_d})', f'Coronal (y={mid_h})', f'Sagittal (x={mid_w})']

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        for col in range(3):
            # Top: stack count
            im1 = axes[0, col].imshow(
                slices_cnt[col], cmap='viridis', origin='lower', aspect='equal',
                vmin=0, vmax=vmax_cnt,
            )
            axes[0, col].set_title(titles[col], fontsize=11)
            axes[0, col].axis('off')

            # Bottom: weighted coverage
            im2 = axes[1, col].imshow(
                slices_wgt[col], cmap='inferno', origin='lower', aspect='equal',
                vmin=0, vmax=vmax_wgt,
            )
            axes[1, col].set_title(titles[col], fontsize=11)
            axes[1, col].axis('off')

        axes[0, 0].set_ylabel('Stack Count', fontsize=12, labelpad=10)
        axes[1, 0].set_ylabel('Weighted Coverage', fontsize=12, labelpad=10)

        cbar1 = fig.colorbar(im1, ax=axes[0, :].tolist(), fraction=0.02, pad=0.02)
        cbar1.set_label('Number of Stacks', fontsize=11)
        cbar2 = fig.colorbar(im2, ax=axes[1, :].tolist(), fraction=0.02, pad=0.02)
        cbar2.set_label('Confidence-Weighted Coverage', fontsize=11)

        fig.suptitle('Voxel-wise Coverage Map', fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logging.info("Saved coverage PNG %s", path)
    except Exception:
        logging.debug("Failed to save coverage PNG %s", path, exc_info=True)


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _resolve_temp_root() -> str:
    svr_tmp = os.environ.get('SVR_TEMP_DIR')
    if svr_tmp:
        return svr_tmp
    output_path = None
    for frame_info in inspect.stack():
        local_args = frame_info.frame.f_locals.get('args', None)
        if local_args and hasattr(local_args, 'output_volume'):
            candidate = getattr(local_args, 'output_volume')
            if candidate:
                output_path = candidate
                break
    if output_path:
        output_dir = os.path.dirname(os.path.abspath(output_path))
        output_base = os.path.splitext(os.path.basename(output_path))[0]
        if output_base.endswith('.nii'):
            output_base = os.path.splitext(output_base)[0]
        svr_tmp = os.path.join(output_dir, f'svr_tmp_{output_base}')
    else:
        svr_tmp = os.path.join(os.getcwd(), 'out', 'tmp')
    os.environ['SVR_TEMP_DIR'] = svr_tmp
    return svr_tmp


def _initial_mask(
    slices: List[Slice],
    output_resolution: float,
    sample_mask: Optional[PathType],
    sample_orientation: Optional[PathType],
    device: DeviceType,
) -> Tuple[Volume, bool]:
    # Determine if any slice originates from an SMS stack
    sms_mode = any(getattr(s, "_source_mb_factor", 1) and getattr(s, "_source_mb_factor", 1) > 1 for s in slices)
    dataset = PointDataset(slices)
    mask = dataset.mask
    if sample_mask is not None:
        mask = load_mask(sample_mask, device)
    transformation = None
    if sample_orientation is not None:
        transformation = load_volume(
            sample_orientation,
            device=device,
        ).transformation
    mask = mask.resample(output_resolution, transformation)
    mask.mask = mask.image > 0
    # If mask is all zero after resampling, force to all ones again
    if mask.mask is None or mask.mask.sum() == 0:
        print("[SVR][DEBUG] Forcing mask to all ones after resampling (simulated or non-brain stack)")
        mask.mask = mask.image > -np.inf  # all True
    return mask, sample_mask is None


def _check_resolution_and_shape(slices: List[Slice]) -> List[Slice]:
    res_inplane = []
    thicknesses = []
    # Save SMS metadata before resampling
    sms_metadata_per_slice = []
    for s in slices:
        res_inplane.append(float(s.resolution_x))
        res_inplane.append(float(s.resolution_y))
        thicknesses.append(float(s.resolution_z))
        # Save SMS metadata
        sms_metadata_per_slice.append({
            '_source_stack_idx': getattr(s, '_source_stack_idx', None),
            '_source_mb_factor': getattr(s, '_source_mb_factor', None),
            '_source_acquisition_order': getattr(s, '_source_acquisition_order', None),
        })

    res_s = min(res_inplane)
    s_thick = np.mean(thicknesses).item()
    slices = [s.resample((res_s, res_s, s_thick)) for s in slices]
    slices = Stack.pad_stacks(slices)
    
    # Restore SMS metadata after resampling and padding
    for s, metadata in zip(slices, sms_metadata_per_slice):
        for key, value in metadata.items():
            if value is not None:
                setattr(s, key, value)

    if max(thicknesses) - min(thicknesses) > 0.001:
        logging.warning("The input data have different thicknesses!")

    return slices


def _normalize(
    stack: Stack, output_intensity_mean: float
) -> Tuple[Stack, float, float, float, float]:
    masked_v = stack.slices[stack.mask]
    if masked_v.numel() == 0:
        logging.warning("Stack mask empty during normalization; skipping scaling")
        mean_intensity = 1.0
        max_intensity = stack.slices.max().item()
        min_intensity = stack.slices.min().item()
        scale = 1.0
    else:
        mean_intensity = masked_v.mean().item()
        max_intensity = masked_v.max().item()
        min_intensity = masked_v.min().item()
        if not np.isfinite(mean_intensity) or abs(mean_intensity) < 1e-6:
            logging.warning(
                "Stack mean intensity %.6f is not usable; skipping scaling", mean_intensity
            )
            scale = 1.0
        else:
            scale = output_intensity_mean / mean_intensity
    stack.slices = stack.slices * scale
    max_intensity = max_intensity * scale
    min_intensity = min_intensity * scale
    return stack, max_intensity, min_intensity, scale, mean_intensity


def slice_to_volume_reconstruction(
    slices: List[Slice],
    *,
    with_background: bool = False,
    output_resolution: float = 0.8,
    output_intensity_mean: float = 700,
    delta: float = 150 / 700,
    n_iter: int = 3,
    n_iter_rec: Optional[List[int]] = None,
    global_ncc_threshold: float = 0.5,
    local_ssim_threshold: float = 0.4,
    no_slice_robust_statistics: bool = False,
    no_pixel_robust_statistics: bool = False,
    no_global_exclusion: bool = False,
    no_local_exclusion: bool = False,
    no_registration: bool = False,
    sample_mask: Optional[PathType] = None,
    sample_orientation: Optional[PathType] = None,
    psf: str = "gaussian",
    device: DeviceType = torch.device("cpu"),
    normalize_stacks: bool = True,
    estimate_uncertainty: bool = False,
    n_iter_quantile: int = 20,
    quantiles: Optional[List[float]] = None,
    **unused
) -> Tuple[Volume, List[Slice], List[Slice]]:
    # check data
    slices = _check_resolution_and_shape(slices)
    if n_iter_rec is None:
        n_iter_rec = [7, 7, 21]
    
    # Debug: Check if slices have SMS metadata attached
    print(f"[SVR][DEBUG] Processing {len(slices)} slices")
    for i, s in enumerate(slices[:3]):  # Check first 3 slices
        print(f"[SVR][DEBUG] Slice {i}: has _source_stack_idx={hasattr(s, '_source_stack_idx')}, "
              f"_source_mb_factor={getattr(s, '_source_mb_factor', 'N/A')}, "
              f"_source_acquisition_order={getattr(s, '_source_acquisition_order', 'N/A')}")
    
    stack = Stack.cat(slices)

    svr_tmp = _resolve_temp_root()
    intermediates_dir = _ensure_dir(os.path.join(svr_tmp, 'intermediates'))
    recon_dir = _ensure_dir(os.path.join(svr_tmp, 'reconstructions'))
    transforms_dir = _ensure_dir(os.path.join(svr_tmp, 'svr'))
    png_dir = _ensure_dir(os.path.join(intermediates_dir, 'png'))

    _save_nifti(stack.slices, os.path.join(intermediates_dir, '00_input_stack.nii.gz'))
    _save_nifti(stack.mask.float(), os.path.join(intermediates_dir, '00_input_mask.nii.gz'))
    _save_stack_png(stack.slices, os.path.join(png_dir, '00_input_stack.png'), title='Input Stack (preprocessed)')
    _save_stack_png(stack.mask.float(), os.path.join(png_dir, '00_input_mask.png'), title='Input Mask')
    
    # Extract SMS metadata from input slices if they came from SMS stacks
    # Each slice may have stack_metadata attached during registration
    mb_factors = []
    acquisition_orders = []
    slice_counts_per_stack = []
    current_stack_idx = None
    current_count = 0
    
    for s in slices:
        stack_idx = getattr(s, '_source_stack_idx', None)
        if stack_idx != current_stack_idx and current_stack_idx is not None:
            # New stack encountered, save previous stack info
            if current_count > 0:
                slice_counts_per_stack.append(current_count)
                current_count = 0
        current_stack_idx = stack_idx
        current_count += 1
        
        # Collect SMS metadata from first slice of each stack
        if current_count == 1 and stack_idx is not None:
            mb_factors.append(getattr(s, '_source_mb_factor', 1))
            acquisition_orders.append(getattr(s, '_source_acquisition_order', None))
    
    # Don't forget the last stack
    if current_count > 0:
        slice_counts_per_stack.append(current_count)
    
    print(f"[SVR][DEBUG] Extracted SMS metadata: mb_factors={mb_factors}, slice_counts={slice_counts_per_stack}")
    
    # Attach SMS metadata to the concatenated stack if any SMS stacks were found
    if mb_factors and any(mb > 1 for mb in mb_factors) and len(slice_counts_per_stack) == len(mb_factors):
        stack.sms_metadata = [(mb, ao, sc) for mb, ao, sc in zip(mb_factors, acquisition_orders, slice_counts_per_stack)]
        print(f"[SVR][DEBUG] Attached SMS metadata to stack: {stack.sms_metadata}")
    
    # Check for SMS in multiple ways:
    # 1. sms_metadata attribute (set above for multi-stack concatenated case)
    # 2. mb_factor > 1 directly on stack (for single-stack case or if Stack.cat preserved it)
    has_sms_metadata = (
        bool(getattr(stack, "sms_metadata", None)) or 
        (hasattr(stack, 'mb_factor') and stack.mb_factor > 1)
    )
    if has_sms_metadata:
        logging.info("SMS stack detected; disabling robust slice/pixel outlier weighting.")

    slices_mask_backup = stack.mask.clone()

    # init volume
    volume, is_refine_mask = _initial_mask(
        slices,
        output_resolution,
        sample_mask,
        sample_orientation,
        device,
    )
    # Save initial mask/volume
    _save_nifti(volume.image, os.path.join(intermediates_dir, '01_initial_volume_mask.nii.gz'))
    _save_volume_png(volume.image, os.path.join(png_dir, '01_initial_volume_mask.png'), title='Initial Volume Mask')

    # data normalization
    if normalize_stacks:
        stack, max_intensity, min_intensity, scale, mean_intensity = _normalize(
            stack, output_intensity_mean
        )
        intensity_scale_ref = (
            output_intensity_mean if abs(scale - 1.0) > 1e-6 else mean_intensity
        )
    else:
        masked_v = stack.slices[stack.mask]
        if masked_v.numel() == 0:
            logging.warning("Stack mask empty while skipping normalization; using defaults")
            mean_intensity = 1.0
            max_intensity = stack.slices.max().item()
            min_intensity = stack.slices.min().item()
        else:
            mean_intensity = masked_v.mean().item()
            max_intensity = masked_v.max().item()
            min_intensity = masked_v.min().item()
        intensity_scale_ref = (
            mean_intensity if abs(mean_intensity) > 1e-6 else output_intensity_mean
        )
    _save_nifti(stack.slices, os.path.join(intermediates_dir, '02_normalized_stack.nii.gz'))
    _save_stack_png(stack.slices, os.path.join(png_dir, '02_normalized_stack.png'), title='Normalized Stack')

    # define psf
    psf_tensor = get_PSF(
        res_ratio=(
            stack.resolution_x / output_resolution,
            stack.resolution_y / output_resolution,
            stack.thickness / output_resolution,
        ),
        device=volume.device,
        psf_type=psf,
    )
    _save_numpy(psf_tensor, os.path.join(intermediates_dir, '03_psf_kernel.npy'))
    _save_volume_png(psf_tensor, os.path.join(png_dir, '03_psf_kernel.png'), title='PSF Kernel', cmap='hot')

    for i in range(n_iter):
        logging.info("outer %d", i)
        # Save volume at the start of outer iteration
        affine = getattr(volume, 'affine', None)
        _save_nifti(volume.image, os.path.join(recon_dir, f'outer{i:02d}_00_volume_start.nii.gz'), affine)
        _save_volume_png(volume.image, os.path.join(png_dir, f'outer{i:02d}_00_volume_start.png'), title=f'Volume at Start of Outer Iter {i}')

        # slice-to-volume registration
        if i > 0 and not no_registration:  # skip slice-to-volume registration for the first iteration
            svr = SliceToVolumeRegistration(
                num_levels=3,
                num_steps=5,
                step_size=2,
                max_iter=10,
            )
            # Pass the actual PSF to the registration for accurate slice simulation
            svr.psf = psf_tensor
            slices_transform, _ = svr(
                stack,
                volume,
                use_mask=False,
            )
            stack.transformation = slices_transform
            # Save stack transformation after registration
            _save_numpy(stack.transformation.matrix(), os.path.join(intermediates_dir, f'outer{i:02d}_01_transforms_after_registration.npy'))

        # global structual exclusion
        if i > 0 and not no_global_exclusion:
            stack.mask = slices_mask_backup.clone()
            excluded = global_ncc_exclusion(stack, volume, global_ncc_threshold)
            stack.mask[excluded] = False
            # Save mask after global exclusion
            _save_nifti(stack.mask.float(), os.path.join(intermediates_dir, f'outer{i:02d}_02_mask_after_ncc_exclusion.nii.gz'))
            _save_stack_png(stack.mask.float(), os.path.join(png_dir, f'outer{i:02d}_02_mask_after_ncc_exclusion.png'), title=f'Mask After Global NCC Exclusion (Outer {i})')

        # PSF reconstruction & volume mask
        # Only rebuild volume from scratch in the first iteration (i==0)
        # In subsequent iterations, keep the refined volume from the previous iteration
        if i == 0:
            volume = psf_reconstruction(
                stack,
                volume,
                update_mask=is_refine_mask,
                use_mask=not with_background,
                psf=psf_tensor,
            )
            _save_nifti(volume.image, os.path.join(intermediates_dir, f'outer{i:02d}_03_volume_after_psf_init.nii.gz'))
            _save_volume_png(volume.image, os.path.join(png_dir, f'outer{i:02d}_03_volume_after_psf_init.png'), title=f'Volume After PSF Init (Outer {i})')

        # init EM
        em = EM(max_intensity, min_intensity)
        p_voxel = torch.ones_like(stack.slices)
        # super-resolution reconstruction (inner loop)
        for j in range(n_iter_rec[i]):
            logging.info("inner %d", j)
            # simulate slices
            slices_sim, slices_weight = cast(
                Tuple[Stack, Stack],
                simulate_slices(
                    stack,
                    volume,
                    return_weight=True,
                    use_mask=not with_background,
                    psf=psf_tensor,
                ),
            )
            _save_nifti(
                slices_sim.slices,
                os.path.join(intermediates_dir, f'outer{i:02d}_inner{j:02d}_01_simulated_slices.nii.gz'),
            )
            _save_nifti(
                slices_weight.slices,
                os.path.join(intermediates_dir, f'outer{i:02d}_inner{j:02d}_02_slice_weights.nii.gz'),
            )
            # scale
            scale = slices_scale(stack, slices_sim, slices_weight, p_voxel, True)
            _save_numpy(scale, os.path.join(intermediates_dir, f'outer{i:02d}_inner{j:02d}_03_scale_factors.npy'))
            
            # SVRTK-style scale-based exclusion: exclude slices with unrealistic scales
            # This catches misregistered slices that would otherwise corrupt the reconstruction
            scale_excluded = (scale < 0.2) | (scale > 5.0)
            num_scale_excluded = scale_excluded.sum().item()
            if num_scale_excluded > 0:
                logging.info(
                    "Scale-based exclusion: %d/%d slices have unrealistic scale (outside [0.2, 5])",
                    num_scale_excluded, len(scale)
                )
            
            # err
            err = simulated_error(stack, slices_sim, scale)
            _save_nifti(
                err.slices,
                os.path.join(intermediates_dir, f'outer{i:02d}_inner{j:02d}_04_residual_error.nii.gz'),
            )
            # Determine whether to apply robust statistics (SMS stacks skip by default)
            pixel_robust_enabled = not no_pixel_robust_statistics and not has_sms_metadata
            slice_robust_enabled = not no_slice_robust_statistics and not has_sms_metadata

            # EM robust statistics
            if pixel_robust_enabled or slice_robust_enabled:
                p_voxel, p_slice = em(
                    err,
                    slices_weight,
                    scale,
                    1,
                    disable_slice_updates=not slice_robust_enabled,
                )
                _save_nifti(
                    p_voxel,
                    os.path.join(intermediates_dir, f'outer{i:02d}_inner{j:02d}_05_voxel_confidence.nii.gz'),
                )
                _save_numpy(
                    p_slice,
                    os.path.join(intermediates_dir, f'outer{i:02d}_inner{j:02d}_06_slice_confidence.npy'),
                )
                if not pixel_robust_enabled:
                    p_voxel = torch.ones_like(p_voxel)
                if not slice_robust_enabled:
                    p_slice = torch.ones_like(p_slice)
            else:
                p_voxel = torch.ones_like(stack.slices)
                p_slice = torch.ones(
                    stack.slices.shape[0],
                    device=stack.slices.device,
                    dtype=stack.slices.dtype,
                )
            
            # Apply SVRTK-style scale-based exclusion to p_slice
            # Slices with unrealistic scales (outside [0.2, 5]) get weight 0
            p_slice = p_slice * (~scale_excluded).float()
            
            p = p_voxel
            if slice_robust_enabled:
                p = p * p_slice.view(-1, 1, 1, 1)
            _save_nifti(
                p,
                os.path.join(intermediates_dir, f'outer{i:02d}_inner{j:02d}_07_combined_confidence.nii.gz'),
            )
            # local structural exclusion
            if not no_local_exclusion:
                p = p * local_ssim_exclusion(stack, slices_sim, local_ssim_threshold)
                _save_nifti(
                    p,
                    os.path.join(intermediates_dir, f'outer{i:02d}_inner{j:02d}_08_confidence_after_ssim.nii.gz'),
                )
            # super-resolution update
            beta = max(0.01, 0.08 / (2**i))
            alpha = min(1, 0.05 / beta)
            volume = srr_update(
                err,
                volume,
                p,
                alpha,
                beta,
                delta * intensity_scale_ref,
                scale,
                use_mask=not with_background,
                psf=psf_tensor,
            )
            # Save volume after inner iteration
            affine = getattr(volume, 'affine', None)
            _save_nifti(volume.image, os.path.join(recon_dir, f'outer{i:02d}_inner{j:02d}_09_reconstructed_volume.nii.gz'), affine)
            # Save PNG for first and last inner iteration per outer
            is_first_inner = (j == 0)
            is_last_inner = (j == n_iter_rec[i] - 1)
            if is_first_inner or is_last_inner:
                _save_volume_png(
                    volume.image,
                    os.path.join(png_dir, f'outer{i:02d}_inner{j:02d}_09_reconstructed_volume.png'),
                    title=f'Reconstructed Volume (Outer {i}, Inner {j})',
                )
                _save_stack_png(
                    err.slices,
                    os.path.join(png_dir, f'outer{i:02d}_inner{j:02d}_04_residual_error.png'),
                    title=f'Residual Error (Outer {i}, Inner {j})',
                    cmap='RdBu_r',
                )

    # reconstruction finished
    # Save final transforms after SVR completes
    _save_numpy(stack.transformation.matrix(), os.path.join(transforms_dir, 'final_transforms.npy'))
    _save_nifti(volume.image, os.path.join(intermediates_dir, 'final_volume.nii.gz'))
    _save_nifti(volume.mask.float(), os.path.join(intermediates_dir, 'final_mask.nii.gz'))
    _save_volume_png(volume.image, os.path.join(png_dir, 'final_volume.png'), title='Final Reconstructed Volume')
    _save_volume_png(volume.mask.float(), os.path.join(png_dir, 'final_mask.png'), title='Final Volume Mask')

    # ========================================================================
    # Coverage map: how many stacks contribute to each voxel
    # ========================================================================
    logging.info("Computing voxel-wise coverage map...")
    try:
        from ..slice_acquisition import slice_acquisition_adjoint
        from ..transform import mat_update_resolution

        transforms_all = stack.transformation
        volume_transform = volume.transformation
        rel_transform = volume_transform.inv().compose(transforms_all)
        transform_mat = mat_update_resolution(rel_transform.matrix(), 1, float(volume.resolution_x))

        res_s = float(stack.resolution_x)
        res_r = float(volume.resolution_x)

        # --- Stack-count coverage: number of distinct stacks per voxel ---
        n_slices_total = stack.slices.shape[0]
        n_stacks = len(slice_counts_per_stack)
        stack_count_volume = torch.zeros_like(volume.image)  # 3D
        coverage_threshold = 0.01

        offset = 0
        for si, sc in enumerate(slice_counts_per_stack):
            # Build ones tensor for this stack's slices only
            indicator = torch.zeros_like(stack.slices)  # (N_total, 1, H, W)
            indicator[offset:offset + sc] = 1.0
            # Mask: indicator is zero outside [offset:offset+sc], so multiplying
            # by the full mask effectively masks only this stack's slices.
            # stack.mask shape matches stack.slices: (N_total, 1, H, W)
            indicator = indicator * stack.mask.float()

            contrib = slice_acquisition_adjoint(
                transform_mat,
                psf_tensor,
                indicator,
                stack.mask if not with_background else None,
                volume.mask[None, None] if not with_background else None,
                volume.shape,
                res_s / res_r,
                False,
                False,
            )
            # Binarize: does this stack contribute here?
            stack_count_volume += (contrib[0, 0] > coverage_threshold).float()
            offset += sc

        # --- Weighted coverage: sum of confidence across all slices ---
        # Use the final p (confidence map) from the last SRR iteration
        if p is not None:
            p_for_cov = p if not isinstance(p, Stack) else p.slices
        else:
            p_for_cov = torch.ones_like(stack.slices)

        weighted_coverage = slice_acquisition_adjoint(
            transform_mat,
            psf_tensor,
            p_for_cov,
            stack.mask if not with_background else None,
            volume.mask[None, None] if not with_background else None,
            volume.shape,
            res_s / res_r,
            False,
            False,
        )[0, 0]  # squeeze batch+channel

        coverage_volume = Volume.like(volume, stack_count_volume, deep=False)
        weighted_coverage_volume = Volume.like(volume, weighted_coverage, deep=False)

        # Stats
        if volume.mask.any():
            cnt_vals = stack_count_volume[volume.mask]
            wgt_vals = weighted_coverage[volume.mask]
            logging.info(
                "Coverage stats (masked): stack-count mean=%.1f, min=%d, max=%d; "
                "weighted-coverage mean=%.1f, max=%.1f",
                cnt_vals.mean().item(), int(cnt_vals.min().item()), int(cnt_vals.max().item()),
                wgt_vals.mean().item(), wgt_vals.max().item(),
            )

        _save_nifti(stack_count_volume, os.path.join(intermediates_dir, 'coverage_stack_count.nii.gz'))
        _save_nifti(weighted_coverage, os.path.join(intermediates_dir, 'coverage_weighted.nii.gz'))
    except Exception:
        logging.warning("Failed to compute coverage map", exc_info=True)
        coverage_volume = None
        weighted_coverage_volume = None

    # ========================================================================
    # Quantile regression for uncertainty estimation
    # ========================================================================
    uncertainty_volume = None
    if estimate_uncertainty:
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        logging.info(
            "Estimating uncertainty via quantile regression: quantiles=%s, n_iter=%d",
            quantiles, n_iter_quantile,
        )
        # Use the last outer-iteration's beta/alpha for the quantile passes
        last_outer = n_iter - 1
        beta_q = max(0.01, 0.08 / (2 ** last_outer))
        alpha_q = min(1, 0.05 / beta_q)

        quantile_volumes = {}
        for tau in quantiles:
            logging.info("Quantile τ=%.2f: starting %d SRR iterations", tau, n_iter_quantile)
            # Start from the converged volume
            vol_q = Volume.like(volume, volume.image.clone(), deep=False)
            for qj in range(n_iter_quantile):
                # simulate slices from current quantile volume
                slices_sim_q, slices_weight_q = cast(
                    Tuple[Stack, Stack],
                    simulate_slices(
                        stack, vol_q, return_weight=True,
                        use_mask=not with_background, psf=psf_tensor,
                    ),
                )
                # scale
                scale_q = slices_scale(stack, slices_sim_q, slices_weight_q, p_voxel, True)
                # error
                err_q = simulated_error(stack, slices_sim_q, scale_q)
                # quantile SRR update
                vol_q = srr_update_quantile(
                    err_q, vol_q, p, alpha_q, beta_q,
                    delta * intensity_scale_ref, scale_q,
                    tau=tau,
                    use_mask=not with_background,
                    psf=psf_tensor,
                )
                if qj % 5 == 0 or qj == n_iter_quantile - 1:
                    logging.info("  τ=%.2f iter %d/%d done", tau, qj + 1, n_iter_quantile)
            quantile_volumes[tau] = vol_q
            _save_nifti(
                vol_q.image,
                os.path.join(intermediates_dir, f'quantile_tau{tau:.2f}_volume.nii.gz'),
            )
            _save_volume_png(
                vol_q.image,
                os.path.join(png_dir, f'quantile_tau{tau:.2f}_volume.png'),
                title=f'Quantile τ={tau:.2f}',
            )

        # Use median quantile (0.5) as the main output
        tau_mid = quantiles[len(quantiles) // 2]
        volume = quantile_volumes[tau_mid]
        logging.info("Using τ=%.2f quantile as the SVR output", tau_mid)

        # Uncertainty = upper quantile - lower quantile
        tau_low, tau_high = quantiles[0], quantiles[-1]
        uncertainty_image = quantile_volumes[tau_high].image - quantile_volumes[tau_low].image
        # Clamp to non-negative (upper should be >= lower)
        uncertainty_image = torch.clamp(uncertainty_image, min=0)
        uncertainty_volume = Volume.like(volume, uncertainty_image, deep=False)

        # Stats for diagnostics
        if volume.mask.any():
            mid_vals = quantile_volumes[tau_mid].image[volume.mask]
            lo_vals = quantile_volumes[tau_low].image[volume.mask]
            hi_vals = quantile_volumes[tau_high].image[volume.mask]
            unc_vals = uncertainty_image[volume.mask]
            logging.info(
                "Quantile stats (masked): Q%.1f mean=%.1f, Q%.1f mean=%.1f, "
                "Q%.1f mean=%.1f, uncertainty mean=%.1f, max=%.1f",
                tau_low, lo_vals.mean().item(),
                tau_mid, mid_vals.mean().item(),
                tau_high, hi_vals.mean().item(),
                unc_vals.mean().item(), unc_vals.max().item(),
            )

        _save_nifti(
            uncertainty_image,
            os.path.join(intermediates_dir, 'uncertainty_volume.nii.gz'),
        )

        logging.info("Uncertainty estimation complete")

    # Save combined summary PNG (volume + optional uncertainty + optional coverage)
    _save_summary_png(
        volume.image, volume.mask,
        os.path.join(png_dir, 'summary.png'),
        uncertainty_data=uncertainty_volume.image if uncertainty_volume is not None else None,
        coverage_count=coverage_volume.image if coverage_volume is not None else None,
        coverage_weighted=weighted_coverage_volume.image if weighted_coverage_volume is not None else None,
    )

    # prepare outputs
    slices_sim = cast(
        Stack,
        simulate_slices(
            stack, volume, return_weight=False, use_mask=True, psf=psf_tensor
        ),
    )
    simulated_slices = stack[:]
    output_slices = slices_sim[:]
    return volume, output_slices, simulated_slices, uncertainty_volume, coverage_volume, weighted_coverage_volume
