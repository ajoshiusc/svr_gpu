import logging
from typing import List, Optional, Tuple, cast
import torch
import numpy as np
from .registration import SliceToVolumeRegistration
from .outlier import EM, global_ncc_exclusion, local_ssim_exclusion
from .reconstruction import (
    psf_reconstruction,
    srr_update,
    simulate_slices,
    slices_scale,
    simulated_error,
)
from ..types import DeviceType, PathType
from ..psf import get_PSF
import os
import nibabel as nib
from ..image import Volume, Slice, load_volume, load_mask, Stack
from ..inr.data import PointDataset
import inspect


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
) -> Tuple[Stack, float, float]:
    masked_v = stack.slices[stack.mask]
    mean_intensity = masked_v.mean().item()
    max_intensity = masked_v.max().item()
    min_intensity = masked_v.min().item()
    stack.slices = stack.slices * (output_intensity_mean / mean_intensity)
    max_intensity = max_intensity * (output_intensity_mean / mean_intensity)
    min_intensity = min_intensity * (output_intensity_mean / mean_intensity)
    return stack, max_intensity, min_intensity


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
    sample_mask: Optional[PathType] = None,
    sample_orientation: Optional[PathType] = None,
    psf: str = "gaussian",
    device: DeviceType = torch.device("cpu"),
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

    _save_nifti(stack.slices, os.path.join(intermediates_dir, 'stack_preprocessed.nii.gz'))
    _save_nifti(stack.mask.float(), os.path.join(intermediates_dir, 'stack_preprocessed_mask.nii.gz'))
    
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
    _save_nifti(volume.image, os.path.join(intermediates_dir, 'initial_mask_volume.nii.gz'))

    # data normalization
    stack, max_intensity, min_intensity = _normalize(stack, output_intensity_mean)
    _save_nifti(stack.slices, os.path.join(intermediates_dir, 'stack_normalized.nii.gz'))

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
    _save_numpy(psf_tensor, os.path.join(intermediates_dir, 'psf_tensor.npy'))

    for i in range(n_iter):
        logging.info("outer %d", i)
        # Save volume at the start of outer iteration
        affine = getattr(volume, 'affine', None)
        _save_nifti(volume.image, os.path.join(recon_dir, f'recon_outer_{i:02d}.nii.gz'), affine)
        # slice-to-volume registration
        if i > 0:  # skip slice-to-volume registration for the first iteration
            svr = SliceToVolumeRegistration(
                num_levels=3,
                num_steps=5,
                step_size=2,
                max_iter=30,
            )
            slices_transform, _ = svr(
                stack,
                volume,
                use_mask=True,
            )
            stack.transformation = slices_transform
            # Save stack transformation after registration
            _save_numpy(stack.transformation.matrix(), os.path.join(intermediates_dir, f'stack_transform_outer_{i:02d}.npy'))

        # global structual exclusion
        if i > 0 and not no_global_exclusion:
            stack.mask = slices_mask_backup.clone()
            excluded = global_ncc_exclusion(stack, volume, global_ncc_threshold)
            stack.mask[excluded] = False
            # Save mask after global exclusion
            _save_nifti(stack.mask.float(), os.path.join(intermediates_dir, f'mask_after_global_exclusion_outer_{i:02d}.nii.gz'))

        # PSF reconstruction & volume mask
        volume = psf_reconstruction(
            stack,
            volume,
            update_mask=is_refine_mask,
            use_mask=not with_background,
            psf=psf_tensor,
        )
        _save_nifti(volume.image, os.path.join(intermediates_dir, f'volume_after_psf_outer_{i:02d}.nii.gz'))

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
                os.path.join(intermediates_dir, f'slices_sim_outer_{i:02d}_inner_{j:02d}.nii.gz'),
            )
            _save_nifti(
                slices_weight.slices,
                os.path.join(intermediates_dir, f'slices_weight_outer_{i:02d}_inner_{j:02d}.nii.gz'),
            )
            # scale
            scale = slices_scale(stack, slices_sim, slices_weight, p_voxel, True)
            _save_numpy(scale, os.path.join(intermediates_dir, f'scale_outer_{i:02d}_inner_{j:02d}.npy'))
            # err
            err = simulated_error(stack, slices_sim, scale)
            _save_nifti(
                err.slices,
                os.path.join(intermediates_dir, f'err_outer_{i:02d}_inner_{j:02d}.nii.gz'),
            )
            # EM robust statistics
            if (not no_pixel_robust_statistics) or (not no_slice_robust_statistics):
                p_voxel, p_slice = em(err, slices_weight, scale, 1)
                _save_nifti(
                    p_voxel,
                    os.path.join(intermediates_dir, f'p_voxel_outer_{i:02d}_inner_{j:02d}.nii.gz'),
                )
                _save_numpy(
                    p_slice,
                    os.path.join(intermediates_dir, f'p_slice_outer_{i:02d}_inner_{j:02d}.npy'),
                )
                if no_pixel_robust_statistics:  # reset p_voxel
                    p_voxel = torch.ones_like(stack.slices)
            p = p_voxel
            if not no_slice_robust_statistics:
                p = p_voxel * p_slice.view(-1, 1, 1, 1)
            _save_nifti(
                p,
                os.path.join(intermediates_dir, f'p_combined_outer_{i:02d}_inner_{j:02d}.nii.gz'),
            )
            # local structural exclusion
            if not no_local_exclusion:
                p = p * local_ssim_exclusion(stack, slices_sim, local_ssim_threshold)
                _save_nifti(
                    p,
                    os.path.join(intermediates_dir, f'p_after_local_exclusion_outer_{i:02d}_inner_{j:02d}.nii.gz'),
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
                delta * output_intensity_mean,
                use_mask=not with_background,
                psf=psf_tensor,
            )
            # Save volume after inner iteration
            affine = getattr(volume, 'affine', None)
            _save_nifti(volume.image, os.path.join(recon_dir, f'recon_outer_{i:02d}_inner_{j:02d}.nii.gz'), affine)

    # reconstruction finished
    # Save final transforms after SVR completes
    _save_numpy(stack.transformation.matrix(), os.path.join(transforms_dir, 'transforms_svr_final.npy'))
    _save_nifti(volume.image, os.path.join(intermediates_dir, 'volume_final.nii.gz'))
    _save_nifti(volume.mask.float(), os.path.join(intermediates_dir, 'volume_mask_final.nii.gz'))
    
    # prepare outputs
    slices_sim = cast(
        Stack,
        simulate_slices(
            stack, volume, return_weight=False, use_mask=True, psf=psf_tensor
        ),
    )
    simulated_slices = stack[:]
    output_slices = slices_sim[:]
    return volume, output_slices, simulated_slices
