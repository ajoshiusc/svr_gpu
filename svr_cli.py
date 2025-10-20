#!/usr/bin/env python3
"""
Complete Standalone SVR with inlined preprocessing (minimal package version)

This version inlines the inputs/outputs functionality to remove dependency on
nesvor_extracted.cli.io module while maintaining identical outputs.
"""

import argparse
import logging
import sys
import os
import tempfile
import shutil
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from argparse import Namespace
from typing import List, Dict, Any, Optional, Tuple
from skimage.filters import threshold_multiotsu
from skimage.morphology import dilation, ball

from standalone_inlined import Stack, Volume, RigidTransform
from standalone_inlined.pipeline import (
    _register,
    _segment_stack,
    _correct_bias_field,
    _assess,
)
from reorient_stacks import reorient_to_axial
from standalone_inlined.svr import slice_to_volume_reconstruction

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def _configure_cuda_linalg() -> None:
    if not torch.cuda.is_available():
        return
    preferred = getattr(torch.backends.cuda, "preferred_linalg_library", None)
    if preferred is None:
        return
    try:
        preferred("magma")
        logger.debug("Configured CUDA MAGMA backend for linear algebra ops")
    except RuntimeError as exc:  # pragma: no cover - best effort safeguard
        logger.warning("Failed to switch CUDA linalg backend: %s", exc)


_configure_cuda_linalg()


# ============================================================================
# STACK REORIENTATION WRAPPER
# ============================================================================

def preprocess_stacks_orientation(input_stacks: List[str], temp_dir: str) -> List[str]:
    """
    Preprocess input stacks to ensure they are all oriented with slices in XY plane.
    
    Args:
        input_stacks: List of input stack file paths
        temp_dir: Temporary directory to store reoriented stacks
        
    Returns:
        List of paths to reoriented stacks (in temp_dir)
    """
    logger.info("Reorienting stacks to ensure slices are in XY plane...")
    reoriented_paths = []
    
    for i, input_path in enumerate(input_stacks):
        # Create output path in temp directory
        filename = os.path.basename(input_path)
        output_path = os.path.join(temp_dir, f"reoriented_{i}_{filename}")
        
        # Reorient the stack using imported function
        reorient_to_axial(input_path, output_path, verbose=False)
        reoriented_paths.append(output_path)
    
    logger.info(f"  Reoriented {len(reoriented_paths)} stacks")
    return reoriented_paths


# ============================================================================
# INLINED I/O FUNCTIONS (from nesvor_extracted.image and image_utils)
# ============================================================================

def load_nii_volume(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load NIfTI volume and return data, resolutions, and affine matrix"""
    img = nib.load(path)
    
    dim = img.header["dim"]
    assert dim[0] == 3 or (dim[0] > 3 and all(d == 1 for d in dim[4:])), (
        "Expect a 3D volume but the input is %dD" % dim[0]
    )
    
    volume = img.get_fdata().astype(np.float32)
    while volume.ndim > 3:
        volume = volume.squeeze(-1)
    volume = volume.transpose(2, 1, 0)
    
    resolutions = img.header["pixdim"][1:4]
    
    affine = img.affine
    if np.any(np.isnan(affine)):
        affine = img.get_qform()
    
    return volume, resolutions, affine


def compare_resolution_affine(r1, a1, r2, a2, s1, s2) -> bool:
    """Compare resolutions and affine matrices"""
    r1 = np.array(r1)
    a1 = np.array(a1)
    r2 = np.array(r2)
    a2 = np.array(a2)
    if s1 != s2:
        return False
    if r1.shape != r2.shape:
        return False
    if np.amax(np.abs(r1 - r2)) > 1e-3:
        return False
    if a1.shape != a2.shape:
        return False
    if np.amax(np.abs(a1 - a2)) > 1e-3:
        return False
    return True


def affine2transformation(
    volume: torch.Tensor,
    mask: torch.Tensor,
    resolutions: np.ndarray,
    affine: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor, RigidTransform]:
    """Convert NIfTI affine matrix to NeSVoR transformation"""
    device = volume.device
    d, h, w = volume.shape

    R = affine[:3, :3]
    negative_det = np.linalg.det(R) < 0

    T = affine[:3, -1:]  # T = R @ (-T0 + T_r)
    R = R @ np.linalg.inv(np.diag(resolutions))

    T0 = np.array([(w - 1) / 2 * resolutions[0], (h - 1) / 2 * resolutions[1], 0])
    T = np.linalg.inv(R) @ T + T0.reshape(3, 1)

    tz = (
        torch.arange(0, d, device=device, dtype=torch.float32) * resolutions[2]
        + T[2].item()
    )
    tx = torch.ones_like(tz) * T[0].item()
    ty = torch.ones_like(tz) * T[1].item()
    t = torch.stack((tx, ty, tz), -1).view(-1, 3, 1)
    R = torch.tensor(R, device=device).unsqueeze(0).repeat(d, 1, 1)

    if negative_det:
        volume = torch.flip(volume, (-1,))
        mask = torch.flip(mask, (-1,))
        t[:, 0, -1] *= -1
        R[:, :, 0] *= -1

    transformation = RigidTransform(
        torch.cat((R, t), -1).to(torch.float32), trans_first=True
    )

    return volume, mask, transformation


def transformation2affine(
    volume: torch.Tensor,
    transformation: RigidTransform,
    resolution_x: float,
    resolution_y: float,
    resolution_z: float,
) -> np.ndarray:
    """Convert NeSVoR transformation to NIfTI affine matrix"""
    mat = transformation.matrix(trans_first=True).detach().cpu().numpy()
    assert mat.shape[0] == 1
    R = mat[0, :, :-1]
    T = mat[0, :, -1:]
    d, h, w = volume.shape
    affine = np.eye(4)
    T[0] -= (w - 1) / 2 * resolution_x
    T[1] -= (h - 1) / 2 * resolution_y
    T[2] -= (d - 1) / 2 * resolution_z
    T = R @ T.reshape(3, 1)
    R = R @ np.diag([resolution_x, resolution_y, resolution_z])
    affine[:3, :] = np.concatenate((R, T), -1)
    return affine


def save_nii_volume(
    path: str,
    volume: torch.Tensor,
    affine: np.ndarray,
) -> None:
    """Save volume as NIfTI file"""
    assert len(volume.shape) == 3 or (len(volume.shape) == 4 and volume.shape[1] == 1)
    if len(volume.shape) == 4:
        volume = volume.squeeze(1)
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy().transpose(2, 1, 0)
    else:
        volume = volume.transpose(2, 1, 0)
    if isinstance(affine, torch.Tensor):
        affine = affine.detach().cpu().numpy()
    if affine is None:
        affine = np.eye(4)
    if volume.dtype == bool and isinstance(volume, np.ndarray):
        volume = volume.astype(np.int16)
    img = nib.nifti1.Nifti1Image(volume, affine)
    img.header.set_xyzt_units(2)
    img.header.set_qform(affine, code="aligned")
    img.header.set_sform(affine, code="scanner")
    nib.save(img, path)


def load_stack(
    path_vol: str,
    path_mask: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
) -> Stack:
    """Load NIfTI stack file"""
    slices, resolutions, affine = load_nii_volume(path_vol)
    if path_mask is None:
        mask = np.ones_like(slices, dtype=bool)
    else:
        mask, resolutions_m, affine_m = load_nii_volume(path_mask)
        mask = mask > 0
        if not compare_resolution_affine(
            resolutions, affine, resolutions_m, affine_m, slices.shape, mask.shape
        ):
            raise Exception(
                "Error: the sizes/resolutions/affine transformations of the input stack and stack mask do not match!"
            )

    slices_tensor = torch.tensor(slices, device=device)
    mask_tensor = torch.tensor(mask, device=device)

    slices_tensor, mask_tensor, transformation = affine2transformation(
        slices_tensor, mask_tensor, resolutions, affine
    )

    # Try to load SMS metadata from JSON sidecar
    mb_factor = 1
    acquisition_order = None
    import json
    json_path = str(path_vol).replace('.nii.gz', '.json').replace('.nii', '.json')
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                metadata = json.load(f)
                mb_factor = metadata.get('mb_factor', 1)
                acquisition_order = metadata.get('acquisition_order', None)
        except Exception:
            pass  # Fail silently if JSON can't be loaded

    return Stack(
        slices=slices_tensor.unsqueeze(1),
        mask=mask_tensor.unsqueeze(1),
        transformation=transformation,
        resolution_x=resolutions[0],
        resolution_y=resolutions[1],
        thickness=resolutions[2],
        gap=resolutions[2],
        name=str(path_vol),
        mb_factor=mb_factor,
        acquisition_order=acquisition_order,
    )


def load_volume(
    path_vol: str,
    path_mask: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
) -> Volume:
    """Load NIfTI volume file"""
    vol, resolutions, affine = load_nii_volume(path_vol)
    if path_mask is None:
        mask = np.ones_like(vol, dtype=bool)
    else:
        mask, resolutions_m, affine_m = load_nii_volume(path_mask)
        mask = mask > 0
        if not compare_resolution_affine(
            resolutions, affine, resolutions_m, affine_m, vol.shape, mask.shape
        ):
            raise Exception(
                "Error: the sizes/resolutions/affine transformations of the input stack and stack mask do not match!"
            )

    vol_tensor = torch.tensor(vol, device=device)
    mask_tensor = torch.tensor(mask, device=device)

    vol_tensor, mask_tensor, transformation = affine2transformation(
        vol_tensor, mask_tensor, resolutions, affine
    )

    transformation = RigidTransform(transformation.axisangle().mean(0, keepdim=True))

    return Volume(
        image=vol_tensor,
        mask=mask_tensor,
        transformation=transformation,
        resolution_x=resolutions[0],
        resolution_y=resolutions[1],
        resolution_z=resolutions[2],
    )


def load_mask(path_mask: str, device: torch.device = torch.device("cpu")) -> Volume:
    """Load NIfTI mask file"""
    return load_volume(path_mask, path_mask, device)


# ============================================================================
# INLINED PREPROCESSING FUNCTIONS (from nesvor_extracted.preprocessing.masking)
# ============================================================================

def otsu_thresholding(stacks: List[Stack], nbins: int = 256) -> List[Stack]:
    """Apply Otsu thresholding to stacks"""
    for stack in stacks:
        thresholds = threshold_multiotsu(
            image=stack.slices.cpu().numpy(), classes=2, nbins=nbins
        )
        assert len(thresholds) == 1
        mask = stack.slices > thresholds[0]
        mask = torch.tensor(
            dilation(mask.squeeze().cpu().numpy(), footprint=ball(3)),
            dtype=mask.dtype,
            device=mask.device,
        ).view(mask.shape)
        stack.mask = torch.logical_and(stack.mask, mask)
    return stacks


def thresholding(stacks: List[Stack], threshold: float) -> List[Stack]:
    """Apply threshold to stacks"""
    for stack in stacks:
        mask = stack.slices > threshold
        stack.mask = torch.logical_and(stack.mask, mask)
    return stacks


def stack_intersect(stacks: List[Stack], box: bool) -> Volume:
    """Get intersection of stacks as a volume mask"""
    return volume_intersect([stack.get_mask_volume() for stack in stacks], box)


def volume_intersect(volumes: List[Volume], box: bool) -> Volume:
    """Get intersection of volumes"""
    volume = volumes[0].clone()
    for i in range(1, len(volumes)):
        assign_mask = volume.mask.clone()
        volume.mask[assign_mask] = volumes[i].sample_points(volume.xyz_masked) > 0
    mask = volume.mask
    if not mask.any():
        raise ValueError("The intersection of inputs is empty!")
    if box:
        nz = torch.nonzero(mask.sum(dim=[1, 2]))
        i0 = int(nz[0, 0])
        i1 = int(nz[-1, 0] + 1)
        nz = torch.nonzero(mask.sum(dim=[0, 2]))
        j0 = int(nz[0, 0])
        j1 = int(nz[-1, 0] + 1)
        nz = torch.nonzero(mask.sum(dim=[0, 1]))
        k0 = int(nz[0, 0])
        k1 = int(nz[-1, 0] + 1)
        mask[i0:i1, j0:j1, k0:k1] = True
    volume.image = mask.float()
    return volume


# ============================================================================
# INLINED INPUT/OUTPUT FUNCTIONS (from nesvor_extracted.cli.io)
# ============================================================================

def load_inputs(args: Namespace) -> Tuple[Dict, Namespace]:
    """
    Load and preprocess input stacks (inlined from nesvor_extracted.cli.io.inputs)
    """
    input_dict: Dict[str, Any] = dict()
    
    # Track if we need to clean up temp directory
    temp_dir = None
    
    if getattr(args, "input_stacks", None) is not None:
        # Preprocess stacks: reorient to axial if needed (unless disabled)
        auto_reorient = not getattr(args, "no_auto_reorient", False)
        
        if auto_reorient:
            logger.info("Auto-reorientation enabled")
            # Create temp directory for reoriented stacks
            temp_dir = tempfile.mkdtemp(prefix="svr_reoriented_")
            logger.debug(f"  Created temp directory: {temp_dir}")
            
            # Reorient all stacks
            reoriented_stack_paths = preprocess_stacks_orientation(args.input_stacks, temp_dir)
            
            # Replace input stack paths with reoriented versions
            stack_paths = reoriented_stack_paths
        else:
            logger.info("Auto-reorientation disabled")
            stack_paths = args.input_stacks
        
        # Load stacks from (possibly reoriented) paths
        input_stacks = []
        logger.info("Loading stacks")
        for i, f in enumerate(stack_paths):
            stack = load_stack(
                f,
                args.stack_masks[i]
                if getattr(args, "stack_masks", None) is not None
                else None,
                device=args.device,
            )
            if getattr(args, "thicknesses", None) is not None:
                stack.thickness = args.thicknesses[i]
            input_stacks.append(stack)
        
        # Stack thresholding
        logger.info("Background thresholding")
        input_stacks = thresholding(input_stacks, args.background_threshold)
        if getattr(args, "otsu_thresholding", False):
            logger.info("Applying Otsu thresholding")
            input_stacks = otsu_thresholding(input_stacks)
        
        # Volume mask
        volume_mask: Optional[Volume]
        if getattr(args, "volume_mask", None):
            logger.info("Loading volume mask")
            volume_mask = load_mask(args.volume_mask, device=args.device)
        elif getattr(args, "stacks_intersection", False):
            logger.info("Creating volume mask using intersection of stacks")
            volume_mask = stack_intersect(input_stacks, box=True)
        else:
            volume_mask = None
        
        if volume_mask is not None:
            logger.info("Applying volume mask")
            for stack in input_stacks:
                stack.apply_volume_mask(volume_mask)
        
        input_dict["input_stacks"] = input_stacks
        input_dict["volume_mask"] = volume_mask
        
        # Clean up temp directory if it was created
        if temp_dir is not None:
            logger.debug(f"Cleaning up temp directory: {temp_dir}")
            shutil.rmtree(temp_dir)
    else:
        raise ValueError("No input stacks provided!")
    
    return input_dict, args


def save_outputs(data: Dict, args: Namespace) -> None:
    """
    Save outputs (inlined from nesvor_extracted.cli.io.outputs)
    """
    if getattr(args, "output_volume", None) and "output_volume" in data:
        output_volume = data["output_volume"]
        
        # Rescale intensity
        if args.output_intensity_mean:
            if output_volume.mask.any():
                scale_factor = args.output_intensity_mean / output_volume.image[output_volume.mask].mean()
            else:
                scale_factor = args.output_intensity_mean / output_volume.image.mean()
            output_volume.image *= scale_factor
        
        # Prepare for saving
        affine = transformation2affine(
            output_volume.image,
            output_volume.transformation,
            float(output_volume.resolution_x),
            float(output_volume.resolution_y),
            float(output_volume.resolution_z),
        )
        
        # Apply mask if requested
        if not getattr(args, "with_background", False):
            volume_data = output_volume.image * output_volume.mask.to(output_volume.image.dtype)
        else:
            volume_data = output_volume.image
        
        # Save to NIfTI
        save_nii_volume(args.output_volume, volume_data, affine)
        logger.info(f"Saved output volume to {args.output_volume}")


# ============================================================================
# PREPROCESSING PIPELINE
# ============================================================================

# ============================================================================
# PREPROCESSING PIPELINE
# ============================================================================

def preprocess_inputs(args: Namespace):
    """
    Fully replicate the Reconstruct.preprocess() function from NeSVoR.
    This ensures exact same preprocessing as standalone_svr.py
    """
    # Load inputs using inlined function (replaces nesvor_extracted.cli.io.inputs)
    # This handles: loading stacks, thresholding, volume masking
    input_dict, args = load_inputs(args)
    
    if "input_stacks" in input_dict and input_dict["input_stacks"]:
        # Segmentation (if enabled)
        if args.segmentation and str(args.segmentation).lower() != 'none':
            logger.info("Running segmentation")
            input_dict["input_stacks"] = _segment_stack(
                args, input_dict["input_stacks"]
            )
        
        # Bias field correction (if enabled)
        if args.bias_field_correction:
            logger.info("Running bias field correction")
            input_dict["input_stacks"] = _correct_bias_field(
                args, input_dict["input_stacks"]
            )
        
        # Assessment (if enabled)
        if args.metric != "none":
            logger.info("Running slice assessment")
            input_dict["input_stacks"], _ = _assess(
                args, input_dict["input_stacks"], False
            )
        
        # Registration - convert stacks to slices with motion correction
        logger.info(f"Running registration: {args.registration}")
        input_dict["input_slices"] = _register(
            args, input_dict["input_stacks"]
        )
    elif "input_slices" in input_dict and input_dict["input_slices"]:
        pass
    else:
        raise ValueError("No data found!")
    
    return input_dict, args


def run_svr(args: Namespace):
    """
    Run the complete SVR pipeline using exact NeSVoR components.
    """
    # Preprocess inputs (loads, thresholds, masks, segments, corrects bias, assesses, registers)
    logger.info("Preprocessing inputs...")
    input_dict, args = preprocess_inputs(args)
    
    # Run SVR reconstruction (exact NeSVoR implementation)
    logger.info("Running SVR reconstruction...")
    output_volume, output_slices, mask = slice_to_volume_reconstruction(
        slices=input_dict["input_slices"],
        sample_mask=input_dict.get("volume_mask", None),
        with_background=args.with_background,
        output_resolution=args.output_resolution,
        output_intensity_mean=args.output_intensity_mean,
        delta=args.delta,
        n_iter=args.n_iter,
        n_iter_rec=args.n_iter_rec,
        global_ncc_threshold=args.global_ncc_threshold,
        local_ssim_threshold=args.local_ssim_threshold,
        no_slice_robust_statistics=args.no_slice_robust_statistics,
        no_pixel_robust_statistics=args.no_pixel_robust_statistics,
        no_global_exclusion=args.no_global_exclusion,
        no_local_exclusion=args.no_local_exclusion,
        psf=args.psf,
        device=args.device,
    )
    
    # Save outputs
    logger.info("Saving outputs...")
    output_data = {
        "output_volume": output_volume,
        "mask": mask,
    }
    save_outputs(output_data, args)
    
    logger.info("Done!")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Complete Standalone SVR using exact NeSVoR preprocessing'
    )

    # Input/output arguments
    parser.add_argument('--input-stacks', required=True, nargs='+',
                        help='Input stack files (NIFTI format)')
    parser.add_argument('--output', '--output-volume', dest='output_volume',
                        help='Output volume path')
    parser.add_argument('--device', type=int, default=0,
                        help='CUDA device ID (default: 0). Use -1 to force CPU.')

    # Preprocessing arguments
    parser.add_argument('--stack-masks', nargs='+', default=None,
                        help='Stack masks (optional)')
    parser.add_argument('--thicknesses', type=float, nargs='+', default=None,
                        help='Slice thicknesses for each stack')
    parser.add_argument('--stacks-intersection', action='store_true',
                        help='Only consider intersection of input stacks')
    parser.add_argument('--background-threshold', type=float, default=0.0,
                        help='Background threshold for stack masking')
    parser.add_argument('--otsu-thresholding', action='store_true',
                        help='Apply Otsu thresholding to each input stack')
    parser.add_argument('--volume-mask', default=None,
                        help='Volume mask file (optional)')
    parser.add_argument('--segmentation', default=None,
                        help='Segmentation method (twai, multi-label, etc.)')
    parser.add_argument('--no-auto-reorient', action='store_true',
                        help='Disable automatic reorientation of stacks to axial orientation (slices in XY plane)')
    parser.add_argument('--bias-field-correction', action='store_true',
                        help='Apply N4 bias field correction')
    parser.add_argument('--n-proc-n4', type=int, default=1,
                        help='Number of processes for N4 bias field correction (default: 1)')
    parser.add_argument('--shrink-factor-n4', type=int, default=4,
                        help='Shrink factor for N4 bias field correction (default: 4)')
    parser.add_argument('--n-iterations-n4', nargs='+', type=int, default=[50, 50, 50, 50],
                        help='Number of iterations for N4 bias field correction per level')
    parser.add_argument('--n-fitting-levels-n4', type=int, default=4,
                        help='Number of fitting levels for N4 bias field correction (default: 4)')
    parser.add_argument('--convergence-threshold-n4', type=float, default=0.001,
                        help='Convergence threshold for N4 bias field correction (default: 0.001)')
    parser.add_argument('--spline-order-n4', type=int, default=3,
                        help='Spline order for N4 bias field correction (default: 3)')
    parser.add_argument('--wiener-filter-noise-n4', type=float, default=0.01,
                        help='Wiener filter noise for N4 bias field correction (default: 0.01)')
    parser.add_argument('--n-histogram-bins-n4', type=int, default=200,
                        help='Number of histogram bins for N4 bias field correction (default: 200)')
    parser.add_argument('--full-width-at-half-maximum-n4', type=float, default=0.15,
                        help='Full width at half maximum for N4 bias field correction (default: 0.15)')
    parser.add_argument('--mask-label-n4', type=int, default=1,
                        help='Mask label for N4 bias field correction (default: 1)')
    parser.add_argument('--metric', default='none',
                        help='Assessment metric (ncc, ssim, nmi, none)')
    parser.add_argument('--registration', default='svort',
                        help='Registration method (svort, none)')
    parser.add_argument('--svort-version', type=str, default='v2', choices=['v1', 'v2'],
                        help='Version of SVoRT model')
    parser.add_argument('--scanner-space', action='store_true',
                        help='Perform registration in scanner space')

    # Segmentation arguments
    parser.add_argument('--batch-size-seg', type=int, default=16,
                        help='Batch size for segmentation')
    parser.add_argument('--no-augmentation-seg', action='store_true',
                        help='Disable inference augmentation in segmentation')
    parser.add_argument('--dilation-radius-seg', type=float, default=1.0,
                        help='Dilation radius for segmentation mask (mm)')
    parser.add_argument('--threshold-small-seg', type=float, default=0.1,
                        help='Threshold for removing small masks')

    # Assessment arguments
    parser.add_argument('--no-augmentation-assess', action='store_true',
                        help='Disable augmentation in IQA network')
    parser.add_argument('--batch-size-assess', type=int, default=8,
                        help='Batch size for assessment (default: 8)')
    parser.add_argument('--filter-method', type=str, default='none',
                        choices=['none', 'top', 'bottom', 'percentage', 'threshold'],
                        help='Method for filtering stacks based on quality (default: none)')
    parser.add_argument('--cutoff', type=float, default=0.0,
                        help='Cutoff value for filtering (interpretation depends on filter-method)')

    # SVR reconstruction arguments
    parser.add_argument('--output-resolution', type=float, default=0.8,
                        help='Output resolution in mm (default: 0.8)')
    parser.add_argument('--n-iter', type=int, default=3,
                        help='Number of outer iterations (default: 3)')
    parser.add_argument('--n-iter-rec', type=int, nargs='+', default=[7, 7, 21],
                        help='Number of reconstruction iterations per outer iteration')
    parser.add_argument('--delta', type=float, default=150.0/700.0,
                        help='Delta for robust loss (default: 0.214 = 150/700)')
    parser.add_argument('--output-intensity-mean', type=float, default=700.0,
                        help='Target mean intensity for output normalization')
    parser.add_argument('--with-background', action='store_true',
                        help='Include background in output volume')
    parser.add_argument('--psf', type=str, default='gaussian', choices=['gaussian', 'box'],
                        help='PSF type (default: gaussian)')
    
    # Robust statistics and outlier rejection arguments
    parser.add_argument('--global-ncc-threshold', type=float, default=0.5,
                        help='Global NCC threshold for slice exclusion (default: 0.5, lower=more permissive)')
    parser.add_argument('--local-ssim-threshold', type=float, default=0.4,
                        help='Local SSIM threshold for pixel exclusion (default: 0.4, lower=more permissive)')
    parser.add_argument('--no-slice-robust-statistics', action='store_true',
                        help='Disable per-slice robust statistics (outlier weighting)')
    parser.add_argument('--no-pixel-robust-statistics', action='store_true',
                        help='Disable per-pixel robust statistics (outlier weighting)')
    parser.add_argument('--no-global-exclusion', action='store_true',
                        help='Disable global structural exclusion (NCC-based slice rejection)')
    parser.add_argument('--no-local-exclusion', action='store_true',
                        help='Disable local structural exclusion (SSIM-based pixel rejection)')

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # Setup device
    if isinstance(args.device, int) and args.device == -1:
        args.device = torch.device('cpu')
        logger.info("Using CPU device (requested by --device -1)")
    elif torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.device}')
        logger.info(f"Using CUDA device: {args.device}")
    else:
        args.device = torch.device('cpu')
        logger.info("CUDA not available, using CPU")

    # Run SVR
    run_svr(args)


if __name__ == '__main__':
    main()
