import numpy as np
import nibabel as nib
import os
import json
from typing import Optional, List, Tuple
import torch

# MONAI imports
from monai.data import NibabelReader, MetaTensor
from monai.transforms import (
    LoadImage, SaveImage, RandGaussianNoise, 
    RandSpatialCrop, RandAffine, Compose,
    EnsureChannelFirst, AsChannelLast, 
    ToTensor, ToNumpy, NormalizeIntensity,
    Orientation, Spacing, Resize,
    Rand3DElastic, RandSmoothDeform
)
from monai.utils import GridSampleMode, InterpolateMode

def generate_simulated_stacks(
    mri_path: str,
    out_dir: str,
    n_stacks: int = 5,
    slices_per_stack: int = 20,
    orientations: Optional[List[Tuple[float, str]]] = None,
    noise_std: float = 0.01,
    max_disp: float = 5.0,
    acq_order: str = "interleaved-odd-even",
    max_rot_deg: float = 3.0,
    max_trans_mm: float = 1.0,
    mb_factor: int = 1,
    inplane_res: Optional[float] = None,
    slice_thickness: Optional[float] = None,
    # Nonlinear transformation parameters
    enable_nonlinear: bool = True,
    nonlinear_prob: float = 0.7,
    elastic_sigma_range: Tuple[float, float] = (3.0, 5.0),
    elastic_magnitude_range: Tuple[float, float] = (0.0, 2.0),
    smooth_deform_prob: float = 0.3,
    smooth_deform_range: float = 1.0,
):
    """
    Generate simulated stacks from a 3D MRI volume using MONAI transforms,
    with random in-plane displacement per slice and nonlinear deformations,
    ensuring the output stack correctly aligns with the input volume.

    Args:
        mri_path (str): Path to the input NIfTI MRI.
        out_dir (str): Output directory for the generated stacks.
        n_stacks (int): Number of stacks to generate.
        slices_per_stack (int): Number of slices per stack.
        orientations (list, optional): List of (angle, axis) tuples for stack orientations. If None, random orientations are used.
        noise_std (float): Standard deviation of Gaussian noise to add, as a fraction of the 99th percentile of image intensity.
        max_disp (float): Maximum random in-plane displacement in pixels per slice.
        acq_order (str): Acquisition order pattern.
        max_rot_deg (float): Maximum rotation jitter in degrees.
        max_trans_mm (float): Maximum translation jitter in mm.
        mb_factor (int): Multi-band factor for SMS.
        inplane_res (float, optional): In-plane resolution override.
        slice_thickness (float, optional): Slice thickness override.
        enable_nonlinear (bool): Enable nonlinear transformations to simulate underlying MRI changes.