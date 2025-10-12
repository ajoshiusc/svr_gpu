"""Trimmed standalone_inlined package for the minimal svr_cli distribution."""

from .types import PathType, DeviceType
from .config import CHECKPOINT_DIR, MONAIFBS_URL, SVORT_URL_DICT, IQA2D_URL, IQA3D_URL
from .misc import meshgrid, resample, gaussian_blur
from .transform import (
    RigidTransform,
    transform_points,
    init_stack_transform,
    init_zero_transform,
    mat_update_resolution,
    ax_update_resolution,
    mat2euler,
    euler2mat,
    point2mat,
    mat2point,
    mat_transform_points,
    ax_transform_points,
)
from .image import (
    Image,
    Slice,
    Volume,
    Stack,
)
from .image_utils import (
    compare_resolution_affine,
    affine2transformation,
    transformation2affine,
    save_nii_volume,
    load_nii_volume,
)
from .loss import ncc_loss, ssim_loss
from .psf import get_PSF, resolution2sigma
from .inr.data import PointDataset
from .slice_acquisition import (
    slice_acquisition,
    slice_acquisition_adjoint,
)
from .svr.pipeline import slice_to_volume_reconstruction
from .svr.reconstruction import (
    psf_reconstruction,
    SRR_CG,
    srr_update,
    simulate_slices,
    slices_scale,
    simulated_error,
)
from .svr.outlier import EM, global_ncc_exclusion, local_ssim_exclusion
from .svr.registration import (
    SliceToVolumeRegistration,
    VolumeToVolumeRegistration,
    stack_registration,
)

__all__ = [
    "PathType",
    "DeviceType",
    "CHECKPOINT_DIR",
    "MONAIFBS_URL",
    "SVORT_URL_DICT",
    "IQA2D_URL",
    "IQA3D_URL",
    "meshgrid",
    "resample",
    "gaussian_blur",
    "RigidTransform",
    "transform_points",
    "init_stack_transform",
    "init_zero_transform",
    "mat_update_resolution",
    "ax_update_resolution",
    "mat2euler",
    "euler2mat",
    "point2mat",
    "mat2point",
    "mat_transform_points",
    "ax_transform_points",
    "Image",
    "Slice",
    "Volume",
    "Stack",
    "compare_resolution_affine",
    "affine2transformation",
    "transformation2affine",
    "save_nii_volume",
    "load_nii_volume",
    "ncc_loss",
    "ssim_loss",
    "get_PSF",
    "resolution2sigma",
    "PointDataset",
    "slice_acquisition",
    "slice_acquisition_adjoint",
    "slice_to_volume_reconstruction",
    "psf_reconstruction",
    "SRR_CG",
    "srr_update",
    "simulate_slices",
    "slices_scale",
    "simulated_error",
    "EM",
    "global_ncc_exclusion",
    "local_ssim_exclusion",
    "SliceToVolumeRegistration",
    "VolumeToVolumeRegistration",
    "stack_registration",
]
