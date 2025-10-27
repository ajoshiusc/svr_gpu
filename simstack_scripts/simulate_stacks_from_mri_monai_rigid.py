import numpy as np
import nibabel as nib
import os
import torch
import json

from monai.transforms import (
    LoadImaged,
    Resample,
    RandAffine,
    RandGaussianNoise,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    SpatialResample    
)

from nibabel import Nifti1Image

from nibabel.affines import apply_affine

def generate_simulated_stacks_monai_rigid(
    mri_path,
    out_dir,
    n_stacks=5,
    slices_per_stack=20,
    noise_std=0.01,
    max_disp=5.0,  # Note: This parameter is unused in a pure rigid transform model
    acq_order="interleaved-odd-even",
    max_rot_deg=3.0,
    max_trans_mm=1.0,
    mb_factor: int = 1,
    inplane_res: float = None,
    slice_thickness: float = None,
):
    """
    Generate simulated stacks from a 3D MRI volume, using MONAI random rigid
    transforms applied to the 3D volume before slice sampling.
    """
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load data using MONAI (Corrected with dictionary transforms)
    loader = Compose([
        LoadImaged(keys="image", image_only=False, reader="NibabelReader"),
        EnsureTyped(keys="image", dtype=torch.float32),
    ])
    

    img_data = loader({"image": mri_path})
    vol_tensor = img_data["image"].to(device)
    vol_affine = img_data["image_meta_dict"]["affine"]
    orig_zooms = np.array(img_data["image_meta_dict"]["pixdim"][1:4])

    # Robustly squeeze all singleton dimensions
    vol_tensor = vol_tensor.squeeze()

    # Get spatial shape (H, W, D)
    vol_shape = vol_tensor.shape[-3:]