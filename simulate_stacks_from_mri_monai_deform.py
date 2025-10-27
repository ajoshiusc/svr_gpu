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
        nonlinear_prob (float): Probability of applying nonlinear transformations per stack.
        elastic_sigma_range (tuple): Sigma range for elastic deformation (controls smoothness).
        elastic_magnitude_range (tuple): Magnitude range for elastic deformation displacement.
        smooth_deform_prob (float): Probability of applying smooth deformation.
        smooth_deform_range (float): Range for smooth deformation displacement.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Load image using MONAI
    loader = LoadImage(reader=NibabelReader, ensure_channel_first=True, dtype=np.float32)
    img_data = loader(mri_path)
    vol = img_data.numpy()  # Get numpy array
    # Remove channel dimension if present
    if vol.ndim == 4 and vol.shape[0] == 1:
        vol = vol[0]  # Remove channel dimension
    # Get metadata from the MetaTensor
    affine = img_data.meta['affine'] if 'affine' in img_data.meta else img_data.affine
    orig_zooms = img_data.meta['pixdim'][1:4] if 'pixdim' in img_data.meta else img_data.pixdim[1:4]
    
    # Calculate scaled noise level based on image intensity
    if vol.max() > 0:
        data_max = np.percentile(vol[vol > 0], 99)
        scaled_noise_std = noise_std * data_max
    else:
        scaled_noise_std = 0

    def build_acq_order(nz: int, mode: str) -> List[int]:
        """Build acquisition order based on mode."""
        if mode == "sequential-asc":
            return list(range(nz))
        if mode == "sequential-desc":
            return list(range(nz - 1, -1, -1))
        if mode == "interleaved-even-odd":
            return list(range(0, nz, 2)) + list(range(1, nz, 2))
        # default: interleaved-odd-even
        return list(range(0, nz, 2)) + list(range(1, nz, 2))

    def apply_affine(affine_matrix: np.ndarray, coords: np.ndarray) -> np.ndarray:
        """Apply affine transformation to coordinates."""
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)
        return (affine_matrix @ np.hstack([coords, np.ones((coords.shape[0], 1))]).T)[:3].T

    def shift_image(data: np.ndarray, shift_vec: Tuple[float, float]) -> np.ndarray:
        """Apply in-plane shift using MONAI-compatible approach."""
        # Convert to tensor for MONAI transforms
        data_tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        
        # Create affine matrix for translation
        shift_affine = torch.eye(4, dtype=torch.float32)
        shift_affine[0, 3] = shift_vec[0]  # x translation
        shift_affine[1, 3] = shift_vec[1]  # y translation
        
        # Apply transformation using MONAI's RandAffine
        transform = RandAffine(
            prob=1.0,
            translate_range=[(shift_vec[0], shift_vec[0]), (shift_vec[1], shift_vec[1]), (0, 0)],
            mode=InterpolateMode.BILINEAR,
            padding_mode="nearest"
        )
        
        # Apply the transformation
        shifted_tensor = transform(data_tensor, mode="bilinear")
        return shifted_tensor.squeeze(0).squeeze(0).numpy()

    def apply_nonlinear_deformation(volume: np.ndarray, save_deformation: bool = False, stack_idx: int = 0) -> tuple:
        """Apply random nonlinear deformation using MONAI's deformation functions."""
        if not enable_nonlinear or np.random.random() > nonlinear_prob:
            if save_deformation:
                # Create zero deformation field
                deformation_field = np.zeros((*volume.shape, 3), dtype=np.float32)
                return volume, deformation_field
            return volume
            
        # Convert to tensor with proper dimensions for MONAI
        # MONAI expects (batch, channel, spatial_dims) - so we need 5D tensor for 3D volume
        volume_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W, D)

        # Apply MONAI's Rand3DElastic
        elastic_transform = Rand3DElastic(
            sigma_range=[6,8],#elastic_sigma_range,
            magnitude_range=[100,300],#elastic_magnitude_range,
            prob=1.0,
            mode=InterpolateMode.BILINEAR,
            padding_mode="border"
        )
        deformed_tensor = elastic_transform(volume_tensor[0])
        deformed_vol = deformed_tensor.squeeze(0).numpy()
        # Rand3DElastic does not return the deformation field, so we cannot save it directly
        if save_deformation:
            deformation_field = None
            return deformed_vol, deformation_field
        return deformed_vol

    for i in range(n_stacks):
        # Apply nonlinear deformation to simulate underlying MRI changes
        # Each stack gets its own deformation to simulate different acquisition states
        result = apply_nonlinear_deformation(vol.copy(), save_deformation=True, stack_idx=i)
        if isinstance(result, tuple):
            deformed_vol, deformation_field = result
        else:
            deformed_vol = result
            deformation_field = None
        
        ## Determine orientation
        #if orientations is not None:
        #    angle, axis = orientations[i % len(orientations)]
        #else:
        #    axis = np.random.choice(['x', 'y'])
        #    angle = np.random.uniform(15, 345)

        #theta = np.deg2rad(angle)
        #if axis == 'x':
        #    rotmat = np.array([
        #        [1, 0, 0],
        #        [0, np.cos(theta), -np.sin(theta)],
        #        [0, np.sin(theta), np.cos(theta)]])
        #elif axis == 'y':
        #    rotmat = np.array([
        #        [np.cos(theta), 0, np.sin(theta)],
        #        [0, 1, 0],
        #        [-np.sin(theta), 0, np.cos(theta)]])
        #else:  # 'z'
        #    rotmat = np.array([
        #        [np.cos(theta), -np.sin(theta), 0],
        #        [np.sin(theta), np.cos(theta), 0],
        #        [0, 0, 1]
        #    ])

        # Determine voxel sizes
        if inplane_res is None:
            res_x = orig_zooms[0]
            res_y = orig_zooms[1]
        else:
            res_x = float(inplane_res)
            res_y = float(inplane_res)
        if slice_thickness is None:
            gap = (vol.shape[2] / slices_per_stack) * orig_zooms[2]
        else:
            gap = float(slice_thickness)
        stack_zooms = np.array([res_x, res_y, gap])

        # Decompose original affine to get base rotation
        A = affine[:3, :3]
        #S_pix = np.diag(orig_zooms)
        #U, _, Vt = np.linalg.svd(A @ np.linalg.inv(S_pix))
        #R0 = U @ Vt
        #if np.linalg.det(R0) < 0:
        #    U[:, -1] *= -1
        #    R0 = U @ Vt

        # Create new stack's rotation matrix
        #R_stack = R0 @ rotmat
        stack_affine_3x3 = A #R_stack @ np.diag(stack_zooms)

        # Calculate output shape to contain entire rotated volume
        corners_vox = np.array(np.meshgrid([0, vol.shape[0]], [0, vol.shape[1]], [0, vol.shape[2]])).T.reshape(-1, 3)
        corners_world = apply_affine(affine, corners_vox)

        temp_affine = np.eye(4)
        temp_affine[:3, :3] = stack_affine_3x3
        inv_temp_affine = np.linalg.inv(temp_affine)

        corners_new_vox = apply_affine(inv_temp_affine, corners_world)
        min_coords = corners_new_vox.min(axis=0)
        max_coords = corners_new_vox.max(axis=0)

        new_shape_3d = np.ceil(max_coords - min_coords).astype(int)
        stack_shape = (new_shape_3d[0], new_shape_3d[1], slices_per_stack)

        # Center the new stack's field of view
        stack_affine = temp_affine.copy()
        orig_center_world = apply_affine(affine, (np.array(vol.shape) - 1) / 2)
        stack_center_vox = (np.array(stack_shape) - 1) / 2
        T = orig_center_world - apply_affine(stack_affine, stack_center_vox)
        stack_affine[:3, 3] = T

        # Simulate multi-slice acquisition
        nx, ny, nz = stack_shape
        stack = np.zeros(stack_shape, dtype=np.float32)
        order = build_acq_order(nz, acq_order)
        
        # SMS grouping
        mb = int(max(1, mb_factor))
        if mb > nz:
            mb = nz
        groups = [[s for s in order if (s % mb) == r] for r in range(mb)] if mb > 1 else [order]

        base_3x3 = stack_affine[:3, :3]
        cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0

        # Create noise transform for MONAI
        noise_transform = RandGaussianNoise(prob=1.0, std=scaled_noise_std)

        # Loop over SMS groups
        for group in groups:
            # Group-wise motion jitter
            rx, ry, rz = np.deg2rad(np.random.uniform(-max_rot_deg, max_rot_deg, 3))
            tx, ty, tz = np.random.uniform(-max_trans_mm, max_trans_mm, 3)
            Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
            Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
            Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
            R_delta = Rz @ Ry @ Rx

            for s_acq in group:
                # Calculate slice transformation
                world_center_nom = apply_affine(stack_affine, np.array([cx, cy, s_acq]))
                slice_R = (R_delta @ (base_3x3 @ np.linalg.inv(np.diag(stack_zooms))))
                slice_3x3 = slice_R @ np.diag(stack_zooms)
                t_delta = np.array([tx, ty, tz])
                t_slice = world_center_nom + t_delta - (slice_3x3 @ np.array([cx, cy, 0.0]))

                slice_affine = np.eye(4, dtype=np.float64)
                slice_affine[:3, :3] = slice_3x3
                slice_affine[:3, 3] = t_slice

                # Resample slice using MONAI
                # Create a temporary image for this slice
                slice_target_shape = (nx, ny, 1)
                
                # Use MONAI's resampling capabilities
                # For now, we'll use the original nibabel approach for precise affine handling
                # but apply MONAI transforms for noise and shifting
                import nibabel as nib
                from nibabel.processing import resample_from_to
                
                # Create temporary image for resampling using the deformed volume
                temp_img = nib.Nifti1Image(deformed_vol, affine)
                slice_img = resample_from_to(temp_img, (slice_target_shape, slice_affine), 
                                           order=1, cval=float(deformed_vol.min()))
                slice_data = slice_img.get_fdata(dtype=np.float32)[:, :, 0]

                # Apply in-plane displacement and noise using MONAI
                dx, dy = np.random.uniform(-max_disp, max_disp, 2)
                if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                    slice_data = shift_image(slice_data, (dx, dy))
                
                # Apply noise using MONAI
                if scaled_noise_std > 0:
                    slice_tensor = torch.from_numpy(slice_data).unsqueeze(0).unsqueeze(0)
                    noisy_slice = noise_transform(slice_tensor)
                    slice_data = noisy_slice.squeeze(0).squeeze(0).numpy()

                stack[:, :, s_acq] = slice_data

        if np.mean(np.abs(stack)) < 1e-6:
            print(f"WARNING: Stack {i+1} is mostly empty. Check affine and FOV.")

        # Save using MONAI
        out_path = os.path.join(out_dir, f"sim_stack_{i+1:02d}.nii.gz")
        
        # Create NIfTI image with correct affine
        out_img = nib.Nifti1Image(stack, stack_affine)
        out_img.set_sform(stack_affine, code=1)
        out_img.set_qform(stack_affine, code=1)
        nib.save(out_img, out_path)
        
        # Save deformation field if available
        if deformation_field is not None:
            # Create a deformation field image with the same affine as the original volume
            def_path = os.path.join(out_dir, f"deformation_field_{i+1:02d}.nii.gz")
            def_img = nib.Nifti1Image(deformation_field, affine)
            def_img.set_sform(affine, code=1)
            def_img.set_qform(affine, code=1)
            nib.save(def_img, def_path)
            print(f"Saved deformation field {i+1} to {def_path}")
        
        # Save SMS metadata including nonlinear parameters
        json_path = out_path.replace('.nii.gz', '.json')
        metadata = {
            'mb_factor': mb_factor,
            'acquisition_order': acq_order,
            'max_rot_deg': max_rot_deg,
            'max_trans_mm': max_trans_mm,
            'noise_std': noise_std,
            'max_disp': max_disp,
            'nonlinear_enabled': enable_nonlinear,
            'nonlinear_prob': nonlinear_prob,
            'elastic_sigma_range': list(elastic_sigma_range),
            'elastic_magnitude_range': list(elastic_magnitude_range),
            'smooth_deform_prob': smooth_deform_prob,
            'smooth_deform_range': smooth_deform_range,
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved stack {i+1} to {out_path} with shape {stack_shape}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate simulated stacks from a 3D MRI volume using MONAI with nonlinear transformations.")
    parser.add_argument("mri_path", help="Path to input NIfTI MRI volume")
    parser.add_argument("out_dir", help="Output directory for simulated stacks")
    parser.add_argument("--n-stacks", type=int, default=5, help="Number of stacks to generate")
    parser.add_argument("--slices-per-stack", type=int, default=20, help="Number of slices per stack")
    parser.add_argument("--noise-std", type=float, default=0.01, help="Std dev of Gaussian noise, as a fraction of 99th percentile intensity")
    parser.add_argument("--max-disp", type=float, default=5.0, help="Maximum random in-plane displacement per slice (pixels)")
    parser.add_argument("--acq-order", type=str, default="interleaved-odd-even", choices=[
        "sequential-asc", "sequential-desc", "interleaved-odd-even", "interleaved-even-odd"
    ], help="Slice acquisition order")
    parser.add_argument("--max-rot-deg", type=float, default=3.0, help="Max per-slice rotation jitter in degrees")
    parser.add_argument("--max-trans-mm", type=float, default=1.0, help="Max per-slice translation jitter in mm")
    parser.add_argument("--mb-factor", type=int, default=1, help="Simultaneous Multi-Slice factor (1 = no SMS)")
    parser.add_argument("--inplane-res", type=float, default=None, help="In-plane resolution for simulated stacks (mm)")
    parser.add_argument("--slice-thickness", type=float, default=None, help="Slice thickness (gap) for simulated stacks (mm)")
    
    # Nonlinear transformation arguments
    parser.add_argument("--enable-nonlinear", action="store_true", default=True, help="Enable nonlinear transformations")
    parser.add_argument("--disable-nonlinear", action="store_true", help="Disable nonlinear transformations")
    parser.add_argument("--nonlinear-prob", type=float, default=0.7, help="Probability of applying nonlinear transformations per stack")
    parser.add_argument("--elastic-sigma-range", type=float, nargs=2, default=[3.0, 5.0], help="Sigma range for elastic deformation (controls smoothness)")
    parser.add_argument("--elastic-magnitude-range", type=float, nargs=2, default=[0.0, 2.0], help="Magnitude range for elastic deformation displacement")
    parser.add_argument("--smooth-deform-prob", type=float, default=0.3, help="Probability of applying smooth deformation")
    parser.add_argument("--smooth-deform-range", type=float, default=1.0, help="Range for smooth deformation displacement")
    
    args = parser.parse_args()
    
    # Handle nonlinear enable/disable
    enable_nonlinear = args.enable_nonlinear and not args.disable_nonlinear

    generate_simulated_stacks(
        args.mri_path, args.out_dir,
        n_stacks=args.n_stacks,
        slices_per_stack=args.slices_per_stack,
        noise_std=args.noise_std,
        max_disp=args.max_disp,
        acq_order=args.acq_order,
        max_rot_deg=args.max_rot_deg,
        max_trans_mm=args.max_trans_mm,
        mb_factor=args.mb_factor,
        inplane_res=args.inplane_res, 
        slice_thickness=args.slice_thickness,
        enable_nonlinear=enable_nonlinear,
        nonlinear_prob=args.nonlinear_prob,
        elastic_sigma_range=tuple(args.elastic_sigma_range),
        elastic_magnitude_range=tuple(args.elastic_magnitude_range),
        smooth_deform_prob=args.smooth_deform_prob,
        smooth_deform_range=args.smooth_deform_range
    )