import numpy as np
import nibabel as nib
from scipy.ndimage import shift
import os
from nibabel.processing import resample_from_to
from nibabel.affines import apply_affine

def generate_simulated_stacks(mri_path, out_dir, n_stacks=5, slices_per_stack=20, orientations=None, noise_std=0.01, max_disp=5.0):
    """
    Generate simulated stacks from a 3D MRI volume, with random in-plane displacement per slice,
    ensuring the output stack correctly aligns with the input volume.

    Args:
        mri_path (str): Path to the input NIfTI MRI.
        out_dir (str): Output directory for the generated stacks.
        n_stacks (int): Number of stacks to generate.
        slices_per_stack (int): Number of slices per stack.
        orientations (list, optional): List of (angle, axis) tuples for stack orientations. If None, random orientations are used.
        noise_std (float): Standard deviation of Gaussian noise to add, as a fraction of the 99th percentile of image intensity.
        max_disp (float): Maximum random in-plane displacement in pixels per slice.
    """
    os.makedirs(out_dir, exist_ok=True)
    img = nib.load(mri_path)
    # 1. Load data as float32 to prevent issues with noise addition
    vol = img.get_fdata(dtype=np.float32)
    affine = img.affine.copy()
    orig_zooms = img.header.get_zooms()[:3]

    # 2. Calculate a scaled noise level based on image intensity
    # This makes noise_std an intuitive percentage, robust to intensity ranges.
    if vol.max() > 0:
        data_max = np.percentile(vol[vol > 0], 99)
        scaled_noise_std = noise_std * data_max
    else:
        scaled_noise_std = 0

    for i in range(n_stacks):
        if orientations is not None:
            angle, axis = orientations[i % len(orientations)]
        else:
            axis = np.random.choice(['x', 'y'])  # Use x or y for oblique stacks
            angle = np.random.uniform(15, 345)   # Avoid angles too close to axis-aligned

        theta = np.deg2rad(angle)
        if axis == 'x':
            rotmat = np.array([
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)]])
        elif axis == 'y':
            rotmat = np.array([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]])
        else: # 'z'
             rotmat = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])

        # Determine voxel sizes for the new stack
        gap = (vol.shape[2] / slices_per_stack) * orig_zooms[2]
        stack_zooms = np.array([orig_zooms[0], orig_zooms[1], gap])

        # Decompose original affine to get its base rotation
        A = affine[:3, :3]
        S_pix = np.diag(orig_zooms)
        U, _, Vt = np.linalg.svd(A @ np.linalg.inv(S_pix))
        R0 = U @ Vt
        if np.linalg.det(R0) < 0:
            U[:, -1] *= -1
            R0 = U @ Vt

        # Create the new stack's rotation matrix
        R_stack = R0 @ rotmat
        stack_affine_3x3 = R_stack @ np.diag(stack_zooms)

        # 3. (MAIN FIX) Calculate the output shape needed to contain the entire rotated volume
        # This prevents the rotated image from being cropped.
        corners_vox = np.array(np.meshgrid([0, vol.shape[0]], [0, vol.shape[1]], [0, vol.shape[2]])).T.reshape(-1, 3)
        corners_world = apply_affine(affine, corners_vox)

        # Create a temporary affine with just the new rotation and scaling
        temp_affine = np.eye(4)
        temp_affine[:3,:3] = stack_affine_3x3
        inv_temp_affine = np.linalg.inv(temp_affine)

        # Map the world corners into the new voxel grid to find its extent
        corners_new_vox = apply_affine(inv_temp_affine, corners_world)
        min_coords = corners_new_vox.min(axis=0)
        max_coords = corners_new_vox.max(axis=0)

        # The new shape is the size of this bounding box
        new_shape_3d = np.ceil(max_coords - min_coords).astype(int)
        stack_shape = (new_shape_3d[0], new_shape_3d[1], slices_per_stack)

        # 4. (MAIN FIX) Center the new stack's field of view on the original volume's center
        stack_affine = temp_affine.copy()
        orig_center_world = apply_affine(affine, (np.array(vol.shape) - 1) / 2)
        stack_center_vox = (np.array(stack_shape) - 1) / 2
        # Calculate the translation required to align the centers
        T = orig_center_world - apply_affine(stack_affine, stack_center_vox)
        stack_affine[:3, 3] = T

        # Resample the original volume onto the new grid
        stack_img = resample_from_to(img, (stack_shape, stack_affine), order=1, cval=vol.min())
        stack = stack_img.get_fdata(dtype=np.float32)

        if np.mean(np.abs(stack)) < 1e-6:
            print(f"WARNING: Stack {i+1} is mostly empty. Check affine and FOV.")

        # Add random in-plane displacement and noise per slice
        for z in range(stack.shape[2]):
            dx, dy = np.random.uniform(-max_disp, max_disp, 2)
            stack[:, :, z] = shift(stack[:, :, z], shift=(dx, dy), order=1, mode='nearest')
            if scaled_noise_std > 0:
                stack[:, :, z] += np.random.normal(0, scaled_noise_std, stack[:, :, z].shape)

        # Save as NIfTI with correct affine
        out_img = nib.Nifti1Image(stack, stack_affine)
        # 5. Set sform and qform directly; avoid unsafe try/except pass
        out_img.set_sform(stack_affine, code=1) # NIFTI_XFORM_SCANNER_ANAT
        out_img.set_qform(stack_affine, code=1) # NIFTI_XFORM_SCANNER_ANAT

        out_path = os.path.join(out_dir, f"sim_stack_{i+1:02d}.nii.gz")
        nib.save(out_img, out_path)
        print(f"Saved stack {i+1} to {out_path} with shape {stack_shape}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate simulated stacks from a 3D MRI volume.")
    parser.add_argument("mri_path", help="Path to input NIfTI MRI volume")
    parser.add_argument("out_dir", help="Output directory for simulated stacks")
    parser.add_argument("--n-stacks", type=int, default=5, help="Number of stacks to generate")
    parser.add_argument("--slices-per-stack", type=int, default=20, help="Number of slices per stack")
    parser.add_argument("--noise-std", type=float, default=0.01, help="Std dev of Gaussian noise, as a fraction of 99th percentile intensity")
    parser.add_argument("--max-disp", type=float, default=5.0, help="Maximum random in-plane displacement per slice (pixels)")
    args = parser.parse_args()

    generate_simulated_stacks(
        args.mri_path, args.out_dir,
        n_stacks=args.n_stacks,
        slices_per_stack=args.slices_per_stack,
        noise_std=args.noise_std,
        max_disp=args.max_disp
    )