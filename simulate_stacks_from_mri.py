import numpy as np
import nibabel as nib
from scipy.ndimage import shift
import os
from nibabel.processing import resample_from_to
from nibabel.affines import apply_affine

def generate_simulated_stacks(
    mri_path,
    out_dir,
    n_stacks=5,
    slices_per_stack=20,
    orientations=None,
    noise_std=0.01,
    max_disp=5.0,
    acq_order="interleaved-odd-even",
    max_rot_deg=3.0,
    max_trans_mm=1.0,
    mb_factor: int = 1,
    inplane_res: float = None,
    slice_thickness: float = None,
):
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

    def build_acq_order(nz: int, mode: str):
        if mode == "sequential-asc":
            return list(range(nz))
        if mode == "sequential-desc":
            return list(range(nz - 1, -1, -1))
        if mode == "interleaved-even-odd":
            # 0-based: even first
            return list(range(0, nz, 2)) + list(range(1, nz, 2))
        # default: interleaved-odd-even (1-based odd first => 0,2,4 in 0-based)
        return list(range(0, nz, 2)) + list(range(1, nz, 2))

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

        # Determine voxel sizes for the new stack. Allow optional overrides for in-plane resolution and slice thickness.
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

        # Simulate multi-slice acquisition: per-slice resampling with slice-wise motion
        nx, ny, nz = stack_shape
        stack = np.zeros(stack_shape, dtype=np.float32)
        order = build_acq_order(nz, acq_order)
        # Simultaneous Multi-Slice grouping: slices that are acquired at the same time share the same motion
        mb = int(max(1, mb_factor))
        if mb > nz:
            mb = nz
        # Build groups by modulo classes to space slices across the slab
        groups = [[s for s in order if (s % mb) == r] for r in range(mb)] if mb > 1 else [order]

        # Base 3x3 for slice sampling (without translation)
        base_3x3 = stack_affine[:3, :3]
        cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0

        # Loop over SMS groups; apply shared motion within each group
        for group in groups:
            # Group-wise motion jitter
            rx, ry, rz = np.deg2rad(np.random.uniform(-max_rot_deg, max_rot_deg, 3))
            tx, ty, tz = np.random.uniform(-max_trans_mm, max_trans_mm, 3)
            Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
            Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
            Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
            R_delta = Rz @ Ry @ Rx

            for s_acq in group:
                # Nominal center of slice s in world space using the planned stack affine
                world_center_nom = apply_affine(stack_affine, np.array([cx, cy, s_acq]))

                # Compose with base orientation; keep orthonormal by construction
                slice_R = (R_delta @ (base_3x3 @ np.linalg.inv(np.diag(stack_zooms))))
                # Re-apply voxel scaling
                slice_3x3 = slice_R @ np.diag(stack_zooms)

                # Translation so that the slice center hits world_center_nom with added motion
                t_delta = np.array([tx, ty, tz])
                t_slice = world_center_nom + t_delta - (slice_3x3 @ np.array([cx, cy, 0.0]))

                slice_affine = np.eye(4, dtype=np.float64)
                slice_affine[:3, :3] = slice_3x3
                slice_affine[:3, 3] = t_slice

                # Resample just this slice (depth 1) then insert into stack at index s_acq
                slice_img = resample_from_to(img, ((nx, ny, 1), slice_affine), order=1, cval=float(vol.min()))
                slice_data = slice_img.get_fdata(dtype=np.float32)[:, :, 0]

                # Optional additional in-plane displacement and noise
                dx, dy = np.random.uniform(-max_disp, max_disp, 2)
                slice_data = shift(slice_data, shift=(dx, dy), order=1, mode='nearest')
                if scaled_noise_std > 0:
                    slice_data += np.random.normal(0, scaled_noise_std, slice_data.shape).astype(np.float32)

                stack[:, :, s_acq] = slice_data

        if np.mean(np.abs(stack)) < 1e-6:
            print(f"WARNING: Stack {i+1} is mostly empty. Check affine and FOV.")

        # Save as NIfTI with correct affine
        out_img = nib.Nifti1Image(stack, stack_affine)
        # 5. Set sform and qform directly; avoid unsafe try/except pass
        out_img.set_sform(stack_affine, code=1) # NIFTI_XFORM_SCANNER_ANAT
        out_img.set_qform(stack_affine, code=1) # NIFTI_XFORM_SCANNER_ANAT

        out_path = os.path.join(out_dir, f"sim_stack_{i+1:02d}.nii.gz")
        nib.save(out_img, out_path)
        
        # Save SMS metadata as JSON sidecar
        import json
        json_path = out_path.replace('.nii.gz', '.json')
        metadata = {
            'mb_factor': mb_factor,
            'acquisition_order': acq_order,
            'max_rot_deg': max_rot_deg,
            'max_trans_mm': max_trans_mm,
            'noise_std': noise_std,
            'max_disp': max_disp,
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
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
    parser.add_argument("--acq-order", type=str, default="interleaved-odd-even", choices=[
        "sequential-asc", "sequential-desc", "interleaved-odd-even", "interleaved-even-odd"
    ], help="Slice acquisition order")
    parser.add_argument("--max-rot-deg", type=float, default=3.0, help="Max per-slice rotation jitter in degrees")
    parser.add_argument("--max-trans-mm", type=float, default=1.0, help="Max per-slice translation jitter in mm")
    parser.add_argument("--mb-factor", type=int, default=1, help="Simultaneous Multi-Slice factor (1 = no SMS)")
    parser.add_argument("--inplane-res", type=float, default=None, help="In-plane resolution for simulated stacks (mm)")
    parser.add_argument("--slice-thickness", type=float, default=None, help="Slice thickness (gap) for simulated stacks (mm)")
    args = parser.parse_args()

    generate_simulated_stacks(
        args.mri_path, args.out_dir,
        n_stacks=args.n_stacks,
        slices_per_stack=args.slices_per_stack,
        noise_std=args.noise_std,
        max_disp=args.max_disp,
        acq_order=args.acq_order,
        max_rot_deg=args.max_rot_deg,
        max_trans_mm=args.max_trans_mm,
        mb_factor=args.mb_factor
        , inplane_res=args.inplane_res, slice_thickness=args.slice_thickness
    )