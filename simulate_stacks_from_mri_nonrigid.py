import numpy as np
import nibabel as nib
from scipy.ndimage import shift, map_coordinates
import os
from nibabel.processing import resample_from_to
from nibabel.affines import apply_affine

def random_deformation_field(shape, max_disp):
    """
    Generate a smooth random deformation field for nonrigid transformation.
    Returns a displacement field of shape (3, *shape).
    """
    # Low-res random field, upsampled and smoothed
    grid_shape = tuple(max(4, s // 8) for s in shape)
    field = np.random.uniform(-max_disp, max_disp, (3,) + grid_shape).astype(np.float32)
    # Upsample to full size
    coords = [np.linspace(0, gs-1, s) for gs, s in zip(grid_shape, shape)]
    mesh = np.meshgrid(*coords, indexing='ij')
    up_field = np.stack([
        map_coordinates(field[c], mesh, order=3, mode='reflect')
        for c in range(3)
    ], axis=0)
    # Smooth (optional): could add gaussian_filter here
    return up_field

def generate_simulated_stacks_nonrigid(
    mri_path,
    out_dir,
    n_stacks=5,
    slices_per_stack=20,
    orientations=None,
    noise_std=0.01,
    max_disp=5.0,
    acq_order="interleaved-odd-even",
    max_deform=3.0,
    mb_factor: int = 1,
    inplane_res: float = None,
    slice_thickness: float = None,
):
    """
    Generate simulated stacks from a 3D MRI volume, with random nonrigid deformation per slice.
    """
    os.makedirs(out_dir, exist_ok=True)
    img = nib.load(mri_path)
    vol = img.get_fdata(dtype=np.float32)
    affine = img.affine.copy()
    orig_zooms = img.header.get_zooms()[:3]

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
            return list(range(0, nz, 2)) + list(range(1, nz, 2))
        return list(range(0, nz, 2)) + list(range(1, nz, 2))

    for i in range(n_stacks):
        if orientations is not None:
            angle, axis = orientations[i % len(orientations)]
        else:
            axis = np.random.choice(['x', 'y'])
            angle = np.random.uniform(15, 345)

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
        else:
            rotmat = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])

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

        A = affine[:3, :3]
        S_pix = np.diag(orig_zooms)
        U, _, Vt = np.linalg.svd(A @ np.linalg.inv(S_pix))
        R0 = U @ Vt
        if np.linalg.det(R0) < 0:
            U[:, -1] *= -1
            R0 = U @ Vt

        R_stack = R0 @ rotmat
        stack_affine_3x3 = R_stack @ np.diag(stack_zooms)

        corners_vox = np.array(np.meshgrid([0, vol.shape[0]], [0, vol.shape[1]], [0, vol.shape[2]])).T.reshape(-1, 3)
        corners_world = apply_affine(affine, corners_vox)
        temp_affine = np.eye(4)
        temp_affine[:3,:3] = stack_affine_3x3
        inv_temp_affine = np.linalg.inv(temp_affine)
        corners_new_vox = apply_affine(inv_temp_affine, corners_world)
        min_coords = corners_new_vox.min(axis=0)
        max_coords = corners_new_vox.max(axis=0)
        new_shape_3d = np.ceil(max_coords - min_coords).astype(int)
        stack_shape = (new_shape_3d[0], new_shape_3d[1], slices_per_stack)
        stack_affine = temp_affine.copy()
        orig_center_world = apply_affine(affine, (np.array(vol.shape) - 1) / 2)
        stack_center_vox = (np.array(stack_shape) - 1) / 2
        T = orig_center_world - apply_affine(stack_affine, stack_center_vox)
        stack_affine[:3, 3] = T

        nx, ny, nz = stack_shape
        stack = np.zeros(stack_shape, dtype=np.float32)
        order = build_acq_order(nz, acq_order)
        mb = int(max(1, mb_factor))
        if mb > nz:
            mb = nz
        groups = [[s for s in order if (s % mb) == r] for r in range(mb)] if mb > 1 else [order]
        base_3x3 = stack_affine[:3, :3]
        cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0

        for group in groups:
            # Nonrigid deformation field for this group
            deform_field = random_deformation_field((nx, ny), max_deform)
            for s_acq in group:
                world_center_nom = apply_affine(stack_affine, np.array([cx, cy, s_acq]))
                slice_3x3 = base_3x3
                t_slice = world_center_nom - (slice_3x3 @ np.array([cx, cy, 0.0]))
                slice_affine = np.eye(4, dtype=np.float64)
                slice_affine[:3, :3] = slice_3x3
                slice_affine[:3, 3] = t_slice
                slice_img = resample_from_to(img, ((nx, ny, 1), slice_affine), order=1, cval=float(vol.min()))
                slice_data = slice_img.get_fdata(dtype=np.float32)[:, :, 0]
                # Apply nonrigid deformation
                coords = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
                coords_deformed = [
                    coords[0] + deform_field[0],
                    coords[1] + deform_field[1]
                ]
                slice_data = map_coordinates(slice_data, coords_deformed, order=3, mode='reflect')
                # Optional additional in-plane displacement and noise
                dx, dy = np.random.uniform(-max_disp, max_disp, 2)
                slice_data = shift(slice_data, shift=(dx, dy), order=1, mode='nearest')
                if scaled_noise_std > 0:
                    slice_data += np.random.normal(0, scaled_noise_std, slice_data.shape).astype(np.float32)
                stack[:, :, s_acq] = slice_data

        if np.mean(np.abs(stack)) < 1e-6:
            print(f"WARNING: Stack {i+1} is mostly empty. Check affine and FOV.")
        out_img = nib.Nifti1Image(stack, stack_affine)
        out_img.set_sform(stack_affine, code=1)
        out_img.set_qform(stack_affine, code=1)
        out_path = os.path.join(out_dir, f"sim_stack_nonrigid_{i+1:02d}.nii.gz")
        nib.save(out_img, out_path)
        import json
        json_path = out_path.replace('.nii.gz', '.json')
        metadata = {
            'mb_factor': mb_factor,
            'acquisition_order': acq_order,
            'max_deform': max_deform,
            'noise_std': noise_std,
            'max_disp': max_disp,
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved nonrigid stack {i+1} to {out_path} with shape {stack_shape}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate simulated stacks from a 3D MRI volume with nonrigid deformation.")
    parser.add_argument("mri_path", help="Path to input NIfTI MRI volume")
    parser.add_argument("out_dir", help="Output directory for simulated stacks")
    parser.add_argument("--n-stacks", type=int, default=5, help="Number of stacks to generate")
    parser.add_argument("--slices-per-stack", type=int, default=20, help="Number of slices per stack")
    parser.add_argument("--noise-std", type=float, default=0.01, help="Std dev of Gaussian noise, as a fraction of 99th percentile intensity")
    parser.add_argument("--max-disp", type=float, default=5.0, help="Maximum random in-plane displacement per slice (pixels)")
    parser.add_argument("--acq-order", type=str, default="interleaved-odd-even", choices=[
        "sequential-asc", "sequential-desc", "interleaved-odd-even", "interleaved-even-odd"
    ], help="Slice acquisition order")
    parser.add_argument("--max-deform", type=float, default=3.0, help="Max nonrigid deformation per slice (pixels)")
    parser.add_argument("--mb-factor", type=int, default=1, help="Simultaneous Multi-Slice factor (1 = no SMS)")
    parser.add_argument("--inplane-res", type=float, default=None, help="In-plane resolution for simulated stacks (mm)")
    parser.add_argument("--slice-thickness", type=float, default=None, help="Slice thickness (gap) for simulated stacks (mm)")
    args = parser.parse_args()

    generate_simulated_stacks_nonrigid(
        args.mri_path, args.out_dir,
        n_stacks=args.n_stacks,
        slices_per_stack=args.slices_per_stack,
        noise_std=args.noise_std,
        max_disp=args.max_disp,
        acq_order=args.acq_order,
        max_deform=args.max_deform,
        mb_factor=args.mb_factor,
        inplane_res=args.inplane_res,
        slice_thickness=args.slice_thickness
    )
