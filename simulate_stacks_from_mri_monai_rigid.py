

    # 2. Calculate noise level
    if vol_tensor.max() > 0:
        data_max = np.percentile(vol_tensor[vol_tensor > 0].cpu().numpy(), 99)
        scaled_noise_std = noise_std * data_max
    else:
        scaled_noise_std = 0

    # 3. Define MONAI transforms
    # 3a. Rigid motion transform (applied per group)
    rand_affine = RandAffine(
        prob=1.0,
        rotate_range=[np.deg2rad(max_rot_deg)] * 3,
        translate_range=[max_trans_mm] * 3,  # In mm
        scale_range=None,  # No scaling for rigid
        mode="bilinear",
        padding_mode="zeros",
        device=device,
    )
    # 3b. Affine resampler (applied per slice)
    from monai.transforms import Affine
    resampler = SpatialResample(
        mode="bilinear",
        padding_mode="zeros")
    #Affine(
    #    mode="bilinear",
    #    padding_mode="zeros",
    #    device=device,
    #)
    # 3c. Noise transform (applied per slice)
    noise_adder = RandGaussianNoise(
        prob=1.0,
        std=scaled_noise_std,
        mean=0.0,
    )

    def build_acq_order(nz: int, mode: str):
        if mode == "sequential-asc":
            return list(range(nz))
        if mode == "sequential-desc":
            return list(range(nz - 1, -1, -1))
        if mode == "interleaved-even-odd":
            return list(range(0, nz, 2)) + list(range(1, nz, 2))
        # default: interleaved-odd-even
        return list(range(0, nz, 2)) + list(range(1, nz, 2))

    for i in range(n_stacks):
        # 4. Determine target stack geometry
        if inplane_res is None:
            res_x = orig_zooms[0]
            res_y = orig_zooms[1]
        else:
            res_x = float(inplane_res)
            res_y = float(inplane_res)

        if slice_thickness is None:
            # Default: gap fills the original volume's Z-extent
            gap = (vol_shape[2] / slices_per_stack) * orig_zooms[2]
        else:
            gap = float(slice_thickness)
        
        stack_zooms = np.array([res_x, res_y, gap])

        # Calculate new stack shape based on resolution change
        nx = int(np.round(vol_shape[0] * orig_zooms[0] / res_x))
        ny = int(np.round(vol_shape[1] * orig_zooms[1] / res_y))
        nz = slices_per_stack
        stack_shape = (nx, ny, nz)

        # Calculate the new stack's affine
        # Start from original affine, scale by zoom ratios
        stack_affine = vol_affine.clone().numpy()
        zoom_ratios = np.array([
            res_x / orig_zooms[0],
            res_y / orig_zooms[1],
            gap / orig_zooms[2],
            1.0
        ])
        stack_affine = stack_affine @ np.diag(zoom_ratios)

        # 5. Simulate stack acquisition
        stack = np.zeros(stack_shape, dtype=np.float32)
        order = build_acq_order(nz, acq_order)
        mb = int(max(1, mb_factor))
        if mb > nz: mb = nz
        groups = [[s for s in order if (s % mb) == r] for r in range(mb)] if mb > 1 else [order]

        for group in groups:
            # 5a. Apply one random 3D rigid transform for this group
            # This simulates subject motion between acquisitions
            # vol_tensor is now guaranteed 5D [B, C, H, W, D], so RandAffine is 3D
            transformed_vol_tensor = rand_affine(vol_tensor)

            for s_acq in group:
                # 5b. Define the target grid for this *single slice*
                
                # Create the affine for this specific slice
                # Start with the stack's base affine
                slice_affine_np = stack_affine.copy()
                
                # Calculate the world coordinate of this slice's origin [0,0,0]
                # by applying the stack affine to the slice's voxel coord [0,0,s_acq]
                slice_origin_world = apply_affine(stack_affine, [0, 0, s_acq])
                
                # Set the translation part of the slice affine
                slice_affine_np[:3, 3] = slice_origin_world

                # make transformed_vol_tensor into monai image with affine
                transformed_vol_tensor = img_data # Nifti1Image(transformed_vol_tensor, affine=stack_affine)

                # 5c. Resample from the *transformed 3D volume* to the *target 2D slice grid* using Affine


                slice_tensor = resampler(
                    img=transformed_vol_tensor,  # [B, C, H, W, D]
                    #dst_affine=torch.tensor(vol_affine, dtype=torch.float64, device=device).unsqueeze(0),
                    spatial_size=(nx, ny, 1),
                    dst_affine=torch.tensor(slice_affine_np, dtype=torch.float64, device=device).unsqueeze(0)
                ).squeeze(0).squeeze(0)

                # 5d. Add noise
                if scaled_noise_std > 0:
                    slice_tensor = noise_adder(slice_tensor)

                stack[:, :, s_acq] = slice_tensor.squeeze().cpu().numpy()

        # 6. Save the output stack
        out_img = nib.Nifti1Image(stack, stack_affine)
        out_img.set_sform(stack_affine, code=1)
        out_img.set_qform(stack_affine, code=1)
        out_path = os.path.join(out_dir, f"sim_stack_monai_rigid_{i+1:02d}.nii.gz")
        nib.save(out_img, out_path)

        json_path = out_path.replace('.nii.gz', '.json')
        metadata = {
            'mb_factor': mb_factor,
            'acquisition_order': acq_order,
            'max_rot_deg': max_rot_deg,
            'max_trans_mm': max_trans_mm,
            'noise_std': noise_std,
            'max_disp': max_disp, # Included for consistency, though unused here
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Saved MONAI rigid stack {i+1} to {out_path} with shape {stack_shape}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate simulated stacks from a 3D MRI volume using MONAI random rigid transforms.")
    parser.add_argument("mri_path", help="Path to input NIfTI MRI volume")
    parser.add_argument("out_dir", help="Output directory for simulated stacks")
    parser.add_argument("--n-stacks", type=int, default=5, help="Number of stacks to generate")
    parser.add_argument("--slices-per-stack", type=int, default=20, help="Number of slices per stack")
    parser.add_argument("--noise-std", type=float, default=0.01, help="Std dev of Gaussian noise, as a fraction of 99th percentile intensity")
    parser.add_argument("--max-disp", type=float, default=5.0, help="Maximum random in-plane displacement per slice (pixels) [UNUSED IN THIS SCRIPT]")
    parser.add_argument("--acq-order", type=str, default="interleaved-odd-even", choices=[
        "sequential-asc", "sequential-desc", "interleaved-odd-even", "interleaved-even-odd"
    ], help="Slice acquisition order")
    parser.add_argument("--max-rot-deg", type=float, default=3.0, help="Max per-slice rotation jitter in degrees")
    parser.add_argument("--max-trans-mm", type=float, default=1.0, help="Max per-slice translation jitter in mm")
    parser.add_argument("--mb-factor", type=int, default=1, help="Simultaneous Multi-Slice factor (1 = no SMS)")
    parser.add_argument("--inplane-res", type=float, default=None, help="In-plane resolution for simulated stacks (mm)")
    parser.add_argument("--slice-thickness", type=float, default=None, help="Slice thickness (gap) for simulated stacks (mm)")
    args = parser.parse_args()

    generate_simulated_stacks_monai_rigid(
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
        slice_thickness=args.slice_thickness
    )