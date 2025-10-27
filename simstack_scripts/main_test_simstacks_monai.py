# Test script to generate simulated stacks using MONAI rigid transforms
from simulate_stacks_from_mri_monai import generate_simulated_stacks

if __name__ == "__main__":
    generate_simulated_stacks(
        mri_path="SVR001_brain9_20251013_125502/out/tmp/svr_output.nii.gz",
        out_dir="SVR001_brain9_20251013_125502/sim_stacks_monai_rigid_test3/",
        n_stacks=6,
        slices_per_stack=20,
        noise_std=0.01,
        max_disp=5.0,
        acq_order="interleaved-odd-even",
        max_rot_deg=3.0,
        max_trans_mm=1.0,
        mb_factor=1,
        inplane_res=None,
        slice_thickness=None,
        orientations=[(0, 'x'), (90, 'y'), (180, 'x'), (270, 'y'), (45, 'z'), (135, 'z')]
    )