# Test script to generate simulated stacks using MONAI rigid transforms
from simulate_stacks import generate_simulated_stacks

if __name__ == "__main__":
	generate_simulated_stacks(
		mri_path="/deneb_disk/for_atu/mre_scans/Cor_2D_FIESTA_Entrography_BH_20250711195358_3.nii.gz", #"/deneb_disk/for_atu/mre_scans/COR_BTFE_FS_COR_BTFE_FS_20250619210051_401.nii.gz",
		out_dir="/deneb_disk/for_atu/mre_scans/Cor_2D_FIESTA_Entrography_BH_20250711195358_3_sim_stacks_nonlin",
		n_stacks=6,
		slices_per_stack=60,
		noise_std=0.01,
		max_disp=5.0,
		acq_order="interleaved-odd-even",
		max_rot_deg=3.0,
		max_trans_mm=1.0,
		mb_factor=1,
		inplane_res=2,
		slice_thickness=6,
		orientations=[(0, 'x'), (90, 'y'), (180, 'x'), (270, 'y'), (45, 'z'), (135, 'z')],
		enable_nonlinear=True
	)

