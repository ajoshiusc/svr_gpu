# Test script to generate simulated stacks using MONAI rigid transforms
from simulate_stacks_from_mri_monai import generate_simulated_stacks

if __name__ == "__main__":
	generate_simulated_stacks(
		mri_path="/home/ajoshi/Downloads/test_abdomen/39/t1outphase.nii.gz",
		out_dir="/home/ajoshi/Projects/svr_gpu/sim_stacks_rigid/",
		n_stacks=6,
		slices_per_stack=60,
		noise_std=0.01,
		max_disp=5.0,
		acq_order="interleaved-odd-even",
		max_rot_deg=3.0,
		max_trans_mm=1.0,
		mb_factor=1,
		inplane_res=2,
		slice_thickness=9,
		orientations=[(0, 'x'), (90, 'y'), (180, 'x'), (270, 'y'), (45, 'z'), (135, 'z')]
	)

