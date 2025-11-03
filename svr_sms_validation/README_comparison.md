# SMS vs Sequential SVR Comparison

This directory contains tools for validating SMS (Simultaneous Multi-Slice) aware SVR reconstruction.

## Scripts

### 1. `validate_svr_sms.py`
Validates that SMS group transforms are properly averaged during SVR reconstruction.

**Usage:**
```bash
python validate_svr_sms.py <svr_output_dir> <stack1.nii.gz> [stack2.nii.gz ...]
```

**What it checks:**
- Loads final SVR transforms from `svr_output_dir/svr/transforms_svr_final.npy`
- Groups slices by SMS acquisition (based on mb_factor in JSON metadata)
- Verifies that all transforms within an SMS group are equal (within tolerance)

**Expected result:**
All SMS groups should PASS, meaning slices acquired simultaneously share the same rigid transformation.

---

### 2. `compare_sms_vs_sequential.py`
Compares SVR reconstruction quality between SMS and sequential acquisitions from the same ground truth.

**Purpose:**
- Validates that SMS averaging doesn't degrade reconstruction quality
- Compares metrics (PSNR, SSIM, NRMSE) against ground truth
- Ensures SMS and sequential reconstructions are equivalent

**Usage:**
```bash
# Full pipeline: simulate stacks, run SVR, compare
python compare_sms_vs_sequential.py \
    --ground-truth <high_res_volume.nii.gz> \
    --output-dir <output_directory> \
    --num-stacks 3 \
    --mb-factor 2 \
    --motion-level moderate \
    --n-iter 2

# Use existing stacks
python compare_sms_vs_sequential.py \
    --ground-truth <ground_truth.nii.gz> \
    --sms-stacks sms_dir/sim_stack_*.nii.gz \
    --sequential-stacks seq_dir/sim_stack_*.nii.gz \
    --output-dir comparison \
    --skip-simulation

# Use existing reconstructions
python compare_sms_vs_sequential.py \
    --ground-truth <ground_truth.nii.gz> \
    --output-dir existing_comparison \
    --skip-svr
```

**Parameters:**
- `--ground-truth`: High-resolution reference volume (e.g., previous SVR output)
- `--output-dir`: Directory for all outputs
- `--num-stacks`: Number of stacks to simulate (default: 3)
- `--mb-factor`: Multiband factor for SMS stacks (default: 2)
- `--motion-level`: Motion severity - none/mild/moderate/severe (default: moderate)
- `--output-resolution`: SVR output resolution in mm (default: 2.0)
- `--n-iter`: Number of SVR iterations (default: 2)
- `--skip-simulation`: Skip simulation, use existing stacks
- `--skip-svr`: Skip SVR, use existing reconstructions

**Output structure:**
```
output_dir/
├── sms_stacks/
│   ├── sim_stack_01.nii.gz
│   ├── sim_stack_01.json
│   └── ...
├── sequential_stacks/
│   ├── sim_stack_01.nii.gz
│   ├── sim_stack_01.json
│   └── ...
├── sms_svr_temp/
│   └── svr/
│       └── transforms_svr_final.npy
├── sequential_svr_temp/
│   └── svr/
│       └── transforms_svr_final.npy
├── svr_sms_reconstruction.nii.gz
├── svr_sequential_reconstruction.nii.gz
└── comparison_results.json
```

**Metrics computed:**
- **PSNR** (Peak Signal-to-Noise Ratio): Higher is better, measures reconstruction accuracy
- **SSIM** (Structural Similarity Index): Closer to 1.0 is better, measures perceptual quality
- **NRMSE** (Normalized Root Mean Squared Error): Lower is better, measures pixel-wise error

**Expected results:**
- SMS and sequential reconstructions should have similar PSNR/SSIM (difference < 0.5 dB)
- This validates that SMS averaging preserves reconstruction accuracy
- If SMS is better, it suggests SMS constraints help regularize the solution
- If sequential is significantly better, it may indicate an issue with SMS averaging

---

## Workflow Example

### Complete validation workflow:

```bash
# 1. Get a high-quality ground truth volume
GROUND_TRUTH="test_data/39/SVR001_brain9_20251013_130312/out/tmp/svr_output.nii.gz"

# 2. Run comparison
python svr_sms_validation/compare_sms_vs_sequential.py \
    --ground-truth $GROUND_TRUTH \
    --output-dir test_data/39/sms_vs_seq_comparison \
    --num-stacks 3 \
    --mb-factor 2 \
    --motion-level moderate \
    --n-iter 2

# 3. Check results
cat test_data/39/sms_vs_seq_comparison/comparison_results.json
```

### Quick validation on existing SMS reconstruction:

```bash
# Validate that SMS transforms were averaged correctly
python svr_sms_validation/validate_svr_sms.py \
    test_data/39/sim_stacks_sms/svr_temp \
    test_data/39/sim_stacks_sms/sim_stack_*.nii.gz
```

---

## Implementation Details

### SMS Metadata Flow
1. **JSON sidecar files** contain `mb_factor` and `acquisition_order`
2. **Stack objects** have these as attributes after loading
3. **Slice extraction** attaches metadata as attributes (`_source_mb_factor`, etc.)
4. **Concatenation** collects metadata into `sms_metadata` list
5. **Registration** detects `sms_metadata` and applies group averaging
6. **Validation** checks that group transforms are equal

### SMS Transform Averaging
In `standalone_inlined/svr/registration.py`:
- Slices are grouped by stack and SMS acquisition pattern
- For mb_factor=2 with interleaved-odd-even: 
  - Group 1: slices [0, 2, 4, 6, ...]
  - Group 2: slices [1, 3, 5, 7, ...]
- Transforms within each group are averaged: `theta_avg = theta_group.mean(dim=0)`
- All slices in group receive the same averaged transform

### Key Files Modified for SMS Support
- `svr_cli.py`: Load SMS metadata from JSON
- `standalone_inlined/svort/inference.py`: Attach SMS metadata to slices
- `standalone_inlined/svr/pipeline.py`: Preserve metadata through pipeline
- `standalone_inlined/svr/registration.py`: Implement SMS group averaging
- `standalone_inlined/image.py`: Preserve metadata through Stack.cat()

---

## Troubleshooting

### Validation fails: SMS transforms differ
- Check debug logs for "SMS registration" messages
- Verify JSON files have correct `mb_factor`
- Ensure slices have `_source_mb_factor` attribute after extraction
- Check that `sms_metadata` is attached to concatenated stack

### Comparison shows large differences
- Verify same motion parameters used for both SMS and sequential
- Check that ground truth is high quality (not corrupted)
- Ensure sufficient SVR iterations (try --n-iter 3 or more)
- Motion level may be too severe for reliable reconstruction

### Missing transforms file
- Ensure SVR completed successfully
- Check `SVR_TEMP_DIR` environment variable
- Look for transforms in: `output_dir/svr_temp/svr/transforms_svr_final.npy`
