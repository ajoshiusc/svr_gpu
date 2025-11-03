# SMS Validation with Multiple Ground Truths

## Overview

The SMS-aware SVR validation notebook (`sms_detailed_validation.ipynb`) has been updated to support **multiple ground truth volumes**. This allows for more robust statistical validation by testing the SMS reconstruction pipeline across different brain anatomies.

## Key Changes

### 1. Multiple Ground Truth Support

- **Input**: Reads all NIfTI files (`.nii.gz`, `.nii`) from a specified directory
- **Processing**: Runs the full experimental suite (motion levels × MB factors × stack counts) for each ground truth
- **Output**: Organized results by ground truth with comprehensive statistics

### 2. Updated Parameters

```python
# OLD (single ground truth)
GROUND_TRUTH = "test_data/39/sim_stacks_sms/svr_output.nii.gz"

# NEW (multiple ground truths)
GROUND_TRUTH_DIR = "test_data/ground_truths"
ground_truth_files = sorted(Path(GROUND_TRUTH_DIR).glob("*.nii.gz"))
```

### 3. Enhanced Statistics

The notebook now computes:

- **Per-ground-truth statistics**: Mean, std, min, max for each volume
- **Cross-ground-truth statistics**: Aggregated metrics across all volumes
- **Consistency analysis**: Variance and coefficient of variation across volumes
- **Statistical tests**: ANOVA to detect significant differences between volumes
- **Best configuration per volume**: Identifies optimal parameters for each anatomy

### 4. Enhanced Visualizations

New visualizations include:

- **Per-GT quality comparison**: Boxplots showing PSNR distribution for each volume
- **Mean quality by GT**: Bar charts comparing average metrics across volumes
- **SMS vs Sequential by GT**: Comparative analysis for each volume
- **Consistency analysis**: Variance across ground truths by MB factor

### 5. Organized Output Structure

```
test_data/sms_comprehensive_validation/
├── comprehensive_results.csv                 # All results
├── validation_report.txt                     # Summary report
├── per_ground_truth_comparison.png          # New visualization
├── <ground_truth_name_1>/
│   ├── <ground_truth_name_1>_results.csv   # Results for this GT
│   ├── exp_001_mnone_mb1_s3/
│   ├── exp_002_mnone_mb1_s4/
│   └── ...
├── <ground_truth_name_2>/
│   ├── <ground_truth_name_2>_results.csv
│   ├── exp_065_mnone_mb1_s3/
│   └── ...
└── ...
```

## Usage

### Step 1: Prepare Ground Truth Files

Create a directory with your ground truth volumes:

```bash
mkdir -p test_data/ground_truths
cp /path/to/brain_01.nii.gz test_data/ground_truths/
cp /path/to/brain_02.nii.gz test_data/ground_truths/
cp /path/to/brain_03.nii.gz test_data/ground_truths/
```

Or use the setup script:

```bash
python setup_ground_truths_example.py
```

### Step 2: Run the Validation Notebook

Open and run `sms_detailed_validation.ipynb`. The notebook will:

1. Automatically detect all NIfTI files in `GROUND_TRUTH_DIR`
2. Display the list of ground truths to be processed
3. Run experiments for each ground truth sequentially
4. Compute statistics both per-GT and across all GTs
5. Generate comprehensive visualizations and reports

### Step 3: Review Results

Check the output directory for:

- `comprehensive_results.csv`: Full dataset with `ground_truth` column
- Per-GT CSV files: Individual results for each volume
- Visualization PNGs: Including new per-GT comparison plots
- `validation_report.txt`: Updated with ground truth statistics

## Results Structure

The DataFrame now includes a `ground_truth` column:

| exp_id | ground_truth | motion_level | mb_factor | num_stacks | psnr | ssim | nrmse | transform_valid |
|--------|-------------|--------------|-----------|------------|------|------|-------|-----------------|
| 1      | brain_01    | none         | 1         | 3          | 28.5 | 0.95 | 0.04  | True            |
| 2      | brain_01    | none         | 1         | 4          | 29.1 | 0.96 | 0.03  | True            |
| ...    | ...         | ...          | ...       | ...        | ...  | ...  | ...   | ...             |
| 65     | brain_02    | none         | 1         | 3          | 27.8 | 0.94 | 0.05  | True            |
| ...    | ...         | ...          | ...       | ...        | ...  | ...  | ...   | ...             |

## Statistical Analysis

### Aggregated Statistics

All visualizations and summary statistics now aggregate across ground truths:

- **Mean ± Std**: Averaged across all experiments and all volumes
- **SMS vs Sequential**: Pooled comparison with t-tests
- **Motion level analysis**: Combined data from all ground truths

### Per-Ground-Truth Analysis

New section (10b) provides:

- **ANOVA test**: Checks for significant differences across volumes
- **Coefficient of variation**: Measures consistency across GTs
- **Individual comparisons**: SMS vs sequential for each volume

## Performance Considerations

- **Single GT**: 64 experiments (~2-4 hours depending on hardware)
- **Multiple GTs**: 64 × N experiments (where N = number of ground truths)
- **Recommendation**: Start with 2-3 ground truths for initial validation

## Example Output

```
Experimental Design:
  Ground truth files: 3
    1. brain_01.nii.gz
    2. brain_02.nii.gz
    3. brain_03.nii.gz
  Motion levels: ['none', 'mild', 'moderate', 'severe']
  MB factors: [1, 2, 3, 4]
  Stack counts: [3, 4, 5, 6]
  Total experiments per GT: 64
  Total experiments: 192
  Output directory: test_data/sms_comprehensive_validation

...

STATISTICS BY GROUND TRUTH
================================================================================

brain_01:
  Experiments: 64, PSNR: 28.45 ± 3.21 dB

brain_02:
  Experiments: 64, PSNR: 27.89 ± 3.15 dB

brain_03:
  Experiments: 64, PSNR: 28.12 ± 3.18 dB

...

Statistical Analysis Across Ground Truths:
  Number of ground truths: 3
  ANOVA F-statistic: 2.451, p-value: 0.0877
  → No significant differences in quality across ground truths (p >= 0.05)
  Coefficient of variation: 1.23%
```

## Backward Compatibility

If only one ground truth file is present in the directory, the notebook will:

- Process that single volume
- Skip multi-GT comparison visualizations
- Display a message suggesting adding more volumes for robust statistics

## Troubleshooting

### No ground truth files found

```
ValueError: No NIfTI files found in test_data/ground_truths
```

**Solution**: Add `.nii.gz` or `.nii` files to the ground truth directory.

### Memory issues with many ground truths

**Solution**: Process ground truths in batches or reduce the parameter space:

```python
MB_FACTORS = [1, 2]  # Instead of [1, 2, 3, 4]
NUM_STACKS_OPTIONS = [3, 6]  # Instead of [3, 4, 5, 6]
```

### Inconsistent results across ground truths

Check the ANOVA p-value in section 10b. If p < 0.05, there are significant differences. This could be due to:

- Different brain sizes/anatomies
- Different noise levels
- Different resolutions
- Orientation differences

## Future Enhancements

Possible extensions:

1. **Parallel processing**: Run multiple GTs simultaneously
2. **Batch processing**: Process large GT datasets in chunks
3. **Cross-GT validation**: Train on some GTs, validate on others
4. **Metadata tracking**: Record GT properties (resolution, size, SNR)
5. **Automated GT selection**: Choose diverse set based on image properties

## Contact

For questions or issues with multi-GT validation, please refer to the main project documentation or open an issue.
