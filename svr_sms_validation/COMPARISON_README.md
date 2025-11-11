# SMS vs Non-SMS SVR Comparison Analysis

## Quick Start

```bash
# Install dependencies
pip install numpy nibabel pandas matplotlib seaborn scipy scikit-image

# Run comparison
python compare_sms_vs_nonsms.py
```

## Overview

Comprehensive analysis comparing SMS (Simultaneous Multi-Slice) vs non-SMS SVR reconstructions.

**Comparison Dimensions:**
- Motion levels: none, mild, moderate, severe
- MB factors: 1 (non-SMS), 2, 3 (SMS)
- Stack counts: 3, 6, 9, 12
- Multiple permutations per condition

## Output

All results saved to: `{BASEPATH}/data/sms_comparison_results/`

### Files Generated

1. **metrics_data.csv** - Raw metrics for all reconstructions
2. **comparison_report.txt** - Comprehensive statistical report
3. **tables/** - Summary statistics in CSV format
4. **plots/** - 5 publication-quality figures

### Plots

1. **sms_vs_nonsms_boxplots.png** - Box plots for all metrics
2. **performance_vs_stacks.png** - Quality vs number of stacks
3. **ncc_heatmap_motion_mb.png** - Motion level × MB factor heatmap
4. **ncc_violin_by_mb.png** - Distribution by MB factor
5. **sms_benefit_by_motion.png** - SMS improvement scatter plots

## Metrics

- **NCC** (↑): Normalized Cross-Correlation (0-1, higher better)
- **RMSE** (↓): Root Mean Square Error (lower better)
- **PSNR** (↑): Peak Signal-to-Noise Ratio in dB (higher better)
- **SSIM** (↑): Structural Similarity (0-1, higher better)

## Configuration

Edit paths in `compare_sms_vs_nonsms.py`:

```python
OUTPUT_DIR = Path(BASEPATH) / "data" / "svr_reconstructions"
GT_DIR = Path(BASEPATH) / "data" / "ground_truths"
RESULTS_DIR = Path(BASEPATH) / "data" / "sms_comparison_results"
```

## Requirements

- Python 3.7+
- numpy
- nibabel
- pandas
- matplotlib
- seaborn
- scipy
- scikit-image

See full documentation in README_comparison.md
