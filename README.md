# SVR GPU - Slice-to-Volume Reconstruction

A GPU-accelerated slice-to-volume reconstruction (SVR) pipeline for fetal/neonatal MRI with automatic stack reorientation and DICOM support.

## Features

- 🚀 **GPU-accelerated reconstruction** using PyTorch and CUDA
- 🔄 **Automatic stack reorientation** - detects and corrects slice orientations
- 🏥 **DICOM support** - direct DICOM input/output for clinical workflows
- 🧠 **Brain segmentation** - MONAI-based fetal brain masking
- 📊 **Quality metrics** - SVoRT similarity tracking and motion estimation
- 🛠️ **Flexible workflows** - NIfTI-based CLI or DICOM wrapper

## Installation

### Prerequisites

- Python 3.8 or later (Python 3.10+ recommended)
- CUDA-compatible GPU with drivers installed (optional but recommended)
- 8GB+ GPU memory for typical fetal MRI reconstruction

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/ajoshiusc/svr_gpu.git
   cd svr_gpu
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
   ```

### Notes on Dependencies

- **PyTorch with CUDA**: The default `requirements.txt` installs PyTorch with CUDA 12 support (~3GB download)
- **CPU-only installation**: If you don't have a GPU, install CPU-only PyTorch:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  pip install numpy nilearn scipy monai scikit-image
  ```
- **Different CUDA version**: See [PyTorch installation guide](https://pytorch.org/get-started/locally/) for other CUDA versions



## Quick Start

### Option 1: NIfTI Workflow (Recommended for Research)

Use `svr_cli.py` for NIfTI input/output with full control over parameters.

```bash
# Activate virtual environment
source .venv/bin/activate

# Basic reconstruction with automatic reorientation
python svr_cli.py \
  --input-stacks axial.nii.gz coronal.nii.gz sagittal.nii.gz \
  --output reconstructed.nii.gz \
  --segmentation twai  # Use 'none' to disable segmentation

# Advanced: disable auto-reorientation if stacks are already aligned
python svr_cli.py \
  --input-stacks stack1.nii.gz stack2.nii.gz \
  --output result.nii.gz \
  --segmentation twai \
  # Run without segmentation
  python svr_cli.py \
    --input-stacks stack1.nii.gz stack2.nii.gz \
    --output result.nii.gz \
    --segmentation none
  --no-auto-reorient
```

### Option 2: DICOM Workflow (Clinical Use)

Use `svr_dicom.py` for direct DICOM input/output.

```bash
# Activate virtual environment
source .venv/bin/activate

# Process DICOM series directly
python svr_dicom.py \
  --input-series \
    /path/to/dicom/series_axial/ \
    /path/to/dicom/series_coronal/ \
    /path/to/dicom/series_sagittal/ \
  --output-dir /path/to/output/ \
  --segmentation twai
```

### Option 3: Complete Pipeline with `run_svr_gpu.py`

Use `run_svr_gpu.py` for a fully automated workflow that organizes inputs, runs reconstruction, and saves all outputs in a timestamped directory.

**Command syntax:**
```bash
python run_svr_gpu.py DICOM_DIR OUTPUT_PARENT [OPTIONS]
```

**Key options:**
- `--device N`: GPU device ID (0, 1, ...) or -1 for CPU-only
- `--batch-size-seg N`: Segmentation batch size (default: 16, use 4-8 for memory-constrained systems)
- `--max-series N`: Limit number of series to process (prioritizes brain+TSE sequences) (default: 4)
- `--no-augmentation-seg`: Disable segmentation augmentation for faster CPU processing
- `--keep-temp`: Keep intermediate NIfTI files for inspection
- `--study-name NAME`: Custom name for output directory
- `--include-series-keyword KEY`: Only include series whose SeriesDescription contains ALL provided keywords (case-insensitive). Repeatable.

**Example:**
```bash
# Activate virtual environment
source .venv/bin/activate

# Run complete pipeline
python run_svr_gpu.py \
  /path/to/dicom_folder \
  ./results \
  --study-name MyStudy \
  --device 0 \
  --keep-temp
```

**Real example with CHLA data:**
```bash
# GPU run with memory optimization
python run_svr_gpu.py \
  /deneb_disk/chla_data_2_21_2023/unzipped_dicomms/SVR001 \
  . \
  --study-name SVR001 \
  --device 0 \
  --batch-size-seg 8 \
  --include-series-keyword brain \
  --include-series-keyword tse \
  --max-series 5 \
  --keep-temp

# CPU run (slower but no GPU required)
python run_svr_gpu.py \
  /deneb_disk/chla_data_2_21_2023/unzipped_dicomms/SVR005 \
  . \
  --study-name SVR005 \
  --device -1 \
  --batch-size-seg 4 \
  --max-series 7 \
  --no-augmentation-seg \
  --keep-temp
```

**What this does:**
- Creates timestamped run directory: `./results/MyStudy_YYYYMMDD_HHMMSS/`
- Organizes DICOM files by series
- Auto-detects and prioritizes brain T2-weighted sequences (SSH_TSE/T2)
- Limits series count to avoid memory issues (via `--max-series`)
- Converts to NIfTI with preserved original series names
- Runs SVR reconstruction with GPU or CPU
- Saves intermediate files and final DICOM outputs

**Output structure:**
```
MyStudy_20251012_153756/
├── in/dicom/           # Organized DICOM input by series
├── out/
│   ├── tmp/            # Intermediate files (if --keep-temp)
│   │   ├── input/      # NIfTI stacks with original names
│   │   └── reconstructions/  # Per-iteration volumes
│   └── dicom/          # Final DICOM output series
```




## Key Features Explained

### Automatic Stack Reorientation

**Enabled by default** - automatically detects and corrects stack orientations before reconstruction.

**How it works:**
- Detects the slice-stacking axis (dimension with largest voxel spacing)
- Reorients all stacks to axial orientation (slices in XY plane, stacked along Z)
- Ensures consistent coordinate systems across mixed-orientation inputs

### Brain Segmentation

- Uses MONAI-based deep learning models for fetal brain segmentation
- Compatible with MONAI 1.3.0 checkpoint architectures
- Automatically generates brain masks to improve reconstruction quality
- Automatically reorients the reconstructed brain to the standard orientation

### Intelligent Series Selection

The pipeline automatically prioritizes DICOM series for optimal reconstruction:

**Priority order:**
1. **Highest**: Series containing both "BRAIN" and T2-weighted terms (TSE/SSH_TSE/T2)
2. **High**: Series containing "BRAIN" or "HEAD" keywords
3. **Medium**: Series containing T2-weighted sequence terms
4. **Low**: Other series

**Features:**
- Preserves original series names in NIfTI output (e.g., `BRAIN_AXIAL_SSh_TSE_esp5_6.nii.gz`)
- Attempts to include different orientations (axial, coronal, sagittal)
- Respects `--max-series` limit to avoid memory issues
- Skips invalid or failed DICOM-to-NIfTI conversions

### GPU/CPU Acceleration

- **GPU mode**: PyTorch + CUDA for fast reconstruction (5-15 minutes per case)
- **CPU mode**: Use `--device -1` for systems without GPU (30-60 minutes per case)
- **Memory optimization**: Use `--batch-size-seg 4-8` to reduce memory usage
- **Series limiting**: Use `--max-series N` to process only the best N series
- Supports multi-GPU systems (specify GPU with `--device 0`, `--device 1`, etc.)

## Utilities and Tools

### Standalone Stack Reorientation

Reorient stacks without running full reconstruction:

```bash
python reorient_stacks.py \
  --input stack1.nii.gz stack2.nii.gz stack3.nii.gz \
  --output-dir reoriented/
```

### DICOM Organization

Organize DICOM files by study and series:

```bash
python organize_dicom_files.py \
  --input-dir /path/to/dicoms \
  --output-dir /path/to/organized
```

## Project Structure

```
svr_gpu/
├── svr_cli.py              # Main NIfTI reconstruction script
├── svr_dicom.py            # DICOM wrapper script
├── run_svr_gpu.py          # Complete automated pipeline
├── reorient_stacks.py      # Standalone reorientation utility
├── organize_dicom_files.py # DICOM organization utility
├── requirements.txt        # Python dependencies
└── standalone_inlined/     # Core algorithms and models
    ├── assessment/         # Quality metrics (SVoRT, IQA)
    ├── checkpoints/        # Pre-trained model weights
    ├── inr/                # Implicit neural representation
    ├── preprocessing/      # Brain segmentation, bias correction
    ├── slice_acquisition/  # Forward imaging model
    ├── svort/              # Slice-to-volume registration transformers
    ├── svr/                # SVR reconstruction algorithms
    └── utils/              # Helper functions
```



## Troubleshooting

### CUDA/GPU Issues

**Problem:** "CUDA out of memory"
- **Solution:** 
  ```bash
  # Reduce segmentation batch size
  --batch-size-seg 4
  
  # Limit number of series processed
  --max-series 5
  
  # Or use CPU-only mode
  --device -1
  ```

**Problem:** "CUDA not available" despite having a GPU
- **Solution:** Check driver compatibility and reinstall PyTorch with matching CUDA version:
  ```bash
  python -c "import torch; print(torch.version.cuda)"  # Check PyTorch CUDA version
  nvidia-smi  # Check driver CUDA version
  ```

**Problem:** Very slow processing
- **Solution for CPU runs:** Use optimization flags:
  ```bash
  --device -1 --batch-size-seg 4 --no-augmentation-seg --max-series 5
  ```

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'monai'` or similar
- **Solution:** Ensure virtual environment is activated and dependencies are installed:
  ```bash
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

### DICOM Processing Issues

**Problem:** "No T2-weighted series found"
- **Solution:** Check series descriptions in your DICOM files. You may need to manually specify series with `--input-series`

## Additional Documentation

For more detailed information on specific components:
- **Reorientation details**: See code comments in `reorient_stacks.py`
- **DICOM wrapper**: Check `svr_dicom.py` for all available options
- **Algorithm details**: Refer to the NeSVoR paper and documentation (links below)



## Acknowledgements

This project is based on the [NeSVoR](https://github.com/daviddmc/NeSVoR) codebase and was adapted to create a self-contained, DICOM-focused SVR pipeline with automatic stack reorientation. Many core algorithms and design choices originate from NeSVoR.

**NeSVoR Resources:**
- GitHub: https://github.com/daviddmc/NeSVoR
- Documentation: https://nesvor.readthedocs.io/
- Paper: [NeSVoR: Implicit Neural Representation for Slice-to-Volume Reconstruction in MRI](https://ieeexplore.ieee.org/document/10015091)

We thank the NeSVoR authors for providing an excellent foundation for GPU-accelerated slice-to-volume reconstruction.

## License

This project maintains the same license as the original NeSVoR codebase. Please refer to the LICENSE file and the original NeSVoR repository for details.

## Citation

If you use this code in your research, please cite the original NeSVoR paper:

```bibtex
@article{xu2023nesvor,
  title={NeSVoR: Implicit Neural Representation for Slice-to-Volume Reconstruction in MRI},
  author={Xu, Junshen and Moyer, Daniel and Gagoski, Borjan and Iglesias, Juan Eugenio and Grant, P Ellen and Golland, Polina and Adalsteinsson, Elfar},
  journal={IEEE Transactions on Medical Imaging},
  year={2023},
  publisher={IEEE}
}
```

