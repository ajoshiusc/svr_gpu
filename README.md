# SVR CLI Minimal - Complete Package

## Overview

A complete Super-Resolution Reconstruction (SVR) pipeline for fetal/neonatal MRI with automatic stack reorientation and DICOM support.

## Components

### 1. Core SVR Pipeline
- **`svr_cli.py`** - Main SVR reconstruction script (NIfTI input/output)
- **`svr_dicom.py`** - DICOM wrapper for clinical workflows
- **`standalone_inlined/`** - Core reconstruction algorithms

### 2. Stack Reorientation
- **`reorient_stacks.py`** - Standalone stack reorientation utility
- Uses SimpleITK for consistency with existing workflows
- Automatically detects and corrects slice-stacking orientation
- Integrated into both `svr_cli.py` and `svr_dicom.py` by default

### 3. Documentation
- **`REORIENTATION_INTEGRATION.md`** - How automatic reorientation works
- **`COMPARISON_WITH_WITHOUT_REORIENT.md`** - Quality comparison with/without reorientation
- **`SVR_DICOM_README.md`** - DICOM wrapper usage guide

## Quick Start

### NIfTI Workflow

```bash
# Basic reconstruction with automatic reorientation
python svr_cli.py \
  --input-stacks axial.nii.gz coronal.nii.gz sagittal.nii.gz \
  --output reconstructed.nii.gz \
  --segmentation twai

# Disable auto-reorientation
python svr_cli.py \
  --input-stacks stack1.nii.gz stack2.nii.gz \
  --output result.nii.gz \
  --segmentation twai \
  --no-auto-reorient
```

### DICOM Workflow

```bash
# DICOM input/output
python svr_dicom.py \
  --input-series \
    /path/to/study/series_axial/ \
    /path/to/study/series_coronal/ \
    /path/to/study/series_sagittal/ \
  --output-dir /path/to/output/ \
  --segmentation twai
```

### Standalone Reorientation

```bash
# Reorient stacks only (no reconstruction)
python reorient_stacks.py \
  --input stack1.nii.gz stack2.nii.gz stack3.nii.gz \
  --output-dir reoriented/
```

## Key Features

### ✅ Automatic Stack Reorientation (NEW!)

**Enabled by default** - automatically detects and corrects stack orientations:
- ✅ Detects slice-stacking axis (largest voxel dimension)
- ✅ Reorients to axial (slices in XY plane, stacked along Z)
- ✅ Improves registration quality (10-15% better similarity scores)
- ✅ Eliminates "different thicknesses" warnings
- ✅ Works with mixed orientations (axial, coronal, sagittal)
- ✅ Uses SimpleITK for consistency with `rotate2makeslice_z.py`
- ✅ Can be disabled with `--no-auto-reorient` flag

**Comparison:**
| Metric | With Reorientation | Without Reorientation |
|--------|-------------------|----------------------|
| SVoRT similarity | 0.853 | 0.718 |
| Stack similarity | 0.832 | 0.811 |
| Warnings | None | "different thicknesses" |

### 🔬 Brain Segmentation
- MONAI-based fetal brain segmentation
- Compatible with MONAI 0.3.0 and 1.3.0 architectures
- Automatic brain masking for reconstruction
# SVR CLI Minimal

This repository contains utilities to run the SVR (slice-to-volume reconstruction) pipeline.

This README provides a minimal usage example for the helper script `run_svr_gpu.py`, which
creates a timestamped run directory, organizes DICOMs, converts selected series to NIfTI, and
invokes the DICOM-driven SVR workflow (`svr_dicom.py`) with `SVR_TEMP_DIR` set so intermediate
files are stored under the run directory.

Prerequisites
 - Python 3.8+
 - A virtual environment with project dependencies installed (see `requirements.txt` if present).
 - Recommended: CUDA-enabled GPU and PyTorch installed in the same environment

Minimal example: run with a DICOM folder and output parent directory

```bash
# from project root
.venv/bin/python run_svr_gpu.py \
  /path/to/dicom_input_folder \
  --output-parent . \
  --study-name SVR002 \
  --device 0 \
  --keep-temp
```

What the script does
 - Creates a run superdirectory under `--output-parent` named `<study>_YYYYMMDD_HHMMSS`
 - Creates `in/dicom/`, `out/tmp/`, and `out/dicom/` inside the run directory
 - Organizes DICOM series into `in/dicom/<StudyUID>/<SeriesUID>/`
 - Selects SSH_TSE / T2 series (auto-detect) and converts them to NIfTI under `out/tmp/input/`
 - Sets `SVR_TEMP_DIR` to the run `out/tmp` and calls `svr_dicom.py --input-series ... --output-dir <rundir>`

Checking outputs
 - The run directory (example): `./SVR002_20251012_153756`
 - Intermediate NIfTI files and per-iteration reconstructions are saved under:
   `./<rundir>/out/tmp/` (and `./<rundir>/out/tmp/reconstructions/` for per-iteration NIfTIs)
 - Final DICOM output (if enabled) is in `./<rundir>/out/dicom/`

Notes
 - This README intentionally focuses on `run_svr_gpu.py` usage. Other documentation in the
   repository has been removed to keep this project minimal. If you need extended documentation,
   ask and I can re-add targeted docs for a specific component (DICOM wrapper, reorientation, etc.).

License: (unchanged)

## Acknowledgements

This project started from the NeSVoR codebase and was adapted to produce a minimal, self-contained
SVR CLI package focused on DICOM-driven slice-to-volume reconstruction and automatic stack
reorientation. Many core algorithms and high-level design choices were informed by NeSVoR — thanks
to its authors for providing a solid starting point.

Useful links:
- NeSVoR GitHub: https://github.com/daviddmc/NeSVoR
- NeSVoR documentation: https://nesvor.readthedocs.io/
