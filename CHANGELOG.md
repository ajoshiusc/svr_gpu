# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2025-10-12

### Added
- Initial release of SVR GPU pipeline
- Complete DICOM workflow with `run_svr_gpu.py`
- Automatic stack reorientation (10-15% quality improvement)
- GPU-accelerated reconstruction using PyTorch
- MONAI-based brain segmentation
- SVoRT quality metrics and motion estimation
- Three workflow options: automated pipeline, NIfTI CLI, DICOM wrapper
- Comprehensive documentation and user guide
- LICENSE (MIT) and VERSION files

### Features
- Auto-detection of T2-weighted SSH_TSE sequences
- Organized output directory structure with timestamps
- Intermediate file preservation with `--keep-temp` flag
- Multi-GPU support via `--device` parameter
- Compatible with MONAI 0.3.0 and 1.3.0 checkpoints
- Automatic DICOM organization by study/series

### Dependencies
- PyTorch 2.8+ with CUDA 12 support
- pydicom 3.0+ with JPEG decompression (pylibjpeg)
- SimpleITK for reorientation
- MONAI for segmentation
- NiBabel, nilearn, scipy, scikit-image

### Based On
- Original NeSVoR implementation by Xu et al. (MIT CSAIL)
- Paper: "NeSVoR: Implicit Neural Representation for Slice-to-Volume Reconstruction in MRI"
- IEEE Transactions on Medical Imaging, 2023

### Known Issues
- Requires JPEG decompression libraries for some DICOM formats
- CPU-only mode is significantly slower (not recommended for production)
- Large GPU memory requirement (8GB+ recommended)

## [Unreleased]

### Planned Features
- Batch processing for multiple studies
- REST API for clinical integration
- Docker container for easy deployment
- Extended DICOM compatibility testing
- Performance optimizations for CPU mode
