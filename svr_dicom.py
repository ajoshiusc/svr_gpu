#!/usr/bin/env python3
"""
DICOM wrapper for SVR reconstruction.

This script handles DICOM input/output and uses svr_cli.py under the hood.
DICOM files are organized as: studyUID/seriesUID/instanceUID.dcm

Usage:
    python svr_dicom.py --input-series <series_dir1> <series_dir2> ... --output-dir <output_dir>
"""

import argparse
import logging
import os
import sys
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple
import pydicom
import numpy as np
import nibabel as nib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def collect_dicom_files(series_dir: str) -> List[str]:
    """
    Collect all DICOM files from a series directory.
    
    Args:
        series_dir: Path to series directory (studyUID/seriesUID/)
        
    Returns:
        List of DICOM file paths sorted by instance number
    """
    dicom_files = []
    series_path = Path(series_dir)
    
    if not series_path.exists():
        raise ValueError(f"Series directory does not exist: {series_dir}")
    
    # Collect all .dcm files
    for file_path in series_path.glob("*.dcm"):
        dicom_files.append(str(file_path))
    
    if not dicom_files:
        raise ValueError(f"No DICOM files found in: {series_dir}")
    
    # Sort by instance number
    def get_instance_number(dcm_path):
        try:
            ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
            return int(ds.InstanceNumber)
        except:
            return 0
    
    dicom_files.sort(key=get_instance_number)
    logger.info(f"Found {len(dicom_files)} DICOM files in {series_path.name}")
    
    return dicom_files


def dicom_series_to_nifti(dicom_files: List[str], output_path: str) -> Tuple[str, dict]:
    """
    Convert DICOM series to NIfTI format.
    
    Args:
        dicom_files: List of DICOM file paths
        output_path: Output NIfTI file path
        
    Returns:
        Tuple of (output_path, metadata_dict)
    """
    # Skip conversion if less than 3 slices
    if len(dicom_files) < 3:
        logger.info(f"Skipping series with only {len(dicom_files)} slices (less than 3): {os.path.dirname(dicom_files[0])}")
        return None, None

    import dicom2nifti
    import dicom2nifti.settings as settings
    import glob
    # Assume all DICOM files are from the same series directory
    series_dir = os.path.dirname(dicom_files[0])
    temp_out_dir = tempfile.mkdtemp(prefix="dcm2nii_")
    try:
        # Disable strict slice increment validation to handle variable spacing
        settings.disable_validate_slice_increment()
        dicom2nifti.convert_directory(series_dir, temp_out_dir, compression=True)
        # Find the generated NIfTI file (should be only one for a single series)
        nii_files = glob.glob(os.path.join(temp_out_dir, '*.nii.gz'))
        if not nii_files:
            raise RuntimeError(f"dicom2nifti did not produce a NIfTI file for {series_dir}")
        # Robust NIfTI validation
        import nibabel as nib
        import numpy as np
        try:
            nii_img = nib.load(nii_files[0])
            shape = nii_img.shape
            logger.info(f"Loaded NIfTI shape: {shape} from {nii_files[0]}")
            # Explicitly skip if any dimension is zero
            if any(dim == 0 for dim in shape):
                logger.info(f"Skipping series at {series_dir} because NIfTI has a zero dimension (shape: {shape})")
                return None, None
            # Only accept 3D (or 4D with 1 in 4th dim) NIfTI
            if not (len(shape) == 3 or (len(shape) == 4 and shape[3] == 1)):
                logger.info(f"Skipping series at {series_dir} because NIfTI is not strictly 3D (shape: {shape})")
                return None, None
            if any(dim <= 1 for dim in shape[:3]):
                logger.info(f"Skipping series at {series_dir} because NIfTI has a singular or one-sized dimension (shape: {shape})")
                return None, None
            data = nii_img.get_fdata()
            if np.isnan(data).any() or np.isinf(data).any():
                logger.info(f"Skipping series at {series_dir} because NIfTI contains NaN or Inf values (shape: {shape})")
                return None, None
            if np.all(data == 0):
                logger.info(f"Skipping series at {series_dir} because NIfTI is all zeros (shape: {shape})")
                return None, None
        except Exception as e:
            logger.info(f"Skipping series at {series_dir} due to NIfTI load/validation error: {e}")
            return None, None
        shutil.move(nii_files[0], output_path)
        logger.info(f"  Converted to NIfTI using dicom2nifti: {os.path.basename(output_path)}")
        # Read first DICOM for metadata
        ds_first = pydicom.dcmread(dicom_files[0])
        metadata = {
            'StudyInstanceUID': str(ds_first.StudyInstanceUID),
            'SeriesInstanceUID': str(ds_first.SeriesInstanceUID),
            'SeriesDescription': str(getattr(ds_first, 'SeriesDescription', 'Unknown')),
            'PatientID': str(getattr(ds_first, 'PatientID', 'Unknown')),
            'PatientName': str(getattr(ds_first, 'PatientName', 'Unknown')),
            'StudyDate': str(getattr(ds_first, 'StudyDate', '')),
            'Modality': str(getattr(ds_first, 'Modality', 'MR')),
        }
        return output_path, metadata
    finally:
        shutil.rmtree(temp_out_dir)



def nifti_to_dicom_series(nifti_path: str, output_dir: str, reference_metadata: dict, 
                          series_description: str = "SVR Reconstruction") -> str:
    """
    Convert NIfTI volume back to DICOM series.
    
    Args:
        nifti_path: Path to NIfTI file
        output_dir: Output directory for DICOM series (will create studyUID/seriesUID/)
        reference_metadata: Metadata from original DICOM series
        series_description: Description for the new series
        
    Returns:
        Path to output series directory
    """
    # Load NIfTI
    nii_img = nib.load(nifti_path)
    volume = nii_img.get_fdata()
    
    # Create output directory structure
    study_uid = reference_metadata.get('StudyInstanceUID', pydicom.uid.generate_uid())
    series_uid = pydicom.uid.generate_uid()  # New series UID for reconstruction
    
    output_series_dir = Path(output_dir) / study_uid / series_uid
    output_series_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating DICOM series: {output_series_dir}")
    
    # Get reference DICOM for metadata (if available)
    # For now, create minimal DICOM

    # Convert volume orientation: use NIfTI affine to compute DICOM ImagePositionPatient
    # and ImageOrientationPatient. Handle NIfTI (usually RAS) to DICOM (LPS) conversion.
    affine = nii_img.affine

    # Helper: convert voxel (i, j, k) indices to patient coordinates (LPS)
    def voxel_to_lps(i, j, k):
        v = np.array([i, j, k, 1.0])
        xyz = affine.dot(v)[:3]
        # NIfTI affine is usually RAS; DICOM expects LPS -> flip x and y
        xyz_lps = xyz.copy()
        xyz_lps[0] = -xyz_lps[0]
        xyz_lps[1] = -xyz_lps[1]
        return xyz_lps

    num_slices = volume.shape[2]

    # Precompute spacing and direction cosines from affine
    # p00: voxel (0,0,0)
    p00 = voxel_to_lps(0, 0, 0)
    p01 = voxel_to_lps(0, 1, 0)  # move +1 in column (j)
    p10 = voxel_to_lps(1, 0, 0)  # move +1 in row (i)

    # Direction cosines: first triplet = direction of increasing column index
    # (from pixel (0,0) to (0,1)); second triplet = direction of increasing row index
    row_vec = p01 - p00
    col_vec = p10 - p00
    # Normalize to get direction cosines
    def safe_normalize(v):
        norm = np.linalg.norm(v)
        return (v / norm) if norm > 0 else v

    dir_col = safe_normalize(row_vec)
    dir_row = safe_normalize(col_vec)

    # Pixel spacing: [row spacing (distance between rows), column spacing]
    row_spacing = float(np.linalg.norm(col_vec))
    col_spacing = float(np.linalg.norm(row_vec))

    # Slice spacing: distance between adjacent slices along k
    if num_slices > 1:
        p0_slice = voxel_to_lps(0, 0, 0)
        p1_slice = voxel_to_lps(0, 0, 1)
        slice_spacing = float(np.linalg.norm(p1_slice - p0_slice))
    else:
        slice_spacing = 0.0

    for slice_idx in range(num_slices):
        # Create DICOM dataset
        ds = pydicom.Dataset()
        
        # Patient module
        ds.PatientName = reference_metadata.get('PatientName', 'Unknown')
        ds.PatientID = reference_metadata.get('PatientID', 'Unknown')
        
        # Study module
        ds.StudyInstanceUID = study_uid
        ds.StudyDate = reference_metadata.get('StudyDate', '')
        ds.StudyTime = ''
        ds.AccessionNumber = ''
        ds.StudyID = '1'
        
        # Series module
        ds.SeriesInstanceUID = series_uid
        ds.SeriesNumber = '999'
        ds.Modality = reference_metadata.get('Modality', 'MR')
        ds.SeriesDescription = series_description
        
        # Image module
        ds.InstanceNumber = str(slice_idx + 1)
        ds.SOPInstanceUID = pydicom.uid.generate_uid()
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.4'  # MR Image Storage
        
        # Image pixel module
        slice_data = volume[:, :, slice_idx].astype(np.uint16)
        ds.Rows = slice_data.shape[0]
        ds.Columns = slice_data.shape[1]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        
        # Spacing and orientation from affine-derived values
        ds.PixelSpacing = [row_spacing, col_spacing]
        ds.SliceThickness = slice_spacing if slice_spacing > 0 else 0.0
        ds.SpacingBetweenSlices = slice_spacing if slice_spacing > 0 else ds.SliceThickness

        # ImagePositionPatient: world coordinates (LPS) of the first pixel (0,0,slice)
        ipp = voxel_to_lps(0, 0, slice_idx)
        ds.ImagePositionPatient = [float(ipp[0]), float(ipp[1]), float(ipp[2])]

        # ImageOrientationPatient: two direction cosines (column vector, row vector)
        ds.ImageOrientationPatient = [
            float(dir_col[0]), float(dir_col[1]), float(dir_col[2]),
            float(dir_row[0]), float(dir_row[1]), float(dir_row[2])
        ]

        # Set pixel data
        ds.PixelData = slice_data.tobytes()

        # Set file meta information
        ds.file_meta = pydicom.dataset.FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.file_meta.ImplementationClassUID = pydicom.uid.generate_uid()

        ds.is_little_endian = True
        ds.is_implicit_VR = False

        # Save DICOM file
        output_file = output_series_dir / f"{ds.SOPInstanceUID}.dcm"
        ds.save_as(str(output_file), write_like_original=False)
    
    logger.info(f"  Created {num_slices} DICOM slices")
    logger.info(f"  Series directory: {output_series_dir}")
    
    return str(output_series_dir)


def run_svr_cli(nifti_inputs, nifti_output, args, temp_dir=None):
    """
    Run svr_cli.py with NIfTI files.
    
    Args:
        nifti_inputs: List of input NIfTI files
        nifti_output: Output NIfTI file path
        args: Command-line arguments from main script
        temp_dir: Temporary directory for intermediate outputs (optional)
        
    Returns:
        Return code from svr_cli.py
    """
    # Build svr_cli.py command
    svr_cli_path = Path(__file__).parent / "svr_cli.py"
    
    cmd = [
        sys.executable,
        str(svr_cli_path),
        "--input-stacks"
    ] + nifti_inputs + [
        "--output", nifti_output,
    ]
    
    # Pass through optional arguments
    if args.segmentation:
        cmd += ["--segmentation", args.segmentation]
    if args.device is not None:
        cmd += ["--device", str(args.device)]
    if args.no_auto_reorient:
        cmd += ["--no-auto-reorient"]
    if args.bias_field_correction:
        cmd += ["--bias-field-correction"]
    if hasattr(args, 'batch_size_seg') and args.batch_size_seg is not None:
        cmd += ["--batch-size-seg", str(args.batch_size_seg)]
    
    # Add robust reconstruction settings to eliminate holes
    # Disable local SSIM-based exclusion which causes scattered holes
    cmd += [
        "--global-ncc-threshold", "0.3",  # More permissive (default 0.5)
        "--no-local-exclusion",           # Disable SSIM-based pixel rejection
    ]
    
    logger.info(f"Running SVR reconstruction...")
    logger.debug(f"Command: {' '.join(cmd)}")
    
    # Set environment variable for temp directory if provided
    env = os.environ.copy()
    if temp_dir:
        env['SVR_TEMP_DIR'] = str(temp_dir)
    
    # Run svr_cli.py
    result = subprocess.run(cmd, env=env)
    
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description='SVR reconstruction with DICOM input/output'
    )
    
    # Input/output arguments
    parser.add_argument('--input-series', required=True, nargs='+',
                        help='Input DICOM series directories (studyUID/seriesUID/)')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for DICOM results')
    
    # SVR arguments (passed through to svr_cli.py)
    parser.add_argument('--segmentation', default='twai',
                        help='Segmentation method (default: twai)')
    parser.add_argument('--device', type=int, default=0,
                        help='CUDA device ID (default: 0)')
    parser.add_argument('--no-auto-reorient', action='store_true',
                        help='Disable automatic stack reorientation')
    parser.add_argument('--bias-field-correction', action='store_true',
                        help='Apply N4 bias field correction')
    
    # DICOM-specific arguments
    parser.add_argument('--series-description', default='SVR Reconstruction',
                        help='Description for output DICOM series (default: "SVR Reconstruction")')
    parser.add_argument('--keep-temp', action='store_true',
                        help='Keep temporary NIfTI files for debugging')
    
    parser.add_argument('--batch-size-seg', type=int, default=None,
                        help='Segmentation batch size to pass to SVR CLI')
    args = parser.parse_args()
    
    # Create temp directory for NIfTI conversion
    # Priority:
    # 1. Use SVR_TEMP_DIR environment variable if set (passed by workflow scripts)
    # 2. Try to find */out/tmp directory in workspace (for unique super dirs)
    # 3. Fall back to system temp
    if 'SVR_TEMP_DIR' in os.environ:
        workspace_tmp = Path(os.environ['SVR_TEMP_DIR'])
        temp_dir = str(workspace_tmp)
        temp_input_dir = workspace_tmp / "input"
        temp_input_dir.mkdir(parents=True, exist_ok=True)
        temp_output_nifti = workspace_tmp / "svr_output.nii.gz"
        logger.info(f"Using temp directory from environment: {workspace_tmp}")
    else:
        # Look for any */out/tmp directory in workspace
        workspace_root = Path(__file__).parent
        tmp_dirs = list(workspace_root.glob("*/out/tmp"))
        if tmp_dirs:
            workspace_tmp = tmp_dirs[0]  # Use the first one found
            temp_dir = str(workspace_tmp)
            temp_input_dir = workspace_tmp / "input"
            temp_input_dir.mkdir(exist_ok=True)
            temp_output_nifti = workspace_tmp / "svr_output.nii.gz"
            logger.info(f"Using workspace temp directory: {workspace_tmp}")
        else:
            temp_dir = tempfile.mkdtemp(prefix="svr_dicom_")
            temp_input_dir = Path(temp_dir) / "input"
            temp_input_dir.mkdir()
            temp_output_nifti = Path(temp_dir) / "svr_output.nii.gz"
            logger.info(f"Using system temp directory: {temp_dir}")
    
    try:
        logger.info("=" * 60)
        logger.info("SVR DICOM Reconstruction")
        logger.info("=" * 60)
        
        # Step 1: Convert DICOM series to NIfTI
        logger.info("\nStep 1: Converting DICOM to NIfTI...")
        nifti_inputs = []
        all_metadata = []
        
        for i, series_dir in enumerate(args.input_series):
            logger.info(f"\nProcessing series {i+1}/{len(args.input_series)}: {series_dir}")
            
            # Collect DICOM files
            dicom_files = collect_dicom_files(series_dir)
            
            # Convert to NIfTI
            nifti_path = temp_input_dir / f"stack_{i:02d}.nii.gz"
            nifti_path, metadata = dicom_series_to_nifti(dicom_files, str(nifti_path))
            nifti_inputs.append(nifti_path)
            all_metadata.append(metadata)
        
        # Step 2: Run SVR reconstruction
        logger.info("\n" + "=" * 60)
        logger.info("Step 2: Running SVR reconstruction...")
        logger.info("=" * 60)
        
        returncode = run_svr_cli(nifti_inputs, str(temp_output_nifti), args, temp_dir=temp_dir)
        
        if returncode != 0:
            logger.error(f"SVR reconstruction failed with return code {returncode}")
            return returncode
        
        if not temp_output_nifti.exists():
            logger.error("SVR output file was not created")
            return 1
        
        # Step 3: Convert output NIfTI back to DICOM
        logger.info("\n" + "=" * 60)
        logger.info("Step 3: Converting output to DICOM...")
        logger.info("=" * 60)
        
        # Use metadata from first input series as reference
        reference_metadata = all_metadata[0]
        
        output_series_dir = nifti_to_dicom_series(
            str(temp_output_nifti),
            args.output_dir,
            reference_metadata,
            args.series_description
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("SUCCESS!")
        logger.info("=" * 60)
        logger.info(f"Output DICOM series: {output_series_dir}")
        
        # Clean up temp files
        # Only clean up if using system temp directory (not workspace out/tmp)
        workspace_tmp = Path(__file__).parent / "out" / "tmp"
        is_workspace_tmp = Path(temp_dir).resolve() == workspace_tmp.resolve()
        
        if args.keep_temp or is_workspace_tmp:
            logger.info(f"\nTemporary files kept in: {temp_dir}")
        else:
            shutil.rmtree(temp_dir)
            logger.debug(f"Cleaned up temporary directory: {temp_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up on error (only if using system temp)
        workspace_tmp = Path(__file__).parent / "out" / "tmp"
        is_workspace_tmp = Path(temp_dir).resolve() == workspace_tmp.resolve()
        
        if not args.keep_temp and not is_workspace_tmp:
            shutil.rmtree(temp_dir)
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
