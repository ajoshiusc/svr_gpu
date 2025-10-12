#!/usr/bin/env python3
"""
Organize DICOM files into studyUID/seriesUID/instanceUID.dcm structure.
Filters for brain series and prepares them for SVR reconstruction.
"""
import os
import pydicom
import shutil
from pathlib import Path
import argparse

def organize_dicom_directory(input_dir, output_dir, series_filter=None, verbose=True):
    """
    Organize DICOM files into studyUID/seriesUID/instanceUID.dcm structure.
    
    Args:
        input_dir: Source directory containing DICOM files (any structure)
        output_dir: Target directory for organized structure
        series_filter: Optional list of keywords to filter series descriptions (e.g., ['BRAIN'])
        verbose: Print progress information
    
    Returns:
        Dictionary mapping series paths to their metadata
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all DICOM files recursively
    dicom_files = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith('.dcm') or file.startswith('IM-'):
                dicom_files.append(Path(root) / file)
    
    if verbose:
        print(f"Found {len(dicom_files)} DICOM files in {input_dir}")
    
    # Organize files by series
    series_info = {}
    processed_files = 0
    skipped_files = 0
    
    for dcm_file in dicom_files:
        try:
            # Read DICOM header without pixel data
            ds = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)
            
            # Extract UIDs
            study_uid = ds.StudyInstanceUID
            series_uid = ds.SeriesInstanceUID
            instance_uid = ds.SOPInstanceUID
            
            # Get series description for filtering
            series_desc = getattr(ds, 'SeriesDescription', 'Unknown')
            
            # Apply series filter if specified
            if series_filter:
                if not any(keyword.lower() in series_desc.lower() for keyword in series_filter):
                    skipped_files += 1
                    continue
            
            # Create organized directory structure
            series_dir = output_path / study_uid / series_uid
            series_dir.mkdir(parents=True, exist_ok=True)
            
            # Target filename: instanceUID.dcm
            target_file = series_dir / f"{instance_uid}.dcm"
            
            # Copy file if it doesn't exist
            if not target_file.exists():
                shutil.copy2(str(dcm_file), str(target_file))
                processed_files += 1
            
            # Store series info
            if str(series_dir) not in series_info:
                series_info[str(series_dir)] = {
                    'study_uid': study_uid,
                    'series_uid': series_uid,
                    'series_description': series_desc,
                    'patient_id': getattr(ds, 'PatientID', 'Unknown'),
                    'study_date': getattr(ds, 'StudyDate', 'Unknown'),
                    'file_count': 0
                }
            series_info[str(series_dir)]['file_count'] += 1
            
        except Exception as e:
            if verbose:
                print(f"Warning: Could not process {dcm_file}: {e}")
            skipped_files += 1
            continue
    
    if verbose:
        print(f"\nProcessed {processed_files} files")
        print(f"Skipped {skipped_files} files")
        print(f"\nOrganized into {len(series_info)} series:")
        for series_path, info in sorted(series_info.items()):
            print(f"\n  {info['series_description']}")
            print(f"    Path: {series_path}")
            print(f"    Files: {info['file_count']}")
            print(f"    Study: {info['study_uid']}")
            print(f"    Series: {info['series_uid']}")
    
    return series_info

def main():
    parser = argparse.ArgumentParser(
        description='Organize DICOM files into studyUID/seriesUID/instanceUID.dcm structure'
    )
    parser.add_argument(
        'input_dir',
        help='Input directory containing DICOM files (any structure)'
    )
    parser.add_argument(
        'output_dir',
        help='Output directory for organized DICOM structure'
    )
    parser.add_argument(
        '--filter',
        nargs='+',
        help='Filter series by keywords in description (e.g., --filter BRAIN)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("DICOM File Organization")
    print("=" * 70)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    if args.filter:
        print(f"Filter: {', '.join(args.filter)}")
    print("=" * 70)
    print()
    
    series_info = organize_dicom_directory(
        args.input_dir,
        args.output_dir,
        series_filter=args.filter,
        verbose=not args.quiet
    )
    
    print("\n" + "=" * 70)
    print(f"Organization complete! {len(series_info)} series organized.")
    print("=" * 70)
    
    return series_info

if __name__ == "__main__":
    main()
