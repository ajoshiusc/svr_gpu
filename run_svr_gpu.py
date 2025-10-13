#!/usr/bin/env python3
"""
Prepare a run directory structure and execute the SVR workflow.

Usage:
  run_svr_gpu.py /path/to/raw_dicom_dir /path/to/output_parent_dir [--study-name NAME] [--segmentation twai]

This script will create a timestamped super-directory under the provided output parent dir
and populate the following structure inside it:
  <superdir>/in/dicom
  <superdir>/out/tmp
  <superdir>/out/dicom

It will call `organize_dicom_files.py` to organize the raw DICOMs into `in/dicom`, then run
the existing `svr_dicom.py` workflow, setting `SVR_TEMP_DIR` to the run tmp directory.
"""
import sys
from pathlib import Path
import subprocess
from datetime import datetime
import argparse


def main():
    p = argparse.ArgumentParser(description="Run SVR workflow for a DICOM directory and write outputs under an output folder")
    p.add_argument("dicom_dir", help="Path to raw DICOM directory (unzipped)")
    p.add_argument("output_parent", help="Path where the SVR superdir will be created")
    p.add_argument("--study-name", default="SVR", help="Study prefix for the superdir (default: SVR)")
    p.add_argument("--max-stacks", type=int, default=50,
                   help="Maximum number of stacks to select (default: 50, use 0 for unlimited)")
    p.add_argument("--segmentation", default="twai", help="Segmentation method to pass to svr_dicom.py")
    p.add_argument("--keep-temp", action="store_true", help="Pass --keep-temp to svr_dicom.py to retain temp files")
    p.add_argument("--device", default=None, help="Device id to pass to svr_dicom.py (optional)")
    args = p.parse_args()

    raw_input = Path(args.dicom_dir).expanduser().resolve()
    output_parent = Path(args.output_parent).expanduser().resolve()

    if not raw_input.exists():
        print(f"ERROR: input DICOM directory does not exist: {raw_input}")
        sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    superdir = output_parent / f"{args.study_name}_{ts}"

    in_dir = superdir / "in"
    in_dicom_dir = in_dir / "dicom"
    out_dir = superdir / "out"
    tmp_dir = out_dir / "tmp"
    dicom_out_dir = out_dir / "dicom"

    # create directories
    in_dicom_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dicom_out_dir.mkdir(parents=True, exist_ok=True)

    print("Run directory created:")
    print(f"  superdir: {superdir}")
    print(f"  in/dicom: {in_dicom_dir}")
    print(f"  out/tmp:  {tmp_dir}")
    print(f"  out/dicom:{dicom_out_dir}")

    # Step 1: organize DICOMs into in/dicom
    print("\nSTEP 1: Organizing DICOM files into in/dicom/ structure...")
    organize_cmd = [
        str(Path.cwd() / ".venv" / "bin" / "python"),
        str(Path.cwd() / "organize_dicom_files.py"),
        str(raw_input),
        str(in_dicom_dir),
    ]

    # Run organize command
    rc = subprocess.run(organize_cmd, check=False)
    if rc.returncode != 0:
        print("ERROR: organize_dicom_files.py failed")
        sys.exit(1)

    # Step 2: find series inside in/dicom and select suitable brain/T2 series
    print("\nSTEP 2: Finding brain series in in/dicom/... and running svr_dicom.py")
    import os
    import pydicom

    env = os.environ.copy()
    env['SVR_TEMP_DIR'] = str(tmp_dir)

    # collect series dirs: in/dicom/<studyUID>/<seriesUID>
    series_dirs = []
    for study_dir in in_dicom_dir.iterdir():
        if study_dir.is_dir():
            for series_dir in study_dir.iterdir():
                if series_dir.is_dir():
                    series_dirs.append(series_dir)

    series_info = []
    for series_dir in series_dirs:
        dcm_files = list(series_dir.glob('*.dcm'))
        if not dcm_files:
            continue
        try:
            ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
            desc = getattr(ds, 'SeriesDescription', '')
            series_info.append({'path': series_dir, 'description': desc, 'num_files': len(dcm_files)})
        except Exception:
            continue

    # prefer SSH_TSE T2-weighted sequences
    ssh_tse_series = [s for s in series_info if 'SSH_TSE' in s['description'].upper() or 'SSHTSE' in s['description'].upper()]
    exclude_keywords = ['LOCAL', 'DTI', 'DWI', 'TRACE', 'ADC', 'FA', 'MRCP', 'T1', 'FLAIR', 'BTFE', 'BH']
    ssh_tse_series = [s for s in ssh_tse_series if not any(ex in s['description'].upper() for ex in exclude_keywords)]

    priority_series = []
    if ssh_tse_series:
        # First, prioritize BRAIN series over WHOLE BODY
        brain_ssh_tse = [s for s in ssh_tse_series if 'BRAIN' in s['description'].upper()]
        whole_body_ssh_tse = [s for s in ssh_tse_series if 'WHOLE BODY' in s['description'].upper()]
        
        # Include all brain SSH_TSE series first
        for s in brain_ssh_tse:
            if s not in priority_series:
                priority_series.append(s)
        
        # Then add whole body series by orientation if we need more
        for orientation in ['AXIAL', 'AX ', 'COR', 'SAG']:
            for s in whole_body_ssh_tse:
                desc_words = s['description'].upper().split()
                if orientation in desc_words or orientation in s['description'].upper()[:20]:
                    if s not in priority_series:
                        priority_series.append(s)
                        break
    else:
        # fallback: any T2-weighted series
        t2_keywords = ['T2', 'TSE', 'HASTE', 'SSFSE', 'FIESTA']
        exclude_keywords = ['LOCAL', 'DTI', 'DWI', 'TRACE', 'ADC', 'FA', 'T1', 'FLAIR', 'MRCP']
        t2_series = [s for s in series_info if any(kw in s['description'].upper() for kw in t2_keywords) and not any(ex in s['description'].upper() for ex in exclude_keywords)]
        for orientation in ['AXIAL', 'AX', 'COR', 'SAG']:
            for s in t2_series:
                if orientation in s['description'].upper() and s not in priority_series:
                    priority_series.append(s)
        if len(priority_series) < 3:
            for s in t2_series:
                if s not in priority_series:
                    priority_series.append(s)

    if not priority_series:
        # very last fallback: any series with BRAIN or HEAD in description
        priority_series = [s for s in series_info if 'BRAIN' in s['description'].upper() or 'HEAD' in s['description'].upper()]

    # Apply max_stacks limit if specified
    if args.max_stacks > 0 and len(priority_series) > args.max_stacks:
        priority_series = priority_series[:args.max_stacks]

    if not priority_series:
        print('ERROR: No suitable series found in in/dicom/')
        sys.exit(1)

    print(f"Selected {len(priority_series)} series for reconstruction:")
    for s in priority_series:
        print(f"  - {s['description']} -> {s['path']}")

    # build command with --input-series
    svr_cmd = [
        str(Path.cwd() / '.venv' / 'bin' / 'python'),
        str(Path.cwd() / 'svr_dicom.py'),
        '--input-series'
    ] + [str(s['path']) for s in priority_series] + [
        '--output-dir', str(dicom_out_dir),
        '--segmentation', args.segmentation
    ]

    if args.keep_temp:
        svr_cmd.append('--keep-temp')
    if args.device is not None:
        svr_cmd.extend(['--device', str(args.device)])

    print('\nRunning command:')
    print(' '.join(svr_cmd))
    print(f'Using temp directory: {tmp_dir}')

    rc = subprocess.run(svr_cmd, check=False, env=env)
    if rc.returncode != 0:
        print('ERROR: svr_dicom.py failed')
        sys.exit(1)

    print("\nSVR run complete")
    print(f"Super directory: {superdir}")


if __name__ == '__main__':
    main()
