#!/usr/bin/env python3
"""
Complete automated SVR pipeline with DICOM organization and reconstruction.

Usage:
  run_svr_gpu.py /path/to/raw_dicom_dir /path/to/output_parent_dir [OPTIONS]

This script provides a complete end-to-end SVR workflow:
1. Creates timestamped run directory under output_parent_dir
2. Organizes DICOM files by series with intelligent T2/brain sequence detection
3. Prioritizes series containing both "BRAIN" and T2-weighted terms
4. Converts to NIfTI with preserved original series names
5. Runs SVR reconstruction with GPU or CPU
6. Outputs final DICOM series

Directory structure created:
  <superdir>/in/dicom/        # Organized DICOM input by series
  <superdir>/out/tmp/input/   # NIfTI stacks with original names (if --keep-temp)
  <superdir>/out/dicom/       # Final DICOM output series

Key features:
- Memory optimization via --batch-size-seg and --max-series
- CPU support with --device -1
- Original series name preservation
- Intelligent series prioritization for brain T2-weighted sequences
- Optional manual filtering with --include-series-keyword
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
    p.add_argument("--segmentation", default="twai", help="Segmentation method to pass to svr_dicom.py")
    p.add_argument("--keep-temp", action="store_true", help="Pass --keep-temp to svr_dicom.py to retain temp files")
    p.add_argument("--device", default=None, help="Device id to pass to svr_dicom.py (optional)")
    p.add_argument("--batch-size-seg", type=int, default=None, help="Segmentation batch size to pass to svr_dicom.py")
    p.add_argument("--max-series", type=int, default=4, help="Maximum number of series to use (prioritized) (default: 4)")
    p.add_argument(
        "--include-series-keyword",
        action="append",
        default=None,
        help=(
            "Only include DICOM series whose SeriesDescription contains ALL of the provided keywords. "
            "Case-insensitive. Can be specified multiple times."
        ),
    )
    p.add_argument("--te", type=float, default=None, help="Filter series by Echo Time (TE)")
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
            te = getattr(ds, 'EchoTime', None)
            series_info.append({'path': series_dir, 'description': desc, 'num_files': len(dcm_files), 'te': te})
        except Exception:
            continue

    # Optional manual filtering by TE
    if args.te is not None:
        before = len(series_info)
        series_info = [
            s for s in series_info
            if s['te'] is not None and abs(float(s['te']) - args.te) < 1.0
        ]
        print(f"Filtered series by TE={args.te}: {len(series_info)} of {before} remain")

    # Optional manual filtering by SeriesDescription keyword(s)
    if args.include_series_keyword:
        kws = [kw.lower() for kw in args.include_series_keyword if kw and kw.strip()]
        original_series_info = list(series_info)
        if kws:
            before = len(series_info)
            series_info = [
                s for s in series_info
                if all(kw in (s['description'] or '').lower() for kw in kws)
            ]
            print(f"Filtered series by ALL keywords {kws}: {len(series_info)} of {before} remain")
        if not series_info:
            print("ERROR: No series matched --include-series-keyword (ALL keywords must match).")
            # Print a few available descriptions to help user
            unique_desc = sorted({(s.get('description') or '').strip() for s in original_series_info if s.get('description')})
            if unique_desc:
                print("Available series descriptions (sample):")
                for d in unique_desc[:10]:
                    print(f"  - {d}")
            sys.exit(1)


    # Highest priority: series with BOTH 'BRAIN' and ('TSE', 'SSH_TSE', 'SSHTSE', 'T2') in description
    tse_keywords = ['TSE', 'SSH_TSE', 'SSHTSE', 'T2']
    exclude_keywords = ['LOCAL', 'DTI', 'DWI', 'TRACE', 'ADC', 'FA', 'MRCP', 'T1', 'FLAIR', 'BTFE', 'BH']
    both_brain_tse = [
        s for s in series_info
        if 'BRAIN' in s['description'].upper()
        and any(kw in s['description'].upper() for kw in tse_keywords)
        and not any(ex in s['description'].upper() for ex in exclude_keywords)
    ]
    # Next: series with any of BRAIN/HEAD/TSE/SSH_TSE/SSHTSE/T2
    priority_keywords = ['BRAIN', 'HEAD', 'TSE', 'SSH_TSE', 'SSHTSE', 'T2']
    priority_series_candidates = [
        s for s in series_info
        if any(kw in s['description'].upper() for kw in priority_keywords)
        and not any(ex in s['description'].upper() for ex in exclude_keywords)
        and s not in both_brain_tse
    ]
    priority_series = []
    # Add all both_brain_tse series first (by orientation)
    for orientation in ['AXIAL', 'AX ', 'COR', 'SAG']:
        for s in both_brain_tse:
            desc_words = s['description'].upper().split()
            if orientation in desc_words or orientation in s['description'].upper()[:20]:
                if s not in priority_series:
                    priority_series.append(s)
                    break
    for s in both_brain_tse:
        if s not in priority_series:
            priority_series.append(s)
            if args.max_series is not None and len(priority_series) >= args.max_series:
                break
    # Then add other priority candidates (by orientation)
    for orientation in ['AXIAL', 'AX ', 'COR', 'SAG']:
        for s in priority_series_candidates:
            desc_words = s['description'].upper().split()
            if orientation in desc_words or orientation in s['description'].upper()[:20]:
                if s not in priority_series:
                    priority_series.append(s)
                    break
    for s in priority_series_candidates:
        if s not in priority_series:
            priority_series.append(s)
            if args.max_series is not None and len(priority_series) >= args.max_series:
                break
    # If still not enough and max_series not reached, continue as before
    if not priority_series:
        # 3. Fallback: any T2-weighted series
        t2_keywords = ['T2', 'TSE', 'HASTE', 'SSFSE', 'FIESTA']
        t2_exclude_keywords = ['LOCAL', 'DTI', 'DWI', 'TRACE', 'ADC', 'FA', 'T1', 'FLAIR', 'MRCP']
        t2_series = [s for s in series_info if any(kw in s['description'].upper() for kw in t2_keywords) and not any(ex in s['description'].upper() for ex in t2_exclude_keywords)]
        for orientation in ['AXIAL', 'AX', 'COR', 'SAG']:
            for s in t2_series:
                if orientation in s['description'].upper() and s not in priority_series:
                    priority_series.append(s)
                    if len(priority_series) >= 4:
                        break
            if len(priority_series) >= 4:
                break
        if len(priority_series) < 3:
            for s in t2_series:
                if s not in priority_series:
                    priority_series.append(s)
                    if len(priority_series) >= 4:
                        break
            # 3. Fallback: any T2-weighted series
            t2_keywords = ['T2', 'TSE', 'HASTE', 'SSFSE', 'FIESTA']
            exclude_keywords = ['LOCAL', 'DTI', 'DWI', 'TRACE', 'ADC', 'FA', 'T1', 'FLAIR', 'MRCP']
            t2_series = [s for s in series_info if any(kw in s['description'].upper() for kw in t2_keywords) and not any(ex in s['description'].upper() for ex in exclude_keywords)]
            for orientation in ['AXIAL', 'AX', 'COR', 'SAG']:
                for s in t2_series:
                    if orientation in s['description'].upper() and s not in priority_series:
                        priority_series.append(s)
                        if len(priority_series) >= 4:
                            break
                if len(priority_series) >= 4:
                    break
            if len(priority_series) < 3:
                for s in t2_series:
                    if s not in priority_series:
                        priority_series.append(s)
                        if len(priority_series) >= 4:
                            break

    # Apply max-series limit if set and not already applied
    if args.max_series is not None:
        priority_series = priority_series[:args.max_series]

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

    if args.batch_size_seg is not None:
        svr_cmd.extend(['--batch-size-seg', str(args.batch_size_seg)])
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
