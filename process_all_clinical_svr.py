import os
import subprocess
from pathlib import Path
import dicom2nifti
import shutil
import tempfile

def find_dicom_series_dirs(root_dir):
    """Find directories containing .dcm files."""
    series_dirs = set()
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.dcm') or file.startswith('IM-'):
                series_dirs.add(root)
                break
    return sorted(series_dirs, key=lambda p: Path(p).name.upper())

def convert_to_nifti(series_dir, output_nii):
    """Convert DICOM series to NIfTI."""
    try:
        with tempfile.TemporaryDirectory() as td:
            dicom2nifti.convert_directory(str(series_dir), td, compression=True)
            nii_files = list(Path(td).glob("*.nii.gz"))
            if nii_files:
                shutil.move(str(nii_files[0]), str(output_nii))
                return True
    except Exception as e:
        print(f"Failed to convert {series_dir}: {e}")
    return False

def main():
    base_data_dir = Path("/home/ajoshi/project2_ajoshi_27/data/clinical_svr_chla_data")
    dicom_dir = base_data_dir / "clinical_svr_dicom"
    nifti_in_base = base_data_dir / "svr" / "nifti_in"
    nifti_svr_base = base_data_dir / "svr" / "nifti_svr"

    # Create base output directories
    nifti_in_base.mkdir(parents=True, exist_ok=True)
    nifti_svr_base.mkdir(parents=True, exist_ok=True)

    scripts_dir = Path("/home/ajoshi/Projects/svr_gpu")
    svr_cli_path = scripts_dir / "svr_cli.py"

    subjects = [d for d in dicom_dir.iterdir() if d.is_dir() and d.name.upper().startswith("SVR")]
    subjects.sort()
    
    for subj_dir in subjects:
        subj_id = subj_dir.name
        print(f"\nProcessing {subj_id}...")
        
        subj_nifti_in = nifti_in_base / subj_id
        subj_nifti_svr = nifti_svr_base / subj_id
        
        subj_nifti_in.mkdir(parents=True, exist_ok=True)
        subj_nifti_svr.mkdir(parents=True, exist_ok=True)
        
        series_dirs = find_dicom_series_dirs(subj_dir)
        nifti_inputs = []
        
        # Priority filter: Only process T2 Brain SSH/TSE series
        valid_keywords = ['BRAIN', 'TSE', 'SSH', 'HASTE', 'T2']
        exclude_keywords = ['DTI', 'DWI', 'TRAC', 'ADC', 'FA', 'T1', 'MRCP', 'LOCAL', 'SURVEY']

        for sdir in series_dirs:
            series_name = Path(sdir).name.replace(" ", "_")
            series_upper = series_name.upper()
            
            # Simple heuristic to identify valid structural brain stacks
            if any(ex in series_upper for ex in exclude_keywords):
                continue
            if not ("BRAIN" in series_upper):
                continue
            if not ("TSE" in series_upper or "SSH" in series_upper):
                continue
                
            out_nii = subj_nifti_in / f"{series_name}.nii.gz"
            
            if not out_nii.exists():
                print(f"  Converting {series_name} to NIfTI...")
                if convert_to_nifti(sdir, out_nii):
                    nifti_inputs.append(str(out_nii))
            else:
                print(f"  NIfTI already exists for {series_name}")
                nifti_inputs.append(str(out_nii))
                
        # Keep deterministic discovery order here; svr_cli.py now chooses a
        # geometry-aware registration anchor order internally.
        # nifti_inputs = nifti_inputs[:4]
        
        if not nifti_inputs:
            print(f"  No valid NIfTI files found for {subj_id}, skipping.")
            continue

        out_volume = subj_nifti_svr / "output.nii.gz"
        
        import sys
        # Build command to run svr_cli.py
        cmd = [
            sys.executable, str(svr_cli_path),
            "--input-stacks"
        ] + nifti_inputs + [
            "--output", str(out_volume),
            "--segmentation", "twai",
            "--batch-size-seg", "4"
        ]

        print("  Input NIfTI stacks (discovery order; svr_cli.py may reorder for registration):")
        for nii in nifti_inputs:
            print(f"    {Path(nii).name}")
        print(f"  Running svr_cli.py for {subj_id}...")
        try:
            # We don't want to actually wait for all of them to finish synchronously during dry run, 
            # but this script will do it sequentially when run.
            subprocess.run(cmd, check=True, cwd=str(scripts_dir))
            print(f"  Successfully processed {subj_id}.")
        except subprocess.CalledProcessError as e:
            print(f"  Error processing {subj_id}: svr_cli.py failed.")

if __name__ == "__main__":
    main()
