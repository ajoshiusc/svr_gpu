import argparse
import os
from typing import Tuple

import nibabel as nib
import numpy as np
from skimage import io as skio


def _normalize_uint8(img: np.ndarray, clip_percentiles: Tuple[float, float] = (1.0, 99.0)) -> np.ndarray:
    """
    Normalize a numpy array image to uint8 for visualization.

    - Applies percentile clipping to reduce outlier influence.
    - Scales to [0, 255] and converts to uint8.
    """
    img = np.asarray(img)
    if img.size == 0:
        return np.zeros_like(img, dtype=np.uint8)

    # Handle NaNs/Infs gracefully
    img = np.nan_to_num(img, copy=False)

    lo, hi = np.percentile(img, clip_percentiles)
    if hi <= lo:
        lo, hi = float(np.min(img)), float(np.max(img))
        if hi <= lo:
            return np.zeros_like(img, dtype=np.uint8)

    img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    img = (img * 255.0 + 0.5).astype(np.uint8)
    return img


def save_mip(volume: np.ndarray, axis: int, out_path: str) -> None:
    """Save maximum intensity projection along given axis as PNG."""
    mip = np.max(volume, axis=axis)
    mip_u8 = _normalize_uint8(mip)
    skio.imsave(out_path, mip_u8)


def save_center_slice(volume: np.ndarray, axis: int, out_path: str) -> None:
    """Save the central slice along given axis as PNG."""
    idx = volume.shape[axis] // 2
    if axis == 0:
        sl = volume[idx, :, :]
    elif axis == 1:
        sl = volume[:, idx, :]
    else:
        sl = volume[:, :, idx]
    sl_u8 = _normalize_uint8(sl)
    skio.imsave(out_path, sl_u8)


def main():
    parser = argparse.ArgumentParser(description="Generate quick QC images (MIPs and center slices) for a NIfTI volume.")
    parser.add_argument("--input", required=True, help="Path to input NIfTI file (e.g., svr_output.nii.gz)")
    parser.add_argument("--out-dir", required=True, help="Directory to save QC PNGs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load NIfTI
    img = nib.load(args.input)
    vol = img.get_fdata(dtype=np.float32)

    # Ensure volume is 3D
    if vol.ndim > 3:
        # Collapse channels or time if present via max projection
        vol = np.max(vol, axis=tuple(range(3, vol.ndim))) if vol.ndim > 3 else vol
    if vol.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {vol.shape}")

    # Save MIPs
    save_mip(vol, axis=0, out_path=os.path.join(args.out_dir, "mip_axial.png"))
    save_mip(vol, axis=1, out_path=os.path.join(args.out_dir, "mip_coronal.png"))
    save_mip(vol, axis=2, out_path=os.path.join(args.out_dir, "mip_sagittal.png"))

    # Save center slices
    save_center_slice(vol, axis=0, out_path=os.path.join(args.out_dir, "center_axial.png"))
    save_center_slice(vol, axis=1, out_path=os.path.join(args.out_dir, "center_coronal.png"))
    save_center_slice(vol, axis=2, out_path=os.path.join(args.out_dir, "center_sagittal.png"))

    print(f"QC images saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
