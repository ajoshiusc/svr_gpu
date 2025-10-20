from __future__ import annotations

import logging
import os
from argparse import Namespace
from typing import Any, Dict, List, Tuple, Optional

from .image import Stack, Slice
from .transform import RigidTransform
from .assessment import assess as assess_stacks
from .preprocessing.masking.brain_segmentation import brain_segmentation
from .preprocessing.masking.thresholding import otsu_thresholding, thresholding
from .preprocessing.bias_field import n4_bias_field_correction
from .svort.inference import svort_predict

logger = logging.getLogger(__name__)


def _segment_stack(args: Namespace, stacks: List[Stack]) -> List[Stack]:
    """Apply segmentation to each stack according to the selected method.

    Supported methods:
      - 'twai' (default): MONAI/DynUNet-based brain segmentation
      - 'threshold': Simple intensity thresholding (uses --segmentation-threshold or --background-threshold)
      - 'otsu': Otsu multi-level thresholding
    """
    seg_method = str(args.segmentation).lower() if args.segmentation is not None else "none"

    if seg_method in ["none", "no", ""]:
        return stacks

    if seg_method in ("twai", "monaifbs"):
        stacks_out = brain_segmentation(
            stacks,
            args.device,
            args.batch_size_seg,
            not args.no_augmentation_seg,
            args.dilation_radius_seg,
            args.threshold_small_seg,
        )
    elif seg_method in ("threshold", "simple"):
        # Use segmentation threshold if provided, otherwise fall back to background threshold
        seg_thresh = getattr(args, "segmentation_threshold", None)
        if seg_thresh is None:
            seg_thresh = args.background_threshold
        stacks_out = thresholding(stacks, seg_thresh)
    elif seg_method in ("otsu", "otsu_threshold"):
        stacks_out = otsu_thresholding(stacks)
    else:
        raise ValueError(f"Unknown segmentation method '{args.segmentation}'")

    # Persist masks to SVR_TEMP_DIR if provided in environment
    temp_dir = os.environ.get("SVR_TEMP_DIR")
    if temp_dir:
        try:
            masks_dir = os.path.join(temp_dir, "masks")
            os.makedirs(masks_dir, exist_ok=True)
            for i, s in enumerate(stacks_out):
                try:
                    mask_vol = s.get_mask_volume()
                    out_path = os.path.join(masks_dir, f"stack_{i}_mask.nii.gz")
                    mask_vol.save(out_path)
                except Exception:
                    logging.debug("Could not save mask for stack %d", i)
            logging.info("Saved segmentation masks to %s", masks_dir)
        except Exception as e:
            logging.warning("Failed to persist segmentation masks: %s", e)

    return stacks_out


def _correct_bias_field(args: Namespace, stacks: List[Stack]) -> List[Stack]:
    """Run N4 bias field correction on the provided stacks."""
    n4_params: Dict[str, Any] = {
        k: getattr(args, k)
        for k in vars(args)
        if k.endswith("_n4")
    }
    return n4_bias_field_correction(stacks, n4_params)


def _register(args: Namespace, stacks: List[Stack]) -> List[Slice]:
    """Register stacks using the configured SVoRT/VVR workflow."""
    registration = args.registration
    
    if registration == "svort":
        svort = True
        vvr = True
        force_vvr = False
    elif registration == "svort-stack":
        svort = True
        vvr = True
        force_vvr = True
    elif registration == "svort-only":
        svort = True
        vvr = False
        force_vvr = False
    elif registration == "stack":
        svort = False
        vvr = True
        force_vvr = False
    elif registration == "none":
        svort = False
        vvr = False
        force_vvr = False
    else:
        raise ValueError("Unknown registration method '%s'" % registration)

    force_scanner = args.scanner_space
    slices = svort_predict(
        stacks,
        args.device,
        args.svort_version,
        svort,
        vvr,
        force_vvr,
        force_scanner,
    )
    return slices


def _assess(
    args: Namespace, stacks: List[Stack], print_results: bool = False
) -> Tuple[List[Stack], List[Dict[str, Any]]]:
    """Assess stack quality and optionally filter according to user settings."""
    filtered_stacks, results = assess_stacks(
        stacks,
        args.metric,
        args.filter_method,
        args.cutoff,
        args.batch_size_assess,
        not args.no_augmentation_assess,
        args.device,
    )

    if results:
        descending = results[0]["descending"]
        arrow = "\u2191" if descending else "\u2193"
        template = "\n%15s %25s %15s %15s %15s"
        header = template % (
            "stack",
            "name",
            f"score ({arrow})",
            "rank",
            "",
        )
        result_log = "stack assessment results (metric = %s):" % args.metric
        result_log += header
        for item in results:
            name = item["name"].replace(".gz", "").replace(".nii", "")
            if len(name) > 20:
                name = "..." + name[-17:]
            score = item["score"]
            if isinstance(score, float):
                score_str = f"{score:1.4f}"
            else:
                score_str = str(score)
            result_log += template % (
                item["input_id"],
                name,
                score_str,
                item["rank"],
                "excluded" if item["excluded"] else "",
            )
        if print_results:
            logger.info(result_log)
        else:
            logger.info(result_log)

    logger.debug(
        "Input stacks after assessment and filtering: %s",
        [s.name for s in filtered_stacks],
    )

    return filtered_stacks, results


__all__ = [
    "_segment_stack",
    "_correct_bias_field",
    "_register",
    "_assess",
]
