from __future__ import annotations

import logging
import os
from argparse import Namespace
from typing import Any, Dict, List, Tuple, Optional

import torch

from .image import Stack, Slice
from .transform import RigidTransform
from .assessment import assess as assess_stacks
from .preprocessing.masking.brain_segmentation import brain_segmentation
from .preprocessing.masking.thresholding import otsu_thresholding, thresholding
from .preprocessing.bias_field import n4_bias_field_correction
from .svort.inference import svort_predict

logger = logging.getLogger(__name__)

_ORIENTATION_LABELS = ("sagittal", "coronal", "axial")
_ORIENTATION_PRIORITY = {0: 0, 2: 1, 1: 2}


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
                    stack_vol = s.get_volume()
                    mask_vol = s.get_mask_volume()
                    mask_path = os.path.join(masks_dir, f"stack_{i}_mask.nii.gz")
                    masked_stack_path = os.path.join(
                        masks_dir, f"stack_{i}_masked.nii.gz"
                    )
                    mask_vol.save(mask_path)
                    stack_vol.image = stack_vol.image * mask_vol.image.to(
                        stack_vol.image.device
                    )
                    stack_vol.mask = mask_vol.mask.clone()
                    stack_vol.save(masked_stack_path)
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


def _stack_slice_normal(stack: Stack) -> torch.Tensor:
    """Return the physical slice-normal direction for a stack."""
    matrices = stack.transformation.matrix()
    if matrices.numel() == 0:
        return torch.tensor((0.0, 0.0, 1.0), dtype=torch.float32)
    normal = matrices[0, :, 2].detach().cpu().to(torch.float32)
    norm = torch.linalg.vector_norm(normal).item()
    if norm <= 1e-6:
        return torch.tensor((0.0, 0.0, 1.0), dtype=torch.float32)
    return normal / norm


def _orientation_axis(normal: torch.Tensor) -> int:
    return int(torch.argmax(torch.abs(normal)).item())


def _stack_display_name(stack: Stack, fallback_idx: int) -> str:
    name = getattr(stack, "name", None)
    if not name:
        return f"stack_{fallback_idx}"
    return os.path.basename(str(name))


def _registration_order(stacks: List[Stack]) -> Tuple[List[int], List[Tuple[str, str, Tuple[float, float, float]]]]:
    """Choose a deterministic, geometry-aware stack order for registration."""
    count = len(stacks)
    if count <= 1:
        normal = (0.0, 0.0, 1.0)
        summary = [(_stack_display_name(stacks[0], 0), "axial", normal)] if stacks else []
        return list(range(count)), summary

    normals = [_stack_slice_normal(stack) for stack in stacks]
    axes = [_orientation_axis(normal) for normal in normals]
    axis_counts = {axis: axes.count(axis) for axis in range(3)}

    best_pair: Optional[Tuple[int, int]] = None
    best_pair_key: Optional[Tuple[float, int, int, int, int, int]] = None
    for i in range(count):
        for j in range(i + 1, count):
            dot = abs(float(torch.dot(normals[i], normals[j])))
            coverage = axis_counts[axes[i]]
            if axes[j] != axes[i]:
                coverage += axis_counts[axes[j]]
            pair_key = (
                round(dot, 6),
                -coverage,
                min(_ORIENTATION_PRIORITY[axes[i]], _ORIENTATION_PRIORITY[axes[j]]),
                max(_ORIENTATION_PRIORITY[axes[i]], _ORIENTATION_PRIORITY[axes[j]]),
                i,
                j,
            )
            if best_pair_key is None or pair_key < best_pair_key:
                best_pair_key = pair_key
                best_pair = (i, j)

    assert best_pair is not None
    order = sorted(
        best_pair,
        key=lambda idx: (
            axis_counts[axes[idx]],
            _ORIENTATION_PRIORITY[axes[idx]],
            idx,
        ),
    )

    remaining = [idx for idx in range(count) if idx not in order]
    while remaining:
        next_idx = min(
            remaining,
            key=lambda idx: (
                max(abs(float(torch.dot(normals[idx], normals[chosen]))) for chosen in order),
                axis_counts[axes[idx]],
                _ORIENTATION_PRIORITY[axes[idx]],
                idx,
            ),
        )
        order.append(next_idx)
        remaining.remove(next_idx)

    summary = []
    for idx in order:
        normal = tuple(round(float(v), 3) for v in normals[idx].tolist())
        summary.append(
            (
                _stack_display_name(stacks[idx], idx),
                _ORIENTATION_LABELS[axes[idx]],
                normal,
            )
        )
    return order, summary


def _order_stacks_for_registration(stacks: List[Stack]) -> List[Stack]:
    """Promote orthogonal stack anchors before SVoRT / stack registration."""
    order, summary = _registration_order(stacks)
    if not summary:
        return stacks

    original_names = [_stack_display_name(stack, i) for i, stack in enumerate(stacks)]
    ordered_names = [original_names[i] for i in order]
    if order != list(range(len(stacks))):
        logger.info(
            "Geometry-aware registration reorder: %s -> %s",
            " -> ".join(original_names),
            " -> ".join(ordered_names),
        )
    else:
        logger.info("Input stack order already suitable for geometry-aware registration anchors")

    for rank, (name, orientation, normal) in enumerate(summary, start=1):
        logger.info(
            "  registration stack %d: %s [%s normal=(%.3f, %.3f, %.3f)]",
            rank,
            name,
            orientation,
            normal[0],
            normal[1],
            normal[2],
        )

    return [stacks[idx] for idx in order]


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

    registration_stacks = stacks
    if svort or vvr:
        registration_stacks = _order_stacks_for_registration(stacks)

    force_scanner = args.scanner_space
    slices = svort_predict(
        registration_stacks,
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
