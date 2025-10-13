from typing import Optional, Sequence, cast
import logging
import torch
import torch.nn.functional as F

from ..image import Volume, Slice
from ..transform import mat_transform_points

BATCH_SIZE = 64


def _construct_coef(
    idxs,
    transforms,
    vol_shape,
    slice_shape,
    vol_mask,
    slice_mask,
    psf,
    res_slice,
):
    slice_ids = []
    volume_ids = []
    psf_vs = []
    for i in range(len(idxs)):
        slice_id, volume_id, psf_v = _construct_slice_coef(
            i,
            transforms[idxs[i]],
            vol_shape,
            slice_shape,
            vol_mask,
            slice_mask[idxs[i]] if slice_mask is not None else None,
            psf,
            res_slice,
        )
        slice_ids.append(slice_id)
        volume_ids.append(volume_id)
        psf_vs.append(psf_v)

    slice_id = torch.cat(slice_ids)
    del slice_ids
    volume_id = torch.cat(volume_ids)
    del volume_ids
    ids = torch.stack((slice_id, volume_id), 0)
    del slice_id, volume_id
    psf_v = torch.cat(psf_vs)
    del psf_vs
    coef = torch.sparse_coo_tensor(
        ids,
        psf_v,
        [
            slice_shape[0] * slice_shape[1] * len(idxs),
            vol_shape[0] * vol_shape[1] * vol_shape[2],
        ],
    ).coalesce()
    return coef


def _construct_slice_coef(
    i,
    transform,
    vol_shape,
    slice_shape,
    vol_mask,
    slice_mask,
    psf,
    res_slice,
):
    transform = transform[None]
    psf_volume = Volume(psf, psf > 0, resolution_x=1)
    psf_xyz = psf_volume.xyz_masked_untransformed
    psf_v = psf_volume.v_masked
    if slice_mask is not None:
        _slice = slice_mask
    else:
        _slice = torch.ones((1,) + slice_shape, dtype=torch.bool, device=psf.device)

    slice_xyz = Slice(_slice, _slice, resolution_x=res_slice).xyz_masked_untransformed
    slice_xyz = mat_transform_points(transform, slice_xyz, trans_first=True)
    psf_xyz = mat_transform_points(
        transform, psf_xyz - transform[:, :, -1], trans_first=True
    )
    shift_xyz = (
        torch.tensor(vol_shape[::-1], dtype=psf.dtype, device=psf.device) - 1
    ) / 2.0
    slice_xyz = shift_xyz + psf_xyz.reshape((1, -1, 3)) + slice_xyz.reshape((-1, 1, 3))
    inside_mask = torch.all((slice_xyz > 0) & (slice_xyz < (shift_xyz * 2)), -1)
    slice_xyz = slice_xyz[inside_mask].round().long()
    slice_id = torch.arange(
        i * slice_shape[0] * slice_shape[1],
        (i + 1) * slice_shape[0] * slice_shape[1],
        dtype=torch.long,
        device=psf.device,
    )
    if slice_mask is not None:
        slice_id = slice_id.view_as(slice_mask)[slice_mask]
    slice_id = slice_id[..., None].expand(-1, psf_v.shape[0])[inside_mask]
    psf_v = psf_v[None].expand(inside_mask.shape[0], -1)[inside_mask]
    volume_id = (
        slice_xyz[:, 0]
        + slice_xyz[:, 1] * vol_shape[2]
        + slice_xyz[:, 2] * (vol_shape[1] * vol_shape[2])
    )
    return slice_id, volume_id, psf_v


def slice_acquisition_torch(
    transforms: torch.Tensor,
    vol: torch.Tensor,
    vol_mask: Optional[torch.Tensor],
    slices_mask: Optional[torch.Tensor],
    psf: torch.Tensor,
    slice_shape: Sequence,
    res_slice: float,
    need_weight: bool,
):
    slice_shape = tuple(slice_shape)
    global BATCH_SIZE
    if psf.numel() == 1 and not need_weight:
        return slice_acquisition_no_psf_torch(
            transforms, vol, vol_mask, slices_mask, slice_shape, res_slice
        )
    if vol_mask is not None:
        vol = vol * vol_mask
    vol_shape = vol.shape[-3:]
    _slices = []
    _weights = []
    i = 0
    while i < transforms.shape[0]:
        succ = False
        try:
            # Build sparse coefficients in full precision for numerical stability
            coef = _construct_coef(
                list(range(i, min(i + BATCH_SIZE, transforms.shape[0]))),
                transforms,
                vol_shape,
                slice_shape,
                vol_mask,
                slices_mask,
                psf,
                res_slice,
            )
            s = torch.mv(coef, vol.view(-1)).to_dense().reshape((-1, 1) + slice_shape)
            weight = torch.sparse.sum(coef, 1).to_dense().reshape_as(s)
            del coef
            succ = True
        except RuntimeError as e:
            if "out of memory" in str(e) and BATCH_SIZE > 0:
                logging.debug("slice_acquisition_torch: OOM, reducing batch size")
                BATCH_SIZE = BATCH_SIZE // 2
                torch.cuda.empty_cache()
            else:
                raise
        if succ:
            _slices.append(s)
            _weights.append(weight)
            i += BATCH_SIZE
            torch.cuda.empty_cache()

    slices = torch.cat(_slices)
    weights = torch.cat(_weights)
    m = weights > 1e-2
    slices[m] = slices[m] / weights[m]
    if slices_mask is not None:
        slices = slices * slices_mask
    if need_weight:
        return slices, weights
    return slices


def slice_acquisition_adjoint_torch(
    transforms: torch.Tensor,
    psf: torch.Tensor,
    slices: torch.Tensor,
    slices_mask: Optional[torch.Tensor],
    vol_mask: Optional[torch.Tensor],
    vol_shape: Sequence,
    res_slice: float,
    equalize: bool,
):
    vol_shape = tuple(vol_shape)
    global BATCH_SIZE
    if slices_mask is not None:
        slices = slices * slices_mask
    vol = None
    weight = None
    slice_shape = slices.shape[-2:]
    i = 0
    while i < transforms.shape[0]:
        succ = False
        try:
            # Build sparse coefficients in full precision for numerical stability
            coef = _construct_coef(
                list(range(i, min(i + BATCH_SIZE, transforms.shape[0]))),
                transforms,
                vol_shape,
                slice_shape,
                vol_mask,
                slices_mask,
                psf,
                res_slice,
            ).t()
            v = torch.mv(coef, slices[i : i + BATCH_SIZE].view(-1))
            if equalize:
                w = torch.sparse.sum(coef, 1)
            del coef
            succ = True
        except RuntimeError as e:
            if "out of memory" in str(e) and BATCH_SIZE > 0:
                logging.debug("slice_acquisition_adjoint_torch: OOM, reducing batch size")
                BATCH_SIZE = BATCH_SIZE // 2
                torch.cuda.empty_cache()
            else:
                raise
        if succ:
            if vol is None:
                vol = v
            else:
                vol += v
            if equalize:
                if weight is None:
                    weight = w
                else:
                    weight += w
            i += BATCH_SIZE
            torch.cuda.empty_cache()
    vol = cast(torch.Tensor, vol)
    vol = vol.to_dense().reshape((1, 1) + vol_shape)
    if equalize:
        weight = cast(torch.Tensor, weight)
        weight = weight.to_dense().reshape_as(vol)
        m = weight > 1e-2
        vol[m] = vol[m] / weight[m]
    if vol_mask is not None:
        vol = vol * vol_mask
    return vol


def slice_acquisition_no_psf_torch(
    transforms: torch.Tensor,
    vol: torch.Tensor,
    vol_mask: Optional[torch.Tensor],
    slices_mask: Optional[torch.Tensor],
    slice_shape: Sequence,
    res_slice: float,
) -> torch.Tensor:
    slice_shape = tuple(slice_shape)
    device = transforms.device
    _slice = torch.ones((1,) + slice_shape, dtype=torch.bool, device=device)
    slice_xyz = Slice(_slice, _slice, resolution_x=res_slice).xyz_masked_untransformed
    
    # Perform transforms and interpolation in full precision
    slice_xyz = mat_transform_points(
        transforms[:, None], slice_xyz[None], trans_first=True
    ).view((transforms.shape[0], 1) + slice_shape + (3,))

    output_slices = torch.zeros_like(slice_xyz[..., 0])

    if slices_mask is not None:
        masked_xyz = slice_xyz[slices_mask]
    else:
        masked_xyz = slice_xyz

    masked_xyz = masked_xyz / (
        (torch.tensor(vol.shape[-3:][::-1], dtype=masked_xyz.dtype, device=device) - 1)
        / 2
    )
    if vol_mask is not None:
        vol = vol * vol_mask
    masked_v = F.grid_sample(vol, masked_xyz.view(1, 1, 1, -1, 3), align_corners=True)
    
    if slices_mask is not None:
        output_slices[slices_mask] = masked_v.to(output_slices.dtype)
    else:
        output_slices = masked_v.reshape((transforms.shape[0], 1) + slice_shape)
    return output_slices
