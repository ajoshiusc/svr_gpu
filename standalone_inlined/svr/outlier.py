import torch
import torch.nn as nn
import logging
from typing import Optional, Union, Tuple, cast
from ..image import Stack, Volume
from .reconstruction import simulate_slices
from ..loss import ncc_loss, ssim_loss


class EM(nn.Module):
    def __init__(
        self,
        max_intensity: float,
        min_intensity: float,
        c_voxel: float = 0.9,
        c_slice: float = 0.9,
    ) -> None:
        super().__init__()
        self.c_voxel_init: Union[float, torch.Tensor] = c_voxel
        self.c_voxel: Union[float, torch.Tensor] = c_voxel
        self.c_slice: Union[float, torch.Tensor] = c_slice
        self.sigma_voxel: Union[float, torch.Tensor] = -1
        self.m: float = -1
        self.max_intensity = max_intensity
        self.min_intensity = min_intensity
        self.p_voxel: Optional[torch.Tensor] = None
        self.p_slice: Optional[torch.Tensor] = None

    def forward(
        self,
        e: Stack,
        weight: Optional[Union[Stack, torch.Tensor]] = None,
        scale: Optional[torch.Tensor] = None,
        n_iter: int = 3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # get inputs
        err = e.slices
        slices_mask = e.mask
        if weight is not None:
            if isinstance(weight, Stack):
                weight = weight.slices
        else:
            weight = torch.ones_like(err)

        mask_voxel_low = weight > 0
        mask_voxel_high = weight > 0.99

        # init
        if self.sigma_voxel < 0:
            self.sigma_voxel = err[mask_voxel_high].std()
        if self.m < 0:
            self.m = 1 / (2 * (self.max_intensity - self.min_intensity))
        # fit mixture model for voxel error
        for _ in range(n_iter):
            _err = err[torch.logical_and(slices_mask, mask_voxel_high)]
            _p = voxel_outlier_update(_err, self.sigma_voxel, self.m, self.c_voxel)
            self.c_voxel = _p.mean()
            if self.c_voxel < 0.1:
                logging.warning(
                    "The proportion of outlier voxel is too high: %f, reset to %f.",
                    1 - self.c_voxel,
                    1 - self.c_voxel_init,
                )
                self.c_voxel = self.c_voxel_init
            self.sigma_voxel = torch.sqrt(torch.mean(_err * _err * _p) / self.c_voxel)
            if self.sigma_voxel < 0.0001:
                logging.warning(
                    "The std of outlier voxel is too low: %f.", self.sigma_voxel
                )
                self.sigma_voxel = 0.0001
            self.m = 1 / (_err.max() - _err.min())
            logging.debug(
                "c_voxel = %f, sigma_voxel = %f, m = %f",
                self.c_voxel,
                self.sigma_voxel,
                self.m,
            )
        # get voxel weights
        self.p_voxel = torch.zeros_like(err)
        self.p_voxel[mask_voxel_low] = voxel_outlier_update(
            err[mask_voxel_low], self.sigma_voxel, self.m, self.c_voxel
        )

        denom = mask_voxel_high.sum((1, 2, 3))
        potential = ((1 - self.p_voxel) ** 2) * mask_voxel_high
        potential = torch.sqrt(potential.sum((1, 2, 3)) / denom)
        mask_1 = denom > 0
        mask_slice = mask_1
        slices_mask_sum = slices_mask.sum((1, 2, 3))
        mask_2 = slices_mask_sum > slices_mask_sum.median() * 0.1
        mask_slice = mask_slice & mask_2

        if scale is not None:
            mask_3 = (scale > 0.2) & (scale < 5)
            mask_slice = mask_slice & mask_3

        if self.p_slice is None:
            self.p_slice = torch.ones_like(mask_slice, dtype=torch.float)
        self.p_slice[~mask_slice] = 0
        self.p_slice[torch.logical_and(mask_slice, self.p_slice == 0)] = 1
        if self.p_slice.sum() == 0:
            logging.warning("All slices are excluded, reset")
            self.p_slice = torch.ones_like(mask_slice, dtype=torch.float)
            mask_slice = torch.ones_like(mask_slice)
        for _ in range(n_iter):
            self.c_slice, self.p_slice[mask_slice] = slice_outlier_update(
                potential[mask_slice], self.p_slice[mask_slice], self.c_slice
            )

        return self.p_voxel, self.p_slice


def voxel_outlier_update(
    x: torch.Tensor,
    sigma: Union[float, torch.Tensor],
    m: float,
    c: Union[float, torch.Tensor],
) -> torch.Tensor:
    g = torch.distributions.normal.Normal(0, sigma).log_prob(x).exp()
    gc = g * c
    # Add epsilon to prevent division by zero
    p = gc / (gc + (1 - c) * m + 1e-10)
    # Check for NaN/Inf and replace with ones (assume inlier if uncertain)
    p = torch.where(torch.isfinite(p), p, torch.ones_like(p))
    return p


def slice_outlier_update(
    x: torch.Tensor, p: torch.Tensor, c: Union[float, torch.Tensor]
):
    sum_in = torch.dot(x, p)
    sum_out = torch.dot(x, 1 - p)
    N_in = p.sum()
    N_out = x.numel() - N_in

    # debug before computing means
    if not torch.isfinite(sum_in) or not torch.isfinite(sum_out):
        # Use print to guarantee visibility during debugging (temporary)
        print(
            "DEBUG slice_outlier_update early:",
            sum_in.item() if isinstance(sum_in, torch.Tensor) else sum_in,
            sum_out.item() if isinstance(sum_out, torch.Tensor) else sum_out,
            N_in.item() if isinstance(N_in, torch.Tensor) else N_in,
            N_out,
            p.nonzero().size(0),
            x.min().item(),
            x.max().item(),
            x.mean().item(),
        )
        logging.debug(
            "slice_outlier_update early debug: sum_in=%s, sum_out=%s, N_in=%s, N_out=%s, p_nonzero=%s, x_stats=(min=%s,max=%s,mean=%s)",
            sum_in.item() if isinstance(sum_in, torch.Tensor) else sum_in,
            sum_out.item() if isinstance(sum_out, torch.Tensor) else sum_out,
            N_in.item() if isinstance(N_in, torch.Tensor) else N_in,
            N_out,
            p.nonzero().size(0),
            x.min().item(),
            x.max().item(),
            x.mean().item(),
            )
    mu_in = sum_in / (N_in + 1e-10) if N_in > 0 else x.min()
    mu_out = sum_out / (N_out + 1e-10) if N_out > 0 else (x.max() + mu_in) / 2
    
    # Check for NaN/Inf
    # Ensure mu_in and mu_out are tensors for consistency
    if not isinstance(mu_in, torch.Tensor):
        mu_in = torch.tensor(mu_in, device=x.device, dtype=x.dtype)
    if not isinstance(mu_out, torch.Tensor):
        mu_out = torch.tensor(mu_out, device=x.device, dtype=x.dtype)
    mu_in = mu_in if torch.isfinite(mu_in) else x.mean()
    mu_out = mu_out if torch.isfinite(mu_out) else x.max()
    # Final guard: ensure mu_in and mu_out are finite
    if not torch.isfinite(mu_in) or not torch.isfinite(mu_out):
        logging.debug(
            "slice_outlier_update debug: sum_in=%s, sum_out=%s, N_in=%s, N_out=%s, x_min=%s, x_max=%s, x_mean=%s",
            sum_in.item() if isinstance(sum_in, torch.Tensor) else sum_in,
            sum_out.item() if isinstance(sum_out, torch.Tensor) else sum_out,
            N_in.item() if isinstance(N_in, torch.Tensor) else N_in,
            N_out,
            x.min().item(),
            x.max().item(),
            x.mean().item(),
        )
    if not torch.isfinite(mu_in):
        mu_in = x.mean()
    if not torch.isfinite(mu_out):
        mu_out = x.max()
    # Final guard: ensure mu_in and mu_out are finite
    if not torch.isfinite(mu_in):
        mu_in = x.mean()
    if not torch.isfinite(mu_out):
        mu_out = x.max()

    sum2_in = torch.dot((x - mu_in) ** 2, p)
    sum2_out = torch.dot((x - mu_out) ** 2, p)

    sigma_in: Union[float, torch.Tensor] = 0
    if sum2_in > 0 and N_in > 0:
        sigma_in = torch.sqrt(sum2_in / (N_in + 1e-10))
    else:
        sigma_in = torch.tensor(0.025, device=x.device, dtype=x.dtype)
    # Ensure sigma_in is a tensor before using torch.isfinite
    if not isinstance(sigma_in, torch.Tensor):
        sigma_in = torch.tensor(sigma_in, device=x.device, dtype=x.dtype)
    if sigma_in < 0.0001 or not torch.isfinite(sigma_in):
        sigma_in = torch.tensor(0.0001, device=x.device, dtype=x.dtype)

    sigma_out: Union[float, torch.Tensor] = 0
    if sum2_out > 0 and N_out > 0:
        sigma_out = torch.sqrt(sum2_out / (N_out + 1e-10))
    else:
        sigma_out = (mu_out - mu_in) ** 2 / 4
    # Ensure sigma_out is a tensor and finite
    if not isinstance(sigma_out, torch.Tensor):
        sigma_out = torch.tensor(sigma_out, device=x.device, dtype=x.dtype)
    if sigma_out < 0.0001 or not torch.isfinite(sigma_out):
        sigma_out = torch.tensor(0.0001, device=x.device, dtype=x.dtype)

    if N_in <= 0 or mu_out <= mu_in:
        logging.warning("All slices are classified as outlier, reset")
        p = torch.ones_like(p)
    else:
        g_in = torch.distributions.normal.Normal(mu_in, sigma_in).log_prob(x).exp()
        g_out = torch.distributions.normal.Normal(mu_out, sigma_out).log_prob(x).exp()
        # Add epsilon to prevent division by zero
        p = c * g_in / (c * g_in + (1 - c) * g_out + 1e-10)
        # Check for NaN/Inf
        p = torch.where(torch.isfinite(p), p, torch.ones_like(p))
        mask_p = p > 0
        p[~mask_p] = 1
        p[torch.logical_and(x > mu_out, ~mask_p)] = 0

    c = p.mean()
    logging.debug("N_in = %d, mu_in = %f, mu_out = %f, c=%f", N_in, mu_in, mu_out, c)

    return c, p


def global_ncc_exclusion(stack: Stack, volume: Volume, threshold) -> torch.Tensor:
    ncc = -ncc_loss(
        cast(Stack, simulate_slices(stack, volume, True, True)[0]).slices,
        stack.slices,
        stack.mask,
        win=None,
        reduction="none",
    )
    excluded = ncc < threshold
    num_excluded = torch.count_nonzero(excluded).item()
    if num_excluded == excluded.shape[0]:
        logging.warning("All slices excluded according to global NCC. Reset.")
        excluded = torch.zeros_like(excluded)
        num_excluded = 0
    logging.info(
        "global structural exlusion: mean ncc = %f, num_excluded = %d, mean ncc after exclusion = %f",
        ncc.mean().item(),
        num_excluded,
        ncc[~excluded].mean().item(),
    )
    return excluded


def local_ssim_exclusion(
    stack: Stack, slices_sim: Union[Stack, torch.Tensor], threshold: float
) -> torch.Tensor:
    if isinstance(slices_sim, Stack):
        slices_sim = slices_sim.slices
    ssim_map = -ssim_loss(slices_sim, stack.slices, stack.mask)
    logging.info(
        "local structural exlusion: mean ssim = %f, ratio downweighted = %f",
        ssim_map[stack.mask].mean().item(),
        (ssim_map[stack.mask] <= threshold).float().mean().item(),
    )
    return torch.where(ssim_map > threshold, 1.0, 0.1)
