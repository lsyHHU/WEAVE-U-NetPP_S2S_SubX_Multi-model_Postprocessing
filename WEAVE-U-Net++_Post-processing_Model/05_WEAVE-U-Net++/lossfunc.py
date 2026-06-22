
#%%
import sys
import numpy as np
from tqdm import tqdm
import copy
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import Dataset

sys.path.append('/home/liusy/MachineLearning_CSGD/S2S_SubX_Multimodel_250525')
from Gamma_CDF_aux_copy import GammaCDF


class LossCRPS_CSGD_ngrid(nn.Module):
    def __init__(self, n_grid_per_batch=1024):
        """
        n_grid_per_batch: 每次计算时使用的网格数
        """
        super(LossCRPS_CSGD_ngrid, self).__init__()
        self.n_grid_per_batch = n_grid_per_batch

    def forward(self, obs, predict_shift, predict_mu, predict_sigma, mask=None):
        """
        obs: (b, H, W)的图片
        predict_shift, predict_mu, predict_sigma: (b, H, W)的图片
        mask: (H, W) or (1, H, W) or (B, H, W) 的 numpy array 或 tensor，值为 0 或 1，只计算中国陆地区域
        """
        bs = obs.size(0)  # batch size
        H, W = obs.size(1), obs.size(2)
        ngrid = H * W 

        # 处理mask
        if mask is None:
            mask = torch.ones((1, H, W), dtype=torch.float32, device=obs.device)
        if isinstance(mask, np.ndarray):
            mask = torch.tensor(mask, dtype=torch.float32, device=obs.device)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # (H, W) -> (1, H, W)
        if mask.size(0) == 1:
            mask = mask.expand(bs, -1, -1)  # (1, H, W) -> (b, H, W)
        
        mask = mask.view(bs, -1)  # -> (b, H*W)

        # 处理网络输出的分布参数（二维）
        shift = predict_shift - 1e-6  # negative value
        mu = predict_mu
        sigma = predict_sigma

        # shape 和 scale 计算
        shape = (mu / (sigma + 1e-6)) ** 2
        scale = (sigma ** 2) / (mu + 1e-6)

        # reshape
        obs = obs.view(bs, -1)
        shift = shift.view(bs, -1)
        shape = shape.view(bs, -1)
        scale = scale.view(bs, -1)

        # 将网格分为多个批次，每批次计算 n_grid_per_batch 个网格的 CRPS
        crps_batch_losses = []
        valid_counts = 0  # 有效网格数量统计

        for i in range(0, ngrid, self.n_grid_per_batch):
            # 取每次计算的网格批次
            end_idx = min(i + self.n_grid_per_batch, ngrid)
            obs_batch = obs[:, i:end_idx]
            shift_batch = shift[:, i:end_idx]
            shape_batch = shape[:, i:end_idx]
            scale_batch = scale[:, i:end_idx]
            mask_batch = mask[:, i:end_idx]

            # 展平
            obs_flat = obs_batch.flatten()
            shift_flat = shift_batch.flatten()
            shape_flat = shape_batch.flatten()
            scale_flat = scale_batch.flatten()
            mask_flat = mask_batch.flatten()

            # 只保留mask==1的网格
            valid_idx = (mask_flat > 0.5)
            if valid_idx.sum() == 0:
                continue  # 当前批次没有有效网格

            obs_valid = obs_flat[valid_idx]
            shift_valid = shift_flat[valid_idx]
            shape_valid = shape_flat[valid_idx]
            scale_valid = scale_flat[valid_idx]
            
            # 计算 Beta 函数项
            betaf = torch.exp(torch.lgamma(torch.tensor(0.5, device=obs.device)) + 
                              torch.lgamma(shape_valid + 0.5) - 
                              torch.lgamma(shape_valid + 1.0))
            
            # Gamma CDF
            Fyk = GammaCDF.apply(obs_valid - shift_valid, shape_valid, scale_valid)
            Fck = GammaCDF.apply(-shift_valid, shape_valid, scale_valid)
            FykP1 = GammaCDF.apply(obs_valid - shift_valid, shape_valid + 1.0, scale_valid)
            FckP1 = GammaCDF.apply(-shift_valid, shape_valid + 1.0, scale_valid)
            F2c2k = GammaCDF.apply(2 * (-shift_valid), shape_valid * 2, scale_valid)

            crps_scaled = (obs_valid - shift_valid) * (2.0 * Fyk - 1.0) + \
                        shift_valid * (Fck ** 2) + \
                        shape_valid * scale_valid * (1.0 + 2.0 * Fck * FckP1 - Fck ** 2 - 2.0 * FykP1) - \
                        (shape_valid * scale_valid / 3.141592653) * betaf * (1.0 - F2c2k)

            crps_batch_losses.append(crps_scaled.sum())
            valid_counts += valid_idx.sum().item()

        if valid_counts == 0:
            return torch.tensor(0.0, device=obs.device)

        total_loss = torch.stack(crps_batch_losses).sum() / valid_counts
        return total_loss
    

######################################################
######################################################
######################################################

def compute_oracle_weights(err, temperatue=1.0):
    """
    构造 oracle 权重（不进行 detach）。
    err: (B, M, H, W) - 每个模型的降水预报与观测之间计算的crps
    temperature: scalar >0
    返回:
      oracle_w: (B, M, H, W)  每像素归一化的概率分布
      oracle_logits: (B, M, H, W)  softmax 前 logits = -err/temperature
    """
    EPS = 1e-6
    logits = -err / (temperatue + EPS)   # smaller error, larger logits
    # softmax over model dim (dim=1)
    logits_flat = logits.view(logits.shape[0], logits.shape[1], -1)  # (B, M, H*W)
    oracle_w_flat = F.softmax(logits_flat, dim=1)  # (B, M, H*W)
    oracle_w = oracle_w_flat.view_as(logits)  # (B, M, H, W)
    return oracle_w, logits
    

def kl_divergence_oracle_w(oracle_w, pred_w, mask, reduction='mean', eps=1e-8, clamp_negative=True):
    """
    Robust KL(oracle || pred) over masked pixels only.

    Args:
      oracle_w: (B, M, H, W) - may contain NaN
      pred_w:   (B, M, H, W) - predicted probabilities (ideally softmax outputs)
      mask:     (H, W) or (B, H, W) with 0/1 indicating valid pixels
      reduction: 'mean'|'sum'|'none'
      eps: numeric epsilon for stability
      clamp_negative: if True, clamp tiny negative kl values to 0

    Returns:
      scalar (if reduction in {'mean','sum'}) or (B, H, W) tensor if reduction == 'none'
    """
    if oracle_w.shape != pred_w.shape:
        raise ValueError("oracle_w and pred_w must have same shape (B, M, H, W)")
    B, M, H, W = oracle_w.shape

    # prepare mask -> (B, H, W) float
    mask_t = torch.as_tensor(mask, device=oracle_w.device)
    if mask_t.dim() == 2:
        mask_b = mask_t.unsqueeze(0).expand(B, -1, -1).to(dtype=oracle_w.dtype)
    elif mask_t.dim() == 3:
        if mask_t.shape[0] != B:
            raise ValueError("mask batch dim != B")
        mask_b = mask_t.to(dtype=oracle_w.dtype)
    else:
        raise ValueError("mask must be (H,W) or (B,H,W)")

    # 1) replace NaN in oracle_w with 0 (we'll exclude these pixels later)
    oracle = torch.nan_to_num(oracle_w, nan=0.0, posinf=0.0, neginf=0.0)  # (B,M,H,W)

    # 2) If a pixel has all-zero oracle (e.g., was all NaN), mark mask as invalid for that pixel
    sum_oracle = oracle.sum(dim=1)  # (B,H,W)
    valid_oracle = (sum_oracle > 0).to(oracle.dtype)  # 1 if there is some oracle mass
    mask_b = mask_b * valid_oracle  # drop pixels where oracle had no mass

    # 3) Normalize oracle over model dim, only where sum_oracle>0
    denom_oracle = sum_oracle.unsqueeze(1) + eps  # (B,1,H,W)
    oracle_norm = oracle / denom_oracle  # safe division (pixels with denom near 0 get near-zero oracle_norm)

    # 4) Prepare pred: replace NaN with small eps and normalize over model dim
    pred = torch.nan_to_num(pred_w, nan=0.0, posinf=0.0, neginf=0.0)
    sum_pred = pred.sum(dim=1, keepdim=True)  # (B,1,H,W)
    # avoid division by zero: add eps
    pred_norm = pred / (sum_pred + eps)

    # 5) clamp lower bound for log stability
    oracle_clamped = oracle_norm.clamp(min=eps)
    pred_clamped = pred_norm.clamp(min=eps)

    # 6) KL per-pixel: sum over model dim -> (B, H, W)
    kl_per_pixel = (oracle_clamped * (torch.log(oracle_clamped) - torch.log(pred_clamped))).sum(dim=1)

    # 7) optionally clamp tiny negative numerical noise to 0
    if clamp_negative:
        kl_per_pixel = torch.clamp(kl_per_pixel, min=0.0)

    # 8) apply mask
    kl_masked = kl_per_pixel * mask_b  # (B,H,W)

    if reduction == 'none':
        return kl_masked

    total_mask = mask_b.sum()  # scalar

    if reduction == 'sum':
        return kl_masked.sum()

    if reduction == 'mean':
        if total_mask.item() == 0:
            # no valid pixel -> return zero scalar
            return torch.tensor(0.0, dtype=kl_masked.dtype, device=kl_masked.device)
        return kl_masked.sum() / total_mask

    raise ValueError("reduction must be 'mean'|'sum'|'none'")



# crps_arr = np.load('/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/00_Results/crps_by_model.npy')[:10, ...]
# crps_arr_t = torch.tensor(crps_arr)
# oracle_w, logits = compute_oracle_weights(err=crps_arr_t, temperatue=1.0)

# oracle_w.detach()

# pred_w = torch.rand((10, 18, 48, 80))
# mask = np.load('/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/CN_land_coords/land_mask.npy')
# mask_t = torch.tensor(mask, dtype=torch.float32)
# kl_loss = kl_divergence_oracle_w(oracle_w, pred_w, mask_t)


# %%
