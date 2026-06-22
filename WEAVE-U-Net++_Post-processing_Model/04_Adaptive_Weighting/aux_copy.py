
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


def initialize_weights(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight)  # 对 Linear 层进行 Xavier 初始化
            if module.bias is not None:
                init.zeros_(module.bias)  # 偏置项初始化为零
        elif isinstance(module, nn.Conv2d):
            init.xavier_normal_(module.weight)  # 对 Conv2d 层进行 Xavier 正态初始化
            if module.bias is not None:
                init.zeros_(module.bias)  # 偏置项初始化为零
                

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
    

class CustomDataset(Dataset):
    def __init__(self, fcst_images, geo_images, labels, leads, pr_ensmean, pr_ensstd):
        self.fcst_images = fcst_images
        self.geo_images = geo_images
        self.labels = labels
        self.leads = leads
        self.pr_ensmean = pr_ensmean
        self.pr_ensstd = pr_ensstd
    
    def __len__(self):
        return len(self.fcst_images)
    
    def __getitem__(self, idx):
        return self.fcst_images[idx], self.geo_images[idx], self.labels[idx], self.leads[idx], \
                self.pr_ensmean[idx], self.pr_ensstd[idx]


def compute_channel_min_max(dataset):
    fcst_images = []
    pr_ensmean_images = []
    pr_ensstd_images = []
    for fcst_image, _, _, _, pr_ensmean_image, pr_ensstd_image in dataset:
        fcst_images.append(fcst_image)
        pr_ensmean_images.append(pr_ensmean_image)
        pr_ensstd_images.append(pr_ensstd_image)
    
    # 将所有图像拼接在一起，计算每个通道的最大最小值
    all_fcst_images = torch.stack(fcst_images)
    fcst_min = all_fcst_images.amin(dim=(0,2,3))  # 每个通道的最小值
    fcst_max = all_fcst_images.amax(dim=(0,2,3))  # 每个通道的最大值

    all_pr_ensmean_images = torch.stack(pr_ensmean_images)
    pr_ensmean_min = all_pr_ensmean_images.amin(dim=(0,2,3))  # 每个通道的最小值
    pr_ensmean_max = all_pr_ensmean_images.amax(dim=(0,2,3))  # 每个通道的最大值

    all_pr_ensstd_images = torch.stack(pr_ensstd_images)
    pr_ensstd_min = all_pr_ensstd_images.amin(dim=(0,2,3))  # 每个通道的最小值
    pr_ensstd_max = all_pr_ensstd_images.amax(dim=(0,2,3))  # 每个通道的最大值
    
    return fcst_min, fcst_max, pr_ensmean_min, pr_ensmean_max, pr_ensstd_min, pr_ensstd_max

def normalize(tensor, fcst_min, fcst_max):
    for i in range(tensor.shape[1]):
        tensor[:, i, :, :] = (tensor[:, i, :, :] - fcst_min[i]) / (fcst_max[i] - fcst_min[i] + 1e-6)
    return tensor

def pr_ensmean_normalize(tensor, pr_ensmean_min, pr_ensmean_max):
    for i in range(tensor.shape[1]):
        tensor[:, i, :, :] = (tensor[:, i, :, :] - pr_ensmean_min[i]) / (pr_ensmean_max[i] - pr_ensmean_min[i] + 1e-6)
    return tensor

def pr_ensstd_normalize(tensor, pr_ensstd_min, pr_ensstd_max):
    for i in range(tensor.shape[1]):
        tensor[:, i, :, :] = (tensor[:, i, :, :] - pr_ensstd_min[i]) / (pr_ensstd_max[i] - pr_ensstd_min[i] + 1e-6)
    return tensor

def collate_fn(batch, fcst_min, fcst_max, pr_ensmean_min, pr_ensmean_max, pr_ensstd_min, pr_ensstd_max):
    # batch 是一个由 (fcst_images, geo_images, labels, leads, pr_ensmean_images, pr_ensstd_images) 组成的列表
    fcst_images, geo_images, labels, leads, pr_ensmean_images, pr_ensstd_images = zip(*batch)
    
    # 将数据堆叠成一个批次
    fcst_images = torch.stack(fcst_images)
    geo_images = torch.stack(geo_images)
    labels = torch.stack(labels)
    leads = torch.stack(leads)
    pr_ensmean_images = torch.stack(pr_ensmean_images)
    pr_ensstd_images = torch.stack(pr_ensstd_images)
    
    # 对 fcst_images 进行归一化处理
    fcst_images = normalize(fcst_images, fcst_min, fcst_max)
    pr_ensmean_images = pr_ensmean_normalize(pr_ensmean_images, pr_ensmean_min, pr_ensmean_max)
    pr_ensstd_images = pr_ensstd_normalize(pr_ensstd_images, pr_ensstd_min, pr_ensstd_max)
    return fcst_images, geo_images, labels, leads, pr_ensmean_images, pr_ensstd_images

class EarlyStopping:
    def __init__(self, patience, delta, verbose=False, start_early_stop_epoch=100):
        """
        patience (int): 在早停之前, 验证损失可以不改善的epoch数。
        verbose (bool): 如果为True, 则每次更新早停时打印一条消息。
        delta (float): 提高的最小变化, 视为改善。
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        # self.val_loss_min = float('inf')
        self.delta = delta
        self.start_early_stop_epoch = start_early_stop_epoch

    def __call__(self, valid_loss, epoch):
        # 如果当前epoch小于start_early_stop_epoch，则不执行早停逻辑
        if epoch < self.start_early_stop_epoch:
            return
        
        score = valid_loss

        if self.best_score == None:
            self.best_score = score
        
        if score > self.best_score - self.delta:
            self.counter += 1
            print(f'Early Stopping Counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0

        if score < self.best_score:
            self.best_score = score
            self.counter = 0
        
            # self.val_loss_min = valid_loss
            # if self.verbose:
            #     print(f'Validation Loss Decrease ({self.val_loss_min:.6f} --> {valid_loss:.6f}).  Saving Model ...')


def model_train(epoch, num_epochs, model, optimizer, loss_fn, train_dataloader, mask, device):
    train_loss_batch = []
    """模型训练阶段"""
    for batch_idx, (fcst_images, geo_images, labels, leads, pr_ensmean_images, pr_ensstd_images) in enumerate(
        tqdm(train_dataloader,
             desc=f'Epoch {epoch+1}/{num_epochs}',
             unit='batch')
    ):
        if device == 'cpu':
            pass
        else:
            fcst_images, geo_images = fcst_images.to(device), geo_images.to(device)
            labels, leads = labels.to(device), leads.to(device)
            mask = mask.to(device)
            pr_ensmean_images, pr_ensstd_images = pr_ensmean_images.to(device), pr_ensstd_images.to(device)

        model.train()
        optimizer.zero_grad()
        # 前向传播
        _, predict_par = model(fcst_images, geo_images, leads, pr_ensmean_images, pr_ensstd_images)
        predict_shift = predict_par[:, 0, :, :]
        predict_mu = predict_par[:, 1, :, :]
        predict_sigma = predict_par[:, 2, :, :]
        # 计算损失
        y_true = labels.float()
        loss = loss_fn(y_true, predict_shift, predict_mu, predict_sigma, mask)
        # 后向传播
        with torch.autograd.detect_anomaly():
            loss.backward()
        optimizer.step()
        # 存储每个batch的训练损失
        train_loss_batch.append(loss.item())
    # 返回该epoch中所有batch的平均训练损失
    return np.mean(train_loss_batch)


def model_valid(epoch, num_epochs, model, loss_fn, valid_dataloader, mask, device):
    valid_loss_batch = []
    """模型验证阶段"""
    for batch_idx, (fcst_images, geo_images, labels, leads, pr_ensmean_images, pr_ensstd_images) in enumerate(
        tqdm(valid_dataloader,
             desc=f'Epoch {epoch+1}/{num_epochs}',
             unit='batch')
    ):
        if device == 'cpu':
            pass
        else:
            fcst_images, geo_images = fcst_images.to(device), geo_images.to(device)
            labels, leads = labels.to(device), leads.to(device)
            mask = mask.to(device)
            pr_ensmean_images, pr_ensstd_images = pr_ensmean_images.to(device), pr_ensstd_images.to(device)
        
        model.eval()
        with torch.no_grad():
            # 前向传播
            _, predict_par = model(fcst_images, geo_images, leads, pr_ensmean_images, pr_ensstd_images)
            predict_shift = predict_par[:, 0, :, :]
            predict_mu = predict_par[:, 1, :, :]
            predict_sigma = predict_par[:, 2, :, :]
            # 计算损失
            y_true = labels.float()
            loss = loss_fn(y_true, predict_shift, predict_mu, predict_sigma, mask)
            # 存储每个batch的验证损失
            valid_loss_batch.append(loss.item())
    # 返回该epoch中所有batch的平均验证损失
    return np.mean(valid_loss_batch)


def save_model(epoch, model, optimizer, train_loss_vec, valid_loss_vec, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss_vec': train_loss_vec,
        'valid_loss_vec': valid_loss_vec,
    }, path)


def load_model(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch'] + 1
    train_loss_vec = checkpoint['train_loss_vec']
    valid_loss_vec = checkpoint['valid_loss_vec']
    return epoch, train_loss_vec, valid_loss_vec





# %%
