
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

from lossfunc import compute_oracle_weights, kl_divergence_oracle_w


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
                

class CustomDataset(Dataset):
    def __init__(self, fcst_images, geo_images, labels, leads, pr_ensmean, pr_ensstd, crps):
        self.fcst_images = fcst_images
        self.geo_images = geo_images
        self.labels = labels
        self.leads = leads
        self.pr_ensmean = pr_ensmean
        self.pr_ensstd = pr_ensstd
        self.crps = crps
    
    def __len__(self):
        return len(self.fcst_images)
    
    def __getitem__(self, idx):
        return self.fcst_images[idx], self.geo_images[idx], self.labels[idx], self.leads[idx], \
            self.pr_ensmean[idx], self.pr_ensstd[idx], self.crps[idx]


def compute_channel_min_max(dataset):
    fcst_images = []
    pr_ensmean_images = []
    pr_ensstd_images = []
    for fcst_image, _, _, _, pr_ensmean_image, pr_ensstd_image, _ in dataset:
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
    # batch 是一个由 (fcst_images, geo_images, labels, leads, pr_ensmean_images, pr_ensstd_images, crps) 组成的列表
    fcst_images, geo_images, labels, leads, pr_ensmean_images, pr_ensstd_images, crps = zip(*batch)
    
    # 将数据堆叠成一个批次
    fcst_images = torch.stack(fcst_images)
    geo_images = torch.stack(geo_images)
    labels = torch.stack(labels)
    leads = torch.stack(leads)
    pr_ensmean_images = torch.stack(pr_ensmean_images)
    pr_ensstd_images = torch.stack(pr_ensstd_images)
    crps = torch.stack(crps)
    
    # 对 fcst_images 进行归一化处理
    fcst_images = normalize(fcst_images, fcst_min, fcst_max)
    pr_ensmean_images = pr_ensmean_normalize(pr_ensmean_images, pr_ensmean_min, pr_ensmean_max)
    pr_ensstd_images = pr_ensstd_normalize(pr_ensstd_images, pr_ensstd_min, pr_ensstd_max)
    return fcst_images, geo_images, labels, leads, pr_ensmean_images, pr_ensstd_images, crps

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


def model_train(epoch, num_epochs, TEMPERATURE, ALPHA, model, optimizer, loss_fn, train_dataloader, mask, device):
    train_crps_loss_batch = []
    train_kl_loss_batch = []
    train_loss_batch = []
    """模型训练阶段"""
    for batch_idx, (fcst_images, geo_images, labels, leads, pr_ensmean_images, pr_ensstd_images, crps) in enumerate(
        tqdm(train_dataloader,
             desc=f'Epoch {epoch+1}/{num_epochs}',
             unit='batch')
    ):
        if device == 'cpu':
            pass
        else:
            fcst_images, geo_images = fcst_images.to(device), geo_images.to(device)
            labels, leads = labels.to(device), leads.to(device)
            crps = crps.to(device)
            mask = mask.to(device)
            pr_ensmean_images, pr_ensstd_images = pr_ensmean_images.to(device), pr_ensstd_images.to(device)

        model.train()
        optimizer.zero_grad()
        # 前向传播
        pred_w, predict_par = model(fcst_images, geo_images, leads, pr_ensmean_images, pr_ensstd_images)
        predict_shift = predict_par[:, 0, :, :]
        predict_mu = predict_par[:, 1, :, :]
        predict_sigma = predict_par[:, 2, :, :]
        # 计算CRPS损失
        y_true = labels.float()
        crps_loss = loss_fn(y_true, predict_shift, predict_mu, predict_sigma, mask)
        # 计算权重KL损失
        oracle_w, _ = compute_oracle_weights(err=crps, temperatue=TEMPERATURE)
        kl_loss = ALPHA * kl_divergence_oracle_w(oracle_w.detach(), pred_w, mask)
        kl_loss_value = kl_loss.item()
        total_loss = crps_loss + kl_loss
        total_loss.backward()
        optimizer.step()
        # 存储每个batch的训练损失
        train_crps_loss_batch.append(crps_loss.item())
        train_kl_loss_batch.append(kl_loss_value)
        train_loss_batch.append(total_loss.item())
    # 返回该epoch中所有batch的平均训练损失
    return np.mean(train_crps_loss_batch), np.mean(train_kl_loss_batch), np.mean(train_loss_batch)


def model_valid(epoch, num_epochs, TEMPERATURE, ALPHA, model, loss_fn, valid_dataloader, mask, device):
    valid_crps_loss_batch = []
    valid_kl_loss_batch = []
    valid_loss_batch = []
    """模型验证阶段"""
    for batch_idx, (fcst_images, geo_images, labels, leads, pr_ensmean_images, pr_ensstd_images, crps) in enumerate(
        tqdm(valid_dataloader,
             desc=f'Epoch {epoch+1}/{num_epochs}',
             unit='batch')
    ):
        if device == 'cpu':
            pass
        else:
            fcst_images, geo_images = fcst_images.to(device), geo_images.to(device)
            labels, leads = labels.to(device), leads.to(device)
            crps = crps.to(device)
            mask = mask.to(device)
            pr_ensmean_images, pr_ensstd_images = pr_ensmean_images.to(device), pr_ensstd_images.to(device)
        
        model.eval()
        with torch.no_grad():
            # 前向传播
            pred_w, predict_par = model(fcst_images, geo_images, leads, pr_ensmean_images, pr_ensstd_images)
            predict_shift = predict_par[:, 0, :, :]
            predict_mu = predict_par[:, 1, :, :]
            predict_sigma = predict_par[:, 2, :, :]
            # 计算CRPS损失
            y_true = labels.float()
            crps_loss = loss_fn(y_true, predict_shift, predict_mu, predict_sigma, mask)
            # 计算权重KL损失
            oracle_w, _ = compute_oracle_weights(err=crps, temperatue=TEMPERATURE)
            kl_loss = ALPHA * kl_divergence_oracle_w(oracle_w.detach(), pred_w, mask)
            # 计算总损失
            loss = crps_loss + kl_loss
            # 存储每个batch的验证损失
            valid_crps_loss_batch.append(crps_loss.item())
            valid_kl_loss_batch.append(kl_loss.item())
            valid_loss_batch.append(loss.item())
    # 返回该epoch中所有batch的平均验证损失
    return np.mean(valid_crps_loss_batch), np.mean(valid_kl_loss_batch), np.mean(valid_loss_batch)


def save_model(epoch, model, optimizer, 
                train_crps_loss_vec, train_kl_loss_vec, train_loss_vec, 
                valid_crps_loss_vec, valid_kl_loss_vec, valid_loss_vec, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_crps_loss_vec': train_crps_loss_vec,
        'train_kl_loss_vec': train_kl_loss_vec,
        'train_loss_vec': train_loss_vec,
        'valid_crps_loss_vec': valid_crps_loss_vec,
        'valid_kl_loss_vec': valid_kl_loss_vec,
        'valid_loss_vec': valid_loss_vec,
    }, path)


def load_model(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch'] + 1
    train_crps_loss_vec = checkpoint['train_crps_loss_vec']
    train_kl_loss_vec = checkpoint['train_kl_loss_vec']
    train_loss_vec = checkpoint['train_loss_vec']
    valid_crps_loss_vec = checkpoint['valid_crps_loss_vec']
    valid_kl_loss_vec = checkpoint['valid_kl_loss_vec']
    valid_loss_vec = checkpoint['valid_loss_vec']
    return epoch, train_crps_loss_vec, train_kl_loss_vec, train_loss_vec, \
        valid_crps_loss_vec, valid_kl_loss_vec, valid_loss_vec





# %%
