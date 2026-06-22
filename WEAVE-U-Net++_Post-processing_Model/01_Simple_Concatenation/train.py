
#%%

import sys
import copy
import os
import time
import gc
import xarray as xr
import numpy as np
import pandas as pd
from scipy.stats import gamma

import torch
from torch import optim
from torch.utils.data import DataLoader, random_split

sys.path.append('/home/liusy/MachineLearning_CSGD/SubX_Multimodel_250525/Unet++/UNetPP-A')
from model import UnetPlusPlus
from aux_copy import LossCRPS_CSGD_ngrid, CustomDataset, EarlyStopping
from aux_copy import model_train, model_valid, save_model, load_model
from aux_copy import compute_channel_min_max, collate_fn
from aux_copy import initialize_weights


device = torch.device('cuda:0')

file_path = '/data/selfdata/datalsy/SubX_Multimodel_250525/00_Results/UNetPP-A/'


##################################
### Step 1: Data Preparation
##################################

tscale = 7

# 经纬度序列
lon = np.linspace(65, 144, 80)
nlon = len(lon)
lat = np.linspace(10, 57, 48)
nlat = len(lat)

# 读取预报数据
datadir = '/data/selfdata/datalsy/SubX_Multimodel_250525/'
fcst_arr = np.load(datadir + f'00_npy/S2S_SubX_fcst_{tscale}days.npy') # 包括降水和其他大气变量
obs_arr = np.load(datadir + f'00_npy/obs_{tscale}days.npy')
issuetime = pd.to_datetime(np.load(datadir + f'common_issue/common_issue_{tscale}days.npy'))
mask_arr = np.load(datadir + 'CN_land_coords/land_mask.npy')
mask_t = torch.tensor(mask_arr, dtype=torch.float32, device=device)

# 年份序列
year_list = np.arange(2001, 2014)  
nyear = len(year_list)
year_index = np.arange(nyear).tolist()   # 原始年份索引


####################################
### Step 2: 13-fold Cross Validation
####################################
# 1 years for test

for ib in range(1,2):   # 重复实验，最后算5次评价指标的平均值

    for iCV in range(5,13):  
        print(f"Starting training for fold {iCV+1}")

        # 测试集年份索引
        test_index = year_index[iCV]
        # 从原始年份序列中，剔除测试年份
        train_index = np.delete(year_index, iCV)
        
        # 将年份索引转换为年份
        train_year = [(x+2001) for x in train_index]
        test_year = test_index + 2001
        print('train_year: ', train_year)
        print('test_year: ', test_year)
        
        # 获取年份内日期的索引
        train_issuedate_idx = issuetime.get_indexer(issuetime[issuetime.year.isin(train_year)])
        issuedate_in_testyear = issuetime[issuetime.year == test_year]
        test_issuedate_idx = issuetime.get_indexer(issuedate_in_testyear)
        nday_test = len(issuedate_in_testyear)


        # 模型参数存储地址
        checkpoint_savepath = file_path + f'Output_{ib}/{tscale}days/checkpoint_savepath/CV{iCV}/'
        os.makedirs(checkpoint_savepath, exist_ok=True)
        
        # 创建日志文件
        log_savepath = file_path + f'Output_{ib}/{tscale}days/log_savepath/'
        os.makedirs(log_savepath, exist_ok=True)
        log_file_path = os.path.join(log_savepath, f'CV{iCV}.txt')
        
        
        ##############################################
        ### Step 3: Dataset Construction for ML Model
        ##############################################

        # train dataset
        fcst_arr_train = fcst_arr[train_issuedate_idx, :, :, :]
        obs_arr_train = obs_arr[train_issuedate_idx, :, :]

        train_dataset = CustomDataset(
            torch.tensor(fcst_arr_train, dtype=torch.float32),
            torch.tensor(obs_arr_train, dtype=torch.float32),
        )

        # 按照7：3拆分为训练集和验证集
        torch_seed = 42
        torch.manual_seed(torch_seed)
        train_size = int(0.7 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_set, val_set = random_split(train_dataset, [train_size, val_size])

        # test dataset
        fcst_arr_test = fcst_arr[test_issuedate_idx, :, :, :]
        obs_arr_test = obs_arr[test_issuedate_idx, :, :]

        test_dataset = CustomDataset(
            torch.tensor(fcst_arr_test, dtype=torch.float32),
            torch.tensor(obs_arr_test, dtype=torch.float32),
        )
        gc.collect()
        
        # 计算训练集数据的最大最小值，用于归一化
        train_fcst_min, train_fcst_max = compute_channel_min_max(train_set)

        # 定义存储结果的数组
        predict_par_all_mat = np.zeros((nday_test, 1, 3, nlat, nlon))    # nday, ntscale, npar, nlat, nlon
        
        ##############################################
        ### Step 4: Model Initialization
        ##############################################

        training_successful = False
        latest_checkpoint = None
        
        while not training_successful:

            # 初始化模型
            model = UnetPlusPlus(False).to(device)
            # initialize_weights(model)
            
            # 超参数设置
            EPOCHS = 50
            BATCHSIZE = 16
            LR = 1e-5
            
            # 早停、优化器、损失函数
            early_stopping = EarlyStopping(patience=20, delta=-0.2, verbose=False, start_early_stop_epoch=25)
            loss_fn = LossCRPS_CSGD_ngrid(n_grid_per_batch=1024)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-6)
            
            # 创建dataloader
            train_dataloader = DataLoader(train_set, batch_size=BATCHSIZE, shuffle=True,
                                        collate_fn=lambda batch: collate_fn(batch, train_fcst_min, train_fcst_max))
            val_dataloader = DataLoader(val_set, batch_size=BATCHSIZE, shuffle=True,
                                        collate_fn=lambda batch: collate_fn(batch, train_fcst_min, train_fcst_max))

            # 训练
            with open(log_file_path, 'w', buffering=1) as log_file:
                
                log_file.write(f"CV {iCV}\n")
                log_file.write(f"Test year {test_year}\n")
                log_file.write(f"Torch manual seed (to split train and val set): {torch_seed}\n")
                log_file.write(f"{'='*40}\n")
                log_file.write(f"Batch size: {BATCHSIZE}\n")
                log_file.write(f"Learning rate: {LR}\n")
                log_file.write(f"{'='*40}\n")
                
                # 当前折的损失序列
                if latest_checkpoint is not None:
                    epoch, train_loss_vec, valid_loss_vec = load_model(latest_checkpoint, model, optimizer, device)
                else:
                    epoch = 0
                    train_loss_vec = [] # 用于存储：所有epoch的损失组成的序列
                    valid_loss_vec = []
                
                # Epoch循环
                while epoch < EPOCHS:

                    train_loss = 0   
                    valid_loss = 0
                    loss_abnormal = False
                    
                    # 模型训练
                    train_loss = model_train(epoch, EPOCHS, model, optimizer, loss_fn, train_dataloader, mask_t, device)
                    # 模型验证
                    valid_loss = model_valid(epoch, EPOCHS, model, loss_fn, val_dataloader, mask_t, device)
                    
                    # 判断损失是否为nan / 判断损失是否异常
                    if np.isnan(train_loss) or np.isnan(valid_loss):
                        print(f"Loss is NaN at epoch {epoch + 1}.")
                        loss_abnormal = True
                    elif epoch > 0: # 只有在epoch > 0时才比较上一轮的损失
                        if (train_loss > 1.3*(train_loss_vec[-1])) or (valid_loss > 1.3*(valid_loss_vec[-1])):
                            print(f"Loss is abnormal at epoch {epoch + 1}.")
                            loss_abnormal = True
                    
                    if loss_abnormal:
                        if len(os.listdir(checkpoint_savepath)) == 0:
                            # 如果没有存储任何checkpoint
                            training_successful = False
                            print('There is no checkpoint. Restart training from Epoch 1...')
                            break   # 跳出epoch循环，重新开始训练
                        else:
                            # 寻找最新的checkpoint文件
                            latest_checkpoint = max([checkpoint_savepath + f for f in os.listdir(checkpoint_savepath)], 
                                                    key=os.path.getctime)
                            print(f'Loading checkpoint from {latest_checkpoint}...')
                            break
                        
                    
                    # 打印损失
                    print(f" Epoch {epoch + 1} | Train Loss: {train_loss} | Valid Loss: {valid_loss}")
                    train_loss_vec.append(np.round(train_loss, 3))
                    valid_loss_vec.append(np.round(valid_loss, 3))
                    
                    template = 'Epoch {}, Train Loss: {:.4f}, Valid Loss: {:.4f}\n'
                    epoch_info = template.format(epoch + 1, train_loss, valid_loss)
                    print(epoch_info.strip())
                    log_file.write(epoch_info)
                    
                    # 每隔10个epoch存储一次模型参数
                    if ((epoch+1) % 10 == 0) and (epoch != 0):
                        # 存储模型参数
                        save_model(epoch+1, model, optimizer,
                                    train_loss_vec, valid_loss_vec,
                                    checkpoint_savepath + f'ckpt_latest_CV{iCV}.pth')
                        print(f'Saving checkpoint... Epoch {epoch+1}...')
                        log_file.write(f"Saved checkpoint at Epoch {epoch + 1}.\n")
                    
                    # early stop
                    early_stopping(valid_loss, epoch)
                    if early_stopping.early_stop:
                        training_successful = True
                        log_file.write(f"Early stopping at Epoch {epoch + 1}.\n")
                        print(f'Early stop at Epoch {epoch+1}... Training Success !')
                        break   # 跳出epoch循环
                    
                    # epoch reach the max EPOCH
                    if epoch == EPOCHS-1:
                        training_successful = True
                        log_file.write(f"Max Epoch reached. Training Success !\n")
                        print('Epoch reaches the maximum... Training Success !')
                    
                    # scheduler.step()
                    epoch += 1
                
                if not training_successful:
                    continue   # 重新开始最外层while循环
                    
                    
                # 训练成功
                # 写入最终的损失序列
                log_file.write("\nTrain Loss Sequence:\n")
                log_file.write(", ".join(map(str, train_loss_vec)) + "\n")
                log_file.write("\nValidation Loss Sequence:\n")
                log_file.write(", ".join(map(str, valid_loss_vec)) + "\n")
                # 存储损失序列
                loss_savepath = file_path + f'Output_{ib}/{tscale}days/loss/'
                os.makedirs(loss_savepath, exist_ok=True)
                np.save(loss_savepath + f'loss_CV{iCV}.npy', np.array([train_loss_vec, valid_loss_vec]))
                
                # 存储模型参数
                model_savepath = file_path + f'Output_{ib}/{tscale}days/unet_model/'
                os.makedirs(model_savepath, exist_ok=True)
                torch.save(model.state_dict(), model_savepath + f'unet_CV{iCV}.pth')

                log_file.write(f"CV {iCV} training completed.\n")
                print(f'CV {iCV} training completed.')
                

                ##############################################
                ### Step 4: Predict
                ##############################################
                
                test_batchsize = len(test_dataset)
                test_dataloader = DataLoader(test_dataset, batch_size=test_batchsize, shuffle=False, 
                                            collate_fn=lambda batch: collate_fn(batch, train_fcst_min, train_fcst_max))
                
                # predict
                for fcst_images, labels in test_dataloader:
                    fcst_images, labels = fcst_images.to(device), labels.to(device)
                        
                    model.eval()
                    with torch.no_grad():
                        
                        # 避免CUDA out of memory，逐日预测
                        for iday in range(nday_test):
                            print(f'Predicting day {iday}...')

                            predict_par_1d = model(fcst_images[iday, ...].unsqueeze(0))  # (1, 3, H, W)
                            predict_shift_1d = predict_par_1d[:, 0, :, :]
                            predict_mu_1d = predict_par_1d[:, 1, :, :]
                            predict_sigma_1d = predict_par_1d[:, 2, :, :]

                            # 将mask=0的位置负值为nan，只保存mask=1的位置的输出
                            mask_bool = mask_t.bool()
                            # 调整维度为 (B, H, W)
                            if mask_bool.dim() == 2:
                                mask_bool = mask_bool.unsqueeze(0)  # -> (1, H, W)
                            if mask_bool.size(0) == 1:
                                mask_bool = mask_bool.expand(predict_par_1d.size(0), -1, -1)

                            predict_shift_1d = predict_shift_1d.masked_fill(~mask_bool, float('nan')).cpu().numpy()
                            predict_mu_1d = predict_mu_1d.masked_fill(~mask_bool, float('nan')).cpu().numpy()
                            predict_sigma_1d = predict_sigma_1d.masked_fill(~mask_bool, float('nan')).cpu().numpy()

                            predict_par_1CV_1d = np.concatenate([
                                predict_shift_1d[:, np.newaxis, :, :],
                                predict_mu_1d[:, np.newaxis, :, :],
                                predict_sigma_1d[:, np.newaxis, :, :]
                            ], axis=1)   # shape: (1, 3, nlat, nlon)

                            # 将本次交叉验证的测试集结果存储到最终结果中
                            predict_par_all_mat[iday, 0, :, :, :] = predict_par_1CV_1d
                
                log_file.write(f"CV {iCV} prediction completed.\n")
                print(f'CV {iCV} prediction completed.')
                
                
            ################################################
            ### Step 5: Save Post-Processing Results
            ################################################

            predict_par_all_da = xr.DataArray(
                predict_par_all_mat,
                coords={
                    'issuetime': issuedate_in_testyear,
                    'tscale': [tscale],
                    'par': ['shift', 'mu', 'sigma'],
                    'lat': lat,
                    'lon': lon
                },
                dims=['issuetime', 'tscale', 'par', 'lat', 'lon']
            )            
            predictpar_savepath = file_path + f'/Output_{ib}/{tscale}days/Unetfcst_CSGDPar/'
            os.makedirs(predictpar_savepath, exist_ok=True)
            predict_par_all_da.to_netcdf(predictpar_savepath + f'par_CV{iCV}.nc')
            print('CSGDPar have been saved !')
        

            # 该CV训练结束
            # 清理显存
            del model, optimizer, train_dataset, test_dataset
            torch.cuda.empty_cache()
            gc.collect()
                




                            



# %%
