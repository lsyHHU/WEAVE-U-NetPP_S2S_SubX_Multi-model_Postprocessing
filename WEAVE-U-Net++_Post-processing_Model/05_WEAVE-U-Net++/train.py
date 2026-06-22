
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

sys.path.append('/home/liusy/MachineLearning_CSGD/S2S_SubX_Multimodel_250525/03_Unet++_newTest_251209/UNetPP-A-WeightPredictor-T-KLloss-ContinueTraining_3')
from model import UnetPlusPlus
from aux_copy import CustomDataset, EarlyStopping
from aux_copy import model_train, model_valid, save_model, load_model
from aux_copy import compute_channel_min_max, collate_fn
from aux_copy import initialize_weights
from lossfunc import LossCRPS_CSGD_ngrid


device = torch.device('cuda:0')

file_path = '/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/00_Results_newTest_251209/UNetPP-A-WeightPredictor-T-KLloss-ContinueTraining_3/'


##################################
### Step 1: Data Preparation
##################################

tscale = 14

# 经纬度序列
lon = np.linspace(65, 144, 80)
nlon = len(lon)
lat = np.linspace(10, 57, 48)
nlat = len(lat)

# 读取预报数据
datadir = '/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/'
fcst_arr = np.load(datadir + f'00_npy/S2S_SubX_fcst_{tscale}days_1.npy') # 包括降水和其他大气变量
geo_arr = np.load(datadir + f'00_npy/geo_{tscale}days.npy')  # 地理数据
obs_arr = np.load(datadir + f'00_npy/obs_{tscale}days.npy')

issuetime = pd.to_datetime(np.load(datadir + f'common_issue/common_issue_{tscale}days.npy'))

pr_ensmean_arr = np.load(datadir + f'00_npy/S2S_SubX_pr_ensmean_{tscale}days.npy')
pr_ensstd_arr = np.load(datadir + f'00_npy/S2S_SubX_pr_ensstd_{tscale}days.npy')

lead = np.load(datadir + f'common_issue/lead_allmodels_list_{tscale}days.npy')  # (nday, nmodel)
mask_arr = np.load(datadir + 'CN_land_coords/land_mask.npy')
mask_t = torch.tensor(mask_arr, dtype=torch.float32, device=device)

# 读取原始降水预报crps数据
crps_arr = np.load(datadir + f'00_Results/crps_by_model_{tscale}days.npy')  # (nday, nmodel, nlat, nlon)

# 年份序列
year_list = np.arange(2001, 2014)  
nyear = len(year_list)
year_index = np.arange(nyear).tolist()   # 原始年份索引


####################################
### Step 2: 13-fold Cross Validation
####################################
# 1 years for test

# seed = [22, 50, 66, 77, 88]

for ib in range(1):   # 重复实验，最后算10次评价指标的平均值

    for iCV in range(4):
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
        geo_arr_train = geo_arr[train_issuedate_idx, :, :, :]
        obs_arr_train = obs_arr[train_issuedate_idx, :, :]
        lead_arr_train = lead[train_issuedate_idx, :]
        pr_ensmean_arr_train = pr_ensmean_arr[train_issuedate_idx, :, :, :]
        pr_ensstd_arr_train = pr_ensstd_arr[train_issuedate_idx, :, :, :]
        crps_arr_train = crps_arr[train_issuedate_idx, :, :, :]

        train_dataset = CustomDataset(
            torch.tensor(fcst_arr_train, dtype=torch.float32),
            torch.tensor(geo_arr_train, dtype=torch.float32),
            torch.tensor(obs_arr_train, dtype=torch.float32),
            torch.tensor(lead_arr_train, dtype=torch.float32),
            torch.tensor(pr_ensmean_arr_train, dtype=torch.float32),
            torch.tensor(pr_ensstd_arr_train, dtype=torch.float32),
            torch.tensor(crps_arr_train, dtype=torch.float32)
        )

        # 按照7：3拆分为训练集和验证集
        torch_seed = 22
        # torch_seed = seed[ib-3]
        torch.manual_seed(torch_seed)
        train_size = int(0.7 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_set, val_set = random_split(train_dataset, [train_size, val_size])
    
        # test dataset
        fcst_arr_test = fcst_arr[test_issuedate_idx, :, :, :]
        geo_arr_test = geo_arr[test_issuedate_idx, :, :, :]
        obs_arr_test = obs_arr[test_issuedate_idx, :, :]
        lead_arr_test = lead[test_issuedate_idx, :]
        pr_ensmean_arr_test = pr_ensmean_arr[test_issuedate_idx, :, :, :]
        pr_ensstd_arr_test = pr_ensstd_arr[test_issuedate_idx, :, :, :]
        crps_arr_test = crps_arr[test_issuedate_idx, :, :, :]

        test_dataset = CustomDataset(
            torch.tensor(fcst_arr_test, dtype=torch.float32),
            torch.tensor(geo_arr_test, dtype=torch.float32),
            torch.tensor(obs_arr_test, dtype=torch.float32),
            torch.tensor(lead_arr_test, dtype=torch.float32),
            torch.tensor(pr_ensmean_arr_test, dtype=torch.float32),
            torch.tensor(pr_ensstd_arr_test, dtype=torch.float32),
            torch.tensor(crps_arr_test, dtype=torch.float32)
        )
        gc.collect()
        
        # 计算训练集数据的最大最小值，用于归一化
        train_fcst_min, train_fcst_max, train_pr_ensmean_min, train_pr_ensmean_max, train_pr_ensstd_min, train_pr_ensstd_max = \
            compute_channel_min_max(train_set)

        # 定义存储结果的数组
        predict_par_all_mat = np.zeros((nday_test, 1, 3, nlat, nlon))    # nday, ntscale, npar, nlat, nlon
        w_all_mat = np.zeros((nday_test, 18, nlat, nlon))    # nday, nmodel, nlat, nlon
        
        ##############################################
        ### Step 4: Model Initialization
        ##############################################

        training_successful = False
        latest_checkpoint = None
        
        while not training_successful:

            # 初始化模型
            WEIGHT_PREDICTOR_T = 0.1
            in_ch_list = [10, 3, 4, 4, 11, 1, 4, 33, 4, 10, 11, 4, 11, 10, 4, 8, 3, 7]
            model = UnetPlusPlus(in_ch_list, temperature=WEIGHT_PREDICTOR_T, deep_supervision=False).to(device)
            # initialize_weights(model)
            
            # 加载UNetPP-A-WeightPredictor-T已经训练好的模型
            model_path = f'/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/00_Results_newTest_251209/UNetPP-A-WeightPredictor-T/Output_{ib}/{tscale}days/unet_model/unet_CV{iCV}.pth'
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

            # 超参数设置
            EPOCHS = 150
            BATCHSIZE = 4
            LR = 5e-5

            # 损失超参数
            LOSS_TEMPERATURE = 5.0
            ALPHA = 5.0  # 2.0

            # 早停、优化器、损失函数
            early_stopping = EarlyStopping(patience=10, delta=-0.1, verbose=False, start_early_stop_epoch=30)
            loss_fn = LossCRPS_CSGD_ngrid(n_grid_per_batch=2048)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
            
            # 创建dataloader
            train_dataloader = DataLoader(train_set, batch_size=BATCHSIZE, shuffle=True,
                                        collate_fn=lambda batch: collate_fn(batch, train_fcst_min, train_fcst_max, train_pr_ensmean_min, train_pr_ensmean_max, train_pr_ensstd_min, train_pr_ensstd_max))
            val_dataloader = DataLoader(val_set, batch_size=BATCHSIZE, shuffle=True,
                                        collate_fn=lambda batch: collate_fn(batch, train_fcst_min, train_fcst_max, train_pr_ensmean_min, train_pr_ensmean_max, train_pr_ensstd_min, train_pr_ensstd_max))

            # 训练
            with open(log_file_path, 'w', buffering=1) as log_file:
                
                log_file.write(f"CV {iCV}\n")
                log_file.write(f"Test year {test_year}\n")
                log_file.write(f"Torch manual seed (to split train and val set): {torch_seed}\n")
                log_file.write(f"{'='*40}\n")
                log_file.write(f"Batch size: {BATCHSIZE}\n")
                log_file.write(f"Learning rate: {LR}\n")
                log_file.write(f"Weight Predictor - Temperature: {WEIGHT_PREDICTOR_T}\n")
                log_file.write(f"KL Loss - Temperature: {LOSS_TEMPERATURE}\n")
                log_file.write(f"KL Loss - Alpha: {ALPHA}\n")
                log_file.write(f"{'='*40}\n")
                
                # 当前折的损失序列
                if latest_checkpoint is not None:
                    epoch, train_crps_loss_vec, train_kl_loss_vec, train_loss_vec, \
                        valid_crps_loss_vec, valid_kl_loss_vec, valid_loss_vec = load_model(latest_checkpoint, model, optimizer, device)
                else:
                    epoch = 0
                    train_crps_loss_vec = []
                    train_kl_loss_vec = []
                    train_loss_vec = [] # 用于存储：所有epoch的损失组成的序列
                    valid_crps_loss_vec = []
                    valid_kl_loss_vec = []
                    valid_loss_vec = []
                
                # Epoch循环
                while epoch < EPOCHS:

                    train_crps_loss, train_kl_loss, train_loss = 0, 0, 0
                    valid_crps_loss, valid_kl_loss, valid_loss = 0, 0, 0
                    loss_abnormal = False
                    
                    # 模型训练
                    train_crps_loss, train_kl_loss, train_loss = \
                        model_train(epoch, EPOCHS, LOSS_TEMPERATURE, ALPHA, model, optimizer, loss_fn, train_dataloader, mask_t, device)
                    # 模型验证
                    valid_crps_loss, valid_kl_loss, valid_loss = \
                        model_valid(epoch, EPOCHS, LOSS_TEMPERATURE, ALPHA, model, loss_fn, val_dataloader, mask_t, device)
                    
                    # 判断损失是否为nan / 判断损失是否异常
                    if np.isnan(train_loss) or np.isnan(valid_loss):
                        print(f"Loss is NaN at epoch {epoch + 1}.")
                        loss_abnormal = True
                    elif epoch > 0: # 只有在epoch > 0时才比较上一轮的损失
                        if (train_loss > 1.5*(train_loss_vec[-1])) or (valid_loss > 1.5*(valid_loss_vec[-1])):
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
                    print(f" Epoch {epoch + 1}")
                    print(f" Train CRPS Loss: {train_crps_loss} | Train KL Loss: {train_kl_loss} | Train Loss: {train_loss}")
                    print(f" Valid CRPS Loss: {valid_crps_loss} | Valid KL Loss: {valid_kl_loss} | Valid Loss: {valid_loss}")
                    train_crps_loss_vec.append(np.round(train_crps_loss, 3))
                    train_kl_loss_vec.append(np.round(train_kl_loss, 3))
                    train_loss_vec.append(np.round(train_loss, 3))
                    valid_crps_loss_vec.append(np.round(valid_crps_loss, 3))
                    valid_kl_loss_vec.append(np.round(valid_kl_loss, 3))
                    valid_loss_vec.append(np.round(valid_loss, 3))
                    
                    template = 'Epoch {}, Train CRPS Loss: {:.4f}, Train KL Loss: {:.4f}, Train Loss: {:.4f}, Valid CRPS Loss: {:.4f}, Valid KL Loss: {:.4f}, Valid Loss: {:.4f}\n'
                    epoch_info = template.format(epoch + 1, train_crps_loss, train_kl_loss, train_loss, valid_crps_loss, valid_kl_loss, valid_loss)
                    print(epoch_info.strip())
                    log_file.write(epoch_info)
                    
                    # 每隔5个epoch存储一次模型参数
                    if ((epoch+1) % 20 == 0) and (epoch != 0):
                        # 存储模型参数
                        save_model(epoch+1, model, optimizer,
                                    train_crps_loss_vec, train_kl_loss_vec, train_loss_vec, 
                                    valid_crps_loss_vec, valid_kl_loss_vec, valid_loss_vec,
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
                log_file.write("\nTrain CRPS Loss Sequence:\n")
                log_file.write(", ".join(map(str, train_crps_loss_vec)) + "\n")
                log_file.write("\nTrain KL Loss Sequence:\n")
                log_file.write(", ".join(map(str, train_kl_loss_vec)) + "\n")
                log_file.write("\nTrain Loss Sequence:\n")
                log_file.write(", ".join(map(str, train_loss_vec)) + "\n")
                log_file.write("\nValidation CRPS Loss Sequence:\n")
                log_file.write(", ".join(map(str, valid_crps_loss_vec)) + "\n")
                log_file.write("\nValidation KL Loss Sequence:\n")
                log_file.write(", ".join(map(str, valid_kl_loss_vec)) + "\n")
                log_file.write("\nValidation Loss Sequence:\n")
                log_file.write(", ".join(map(str, valid_loss_vec)) + "\n")
                # 存储损失序列
                loss_savepath = file_path + f'Output_{ib}/{tscale}days/loss/'
                os.makedirs(loss_savepath, exist_ok=True)
                np.save(loss_savepath + f'loss_CV{iCV}.npy', np.array([train_crps_loss_vec, train_kl_loss_vec, train_loss_vec, 
                                                                        valid_crps_loss_vec, valid_kl_loss_vec, valid_loss_vec]))
                
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
                                            collate_fn=lambda batch: collate_fn(batch, train_fcst_min, train_fcst_max, train_pr_ensmean_min, train_pr_ensmean_max, train_pr_ensstd_min, train_pr_ensstd_max))
                
                # predict
                for fcst_images, geo_images, labels, leads, pr_ensmean_images, pr_ensstd_images, crps in test_dataloader:
                    fcst_images, geo_images = fcst_images.to(device), geo_images.to(device)
                    labels, leads = labels.to(device), leads.to(device)
                    crps = crps.to(device)
                    pr_ensmean_images, pr_ensstd_images = pr_ensmean_images.to(device), pr_ensstd_images.to(device)
                        
                    model.eval()
                    with torch.no_grad():
                        
                        # 避免CUDA out of memory，逐日预测
                        for iday in range(nday_test):
                            print(f'Predicting day {iday}...')

                            pred_w, predict_par_1d = model(fcst_images[iday, ...].unsqueeze(0), 
                                                            geo_images[iday, ...].unsqueeze(0),  
                                                            leads[iday, ...].unsqueeze(0),
                                                            pr_ensmean_images[iday, ...].unsqueeze(0),
                                                            pr_ensstd_images[iday, ...].unsqueeze(0))  # (1, 3, H, W)
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
                            w_all_mat[iday] = pred_w.cpu().numpy()

                
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

            w_all_da = xr.DataArray(
                w_all_mat,
                coords={
                    'issuetime': issuedate_in_testyear,
                    'model': ['46LCESM1', 'CCSM4', 'CFSv2', 'FIMr1p1', 'GEFS', 'NESM', 'GEOS_V2p1',
                              'BoM', 'CMA', 'CNRM', 'CPTEC', 'ECCC', 'ECMWF', 'HMCR', 'IAPCAS', 'ISACCNR', 'KMA', 'UKMO'],
                    'lat': lat,
                    'lon': lon
                }, dims=['issuetime', 'model', 'lat', 'lon']
            )
            predictw_savepath = file_path + f'/Output_{ib}/{tscale}days/Unetfcst_w/'
            os.makedirs(predictw_savepath, exist_ok=True)
            w_all_da.to_netcdf(predictw_savepath + f'weights_CV{iCV}.nc')
            print('weights have been saved !')


            # 该CV训练结束
            # 清理显存
            del model, optimizer, train_dataset, test_dataset
            torch.cuda.empty_cache()
            gc.collect()
                




   



# %%
