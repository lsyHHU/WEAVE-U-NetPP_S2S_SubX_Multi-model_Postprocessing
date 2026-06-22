#%%

import os
import time
import math
import itertools
import xarray as xr
import numpy as np
import pandas as pd
import scipy as sp
from multiprocessing import Pool

from numpy import ma
from scipy.stats import gamma
from scipy.special import beta
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from pandas.tseries.offsets import MonthEnd

import sys
sys.path.append('/home/liusy/MachineLearning_CSGD/S2S_SubX_Multimodel_250525/EMOS_py')
from CSGD_EMOS_auxiliary_func import crpsCondCSGD


def multiprocess_Fit_crps_CondCSGD(igrid):
# for igrid in range(1):

    print('------------------Func: multiprocess_Fit_crps_CondCSGD is running! ------------------')
    print(f'------------------------- Grid No. {igrid} has begun! -------------------------')

    year = np.arange(2001, 2014)
    nyear = len(year)

    month = np.arange(1, 13)
    nmonth = len(month)

    tscale = [7, 14, 21, 28]
    ntscale = len(tscale)

    # 读取经纬度csv文件
    coords_file = pd.read_csv('/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/CN_land_coords/land_coords.csv')
    grid_lat, grid_lon = coords_file['lat'][igrid], coords_file['lon'][igrid]


    # --------------------- Step 0: 数据读取 ---------------------

    # 读取当前网格的climatologyCSGD分布参数
    climaCSGDPar_path = '/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/01_EMOS_py/ClimaCSGD_param_20012013_monthly/'
    scale_cl_arr = np.load(climaCSGDPar_path + f'scale_cl/par_lat{grid_lat}_lon{grid_lon}.npy')
    shape_cl_arr = np.load(climaCSGDPar_path + f'shape_cl/par_lat{grid_lat}_lon{grid_lon}.npy')
    shift_cl_arr = np.load(climaCSGDPar_path + f'shift_cl/par_lat{grid_lat}_lon{grid_lon}.npy')

    # 时间尺度循环
    for t in range(ntscale):
        _t = tscale[t]

        # 读取当前时间尺度和当前网格的数据文件
        data_file = pd.read_csv(f'/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/01_BMA/{_t}days/raw_csv/rawfcst_obs_Lat{grid_lat}_Lon{grid_lon}.csv')
        all_dates = pd.to_datetime(data_file['dates'].values)

        # 存储回归参数
        par_reg = np.zeros((nyear, nmonth, 1, 23, 1, 1))  # 回归模型有18+5=23个参数，最后两个维度存储经纬度
        # 存储条件CSGD分布参数
        mu_CondDis_list, sigma_CondDis_list, shift_CondDis_list = [], [], []   # 先将三个参数存成列表，等循环结束后再reshape成指定的形状

        # 年份循环
        for y in range(nyear):
            # 月份循环
            for m in range(nmonth):
                _y, _m = year[y], month[m]
                # 挑选出训练日期序列
                train_central_date = all_dates[(all_dates.month == _m) & (all_dates.year != _y)]
                # 用 DatetimeIndex 的 union 来累积所有训练日期（自动去重）
                train_dates_idx = pd.DatetimeIndex([], freq=None)
                for d in train_central_date:
                    # 前一个月的第一天
                    start = (d - pd.DateOffset(months=1)).replace(day=1)
                    # 后一个月的最后一天：先加1个月，再移动到该月的月末
                    end = (d + pd.DateOffset(months=1)) + MonthEnd(0)
                    # 该区间的完整日期序列
                    date_range = pd.date_range(start=start, end=end, freq='D')
                    # 与 all_dates 取交集（只保留实际存在于 allDates 的日子）
                    in_all = date_range.intersection(all_dates)
                    # 累加（union 自动去重）
                    train_dates_idx = train_dates_idx.union(in_all)
                
                # --------------------- Step 1: 数据准备 ---------------------

                # 挑选出训练日期的预报数据
                fcst_train_data = data_file.iloc[:, :18][all_dates.isin(train_dates_idx)]
                fcst_train_arr = fcst_train_data.values    # shape: nday, nmember

                # 挑选出训练日期的观测数据
                obs_train_data = data_file['obs'][all_dates.isin(train_dates_idx)]
                obs_train_arr = obs_train_data.values   

                # --------------------- Step 2: 计算预报因子 ensmean anomaly and MD ---------------------

                fcst_ensmeanano_train = fcst_train_arr / np.nanmean(fcst_train_arr, axis=1, keepdims=True)   # nday, 18
                M = fcst_ensmeanano_train.shape[1]   # 模型数 = 18

                fcst_diff_train = np.abs(fcst_train_arr[:, :, np.newaxis] - fcst_train_arr[:, np.newaxis, :])
                fcst_MD_train = np.nansum(fcst_diff_train, axis=(1, 2)) / (fcst_diff_train.shape[-1] ** 2)   # nday

                # --------------------- Step 3: 加载climatologyCSGD分布参数 ---------------------

                scale_cl = scale_cl_arr[y, m, t]
                shape_cl = shape_cl_arr[y, m, t]
                shift_cl = shift_cl_arr[y, m, t]
                # 将以上参数转换为 mu, sigma
                muc = (shape_cl * scale_cl).astype(np.float64)
                sigmac = (np.sqrt(shape_cl) * scale_cl).astype(np.float64)
                shiftc = (shift_cl).astype(np.float64)

                # --------------------- Step 4: 拟合线性回归模型 ---------------------

                param_initial = [0.05, 0.5, 0.5, 0.5, 0.9] + [0.5]*M
                # param_ranges = [               
                #     (0.001, 10),
                #     (0.0001, 1.0),
                #     (0.1, 1.0),
                #     (0.0001, 1.0),
                #     (0.0001, 1.5)
                # ] + [(0.0, 10)]*M

                param_ranges = [               
                    (0.0001, 10),
                    (0.0001, 10),
                    (0.0001, 10),
                    (0.0001, 10),
                    (0.0001, 10)
                ] + [(0.0, 10)]*M

                par_reg[y, m, 0, :, 0, 0] = minimize(
                    fun=crpsCondCSGD,
                    x0=param_initial,
                    args=(obs_train_arr, fcst_ensmeanano_train, fcst_MD_train, muc, sigmac, shiftc),
                    method='L-BFGS-B',
                    bounds=param_ranges,
                    tol=1e-6
                ).x

                # --------------------- Step 5: 计算CondCSGD分布参数 ---------------------

                # 验证集日期
                valid_date = all_dates[(all_dates.month == _m) & (all_dates.year == _y)]

                # 挑选出验证集的预报数据
                fcst_valid_data = data_file.iloc[:, :18][all_dates.isin(valid_date)]
                fcst_valid_arr = fcst_valid_data.values    # shape: nday, nmember

                # 计算验证集的 ensmean anomaly 和 MD
                fcst_ensmeanano_valid = fcst_valid_arr / np.nanmean(fcst_valid_arr, axis=1, keepdims=True)

                fcst_diff_valid = np.abs(fcst_valid_arr[:, :, np.newaxis] - fcst_valid_arr[:, np.newaxis, :])
                fcst_MD_valid = np.nansum(fcst_diff_valid, axis=(1, 2)) / (fcst_diff_valid.shape[-1] ** 2)

                # 计算CondCSGD分布参数
                a1 = par_reg[y, m, 0, 0, 0, 0]
                a2 = par_reg[y, m, 0, 1, 0, 0]
                a3 = par_reg[y, m, 0, 2, 0, 0]
                a4 = par_reg[y, m, 0, 3, 0, 0]
                a5 = par_reg[y, m, 0, 4, 0, 0]
                m = par_reg[y, m, 0, 5:5+M, 0, 0]
                m = np.array(m).reshape(1, M)
                eps = 1e-6
                mu_vec = muc * np.log1p( np.expm1(a1) * (a2 + np.sum(m * fcst_ensmeanano_valid, axis=1)) ) / (a1+eps)
                sigma_vec = a3 * sigmac * (mu_vec / (muc+eps))**a4 + a5 * fcst_MD_valid
                shift_vec = np.ones_like(mu_vec) * shiftc   # nday

                mu_CondDis_list.extend(mu_vec)   # shape: (nyear*nmonth*nday)
                sigma_CondDis_list.extend(sigma_vec)
                shift_CondDis_list.extend(shift_vec)

                print(f'------ Year {_y} Month {_m} Timescale {_t} has finished! ------')


        # 循环结束
        # 每个tscale都分别存储结果

        # 1. 存储回归参数结果
        LinearReg_par_savepath = f'/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/01_EMOS_py/LinearReg_param_20012013_monthly/{_t}days/'
        os.makedirs(LinearReg_par_savepath, exist_ok=True)
        LinearReg_par_filename = LinearReg_par_savepath + f'par_lat{grid_lat}_lon{grid_lon}.nc'
        par_reg_da = xr.DataArray(
            par_reg,
            coords={
                'year': year,
                'month': month,
                'tscale': [_t],
                'par': np.arange(23),
                'lat': [grid_lat],
                'lon': [grid_lon]
            }, dims=['year', 'month', 'tscale', 'par', 'lat', 'lon']
        )
        par_reg_da.to_netcdf(LinearReg_par_filename)
        par_reg_da.close()

        # 2. 存储CondCSGD分布参数结果
        mu_CondDis_arr = np.array(mu_CondDis_list).reshape((1, -1))   # shape: (ntscale, nday)
        sigma_CondDis_arr = np.array(sigma_CondDis_list).reshape((1, -1))
        shift_CondDis_arr = np.array(shift_CondDis_list).reshape((1, -1))
        # 将三个参数组合成一个array
        par_CondDis_arr = np.stack((mu_CondDis_arr, sigma_CondDis_arr, shift_CondDis_arr), 
                                    axis=2)   # shape: (1, nday, 3)
        # 最后添加两个维度，存放经纬度
        par_CondDis_arr = par_CondDis_arr[..., None, None]   # shape: (1, nday, 3, 1, 1)
        # 存储地址
        CondCSGD_par_savepath = f'/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/01_EMOS_py/CondCSGD_param_20012013_commonIssuetime/{_t}days/'
        os.makedirs(CondCSGD_par_savepath, exist_ok=True)
        CondCSGD_par_filename = CondCSGD_par_savepath + f'par_lat{grid_lat}_lon{grid_lon}.nc'
        par_CondDis_da = xr.DataArray(
            par_CondDis_arr,
            coords={
                'tscale': [_t],
                'issuetime': all_dates,
                'par': np.arange(3),
                'lat': [grid_lat],
                'lon': [grid_lon]
            }, dims=['tscale', 'issuetime', 'par', 'lat', 'lon']
        )
        par_CondDis_da.to_netcdf(CondCSGD_par_filename)
        par_CondDis_da.close()


        print(f'------------------------- Grid No. {igrid} has completed! -------------------------')
    






if __name__ == '__main__':

    time1 = time.time()

    # 读取经纬度csv文件
    coords_file = pd.read_csv('/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/CN_land_coords/land_coords.csv')
    ngrid = len(coords_file['lat'])

    n_process = 40
    pool = Pool(n_process)
    jobs = np.arange(1,ngrid)
    pool.map_async(multiprocess_Fit_crps_CondCSGD, jobs)
    print('##################################################################')
    print('      Func: multiprocess_Fit_crps_CondCSGD has completed !       ')
    print('##################################################################')
    pool.close()
    pool.join()

    time2 = time.time()
    print(time2 - time1)
    




# %%
