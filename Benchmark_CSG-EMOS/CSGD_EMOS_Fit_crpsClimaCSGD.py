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
from CSGD_EMOS_auxiliary_func import calculate_climaCSGD_par


def multiprocess_Fit_crps_ClimaCSGD(igrid):
# for igrid in range(1):

    print('------------------Func: multiprocess_Fit_crps_ClimaCSGD is running! ------------------')
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

    # 定义ClimaCSGD分布参数
    shape_cl = np.zeros((nyear, nmonth, ntscale))
    scale_cl = np.zeros((nyear, nmonth, ntscale))
    shift_cl = np.zeros((nyear, nmonth, ntscale))

    # 率定ClimatologyCSGD分布参数
    # 使用leave-one-year-out cross-validation，每个月份、每个时间尺度、每个网格都拟合一个参数

    # 时间尺度循环
    for t in range(ntscale):
        # 当前时间尺度
        _t = tscale[t]

        # 读取当前时间尺度和当前网格的数据文件
        data_file = pd.read_csv(f'/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/01_BMA/{_t}days/raw_csv/rawfcst_obs_Lat{grid_lat}_Lon{grid_lon}.csv')
        all_dates = pd.to_datetime(data_file['dates'].values)
        
        # 年份循环
        for y in range(nyear):
            # 月份循环
            for m in range(nmonth):
                # 当前年份月份
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

                # 挑选出当前网格、当前时间尺度、训练日期的观测序列
                obs_1grid = data_file['obs'][all_dates.isin(train_dates_idx)]
                obs_1grid_arr = obs_1grid.values

                # 拟合ClimatologyCSGD分布参数
                par_clima = calculate_climaCSGD_par(obs_1grid_arr)
                mu_cl, sigma_cl = par_clima[0], par_clima[1]
                shape_cl[y, m, t] = (mu_cl / sigma_cl) ** 2
                scale_cl[y, m, t] = (sigma_cl ** 2) / mu_cl
                shift_cl[y, m, t] = par_clima[2]

                print(f'------ Year {_y} Month {_m} Timescale {_t} has finished! ------')

    
    # 循环结束

    # 存储climatologyCSGD分布参数
    # shape_cl
    ClimatologyCSGD_shape_savepath = '/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/01_EMOS_py/ClimaCSGD_param_20012013_monthly/shape_cl/'
    os.makedirs(ClimatologyCSGD_shape_savepath, exist_ok=True)
    ClimatologyCSGD_shape_filename = ClimatologyCSGD_shape_savepath + f'par_lat{grid_lat}_lon{grid_lon}.npy'
    np.save(ClimatologyCSGD_shape_filename, shape_cl)

    # scale_cl
    ClimatologyCSGD_scale_savepath = '/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/01_EMOS_py/ClimaCSGD_param_20012013_monthly/scale_cl/'
    os.makedirs(ClimatologyCSGD_scale_savepath, exist_ok=True)
    ClimatologyCSGD_scale_filename = ClimatologyCSGD_scale_savepath + f'par_lat{grid_lat}_lon{grid_lon}.npy'
    np.save(ClimatologyCSGD_scale_filename, scale_cl)

    # shift_cl
    ClimatologyCSGD_shift_savepath = '/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/01_EMOS_py/ClimaCSGD_param_20012013_monthly/shift_cl/'
    os.makedirs(ClimatologyCSGD_shift_savepath, exist_ok=True)
    ClimatologyCSGD_shift_filename = ClimatologyCSGD_shift_savepath + f'par_lat{grid_lat}_lon{grid_lon}.npy'
    np.save(ClimatologyCSGD_shift_filename, shift_cl)

    print(f'------------------------- Grid No. {igrid} has completed! -------------------------')






if __name__ == '__main__':

    time1 = time.time()

    # 读取经纬度csv文件
    coords_file = pd.read_csv('/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/CN_land_coords/land_coords.csv')
    ngrid = len(coords_file['lat'])

    n_process = 20
    pool = Pool(n_process)
    jobs = np.arange(1,ngrid)
    pool.map_async(multiprocess_Fit_crps_ClimaCSGD, jobs)
    print('##################################################################')
    print('      Func: multiprocess_Fit_crps_ClimaCSGD has completed !       ')
    print('##################################################################')
    pool.close()
    pool.join()

    time2 = time.time()
    print(time2 - time1)







# %%
