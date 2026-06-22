
# 计算 CN05.1 tp 观测数据的 气候态预报

#%%

import os
import calendar
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime


# 需要计算的时间尺度
tscale = 28

# 年份序列
year = np.arange(2001, 2014)
nyear = len(year)

# 2001-2013年日期序列
start_date = pd.to_datetime('2001-01-01')
end_date = pd.to_datetime('2013-12-31')
date_range = pd.date_range(start_date, end_date)

nday = len(date_range)   # 总天数
nens = 100   # 气候态预报的集合成员数

# 读取CN05.1 tp观测数据
obs = xr.open_dataarray(f'/data/selfdata/datalsy/SubX_Multimodel_250525/obs/CN05.1_totalPre_1995_2015_{tscale}days_1x1.nc')
obs.close()

# ntscale = len(obs.tscale)
nlat = len(obs.lat)
nlon = len(obs.lon)

# 定义空数组用于存储气候态预报
tp_climo = np.zeros((nday, nens, 1, nlat, nlon))   # 这里先只算1个tscale

# 日期循环
for iday in range(nday):
    # 挑选出以当前日期为中心，前4天 - 后4天在内的9天，且年份为其余年的日期
    central_date = date_range[iday]   # 当前日期
    central_date_list = []    # 2001-2013所有central_date组成的序列

    for year in range(2001, 2014):
        # 判断是否是2.29
        if central_date.month == 2 and central_date.day == 29:
            central_date_list.append(
                pd.Timestamp(f'{year}-02-01') + pd.offsets.MonthEnd(1)
            )    # 2.29的气候态预报用2.28的代替
        else:  # 正常情况
            central_date_list.append(
                pd.Timestamp(f'{year}-{central_date.month}-{central_date.day}')
            )
    # 删掉当前年份的日期
    central_date_list.remove(central_date)

    # central date 循环
    date_window = []
    for i_ct_date in central_date_list:
        date_window_1year = pd.date_range(start=i_ct_date - pd.Timedelta(days=4),
                                            end=i_ct_date + pd.Timedelta(days=4))     # 9 days window
        date_window.extend(date_window_1year)

    # 一共挑出12年*9天 = 108天，取前100天作为气候态预报的100个集合成员
    # 挑选出date_window中的观测数据
    obs_in_date_window = obs.sel(time=date_window).to_numpy()
    # 取前100天
    obs_climo_100day = obs_in_date_window[:100, :, :]
    tp_climo[iday, :, 0, :, :] = obs_climo_100day

    print(central_date)


# 存成nc
tp_climo_reshape_da = xr.DataArray(
    tp_climo,
    coords={
        'issuetime': date_range,
        'ens': np.arange(100),
        'tscale': [tscale],
        'lat': obs.lat,
        'lon': obs.lon
    }, 
    dims=['issuetime', 'ens', 'tscale', 'lat', 'lon']
)
tp_climo_savepath = '/data/selfdata/datalsy/SubX_Multimodel_250525/obs_climo/'
if not os.path.exists(tp_climo_savepath):
    os.makedirs(tp_climo_savepath)
tp_climo_reshape_da.to_netcdf(tp_climo_savepath + f'CN05.1_totalPre_climo_2001_2013_{tscale}days_1x1.nc')

print('climo saving completed !')





# %%







# %%
