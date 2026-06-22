#%%
"""
CN05.1 观测降水数据
"""


import xarray as xr
import numpy as np
import pandas as pd

obs = xr.open_dataarray('/data/selfdata/datalsy/SubX_Multimodel_250525/obs/CN05.1_Pre_1961_2022_daily_1x1.nc')
obs.close()


time = pd.to_datetime(obs.time)
filter_time = time[(time.year >= 1995) & (time.year <= 2016)]
nday = len(filter_time)

# 计算不同时间尺度的累积降水量
tscale = [7, 14, 21, 28]
for t in tscale: 
    acc_pre = np.zeros((nday, len(obs.lat), len(obs.lon)))
    for d in range(nday):
        _d = filter_time[d]
        acc_pre[d, :, :] = obs.sel(time=slice(_d, _d + pd.DateOffset(days=t-1))).sum(dim='time', skipna=False)  # 国界外的值为nan保留
        print(d)

    # 存储
    acc_pre_da = xr.DataArray(
        acc_pre,
        coords={
            'time': filter_time,
            'lat': obs.lat,
            'lon': obs.lon
        }, dims=['time', 'lat', 'lon']
    )
    acc_pre_da.to_netcdf(f'/data/selfdata/datalsy/SubX_Multimodel_250525/obs/CN05.1_totalPre_1995_2015_{t}days_1x1.nc')
    print(t)
    




#%%
"""
将nc文件存成npy
"""

import xarray as xr
import numpy as np
import pandas as pd

t = 28
lat_range = slice(10, 57)  
lon_range = slice(65, 144)

obs = xr.open_dataarray(f'/data/selfdata/datalsy/SubX_Multimodel_250525/obs/CN05.1_totalPre_1995_2015_{t}days_1x1.nc')
obs.close()

# obs_expanded = obs.reindex_like(
#     obs.sel(lat=lat_range, lon=lon_range).reindex(
#         lat=np.arange(10, 57.1, 1),  
#         lon=np.arange(65, 144.1, 1),
#         method=None,
#         fill_value=np.nan
#     ),
#     method=None,
#     fill_value=np.nan
# )

common_issue = pd.to_datetime(
    np.load(f'/data/selfdata/datalsy/SubX_Multimodel_250525/common_issue/common_issue_{t}days.npy', allow_pickle=True)
)
common_obs = obs.sel(time=common_issue)
common_obs_arr = common_obs.values

# Nan detection
# print("Nan detection: ", np.isnan(common_obs_arr).any())
np.save(f'/data/selfdata/datalsy/SubX_Multimodel_250525/00_npy/obs_{t}days.npy', common_obs_arr)




# %%
import numpy as np
import xarray as xr

a = np.load('/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/00_npy/S2S_SubX_fcst_7days_1.npy')
a_arr = a[0, ...]

#%%
a_da = xr.DataArray(
    a_arr,
    coords={
        'channel': np.arange(a_arr.shape[0]),
        'lat': np.linspace(10, 57, 48),
        'lon': np.linspace(65, 144, 80)
    }, dims=['channel', 'lat', 'lon']
)
a_da.to_netcdf('/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/00_npy/S2S_SubX_fcst_7days_2.nc')

# %%
