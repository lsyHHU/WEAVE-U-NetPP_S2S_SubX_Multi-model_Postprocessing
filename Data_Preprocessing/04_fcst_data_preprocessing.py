#%%
"""
S2S SubX Multimodel Forecast Data
"""

import xarray as xr
import numpy as np
import pandas as pd


t = 28

model_var = {
    '46LCESM1': ['pr', 'tas', 'zg200', 'zg500'],
    'CCSM4': ['pr', 'tas', 'mslp', 'zg200', 'zg500', 'sh850'],
    'CFSv2': ['pr', 'tas'],
    'FIMr1p1': ['pr', 'tas', 'mslp', 'zg200', 'zg500', 'zg850', 'sh850'],
    'GEFS': ['pr', 'tas', 'mslp', 'zg200', 'zg500', 'zg850', 'sh850'],
    'NESM': ['pr', 'tas', 'mslp', 'zg200', 'zg500', 'zg850'],
    'GEOS_V2p1': ['pr', 'tas', 'mslp', 'zg200', 'zg500'],
    'BoM': ['pr', 'tas', 'tcc', 'tcw', 'mslp', 'zg200', 'zg500', 'zg850', 'sh200', 'sh500', 'sh850'],
    'CMA': ['pr', 'tas', 'tcc', 'tcw', 'mslp', 'zg200', 'zg500', 'zg850', 'sh200', 'sh500', 'sh850'],
    'CNRM': ['pr', 'tas', 'tcc', 'tcw', 'mslp', 'zg200', 'zg500', 'zg850', 'sh200', 'sh500', 'sh850'],
    'CPTEC': ['pr', 'tas', 'tcc', 'tcw', 'mslp', 'zg200', 'zg500', 'zg850', 'sh200', 'sh500', 'sh850'],
    'ECCC': ['pr', 'tas', 'tcc', 'tcw', 'mslp', 'zg200', 'zg500', 'zg850', 'sh200', 'sh500', 'sh850'],
    'ECMWF': ['pr', 'tas', 'tcc', 'tcw', 'mslp', 'zg200', 'zg500', 'zg850', 'sh200', 'sh500', 'sh850'],
    'HMCR': ['pr', 'tas', 'tcc', 'mslp', 'zg200', 'zg500', 'zg850', 'sh200', 'sh500', 'sh850'],
    'IAPCAS': ['pr', 'tas', 'tcc', 'zg200', 'zg500', 'zg850', 'sh200', 'sh500', 'sh850'],
    'ISACCNR': ['pr', 'tas', 'tcc', 'mslp', 'zg200', 'zg500', 'zg850'],
    'KMA': ['pr', 'tas', 'tcc', 'mslp', 'zg200', 'zg500', 'zg850', 'sh200', 'sh500', 'sh850'],
    'UKMO': ['pr', 'tas', 'tcc', 'zg200', 'zg500', 'zg850', 'sh200', 'sh500', 'sh850'],
}



combined_ndarray = np.zeros((1330, 272, 48, 80))   # 28 days
# combined_ndarray = np.zeros((2050, 272, 48, 80))
n = 0
# 模型循环
for model in model_var.keys():
    # 变量循环
    for var in model_var[model]:
        
        # 打开数据文件
        if model in ['46LCESM1', 'CCSM4', 'CFSv2', 'FIMr1p1', 'GEFS', 'NESM', 'GEOS_V2p1']:
            data_dir = f'/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/SubX/{t}days/'
        else:
            data_dir = f'/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/S2S/{t}days/'
        data_file = xr.open_dataarray(data_dir + f'{model}_{var}_{t}days.nc')

        # 拼接
        if data_file.ndim == 4:  # ensemble 
            _n = len(data_file.number)
        elif data_file.ndim == 3:  # ensemble mean
            data_file = data_file.expand_dims(number=[0]).transpose('issuetime', 'number', 'lat', 'lon')
            _n = 1

        combined_ndarray[:, n:n+_n, :, :] = data_file.values
        n += _n

        print(f'{model}_{var}')


# Nan detection
print("Nan detection: ", np.isnan(combined_ndarray).any())
# Shape detection
print("Shape detection: ", combined_ndarray.shape)
# Save to file
np.save(f'/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/00_npy/S2S_SubX_fcst_{t}days_1.npy', combined_ndarray)




# %%
