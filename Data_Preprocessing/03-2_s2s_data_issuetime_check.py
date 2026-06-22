#%%
"""

"""

import xarray as xr
import numpy as np
import pandas as pd

datadir = '/data/selfdata/datalsy/SubX_Multimodel_250525/S2S/'

model_var = {
    'name': ['BoM', 'CMA', 'CNRM', 'CPTEC', 'ECCC', 'ECMWF', 'HMCR', 'IAPCAS', 'ISACCNR', 'KMA', 'UKMO'],
    'var': [
        ['pr', 'tas', 'tcc', 'tcw', 'mslp', 'zg200', 'zg500', 'zg850', 'sh200', 'sh500', 'sh850'],
        ['pr', 'tas', 'tcc', 'tcw', 'mslp', 'zg200', 'zg500', 'zg850', 'sh200', 'sh500', 'sh850'],
        ['pr', 'tas', 'tcc', 'tcw', 'mslp', 'zg200', 'zg500', 'zg850', 'sh200', 'sh500', 'sh850'],
        ['pr', 'tas', 'tcc', 'tcw', 'mslp', 'zg200', 'zg500', 'zg850', 'sh200', 'sh500', 'sh850'],
        # ['pr', 'tas', 'mslp', 'zg200', 'zg500', 'zg850', 'sh200', 'sh500', 'sh850'],
        ['pr', 'mslp'],
        ['pr', 'tas', 'tcc', 'tcw', 'mslp', 'zg200', 'zg500', 'zg850', 'sh200', 'sh500', 'sh850'],
        ['pr', 'tas', 'tcc', 'mslp', 'zg200', 'zg500', 'zg850', 'sh200', 'sh500', 'sh850'],
        ['pr', 'tas', 'tcc', 'zg200', 'zg500', 'zg850', 'sh200', 'sh500', 'sh850'],
        ['pr', 'tas', 'tcc', 'mslp', 'zg200', 'zg500', 'zg850'],
        ['pr', 'tas', 'tcc', 'mslp', 'zg200', 'zg500', 'zg850', 'sh200', 'sh500', 'sh850'],
        ['pr', 'tas', 'tcc', 'zg200', 'zg500', 'zg850', 'sh200', 'sh500', 'sh850'],
    ]
}


model_name = model_var['name']
model_var = model_var['var']


t = 7


# 模型循环
for m in range(2, len(model_name)):

    # 变量循环
    issuedate_list = []
    for v in range(len(model_var[m])):

        _m = model_name[m]
        _v = model_var[m][v]

        print('Model:', _m)
        print('Variable:', _v)

        # 读取数据
        da = xr.open_dataarray(datadir + f'/{_m}/{_m}_{_v}_{t}days.nc')
        da.close()

        # 检查是否存在nan
        print('Nan check:')
        print(np.isnan(da).any())

        # 时间检查
        print('Time length:', len(da.issuetime))

        print('##################################################################')




#%%

da1 = xr.open_dataarray(datadir + f'/UKMO/UKMO_pr_7days.nc')
# da2 = xr.open_dataarray(datadir + f'/GEOS_V2p1/GEOS_V2p1_tas_7days.nc')
# da3 = xr.open_dataarray(datadir + f'/GEOS_V2p1/GEOS_V2p1_zg200_7days.nc')
# da4 = xr.open_dataarray(datadir + f'/GEOS_V2p1/GEOS_V2p1_zg500_7days.nc')

# issue = np.intersect1d(da1.issuetime.values, da2.issuetime.values)
# issue = np.intersect1d(issue, da3.issuetime.values)
# issue = np.intersect1d(issue, da4.issuetime.values)



issue = da1.issuetime.values
np.save(datadir + f'/UKMO/UKMO_issuelist_7days.npy', issue)



# %%
