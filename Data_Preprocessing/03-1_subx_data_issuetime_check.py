#%%
"""

"""

import xarray as xr
import numpy as np
import pandas as pd

datadir = '/data/selfdata/datalsy/SubX_Multimodel_250525/SubX/'

model_var = {
    'name': ['46LCESM1', 'CCSM4', 'CFSv2', 'FIMr1p1', 'GEFS', 'GEM', 'GEOS_V2p1'],
    'var': [
        ['pr', 'tas', 'zg200', 'zg500'],
        ['pr', 'tas', 'zg200', 'zg500', 'sh850'],
        ['pr', 'tas', 'zg200', 'zg500'],
        ['pr', 'tas', 'zg200', 'zg500', 'zg850', 'sh850'],
        ['pr', 'tas', 'zg200', 'zg500', 'zg850', 'sh850'],
        ['pr', 'tas', 'zg200', 'zg500', 'sh850'],
        ['pr', 'tas', 'zg200', 'zg500']
    ]
}


model_name = model_var['name']
model_var = model_var['var']

# 模型循环
# for m in range(len(model_name)):
m = 6
t = 7

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

da1 = xr.open_dataarray(datadir + f'/GEOS_V2p1/GEOS_V2p1_pr_7days.nc')
da2 = xr.open_dataarray(datadir + f'/GEOS_V2p1/GEOS_V2p1_tas_7days.nc')
da3 = xr.open_dataarray(datadir + f'/GEOS_V2p1/GEOS_V2p1_zg200_7days.nc')
da4 = xr.open_dataarray(datadir + f'/GEOS_V2p1/GEOS_V2p1_zg500_7days.nc')

issue = np.intersect1d(da1.issuetime.values, da2.issuetime.values)
issue = np.intersect1d(issue, da3.issuetime.values)
issue = np.intersect1d(issue, da4.issuetime.values)



# issue = da1.issuetime.values
np.save(datadir + f'/GEOS_V2p1/GEOS_V2p1_issuelist_7days.npy', issue)



# %%
