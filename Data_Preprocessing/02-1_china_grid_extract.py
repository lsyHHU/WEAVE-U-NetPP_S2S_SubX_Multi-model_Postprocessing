#%%
"""
通过CN05.1观测降水数据，将中国陆地区域网格坐标提取出来，并存成为csv文件
"""

import xarray as xr
import numpy as np
import pandas as pd


obs = xr.open_dataarray('/data/selfdata/datalsy/SubX_Multimodel_250525/obs/CN05.1_totalPre_1995_2015_7days_1x1.nc')
obs.close()

lat = obs.lat.values
lon = obs.lon.values

land_coords = []
for ilat in range(len(lat)):
    for ilon in range(len(lon)):
        if obs.values[0, ilat, ilon] != np.nan: # 筛选出有观测的网格
            land_coords.append([lat[ilat], lon[ilon]])


# 存成csv文件
df = pd.DataFrame(land_coords, columns=['lat', 'lon'])
df.to_csv('/data/selfdata/datalsy/SubX_Multimodel_250525/CN_land_coords/land_coords.csv', index=False)



# %%
"""
在 Lat10-57, Lon65-144 的区域内，生成一个中国陆地区域的0-1mask，用于方便NN计算损失
"""

import xarray as xr
import numpy as np
import pandas as pd


obs = xr.open_dataarray('/data/selfdata/datalsy/SubX_Multimodel_250525/obs/CN05.1_totalPre_1995_2015_7days_1x1.nc')
obs.close()  # lat: 14-56, lon: 69-141

lat_range = slice(10, 57)  
lon_range = slice(65, 144)

# 对 obs_data 进行经纬度重索引
obs_expanded = obs.reindex_like(
    obs.sel(lat=lat_range, lon=lon_range).reindex(
        lat=np.arange(10, 57.1, 1),  
        lon=np.arange(65, 144.1, 1),
        method=None,
        fill_value=np.nan
    ),
    method=None,
    fill_value=np.nan
)

# 创建 mask：有值为 1，nan 为 0
mask = xr.where(~np.isnan(obs_expanded), 1, 0)
mask_1t = mask.isel(time=0).to_numpy()
np.save('/data/selfdata/datalsy/SubX_Multimodel_250525/CN_land_coords/land_mask.npy', mask_1t)








# %%
