#%%

import numpy as np
import xarray as xr



tscale = [7, 14, 21, 28]


# 打开地理nc文件
dem_nc = xr.open_dataarray('/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/geo/dem.nc')
lat_nc = xr.open_dataarray('/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/geo/lat.nc')
lon_nc = xr.open_dataarray('/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/geo/lon.nc')

# 时间尺度循环
for t in tscale:
    # 打开common issue文件
    common_issue = np.load(f'/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/common_issue/common_issue_{t}days.npy', allow_pickle=True)
    nday = len(common_issue)

    # 将地理nc文件转换成array
    dem = dem_nc.values
    lat = lat_nc.values
    lon = lon_nc.values

    # 每个array复制nday份
    dem = np.tile(dem, (nday, 1, 1))[:, np.newaxis, ...]
    lat = np.tile(lat, (nday, 1, 1))[:, np.newaxis, ...]
    lon = np.tile(lon, (nday, 1, 1))[:, np.newaxis, ...]

    # 在axis=1拼接
    geo_arr = np.concatenate([dem, lat, lon], axis=1)
    np.save(f'/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/00_npy/geo_{t}days.npy', geo_arr)

    print(t)














# %%
