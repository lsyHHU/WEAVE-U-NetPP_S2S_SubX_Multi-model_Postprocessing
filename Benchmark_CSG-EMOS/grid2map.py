

# func_tool: 用于将评价指标的 [igrid, ilead] .npy文件，转换为带经纬度的 [ilead, ilat, ilon] .nc文件


#%%

import os
import pandas as pd
import numpy as np
import xarray as xr


coords_file = pd.read_csv('/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/CN_land_coords/land_coords.csv')
lat, lon = coords_file['lat'], coords_file['lon']
ngrid = len(lat)   # 总格点数


lat_list = np.linspace(10, 57, 48)
nlat = len(lat_list)
lon_list = np.linspace(65, 144, 80)
nlon = len(lon_list)

tscale = [7, 14, 21, 28]
ntscale = len(tscale)

# 时间尺度循环
for t in tscale:

    common_issue = np.load(f'/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/common_issue/common_issue_{t}days.npy', allow_pickle=True)
    nday = len(common_issue)
    n_mem = 100

    # 该时间尺度的集合预报空间分布
    ens_map = np.full((nday, n_mem, 1, nlat, nlon), np.nan)

    # 网格循环
    for ilat in range(nlat):
        for ilon in range(nlon):
            
            _lat = lat_list[ilat]
            _lon = lon_list[ilon]
            
            ens_path = f'/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/01_EMOS_py/predict_ensfcst100/{t}days/'
            ens_filename = ens_path + f'ensfcst_lat{_lat}_lon{_lon}.nc'

            if not os.path.exists(ens_filename):
                print(f'ensfcst_lat{_lat}_lon{_lon}.nc NOT exist!')
                pass
            else:
                ens_1grid = xr.open_dataarray(ens_path + f'ensfcst_lat{_lat}_lon{_lon}.nc').squeeze()
                ens_1grid.close()   # (issuetime, ens)
                print(f'ensfcst_lat{_lat}_lon{_lon}.nc exist!')
                ens_map[:, :, 0, ilat, ilon] = ens_1grid.to_numpy()
     
    # 存成nc
    ens_map_da = xr.DataArray(
        ens_map,
        coords={
            'issuetime': pd.to_datetime(common_issue),
            'ens': np.arange(n_mem),
            'tscale': [t],
            'lat': lat_list,
            'lon': lon_list
        }, dims=['issuetime', 'ens', 'tscale', 'lat', 'lon']
    )
    savepath = '/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/01_EMOS_py/predict_ensfcst100_map/'
    os.makedirs(savepath, exist_ok=True)
    ens_map_da.to_netcdf(savepath + f'ensfcst_{t}daysTP.nc')
    print(f'tscale: {t} done!')


















# %%
