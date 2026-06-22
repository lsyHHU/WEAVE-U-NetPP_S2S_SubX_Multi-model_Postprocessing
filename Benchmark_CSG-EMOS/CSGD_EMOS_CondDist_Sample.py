#%%

import os
import time
import xarray as xr
import numpy as np
import pandas as pd
from multiprocessing import Pool

import sys
sys.path.append('/home/liusy/MachineLearning_CSGD/S2S_SubX_Multimodel_250525/EMOS_py')
from CSGD_EMOS_auxiliary_func import qctg



def multiprocess_CondCSGD_Sample(igrid):
# for igrid in range(1):

    print('------------------Func: multiprocess_CondCSGD_Sample is running! ------------------')
    print(f'------------------------- Grid No. {igrid} has begun! -------------------------')

    year = np.arange(2001, 2014)
    nyear = len(year)

    tscale = [7, 14, 21, 28]
    ntscale = len(tscale)

    # 读取经纬度csv文件
    coords_file = pd.read_csv('/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/CN_land_coords/land_coords.csv')
    grid_lat, grid_lon = coords_file['lat'][igrid], coords_file['lon'][igrid]

    # 时间尺度循环
    for t in range(ntscale):
        _t = tscale[t]

        # 读取当前时间尺度、当前网格的CondCSGD分布参数
        CondCSGD_par_savepath = '/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/01_EMOS_py/CondCSGD_param_20012013_commonIssuetime/'
        CondCSGD_par_da = xr.open_dataarray(CondCSGD_par_savepath + f'{_t}days/par_lat{grid_lat}_lon{grid_lon}.nc').squeeze()
        CondCSGD_par_da.close()    # issuetime, par

        mu_vec = CondCSGD_par_da.sel(par=0).to_numpy()   # shape: (nday,)
        sigma_vec = CondCSGD_par_da.sel(par=1).to_numpy()
        shift_vec = CondCSGD_par_da.sel(par=2).to_numpy()

        # 天数
        nday = len(CondCSGD_par_da.issuetime)
        # 集合成员数
        n_member = 100

        assert len(mu_vec) == len(sigma_vec) == len(shift_vec) == nday

        # 定义集合预报
        ensfcst = np.zeros((nday, n_member, 1, 1, 1))   # 最后三个维度分别存储时间尺度、经纬度

        # 等百分比抽样生成集合预报
        # sample from quantiles of 1/(n_member+1), 2/(n_member+1), ..., n_member/(n_member+1)
        for i_mem in range(n_member):
            ensfcst[:, i_mem, 0, 0, 0] = qctg((i_mem+1)/(n_member+1), mu_vec, sigma_vec, shift_vec)


        # 保存集合预报
        ensfcst_savepath = f'/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/01_EMOS_py/predict_ensfcst100/{_t}days/'
        os.makedirs(ensfcst_savepath, exist_ok=True)
        ensfcst_filename = ensfcst_savepath + f'ensfcst_lat{grid_lat}_lon{grid_lon}.nc'
        ensfcst_da = xr.DataArray(
            ensfcst,
            coords={
                'issuetime': CondCSGD_par_da.issuetime,
                'member': np.arange(n_member),
                'tscale': [_t],
                'lat': [grid_lat],
                'lon': [grid_lon]
            }, dims=['issuetime', 'member', 'tscale', 'lat', 'lon']
        )
        ensfcst_da.to_netcdf(ensfcst_filename)
        ensfcst_da.close()

        
        print(f'------------------------- Grid No. {igrid} Timescale {_t} has completed! -------------------------')






if __name__ == '__main__':

    time1 = time.time()

    # 读取经纬度csv文件
    coords_file = pd.read_csv('/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/CN_land_coords/land_coords.csv')
    ngrid = len(coords_file['lat'])

    n_process = 10
    pool = Pool(n_process)
    jobs = np.arange(ngrid)
    pool.map_async(multiprocess_CondCSGD_Sample, jobs)
    print('##################################################################')
    print('      Func: multiprocess_CondCSGD_Sample has completed !       ')
    print('##################################################################')
    pool.close()
    pool.join()

    time2 = time.time()
    print(time2 - time1)











# %%
