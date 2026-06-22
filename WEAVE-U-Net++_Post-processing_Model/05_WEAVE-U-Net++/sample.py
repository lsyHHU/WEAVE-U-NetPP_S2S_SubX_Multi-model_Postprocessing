
#%%
import os
import xarray as xr
import numpy as np
import pandas as pd
from scipy.stats import gamma
from scipy.interpolate import griddata


# code_name = 'UNetPP-A-WeightPredictor-T-KLloss-ContinueTraining_4'
# code_name = 'UNetPP-A-WeightPredictor-T-new_1'
# code_name = 'UNetPP-A-WeightPredictor-T-KLloss-ContinueTraining-new_3'
# code_name = 'UNetPP-A-SimpleConvWeightPredictor-T'
# code_name = 'UNetPP-A-SkillBasedWeight-NoWeightLearning'
# code_name = 'UNetPP-A-SimpleConvWeightPredictor-T-KLloss-normCRPS'
# code_name = 'UNetPP-A-PixelWiseQueryAttentionWeightPredictor-T'
# code_name = 'UNetPP-A-LocalGridAttentionWeightPredictor-T'
code_name = 'UNetPP-A-LocalGridAttentionWeightPredictor-T-KLloss-ContinueTraining'

# tscale = [7, 14, 21, 28]
tscale = [21, 28]

year = np.arange(2001,2014)
nyear = len(year)


for ib in range(1):
    
    file_path = f'/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/00_Results_newTest_251209/{code_name}/Output_{ib}/'
 
    for t in tscale:

        # 年份循环
        # 分别打开每一年的参数文件，然后进行抽样，生成每一年的后处理集合预报
        for iyr in range(nyear):
        
            yr = year[iyr]   # 当前年份
 
            # 打开CSGD参数文件
            par_path = file_path + f'{t}days/Unetfcst_CSGDPar/par_CV{iyr}.nc'
            par_da = xr.open_dataarray(par_path)
            par_da.close()
            # issuetime, tscale, par, lat, lon
 
            shift = par_da.sel(par='shift').to_numpy().squeeze()    # (issuetime, lat, lon)
            mu = par_da.sel(par='mu').to_numpy().squeeze()
            sigma = par_da.sel(par='sigma').to_numpy().squeeze()

            shape = (mu / (sigma+1e-6)) ** 2
            scale = (sigma**2) / (mu+1e-6)
 
            # 定义空数组存放100个成员的后处理预报
            nday = len(par_da.issuetime)
            nlat = len(par_da.lat)
            nlon = len(par_da.lon)
            nens = 100
            ensfcst = np.zeros((nday, 1, nens, nlat, nlon))
 
            # 百分位用于抽样
            bin_quantile = 1 / nens
            quantile_vec = np.linspace(bin_quantile/2, 1-bin_quantile/2, nens)
            # 抽样
            for iday in range(nday):
                for ilat in range(nlat):
                    for ilon in range(nlon):
                        
                        # 如果当前网格的参数为nan
                        if np.isnan(shift[iday, ilat, ilon]):
                            ensfcst[iday, 0, :, ilat, ilon] = np.nan
                        else:
                            eps = 1e-6
                            ensfcst[iday, 0, :, ilat, ilon] = np.maximum(
                                0, 
                                shift[iday, ilat, ilon] + gamma.ppf(quantile_vec, 
                                                                    scale=scale[iday, ilat, ilon] + eps, 
                                                                    a=shape[iday, ilat, ilon] + eps)
                            )
                print(iday)
            # 集合平均
            ensfcst_mean = np.nanmean(ensfcst, axis=2)
 
            # 存成nc
            ensfcst_da = xr.DataArray(
                ensfcst,
                coords={
                    'issuetime': par_da.issuetime,
                    'tscale': [t],
                    'ens': np.arange(100),
                    'lat': par_da.lat,
                    'lon': par_da.lon
                },
                dims=['issuetime', 'tscale', 'ens', 'lat', 'lon']
            )
            ens_savepath = file_path + f'{t}days/Unetfcst_ENS/'
            if not os.path.exists(ens_savepath):
                os.makedirs(ens_savepath)
            ensfcst_da.to_netcdf(ens_savepath + f'ensfcst_Year{yr}.nc')
            print('ENS saving completed !')
 
            ensfcst_mean_da = xr.DataArray(
                ensfcst_mean,
                coords={
                    'issuetime': par_da.issuetime,
                    'tscale': [t],
                    'lat': par_da.lat,
                    'lon': par_da.lon
                },
                dims=['issuetime', 'tscale', 'lat', 'lon']
            )
            ensmean_savepath = file_path + f'{t}days/Unetfcst_MEAN/'
            if not os.path.exists(ensmean_savepath):
                os.makedirs(ensmean_savepath)
            ensfcst_mean_da.to_netcdf(ensmean_savepath + f'ensmean_Year{yr}.nc')
            print('MEAN saving completed !')

            print(f'Year {yr} has completed !')

        # 所有年份合并
        ens_path = file_path + f'{t}days/Unetfcst_ENS/'
        ens_combine = xr.open_mfdataset(ens_path + '*.nc')
        ens_combine.to_netcdf(ens_path + f'ensfcst_pr_{t}days_Year20012013.nc')
        print(t, 'ens done')

        mean_path = file_path + f'{t}days/Unetfcst_MEAN/'
        mean_combine = xr.open_mfdataset(mean_path + '*.nc')
        mean_combine.to_netcdf(mean_path + f'ensmean_pr_{t}days_Year20012013.nc')
        print(t, 'mean done')

        # 如果是不同模型不同权重的，把权重文件也做合并
        if ('LW' in code_name) or ('Weight' in code_name):
            weight_path = file_path + f'{t}days/Unetfcst_w/'
            w_combine = xr.open_mfdataset(weight_path + '*.nc')
            w_combine.to_netcdf(weight_path + f'weights_{t}days_Year20012013.nc')
            print(t, 'weight done')

    print(ib, 'done')







# %%
