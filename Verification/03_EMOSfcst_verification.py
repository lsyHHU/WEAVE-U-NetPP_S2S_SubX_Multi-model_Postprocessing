#%%

import os, sys
import xarray as xr
import numpy as np
import pandas as pd
sys.path.append('/home/liusy')
from pyNMME import module_verification

def get_ETS(obs, fcst, threshold, method):
    if method == 'right':
        obs = np.where(obs >= threshold, 1, 0)
        fcst = np.where(fcst >= threshold, 1, 0)
    elif method == 'left':
        obs = np.where(obs <= threshold, 1, 0)
        fcst = np.where(fcst <= threshold, 1, 0)

    hits = np.sum((obs == 1) & (fcst == 1))
    misses = np.sum((obs == 1) & (fcst == 0))
    false_alarms = np.sum((obs == 0) & (fcst == 1))
    correct_negatives = np.sum((obs == 0) & (fcst == 0))

    num = (hits + misses) * (hits + false_alarms)
    den = hits + misses + false_alarms + correct_negatives
    Dr = num / (den + 1e-6)

    ETS = (hits - Dr) / ((hits + misses + false_alarms - Dr) + 1e-6)

    return ETS

def get_ACC(fcst, clima, obs):
    
    """
    计算距平相关系数
    fcst, clima, obs: shape: (n,)
    """
    
    fcst_anomaly = fcst - clima
    obs_anomaly = obs - clima
    
    fcst_a1 = fcst_anomaly - np.nanmean(fcst_anomaly)
    obs_a1 = obs_anomaly - np.nanmean(obs_anomaly)
    
    a1 = np.nansum(fcst_a1 * obs_a1)
    a2 = np.sqrt(np.nansum(fcst_anomaly**2)) * np.sqrt(np.nansum(obs_anomaly**2))
    acc = a1 / a2
    
    return acc

def get_TCC(fcst, obs):
    """
    fcst, obs: numpy array of shape (time, lat, lon)
    """
    # 去均值
    fcst_mean = np.nanmean(fcst, axis=0)
    obs_mean = np.nanmean(obs, axis=0)
    fcst_anom = fcst - fcst_mean
    obs_anom = obs - obs_mean
    # 分子：协方差
    numerator = np.nansum(fcst_anom * obs_anom, axis=0)
    # 分母：标准差乘积
    denominator = np.sqrt(np.nansum(fcst_anom**2, axis=0) * np.nansum(obs_anom**2, axis=0))
    # 避免除以0
    with np.errstate(divide='ignore', invalid='ignore'):
        tcc = np.where(denominator == 0, np.nan, numerator / denominator)
    return tcc

def get_SCC(fcst, obs):
    """
    fcst, obs: numpy array of shape (time, lat, lon)
    Returns: array of SCC values for each time step (length T)
    """
    T = fcst.shape[0]
    fcst_flat = fcst.reshape(T, -1)   # (time, lat*lon)
    obs_flat = obs.reshape(T, -1)
    fcst_mean = np.nanmean(fcst_flat, axis=1, keepdims=True)
    obs_mean = np.nanmean(obs_flat, axis=1, keepdims=True)
    fcst_anom = fcst_flat - fcst_mean
    obs_anom = obs_flat - obs_mean
    numerator = np.nansum(fcst_anom * obs_anom, axis=1)
    denominator = np.sqrt(np.nansum(fcst_anom**2, axis=1) * np.nansum(obs_anom**2, axis=1))
    with np.errstate(divide='ignore', invalid='ignore'):
        scc = np.where(denominator == 0, np.nan, numerator / denominator)
    return scc  # shape (T,)


####################################################################################
####################################################################################
####################################################################################

code_name = 'EMOS'

tscale = [7, 14, 21, 28]


for t in tscale:

    # issuetime
    common_issue = np.load(f'/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/common_issue/common_issue_{t}days.npy', allow_pickle=True)
    T = len(common_issue)

    # 读取气候态预报
    climofcst = xr.open_dataarray(f'/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/obs_climo/CN05.1_totalPre_climo_2001_2013_{t}days_1x1.nc').squeeze()
    climofcst.close()  # (issuetime, ens, lat, lon)

    # 读取MSWEP观测数据
    obs = xr.open_dataarray(f'/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/obs/CN05.1_totalPre_1995_2015_{t}days_1x1.nc')
    obs.close()   # (time, lat, lon)

    # 经纬度范围
    lat = np.linspace(10, 57, 48)
    nlat = len(lat)
    lon = np.linspace(65, 144, 80)
    nlon = len(lon)

    # 年份序列
    year = np.arange(2001, 2014)
    nyear = len(year)

    # 定义空数组，存储评价结果
    pcc = np.zeros((1, nyear+1, 1, nlat, nlon))
    acc = np.zeros((1, nyear+1, 1, nlat, nlon))
    rmse = np.zeros((1, nyear+1, 1, nlat, nlon))
    rb = np.zeros((1, nyear+1, 1, nlat, nlon))
    crpss = np.zeros((1, nyear+1, 1, nlat, nlon))
    rpss = np.zeros((1, nyear+1, 1, nlat, nlon))
    aindex = np.zeros((1, nyear+1, 1, nlat, nlon))
    nse = np.zeros((1, nyear+1, 1, nlat, nlon))
    kge = np.zeros((1, nyear+1, 1, nlat, nlon))
    ets = np.zeros((1, nyear+1, 4, 1, nlat, nlon))
    rocss = np.zeros((1, nyear+1, 4, 1, nlat, nlon))
    bss = np.zeros((1, nyear+1, 4, 1, nlat, nlon))
    tcc = np.zeros((1, nyear+1, 1, nlat, nlon))

    for ilat in range(nlat):
        for ilon in range(nlon):

            postfcst_savepath = f'/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/01_EMOS_py_test1/predict_ensfcst100_test2/{t}days/'
            postfcst_filename = postfcst_savepath + f'ensfcst_lat{lat[ilat]}_lon{lon[ilon]}.nc'

            obs_1grid = obs.sel(lat=lat[ilat], lon=lon[ilon])

            if (not os.path.exists(postfcst_filename)) or (np.isnan(obs_1grid).any()):
                pcc[0, :, 0, ilat, ilon] = np.nan
                acc[0, :, 0, ilat, ilon] = np.nan
                rmse[0, :, 0, ilat, ilon] = np.nan
                rb[0, :, 0, ilat, ilon] = np.nan
                crpss[0, :, 0, ilat, ilon] = np.nan
                rpss[0, :, 0, ilat, ilon] = np.nan
                aindex[0, :, 0, ilat, ilon] = np.nan
                nse[0, :, 0, ilat, ilon] = np.nan
                kge[0, :, 0, ilat, ilon] = np.nan
                ets[0, :, :, 0, ilat, ilon] = np.nan
                rocss[0, :, :, 0, ilat, ilon] = np.nan
                bss[0, :, :, 0, ilat, ilon] = np.nan
                tcc[0, :, 0, ilat, ilon] = np.nan
                print(f'No postfcst data at lat{lat[ilat]}_lon{lon[ilon]}')
            
            else:
                # 读取预报数据
                postfcst = xr.open_dataarray(postfcst_filename).squeeze()
                postfcst.close()   # issuetime, member

                if np.isnan(postfcst).any():
                    pcc[0, :, 0, ilat, ilon] = np.nan
                    acc[0, :, 0, ilat, ilon] = np.nan
                    rmse[0, :, 0, ilat, ilon] = np.nan
                    rb[0, :, 0, ilat, ilon] = np.nan
                    crpss[0, :, 0, ilat, ilon] = np.nan
                    rpss[0, :, 0, ilat, ilon] = np.nan
                    aindex[0, :, 0, ilat, ilon] = np.nan
                    nse[0, :, 0, ilat, ilon] = np.nan
                    kge[0, :, 0, ilat, ilon] = np.nan
                    ets[0, :, :, 0, ilat, ilon] = np.nan
                    rocss[0, :, :, 0, ilat, ilon] = np.nan
                    bss[0, :, :, 0, ilat, ilon] = np.nan
                    tcc[0, :, 0, ilat, ilon] = np.nan
                else:
                    for iyr in range(nyear+1):
                        if iyr == 0:  # All years
                            postfcst_da = postfcst
                        else:
                            postfcst_da = postfcst.sel(issuetime=postfcst.issuetime.dt.year == year[iyr-1])
                        postfcst_arr = postfcst_da.to_numpy()

                        climofcst_da = climofcst.sel(issuetime=np.intersect1d(climofcst.issuetime, postfcst_da.issuetime),
                                                    lat=lat[ilat], lon=lon[ilon])
                        climofcst_arr = climofcst_da.to_numpy()
                        obs_da = obs_1grid.sel(time=np.intersect1d(obs.time, postfcst_da.issuetime))
                        obs_arr = obs_da.to_numpy()

                        pcc[0, iyr, 0, ilat, ilon] = module_verification.pcc(postfcst_arr, obs_arr)
                        acc[0, iyr, 0, ilat, ilon] = get_ACC(fcst=np.nanmean(postfcst_arr, axis=1), 
                                                            clima=np.nanmean(climofcst_arr, axis=1),
                                                            obs=obs_arr)
                        rmse[0, iyr, 0, ilat, ilon] = module_verification.rmse(postfcst_arr, obs_arr)
                        rb[0, iyr, 0, ilat, ilon] = module_verification.rb(postfcst_arr, obs_arr)
                        crpss[0, iyr, 0, ilat, ilon] = module_verification.skill(postfcst_arr, obs_arr, climofcst_arr)     
                        rpss[0, iyr, 0, ilat, ilon] = module_verification.skill(postfcst_arr, obs_arr, climofcst_arr, skill_type='rps')
                        aindex[0, iyr, 0, ilat, ilon] = module_verification.alpha_index(postfcst_arr, obs_arr, censor_value=np.array([0., 0.]))
                        nse[0, iyr, 0, ilat, ilon] = module_verification.nse(postfcst_arr, obs_arr)
                        kge[0, iyr, 0, ilat, ilon] = module_verification.kge(postfcst_arr, obs_arr)
                        tcc[0, iyr, 0, ilat, ilon] = get_TCC(fcst=np.nanmean(postfcst_arr, axis=1), 
                                                            obs=obs_arr)

                        ths = [10, 33.3, 66.7, 90]
                        event = ['left', 'left', 'right', 'right']
                        for its in range(4):
                            ets[0, iyr, its, 0, ilat, ilon] = get_ETS(obs_arr, np.nanmean(postfcst_arr, axis=1),
                                                                        threshold=np.percentile(obs_arr, ths[its]), method=event[its])

                            rocss[0, iyr, its, 0, ilat, ilon] = module_verification.rocss(postfcst_arr, obs_arr, 
                                                                                            threshold=np.percentile(obs_arr, ths[its]), 
                                                                                            event=event[its])

                            postbs = module_verification.bs(postfcst_arr, obs_arr, 
                                                            threshold=np.percentile(obs_arr, ths[its]), event=event[its])
                            climabs = module_verification.bs(climofcst_arr, obs_arr,
                                                            threshold=np.percentile(obs_arr, ths[its]), event=event[its])
                            bss[0, iyr, its, 0, ilat, ilon] = (climabs - postbs) / climabs * 100.
                        

                        print(f'Model: {code_name}, Year: {iyr}, Lat: {ilat}, Lon: {ilon}')


    # 存成nc
    savepath = f'/home/liusy/MachineLearning_CSGD/S2S_SubX_Multimodel_250525/pre_verif_result_EMOS(py)fcst_test2/{t}days/'
    os.makedirs(savepath, exist_ok=True)

    pcc_da = xr.DataArray(
        pcc,
        coords={
            'model': [code_name],
            'year': ['AllYears'] + year.tolist(),
            'tscale': [t],
            'lat': lat,
            'lon': lon
        }, dims=['model', 'year', 'tscale', 'lat', 'lon']
    )
    pcc_da.to_netcdf(savepath + f'emos_pcc_{t}daysTP_Year20012013.nc')

    acc_da = xr.DataArray(
        acc,
        coords={
            'model': [code_name],
            'year': ['AllYears'] + year.tolist(),
            'tscale': [t],
            'lat': lat,
            'lon': lon
        }, dims=['model', 'year', 'tscale', 'lat', 'lon']
    )
    acc_da.to_netcdf(savepath + f'emos_acc_{t}daysTP_Year20012013.nc')

    rmse_da = xr.DataArray(
        rmse,
        coords={
            'model': [code_name],
            'year': ['AllYears'] + year.tolist(),
            'tscale': [t],
            'lat': lat,
            'lon': lon
        }, dims=['model', 'year', 'tscale', 'lat', 'lon']
    )
    rmse_da.to_netcdf(savepath + f'emos_rmse_{t}daysTP_Year20012013.nc')

    rb_da = xr.DataArray(
        rb,
        coords={
            'model': [code_name],
            'year': ['AllYears'] + year.tolist(),
            'tscale': [t],
            'lat': lat,
            'lon': lon
        }, dims=['model', 'year', 'tscale', 'lat', 'lon']
    )
    rb_da.to_netcdf(savepath + f'emos_rb_{t}daysTP_Year20012013.nc')

    crpss_da = xr.DataArray(
        crpss,
        coords={
            'model': [code_name],
            'year': ['AllYears'] + year.tolist(),
            'tscale': [t],
            'lat': lat,
            'lon': lon
        }, dims=['model', 'year', 'tscale', 'lat', 'lon']
    )
    crpss_da.to_netcdf(savepath + f'emos_crpss_{t}daysTP_Year20012013.nc')

    rpss_da = xr.DataArray(
        rpss,
        coords={
            'model': [code_name],
            'year': ['AllYears'] + year.tolist(),
            'tscale': [t],
            'lat': lat,
            'lon': lon
        }, dims=['model', 'year', 'tscale', 'lat', 'lon']
    )
    rpss_da.to_netcdf(savepath + f'emos_rpss_{t}daysTP_Year20012013.nc')

    aindex_da = xr.DataArray(
        aindex,
        coords={
            'model': [code_name],
            'year': ['AllYears'] + year.tolist(),
            'tscale': [t],
            'lat': lat,
            'lon': lon
        }, dims=['model', 'year', 'tscale', 'lat', 'lon']
    )
    aindex_da.to_netcdf(savepath + f'emos_aindex_{t}daysTP_Year20012013.nc')

    nse_da = xr.DataArray(
        nse,
        coords={
            'model': [code_name],
            'year': ['AllYears'] + year.tolist(),
            'tscale': [t],
            'lat': lat,
            'lon': lon
        }, dims=['model', 'year', 'tscale', 'lat', 'lon']
    )
    nse_da.to_netcdf(savepath + f'emos_nse_{t}daysTP_Year20012013.nc')

    kge_da = xr.DataArray(
        kge,
        coords={
            'model': [code_name],
            'year': ['AllYears'] + year.tolist(),
            'tscale': [t],
            'lat': lat,
            'lon': lon
        }, dims=['model', 'year', 'tscale', 'lat', 'lon']
    )
    kge_da.to_netcdf(savepath + f'emos_kge_{t}daysTP_Year20012013.nc')

    ets_da = xr.DataArray(
        ets,
        coords={
            'model': [code_name],
            'year': ['AllYears'] + year.tolist(),
            'threshold': [10, 33.3, 66.7, 90],
            'tscale': [t],
            'lat': lat,
            'lon': lon
        }, dims=['model', 'year', 'threshold', 'tscale', 'lat', 'lon']
    )
    ets_da.to_netcdf(savepath + f'emos_ets_{t}daysTP_Year20012013.nc')

    rocss_da = xr.DataArray(
        rocss,
        coords={
            'model': [code_name],
            'year': ['AllYears'] + year.tolist(),
            'threshold': [10, 33.3, 66.7, 90],
            'tscale': [t],
            'lat': lat,
            'lon': lon
        }, dims=['model', 'year', 'threshold', 'tscale', 'lat', 'lon']
    )
    rocss_da.to_netcdf(savepath + f'emos_rocss_{t}daysTP_Year20012013.nc')

    bss_da = xr.DataArray(
        bss,
        coords={
            'model': [code_name],
            'year': ['AllYears'] + year.tolist(),
            'threshold': [10, 33.3, 66.7, 90],
            'tscale': [t],
            'lat': lat,
            'lon': lon
        }, dims=['model', 'year', 'threshold', 'tscale', 'lat', 'lon']
    )
    bss_da.to_netcdf(savepath + f'emos_bss_{t}daysTP_Year20012013.nc')

    tcc_da = xr.DataArray(
        tcc,
        coords={
            'model': [code_name],
            'year': ['AllYears'] + year.tolist(),
            'tscale': [t],
            'lat': lat,
            'lon': lon
        }, dims=['model', 'year', 'tscale', 'lat', 'lon']
    )
    tcc_da.to_netcdf(savepath + f'emos_tcc_{t}daysTP_Year20012013.nc')

    # scc_da = xr.DataArray(
    #     scc,
    #     coords={
    #         'model': [code_name],
    #         'year': ['AllYears'] + year.tolist(),
    #         'tscale': [t],
    #         'issuetime': pd.to_datetime(postfcst.issuetime),
    #     }, dims=['model', 'year', 'tscale', 'issuetime']
    # )
    # scc_da.to_netcdf(savepath + f'emos_scc_{t}daysTP_Year20012013.nc')

    print('saving results completed !')





# %%
