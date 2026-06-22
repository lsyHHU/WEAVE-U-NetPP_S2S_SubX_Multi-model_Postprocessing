#%%
"""
所有模型的原始降水预报组成的大集合预报（Multimodel ensemble）的技能评价
"""
import sys, os
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

tscale = [7, 14, 21, 28]


for t in tscale:

    # 读取公共预报时间序列
    common_issue = pd.to_datetime(
        np.load(f'/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/common_issue/common_issue_{t}days.npy')
    )

    # 经纬度范围
    lat_list = np.linspace(10, 57, 48)
    nlat = len(lat_list)
    lon_list = np.linspace(65, 144, 80)
    nlon = len(lon_list)

    # 年份序列
    year = np.arange(2001, 2014)
    nyear = len(year)

    T = len(common_issue)

    # 定义空数组，存储评价结果
    pcc = np.zeros((1, nyear+1, 1, nlat, nlon))   # 1model(MME), nyear, ntscale, nlat, nlon
    acc = np.zeros((1, nyear+1, 1, nlat, nlon))
    rmse = np.zeros((1, nyear+1, 1, nlat, nlon))
    rb = np.zeros((1, nyear+1, 1, nlat, nlon))
    crpss = np.zeros((1, nyear+1, 1, nlat, nlon))
    rpss = np.zeros((1, nyear+1, 1, nlat, nlon))
    nse = np.zeros((1, nyear+1, 1, nlat, nlon))
    kge = np.zeros((1, nyear+1, 1, nlat, nlon))
    aindex = np.zeros((1, nyear+1, 1, nlat, nlon))
    ets = np.zeros((1, nyear+1, 4, 1, nlat, nlon))
    rocss = np.zeros((1, nyear+1, 4, 1, nlat, nlon))
    bss = np.zeros((1, nyear+1, 4, 1, nlat, nlon))
    tcc = np.zeros((1, nyear+1, 1, nlat, nlon))
    scc = np.zeros((1, nyear+1, 1, T))

    # 读取原始预报数据
    # fcst_npy = np.load(f'/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/00_npy/S2S_SubX_fcst_{t}days_1.npy')


    # 读取原始预报数据
    MMEfcst = xr.open_dataarray(f'/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/MME/MME_pr_{t}days_1.nc').sel(issuetime=common_issue)
    MMEfcst.close()  # (issuetime, number, lat, lon)

    # 读取观测数据
    obs = xr.open_dataarray(f'/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/obs/CN05.1_totalPre_1995_2015_{t}days_1x1.nc').sel(time=common_issue)
    obs.close()   # # (issuetime, lat, lon)

    # 读取气候态预报数据
    clima = xr.open_dataarray(f'/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/obs_climo/CN05.1_totalPre_climo_2001_2013_{t}days_1x1.nc').sel(issuetime=common_issue).squeeze()
    clima.close()   # # (issuetime, ens, lat, lon)

    # 年份循环
    for iyr in range(nyear+1):
        if iyr == 0:  # all years
            MMEfcst_da = MMEfcst
        else:
            MMEfcst_da = MMEfcst.sel(issuetime=common_issue[common_issue.year == year[iyr-1]])
        MMEfcst_arr = MMEfcst_da.to_numpy()

        obs_da = obs.sel(time=np.intersect1d(obs.time, MMEfcst_da.issuetime))
        obs_arr = obs_da.to_numpy()

        clima_da = clima.sel(issuetime=np.intersect1d(clima.issuetime, MMEfcst_da.issuetime))
        clima_arr = clima_da.to_numpy()

        # 计算评价指标
        for ilat in range(nlat):
            for ilon in range(nlon):
                # 判断当前网格是否存在观测数据，如果没有观测数据则评价指标设置为nan
                if np.isnan(obs_arr[:, ilat, ilon]).any():
                    pcc[0, iyr, 0, ilat, ilon] = np.nan
                    acc[0, iyr, 0, ilat, ilon] = np.nan
                    rmse[0, iyr, 0, ilat, ilon] = np.nan
                    rb[0, iyr, 0, ilat, ilon] = np.nan
                    crpss[0, iyr, 0, ilat, ilon] = np.nan
                    rpss[0, iyr, 0, ilat, ilon] = np.nan
                    nse[0, iyr, 0, ilat, ilon] = np.nan
                    kge[0, iyr, 0, ilat, ilon] = np.nan
                    aindex[0, iyr, 0, ilat, ilon] = np.nan
                    ets[0, iyr, :, 0, ilat, ilon] = np.nan
                    rocss[0, iyr, :, 0, ilat, ilon] = np.nan
                    bss[0, iyr, :, 0, ilat, ilon] = np.nan
                    tcc[0, iyr, 0, ilat, ilon] = np.nan
                
                else:
                    pcc[0, iyr, 0, ilat, ilon] = module_verification.pcc(MMEfcst_arr[:, :, ilat, ilon], obs_arr[:, ilat, ilon])
                    rmse[0, iyr, 0, ilat, ilon] = module_verification.rmse(MMEfcst_arr[:, :, ilat, ilon], obs_arr[:, ilat, ilon])
                    rb[0, iyr, 0, ilat, ilon] = module_verification.rb(MMEfcst_arr[:, :, ilat, ilon], obs_arr[:, ilat, ilon])
                    crpss[0, iyr, 0, ilat, ilon] = module_verification.skill(MMEfcst_arr[:, :, ilat, ilon], obs_arr[:, ilat, ilon], clima_arr[:, :, ilat, ilon])   
                    rpss[0, iyr, 0, ilat, ilon] = module_verification.skill(MMEfcst_arr[:, :, ilat, ilon], obs_arr[:, ilat, ilon], clima_arr[:, :, ilat, ilon], skill_type='rps')
                    nse[0, iyr, 0, ilat, ilon] = module_verification.nse(MMEfcst_arr[:, :, ilat, ilon], obs_arr[:, ilat, ilon])
                    kge[0, iyr, 0, ilat, ilon] = module_verification.kge(MMEfcst_arr[:, :, ilat, ilon], obs_arr[:, ilat, ilon])
                    aindex[0, iyr, 0, ilat, ilon] = module_verification.alpha_index(MMEfcst_arr[:, :, ilat, ilon], obs_arr[:, ilat, ilon], censor_value=np.array([0., 0.]))
                    acc[0, iyr, 0, ilat, ilon] = get_ACC(fcst=np.nanmean(MMEfcst_arr[:, :, ilat, ilon], axis=1), 
                                                        clima=np.nanmean(clima_arr[:, :, ilat, ilon], axis=1),
                                                        obs=obs_arr[:, ilat, ilon])  

                    ths = [10, 33.3, 66.7, 90]
                    event = ['left', 'left', 'right', 'right']
                    for its in range(4):
                        ets[0, iyr, its, 0, ilat, ilon] = get_ETS(obs_arr[:, ilat, ilon], np.nanmean(MMEfcst_arr[:, :, ilat, ilon], axis=1),
                                                                    threshold=np.percentile(obs_arr[:, ilat, ilon], ths[its]), method=event[its])

                        rocss[0, iyr, its, 0, ilat, ilon] = module_verification.rocss(MMEfcst_arr[:, :, ilat, ilon], obs_arr[:, ilat, ilon], 
                                                                                    threshold=np.percentile(obs_arr[:, ilat, ilon], ths[its]), 
                                                                                    event=event[its])

                        fcstbs = module_verification.bs(MMEfcst_arr[:, :, ilat, ilon], obs_arr[:, ilat, ilon], 
                                                        threshold=np.percentile(obs_arr[:, ilat, ilon], ths[its]), event=event[its])
                        climabs = module_verification.bs(clima_arr[:, :, ilat, ilon], obs_arr[:, ilat, ilon],
                                                        threshold=np.percentile(obs_arr[:, ilat, ilon], ths[its]), event=event[its])
                        bss[0, iyr, its, 0, ilat, ilon] = (climabs - fcstbs) / climabs * 100.

                    print(f'MME, Year: {iyr}, Lat: {ilat}, Lon: {ilon}')

        # 当nyear==0时计算SCC（计算所有日期，不分年）
        if iyr == 0:
            scc[0, iyr, 0, :] = get_SCC(fcst=np.nanmean(MMEfcst_arr, axis=1),
                                        obs=obs_arr)


    # 存成nc
    savepath = f'/home/liusy/MachineLearning_CSGD/SubX_Multimodel_250525/pre_verif_result_MMEfcst_1/{t}days/'
    os.makedirs(savepath, exist_ok=True)

    pcc_da = xr.DataArray(
        pcc,
        coords={
            'model': ['MME'],
            'year': ['AllYears'] + year.tolist(),
            'tscale': [t],
            'lat': lat_list,
            'lon': lon_list
        }, dims=['model', 'year', 'tscale', 'lat', 'lon']
    )
    pcc_da.to_netcdf(savepath + f'mme_pcc_{t}daysTP_Year20012013.nc')

    acc_da = xr.DataArray(
        acc,
        coords={
            'model': ['MME'],
            'year': ['AllYears'] + year.tolist(),
            'tscale': [t],
            'lat': lat_list,
            'lon': lon_list
        }, dims=['model', 'year', 'tscale', 'lat', 'lon']
    )
    acc_da.to_netcdf(savepath + f'mme_acc_{t}daysTP_Year20012013.nc')

    rmse_da = xr.DataArray(
        rmse,
        coords={
            'model': ['MME'],
            'year': ['AllYears'] + year.tolist(),
            'tscale': [t],
            'lat': lat_list,
            'lon': lon_list
        }, dims=['model', 'year', 'tscale', 'lat', 'lon']
    )
    rmse_da.to_netcdf(savepath + f'mme_rmse_{t}daysTP_Year20012013.nc')

    rb_da = xr.DataArray(
        rb,
        coords={
            'model': ['MME'],
            'year': ['AllYears'] + year.tolist(),
            'tscale': [t],
            'lat': lat_list,
            'lon': lon_list
        }, dims=['model', 'year', 'tscale', 'lat', 'lon']
    )
    rb_da.to_netcdf(savepath + f'mme_rb_{t}daysTP_Year20012013.nc')

    crpss_da = xr.DataArray(
        crpss,
        coords={
            'model': ['MME'],
            'year': ['AllYears'] + year.tolist(),
            'tscale': [t],
            'lat': lat_list,
            'lon': lon_list
        }, dims=['model', 'year', 'tscale', 'lat', 'lon']
    )
    crpss_da.to_netcdf(savepath + f'mme_crpss_{t}daysTP_Year20012013.nc')

    rpss_da = xr.DataArray(
        rpss,
        coords={
            'model': ['MME'],
            'year': ['AllYears'] + year.tolist(),
            'tscale': [t],
            'lat': lat_list,
            'lon': lon_list
        }, dims=['model', 'year', 'tscale', 'lat', 'lon']
    )
    rpss_da.to_netcdf(savepath + f'mme_rpss_{t}daysTP_Year20012013.nc')

    nse_da = xr.DataArray(
        nse,
        coords={
            'model': ['MME'],
            'year': ['AllYears'] + year.tolist(),
            'tscale': [t],
            'lat': lat_list,
            'lon': lon_list
        }, dims=['model', 'year', 'tscale', 'lat', 'lon']
    )
    nse_da.to_netcdf(savepath + f'mme_nse_{t}daysTP_Year20012013.nc')

    kge_da = xr.DataArray(
        kge,
        coords={
            'model': ['MME'],
            'year': ['AllYears'] + year.tolist(),
            'tscale': [t],
            'lat': lat_list,
            'lon': lon_list
        }, dims=['model', 'year', 'tscale', 'lat', 'lon']
    )
    kge_da.to_netcdf(savepath + f'mme_kge_{t}daysTP_Year20012013.nc')

    aindex_da = xr.DataArray(
        aindex,
        coords={
            'model': ['MME'],
            'year': ['AllYears'] + year.tolist(),
            'tscale': [t],
            'lat': lat_list,
            'lon': lon_list
        }, dims=['model', 'year', 'tscale', 'lat', 'lon']
    )
    aindex_da.to_netcdf(savepath + f'mme_aindex_{t}daysTP_Year20012013.nc')

    ets_da = xr.DataArray(
        ets,
        coords={
            'model': ['MME'],
            'year': ['AllYears'] + year.tolist(),
            'threshold': [10, 33.3, 66.7, 90],
            'tscale': [t],
            'lat': lat_list,
            'lon': lon_list
        }, dims=['model', 'year', 'threshold', 'tscale', 'lat', 'lon']
    )
    ets_da.to_netcdf(savepath + f'mme_ets_{t}daysTP_Year20012013.nc')

    rocss_da = xr.DataArray(
        rocss,
        coords={
            'model': ['MME'],
            'year': ['AllYears'] + year.tolist(),
            'threshold': [10, 33.3, 66.7, 90],
            'tscale': [t],
            'lat': lat_list,
            'lon': lon_list
        }, dims=['model', 'year', 'threshold', 'tscale', 'lat', 'lon']
    )
    rocss_da.to_netcdf(savepath + f'mme_rocss_{t}daysTP_Year20012013.nc')

    bss_da = xr.DataArray(
        bss,
        coords={
            'model': ['MME'],
            'year': ['AllYears'] + year.tolist(),
            'threshold': [10, 33.3, 66.7, 90],
            'tscale': [t],
            'lat': lat_list,
            'lon': lon_list
        }, dims=['model', 'year', 'threshold', 'tscale', 'lat', 'lon']
    )
    bss_da.to_netcdf(savepath + f'mme_bss_{t}daysTP_Year20012013.nc')

    tcc_da = xr.DataArray(
        tcc,
        coords={
            'model': ['MME'],
            'year': ['AllYears'] + year.tolist(),
            'tscale': [t],
            'lat': lat_list,
            'lon': lon_list
        }, dims=['model', 'year', 'tscale', 'lat', 'lon']
    )
    tcc_da.to_netcdf(savepath + f'mme_tcc_{t}daysTP_Year20012013.nc')

    scc_da = xr.DataArray(
        scc,
        coords={
            'model': ['MME'],
            'year': ['AllYears'] + year.tolist(),
            'tscale': [t],
            'issuetime': MMEfcst.issuetime,
        }, dims=['model', 'year', 'tscale', 'issuetime']
    )
    scc_da.to_netcdf(savepath + f'mme_scc_{t}daysTP_Year20012013.nc')

    print('saving results completed !')




# %%
#%%
# # 挑选出降水预报数据
# selected_indices = list(range(0, 10)) + list(range(13, 16)) + list(range(21, 25)) + list(range(26, 30)) + \
#     list(range(36, 47)) + list(range(53, 54)) + list(range(59, 63)) + list(range(67, 100)) + \
#     list(range(110, 114)) + list(range(124, 134)) + list(range(144, 155)) + list(range(165, 169)) + \
#     list(range(179, 190)) + list(range(200, 210)) + list(range(219, 223)) + list(range(231, 239)) + \
#     list(range(245, 248)) + list(range(257, 264))
# MMEfcst_npy = fcst_npy[:, selected_indices, :, :]  # (nday, 142, nlat, nlon)

# # #%%
# # 将提取出的降水预报数据转换成DataArray
# MMEfcst_nc = xr.DataArray(
#     MMEfcst_npy,
#     coords={
#         'issuetime': common_issue,
#         'number': np.arange(142),
#         'lat': lat_list,
#         'lon': lon_list
#     }, dims=['issuetime', 'number', 'lat', 'lon']
# )
# MMEfcst_nc.to_netcdf(f'/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/MME/MME_pr_{t}days_1.nc')