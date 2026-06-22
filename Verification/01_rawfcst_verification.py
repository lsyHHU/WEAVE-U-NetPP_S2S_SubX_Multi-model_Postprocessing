#%%
"""
各模型原始预报技能评价
"""
import sys, os
import xarray as xr
import numpy as np
import pandas as pd

sys.path.append('/home/liusy')
from pyNMME import module_verification


def get_ETS(obs, fcst, threshold, method):
    if method == 'upper':
        obs = np.where(obs >= threshold, 1, 0)
        fcst = np.where(fcst >= threshold, 1, 0)
    elif method == 'lower':
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


####################################################################################
####################################################################################
####################################################################################

tscale = [21, 28]

model_name = ['46LCESM1', 'CCSM4', 'CFSv2', 'FIMr1p1', 'GEFS', 'NESM', 'GEOS_V2p1', 
              'BoM', 'CMA', 'CNRM', 'CPTEC', 'ECCC', 'ECMWF', 'HMCR', 'IAPCAS', 'ISACCNR', 'KMA', 'UKMO']
nmodel = len(model_name)


# 时间尺度循环
for t in tscale:

    # 读取公共预报时间序列
    common_issue = pd.to_datetime(
        np.load(f'/data/selfdata/datalsy/SubX_Multimodel_250525/common_issue/common_issue_{t}days.npy')
    )

    # 经纬度范围
    lat_list = np.linspace(10, 57, 48)
    nlat = len(lat_list)
    lon_list = np.linspace(65, 144, 80)
    nlon = len(lon_list)

    # 年份序列
    year = np.arange(2001, 2014)
    nyear = len(year)

    # 定义空数组，存储评价结果
    pcc = np.zeros((nmodel, nyear+1, 1, nlat, nlon))
    rmse = np.zeros((nmodel, nyear+1, 1, nlat, nlon))
    rb = np.zeros((nmodel, nyear+1, 1, nlat, nlon))
    crpss = np.zeros((nmodel, nyear+1, 1, nlat, nlon))
    ets = np.zeros((nmodel, nyear+1, 2, 1, nlat, nlon))
    rocss = np.zeros((nmodel, nyear+1, 2, 1, nlat, nlon))
    bss = np.zeros((nmodel, nyear+1, 3, 1, nlat, nlon))


    # 模型循环
    for m in range(nmodel):
        _m = model_name[m]

        # 读取原始预报数据
        if _m in ['46LCESM1', 'CCSM4', 'CFSv2', 'FIMr1p1', 'GEFS', 'NESM', 'GEOS_V2p1']:
            rawfcst_dir = f'/data/selfdata/datalsy/SubX_Multimodel_250525/SubX/{t}days/'
        else:
            rawfcst_dir = f'/data/selfdata/datalsy/SubX_Multimodel_250525/S2S/{t}days/'
        rawfcst = xr.open_dataarray(rawfcst_dir + f'{_m}_pr_{t}days.nc').sel(issuetime=common_issue)
        rawfcst.close()  # (issuetime, number, lat, lon)

        # 读取观测数据
        obs = xr.open_dataarray(f'/data/selfdata/datalsy/SubX_Multimodel_250525/obs/CN05.1_totalPre_1995_2015_{t}days_1x1.nc').sel(time=common_issue)
        obs.close()   # # (issuetime, lat, lon)

        # 读取气候态预报数据
        clima = xr.open_dataarray(f'/data/selfdata/datalsy/SubX_Multimodel_250525/obs_climo/CN05.1_totalPre_climo_2001_2013_{t}days_1x1.nc').sel(issuetime=common_issue).squeeze()
        clima.close()   # # (issuetime, ens, lat, lon)

        # 年份循环
        for iyr in range(nyear+1):
            if iyr == 0:  # all years
                rawfcst_da = rawfcst
            else:
                rawfcst_da = rawfcst.sel(issuetime=common_issue[common_issue.year == year[iyr-1]])
            rawfcst_arr = rawfcst_da.to_numpy()

            obs_da = obs.sel(time=np.intersect1d(obs.time, rawfcst_da.issuetime))
            obs_arr = obs_da.to_numpy()

            clima_da = clima.sel(issuetime=np.intersect1d(clima.issuetime, rawfcst_da.issuetime))
            clima_arr = clima_da.to_numpy()

            # 计算评价指标
            for ilat in range(nlat):
                for ilon in range(nlon):
                    # 判断当前网格是否存在观测数据，如果没有观测数据则评价指标设置为nan
                    if np.isnan(obs_arr[:, ilat, ilon]).any():
                        pcc[m, iyr, 0, ilat, ilon] = np.nan
                        rmse[m, iyr, 0, ilat, ilon] = np.nan
                        rb[m, iyr, 0, ilat, ilon] = np.nan
                        crpss[m, iyr, 0, ilat, ilon] = np.nan
                        ets[m, iyr, :, 0, ilat, ilon] = np.nan
                        rocss[m, iyr, :, 0, ilat, ilon] = np.nan
                        bss[m, iyr, :, 0, ilat, ilon] = np.nan
                    
                    else:
                        pcc[m, iyr, 0, ilat, ilon] = module_verification.pcc(rawfcst_arr[:, :, ilat, ilon], obs_arr[:, ilat, ilon])
                        rmse[m, iyr, 0, ilat, ilon] = module_verification.rmse(rawfcst_arr[:, :, ilat, ilon], obs_arr[:, ilat, ilon])
                        rb[m, iyr, 0, ilat, ilon] = module_verification.rb(rawfcst_arr[:, :, ilat, ilon], obs_arr[:, ilat, ilon])
                        crpss[m, iyr, 0, ilat, ilon] = module_verification.skill(rawfcst_arr[:, :, ilat, ilon], obs_arr[:, ilat, ilon], clima_arr[:, :, ilat, ilon])     

                        ets_rocss_ths = [33.3, 66.7]
                        for its in range(2):
                            ets_event = 'lower' if its == 0 else 'upper'
                            ets[m, iyr, its, 0, ilat, ilon] = get_ETS(obs_arr[:, ilat, ilon], np.nanmean(rawfcst_arr[:, :, ilat, ilon], axis=1),
                                                                    threshold=np.percentile(obs_arr[:, ilat, ilon], ets_rocss_ths[its]), method=ets_event)

                            rocss_event = 'left' if its == 0 else 'right'
                            rocss[m, iyr, its, 0, ilat, ilon] = module_verification.rocss(rawfcst_arr[:, :, ilat, ilon], obs_arr[:, ilat, ilon], 
                                                                                        threshold=np.percentile(obs_arr[:, ilat, ilon], ets_rocss_ths[its]), 
                                                                                        event=rocss_event)

                        bss_ths = [10, 50, 90]
                        for its in range(3):
                            fcstbs = module_verification.bs(rawfcst_arr[:, :, ilat, ilon], obs_arr[:, ilat, ilon], 
                                                            threshold=np.percentile(obs_arr[:, ilat, ilon], bss_ths[its]), event='right')
                            climabs = module_verification.bs(clima_arr[:, :, ilat, ilon], obs_arr[:, ilat, ilon],
                                                            threshold=np.percentile(obs_arr[:, ilat, ilon], bss_ths[its]), event='right')
                            bss[m, iyr, its, 0, ilat, ilon] = (climabs - fcstbs) / climabs * 100.


                        print(f'Model: {_m}, Year: {iyr}, Lat: {ilat}, Lon: {ilon}')


    # 存成nc
    savepath = f'/home/liusy/MachineLearning_CSGD/SubX_Multimodel_250525/pre_verif_result_rawfcst/{t}days/'
    os.makedirs(savepath, exist_ok=True)

    pcc_da = xr.DataArray(
        pcc,
        coords={
            'model': model_name,
            'year': ['AllYears'] + year.tolist(),
            'tscale': [t],
            'lat': lat_list,
            'lon': lon_list
        }, dims=['model', 'year', 'tscale', 'lat', 'lon']
    )
    pcc_da.to_netcdf(savepath + f'raw_pcc_{t}daysTP_Year20012013.nc')

    rmse_da = xr.DataArray(
        rmse,
        coords={
            'model': model_name,
            'year': ['AllYears'] + year.tolist(),
            'tscale': [t],
            'lat': lat_list,
            'lon': lon_list
        }, dims=['model', 'year', 'tscale', 'lat', 'lon']
    )
    rmse_da.to_netcdf(savepath + f'raw_rmse_{t}daysTP_Year20012013.nc')

    rb_da = xr.DataArray(
        rb,
        coords={
            'model': model_name,
            'year': ['AllYears'] + year.tolist(),
            'tscale': [t],
            'lat': lat_list,
            'lon': lon_list
        }, dims=['model', 'year', 'tscale', 'lat', 'lon']
    )
    rb_da.to_netcdf(savepath + f'raw_rb_{t}daysTP_Year20012013.nc')

    crpss_da = xr.DataArray(
        crpss,
        coords={
            'model': model_name,
            'year': ['AllYears'] + year.tolist(),
            'tscale': [t],
            'lat': lat_list,
            'lon': lon_list
        }, dims=['model', 'year', 'tscale', 'lat', 'lon']
    )
    crpss_da.to_netcdf(savepath + f'raw_crpss_{t}daysTP_Year20012013.nc')

    ets_da = xr.DataArray(
        ets,
        coords={
            'model': model_name,
            'year': ['AllYears'] + year.tolist(),
            'threshold': [33.3, 66.7],
            'tscale': [t],
            'lat': lat_list,
            'lon': lon_list
        }, dims=['model', 'year', 'threshold', 'tscale', 'lat', 'lon']
    )
    ets_da.to_netcdf(savepath + f'raw_ets_{t}daysTP_Year20012013.nc')

    rocss_da = xr.DataArray(
        rocss,
        coords={
            'model': model_name,
            'year': ['AllYears'] + year.tolist(),
            'threshold': [33.3, 66.7],
            'tscale': [t],
            'lat': lat_list,
            'lon': lon_list
        }, dims=['model', 'year', 'threshold', 'tscale', 'lat', 'lon']
    )
    rocss_da.to_netcdf(savepath + f'raw_rocss_{t}daysTP_Year20012013.nc')

    bss_da = xr.DataArray(
        bss,
        coords={
            'model': model_name,
            'year': ['AllYears'] + year.tolist(),
            'threshold': [10, 50, 90],
            'tscale': [t],
            'lat': lat_list,
            'lon': lon_list
        }, dims=['model', 'year', 'threshold', 'tscale', 'lat', 'lon']
    )
    bss_da.to_netcdf(savepath + f'raw_bss_{t}daysTP_Year20012013.nc')

    print('saving results completed !')




# %%
import numpy as np

a = np.load('/data/selfdata/datalsy/SubX_Multimodel_250525/CN_land_coords/land_mask.npy')
a = a.flatten()

a_mask = np.where(a == 1)[0]







# %%
