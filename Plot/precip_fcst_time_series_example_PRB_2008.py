#%%
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.offsetbox import AnchoredText
import sys
sys.path.append('/home/liusy')
from pyNMME import module_verification

def calc_coverage(forecast, obs, interval=0.90):
    """
    Calculate prediction interval coverage for ensemble forecasts.

    Parameters
    ----------
    forecast : np.ndarray
        Ensemble forecast array with shape (n, m),
        where n is issuetime and m is ensemble member.
    obs : np.ndarray
        Observation array with shape (n,).
    interval : float
        Prediction interval level, e.g., 0.90 for 90% interval,
        0.50 for 50% interval.

    Returns
    -------
    coverage : float
        Fraction of observations falling within the prediction interval.
    lower : np.ndarray
        Lower bound of prediction interval.
    upper : np.ndarray
        Upper bound of prediction interval.
    is_covered : np.ndarray
        Boolean array indicating whether each observation is covered.
    """

    forecast = np.asarray(forecast)
    obs = np.asarray(obs)

    # 例如 90% interval: lower=5%, upper=95%
    alpha = 1 - interval
    lower_q = alpha / 2
    upper_q = 1 - alpha / 2

    lower = np.nanquantile(forecast, lower_q, axis=1)
    upper = np.nanquantile(forecast, upper_q, axis=1)

    # 去除 forecast 或 obs 中存在 NaN 的样本
    valid = np.isfinite(lower) & np.isfinite(upper) & np.isfinite(obs)

    is_covered = np.full(obs.shape, np.nan)
    is_covered[valid] = (obs[valid] >= lower[valid]) & (obs[valid] <= upper[valid])

    coverage = np.nanmean(is_covered)

    return coverage, lower, upper, is_covered

def add_metric_text(ax, pcc, rb, picp50, picp90, fontsize=12):
    """
    Add aligned verification metrics to the upper-right corner of an axis.
    """

    text = (
        f"{'PCC':<6} = {pcc:>6.2f}\n"
        f"{'RB':<6} = {rb:>6.1f}%\n"
        f"{'PICP50':<6} = {picp50 * 100:>6.1f}%\n"
        f"{'PICP90':<6} = {picp90 * 100:>6.1f}%"
    )

    ax.text(
        0.97, 0.95, text,
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=fontsize,
        # fontfamily='DejaVu Sans Mono',
        linespacing=1.25,
        bbox=dict(
            facecolor='white',
            edgecolor='0.75',
            linewidth=0.6,
            alpha=0.75,
            boxstyle='round,pad=0.35'
        )
    )




tscale = [7, 14, 21, 28]
basin = 'PRB'

code_name = 'UNetPP-A-WeightPredictor-T-KLloss-ContinueTraining_3'

coords_csv = pd.read_csv(f'/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/CN_land_coords/land_coords_{basin}.csv')
basin_lat = xr.DataArray(coords_csv['lat'].values, dims='new')
basin_lon = xr.DataArray(coords_csv['lon'].values, dims='new')


fig, ax = plt.subplots(4, 5, figsize=(28, 20))

colors = ['#DB9C4D', '#70B48F', '#293890', '#BF1D2D']
row_title = ['7 days', '14 days', '21 days', '28 days']
col_title = ['Ensemble mean\nof 4 Models', 'Forecast intervals\n of ECMWF', 'Forecast intervals\n of MME',
             'Forecast intervals\n of EMOS', 'Forecast intervals\n of WEAVE-U-Net++']

for t in range(4):
    _t = tscale[t]

    # 读取后处理预报
    postfcst = xr.open_dataarray(f'/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/00_Results_newTest_251209/{code_name}/Output_0/{_t}days/Unetfcst_ENS/ensfcst_pr_{_t}days_Year20012013.nc')
    postfcst.close()
    postfcst = postfcst.sel(lat=basin_lat, lon=basin_lon, issuetime=slice('2008-1-1', '2008-12-31')).squeeze()
    postfcst_regional_mean = postfcst.mean(dim=['new'], skipna=True)
    postfcst_regional_mean_arr = postfcst_regional_mean.values   # (n, m)
    # 集合平均
    postfcst_regional_ensmean_arr = np.nanmean(postfcst_regional_mean_arr, axis=1)
    # 预报区间
    postfcst_regional_5th_arr = np.percentile(postfcst_regional_mean_arr, 5, axis=1)
    postfcst_regional_25th_arr = np.percentile(postfcst_regional_mean_arr, 25, axis=1)
    postfcst_regional_75th_arr = np.percentile(postfcst_regional_mean_arr, 75, axis=1)
    postfcst_regional_95th_arr = np.percentile(postfcst_regional_mean_arr, 95, axis=1)
    dates = pd.to_datetime(postfcst_regional_mean.issuetime.values)

    # 读取EMOS预报
    emosfcst = xr.open_dataarray(f'/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/01_EMOS_py_test1/predict_ensfcst100_map/ensfcst_{_t}daysTP.nc')
    emosfcst.close()
    emosfcst = emosfcst.sel(lat=basin_lat, lon=basin_lon, issuetime=np.intersect1d(emosfcst.issuetime, postfcst.issuetime)).squeeze()
    emosfcst_regional_mean = emosfcst.mean(dim=['new'], skipna=True)
    emosfcst_regional_mean_arr = emosfcst_regional_mean.values   # (n,m)
    # 集合平均
    emosfcst_regional_ensmean_arr = np.nanmean(emosfcst_regional_mean_arr, axis=1)
    # 预报区间
    emosfcst_regional_5th_arr = np.percentile(emosfcst_regional_mean_arr, 5, axis=1)
    emosfcst_regional_25th_arr = np.percentile(emosfcst_regional_mean_arr, 25, axis=1)
    emosfcst_regional_75th_arr = np.percentile(emosfcst_regional_mean_arr, 75, axis=1)
    emosfcst_regional_95th_arr = np.percentile(emosfcst_regional_mean_arr, 95, axis=1)

    # 读取MME预报
    mmefcst = xr.open_dataarray(f'/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/MME/MME_pr_{_t}days_1.nc')
    mmefcst.close()
    mmefcst = mmefcst.sel(lat=basin_lat, lon=basin_lon, issuetime=np.intersect1d(mmefcst.issuetime, postfcst.issuetime)).squeeze()
    mmefcst_regional_mean = mmefcst.mean(dim=['new'], skipna=True)
    mmefcst_regional_mean_arr = mmefcst_regional_mean.values   # (n,m)
    # 集合平均
    mmefcst_regional_ensmean_arr = np.nanmean(mmefcst_regional_mean_arr, axis=1)
    # 预报区间
    mmefcst_regional_5th_arr = np.percentile(mmefcst_regional_mean_arr, 5, axis=1)
    mmefcst_regional_25th_arr = np.percentile(mmefcst_regional_mean_arr, 25, axis=1)
    mmefcst_regional_75th_arr = np.percentile(mmefcst_regional_mean_arr, 75, axis=1)
    mmefcst_regional_95th_arr = np.percentile(mmefcst_regional_mean_arr, 95, axis=1)

    # 读取原始ECMWF预报
    rawfcst = xr.open_dataarray(f'/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/S2S/{_t}days/ECMWF_pr_{_t}days.nc')
    rawfcst.close()
    rawfcst = rawfcst.sel(lat=basin_lat, lon=basin_lon, issuetime=np.intersect1d(rawfcst.issuetime, postfcst.issuetime)).squeeze()
    rawfcst_regional_mean = rawfcst.mean(dim=['new'], skipna=True)
    rawfcst_regional_mean_arr = rawfcst_regional_mean.values   # (n,m)
    # 集合平均
    rawfcst_regional_ensmean_arr = np.nanmean(rawfcst_regional_mean_arr, axis=1)
    # 预报区间
    rawfcst_regional_5th_arr = np.percentile(rawfcst_regional_mean_arr, 5, axis=1)
    rawfcst_regional_25th_arr = np.percentile(rawfcst_regional_mean_arr, 25, axis=1)
    rawfcst_regional_75th_arr = np.percentile(rawfcst_regional_mean_arr, 75, axis=1)
    rawfcst_regional_95th_arr = np.percentile(rawfcst_regional_mean_arr, 95, axis=1)
    
    # 读取观测数据
    obs = xr.open_dataarray(f'/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/obs/CN05.1_totalPre_1995_2015_{_t}days_1x1.nc')
    obs.close()
    obs = obs.sel(lat=basin_lat, lon=basin_lon, time=np.intersect1d(obs.time, postfcst.issuetime))
    obs_regional_mean = obs.mean(dim=['new'], skipna=True)
    obs_regional_mean_arr = obs_regional_mean.values   # (n,)

    # 计算评价指标
    pcc_ecmwf = module_verification.pcc(rawfcst_regional_ensmean_arr, obs_regional_mean_arr)
    pcc_mme = module_verification.pcc(mmefcst_regional_ensmean_arr, obs_regional_mean_arr)
    pcc_emos = module_verification.pcc(emosfcst_regional_ensmean_arr, obs_regional_mean_arr)
    pcc_post = module_verification.pcc(postfcst_regional_ensmean_arr, obs_regional_mean_arr)

    rb_ecmwf = module_verification.rb(rawfcst_regional_ensmean_arr, obs_regional_mean_arr)
    rb_mme = module_verification.rb(mmefcst_regional_ensmean_arr, obs_regional_mean_arr)
    rb_emos = module_verification.rb(emosfcst_regional_ensmean_arr, obs_regional_mean_arr)
    rb_post = module_verification.rb(postfcst_regional_ensmean_arr, obs_regional_mean_arr)

    picp50_ecmwf, _, _, _ = calc_coverage(rawfcst_regional_mean_arr, obs_regional_mean_arr, 0.5)
    picp50_mme, _, _, _ = calc_coverage(mmefcst_regional_mean_arr, obs_regional_mean_arr, 0.5)
    picp50_emos, _, _, _ = calc_coverage(emosfcst_regional_mean_arr, obs_regional_mean_arr, 0.5)
    picp50_post, _, _, _ = calc_coverage(postfcst_regional_mean_arr, obs_regional_mean_arr, 0.5)

    picp90_ecmwf, _, _, _ = calc_coverage(rawfcst_regional_mean_arr, obs_regional_mean_arr, 0.9)
    picp90_mme, _, _, _ = calc_coverage(mmefcst_regional_mean_arr, obs_regional_mean_arr, 0.9)
    picp90_emos, _, _, _ = calc_coverage(emosfcst_regional_mean_arr, obs_regional_mean_arr, 0.9)
    picp90_post, _, _, _ = calc_coverage(postfcst_regional_mean_arr, obs_regional_mean_arr, 0.9)


    # 绘制观测序列
    for i in range(5):
        ax[t, i].plot(dates, obs_regional_mean_arr, color='k', lw=1.5, label='Obs')
        ax[t, i].set_xlim(pd.to_datetime('2008-01-01'), pd.to_datetime('2008-12-31'))
        ax[0, i].set_ylim(-10, 350)
        ax[1, i].set_ylim(-10, 450)
        ax[2, i].set_ylim(-10, 600)
        ax[3, i].set_ylim(-10, 700)

    # 绘制第一列（4个模型的集合均值预报）
    ax[t, 0].plot(dates, rawfcst_regional_ensmean_arr, color=colors[0], lw=1, label='ECMWF')
    ax[t, 0].plot(dates, mmefcst_regional_ensmean_arr, color=colors[1], lw=1, label='MME')
    ax[t, 0].plot(dates, emosfcst_regional_ensmean_arr, color=colors[2], lw=1, label='EMOS')
    ax[t, 0].plot(dates, postfcst_regional_ensmean_arr, color=colors[3], lw=1, label='Weave-U-Net++')

    # 绘制后四列（每个模型的预报区间）
    ax[t, 1].plot(dates, rawfcst_regional_ensmean_arr, color=colors[0], lw=1.5, label='Ens. Mean')
    ax[t, 1].fill_between(dates, rawfcst_regional_5th_arr, rawfcst_regional_95th_arr, color=colors[0], alpha=0.2, label='90% Interval')
    ax[t, 1].fill_between(dates, rawfcst_regional_25th_arr, rawfcst_regional_75th_arr, color=colors[0], alpha=0.5, label='50% Interval')

    ax[t, 2].plot(dates, mmefcst_regional_ensmean_arr, color=colors[1], lw=1.5, label='Ens. Mean')
    ax[t, 2].fill_between(dates, mmefcst_regional_5th_arr, mmefcst_regional_95th_arr, color=colors[1], alpha=0.2, label='90% Interval')
    ax[t, 2].fill_between(dates, mmefcst_regional_25th_arr, mmefcst_regional_75th_arr, color=colors[1], alpha=0.5, label='50% Interval')

    ax[t, 3].plot(dates, emosfcst_regional_ensmean_arr, color=colors[2], lw=1.5, label='Ens. Mean')
    ax[t, 3].fill_between(dates, emosfcst_regional_5th_arr, emosfcst_regional_95th_arr, color=colors[2], alpha=0.2, label='90% Interval')
    ax[t, 3].fill_between(dates, emosfcst_regional_25th_arr, emosfcst_regional_75th_arr, color=colors[2], alpha=0.5, label='50% Interval')

    ax[t, 4].plot(dates, postfcst_regional_ensmean_arr, color=colors[3], lw=1.5, label='Ens. Mean')
    ax[t, 4].fill_between(dates, postfcst_regional_5th_arr, postfcst_regional_95th_arr, color=colors[3], alpha=0.2, label='90% Interval')
    ax[t, 4].fill_between(dates, postfcst_regional_25th_arr, postfcst_regional_75th_arr, color=colors[3], alpha=0.5, label='50% Interval')

    # 添加评价指标：仅添加到第2–5列，第一列ensemble mean不添加
    add_metric_text(ax[t, 1], pcc_ecmwf, rb_ecmwf, picp50_ecmwf, picp90_ecmwf, fontsize=14)
    add_metric_text(ax[t, 2], pcc_mme, rb_mme, picp50_mme, picp90_mme,fontsize=14)
    add_metric_text(ax[t, 3], pcc_emos, rb_emos, picp50_emos, picp90_emos, fontsize=14)
    add_metric_text(ax[t, 4], pcc_post, rb_post, picp50_post, picp90_post, fontsize=14)

for t in range(4):
    for i in range(5):
        ax[t, i].grid(True, axis='y', lw=0.5, alpha=0.5)
        ax[t, i].tick_params(labelsize=16)
        ax[t, i].legend(loc='upper left', fontsize=14) if t == 0 else None
        ax[t, i].set_xlabel('Date', fontsize=16, labelpad=15) if t == 3 else None
        ax[t, i].set_ylabel('Precipitation (mm)', fontsize=16, labelpad=15) if i == 0 else None
        ax[t, i].set_title(col_title[i], fontsize=22, fontweight='bold', pad=15) if t == 0 else None
        ax[t, i].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax[t, i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

ax[0, 0].text(pd.to_datetime('2007-05-20'), 175, '7 days', fontsize=22, fontweight='bold')
ax[1, 0].text(pd.to_datetime('2007-05-20'), 225, '14 days', fontsize=22, fontweight='bold')
ax[2, 0].text(pd.to_datetime('2007-05-20'), 300, '21 days', fontsize=22, fontweight='bold')
ax[3, 0].text(pd.to_datetime('2007-05-20'), 350, '28 days', fontsize=22, fontweight='bold')
    
# 保存图像
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.08, top=0.92)
savedir = '/home/liusy/MachineLearning_CSGD/S2S_SubX_Multimodel_250525/03_Unet++_newTest_251209/00_pic/'
fig.savefig(savedir + f'82_precip_fcst_time_series_example_{basin}_2008_new.jpg', dpi=600)





# %%
