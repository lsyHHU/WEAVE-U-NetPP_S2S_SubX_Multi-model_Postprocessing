#%%
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def shade_seasons(ax, dates, summer=(4, 1, 10, 1),  # Apr 1 -> Oct 1
                  winter=(10, 1, 4, 1),            # Oct 1 -> next Apr 1
                  summer_kwargs=None, winter_kwargs=None,
                  add_legend=False):
    """
    Shade summer and winter periods on a matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    dates : array-like of datetime64/pandas Timestamp
        Used to determine year range and x-limits.
    summer : (start_month, start_day, end_month, end_day)
        Summer period within same year.
    winter : (start_month, start_day, end_month, end_day)
        Winter period spans year boundary (Oct->Apr of next year by default).
    summer_kwargs / winter_kwargs : dict
        Passed to ax.axvspan.
    add_legend : bool
        If True, add dummy handles for legend.
    """
    dates = pd.to_datetime(dates)
    y0, y1 = dates.min().year, dates.max().year

    if summer_kwargs is None:
        summer_kwargs = dict(alpha=0.12, color="orange", lw=0)
    if winter_kwargs is None:
        winter_kwargs = dict(alpha=0.10, color="steelblue", lw=0)

    sm, sd, em, ed = summer
    wm, wd, w_em, w_ed = winter

    # Summer shading: Apr 1 -> Oct 1 (same year)
    for y in range(y0, y1 + 1):
        s0 = pd.Timestamp(y, sm, sd)
        s1 = pd.Timestamp(y, em, ed)
        ax.axvspan(s0, s1, **summer_kwargs)

    # Winter shading: Oct 1 -> Apr 1 (next year)
    for y in range(y0, y1 + 1):
        w0 = pd.Timestamp(y, wm, wd)
        w1 = pd.Timestamp(y + 1, w_em, w_ed)
        ax.axvspan(w0, w1, **winter_kwargs)

    # Optional legend handles
    if add_legend:
        import matplotlib.patches as mpatches
        summer_patch = mpatches.Patch(color=summer_kwargs.get("color", "orange"),
                                      alpha=summer_kwargs.get("alpha", 0.01),
                                      label="Summer (Apr–Sep)")
        winter_patch = mpatches.Patch(color=winter_kwargs.get("color", "steelblue"),
                                      alpha=winter_kwargs.get("alpha", 0.01),
                                      label="Winter (Oct–Mar)")
        ax.legend(handles=[summer_patch, winter_patch], loc="upper left")





tscale = [7, 14, 21, 28]

fig, ax = plt.subplots(1, 1, figsize=(15, 4))



code_name1 = 'UNetPP-A-WeightPredictor-T-KLloss-ContinueTraining_3'
coed_name2 = 'UNetPP-A-WeightPredictor-T-new-new'

# 读取中国陆地网格csv
coords_csv = pd.read_csv('/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/CN_land_coords/land_coords.csv')
lat_mask = xr.DataArray(coords_csv['lat'].values, dims='new')
lon_mask = xr.DataArray(coords_csv['lon'].values, dims='new')

# 读取 weights-crps corr 数据
corr_dir1 = f'/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/00_Results_newTest_251209/{code_name1}/Output_0/'
corr_dir2 = f'/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/00_Results/{coed_name2}/Output_0/'


colors = ['#DB9C4D', '#70B48F', '#293890', '#BF1D2D']

for t in range(4):
    _t = tscale[t]
    corr_1t_1 = xr.open_dataarray(corr_dir1 + f'{_t}days/weights_crps_Spearman_corr(alongTime)_{_t}days.nc').sel(lat=lat_mask, lon=lon_mask)
    corr_1t_1.close()
    corr_1t_mean_1 = corr_1t_1.mean(dim='new')   # (nday,)
    dates = pd.to_datetime(corr_1t_1.issuetime.values)
    # 绘制折线图
    ax.plot(dates, corr_1t_mean_1, label=f'{_t} days (with KL loss)', lw=1.0, color=colors[t], alpha=0.8, ls='-')

    corr_1t_2 = xr.open_dataarray(corr_dir2 + f'{_t}days/weights_crps_Spearman_corr(alongTime)_{_t}days.nc').sel(lat=lat_mask, lon=lon_mask)
    corr_1t_2.close()
    corr_1t_mean_2 = corr_1t_2.mean(dim='new')   # (nday,)
    dates = pd.to_datetime(corr_1t_2.issuetime.values)
    # 绘制折线图
    ax.plot(dates, corr_1t_mean_2, label=f'{_t} days (without KL loss)', lw=1.0, color=colors[t], alpha=0.8, ls='--')
    # 标注出夏季月份和冬季月份的范围
    shade_seasons(ax, dates, 
                summer_kwargs=dict(color="gold", alpha=0.12, lw=0),
                winter_kwargs=dict(color="lightskyblue", alpha=0.10, lw=0),
                add_legend=True)


ax.legend(loc='upper right', ncol=4)
ax.set_ylim(-1, 0.5)
ax.set_ylabel('Spearman Correlation')
ax.set_xlim(dates[0], dates[-1])
ax.set_xlabel('Date')

fig.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.15)
# fig.savefig(f'/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/00_Results_newTest_251209/{code_name1}/Output_0/weights_crps_Spearman_corr(alongTime)_lineplot_TwoModelCompare.jpg', dpi=600)







# %%
