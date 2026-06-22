#%%
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from matplotlib import colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['figure.dpi'] = 600
font1 = {'weight': 'normal', 'size': 18}
font1_bold = {'weight': 'bold', 'size': 18, 'color': 'blue'}
font2 = {'weight': 'bold', 'size': 24}
font3 = {'weight': 'normal', 'size': 30}

# 水文气候分区shp文件
shp = shpreader.Reader("/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/shp/China_nine_basin_Dissolve_WGS.shp")
shp_feature = cfeature.ShapelyFeature(shp.geometries(),
                                      ccrs.PlateCarree(),
                                      edgecolor="black",
                                      facecolor="none")

# 生成画布
fig = plt.figure(figsize=(18, 8))
gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 1], wspace=0.0, hspace=0.0)
axes = [[None for _ in range(4)] for _ in range(2)]  # 二维列表
for i in range(2):       # 行
    for j in range(4):   # 列
        ax = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree())
        axes[i][j] = ax


tscale = [7, 14, 21, 28]
row_title = ['Summer\n(Apr-Sept)', 'Winter\n(Oct-Mar)']
column_title = ['7 days', '14 days', '21 days', '28 days']


# 时间尺度循环
for t in range(4):
    _t = tscale[t]
    # 读取cprs数据
    cprs_nc = xr.open_dataarray(f'/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/00_Results/crps_by_model_{_t}days.nc')  # (issuetime, model, lat, lon)
    cprs_nc.close()

    crps_vals = cprs_nc.values  # (T, M, H, W)
    T, M, H, W = crps_vals.shape

    # 输出 rank：同形状 (T, M, H, W)
    rank_vals = np.full((T, M, H, W), np.nan, dtype=float)

    # 把空间拍平：对每个 time 的所有网格一起做 rank（按网格循环）
    # flat shape: (T, M, P) where P = H*W
    flat = crps_vals.reshape(T, M, -1)
    rank_flat = rank_vals.reshape(T, M, -1)

    for i in range(T):
        # row shape: (M, P)
        row = flat[i, :, :]
        # valid mask per gridpoint (P,)：要求该格点所有模式都有限；你也可以放宽为 >=2 个有效模式
        valid_grid = np.isfinite(row).all(axis=0)

        if valid_grid.sum() == 0:
            continue

        # 对每个有效格点 p，按 model 维排序并给 rank
        # sub shape: (M, P_valid)
        sub = row[:, valid_grid]

        # stable sort along model axis
        order = np.argsort(sub, axis=0, kind="mergesort")  # (M, P_valid)

        # ranks：1..M
        ranks = np.empty_like(order, dtype=float)
        ranks[order, np.arange(order.shape[1])[None, :]] = np.arange(1, M + 1)[:, None]

        rank_flat[i, :, valid_grid] = ranks.T

    rank_da = xr.DataArray(
        rank_vals,
        coords=cprs_nc.coords,
        dims=cprs_nc.dims,
        name="crps_rank"
    )
    
    
    # 季节循环
    for isea in range(2):
        if isea == 0:
            std_rank_da_1sea = rank_da.sel(issuetime=rank_da.issuetime.dt.month.isin([4, 5, 6, 7, 8, 9])).std(dim='issuetime', skipna=True)
        else:
            std_rank_da_1sea = rank_da.sel(issuetime=rank_da.issuetime.dt.month.isin([10, 11, 12, 1, 2, 3])).std(dim='issuetime', skipna=True)
        # 取model维度平均
        std_rank_da_1sea_mean = std_rank_da_1sea.mean(dim='model', skipna=True)


        # 绘图
        p = std_rank_da_1sea_mean.plot.imshow(
            ax = axes[isea][t],
            transform = ccrs.PlateCarree(), 
            x = 'lon', y = 'lat',
            vmin = 2, vmax = 5.5,
            extend = 'both', 
            cmap = 'rainbow',
            add_colorbar = False,
        )

        axes[isea][t].set_facecolor("lightgray")
        axes[isea][t].set_title(None)
        # cartopy绘制地图
        axes[isea][t].set_extent([70, 139, 10, 62], crs=ccrs.PlateCarree())    
        # 绘制水文气象分区shp
        axes[isea][t].add_feature(shp_feature, linestyle='-', linewidth=1.0)


        # 添加列标题和行标题
        axes[isea][t].text(45, 30, row_title[isea], fontdict=font2, ha='center') if t == 0 else None
        axes[isea][t].set_title(column_title[t], fontdict=font2, pad=30.0) if isea == 0 else None


fig.subplots_adjust(left=0.15, right=0.95, bottom=0.17, top=0.9)

# 添加colorbar
p_ax_anchor = axes[1][0].get_position()
p_ax_cb = fig.add_axes([p_ax_anchor.x0, p_ax_anchor.y0-0.07, p_ax_anchor.width*4, 0.03])
p_cb = fig.colorbar(p, 
                    cax=p_ax_cb, 
                    extend='both', 
                    orientation='horizontal')
p_cb.set_label(label='Standard deviation of rank of CRPS across 18 NWP models', fontdict=font3)
p_ax_cb.tick_params(which='major', direction='in', length=7)
for ticklabel in p_ax_cb.xaxis.get_ticklabels():
    ticklabel.set_fontsize(26)

# 保存
savedir = '/home/liusy/MachineLearning_CSGD/S2S_SubX_Multimodel_250525/03_Unet++_newTest_251209/00_pic/'
fig.savefig(savedir + f'94-2_rawfcst_crps_std(rank)_map_plot.jpg')











# %%
