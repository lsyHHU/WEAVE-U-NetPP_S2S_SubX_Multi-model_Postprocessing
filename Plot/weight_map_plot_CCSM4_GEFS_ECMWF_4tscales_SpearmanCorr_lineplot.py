#%%
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['figure.dpi'] = 600
font1 = {'weight': 'normal', 'size': 22}
font1_bold = {'weight': 'bold', 'size': 22, 'color': 'blue'}
font2 = {'weight': 'bold', 'size': 24}
font3 = {'weight': 'normal', 'size': 30}

def omit_leading_zero(x, pos):
    return f'{x:.2f}'.lstrip('0') if x < 1 else f'{x:.2f}'

def add_hist_inset(parent_ax, data, bins, xlim, ylim, cmap, norm, if_axvline=True):
    """
    在地图左下角添加一个频率直方图小子图
    data: 2D ndarray 或 DataArray.values
    """
    vals = np.asarray(data).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return

    vals = np.clip(vals, xlim[0], xlim[1])

    hist_ax = inset_axes(
        parent_ax,
        width="34%",
        height="26%",
        loc='lower left',
        bbox_to_anchor=(0.065, 0.06, 1, 1),   # 左边界，下边界
        bbox_transform=parent_ax.transAxes,
        borderpad=0.0,  # 子图与父图边界之间的距离
    )

    # 计算相对频率
    counts, edges = np.histogram(vals, bins=bins)
    freqs = counts / vals.size
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    cmap_obj = plt.get_cmap(cmap)
    bar_colors = [cmap_obj(norm(c)) for c in centers]

    hist_ax.bar(
        edges[:-1], freqs, width=widths, align='edge',
        color=bar_colors, edgecolor='black', linewidth=0.4, alpha=0.8,
        zorder=3
    )

    if if_axvline:
        hist_ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, zorder=4)

    # 透明背景
    hist_ax.set_facecolor('none')
    hist_ax.patch.set_alpha(0.0)

    # 固定坐标范围
    hist_ax.set_xlim(*xlim)
    hist_ax.set_ylim(*ylim)
    # hist_ax.set_xticks([xlim[0], 0, xlim[1]])
    hist_ax.set_yticks(np.linspace(ylim[0], ylim[1], 3))
    hist_ax.yaxis.set_major_formatter(FuncFormatter(omit_leading_zero))

    # 仅保留左轴和下轴
    hist_ax.spines['top'].set_visible(False)
    hist_ax.spines['right'].set_visible(False)
    hist_ax.spines['left'].set_linewidth(0.8)
    hist_ax.spines['bottom'].set_linewidth(0.8)
    hist_ax.tick_params(axis='both', labelsize=7, length=2, pad=1,
                        top=False, right=False)

    # 不再加背景网格和标题，尽量简洁
    hist_ax.grid(False)

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
    for y in range(y0 - 1, y1 + 1):
        s0 = pd.Timestamp(y, sm, sd)
        s1 = pd.Timestamp(y, em, ed)
        ax.axvspan(s0, s1, **summer_kwargs)

    # Winter shading: Oct 1 -> Apr 1 (next year)
    for y in range(y0 - 1, y1 + 1):
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
        
        # 先获取已有的线图图例
        handles, labels = ax.get_legend_handles_labels()

        # 追加shading图例
        handles.extend([summer_patch, winter_patch])
        labels.extend(["Apr–Sep", "Oct–Mar"])
        
        ax.legend(handles, labels, loc='upper right', fontsize=17, ncol=5, columnspacing=1.0, handlelength=1.5)





# 水文气候分区shp文件
shp = shpreader.Reader("/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/shp/China_nine_basin_Dissolve_WGS.shp")
shp_feature = cfeature.ShapelyFeature(shp.geometries(),
                                      ccrs.PlateCarree(),
                                      edgecolor="black",
                                      facecolor="none")

# 读取中国陆地mask
mask = np.load('/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/CN_land_coords/land_mask.npy')
mask_da = xr.DataArray(mask, coords=[np.linspace(10, 57, 48), np.linspace(65, 144, 80)], dims=['lat', 'lon'])
mask_da = mask_da.where(mask_da == 1, np.nan)

# 读取中国陆地网格csv
coords_csv = pd.read_csv('/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/CN_land_coords/land_coords.csv')
lat_mask = xr.DataArray(coords_csv['lat'].values, dims='new')
lon_mask = xr.DataArray(coords_csv['lon'].values, dims='new')


# =========================
# 1. 建立画布
# =========================
fig = plt.figure(figsize=(25, 18))

# 外层：分成上下两块
# 上块 = 前4行地图
# 下块 = 第3行 lineplot
outer_gs = gridspec.GridSpec(
    2, 1,
    height_ratios=[4.0, 0.9],   # 上面地图块更高，下面折线图稍矮
    hspace=0.1                 # 第二行和第三行之间留出较大空隙
)

# 内层：前4行6列地图，行间距/列间距都设为0
map_gs = outer_gs[0].subgridspec(
    4, 6,
    wspace=0.0,
    hspace=0.0
)

# =========================
# 2. 前4行：weight空间分布图
# =========================
map_axes = [[None for _ in range(6)] for _ in range(4)]

for i in range(4):
    for j in range(6):
        ax = fig.add_subplot(map_gs[i, j], projection=ccrs.PlateCarree())
        map_axes[i][j] = ax

# =========================
# 3. 第5行：一个长的 lineplot
# =========================
line_ax = fig.add_subplot(outer_gs[1, 0])


####################################################################
# -------------- 读取数据，绘制weight空间分布图

root_dir = '/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/00_Results_newTest_251209/'


code_name = [
    'UNetPP-A-WeightPredictor-T-KLloss-ContinueTraining_3',
    'UNetPP-A-WeightPredictor-T'
]

tscale = [7, 14, 21, 28]
model_name = ['CCSM4', 'GEFS', 'ECMWF', 'CCSM4', 'GEFS', 'ECMWF']
row_title = ['7 days', '14 days', '21 days', '28 days']
column_title = ['CCSM4\n(w/ KL loss)', 'GEFS\n(w/ KL loss)', 'ECMWF\n(w/ KL loss)', 
               'CCSM4\n(w/o KL loss)', 'GEFS\n(w/o KL loss)', 'ECMWF\n(w/o KL loss)']

# 时间尺度循环
for t in range(len(tscale)):
    _t = tscale[t]

    # 方法循环/模型循环
    for m in range(len(model_name)):
        _c = code_name[m//3]
        _m = model_name[m]
    
        # 读取权重数据
        weights_da = xr.open_dataarray(root_dir + f'{_c}/Output_0/{_t}days/Unetfcst_w/weights_{_t}days_Year20012013.nc')
        weights_da.close()

        weights_issuetime_mean_da = weights_da.sel(model=_m).mean(dim='issuetime', skipna=True)
        weights_masked = weights_issuetime_mean_da * mask_da
        weights_masked = weights_masked * 100.   # 权重乘100

        # 绘制权重图
        p = weights_masked.plot.imshow(
            ax = map_axes[t][m],
            transform = ccrs.PlateCarree(), 
            x = 'lon', y = 'lat',
            vmin = 3, vmax = 8,
            extend = 'both', 
            cmap = 'BrBG',
            add_colorbar = False,
        )

        map_axes[t][m].set_facecolor("lightgray")
        map_axes[t][m].set_title(None)
        # cartopy绘制地图
        map_axes[t][m].set_extent([70, 139, 10, 62], crs=ccrs.PlateCarree())    
        # 绘制水文气象分区shp
        map_axes[t][m].add_feature(shp_feature, linestyle='-', linewidth=1.0)

        # 添加列标题和行标题
        map_axes[t][m].text(50, 30, row_title[t], fontdict=font2, ha='center') if m == 0 else None
        map_axes[t][m].set_title(column_title[m], fontdict=font2, pad=15.0, ha='center') if t == 0 else None

        # 在每个子图中再插入一个小子图，绘制当前空间分布的频率直方图
        HIST_BINS = np.arange(3, 8, 0.5)
        HIST_XLIM = (3, 8)
        HIST_YLIM = (0.0, 0.4) if m <=2 else (0.0, 1.0)
        HIST_CMAP = 'BrBG'
        HIST_NORM = Normalize(vmin=3, vmax=8)
        add_hist_inset(map_axes[t][m], weights_masked.values,
                        bins=HIST_BINS, 
                        xlim=HIST_XLIM, ylim=HIST_YLIM,
                        cmap=HIST_CMAP, norm=HIST_NORM,
                        if_axvline=False)


fig.subplots_adjust(left=0.1, right=0.88, bottom=0.07, top=0.93)

# 添加colorbar
map_ax_0_2_pos = map_axes[0][5].get_position()
map_ax_1_2_pos = map_axes[3][5].get_position()

p_ax_cb = fig.add_axes([map_ax_1_2_pos.x0 + map_ax_1_2_pos.width + 0.02, map_ax_1_2_pos.y0, 
                        0.02, map_ax_0_2_pos.y0 + map_ax_0_2_pos.height - map_ax_1_2_pos.y0])
p_cb = fig.colorbar(
    p, 
    cax=p_ax_cb, 
    extend='both', 
    orientation='vertical'
)
p_cb.set_label(label=r'Weights ($\times 10^{-2}$)', fontdict=font3, labelpad=20)
p_ax_cb.tick_params(which='major', direction='in', length=7)
for ticklabel in p_ax_cb.yaxis.get_ticklabels():
    ticklabel.set_fontsize(26)


####################################################################
# -------------- 读取数据，绘制spearman corr lineplot

colors = ['#DB9C4D', '#70B48F', '#293890', '#BF1D2D']
marker = ['o', 's', '^', 'P']
tscale_list = [7, 14, 21, 28]

for t in range(4):
    _t = tscale_list[t]
    corr_1t_1 = xr.open_dataarray(root_dir + f'{code_name[0]}/Output_0/{_t}days/weights_crps_Spearman_corr(alongTime)_{_t}days.nc').sel(lat=lat_mask, lon=lon_mask)
    corr_1t_1.close()
    corr_1t_mean_1 = corr_1t_1.mean(dim='new')   # (nday,)
    corr_1t_mean_1_monthly = corr_1t_mean_1.resample(issuetime='M').mean()  # 对同一月份的数据进行平均
    dates = pd.to_datetime(corr_1t_mean_1_monthly.issuetime.values)
    # 绘制折线图
    line_ax.plot(dates, corr_1t_mean_1_monthly, label=f'{_t} days (w/ KL loss)', lw=1.0, color=colors[t], alpha=0.8, ls='-', zorder=2,
                    marker=marker[t], markevery=3, markersize=6)

    corr_1t_2 = xr.open_dataarray(root_dir + f'{code_name[1]}/Output_0/{_t}days/weights_crps_Spearman_corr(alongTime)_{_t}days.nc').sel(lat=lat_mask, lon=lon_mask)
    corr_1t_2.close()
    corr_1t_mean_2 = corr_1t_2.mean(dim='new')   # (nday,)
    corr_1t_mean_2_monthly = corr_1t_mean_2.resample(issuetime='M').mean()  # 对同一月份的数据进行平均
    dates = pd.to_datetime(corr_1t_mean_2_monthly.issuetime.values)
    # 绘制折线图
    line_ax.plot(dates, corr_1t_mean_2_monthly, label=f'{_t} days (w/o KL loss)', lw=1.0, color=colors[t], alpha=0.6, ls='--', zorder=2,
                    marker=marker[t], markevery=3, markersize=6)

line_ax.axhline(0, color='k', lw=1.0, ls='--', zorder=1)
# line_ax.legend(loc='upper right', fontsize=15, ncol=4, columnspacing=1.0)
line_ax.set_ylim(-0.7, 0.5)
line_ax.set_ylabel('Correlation', fontdict=font1, labelpad=20)
line_ax.set_xlim(dates[0], dates[-1])
line_ax.set_xlabel('Date', fontdict=font1)
line_ax.tick_params(labelsize=22)
line_ax.set_title('Spearman correlation between model weights and CRPS', fontdict=font2, pad=15.0)

# 标注出夏季月份和冬季月份的范围
shade_seasons(line_ax, dates, 
            summer_kwargs=dict(color="gold", alpha=0.25, lw=0),
            winter_kwargs=dict(color="lightskyblue", alpha=0.25, lw=0),
            add_legend=True)  # 在该函数中控制legend参数



# # 保存
fig.savefig(f'/home/liusy/MachineLearning_CSGD/S2S_SubX_Multimodel_250525/03_Unet++_newTest_251209/00_pic/93-8-1_weight_map_plot_CCSM4_GEFS_ECMWF_4tscales_SpearmanCorr_lineplot.jpg')





















# %%
