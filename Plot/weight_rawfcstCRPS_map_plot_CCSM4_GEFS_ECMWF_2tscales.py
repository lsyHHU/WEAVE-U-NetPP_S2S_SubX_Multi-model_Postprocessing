#%%
import string
import xarray as xr
import numpy as np
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
font1 = {'weight': 'normal', 'size': 18}
font1_bold = {'weight': 'bold', 'size': 18, 'color': 'blue'}
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

def add_panel_labels(
    axes,
    x=0.02,
    y=0.97,
    fontsize=22,
    fontweight='bold',
):
    """
    给主 axes 添加 (a), (b), (c) ... 编号。
    只遍历 axes 二维列表，因此不会给 inset axes 或 colorbar axes 编号。
    """
    labels = [f'({letter})' for letter in string.ascii_lowercase]

    k = 0
    for i in range(len(axes)):
        for j in range(len(axes[i])):
            ax = axes[i][j]

            ax.text(
                x, y, labels[k],
                transform=ax.transAxes,   # 使用 axes 坐标，而不是经纬度/数据坐标
                ha='left',
                va='top',
                fontsize=fontsize,
                fontweight=fontweight,
                color='black',
                zorder=100,
            )
            k += 1



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


# 生成画布
fig = plt.figure(figsize=(20, 12))
gs = gridspec.GridSpec(3, 4, width_ratios=[1, 1, 1, 1], wspace=0.0, hspace=0.0)
axes = [[None for _ in range(4)] for _ in range(3)]  # 二维列表
for i in range(3):       # 行
    for j in range(4):   # 列
        ax = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree())
        axes[i][j] = ax


root_dir = '/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/00_Results_newTest_251209/'

code_name = 'UNetPP-A-WeightPredictor-T-KLloss-ContinueTraining_3'

tscale = [7, 28, 7, 28]
row_title = ['CCSM4', 'GEFS', 'ECMWF']
column_title = ['Weights (7 days)', 'Weights (28 days)', 'CRPS (7 days)', 'CRPS (28 days)']


# 模型循环
for m in range(3):
    
    # 时间尺度循环
    for t in range(4):
        _t = tscale[t]

        if t == 0 or t == 1:
            # 读取权重文件
            weights_da = xr.open_dataarray(root_dir + f'{code_name}/Output_0/{_t}days/Unetfcst_w/weights_{_t}days_Year20012013.nc').sel(model=row_title[m])
            weights_issuetime_mean_da = weights_da.mean(dim='issuetime', skipna=True)
            weights_masked = weights_issuetime_mean_da * mask_da
            weights_masked = weights_masked * 100.   # 权重乘100

            # 绘制权重图
            p = weights_masked.plot.imshow(
                ax = axes[m][t],
                transform = ccrs.PlateCarree(), 
                x = 'lon', y = 'lat',
                vmin = 3, vmax = 8,
                extend = 'both', 
                cmap = 'BrBG',
                add_colorbar = False,
            )

            # 在每个子图中再插入一个小子图，绘制当前空间分布的频率直方图
            HIST_BINS = np.arange(3, 8, 0.5)
            HIST_XLIM = (3, 8)
            HIST_YLIM = (0.0, 0.6)
            HIST_CMAP = 'BrBG'
            HIST_NORM = Normalize(vmin=3, vmax=8)
            add_hist_inset(axes[m][t], weights_masked.values,
                            bins=HIST_BINS, 
                            xlim=HIST_XLIM, ylim=HIST_YLIM,
                            cmap=HIST_CMAP, norm=HIST_NORM,
                            if_axvline=False)

        else:
            # 读取CRPS文件
            crps_da = xr.open_dataarray(f'/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/00_Results/crps_by_model_{_t}days.nc').sel(model=row_title[m])
            crps_issuetime_mean_da = crps_da.mean(dim='issuetime', skipna=True)
            crps_masked = crps_issuetime_mean_da * mask_da

            # 绘制crps分布图
            q = crps_masked.plot.imshow(
                ax = axes[m][t],
                transform = ccrs.PlateCarree(), 
                x = 'lon', y = 'lat',
                vmin = 0, vmax = 60,
                extend = 'max', 
                cmap = 'RdBu_r',
                add_colorbar = False,
            )

            # 在每个子图中再插入一个小子图，绘制当前空间分布的频率直方图
            HIST_BINS = np.arange(0, 60, 5)
            HIST_XLIM = (0, 60)
            HIST_YLIM = (0.0, 0.7)
            HIST_CMAP = 'RdBu_r'
            HIST_NORM = Normalize(vmin=0, vmax=60)
            add_hist_inset(axes[m][t], crps_masked.values,
                            bins=HIST_BINS, 
                            xlim=HIST_XLIM, ylim=HIST_YLIM,
                            cmap=HIST_CMAP, norm=HIST_NORM,
                            if_axvline=False)
    

        axes[m][t].set_facecolor("lightgray")
        axes[m][t].set_title(None)
        # cartopy绘制地图
        axes[m][t].set_extent([70, 139, 10, 62], crs=ccrs.PlateCarree())    
        # 绘制水文气象分区shp
        axes[m][t].add_feature(shp_feature, linestyle='-', linewidth=1.0)

        # 添加列标题和行标题
        axes[m][t].text(25, 30, row_title[m], fontdict=font2) if t == 0 else None
        axes[m][t].set_title(column_title[t], fontdict=font2, pad=30.0) if m == 0 else None


# 添加字母编号
add_panel_labels(axes)

fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)

# 添加colorbar
ax_2_0_pos = axes[2][0].get_position()
ax_2_2_pos = axes[2][2].get_position()

p_ax_cb = fig.add_axes([ax_2_0_pos.x0 * 1.15, 0.1, ax_2_0_pos.width * 1.6, 0.02])
p_cb = fig.colorbar(
    p, 
    cax=p_ax_cb, 
    extend='both', 
    orientation='horizontal'
)
p_cb.set_label(label=r'Weights ($\times 10^{-2}$)', fontdict=font3)
p_ax_cb.tick_params(which='major', direction='in', length=7)
for ticklabel in p_ax_cb.xaxis.get_ticklabels():
    ticklabel.set_fontsize(26)

q_ax_cb = fig.add_axes([ax_2_2_pos.x0 * 1.1, 0.1, ax_2_2_pos.width * 1.6, 0.02])
q_cb = fig.colorbar(
    q,
    cax=q_ax_cb,
    extend='max',
    orientation='horizontal'
)
q_cb.set_label(label='CRPS (mm)', fontdict=font3)
q_ax_cb.tick_params(which='major', direction='in', length=7)
for ticklabel in q_ax_cb.xaxis.get_ticklabels():
    ticklabel.set_fontsize(26)

#%%
# 保存
fig.savefig(f'/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/00_Results_newTest_251209/{code_name}/Output_0/93-1_weight_rawfcstCRPS_map_plot_CCSM4_GEFS_ECMWF_2tscales.jpg')









# %%













