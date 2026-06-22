#%%
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


# =========================
# 生成画布：18行地图 + 1行colorbar；1列行标签 + 4列地图
# =========================

fig = plt.figure(figsize=(15, 30))

n_model = 18
n_col = 4

gs = gridspec.GridSpec(
    nrows=n_model + 1,          # 18行地图 + 1行colorbar
    ncols=n_col + 1,            # 1列模式名称 + 4列地图
    figure=fig,
    width_ratios=[0.75, 1, 1, 1, 1],
    height_ratios=[1] * n_model + [0.18],
    left=0.05,
    right=0.98,
    bottom=0.1,
    top=0.94,
    wspace=0.08,
    hspace=0.02
)

axes = [[None for _ in range(n_col)] for _ in range(n_model)]
label_axes = [None for _ in range(n_model)]
cbar_axes = [None for _ in range(n_col)]

# 左侧行标签轴
for m in range(n_model):
    label_ax = fig.add_subplot(gs[m, 0])
    label_ax.axis('off')
    label_ax.text(
        0.98, 0.5, model_name[m],
        ha='right',
        va='center',
        fontsize=16,
        fontweight='bold'
    )
    label_axes[m] = label_ax

# 地图轴
for m in range(n_model):
    for t in range(n_col):
        ax = fig.add_subplot(gs[m, t + 1], projection=ccrs.PlateCarree())
        axes[m][t] = ax

# 底部colorbar轴
for t in range(n_col):
    cbar_axes[t] = fig.add_subplot(gs[-1, t + 1])



# =========================
# 基本参数
# =========================

root_dir = '/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/00_Results_newTest_251209/'
code_name = 'UNetPP-A-WeightPredictor-T-KLloss-ContinueTraining_3'

tscale = [7, 7, 14, 14]
column_title = [
    'Weights (7 days)',
    'CRPS (7 days)',
    'Weights (14 days)',
    'CRPS (14 days)'
]

model_name = [
    '46LCESM1', 'CCSM4', 'CFSv2', 'FIMr1p1', 'GEFS', 'NESM', 'GEOS_V2p1',
    'BoM', 'CMA', 'CNRM', 'CPTEC', 'ECCC', 'ECMWF', 'HMCR', 'IAPCAS',
    'ISACCNR', 'KMA', 'UKMO'
]

# 保存每一列的mappable，用于底部colorbar
mappables = [None for _ in range(n_col)]



# =========================
# 绘图循环
# =========================

# 模型循环
for m in range(n_model):
    
    # 时间尺度循环
    for t in range(n_col):
        _t = tscale[t]
        ax = axes[m][t]

        # -------------------------
        # 第1列和第3列：Weights
        # -------------------------
        if t == 0 or t == 2:
            # 读取权重文件
            weights_da = xr.open_dataarray(root_dir + f'{code_name}/Output_0/{_t}days/Unetfcst_w/weights_{_t}days_Year20012013.nc').sel(model=model_name[m])
            weights_issuetime_mean_da = weights_da.mean(dim='issuetime')
            weights_masked = weights_issuetime_mean_da * mask_da
            weights_masked = weights_masked * 100.   # 权重乘100

            # 绘制权重图
            p = weights_masked.plot.imshow(
                ax = ax,
                transform = ccrs.PlateCarree(), 
                x = 'lon', y = 'lat',
                vmin = 3, vmax = 8,
                extend = 'both', 
                cmap = 'BrBG',
                add_colorbar = False,
            )

            mappables[t] = p

        # -------------------------
        # 第2列和第4列：CRPS
        # -------------------------
        else:
            # 读取CRPS文件
            crps_da = xr.open_dataarray(f'/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/00_Results/crps_by_model_{_t}days.nc').sel(model=model_name[m])
            crps_issuetime_mean_da = crps_da.mean(dim='issuetime', skipna=True)
            crps_masked = crps_issuetime_mean_da * mask_da

            if t == 1:
                crps_vmin, crps_vmax = 0, 20
            elif t == 3:
                crps_vmin, crps_vmax = 0, 40

            # 绘制crps分布图
            q = crps_masked.plot.imshow(
                ax = ax,
                transform = ccrs.PlateCarree(), 
                x = 'lon', y = 'lat',
                vmin = crps_vmin, vmax = crps_vmax,
                extend = 'max', 
                cmap = 'RdBu_r',
                add_colorbar = False,
            )
            
            mappables[t] = q
            
        # -------------------------
        # 子图通用设置
        # -------------------------

        axes[m][t].set_facecolor("lightgray")
        ax.set_extent([70, 139, 15, 57], crs=ccrs.PlateCarree())
        ax.add_feature(shp_feature, linestyle='-', linewidth=0.8)

        # 去掉xarray自动添加的坐标轴标签
        ax.set_xlabel('')
        ax.set_ylabel('')

        # 去掉地图坐标刻度，避免18行图过于拥挤
        ax.set_xticks([])
        ax.set_yticks([])

        # 去掉默认标题
        ax.set_title(None)

        # 第一行添加列标题
        if m == 0:
            ax.set_title(
                column_title[t],
                fontsize=18,
                fontweight='bold',
                pad=12
            )

        # 控制地图边框线宽
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

        # 地图变扁
        ax.set_aspect('auto')
        ax.set_box_aspect(0.5)


# =========================
# 添加底部4个colorbar
# =========================
cbar_labels = [
    r'Weights ($\times 10^{-2}$)',
    'CRPS (mm)',
    r'Weights ($\times 10^{-2}$)',
    'CRPS (mm)'
]

cbar_ticks = [
    np.arange(3, 9, 1),
    np.arange(0, 21, 5),
    np.arange(3, 9, 1),
    np.arange(0, 41, 10)
]

for t in range(n_col):

    extend = 'both' if t == 0 or t == 2 else 'max'

    cb = fig.colorbar(
        mappables[t],
        cax=cbar_axes[t],
        orientation='horizontal',
        extend = extend,
    )

    cb.set_label(
        cbar_labels[t],
        fontsize=15,
        labelpad=8
    )

    cb.set_ticks(cbar_ticks[t])
    cb.ax.tick_params(
        labelsize=13,
        direction='in',
        length=5,
        width=0.8
    )

    for spine in cbar_axes[t].spines.values():
        spine.set_linewidth(0.6)



# 保存
fig.savefig(f'/home/liusy/MachineLearning_CSGD/S2S_SubX_Multimodel_250525/03_Unet++_newTest_251209/00_pic/93-9-1_weight_crps_map_plot_7_14days.jpg', 
    dpi=600, bbox_inches='tight')









# %%













