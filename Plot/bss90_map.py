#%%
import string
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['figure.dpi'] = 600
font1 = {'weight': 'normal', 'size': 18}
font1_bold = {'weight': 'bold', 'size': 18, 'color': 'blue'}
font2 = {'weight': 'bold', 'size': 28}
font3 = {'weight': 'normal', 'size': 30}

# 水文气候分区shp文件
shp = shpreader.Reader(f"D:/shp file/全国shp/9大流域片/liuyu_WGS84_line.shp")
shp_feature = cfeature.ShapelyFeature(shp.geometries(),
                                      ccrs.PlateCarree(),
                                      edgecolor="black",
                                      facecolor="none")

def add_panel_labels(
    axes,
    x=0.02,
    y=1.12,
    fontsize=18,
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

#######################################################################################
#####    辅助函数
#######################################################################################

def omit_leading_zero(x, pos):
    return f'{x:.2f}'.lstrip('0') if x < 1 else f'{x:.2f}'

def calculate_improvement(metrix_name, raw_metrix_da, post_metrix_da):
    """
    计算一个空间范围内，评价指标改善的网格比例
    metrix_name: 'PCC', 'ACC', 'RMSE', 'RB', 'CRPSS', 'RPSS'
    raw_metrix_da, post_metrix_da: 原始预报/后处理预报评价指标, DataArray
    """
    ngrid = 928
    if metrix_name in ['PCC', 'ACC', 'CRPSS', 'RPSS', 'BSS', 'SCC', 'ROCSS', 'ETS']:
        metrix_improve = (post_metrix_da > raw_metrix_da).sum().values / ngrid * 100.
    elif metrix_name == 'RMSE':
        metrix_improve = (post_metrix_da < raw_metrix_da).sum().values / ngrid * 100.
    elif metrix_name == 'RB':
        metrix_improve = (abs(post_metrix_da) < abs(raw_metrix_da)).sum().values / ngrid * 100.
    return metrix_improve

def get_bold_value_index(metrix_name, metrix_values_arr):
    """
    给出评价指标的列表，找到最优值的索引
    """
    if metrix_name in ['PCC', 'ACC', 'CRPSS', 'RPSS', 'BSS', 'SCC', 'ROCSS', 'ETS']:
        return np.argmax(metrix_values_arr, axis=1)
    elif metrix_name == 'RMSE':
        return np.argmin(metrix_values_arr, axis=1)
    elif metrix_name == 'RB':
        return np.argmin(np.abs(metrix_values_arr), axis=1)

def add_hist_inset(parent_ax, data, bins, xlim, ylim, cmap, norm):
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

    hist_ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, zorder=4)

    # 透明背景
    hist_ax.set_facecolor('none')
    hist_ax.patch.set_alpha(0.0)

    # 固定坐标范围
    hist_ax.set_xlim(*xlim)
    hist_ax.set_ylim(*ylim)
    hist_ax.set_xticks([xlim[0], 0, xlim[1]])
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
    
#######################################################################################


# 生成画布
fig = plt.figure(figsize=(22, 15))
gs = gridspec.GridSpec(4, 5, width_ratios=[1, 1, 1, 1, 1.3], wspace=0.0, hspace=0.2)
axes = [[None for _ in range(5)] for _ in range(4)]  # 二维列表
for i in range(4):       # 行
    for j in range(5):   # 列
        if j != 4:
            ax = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree())
        else:
            ax = fig.add_subplot(gs[i, j])
        axes[i][j] = ax


root_dir = 'D:/00_Code_lsy/MachineLearning_CSGD/S2S_SubX_Multimodel_251209/'

tscale = [7, 14, 21, 28]

model_name = [
    'pre_verif_result_rawfcst', 'pre_verif_result_MMEfcst', 
    'pre_verif_result_EMOSfcst', 'pre_verif_result_UNetPP-A-WeightPredictor-T-KLloss-ContinueTraining_3'
]
nmodel = len(model_name)

column_title = ['ECMWF', 'MME', 'EMOS', 'WEAVE-U-Net++', 'CDF curves']
row_title = ['7 days', '14 days', '21 days', '28 days']
colors = ['#DB9C4D', '#70B48F', '#293890', '#BF1D2D']
# colors = ['#B8945E', '#7FA08A', '#5E7392', '#9C5A66']


raw_metrix_file = []   # 存放原始预报指标da的列表
mme_metrix_file = []   # 存放mme模型指标da的列表
emos_metrix_file = []   # 存放emos模型指标da的列表
unet_metrix_file = []   # 存放unet模型指标da的列表
metrix_mean_arr = np.zeros((4, 4))
metrix_improvment_arr = np.zeros((4, 3))

# 时间尺度循环
for t in range(4):
    _t = tscale[t]

    # 模型循环
    for m in range(5):

        ################
        # 绘制3个模型的评价指标的空间分布
        ################
        if m != 4:
            if m == 0:  # raw (ECMWF)
                file_name = f'D:/00_Code_lsy/MachineLearning_CSGD/S2S_SubX_Multimodel_250525/{model_name[m]}/{_t}days/raw_bss_{_t}daysTP_Year20012013.nc'
                metrix_file = xr.open_dataarray(file_name).sel(year='AllYears', model='ECMWF', threshold=90).squeeze()
                raw_metrix_file.append(metrix_file)
            elif m == 1:  # MME
                file_name = root_dir + f'{model_name[m]}/{_t}days/mme_bss_{_t}daysTP_Year20012013.nc'
                metrix_file = xr.open_dataarray(file_name).sel(year='AllYears', threshold=90).squeeze()
                mme_metrix_file.append(metrix_file)
            elif m == 2:  # emos
                file_name = root_dir + f'{model_name[m]}/{_t}days/emos_bss_{_t}daysTP_Year20012013.nc'
                metrix_file = xr.open_dataarray(file_name).sel(year='AllYears', threshold=90).squeeze()
                emos_metrix_file.append(metrix_file)
            else:   # Unet-based
                file_name = root_dir + f'{model_name[m]}/Output_0/{_t}days/post_bss_{_t}daysTP_Year20012013.nc'
                metrix_file = xr.open_dataarray(file_name).sel(year='AllYears', threshold=90).squeeze()
                unet_metrix_file.append(metrix_file)
            metrix_file.close()

            axes[t][m].set_facecolor("lightgray")

            # 绘制metrix map
            p = metrix_file.plot.imshow(
                ax = axes[t][m],
                transform = ccrs.PlateCarree(), 
                x = 'lon', y = 'lat',
                vmin = -30, vmax = 30,
                extend = 'both', 
                cmap = 'RdBu_r', 
                add_colorbar = False,
                levels = np.linspace(-30, 30, 31)
            )

            axes[t][m].set_title(None)
            # cartopy绘制地图
            axes[t][m].set_extent([70, 139, 10, 62], crs=ccrs.PlateCarree())    
            # 绘制水文气象分区shp
            axes[t][m].add_feature(shp_feature, linestyle='-', linewidth=1.0)

            # 在每个子图中再插入一个小子图，绘制当前空间分布BSS的频率直方图
            HIST_BINS = np.arange(-30, 35, 5)
            HIST_XLIM = (-30, 30)
            HIST_YLIM = (0.0, 0.50)
            HIST_CMAP = 'RdBu_r'
            HIST_NORM = Normalize(vmin=-30, vmax=30)
            add_hist_inset(axes[t][m], metrix_file.values, 
                           bins=HIST_BINS, 
                           xlim=HIST_XLIM, ylim=HIST_YLIM, 
                           cmap=HIST_CMAP, norm=HIST_NORM)
        
            # 计算每个指标在整个区域上的平均值
            metrix_mean = metrix_file.mean(skipna=True).values
            metrix_mean_arr[t][m] = metrix_mean

            # 计算改善网格比例
            if m != 0:
                metrix_improve = calculate_improvement(
                    metrix_name='BSS',
                    raw_metrix_da=raw_metrix_file[t],
                    post_metrix_da=metrix_file
                )
                metrix_improvment_arr[t, m-1] = metrix_improve

        ################
        # 绘制3个模型的CDF曲线
        ################
        else:
            raw_metrix_arr = raw_metrix_file[t].values.flatten()
            mme_matrix_arr = mme_metrix_file[t].values.flatten()
            emos_metrix_arr = emos_metrix_file[t].values.flatten()
            unet_metrix_arr = unet_metrix_file[t].values.flatten()

            raw_metrix_sorted = np.sort(raw_metrix_arr[~np.isnan(raw_metrix_arr)])
            mme_matrix_sorted = np.sort(mme_matrix_arr[~np.isnan(mme_matrix_arr)])
            emos_metrix_sorted = np.sort(emos_metrix_arr[~np.isnan(emos_metrix_arr)])
            unet_metrix_sorted = np.sort(unet_metrix_arr[~np.isnan(unet_metrix_arr)])

            raw_cdf = np.linspace(0, 1, len(raw_metrix_sorted), endpoint=False)
            mme_cdf = np.linspace(0, 1, len(mme_matrix_sorted), endpoint=False)
            emos_cdf = np.linspace(0, 1, len(emos_metrix_sorted), endpoint=False)
            unet_cdf = np.linspace(0, 1, len(unet_metrix_sorted), endpoint=False)

            axes[t][m].plot(raw_metrix_sorted, raw_cdf, color=colors[0], label='ECMWF', marker='o', markevery=100, lw=1.5, ms=6)
            axes[t][m].plot(mme_matrix_sorted, mme_cdf, color=colors[1], label='MME', marker='s', markevery=100, lw=1.5, ms=6)
            axes[t][m].plot(emos_metrix_sorted, emos_cdf, color=colors[2], label='EMOS', marker='^', markevery=100, lw=1.5, ms=6)
            axes[t][m].plot(unet_metrix_sorted, unet_cdf, color=colors[3], label='WEAVE', marker='P', markevery=100, lw=1.5, ms=6)
            axes[t][m].axvline(0, color='gray', linewidth=1.5)
            axes[t][m].set_xlim(-35, 35)
            axes[t][m].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            axes[t][m].set_yticklabels([0, 0.25, 0.5, 0.75, 1.0])

            axes[t][m].set_ylabel('CDF', fontsize=24, labelpad=20)
            axes[t][m].set_xlabel('BSS (%)', fontsize=24) if t == 3 else None
            axes[t][m].tick_params(labelsize=20)
            axes[t][m].legend(loc='upper left', fontsize=16) if t == 0 else None
            # axes[t][m].grid(True, alpha=0.5, which='both')

            # 启用右侧 y 轴
            axes[t][m].yaxis.set_label_position('right')
            axes[t][m].yaxis.tick_right()
            axes[t][m].yaxis.set_major_formatter(FuncFormatter(omit_leading_zero))
            axes[t][m].tick_params(axis='y', which='both', left=False)

            # 绘制指标等于0处的水平线
            raw_idx = np.argmin(np.abs(raw_metrix_sorted - 0))
            raw_y0 = raw_cdf[raw_idx]
            axes[t][m].axhline(y=raw_y0, xmin=0.5, xmax=max(raw_metrix_sorted), color=colors[0], linestyle='--', linewidth=1.5)

            mme_idx = np.argmin(np.abs(mme_matrix_sorted - 0))
            mme_y0 = mme_cdf[mme_idx]
            axes[t][m].axhline(y=mme_y0, xmin=0.5, xmax=max(mme_matrix_sorted), color=colors[1], linestyle='--', linewidth=1.5)

            emos_idx = np.argmin(np.abs(emos_metrix_sorted - 0))
            emos_y0 = emos_cdf[emos_idx]
            axes[t][m].axhline(y=emos_y0, xmin=0.5, xmax=max(emos_metrix_sorted), color=colors[2], linestyle='--', linewidth=1.5)

            unet_idx = np.argmin(np.abs(unet_metrix_sorted - 0))
            unet_y0 = unet_cdf[unet_idx]
            axes[t][m].axhline(y=unet_y0, xmin=0.5, xmax=max(unet_metrix_sorted), color=colors[3], linestyle='--', linewidth=1.5)


        ################
        # 子图设置
        ################
        # 添加列标题和行标题
        axes[t][m].text(25, 30, row_title[t], fontdict=font2, ha='center') if m == 0 else None
        axes[t][m].set_title(column_title[m], fontdict=font2, pad=45.0) if t == 0 else None

        
# 添加统计数据
bold_mean_idx = get_bold_value_index(metrix_name='BSS', metrix_values_arr=metrix_mean_arr)
bold_improment_idx = np.argmax(metrix_improvment_arr, axis=1)
for t in range(4):
    for m in range(4):
        # 添加均值（最大值加粗）
        if m == bold_mean_idx[t]:
            axes[t][m].text(72, 56, f'{metrix_mean_arr[t, m]:.2f}', fontdict=font1_bold)
        else:
            axes[t][m].text(72, 56, f'{metrix_mean_arr[t, m]:.2f}', fontdict=font1)
    for m in range(3):
        # 添加改善网格比例（最大值加粗）
        if m == bold_improment_idx[t]:
            axes[t][m+1].text(115, 56, f'{metrix_improvment_arr[t, m]:.2f}%', fontdict=font1_bold)
        else:
            axes[t][m+1].text(115, 56, f'{metrix_improvment_arr[t, m]:.2f}%', fontdict=font1)


fig.set_constrained_layout(False)
fig.subplots_adjust(left=0.15, right=0.9, bottom=0.15, top=0.9)
fig.canvas.draw()
for i in range(4):
    # 以每行第一个 map 轴为参考（axes[i][0]），将该行 CDF 轴（axes[i][4]）的 y0 和 height 与参考一致
    ref_pos = axes[i][0].get_position()   # BBox in figure coords
    cdf_ax = axes[i][4]
    cpos = cdf_ax.get_position()
    # 保持 cdf_ax 的 x0 和 width，不改变水平位置，只对齐垂直位置和高度
    new_pos = [cpos.x0, ref_pos.y0, cpos.width, ref_pos.height]
    cdf_ax.set_position(new_pos)
fig.canvas.draw()

# 添加字母编号
add_panel_labels(axes)

# 添加colorbar
p_ax_cb = fig.add_axes([0.15, 0.1, 0.55, 0.015])
p_cb = fig.colorbar(p, 
                    cax=p_ax_cb, 
                    extend='both', 
                    drawedges=True, 
                    orientation='horizontal')
p_cb.set_label(label=r'$\mathrm{BSS\ _{\geq90th}}$' + ' (%)', fontdict=font3)
p_ax_cb.tick_params(which='major', direction='in', length=7)
for ticklabel in p_ax_cb.xaxis.get_ticklabels():
    ticklabel.set_fontsize(26)



# 保存图片
fig.savefig(root_dir + 'draw_251226/pic_251226/01_bss90_map_emos_new.jpg')



# %%
