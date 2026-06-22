#%%
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
font2 = {'weight': 'bold', 'size': 18}
font3 = {'weight': 'normal', 'size': 24}

# 水文气候分区shp文件
shp = shpreader.Reader(f"D:/shp file/全国shp/9大流域片/liuyu_WGS84_line.shp")
shp_feature = cfeature.ShapelyFeature(shp.geometries(),
                                      ccrs.PlateCarree(),
                                      edgecolor="black",
                                      facecolor="none")



#######################################################################################
#####    辅助函数
#######################################################################################

def omit_leading_zero(x, pos):
    return f'{x:.2f}'.lstrip('0') if x < 1 else f'{x:.2f}'

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


column_title = [
    'Adaptive Weighting\nwith KL Regularization\n(WEAVE-U-Net++)', 
    'Simple Concatenation', 
    'Equal Weighting', 
    'Prescribed Skill-Based\nWeighting', 
    'Adaptive Weighting'
]

model_name = [
    'pre_verif_result_UNetPP-A-WeightPredictor-T-KLloss-ContinueTraining_3',
    'pre_verif_result_UNetPP-A', 
    'pre_verif_result_UNetPP-A-FW-new', 
    'pre_verif_result_UNetPP-A-SkillBasedWeight-NoWeightLearning',
    'pre_verif_result_UNetPP-A-WeightPredictor-T', 
]

tscale = [7, 14, 21, 28]
row_title = ['7 days', '14 days', '21 days', '28 days']




figsize = (26, 12)
nrows, ncols_logical = 4, 5

col_width_ratio = 1.0   # 每个真正子图列的宽度比（可调整）
gap_width_ratio = 0.9  # 间隙列的相对宽度（0.0 表示无间隙，0.12 表示留出一点空白）

# margins（可按需微调）
# left, right = 0.03, 0.02
# top, bottom = 0.98, 0.02
left, right = 0.1, 0.12
top, bottom = 0.9, 0.03

# 我们用 5 个 GridSpec 列： [col0, gap_col, col1, col2, col3, col4]
widths = [col_width_ratio, gap_width_ratio, col_width_ratio, col_width_ratio, col_width_ratio, col_width_ratio]

fig = plt.figure(figsize=figsize)
# 注意：将 wspace 和 hspace 都设为 0，保证 GridSpec 各单元格间默认无额外空隙
gs = fig.add_gridspec(nrows=nrows, ncols=len(widths), width_ratios=widths,
                      left=left, right=1-right, top=top, bottom=bottom,
                      hspace=0.0, wspace=0.0)

axes = []
for i in range(nrows):
    row_axes = []  # 保存当前行的所有列 ax
    # logical 列映射到实际的 GridSpec 列索引
    mapping = [0, 2, 3, 4, 5]  # logical col0->gs[:,0], col1->gs[:,2], col2->gs[:,3], col3->gs[:,4], col4->gs[:,5]
    for j_logical, j_gs in enumerate(mapping):
        ax = fig.add_subplot(gs[i, j_gs], projection=ccrs.PlateCarree())
        # 关闭刻度标签以避免拥挤（按需打开）
        ax.set_xticks([])
        ax.set_yticks([])
        row_axes.append(ax)
    axes.append(row_axes)


# 绘制评价指标的空间分布

# 时间尺度循环
for t in range(4):
    _t = tscale[t]

    metrix_file_arr = []
    # 模型循环
    for m in range(5):
        # 读取评价指标文件
        filedir = 'D:/00_Code_lsy/MachineLearning_CSGD/S2S_SubX_Multimodel_251209/'
        if m == 1:
            file_name = f'D:/00_Code_lsy/MachineLearning_CSGD/S2S_SubX_Multimodel_250525/{model_name[m]}/Output_0/{_t}days/post_bss_{_t}daysTP_Year20012013.nc'
        else:
            file_name = filedir + f'{model_name[m]}/Output_0/{_t}days/post_bss_{_t}daysTP_Year20012013.nc'
        metrix_file = xr.open_dataarray(file_name).sel(year='AllYears', threshold=90).squeeze()
        metrix_file.close()
        metrix_file_arr.append(metrix_file)

        axes[t][m].set_facecolor("lightgray")

        # WEAVE-U-Net++ 模型绘制bss地图
        if m == 0:
            p_bss = metrix_file.plot.imshow(ax=axes[t][m], 
                        transform=ccrs.PlateCarree(), x='lon', y='lat', 
                        vmin=-30, vmax=30, 
                        extend='both', cmap='RdBu_r', add_colorbar=False,
                        levels=np.linspace(-30, 30, 21))
        # 其他模型，绘制与WEAVE-U-Net++之间的bss差值
        else:
            diff = metrix_file - metrix_file_arr[0]
            q_bss = diff.plot.imshow(ax=axes[t][m], 
                        transform=ccrs.PlateCarree(), x='lon', y='lat', 
                        vmin=-10, vmax=10, 
                        extend='both', cmap='RdBu_r', add_colorbar=False,
                        levels=np.linspace(-10, 10, 11))
            # 统计 diff > 0 (即WEAVE-U-Net++模型表现更好) 的网格占比, delta bss > 0 代表技能变差
            per = np.sum(np.where(diff > 0, 1, 0)) / 928 * 100
            axes[t][m].text(120, 56, f'{per:.2f}%', fontdict=font1)
            # 在每个子图中再插入一个小子图，绘制当前空间分布BSS的频率直方图
            HIST_BINS = np.linspace(-10, 10, 11)
            HIST_XLIM = (-10, 10)
            HIST_YLIM = (0.0, 0.4)
            HIST_CMAP = 'RdBu_r'
            HIST_NORM = Normalize(vmin=-10, vmax=10)
            add_hist_inset(axes[t][m], diff.values, 
                            bins=HIST_BINS, 
                            xlim=HIST_XLIM, ylim=HIST_YLIM, 
                            cmap=HIST_CMAP, norm=HIST_NORM)

        # 其他设置
        axes[t][m].set_xlabel(None)
        axes[t][m].set_ylabel(None)
        axes[t][m].set_title(None)
        axes[t][m].set_extent([70, 139, 10, 62], crs=ccrs.PlateCarree())
        # 绘制水文气象分区shp
        axes[t][m].add_feature(shp_feature, linestyle='-', linewidth=1.5)

        # 添加列标题和行标题
        axes[t][m].text(45, 30, row_title[t], fontdict={'size': 22, 'weight': 'bold'}, ha='center') if m == 0 else None
        axes[t][m].set_title(column_title[m], fontdict=font2, va='center', pad=40.0) if t == 0 else None


# 添加颜色条
ax_3_0_pos = axes[3][0].get_position()
p_bss_ax_cb = fig.add_axes([ax_3_0_pos.x0+ax_3_0_pos.width+0.02, ax_3_0_pos.y0+0.02, 0.015, 0.82])
p_bss_cb = fig.colorbar(
    p_bss, 
    cax=p_bss_ax_cb, 
    extend='max', 
    drawedges=True, 
    orientation='vertical'
)
p_bss_cb.set_label(label=r'$\mathrm{BSS\ _{\geq90th}}$' + ' (%)', fontdict=font3, labelpad=15)
p_bss_ax_cb.tick_params(which='major', direction='in', length=7)
for ticklabel in p_bss_ax_cb.yaxis.get_ticklabels():
    ticklabel.set_fontsize(22)

# delta bss
ax_3_4_pos = axes[3][4].get_position()
q_bss_ax_cb = fig.add_axes([ax_3_4_pos.x0+ax_3_4_pos.width+0.02, ax_3_4_pos.y0+0.02, 0.015, 0.82])
q_bss_cb = fig.colorbar(
    q_bss, 
    cax=q_bss_ax_cb, 
    extend='both', 
    drawedges=True, 
    orientation='vertical'
)
q_bss_cb.set_label(label=r'$\Delta_{\mathrm{BSS}}$' + '(%)', fontdict=font3, labelpad=15)
q_bss_ax_cb.tick_params(which='major', direction='in', length=7)
for ticklabel in q_bss_ax_cb.yaxis.get_ticklabels():
    ticklabel.set_fontsize(22)


# 存储
fig.savefig(filedir + f'draw_251226/supple_pic_260521/s10_bss90_ablation_comparison_map.jpg')




# %%
