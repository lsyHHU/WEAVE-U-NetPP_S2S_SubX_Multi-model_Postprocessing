# %%
"""
只绘制陆地网格的可靠性图
"""

import sys, os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


tscale = [7, 14, 21, 28]

model_name = ['Raw', 'MME', 'EMOS', 'UNetPP-A-WeightPredictor-T-KLloss-ContinueTraining']
model_label = ['ECMWF', 'MME', 'EMOS', 'WEAVE-UNet++']
nmodel = len(model_name)

# 阈值
ths = 10
# climatological无技能线
clima_line = 0.1

# 读取数据
ens_prob_arr = np.zeros((4, 4, 10))  # 4 tscale, 4 model
obs_freq_arr = np.zeros((4, 4, 10))
bins_sample_arr = np.zeros((4, 4, 10))
rel_arr = np.zeros((4, 4))
res_arr = np.zeros((4, 4))

datadir = f'/home/liusy/MachineLearning_CSGD/S2S_SubX_Multimodel_250525/01_reliability_diagram/'
for t in range(4):
    for m in range(4):
        _m = model_name[m]
        _t = tscale[t]

        if m == 0:  # raw
            data_subdir = datadir + f'{_m}/{model_label[m]}/{_t}days/'
        else:
            data_subdir = datadir + f'{_m}/{_t}days/'

        ens_prob = np.load(data_subdir + f'ens_prob_{ths}.npy', allow_pickle=True)
        obs_freq = np.load(data_subdir + f'obs_freq_{ths}.npy', allow_pickle=True)
        bins_sample = np.load(data_subdir + f'bins_sample_{ths}.npy', allow_pickle=True)
        rel = np.load(data_subdir + f'rel_{ths}.npy', allow_pickle=True)
        res = np.load(data_subdir + f'res_{ths}.npy', allow_pickle=True)

        if len(ens_prob) != 10 and len(obs_freq) != 10:
            idx = np.where(bins_sample == 0)[0]
            for idx_ in idx:
                ens_prob = np.insert(ens_prob, idx_, np.nan)
                obs_freq = np.insert(obs_freq, idx_, np.nan)

        ens_prob_arr[t, m, :] = ens_prob
        obs_freq_arr[t, m, :] = obs_freq
        bins_sample_arr[t, m, :] = bins_sample
        rel_arr[t, m] = rel
        res_arr[t, m] = res

# 找出各个时间尺度下，三个模型中REL的最小值，RES的最大值
rel_optimal = np.nanmin(rel_arr, axis=1)    # （4，）
res_optimal = np.nanmax(res_arr, axis=1)






#%%

# 绘图

upper, lower = 1.05, -0.05
x = np.linspace(0, 1, 11)

# 创建总图像
fig = plt.figure(figsize=(18, 18))
outer = gridspec.GridSpec(2, 2, wspace=0.4, hspace=0.4)

# 存储主图与小图
axes_main = []
axes_small = []

for i in range(2):  # 两行
    for j in range(2):  # 两列
        # 对于每个主图+小图组合，在每个 outer cell 中创建 2 行 1 列的子网格，高度比例为 3:1
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=outer[i, j],
            height_ratios=[3, 1],  # 主图:小图 = 3:1
            hspace=0  # 无间距
        )

        # 主图
        ax_main = fig.add_subplot(inner[0])
        axes_main.append(ax_main)

        # 小图，共享 x 轴
        ax_small = fig.add_subplot(inner[1], sharex=ax_main)
        ax_small.set_yscale('log')   # 对数坐标
        axes_small.append(ax_small)

        # 可选：隐藏小图的 x ticklabels，防止重叠
        plt.setp(ax_main.get_xticklabels(), visible=False)


colors = ['#DB9C4D', '#70B48F', '#293890', '#BF1D2D']
title = ['Timescale: 7 days', 'Timescale: 14 days', 'Timescale: 21 days', 'Timescale: 28 days']

for t in range(4):
    # 绘制perfect 1：1对角线
    axes_main[t].plot([lower, upper], [lower, upper], ls='--', color='grey', lw=2.0)   # 对角线
    
    # 绘制三个模型的可靠性图
    axes_main[t].plot(ens_prob_arr[t, 0, :], obs_freq_arr[t, 0, :], marker='o', color=colors[0], label=model_label[0], ms=10, lw=2)
    axes_main[t].plot(ens_prob_arr[t, 1, :], obs_freq_arr[t, 1, :], marker='s', color=colors[1], label=model_label[1], ms=10, lw=2)
    axes_main[t].plot(ens_prob_arr[t, 2, :], obs_freq_arr[t, 2, :], marker='^', color=colors[2], label=model_label[2], ms=10, lw=2)
    axes_main[t].plot(ens_prob_arr[t, 3, :], obs_freq_arr[t, 3, :], marker='*', color=colors[3], label=model_label[3], ms=10, lw=2)
    
    # 绘制climatological无技能线
    x1 = np.arange(lower, clima_line + 0.0001, 0.01)
    y1 = 0.5 * (x1 + clima_line)  # point and slope
    y2 = lower
    axes_main[t].fill_between(x1, y1, y2, facecolor = 'grey', alpha = 0.2)

    x2 = np.arange(clima_line, upper + 0.0001, 0.01)
    y3 = 0.5 * (x2 + clima_line)
    y4 = upper
    axes_main[t].fill_between(x2, y3, y4, facecolor = 'grey', alpha = 0.2)

    axes_main[t].axvline(clima_line, color = 'grey', lw = 0.3)
    axes_main[t].axhline(clima_line, color = 'grey', lw = 0.3)
    axes_main[t].plot([lower, upper], [y1[0], y3[-1]], color='grey', lw=2.0)

    # axes_main设置
    axes_main[t].set_xticks(x)
    axes_main[t].set_xlim(lower, upper)
    axes_main[t].set_ylim(lower, upper)
    axes_main[t].tick_params(labelsize=20, which='major', direction='in', length=5)
    axes_main[t].set_ylabel('Observed frequency', fontsize=22, labelpad=20)
    axes_main[t].set_title(title[t], fontsize=26, fontweight='bold', pad=30)
    axes_main[t].legend(loc='lower right', fontsize=18)

    # 绘制频率柱状图
    x_bar = np.linspace(0.05, 0.95, 10)
    w = 0.015
    axes_small[t].bar(x_bar-1.5*w, bins_sample_arr[t, 0, :], width=w, color=colors[0], label=model_name[0], alpha=0.9)
    axes_small[t].bar(x_bar-0.5*w, bins_sample_arr[t, 1, :], width=w, color=colors[1], label=model_name[1], alpha=0.9)
    axes_small[t].bar(x_bar+0.5*w, bins_sample_arr[t, 2, :], width=w, color=colors[2], label=model_name[2], alpha=0.9)
    axes_small[t].bar(x_bar+1.5*w, bins_sample_arr[t, 3, :], width=w, color=colors[3], label=model_name[3], alpha=0.9)

    # axes_small设置
    axes_small[t].set_xticks(x)
    axes_small[t].set_xticklabels(np.round(x, 1))
    axes_small[t].tick_params(labelsize=20, which='major', direction='in', length=5)
    axes_small[t].set_xlabel('Forecast probability', fontsize=22, labelpad=20)
    axes_small[t].set_ylabel('Samples', fontsize=22, labelpad=20)

    # 绘制rel和res表格
    col_labels = ['REL', 'RES']
    cell_text = [
        [f'{np.round(rel_arr[t, 0], 4)}', f'{np.round(res_arr[t, 0], 4)}'],
        [f'{np.round(rel_arr[t, 1], 4)}', f'{np.round(res_arr[t, 1], 4)}'],
        [f'{np.round(rel_arr[t, 2], 4)}', f'{np.round(res_arr[t, 2], 4)}'],  
        [f'{np.round(rel_arr[t, 3], 4)}', f'{np.round(res_arr[t, 3], 4)}']
    ]
    table = axes_main[t].table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc='center',
        colLoc='center',
        loc='upper left',
        colWidths=[0.2, 0.2],
        zorder=10
    )
    table.scale(1, 1.8)  # 宽度不变，高度变为原来的1.8倍
    # 遍历所有单元格，设置颜色
    for (row, col), cell in table.get_celld().items():
        if row == 0:    # 第0行是列标题，字体黑色
            cell.get_text().set_color('black')
            cell.set_fontsize(16)
            cell.set_text_props(weight='bold')  # 可以加粗标题
        else:   # 数据行，设置对应颜色
            cell.get_text().set_color(colors[row-1])
            cell.set_fontsize(16)

        # 判断当前的rel和res是否为最优值，最优值加粗
        if row != 0 and col == 0:   # rel
            if cell.get_text().get_text() == f'{np.round(rel_optimal[t], 4)}':
                cell.set_text_props(weight='bold')
        if row != 0 and col == 1:   # res
            if cell.get_text().get_text() == f'{np.round(res_optimal[t], 4)}':
                cell.set_text_props(weight='bold')


# 存储
fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9)
# fig.savefig(f'/home/liusy/MachineLearning_CSGD/S2S_SubX_Multimodel_250525/01_reliability_diagram/reliability_diagram_WEAVEUnet_{ths}.jpg', dpi=600)




# %%
