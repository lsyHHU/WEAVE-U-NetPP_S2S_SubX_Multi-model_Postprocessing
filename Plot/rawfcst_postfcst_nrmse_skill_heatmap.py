#%%
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


tscale = [7, 14, 21, 28]
region_name = ['Mainland\nChina', 'CB', 'HaiRB', 'HuaiRB', 'PRB', 'SEB', 'SLRB', 'SWB', 'YlRB', 'YtzRB']
title = ['Timescale: 7 days', 'Timescale: 14 days', 'Timescale: 21 days', 'Timescale: 28 days']
model_name = ['ECMWF', 'MME', 'EMOS', 'WEAVE-U-Net++']
model_yticklabel = ['ECMWF', 'MME', 'EMOS', 'WEAVE-\nU-Net++']
metric = 'nrmse'


path = 'D:/00_Code_lsy/MachineLearning_CSGD/S2S_SubX_Multimodel_251209/'


data_arr = np.zeros((4, 4, 10))   # 4 tscales, 4 models, 10 regions

# 读取数据

# 时间尺度循环
for t in range(4):
    _t = tscale[t]

    # 模型循环
    for m in range(4):
        if m == 0:  # rawfcst ECMWF
            filedir = 'D:/00_Code_lsy/MachineLearning_CSGD/S2S_SubX_Multimodel_250525/pre_verif_result_rawfcst/'
            data_nc = xr.open_dataarray(filedir + f'{_t}days/raw_{metric}_{_t}daysTP_Year20012013.nc').sel(model='ECMWF')
        elif m == 1:  # MME
            filedir = 'D:/00_Code_lsy/MachineLearning_CSGD/S2S_SubX_Multimodel_251209/pre_verif_result_MMEfcst/'
            data_nc = xr.open_dataarray(filedir + f'{_t}days/mme_{metric}_{_t}daysTP_Year20012013.nc')
        elif m == 2:  # EMOS
            filedir = 'D:/00_Code_lsy/MachineLearning_CSGD/S2S_SubX_Multimodel_251209/pre_verif_result_EMOSfcst/'
            data_nc = xr.open_dataarray(filedir + f'{_t}days/emos_{metric}_{_t}daysTP_Year20012013.nc')
        elif m == 3:  # WEAVE-U-Net++
            filedir = 'D:/00_Code_lsy/MachineLearning_CSGD/S2S_SubX_Multimodel_251209/pre_verif_result_UNetPP-A-WeightPredictor-T-KLloss-ContinueTraining_3/Output_0/'
            data_nc = xr.open_dataarray(filedir + f'{_t}days/post_{metric}_{_t}daysTP_Year20012013.nc')
        data_nc = data_nc.sel(year='AllYears').squeeze()
        data_nc.close()

        # 区域循环
        for r in range(10):
            _r = region_name[r]
            # 读取区域坐标
            if r == 0:   # 全国
                region_coords = pd.read_csv(path + 'CN_land_coords/land_coords.csv')
            else:   # 各个流域
                region_coords = pd.read_csv(path + f'CN_land_coords/land_coords_{_r}.csv')
            region_coords_lat = xr.DataArray(region_coords['lat'].values, dims='new')
            region_coords_lon = xr.DataArray(region_coords['lon'].values, dims='new')
            # 读取区域CRPSS数据
            crpss_region = data_nc.sel(lat=region_coords_lat, lon=region_coords_lon)
            # 计算区域CRPSS均值
            crpss_regional_mean = crpss_region.mean(dim=['new'], skipna=True)
            data_arr[t, m, r] = crpss_regional_mean.values

# 得到最优值的索引
nrmse_min_idx = np.argmin(data_arr, axis=1)  # 4 tscales, 10 regions



# 绘图
fig, ax = plt.subplots(2, 2, figsize=(24, 12))
ax = ax.flatten()

nrmse_norm = mpl.colors.Normalize(vmin=0.2, vmax=1.5)
nrmse_cmap = mpl.cm.get_cmap('YlOrRd')

# 绘制热力图
for t in range(4):
    p = ax[t].pcolormesh(data_arr[t, :, :], cmap=nrmse_cmap, norm=nrmse_norm, alpha=0.75, edgecolor='k')
    
    # 子图设置
    ax[t].set_yticks(np.arange(4)+0.5)
    ax[t].set_yticklabels(model_yticklabel, fontsize=20)
    ax[t].set_xticks(np.arange(10)+0.5)
    ax[t].set_xticklabels(region_name, fontsize=20)
    ax[t].tick_params(pad=10)
    ax[t].set_xlabel('Hydroclimatic region', fontsize=24) if t >= 2 else None
    ax[t].set_ylabel('Model', fontsize=24, labelpad=20) if t == 0 or t == 2 else None
    ax[t].set_title(title[t], fontsize=26, fontweight='bold', pad=10)

    ax[t].text(
        0.015, 1.125, f'({chr(97+t)})',
        transform=ax[t].transAxes,
        ha='left',
        va='top',
        fontsize=26,
        fontweight='bold'
    )

    # 单元格中添加数据
    for i in range(4):   # 4 models
        for j in range(10):   # 10 regions
            val = data_arr[t, i, j] 
            rgba = nrmse_cmap(nrmse_norm(val))    # 映射到颜色（R,G,B,A）

            # 如果是最大值，则用蓝色加粗
            if nrmse_min_idx[t, j] == i:
                ax[t].text(
                    j + 0.5, i + 0.5, f'{val:.2f}',
                    ha='center', va='center',
                    fontsize=18, color='blue', fontweight='bold'
                )
            else:
                # 计算感知亮度（0=黑, 1=白），阈值可调
                r, g, b, _ = rgba
                luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                txt_color = 'black' if luminance > 0.5 else 'white'
                ax[t].text(
                    j + 0.5, i + 0.5, f'{val:.2f}',
                    ha='center', va='center',
                    fontsize=18, color=txt_color
                )

fig.subplots_adjust(left=0.1, right=0.98, bottom=0.25, top=0.95, hspace=0.45, wspace=0.2)

# 添加colorbar
p_ax_cb = fig.add_axes([ax[2].get_position().x0, 
                        ax[3].get_position().y0-0.15, 
                        ax[3].get_position().x0 + ax[3].get_position().width - ax[2].get_position().x0, 
                        0.02])
p_cb = fig.colorbar(p, 
                    cax=p_ax_cb, 
                    extend='both', 
                    drawedges=False, 
                    orientation='horizontal')
p_cb.set_label(label='nRMSE', fontsize=26)
p_ax_cb.tick_params(which='major', direction='in', length=7)
for ticklabel in p_ax_cb.xaxis.get_ticklabels():
    ticklabel.set_fontsize(24)


# 保存图片
fig.savefig(path + 'draw_251226/pic_251226/72-1_rawfcst_postfcst_nrmse_skill_heatmap_(easy_version).jpg', dpi=600)




# %%
