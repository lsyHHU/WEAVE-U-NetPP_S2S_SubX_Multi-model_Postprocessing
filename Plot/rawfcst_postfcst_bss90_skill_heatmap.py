#%%
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


tscale = [7, 14, 21, 28]
region_name = ['Mainland China', 'CB', 'HaiRB', 'HuaiRB', 'PRB', 'SEB', 'SLRB', 'SWB', 'YlRB', 'YtzRB']
tscale_title = ['Timescale: 7 days', 'Timescale: 14 days',
                'Timescale: 21 days', 'Timescale: 28 days']
model_name = ['46LCESM1', 'CCSM4', 'CFSv2', 'FIMr1p1', 'GEFS', 'NESM', 'GEOS_V2p1',
                'BoM', 'CMA', 'CNRM', 'CPTEC', 'ECCC', 'ECMWF', 'HMCR', 'IAPCAS',
                'ISACCNR', 'KMA', 'UKMO', 
                'MME', 'EMOS', 'WEAVE-U-Net++']


path = 'D:/00_Code_lsy/MachineLearning_CSGD/S2S_SubX_Multimodel_251209/'


bss90_arr = np.zeros((4, 21, 10))   # 4 tscales, 21 models, 10 regions

# 读取数据
# 时间尺度循环
for t in range(4):
    _t = tscale[t]

    # 模型循环
    for m in range(21):
        if m < 18:  # rawfcst
            filedir = 'D:/00_Code_lsy/MachineLearning_CSGD/S2S_SubX_Multimodel_250525/pre_verif_result_rawfcst/'
            bss90_nc = xr.open_dataarray(filedir + f'{_t}days/raw_bss_{_t}daysTP_Year20012013.nc')
        elif m == 18:  # MME
            filedir = 'D:/00_Code_lsy/MachineLearning_CSGD/S2S_SubX_Multimodel_251209/pre_verif_result_MMEfcst/'
            bss90_nc = xr.open_dataarray(filedir + f'{_t}days/mme_bss_{_t}daysTP_Year20012013.nc')
        elif m == 19:  # EMOS
            filedir = 'D:/00_Code_lsy/MachineLearning_CSGD/S2S_SubX_Multimodel_251209/pre_verif_result_EMOSfcst/'
            bss90_nc = xr.open_dataarray(filedir + f'{_t}days/emos_bss_{_t}daysTP_Year20012013.nc')
        elif m == 20:  # WEAVE-U-Net++
            filedir = 'D:/00_Code_lsy/MachineLearning_CSGD/S2S_SubX_Multimodel_251209/pre_verif_result_UNetPP-A-WeightPredictor-T-KLloss-ContinueTraining_3/Output_0/'
            bss90_nc = xr.open_dataarray(filedir + f'{_t}days/post_bss_{_t}daysTP_Year20012013.nc')
        bss90_nc = bss90_nc.sel(year='AllYears', threshold=90).squeeze()
        bss90_nc.close()

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
            # 读取区域bss90数据
            bss90_region = bss90_nc.sel(lat=region_coords_lat, lon=region_coords_lon)
            # 计算区域bss90均值
            bss90_regional_mean = bss90_region.mean(dim=['new'], skipna=True)

            if m < 18:
                bss90_arr[t, :18, r] = bss90_regional_mean.values
            else:
                bss90_arr[t, m, r] = bss90_regional_mean.values

# 找到最优值索引
bss90_min_idx = np.argmax(bss90_arr, axis=1)  # 4 tscales, 10 regions




# 绘图
fig, ax = plt.subplots(4, 1, figsize=(25, 32))

norm = mpl.colors.Normalize(vmin=-50, vmax=50)
cmap = mpl.cm.get_cmap('RdBu_r')

yticklabel_region_name = ['Mainland\nChina', 'CB', 'HaiRB', 'HuaiRB', 'PRB', 'SEB', 'SLRB', 'SWB', 'YlRB', 'YtzRB']

for t in range(4):
    p = ax[t].pcolormesh(bss90_arr[t, :, :].T, cmap=cmap, norm=norm, alpha=0.75, edgecolor='k')
    ax[t].set_xticks(np.arange(21)+0.5)
    ax[t].set_xticklabels(model_name, rotation=30, fontsize=18, ha='right', rotation_mode='anchor')
    ax[t].set_yticks(np.arange(10)+0.5)
    ax[t].set_yticklabels(yticklabel_region_name, fontsize=18)
    ax[t].set_title(tscale_title[t], fontsize=26, fontweight='bold', pad=20)
    ax[t].set_ylabel('Hydroclimatic region', fontsize=24, labelpad=20)
    ax[t].set_xlabel('Model', fontsize=24) if t == 3 else None
    # 单元格中添加数据
    for i in range(21):
        for j in range(10):
            val = bss90_arr[t, i, j] 
            rgba = cmap(norm(val))    # 映射到颜色（R,G,B,A）
            
            if bss90_min_idx[t, j] == i:
                ax[t].text(
                    i + 0.5, j + 0.5, f'{val:.2f}',
                    ha='center', va='center',
                    fontsize=14, color='blue', fontweight='bold'
                )
            else:
                # 计算感知亮度（0=黑, 1=白），阈值可调
                r, g, b, _ = rgba
                luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                txt_color = 'black' if luminance > 0.5 else 'white'
                ax[t].text(
                    i + 0.5, j + 0.5, f'{val:.2f}',
                    ha='center', va='center',
                    fontsize=14, color=txt_color
                )


# 添加colorbar
p_ax_cb = fig.add_axes([0.9, 0.2, 0.015, 0.6])
p_cb = fig.colorbar(p, 
                    cax=p_ax_cb, 
                    extend='both', 
                    drawedges=False, 
                    orientation='vertical')
p_cb.set_label(label=r'$\mathrm{BSS\ _{\geq90th}}$' + ' (%)', fontsize=26)
p_ax_cb.tick_params(which='major', direction='in', length=7)
for ticklabel in p_ax_cb.yaxis.get_ticklabels():
    ticklabel.set_fontsize(24)



fig.subplots_adjust(left=0.15, right=0.88, bottom=0.07, top=0.95, hspace=0.4)
fig.savefig(path + 'draw_251226/supple_pic_260521/s02_rawfcst_postfcst_bss90_skill_heatmap.jpg', dpi=600)











# %%
