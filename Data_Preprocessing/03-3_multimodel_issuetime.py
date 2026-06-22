#%%
"""

"""

import xarray as xr
import numpy as np
import pandas as pd

t = 7

# 读取所有模型的issuetime list
# subx_dir = '/data/selfdata/datalsy/SubX_Multimodel_250525/SubX/'
# lcesm1_issue = np.load(subx_dir + f'46LCESM1/46LCESM1_issuelist_{t}days.npy')
# ccsm4_issue = np.load(subx_dir + f'CCSM4/CCSM4_issuelist_{t}days.npy')
# cfsv2_issue = np.load(subx_dir + f'CFSv2/CFSv2_issuelist_{t}days.npy')
# fimr1p1_issue = np.load(subx_dir + f'FIMr1p1/FIMr1p1_issuelist_{t}days.npy')
# gefs_issue = np.load(subx_dir + f'GEFS/GEFS_issuelist_{t}days.npy')
# gem_issue = np.load(subx_dir + f'GEM/GEM_issuelist_{t}days.npy')
# geosv2p1_issue = np.load(subx_dir + f'GEOS_V2p1/GEOS_V2p1_issuelist_{t}days.npy')

s2s_dir = '/data/selfdata/datalsy/SubX_Multimodel_250525/S2S/'
bom_issue = pd.to_datetime(np.load(s2s_dir + 'BoM_issuedate_list.npy', allow_pickle=True))
cma_issue = pd.to_datetime(np.load(s2s_dir + 'CMA_issuedate_list.npy', allow_pickle=True))
cnrm_issue = pd.to_datetime(np.load(s2s_dir + 'CNRM_issuedate_list.npy', allow_pickle=True))
cptec_issue = pd.to_datetime(np.load(s2s_dir + 'CPTEC_issuedate_list.npy', allow_pickle=True))
eccc_issue = pd.to_datetime(np.load(s2s_dir + 'ECCC_issuedate_list.npy', allow_pickle=True))
ecmwf_issue = pd.to_datetime(np.load(s2s_dir + 'ECMWF_issuedate_list.npy', allow_pickle=True))
hmcr_issue = pd.to_datetime(np.load(s2s_dir + 'HMCR_issuedate_list.npy', allow_pickle=True))
iapcas_issue = pd.to_datetime(np.load(s2s_dir + 'IAPCAS_issuedate_list.npy', allow_pickle=True))
isaccnr_issue = pd.to_datetime(np.load(s2s_dir + 'ISACCNR_issuedate_list.npy', allow_pickle=True))
kma_issue = pd.to_datetime(np.load(s2s_dir + 'KMA_issuedate_list.npy', allow_pickle=True))
ukmo_issue = pd.to_datetime(np.load(s2s_dir + 'UKMO_issuedate_list.npy', allow_pickle=True))

issue_dict = {
    'BoM': bom_issue,
    'CMA': cma_issue,
    'CNRM': cnrm_issue,
    'CPTEC': cptec_issue,
    'ECCC': eccc_issue,
    'ECMWF': ecmwf_issue,
    'HMCR': hmcr_issue,
    'IAPCAS': iapcas_issue,
    'ISACCNR': isaccnr_issue,
    'KMA': kma_issue,
    'UKMO': ukmo_issue
}
lead_dict = {
    'BoM': 40,
    'CMA': 35,
    'CNRM': 40,
    'CPTEC': 35,
    'ECCC': 32,
    'ECMWF': 40,
    'HMCR': 40,
    'IAPCAS': 35,
    'ISACCNR': 35,
    'KMA': 35,
    'UKMO': 35
}


alldate = pd.date_range('1999-1-1', '2014-12-31', freq='D')


def get_common_issue(issue_dict, lead_dict, alldate, t=7, max_start_lead=5):
    """
    issue_dict: dict，例如 {'A': issue_A, ..., 'E': issue_E}
                每个值是 pd.DatetimeIndex，已排序
    lead_dict: dict，例如 {'A': 30, 'B': 32, ..., 'E': 35}
               表示每个模型的最大预见期（forecast length）
    alldate: pd.DatetimeIndex，表示要循环的全部日期
    t: 时间尺度（例如 7 天）
    max_start_lead: 限制每个模型在 d 日期上的预见期不得超过的最大值（如 5 天）
    
    返回：
    - common_issue: pd.DatetimeIndex，所有符合条件的日期
    - lead_record: dict，键是 common_issue 中的每一天，值是 {model: start_lead}
    """
    common_issue = []
    lead_record = {}

    for d in alldate:
        valid = True
        start_leads = {}

        for model in issue_dict:
            issue_dates = issue_dict[model]
            max_lead = lead_dict[model]

            # 找出发布日 ≤ d 的所有日期
            valid_issues = issue_dates[issue_dates <= d]
            if len(valid_issues) == 0:
                valid = False
                break

            # 最近的发布日
            nearest_issue = valid_issues[-1]
            start_lead = (d - nearest_issue).days

            # 条件1：是否超过模型本身的预见期
            if start_lead + t > max_lead:
                valid = False
                break

            # 条件2：是否超过最大允许的start_lead
            if start_lead > max_start_lead:
                valid = False
                break

            start_leads[model] = start_lead

        if valid:
            common_issue.append(d)
            lead_record[d] = start_leads

    return pd.DatetimeIndex(common_issue), lead_record


common_issue, lead_record = get_common_issue(issue_dict, lead_dict, alldate, t=7, max_start_lead=5)

#%%
nday = len(common_issue)

lat = np.linspace(10, 57, 48)
nlat = len(lat)
lon = np.linspace(65, 144, 80)
nlon = len(lon)


model_name = ['BoM', 'CMA', 'CNRM', 'CPTEC', 'ECCC', 'ECMWF',
             'HMCR', 'IAPCAS', 'ISACCNR', 'KMA', 'UKMO']
nmodel = len(model_name)

# 日期循环
for d in range(nday):
    _d = common_issue[d]
    print('common issue:', _d)

    year = str(_d.year)
    month = str(_d.month).zfill(2)
    day = str(_d.day).zfill(2)

    model_lead = lead_record[_d]  # 找到对应当前日期的，各个模型对应的预见期
    # 模型循环
    for m in range(nmodel):
        _m = model_name[m]
        lead = model_lead[_m]  # 当前模型的对应预见期

        # 读取数据
        if _m in ['BoM', 'CMA', 'CPTEC', 'ECCC', 'ECMWF', 'HMCR', 'ISACCNR']:
            datadir = f'/home/data/fcst/S2S/{_m}/Control/2mt/'
        elif _m == 'CNRM':
            datadir = f'/home/data/fcst/S2S/{_m}/Control/2mt_tcc_tcw/'
        elif _m in ['IAPCAS', 'KMA', 'UKMO']:
            datadir = f'/home/data/fcst/S2S/{_m}/Control/2mt_tcc/'
        
        if _m in ['BoM', 'CMA', 'CPTEC', 'ECCC', 'ECMWF', 'HMCR', 'ISACCNR']:
            data = xr.open_dataarray(datadir + f'{_m}_{year}-{month}-{day}.nc')
        else:
            data = xr.open_dataset(datadir + f'{_m}_{year}-{month}-{day}.nc')['t2m']
        data.close()
        data = data.interp(latitude=lat, longitude=lon)
        data = data.expand_dims(issuetime=[_d])
        data = data.rename({'time': 'leadtime'})
        data = data.assign_coords(leadtime=np.arange(len(data.leadtime)))


        break
    break
    











# %%
