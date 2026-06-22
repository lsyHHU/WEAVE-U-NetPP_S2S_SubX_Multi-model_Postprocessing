# %%
"""
只计算陆地网格的可靠性图
"""

import sys, os
import numpy as np
import pandas as pd
import xarray as xr
import multiprocessing as mp
sys.path.append('/home/liusy')
from pyNMME import module_verification



# -----------------------------
# multiprocessing worker globals
# -----------------------------
_FCST = None   # shape: (ntime, nens, ngrid)
_OBS  = None   # shape: (ntime, ngrid)
_EVENT = None
_THRESHOLD = None

def init_worker_gridwise(fcst_np, obs_np, event, ths):
    """Runs once per process."""
    global _FCST, _OBS, _EVENT, _THS
    _FCST = fcst_np
    _OBS = obs_np
    _EVENT = event
    _THS = ths

def calc_one_grid_gridwise(igrid: int):
    """Compute ens_p and obs_p for one grid. Threshold is grid-wise percentile."""
    fcst_grid = _FCST[:, :, igrid]  # (ntime, nens)
    obs_grid  = _OBS[:, igrid]      # (ntime,)

    threshold = np.percentile(obs_grid, _THS)
    ens_p_1grid = module_verification.ens2p(fcst_grid, threshold, event=_EVENT)
    obs_p_1grid = module_verification.value2bool(obs_grid, threshold, event=_EVENT)
    return ens_p_1grid, obs_p_1grid



def main():
    # 读取mask
    mask = pd.read_csv('/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/CN_land_coords/land_coords.csv')
    mask_lat = xr.DataArray(mask['lat'].values, dims='new')
    mask_lon = xr.DataArray(mask['lon'].values, dims='new')
    ngrid = len(mask)

    tscale = [7, 14, 21, 28]
    # tscale = [14]
    model_name = ['UNetPP-SkillBasedWeight-NoWeightLearning']
    ths_list = [10, 33.3, 66.7, 90]
    event_list = ['left', 'left', 'right', 'right']

    ib = 0

    # 建议在 HPC 上用 spawn 更稳（尤其你依赖的库比较多时）
    # mp.set_start_method("spawn", force=True)

    nproc = 10

    for t in tscale:
        for _m in model_name:
            # 打开预报数据
            datadir = f'/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/00_Results_newTest_251209/{_m}/Output_{ib}/{t}days/Unetfcst_ENS/'
            fcst_da = xr.open_dataarray(datadir + f'ensfcst_pr_{t}days_Year20012013.nc') \
                        .sel(lat=mask_lat, lon=mask_lon).squeeze()

            # 打开观测数据
            obs_da = xr.open_dataarray(
                f'/data/selfdata/datalsy/S2S_SubX_Multimodel_250525/obs/CN05.1_totalPre_1995_2015_{t}days_1x1.nc'
            ).sel(lat=mask_lat, lon=mask_lon)

            # 对齐时间
            common_time = np.intersect1d(fcst_da.issuetime.values, obs_da.time.values)
            fcst_da = fcst_da.sel(issuetime=common_time)
            obs_da  = obs_da.sel(time=common_time)

            fcst_np = np.asarray(fcst_da.values)  # (ntime, nens, ngrid)
            obs_np  = np.asarray(obs_da.values)   # (ntime, ngrid)

            try: fcst_da.close()
            except Exception: pass
            try: obs_da.close()
            except Exception: pass

            for iths, (ths, event) in enumerate(zip(ths_list, event_list)):
                ens_p_parts = []
                obs_p_parts = []

                with mp.Pool(
                    processes=nproc,
                    initializer=init_worker_gridwise,
                    initargs=(fcst_np, obs_np, event, ths),
                ) as pool:
                    chunksize = max(1, ngrid // (4 * nproc))

                    for i, (ens_1, obs_1) in enumerate(
                        pool.imap_unordered(calc_one_grid_gridwise, range(ngrid), chunksize=chunksize),
                        start=1
                    ):
                        ens_p_parts.append(np.asarray(ens_1, dtype=np.float64))
                        obs_p_parts.append(np.asarray(obs_1, dtype=np.float64))

                        if i % 200 == 0 or i == ngrid:
                            print(f"{t}days {_m} ths={ths}: {i}/{ngrid} grids done")

                # 一次性拼接（比 extend 更快）
                ens_p = np.concatenate(ens_p_parts, axis=0)
                obs_p = np.concatenate(obs_p_parts, axis=0)

                obs_clim = float(obs_p.mean())
                unc = obs_clim * (1 - obs_clim)
                n = ens_p.size

                bins = (0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.)
                inds = np.digitize(ens_p, bins=bins, right=False)
                inds[inds == len(bins)] = len(bins) - 1

                ens_prob, obs_freq, prop_list = [], [], []
                rel_list, res_list = [], []

                for ind in np.unique(inds):
                    temp_mask = inds == ind
                    ep = ens_p[temp_mask].mean()
                    of = obs_p[temp_mask].mean()
                    prop = temp_mask.mean()

                    ens_prob.append(ep)
                    obs_freq.append(of)
                    prop_list.append(prop)

                    rel_list.append((ep - of) ** 2)
                    res_list.append((of - obs_clim) ** 2)

                rel = float(np.sum(np.array(rel_list) * np.array(prop_list)))
                res = float(np.sum(np.array(res_list) * np.array(prop_list)))

                bins_prop = np.zeros(len(bins) - 1)
                bins_prop[np.unique(inds) - 1] = np.array(prop_list)
                bins_sample = (bins_prop * n).tolist()

                savedir = f'/home/liusy/MachineLearning_CSGD/S2S_SubX_Multimodel_250525/03_Unet++_newTest_251209/03_1_reliability_diagram/{_m}/{t}days/'
                os.makedirs(savedir, exist_ok=True)
                np.save(savedir + f'ens_prob_{ths}.npy', ens_prob)
                np.save(savedir + f'obs_freq_{ths}.npy', obs_freq)
                np.save(savedir + f'bins_prop_{ths}.npy', bins_prop)
                np.save(savedir + f'bins_sample_{ths}.npy', bins_sample)
                np.save(savedir + f'rel_{ths}.npy', rel)
                np.save(savedir + f'res_{ths}.npy', res)
                np.save(savedir + f'unc_{ths}.npy', unc)

                print(t, _m, iths, 'done')


if __name__ == "__main__":
    main()







        


# %%
