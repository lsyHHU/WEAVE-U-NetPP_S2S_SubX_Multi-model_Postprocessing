#%%

# CSGD-EMOS统计后处理 辅助函数

import math
import numpy as np
from numpy import ma
from scipy.stats import gamma
from scipy.special import beta
from scipy.optimize import minimize



def crpsclimaCSGD_fit3params(par, obs):
    """
    最小化CRPS，拟合climatology CSGD分布的三参数(shape / scale / shift)
    使用前提:观测降水的降水发生概率pop<1
    par[0]: mu
    par[1]: sigma
    par[2]: shift

    输入参数：
    obs: 观测降水序列
    """

    obs = np.sort(obs[~np.isnan(obs)])
    n = len(obs)
    k0 = np.sum(obs == 0)

    shape = (par[0] / par[1]) ** 2
    scale = par[0] / shape
    shift = par[2]

    crps = np.zeros(n)

    c_std = -shift / scale
    y_std = (obs[k0:] - shift) / scale

    F_k_c = gamma.cdf(c_std, a=shape)
    F_kp1_c = gamma.cdf(c_std, a=shape+1.)
    F_2k_2c = gamma.cdf(2.*c_std, a=2.*shape)
    B_05_kp05 = beta(0.5, shape+0.5)

    F_k_y = gamma.cdf(y_std, a=shape)
    F_kp1_y = gamma.cdf(y_std, a=shape+1.)

    crps[:k0] = (
        c_std * (2. * F_k_c - 1.) - shape * (2. * F_kp1_c - 1. + F_k_c ** 2. - 2. *F_kp1_c * F_k_c)
        - c_std * F_k_c ** 2. - (shape / np.pi) * B_05_kp05 * (1. - F_2k_2c)
    )
    
    return scale * np.mean(crps)


def crpsclimaCSGD_fit2params(par, obs):
    """
    最小化CRPS，拟合climatology CSGD分布的三参数(shape / scale)
    使用前提:观测降水的降水发生概率pop==1, shift参数不参与优化, 默认为0
    par[0]: mu
    par[1]: sigma

    输入参数：
    obs: 观测降水序列
    """

    obs = np.sort(obs[~np.isnan(obs)])
    n = len(obs)
    k0 = np.sum(obs == 0)

    shape = (par[0] / par[1]) ** 2
    scale = par[0] / shape
    shift = 0.0

    crps = np.zeros(n)

    c_std = -shift / scale
    y_std = (obs[k0:] - shift) / scale

    F_k_c = gamma.cdf(c_std, a=shape)
    F_kp1_c = gamma.cdf(c_std, a=shape+1.)
    F_2k_2c = gamma.cdf(2.*c_std, a=2.*shape)
    B_05_kp05 = beta(0.5, shape+0.5)

    F_k_y = gamma.cdf(y_std, a=shape)
    F_kp1_y = gamma.cdf(y_std, a=shape+1.)

    crps[:k0] = (
        c_std * (2. * F_k_c - 1.) - shape * (2. * F_kp1_c - 1. + F_k_c ** 2. - 2. *F_kp1_c * F_k_c)
        - c_std * F_k_c ** 2. - (shape / np.pi) * B_05_kp05 * (1. - F_2k_2c)
    )

    crps[k0:] = (
        y_std * (2. * F_k_y - 1.) - shape * (2. * F_kp1_y - 1. + F_k_c ** 2. - 2. *F_kp1_c * F_k_c)
        - c_std * F_k_c ** 2. - (shape / np.pi) * B_05_kp05 * (1. - F_2k_2c)
    )

    return scale * np.mean(crps)


def calculate_climaCSGD_par(obs):
    """
    调用crpsClimaCSGD函数，拟合其参数
    """

    obs_mean = np.mean(obs)
    obs_pop = np.mean(obs > 0.1)
    sigma = obs_mean   # 初始值, 假设 sigma = mu

    # initial value of mu, sigma, shift
    for mu in (np.arange(40, 0, -1) * (sigma / 40)):
        shape = (mu / sigma) ** 2
        scale = mu / shape
        shift = -gamma.ppf(1.-obs_pop, a=shape, scale=scale, loc=0)
        # print(shift)
        if shift > -mu / 2.:
            break

    # 初始值
    if obs_pop == 1.:
        par0 = np.array([mu, sigma])
    else:
        par0 = np.array([mu, sigma, shift])

    # extremely dry
    # 使用ad hoc values 作为气候态分布的参数
    if obs_pop < 0.005:
        par_clima = np.array([0.0005, 0.0182, -0.00049])
        return par_clima
    # dry
    # 使用初始值作为气候态分布的参数
    if obs_pop < 0.02:
        par_clima = par0
        return par_clima

    if obs_pop == 1.:
        par_clima = minimize(crpsclimaCSGD_fit2params, par0, 
                            args=(obs,),
                            method='L-BFGS-B',
                            bounds=(
                                (0.1*par0[0], 5*par0[0]), 
                                (0.1*par0[1], 5*par0[1])
                            ),
                            tol=1e-6).x
        par_clima = np.append(par_clima, 0.0)
    else:
        par_clima = minimize(crpsclimaCSGD_fit3params, par0, 
                            args=(obs,),
                            method='L-BFGS-B',
                            bounds=(
                                (0.1*par0[0], 5*par0[0]), 
                                (0.1*par0[1], 5*par0[1]),
                                (5*par0[2], 0.1*par0[2])
                            ),
                            tol=1e-6).x
    return par_clima


#################################################################################################


def crpsCondCSGD(par, obs, ensmeanano_arr, MD, muc, sigmac, shiftc):
    """
    最小化crps, 拟合线性回归的参数
    ensmeanano_arr: M个模型的集合平均值组成的1维arr

    μ = μc * log1p[ expm1(a1) * (a2 + m1*ensmeanano_arr[0] + m2*ensmeanano_arr[1] + ... + mM*ensmeanano_arr[M-1]) ] / a1
    σ = a3 * σc * (μ / μc)**a4 + a5*MD
    δ = δc
    """

    M = ensmeanano_arr.shape[1]  # 模型数 = 18
    a1 = par[0]
    a2 = par[1]
    a3 = par[2]
    a4 = par[3]
    a5 = par[4]
    m = par[5:5+M]
    eps = 1e-6

    m = np.array(m).reshape(1, M)
    mu = muc * np.log1p(np.expm1(a1) * (a2 + np.sum(m * ensmeanano_arr, axis=1))) / (a1+eps)
    sigma = a3 * sigmac * (mu / muc)**a4 + a5 * MD
    
    shape = np.square(mu / (sigma+eps))
    scale = np.square(sigma) / (mu+eps)
    shift = shiftc

    betaf = beta(0.5, shape+0.5)
    cstd = (0.1 - shift) / scale
    ystd = np.maximum(obs-shift, 0.0) / scale   # standardized observation
    Fyk = gamma.cdf(ystd, shape, scale=1.0)
    Fck = gamma.cdf(cstd, shape, scale=1.0)
    FykP1 = gamma.cdf(ystd, shape+1.0, scale=1.0)
    FckP1 = gamma.cdf(cstd, shape+1.0, scale=1.0)
    F2c2k = gamma.cdf(2.0*cstd, 2.0*shape, scale=1.0)
    crps = ystd*(2.0*Fyk-1.0) - cstd*np.square(Fck) + shape*(1.0+2.0*Fck*FckP1-np.square(Fck)-2.0*FykP1) \
           - (shape/float(math.pi))*betaf*(1.0-F2c2k)
    return ma.mean(scale*crps)


##################################################################################################################


def pctg(x, mu, sigma, shift):
    """
    x: values of precipitation
    output: CDF of CSGD
    """
    return gamma.cdf(x-shift,
                     scale=(np.square(sigma)) / (mu+0.0001),
                     a=np.square(mu/(sigma+0.0001)))        # a is shape parameter


def qctg(cdf_value, mu, sigma, shift):
    """
    cdf_value: CDF values to calculate the quantiles
    output: quantiles of CSGD
    """
    return np.maximum(0,
                      shift + gamma.ppf(cdf_value,
                      scale=(np.square(sigma)) / (mu+0.0001),
                      a=np.square(mu/(sigma+0.0001))))
    
                            






# %%

import numpy as np

m = np.arange(18)            # 长度为18的列表 / 向量
arr = np.random.rand(414, 18)

m = np.array(m).reshape(1, 18)
res = np.sum(m * arr, axis=1)




# %%
