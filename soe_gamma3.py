__author__ = 'demos'

from scipy.optimize import minimize
import statistics as st
import itertools as itertools

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import pylppl as lp
import datetime
import math
from numpy import *
from numba import autojit

import sys
sys.path.append('/Users/demos/Documents/Python/ipy (work)/LPPLS - Sloppy/')

import sys
sys.path.append('/Users/demos/Documents/Python/ipy (work)/BACKTESTING - bubble indicator and trading strategies 101/clean_codes/')
sys.path.append('/Users/demos/Documents/Python/ipy (work)/BACKTESTING - bubble indicator and trading strategies 101/clean_strategy/')
sys.path.append('/Users/demos/Documents/Python/ipy (work)/HIERARCHY CALIBRATION - extension analysis to different models')
sys.path.append('/Users/demos/Documents/Python/ipy (work)/BACKTESTING - bubble indicator and trading strategies 101/LPPLS - using the modified profile likelihood method for estimating the bubble status indicator/')

from numpy import log, pi
import sum_of_exponentials as soe
import soe_clean as soec
import numdifftools as ndt

# For parallel loops
from multiprocessing import cpu_count
try:
    CPU_COUNT = cpu_count()
except NotImplementedError:
    CPU_COUNT = 1

try:
    from joblib import Parallel, delayed
    RUN_PARALLEL = CPU_COUNT > 1
except ImportError:
    Parallel = None
    delayed = None
    RUN_PARALLEL = False

import matplotlib as mp
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : 18}

label_size = 16
mp.rcParams['ytick.labelsize'] = label_size
mp.rcParams['xtick.labelsize'] = label_size


########################################################################################################################

# Functions part I #

########################################################################################################################


# ---------------------------------------------------------------
alg = 'Nelder-Mead'
noiseLevel = 0.01
numpts0 = 3 # Number of initial values

# ---------------------------------------------------------------
def fitBest(data, gammaFix=None, llkOnly=False):

    ggg, _ = fitNlinears(data, gammaFix=gammaFix)
    ABC, fit, obj = fitLinears(data, ggg)

    if llkOnly == False:
        return list(ABC) + list(ggg), fit, obj
    else:
        return obj

# ---------------------------------------------------------------
def fitNlinears(data, gammaFix=None):

    # Create initial grid
    pts0, cost = createInitialValues(data, gammaFix=gammaFix)

    res = [minimize(cost, x0,
                    method=alg,
                    options={'disp': False})
           for x0 in pts0]
    objs = [r.fun for r in res]
    objs = np.array(objs)
    objs = objs[~np.isnan(objs)]
    objs = list(objs)
    mfun = min(objs)
    pars = res[objs.index(mfun)].x

    if gammaFix is None:
        return pars, mfun
    else:
        return [pars[0], pars[1], gammaFix], mfun

# ---------------------------------------------------------------
def createInitialValues(data, numpts0=numpts0, gammaFix=None):

    GRID_G1 = np.linspace(0.01, 5., numpts0+2)
    GRID_G2 = np.linspace(0.01, 5., numpts0+2)
    GRID_G3 = np.linspace(0.01, 5., numpts0+2)

    if gammaFix is None:
        def cost(x):
            return fitLinears(data, x)[-1]
        PTS = [GRID_G1, GRID_G2, GRID_G3]
    else:
        def cost(x):
            return fitLinears(data, [x[0], x[1], gammaFix])[-1]
        PTS = [GRID_G1, GRID_G3]

    pars = [list(x) for x in itertools.product(*PTS)]

    return pars, cost

# ---------------------------------------------------------------
def fitLinears(data, gammas):

    """gammas = [gamma1, gamma2, gamma3] """

    T = np.size(data,0)
    t = np.arange(0, T, 1)

    # we convert parameters to float here
    gamma1, gamma2, gamma3 = (np.array(gammas) + 0.0)

    g  = np.exp(-gamma1*t)
    f1 = np.exp(-gamma2*t)
    f2 = np.exp(-gamma3*t)

    inds = ~np.isfinite(g)
    g[inds]  = 0.
    f1[inds] = 0.
    f2[inds] = 0.

    MAT = np.array([[np.sum(g*g), np.sum(g*f1),  np.sum(g*f2)],
                   [0,            np.sum(f1*f1), np.sum(f1*f2)],
                   [0,                        0, np.sum(f2*f2)]]) # IS THIS CORRECT ?

    MAT[1,0] = MAT[0,1]
    MAT[2,0] = MAT[0,2]
    MAT[2,1] = MAT[1,2]

    Y = np.array([np.sum(g * data),
                  np.sum(f1 * data),
                  np.sum(f2 * data)]) # IS THIS CORRECT ?

    try:
        ABC = np.linalg.solve(MAT, Y)
    except np.linalg.LinAlgError:
        return [np.NaN, np.NaN, np.NaN], np.NaN, np.Inf

    fit = ABC[0]*g + ABC[1]*f1 + ABC[2]*f2

    sse = (data - fit)**2.

    return ABC, fit, np.sum(sse)

########################################################################################################################

# Profiling #

########################################################################################################################

# ---------------------------------------------------------------
def profilingGammaThree(data):

    # construct
    gammaRange = np.linspace(.001, 5., 40)
    r = [fitBest(data, gammaFix=i) for i in gammaRange]

    # Get results
    R = [x[2] for x in r]
    resDf = pd.DataFrame(R, index=gammaRange)
    bestGamma = resDf[resDf == resDf.min()].dropna().index[0]

    # Get pars
    hatPars, _, _ = fitBest(data, gammaFix=bestGamma)

    # Log Profile
    log_Lp = -len(data)/2. * np.log(resDf)
    s_tc = resDf / float(len(data))

    return log_Lp, hatPars, s_tc


########################################################################################################################

# Plotting #

########################################################################################################################


def firstDiagnosisOnNoiseLevel(sampleSize, pars):

    # Simulate
    sData = soec.simulate2(pars, sampleSize, noise=True)

    # Estimate
    ParsQml, fit, _ = fitBest(sData)

    # Monte-Carlo:
    #   - Several (10) data simulations
    #   - Several (10) fits
    SD = []
    FIT = []
    estimatedParsQmlMc = []
    for i in range(10):
        sData = soec.simulate2(pars, sampleSize, noise=True)
        # Estimate
        ParsQml, fit, _ = fitBest(sData)
        SD.append(sData)
        FIT.append(fit)
        estimatedParsQmlMc.append(ParsQml)

    # DataFrame with estimated parameters throught the monteCarlo Function
    McParsQml = pd.DataFrame(estimatedParsQmlMc)
    McParsQml.columns = ['a','b','c','g1','g2','g3']

    # Make a dataframe
    SData = pd.DataFrame(SD)
    FITT = pd.DataFrame(FIT)

    # Plot
    f,ax = plt.subplots(1,1,figsize=(6,3))
    plt.plot(FITT.T, color='r', linestyle='-', linewidth=1)
    locs, labels = plt.xticks()
    x = np.arange(0, len(FITT.T), 1)
    plt.boxplot(SData.values, positions=x, notch=True)
    plt.xlabel(r'$N$', fontsize=20)
    plt.ylabel(r'$Y$', fontsize=20)
    plt.grid()
    plt.tight_layout()

    # Profile
    lpDf, lpHat, s_tc = profilingGammaTwo(sData)

    f,ax = plt.subplots(1,1,figsize=(6,3))
    (lpDf).plot(ax=ax, linewidth=3, marker='.', markersize=12)
    plt.ylabel(r'$[L_p(\gamma_1|\hat{\eta})]$', fontsize=20)
    plt.xlabel(r'$\gamma_1$', fontsize=20)
    plt.axvline(lpDf[lpDf==lpDf.max()].dropna().index[0], color='r', linewidth=2)
    plt.axvline(lpHat[3], color='r', linestyle='--', linewidth=5)
    plt.grid()
    plt.tight_layout()

    sDataQml  = soec.simulate2(ParsQml, sampleSize, noise=False)
    sDataTrue = soec.simulate2(pars, sampleSize, noise=False)
    sDataLp   = soec.simulate2(lpHat, sampleSize, noise=False)

    f,ax = plt.subplots(1,1,figsize=(6,3))
    ax.plot(sData, marker='o', linestyle='', color='r', markersize=12)
    ax.plot(sDataQml, linestyle='-',linewidth=4, color='k')
    ax.plot(sDataTrue,linestyle='-',linewidth=4, color='b')
    ax.plot(sDataLp,linestyle='-',linewidth=4, color='c')
    plt.ylabel(r'$Y$', fontsize=20)
    plt.xlabel(r'$N$', fontsize=20)
    plt.legend(['Data','Qml','True','Lp'])
    plt.tight_layout()

    print('RMSE Qml*100 = %.5f'%np.sqrt(np.sum((sDataQml-sDataTrue)**2.)*100))
    print('RMSE Lp*100 = %.5f'%np.sqrt(np.sum((sDataLp-sDataTrue)**2.)*100))

    return ParsQml, lpHat, McParsQml


########################################################################################################################

# Modified Profile Likelihood #

########################################################################################################################


def hessianForGivenGammaThree(pars, data, gammaFix):

    # Getting the Hessian: |profile_gammaOne(pars, data, gamma_one|
    f = lambda x: profile_gammaThree(x, data, gammaFix)

    # Get Hessian
    Hfun = ndt.Hessian(f)
    H = Hfun(np.array([pars[0], pars[1], pars[2], pars[3], pars[4]]))

    # Get scores
    scores = calc_scores_soe(pars, data, gammaFix)

    return H, scores


# ---------------------------------------------------------------
def profile_gammaThree(pars, data, gamma_fix, simulate=False):

    """ enter a fix value for gamma_one """

    A,B,C = pars[0], pars[1], pars[2]
    gamma_one,gamma_two,gamma_three = pars[3], pars[4], gamma_fix

    T = len(data)   # Data size
    Y = np.zeros(T) # initial values

    for t in range(T): # Start looping
        Y[t] = A*np.exp(-gamma_one*t)  # the model
        + B*np.exp(-gamma_two*t)
        + C*np.exp(-gamma_three*t)

    sse = (Y - data)**2.
    obj = np.sum(sse)

    if simulate == False:
        return obj
    else:
        return Y, obj


# ---------------------------------------------------------------
def calc_scores_soe(pars, data, fixedGamma):

    small_pars_set = np.array([pars[0], pars[1], pars[2], pars[3], pars[4]])

    ## Sensitivity analysis
    step = 1e-5 * small_pars_set
    T = np.size(data, 0)
    scores = np.zeros((T,5))

    for i in xrange(len(small_pars_set)):
        h = step[i]
        delta = np.zeros(5)
        delta[i] = h

        _, logliksplus = profile_gammaThree(small_pars_set + delta,
                                                   data, fixedGamma, simulate=True)
        _, loglikminus = profile_gammaThree(small_pars_set - delta,
                                                    data,fixedGamma, simulate=True)
        scores[:,i] = (logliksplus - loglikminus)/(2*h)

    return scores


# ---------------------------------------------------------------
def modLikStep(ind, pars, data, X_hat, s_tc):

    p = 5

    H, X = hessianForGivenGammaThree(pars, data, pars[5])

    detI = np.abs(np.linalg.det(H))  # To avoid duplicated calculations: I_psi
    detS = np.abs(np.linalg.det(np.dot(X_hat.T, X)))

    log_Lm = -(len(data)-p-2.)/2. * s_tc.values[ind][0] + detI/2. - detS

    return log_Lm


# ---------------------------------------------------------------
def calcModLogLik(pars, data, unbiased_s=True, job_pool=None):

     # Profile likelihood and hat{s}_tc
    logLp, lpHat, s_tc = profilingGammaThree(data)
    if unbiased_s:
        p = 5.
        s_tc = s_tc * float(len(data)) / float(len(data) - p)

    # X for MLE
    H, X_hat = hessianForGivenGammaThree(pars, data, lpHat[3])

    # Args
    args = {'pars': lpHat, 'data': data, 'X_hat': X_hat, 's_tc': s_tc}

    # Modified profile likelihood
    if RUN_PARALLEL:
        if job_pool is None:
            job_pool = Parallel(n_jobs=CPU_COUNT)
        log_Lm = job_pool(delayed(modLikStep)(ind, **args)
                          for ind in range(len(logLp)))
    else:
        log_Lm = [modLikStep(ind, **args) for ind in range(len(logLp))]

    # Make a dataframe
    log_Lm = pd.DataFrame(log_Lm, index=logLp.index)

    # Get par
    bestGamma = log_Lm[log_Lm==log_Lm.max()].dropna().index[0]

    # Get pars
    lmHat, _, _ = fitBest(data, gammaFix=bestGamma)

    return log_Lm, logLp, lmHat, lpHat