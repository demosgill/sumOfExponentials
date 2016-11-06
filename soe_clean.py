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
sys.path.append('/Users/demos/Documents/Python/ipy (work)/BACKTESTING - bubble indicator and trading strategies 101/clean_codes/')
sys.path.append('/Users/demos/Documents/Python/ipy (work)/BACKTESTING - bubble indicator and trading strategies 101/clean_strategy/')
sys.path.append('/Users/demos/Documents/Python/ipy (work)/HIERARCHY CALIBRATION - extension analysis to different models')
sys.path.append('/Users/demos/Documents/Python/ipy (work)/BACKTESTING - bubble indicator and trading strategies 101/LPPLS - using the modified profile likelihood method for estimating the bubble status indicator/')
sys.path.append('/Users/demos/Documents/Python/ipy (work)/LPPLS - Sloppy/')
import sloppy_func as fsl

import data_functions as dfn
from numpy import log, pi
import sum_of_exponentials as soe
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

# ---------------------------------------------------------------
# Define:
# - Algo
# - Size of initial grid: numpts0 + 2
alg = 'Nelder-Mead'
noiseLevel = 0.01
numpts0 = 3
# ---------------------------------------------------------------

########################################################################################################################

# Functions part I #

########################################################################################################################

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

    if gammaFix==None:
        # bounds
        bounds = (([0.1, .5],
                   [0.1, .5],
                   [0.1, .5]))
    else:
        bounds = (([0.1, .5],
                   [0.1, .5]))


    # Adding constrains
    cons = ({'type':'ineq',
             'fun' : lambda x: np.array([x[0]])},
           {'type':'ineq',
            'fun': lambda x: np.array([x[1]])},
           {'type':'ineq',
            'fun': lambda x: np.array([x[2]])})

    res = [minimize(cost, x0,
                    method=alg,# bounds=bounds,constraints=cons,
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
        return [gammaFix, pars[0], pars[1]], mfun

# ---------------------------------------------------------------
def createInitialValues(data, numpts0=numpts0, gammaFix=None):

    GRID_G1 = np.linspace(0.01, 1., numpts0+2)
    GRID_G2 = np.linspace(0.01, 1., numpts0+2)
    GRID_G3 = np.linspace(0.01, 1., numpts0+2)

    if gammaFix is None:
        def cost(x):
            return fitLinears(data, x)[-1]
        PTS = [GRID_G1, GRID_G2, GRID_G3]
    else:
        def cost(x):
            return fitLinears(data, [gammaFix, x[0], x[1]])[-1]
        PTS = [GRID_G2, GRID_G3]

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

# Data simulation only #

########################################################################################################################

# ---------------------------------------------------------------
def simulate2(pars, sampleSize, noise=True):

    data = np.random.normal(0., noiseLevel, sampleSize)
    syn = fit_exponential(pars, data, simulate=True)
    sdata = syn + data # add noise

    if noise == True:
        return sdata
    else:
        return syn

# ---------------------------------------------------------------
def fit_exponential(pars, data, simulate=False):

    A,B,C = pars[0], pars[1], pars[2]
    gamma_one,gamma_two,gamma_three = pars[3], pars[4], pars[5]

    T = len(data)   # Data size
    Y = np.zeros(T) # initial values

    for t in range(T): # Start looping
        g = np.exp(-gamma_one*t)
        h = np.exp(-gamma_two*t)
        i = np.exp(-gamma_three*t)
        Y[t] = A*g + B*h + C*i

    sse = (Y - data)**2.

    if simulate == False:
        return np.sum(sse)
    else:
        return Y


########################################################################################################################

# Profiling #

########################################################################################################################

# ---------------------------------------------------------------
def profilingGammaOne(data):

    # construct
    gammaRange = np.linspace(.001, 1., 25)
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

# Modified Profile Likelihood # WITH OPTIMISATION !

########################################################################################################################

def LpAndMlpOptimising(data):

    # Profile likelihood of gamma_1
    lpDf, lpHat, sGammaLp = profilingGammaOne(data)

    ## Hessian and Hica
    H, J = hessianForGivenGammaOne(lpHat, data, lpHat[3])

    # Estimate ModifiedProfile (optimised)

    # Define Gamma Range
    gammaRange = np.linspace(.001, 1., 25)

    job_pool = Parallel(n_jobs=CPU_COUNT)
    res = job_pool(delayed(fitBestLm)(data, H, J,
                                      gammaFix=gammai) for gammai in gammaRange)

    # get results
    minLmp = [x[2] for x in res]

    # Make dataframe of MPL
    mplDf = pd.DataFrame(-np.array(minLmp),
                         index=gammaRange)

    # Get parameters
    bestGamma = mplDf[mplDf == mplDf.max()].dropna().index[0]

    # Get remaining parameters
    lmpHat, _, _ = fitBestLm(data, H, J, gammaFix=bestGamma)


    return lpDf, mplDf, lpHat, lmpHat


# ---------------------------------------------------------------
def hessianForGivenGammaOne(pars, data, gammaOneFix):

    # Getting the Hessian: |profile_gammaOne(pars, data, gamma_one|
    f = lambda x: profile_gammaOne(x, data, gammaOneFix)

    # Get Hessian
    Hfun = ndt.Hessian(f)
    H = Hfun(np.array([pars[0], pars[1], pars[2], pars[4], pars[5]]))

    # Get scores
    scores = calc_scores_soe(pars, data, gammaOneFix)

    return H, scores


# ---------------------------------------------------------------
def calc_scores_soe(pars, data, fixedGammaOne):

    small_pars_set = np.array([pars[0], pars[1], pars[2], pars[4], pars[5]])

    ## Sensitivity analysis
    step = 1e-5 * small_pars_set
    T = np.size(data, 0)
    scores = np.zeros((T,5))

    for i in xrange(len(small_pars_set)):
        h = step[i]
        delta = np.zeros(5)
        delta[i] = h

        _, logliksplus = profile_gammaOne(small_pars_set + delta,
                                                   data, fixedGammaOne, simulate=True)
        _, loglikminus = profile_gammaOne(small_pars_set - delta,
                                                    data,fixedGammaOne, simulate=True)
        scores[:,i] = (logliksplus - loglikminus)/(2*h)

    return scores



# ---------------------------------------------------------------
def fitBestLm(data, H_hat, J_hat, gammaFix=None, llkOnly=False):

    ggg, _ = fitNlinearsLm(data, H_hat, J_hat, gammaFix=gammaFix)
    ABC, fit, obj = fitLinearsLm(data, H_hat, J_hat, ggg)

    if llkOnly == False:
        return list(ABC) + list(ggg), fit, obj
    else:
        return obj


# ---------------------------------------------------------------
def fitNlinearsLm(data, H_hat, J_hat, gammaFix=None):

    # Create initial grid
    pts0, cost = createInitialValuesLm(data, H_hat, J_hat, gammaFix=gammaFix)

    res = [minimize(cost, x0, method=alg,
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
        return [gammaFix, pars[0], pars[1]], mfun


# ---------------------------------------------------------------
def createInitialValuesLm(data, H_hat, J_hat, numpts0=numpts0, gammaFix=None):

    GRID_G1 = np.linspace(0.01, 1., numpts0+2)
    GRID_G2 = np.linspace(0.01, 1., numpts0+2)
    GRID_G3 = np.linspace(0.01, 1., numpts0+2)

    if gammaFix is None:
        def cost(x):
            return fitLinearsLm(data, H_hat, J_hat, x)[-1]
        PTS = [GRID_G1, GRID_G2, GRID_G3]
    else:
        def cost(x):
            return fitLinearsLm(data, H_hat, J_hat, [gammaFix, x[0], x[1]])[-1]
        PTS = [GRID_G2, GRID_G3]

    pars = [list(x) for x in itertools.product(*PTS)]

    return pars, cost

# ---------------------------------------------------------------
def fitLinearsLm(data, H_hat, J_hat, gammas):

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
                  np.sum(f2 * data)])

    try:
        ABC = np.linalg.solve(MAT, Y)
    except np.linalg.LinAlgError:
        return [np.NaN, np.NaN, np.NaN], np.NaN, np.Inf

    # OBJECTIVE 1
    fit = ABC[0]*g + ABC[1]*f1 + ABC[2]*f2
    sse = (data - fit)**2.

    # Modified
    smallP = np.array([ABC[0], ABC[1], ABC[2], gammas[0],
                       gammas[1], gammas[2]])
    Fish_step, J_step = hessianForGivenGammaOne(smallP,data,gammas[0])
    detI = np.abs(np.linalg.det(np.dot(J_step.T, J_step) - Fish_step))
    detS = np.abs(np.linalg.det(np.dot(J_hat.T, J_step)))

    # Objective 2
    loglik = np.sum(sse)
    Lm = (len(data)-2-2.)/2. * np.log(loglik) - np.log(detI)/2. + np.log(detS)

    return ABC, fit, np.sum(sse)


########################################################################################################################

# Metrics #

########################################################################################################################

def calcMetricForGivenValueOfEstimatedParsLIN(ind, parsM, parsHM):
    """Return the folowing metric:
        - (phat - truepar)/truepar
        for linear parameters
    """
    r = [(parsHM['Lin'].values[ind] - parsM['LinTrue'][i])/parsM['LinTrue'][i] for i in range(len(parsM['LinTrue']))]
    minDistance = np.min(np.abs(r))
    return minDistance


def calcMetricForGivenValueOfEstimatedParsNLIN(ind, parsM, parsHM):
    """Return the folowing metric:
        - (phat - truepar)/truepar
        for non-linear parameters
    """
    r = [(parsHM['NLin'].values[ind] - parsM['NLinTrue'][i])/parsM['NLinTrue'][i] for i in range(len(parsM['NLinTrue']))]
    minDistance = np.min(np.abs(r))
    return minDistance


def getCostLinearParsH(parsHM, parsM):
    CostLin = []
    for i in range(len(parsHM['Lin'])):
        c = calcMetricForGivenValueOfEstimatedParsLIN(i, parsM, parsHM)
        CostLin.append(c)

    return np.array(CostLin)


def getCostNLinearParsH(parsHM, parsM):
    CostNLin = []
    for i in range(len(parsHM['NLin'])):
        c = calcMetricForGivenValueOfEstimatedParsNLIN(i, parsM, parsHM)
        CostNLin.append(c)

    return np.array(CostNLin)


def GettingParsAlgo(pars, parsh):
    """ pars -> True
        parsh -> Estimated
    """
    parsh1 = pd.DataFrame(parsh)
    pars1 = pd.DataFrame(pars)

    parsL = parsh1[0:3]
    parsL.columns = ['Lin']
    parsnL = parsh1[3:]
    parsnL.columns = ['NLin']
    parsnL.index = parsL.index

    lrgWght = parsL[np.abs(parsL)>10**-1].dropna()
    RES = pd.concat([lrgWght, parsnL.ix[lrgWght.index]], axis=1)

    parsM = pars1[0:3]
    parsM.columns = ['LinTrue']
    parsMNL = pars1[3:]
    parsMNL.columns = ['NLinTrue']
    parsMNL.index = parsL.index

    lrgWght2 = parsM[np.abs(parsM)>10**-1].dropna()
    RES2 = pd.concat([lrgWght2, parsMNL.ix[lrgWght2.index]], axis=1)

    return RES, RES2


def costAstIntegerForAGivenEstimator(pars, parsH):

    # Munging parameter estimates for computing metric
    parsHM, parsM = GettingParsAlgo(pars, parsH)

    # Getting cost (chi2) for linear parameters
    costLin = getCostLinearParsH(parsHM, parsM)

    # Getting cost (chi2) for non-linear parameters
    costNLin = getCostNLinearParsH(parsHM, parsM)

    # Total cost
    totalCost = np.sum(costLin**2.) + np.sum(costNLin**2.)

    return totalCost


########################################################################################################################

# Simulations #

########################################################################################################################

def simulateAndEstimateViaAllEstimators(pars, sampleSize, MC):

    LOGLM, LOGLP, LMHAT, LPHAT, QMLHAT = [], [], [], [], []
    costQML, costLP, costLM = [], [], []

    for i in range(MC):
        # Simulate
        sData = simulate2(pars, sampleSize)

        # Estimate QML
        ParsQml, _, _ = fitBest(sData)

        # Estimate via Lp and Mpl
        logLm, logLp, lmHat, lpHat = calcModLogLik(pars, sData)

        # CalcMetric
        cqml = costAstIntegerForAGivenEstimator(pars, ParsQml)
        clp  = costAstIntegerForAGivenEstimator(pars, lpHat)
        clmp = costAstIntegerForAGivenEstimator(pars, lmHat)

        # Appending cost
        costQML.append(cqml); costLP.append(clp); costLM.append(clmp)

        # Appending results
        #LOGLM.append(logLm); LOGLP.append(logLp); LMHAT.append(lmHat);
        #QMLHAT.append(ParsQml); LPHAT.append(logLp)

    # Make it a dataframe
    C1 = pd.DataFrame(costQML, columns=[np.str(sampleSize)])
    C2 = pd.DataFrame(costLP,  columns=[np.str(sampleSize)])
    C3 = pd.DataFrame(costLM,  columns=[np.str(sampleSize)])

    return C1, C2, C3


def get_all_files_from_a_folder(path):
    # get files from the folder
    """ GET ALL H5 files from a folder """

    FILES = dfn.getFiles(path)

    # append results
    HUGE = pd.DataFrame()
    K = FILES.keys()[:]; K = np.sort(K)

    # Run loop
    for files in K:
        try:
            DF = pd.read_hdf(path+files,'res')
            HUGE = pd.concat([HUGE,DF],axis=1)
        except:
            pass

    # Fill na with zeros
    HUGE = HUGE.fillna(0)

    return HUGE


def transform_str2float(df):
    df.index = df.index.astype(float)
    df.sort_index(ascending=True,inplace=True)
    return df


########################################################################################################################

# Plotting #

########################################################################################################################

# ---------------------------------------------------------------
def plotAllMethods(data, qmlHat, Lph, Lmph, pars, sampleSize):

    # Plot
    plt.plot(data, marker='o', linestyle='', markersize=16, color='r')
    plt.plot(simulate2(qmlHat, sampleSize, noise=False),
            linestyle='-', linewidth=4, color='k')
    plt.plot(data, color='r', linewidth=4)
    plt.plot(simulate2(pars, sampleSize, noise=False),
             linestyle='-', linewidth=4)
    plt.plot(simulate2(Lph, sampleSize, noise=False), color='g',linestyle='-', linewidth=4)
    plt.plot(simulate2(Lmph, sampleSize, noise=False), color='m',linestyle='-', linewidth=4)
    plt.legend(['Data','res.x','Data','True','Lp','Lmp'])
    plt.grid()
    plt.tight_layout()

# ---------------------------------------------------------------
def firstDiagnosisOnNoiseLevel(sampleSize, pars):

    # Simulate
    sData = simulate2(pars, sampleSize, noise=True)

    # Estimate
    ParsQml, fit, _ = fitBest(sData)

    # Monte-Carlo:
    #   - Several (10) data simulations
    #   - Several (10) fits
    SD = []
    FIT = []
    estimatedParsQmlMc = []
    for i in range(10):
        sData = simulate2(pars, sampleSize, noise=True)
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
    lpDf, lpHat, s_tc = profilingGammaOne(sData)

    f,ax = plt.subplots(1,1,figsize=(6,3))
    (lpDf).plot(ax=ax, linewidth=3, marker='.', markersize=12)
    plt.ylabel(r'$[L_p(\gamma_1|\hat{\eta})]$', fontsize=20)
    plt.xlabel(r'$\gamma_1$', fontsize=20)
    plt.axvline(lpDf[lpDf==lpDf.max()].dropna().index[0], color='r', linewidth=2)
    plt.axvline(lpHat[3], color='r', linestyle='--', linewidth=5)
    plt.grid()
    plt.tight_layout()

    # Plotting all esitimators
    plottingFitsAllEstimators(sData, pars, sampleSize, ParsQml, lpHat, lmHat)

    print('RMSE Qml*100 = %.5f'%np.sqrt(np.sum((sDataQml-sDataTrue)**2.)*100))
    print('RMSE Lp*100 = %.5f'%np.sqrt(np.sum((sDataLp-sDataTrue)**2.)*100))

    return ParsQml, lpHat, McParsQml



def plotMplGamma1And2(pars, log_Lm, log_LmG2, log_LmG3, logLp, logLpG2, logLpG3, lmHat, lmHatG2, lmHatG3, lpHat, lpHatG2, lpHatG3):

    """Comparing the 4 likelihood estimators: lp, lp_gamma2, lm, lm_gamma2"""

    f,ax = plt.subplots(1,3,figsize=(14,4), sharey=True)
    log_Lm.plot(ax=ax[0], logx=False, color='r', marker='s')
    ax[0].axvline(lmHat[3],color='r')
    ax[0].legend(['Lmp'], loc='upper right')
    ax[0].grid()
    a = ax[0].twinx()
    logLp.plot(ax=a,linestyle='--', marker='D')
    a.axvline(lpHat[3],color='b', linestyle=':', linewidth=3)
    a.axvline(pars[3],color='k', linestyle=':', linewidth=3)
    a.legend(['Lp'], loc='center right')
    a.set_yticklabels('')
    ax[0].set_xlabel(r'$\gamma_1$', fontsize=20)
    plt.tight_layout()

    log_LmG2.plot(ax=ax[1], logx=False, color='r', marker='s')
    ax[1].axvline(lmHatG2[4],color='r')
    ax[1].legend(['LmpG2'], loc='upper right')
    ax[1].grid()
    a = ax[1].twinx()
    logLpG2.plot(ax=a,linestyle='--', marker='D')
    a.axvline(pars[4],color='k', linestyle=':', linewidth=3)
    a.axvline(lpHatG2[4],color='b', linestyle=':', linewidth=3)
    a.legend(['LpG2'], loc='center right')
    a.set_yticklabels('')
    ax[1].set_xlabel(r'$\gamma_2$', fontsize=20)
    plt.tight_layout()

    log_LmG3.plot(ax=ax[2], logx=False, color='r', marker='s')
    ax[2].axvline(lmHatG3[5],color='r')
    ax[2].legend(['LmpG3'], loc='upper right')
    ax[2].grid()
    a = ax[2].twinx()
    logLpG3.plot(ax=a,linestyle='--', marker='D')
    a.axvline(pars[5],color='k', linestyle=':', linewidth=3)
    a.axvline(lpHatG3[5],color='b', linestyle=':', linewidth=3)
    a.legend(['LpG3'], loc='center right')
    a.set_yticklabels('')
    ax[2].set_xlabel(r'$\gamma_3$', fontsize=20)
    plt.tight_layout()


def plottingFitsAllEstimators(sData, pars, sampleSize, ParsQml, lpHat, lmHat):

    sDataQml   = simulate2(ParsQml, sampleSize, noise=False)
    sDataTrue  = simulate2(pars, sampleSize, noise=False)
    sDataLp    = simulate2(lpHat, sampleSize, noise=False)
    sDataLmp   = simulate2(lmHat, sampleSize, noise=False)

    f,ax = plt.subplots(1,1,figsize=(5,3))
    ax.plot(sData, marker='o', linestyle='', color='r', markersize=12)
    ax.plot(sDataQml, linestyle='-',linewidth=2, color='k')
    ax.plot(sDataTrue,linestyle='-',linewidth=2, color='b')
    ax.plot(sDataLp,linestyle='-',linewidth=2, color='c')
    ax.plot(sDataLmp,linestyle='--',linewidth=2, color='m')
    plt.ylabel(r'$Y$', fontsize=20)
    plt.xlabel(r'$N$', fontsize=20)
    plt.legend(['Data','Qml','True','Lp','Lmp'])
    plt.tight_layout()


def compareLpAndLmp(lpdf, lmdf, lph, lmh, pars):

    ms = 14
    lw = 5

    f, ax = plt.subplots(1,1, figsize=(8,4))
    lpdf.plot(ax=ax, marker='s', linewidth=lw, color='b', markersize=ms)
    a = plt.twinx()
    lmdf.plot(ax=a, marker='o', linewidth=lw, color='r', markersize=ms)
    ax.axvline(pars[3], color='k', linewidth=lw, linestyle='--')
    ax.axvline(lph[3], color='b', linewidth=lw)
    ax.axvline(lmh[3], color='r', linewidth=lw, linestyle='--')
    ax.grid()
    plt.tight_layout()

########################################################################################################################

# Modified Profile Likelihood # WITHOUT OPTIMISATION !

########################################################################################################################

# ---------------------------------------------------------------
def profile_gammaOne(pars, data, gamma_fix, simulate=False):

    """ enter a fix value for gamma_one """

    A,B,C = pars[0], pars[1], pars[2]
    gamma_one,gamma_two,gamma_three = gamma_fix, pars[3], pars[4]

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
def modLikStep(ind, pars, data, X_hat, s_tc):

    p = 5

    H, X = hessianForGivenGammaOne(pars, data, pars[3])

    detI = np.abs(np.linalg.det(H))  # To avoid duplicated calculations: I_psi
    detS = np.abs(np.linalg.det(np.dot(X_hat.T, X)))

    log_Lm = -(len(data)-p-2.)/2. * s_tc.values[ind][0] + detI/2. - detS

    return log_Lm


# ---------------------------------------------------------------
def calcModLogLik(pars, data, unbiased_s=True, job_pool=None):

     # Profile likelihood and hat{s}_tc
    logLp, lpHat, s_tc = profilingGammaOne(data)
    if unbiased_s:
        p = 5.
        s_tc = s_tc * float(len(data)) / float(len(data) - p)

    # X for MLE
    H, X_hat = hessianForGivenGammaOne(pars, data, lpHat[3])

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

    # Make a dframe
    log_Lm = pd.DataFrame(log_Lm, index=logLp.index)

    # Get pars
    bestGamma = log_Lm[log_Lm==log_Lm.max()].dropna().index[0]

    # Get pars
    lmHat, _, _ = fitBest(data, gammaFix=bestGamma)

    return log_Lm, logLp, lmHat, lpHat