
__author__ = 'demos'

from numpy import size, log, pi, sum, diff, array, zeros, diag, dot, mat, asarray, sqrt, copy
from numpy.linalg import inv
from pandas import read_csv
import pandas as pd
from scipy.optimize import fmin_slsqp
import scipy as sp
import sys
import numdifftools as ndt

import pandas as pd
import htools as htools
from scipy.optimize import minimize

# For parallel loops
from multiprocessing import cpu_count
try:
    CPU_COUNT = 6
except NotImplementedError:
    CPU_COUNT = 1

try:
    from joblib import Parallel, delayed
    RUN_PARALLEL = CPU_COUNT > 1
except ImportError:
    Parallel = None
    delayed = None
    RUN_PARALLEL = False

import itertools
import numpy as np
import os

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


# ---------------------------------------------------------------
def profile_gammaOne(pars, data, gamma_fix, simulate=False):

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


########################################################################################################################

# Modified Profile Likelihood #

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


# ---------------------------------------------------------------
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


########################################################################################################################

# Metrics #

########################################################################################################################

def calcMetricForGivenValueOfEstimatedParsLIN(ind, parsM, parsHM):

    r = [(parsHM['Lin'].values[ind] - parsM['LinTrue'][i])/parsM['LinTrue'][i] for i in range(len(parsM['LinTrue']))]
    minDistance = np.min(np.abs(r))
    return minDistance


# ---------------------------------------------------------------
def calcMetricForGivenValueOfEstimatedParsNLIN(ind, parsM, parsHM):

    r = [(parsHM['NLin'].values[ind] - parsM['NLinTrue'][i])/parsM['NLinTrue'][i] for i in range(len(parsM['NLinTrue']))]
    minDistance = np.min(np.abs(r))
    return minDistance


# ---------------------------------------------------------------
def getCostLinearParsH(parsHM, parsM):
    CostLin = []
    for i in range(len(parsHM['Lin'])):
        c = calcMetricForGivenValueOfEstimatedParsLIN(i, parsM, parsHM)
        CostLin.append(c)

    return np.array(CostLin)


# ---------------------------------------------------------------
def getCostNLinearParsH(parsHM, parsM):
    CostNLin = []
    for i in range(len(parsHM['NLin'])):
        c = calcMetricForGivenValueOfEstimatedParsNLIN(i, parsM, parsHM)
        CostNLin.append(c)

    return np.array(CostNLin)


# ---------------------------------------------------------------
def GettingParsAlgo(pars, parsh):

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


# ---------------------------------------------------------------
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
        logLp, logLm, lpHat, lmHat = LpAndMlpOptimising(sData)

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


#----------------------------------------


def brutus_jobindex():
    return int(os.environ['LSB_JOBINDEX'])

#----------------------------------------


# Get job ID
ID = brutus_jobindex() - 1

# Initial setup
pars = np.array([.2, .3, .4, .3, .15, .075])
MC = 10

# Path (Brutus location)
path = '/cluster/home02/mtec/gdemos/work/mod_likelihood_SOE/res_2/'

# smaplesize
sampleSize = np.arange(3,70,3)
ParForThisRun = sampleSize[ID]
print(ID)

# c1 -> qml
# c2 -> lp
# c3 -> lmp

# Estimate
c1, c2, c3 = simulateAndEstimateViaAllEstimators(pars, ParForThisRun, MC)

# save
c1.to_hdf(path+"testQml_1_MC10_%i.h5" %ID, "res")
c2.to_hdf(path+"testLp_1_MC10_%i.h5" %ID, "res")
c3.to_hdf(path+"testLmp_1_MC10_%i.h5" %ID, "res")

