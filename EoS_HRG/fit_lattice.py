import numpy as np
import pandas as pd 
from math import factorial
import scipy.optimize
import scipy.misc
import os
import re
import argparse
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,ConstantKernel
# for tests
#import matplotlib.pyplot as pl 

# directory where the fit_lattice_test.py is located
dir_path = os.path.dirname(os.path.realpath(__file__))

########################################################################
if __name__ == "__main__": 
    __doc__="""Construct a parametrization (from PhysRevC.100.064910) of the lattice QCD equation of state 
(P/T^4, n/T^3, s/T^3, e/T^4) by calling function:
- param(T,muB,muQ,muS)
  input: temperature and chemical potentials in [GeV]
  output: dictionnary of all quantities ['T','P','s','n_B','n_Q','n_S','e']

Produces lattice data for P/T^4, nB/T^3, s/T^3, e/T^4 as a function of T for a single value of muB:
- lattice_data(EoS,muB)
  input: - EoS: - 'muB' refers to the EoS with the condition \mu_Q = \mu_S = 0
                - 'nS0' refers to the EoS with the condition <n_S> = 0 & <n_Q> = 0.4 <n_B>
         - muB: baryon chemical potential in [GeV]
  output: dictionnary of all quantities + error ['T','P','s','n_B','e']

Calculation of the equation of state under the conditions: <n_S> = 0 ; <n_Q> = factQB*<n_B>:
- EoS_nS0(fun,T,muB,**kwargs)
  input: - fun: any function which calculate an EoS (by ex: param, HRG, full_EoS)
         - T,muB: temperature and baryon chemical potential in [GeV]
  output: dictionnary of all quantities ['T','P','s','n_B','e']
"""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter
    )
    args = parser.parse_args()

###############################################################################
# import data from lattice at muB = 0
###############################################################################
WB_EoS0 = pd.read_csv(dir_path+"/data/WB-EoS_muB0.csv", skiprows=1).to_dict('list')
for chi in WB_EoS0:
    WB_EoS0[chi] = np.array(WB_EoS0[chi])
    
###############################################################################
# import data for the parametrization of susceptibilities
###############################################################################

param_chi_a = pd.read_csv(dir_path+"/data/chi_a.csv", skiprows=1)
chi_a = {}
for i,chi in enumerate(param_chi_a['chi']):
    values = np.zeros(len(param_chi_a)-1)
    for j,coeff in enumerate(param_chi_a):
        if(coeff=='chi'):
            continue
        values[j-1] = param_chi_a[coeff][i]
    chi_a[chi] = values
    
param_chi_b = pd.read_csv(dir_path+"/data/chi_b.csv", skiprows=1)
chi_b = {}
for i,chi in enumerate(param_chi_b['chi']):
    values = np.zeros(len(param_chi_b)-1)
    for j,coeff in enumerate(param_chi_b):
        if(coeff=='chi'):
            continue
        values[j-1] = param_chi_b[coeff][i]
    chi_b[chi] = values

########################################################################
def param_chi(T,quant):
    """
    Parametriation of the susceptibilities at as a function of temperature
    Ex: param_chi(T,'chiBQS121')
    input quant is a string with the format: chiBQS121
    input T being a list or a float
    Parametrizations from: PhysRevC.100.064910
    """
    if(quant!='chiB2'):
        tt = T/0.154
        numerator = sum([ai/(tt)**i for i,ai in enumerate(chi_a[quant])])
        denominator = sum([bi/(tt)**i for i,bi in enumerate(chi_b[quant][:-1])])
        c0 = chi_b[quant][-1]
        param_chi = numerator/denominator + c0
    else:
    # for chiB2, other parametrization
        tt = T/0.2
        h1 = -0.325372
        h2 = 0.497729
        f3 = 0.148987
        f4 = 6.66388
        f5 = -5.07725
        param_chi = np.exp(-h1/tt-h2/tt**2.)*f3*(1.+np.tanh(f4*tt+f5))
    return param_chi

########################################################################
# list of all susceptibilities
list_chi = list(param_chi_a['chi'])
list_chi.append('chiB2')

"""
########################################################################
# test and plot parametrization of each susceptibility
xtemp = np.linspace(0.00001,0.6,100)
for chi in list_chi:
    f,ax = pl.subplots(figsize=(10,7))
    y_chi = param_chi(xtemp,chi)
    ax.plot(xtemp,y_chi)
    ax.set(xlabel='T [GeV]',ylabel=chi)
    f.savefig(f'{dir_path}/{chi}.png')
    pl.show()
"""

########################################################################
# for each susceptibility, get the order of the derivative wrt B,Q,S
########################################################################

BQS = dict(zip(list_chi,[{'B': 0., 'Q': 0., 'S': 0.} for i in range(len(list_chi))]))
for chi in list_chi:
    # derivatives wrt to each charge
    if(chi!='chi0'):
        # decompose chiBQS234 in [B,Q,S] and [2,3,4]
        chi_match = re.match('chi([A-Z]+)([0-9]+)', chi)
        list_charge = list(chi_match.group(1)) # contains the charges
        list_der = list(chi_match.group(2)) # contains the derivatives
        for ich,xcharge in enumerate(list_charge):
            BQS[chi][xcharge] = int(list_der[ich]) # match each charge to its derivative

########################################################################
def param(T,muB,muQ,muS):
    """
    Parametrization of thermodynamic quantities from lQCD
    as a function of T, \mu_B, \mu_Q, \mu_S
    Parametrization from: PhysRevC.100.064910
    """
    # if input is a single temperature value T
    if(isinstance(T,float)):
        p = 0.
        nB = 0.
        nQ = 0.
        nS = 0.
        s = 0.
        e = 0.

        for chi in list_chi:
            i = BQS[chi]['B']
            j = BQS[chi]['Q']
            k = BQS[chi]['S']
            fact = 1./(factorial(i)*factorial(j)*factorial(k))
            xchi = param_chi(T,chi)
            pow_muB = ((muB/T)**i)
            pow_muQ = ((muQ/T)**j)
            pow_muS = ((muS/T)**k)
            # pressure P/T^4
            p += fact*xchi*pow_muB*pow_muQ*pow_muS
            # baryon density n_B/T^3 when i > 1
            if(i >= 1):
                nB += fact*xchi*i*((muB/T)**(i-1.))*pow_muQ*pow_muS
            # charge density n_Q/T^3 when i > 1
            if(j >= 1):
                nQ += fact*xchi*pow_muB*j*((muQ/T)**(j-1.))*pow_muS
            # strangeness density n_S/T^3 when k > 1
            if(k >= 1):
                nS += fact*xchi*pow_muB*pow_muQ*k*((muS/T)**(k-1.))
            # derivative of the susceptibility wrt temperature
            der = scipy.misc.derivative(lambda xT : param_chi(xT,chi),T,dx=1e-5)
            # s/T^3 = T d(P/T^4)/d(T) + 4 P/T^4
            # here we add just the 1st part
            s += fact*(T*der-(i+j+k)*xchi)*pow_muB*pow_muQ*pow_muS
        # add 2nd piece to s/T^3
        s += 4.*p
        # energy density e/T^4
        e = s-p+(muB/T)*nB+(muQ/T)*nQ+(muS/T)*nS
    
    # if the input is a list of temperature values
    elif(isinstance(e,np.ndarray) or isinstance(e,list)):
        p = np.zeros_like(T)
        s = np.zeros_like(T)
        nB = np.zeros_like(T)
        nQ = np.zeros_like(T)
        nS = np.zeros_like(T)
        e = np.zeros_like(T)
        for i,xT in enumerate(T):
            result = param(xT,muB,muQ,muS)
            p[i] = result['P']
            s[i] = result['s']
            nB[i] = result['n_B']
            nQ[i] = result['n_Q']
            nS[i] = result['n_S']
            e[i] = result['e']

    else:
        raise Exception('Problem with input')
                
    return {'T': T,'P':p, 's':s, 'n_B':nB, 'n_Q':nQ, 'n_S':nS, 'e':e}

###############################################################################
# critical temperature from lattice at \mu_B = 0
Tc0 = 0.158 
# expansion coefficients of T_c(\mu_B)
kappa2 = 0.015
kappa4 = 0.
###############################################################################
def Tc_lattice(muB):
    """
    Critical temperature as a function of muB
    """
    # lattice evaluation of Tc(muB)
    return Tc0*(1.-kappa2*(muB/Tc0)**2.-kappa4*(muB/Tc0)**4.)

###############################################################################
def dTcdmuB_lattice(muB):
    """
    Derivative of the critical temperature wrt \mu_B
    """
    dTc = -2.*muB*kappa2/Tc0 -4.*(muB**3.)*kappa4/Tc0**3.
    return dTc

###############################################################################
def lattice_data(EoS,muB):
    """
    Produces the lattice data for P/T^4, nB/T^3, s/T^3, e/T^4 as a function of T for a single value of muB
    """

    # select which susceptibilities to use according to the EoS to evaluate
    if(EoS=='nS0'):
    # import data from lattice for the susceptibilities 
    # case when <n_S> = 0 & <n_Q> = 0.4 <n_B>
        WB_chi = pd.read_csv(dir_path+"/data/WB_chi_T_nS0.csv", skiprows=1).to_dict('list')
        for chi in WB_chi:
            WB_chi[chi] = np.array(WB_chi[chi])
    # case when \mu_u = \mu_d = \mu_s = \mu_B/3
    elif(EoS=='muB'):
        WB_chi = pd.read_csv(dir_path+"/data/WB_chi_T_muB.csv", skiprows=1).to_dict('list')
        for chi in WB_chi:
            WB_chi[chi] = np.array(WB_chi[chi])

    # Maximum order in the susceptibilities from lattice
    ord_max = 6

    # case when \mu_B = 0
    if(muB == 0.):
        xtemp = WB_EoS0['T']
        result_P = WB_EoS0['P']
        err_P = WB_EoS0['P_err']
        # return n_B = 0 
        result_nB = np.zeros_like(xtemp)
        err_nB = np.zeros_like(xtemp)
        result_s = WB_EoS0['s']
        err_s = WB_EoS0['s_err']
        result_e = WB_EoS0['e']
        err_e = WB_EoS0['e_err']
            
    # case when \mu_B != 0
    else:
        # now range in T is defined by the range covered by the susceptibilities (chi)
        xtemp = WB_chi['T']
        result_p0 = np.zeros_like(xtemp)
        err_p0 = np.zeros_like(xtemp)
        result_s0 = np.zeros_like(xtemp)
        err_s0 = np.zeros_like(xtemp)
        result_e0 = np.zeros_like(xtemp)
        err_e0 = np.zeros_like(xtemp)

        # scan values of T in the values of chi
        for i,xT1 in enumerate(xtemp):
            # scan values of T in the EoS at muB = 0
            for j,xT2 in enumerate(WB_EoS0['T']):
                # search for a match
                if(xT1 == xT2):
                    result_p0[i] = WB_EoS0['P'][j]
                    err_p0[i] = WB_EoS0['P_err'][j]

                    result_s0[i] = WB_EoS0['s'][j]
                    err_s0[i] = WB_EoS0['s_err'][j]

                    result_e0[i] = WB_EoS0['e'][j]
                    err_e0[i] = WB_EoS0['e_err'][j]
    
        def dchidT(i):
            """
            Return the list corresponding to the estimation of the derivative wrt T of the susceptibility of order i
            """            
            # Instantiate a Gaussian Process model
            kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3))*RBF(length_scale=0.01, length_scale_bounds=(0.001, 0.1))
            # alpha (noise) proportionnal to the actual error in lQCD data for chi(T)
            gp = GaussianProcessRegressor(kernel=kernel, alpha=WB_chi[f'P{i}_err']/100., n_restarts_optimizer=1)
            # Fit to data using Maximum Likelihood Estimation of the parameters
            gp.fit(np.atleast_2d(xtemp).T, np.atleast_2d(WB_chi[f'P{i}']).T)
            # calculate derivative by using prediction from GP
            dchidT = np.zeros_like(xtemp)
            dT=1e-5
            for iT,xT in enumerate(xtemp):
                eval = gp.predict(np.atleast_2d([xT+dT,xT-dT]).T, return_std=False)
                dchidT[iT] = (eval[0]-eval[1])/(2.*dT)

            """
            # tests : plot to see the accuracy of GP
            y_pred, sigma = gp.predict(np.atleast_2d(xtemp).T, return_std=True)
            f,ax = pl.subplots(figsize=(10,7))
            ax.plot(xtemp, WB_chi[f'P{i}'],'o')
            ax.errorbar(xtemp, WB_chi[f'P{i}'], yerr=WB_chi[f'P{i}_err'], xerr=None)
            ax.plot(xtemp, y_pred.flat)
            ax.fill_between(xtemp, y_pred.flat-2*sigma, y_pred.flat+2*sigma, color="#dddddd")
            ax.plot(xtemp,dchidT/30.)
            f.savefig(f'{dir_path}/test_chi{i}.png')
            pl.show()
            """
            
            return dchidT

        # error is calculated such as f = a_1*x_1 + a_2*x_2: \delta(f) = sqrt((a_1*\delta(x_1))**2. + (a_2*\delta(x_2))**2.)
        if(EoS == 'nS0'):
            # pressure P/T^4
            result_P = result_p0 + sum([((muB/xtemp)**i)*WB_chi[f'P{i}'] for i in range(2,ord_max+1,2)])
            err_P = np.sqrt(err_p0**2. + sum([(((muB/xtemp)**i)*WB_chi[f'P{i}_err'])**2. for i in range(2,ord_max+1,2)]))
            # baryon density n_B\T^3
            result_nB = sum([(i*(muB/xtemp)**(i-1.))*WB_chi[f'P{i}'] for i in range(2,ord_max+1,2)])
            err_nB = np.sqrt(sum([((i*(muB/xtemp)**(i-1.))*WB_chi[f'P{i}_err'])**2. for i in range(2,ord_max+1,2)]))
            # entropy density s/T^3 = s_0/T^3 + 4*\Delta P/T^4 + T d(\Delta P/T^4)/dT
            result_s = result_s0 + sum([((muB/xtemp)**i)*(xtemp*dchidT(i)+(4.-i)*WB_chi[f'P{i}']) for i in range(2,ord_max+1,2)])
            err_s = None
            # energy density e/T^4 = s/T^3 - P/T^4 + \mu_B/T n_B/T^3
            result_e = result_e0 + sum([((muB/xtemp)**i)*(xtemp*dchidT(i)+3.*WB_chi[f'P{i}']) for i in range(2,ord_max+1,2)])
            err_e = None
        elif(EoS == 'muB'):
            # pressure P/T^4
            result_P = result_p0 + sum([((muB/xtemp)**i)*WB_chi[f'P{i}']/factorial(i) for i in range(2,ord_max+1,2)])
            err_P = np.sqrt(err_p0**2. + sum([(((muB/xtemp)**i)*WB_chi[f'P{i}_err']/factorial(i))**2. for i in range(2,ord_max+1,2)]))
            # baryon density n_B\T^3
            result_nB = sum([(i*(muB/xtemp)**(i-1.))*WB_chi[f'P{i}']/factorial(i) for i in range(2,ord_max+1,2)])
            err_nB = np.sqrt(sum([((i*(muB/xtemp)**(i-1.))*WB_chi[f'P{i}_err']/factorial(i))**2. for i in range(2,ord_max+1,2)]))
            # entropy density s/T^3
            result_s = result_s0 + sum([((muB/xtemp)**i)*(xtemp*dchidT(i)+(4.-i)*WB_chi[f'P{i}'])/factorial(i) for i in range(2,ord_max+1,2)])
            err_s = None
            # energy density e/T^4 = s/T^3 - P/T^4 + \mu_B/T n_B/T^3
            result_e = result_e0 + sum([((muB/xtemp)**i)*(xtemp*dchidT(i)+3.*WB_chi[f'P{i}'])/factorial(i) for i in range(2,ord_max+1,2)])
            err_e = None
    
    return {'T': xtemp, 'P': [result_P,err_P], 'n_B': [result_nB,err_nB],'s': [result_s,err_s], 'e': [result_e,err_e]}   

###############################################################################
def EoS_nS0(fun,xT,muB,**kwargs):
    """
    Calculation of the EoS defined by the input function at (T,muB) with the conditions:
    <n_S> = 0
    <n_Q> = factQB*<n_B>
    """
    factQB = 0.4

    if(isinstance(xT,float)):
        T = xT
        p = 0.
        nB = 0.
        s = 0.
        e = 0.
        
        def system(mu):
            """
            Define the system to be solved
            <n_S> = 0 
            <n_Q> = factQB * <n_B>
            """
            thermo = fun(T,muB,mu[0],mu[1],**kwargs)
            return [thermo['n_S'], thermo['n_Q']-factQB*thermo['n_B']]
            
        solution = scipy.optimize.root(system,[-0.08*muB,0.03*muB],method='lm').x
        muQ = solution[0]
        muS = solution[1]
        result = fun(T,muB,muQ,muS,**kwargs)
            
        p = result['P']
        s = result['s']
        nB = result['n_B']
        e = result['e'] 
        
    elif(isinstance(xT,np.ndarray) or isinstance(e,list)):
        p = np.zeros_like(xT)
        s = np.zeros_like(xT)
        nB = np.zeros_like(xT)
        e = np.zeros_like(xT)
        muQ = np.zeros_like(xT)
        muS = np.zeros_like(xT)
        for i,T in enumerate(xT):
            result = EoS_nS0(fun,T,muB,**kwargs)
            p[i] = result['P']
            s[i] = result['s']
            nB[i] = result['n_B']
            e[i] = result['e']
            muQ[i] = result['muQ']
            muS[i] = result['muS']
    
    else:
        raise Exception('Problem with input')
    
    return {'T':xT, 'muQ': muQ, 'muS': muS, 'P':p, 's':s, 'n_B':nB, 'e':e} 
