import numpy as np
import pandas as pd
from math import factorial, pi
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
# J. Phys.: Conf. Ser. 1602 012011
# critical temperature from lattice at \mu_B = 0
Tc0 = 0.158 
# expansion coefficients of T_c(\mu_B)
kappa2 = 0.0153
kappa4 = 0.00032 
###############################################################################
def Tc_lattice(muB):
    """
    Critical temperature as a function of muB from lQCD
    J. Phys.: Conf. Ser. 1602 012011
    """   
    return Tc0*(1.-kappa2*(muB/Tc0)**2.-kappa4*(muB/Tc0)**4.)

###############################################################################
def dTcdmuB_lattice(muB):
    """
    Derivative of the critical temperature wrt \mu_B
    """
    dTc = -2.*muB*kappa2/Tc0 -4.*(muB**3.)*kappa4/Tc0**3.
    return dTc

def Tc_lattice_muBoT(muBoT):
    """
    Find the critical temperature Tc for a fixed muB/T
    """
    if(muBoT==0):
        return Tc_lattice(0.)
    else:
        xmuB = scipy.optimize.root(lambda muB: muB/Tc_lattice(muB)-muBoT,[muBoT*Tc0],method='lm').x[0]
        return Tc_lattice(xmuB)

###############################################################################
# import data for the parametrization of susceptibilities
###############################################################################

chi_a = {}
for chi_file in ["/data/chi_a_nS0.csv","/data/chi_a.csv"]:
    param_chi_a = pd.read_csv(dir_path+chi_file).to_dict(orient='list')
    # scan rows for each chi
    for i,chi in enumerate(param_chi_a['chi']):
        values = []
        # scan columns with coefficients
        for j,coeff in enumerate(param_chi_a):
            # skip first column which is chi string
            if(coeff=='chi'):
                continue
            # append values
            values.append(param_chi_a[coeff][i])
        chi_a.update({chi:values})

chi_b = {}
for chi_file in ["/data/chi_b_nS0.csv","/data/chi_b.csv"]:
    param_chi_b = pd.read_csv(dir_path+chi_file).to_dict(orient='list')
    # scan rows for each chi
    for i,chi in enumerate(param_chi_b['chi']):
        values = []
        # scan columns with coefficients
        for j,coeff in enumerate(param_chi_b):
            # skip first column which is chi string
            if(coeff=='chi'):
                continue
            # append values
            values.append(param_chi_b[coeff][i])
        chi_b.update({chi:values})

# list of all susceptibilities
list_chi = list(param_chi_a['chi'])
list_chi_nS0 = ['chiB2_nS0','chiB4_nS0']

########################################################################
# Stefan Boltzmann limit for the susceptibilities
# can be found in PhysRevC.100.064910
chi_SB = dict(zip(list_chi,[19.*pi**2./36.,\
    1./3.,2./3.,1.,\
    0.,-1./3.,1./3.,\
    2./(9.*pi**2.),4./(3*pi**2.),6./pi**2.,\
    0.,-2./(9.*pi**2.),2./(9.*pi**2.),\
    4./(9.*pi**2.),-2./pi**2.,2./pi**2.,\
    4./(9.*pi**2.),2./(3.*pi**2.),2./(3.*pi**2.),\
    2./(9.*pi**2.),-2./(9.*pi**2.),-2./(3.*pi**2.)]))

chi_SB.update(dict(zip(list_chi_nS0,[0.1067856506125367,0.0006673764465596013])))

########################################################################
def param_chi(T,quant):
    """
    Parametriation of the susceptibilities at as a function of temperature
    Ex: param_chi(T,'chiBQS121')
    input quant is a string with the format: chiBQS121
    input T being a list or a float
    """
    tt = T/Tc_lattice(0.)
    numerator = sum([ai/(tt)**i for i,ai in enumerate(chi_a[quant])])
    denominator = sum([bi/(tt)**i for i,bi in enumerate(chi_b[quant])])
    c0 = chi_SB[quant]-chi_a[quant][0]/chi_b[quant][0]
    return numerator/denominator + c0

########################################################################
# for each susceptibility, get the order of the derivative wrt B,Q,S
########################################################################

BQS = dict(zip(list_chi,[{'B': 0, 'Q': 0, 'S': 0} for i in range(len(list_chi))]))
chi_latex = {'chi0':r'$\chi_0$'}
for chi in list_chi:
    # derivatives wrt to each charge
    if(chi!='chi0'):
        # decompose chiBQS234 in [B,Q,S] and [2,3,4]
        chi_match = re.match('chi([A-Z]+)([0-9]+)', chi)
        list_charge = list(chi_match.group(1)) # contains the charges
        list_der = list(chi_match.group(2)) # contains the derivatives
        chi_latex.update({chi:r'$\chi^{'+"".join(list_charge)+'}_{'+"".join(list_der)+'}$'})
        for ich,xcharge in enumerate(list_charge):
            BQS[chi][xcharge] = int(list_der[ich]) # match each charge to its derivative

chi_latex.update({'chiB2_nS0':r'$c_2$', 'chiB4_nS0':r'$c_4$'})

########################################################################
def param(T,muB,muQ,muS):
    """
    Parametrization of thermodynamic quantities from lQCD
    as a function of T, \mu_B, \mu_Q, \mu_S
    """
    # if input is a single temperature value T
    if(isinstance(T,float)):
        p = 0.
        nB = 0.
        nQ = 0.
        nS = 0.
        s = 0.
        e = 0.

        if(muB==0. and muQ==0. and muS==0.):
            p = param_chi(T,'chi0')
            der = scipy.misc.derivative(param_chi,T,dx=1e-5,args=('chi0',))
            s = T*der
        else:
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
                der = scipy.misc.derivative(param_chi,T,dx=1e-5,args=(chi,))
                # s/T^3 = T d(P/T^4)/d(T) + 4 P/T^4
                # here we add just the 1st part
                s += fact*(T*der-(i+j+k)*xchi)*pow_muB*pow_muQ*pow_muS
        # add 2nd piece to s/T^3
        s += 4.*p
        # energy density e/T^4
        e = s-p+(muB/T)*nB+(muQ/T)*nQ+(muS/T)*nS
    
    # if the input is a list of temperature values
    elif(isinstance(T,np.ndarray) or isinstance(T,list)):
        p = np.zeros_like(T)
        s = np.zeros_like(T)
        nB = np.zeros_like(T)
        nQ = np.zeros_like(T)
        nS = np.zeros_like(T)
        e = np.zeros_like(T)
        for i,xT in enumerate(T):
            # see if arrays are also given for chemical potentials
            try:
                xmuB = muB[i]
            except:
                xmuB = muB
            try:
                xmuQ = muQ[i]
            except:
                xmuQ = muQ
            try:
                xmuS = muS[i]
            except:
                xmuS = muS
            result = param(xT,xmuB,xmuQ,xmuS)
            p[i] = result['P']
            s[i] = result['s']
            nB[i] = result['n_B']
            nQ[i] = result['n_Q']
            nS[i] = result['n_S']
            e[i] = result['e']

    else:
        raise Exception('Problem with input')
                
    return {'T': T,'P':p, 's':s, 'n_B':nB, 'n_Q':nQ, 'n_S':nS, 'e':e, 'I':e-3*p}

########################################################################
def param_nS0(T,muB):
    """
    Parametrization of thermodynamic quantities from lQCD
    as a function of T, \mu_B for the case <n_S>=0 & <n_Q>=0.4<n_B>
    """
    # if input is a single temperature value T
    if(isinstance(T,float)):
        p = 0.
        nB = 0.
        nQ = 0.
        nS = 0.
        s = 0.
        e = 0.

        p = param_chi(T,'chi0')
        der = scipy.misc.derivative(param_chi,T,dx=1e-5,args=('chi0',))
        s = T*der
        if(muB!=0.):
            for ichi,chi in enumerate(list_chi_nS0):
                i = 2*(ichi+1)
                xchi = param_chi(T,chi)
                pow_muB = ((muB/T)**i)
                # pressure P/T^4
                p += xchi*pow_muB
                # baryon density n_B/T^3 when i > 1
                nB += xchi*i*((muB/T)**(i-1.))
                # derivative of the susceptibility wrt temperature
                der = scipy.misc.derivative(param_chi,T,dx=1e-5,args=(chi,))
                # s/T^3 = T d(P/T^4)/d(T) + 4 P/T^4
                # here we add just the 1st part
                s += (T*der-(i)*xchi)*pow_muB
        # add 2nd piece to s/T^3
        s += 4.*p
        # energy density e/T^4
        e = s-p+(muB/T)*nB
        # charge density
        nQ = 0.4*nB
    
    # if the input is a list of temperature values
    elif(isinstance(T,np.ndarray) or isinstance(T,list)):
        p = np.zeros_like(T)
        s = np.zeros_like(T)
        nB = np.zeros_like(T)
        nQ = np.zeros_like(T)
        nS = np.zeros_like(T)
        e = np.zeros_like(T)
        for i,xT in enumerate(T):
            # see if arrays are also given for chemical potentials
            try:
                xmuB = muB[i]
            except:
                xmuB = muB
            result = param_nS0(xT,xmuB)
            p[i] = result['P']
            s[i] = result['s']
            nB[i] = result['n_B']
            nQ[i] = result['n_Q']
            nS[i] = result['n_S']
            e[i] = result['e']

    else:
        raise Exception('Problem with input')
                
    return {'T': T,'P':p, 's':s, 'n_B':nB, 'n_Q':nQ, 'n_S':nS, 'e':e, 'I':e-3*p}

###############################################################################
# import data from lattice at muB = 0
###############################################################################

# read chi0 
WB_EoS0 = pd.read_csv(dir_path+"/data/WB-EoS_muB0_j.physletb.2014.01.007.csv").to_dict(orient='list')
chi_lattice2014 = {'chi0':np.array(list(zip(WB_EoS0['T'],WB_EoS0['P'],WB_EoS0['P_err'])))}
# save all other thermodynamic quantities
for quant in WB_EoS0:
    WB_EoS0[quant] = np.array(WB_EoS0[quant])
# read data from 2012 (chiB2,chiQ2,chiS2)
chi_lattice2012 = {}
try:
    df = pd.read_csv(dir_path+"/data/WB_chi_T_JHEP01(2012)138.csv").to_dict(orient='list')
    for entry in df:
        if(entry=='T' or '_err' in entry):
            continue
        chi_lattice2012.update({entry:np.array(list(zip(df['T'],df[entry],df[entry+'_err'])))})
except:
    pass
# read data from 2015 (chiB2,chiB4,chiS2)
chi_lattice2015 = {}
try:
    df = pd.read_csv(dir_path+"/data/WB_chi_T_PhysRevD.92.114505.csv").to_dict(orient='list')
    for entry in df:
        if(entry=='T' or '_err' in entry):
            continue
        chi_lattice2015.update({entry:np.array([[df['T'][iT],df[entry][iT],df[entry+'_err'][iT]] for iT,_ in enumerate(df[entry]) if np.logical_not(np.isnan(df[entry][iT]))])})
except:
    pass
# read data from 2017 (chiB2,chiB4,chiB2) for <nS>=0 & <nQ>=0.4<nB>
chi_lattice2017 = {}
try:
    df = pd.read_csv(dir_path+"/data/WB_chi_nS0_T_EPJWebConf.137(2017)07008.csv").to_dict(orient='list')
    for entry in df:
        if(entry=='T' or '_err' in entry):
            continue
        chi_lattice2017.update({entry:np.array([[df['T'][iT],df[entry][iT],df[entry+'_err'][iT]] for iT,_ in enumerate(df[entry]) if np.logical_not(np.isnan(df[entry][iT]))])})
except:
    pass
# read data from 2018
chi_lattice2018 = {}
try:
    df = pd.read_csv(dir_path+"/data/WB_chi_T_JHEP10(2018)205.csv").to_dict(orient='list')
    for entry in df:
        if(entry=='T' or '_err' in entry):
            continue
        chi_lattice2018.update({entry:np.array(list(zip(df['T'],df[entry],df[entry+'_err'])))})
except:
    pass
# read data from 2020 (chiBQ11,chiBS11,chiQS11)
chi_lattice2020 = {}
try:
    df = pd.read_csv(dir_path+"/data/WB_chi_T_PhysRevD.101.034506.csv").to_dict(orient='list')
    for entry in df:
        if(entry=='T' or '_err' in entry):
            continue
        chi_lattice2020.update({entry:np.array(list(zip(df['T'],df[entry],df[entry+'_err'])))})
except:
    pass
# read data from 2021
WB_EoS_muBoT2021 = {}
try:
    df = pd.read_csv(dir_path+"/data/WB-EoS_muBoT_2102.06660.csv").to_dict(orient='list')
    for entry in df:
        if(entry=='T' or '_err' in entry):
            continue
        WB_EoS_muBoT2021.update({entry:np.array(list(zip(df['T'],df[entry],df[entry+'_err'])))})
except:
    pass

###############################################################################
def EoS_nS0(fun,T,muB,**kwargs):
    """
    Calculation of the EoS defined by the input function at (T,muB) with the conditions:
    <n_S> = 0
    <n_Q> = factQB*<n_B>
    """
    factQB = 0.4

    if(isinstance(T,float)):
        p = 0.
        nB = 0.
        nQ = 0.
        nS = 0.
        s = 0.
        e = 0.
        n = 0.
        chi = np.zeros(len(list_chi))
        
        def system(mu):
            """
            Define the system to be solved
            <n_S> = 0 
            <n_Q> = factQB * <n_B>
            """
            thermo = fun(T,muB,mu[0],mu[1],**kwargs)
            return [thermo['n_S']*T**3, thermo['n_Q']*T**3-factQB*thermo['n_B']*T**3]
            
        solution = scipy.optimize.root(system,[-0.08*muB,0.03*muB],method='lm').x
        muQ = solution[0]
        muS = solution[1]
        result = fun(T,muB,muQ,muS,**kwargs)
            
        p = result['P']
        s = result['s']
        nB = result['n_B']
        nQ = factQB*nB
        nS = 0.
        e = result['e'] 
        # some extra quantities are calculated within HRG function
        try:
            n = result['n']
            chi = result['chi']
        except:
            pass
        
    elif(isinstance(T,np.ndarray) or isinstance(T,list)):
        p = np.zeros_like(T)
        s = np.zeros_like(T)
        nB = np.zeros_like(T)
        nQ = np.zeros_like(T)
        nS = np.zeros_like(T)
        n = np.zeros_like(T)
        e = np.zeros_like(T)
        muQ = np.zeros_like(T)
        muS = np.zeros_like(T)
        chi = np.zeros((len(list_chi),len(T)))
        for i,xT in enumerate(T):
            # see if arrays are also given for chemical potentials
            try:
                xmuB = muB[i]
            except:
                xmuB = muB
            result = EoS_nS0(fun,xT,xmuB,**kwargs)
            p[i] = result['P']
            s[i] = result['s']
            nB[i] = result['n_B']
            nQ[i] = result['n_Q']
            nS[i] = result['n_S']
            n[i] = result['n']
            e[i] = result['e']
            muQ[i] = result['muQ']
            muS[i] = result['muS']
            chi[:,i] = result['chi']
    
    else:
        raise Exception('Problem with input')
    
    return {'T':T, 'muQ': muQ, 'muS': muS, 'P':p, 's':s, 'n_B':nB, 'n_Q':nQ, 'n_S':nS, 'n':n, 'e':e, 'chi':chi, 'I':e-3*p} 
