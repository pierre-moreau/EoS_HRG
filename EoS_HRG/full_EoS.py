import numpy as np
import os
import argparse
import scipy.optimize

from EoS_HRG.HRG import HRG
from EoS_HRG.fit_lattice import Tc_lattice, param, dTcdmuB_lattice, EoS_nS0

########################################################################
if __name__ == "__main__": 
    __doc__="""Construct the EoS (P/T^4, n/T^3, s/T^3, e/T^4) 
resulting from a matching between the lattice QCD EoS
from EoS_HRG.fit_lattice and the HRG EoS from EoS_HRG.HRG:
- full_EoS(T,muB,muQ,muS,**kwargs)
  input: temperature and chemical potentials in [GeV]
  kwargs: - species = all, mesons, baryons -> which particles to include in the HRG?
          - offshell = True, False -> integration over mass for unstable particles in the HRG?
  output: dictionnary of all quantities ['T','P','s','n_B','n_Q','n_S','e']

From function full_EoS, find [T,muB,muQ,muS] from thermodynamic quantities:
- find_param(EoS,**kwargs)
  EoS: - 'full': find T,muB,muQ,muS from e,n_B,n_Q,n_S
       - 'muB': find T,muB from e,n_B with the condition \mu_Q = \mu_S = 0
       - 'nS0': find T,muB from e,n_B with the condition <n_S> = 0 & <n_Q> = 0.4 <n_B>
  kwargs: - e: energy density in GeV.fm^-3
          - n_B: baryon density in fm^-3 
          - n_Q: baryon density in fm^-3 
          - n_S: baryon density in fm^-3 
  output: dictionnary of quantities in [GeV]: ['T','muB','muQ','muS'] for EoS='full'
          and ['T','muB'] for EoS='muB' or 'nS0'
"""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter
    )
    args = parser.parse_args()

########################################################################
def full_EoS(T,muB,muQ,muS,**kwargs):
    """
    full EoS (matching between lQCD and HRG) at a fixed T,muB,muQ,muS
    """

    if(isinstance(T,float)):

        try:
            Ttrans = kwargs['Ttrans']
            dTtransdmuB = Ttrans.derivative(nu=1)
        except:
            Ttrans = Tc_lattice # default function for transition temperature as a function of muB
            dTtransdmuB = dTcdmuB_lattice
        
        dT = 0.1*Ttrans(0.)  # matching interval

        # if T is large, EoS from lattice only
        if(T>Ttrans(muB)+3.*dT):
            result = param(T,muB,muQ,muS)
            p = result['P']
            nB = result['n_B']
            nQ = result['n_Q']
            nS = result['n_S']
            s = result['s']
            e = result['e']
        # if T is small, EoS from HRG only
        elif(T<Ttrans(muB)-3.*dT):
            result = HRG(T,muB,muQ,muS,**kwargs)
            p = result['P']
            nB = result['n_B']
            nQ = result['n_Q']
            nS = result['n_S']
            s = result['s']
            e = result['e']
        # else, matching between HRG and lattice
        else:
            # matching function, and its derivative wrt T and muB
            fmatch = np.tanh((T-Ttrans(muB))/dT)
            fmatchdT = (1.-(np.tanh((T-Ttrans(muB))/dT))**2.)/dT
            fmatchdmu = -dTtransdmuB(muB)*(1.-(np.tanh((T-Ttrans(muB))/dT))**2.)/dT

            # HRG and lattice EoS
            EoSH = HRG(T,muB,muQ,muS,**kwargs)
            EoSL = param(T,muB,muQ,muS)
            
            # Matching done as in DOI: 10.1103/PhysRevC.100.024907
            p = 0.5*(1.-fmatch)*EoSH['P']+0.5*(1.+fmatch)*EoSL['P']
            nB = 0.5*(1.-fmatch)*EoSH['n_B']+0.5*(1.+fmatch)*EoSL['n_B']+0.5*T*(EoSL['P']-EoSH['P'])*fmatchdmu
            nQ = 0.5*(1.-fmatch)*EoSH['n_Q']+0.5*(1.+fmatch)*EoSL['n_Q']
            nS = 0.5*(1.-fmatch)*EoSH['n_S']+0.5*(1.+fmatch)*EoSL['n_S']
            s = 0.5*(1.-fmatch)*EoSH['s']+0.5*(1.+fmatch)*EoSL['s']+0.5*T*(EoSL['P']-EoSH['P'])*fmatchdT
            e = s-p+(muB/T)*nB+(muQ/T)*nQ+(muS/T)*nS

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
            result = full_EoS(xT,xmuB,xmuQ,xmuS,**kwargs)
            p[i] = result['P']
            s[i] = result['s']
            nB[i] = result['n_B']
            nQ[i] = result['n_Q']
            nS[i] = result['n_S']
            e[i] = result['e']
    else:
        raise Exception('Problem with input')
    
    return {'P':p, 's':s, 'n_B':nB, 'n_Q':nQ, 'n_S':nS, 'e':e, 'I':e-3*p}

def full_EoS_nS0(T,muB,**kwargs):
    """
    full EoS (matching between lQCD and HRG) at a fixed T,muB
    <n_S> = 0
    <n_Q> = factQB*<n_B>
    """
    return EoS_nS0(full_EoS,T,muB,**kwargs)

########################################################################
def find_param(EoS,**kwargs):
    """
    Find T,muB,muQ,muS [GeV] from e [GeV/fm^3], nB, nQ, nS [/fm^3]
    using the full_EoS function.
    Three cases: - full -> find T,muB,muQ,muS
                 - muB -> find T,muB and muQ = muS = 0
                 - nS0 -> find T,muB with <n_S> = 0 & <n_Q> = 0.4 <n_B>
    """

    hbarc = 0.1973269804 # GeV.fm

    # read the input (e,n_B,n_Q,n_S) from kwargs
    e = kwargs['e']
    nB = kwargs['n_B']
    try:
        nQ = kwargs['n_Q']
        nS = kwargs['n_S']
    except:
        nQ = None
        nS = None
        if(EoS=='full'):
            raise Exception('Please input n_Q & n_S to use the full EoS')  

    # if input is a single value
    if(isinstance(e,float)):

        if(EoS=='full'):
            init_guess = [0.2,0.,0.,0.] # initial guess
            output = ['T','muB','muS','muQ']
            def system(Tmu):
                """
                Define the system to be solved
                e(T,muB,muQ,muS) = e_input
                n_B(T,muB,muQ,muS) = n_B_input
                n_Q(T,muB,muQ,muS) = n_Q_input
                n_S(T,muB,muQ,muS) = n_S_input
                """
                thermo = full_EoS(Tmu[0],Tmu[1],Tmu[2],Tmu[3],**kwargs) # this is unitless
                return [thermo['e']*Tmu[0]**4./(hbarc**3.) - e, thermo['n_B']*Tmu[0]**3./(hbarc**3.) - nB, thermo['n_Q']*Tmu[0]**3./(hbarc**3.) - nQ, thermo['n_S']*Tmu[0]**3./(hbarc**3.) - nS]
        elif(EoS=='muB'):
            init_guess = [0.2,0.] # initial guess
            output = ['T','muB']
            def system(Tmu):
                """
                Define the system to be solved
                e(T,muB,0,0) = e_input
                n_B(T,muB,0,0) = n_B_input
                """
                thermo = full_EoS(Tmu[0],Tmu[1],0.,0.,**kwargs) # this is unitless
                return [thermo['e']*Tmu[0]**4./(hbarc**3.) - e, thermo['n_B']*Tmu[0]**3./(hbarc**3.) - nB]
        elif(EoS=='nS0'):
            init_guess = [0.2,0.]
            output = ['T','muB']
            def system(Tmu):
                """
                Define the system to be solved
                e_nS0(T,muB) = e_input
                n_B_nS0(T,muB) = n_B_input
                """
                thermo = EoS_nS0(full_EoS,Tmu[0],Tmu[1],**kwargs) # this is unitless
                return [thermo['e']*Tmu[0]**4./(hbarc**3.) - e, thermo['n_B']*Tmu[0]**3./(hbarc**3.) - nB]

        solution = scipy.optimize.root(system,init_guess,method='lm').x

    # if input is an array
    elif(isinstance(e,np.ndarray) or isinstance(e,list)):

        if(EoS=='full'):
            output = ['T','muB','muS','muQ']
        elif(EoS=='muB' or EoS=='nS0'):
            output = ['T','muB']
        else:
            raise Exception(f'EoS not valid: {EoS},{type(EoS)}')

        solution = np.zeros((len(output),len(e))) # initialize the resulting arrray

        for i,_ in enumerate(e):
            if(EoS=='full'):
                result = find_param(EoS,e=e[i],n_B=nB[i],n_Q=nQ[i],n_S=nS[i])
            elif(EoS=='muB' or EoS=='nS0'):
                result = find_param(EoS,e=e[i],n_B=nB[i])
            for iq, quant in enumerate(output):
                solution[iq,i] = result[quant]
    else:
        raise Exception('Problem with input')

    return dict(zip(output,solution))

def isentropic(EoS,snB):
    """
    Calculate isentropic trajectories
    """
    if(EoS=='muB'):

        def system(muB,xT):
            """
            Define the system to be solved
            <s> = fact*<n_B> 
            """
            thermo = full_EoS(xT,muB,0.,0.)
            return thermo['s']*xT**3-snB*thermo['n_B']*xT**3

    elif(EoS=='nS0'):

        def system(Tmu,xT):
            """
            Define the system to be solved
            <s> = fact*<n_B> 
            <n_S> = 0
            <n_Q> = 0.4*<n_B>
            """
            thermo = full_EoS(xT,Tmu[0],Tmu[1],Tmu[2]) # this is unitless
            return [thermo['s']*xT**3-snB*thermo['n_B']*xT**3, thermo['n_S']*xT**3, thermo['n_Q']*xT**3-0.4*thermo['n_B']*xT**3]

    # initialize values of T
    xtemp = np.linspace(0.6,0.2,10)
    xtemp = np.append(xtemp,np.linspace(0.18,0.1,15))
    xtemp = np.append(xtemp,np.linspace(0.1,0.05,15))

    # now calculate values of muB along isentropic trajectories
    # loop over T
    if(EoS=='muB'):
        xmuB = np.zeros((len(xtemp),1))
    elif(EoS=='nS0'):
        xmuB = np.zeros((len(xtemp),3))
    for iT,xT in enumerate(xtemp):

        if(EoS=='muB'):
            xmuB[iT,0] = scipy.optimize.root(system,[0.2],args=(xT),method='lm').x
        elif(EoS=='nS0'):
            xmuB[iT,0],xmuB[iT,1],xmuB[iT,2] = scipy.optimize.root(system,[0.2,-0.08*0.2,0.03*0.2],args=(xT),method='lm').x

    # also output trajectories for mu_Q & mu_S
    if(EoS=='muB'):
        xmuQ = np.zeros_like(xtemp)
        xmuS = np.zeros_like(xtemp)
        return xmuB,xtemp,xmuQ,xmuS
    if(EoS=='nS0'):
        return xmuB[:,0],xtemp,xmuB[:,1],xmuB[:,2]
