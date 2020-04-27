import numpy as np
import os
import scipy.optimize

from EoS_HRG.HRG import HRG
from EoS_HRG.fit_lattice import Tc_lattice, param, dTcdmuB_lattice, EoS_nS0

########################################################################
def full_EoS(xT,muB,muQ,muS,**kwargs):
    """
    full EoS (matching between lQCD and HRG) at a fixed T,muB,muQ,muS
    """

    if(isinstance(xT,float)):
        T = xT
        dT = 0.1*Tc_lattice(0.)
        
        # if T is large, EoS from lattice only
        if(T>Tc_lattice(muB)+3.*dT):
            result = param(T,muB,muQ,muS)
            p = result['P']
            nB = result['n_B']
            nQ = result['n_Q']
            nS = result['n_S']
            s = result['s']
            e = result['e']
        # if T is small, EoS from HRG only
        elif(T<Tc_lattice(muB)-3.*dT):
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
            fmatch = lambda xT,xmuB : np.tanh((xT-Tc_lattice(xmuB))/dT)
            fmatchdT = lambda xT,xmuB : (1.-(np.tanh((xT-Tc_lattice(xmuB))/dT))**2.)/dT
            fmatchdmu = lambda xT,xmuB : (-dTcdmuB_lattice(xmuB)*(1.-(np.tanh((xT-Tc_lattice(xmuB))/dT))**2.))/dT

            # HRG and lattice EoS
            EoSH = HRG(T,muB,muQ,muS,**kwargs)
            EoSL = param(T,muB,muQ,muS)
            
            # Matching done as in DOI: 10.1103/PhysRevC.100.024907
            p = 0.5*(1.-fmatch(T,muB))*EoSH['P']+0.5*(1.+fmatch(T,muB))*EoSL['P']
            nB = 0.5*(1.-fmatch(T,muB))*EoSH['n_B']+0.5*(1.+fmatch(T,muB))*EoSL['n_B']+0.5*T*(EoSL['P']-EoSH['P'])*fmatchdmu(T,muB)
            nQ = 0.5*(1.-fmatch(T,muB))*EoSH['n_Q']+0.5*(1.+fmatch(T,muB))*EoSL['n_Q']
            nS = 0.5*(1.-fmatch(T,muB))*EoSH['n_S']+0.5*(1.+fmatch(T,muB))*EoSL['n_S']
            s = 0.5*(1.-fmatch(T,muB))*EoSH['s']+0.5*(1.+fmatch(T,muB))*EoSL['s']+0.5*T*(EoSL['P']-EoSH['P'])*fmatchdT(T,muB)
            e = s-p+(muB/T)*nB+(muQ/T)*nQ+(muS/T)*nS

    elif(isinstance(xT,np.ndarray)):
        p = np.zeros_like(xT)
        s = np.zeros_like(xT)
        nB = np.zeros_like(xT)
        nQ = np.zeros_like(xT)
        nS = np.zeros_like(xT)
        e = np.zeros_like(xT)
        for i,T in enumerate(xT):
            result = full_EoS(T,muB,muQ,muS)
            p[i] = result['P']
            s[i] = result['s']
            nB[i] = result['n_B']
            nQ[i] = result['n_Q']
            nS[i] = result['n_S']
            e[i] = result['e']
    
    return {'P':p, 's':s, 'n_B':nB, 'n_Q':nQ, 'n_S':nS, 'e':e}

def full_EoS_nS0(xT,muB,**kwargs):
    """
    full EoS (matching between lQCD and HRG) at a fixed T,muB
    <n_S> = 0
    <n_Q> = factQB*<n_B>
    """
    return EoS_nS0(full_EoS,xT,muB,**kwargs)

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
    
    return dict(zip(output,solution))
