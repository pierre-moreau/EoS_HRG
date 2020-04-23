import numpy as np
import os

from EoS_HRG.HRG import HRG
from EoS_HRG.fit_lattice import Tc_lattice, param, dTcdmuB_lattice

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
