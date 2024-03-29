import numpy as np
import scipy.integrate as integrate
from scipy.special import kn
import re
from math import pi
import os
import argparse
from iminuit import Minuit
from iminuit.cost import LeastSquares
from EoS_HRG.fit_lattice import EoS_nS0,BQS,list_chi
# import from __init__.py
from . import *

# directory where the fit_lattice_test.py is located
dir_path = os.path.dirname(os.path.realpath(__file__))

########################################################################
if __name__ == "__main__": 
    __doc__="""Construct the HRG equation of state (P/T^4, n/T^3, s/T^3, e/T^4) by calling function:
- HRG(T,muB,muQ,muS,**kwargs)
  input: temperature and chemical potentials in [GeV]
  kwargs: - species = all, mesons, baryons -> which particles to include in the HRG?
          - offshell = True, False -> integration over mass for unstable particles in the HRG?
  output: dictionnary of all quantities ['T','P','s','n_B','n_Q','n_S','e']
"""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter
    )
    args = parser.parse_args()

########################################################################
class parton:
    """
    Define properties of partons
    """
    def __init__(self,name,ID,Bcharge,Qcharge,Scharge):
        self.name = name
        self.ID = ID
        self.Bcharge = Bcharge
        self.Qcharge = Qcharge
        self.Scharge = Scharge

# attribute properties to parton objects
u = parton("u",1,1./3.,2./3.,0.)
d = parton("d",2,1./3.,-1./3.,0.)
s = parton("s",3,1./3.,-1./3.,-1.)
ubar = parton("U",-1,-1./3.,-2./3.,-0.)
dbar = parton("D",-2,-1./3.,1./3.,-0.)
sbar = parton("S",-3,-1./3.,1./3.,1.)
g = parton("g",10,0.,0.,0.)
list_partons = [u,d,s,ubar,dbar,sbar,g]

########################################################################
def from_name_to_parton(name_parton):
    """
    from string name, to parton object
    """
    for parton in list_partons:
        if(name_parton==parton.name):
            return parton

########################################################################
def Bcharge(particle):
    """
    Return Baryon charge of the particle object
    """
    if(is_baryon(particle)):
        pdg = particle.pdgid
        if(pdg>0):
            Bcharge = 1
        elif(pdg<0):
            Bcharge = -1
    else:
        Bcharge = 0
    return Bcharge
    
def Qcharge(particle):
    """
    Return electric charge of the paricle object
    """
    Qcharge = particle.charge
    return int(Qcharge)

def Scharge(particle):
    """
    Return strangeness of the particle object
    """
    pdg = particle.pdgid
    # check whether particle has a strange quark or not
    # also check particles whose PDG ID is not valid
    if(pdg.has_strange or not pdg.is_valid):
        if(is_meson(particle)):
            try:
                match = re.match('([A-Z,a-z]?)([A-Z,a-z]?)', particle.quarks)
                quark1 = from_name_to_parton(match.group(1))
                quark2 = from_name_to_parton(match.group(2))
                Scharge = quark1.Scharge + quark2.Scharge
            except:
                Scharge = 0
        elif(is_baryon(particle)):
            match = re.match('([A-Z,a-z]?)([A-Z,a-z]?)([A-Z,a-z]?)', particle.quarks)
            quark1 = from_name_to_parton(match.group(1))
            quark2 = from_name_to_parton(match.group(2))
            quark3 = from_name_to_parton(match.group(3))
            Scharge = quark1.Scharge + quark2.Scharge + quark3.Scharge
        else:
            Scharge = 0
    else:
        Scharge = 0
    return int(Scharge)
    
########################################################################
def muk(particle,muB,muQ,muS):
    """
    Return the chemical potential of the particle object
    \mu = B*_mu_B + Q*_mu_Q + S*_mu_S
    """
    muk = Bcharge(particle)*muB + Qcharge(particle)*muQ + Scharge(particle)*muS
    return muk

########################################################################
def J(particle):
    """
    spin of the particle object
    """
    xJ = particle.J
    # for particles whose PDG ID is not recognized
    if(xJ==None):
        if('N(22' in particle.name or 'Lambda(2350)' in particle.name):
            xJ = 9/2
        if('Delta(2420)' in particle.name or 'N(2600)' in particle.name):
            xJ = 11/2
    return xJ

########################################################################
def d_spin(particle):
    """
    degeneracy factor of the particle object
    d = 2*J+1
    """
    return 2*J(particle)+1

########################################################################
def BW(m,M0,gamma):
    """
    Breit-Wigner spectral function
    PHYSICAL REVIEW C 98, 034906 (2018)
    """
    BW = (2.*gamma*M0*m)/((m**2.-M0**2.)**2. + (M0*gamma)**2.)/pi
    return BW

########################################################################
def print_info(part):
    """
    Print info of a particle object
    """
    if not(isinstance(part, list)):
        print(f'{part} {part.pdgid}; mass {mass(part)} [GeV]; width {width(part)} [GeV]; J = {J(part)}; {part.quarks}; B,Q,S = {Bcharge(part)},{Qcharge(part)},{Scharge(part)}; anti = {to_antiparticle(part) if has_anti(part) else False}')
    else:
        for xpart in part:
            print_info(xpart)

########################################################################
# import mesons and baryons to include in the HRG
########################################################################
HRG_mesons = PDG_mesons[:]
HRG_baryons = PDG_baryons[:]

#print_info(HRG_mesons)
#print_info(HRG_baryons)

#for part in HRG_mesons+HRG_baryons:
#for part in to_particle(['Lambda','Lambda~','N(1875)+','N(1875)~-','N(1875)~0','N(1880)~-','N(1880)~0']):
#    print_info(part)
#    print_decays(part)
#    input("\npause")

########################################################################
# Define threshold for decays of unstable mesons and baryons
########################################################################

def threshold(list_part):
    """
    Average threshold energy for the particle
    sum of decay products weighted by the corresponding branching ratios (branch)
    """
    # calculate threshold energy as an average:
    # sum_n br_n*(mass_n1 + mass_n2 + mass_n3 ...)
    # with n being decay channel number n with branching br_n
    # and mass_n are the particle resulting from the decay
    mth_dict = {}
    for hadron in list_part:
        # initialize average threshold mass
        thres = 0.
        # see if this particle has decay info
        list_decays = part_decay(hadron)

        if(list_decays!=None):
            # loop over decay channels
            for decay in list_decays:
                # branching
                br = decay[0]
                children = decay[1]
                # sum mass of child particles
                thres += br*sum([mass(child) for child in children])
        
            mth_dict.update({hadron.name: thres})

    return mth_dict

########################################################################
# Define decay particles of unstable particles and their branching ratio
# add entry to the dictionnary by calling function threshold
# (dict, particle, [fraction in %,decay products])
########################################################################

# calculate threshold energy as an average:
# sum_n br_n*(mass_n1 + mass_n2 + mass_n3 ...)
# with n being decay channel number n with branching br_n
# and mass_n are the particle resulting from the decay

mth_mesons = threshold(HRG_mesons)
mth_baryons = threshold(HRG_baryons)
#print(mth_mesons)
#print(mth_baryons)

mth_all = mth_mesons
mth_all.update(mth_baryons)

########################################################################
# threshold mass/width for considering particle as unstable
thres_off = 0.05

########################################################################
def norm_BW():
    """
    Normalization factor for the spectral function of each particle
    """
    norm = np.zeros(len(HRG_mesons+HRG_baryons))
    for ip,part in enumerate(HRG_mesons+HRG_baryons):
        xmass = mass(part)
        xwidth = width(part)
        if(xwidth/xmass <= thres_off):
            continue

        # integration over mass runing from {mmin,mmax}
        # doi:10.1016/j.cpc.2008.08.001
        try:
            mthres = mth_all[part.name]
        except:
            # when decays are not precisely known
            mthres = xmass-2.*xwidth
            mth_all[part.name] = mthres
        mmin = max(mthres,xmass-2.*xwidth)
        mmax = xmass+2.*xwidth
                
        # normalisation of the spectral function
        # PHYSICAL REVIEW C 98, 034906 (2018)
        norm[ip] = integrate.quad(BW, mmin, mmax, args=(xmass,xwidth))[0]

    return dict(zip(HRG_mesons+HRG_baryons,norm))

# evaluate the normalization factor for the spectral function of each particle
norm = norm_BW()

#@memory.cache
########################################################################
def HRG(T,muB,muQ,muS,**kwargs):
    """
    Calculation of the HRG EoS as a function of T,muB,muQ,muS
    kwargs:
        species = all, mesons, baryons -> which particles to include?
        offshell = True, False -> integration over mass for unstable particles?
    """
    # consider integration over mass?
    try:
        offshell = kwargs['offshell']
    except:
        offshell = False # default

    # evaluate susceptibilities as well?
    try:
        eval_chi = kwargs['eval_chi']
    except:
        eval_chi = False # default

    # strangeness suppression factor?
    try:
        gammaS = kwargs['gammaS']
    except:
        gammaS = 1 # default no suppression
        
    # which particles to consider in the HRG EoS?
    try:
        species = kwargs['species']
    except:
        species = 'all' # default - consider all particles

    if(isinstance(T,float) or isinstance(T,np.float64)):
        p = 0.
        ndens = 0.
        nB = 0.
        nQ = 0.
        nS = 0.
        s = 0.
        e = 0.
        chi = np.zeros(len(list_chi))
        flag_1part = False
        if(species=='all'):
            list_part = HRG_mesons + HRG_baryons
        elif(species=='mesons'):
            list_part = HRG_mesons
        elif(species=='baryons'):
            list_part = HRG_baryons
        else:
            # if just one particle name is specified
            list_part = [to_particle(species)]
            flag_1part = True
        
        maxk = 100 # max value of k for sum over k

        for part in list_part:
            # initialize quantities for this particle, sum over k
            resultp = 0.
            resultn = 0.
            results = 0.
            resultpder = np.zeros(4)

            if(flag_1part):
                # if just one particle selected, don't count antiparticle contribution
                antip = 0
            else:
                # to account for baryon/antibaryon and meson/antimesons
                antip = float(has_anti(part))

            xmass = mass(part) # pole mass of the particle
            xwidth = width(part) # width of the particle
            dg = d_spin(part) # degeneracy factor of the particle
            xmu = muk(part,muB,muQ,muS) + np.log(gammaS**(abs(Scharge(part)))) # chemical potential of the particle
            fug = np.exp(xmu/T) # fugacity
            factB = (-1.)**(Bcharge(part)) # should be -1 for fermions, +1 for bosons

            # stable particles
            if(xwidth/xmass <= thres_off or not(offshell)):

                # precalculate different factors entering thermodynamic quantities
                factp = dg/(2.*pi**2.)*(xmass**2.)*(T**2.)
                facts = dg/(2.*pi**2.)*(xmass**2.)

                for k in range(1,maxk+1):
                    # precalculate different factors entering thermodynamic quantities
                    kn2 = kn(2,k*xmass/T)

                    resultpk0 = factp*(factB**(k+1.))/(k**2.)*kn2 # pressure at mu=0
                    resultpk = resultpk0*(fug**k+antip*fug**(-k)) # pressure finite mu

                    # evaluate if the contribution of the particle is significant or not
                    if(not eval_chi and abs(resultpk/(resultp+resultpk))<=0.005):
                        break
                    elif(eval_chi and abs(resultpk*k**4./(resultpk*k**4.+resultpder[1]))<=0.005\
                         and abs(resultpk*k**6./(resultpk*k**6.+resultpder[2]))<=0.005\
                         and abs(resultpk*k**8./(resultpk*k**8.+resultpder[3]))<=0.005):
                        break

                    resultp += resultpk
                    resultn += resultpk0*k/T*(fug**k-antip*fug**(-k)) 
                    kn1 = kn(1,k*xmass/T)
                    results += facts*factB**(k+1.)/(k**2.)*((fug**k)*(k*xmass*kn1+(4.*T-k*xmu)*kn2) \
                                         + antip*(fug**(-k))*(k*xmass*kn1+(4.*T+k*xmu)*kn2))

                    if(eval_chi):
                        # derivative of the pressure wrt mu**(2,4,6,8)
                        resultpder += resultpk*np.array([k**2.,k**4.,k**6.,k**8.])

            # unstable particles, integration over mass, weighted with Breit-Wigner spectral function
            else:
                mthres = mth_all[part.name]
                
                # integration over mass runing from {mmin,mmax}
                # doi:10.1016/j.cpc.2008.08.001
                mmin = max(mthres,xmass-2.*xwidth)
                mmax = xmass+2.*xwidth

                # normalisation of the spectral function
                # PHYSICAL REVIEW C 98, 034906 (2018)
                xnorm = norm[part]

                # now perform the integration to calculate thermodynamic quantities
                def fp(m,k):
                    return BW(m,xmass,xwidth)*(m**2.)*kn(2,k*m/T)
                def fs(m,k):
                    return BW(m,xmass,xwidth)*(m**2.)*((fug**k)*(k*m*kn(1,k*m/T)+(4.*T-k*xmu)*kn(2,k*m/T)) \
                    + antip*(fug**(-k))*(k*m*kn(1,k*m/T)+(4.*T+k*xmu)*kn(2,k*m/T)))

                # precalculate different factors entering thermodynamic quantities
                factp = dg/(2.*pi**2.)*(T**2.)/xnorm
                facts = dg/(2.*pi**2.)/xnorm

                for k in range(1,maxk+1):

                    resultpk0 = integrate.quad(fp, mmin, mmax, epsrel=0.01, args=(k))[0]*factp*(factB**(k+1.))/(k**2.)
                    resultpk = resultpk0*((fug**k)+antip*(fug**(-k)))

                    # evaluate if the contribution of the particle is significant or not
                    if(not eval_chi and abs(resultpk/(resultp+resultpk))<=0.005):
                        break
                    elif(eval_chi and abs(resultpk*k**4./(resultpk*k**4.+resultpder[1]))<=0.005\
                         and abs(resultpk*k**6./(resultpk*k**6.+resultpder[2]))<=0.005\
                         and abs(resultpk*k**8./(resultpk*k**8.+resultpder[3]))<=0.005):
                        break

                    resultp += resultpk
                    resultn += resultpk0*k/T*(fug**k-antip*fug**(-k))
                    results += facts*(factB**(k+1.))/(k**2.)*integrate.quad(fs, mmin, mmax, epsrel=0.01, args=(k))[0]
                    if(eval_chi):
                        # derivative of the pressure wrt mu**(2,4,6,8)
                        resultpder += resultpk*np.array([k**2.,k**4.,k**6.,k**8.])

            # dimensioneless quantities
            # sum over particles
            p += resultp/T**4.
            ndens += resultn/T**3.
            nB += Bcharge(part)*resultn/T**3.
            nQ += Qcharge(part)*resultn/T**3.
            nS += Scharge(part)*resultn/T**3.
            s += results/T**3.

            if(eval_chi):
                for ichi,xchi in enumerate(list_chi):
                    if(ichi==0):
                        chi[ichi] += resultp/T**4.
                        continue
                    ii = BQS[xchi]['B']
                    jj = BQS[xchi]['Q']
                    kk = BQS[xchi]['S']
                    factBQS = ((Bcharge(part))**ii)*((Qcharge(part))**jj)*((Scharge(part))**kk)
                    chi[ichi] += factBQS*resultpder[int((ii+jj+kk)/2-1)]/T**4.

        e = s-p+(muB/T)*nB+(muQ/T)*nQ+(muS/T)*nS
    
    # if the temperature input is a list
    elif(isinstance(T,np.ndarray) or isinstance(T,list)):
        p = np.zeros_like(T)
        s = np.zeros_like(T)
        ndens = np.zeros_like(T)
        nB = np.zeros_like(T)
        nQ = np.zeros_like(T)
        nS = np.zeros_like(T)
        e = np.zeros_like(T)
        chi = np.zeros((len(list_chi),len(T)))
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
            result = HRG(xT,xmuB,xmuQ,xmuS,**kwargs)
            p[i] = result['P']
            s[i] = result['s']
            ndens[i] = result['n']
            nB[i] = result['n_B']
            nQ[i] = result['n_Q']
            nS[i] = result['n_S']
            e[i] = result['e']
            chi[:,i] = result['chi']

    else:
        raise Exception('Problem with input')
    
    return {'T': T,'P':p, 's':s, 'n':ndens, 'n_B':nB, 'n_Q':nQ, 'n_S':nS, 'e':e, 'chi':chi, 'I':e-3*p}

########################################################################
def HRG_freezout(T,muB,muQ,muS,gammaS,EoS='full',**kwargs):
    """
    Calculate all particle number densities from HRG
    Includes decays as well.
    """

    # list of particles to consider
    list_particles = HRG_mesons + HRG_baryons + to_antiparticle(HRG_mesons) + to_antiparticle(HRG_baryons)
    # particles are listed from the heaviest to the lightest
    # that's for the decays
    list_particles.sort(reverse=True,key=lambda part: mass(part))

    # for tests only
    #for part in list_particles:
    #    print_decays(part)
    #input('pause')

    # consider integration over mass?
    try:
        offshell = kwargs['offshell']
    except:
        offshell = False # default

    # which particles are corrected for feed-down weak decays?
    try:
        no_feeddown = kwargs['no_feeddown']
        # if all particles are corrected for feed-down weak decays
        if(no_feeddown=='all'):
            no_feeddown = [part.name for part in list_particles]
    except:
        # default (like BES data, pions and lambda are corrected, protons are inclusive)
        no_feeddown = ['pi+','pi-','Lambda','Lambda~']#,'p','p~']

    # consider decay from unstable particles in the calculation of densities?
    try:
        freezeout_decay = kwargs['freezeout_decay']
        # in PHSD, the eta and strange baryons haven't decayed in the final output
        if(freezeout_decay=='PHSD'):
            # stable particles
            stables = ['eta']
            freezeout_decay = True
            # all particles should be corrected for feed-down weak decays
            no_feeddown = [part.name for part in list_particles]
    except:
        freezeout_decay = True # default
        stables = [] # all particles decay

    # initial densities of particles
    init_dens = {}
    # dictionnary containing the densities from weak decays
    feeddown_dens = {}
    # dictionnary containing final densities (incuding possible decays) from HRG
    final_dens = {}
    # fill dictionnaries with densities
    if(EoS=='nS0'):
        init_EoS = EoS_nS0(HRG,T,muB,gammaS=gammaS,offshell=offshell)
        # keep the values of muQ and muS for later
        muQ = init_EoS['muQ']
        muS = init_EoS['muS']
    for part in list_particles:
        part_dens = HRG(T,muB,muQ,muS,gammaS=gammaS,offshell=offshell,species=part.name)['n']
        init_dens[part.name] = part_dens
        feeddown_dens[part.name] = 0.
        final_dens[part.name] = part_dens

    # particles which contribute to feed-down weak decays
    weak_decays = ['K0','Lambda','Sigma+','Sigma-','Xi-','Xi0','Omega-']
    # add their corresponding antiparticles
    weak_decays += [to_antiparticle(to_particle(part)).name for part in weak_decays]

    # decay of unstable particles, which can be added to the density
    # loop over all the considered particles
    if(freezeout_decay):
        for parent in list_particles:
            # see if this particle can decay
            list_decays = part_decay(parent)
            #print_decays(parent)
            # if a particle is considered stable, skip
            if(parent.name in stables):
                continue
            # half of the K0 & K~0 go to K(S)0
            if(parent.name=='K0' or parent.name=='K~0'):
                list_decays = part_decay(to_particle('K(S)0'))
                # half of the K0 & K~0 go to K(S)0
                # so divide the K(S)0 branchings by 2
                fact_br = 0.5
            else:
                fact_br = 1.
                #print_decays(to_particle('K(S)0'))
            # loop over decay channels if not None
            if(list_decays!=None):
                for decay in list_decays:
                    br = fact_br*decay[0] # branching
                    children = decay[1]
                    # loop over child particles
                    for child in children:
                        # try if child particle is in dictionnary
                        try:
                            final_dens[child.name] += br*final_dens[parent.name]
                            # count feed-down weak decays for this particle
                            if(parent.name in weak_decays):
                                feeddown_dens[child.name] += br*final_dens[parent.name]
                        except:
                            pass

    # correct (substract) densities from weak decays for particles in list no_feeddown
    for part in no_feeddown:
        final_dens[part] -= feeddown_dens[part]

    # output to compare final and initial densities
    #for part in list_particles:
    #    print(part.name,init_dens[part.name],final_dens[part.name],'from decays [%]:',(final_dens[part.name]-init_dens[part.name])*100./final_dens[part.name])
    #input('pause')

    return final_dens

########################################################################
def fit_freezeout(dict_yield,**kwargs):
    """
    Extract freeze out parameters by fitting final heavy ion data (dN/dy)
    given in dict_yield. Construct ratios of different particles.
    """
    # additional plots of chi^2 (takes time)
    try:
        chi2_plot = kwargs['chi2_plot']
    except:
        chi2_plot = False # default

    # consider decay from unstable particles in the calculation of densities?
    try:
        freezeout_decay = kwargs['freezeout_decay']
    except:
        freezeout_decay = True

    # apply fit to both yields and ratios?
    try:
        method = kwargs['method']
    except:
        method = 'all'

    # consider integration over mass?
    try:
        offshell = kwargs['offshell']
    except:
        offshell = False # default

    # evaluate freeze out parameters for which EoS? full or strangeness neutrality ns0 ?
    try:
        EoS = kwargs['EoS']
    except:
        EoS = 'all' # default

    # we fit the HRG EoS to the ratios of particle yields as list_part1/list_part2
    # as in BES STAR paper: PHYSICAL REVIEW C 96, 044904 (2017)
    list_part1 = ['pi-','K-','p~','Lambda~','Xi~+','K-','p~','Lambda','Xi~+']
    list_part2 = ['pi+','K+','p','Lambda','Xi-','pi-','pi-','pi-','pi-']

    # unique list of particles (for the yields)
    list_part = ['pi+','pi-','K+','K-','p','p~','Lambda','Lambda~','Xi-','Xi~+']

    # construct yields from input
    data_yields = []
    err_yields = []
    final_part = []
    for part in list_part:
        try:
            # check if the particles are given in dict_yield
            if(dict_yield[part]!=None and dict_yield[part]>0.):
                data_yields.append(dict_yield[part])
                err_yields.append(dict_yield[part+'_err'])
                final_part.append(part)
        except:
            pass

    # construct ratios from input yields
    data_ratios = []
    err_ratios = []
    final_part1 = []
    final_part2 = []
    # loop over particles in list_part1 and list_part2
    for part1,part2 in zip(list_part1,list_part2):
        try:
            # check if the particles are given in dict_yield
            if(dict_yield[part1]!=None and dict_yield[part1]>0. and dict_yield[part2]!=None and dict_yield[part2]>0.):
                ratio = dict_yield[part1]/dict_yield[part2]
                data_ratios.append(ratio)
                err_ratios.append(abs(ratio)*np.sqrt((dict_yield[part1+'_err']/dict_yield[part1])**2.+(dict_yield[part2+'_err']/dict_yield[part2])**2.))
                final_part1.append(part1)
                final_part2.append(part2)
        except:
            pass

    def f_yields(x,T,muB,muQ,muS,gammaS,dVdy):
        """
        Calculate the particle yields for fixed T,muB,muQ,muS,gammaS,volume
        x is a dummy argument
        """
        result = np.zeros(len(final_part))
        # calculate all densities
        result_HRG = HRG_freezout(T,muB,muQ,muS,gammaS,EoS='full',**kwargs)
        # loop over particles
        for i,part in enumerate(final_part):
            yval = result_HRG[part]
            #print('part,yval=',part,yval)
            
            # if no decays, then Sigma0 should be added to Lambda
            # if decays are activated, then Sigma0 decays to Lambda
            if(not(freezeout_decay)):
                # include Sigma0 with Lambda
                if(part=='Lambda'):
                    yval += result_HRG['Sigma0']
                # include Sigma~0 with Lambda~
                elif(part=='Lambda~'):
                    yval += result_HRG['Sigma~0']
                    
            # number of particles
            result[i] = yval*T**3.*dVdy/(0.197**3.)

        # return the list of yields
        return result

    def f_yields_nS0(x,T,muB,gammaS,dVdy):
        """
        Calculate the particle yields for fixed T,muB,gammaS,volume
        x is a dummy argument
        """
        result = np.zeros(len(final_part))
        # calculate all densities
        result_HRG = HRG_freezout(T,muB,0.,0.,gammaS,EoS='nS0',**kwargs)
        # loop over particles
        for i,part in enumerate(final_part):
            yval = result_HRG[part]
            #print('part,yval=',part,yval)
            
            # if no decays, then Sigma0 should be added to Lambda
            # if decays are activated, then Sigma0 decays to Lambda
            if(not(freezeout_decay)):
                # include Sigma0 with Lambda
                if(part=='Lambda'):
                    yval += result_HRG['Sigma0']
                # include Sigma~0 with Lambda~
                elif(part=='Lambda~'):
                    yval += result_HRG['Sigma~0']
                    
            # number of particles
            result[i] = yval*T**3.*dVdy/(0.197**3.)

        # return the list of yields
        return result

    def f_ratios(x,T,muB,muQ,muS,gammaS):
        """
        Calculate the ratios of particle yields for fixed T,muB,muQ,muS,gammaS
        x is a dummy argument
        """
        result = np.zeros(len(data_ratios))
        # calculate all densities
        result_HRG = HRG_freezout(T,muB,muQ,muS,gammaS,EoS='full',**kwargs)
        # loop over different ratios
        for i,(part1,part2) in enumerate(zip(final_part1,final_part2)):
            yval1 = result_HRG[part1]
            yval2 = result_HRG[part2]
            #print('part,yval1=',part1,yval1)
            #print('part,yval2=',part2,yval2)
            
            # if no decays, then Sigma0 should be added to Lambda
            # if decays are activated, then Sigma0 decays to Lambda
            if(not(freezeout_decay)):
                # include Sigma0 with Lambda
                if(part1=='Lambda'):
                    yval1 += result_HRG['Sigma0']
                # include Sigma~0 with Lambda~
                elif(part1=='Lambda~'):
                    yval1 += result_HRG['Sigma~0']
                # include Sigma0 with Lambda
                if(part2=='Lambda'):
                    yval2 += result_HRG['Sigma0']
                # include Sigma~0 with Lambda~
                elif(part2=='Lambda~'):
                    yval2 += result_HRG['Sigma~0']
                    
            # ratio of particle1/particle2
            result[i] = yval1/yval2

        # return the list of ratios
        return result

    def f_ratios_nS0(x,T,muB,gammaS):
        """
        Calculate the ratios of particle yields for fixed T,muB,gammaS
        x is a dummy argument
        """
        result = np.zeros(len(data_ratios))
        # calculate all densities
        result_HRG = HRG_freezout(T,muB,0.,0.,gammaS,EoS='nS0',**kwargs)
        # loop over different ratios
        for i,(part1,part2) in enumerate(zip(final_part1,final_part2)):
            yval1 = result_HRG[part1]
            yval2 = result_HRG[part2]
            #print('part,yval1=',part1,yval1)
            #print('part,yval2=',part2,yval2)
            
            # if no decays, then Sigma0 should be added to Lambda
            # if decays are activated, then Sigma0 decays to Lambda
            if(not(freezeout_decay)):
                # include Sigma0 with Lambda
                if(part1=='Lambda'):
                    yval1 += result_HRG['Sigma0']
                # include Sigma~0 with Lambda~
                elif(part1=='Lambda~'):
                    yval1 += result_HRG['Sigma~0']
                # include Sigma0 with Lambda
                if(part2=='Lambda'):
                    yval2 += result_HRG['Sigma0']
                # include Sigma~0 with Lambda~
                elif(part2=='Lambda~'):
                    yval2 += result_HRG['Sigma~0']
                    
            # ratio of particle1/particle2
            result[i] = yval1/yval2

        # return the list of ratios
        return result

    # initialize the parameters
    # which parameters to be fixed?
    fix_T=False
    fix_muB=False
    fix_muQ=False
    fix_muS=False
    fix_gammaS=False
    fix_dVdy=False
    # first guesses for T,muB,muQ,muS,gammaS,dVdy
    guess = (0.150, 0.05, 0., 0.05, 1., 2000.)
    # bounds
    bounds = ((0.100, 0.200), (0, 0.6), (-0.2,0.2), (0,0.2), (0.0,1.2), (100.,10000.))

    # fit with yields
    if((EoS=='all' or EoS=='full') and (method=='all' or method=='yields')):
        # x-values, just the indexes of ratios [1,2,...,N_particles]
        xyields = np.arange(len(final_part))
        # initialize Minuit least_squares class
        least_squares = LeastSquares(xyields, data_yields, err_yields, f_yields)
        m = Minuit(least_squares, T=guess[0], muB=guess[1], muQ=guess[2], muS=guess[3], gammaS=guess[4], dVdy=guess[5],
                           limit_T=bounds[0],limit_muB=bounds[1],limit_muQ=bounds[2],limit_muS=bounds[3],limit_gammaS=bounds[4],limit_dVdy=bounds[5],
                           fix_T=fix_T,fix_muB=fix_muB,fix_muQ=fix_muQ,fix_muS=fix_muS,fix_gammaS=fix_gammaS,fix_dVdy=fix_dVdy)
        m.migrad() # finds minimum of least_squares function
        m.hesse()  # computes errors
        #print(m.params) # minuit output

        # display values and errors
        popt1 = m.values.values()
        perr1 = m.errors.values()
        print('\nfit from yields, full EoS:')
        fit_string1 = f'$T_{{ch}}={popt1[0]:.4f} \pm {perr1[0]:.4f}\ GeV$\
            \n$\mu_{{B}}={popt1[1]:.4f} \pm {perr1[1]:.4f}\ GeV$\
            \n$\mu_{{Q}}={popt1[2]:.4f} \pm {perr1[2]:.4f}\ GeV$\
            \n$\mu_{{S}}={popt1[3]:.4f} \pm {perr1[3]:.4f}\ GeV$\
            \n$\gamma_{{S}}={popt1[4]:.2f} \pm {perr1[4]:.2f}$\
            \n$dV/dy={popt1[5]:.1f} \pm {perr1[5]:.1f} \ fm^3$'
        print(fit_string1)

        thermo = HRG(popt1[0],popt1[1],popt1[2],popt1[3],gammaS=popt1[4],offshell=offshell)
        snB1 = thermo['s']/thermo['n_B']
        snB1_err = 0.
        # derivative wrt T
        thermoT1 = HRG(popt1[0]+perr1[0]/2.,popt1[1],popt1[2],popt1[3],gammaS=popt1[4],offshell=offshell)
        thermoT2 = HRG(popt1[0]-perr1[0]/2.,popt1[1],popt1[2],popt1[3],gammaS=popt1[4],offshell=offshell)
        if(thermoT1['n_B']!=0. and thermoT2['n_B']!=0.):
            snB1_err += (thermoT1['s']/thermoT1['n_B']-thermoT2['s']/thermoT2['n_B'])**2.
        # derivative wrt mu_B
        thermomuB1 = HRG(popt1[0],popt1[1]+perr1[1]/2.,popt1[2],popt1[3],gammaS=popt1[4],offshell=offshell)
        thermomuB2 = HRG(popt1[0],popt1[1]-perr1[1]/2.,popt1[2],popt1[3],gammaS=popt1[4],offshell=offshell)
        if(thermomuB1['n_B']!=0. and thermomuB2['n_B']!=0.):
            snB1_err += (thermomuB1['s']/thermomuB1['n_B']-thermomuB2['s']/thermomuB2['n_B'])**2.
        # derivative wrt mu_Q
        thermomuQ1 = HRG(popt1[0],popt1[1],popt1[2]+perr1[2]/2.,popt1[3],gammaS=popt1[4],offshell=offshell)
        thermomuQ2 = HRG(popt1[0],popt1[1],popt1[2]-perr1[2]/2.,popt1[3],gammaS=popt1[4],offshell=offshell)
        if(thermomuQ1['n_B']!=0. and thermomuQ2['n_B']!=0.):
            snB1_err += (thermomuQ1['s']/thermomuQ1['n_B']-thermomuQ2['s']/thermomuQ2['n_B'])**2.
        # derivative wrt mu_S
        thermomuS1 = HRG(popt1[0],popt1[1],popt1[2],popt1[3]+perr1[3]/2.,gammaS=popt1[4],offshell=offshell)
        thermomuS2 = HRG(popt1[0],popt1[1],popt1[2],popt1[3]-perr1[3]/2.,gammaS=popt1[4],offshell=offshell)
        if(thermomuS1['n_B']!=0. and thermomuS2['n_B']!=0.):
            snB1_err += (thermomuS1['s']/thermomuS1['n_B']-thermomuS2['s']/thermomuS2['n_B'])**2.
        # derivative wrt gamma_S
        thermogammaS1 = HRG(popt1[0],popt1[1],popt1[2],popt1[3],gammaS=popt1[4]+perr1[4]/2.,offshell=offshell)
        thermogammaS2 = HRG(popt1[0],popt1[1],popt1[2],popt1[3],gammaS=popt1[4]-perr1[4]/2.,offshell=offshell)
        if(thermogammaS1['n_B']!=0. and thermogammaS2['n_B']!=0.):
            snB1_err += (thermogammaS1['s']/thermogammaS1['n_B']-thermogammaS2['s']/thermogammaS2['n_B'])**2.
        # error as sqrt((df/dT)**2. dT+(df/dmuB)**2.+...) with f = s/n_B
        snB1_err = np.sqrt(snB1_err)
        print(f's/n_B = {snB1} \pm {snB1_err}')

        # evaluate the chi^2 values for each parameter
        if(chi2_plot):
            dT, fT = m.profile('T')
            dmuB, fmuB = m.profile('muB')
            dmuQ, fmuQ = m.profile('muQ')
            dmuS, fmuS = m.profile('muS')  
            dgammaS, fgammaS = m.profile('gammaS') 
            ddVdy, fdVdy = m.profile('dVdy') 
            output_chi21 = [[dT,fT],[dmuB,fmuB],[dmuQ,fmuQ],[dmuS,fmuS],[dgammaS,fgammaS],[ddVdy,fdVdy]]
        else:
            output_chi21 = None

        output_yields = {'fit_yields':np.array(list(zip(popt1,perr1))),\
                         'fit_string_yields':fit_string1,\
                         'result_yields':f_yields(xyields,*popt1),\
                         'data_yields':np.array(list(zip(data_yields,err_yields))),\
                         'particle_yields':list(latex(final_part)),\
                         'chi2_yields':output_chi21,\
                         'snB_yields':np.array([snB1,snB1_err])}
    else:
        output_yields = {}

    # fit with yields
    # strangeness neutrality 
    if((EoS=='all' or EoS=='nS0') and (method=='all' or method=='yields')):
        # x-values, just the indexes of ratios [1,2,...,N_particles]
        xyields = np.arange(len(final_part))
        # initialize Minuit least_squares class
        least_squares = LeastSquares(xyields, data_yields, err_yields, f_yields_nS0)
        m = Minuit(least_squares, T=guess[0], muB=guess[1], gammaS=guess[4], dVdy=guess[5],
                           limit_T=bounds[0],limit_muB=bounds[1],limit_gammaS=bounds[4],limit_dVdy=bounds[5],
                           fix_T=fix_T,fix_muB=fix_muB,fix_gammaS=fix_gammaS,fix_dVdy=fix_dVdy)
        m.migrad() # finds minimum of least_squares function
        m.hesse()  # computes errors
        #print(m.params) # minuit output

        # display values and errors
        popt1 = m.values.values()
        perr1 = m.errors.values()
        thermo = EoS_nS0(HRG,popt1[0],popt1[1],gammaS=popt1[2],offshell=offshell)

        print('\nfit from yields, nS0 EoS:')
        fit_string1 = f'$T_{{ch}}={popt1[0]:.4f} \pm {perr1[0]:.4f}\ GeV$\
            \n$\mu_{{B}}={popt1[1]:.4f} \pm {perr1[1]:.4f}\ GeV$\
            \n$\gamma_{{S}}={popt1[2]:.2f} \pm {perr1[2]:.2f}$\
            \n$dV/dy={popt1[3]:.1f} \pm {perr1[3]:.1f} \ fm^3$\
            \n$\mu_{{Q}}={thermo["muQ"]:.4f}\ GeV$\
            \n$\mu_{{S}}={thermo["muS"]:.4f}\ GeV$'
        print(fit_string1)

        snB1 = thermo['s']/thermo['n_B']
        snB1_err = 0.
        # derivative wrt T
        thermoT1 = EoS_nS0(HRG,popt1[0]+perr1[0]/2.,popt1[1],gammaS=popt1[2],offshell=offshell)
        thermoT2 = EoS_nS0(HRG,popt1[0]-perr1[0]/2.,popt1[1],gammaS=popt1[2],offshell=offshell)
        if(thermoT1['n_B']!=0. and thermoT2['n_B']!=0.):
            snB1_err += (thermoT1['s']/thermoT1['n_B']-thermoT2['s']/thermoT2['n_B'])**2.
        # derivative wrt mu_B
        thermomuB1 = EoS_nS0(HRG,popt1[0],popt1[1]+perr1[1]/2.,gammaS=popt1[2],offshell=offshell)
        thermomuB2 = EoS_nS0(HRG,popt1[0],popt1[1]-perr1[1]/2.,gammaS=popt1[2],offshell=offshell)
        if(thermomuB1['n_B']!=0. and thermomuB2['n_B']!=0.):
            snB1_err += (thermomuB1['s']/thermomuB1['n_B']-thermomuB2['s']/thermomuB2['n_B'])**2.
        # derivative wrt gamma_S
        thermogammaS1 = EoS_nS0(HRG,popt1[0],popt1[1],gammaS=popt1[2]+perr1[2]/2.,offshell=offshell)
        thermogammaS2 = EoS_nS0(HRG,popt1[0],popt1[1],gammaS=popt1[2]-perr1[2]/2.,offshell=offshell)
        if(thermogammaS1['n_B']!=0. and thermogammaS2['n_B']!=0.):
            snB1_err += (thermogammaS1['s']/thermogammaS1['n_B']-thermogammaS2['s']/thermogammaS2['n_B'])**2.
        # error as sqrt((df/dT)**2. dT+(df/dmuB)**2.+...) with f = s/n_B
        snB1_err = np.sqrt(snB1_err)
        print(f's/n_B = {snB1} \pm {snB1_err}')

        # evaluate the chi^2 values for each parameter
        if(chi2_plot):
            dT, fT = m.profile('T')
            dmuB, fmuB = m.profile('muB')
            dgammaS, fgammaS = m.profile('gammaS') 
            ddVdy, fdVdy = m.profile('dVdy') 
            output_chi21 = [[dT,fT],[dmuB,fmuB],[dgammaS,fgammaS],[ddVdy,fdVdy]]
        else:
            output_chi21 = None

        result_yields_nS0 = f_yields_nS0(xyields,*popt1)
        Tch,muB,gammaS,dVdy = popt1
        Tch_err,muB_err,gammaS_err,dVdy_err = perr1
        popt1 = np.array([Tch,muB,thermo['muQ'],thermo['muS'],gammaS,dVdy])
        perr1 = np.array([Tch_err,muB_err,0.,0.,gammaS_err,dVdy_err])

        output_yields_nS0 = {'fit_yields_nS0':np.array(list(zip(popt1,perr1))),\
                         'fit_string_yields_nS0':fit_string1,\
                         'result_yields_nS0':result_yields_nS0,\
                         'data_yields':np.array(list(zip(data_yields,err_yields))),\
                         'particle_yields':list(latex(final_part)),\
                         'chi2_yields_nS0':output_chi21,\
                         'snB_yields_nS0':np.array([snB1,snB1_err])}
    else:
        output_yields_nS0 = {}

    # fit with ratios
    if((EoS=='all' or EoS=='full') and (method=='all' or method=='ratios')):
        # x-values, just the indexes of ratios [1,2,...,N_ratios]
        xratios = np.arange(len(data_ratios))
        # initialize Minuit least_squares class
        least_squares = LeastSquares(xratios, data_ratios, err_ratios, f_ratios)
        m = Minuit(least_squares, T=guess[0], muB=guess[1], muQ=guess[2], muS=guess[3], gammaS=guess[4],
                           limit_T=bounds[0],limit_muB=bounds[1],limit_muQ=bounds[2],limit_muS=bounds[3],limit_gammaS=bounds[4],
                           fix_T=fix_T,fix_muB=fix_muB,fix_muQ=fix_muQ,fix_muS=fix_muS,fix_gammaS=fix_gammaS)
        m.migrad() # finds minimum of least_squares function
        m.hesse()  # computes errors
        #print(m.params) # minuit output

        # display values and errors
        popt2 = m.values.values()
        perr2 = m.errors.values()
        print('\nfit from ratios, full EoS:')
        fit_string2 = f'$T_{{ch}}={popt2[0]:.4f} \pm {perr2[0]:.4f}\ GeV$\
            \n$\mu_{{B}}={popt2[1]:.4f} \pm {perr2[1]:.4f}\ GeV$\
            \n$\mu_{{Q}}={popt2[2]:.4f} \pm {perr2[2]:.4f}\ GeV$\
            \n$\mu_{{S}}={popt2[3]:.4f} \pm {perr2[3]:.4f}\ GeV$\
            \n$\gamma_{{S}}={popt2[4]:.2f} \pm {perr2[4]:.2f}$'
        print(fit_string2)

        thermo = HRG(popt2[0],popt2[1],popt2[2],popt2[3],gammaS=popt2[4],offshell=offshell)
        snB2 = thermo['s']/thermo['n_B']
        snB2_err = 0.
        # derivative wrt T
        thermoT1 = HRG(popt2[0]+perr2[0]/2.,popt2[1],popt2[2],popt2[3],gammaS=popt2[4],offshell=offshell)
        thermoT2 = HRG(popt2[0]-perr2[0]/2.,popt2[1],popt2[2],popt2[3],gammaS=popt2[4],offshell=offshell)
        if(thermoT1['n_B']!=0. and thermoT2['n_B']!=0.):
            snB2_err += (thermoT1['s']/thermoT1['n_B']-thermoT2['s']/thermoT2['n_B'])**2.
        # derivative wrt mu_B
        thermomuB1 = HRG(popt2[0],popt2[1]+perr2[1]/2.,popt2[2],popt2[3],gammaS=popt2[4],offshell=offshell)
        thermomuB2 = HRG(popt2[0],popt2[1]-perr2[1]/2.,popt2[2],popt2[3],gammaS=popt2[4],offshell=offshell)
        if(thermomuB1['n_B']!=0. and thermomuB2['n_B']!=0.):
            snB2_err += (thermomuB1['s']/thermomuB1['n_B']-thermomuB2['s']/thermomuB2['n_B'])**2.
        # derivative wrt mu_Q
        thermomuQ1 = HRG(popt2[0],popt2[1],popt2[2]+perr2[2]/2.,popt2[3],gammaS=popt2[4],offshell=offshell)
        thermomuQ2 = HRG(popt2[0],popt2[1],popt2[2]-perr2[2]/2.,popt2[3],gammaS=popt2[4],offshell=offshell)
        if(thermomuQ1['n_B']!=0. and thermomuQ2['n_B']!=0.):
            snB2_err += (thermomuQ1['s']/thermomuQ1['n_B']-thermomuQ2['s']/thermomuQ2['n_B'])**2.
        # derivative wrt mu_S
        thermomuS1 = HRG(popt2[0],popt2[1],popt2[2],popt2[3]+perr2[3]/2.,gammaS=popt2[4],offshell=offshell)
        thermomuS2 = HRG(popt2[0],popt2[1],popt2[2],popt2[3]-perr2[3]/2.,gammaS=popt2[4],offshell=offshell)
        if(thermomuS1['n_B']!=0. and thermomuS2['n_B']!=0.):
            snB2_err += (thermomuS1['s']/thermomuS1['n_B']-thermomuS2['s']/thermomuS2['n_B'])**2.
        # derivative wrt gamma_S
        thermogammaS1 = HRG(popt2[0],popt2[1],popt2[2],popt2[3],gammaS=popt2[4]+perr2[4]/2.,offshell=offshell)
        thermogammaS2 = HRG(popt2[0],popt2[1],popt2[2],popt2[3],gammaS=popt2[4]-perr2[4]/2.,offshell=offshell)
        if(thermogammaS1['n_B']!=0. and thermogammaS2['n_B']!=0.):
            snB2_err += (thermogammaS1['s']/thermogammaS1['n_B']-thermogammaS2['s']/thermogammaS2['n_B'])**2.
        # error as sqrt((df/dT)**2. dT+(df/dmuB)**2.+...) with f = s/n_B
        snB2_err = np.sqrt(snB2_err)
        print(f's/n_B = {snB2} \pm {snB2_err}')

        # evaluate the chi^2 values for each parameter
        if(chi2_plot):
            dT, fT = m.profile('T')
            dmuB, fmuB = m.profile('muB')
            dmuQ, fmuQ = m.profile('muQ')
            dmuS, fmuS = m.profile('muS')  
            dgammaS, fgammaS = m.profile('gammaS') 
            output_chi22 = [[dT,fT],[dmuB,fmuB],[dmuQ,fmuQ],[dmuS,fmuS],[dgammaS,fgammaS]]
        else:
            output_chi22 = None

        output_ratios = {'fit_ratios':np.array(list(zip(popt2,perr2))),\
                         'fit_string_ratios':fit_string2,\
                         'result_ratios':f_ratios(xratios,*popt2),\
                         'data_ratios':np.array(list(zip(data_ratios,err_ratios))),\
                         'particle_ratios':list(zip(latex(final_part1),latex(final_part2))),\
                         'chi2_ratios':output_chi22,\
                         'snB_ratios':np.array([snB2,snB2_err])}
    else:
        output_ratios = {}

    # fit with ratios
    if((EoS=='all' or EoS=='nS0') and (method=='all' or method=='ratios')):
        # x-values, just the indexes of ratios [1,2,...,N_ratios]
        xratios = np.arange(len(data_ratios))
        # initialize Minuit least_squares class
        least_squares = LeastSquares(xratios, data_ratios, err_ratios, f_ratios_nS0)
        m = Minuit(least_squares, T=guess[0], muB=guess[1], gammaS=guess[4],
                           limit_T=bounds[0],limit_muB=bounds[1],limit_gammaS=bounds[4],
                           fix_T=fix_T,fix_muB=fix_muB,fix_gammaS=fix_gammaS)
        m.migrad() # finds minimum of least_squares function
        m.hesse()  # computes errors
        #print(m.params) # minuit output

        # display values and errors
        popt2 = m.values.values()
        perr2 = m.errors.values()
        thermo = EoS_nS0(HRG,popt2[0],popt2[1],gammaS=popt2[2],offshell=offshell)
        print('\nfit from ratios, nS0 EoS:')
        fit_string2 = f'$T_{{ch}}={popt2[0]:.4f} \pm {perr2[0]:.4f}\ GeV$\
            \n$\mu_{{B}}={popt2[1]:.4f} \pm {perr2[1]:.4f}\ GeV$\
            \n$\gamma_{{S}}={popt2[2]:.2f} \pm {perr2[2]:.2f}$\
            \n$\mu_{{Q}}={thermo["muQ"]:.4f}\ GeV$\
            \n$\mu_{{S}}={thermo["muS"]:.4f}\ GeV$'
        print(fit_string2)

        snB2 = thermo['s']/thermo['n_B']
        snB2_err = 0.
        # derivative wrt T
        thermoT1 = EoS_nS0(HRG,popt2[0]+perr2[0]/2.,popt2[1],gammaS=popt2[2],offshell=offshell)
        thermoT2 = EoS_nS0(HRG,popt2[0]-perr2[0]/2.,popt2[1],gammaS=popt2[2],offshell=offshell)
        if(thermoT1['n_B']!=0. and thermoT2['n_B']!=0.):
            snB2_err += (thermoT1['s']/thermoT1['n_B']-thermoT2['s']/thermoT2['n_B'])**2.
        # derivative wrt mu_B
        thermomuB1 = EoS_nS0(HRG,popt2[0],popt2[1]+perr2[1]/2.,gammaS=popt2[2],offshell=offshell)
        thermomuB2 = EoS_nS0(HRG,popt2[0],popt2[1]-perr2[1]/2.,gammaS=popt2[2],offshell=offshell)
        if(thermomuB1['n_B']!=0. and thermomuB2['n_B']!=0.):
            snB2_err += (thermomuB1['s']/thermomuB1['n_B']-thermomuB2['s']/thermomuB2['n_B'])**2.
        # derivative wrt gamma_S
        thermogammaS1 = EoS_nS0(HRG,popt2[0],popt2[1],gammaS=popt2[2]+perr2[2]/2.,offshell=offshell)
        thermogammaS2 = EoS_nS0(HRG,popt2[0],popt2[1],gammaS=popt2[2]-perr2[2]/2.,offshell=offshell)
        if(thermogammaS1['n_B']!=0. and thermogammaS2['n_B']!=0.):
            snB2_err += (thermogammaS1['s']/thermogammaS1['n_B']-thermogammaS2['s']/thermogammaS2['n_B'])**2.
        # error as sqrt((df/dT)**2. dT+(df/dmuB)**2.+...) with f = s/n_B
        snB2_err = np.sqrt(snB2_err)
        print(f's/n_B = {snB2} \pm {snB2_err}')

        # evaluate the chi^2 values for each parameter
        if(chi2_plot):
            dT, fT = m.profile('T')
            dmuB, fmuB = m.profile('muB')
            dgammaS, fgammaS = m.profile('gammaS') 
            output_chi22 = [[dT,fT],[dmuB,fmuB],[dgammaS,fgammaS]]
        else:
            output_chi22 = None

        result_ratios_nS0 = f_ratios_nS0(xratios,*popt2)
        Tch,muB,gammaS = popt2
        Tch_err,muB_err,gammaS_err = perr2
        popt2 = np.array([Tch,muB,thermo['muQ'],thermo['muS'],gammaS])
        perr2 = np.array([Tch_err,muB_err,0.,0.,gammaS_err])

        output_ratios_nS0 = {'fit_ratios_nS0':np.array(list(zip(popt2,perr2))),\
                         'fit_string_ratios_nS0':fit_string2,\
                         'result_ratios_nS0':result_ratios_nS0,\
                         'data_ratios':np.array(list(zip(data_ratios,err_ratios))),\
                         'particle_ratios':list(zip(latex(final_part1),latex(final_part2))),\
                         'chi2_ratios_nS0':output_chi22,\
                         'snB_ratios_nS0':np.array([snB2,snB2_err])}
    else:
        output_ratios_nS0 = {}

    output = {}
    output.update(output_yields)
    output.update(output_ratios)
    output.update(output_yields_nS0)
    output.update(output_ratios_nS0)

    return output
