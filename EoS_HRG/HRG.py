import numpy as np
import scipy.integrate as integrate
from scipy.special import kn
import re
import math
import os
import argparse
from iminuit import Minuit
from iminuit.cost import LeastSquares
# import from __init__.py
from . import *

# define Pi variable
Pi = math.pi

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
    pdg = particle.pdgid
    if(pdg.is_meson):
        Bcharge = 0.
    elif(pdg.is_baryon):
        match = re.match('([A-Z,a-z]?)([A-Z,a-z]?)([A-Z,a-z]?)', particle.quarks)
        quark1 = from_name_to_parton(match.group(1))
        quark2 = from_name_to_parton(match.group(2))
        quark3 = from_name_to_parton(match.group(3))
        Bcharge = quark1.Bcharge + quark2.Bcharge + quark3.Bcharge
    return Bcharge
    
def Qcharge(particle):
    """
    Return electric charge of the paricle object
    """
    Qcharge = particle.charge
    return Qcharge

def Scharge(particle):
    """
    Return strangeness of the particle object
    """
    pdg = particle.pdgid
    if(not(pdg.has_strange)):
        Scharge = 0.
    else:
        if(pdg.is_meson):
            try:
                match = re.match('([A-Z,a-z]?)([A-Z,a-z]?)', particle.quarks)
                quark1 = from_name_to_parton(match.group(1))
                quark2 = from_name_to_parton(match.group(2))
                Scharge = quark1.Scharge + quark2.Scharge
            except:
                Scharge = 0.
        elif(pdg.is_baryon):
            match = re.match('([A-Z,a-z]?)([A-Z,a-z]?)([A-Z,a-z]?)', particle.quarks)
            quark1 = from_name_to_parton(match.group(1))
            quark2 = from_name_to_parton(match.group(2))
            quark3 = from_name_to_parton(match.group(3))
            Scharge = quark1.Scharge + quark2.Scharge + quark3.Scharge
    return Scharge
    
########################################################################
def muk(particle,muB,muQ,muS):
    """
    Return the chemical potential of the particle object
    \mu = B*_mu_B + Q*_mu_Q + S*_mu_S
    """
    muk = Bcharge(particle)*muB + Qcharge(particle)*muQ + Scharge(particle)*muS
    return muk

########################################################################
def d_spin(particle):
    """
    degeneracy factor of the particle object
    d = 2*J+1
    """
    J = particle.J
    return 2*J+1

########################################################################
def BW(m,M0,gamma):
    """
    Breit-Wigner spectral function
    PHYSICAL REVIEW C 98, 034906 (2018)
    """
    BW = (2.*gamma*M0*m)/((m**2.-M0**2.)**2. + (M0*gamma)**2.)/Pi
    return BW

########################################################################
def print_info(part):
    """
    Print info of a particle object
    """
    if not(isinstance(part, list)):
        print(part,part.pdgid,mass(part),width(part),part.J,part.quarks,Bcharge(part),Qcharge(part),Scharge(part))
    else:
        for xpart in part:
            print_info(xpart)

########################################################################
# import mesons and baryons to include in the HRG
########################################################################
HRG_mesons = []
with open(dir_path+'/mesons_HRG.dat', 'r') as f:
    for line in f.readlines()[2:]:
        str_line = line.rstrip('\n')
        HRG_mesons.append(to_particle(str_line))

HRG_baryons = []
with open(dir_path+'/baryons_HRG.dat', 'r') as f:
    for line in f.readlines()[2:]:
        str_line = line.rstrip('\n')
        HRG_baryons.append(to_particle(str_line))

#print_info(HRG_mesons)
#print_info(HRG_baryons)

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
thres_off = 0.06

########################################################################
# check that all threshold energies for unstable particles have been given
########################################################################
for part in HRG_mesons + HRG_baryons:
    if (width(part)/mass(part) > thres_off):
        try:
            mth = mth_all[part.name]
        except:
            raise Exception(f'decay information missing for particle: {part.name}')

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
        mthres = mth_all[part.name]
        mmin = max(mthres,xmass-2.*xwidth)
        mmax = xmass+2.*xwidth
                
        # normalisation of the spectral function
        # PHYSICAL REVIEW C 98, 034906 (2018)
        norm[ip] = integrate.quad(BW, mmin, mmax, args=(xmass,xwidth))[0]

    return dict(zip(HRG_mesons+HRG_baryons,norm))

# evaluate the normalization factor for the spectral function of each particle
norm = norm_BW()

########################################################################
def HRG(xT,muB,muQ,muS,**kwargs):
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

    if(isinstance(xT,float)):
        T = xT
        p = 0.
        ndens = 0.
        nB = 0.
        nQ = 0.
        nS = 0.
        s = 0.
        e = 0.
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
        
        for part in list_part:
            resultp = 0.
            resultn = 0.
            results = 0.

            # to account for baryon/antibaryon and meson/antimesons
            antip = float(has_anti(part))
            # if just one particle selected, don't count antiparticle contribution
            if(flag_1part):
                antip = 0

            xmass = mass(part) # pole mass of the particle
            maxk = 3 # max value of k for sum over k
            # evaluate if the contribution of the particle is significant or not
            if(xmass/T>14.):
                continue
            elif(xmass/T>6.):
                maxk = 1
            elif(xmass/T>3.):
                maxk = 2

            xmu = muk(part,muB,muQ,muS) + np.log(gammaS**(abs(Scharge(part)))) # chemical potential of the particle           
            xwidth = width(part) # width of the particle
            dg = d_spin(part) # degeneracy factor of the particle

            # precalculate different factors entering thermodynamic quantities
            fug1 = np.array([np.exp(k*xmu/T) for k in range(0,maxk+1)])
            fug2 = np.array([np.exp(-k*xmu/T) for k in range(0,maxk+1)])
            kn2 = np.array([kn(2,k*xmass/T) for k in range(0,maxk+1)])
            kn1 = np.array([kn(1,k*xmass/T) for k in range(0,maxk+1)])
            factk = np.array([(-((-1.)**(Bcharge(part)+1.)))**(k+1.) for k in range(0,maxk+1)])

            # stable particles
            if(xwidth/xmass <= thres_off or not(offshell)):

                fp = dg/(2.*Pi**2.)*(xmass**2.)*(T**2.)
                fn = dg/(2.*Pi**2.)*(xmass**2.)*T
                fs = dg/(2.*Pi**2.)*(xmass**2.) 
                 
                resultp += sum([fp*factk[k]/(k**2.)*kn2[k]*(fug1[k]+antip*fug2[k]) for k in range(1,maxk+1)])
                resultn += sum([fn*factk[k]/k*kn2[k]*(fug1[k]-antip*fug2[k]) for k in range(1,maxk+1)])
                results += sum([fs*factk[k]/(k**2.)*(fug1[k]*(k*xmass*kn1[k]+(4.*T-k*xmu)*kn2[k]) \
                 + antip*fug2[k]*(k*xmass*kn1[k]+(4.*T+k*xmu)*kn2[k])) for k in range(1,maxk+1)])
            
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

                xBW = lambda m : BW(m,xmass,xwidth)
                # now perform the integration to calculate thermodynamic quantities
                fp = lambda m,k : xBW(m)*(m**2.)*kn(2,k*m/T)
                fn = lambda m,k : xBW(m)*(m**2.)*kn(2,k*m/T)
                fs = lambda m,k : xBW(m)*(m**2.)*(fug1[k]*(k*m*kn(1,k*m/T)+(4.*T-k*xmu)*kn(2,k*m/T)) \
                    + antip*fug2[k]*(k*m*kn(1,k*m/T)+(4.*T+k*xmu)*kn(2,k*m/T)))

                factp = dg/(2.*Pi**2.)*(T**2.)/xnorm
                factn = dg/(2.*Pi**2.)*T/xnorm
                facts = dg/(2.*Pi**2.)/xnorm
                
                resultp += sum([integrate.quad(fp, mmin, mmax, epsrel=0.01, args=(k))[0]*factp*factk[k]/(k**2.)*(fug1[k]+antip*fug2[k]) for k in range(1,maxk+1)])
                resultn += sum([integrate.quad(fn, mmin, mmax, epsrel=0.01, args=(k))[0]*factn*factk[k]/k*(fug1[k]-antip*fug2[k]) for k in range(1,maxk+1)])
                results += sum([integrate.quad(fs, mmin, mmax, epsrel=0.01, args=(k))[0]*facts*factk[k]/(k**2.) for k in range(1,maxk+1)])
            
            # dimensioneless quantities
            p += resultp/T**4.
            ndens += resultn/T**3.
            nB += Bcharge(part)*resultn/T**3.   
            nQ += Qcharge(part)*resultn/T**3.  
            nS += Scharge(part)*resultn/T**3.  
            s += results/T**3.     
        
        e = s-p+(muB/T)*nB+(muQ/T)*nQ+(muS/T)*nS
    
    # if the temperature input is a list
    elif(isinstance(xT,np.ndarray) or isinstance(xT,list)):
        p = np.zeros_like(xT)
        s = np.zeros_like(xT)
        ndens = np.zeros_like(xT)
        nB = np.zeros_like(xT)
        nQ = np.zeros_like(xT)
        nS = np.zeros_like(xT)
        e = np.zeros_like(xT)
        for i,T in enumerate(xT):
            # see if arrays are also given for chemical potentials
            try:
                valmuB = muB[i]
            except:
                valmuB = muB
            try:
                valmuQ = muQ[i]
            except:
                valmuQ = muQ
            try:
                valmuS = muS[i]
            except:
                valmuS = muS
            result = HRG(T,valmuB,valmuQ,valmuS,**kwargs)
            p[i] = result['P']
            s[i] = result['s']
            ndens[i] = result['n']
            nB[i] = result['n_B']
            nQ[i] = result['n_Q']
            nS[i] = result['n_S']
            e[i] = result['e']

    else:
        raise Exception('Problem with input')
    
    return {'T': xT,'P':p, 's':s, 'n':ndens, 'n_B':nB, 'n_Q':nQ, 'n_S':nS, 'e':e}

########################################################################
def HRG_freezout(T,muB,muQ,muS,gammaS,**kwargs):
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
        result_HRG = HRG_freezout(T,muB,muQ,muS,gammaS,**kwargs)
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
        result_HRG = HRG_freezout(T,muB,muQ,muS,gammaS,**kwargs)
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
    if(method=='all' or method=='yields'):
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
        print('\nfit from yields:')
        fit_string1 = f'$T_{{ch}}={popt1[0]:.3f} \pm {perr1[0]:.3f}\ GeV$\
            \n$\mu_{{B}}={popt1[1]:.3f} \pm {perr1[1]:.3f}\ GeV$\
            \n$\mu_{{Q}}={popt1[2]:.3f} \pm {perr1[2]:.3f}\ GeV$\
            \n$\mu_{{S}}={popt1[3]:.3f} \pm {perr1[3]:.3f}\ GeV$\
            \n$\gamma_{{S}}={popt1[4]:.3f} \pm {perr1[4]:.3f}$\
            \n$dV/dy={popt1[5]:.3f} \pm {perr1[5]:.3f} \ fm^3$'
        print(fit_string1)

        thermo = HRG(popt1[0],popt1[1],popt1[2],popt1[3],gammaS=popt1[4],offshell=offshell)
        snB1 = thermo['s']/thermo['n_B']
        # derivative wrt T
        thermoT1 = HRG(popt1[0]+perr1[0]/2.,popt1[1],popt1[2],popt1[3],gammaS=popt1[4],offshell=offshell)
        thermoT2 = HRG(popt1[0]-perr1[0]/2.,popt1[1],popt1[2],popt1[3],gammaS=popt1[4],offshell=offshell)
        # derivative wrt mu_B
        thermomuB1 = HRG(popt1[0],popt1[1]+perr1[1]/2.,popt1[2],popt1[3],gammaS=popt1[4],offshell=offshell)
        thermomuB2 = HRG(popt1[0],popt1[1]-perr1[1]/2.,popt1[2],popt1[3],gammaS=popt1[4],offshell=offshell)
        # derivative wrt mu_Q
        thermomuQ1 = HRG(popt1[0],popt1[1],popt1[2]+perr1[2]/2.,popt1[3],gammaS=popt1[4],offshell=offshell)
        thermomuQ2 = HRG(popt1[0],popt1[1],popt1[2]-perr1[2]/2.,popt1[3],gammaS=popt1[4],offshell=offshell)
        # derivative wrt mu_S
        thermomuS1 = HRG(popt1[0],popt1[1],popt1[2],popt1[3]+perr1[3]/2.,gammaS=popt1[4],offshell=offshell)
        thermomuS2 = HRG(popt1[0],popt1[1],popt1[2],popt1[3]-perr1[3]/2.,gammaS=popt1[4],offshell=offshell)
        # derivative wrt gamma_S
        thermogammaS1 = HRG(popt1[0],popt1[1],popt1[2],popt1[3],gammaS=popt1[4]+perr1[4]/2.,offshell=offshell)
        thermogammaS2 = HRG(popt1[0],popt1[1],popt1[2],popt1[3],gammaS=popt1[4]-perr1[4]/2.,offshell=offshell)
        # error as sqrt((df/dT)**2. dT+(df/dmuB)**2.+...) with f = s/n_B
        snB1_err = np.sqrt((thermoT1['s']/thermoT1['n_B']-thermoT2['s']/thermoT2['n_B'])**2.\
                           +(thermomuB1['s']/thermomuB1['n_B']-thermomuB2['s']/thermomuB2['n_B'])**2.\
                           +(thermomuQ1['s']/thermomuQ1['n_B']-thermomuQ2['s']/thermomuQ2['n_B'])**2.\
                           +(thermomuS1['s']/thermomuS1['n_B']-thermomuS2['s']/thermomuS2['n_B'])**2.\
                           +(thermogammaS1['s']/thermogammaS1['n_B']-thermogammaS2['s']/thermogammaS2['n_B'])**2.)
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

    # fit with ratios
    if(method=='all' or method=='ratios'):
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
        print('\nfit from ratios:')
        fit_string2 = f'$T_{{ch}}={popt2[0]:.3f} \pm {perr2[0]:.3f}\ GeV$\
            \n$\mu_{{B}}={popt2[1]:.3f} \pm {perr2[1]:.3f}\ GeV$\
            \n$\mu_{{Q}}={popt2[2]:.3f} \pm {perr2[2]:.3f}\ GeV$\
            \n$\mu_{{S}}={popt2[3]:.3f} \pm {perr2[3]:.3f}\ GeV$\
            \n$\gamma_{{S}}={popt2[4]:.3f} \pm {perr2[4]:.3f}$'
        print(fit_string2)

        thermo = HRG(popt2[0],popt2[1],popt2[2],popt2[3],gammaS=popt2[4],offshell=offshell)
        snB2 = thermo['s']/thermo['n_B']
        # derivative wrt T
        thermoT1 = HRG(popt2[0]+perr2[0]/2.,popt2[1],popt2[2],popt2[3],gammaS=popt2[4],offshell=offshell)
        thermoT2 = HRG(popt2[0]-perr2[0]/2.,popt2[1],popt2[2],popt2[3],gammaS=popt2[4],offshell=offshell)
        # derivative wrt mu_B
        thermomuB1 = HRG(popt2[0],popt2[1]+perr2[1]/2.,popt2[2],popt2[3],gammaS=popt2[4],offshell=offshell)
        thermomuB2 = HRG(popt2[0],popt2[1]-perr2[1]/2.,popt2[2],popt2[3],gammaS=popt2[4],offshell=offshell)
        # derivative wrt mu_Q
        thermomuQ1 = HRG(popt2[0],popt2[1],popt2[2]+perr2[2]/2.,popt2[3],gammaS=popt2[4],offshell=offshell)
        thermomuQ2 = HRG(popt2[0],popt2[1],popt2[2]-perr2[2]/2.,popt2[3],gammaS=popt2[4],offshell=offshell)
        # derivative wrt mu_S
        thermomuS1 = HRG(popt2[0],popt2[1],popt2[2],popt2[3]+perr2[3]/2.,gammaS=popt2[4],offshell=offshell)
        thermomuS2 = HRG(popt2[0],popt2[1],popt2[2],popt2[3]-perr2[3]/2.,gammaS=popt2[4],offshell=offshell)
        # derivative wrt gamma_S
        thermogammaS1 = HRG(popt2[0],popt2[1],popt2[2],popt2[3],gammaS=popt2[4]+perr2[4]/2.,offshell=offshell)
        thermogammaS2 = HRG(popt2[0],popt2[1],popt2[2],popt2[3],gammaS=popt2[4]-perr2[4]/2.,offshell=offshell)
        # error as sqrt((df/dT)**2. dT+(df/dmuB)**2.+...) with f = s/n_B
        snB2_err = np.sqrt((thermoT1['s']/thermoT1['n_B']-thermoT2['s']/thermoT2['n_B'])**2.\
                           +(thermomuB1['s']/thermomuB1['n_B']-thermomuB2['s']/thermomuB2['n_B'])**2.\
                           +(thermomuQ1['s']/thermomuQ1['n_B']-thermomuQ2['s']/thermomuQ2['n_B'])**2.\
                           +(thermomuS1['s']/thermomuS1['n_B']-thermomuS2['s']/thermomuS2['n_B'])**2.\
                           +(thermogammaS1['s']/thermogammaS1['n_B']-thermogammaS2['s']/thermogammaS2['n_B'])**2.)
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

    output_yields.update(output_ratios)
    return output_yields
