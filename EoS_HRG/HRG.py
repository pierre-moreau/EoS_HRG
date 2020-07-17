import numpy as np
import scipy.integrate as integrate
from scipy.special import kn
from particle import Particle
import re
import math
import os
import argparse

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
def to_particle(list_name):
    """
    Convert a list of particle names to a list of particle objects
    """
    if not(isinstance(list_name, list)):
        list_part = Particle.find(lambda p: p.name==list_name)
    else:
        list_part = [None]*len(list_name)
        for i,part in enumerate(list_name):
            list_part[i] = to_particle(part)
    return list_part

########################################################################
def to_antiparticle(list_part):
    """
    Convert a list of particle object to a list of their corresponding antiparticle
    """
    if not(isinstance(list_part, list)):
        list_anti = Particle.from_pdgid(-list_part.pdgid)
    else:
        list_anti = [None]*len(list_part)
        for i,part in enumerate(list_part):
            list_anti[i] = to_antiparticle(part)
    return list_anti

########################################################################
def has_anti(part):
    """
    Return True or False depending if the particle has an antiparticle or not
    """
    try:
        Particle.from_pdgid(-part.pdgid)
        result = True
    except:
        result = False
    return result

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
def mass(particle):
    """
    mass of a particle object, in GeV
    """
    return particle.mass/1000. # convert particle mass in GeV

def width(particle):
    """
    width of a particle object, in GeV
    """
    # for K0, K~0, return width=0
    if(particle.width==None):
        return 0.
    return abs(particle.width)/1000. # convert particle width in GeV

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

def threshold(mth_dict,part_name,*args):
    """
    Average threshold energy for the particle
    sum of decay products weighted by the corresponding branching ratios (branch)
    input is: particle, [branch1, part11, part12, ....], [branch2, part21, part22,...]
    """
    # calculate threshold energy as an average of
    # sum_n %_n*(part_n1 + part_n2 + part_n3 ...)
    # with n being decay channel number n with branching %n
    # and part_n are the particle resulting from the decay
    thres = 0.
    for dec in args:
        m_decay = np.array([mass(to_particle(part_name)) for part_name in dec[1:len(dec)]])
        thres += dec[0]/100.*sum(m_decay)
    return mth_dict.update({part_name: thres})

########################################################################
# Define decay particles of unstable particles and their branching ratio
# add entry to the dictionnary by calling function threshold
# (dict, particle, [fraction in %,decay products])
########################################################################
mth_mesons = {}
threshold(mth_mesons,"eta'(958)",[42.9,"pi+","pi-","eta"],[29.1,"rho(770)0"],[22.2,"pi0","pi0","eta"],[2.75,"omega(782)"])
threshold(mth_mesons,"rho(770)+", [100, "pi+","pi0"])
threshold(mth_mesons,"rho(770)0", [100, "pi0","pi0"])
threshold(mth_mesons,"omega(782)", [89.2,"pi+","pi-","pi0"], [8.28,"pi0"], [1.53,"pi+","pi-"])
threshold(mth_mesons,"phi(1020)", [48.9,"K+","K-"], [34.2,"K0","K0"], [15.32,"rho(770)+","pi-","pi+","pi-","pi0"])
threshold(mth_mesons,"K*(892)+", [100,"K+","pi0"])
threshold(mth_mesons,"K*(892)0", [100,"K0","pi0"])
threshold(mth_mesons,"a(1)(1260)+", [100,"rho(770)+","pi0"])
threshold(mth_mesons,"a(1)(1260)0", [100,"rho(770)0","pi0"])

mth_baryons = {}
threshold(mth_baryons,"N(1440)+", [65,"p","pi0"], [35,"p","pi0","pi0"])
threshold(mth_baryons,"N(1440)0", [65,"n","pi0"], [35,"n","pi0","pi0"])
threshold(mth_baryons,"N(1535)+", [45,"p","pi0"], [42,"p","eta"], [5,"p","pi0","pi0"])
threshold(mth_baryons,"N(1535)0", [45,"n","pi0"], [42,"n","eta"], [5,"n","pi0","pi0"])
threshold(mth_baryons,"Delta(1232)++", [100,"p","pi+"])
threshold(mth_baryons,"Delta(1232)+", [100,"p","pi0"])
threshold(mth_baryons,"Delta(1232)0", [100,"n","pi0"])
threshold(mth_baryons,"Delta(1232)-", [100,"n","pi-"])
threshold(mth_baryons,"Sigma(1385)+", [87,"Lambda","pi+"], [11.7,"Sigma+","pi0"])
threshold(mth_baryons,"Sigma(1385)0", [87,"Lambda","pi0"], [11.7,"Sigma0","pi0"])
threshold(mth_baryons,"Sigma(1385)-", [87,"Lambda","pi-"], [11.7,"Sigma-","pi0"])
threshold(mth_baryons,"Xi(1530)0", [100,"Xi0","pi0"])
threshold(mth_baryons,"Xi(1530)-", [100,"Xi-","pi0"])

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
        
    # which particles to consider in the HRG EoS?
    try:
        species = kwargs['species']
    except:
        species = 'all' # default - consider all particles

    if(isinstance(xT,float)):
        T = xT
        p = 0.
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

            xmu = muk(part,muB,muQ,muS) # chemical potential of the particle           
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
            nB += Bcharge(part)*resultn/T**3.   
            nQ += Qcharge(part)*resultn/T**3.  
            nS += Scharge(part)*resultn/T**3.  
            s += results/T**3.
        
        e = s-p+(muB/T)*nB+(muQ/T)*nQ+(muS/T)*nS
    
    # if the temperature input is a list
    elif(isinstance(xT,np.ndarray) or isinstance(xT,list)):
        p = np.zeros_like(xT)
        s = np.zeros_like(xT)
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
            nB[i] = result['n_B']
            nQ[i] = result['n_Q']
            nS[i] = result['n_S']
            e[i] = result['e']

    else:
        raise Exception('Problem with input')
    
    return {'T': xT,'P':p, 's':s, 'n_B':nB, 'n_Q':nQ, 'n_S':nS, 'e':e}
