"""
Equation of state (EoS) from a matching between lattice QCD (lQCD) and the Hadron Resonance Gas model (HRG). 
The parametrization of the lattice QCD susceptibilities is adapted from Phys. Rev. C 100, 064910, 
and the matching procedure is based on Phys. Rev. C 100, 024907.
Possibility to fit HRG parameters to final heavy-ion particle yields.
"""

__version__ = '2.3.1'

# import decay data here
import re
import os
from decaylanguage import data, DecFileParser
from decaylanguage.dec.dec import get_branching_fraction,get_final_state_particle_names
from particle import Particle
import numpy as np
from joblib import Memory

# directory where the fit_lattice_test.py is located
dir_path = os.path.dirname(os.path.realpath(__file__))

# joblib directory to store results
memory = Memory(location=dir_path,verbose=0)

########################################################################
def to_particle(list_name):
    """
    Convert a list of particle names to a list of particle objects
    """
    if not(isinstance(list_name, list)):
        try:
            list_part = Particle.find(lambda p: p.name==list_name)
        except:
            list_part = Particle.findall(lambda p: p.name==list_name)[0]
    else:
        list_part = [None]*len(list_name)
        for i,part in enumerate(list_name):
            list_part[i] = to_particle(part)
    return list_part

########################################################################
def to_name(list_part):
    """
    Convert a list of particle objects to a list of particle names
    """
    if not(isinstance(list_part, list)):
        list_name = list_part.name
    else:
        list_name = [None]*len(list_part)
        for i,part in enumerate(list_part):
            list_name[i] = to_name(part)
    return list_name

########################################################################
def pdgid_to_particle(pdg_id):
    """
    Convert a list of pdg id to a list of particle objects
    """
    if not(isinstance(pdg_id, list)):
        list_part = Particle.find(lambda p: p.pdgid==pdg_id)
    else:
        list_part = [None]*len(pdg_id)
        for i,pdg in enumerate(pdg_id):
            list_part[i] = pdgid_to_particle(pdg)
    return list_part

########################################################################
def has_anti(part):
    """
    Return True or False depending if the particle object has an antiparticle or not
    """
    try:
        Particle.find(lambda p: p.pdgid==-part.pdgid)
        return True
    except:
        return False

########################################################################
def to_antiparticle(part,same=False):
    """
    Convert a list of particle object to a list of their corresponding antiparticle
    """
    if not(isinstance(part, list)):
        # if particle has an antiparticle, return particle object
        if(has_anti(part)):
            output_anti = pdgid_to_particle(-part.pdgid)
            return output_anti
        # if antiparticle doesn't exist, return same particle
        # if flag same is set to true
        elif(same==True):
            return part
    else:
        output_anti = [None]*len(part)
        for i,ipart in enumerate(part):
            output_anti[i] = to_antiparticle(ipart)

        return list(filter(None, output_anti)) 

########################################################################
def mass(particle):
    """
    mass of a particle object, in GeV
    """
    try:
        xmass = particle.mass/1000. # convert particle mass in GeV
    except:
        xmass = 0.
    return xmass

def width(particle):
    """
    width of a particle object, in GeV
    """
    # for K0, K~0, return width=0
    if(particle.width==None):
        return 0.
    return abs(particle.width)/1000. # convert particle width in GeV

def tau(particle):
    """
    lifetime of a particle object, in fm/c
    """
    part_width = width(particle)
    if(part_width>0.):
        return 0.1973269804/part_width
    else:
        return float('inf')

def ctau(particle):
    """
    ctau of the particle in meters
    """
    lifetime = tau(particle) # in fm/c
    return lifetime*10.**(-15.) # now in meters with c=1

########################################################################
def latex(part_name):
    """
    input is a particle name
    output is latex name
    """
    if not(isinstance(part_name, list)):
        part = to_particle(part_name)
        latex_name = part.latex_name
    else:
        latex_name = [None]*len(part_name)
        for i,part in enumerate(part_name):
            latex_name[i] = latex(part)

    return latex_name

########################################################################
# some baryons don't have a valid PDG ID
PDG_ID_not_valid = ['N(2220)','N(2250)','N(2600)','Delta(2420)','Lambda(2350)0']

def is_baryon(part):
    PDG_ID = part.pdgid
    if(PDG_ID.is_baryon):
        return True
    if(not PDG_ID.is_valid):
        for partname in PDG_ID_not_valid:
            if(partname in part.name):
                return True
    return False

def is_meson(part):
    PDG_ID = part.pdgid
    if(PDG_ID.is_meson):
        return True
    return False

def is_hadron(part):
    return is_baryon(part) or is_meson(part)

########################################################################
# import all known hadrons from the PDG 2021
########################################################################
#@memory.cache
def list_pdg():
    print('Reading particle information...\n')
    # load particles not included in the PDG 2020 online tables
    Particle.load_table(filename=dir_path+'/data/extra_particles.csv',append=True)
    # initialize arrays
    PDG_mesons = []
    PDG_baryons = []
    PDG_all = []
    PDG_evtgen_name = {}
    for part in Particle.all():
        no_evtgen = False
        PDG_ID = part.pdgid
        # exclude charm and bottom hadrons
        if(PDG_ID.has_charm or PDG_ID.has_bottom):
            continue
        # store all particles
        if(is_hadron(part) and mass(part)>0.):
            if(part not in PDG_all): # discard if particle is already saved
                PDG_all.append(part)
        # store info to convert evtgen_name to part objects
        try:
            PDG_evtgen_name.update({part.evtgen_name:part})
        except:
            if(is_hadron(part)):
                no_evtgen = True
                PDG_evtgen_name.update({part.name:part}) # if no evtgen, just put particle name instead
        # just keep K0 and discard K(L)0 and K(S)0
        if(PDG_ID==130 or PDG_ID==310):
            continue
        # discard f(0)(500) sigma
        if(part.name=='f(0)(500)'):
            continue
        # discard K(0)*(700)0 K(0)*(700)+ alias kappa
        if(part.name=='K(0)*(700)0' or part.name=='K(0)*(700)+'):
            continue
        # discard antiparticles because automatically included in HRG
        if(PDG_ID < 0):
            continue
        # sometimes mass isn't indicated properly
        if(mass(part)<0 or part.mass==None):
            continue
        # mesons
        if(is_meson(part)):
            if(part not in PDG_mesons):
                # just keep well established mesons
                if(part.status.value==0 or part.status.value==1):
                    PDG_mesons.append(part)
                elif(part.status.value==2 and (part.name=='h(1)(1415)' or 'K(1460)' in part.name or 'K(1)(1650)' in part.name or 'a(1)(1640)' in part.name or 'a(2)(1700)' in part.name)):
                    PDG_mesons.append(part)
                else:
                    continue
        # baryons
        # some in extra_particles.csv are not recognized as baryons from their pdgid
        elif(is_baryon(part)):
            if(part not in PDG_baryons):
                if((part.name=='p' and PDG_ID!=2212) or (part.name=='n' and PDG_ID!=2112)):
                    continue
                # just keep well established baryons
                if(part.rank>=3):
                    PDG_baryons.append(part)
                else:
                    continue
        if(not PDG_ID.is_valid):
            print('not valid pdgid:', part)
        if(no_evtgen):
            print('no evtgen',part.name)
            print('------------------------')

    # sort particle lists by mass
    PDG_mesons = [PDG_mesons[ipart] for ipart in np.argsort(np.array([(mass(part),part.I,part.J,part.charge) for part in PDG_mesons],dtype=[('m', np.float64), ('I', np.float64), ('J', np.float64), ('charge', np.float64)]),order=('m','I','J','charge'))]
    PDG_baryons = [PDG_baryons[ipart] for ipart in np.argsort(np.array([(mass(part),part.I,part.J,part.charge) for part in PDG_baryons],dtype=[('m', np.float64), ('I', np.float64), ('J', np.float64), ('charge', np.float64)]),order=('m','I','J','charge'))]
    PDG_all = [PDG_all[ipart] for ipart in np.argsort(np.array([(mass(part),part.I,part.J,part.charge) for part in PDG_all],dtype=[('m', np.float64), ('I', np.float64), ('J', np.float64), ('charge', np.float64)]),order=('m','I','J','charge'))]

    return PDG_mesons,PDG_baryons,PDG_all,PDG_evtgen_name

PDG_mesons,PDG_baryons,PDG_all,PDG_evtgen_name = list_pdg()

########################################################################
# read files to read decay data
########################################################################
print('Reading decay information...')
# read decay data
path_decay_data = os.path.dirname(os.path.realpath(data.__file__))
parser = DecFileParser(f'{path_decay_data}/DECAY_LHCB.DEC')
parser.parse()

extra_parser = DecFileParser(dir_path+'/data/extra_decays.DEC')
extra_parser.parse()

########################################################################
def part_decay(part):
    """
    input is a particle object
    output is the list of decays and branching
    """
    list_decays = None
    antip = False
    for pars in [parser,extra_parser]:
        try:
            list_decays = pars._find_decay_modes(part.evtgen_name)
            break
        # if not found with evtgen or if it doesn't exist, try with the particle name
        except:
            try:
                if(part.pdgid > 0):
                    list_decays = pars._find_decay_modes(part.name)
                else:
                    anti_part = to_antiparticle(part)
                    list_decays = pars._find_decay_modes(anti_part.name)
                    antip = True
                break
            except:
                continue

    if(list_decays==None):
        return None

    # total branching ratio (should be 1 in principle)
    br_tot = 0
    for decay in list_decays:
        br_tot += get_branching_fraction(decay) # branching
    # construct final decay list
    final_decays = []
    for decay in list_decays:
        br = get_branching_fraction(decay) # branching
        children = get_final_state_particle_names(decay) # child particle names

        # convert evt gen names to particle objects
        final_decays.append((br/br_tot,(to_antiparticle(PDG_evtgen_name[child],same=True) if antip else PDG_evtgen_name[child] for child in children)))
    return final_decays

########################################################################
def print_decays(part):
    if not(isinstance(part, list)):
        decays = part_decay(part)
        print(f'\n{part.name}: width {width(part)} [GeV] = {width(part)/mass(part)*100:5.2f}% mass; tau {tau(part):5.2f} [fm/c]; ctau {ctau(part)} [m]')
        if(decays!=None):
            tot_br = 0.
            for idecay,decay in enumerate(decays):
                br = decay[0]
                tot_br += br
                children = [child.name for child in decay[1]]
                if(idecay==0):
                    print(f'{part.name} -> {" + ".join(children)} [{br}]')
                else:
                    print(f'{" "*len(part.name)} -> {" + ".join(children)} [{br}]')
            print(f'Total branching ratio = {tot_br}')
        else:
            print('///////////////////////')
            print('////None available/////')
            print('///////////////////////')
    else:
        for xpart in part:
            print_decays(xpart)
