"""
Equation of state (EoS) from a matching between lattice QCD (lQCD) and the Hadron Resonance Gas model (HRG). 
The parametrization of the lattice QCD susceptibilities is adapted from Phys. Rev. C 100, 064910, 
and the matching procedure is based on Phys. Rev. C 100, 024907.
Possibility to fit HRG parameters to final heavy-ion particle yields.
"""

__version__ = '2.3.0'

# import decay data here
import re
import os
from decaylanguage import data,DecFileParser
from lark import Lark, Transformer, Tree
from particle import Particle
import numpy as np

# directory where the fit_lattice_test.py is located
dir_path = os.path.dirname(os.path.realpath(__file__))

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
def pdgid_to_particle(pdg_id):
    """
    Convert a list of pdg id to a list of particle objects
    """
    if not(isinstance(pdg_id, list)):
        list_part = Particle.from_pdgid(pdg_id)
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
        Particle.from_pdgid(-part.pdgid)
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
            output_anti = Particle.from_pdgid(-part.pdgid)
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
# import all known hadrons from the PDG 2020
########################################################################
print('Reading particle information...')
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
    if(PDG_ID.is_hadron and mass(part)>0.):
        PDG_all.append(part)
    # store info to convert evtgen_name to part objects
    try:
        PDG_evtgen_name.update({part.evtgen_name:part})
    except:
        if(PDG_ID.is_hadron):
            no_evtgen = True
    # just keep K0 and discard K(L)0 and K(S)0
    if(PDG_ID==130 or PDG_ID==310):
        continue
    # discard f(0)(500) sigma
    if(part.name=='f(0)(500)'):
        continue
    # discard K(0)*(700)0 K(0)*(700)+ alias kappa
    if(part.name=='K(0)*(700)0' or part.name=='K(0)*(700)+'):
        continue
    # discard antiparticles because included automatically in HRG
    if(PDG_ID < 0):
        continue
    # mesons
    if(PDG_ID.is_meson):
        # just keep well established mesons
        if(part.status.value==0 or part.status.value==1):
            PDG_mesons.append(part)
        elif(part.status.value==2 and (part.name=='h(1)(1415)' or 'a(1)(1640)' in part.name or 'a(2)(1700)' in part.name)):
            PDG_mesons.append(part)
        else:
            continue
    # baryons
    elif(PDG_ID.is_baryon):
        if((part.name=='p' and PDG_ID!=2212) or (part.name=='n' and PDG_ID!=2112)):
            continue
        # just keep well established baryons
        if(part.rank>=3):
            PDG_baryons.append(part)
        else:
            continue
    #if(no_evtgen):
    #    print('  no evtgen',part.name)

# sort particle lists by mass
PDG_mesons = [PDG_mesons[ipart] for ipart in np.argsort(np.array([(mass(part),part.I,part.J,part.charge) for part in PDG_mesons],dtype=[('m', np.float64), ('I', np.float64), ('J', np.float64), ('charge', np.float64)]),order=('m','I','J','charge'))]
PDG_baryons = [PDG_baryons[ipart] for ipart in np.argsort(np.array([(mass(part),part.I,part.J,part.charge) for part in PDG_baryons],dtype=[('m', np.float64), ('I', np.float64), ('J', np.float64), ('charge', np.float64)]),order=('m','I','J','charge'))]
PDG_all = [PDG_all[ipart] for ipart in np.argsort(np.array([(mass(part),part.I,part.J,part.charge) for part in PDG_all],dtype=[('m', np.float64), ('I', np.float64), ('J', np.float64), ('charge', np.float64)]),order=('m','I','J','charge'))]

########################################################################
# read files to read decay data
########################################################################
print('Reading decay information...')
# read decay data
with data.open_text(data, 'DECAY_LHCB.DEC') as f:
    dec_file = f.read()
# read grammar to parse file
with data.open_text(data, 'decfile.lark') as f:
    grammar = f.read()

# parse decay data
l = Lark(grammar, parser='lalr', lexer='standard')
parsed_dec_file = l.parse(dec_file)

# Return a list of the actual decays defined in the .dec file
list_decays = list(parsed_dec_file.find_data('decay'))

########################################################################
def get_decay_mode_details(decay_mode_Tree):
    """
    Parse a decay mode tree and return the relevant information
    """
    list_values = [x for x in decay_mode_Tree.find_data('value')]

    bf = float(list_values[-1].children[0].value) if len(list_values)>=1 else None
    products = tuple([p.children[0].value for p in decay_mode_Tree.children if isinstance(p,Tree) and p.data=='particle'])

    return (bf, products)

########################################################################
# create dictionnary which will contain decay info
decays = {}
print('Processing decay information...')
# loop over each particle in decay list
for tree in list_decays:
    # create entry with parent particle name
    parent_evtgen = tree.children[0].children[0].value
    decays[parent_evtgen] = []

    # loop over each decay line
    for decay_mode in tree.find_data('decayline'):
        # find details about the decay
        details = get_decay_mode_details(decay_mode)
        # check that branching is not None and > 0
        if(details[0]>0.):
            # add to the dict
            decays[parent_evtgen].append(details)

########################################################################
def part_decay(part):
    """
    input is a particle object
    output is the list of decays and branching
    """
    # get evt gen name to search for the decay
    try:
        list_decays = decays[part.evtgen_name]
        antip = False
    # if not found, transform particle decay info from particle to antiparticle
    except:
        try:
            anti_part = to_antiparticle(part)
            list_decays = decays[anti_part.evtgen_name]
            antip = True
        except:
            return None

    # construct final decay list
    final_decays = []
    for decay in list_decays:
        br = decay[0] # branching
        children = decay[1] # child particles

        # convert evt gen names to particle objects
        final_decays.append((br,(to_antiparticle(PDG_evtgen_name[child],same=True) if antip else PDG_evtgen_name[child] for child in children)))
    return final_decays

########################################################################
def print_decays(part):
    if not(isinstance(part, list)):
        decays = part_decay(part)
        print(f'\n{part.name}: width {width(part)} [GeV] = {width(part)/mass(part)*100:5.2f}% mass; tau {tau(part):5.2f} [fm/c]; ctau {ctau(part)} [m]')
        if(decays!=None):
            for idecay,decay in enumerate(decays):
                br = decay[0]
                children = [child.name for child in decay[1]]
                if(idecay==0):
                    print(f'{part.name} -> {" + ".join(children)} [{br}]')
                else:
                    print(f'{" "*len(part.name)} -> {" + ".join(children)} [{br}]')
        else:
            print('None available')
    else:
        for xpart in part:
            print_decays(xpart)
