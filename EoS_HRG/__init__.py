"""
Equation of state (EoS) from a matching between lattice QCD (lQCD) and the Hadron Resonance Gas model (HRG). 
The parametrization of the lattice QCD susceptibilities is adapted from Phys. Rev. C 100, 064910, 
and the matching procedure is based on Phys. Rev. C 100, 024907.
Possibility to fit HRG parameters to final heavy-ion particle yields.
"""

__version__ = '2.1.0'

# import decay data here
import re
import os
from decaylanguage import data,DecFileParser
from lark import Lark, Transformer, Tree
from particle import Particle

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
# read files to read decay data
########################################################################

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
        final_decays.append((br,(to_antiparticle(Particle.find(evtgen_name=child),same=True) if antip else Particle.find(evtgen_name=child) for child in children)))
    return final_decays

########################################################################
def print_decays(part):
    decays = part_decay(part)
    print(f'\n{part.name}: width {width(part)} [GeV]; tau {tau(part)} [fm/c]; ctau {ctau(part)} [m]')
    if(decays!=None):
        for idecay,decay in enumerate(decays):
            br = decay[0]
            children = [child.name for child in decay[1]]
            if(idecay==0):
                print(f'{part.name} -> {" + ".join(children)} [{br}]')
            else:
                print(f'{" "*len(part.name)} -> {" + ".join(children)} [{br}]')
