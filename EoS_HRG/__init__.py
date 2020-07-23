"""
Equation of state (EoS) from a matching between lattice QCD (lQCD) and the Hadron Resonance Gas model (HRG). 
The reference to the lattice QCD parametrization of susceptibilities can be found in Phys. Rev. C 100, 064910, 
and to the matching procedure in Phys. Rev. C 100, 024907.
"""

__version__ = '0.1.0'

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
        list_part = Particle.find(lambda p: p.name==list_name)
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
def to_antiparticle(part):
    """
    Convert a list of particle object to a list of their corresponding antiparticle
    """
    if not(isinstance(part, list)):
        if(has_anti(part)):
            output_anti = Particle.from_pdgid(-part.pdgid)
            return output_anti
        else:
            return None
    else:
        output_anti = [None]*len(part)
        for i,ipart in enumerate(part):
            output_anti[i] = to_antiparticle(ipart)

        return list(filter(None, output_anti)) 

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
    except:
        return None

    # construct final decay list
    final_decays = []
    for decay in list_decays:
        br = decay[0] # branching 
        children = decay[1] # child particles

        # convert evt gen names to particle objects
        final_decays.append((br,(Particle.find(evtgen_name=child) for child in children)))

    return final_decays

########################################################################
def print_decays(part):
    decays = part_decay(part)
    if(decays!=None):
        for idecay,decay in enumerate(decays):
            br = decay[0]
            children = [child.name for child in decay[1]]
            if(idecay==0):
                print(f'{part.name} -> {" + ".join(children)} [{br}]')
            else:
                print(f'{" "*len(part.name)} -> {" + ".join(children)} [{br}]')
