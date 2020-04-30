# EoS_HRG

Equation of state (EoS) from a matching between lattice QCD (lQCD) and the Hadron Resonance Gas model (HRG). The reference to the lattice QCD parametrization of susceptibilities can be found in [Phys. Rev. C 100, 064910](https://journals.aps.org/prc/abstract/10.1103/PhysRevC.100.064910), and to the matching procedure in [Phys. Rev. C 100, 024907](https://journals.aps.org/prc/abstract/10.1103/PhysRevC.100.024907).

## Basic usage

This package can be installed with: ``pip install .`` along with all the necessary packages indicated in [requirements.txt](requirements.txt). 

It can also be used without installation: ``python -m EoS_HRG.<modulename> [--help]``. 

To perform tests, produce plots or data: ``python -m EoS_HRG.test.<modulename> [--help]``.

The ``-h, --help`` argument details the purpose of each file as well as its main functions.

The list of hadrons which are included in the HRG model can be modified in [baryons_HRG.dat](EoS_HRG/baryons_HRG.dat) and [mesons_HRG.dat](EoS_HRG/mesons_HRG.dat).