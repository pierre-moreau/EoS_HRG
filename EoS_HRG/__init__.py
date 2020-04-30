"""
Equation of state (EoS) from a matching between lattice QCD (lQCD) and the Hadron Resonance Gas model (HRG). 
The reference to the lattice QCD parametrization of susceptibilities can be found in Phys. Rev. C 100, 064910, 
and to the matching procedure in Phys. Rev. C 100, 024907.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as pl
from matplotlib.pyplot import rc
import pandas as pd 
import math
import scipy

from scipy import interpolate
import scipy.integrate as integrate
from scipy.optimize import curve_fit

import time

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

__version__ = '0.0.0'