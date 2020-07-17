import numpy as np
import matplotlib.pyplot as pl
import os
import argparse

# import from __init__.py
from . import *
# import the functions to test from HRG.py
from EoS_HRG.HRG import HRG
# to plot
from EoS_HRG.test.plot_lattice import plot_lattice 
from EoS_HRG.fit_lattice import lattice_data, Tc_lattice, EoS_nS0

# directory where the fit_lattice_test.py is located
dir_path = os.path.dirname(os.path.realpath(__file__))

###############################################################################
__doc__ = """Produce plots for the HRG EoS defined in EoS_HRG.HRG 
and compare with lQCD data from EoS_HRG.fit_lattice for two different settings:
    - 'muB' refers to the EoS with the condition \mu_Q = \mu_S = 0
    - 'nS0' refers to the EoS with the condition <n_S> = 0 & <n_Q> = 0.4 <n_B>
"""
parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
        '--Tmin', type=float, default=.1,
        help='minimum temperature [GeV]')
parser.add_argument(
        '--Tmax', type=float, default=.2,
        help='maximum temperature [GeV]')
parser.add_argument(
        '--Npoints', type=int, default=15,
        help='Number of points to plot')
parser.add_argument(
        '--species', choices=['all', 'baryons', 'mesons'], default='all',
        help='include only baryons, mesons, or everything in the HRG EoS')
parser.add_argument(
        '--offshell', type=str2bool, default=False,
        help='account for finite width of resonances and integrate over mass')
parser.add_argument(
        '--output', type=str2bool, default=False,
        help='write the HRG results in output file')
args = parser.parse_args()

Tmin = args.Tmin
Tmax = args.Tmax
species_out = ''
if(args.species!='all'):
    species_out = "_"+args.species
###############################################################################
@timef
def main(EoS,tab):

    print(EoS)
    # quantities to plot
    list_quant = ['P','n_B','s','e']
    # initialize plots with lattice data
    f,ax = plot_lattice(EoS,tab)

    # initialize values of T to evaluate
    xtemp = np.linspace(Tmin,Tmax,args.Npoints)
    for muB,color in tab:
        print('    muB = ', muB, ' GeV')

        if(EoS!='nS0'):
            yval = HRG(xtemp,muB,0.,0.,species=args.species,offshell=args.offshell)
        else:
            yval = EoS_nS0(HRG,xtemp,muB,species=args.species,offshell=args.offshell)

        for ipl,quant in enumerate(list_quant):
            if(quant=='n_B' and muB==0): # don't plot n_B when \mu_B = 0
                continue
            # plot HRG EoS with solid line below Tc
            cond = xtemp <= Tc_lattice(muB)
            ax[ipl].plot(xtemp[cond],yval[quant][cond], color=color, linewidth='2.5', label=r'$ \mu_B = $'+str(muB)+' GeV')
            # plot HRG EoS with dashed line above Tc
            ax[ipl].plot(xtemp,yval[quant], '--', color=color, linewidth='2.5')
            ax[ipl].legend(bbox_to_anchor=(0.05, 0.75),title='PHSD HRG', title_fontsize='25', loc='center left', borderaxespad=0., frameon=False)

        # output data
        if(args.output):
            with open(f"{dir_path}/HRG_muB{int(muB*10):02d}_{EoS}{species_out}.dat",'w') as outfile:
                outfile.write(",".join(yval.keys()))
                outfile.write('\n')
                for i,_ in enumerate(yval['T']):
                    outfile.write(",".join(map(lambda x: f"{yval[x][i]:5.3E}",yval.keys())))
                    outfile.write('\n')
    
    def getmax():
        """
        get max value of lattice data in the range T < Tmax
        """
        ymax = np.zeros(len(list_quant))
        for muB,_ in tab:
            lQCD = lattice_data(EoS,muB)
            for iq,quant in enumerate(list_quant):
                TlQCD = lQCD['T']
                dlQCD, _ = lQCD[quant]
                test = dlQCD[TlQCD < Tmax].max()
                if(test>ymax[iq]):
                    ymax[iq] = test
        return dict(zip(list_quant,ymax*1.1))

    max_val = getmax()
    # for each plot, adapt range in (x,y) and export
    for ipl,quant in enumerate(list_quant):
        ax[ipl].set_xlim(Tmin,Tmax)
        ax[ipl].set_ylim(0.,max_val[quant])
        f[ipl].savefig(f"{dir_path}/HRG_{quant}_T_{EoS}{species_out}.png")

###############################################################################
@timef
def plot_species(EoS,muB):
    """
    Plot the contribution to the HRG EoS from different individual particles
    """

    print(EoS)
    # quantities to plot
    list_quant = ['P','n_B','s','e']
    # initialize plot
    plots = np.array([pl.subplots(figsize=(10,7)) for x in np.arange(len(list_quant))])
    f = plots[:,0]
    ax = plots[:,1]

    # initialize values of T to evaluate
    xtemp = np.linspace(Tmin,Tmax,args.Npoints)
    # EoS including all particles
    if(EoS!='nS0'):
        yval = HRG(xtemp,muB,0.,0.,species='all',offshell=args.offshell)
    else:
        yval = EoS_nS0(HRG,xtemp,muB,species='all',offshell=args.offshell)
        # keep the values of muQ and muS as a function of T for later
        list_muQ = yval['muQ']
        list_muS = yval['muS']
    
    for ipl,quant in enumerate(list_quant):
        ax[ipl].plot(xtemp,yval[quant], linewidth='2.5', color='k', label='all particles')
        ax[ipl].legend(title='PHSD HRG', title_fontsize='20', loc='best', borderaxespad=0., frameon=False)

    list_part = [['pi+',r'$\pi^+$'],['pi-',r'$\pi^-$'],['K+',r'$K^+$'],['K-',r'$K^-$'],['p',r'$p$'],['n',r'$n$'],['p~',r'$\bar{p}$'],['n~',r'$\bar{n}$']]
    line = '-'
    for part,part_latex in list_part:
        print('    particle = ', part)

        if(EoS!='nS0'):
            yval = HRG(xtemp,muB,0.,0.,species=part,offshell=args.offshell)
        else:
            # use stored values of muQ and muS as a function of T
            yval = HRG(xtemp,muB,list_muQ,list_muS,species=part,offshell=args.offshell)

        for ipl,quant in enumerate(list_quant):
            ax[ipl].plot(xtemp,yval[quant], line, linewidth='2.5', label=part_latex)
            ax[ipl].legend(title='PHSD HRG', title_fontsize='20', loc='best', borderaxespad=0., frameon=False)

        if(line=='-'):
            line = '--'
        elif(line=='--'):
            line = '-'

    # for each plot, adapt range in (x,y) and export
    for ipl,quant in enumerate(list_quant):
        if(quant=='n_B' or quant=='s'):
            ylabel = '$'+quant+'/T^3$'
        else:
            ylabel = '$'+quant+'/T^4$'
        if(EoS!='nS0'):
            title = f'$\mu_B =$ {muB} GeV; $\mu_Q = \mu_S = 0$'
        else:
            title = rf'$\mu_B =$ {muB} GeV; $\langle n_S \rangle = 0$ & $\langle n_Q \rangle = 0.4 \langle n_B \rangle$'
        ax[ipl].set(xlabel='T [GeV]', ylabel=ylabel, title=title)
        ax[ipl].set_xlim(Tmin,Tmax)
        ax[ipl].set_yscale('log')
        f[ipl].savefig(f"{dir_path}/HRG_{quant}_T_muB{int(muB*10):02d}_{EoS}_species.png")

###############################################################################
if __name__ == "__main__":
    # values of \mu_B where to test the parametrization of lQCD data
    tab = [[0.,'r'],[0.2,'tab:orange'],[0.3,'b'],[0.4,'g']]

    main('muB',tab)
    main('nS0',tab)

    plot_species('muB',0)
    plot_species('muB',0.4)
    plot_species('nS0',0)
    plot_species('nS0',0.4)