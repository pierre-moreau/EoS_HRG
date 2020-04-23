import numpy as np
import matplotlib.pyplot as pl
import scipy
import os
import argparse

# import from __init__.py
from . import *
# import functions to evaluate
from EoS_HRG.full_EoS import full_EoS
# for the plots, import plots of lQCD data
from EoS_HRG.test.plot_lattice import plot_lattice
from EoS_HRG.fit_lattice import lattice_data, Tc_lattice, EoS_nS0

# directory where the fit_lattice_test.py is located
dir_path = os.path.dirname(os.path.realpath(__file__))

###############################################################################
__doc__ = """
Produce plots for the HRG+lQCD EoS defined in EoS_HRG.full_EoS:
    - EoS as a function of T and muB in comparison with lQCD data from EoS_HRG.fit_lattice
    - isentropic trajectories at fixed values of s/n_B
for two different settings:
    - 'muB' refers to the EoS with the condition \mu_Q = \mu_S = 0
    - 'nS0' refers to the EoS with the condition <n_S> = 0 & <n_Q> = 0.4 <n_B>
"""
########################################################################
@timef
def main(EoS,tab):
    # get the range of lattice data in T to plot the parametrization (default values)
    lQCDdata0 = lattice_data(EoS,0.)

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--Tmin', type=float, default=lQCDdata0['T'].min(),
        help='minimum temperature [GeV]'
    )
    parser.add_argument(
        '--Tmax', type=float, default=lQCDdata0['T'].max(),
        help='maximum temperature [GeV]'
    )
    parser.add_argument(
        '--Npoints', type=int, default=50,
        help='Number of points to plot/output'
    )
    parser.add_argument(
        '--output', type=str2bool, default=False,
        help='write the HRG+lQCD results in output file'
    )
    parser.add_argument(
        '--offshell', type=str2bool, default=False,
        help='account for finite width of resonances and integrate over mass'
    )
    
    args = parser.parse_args()
    Tmin = args.Tmin
    Tmax = args.Tmax

    print(EoS)
    # quantities to plot/output
    list_quant = ['P','n_B','s','e']
    # initialize plots with lattice data
    f,ax = plot_lattice(EoS,tab)

    # initialize temperature values
    xtemp = np.linspace(Tmin,Tmax,args.Npoints)

    for muB,color in tab:
        print('    muB = ', muB, ' GeV')
        # evaluate the EoS for each value of \mu_B
        if(EoS!='nS0'):
            yval = full_EoS(xtemp,muB,0.,0.,offshell=args.offshell)
        else:
            yval = EoS_nS0(full_EoS,xtemp,muB,offshell=args.offshell)

        # plot the resulting curve
        for ipl,quant in enumerate(list_quant):
            if(quant=='n_B' and muB==0.): # don't plot for n_B when \mu_B=0
                continue
            ax[ipl].plot(xtemp,yval[quant], color=color, linewidth='2.5', label=r'$ \mu_B = $'+str(muB)+' GeV')
            ax[ipl].legend(bbox_to_anchor=(0.6, 0.5),title='HRG+lQCD', title_fontsize='25', loc='center left', borderaxespad=0., frameon=False)
        
        # output data
        if(args.output):
            with open(f"{dir_path}/fullEoS_muB{int(muB*10):02d}_{EoS}.dat",'w') as outfile:
                outfile.write(",".join(yval.keys()))
                outfile.write('\n')
                for i,_ in enumerate(yval['T']):
                    outfile.write(",".join(map(lambda x: f"{yval[x][i]:5.3E}",yval.keys())))
                    outfile.write('\n')

    # for each plot, adapt the range in (x,y) and export
    for ipl,quant in enumerate(list_quant):
        ax[ipl].set_xlim(Tmin,Tmax)
        ax[ipl].set_ylim(0.,)
        f[ipl].savefig(f"{dir_path}/fullEoS_{quant}_T_{EoS}.png")
        

########################################################################
@timef
def isentropic(EoS,tab):
    """
    Calculate isentropic trajectories and export the
    corresponding plot in the T-\mu_B plane
    """
    print(EoS)

    # inititialize plot
    f,ax = pl.subplots(figsize=(7,7))

    # plot Tc(muB) first
    ax.plot(np.linspace(0.0,0.3,10),Tc_lattice(np.linspace(0.0,0.3,10)), '--', color='k', linewidth='2.5')

    # plot lines for \mu_B/T
    for muBoT in range(1,5):
        ax.plot(np.linspace(0.0,0.6,10),np.linspace(0.0,0.6,10)/muBoT, '--', color='grey', linewidth='1', alpha=0.5)

    if(EoS!='nS0'):
        fEoS = lambda xT,xmuB : full_EoS(xT,xmuB,0.,0.)
    else:
        fEoS = lambda xT,xmuB : EoS_nS0(full_EoS,xT,xmuB)

    def system(muB,xT,snB):
        """
        Define the system to be solved
        <s> = fact*<n_B> 
        """
        thermo = fEoS(xT,muB)
        return thermo['s']-snB*thermo['n_B']

    # initialize values of T
    xtemp = np.linspace(0.5,0.2,8)
    xtemp = np.append(xtemp,np.linspace(0.18,0.1,12))
    xtemp = np.append(xtemp,np.linspace(0.09,0.01,10))

    for snB,color in tab:
        print('    s/n_B = ', snB)
        # initialize values of \mu_B
        xmuB = np.zeros_like(xtemp)
        for iT,xT in enumerate(xtemp):
            valmuB = np.array(list([imuB*0.01 for imuB in range(100)]))
            fval = np.zeros_like(valmuB)
            for imuB,xxmuB in enumerate(valmuB):
                fval[imuB] = system(xxmuB,xT,snB)
            
            try:
                xmuB[iT] = scipy.optimize.brentq(system,a=0.0001,b=0.7,args=(xT,snB),rtol=0.01)
            except:
                xmuB[iT] = None
        
        # plot the isentropic trajectories
        ax.plot(xmuB,xtemp, color=color, linewidth='2.5', label=r'$ s/n_B = $'+str(snB))
        ax.legend(bbox_to_anchor=(0.55, 0.25), title_fontsize=SMALL_SIZE, loc='center left', borderaxespad=0., frameon=False)
    
    if(EoS == 'nS0'):
        title = r'$\langle n_S \rangle = 0$ and $\langle n_Q \rangle = 0.4 \langle n_B \rangle$'
    elif(EoS == 'muB'):
        title = r'$\mu_Q = \mu_S = 0$'

    ax.set(xlabel=r'$\mu_B$ [GeV]', ylabel=r'$T$ [GeV]', title=title)
    ax.set_xlim(0.,0.6)
    ax.set_ylim(0.,0.5)
    f.savefig(f"{dir_path}/fullEoS_isentropic_{EoS}.png")

###############################################################################
if __name__ == "__main__":    

    # values of \mu_B where to test the lQCD+HRG EoS
    tab = [[0.,'r'],[0.2,'tab:orange'],[0.3,'b'],[0.4,'g']]
    main('muB',tab)
    main('nS0',tab)

    # values of s/n_B to test the lQCD+HRG EoS
    tab = [[420,'r'],[144,'tab:purple'],[94,'m'],[70,'tab:orange'],[51,'tab:olive'],[30,'g']]
    isentropic('muB',tab)
    isentropic('nS0',tab)

