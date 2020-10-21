import numpy as np
import matplotlib.pyplot as pl
import scipy
import os
import argparse
from tqdm import trange

# import from __init__.py
from . import *
# import functions to evaluate
from EoS_HRG.full_EoS import full_EoS, full_EoS_nS0, find_param, isentropic
# for the plots, import plots of lQCD data
from EoS_HRG.test.plot_lattice import plot_lattice
from EoS_HRG.fit_lattice import lattice_data, Tc_lattice

# directory where the fit_lattice_test.py is located
dir_path = os.path.dirname(os.path.realpath(__file__))

###############################################################################
__doc__ = """Produce plots for the HRG+lQCD EoS defined in EoS_HRG.full_EoS:
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
        formatter_class=argparse.RawTextHelpFormatter
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
    dict_plots = plot_lattice(EoS,tab)

    # initialize temperature values
    xtemp = np.linspace(Tmin,Tmax,args.Npoints)

    for muB,color in tab:
        print('    muB = ', muB, ' GeV')
        # evaluate the EoS for each value of \mu_B
        if(EoS!='nS0'):
            yval = full_EoS(xtemp,muB,0.,0.,offshell=args.offshell)
        else:
            yval = full_EoS_nS0(xtemp,muB,offshell=args.offshell)

        # plot the resulting curve
        for ipl,quant in enumerate(list_quant):
            if(quant=='n_B' and muB==0.): # don't plot for n_B when \mu_B=0
                continue
            dict_plots[quant][1].plot(xtemp,yval[quant], color=color, linewidth='2.5', label=r'$ \mu_B = $'+str(muB)+' GeV')
            dict_plots[quant][1].legend(bbox_to_anchor=(0.6, 0.5),title='HRG+lQCD', title_fontsize='25', loc='center left', borderaxespad=0., frameon=False)
        
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
        dict_plots[quant][1].set_xlim(Tmin,Tmax)
        dict_plots[quant][1].set_ylim(0.,)
        dict_plots[quant][0].savefig(f"{dir_path}/fullEoS_{quant}_T_{EoS}.png")
        dict_plots[quant][0].clf()
        pl.close(dict_plots[quant][0])
        

########################################################################
@timef
def plot_isentropic(EoS,tab):
    """
    Calculate isentropic trajectories and export the
    corresponding plot in the T-\mu_B,\mu_Q,\mu_S plane
    """
    print(EoS)

    # initialize plots
    plots = np.array([pl.subplots(figsize=(7,7)) for x in np.arange(3)])
    f = plots[:,0]
    ax = plots[:,1]

    # plot Tc(muB) first
    ax[0].plot(np.linspace(0.0,0.3,10),Tc_lattice(np.linspace(0.0,0.3,10)), '--', color='k', linewidth='2.5')

    # plot lines for \mu_B/T,\mu_Q/T,\mu_S/T
    for muBoT in range(1,5):
        if(muBoT!=0):
            ax[0].plot(np.linspace(0.0,0.6,10),np.linspace(0.0,0.6,10)/muBoT, '--', color='grey', linewidth='1', alpha=0.5)

    for snB,color in tab:
        print('    s/n_B = ', snB)
        # initialize values of \mu_B
        xmuB,xtemp,xmuQ,xmuS = isentropic(EoS,snB)
        
        # plot the isentropic trajectories
        ax[0].plot(xmuB,xtemp, color=color, linewidth='2.5', label=r'$ s/n_B = $'+str(snB))
        ax[0].legend(title_fontsize=SMALL_SIZE, loc='lower right', borderaxespad=0., frameon=False)
        ax[1].plot(-xmuQ,xtemp, color=color, linewidth='2.5', label=r'$ s/n_B = $'+str(snB))
        ax[1].legend(title_fontsize=SMALL_SIZE, loc='lower right', borderaxespad=0., frameon=False)
        ax[2].plot(xmuS,xtemp, color=color, linewidth='2.5', label=r'$ s/n_B = $'+str(snB))
        ax[2].legend(title_fontsize=SMALL_SIZE, loc='lower right', borderaxespad=0., frameon=False)
    
    if(EoS == 'nS0'):
        title = r'$\langle n_S \rangle = 0$ and $\langle n_Q \rangle = 0.4 \langle n_B \rangle$'
    elif(EoS == 'muB'):
        title = r'$\mu_Q = \mu_S = 0$'

    ax[0].set(xlabel=r'$\mu_B$ [GeV]', ylabel=r'$T$ [GeV]', title=title)
    ax[0].set_xlim(0.,0.6)
    ax[0].set_ylim(0.,0.5)
    f[0].savefig(f"{dir_path}/fullEoS_isentropic_TmuB_{EoS}.png")
    f[0].clf()
    pl.close(f[0])

    if(EoS == 'nS0'):
        ax[1].set(xlabel=r'$-\mu_Q$ [GeV]', ylabel=r'$T$ [GeV]', title=title)
        ax[1].set_xlim(0.,0.05)
        ax[1].set_ylim(0.,0.5)
        f[1].savefig(f"{dir_path}/fullEoS_isentropic_TmuQ_{EoS}.png")
        f[1].clf()
        pl.close(f[1])

        ax[2].set(xlabel=r'$\mu_S$ [GeV]', ylabel=r'$T$ [GeV]', title=title)
        ax[2].set_xlim(0.,0.2)
        ax[2].set_ylim(0.,0.5)
        f[2].savefig(f"{dir_path}/fullEoS_isentropic_TmuS_{EoS}.png")
        f[2].clf()
        pl.close(f[2])

@timef
def test_find(EoS):
    """
    test accuracy of the find_param function in EoS_HRG.full_EoS
    """

    hbarc = 0.1973269804 # GeV.fm   

    print(EoS)

    if(EoS=='full'):
        Nunk = 4
        fun = lambda T,muB,muQ,muS : full_EoS(T,muB,muQ,muS)
    elif(EoS=='muB'):
        Nunk = 2
        fun = lambda T,muB,mQ,muS : full_EoS(T,muB,0.,0.)
    elif(EoS=='nS0'):
        Nunk = 2
        fun = lambda T,muB,mQ,muS : full_EoS_nS0(T,muB)

    xtrue = np.zeros(Nunk) # record when a certain accuracy is achieved
    for _ in trange(100):

        # first randomly pick T,muB,muQ,muS
        T = np.random.uniform(0.05,0.8)
        muB = np.random.uniform(-T*4.,T*4.)
        muQ = np.random.uniform(-0.25,0.25)
        muS = np.random.uniform(-0.25,0.25)
        TmuB = [T,muB,muQ,muS]
        #print(TmuB)

        # evaluate thermo quantities
        thermo = fun(T,muB,muQ,muS) # this is unitless
        e = thermo['e']*T**4./(hbarc**3.)
        nB = thermo['n_B']*T**3./(hbarc**3.)
        try:
            nQ = thermo['n_Q']*T**3./(hbarc**3.)
            nS = thermo['n_S']*T**3./(hbarc**3.)
        except:
            nQ = None
            nS = None

        # solve the system to find T,muB,muQ,muS
        sol = find_param(EoS,e=e,n_B=nB,n_Q=nQ,n_S=nS)

       # print(sol)

        # test accuracy of the find_param function
        # -> compare the found T,muB,muQ,muS to their exact values
        # 5% accuracy gives a positive (True) result
        is_close = np.isclose(list(sol.values()),TmuB[:Nunk],rtol=0.05)
        for i in range(Nunk):
            if(is_close[i]):
                xtrue[i] += 1.

    print('5% accuracy in %:', dict(zip(list(sol.keys())[:Nunk],xtrue)))

###############################################################################
if __name__ == "__main__":    

    # values of \mu_B where to test the lQCD+HRG EoS
    tab = [[0.,'r'],[0.2,'tab:orange'],[0.3,'b'],[0.4,'g']]
    main('muB',tab)
    main('nS0',tab)

    # values of s/n_B to test the lQCD+HRG EoS (takes a long time)
    tab = [[420,'r'],[144,'tab:purple'],[94,'m'],[70,'tab:orange'],[51,'tab:olive'],[30,'g']]
    plot_isentropic('muB',tab)
    plot_isentropic('nS0',tab)
    
    # test the accuracy of the find_param function in EoS_HRG.full_EoS
    test_find('full')
    test_find('muB')
    # test_find('nS0') # takes a long time
