import matplotlib.pyplot as pl
import os
import numpy as np
import argparse

# import from __init__.py
from . import *
# import the functions to test from fit_lattice.py
from EoS_HRG.fit_lattice import lattice_data, param, EoS_nS0

# directory where the fit_lattice_test.py is located
dir_path = os.path.dirname(os.path.realpath(__file__))

###############################################################################
def plot_lattice(EoS,tab):
    """
    Produce plots with lQCD data of each quantity = 'P', 'n_B', 's' or 'e'
    as a function of T [GeV] for the values of muB [GeV] indicated in tab
    """
    list_quant = ['P','n_B','s','e']

    # title of the plot
    if(EoS == 'nS0'):
        title = r'$\langle n_S \rangle = 0$ and $\langle n_Q \rangle = 0.4 \langle n_B \rangle$'
    elif(EoS == 'muB'):
        title = r'$\mu_Q = \mu_S = 0$'

    # initialize plots
    plots = np.array([pl.subplots(figsize=(10,7)) for x in np.arange(len(list_quant))])
    f = plots[:,0]
    ax = plots[:,1]
    for i,[muB,color] in enumerate(tab):
        # evaluate lattice data for the value of \mu_B
        lQCDdata = lattice_data(EoS,muB)

        # for each quantity, plot
        for iq,quant in enumerate(list_quant):
            if(quant=='n_B' and muB==0.): continue
            data, err = lQCDdata[quant]
            latticeplot, = ax[iq].plot(lQCDdata['T'],data, 'o', color=color, ms='8', mfc='none')
            ax[iq].errorbar(lQCDdata['T'], data, yerr=err, xerr=None, fmt='none', color=color)
            
            if((quant!='n_B' and i == 0) or (quant=='n_B' and i == 1)):
                first_legend = ax[iq].legend([latticeplot],['lQCD'], bbox_to_anchor=(0.975, 0.075), loc='center right', borderaxespad=0., frameon=False)
                f[iq].gca().add_artist(first_legend)

    dict_plots = {}
    for iq,quant in enumerate(list_quant):
        if(quant=='n_B' or quant=='s'):
            ylabel = '$'+quant+'/T^3$'
        else:
            ylabel = '$'+quant+'/T^4$'
        ax[iq].set(xlabel='T [GeV]', ylabel=ylabel, title=title)
        # return dict of plots
        dict_plots.update({quant:[f[iq],ax[iq]]})

    return dict_plots

###############################################################################
def plot_lattice_all(EoS,muB):
    """
    Produce a plot with lQCD data of all quantities (P,n_B,s,e) for a fixed muB [GeV]
    as a function of temperature T [GeV]
    """

    # evaluate lattice data at a fixed muB
    lQCDdata = lattice_data(EoS,muB)

    f,ax = pl.subplots(figsize=(10,7))
    for  quant,color in [["P",'r'],["n_B",'tab:orange'],["e",'b'],["s",'g']]:

        data, err = lQCDdata[quant]
        ax.plot(lQCDdata['T'], data, 'o', color=color, ms='8', mfc='none')
        ax.errorbar(lQCDdata['T'], data, yerr=err, xerr=None, fmt='none', color=color)

        if(EoS=='muB'):
            paramdata = param(lQCDdata['T'],muB,0.,0.)
        elif(EoS=='nS0'):
            paramdata = EoS_nS0(param,lQCDdata['T'],muB)

        ax.plot(lQCDdata['T'],paramdata[quant],'--', ms='8', color=color)

    # check min/max of T and s/T^3 in the plot
    xmin = lQCDdata['T'].min()
    xmax = lQCDdata['T'].max()
    smax = lQCDdata['s'][0].max()

    for quant,color,dec in [["P",'r',0.],["n_B",'tab:orange',0.15],["e",'b',0.3],["s",'g',0.45]]:
        if(quant=='n_B' or quant=='s'):
            ylabel = '$'+quant+'/T^3$'
        else:
            ylabel = '$'+quant+'/T^4$'
        pl.text(xmin+(0.4+dec)*(xmax-xmin),0.45*smax, ylabel, color=color, fontsize=MEDIUM_SIZE)

    # indicate value of muB on the plot
    pl.text(xmin, 0.9*smax, r'$\mu_B = '+str(muB)+'$ GeV', fontsize=MEDIUM_SIZE)

    if(EoS == 'nS0'):
        title = r'$\langle n_S \rangle = 0$ and $\langle n_Q \rangle = 0.4 \langle n_B \rangle$'
    elif(EoS == 'muB'):
        title = r'EoS: $\mu_Q = \mu_S = 0$'
    ax.set(xlabel='T [GeV]', title=title)
    return f,ax

###############################################################################
__doc__ = """Produce plots to compare lQCD data width their parametrization from EoS_HRG.fit_lattice
as a function of T and muB for two different settings:
    - 'muB' refers to the EoS with the condition \mu_Q = \mu_S = 0
    - 'nS0' refers to the EoS with the condition <n_S> = 0 & <n_Q> = 0.4 <n_B>
"""
###############################################################################
@timef
def main(EoS,tab):
    # get the range of lattice data in T to plot the parametrization (default values)
    lQCDdata0 = lattice_data(EoS,0.)['T']

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--Tmin', type=float, default=lQCDdata0.min(),
        help='minimum temperature [GeV]'
    )
    parser.add_argument(
        '--Tmax', type=float, default=lQCDdata0.max(),
        help='maximum temperature [GeV]'
    )
    parser.add_argument(
        '--Npoints', type=int, default=25,
        help='Number of points to plot/output'
    )
    parser.add_argument(
        '--output', type=str2bool, default=False,
        help='write the HRG results in output file'
    )
    args = parser.parse_args()

    # min/max for the temperature
    Tmin = args.Tmin
    Tmax = args.Tmax

    print(EoS)
    # quantities to plot
    list_quant = ["P","n_B","s","e"]

    # initialize plots for each quantity with lattice data
    dict_plots = plot_lattice(EoS,tab)

    # get the range of lattice data in T to plot the parametrization
    xval = np.linspace(Tmin,Tmax,args.Npoints)

    for j,[muB,color] in enumerate(tab):
        # get the parametrization of lQCD data for each \mu_B
        if(EoS!='nS0'):
            paramdata = param(xval,muB,0.,0.)
        else:
            paramdata = EoS_nS0(param,xval,muB)

        # now plot parametrization for each muB and each quantity
        for i,quant in enumerate(list_quant):
            if(quant=='n_B' and j == 0): # don't plot n_B for \mu_B = 0
                continue
            dict_plots[quant][1].plot(xval,paramdata[quant],'--', ms='8', color=color, label=r'$ \mu_B = $'+str(muB)+' GeV')
            dict_plots[quant][1].legend(bbox_to_anchor=(0.6, 0.5),title='Parametrization', title_fontsize='25', loc='center left', borderaxespad=0., frameon=False)

            # when last value of \mu_B is reached, export plot
            if(j==len(tab)-1):
                dict_plots[quant][0].savefig(f"{dir_path}/lQCD_{EoS}_{quant}_T.png")
                dict_plots[quant][0].clf()
                pl.close(dict_plots[quant][0])

        # output data for each \mu_B
        if(args.output):
            with open(f"{dir_path}/lQCD_muB{int(muB*10):02d}_{EoS}.dat",'w') as outfile:
                outfile.write(",".join(paramdata.keys()))
                outfile.write('\n')
                for i,_ in enumerate(paramdata['T']):
                    outfile.write(",".join(map(lambda x: f"{paramdata[x][i]:5.3E}",paramdata.keys())))
                    outfile.write('\n')


        # plot for P/T^4, nB/T^3, s/T^3, e/T^4 for # muB
        fall,_ = plot_lattice_all(EoS,muB)
        fall.savefig(f"{dir_path}/lQCD_{EoS}_all_muB{int(10.*muB):02d}_T.png")
        fall.clf()
        pl.close(fall)
        print('  plot \mu_B = '+str(muB)+' GeV')

###############################################################################
if __name__ == "__main__":
    # values of \mu_B where to test the parametrization of lQCD data
    tab = [[0.,'r'],[0.2,'tab:orange'],[0.3,'b'],[0.4,'g']]

    main('muB',tab)
    main('nS0',tab)
    