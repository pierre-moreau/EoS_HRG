import numpy as np
import matplotlib.pyplot as pl
import os
import argparse

# import from __init__.py
from . import *
from .. import *
# import the functions to test from HRG.py
from EoS_HRG.HRG import HRG,fit_freezeout
# to plot
from EoS_HRG.test.plot_lattice import plot_lattice 
from EoS_HRG.fit_lattice import lattice_data, Tc_lattice, EoS_nS0

# directory where the fit_lattice_test.py is located
dir_path = os.path.dirname(os.path.realpath(__file__))

###############################################################################
@timef
def main(EoS,tab):

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

    def plot_species(EoS,muB):
        """
        Plot the contribution to the HRG EoS from different individual particles
        """
        # quantities to plot
        list_quant = ['n']#,'P','n_B','s','e']
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

        list_part = ['pi+','pi-','K+','K-','p','n','p~','n~']
        line = '-'
        for part in list_part:
            if(EoS!='nS0'):
                yval = HRG(xtemp,muB,0.,0.,species=part,offshell=args.offshell)
            else:
                # use stored values of muQ and muS as a function of T
                yval = HRG(xtemp,muB,list_muQ,list_muS,species=part,offshell=args.offshell)

            for ipl,quant in enumerate(list_quant):
                ax[ipl].plot(xtemp,yval[quant], line, linewidth='2.5', label=r'$'+latex(part)+'$')
                ax[ipl].legend(title='PHSD HRG', title_fontsize='20', loc='best', borderaxespad=0., frameon=False)

            if(line=='-'):
                line = '--'
            elif(line=='--'):
                line = '-'

        # for each plot, adapt range in (x,y) and export
        for ipl,quant in enumerate(list_quant):
            if(quant=='n_B' or quant=='s' or quant=='n'):
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

    # call function plot_species here
    for muB,_ in tab:
        print('    muB = ', muB, ' GeV')
        plot_species(EoS,muB)

###############################################################################
@timef
def plot_freezeout(dict_yield,**kwargs):
    """
    Plot the fit to particle yields from HRG.fit_freezeout
    """
    # title for plot
    try:
        title = kwargs['title']
    except:
        title = None # default

    # name for the output plot
    try:
        plot_file_name = kwargs['plot_file_name']
    except:
        plot_file_name = f"{dir_path}/HRG_freezeout"

    # additional plots of chi^2 (takes time)
    try:
        chi2_plot = kwargs['chi2_plot']
    except:
        chi2_plot = False # default

    # apply fit to both yields and ratios?
    try:
        method = kwargs['method']
    except:
        method = 'all'

    def make_plot(xdata,ydata,fit_data,fit_string,plot_type):
        """
        Function to plot the freeze out fits
        """
        # initialize plot
        f = pl.figure(figsize=(10,10))
        gs = f.add_gridspec(2, 1, hspace=0, height_ratios=[2,1])

        # plot the fits
        ax1 = f.add_subplot(gs[0])
        ax1.scatter(xdata,fit_data,marker="_",s=1000,color='b', label='fit HRG')
        # plot exp data
        ax1.errorbar(xdata,ydata[:,0],yerr=ydata[:,1],marker='o',linestyle='None',color='r', label='data')
        # legend
        ax1.legend(loc='upper right', borderaxespad=0., frameon=False)
        # text with parameters
        ax1.annotate(fit_string, xy=(0.05, 0.05), xycoords='axes fraction', fontsize='15')
        # other settings
        ax1.set(title=title)
        ax1.set_yscale('log')
        ax1.yaxis.grid(True)
        ax1.set_xticks(range(len(xdata)))

        # difference between data and fit
        ax2 = f.add_subplot(gs[1],sharex=ax1)
        ax2.scatter(xdata,(fit_data-ydata[:,0])/ydata[:,1],marker="_",s=1000,color='k')

        # other settings
        ax2.yaxis.grid(True)
        ax2.set_ylim([-2,2])
        ax2.set_xticks(range(len(xdata)))
        if(plot_type=='yields'):
            ax2.set_xticklabels([rf'${part}$' for part in particle_yields])
        elif(plot_type=='ratios'):
            ax2.set_xticklabels([rf'$\frac{{{part1}}}{{{part2}}}$' for part1,part2 in particle_ratios])
        ax2.set_ylabel('(fit-data)/error')

        # Hide x labels and tick labels for all but bottom plot.
        ax1.label_outer()
        ax2.label_outer()

        # save plots
        f.savefig(plot_file_name+f'_{plot_type}.png')  

    # calculate fits and extract results
    result = fit_freezeout(dict_yield,**kwargs) 

    if(method=='all' or method=='yields'):
        fit_yields = result['fit_yields']
        fit_string_yields = result['fit_string_yields']
        result_yields = result['result_yields']
        data_yields = result['data_yields']
        particle_yields = result['particle_yields']
        # x-values, just the indexes of ratios [1,2,...,N_particles]
        xyields = np.arange(len(data_yields))
        make_plot(xyields,data_yields,result_yields,fit_string_yields,'yields')
    else:
        fit_yields = None

    if(method=='all' or method=='ratios'):
        fit_ratios = result['fit_ratios']
        fit_string_ratios = result['fit_string_ratios']
        result_ratios = result['result_ratios']
        data_ratios = result['data_ratios']
        particle_ratios = result['particle_ratios']
        # x-values, just the indexes of ratios [1,2,...,N_ratios]
        xratios = np.arange(len(data_ratios))
        make_plot(xratios,data_ratios,result_ratios,fit_string_ratios,'ratios')
    else:
        fit_ratios = None

    # plot chi squared results
    if(chi2_plot):

        def make_plot_chi2(all_data_chi2,fit,plot_type):
            """
            Function to plot the chi^2 for each parameter
            """
            list_quant = ['T','muB','muQ','muS','gammaS','dVdy']
            list_latex = ['T','\mu_B','\mu_Q','\mu_S','\gamma_S','dV/dy']
            list_unit = ['GeV','GeV','GeV','GeV','','fm^{3}']

            for i,data_chi2 in enumerate(all_data_chi2):
                f,ax = pl.subplots(figsize=(10,7))
                pl.axvline(x=fit[i,0],color='k')
                ax.axvspan(fit[i,0]-fit[i,1], fit[i,0]+fit[i,1], alpha=0.25, color='k')
                ax.plot(data_chi2[0],data_chi2[1])
                ax.set(xlabel=f'${list_latex[i]}$',ylabel=r'$\chi^2$',title=f'${list_latex[i]}={fit[i,0]:.3f} \pm {fit[i,1]:.3f}\ {list_unit[i]}$') 
                f.savefig(plot_file_name+f'_{plot_type}_{list_quant[i]}.png')

        if(method=='all' or method=='yields'):
            make_plot_chi2(result['chi2_yields'],fit_yields,'yields')
        if(method=='all' or method=='ratios'):
            make_plot_chi2(result['chi2_ratios'],fit_ratios,'ratios')

    return fit_yields,fit_ratios

###############################################################################
if __name__ == "__main__":
    # values of \mu_B where to test the parametrization of lQCD data
    tab = [[0.,'r'],[0.2,'tab:orange'],[0.3,'b'],[0.4,'g']]

    main('muB',tab)
    main('nS0',tab)
    
    # BES STAR data, for tests (PHYSICAL REVIEW C 96, 044904 (2017))
    # just the pions and Lambdas are corrected for feed-down weak decays
    dict_19GeV = {'pi+':161.4,'pi+_err':17.8,'pi-':165.8,'pi-_err':18.3, \
    'K+':29.6,'K+_err':2.9,'K-':18.8,'K-_err':1.9, \
    'p':34.2,'p_err':4.5,'p~':4.2,'p~_err':0.5, \
    'Lambda':12.58,'Lambda_err':0.46,'Lambda~':1.858,'Lambda~_err':0.099, \
    'Xi-':1.62,'Xi-_err':0.1,'Xi~+':0.421,'Xi~+_err':0.03}

    # BES STAR data, for tests (PHYSICAL REVIEW C 96, 044904 (2017))
    # just the pions and Lambdas are corrected for feed-down weak decays
    dict_39GeV = {'pi+':182.3,'pi+_err':20.1,'pi-':185.8,'pi-_err':20.5, \
    'K+':32.,'K+_err':2.9,'K-':25.,'K-_err':2.3, \
    'p':26.5,'p_err':2.9,'p~':8.5,'p~_err':1., \
    'Lambda':11.02,'Lambda_err':0.86,'Lambda~':3.82,'Lambda~_err':0.36, \
    'Xi-':1.54,'Xi-_err':0.19,'Xi~+':0.78,'Xi~+_err':0.11}

    # ALICE data, for tests
    # here all particles are corrected for feed-down weak decays
    dict_2TeV = {'pi+':733,'pi+_err':54,'pi-':732,'pi-_err':52, \
    'K+':109,'K+_err':9,'K-':109,'K-_err':9, \
    'p':34,'p_err':3,'p~':33,'p~_err':3, \
    'Lambda':26,'Lambda_err':3,'Lambda~':None,'Lambda~_err':None, \
    'Xi-':3.34,'Xi-_err':0.27,'Xi~+':3.28,'Xi~+_err':0.26}

    plot_freezeout(dict_19GeV,chi2_plot=False,method='all')