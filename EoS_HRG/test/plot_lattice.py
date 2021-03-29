import matplotlib
import matplotlib.pyplot as pl
import os
import numpy as np
import argparse
from math import factorial

# import from __init__.py
from . import *
# import the functions to test from fit_lattice.py
from EoS_HRG.fit_lattice import *
# susceptibilities from HRG
from EoS_HRG.HRG import HRG

# directory where the fit_lattice_test.py is located
dir_path = os.path.dirname(os.path.realpath(__file__))

###############################################################################
def plot_lattice(EoS,tab,list_quant,wparam=True,all_labels=True,muBoT=False,Tmax=None):
    """
    Produce plots with lQCD data of each quantity = 'P', 'n_B', 's' or 'e'
    as a function of T [GeV] for the values of muB [GeV] or muB/T indicated in tab
    """

    # title of the plot
    if(EoS == 'nS0'):
        title = r'$\langle n_S \rangle = 0$ and $\langle n_Q \rangle = 0.4 \langle n_B \rangle$'
    elif(EoS == 'muB'):
        title = r'$\mu_Q = \mu_S = 0$'

    # get the range of lattice data in T to plot the parametrization (default values)
    lQCDdata0 = WB_EoS0['T']
    # get the range of lattice data in T to plot the parametrization
    if(Tmax==None):
        xval = np.linspace(lQCDdata0.min(),lQCDdata0.max(),50)
    else:
        xval = np.linspace(lQCDdata0.min(),Tmax,50)

    # initialize plots
    plots = np.array([pl.subplots(figsize=(10,7)) for x in np.arange(len(list_quant))])
    f = plots[:,0]
    ax = plots[:,1]

    for imuB,[muB,color] in enumerate(tab):
        if(not muBoT):
            xmuB = muB # fixed value of muB
        elif(muBoT):
            xmuB = muB*xval # fixed value of muB/T
        # parametrization
        # get the parametrization of lQCD data for each \mu_B
        if(EoS!='nS0'):
            paramdata = param(xval,xmuB,0.,0.)
        else:
            paramdata = EoS_nS0(param,xval,xmuB)

        # for each quantity, plot
        for iq,quant in enumerate(list_quant):

            # data at mu = 0 from lQCD
            if(muB==0):
                try:
                    data = WB_EoS0[quant]
                    err = WB_EoS0[quant+'_err']
                    latticeplot, = ax[iq].plot(WB_EoS0['T'],data, 'o', color=color, ms='8', mfc='none')
                    ax[iq].errorbar(WB_EoS0['T'], data, yerr=err, xerr=None, fmt='none', color=color)
                    first_legend = ax[iq].legend([latticeplot],['lQCD'], loc='lower right', borderaxespad=0., frameon=False)
                    if(all_labels):
                        f[iq].gca().add_artist(first_legend)
                except:
                    pass
            
            # data at fixed muB/T from lQCD
            if(muBoT and EoS!='nS0'):
                try:
                    data = WB_EoS_muBoT2021[f'{quant}(muB/T={muB})']
                    latticeplot, = ax[iq].plot(data[:,0],data[:,1], 'o', color=color, ms='8', mfc='none')
                    ax[iq].errorbar(data[:,0], data[:,1], yerr=data[:,2], xerr=None, fmt='none', color=color)
                except:
                    pass

            if(wparam):
                # now plot parametrization for each muB and each quantity
                if(quant=='cs^2'):
                    continue
                if(all_labels):
                    if(not muBoT):
                        label = r'$ \mu_B = $'+str(muB)+' GeV'
                    elif(muBoT):
                        label = r'$ \mu_B/T = $'+str(muB)
                    ax[iq].plot(xval,paramdata[quant],'--', ms='8', color=color, label=label)
                    ax[iq].legend(title='Parametrization', title_fontsize='25', loc='center right', frameon=False)
                else:
                    if(imuB==0):
                        paramplot, = ax[iq].plot(xval,paramdata[quant],'--', ms='8', color=color)
                    else:
                        ax[iq].plot(xval,paramdata[quant],'--', ms='8', color=color)

    dict_plots = {}
    for iq,quant in enumerate(list_quant):
        if(quant=='n_B' or quant=='n_Q' or quant=='n_S' or quant=='s'):
            ylabel = '$'+quant+'/T^3$'
        elif(quant=='cs^2'):
            ylabel = '$c^2_s$'
        else:
            ylabel = '$'+quant+'/T^4$'
        ax[iq].set(xlabel='T [GeV]', ylabel=ylabel, title=title)

        if(not all_labels and quant!='cs^2'):
            if(wparam):
                first_legend = ax[iq].legend([latticeplot,paramplot],['lQCD','lQCD parametrization'], loc='lower right', frameon=False)
            else:
                first_legend = ax[iq].legend([latticeplot],['lQCD'], loc='lower right', frameon=False)
            f[iq].gca().add_artist(first_legend)

        # return dict of plots
        dict_plots.update({quant:[f[iq],ax[iq]]})

    return dict_plots

###############################################################################
def plot_lattice_all(EoS,muB,wparam=True,muBoT=False):
    """
    Produce a plot with lQCD data of all quantities (P,n_B,s,e) for a fixed muB [GeV] or muB/T
    as a function of temperature T [GeV]
    """
    xval = WB_EoS0['T'] # temperature values
    if(not muBoT):
        xmuB = muB # fixed value of muB
    elif(muBoT):
        xmuB = muB*xval # fixed value of muB/T

    f,ax = pl.subplots(figsize=(10,7))
    for quant,color in [["P",'r'],["n_B",'tab:orange'],["e",'b'],["s",'g'],["I",'m']]:
        if(muB==0):
            try:
                data = WB_EoS0[quant]
                err = WB_EoS0[quant+'_err']
                ax.plot(xval, data, 'o', color=color, ms='8', mfc='none')
                ax.errorbar(xval, data, yerr=err, xerr=None, fmt='none', color=color)
            except:
                pass

        # data at fixed muB/T from lQCD
        if(muBoT and EoS!='nS0'):
            try:
                data = WB_EoS_muBoT2021[f'{quant}(muB/T={muB})']
                latticeplot, = ax.plot(data[:,0],data[:,1], 'o', color=color, ms='8', mfc='none')
                ax.errorbar(data[:,0], data[:,1], yerr=data[:,2], xerr=None, fmt='none', color=color)
            except:
                pass

        if(EoS=='muB'):
            paramdata = param(xval,xmuB,0.,0.)
        elif(EoS=='nS0'):
            paramdata = EoS_nS0(param,xval,xmuB)

        if(wparam):
            ax.plot(xval,paramdata[quant],'--', ms='8', color=color)

    # check min/max of T and s/T^3 in the plot
    xmin = xval.min()
    xmax = xval.max()
    smax = WB_EoS0['s'][0].max()

    for quant,color,dec in [["P",'r',0.],["n_B",'tab:orange',0.15],["e",'b',0.3],["s",'g',0.45],["I",'m',0.6]]:
        if(quant=='n_B' or quant=='n_Q' or quant=='n_S' or quant=='s'):
            ylabel = '$'+quant+'/T^3$'
        else:
            ylabel = '$'+quant+'/T^4$'
        pl.text(xmin+(0.3+dec)*(xmax-xmin),0.45*smax, ylabel, color=color, fontsize=MEDIUM_SIZE)

    # indicate value of muB on the plot
    if(not muBoT):
        pl.text(xmin, 0.9*smax, r'$\mu_B = '+str(muB)+'$ GeV', fontsize=MEDIUM_SIZE)
    elif(muBoT):
        pl.text(xmin, 0.9*smax, r'$\mu_B/T = $'+str(muB), fontsize=MEDIUM_SIZE)

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
def main(EoS,tab,muBoT=False):
    # get the range of lattice data in T to plot the parametrization (default values)
    lQCDdata0 = WB_EoS0['T']

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

    # create new directory to save plots if it doesn't already exist
    if not os.path.exists(f'{dir_path}/plot_lattice/'):
        os.makedirs(f'{dir_path}/plot_lattice/')

    # min/max for the temperature
    Tmin = args.Tmin
    Tmax = args.Tmax

    print(EoS)
    # quantities to plot
    list_quant = ["P","n_B","n_S","s","e","I",'cs^2']

    # initialize plots for each quantity with lattice data
    dict_plots = plot_lattice(EoS,tab,list_quant,muBoT=muBoT)
    # export plots
    for quant in list_quant:
        if(EoS=='nS0' and quant=='n_S'):
            continue
        if(not muBoT):
            dict_plots[quant][0].savefig(f"{dir_path}/plot_lattice/lQCD_{EoS}_{quant}_T_muB.png")
        elif(muBoT):
            dict_plots[quant][0].savefig(f"{dir_path}/plot_lattice/lQCD_{EoS}_{quant}_T_muBoT.png")
        dict_plots[quant][0].clf()
        pl.close(dict_plots[quant][0])

    # plot for P/T^4, nB/T^3, s/T^3, e/T^4 for # muB
    for muB,_ in tab:
        fall,_ = plot_lattice_all(EoS,muB,muBoT=muBoT)
        if(not muBoT):
            fall.savefig(f"{dir_path}/plot_lattice/lQCD_{EoS}_all_muB{int(10.*muB):02d}_T.png")
            print('  plot \mu_B = '+str(muB)+' GeV')
        elif(muBoT):
            fall.savefig(f"{dir_path}/plot_lattice/lQCD_{EoS}_all_muBoT{muB}_T.png")
            print('  plot \mu_B/T = '+str(muB))
        fall.clf()
        pl.close(fall)

    # output data for each \mu_B
    if(args.output):
        xval = np.linspace(Tmin,Tmax,args.Npoints)
        for muB,_ in tab:
            if(not muBoT):
                xmuB = muB # fixed value of muB
            elif(muBoT):
                xmuB = muB*xval # fixed value of muB/T
            # get the parametrization of lQCD data for each \mu_B
            if(EoS!='nS0'):
                paramdata = param(xval,xmuB,0.,0.)
            else:
                paramdata = EoS_nS0(param,xval,xmuB)

            # output data for each \mu_B or \mu_B/T
            if(not muBoT):
                filename = f"{dir_path}/plot_lattice/lQCD_muB{int(muB*10):02d}_{EoS}.dat"
            elif(muBoT):
                filename = f"{dir_path}/plot_lattice/lQCD_muBoT{muB}_{EoS}.dat"

            with open(filename,'w') as outfile:
                outfile.write(",".join(paramdata.keys()))
                outfile.write('\n')
                for i,_ in enumerate(paramdata['T']):
                    outfile.write(",".join(map(lambda x: f"{paramdata[x][i]:5.3E}",paramdata.keys())))
                    outfile.write('\n')

@timef
def plot_chi():
    """
    To plot the fit of susceptibilities compared to lQCD results
    """
    # create new directory to save plots if it doesn't already exist
    if not os.path.exists(f'{dir_path}/plot_chi/'):
        os.makedirs(f'{dir_path}/plot_chi/')

    # values in T for the plot
    xtemp = np.arange(0.05,10.,0.0001)
    # x-lim for zoom plot
    Tminz = 0.1
    Tmaxz = 0.2

    # susceptibilities from HRG
    xTHRG = np.arange(0.05,0.18,0.005)
    print('Calculating susceptibilities...')
    HRG_chi = HRG(xTHRG,0.,0.,0.,offshell=True,eval_chi=True)['chi']
    HRG_chi_nS0 = EoS_nS0(HRG,xTHRG,0.,offshell=True,eval_chi=True)['chi']

    # location of the zoom plot
    # 'upper right' = 1
    # 'lower right' = 4
    locz = [4,4,4,4,1,1,4,1,1,4,1,4,1,4,1,4,4,4,4,4,1,1,4,1]

    # count how many times lQCD data have been plotted
    lqcd_data_bool = np.zeros(6)

    # loop over susceptibilities
    for ichi,chi in enumerate(list_chi+list_chi_nS0):
        print(chi)

        f,ax = pl.subplots(figsize=(8,7))
        if(chi=='chiQS22'):
            # shift a tiny bit zoom plot for chiQS22
            axins = inset_axes(ax,width=3, height=2, bbox_to_anchor=(0.1,0.065,0.85,0.935), bbox_transform=ax.transAxes, loc=locz[ichi])
        else:
            axins = inset_axes(ax,width=3, height=2, bbox_to_anchor=(0.1,0.1,0.85,0.85), bbox_transform=ax.transAxes, loc=locz[ichi])

        # loop over lattice data
        for ilattice,[chi_lattice,point] in enumerate([[chi_lattice2020,'s'],[chi_lattice2015,'*'],[chi_lattice2014,'^'],[chi_lattice2012,'o'],[chi_lattice2018,'p'],[chi_lattice2017,'P']]):
            try:
                data = chi_lattice[chi]
                if(chi=='chiQ4'):
                    data = np.delete(data,[0,1,2],axis=0) # remove 1st 3 points for chiQ4

                for xax in [ax,axins]:
                    xax.plot(data[:,0], data[:,1], point, color='b', ms='6', fillstyle='none',label='lQCD')
                    xax.errorbar(data[:,0], data[:,1], yerr=data[:,2], fmt='none', color='b')

                lqcd_data_bool[ilattice] += 1 # count how many times lQCD data are plotted
                break
            except:
                pass

        # plot parametrization
        y_chi = param_chi(xtemp,chi)
        for xax in [ax,axins]:
            xax.plot(xtemp,y_chi,color='r',linewidth=2,label='parametrization')
            
        # select data from HRG
        if(chi=='chiB2_nS0'):
            xHRG_chi = HRG_chi_nS0[1]/factorial(2) 
            ls = 'dotted'
            label = r'limit $T \rightarrow \infty$'
        elif(chi=='chiB4_nS0'):
            xHRG_chi = HRG_chi_nS0[7]/factorial(4)
            ls = 'dotted'
            label = r'limit $T \rightarrow \infty$'
        else:
            xHRG_chi = HRG_chi[ichi]
            ls = '-'
            label = 'SB limit'

        # SB limit
        for xax in [ax,axins]:
            xax.plot([5,10],chi_SB[chi]*np.ones(2),linestyle=ls,color='k',linewidth=5,label=label)

        # HRG
        for xax in [ax,axins]:
            xax.plot(xTHRG,xHRG_chi,'--',color='k',linewidth=2,label='HRG')

        # find min / max values
        ylimm = 1.1*min(xHRG_chi[0],chi_SB[chi],np.amin(y_chi),np.amin(data[:,1]-data[:,2]))
        ylimp = 1.1*max(xHRG_chi[0],chi_SB[chi],np.amax(y_chi),np.amax(data[:,1]+data[:,2]))

        # find min / max values for zoom plot
        condHRGT = np.logical_and(xTHRG >= Tminz,xTHRG <= Tmaxz)
        condparam = np.logical_and(xtemp >= Tminz,xtemp <= Tmaxz)
        conddataT = np.logical_and(data[:,0] >= Tminz,data[:,0] <= Tmaxz)
        if(abs(np.amin(xHRG_chi[condHRGT]))>abs(np.amax(xHRG_chi[condHRGT]))):
            ylimmz = 1.1*min(np.amin(data[:,1][conddataT]-data[:,2][conddataT]),np.amin(y_chi[condparam]))
            ylimpz = 1.1*max(np.amax(xHRG_chi[condHRGT]),np.amax(data[:,1][conddataT]+data[:,2][conddataT]))
        else:
            ylimmz = 1.1*min(np.amin(xHRG_chi[condHRGT]),np.amin(data[:,1][conddataT]-data[:,2][conddataT]))
            ylimpz = 1.1*max(np.amax(data[:,1][conddataT]+data[:,2][conddataT]),np.amax(y_chi[condparam]))

        # adjust limits
        if(ylimp>=0. and abs(ylimm)<0.1*ylimp):
            ylimm = -0.05*ylimp
        else:
            ylimp = max(0.05*abs(ylimm),ylimp)

        # adjust limits for zoom plot
        if(ylimmz>=0.):
            if(ylimmz/ylimp<0.15):
                ylimmz = 0
        ylimpz = min(max(0.025*abs(ylimmz),ylimpz),0.975*ylimp)

        ax.set(xlabel='T [GeV]',ylabel=chi_latex[chi],xscale='log',ylim=[ylimm,ylimp])
        axins.set(xlim=[Tminz,Tmaxz],ylim=[ylimmz,ylimpz])
        ax.set_xticks([round(0.05,2),round(0.1,1),round(0.2,1),round(0.3,1),round(0.5,1),round(1,0),round(2,0),round(3,0),round(5,0),round(10,0)])

        def f_ticks(number,pos):
            if(number>=1):
                return f'{number:.0f}'
            elif(number>=0.1):
                return f'{number:.1f}'
            else:
                return f'{number:.2f}'

        ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(f_ticks))

        # location of lines connecting to the zoom plot
        loc1 = 1
        loc2 = 4
        if(chi=='chi0'):
            loc1 = 2
        elif(chi=='chiQ2'):
            loc2 = 3
        mark_inset(ax, axins, loc1=loc1, loc2=loc2, fc="none", ec="0.5")

        # place legend
        if(chi=='chi0'):
            loc = 'upper left'
            bbox_to_anchor = None
        elif(chi=='chiB2' or chi=='chiB2_nS0'):
            loc = 'center right'
            bbox_to_anchor = (0.,0.35,1.,0.7)
        elif(chi=='chiQ4'):
            loc = 'lower right'
            bbox_to_anchor = None
        elif(chi=='chiQ2'):
            loc = 'center right'
            bbox_to_anchor = (0.,0.35,1.,0.7)
        elif(chi=='chiBQ11'):
            loc = 'lower right'
            bbox_to_anchor = (0.,0.1,1.,0.9)

        # show legend only once per lQCD data
        if(lqcd_data_bool[ilattice]==1):
            ax.legend(title=None, title_fontsize='25', bbox_to_anchor=bbox_to_anchor, loc=loc, borderaxespad=0.5, frameon=False)

        f.savefig(f'{dir_path}/plot_chi/{chi}.png')
        f.clf()
        pl.close(f)

###############################################################################
if __name__ == "__main__":
    # values of \mu_B where to test the parametrization of lQCD data
    tab = [[0,'r'],[0.2,'tab:orange'],[0.3,'b'],[0.4,'g']]
    main('muB',tab)
    main('nS0',tab)
    # values of \mu_B/T where to test the parametrization of lQCD data
    tab = [[0,'r'],[1,'tab:orange'],[2,'b'],[3,'g'],[3.5,'m']]
    main('muB',tab,muBoT=True)
    main('nS0',tab,muBoT=True)
    # plot the susceptibilities
    plot_chi()
    