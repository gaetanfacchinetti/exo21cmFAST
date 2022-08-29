############################################################################
#    Code to plot a synthetic figure of example_run_lightcone.py outputs
#    
#    Copyright (C) 2022  Gaetan Facchinetti
#    gaetan.facchinetti@ulb.be
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>
############################################################################


# Plotting packages:
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from pylab import *
from matplotlib.gridspec import GridSpec

import py21cmfast as p21c
from scipy import interpolate
import py21cmfast.dm_dtb_tools as db_tools
import example_lightcone_analysis as lightcone_analysis

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

matplotlib.rc_file('matplotlibrc')

database_location = "/home/ulb/physth_fi/gfacchin/exo21cmFAST_release/output/test_database"
cache_location = "/scratch/ulb/physth_fi/gfacchin/output_exo21cmFAST/"

# -------------------------------------------------------------------------------------
# We can charge and remake the analysis for certain lightcone if we want before plotting
#lightcone = p21c.LightCone.read(fname=database_location + '/BrightnessTemp_5/Lightcone/Lightcone.h5')
#io_p21c.make_analysis(database_location + '/BrightnessTemp_5', lightcone, n_psbins=20, nchunks=40)
# -------------------------------------------------------------------------------------

# Defines the two database managers according to the 

def db_manager(approx = False) : 
    db_manager_approx = db_tools.ApproxDepDatabase(path = database_location, cache_path = cache_location)
    db_manager_exact  = db_tools.DHDatabase(path = database_location, cache_path = cache_location)
    return db_manager_exact if (approx is False) else db_manager_approx






# -------------------------------------------------------------------------------------
# Define which cases we want to plot
bkr        = [False, True]
process    = ['decay', 'decay']
mDM        = [1.26e+8, 1.26e+8]
primary    = ['elec_delta', 'elec_delta']
boost      = ['none', 'none']
fs_method  = ['no_He', 'no_He']
sigv       = [0, 0]
lifetime   = [1e+26, 1e+26]
comm       = ['large_box', 'large_box']

index_exact = [-1, -1]

approx_shape       = ['schechter']
approx_params      = [[1.5310e-01, -3.2090e-03, 1.2950e-01]]
approx_sigv        = [0.]
approx_lifetime    = [1e+26]
approx_mDM         = [1.26e+8]
approx_process     = ['decay']
fion_H_over_fheat  = [-1.]
fion_He_over_fheat = [-1.]
fexc_over_fheat    = [-1.]
force_init_cond    = [True]
xe_init            = [0.0031]
Tm_init            = [118.0]
approx_comm        = ['large_box']

index_approx = [-1]


# Get the reference plot witout exotic energy injection
index_no_DM_0 = db_manager(approx = False).search(bkr=False, process='none', mDM=0., primary='none', boost='none', fs_method='none', 
                                                sigmav=0., lifetime=0., comment='large_box')[0]
index_no_DM_1 = db_manager(approx = True ).search(process='none', mDM=0., approx_shape='none', approx_params=[0.], 
                                                sigmav=0., lifetime=0., fion_H_over_fheat=0., fion_He_over_fheat=0., fexc_over_feat=0.,
                                                force_init_cond=False, xe_init=0., Tm_init=0.,
                                                comment='large_box')[0]

# Get the index of the corresponding files in the database
for i in range(0, len(index_exact)) : 
    index_exact[i] = db_manager(approx = False).search(bkr=bkr[i], process=process[i], mDM=mDM[i], primary=primary[i], 
                                                       boost=boost[i], fs_method=fs_method[i], sigmav=sigv[i], lifetime=lifetime[i], comment=comm[i])[0]


for i in range(0, len(index_approx)) :
    index_approx[i] = db_manager(approx = True).search(process=approx_process[i], mDM=approx_mDM[i], approx_shape=approx_shape[i], approx_params=approx_params[i],
                                                  sigmav=approx_sigv[i], lifetime=approx_lifetime[i], fion_H_over_fheat=fion_H_over_fheat[i], 
                                                  fion_He_over_fheat=fion_He_over_fheat[i], fexc_over_feat=fexc_over_fheat[i], force_init_cond=force_init_cond[i], 
                                                xe_init=xe_init[i], Tm_init=Tm_init[i], comment=approx_comm[i])[0]

#print(index_exact, index_approx)

# Get the corresponding name of the files (according to the database)
filename = [database_location + '/darkhistory/BrightnessTemp_' + str(index_no_DM_0),
            database_location + '/approx/BrightnessTemp_' + str(index_no_DM_1),
            *[database_location + '/darkhistory/BrightnessTemp_' + str(ind) for ind in index_exact],
            *[database_location + '/approx/BrightnessTemp_' + str(ind) for ind in index_approx],]



# -------------------------------------------------------------------------------------

# Define the variables we will plot 
z_GQ = [[] for f in filename]
dTb = [[] for f in filename]
xH_box = [[] for f in filename]
x_e_box = [[] for f in filename]
Ts_box = [[] for f in filename]
Tk_box = [[] for f in filename]
z_PS = [[] for f in filename]
delta_z = [[] for f in filename]
k_approx = [0 for f in filename]

# Get the relevant quantities from the codes in io_p21c
for i, fname in enumerate(filename) :
    (z_GQ[i], dTb[i], xH_box[i], x_e_box[i], Ts_box[i], Tk_box[i]) = lightcone_analysis.make_plots_global_quantities(fname, plot=False) 
    (z_PS[i], delta_z[i], k_approx[i])  = lightcone_analysis.make_plots_powerspectra(fname, 0.12)


# Get the corresponding name of the files for DM energy injection (according to the database)
filename = [*[database_location + '/darkhistory/result_run_' + str(ind) + '.txt' for ind in index_exact], 
            *[database_location + '/approx/result_run_' + str(ind) + '.txt' for ind in index_approx]]

z_f = []
f_H_ion = []
f_He_ion = []
f_exc = []
f_heat = []
f_cont = []
inj_e_smooth = []
xe = []


for file in filename :
    data = np.loadtxt(file)
    z_f.append(data[:, 0]-1)
    f_H_ion.append(data[:, 1])
    f_He_ion.append(data[:, 2])
    f_exc.append(data[:, 3])
    f_heat.append(data[:, 4])
    f_cont.append(data[:, 5])
    xe.append(data[:,7])

    inj_e_smooth.append(data[:,6]) # in erg/s^{-1} (per baryons)

# -------------------------------------------------------------------------------------


# Prepare the figure for the plot

fig = plt.figure(constrained_layout=False, figsize=(10,8))
fig.subplots_adjust(wspace=0, hspace=0)

gs = GridSpec(6, 4, figure=fig)
ax1 = fig.add_subplot(gs[0:2, 0:2])
ax2 = fig.add_subplot(gs[2:4, 0:2])
ax3 = fig.add_subplot(gs[4:5, 0:2])
ax4 = fig.add_subplot(gs[5:6, 0:2])
ax5 = fig.add_subplot(gs[0:2, 2:4])
ax6 = fig.add_subplot(gs[2:3, 2:4])
ax7 = fig.add_subplot(gs[3:4, 2:4])
ax8 = fig.add_subplot(gs[4:5, 2:4])
ax9 = fig.add_subplot(gs[5:6, 2:4])


ax1.tick_params(labelbottom=False)
ax2.tick_params(labelbottom=False)
ax5.tick_params(labelbottom=False, labelright=True, labelleft=False)
ax6.tick_params(labelbottom=False, labelright=True, labelleft=False)
ax7.tick_params(labelbottom=False, labelright=True, labelleft=False)
ax8.tick_params(labelbottom=False, labelright=True, labelleft=False)
ax9.tick_params(labelright=True, labelleft=False)
ax5.yaxis.set_label_position("right")
ax6.yaxis.set_label_position("right")
ax7.yaxis.set_label_position("right")
ax8.yaxis.set_label_position("right")
ax9.yaxis.set_label_position("right")

ax1.minorticks_on()
ax2.minorticks_on()
ax3.minorticks_on()
ax4.minorticks_on()
ax5.minorticks_on()
ax6.minorticks_on()
ax7.minorticks_on()
ax8.minorticks_on()
ax8.minorticks_on()

ax1.grid(True, alpha = 0.5, linewidth=0.5)
ax2.grid(True, alpha = 0.5, linewidth=0.5)
ax5.grid(True, alpha = 0.5, linewidth=0.5)
ax6.grid(True, alpha = 0.5, linewidth=0.5)
ax7.grid(True, alpha = 0.5, linewidth=0.5)
ax8.grid(True, alpha = 0.5, linewidth=0.5)
ax9.grid(True, alpha = 0.5, linewidth=0.5)

#X and Y ranges
z_max = 35
ax1.set_xlim(4, z_max)
ax2.set_xlim(4, z_max)
ax3.set_xlim(4, z_max)
ax4.set_xlim(4, z_max)
ax5.set_xlim(4, z_max)
ax6.set_xlim(4, z_max)
ax7.set_xlim(4, z_max)
ax8.set_xlim(4, z_max)
ax9.set_xlim(4, z_max)

ax1.set_ylim(2e-2, 1e+3)
ax2.set_ylim(-180, 50)
ax5.set_ylim(1e-28, 5e-30)
ax6.set_ylim(5e-5, 4)
ax7.set_ylim(5e-5, 4)

#Axes' labels
ax1.set_ylabel(r'$\overline{\delta T_b}^2 \Delta_{21}^2~{\rm [mK^2]}$')
ax2.set_ylabel(r'$\overline{\delta T_b}~{\rm [mK]}$')
ax3.set_ylabel(r'$d~{\rm [Mpc]}$')
ax4.set_ylabel(r'$d~{\rm [Mpc]}$')
ax4.set_xlabel(r'$z$')
ax5.set_ylabel(r'$\epsilon_{\rm heat}^{\rm DM}~[\rm erg/s]$', rotation=-90, labelpad=20)
ax6.set_ylabel(r'$\overline{x_{\rm e}}$', rotation=-90, labelpad=20)
ax7.set_ylabel(r'$\overline{x_{\rm HII}}$', rotation=-90, labelpad=20)
ax8.set_ylabel(r'${\rm T_{\rm S}}~{\rm [K]}$', rotation=-90, labelpad=20)
ax9.set_ylabel(r'${\rm T_{\rm K}}~{\rm [K]}$', rotation=-90, labelpad=20)
ax9.set_xlabel(r'$z$')

#Set logscale
ax1.set_yscale('log')
ax5.set_yscale('log')
ax6.set_yscale('log')
ax7.set_yscale('log')
ax8.set_yscale('log')
ax9.set_yscale('log')


my_colors = ['#D90000','#2574d4','#20b356', '#f07000', '#8e006f']
colors=['k', *my_colors]
solid_lines=[ Line2D([],[],color=c,linestyle='-',linewidth=1) for c in colors ]

# -------------------------------------------------------------------------------------



# The power spectrum
ax1.plot(z_PS[0], delta_z[0], '-',  color=colors[0])  # no DM energy injection
ax1.plot(z_PS[1], delta_z[1], '--', color=colors[0]) # no DM energy injection
ax1.plot(z_PS[2], delta_z[2], '-',  color=colors[1])  # DH no bkr
ax1.plot(z_PS[3], delta_z[3], '-',  color=colors[2]) # DH bkr
ax1.plot(z_PS[4], delta_z[4], '--', color=colors[1]) # approx no bkr


# The global brighness temperature
ax2.plot(z_GQ[0], dTb[0], '-',  color=colors[0])  # no DM energy injection
ax2.plot(z_GQ[1], dTb[1], '--', color=colors[0]) # no DM energy injection
ax2.plot(z_GQ[2], dTb[2], '-',  color=colors[1])  # DH no bkr
ax2.plot(z_GQ[3], dTb[3], '-',  color=colors[2]) # DH bkr
ax2.plot(z_GQ[4], dTb[4], '--', color=colors[1]) # approx no bkr

# The energy injected into heat
imax = np.where(z_f[0] == min(z_f[0], key=lambda x:abs(x-z_max)))[0].tolist()[0]
ax5.plot(z_f[0][imax:-1], f_heat[0][imax:-1]*inj_e_smooth[0][imax:-1], '-',  color=colors[1]) # DH no bkr
ax5.plot(z_f[1][imax:-1], f_heat[1][imax:-1]*inj_e_smooth[1][imax:-1], '-',  color=colors[2]) # DH bkr
ax5.plot(z_f[2][imax:-1], f_heat[2][imax:-1]*inj_e_smooth[2][imax:-1], '--',  color=colors[1]) # approx no bkr

# The ionized fractions
ax6.plot(z_GQ[0], x_e_box[0], '-',  color=colors[0]) # no DM energy injection
ax6.plot(z_GQ[1], x_e_box[1], '--', color=colors[0]) # no DM energy injection
ax6.plot(z_GQ[2], x_e_box[2], '-',  color=colors[1])  # DH no bkr
ax6.plot(z_GQ[3], x_e_box[3], '-',  color=colors[2]) # DH bkr
ax6.plot(z_GQ[4], x_e_box[4], '--', color=colors[1]) # approx no bkr

ax7.plot(z_GQ[0], 1-xH_box[0], '-',  color=colors[0]) # no DM energy injection
ax7.plot(z_GQ[1], 1-xH_box[1], '--', color=colors[0]) # no DM energy injection
ax7.plot(z_GQ[2], 1-xH_box[2], '-',  color=colors[1])  # DH no bkr
ax7.plot(z_GQ[3], 1-xH_box[3], '-',  color=colors[2]) # DH bkr
ax7.plot(z_GQ[4], 1-xH_box[4], '--', color=colors[1]) # approx no bkr


# The temperatures
T_CMB = [2.73*(1+zz) for zz in z_GQ[0]]
ax8.plot(z_GQ[0], T_CMB ,":", color='grey', alpha=0.8)
ax9.plot(z_GQ[0], T_CMB ,":", color='grey', alpha=0.8)

ax8.plot(z_GQ[0], Ts_box[0] ,"-",  color=colors[0]) #no DM energy injection
ax8.plot(z_GQ[1], Ts_box[1], '--', color=colors[0]) # no DM energy injection
ax8.plot(z_GQ[2], Ts_box[2], '-',  color=colors[1])  # DH no bkr
ax8.plot(z_GQ[3], Ts_box[3], '-',  color=colors[2]) # DH bkr
ax8.plot(z_GQ[4], Ts_box[4], '--', color=colors[1]) # approx no bkr

ax9.plot(z_GQ[0], Tk_box[0] ,"-",  color=colors[0]) # no DM energy injection
ax9.plot(z_GQ[1], Tk_box[1], '--', color=colors[0]) # no DM energy injection
ax9.plot(z_GQ[2], Tk_box[2], '-',  color=colors[1])  # DH no bkr
ax9.plot(z_GQ[3], Tk_box[3], '-',  color=colors[2]) # DH bkr
ax9.plot(z_GQ[4], Tk_box[4], '--', color=colors[1]) # approx no bkr

# -------------------------------------------------------------------------------------

######
## Plotting the brightness temperature for the case without DM
lightcone_noDM = p21c.LightCone.read(fname = database_location + '/darkhistory/BrightnessTemp_' + str(index_no_DM_0) + '/Lightcone/Lightcone.h5')

slc_Tb = np.take(lightcone_noDM.brightness_temp, 0, axis=0)
x_Tb = linspace(4, lightcone_noDM.cell_size * lightcone_noDM.shape[0], lightcone_noDM.shape[0])
y_Tb = lightcone_noDM.lightcone_redshifts
z_interp_Tb = linspace(4, 34, 400)
slc_interp_Tb = []
for slc_item in slc_Tb:
    slc_interp_Tb.append(interpolate.interp1d(y_Tb, slc_item)(z_interp_Tb).tolist())

axins3 = inset_axes(ax3, width="30%", height="8%", loc="upper right")
c1 = ax3.pcolormesh(z_interp_Tb, x_Tb, slc_interp_Tb, cmap="EoR", vmin = -150, vmax= 30, shading='gouraud')
cbar1 = colorbar(c1, ax=ax3, cax=axins3, orientation='horizontal')
axins3.tick_params(colors="white")
axins3.set_xlabel(r"$\delta T_b~{\rm [mK]}$", color="white")


slc_xH = np.take(lightcone_noDM.xH_box, 0, axis=0)
x_xH = linspace(4, lightcone_noDM.cell_size * lightcone_noDM.shape[0], lightcone_noDM.shape[0])
y_xH = lightcone_noDM.lightcone_redshifts
z_interp_xH = linspace(4, 34, 400)
slc_interp_xH = []
for slc_item in slc_xH:
    slc_interp_xH.append(interpolate.interp1d(y_xH, slc_item)(z_interp_xH).tolist())

axins4 = inset_axes(ax4, width="30%", height="8%", loc="upper right")
c1 = ax4.pcolormesh(z_interp_xH, x_xH, slc_interp_xH, cmap="viridis", vmin = 0, vmax= 1, shading='gouraud')
cbar1 = colorbar(c1, ax=ax4, cax=axins4, orientation='horizontal')
axins4.tick_params(colors="black")
axins4.set_xlabel(r"$x_{\rm H}$", color="black")
######

# -------------------------------------------------------------------------------------

# Legends

line1 = Line2D([],[],color='k',linestyle='-',label='longitude=0',linewidth=1)
line2 = Line2D([],[],color='k',linestyle='--',label='longitude=0',linewidth=1)
line3 = Line2D([],[],color=colors[0],linestyle='-',label='longitude=0',linewidth=1)
line4 = Line2D([],[],color=colors[1],linestyle='-',label='longitude=0',linewidth=1)
line5 = Line2D([],[],color=colors[2],linestyle='-',label='longitude=0',linewidth=1)

no_line = Line2D([],[],color='k',linestyle='-',linewidth=0, alpha = 0) # When we don't really want a handle in legend


def remove_handles_legend(legend) : 
    for item in legend.legendHandles :
        item.set_visible(False)

fig.suptitle(r"$\chi \to e^+ e^-,~m_\chi = 126~{\rm MeV},~\tau = 10^{26}~{\rm s} $", fontsize=18, y=1.01)
legend_text_1 = [r"$\rm DarkHistory$", r"$\rm Templates$"]
legend_text_2 = [r"$\rm without~DM~inj.$", r"$\rm no~backreaction$", r"$\rm with~backreaction$"]
legend11 = ax1.legend([line1, line2], legend_text_1, loc='upper left', bbox_to_anchor=(0.01,1.35), ncol=2)
legend12 = ax1.legend([line3, line4, line5], legend_text_2, loc='upper left', bbox_to_anchor=(0.01,1.20), ncol=3)
legend13 = ax1.legend([no_line], [r"$k = {:2.2f}~\rm Mpc^{{-1}}$".format(k_approx[0])], handlelength=0, handletextpad=0, loc='upper right', bbox_to_anchor=(0.99,0.99))

remove_handles_legend(legend13)
ax1.add_artist(legend11)
ax1.add_artist(legend12)

# -------------------------------------------------------------------------------------

# Saving the figure

fig.savefig('./example_plot_0.pdf', bbox_inches='tight')

## END OF SCRIPT
