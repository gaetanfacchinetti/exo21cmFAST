############################################################################
#    Code to plot comparison between the energy injection templates and
#    the full DarkHistory code
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
from pylab import matplotlib
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D 
from scipy import interpolate
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
matplotlib.rc_file('matplotlibrc')

# Load custom packages
import py21cmfast.dm_dtb_tools as db_tools
import example_lightcone_analysis as lightcone_analysis


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
## PART OF THE CODE TO MODIFY ACCORDING TO WHAT MUST BE PLOTTED
## Define which cases we want to compare

database_location = "/home/ulb/physth_fi/gfacchin/exo21cmFAST_release/output/test_database"
cache_location = "/scratch/ulb/physth_fi/gfacchin/output_exo21cmFAST/"

# Common parameters
process    = ['none', 'decay', 'decay']
mDM        = [0, 1.26e+8, 1.26e+8]
sigv       = [0, 0, 0]
lifetime   = [0, 1e+26, 1e+26]

# Parameters for the exact solution
bkr        = [False, False, True]
primary    = ['none', 'elec_delta', 'elec_delta']
boost      = ['none', 'none', 'none']
fs_method  = ['none', 'no_He', 'no_He']
comm       = ['large_box', 'large_box', 'large_box']

# Parameters for the solution using the templates
approx_shape       = ['none', 'schechter', 'schechter']
approx_params      = [[0], [1.5310e-01, -3.2090e-03, 1.2950e-01], [1.5310e-01, -3.2090e-03, 1.2950e-01]]
approx_comm        = ['large_box', 'large_box', 'large_box']
fion_H_over_fheat  = [0, -1., -1.]
fion_He_over_fheat = [0, -1., -1.]
fexc_over_fheat    = [0, -1., -1.]
force_init_cond    = [False, True, True]
xe_init            = [0, 0.0031, 0.0031]
Tm_init            = [0, 118.0, 118.0]

# We can charge and remake the analysis for certain lightcone if we want before plotting
# lightcone = lightcone_analysis.LightCone.read(fname=database_location + '/BrightnessTemp_5/Lightcone/Lightcone.h5')
# lightcone_analysis.make_analysis(database_location + '/BrightnessTemp_5', lightcone, n_psbins=20, nchunks=40)

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------


index_exact  = [-1 for i in range(len(process))]
index_approx = [-1 for i in range(len(process))]

def db_manager(database_location, cache_location, approx = False) : 
    db_manager_approx = db_tools.ApproxDepDatabase(path = database_location, cache_path = cache_location)
    db_manager_exact  = db_tools.DHDatabase(path = database_location, cache_path = cache_location)
    return db_manager_exact if (approx is False) else db_manager_approx


# Get the index of the corresponding files in the database
for i in range(0, len(index_exact)) : 
    index_exact[i] = db_manager(database_location, cache_location, approx = False).search(bkr=bkr[i], process=process[i], mDM=mDM[i], primary=primary[i], 
                                                    boost=boost[i], fs_method=fs_method[i], sigmav=sigv[i], lifetime=lifetime[i], comment=comm[i])[0]


for i in range(0, len(index_approx)) :
    index_approx[i] = db_manager(database_location, cache_location, approx = True).search(process=process[i], mDM=mDM[i], approx_shape=approx_shape[i], 
                                                    approx_params=approx_params[i], sigmav=sigv[i], lifetime=lifetime[i], fion_H_over_fheat=fion_H_over_fheat[i], 
                                                    fion_He_over_fheat=fion_He_over_fheat[i], fexc_over_feat=fexc_over_fheat[i], force_init_cond=force_init_cond[i], 
                                                    xe_init=xe_init[i], Tm_init=Tm_init[i], comment=approx_comm[i])[0]


position_noDM     = None
index_noDM_exact  = None
index_noDM_approx = None

for i, _ in enumerate(process) : 
    if primary[i] == 'none':
        position_noDM    = i
        index_noDM_exact = index_exact[i]

for i, _ in enumerate(process) : 
    if approx_shape[i] == 'none':
        index_noDM_approx = index_approx[i]

#print(index_exact, index_approx, index_noDM_exact, index_noDM_approx)

# Get the corresponding name of the files (according to the database)
filenames_brightness_exact  = [database_location + '/darkhistory/BrightnessTemp_' + str(ind) for ind in index_exact]
filenames_brightness_approx = [database_location + '/approx/BrightnessTemp_' + str(ind) for ind in index_approx]
filenames_energy_inj_exact =  [database_location + '/darkhistory/result_run_' + str(ind) + ".txt" for ind in index_exact if ind != index_noDM_exact]
filenames_energy_inj_approx = [database_location + '/approx/result_run_' + str(ind) + ".txt" for ind in index_approx if ind != index_noDM_approx]

filenames_brightness = [[filenames_brightness_exact[i], filenames_brightness_approx[i]] for i, _ in enumerate(filenames_brightness_exact)]
filenames_energy_inj = [[filenames_energy_inj_exact[i], filenames_energy_inj_approx[i]] for i, _ in enumerate(filenames_energy_inj_exact)]

# -------------------------------------------------------------------------------------
# Define the variables we will plot and get the data

z_GQ      = [[[] for i in [0, 1]] for f in filenames_brightness]
dTb       = [[[] for i in [0, 1]] for f in filenames_brightness]
xH_box    = [[[] for i in [0, 1]] for f in filenames_brightness]
x_e_box   = [[[] for i in [0, 1]] for f in filenames_brightness]
Ts_box    = [[[] for i in [0, 1]] for f in filenames_brightness]
Tk_box    = [[[] for i in [0, 1]] for f in filenames_brightness]
z_PS      = [[[] for i in [0, 1]] for f in filenames_brightness]
delta_z   = [[[] for i in [0, 1]] for f in filenames_brightness]
k_approx  = [[[] for i in [0, 1]] for f in filenames_brightness]

# Get the relevant quantities from the codes in lightcone_analysis
for i, fname in enumerate(filenames_brightness) :
    for j in [0, 1] :
        (z_GQ[i][j], dTb[i][j], xH_box[i][j], x_e_box[i][j], Ts_box[i][j], Tk_box[i][j]) = lightcone_analysis.make_plots_global_quantities(fname[j], plot=False) 
        (z_PS[i][j], delta_z[i][j], k_approx[i][j])  = lightcone_analysis.make_plots_powerspectra(fname[j], 0.12)


z_f           = [[np.array([]) for i in [0, 1]] for f in filenames_energy_inj]
f_H_ion       = [[np.array([]) for i in [0, 1]] for f in filenames_energy_inj]
f_He_ion      = [[np.array([]) for i in [0, 1]] for f in filenames_energy_inj]
f_exc         = [[np.array([]) for i in [0, 1]] for f in filenames_energy_inj]
f_heat        = [[np.array([]) for i in [0, 1]] for f in filenames_energy_inj]
f_cont        = [[np.array([]) for i in [0, 1]] for f in filenames_energy_inj]
inj_e_smooth  = [[np.array([]) for i in [0, 1]] for f in filenames_energy_inj]
xe            = [[np.array([]) for i in [0, 1]] for f in filenames_energy_inj]


#print(z_f, filenames_energy_inj)

for i, file in enumerate(filenames_energy_inj) :
    for j in [0, 1] :
        data = np.loadtxt(file[j])
        z_f[i][j]       = data[:, 0]-1
        f_H_ion[i][j]   = data[:, 1]
        f_He_ion[i][j]  = data[:, 2]
        f_exc[i][j]     = data[:, 3]
        f_heat[i][j]    = data[:, 4]
        f_cont[i][j]    = data[:, 5]
        xe[i][j]        = data[:,7]

        inj_e_smooth[i][j] = data[:,6] # in erg/s^{-1} (per baryons)

# -------------------------------------------------------------------------------------


# Prepare the figure for the plot

fig = plt.figure(constrained_layout=False, figsize=(10,8))
fig.subplots_adjust(wspace=0, hspace=0)

gs = GridSpec(3, 2, figure=fig)
ax1 = fig.add_subplot(gs[0:1, 0:1])
ax2 = fig.add_subplot(gs[1:2, 0:1])
ax3 = fig.add_subplot(gs[2:3, 0:1])
ax4 = fig.add_subplot(gs[0:1, 1:2])
ax5 = fig.add_subplot(gs[1:2, 1:2])
ax6 = fig.add_subplot(gs[2:3, 1:2])



ax1.tick_params(labelbottom=False)
#ax2.tick_params(labelbottom=False)
#ax3.tick_params(labelbottom=False, labelright=True, labelleft=False)
ax4.tick_params(labelright=True, labelbottom=False, labelleft=False)
ax5.tick_params(labelright=True, labelbottom=False, labelleft=False)
ax6.tick_params(labelright=True, labelleft=False)
ax4.yaxis.set_label_position("right")
ax5.yaxis.set_label_position("right")
ax6.yaxis.set_label_position("right")

ax1.minorticks_on()
ax2.minorticks_on()
ax3.minorticks_on()
ax4.minorticks_on()
ax5.minorticks_on()
ax6.minorticks_on()

ax1.grid(True, alpha = 0.5, linewidth=0.5)
ax2.grid(True, alpha = 0.5, linewidth=0.5)
ax3.grid(True, alpha = 0.5, linewidth=0.5)
ax4.grid(True, alpha = 0.5, linewidth=0.5)
ax5.grid(True, alpha = 0.5, linewidth=0.5)
ax6.grid(True, alpha = 0.5, linewidth=0.5)

#X and Y ranges
z_max = 33
ax1.set_xlim(4, z_max)
ax2.set_xlim(4, z_max)
ax3.set_xlim(4, z_max)
ax4.set_xlim(4, z_max)
ax5.set_xlim(4, z_max)
ax6.set_xlim(4, z_max)


#Axes' labels
#ax2.set_xlabel(r'$z$')
ax1.set_ylabel(r'$\overline{\delta T_b}^2 \Delta_{21}^2~{\rm [\%]}$')
ax2.set_ylabel(r'$\overline{\delta T_b}~{\rm [\%]}$')
ax3.set_ylabel(r'$f_{\rm heat}~{\rm [\%]}$')
ax3.set_xlabel(r'$z$')
ax6.set_xlabel(r'$z$')
ax4.set_ylabel(r'$\rm x_{HII}~{\rm [\%]}$', rotation=-90, labelpad=20)
ax5.set_ylabel(r'${\rm T_{\rm K}}~{\rm [\%]}$', rotation=-90, labelpad=20)
ax6.set_ylabel(r'${\rm T_{\rm S}}~{\rm [\%]}$', rotation=-90, labelpad=20)


my_colors = ['#D90000','#2574d4','#20b356', '#f07000', '#8e006f', 'cyan', 'chartreuse']
colors=['k', *my_colors]
solid_lines=[ Line2D([],[],color=c,linestyle='-',linewidth=1) for c in colors ]
# -------------------------------------------------------------------------------------



# The power spectrum
for i, z in enumerate(z_PS): 
    z_comp = np.logspace(np.log10(5.), np.log10(z_max), 65)
    delta_z_interp_exact  = interpolate.interp1d(z[0], delta_z[i][0])(z_comp)
    delta_z_interp_approx = interpolate.interp1d(z[1], delta_z[i][1])(z_comp)
    res_delta_z = [ 100*(delta_z_interp_approx[j] - dz)/dz for j, dz in enumerate(delta_z_interp_exact) ]
    ax1.plot(z_comp, res_delta_z, color=colors[i])


# The global brighness temperature
for i, z in enumerate(z_GQ): 
    z_comp = np.logspace(np.log10(8.), np.log10(z_max), 65)
    dTb_interp_exact  = interpolate.interp1d(z[0], dTb[i][0])(z_comp)
    dTb_interp_approx = interpolate.interp1d(z[1], dTb[i][1])(z_comp)
    res_dTb = [ 100*(dTb_interp_approx[j] - dTb)/dTb for j, dTb in enumerate(dTb_interp_exact) ]  
    ax2.plot(z_comp, res_dTb ,color=colors[i])



# The energy injected into heat
for i, z in enumerate(z_f):    
    z_comp = np.logspace(np.log10(5.), np.log10(z_max), 400)
    fheat_exact  = f_heat[i][0]
    fheat_approx = f_heat[i][1]
    fheat_interp_exact  = interpolate.interp1d(z[0], fheat_exact )(z_comp)
    fheat_interp_approx = interpolate.interp1d(z[1], fheat_approx)(z_comp)
    res_fheat = [ 100*(fheat_interp_approx[j] - inj)/inj for j, inj in enumerate(fheat_interp_exact) ]  
    index_color = i if i < position_noDM else i+1
    ax3.plot(z_comp, res_fheat, color=colors[index_color])


# The ionized fractions
for i, z in enumerate(z_GQ):
    z_comp = np.logspace(np.log10(5.), np.log10(z_max), 65)
    xHII_interp_exact  = interpolate.interp1d(z[0], 1-xH_box[i][0])(z_comp)
    xHII_interp_approx = interpolate.interp1d(z[1], 1-xH_box[i][1])(z_comp)
    res_xHII = [ 100*(xHII_interp_approx[j] - xHII)/xHII for j, xHII in enumerate(xHII_interp_exact) ] 
    #ax4.plot(z, x_e_box[i],"-", color=colors[i])
    ax4.plot(z_comp, res_xHII, "-", color=colors[i])

# The temperatures
for i, z in enumerate(z_GQ):  
    z_comp = np.logspace(np.log10(5.), np.log10(z_max), 65)
    Tk_interp_exact  = interpolate.interp1d(z[0], Tk_box[i][0])(z_comp)
    Tk_interp_approx = interpolate.interp1d(z[1], Tk_box[i][1])(z_comp)
    res_Tk = [ 100*(Tk_interp_approx[j] - Tk)/Tk for j, Tk in enumerate(Tk_interp_exact) ]   
    ax5.plot(z_comp, res_Tk ,"-", color=colors[i])
    
for i, z in enumerate(z_GQ):   
    z_comp = np.logspace(np.log10(5.), np.log10(z_max), 65)
    Ts_interp_exact  = interpolate.interp1d(z[0], Ts_box[i][0])(z_comp)
    Ts_interp_approx = interpolate.interp1d(z[1], Ts_box[i][1])(z_comp)
    res_Ts = [ 100*(Ts_interp_approx[j] - Ts)/Ts for j, Ts in enumerate(Ts_interp_exact) ]   
    ax6.plot(z_comp, res_Ts ,"-", color=colors[i])



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

fig.suptitle(r"${\rm Residuals:}~\chi \to e^+ e^-,~m_\chi = 126~{\rm MeV},~\tau = 10^{26}~{\rm s} $", fontsize=18, y=1.01)
legend_text_1 = [r"$\rm DarkHistory$", r"$\rm Templates$"]
legend_text_2 = [r"$\rm without~DM~inj.$", r"$\rm no~backreaction$", r"$\rm with~backreaction$"]
legend11 = ax1.legend([line1, line2], legend_text_1, loc='upper left', bbox_to_anchor=(0.01,1.35), ncol=2)
legend12 = ax1.legend([line3, line4, line5], legend_text_2, loc='upper left', bbox_to_anchor=(0.01,1.20), ncol=3)
legend13 = ax1.legend([no_line], [r"$k = {:2.2f}~\rm Mpc^{{-1}}$".format(k_approx[0][0])], handlelength=0, handletextpad=0, loc='lower left', bbox_to_anchor=(0.01,0.01))

remove_handles_legend(legend13)
ax1.add_artist(legend11)
ax1.add_artist(legend12)



# -------------------------------------------------------------------------------------

# Saving the figure

fig.savefig('./example_plot_1.pdf', bbox_inches='tight')

## END OF SCRIPT
