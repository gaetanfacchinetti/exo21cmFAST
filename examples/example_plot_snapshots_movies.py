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

#matplotlib.rc_file('matplotlibrc')

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


# Get the reference plot witout exotic energy injection
#index = db_manager(approx = False).search(bkr=False, process='none', mDM=0., primary='none', boost='none', fs_method='none', 
#                                                sigmav=0., lifetime=0., comment='large_box')[0]

#index = db_manager(approx = False).search(bkr=True, process='decay', mDM=1.26e+8, primary='elec_delta', lifetime=1e+26, comment='large_box_2')[0]
index = db_manager(approx = True).search(process='decay', mDM=1.26e+8, lifetime=1e+26, approx_shape='schechter', approx_params=[1.5310e-01,-3.2090e-03,1.2950e-01], comment='large_box_2')[0]


print(index)

# -------------------------------------------------------------------------------------

fig = plt.figure(figsize=(5,5))
ax = fig.gca()


######
## Plotting the brightness temperature for the case without DM
#lightcone = p21c.LightCone.read(fname = database_location + '/darkhistory/BrightnessTemp_' + str(index) + '/Lightcone/Lightcone.h5')
lightcone = p21c.LightCone.read(fname = database_location + '/approx/BrightnessTemp_' + str(index) + '/Lightcone/Lightcone.h5')


redshift_arr      = lightcone.lightcone_redshifts
redshift_node_arr = lightcone.node_redshifts
Tb = lightcone.brightness_temp
xH = lightcone.xH_box
Tb_global = lightcone.global_quantities['brightness_temp']

power_spectra, redshift_indices_ps = lightcone_analysis.compute_powerspectra(lightcone, n_psbins=24, nchunks=65)

kref = 0.12
power_spectrum = [0 for z in redshift_indices_ps]

for iz, z in enumerate(redshift_indices_ps):
    ps = power_spectra[iz]
    k_approx = min(ps['k'], key=lambda x:abs(x-kref))
    index_k = np.where(ps['k'] == k_approx)[0][0]
    power_spectrum[iz] = ps['delta'][index_k]

func_Tb = interpolate.interp1d(redshift_arr, Tb, axis=2) # z-axis is the number 2
func_xH = interpolate.interp1d(redshift_arr, xH, axis=2) # z-axis is the number 2
func_Tb_global = interpolate.interp1d(redshift_node_arr, Tb_global) 
func_Tb_power  = interpolate.interp1d(redshift_indices_ps, power_spectrum) 

x_arr = linspace(0, lightcone.cell_size * lightcone.shape[0], lightcone.shape[0])
y_arr = linspace(0, lightcone.cell_size * lightcone.shape[1], lightcone.shape[1])

number_images = 1200
i = number_images

"""
for z in np.linspace(4, 30, number_images):
    
    ax.cla()
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])

    index_image_str = f'{i:05d}'


    ax.pcolormesh(x_arr, y_arr, func_Tb(z), cmap="EoR", vmin = -150, vmax= 30, shading='gouraud')
    fig.savefig('./figures/brightness_temperature_with_exo/Tb_index' + index_image_str + '_z' + str("{:.2e}".format(z)) + '.png', dpi=300, bbox_inches='tight',  pad_inches = 0)
    
    #ax.pcolormesh(x_arr, y_arr, func_xH(z), cmap="viridis", vmin = 0, vmax= 1, shading='gouraud')
    #fig.savefig('./figures/ionization_fraction/xH_index' + index_image_str + '_z' + str("{:.2e}".format(z)) + '.png', dpi=300, bbox_inches='tight',  pad_inches = 0)
    
    i = i-1
######
"""


fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 1], 'wspace': 0, 'hspace': 0}, sharex=True, figsize=(5,5))



i = number_images

for z in np.linspace(5, 30, number_images):
    
    ax1.cla()
    ax2.cla()
    ax1.grid(True, alpha = 0.5, linewidth=0.5)
    ax2.grid(True, alpha = 0.5, linewidth=0.5)
    ax2.set_xticks([0, 5, 10, 15, 20, 25, 30])
    ax1.set_yticks([-150, -100, -50, 0, 50])
    ax2.set_xlabel(r"$z$")
    ax1.set_ylabel(r"$\overline{\delta T_{\rm b}}~[\rm mK]$")
    ax2.set_ylabel(r'$\overline{\delta T_b}^2 \Delta_{21}^2~{\rm [mK^2]}$')
    ax2.set_yscale('log')
    ax1.set_xlim(0, 35)
    ax1.set_ylim(-180, 50)
    ax2.set_ylim(2e-2, 1e+3)

    index_image_str = f'{i:05d}'

    z_arr = np.arange(z, 34, 0.1)
    ax1.scatter(z_arr, func_Tb_global(z_arr), c=func_Tb_global(z_arr), cmap="EoR", vmin = -150, vmax= 30)
    ax2.scatter(z_arr, func_Tb_power(z_arr), c='k')
    fig.savefig('./figures/brightness_temperature_global_with_exo/Tb_index' + index_image_str + '_z' + str("{:.4e}".format(z)) + '.png', dpi=300, bbox_inches='tight',  pad_inches = 0.1)
    i = i-1

## END OF SCRIPT
