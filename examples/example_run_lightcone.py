############################################################################
#    Code to run lightcones in command line 
#    with dark matter energy injection
#
#    Copyright (C) 2022 Gaetan Facchinetti
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


import py21cmfast     as p21f
import py21cmanalysis as p21a
#import example_lightcone_analysis as lightcone_analysis

import os
import logging
import numpy as np


import traceback
import argparse

logger = logging.getLogger(__name__)


# Define the database location
database_location = "/home/ulb/physth_fi/gfacchin/exo21cmFAST_release/output/test_database"
cache_location = "/scratch/ulb/physth_fi/gfacchin/output_exo21cmFAST"



####################### INITIALISATION #######################

# Read the input from the command line
args = p21f.dm_dtb_tools.parse_args(argparse.ArgumentParser())
approx = args.approximate

db_manager = p21f.dm_dtb_tools.DHDatabase(path = database_location, cache_path = cache_location) if approx is False else p21f.dm_dtb_tools.ApproxDepDatabase(path = database_location, cache_path = cache_location)
input, force_overwrite, nomp = db_manager.define_models(args)
input_models = db_manager.add_models_database(input, force_overwrite=force_overwrite)
logger.info(" --------------------- ")

#############################################################


# We can loop here on the models
for current_model in input_models :

    ##################### EVALUATION ############################

    try:

        # Set the directory to our custom cache directory
        cache_direc = db_manager.cache_path + "/_cache_" + str(current_model.index)
        if not os.path.exists(cache_direc): os.mkdir(cache_direc)
        p21f.config['direc'] = cache_direc

        # If we do not specify anything it is that we do not care of the DM
        dm_energy_inj = True if (not current_model.process == 'none') else False

        # Run the lightcone accordingly
        lightcone, output_exotic_energy_injection = p21f.run_lightcone(
            redshift = 5,        # Minimal value of redshift -> Here we will go slightly lower (cannot go below z = 4 for DarkHistory)
            max_redshift = None, # Maximal value of redshift -- By default Z_HEAT_MAX for None of when USE_TS_FLUCT = True
            user_params = {
                "BOX_LEN":                  500,   # Default value: 300 (Box length Mpc) 1000
                "DIM":                      None,  # Default value: None / gives DIM=3*HII_DIM (High resolution) None
                "HII_DIM":                  256,   # Default value: 200  (HII cell resolution) 350
                "USE_FFTW_WISDOM":          False, # Default value: False (Speed up FFT)
                "HMF":                      1,     # Default value: 1 (Halo mass function)
                "USE_RELATIVE_VELOCITIES":  False, # Default value: False (Turn on relative velocites)  -> Attention if USE_RELATIVE_VELOCITIES: True, POWER_SPECTRUM: 5 (CLASS) necessarily
                "POWER_SPECTRUM":           0,     # Default value: 0 (Power spectrum used, by default Eisenstein and Hu)
                "N_THREADS":                nomp,  # Default value: 1 (Number of threads)
                "PERTURB_ON_HIGH_RES":      True,  # Default value: False (Turn on perturbations on high res grid)
                "NO_RNG":                   False, # Default value: False (Turn off random number generation -- for debugging)
                "USE_INTERPOLATION_TABLES": True,  # Default value: True (Use interpolated quantities)
                "FAST_FCOLL_TABLES":        False, # Default value: False (Something for mini halos)
                "USE_2LPT":                 True,  # Default value: True (Turn on second order lagrangian perturbation theory)
                "MINIMIZE_MEMORY":          False, # Default value: False (Reduce memory usage -- good for small computers)
            },

            astro_params = {
                "LOG10_XION_at_Z_HEAT_MAX"  : np.log10(current_model.xe_init) if approx else -99, # Only effective is USE_CUSTOM_INIT_COND = True
                "LOG10_TK_at_Z_HEAT_MAX"    : np.log10(current_model.Tm_init) if approx else -99, # Only effective is USE_CUSTOM_INIT_COND = True

                # --------------------------------------------------------------------------------------------------- #
                "DM_LOG10_MASS"     : np.log10(current_model.mDM),                                                    # DM mass in eV
                "DM_LOG10_SIGMAV"   : np.log10(current_model.sigmav)   if current_model.process == 'swave' else -99,  # Annihilation cross-section (in cm^3/s) | relevant only if DM_PROCESS = 'swave' 
                "DM_LOG10_LIFETIME" : np.log10(current_model.lifetime) if current_model.process == 'decay' else -99,  # Lifetime | relevant only if DM_PROCESS = 'decay'

                "DM_FHEAT_APPROX_PARAM_LOG10_F0" : np.log10(current_model.approx_params[0])   if (approx and len(current_model.approx_params) > 0) else -99,    # Parameter to feed to the template of fheat
                "DM_FHEAT_APPROX_PARAM_A"        : current_model.approx_params[1]             if (approx and len(current_model.approx_params) > 1) else -99,    # Parameter to feed to the template of fheat
                "DM_FHEAT_APPROX_PARAM_B"        : current_model.approx_params[2]             if (approx and len(current_model.approx_params) > 2) else -99,    # Parameter to feed to the template of fheat
                "DM_LOG10_FION_H_OVER_FHEAT"     : np.log10(current_model.fion_H_over_fheat)  if approx and current_model.fion_H_over_fheat > 0    else -99,    # Ratio of f_ion_H over fheat  (if < 0 use values tabulated with DarkHistory)
                "DM_LOG10_FION_HE_OVER_FHEAT"    : np.log10(current_model.fion_He_over_fheat) if approx and current_model.fion_He_over_fheat > 0   else -99,    # Ratio of f_ion_He over fheat (if < 0 use values tabulated with DarkHistory)
                "DM_LOG10_FEXC_OVER_FHEAT"       : np.log10(current_model.fexc_over_fheat)    if approx and current_model.fexc_over_fheat > 0      else -99,    # Ratio of fexc over fheat     (if < 0 use values tabulated with DarkHistory)
                # --------------------------------------------------------------------------------------------------- #
            },

            flag_options = {
                "USE_MINI_HALOS"          : False,                            # Something for mini halos
                "USE_MASS_DEPENDENT_ZETA" : True,                             # Set zeta as a mass dependent function
                "SUBCELL_RSD"             : True,                             # Add sub-cell redshift-space-distortion. Only effective if USE_TS_FLUCT:True
                "INHOMO_RECO"             : True,                             # Turn on inhomogeneous recombinations. Increases computation time
                "USE_TS_FLUCT"            : True,                             # Turn on IGM spin temperature fluctuations
                "USE_HALO_FIELD"          : False,                            # Turn on halo field / otherwise mean collase (much faster) 
                "M_MIN_in_Mass"           : False,                            # If False minimal halo mass for virialisation set from temperature
                "PHOTON_CONS"             : False,                            # Turn on a small correction to account for photon non conservation
                "FIX_VCB_AVG"             : False,                            # 
                "FORCE_DEFAULT_INIT_COND" : False,                            # Force the initial condition to that without DM energy injection
                "USE_CUSTOM_INIT_COND"    : current_model.force_init_cond if approx else False,    # Force initial conditions to the value defined by 
                

                # --------------------------------------------------------------------------------------------------- #
                "USE_DM_ENERGY_INJECTION"    : dm_energy_inj,   # Turn on DM energy injection
                "USE_DM_EFFECTIVE_DEP_FUNCS" : approx,          # Treat the energy injection with approximate templates (instead of DarkHistory)
                "USE_DM_CUSTOM_F_RATIOS"     : False,           # Not really important here (this is just used for checks)
                
                "DM_PROCESS"      : current_model.process,                                 # Energy injection process 'swave', 'decay', ... 
                "DM_PRIMARY"      : current_model.primary    if (not approx) else 'none',  # Primary particles (see list in user_params description)
                "DM_BOOST"        : current_model.boost      if (not approx) else 'none',  # Annihilation boost | relevant only if DM_PROCESS = 'swave' 
                "DM_FS_METHOD"    : current_model.fs_method  if (not approx) else 'none',  # Method to compute the energy deposition (see DarkHistory doc)
                "DM_BACKREACTION" : current_model.bkr        if (not approx) else False,   # Turns on backreaction
                
                "DM_FHEAT_APPROX_SHAPE" : current_model.approx_shape  if approx else 'none',  # Shape of the template for f_heat
                # --------------------------------------------------------------------------------------------------- #
            },
            coarsen_factor       = 16,  # Input factor that determine the redshift steps (put 16 to roughly have the default 21cmFAST value)
            lightcone_quantities = ('brightness_temp', 'xH_box',),
            global_quantities    = ('brightness_temp', 'density', 'xH_box', 'x_e_box', 'Ts_box', 'Tk_box'),
            verbose_ntbk = True,
            direc=cache_direc, 
        )

        #############################################################

        ####################### ANALYSIS ############################

        # Save the result of this lightcone
        # The output from DarkHistory
        if dm_energy_inj == True : db_manager.print_f_vs_rs_from21cmFAST(output_exotic_energy_injection, current_model)

        # The output from 21cmFAST is saved in a lightcone file
        # Moreover, we prepare the folder for the analysis of this lightcone
        # We run on first analysis to have something to plot out of the box  
        path_output = db_manager.path_brightness_temp  + str(current_model.index)
        z_centers, power_spectra = p21a.compute_powerspectra_1D(lightcone=lightcone, nchunks=15, n_psbins=None, k_min=0.1, k_max=1, logk=True) # Compute the power spectra
        p21a.export_powerspectra_1D(path=path_output, z_centers = z_centers, power_spectra = power_spectra)
        
        #lightcone_analysis.make_run_directory(path_output)                                # Create/clean the directory where we store everything
        #lightcone_analysis.make_and_save_lightcone_directory(path_output, lightcone)      # Save the lightcone in a folder called Lightcone created here
        #lightcone_analysis.make_analysis(path_output, lightcone, n_psbins=24, nchunks=65) # Run a first analysis to have something to plot out of the box

        #############################################################

    except Exception as inst:
        # If something goes wrong we remove the entry in the database
        db_manager.remove_entries_database(current_model.index, force_deletion=True)
        traceback.print_exc()
        logger.error("\n------------------------------")
        logger.error("Entry not added to the database")
        exit(0)


   
