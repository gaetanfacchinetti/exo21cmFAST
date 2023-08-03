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


import py21cmfast as p21f

##################
## README
##
## In order to run this code according to you configuration you can modify :
## - the output_location variable
## - the cache_location variable
## - the user and astro params as well as the flag options
##
## WARNING: don't forget to change the user param "N_THREADS" 
## according to the value set on the cluster
##
## For HII_DIM =128, with USE_TS_FLUCT = True, USE_MASS_DEPENDENT_ZETA = True,
## USE_MINI_HALOS = False, and USE_DM_ENERGY_INJECTION = True,
## the code takes around 15 minutes to run on 8 cores
## (and asks for around 22Gb of memory) 
##################

# Define the database location
output_location = "/home/ulb/physth_fi/gfacchin/exo21cmFAST_release/output/"
cache_location = "/scratch/ulb/physth_fi/gfacchin/output_exo21cmFAST"
lightcone_name = "lightcone_fiducial"


# Run the lightcone accordingly
lightcone = p21f.run_lightcone(
    redshift = 5,        # Minimal value of redshift -> Here we will go slightly lower (cannot go below z = 4 for DarkHistory)
    max_redshift = 35,   # Maximal value of redshift -- By default Z_HEAT_MAX for None of when USE_TS_FLUCT = True
    user_params = {
        "BOX_LEN":                  250,   # Default value: 300 (Box length Mpc) 1000 / 500
        "HII_DIM":                  128,   # Default value: 200  (HII cell resolution) 350 / 256
        "USE_FFTW_WISDOM":          True,  # Default value: False (Speed up FFT)
        "PERTURB_ON_HIGH_RES":      True,  # Default value: False (Turn on perturbations on high res grid)
        "USE_INTERPOLATION_TABLES": True,  # Default value: True (Use interpolated quantities)
        "N_THREADS":                1,     # Default value: 1 (Number of threads)
    },
    astro_params = {
        "t_STAR"     : 0.5,
        "F_STAR10"   : -1.3,    
        "F_ESC10"    : -1.0,
        "ALPHA_STAR" : 0.5,  
        "ALPHA_ESC"  : -0.5,  
        "L_X"        : 40.0,  
        "M_TURN"     : 8.7,      
        "DM_LOG10_MASS" : 8.0,   # DM mass in eV
        "DM_DECAY_RATE" : 1e-60, # DM Decay rate
        "NU_X_THRESH"   : 500.0, # E_0
    },
    flag_options = {
        "USE_MINI_HALOS"          : False,        # Something for mini halos
        "USE_MASS_DEPENDENT_ZETA" : True,         # Set zeta as a mass dependent function
        "SUBCELL_RSD"             : True,         # Add sub-cell redshift-space-distortion. Only effective if USE_TS_FLUCT:True
        "INHOMO_RECO"             : True,         # Turn on inhomogeneous recombinations. Increases computation time
        "USE_TS_FLUCT"            : True,         # Turn on IGM spin temperature fluctuations
        "USE_HALO_FIELD"          : False,        # Turn on halo field / otherwise mean collase (much faster) 
        "FORCE_DEFAULT_INIT_COND" : False,        # Force the initial condition to that without DM energy injection
        "USE_DM_ENERGY_INJECTION" : True,         # Turn on DM energy injection
        "DM_PROCESS"              : 'decay',      # Energy injection process 'swave', 'decay', ... 
        "DM_PRIMARY"              : 'elec_delta', # Primary particles (see list in user_params description)
        "DM_BACKREACTION"         : False,        # Whether we include heating backreaction on the deposition fractions
        "DM_USE_DECAY_RATE"       : True,         # Parametrize the decay in terms of the decay rate and not the lifetime
    },
    coarsen_factor       = 16,  # Input factor that determine the redshift steps (put 16 to roughly have the default 21cmFAST value)
    lightcone_quantities = ('brightness_temp', 'xH_box',),
    global_quantities    = ('brightness_temp', 'density', 'xH_box', 'x_e_box', 'Ts_box', 'Tk_box'),
    verbose_ntbk = False,
    direc=cache_location,
    random_seed=1993,    # random seed of the run
)

lightcone.save(fname = lightcone_name + ".h5", direc = output_location)