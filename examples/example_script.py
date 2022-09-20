import py21cmfast as p21c
import numpy as np
output_dir = './'

import logging
logger = logging.getLogger('21cmFAST')
logger.setLevel(logging.INFO)

BOX_LEN = 30
HII_DIM = 10
random_seed = 1993
approx = True

p21c.global_params.XION_at_Z_HEAT_MAX = 118.0
p21c.global_params.TK_at_Z_HEAT_MAX   = 0.0031

user_params = {
        "BOX_LEN":                  BOX_LEN,  # Default value: 300  (Box length Mpc) 1000
        "HII_DIM":                  HII_DIM,  # Default value: 200  (HII cell resolution) 350
        "USE_FFTW_WISDOM":          True,     # Default value: False (Speed up FFT)
        "N_THREADS":                1,        # Default value: 1 (Number of threads)
        "PERTURB_ON_HIGH_RES":      True,     # Default value: False (Turn on perturbations on high res grid)
        "USE_INTERPOLATION_TABLES": True,     # Default value: True (Use interpolated quantities)

        ######################## Fiducial models Gaetan chose ######################
        ## -- Parameters of the DM model : specific to exo21cmFAST
        "DM_MASS":         1.26e8,       # DM mass in eV
        "DM_PROCESS":      'decay',   # Energy injection process 'swave', 'decay', ... 
        #"DM_SIGMAV":       0,    # Annihilation cross-section (in cm^3/s) | relevant only if DM_PROCESS = 'swave' 
        "DM_LIFETIME":     1e26,  # Lifetime | relevant only if DM_PROCESS = 'decay'

        # Specific to DarkHistory
        "DM_PRIMARY":      'elec_delta'  if (not approx) else 'none',  # Primary particles (see list in user_params description)
        "DM_BOOST":        'none'        if (not approx) else 'none',  # Annihilation boost | relevant only if DM_PROCESS = 'swave' 
        "DM_FS_METHOD":    'no_He'       if (not approx) else 'none',  # Method to compute the energy deposition (see DarkHistory doc)
        "DM_BACKREACTION":  False        if (not approx) else False,   # Turns on backreaction

        # Specific to an approximative energy deposition
        "DM_FHEAT_APPROX_SHAPE":  'schechter'       if approx else 'none',     # Shape of the template for f_heat
        "DM_FHEAT_APPROX_PARAMS": [1.5310e-01, -3.2090e-03, 1.2950e-01]      if approx else [0.],       # Parameters (list) to feed to the template of fheat 
        'DM_FION_H_OVER_FHEAT':   -1  if approx else -1,         # Ratio of f_ion_H over fheat  (if < 0 use values tabulated with DarkHistory)
        'DM_FION_HE_OVER_FHEAT':  -1  if approx else -1,         # Ratio of f_ion_He over fheat (if < 0 use values tabulated with DarkHistory)
        'DM_FEXC_OVER_FHEAT':     -1  if approx else -1,         # Ratio of fexc over fheat     (if < 0 use values tabulated with DarkHistory)
        ######################## End of Fiducial models Gaetan chose ######################
        }

flag_options = {
        "USE_MINI_HALOS":           True, # Something for mini halos
        "USE_MASS_DEPENDENT_ZETA":  True, # I don't understand this one
        "SUBCELL_RSD":              True, # Add sub-cell redshift-space-distortion. Only effective if USE_TS_FLUCT:True
        "INHOMO_RECO":              True, # Turn on inhomogeneous recombinations. Increases computation time
        "USE_TS_FLUCT":             True,  # Turn on IGM spin temperature fluctuations
        "PHOTON_CONS":              False, # Turn on a small correction to account for photon non conservation

        ## -- Parameters of the DM model : specific to exo21cmFAST
        "USE_DM_ENERGY_INJECTION" : True  , # Turns on DM energy injection
        "USE_EFFECTIVE_DEP_FUNCS" : False   if (not approx) else True ,
        "FORCE_INIT_COND"         : False   if (not approx) else True
        }

initial_conditions = p21c.initial_conditions(user_params=user_params,random_seed=random_seed, direc=output_dir, regenerate=False)

# Run the lightcone accordingly
lightcone, output_DH = p21c.run_lightcone(
    redshift = 6,  # Minimal value of redshift -> Here we will go slightly lower (cannot go below z = 4 for DarkHistory)
    user_params = user_params,
    flag_options = flag_options,
    init_box = initial_conditions,
    coarsen_factor=16,  # Input factor that determine the redshift steps (put 16 to roughly have the default 21cmFAST value)
    random_seed = random_seed,
    lightcone_quantities=('brightness_temp', 'xH_box',),
    global_quantities=('brightness_temp', 'density', 'xH_box', 'x_e_box', 'Ts_box', 'Tk_box'),
    direc=output_dir
)

lightcone.save(direc=output_dir)
np.save('DH', output_DH)