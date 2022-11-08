import os

import numpy as np
from scipy import interpolate
from astropy import units
import ast

try:
    import py21cmsense as p21s
except:
    pass
    
from py21cmfishlite import tools as p21fl_tools


def define_HERA_observations(z_arr):
    """ 
    Define a set of HERA observation objects (see 21cmSense) 
    according to an array of observation redshifts 

    Parameters
    ----------
        z_arr: list (or numpy array) of floats
            redshifts at which the observations are done    
    """

     ## Define the layout for the 
    hera_layout = p21s.antpos.hera(
        hex_num = 11,             # number of antennas along a side
        separation= 14 * units.m,  # separation between antennas (in metres)
        dl=12.12 * units.m        # separation between rows
    )

    hera = []

    for iz, z in enumerate(z_arr):
        beam = p21s.beam.GaussianBeam(
            frequency = 1427.5831 * units.MHz / (1+z),  # just a reference frequency
            dish_size = 14 * units.m
        )

        hera.append(p21s.Observatory(
            antpos = hera_layout,
            beam = beam,
            latitude = 0.536189 * units.radian,
            Trcv = 100 * units.K
        ))

    observations = []

    for iz, z in enumerate(z_arr):
        observations.append(p21s.Observation(
            observatory   = hera[iz],
            n_channels    = 80, 
            bandwidth     = 8 * units.MHz,
            time_per_day  = 6 * units.hour,   # Number of hours of observation per day
            n_days        = 166.6667,         # Number of days of observation
        ))

    return observations


def extract_noise_from_fiducial(z_arr, k_arr, delta_arr, observations) :

    #folder_name = '/home/ulb/physth_fi/gfacchin/exo21cmFAST_release/output/test_database/darkhistory/BrightnessTemp_12/power_spectra/'

    sensitivities = []
    power_std     = []

    for iz, z in enumerate(z_arr):
        sensitivities.append(p21s.PowerSpectrum(observation=observations[iz], k_21 = k_arr[iz] / units.Mpc, delta_21 = delta_arr[iz] * (units.mK**2), foreground_model='moderate')) # Use the default power spectrum here
        power_std.append(sensitivities[iz].calculate_sensitivity_1d())

    return sensitivities, power_std



def evaluate_fisher_matrix(dir_path: str, observatory: str = None):
    """
    Main function that evaluates the Fisher matrix from the set of power spectra in folder

    Parameters:
        evaluate_fis

    """
    
    # Read all the 
    # list to store files
    key_arr = []
    val_arr = []
    dir_arr = []
    dir_fid = ""

    path_arr = sorted(os.listdir(dir_path + '/output_list'))
    #print(path_arr)

    # Iterate directory
    for path in path_arr:
        bit = path.split('_')[3:]
        if bit[-1] == 'fid':
            dir_fid = path
        else:
            dir_arr.append(path)
            key_arr.append('_'.join(bit[:-1]))
            val_arr.append(float(bit[-1]))

    
    #print(key_arr, val_arr, dir_arr)
      # initialize a null list
    key_arr_unique = []
  
    # traverse for all elements
    for x in key_arr:
        # check if exists in unique_list or not
        if x not in key_arr_unique:
            key_arr_unique.append(x)
    
    n_keys = len(key_arr_unique)

    z_arr_m     = dict()
    z_arr_p     = dict()
    k_arr_m     = dict() 
    k_arr_p     = dict()
    delta_arr_m = dict()
    delta_arr_p = dict()

    z_arr_fid     = []
    k_arr_fid     = []
    delta_arr_fid = []

    delta_func_p   = dict()
    delta_func_m   = dict()
    delta_func_fid = []

    val_arr_m     = dict()
    val_arr_p     = dict()

    for ikey, key in enumerate(key_arr): 

        if val_arr[ikey] > 0 : 
            
            # Read power spectra from the file
            z_arr_p[key], k_arr_p[key], delta_arr_p[key], _ = p21fl_tools.read_power_spectra(dir_path + '/output_list/' + dir_arr[ikey])
            
            delta_func_p[key] = []
            for iz, _ in enumerate(z_arr_p[key]):
                # Define the power spectra interpolation of the (+) models
                delta_func_p[key].append(interpolate.interp1d(k_arr_p[key][iz], delta_arr_p[key][iz]))

            val_arr_p[key] = val_arr[ikey]

        elif val_arr[ikey] < 0 : 
            
            # Read power spectra from the file
            z_arr_m[key], k_arr_m[key], delta_arr_m[key], _ = p21fl_tools.read_power_spectra(dir_path + '/output_list/' + dir_arr[ikey])
            
            delta_func_m[key] = []
            for iz, _ in enumerate(z_arr_m[key]):
                # Define the power spectra interpolation of the (-) models
                delta_func_m[key].append(interpolate.interp1d(k_arr_m[key][iz], delta_arr_m[key][iz]))
        
            val_arr_m[key] = val_arr[ikey]
    
    z_arr_fid, k_arr_fid, delta_arr_fid, _ = p21fl_tools.read_power_spectra(dir_path + '/output_list/' + dir_fid)

    for iz, _ in enumerate(z_arr_fid):
        # Define the power spectra interpolation of the fiducial models 
        delta_func_fid.append(interpolate.interp1d(k_arr_fid[iz], delta_arr_fid[iz]))

    ## Getting the fiducial parameters in the fiducial params.txt file
    ## Note that for now the astro params are put at the end of the file
    ## If fiducial params modified by hand need to be careful
    fiducial_params = []
    with open(dir_path + '/fiducial_params.txt') as f:
        data_lines = f.readlines()
        for data in data_lines:
            if data[0] != '#':
                fiducial_params.append(ast.literal_eval(data))

    fiducial_params = fiducial_params[-1] ## Only keep the astro params


    littleh : float = p21s.config.COSMO.h
    
    #### Now from the redshift array and the instrument we get the (k, z) arrays and the noise from 21cmSense
      
    if observatory is None:
        observatory = 'hera'
    
    if observatory.upper() != 'HERA': 
        raise ValueError("This observatory is not preimplemented")
    
    observations             = define_HERA_observations(z_arr_fid)
    sensitivities, power_std = extract_noise_from_fiducial(z_arr_fid, k_arr_fid, delta_arr_fid, observations)
    
    k_fish     = [sensitivity.k1d.value * littleh for sensitivity in sensitivities]


    fisher_mat = np.zeros((n_keys, n_keys))

    file_temp = open(dir_path + "/my_temp_file.txt", 'w')

    for iz, z in enumerate(z_arr_fid):
        for jk, k in enumerate(k_fish[iz]):
            # Now we are cooking with gaz!

            for kkey1, key1 in enumerate(key_arr_unique):

                dp1 = delta_func_p[key1][iz](k)
                dm1 = delta_func_m[key1][iz](k)
                
                deriv_1 = (dp1 - dm1)/((val_arr_p[key1] - val_arr_m[key1])*fiducial_params[key1]) # derivative with respect to the first parameter
                
                for kkey2, key2 in enumerate(key_arr_unique):

                    dp2 = delta_func_p[key2][iz](k)
                    dm2 = delta_func_m[key2][iz](k)
                    
                    deriv_2 = (dp2 - dm2)/((val_arr_p[key2] - val_arr_m[key2])*fiducial_params[key2]) # derivative with respect to the first parameter
                    
                    print(z, k, key1, key2, deriv_1, deriv_2, val_arr_p[key1], val_arr_p[key2], power_std[iz][jk].value, file=file_temp)

                    if not np.isinf(power_std[iz][jk].value):
                        fisher_mat[kkey1, kkey2] =  fisher_mat[kkey1, kkey2] + deriv_1 * deriv_2 / power_std[iz][jk].value


    print(fisher_mat)