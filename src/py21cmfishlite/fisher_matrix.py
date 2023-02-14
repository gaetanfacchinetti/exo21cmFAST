import os
import glob

import numpy as np
from scipy import interpolate
from scipy import optimize
from astropy import units
import ast

import py21cmsense    as p21s
import py21cmfast     as p21f
import py21cmanalysis as p21a

from astropy.cosmology import Planck18 as cosmo
from astropy import units
from astropy import constants

#except:
 #   pass
    
from py21cmfishlite import tools as p21fl_tools


def define_HERA_observation(z):
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


    beam = p21s.beam.GaussianBeam(
        frequency = 1420.40575177 * units.MHz / (1+z),  # just a reference frequency
        dish_size = 14 * units.m
    )

    hera = p21s.Observatory(
        antpos = hera_layout,
        beam = beam,
        latitude = 0.536189 * units.radian,
        Trcv = 100 * units.K
    )

    observation = p21s.Observation(
        observatory   = hera,
        n_channels    = 80, 
        bandwidth     = 8 * units.MHz,
        time_per_day  = 6 * units.hour,   # Number of hours of observation per day
        n_days        = 166.6667,         # Number of days of observation
    )

    return observation



def extract_noise_from_fiducial(k, dsqr, observation) :
    """
    Give the noise associated to power spectra delta_arr

    Params:
    -------
    k       : list of floats 
        array of modes k in [Mpc^{-1}]
    dsqr    : list of floats 
        Power spectrum in [mK^2] ordered with the modes k in 
    observation : Observation object (c.f. 21cmSense)

    Returns:
    --------
    k_sens       : list of list of floats
    std_sens     : the standard deviation of the sensitivity [mK]

    """


    sensitivity       = p21s.PowerSpectrum(observation=observation, k_21 = k / units.Mpc, delta_21 = dsqr * (units.mK**2), foreground_model='moderate') # Use the default power spectrum here
    k_sens            = sensitivity.k1d.value * p21s.config.COSMO.h
    std_21cmSense     = sensitivity.calculate_sensitivity_1d().value

    std = interpolate.interp1d(k_sens, std_21cmSense, bounds_error=False, fill_value=np.nan)(k)

    return std 




def define_grid_modes_redshifts(z_min: float, B: float, k_min = 0.1 / units.Mpc, k_max = 1 / units.Mpc, z_max: float = 19, logk=False) : 
    """
    Defines a grid of modes and redshift on which to define the noise and on which the Fisher matrix will be evaluated
    
    Params:
    ------
    z_min : float
        Minimal redshift on the grid
    B     : float
        Bandwidth of the instrument

    """

    # Definition of the 21 cm frequency (same as in 21cmSense)
    f21 = 1420.40575177 * units.MHz

    def deltak_zB(z, B) : 
        return 2*np.pi * f21 * cosmo.H(z) / constants.c / (1+z)**2 / B * 1000 * units.m / units.km

    def generate_z_bins(z_min, z_max, B):
        fmax = int(f21.value/(1+z_min))*f21.unit
        fmin = f21/(1+z_max)
        f_bins    = np.arange(fmax.value, fmin.value, -B.value) * f21.unit
        f_centers = f_bins[:-1] - B/2
        z_bins    = (f21/f_bins).value - 1
        z_centers = (f21/f_centers).value -1
        return z_bins, z_centers
    
    def generate_k_bins(z_min, k_min, k_max, B):
        
        dk = deltak_zB(z_min, B) 
        _k_min = dk
        n = 1

        while _k_min < k_min:
            _k_min = n*dk
            n = n+1
        
        _k_min = (n-1)*dk

        if logk is False:
            return np.arange(_k_min.value, k_max.value, dk.value) * k_min.unit
        else:
            return np.logspace(np.log10((n-1)*dk.value), )

    # Get the redshift bin edges and centers
    z_bins, _ = generate_z_bins(z_min, z_max, B)
    
    # Get the k-bins edges
    k_bins = generate_k_bins(z_min, k_min, k_max, B)

    return z_bins, k_bins



def compute_power_spectrum_from_bins(lightcone, z_bins, k_bins, logk): 
    """
    ## Generic function to evaluate the powe spectrum from precomputed bins
    """
    # Define the chunck indices according to the definition of the bins
    lc_redshifts = lightcone.lightcone_redshifts
    chunk_indices = [np.argmin(np.abs(lc_redshifts - z)) for z in z_bins]

    # Compute the power spectrum on the redshift chuncks
    z_arr, ps = p21a.compute_powerspectra_1D(lightcone, chunk_indices = chunk_indices, n_psbins=k_bins.value, logk=logk, remove_nans=False, vb=False)
    
    return z_arr, ps




    


class Run:

    def __init__(self, lightcone, z_bins, k_bins, logk): 
        
        self._z_bins    = z_bins
        self._k_bins    = k_bins
        self._logk      = logk

        # Get the power spectrum from the Lightcone
        self._lightcone       = lightcone
        self._z_arr, self._ps = compute_power_spectrum_from_bins(self._lightcone, self._z_bins, self._k_bins, logk=logk)

    @property
    def power_spectrum(self):
        return [data['delta'] for data in self._ps]

    @property
    def power_spectrum_errors(self):
        return [data['err_delta'] for data in self._ps]

    @property
    def z_bins(self):
        return self._z_bins

    @property
    def k_bins(self): 
        return self._k_bins

    @property
    def z_array(self):
        return self._z_arr

    @property
    def k_array(self):
        return [data['k'] for data in self._ps]

    @property
    def logk(self): 
        return self._logk



class Fiducial(Run): 

    def __init__(self, dir_path, z_bins, k_bins, logk, observation = None):

        self._dir_path     = dir_path
        self._lightcone    = p21f.LightCone.read(self._dir_path + "/Lightcone_FIDUCIAL.h5")
        self._astro_params = self._lightcone.astro_params.pystruct
        self._observation  = observation

        super().__init__(self._lightcone, z_bins, k_bins, logk)
        self.compute_sensitivity()
    
    @property
    def dir_path(self):
        return self._dir_path

    @property
    def astro_params(self):
        return self._astro_params

    @property
    def observation(self):
        return self._observation

    @observation.setter
    def observation(self, value):
        _old_value = self._observation
        if _old_value != value : 
            self._observation = value
            self.compute_sensitivity()

    @property
    def ps_std(self):
        return self._ps_std

    def compute_sensitivity(self):

        _std = None

        if self._observation == 'HERA':
            _std = [None] * len(self.z_array)
            for iz, z in enumerate(self.z_array): 
                _hera     = define_HERA_observation(z)
                _std[iz]  = extract_noise_from_fiducial(self.k_array[iz], self.power_spectrum[iz], _hera)

        self._ps_std = _std


    def plot_power_spectrum(self, obs = None):
    
        fig = p21fl_tools.make_figure_power_spectra(self.k_array,  self.power_spectrum,  self.z_array, std = self._ps_std)
        fig.savefig(self._dir_path + "/power_spectrum.pdf", tight_layout=True)


class Parameter:

    def __init__(self, fiducial, name):
        
        self._fiducial = fiducial
        self._name     = name
        
        self._dir_path     = self._fiducial.dir_path
        self._astro_params = self._fiducial.astro_params
        self._z_bins       = self._fiducial.z_bins
        self._k_bins       = self._fiducial.k_bins
        self._logk         = self._fiducial.logk

        if name not in self._astro_params:
            ValueError("ERROR: the name does not corresponds to any varied parameters")

        # get the lightcones from the filenames
        _lightcone_file_name = glob.glob(self._dir_path + "/Lightcone_" + self._name + "_*.h5")
        
        q_value = []
        for file_name in _lightcone_file_name:
            q_value.append(float(file_name.split("_")[-1].split(".")[0]))

        # We get the lightcones and then create the corresponding runs objects
        self._lightcones =  [p21f.LightCone.read(self._dir_path + "/Lightcone_" + self._name + "_" + str(q) + ".h5") for q in q_value]
        self._runs       =  [Run(lightcone, self._z_bins, self._k_bins, self._logk) for lightcone in self._lightcones]


    def compute_derivative():
        ...

    def plot_power_spectra():
        ...




def evaluate_fisher_matrix(dir_path: str, observatory: str = None, k_min: float = 0.1, k_max: float = 1., z_min = 5, z_max = 35):
    
    """
    Main function that evaluates the Fisher matrix from the set of power spectra in folder

    Parameters:
    -----------
    dir_path: str
        path to the directory where the data is saved
    observatory: str (optional)
        observatory we consider to evaluate the experimental sensitivity
        by default HERA configuration is used
    kmin: float (Mpc^{-1}, optional)
        minimal k mode to sum over in the Fisher matrix
        by default kmin = 0.1 Mpc^{-1}
    kmax: float (Mpc^{-1}, optional)
        maximal k mode to sum over in the Fisher matrix
        by default kmax = 1 Mpc^{-1}

    Returns:
    -----------
    fisher_matrix: (n, n) numpy array
        Fisher matrix where n is the number of parameters
    key_arr: list of str (of size n)
        name of the parameters in the order corresponding to that in the Fisher matrix 
    fiducial_params: list of float (of size n)
        list of the fiducial parameters
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

    delta_func_p   = dict()
    delta_func_m   = dict()
    delta_func_fid = []
    err_func_fid   = []

    val_arr_m     = dict()
    val_arr_p     = dict()


    ## Need to change this part to read the lightcones directly

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
    
    z_arr_fid, k_arr_fid, delta_arr_fid, err_arr_fid = p21fl_tools.read_power_spectra(dir_path + '/output_list/' + dir_fid)

    for iz, _ in enumerate(z_arr_fid):
        # Define the power spectra interpolation of the fiducial models 
        delta_func_fid.append(interpolate.interp1d(k_arr_fid[iz], delta_arr_fid[iz]))
        err_func_fid.append(interpolate.interp1d(k_arr_fid[iz], err_arr_fid[iz])) # Apparently a bad idea as stated in 21cmFish to interpolate

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

    
    #### Now from the redshift array and the instrument we get the (k, z) arrays and the noise from 21cmSense
      
    if observatory is None:
        observatory = 'hera'
    
    if observatory.upper() != 'HERA': 
        raise ValueError("This observatory is not preimplemented")
    
    k_sens     = [None] * len(z_arr_fid)
    dsqr_sens  = [None] * len(z_arr_fid)
    std_sens   = [None] * len(z_arr_fid)
   
    for iz, z in enumerate(z_arr_fid) : 
        observation                              = define_HERA_observation(z)
        k_sens[iz], dsqr_sens[iz], std_sens[iz]  = extract_noise_from_fiducial(k_arr_fid[iz], delta_arr_fid[iz], observation)
    



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


    #### EVALUATE THE FISHER MATRIX

    fisher_matrix = np.zeros((n_keys, n_keys))

    file_temp = open(dir_path + "/my_temp_file.txt", 'w')

    for iz, z in enumerate(z_arr_fid):
        for jk, k in enumerate(k_sens[iz]):

            if k > k_min and k < k_max and z > z_min and z < z_max: 
            # Limit the range of k to that between 0.1 and 1 Mpc^{-1}
            # Now we are cooking with gaz!

                for kkey1, key1 in enumerate(key_arr_unique):

                    dp1 = delta_func_p[key1][iz](k)
                    dm1 = delta_func_m[key1][iz](k)
                    
                    deriv_1 = (dp1 - dm1)/((val_arr_p[key1] - val_arr_m[key1])*fiducial_params[key1]) # derivative with respect to the first parameter
                    
                    for kkey2, key2 in enumerate(key_arr_unique):

                        dp2 = delta_func_p[key2][iz](k)
                        dm2 = delta_func_m[key2][iz](k)
                        
                        deriv_2 = (dp2 - dm2)/((val_arr_p[key2] - val_arr_m[key2])*fiducial_params[key2]) # derivative with respect to the second parameter
                        
                        #print(z, k, key1, key2, deriv_1, deriv_2, val_arr_p[key1], val_arr_p[key2], std_sens[iz][jk], file=file_temp)

                        if not np.isinf(std_sens[iz][jk]):
                            sigma2 = (std_sens[iz][jk])**2 + (err_func_fid[iz](k))**2 # sum the errors in quadrature
                            fisher_matrix[kkey1, kkey2] =  fisher_matrix[kkey1, kkey2] + deriv_1 * deriv_2 / sigma2

    return fisher_matrix, key_arr_unique, fiducial_params


 