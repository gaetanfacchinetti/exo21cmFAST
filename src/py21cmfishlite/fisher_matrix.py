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


    


class Run:

    def __init__(self, lightcone, z_bins, k_bins, logk): 
        
        self._z_bins    = z_bins
        self._k_bins    = k_bins
        self._logk      = logk

        # Get the power spectrum from the Lightcone
        self._lightcone       = lightcone
        self._lc_redshifts    = lightcone.lightcone_redshifts
        self._chunk_indices   = [np.argmin(np.abs(self._lc_redshifts - z)) for z in z_bins]
        self._z_arr, self._ps = p21a.compute_powerspectra_1D(lightcone, chunk_indices = self._chunk_indices, n_psbins=self._k_bins.value, logk=logk, remove_nans=False, vb=False)

        _k_arr = np.array([data['k'] for data in self._ps])
        assert np.any(np.diff(_k_arr, axis=0)[0]/_k_arr <= 1e-3)
        self._k_arr = _k_arr[0]


    @property
    def power_spectrum(self):
        return np.array([data['delta'] for data in self._ps])

    @property
    def power_spectrum_errors(self):
        return np.array([data['err_delta'] for data in self._ps])

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
        return self._k_arr

    @property
    def logk(self): 
        return self._logk

    @property
    def chunk_indices(self):
        return self._chunk_indices




class Fiducial(Run): 

    def __init__(self, dir_path, z_bins, k_bins, logk, observation = ""):

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
        return np.array(self._ps_std)

    def compute_sensitivity(self):

        _std = None

        if self._observation == 'HERA':
            _std = [None] * len(self.z_array)
            for iz, z in enumerate(self.z_array): 
                _hera     = define_HERA_observation(z)
                _std[iz]  = extract_noise_from_fiducial(self.k_array, self.power_spectrum[iz], _hera)

        self._ps_std = _std


    def plot_power_spectrum(self, obs = None):
    
        fig = p21fl_tools.plot_func_vs_z_and_k(self.z_array, self.k_array, self.power_spectrum, func_err = self.power_spectrum_errors, std = self._ps_std)
        fig.savefig(self._dir_path + "/power_spectrum.pdf", tight_layout=True)
        return fig




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
        
        self._q_value = []
        for file_name in _lightcone_file_name:
            # Get the value of q from the thing
            self._q_value.append(float(file_name.split("_")[-1][:-3]))

        # Sort the q_values from the smallest to the largest
        self._q_value = np.sort(self._q_value)

        # Check that the q_values are consistant
        assert (len(self._q_value) == 2 and (self._q_value[0] * self._q_value[1]) < 0) or len(self._q_value) == 1

        print(self._name  + " has been varied with q = " + str(self._q_value))
        print("Loading the lightcones and computing the power spectra")

        # We get the lightcones and then create the corresponding runs objects
        self._lightcones =  [p21f.LightCone.read(self._dir_path + "/Lightcone_" + self._name + "_" + str(q) + ".h5") for q in self._q_value]
        self._runs       =  [Run(lightcone, self._z_bins, self._k_bins, self._logk) for lightcone in self._lightcones]

        print("Power spectra computed")
      
        ## Check that the k-arrays and z-arrays correspond 
        for run in self._runs:
            assert np.all(2* np.abs(run.z_array - self._fiducial.z_array)/(run.z_array + self._fiducial.z_array) < 1e-5)
            assert np.all(2* np.abs(run.k_array - self._fiducial.k_array)/(run.k_array + self._fiducial.k_array) < 1e-5)

        ## Define unique k and z arrays
        self._k_array = self._fiducial.k_array
        self._z_array = self._fiducial.z_array

        print("Computing the derivatives")

        self.compute_derivative()
        self.plot_derivatives()


    @property
    def derivative(self):
        return self._derivative

    @property
    def k_array(self):
        return self._k_array

    @property
    def z_array(self):
        return self._z_array

    @property
    def fiducial(self):
        return self._fiducial

    @property
    def name(self):
        return self._name


    def compute_derivative(self):
        
        _der = [None] * len(self._z_array)
        
        _param_fid      = self._astro_params[self._name]
        _params         = np.array([(1+q) * _param_fid for q in self._q_value])
        _params         = np.append(_params, _param_fid)
        _params_sorted  = np.sort(_params)
        _mixing_params  = np.argsort(_params)
       
        for iz, z in enumerate(self._z_array) :   

            _power_spectra        = [run.power_spectrum[iz] for run in self._runs]
            _power_spectra.append(self._fiducial.power_spectrum[iz])
            
            # Rearrange the power spectra in the same order of the parameters
            _power_spectra_sorted = np.array(_power_spectra)[_mixing_params]        
            
            # Evaluate the derivative as a gradient
            _der[iz] = np.gradient(_power_spectra_sorted, _params_sorted, axis=0)

        # Arrange the derivative according to their number
        self._derivative  = {'one_sided_m' : None, 'one_sided_p' : None, 'two_sided' : None}

        if len(self._q_value) == 2:
            self._derivative['one_sided_m'] = [_der[iz][0] for iz, _ in enumerate(self._z_array)]
            self._derivative['two_sided']   = [_der[iz][1] for iz, _ in enumerate(self._z_array)]
            self._derivative['one_sided_p'] = [_der[iz][2] for iz, _ in enumerate(self._z_array)]

        if len(self._q_value) == 1 and self._q_value[0] < 0 :
            self._derivative['one_sided_m'] = [_der[iz][0] for iz, _ in enumerate(self._z_array)]
        
        if len(self._q_value) == 1 and self._q_value[0] > 0 :
            self._derivative['one_sided_p'] = [_der[iz][0] for iz, _ in enumerate(self._z_array)]

    
    def weighted_derivative(self):
        
        ps_std = self._fiducial.ps_std  
        
        # if fiducial as no standard deviation defined yet, return None
        if ps_std is None:
            return None

        der = self._derivative.get('two_sided', None)
        
        # If there is no two_sided derivative
        if der is None:
            print("Weighted derivative computed from the one_sided derivative")
            der = self._derivative.get('one_sided_m', None)
        if der is None:
            der = self._derivative.get('one_sided_p', None)

        return der / ps_std  



    def plot_derivatives(self):

        der_array = [self._derivative[key] for key in self._derivative.keys()]
        fig = p21fl_tools.plot_func_vs_z_and_k(self._z_array, self._k_array, der_array)
        fig.savefig(self._dir_path + "/derivatives_" + self._name + ".pdf", tight_layout=True)
        return fig


def evaluate_fisher_matrix(parameters):
    
    # Get the standard deviation
    n_params = len(parameters)
    fisher_matrix = np.zeros((n_params, n_params))

    name_arr     = [''] * n_params
    weighted_der = [None] * n_params

    for ip, param in enumerate(parameters):
        name_arr[ip]      = param.name
        weighted_der[ip]  = param.weighted_derivative()

    for i in range(0, n_params) :
        for j in range(0, n_params) :        
            fisher_matrix[i][j] = np.nansum(weighted_der[i] * weighted_der[j])
            
    return {'matrix' : fisher_matrix, 'name' : name_arr}
    
   
    


 