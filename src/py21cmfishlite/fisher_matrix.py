import glob

import numpy as np
import pickle
from scipy import interpolate
from astropy import units
import copy

import py21cmsense    as p21s
import py21cmfast     as p21f
import py21cmanalysis as p21a

from astropy.cosmology import Planck18 as cosmo
from astropy import units
from astropy import constants

from py21cmfishlite import tools as p21fl_tools


import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


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


    sensitivity       = p21s.PowerSpectrum(observation=observation, k_21 = k / units.Mpc, 
                                            delta_21 = dsqr * (units.mK**2), foreground_model='moderate') 
    k_sens            = sensitivity.k1d.value * p21s.config.COSMO.h
    std_21cmSense     = sensitivity.calculate_sensitivity_1d(thermal = True, sample = True).value

    std = interpolate.interp1d(k_sens, std_21cmSense, bounds_error=False, fill_value=np.nan)(k)

    return std 




def define_grid_modes_redshifts(z_min: float, B: float, k_min = 0.1 / units.Mpc, k_max = 1 / units.Mpc, z_max: float = 19, logk=False) : 
    """
    ## Defines a grid of modes and redshift on which to define the noise and on which the Fisher matrix will be evaluated
    
    Params:
    -------
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
        fmax = f21/(1+z_min)
        fmin = f21/(1+z_max)
        f_bins    = np.arange(fmax.value, fmin.value, -B.value) * f21.unit
        f_centers = f_bins[:-1] - B/2
        z_bins    = (f21/f_bins).value - 1
        z_centers = (f21/f_centers).value -1
        return z_bins, z_centers
    
    def generate_k_bins(z_min, k_min, k_max, B):
        
        dk = deltak_zB(z_min, B) 
        _k_min = dk

        if _k_min < k_min :
            _k_min = k_min

        if logk is False:
            return np.arange(_k_min.value, k_max.value, dk.value) * k_min.unit
        else:
            ValueError("logarithmic k-bins not implemented yet")

    # Get the redshift bin edges and centers
    z_bins, _ = generate_z_bins(z_min, z_max, B)
    
    # Get the k-bins edges
    k_bins = generate_k_bins(z_min, k_min, k_max, B)

    return z_bins, k_bins





class Run:

    def __init__(self, lightcone, z_bins, k_bins, logk, q: float = 0.): 
        
        self._z_bins    = z_bins
        self._k_bins    = k_bins
        self._logk      = logk
        self._q         = q


        # Get the power spectrum from the Lightcone
        self._lightcone       = lightcone
        self._lc_redshifts    = lightcone.lightcone_redshifts
        self._chunk_indices   = [np.argmin(np.abs(self._lc_redshifts - z)) for z in z_bins]
        self._z_arr, self._ps = p21a.compute_powerspectra_1D(lightcone, chunk_indices = self._chunk_indices, 
                                                                n_psbins=self._k_bins.value, logk=logk, 
                                                                remove_nans=False, vb=False)
        
        _lc_glob_redshifts = self._lightcone.node_redshifts
        self._z_glob       = np.linspace(z_bins[0], z_bins[-1], 100)
        

        _global_signal = lightcone.global_quantities.get('brightness_temp', np.zeros(len(_lc_glob_redshifts), dtype=np.float64))
        _xH_box        = lightcone.global_quantities.get('xH_box', np.zeros(len(_lc_glob_redshifts), dtype=np.float64))

        self._global_signal = interpolate.interp1d(_lc_glob_redshifts, _global_signal)(self._z_glob)
        self._xH_box        = interpolate.interp1d(_lc_glob_redshifts, _xH_box)(self._z_glob)
        
        _k_arr = np.array([data['k'] for data in self._ps])
        assert np.any(np.diff(_k_arr, axis=0)[0]/_k_arr <= 1e-3)
        self._k_arr = _k_arr[0]


    @property
    def power_spectrum(self):
        return np.array([data['delta'] for data in self._ps])

    @property
    def ps_poisson_noise(self):
        return np.array([data['err_delta'] for data in self._ps])
    
    @property
    def global_signal(self):
        return self._global_signal
    
    @property
    def xH_box(self):
        return self._xH_box
    
    @property
    def z_glob(self):
        return self._z_glob

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

    @property
    def q(self):
        return self._q

    ## Lighcone properties
    @property
    def astro_params(self):
        return dict(self._lightcone.astro_params.self)

    @property
    def user_params(self):
        return dict(self._lightcone.user_params.self)

    @property
    def flag_options(self):
        return dict(self._lightcone.flag_options.self)

    @property
    def cosmo_params(self):
        return dict(self._lightcone.cosmo_params.self)



def compare_arrays(array_1, array_2, eps : float):
    if len(array_1) != len(array_2):
        return False
    return np.all(2* np.abs((array_1 - array_2)/(array_1 + array_2)) < eps)



class CombinedRuns:
    """
    ## Smart collection of the same runs with different random seeds
    """

    def __init__(self, dir_path, name, z_bins = None, k_bins = None, logk=False, q : float = 0, save=True, load=True) -> None:
        
        self._name            = name
        self._dir_path        = dir_path
        self._filename_data   = self._dir_path + '/Table_' + self._name + '.npz'    
        self._filename_params = self._dir_path + '/Param_' + self._name + '.pkl' 

        if load is True : 
            _load_successfull = self._load()

            _params_match = True

            # If we load the file correctly and if the input parameter correspond
            # then we don't need to go further and can skip the full computation again
            if _load_successfull is True:
                if z_bins is not None:
                    if compare_arrays(self._z_bins, z_bins, 1e-5) is False:
                        _params_match = False
                        raise ValueError("z-bins in input are different than the one used to precompute the tables")
                if k_bins is not None:
                    if compare_arrays(self._k_bins, k_bins, 1e-5) is False:
                        _params_match = False
                        raise ValueError("z-bins in input are different than the one used to precompute the tables")

                if _params_match is True:
                    return None

            if (z_bins is None or k_bins is None) and _load_successfull is False:
                raise ValueError("Need to pass z_bins and k_bins as inputs")


        self._z_bins  = z_bins
        self._k_bins  = k_bins
        self._logk    = logk
        self._q       = q


        # fetch all the lightcone files that correspond to runs the same parameters but different seed
        _lightcone_file_name = glob.glob(self._dir_path + "/Lightcone_*" + self._name + ".h5")
        
        if len(_lightcone_file_name) > 1:
            print("For " + self._name + ": grouping a total of " + str(len(_lightcone_file_name)) + " runs")
        
        # Create the array of runs
        self._runs =  [Run(p21f.LightCone.read(file_name), self._z_bins, self._k_bins, logk, q) for file_name in _lightcone_file_name]

        assert len(self._runs) > 0, "ERROR when searching for lightcones with a given name" 

        ## check that there is no problem and all z- and k- arrays have the same properties
        for irun in range(1, len(self._runs)) :

            # check that all with the same q-value have the same bins 
            assert compare_arrays(self._runs[0].k_array, self._runs[irun].k_array, 1e-5) 
            assert compare_arrays(self._runs[0].z_array, self._runs[irun].z_array, 1e-5) 
            assert compare_arrays(self._runs[0].k_bins,  self._runs[irun].k_bins,  1e-5) 
            assert compare_arrays(self._runs[0].z_bins,  self._runs[irun].z_bins,  1e-5) 
            
            # check that all with the same q-value have the same astro_params
            assert self._runs[0].astro_params  == self._runs[irun].astro_params
            assert self._runs[0].user_params   == self._runs[irun].user_params
            assert self._runs[0].flag_options  == self._runs[irun].flag_options
            assert self._runs[0].cosmo_params  == self._runs[irun].cosmo_params

        self._z_array = self._runs[0].z_array
        self._k_array = self._runs[0].k_array
        self._z_glob  = self._runs[0].z_glob

        self._average_quantities()

        self._tau_ion = p21f.compute_tau(redshifts=self._z_glob, global_xHI = self.xH_box, user_params=self._user_params, cosmo_params=self._cosmo_params)

        if save is True:
            self._save()


    def _average_quantities(self):
        
        ## compute the average values and the spread 
        self._power_spectrum    = np.average([run.power_spectrum for run in self._runs], axis=0)
        self._ps_poisson_noise  = np.average([run.ps_poisson_noise for run in self._runs], axis=0)
        self._ps_modeling_noise = np.std([run.power_spectrum for run in self._runs], axis=0)
        self._global_signal     = np.average([run.global_signal for run in self._runs], axis=0)
        self._xH_box            = np.average([run.xH_box for run in self._runs], axis=0)
        self._astro_params      = self._runs[0].astro_params 
        self._user_params       = self._runs[0].user_params
        self._flag_options      = self._runs[0].flag_options
        self._cosmo_params      = self._runs[0].cosmo_params


    def _save(self):
        """
        ##  Saves all the attributes of the class to be easily reload later if necessary

        numpy arrays are saved in an .npz format
        scalar parameters / attributes are saved in a dictionnary in a .pkl format
        """

        with open(self._filename_data, 'wb') as file: 
            np.savez(file, power_spectrum     = self.power_spectrum,
                            ps_poisson_noise  = self.ps_poisson_noise,
                            ps_modeling_noise = self.ps_modeling_noise,
                            global_signal     = self.global_signal,
                            xH_box            = self.xH_box,
                            z_array           = self.z_array,
                            k_array           = self.k_array,
                            z_bins            = self.z_bins,
                            k_bins            = self.k_bins,
                            z_glob            = self.z_glob)
        
        # Prepare the dictionnary of parameters
        param_dict = {'logk' : self.logk, 'q' : self.q, 
                    'astro_params': self.astro_params, 
                    'user_params': self.user_params,
                    'flag_options': self.flag_options,
                    'cosmo_params' : self.cosmo_params}
        
        with open(self._filename_params, 'wb') as file:
            pickle.dump(param_dict, file)
        

    def _load(self):
        """
        ##  Loads all the attributes of the class
        """

        data   = None
        params = None

        try:

            with open(self._filename_data, 'rb') as file: 
                data = np.load(file)
                
                self._power_spectrum    = data['power_spectrum']
                self._ps_poisson_noise  = data['ps_poisson_noise']
                self._ps_modeling_noise = data['ps_modeling_noise']
                self._z_array           = data['z_array']
                self._k_array           = data['k_array']
                self._z_bins            = data['z_bins']
                self._k_bins            = data['k_bins']  / units.Mpc
                self._global_signal     = data['global_signal']
                self._xH_box            = data['xH_box']
                self._z_glob            = data['z_glob']

            with open(self._filename_params, 'rb') as file:
                params = pickle.load(file)

                self._logk          = params['logk']
                self._q             = params['q']
                self._astro_params  = params['astro_params']
                self._user_params   = params['user_params']
                self._flag_options  = params['flag_options']  
                self._cosmo_params  = params['cosmo_params']

            ## Recompute tau_ion from the properties read here
            self._tau_ion = p21f.compute_tau(redshifts=self._z_glob, global_xHI = self.xH_box, user_params=self._user_params, cosmo_params=self._cosmo_params)

            return True

        except FileNotFoundError:
            print("No existing data found for " + self._name)
            
            return False
    

    @property
    def tau_ion(self):
        return self._tau_ion

    @property
    def power_spectrum(self):
        return self._power_spectrum
    
    @property
    def ps_poisson_noise(self):
        return self._ps_poisson_noise

    @property
    def ps_modeling_noise(self):
        return self._ps_modeling_noise
    
    @property
    def global_signal(self):
        return self._global_signal
    
    @property
    def xH_box(self):
        return self._xH_box
    
    @property
    def z_glob(self):
        return self._z_glob

    @property
    def z_array(self):
        return self._z_array

    @property
    def k_array(self): 
        return self._k_array
    
    @property
    def z_bins(self):
        return self._z_bins

    @property
    def k_bins(self): 
        return self._k_bins
 
    @property
    def logk(self): 
        return self._logk

    @property
    def q(self):
        return self._q

    # lightcone properties
    @property
    def astro_params(self): 
        return self._astro_params
    
    @property
    def user_params(self):
        return self._user_params

    @property
    def flag_options(self):
        return self._flag_options

    @property
    def cosmo_params(self):
        return self._cosmo_params



    def plot_power_spectrum(self, std = None, figname = None, plot=True) :  

        fig = p21fl_tools.plot_func_vs_z_and_k(self.z_array, self.k_array, self.power_spectrum, 
                                                func_err = np.sqrt(self.ps_poisson_noise**2 + self.ps_modeling_noise**2),
                                                std = std, title=r'$\Delta_{21}^2 ~{\rm [mK^2]}$', 
                                                xlim = [self._k_bins[0].value, self._k_bins[-1].value], logx=self._logk, logy=True)
        
        if plot is True : 

            if figname is None:
                figname = self._dir_path + "/power_spectrum.pdf"
        
            fig.savefig(figname, bbox_layout='tight')

        return fig
    




class Fiducial(CombinedRuns): 

    def __init__(self, dir_path, z_bins, k_bins, logk, observation = "", frac_noise = 0., **kwargs):

        self._dir_path     = dir_path
        super().__init__(self._dir_path, "FIDUCIAL", z_bins, k_bins, logk, **kwargs)
    

        self._frac_noise = frac_noise
        self._astro_params       = self._astro_params
        self._observation        = observation
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
    def frac_noise(self):
        return self._frac_noise

    @frac_noise.setter
    def frac_noise(self, value):
        print("Warning: frac noise has been changed, all related quantities should be recomputed")
        self._frac_noise = value

    @property
    def ps_exp_noise(self):
        return np.array(self._ps_exp_noise)
    

    def compute_sensitivity(self):

        _std = None

        if self._observation == 'HERA':
            _std = [None] * len(self.z_array)
            for iz, z in enumerate(self.z_array): 
                _hera     = define_HERA_observation(z)
                _std[iz]  = extract_noise_from_fiducial(self.k_array, self.power_spectrum[iz], _hera)

        self._ps_exp_noise = _std


    
    def chi2_UV_luminosity_functions(self, data_set = 'Bouwens21', plot = True):

        z_uv, m_uv, l_uv, sigma_l_uv   = p21fl_tools.load_uv_luminosity_functions(data_set)
        m_uv_sim , _ , log10_l_uv_sim  = p21f.compute_luminosity_function(redshifts = z_uv, 
                                                    user_params  = self._user_params, 
                                                    astro_params = self._astro_params, 
                                                    flag_options = self._flag_options)
        
        if plot is True:
            fig = p21fl_tools.plot_func(m_uv_sim, 10**log10_l_uv_sim, ylog=True, xlim=[-14, -25], ylim =[1e-12, 1], xlabel=r'$M_{\rm UV}$', ylabel=r'$\phi_{\rm UV}$')
            for iz, z in enumerate(z_uv) :
                    fig.gca().errorbar(m_uv[iz], l_uv[iz], yerr=sigma_l_uv[iz], linestyle='', capsize=2, elinewidth=0.5, label=r'${}$'.format(z))

            fig.gca().legend()
            fig.savefig(self._dir_path + '/UV_liminosity_functions.pdf', bbox_inches='tight')

        ## The chi2 is given by a sum on the elements where z > 6 (as we cannot trust 21cmFAST below)

        _chi2 = 0
        _nval = 0
        for iz, z in enumerate(z_uv):
            if z > 6:
                log10_l_uv_func = interpolate.interp1d(m_uv_sim[iz], log10_l_uv_sim[iz])
                for im, m in enumerate(m_uv[iz]) :     
                    _chi2 = _chi2 + (10**log10_l_uv_func(m) - l_uv[iz][im])**2/(sigma_l_uv[iz][im]**2)
                    _nval = _nval + 1

        ## Now compute the reduced chi2
        _reduced_chi2 = _chi2 / _nval

        return _reduced_chi2
    

    
    def test_statistic_luminosity_function() :
        pass

    def plot_power_spectrum(self):
        super().plot_power_spectrum(std=self._ps_exp_noise, figname = self._dir_path + "/fiducial_power_spectrum.pdf")

    def plot_xH_box(self):
        fig = p21fl_tools.plot_func(self.z_glob, self.xH_box,
                                    xlabel=r'$z$',
                                    ylabel=r'$x_{\rm H_{I}}$')
        fig.savefig(self._dir_path + '/fiducial_xH.pdf', bbox_inches='tight')
    
    def plot_global_signal(self):
        fig = p21fl_tools.plot_func(self.z_glob, self.global_signal, ylim=[-150, 50],
                                    xlabel=r'$z$',
                                    ylabel=r'$\overline{T_{\rm b}}~\rm [mK]$')
        fig.savefig(self._dir_path + '/fiducial_global_signal.pdf', bbox_inches='tight')



parameters_tex_name = {'F_STAR10' : r'\log_{10} f_{\star, 10}', 
                        'ALPHA_STAR' : r'\alpha_{\star}',
                        'F_ESC10' : r'\log_{10} f_{\rm esc, 10}', 
                        'ALPHA_ESC' : r'\alpha_{\rm esc}',
                        'L_X' : r'\log_{10} L_X',
                        'NU_X_THRESH' : r'E_0',
                        't_STAR': r't_\star',
                        'M_TURN': r'\log_{10} M_{\rm turn}'}



class Parameter:

    def __init__(self, fiducial, name, plot = True, **kwargs):
        
        self._fiducial       = fiducial
        self._name           = name
        self._plot           = plot

        self._dir_path       = self._fiducial.dir_path
        self._astro_params   = self._fiducial.astro_params
        self._z_bins         = self._fiducial.z_bins
        self._k_bins         = self._fiducial.k_bins
        self._logk           = self._fiducial.logk

        self._tex_name       = parameters_tex_name.get(self._name, r'\theta')
        self._load           = kwargs.get('load', True)

        if name not in self._astro_params:
            ValueError("ERROR: the name does not corresponds to any varied parameters")

        # get the lightcones from the filenames
        _lightcone_file_name = glob.glob(self._dir_path + "/Lightcone_*" + self._name + "_*.h5")
        
        # get (from the filenames) the quantity by which the parameter has been varies from the fiducial
        self._q_value = []
        for file_name in _lightcone_file_name:
            # Get the value of q from the thing
            self._q_value.append(float(file_name.split("_")[-1][:-3]))

        # If more than one file with the same q_values we remove all identical numbers
        self._q_value = list(set(self._q_value))

        # Sort the q_values from the smallest to the largest
        self._q_value = np.sort(self._q_value)

        # Check that the q_values are consistant
        assert len(self._q_value) == 1 or (len(self._q_value) == 2 and (self._q_value[0] * self._q_value[1]) < 0), "q_value : " + str(self._q_value)

        print("------------------------------------------")
        print(self._name  + " has been varied with q = " + str(self._q_value))
        print("Loading the lightcones and computing the power spectra")

        # We get the lightcones and then create the corresponding runs objects
        self._runs =  [CombinedRuns(self._dir_path, self._name + "_" + str(q), self._z_bins, self._k_bins, 
                                    self._logk, q, **kwargs) for q in self._q_value]

        print("Power spectra of " + self._name  + " computed")
      
        ## Check that the k-arrays, z-arrays, k-bins and z-bins correspond 
        for run in self._runs:
            assert compare_arrays(run.z_array, self._fiducial.z_array, 1e-5)
            assert compare_arrays(run.k_array, self._fiducial.k_array, 1e-5)
            assert compare_arrays(run.z_bins,  self._fiducial.z_bins,  1e-5)
            assert compare_arrays(run.k_bins,  self._fiducial.k_bins,  1e-5)

        ## Define unique k and z arrays
        self._k_array = self._fiducial.k_array
        self._z_array = self._fiducial.z_array
        self._k_bins  = self._fiducial.k_bins
        self._z_bins  = self._fiducial.z_bins

        print("Computing the derivatives of " + self._name)

        self.compute_ps_derivative()

        print("Derivative of " + self._name  + " computed")

        # Plotting the derivatives
        if self._plot is True:
            self.plot_ps_derivative()
            self.plot_weighted_ps_derivative()

        print("------------------------------------------")



    @property
    def ps_derivative(self):
        return self._ps_derivative

    @property
    def k_array(self):
        return self._k_array

    @property
    def z_array(self):
        return self._z_array

    @property
    def k_bins(self):
        return self._k_bins

    @property
    def z_bins(self):
        return self._z_bins

    @property
    def fiducial(self):
        return self._fiducial

    @property
    def name(self):
        return self._name


    def compute_ps_derivative(self):
        
        _der = [None] * len(self._z_array)
        
        # For convinience some parameters in 21cmFAST have to be defined by their log value
        # HOWEVER astro_params contains the true value which makes things not confusing at ALL
        # For these parameters we need to get the log again
        _param_fid = self._astro_params[self._name]

        # get all the parameters and sort them
        _params         = np.array([(1+run.q) * _param_fid for run in self._runs])
        _params         = np.append(_params, _param_fid)
        _params_sorted  = np.sort(_params)
        _mixing_params  = np.argsort(_params)


        # loop over all the redshift bins
        for iz, z in enumerate(self._z_array) :   

            # get an array of power spectra in the same order as the parameters
            _power_spectra = [run.power_spectrum[iz] for run in self._runs]
            _power_spectra.append(self._fiducial.power_spectrum[iz])

            # rearrange the power spectra in the same order of the parameters
            _power_spectra_sorted = np.array(_power_spectra)[_mixing_params]        

            # evaluate the derivative as a gradient
            _der[iz] = np.gradient(_power_spectra_sorted, _params_sorted, axis=0)


        # arrange the derivative whether they are left, right or centred
        self._ps_derivative  = {'left' : None, 'right' : None, 'centred' : None}

        if len(self._q_value) == 2:
            self._ps_derivative['left']    = [_der[iz][0] for iz, _ in enumerate(self._z_array)]
            self._ps_derivative['centred'] = [_der[iz][1] for iz, _ in enumerate(self._z_array)]
            self._ps_derivative['right']   = [_der[iz][2] for iz, _ in enumerate(self._z_array)]

        if len(self._q_value) == 1 and self._q_value[0] < 0 :
            self._ps_derivative['left'] = [_der[iz][0] for iz, _ in enumerate(self._z_array)]
        
        if len(self._q_value) == 1 and self._q_value[0] > 0 :
            self._ps_derivative['right'] = [_der[iz][0] for iz, _ in enumerate(self._z_array)]

        

    def weighted_ps_derivative(self, kind: str ='centred'):
       
        """
        ## Weighted derivative of the power spectrum with respect to the parameter
        
        Params:
        -------
        kind : str, optional
            choice of derivative ('left', 'right', or 'centred')

        Returns:
        --------
        The value of the derivative devided by the error
        """
        
        # experimental error
        ps_exp_noise   = self._fiducial.ps_exp_noise  

        # theoretical uncertainty from the simulation              
        ps_poisson_noise  = self._fiducial.ps_poisson_noise 
        ps_modeling_noise = np.sqrt(self._fiducial.ps_modeling_noise**2 + (self._fiducial.frac_noise * self._fiducial.power_spectrum)**2)
    
        
        # if fiducial as no standard deviation defined yet, return None
        if ps_exp_noise is None:
            return None

        # Initialize the derivative array to None
        der = None

        # If two sided derivative is asked then we are good here
        if kind == 'centred' :
            der = self._ps_derivative.get('centred', None)
        
        # Value to check if we use the one sided derivative 
        # by choice or because we cannot use the two-sided one
        _force_to_one_side = False

        # If there is no centred derivative
        if der is None or kind == 'left':
            if der is None and kind != 'left' and kind != 'right': 
                # Means that we could not read a value yet 
                # but not that we chose to use the left
                _force_to_one_side = True
            der = self._ps_derivative.get('left', None)
        
        if der is None or kind == 'right':
            if der is None and kind != 'right' and kind != 'left': 
                # Means that we could not read a value yet 
                # but not that we chose to use the right
                _force_to_one_side = True
            der = self._ps_derivative.get('right', None)

        if _force_to_one_side is True:
            print("Weighted derivative computed from the one_sided derivative")

        # We sum (quadratically) the two errors
        return der / np.sqrt(ps_exp_noise**2 + ps_poisson_noise**2 + ps_modeling_noise**2)  


    def plot_ps_derivative(self):

        der_array = [self._ps_derivative[key] for key in self._ps_derivative.keys()]
        fig = p21fl_tools.plot_func_vs_z_and_k(self._z_array, self._k_array, der_array, marker='.', markersize=2, 
                                                title=r'$\frac{\partial \Delta_{21}^2}{\partial ' + self._tex_name + r'}$', 
                                                xlim = [0.1, 1], logx=self._logk, logy=False)
        fig.savefig(self._dir_path + "/derivatives_" + self._name + ".pdf")
        return fig


    def plot_power_spectra(self, **kwargs):

        _ps        = [self._fiducial.power_spectrum]
        _ps_errors = [np.sqrt(self._fiducial.ps_poisson_noise**2 + self._fiducial.ps_modeling_noise**2)]
        _q_vals    = [0]

        for run in self._runs:
            _ps.append(run.power_spectrum)
            _ps_errors.append(np.sqrt(run.ps_poisson_noise**2 + run.ps_modeling_noise**2))
            _q_vals.append(run.q)

        _order     = np.argsort(_q_vals)
        _ps        = np.array(_ps)[_order]
        _ps_errors = np.array(_ps_errors)[_order]

        fig = p21fl_tools.plot_func_vs_z_and_k(self.z_array, self.k_array, _ps, func_err = _ps_errors, 
                                                std = self._fiducial.ps_exp_noise, 
                                                title=r'$\Delta_{21}^2 ~ {\rm [mK^2]}$', 
                                                logx=self._logk, logy=True, istd = _order[0], **kwargs)

        fig.savefig(self._dir_path + "/power_spectra_" + self.name + ".pdf")
        return fig


    def plot_weighted_ps_derivative(self):

        if self.fiducial.ps_exp_noise is None:
            ValueError("Error: cannot plot the weighted derivatives if the error \
                        if the experimental error is not defined in the fiducial")


        der_array = [self.weighted_ps_derivative(kind=key) for key in self._ps_derivative.keys()]
        fig = p21fl_tools.plot_func_vs_z_and_k(self._z_array, self._k_array, der_array, marker='.', markersize=2, 
                                                title=r'$\frac{1}{\sigma}\frac{\partial \Delta_{21}^2}{\partial ' + self._tex_name + r'}$', 
                                                xlim = [0.1, 1], logx=self._logk, logy=False)
        fig.savefig(self._dir_path + "/weighted_derivatives_" + self._name + ".pdf")
        return fig





def evaluate_fisher_matrix(parameters):
    """
    ## Fisher matrix evaluator

    Parameters:
    -----------
    parameters: array of Parameters objects
        parameters on which to compute the Fisher matrix
    
    Return:
    -------
    dictionnary with keys: 'matrix' for the matrix itself and 'name' for 
    the parameter names associated to the matrix in the same order
    """
    
    # Get the standard deviation
    n_params = len(parameters)
    fisher_matrix = np.zeros((n_params, n_params))

    name_arr     = [''] * n_params
    weighted_der = [None] * n_params

    for ip, param in enumerate(parameters):
        name_arr[ip]      = param.name
        weighted_der[ip]  = param.weighted_ps_derivative()

    for i in range(0, n_params) :
        for j in range(0, n_params) :        
            fisher_matrix[i][j] = np.nansum(weighted_der[i] * weighted_der[j])
            
    return {'matrix' : fisher_matrix, 'name' : name_arr}
    

    


 