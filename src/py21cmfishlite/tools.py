##################################################################################
# MIT Licence
# 
# Copyright (c) 2022, Gaétan Facchinetti
#
# This code has been taken and modified from https://github.com/charlottenosam/21cmfish
# 
# # MIT License
# #
# # Copyright (c) 2019, Charlotte Mason
# # 
# # Permission is hereby granted, free of charge, to any person obtaining a copy
# # of this software and associated documentation files (the "Software"), to deal
# # in the Software without restriction, including without limitation the rights
# # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# # copies of the Software, and to permit persons to whom the Software is
# # furnished to do so, subject to the following conditions:
# # 
# # The above copyright notice and this permission notice shall be included in all
# # copies or substantial portions of the Software.
# # 
# # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# # SOFTWARE.
##################################################################################



from pylab import *

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

import numpy as np
import copy
import os

from py21cmanalysis import tools as p21a_tools

def read_config_params(config_items, int_type = True):
    """
    Read ints and booleans from config files
    Use for user_params and flag_options only
    
    Parameters
    ----------
    item : str
        config dictionary item as a string
    Return
    ------
    config dictionary item as an int, bool or str
    """

    output_dict = dict()

    for key, value in dict(config_items).items():

        try:
            if int_type is True:
                cast_val = int(value)
            else:
                cast_val = float(value)
        except:
            if value == 'True':
                cast_val =  True
            elif value == 'False':
                cast_val =  False
            else:
                cast_val = value
    
        output_dict[key] = cast_val
        
    return output_dict



def write_config_params(filename, name, cache_dir, extra_params, user_params, flag_options, astro_params, key):

    with open(filename, 'w') as f:
       
        print("# Parameter file for : " + key, file = f)
        print('', file=f)

        print("[run]", file=f)
        print("name      : " + name, file=f)
        print("run_id    : " + key, file=f)
        print("cache_dir : " + cache_dir, file=f)
        print('', file=f)
        
        print("[extra_params]", file=f)
        
        for key, value in extra_params.items():
            print(key + " : " + str(value), file=f)

        print('', file=f)
        print("[user_params]", file=f)


        for key, value in user_params.items():
            print(key + " : " + str(value), file=f)

        print('', file=f)
        print("[flag_options]", file=f)

        for key, value in flag_options.items():
            print(key + " : " + str(value), file=f)

        print('', file=f)
        print("[astro_params]", file=f)

        for key, value in astro_params.items():
            print(key + " : " + str(value), file=f)



def read_power_spectra(folder_name: str):
    """ 
    Read the power spectra from a folder 
    The folder must contain a redshift array file: folder_name/redshift_chucnks.txt
    The power spectra must be labelled and organised as folder_name/ps_z_<<"{0:.1f}".format(z)>>.txt
    The units must be Mpc for k_arr, mK**2 for delta_arr and err_arr

    Parameters
    ----------
        folder_name: str
            path to the folder where the power_spectra_are_stored
    
    Returns
    -------
        z_arr: list[float]
            list of redshifts where the power_spectra are evaluated
        k_arr: list[list[float]] (Mpc^{-1})
            list of k values for every redshift
        delta_arr: list[list[float]] (mK^2)
            list of power_spectrum value for every redshift (in correspondance to k_arr)
        err_arr: list[list[float]]  (mK^2)
            list of the error on the power spectrum (in correspondance to k_arr and delta_arr)
    """

    z_arr     = np.genfromtxt(folder_name + '/power_spectra_vs_k/redshift_chunks.txt')

    k_arr     = []
    delta_arr = []
    err_arr   = []

    for iz, z in enumerate(z_arr):
        data = np.genfromtxt(folder_name + '/power_spectra_vs_k/ps_z_' + "{0:.1f}".format(z) + '.txt')

        k_arr.append(data[:, 0])
        delta_arr.append(data[:, 1])
        err_arr.append(data[:, 2])

    return z_arr, k_arr, delta_arr, err_arr




def create_from_scracth(path_dir:str) -> None : 
    """ Create the folder that will hold the fisher analysis from scracth """

    p21a_tools.make_directory(path_dir, clean_existing_dir = False)
    p21a_tools.make_directory(path_dir + "/config_files", clean_existing_dir = False)
    p21a_tools.make_directory(path_dir + "/runs", clean_existing_dir = False)
    p21a_tools.make_directory(path_dir + "/exec", clean_existing_dir = False)

    

def prepare_sbatch_file(name_run: str, mail_user:str) -> None:

    content = \
"""#!/bin/bash
#
#SBATCH --job-name=fish1
#SBATCH --output=test_fisher.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --mem=40000
#
#SBATCH --mail-type=ALL
#SBATCH --mail-user=""" + \
mail_user + \
"""
#SBATCH --array=0-15

module load releases/2021b
module load SciPy-bundle/2021.10-foss-2021b
module load GSL/2.7-GCC-11.2.0
module load Pillow/8.3.1-GCCcore-11.2.0
module load h5py/3.6.0-foss-2021b 
module load PyYAML/5.4.1-GCCcore-11.2.0

source ~/exo21cmFAST_release/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

FILES=(../runs/"""  + \
    name_run + \
"""/run_list/*)

srun python ./run_fisher.py ${FILES[$SLURM_ARRAY_TASK_ID]} -nomp $SLURM_CPUS_PER_TASK"""

    with open("submit_run_fisher_" +  name_run +  ".sh", 'w') as f:
        print(content, file = f)



def confidence_ellipse(cov, mean_x, mean_y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)



def ellipse_from_covariance(cov_matrix, fiducial):
    """ Returns arrays for drawing the covariance matrix
        This function is mainly used as a cross-check of confidence_ellipse()"""
    
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    theta = np.linspace(0, 2*np.pi, 200)
    ellipse_x = (np.sqrt(eigenvalues[0])*np.cos(theta)*eigenvectors[0,0] + np.sqrt(eigenvalues[1])*np.sin(theta)*eigenvectors[0,1]) + fiducial[0]
    ellipse_y = (np.sqrt(eigenvalues[0])*np.cos(theta)*eigenvectors[1,0] + np.sqrt(eigenvalues[1])*np.sin(theta)*eigenvectors[1,1]) + fiducial[1]

    return ellipse_x, ellipse_y



def make_triangle_plot(covariance_matrix, name_params, fiducial_params) : 

    #####################################
    ## Choose the data we want to look at
    cov_matrix      = covariance_matrix
    fiducial_params = copy.deepcopy(fiducial_params)
    name_params     = name_params
    ngrid           = len(name_params)

    #####################################
    ##  Prepare the triangle plot
    fig = plt.figure(constrained_layout=False, figsize=(1.2*ngrid, 1.2*ngrid))
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    gs = GridSpec(ngrid, ngrid, figure=fig)
    axs = [[None for j in range(ngrid)] for i in range(ngrid)]


    ## set the parameter range to plot
    min_val_arr  = [None] * len(name_params)
    max_val_arr  = [None] * len(name_params)
    display_arr  = [None] * len(name_params)
    ticks_arr    = [None] * len(name_params)

    ['F_STAR10', 'F_STAR7_MINI', 'ALPHA_STAR', 'ALPHA_STAR_MINI',  't_STAR', 'F_ESC10', 'F_ESC7_MINI', 'ALPHA_ESC', 'L_X', 'L_X_MINI', 
                   'DM_LOG10_LIFETIME', 'DM_FHEAT_APPROX_PARAM_LOG10_F0', 'DM_FHEAT_APPROX_PARAM_A', 'DM_FHEAT_APPROX_PARAM_B', 
                   'LOG10_XION_at_Z_HEAT_MAX', 'LOG10_TK_at_Z_HEAT_MAX']
    
    
    _parameters_plot = {
        'F_STAR10'          : {'tex_name' : r"$\log_{10}[f_{\star, 10}]$"},
        'F_STAR7_MINI'      : {'tex_name' : r"$\log_{10}[f_{\star, 7}^{\rm mini}]$"},
        'ALPHA_STAR'        : {'tex_name' : r"$\alpha_{\star}$", 'min': -0.7, 'max' : 1.7, 'ticks' : []},
        'ALPHA_STAR_MINI'   : {'tex_name' : r"$\alpha_{\star}^{\rm mini}$", 'min': -0.7, 'max' : 1.7, 'ticks' : []},
        't_STAR'            : {'tex_name' : r"$t_{\star}$", 'min': -0.3, 'max' : 1.3, 'ticks' : []},
        'F_ESC10'           : {'tex_name' : r"$\log_{10}[f_{\rm esc, 10}]$", 'min': -2, 'max' : 0.1},
        'F_ESC7_MINI'       : {'tex_name' : r"$\log_{10}[f_{\rm esc, 7}^{\rm mini}]$", 'min': -2, 'max' : 0.1},
        'ALPHA_ESC'         : {'tex_name' : r"$\alpha_{\rm esc}$", 'min': -1.4, 'max': 0.6, 'ticks' : []},
        'L_X'               : {'tex_name' : r"$\log_{10}\left[\frac{L_X}{\rm units}\right]$", 'min': 40, 'max': 41},
        'L_X_MINI'          : {'tex_name' : r"$\log_{10}\left[\frac{L_X^{\rm mini}}{\rm units}\right]$", 'min': 40, 'max': 41},
        'M_TURN'            : {'tex_name' : r"$\log_{10}\left[\frac{M_{\rm turn}}{{\rm M}_\odot}\right]$", 'min': 7.6, 'max': 8.9, 'ticks' : []},
        'NU_X_THRESH'       : {'tex_name' : r"$E_0~[\rm eV]$", 'min': 300, 'max': 700, 'ticks' : []},
        'DM_LOG10_LIFETIME' : {'tex_name' : r"$\log_{\rm 10}\left[\frac{\tau_{\chi}}{\rm s}\right]$", 'min': 25.6, 'max': 26.4, 'ticks' : []},
        'DM_LOG10_MASS'     : {'tex_name' : r"$\log_{10}[\frac{m_{\chi}}{\rm eV}]$", 'min': 3, 'max': 11,  'ticks' : []},
        'DM_FHEAT_APPROX_PARAM_LOG10_F0' : {'tex_name' : r"$\log_{10}[f_0]$", 'min': -2, 'max': 1,  'ticks' : []},
        'DM_FHEAT_APPROX_PARAM_A'  : {'tex_name' : r"$a$", 'min': -2, 'max': 1,  'ticks' : []},
        'DM_FHEAT_APPROX_PARAM_B'  : {'tex_name' : r"$b$", 'min': -2, 'max': 1,  'ticks' : []},
        'LOG10_XION_at_Z_HEAT_MAX' : {'tex_name' : r"$\log_{10}[\bar x_e^{\rm init}]$", 'min': -2, 'max': 1,  'ticks' : []},
        'LOG10_TK_at_Z_HEAT_MAX'   : {'tex_name' : r"$\log_{10}[\frac{\bar T_K^{\rm init}}{\rm K}]$", 'min': -2, 'max': 1,  'ticks' : []},
        }



    #####################################
    ## Go through all the possible cases and corresponding parameters
    for j in range(ngrid) : 
        for i in range(0, j+1) :

            ## Here i represents the x axis while j goes along the y axis
            axs[j][i] = fig.add_subplot(gs[j:j+1, i:i+1])

            ## Information concertning this case
            _default_info  = {'tex_name' : r'$\theta$', 'min' : None, 'max' : None, 'ticks' : []}
            
            _param_info_x  = _parameters_plot.get(name_params[i], _default_info)
            _param_info_y  = _parameters_plot.get(name_params[j], _default_info)

            # add the fiducial values in the infos
            val_x = fiducial_params[name_params[i]]     
            val_y = fiducial_params[name_params[j]]                 
    
            x_min = val_x - 4*np.sqrt(cov_matrix[i, i])
            x_max = val_x + 4*np.sqrt(cov_matrix[i, i])
            y_min = val_y - 4*np.sqrt(cov_matrix[j, j])
            y_max = val_y + 4*np.sqrt(cov_matrix[j, j])
            

            ##############
            # Setting the plot

            # we remove the ticks if necessary for some parts of this case
            if j < ngrid -1 :
                axs[j][i].xaxis.set_ticklabels([])
            if i > 0 : 
                axs[j][i].yaxis.set_ticklabels([])

            if j == ngrid -1 :
                axs[j][i].set_xlabel(_param_info_x.get('tex_name', r'$\theta$'))
                x_ticks = _param_info_x.get('ticks', [])
                if x_ticks != [] :
                    axs[j][i].set_xticks(_param_info_x.get('ticks', []))
                axs[j][i].tick_params(axis='x', labelsize=8)
                for tick in axs[j][i].get_xticklabels():
                    tick.set_rotation(55)

            if i == 0 and j > 0:
                axs[j][i].set_ylabel(_param_info_y.get('tex_name', r'$\theta$'))
                y_ticks = _param_info_y.get('ticks', [])
                if y_ticks != [] :
                    axs[j][i].set_yticks(_param_info_y.get('ticks', []))
                axs[j][i].tick_params(axis='y', labelsize=8)
                for tick in axs[j][i].get_yticklabels():
                    tick.set_rotation(55)
           

            axs[j][i].set_xlim([x_min, x_max])
            ##############

            ## Make the plots now
            if i != j : 
                
                axs[j][i].set_ylim([y_min, y_max])

                # Countour plot for the scatter
                sub_cov = np.zeros((2, 2))
                sub_cov[0, 0] = cov_matrix[i, i]
                sub_cov[0, 1] = cov_matrix[i, j]
                sub_cov[1, 0] = cov_matrix[j, i]
                sub_cov[1, 1] = cov_matrix[j, j]

                #ellipse_x, ellipse_y = ellipse_from_covariance(sub_cov, [val_x, val_y])
                #axs[j][i].plot(ellipse_x, ellipse_y, linewidth=0.5, color='blue')

                confidence_ellipse(sub_cov, val_x, val_y, axs[j][i],  n_std=2, facecolor='dodgerblue', alpha=0.4)
                confidence_ellipse(sub_cov, val_x, val_y, axs[j][i],  n_std=1, facecolor='dodgerblue', alpha=0.8)

            if i == j :


                axs[i][i].set_ylim([0, 1.2])
                axs[i][i].set_title(r'${} \pm {:.2}$'.format(val_x, np.sqrt(cov_matrix[i, i])), fontsize=10)
    
                # Plot the gaussian approximation in that panel
                sigma     = np.sqrt(cov_matrix[i, i])
                val_arr   = np.linspace(val_x-5*sigma, val_x+5*sigma, 100)
                gaussian  = exp(-(val_arr - val_x)**2/2./sigma**2)

                axs[i][i].plot(val_arr, gaussian, color='dodgerblue')
    

    axs[0][0].set_yticks([])  

    return fig



def plot_func_vs_z_and_k(z, k, func, func_err = None, std = None, istd  : float = 0, **kwargs) :

    """ 
        Function that plots the power spectra with the sensitivity bounds from extract_noise_from_fiducial()
        We can plot on top more power spectra for comparison

        Params
        ------
        k : 1D array of floats
            modes 
        z : 1D array of floats
            redshifts
        func : (list of) 2D arrays of floats
            function(s) to plot in terms of the redshift and modes
        std : 1D array of floats
            standard deviation associated to func (or func[istd])
        istd : float, optional
            index of the func array where to attach the standard deviation
    """
    n_lines = int(len(z) / 5)

    if len(z) / 5 > n_lines:
        n_lines = n_lines + 1
    
    fig = plt.figure(constrained_layout=False, figsize=(10, n_lines*2))
    #fig.subplots_adjust()


    gs = GridSpec(n_lines, 5, figure=fig, wspace=0.5, hspace = 0.5)
    axs = [[None for j in range(0, 5)] for i in range(0, n_lines)]

    if not isinstance(func[0][0], (list, np.ndarray)) :
        func = [func]

    if func_err is None:
        func_err = [None] * len(func)
    else:
        if not isinstance(func_err[0][0], (list, np.ndarray)) : 
            func_err = [func_err]


    if len(func) > 1:
        
        cmap = matplotlib.cm.get_cmap('Spectral')
        a_lin = (0.99-0.2)/(len(func)-1) if len(func) > 1 else 1
        b_lin = 0.2 if len(func) > 1 else 0.5

        _default_color_list     = [cmap(i) for i in np.arange(0, len(k))*a_lin + b_lin]
        _default_linestyle_list = ['-' for i in np.arange(0, len(k))]
    
    else:

        _default_color_list     = ['b']
        _default_linestyle_list = ['-']


    
    color_list     = kwargs.get('color', _default_color_list)
    linestyle_list = kwargs.get('linestyle', _default_linestyle_list)
    ylabel         = kwargs.get('ylabel', None)
    title          = kwargs.get('title', None)
    marker         = kwargs.get('marker', None)
    markersize     = kwargs.get('markersize', None)
    
    no_line = Line2D([],[],color='k',linestyle='-',linewidth=0, alpha = 0) 
  

    iz = 0
    for i in range(0, n_lines):
        for j in range(0, 5):
            
            if iz < len(z): 

                axs[i][j] = fig.add_subplot(gs[i:i+1, j:j+1])

                # Plot the power spectrum at every redshift
                for jf, f in enumerate(func) : 
                    if marker is None : 
                        # By default we plot steps
                        axs[i][j].step(k, f[iz], where='mid', alpha = 1, color=color_list[jf], 
                                        linestyle = linestyle_list[jf], linewidth=0.5)
                    else: 
                        # Otherwise we use markers
                        axs[i][j].plot(k, f[iz], alpha = 1, color=color_list[jf], linestyle = linestyle_list[jf], 
                                        marker=marker, markersize=markersize, markerfacecolor=color_list[jf], 
                                        linewidth=0.8)
                    
                    if func_err[jf] is not None:
                        axs[i][j].fill_between(k, f[iz] - func_err[jf][iz], f[iz] + func_err[jf][iz], 
                                                color=color_list[jf], linestyle=linestyle_list[jf], 
                                                step='mid', alpha=0.1)
                

                # Plot the standard deviation bars if standard deviation is given
                if std is not None : 
                    axs[i][j].fill_between(k, func[istd][iz] - std[iz], func[istd][iz] + std[iz], step='mid', alpha = 0.2, color='blue')
                
                
                xlim = kwargs.get('xlim', None)
                ylim = kwargs.get('ylim', None)

                if xlim is not None: 
                    axs[i][j].set_xlim(xlim)
                if ylim is not None: 
                    axs[i][j].set_ylim(ylim)

                logx = kwargs.get('logx', False)
                logy = kwargs.get('logy', False)

                if logx is True:
                    axs[i][j].set_xscale('log')
                if logy is True:
                    axs[i][j].set_yscale('log')
                

                axs[i][j].legend([no_line], [r'$\rm z = {0:.1f}$'.format(z[iz])], loc='lower right', handlelength=0, handletextpad=0, 
                                    bbox_to_anchor=(0.98, 1), fontsize=8, frameon=False,
                                    borderpad = 0, borderaxespad = 0, framealpha=0)

            iz = iz+1

    if ylabel is not None:
        axs[int(n_lines/2)-1][0].set_ylabel('{}'.format(ylabel))

    axs[n_lines-1][0].set_xlabel(r'$k ~{\rm [Mpc^{-1}]}$')


    if title is not None: 
        fig.suptitle('{}'.format(title), fontsize=14)


    return fig



def plot_func(x, func, **kwargs) :

    """ 
        Function that plots simple functions of one variable on the same graph

        Params
        ------
        x : 1D array of floats
            redshifts
        func : (list of) 1D arrays of floats
            function(s) to plot in terms of x
    """

        
    if isinstance(func[0], (int, float, np.float64, np.float32)) : 
        func = [func]

    if isinstance(x[0], (int, float, np.float64, np.float32)) : 
        x = [x]*len(func)

    
    fig = plt.figure(figsize = (5,4))
    ax = fig.gca()

    xlim   = kwargs.get('xlim', [np.min([val[0] for val in x]), np.max([val[-1] for val in x])])
    ylim   = kwargs.get('ylim', None)
    xlog   = kwargs.get('xlog', False)
    ylog   = kwargs.get('ylog', False)
    xlabel = kwargs.get('xlabel', r'$x$')
    ylabel = kwargs.get('ylabel', r'$y$')
    color  = kwargs.get('color', plt.rcParams['axes.prop_cycle'].by_key()['color'])

    ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('{}'.format(xlabel))
    ax.set_ylabel('{}'.format(ylabel))

    if xlog is True:
        ax.set_xscale('log')
    if ylog is True:
        ax.set_yscale('log')


    for ifunc, f in enumerate(func):
        ax.plot(x[ifunc], f, color=color[ifunc])

    return fig



def display_matrix(matrix, names = None):

    if names is not None:
        print('            ', end='')
        for name in names:
            print(name.ljust(10)[:10] + ' | ', end = '')

    print('')
    
    for i in range(0, len(matrix)):
        if names is not None:
            print(names[i].ljust(10)[:10] + ' | ', end='')
        for j in range(0, len(matrix)):
            add = ''
            if matrix[i][j] > 0:
                add = ' '
            print(add + "{:.1e}".format(matrix[i][j]) + '  |  ', end='')
        print('')




def load_uv_luminosity_functions(data_set = 'Bouwens21'):

    file_name = os.path.dirname(os.path.abspath(__file__)) + '/_data/' + data_set + '.npz'

    with open(file_name, 'rb') as file:  
        data = np.load(file, allow_pickle = True)   
       
        z_uv       = data['redshifts']
        m_uv       = data['M_uv']
        l_uv       = data['l_uv']
        sigma_l_uv = data['sigma_l_uv']


    return z_uv, m_uv, l_uv, sigma_l_uv

