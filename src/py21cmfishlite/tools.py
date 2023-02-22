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

    for iname, name in enumerate(name_params): 
        if name in ['M_TURN', 'L_X', 'F_STAR10', 'F_ESC10']: 
            fiducial_params[name] = np.log10(fiducial_params[name])

    #####################################
    ##  Prepare the triangle plot

    fig = plt.figure(constrained_layout=False, figsize=(12,12))
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    ngrid = len(name_params)
    gs = GridSpec(ngrid, ngrid, figure=fig)
    axs = [[None for j in range(ngrid)] for i in range(ngrid)]


    ## set the parameter range to plot
    min_val_arr  = [None] * len(name_params)
    max_val_arr  = [None] * len(name_params)
    display_arr  = [None] * len(name_params)
    ticks_arr    = [None] * len(name_params)

    for iname, name in enumerate(name_params): 
        if name == 'ALPHA_ESC':
            min_val_arr[iname] = -1
            max_val_arr[iname] = 0.5
            display_arr[iname] = r"$\alpha_{\rm esc}$"
            ticks_arr[iname]   = [-0.7, -0.25, 0.2]
        if name == 'ALPHA_STAR': 
            min_val_arr[iname] = -0.5
            max_val_arr[iname] = 1
            display_arr[iname] = r"$\alpha_{\star}$"
            ticks_arr[iname]   = [-0.2, 0.25, 0.7]
        if name == 'DM_LOG10_LIFETIME': 
            min_val_arr[iname] = 25.6
            max_val_arr[iname] = 26.4
            display_arr[iname] = r"$\log_{\rm 10}\left[\frac{\tau_{\chi}}{\rm s}\right]$"
            ticks_arr[iname]   = [25.75, 26, 26.25]
        if name == 'DM_LOG10_MASS': 
            min_val_arr[iname] = 3
            max_val_arr[iname] = 11
            display_arr[iname] = r"$\log_{10}[\frac{m_{\chi}}{\rm eV}]$"
            ticks_arr[iname]   = [5, 7, 9]
        if name == 'F_ESC10':
            min_val_arr[iname] = -3
            max_val_arr[iname] = 0
            display_arr[iname] = r"$\log_{10}[f_{\rm esc, 10}]$"
            ticks_arr[iname]   = [-2, -1]
        if name == 'F_STAR10': 
            min_val_arr[iname] = -3
            max_val_arr[iname] = 0
            display_arr[iname] = r"$\log_{10}[f_{\star, 10}]$"
            ticks_arr[iname]   = [-2, -1]
        if name == 'L_X' : 
            min_val_arr[iname] = 38
            max_val_arr[iname] = 42
            display_arr[iname] = r"$\log_{10}\left[\frac{L_X}{\rm units}\right]$"
            ticks_arr[iname]   = [39, 40, 41]
        if name == 'M_TURN':
            min_val_arr[iname] = 8
            max_val_arr[iname] = 10
            display_arr[iname] = r"$\log_{10}\left[\frac{M_{\rm turn}}{{\rm M}_\odot}\right]$"
            ticks_arr[iname]   = [8.5, 9, 9.5]
        if name == 't_STAR': 
            min_val_arr[iname] = 0
            max_val_arr[iname] = 1
            display_arr[iname] = r"$t_{\star}$"
            ticks_arr[iname]   = [0.25, 0.5, 0.75]
        if name == 'NU_X_THRESH': 
            min_val_arr[iname] = 0
            max_val_arr[iname] = 1
            display_arr[iname] = r"$E_0~[\rm eV]$"
            #ticks_arr[iname]   = [0.25, 0.5, 0.75]


    for i in range(0, ngrid) : 
        for j in range(0, i+1) : 
            axs[i][j] = fig.add_subplot(gs[i:i+1, j:j+1])
            #axs[i][j].set_xscale('log')
            if i != j :
                None
                #axs[i][j].set_yscale('log')
            if i < ngrid -1 :
                axs[i][j].xaxis.set_ticklabels([])
            if j > 0 : 
                axs[i][j].yaxis.set_ticklabels([])

    for i in range(ngrid):
        axs[-1][i].set_xlabel(display_arr[i])
        #axs[-1][i].set_xticks(ticks_arr[i])
        for tick in axs[-1][i].get_xticklabels():
                tick.set_rotation(55)

    for j in range(1, ngrid):
        axs[j][0].set_ylabel(display_arr[j])
        #axs[j][0].set_yticks(ticks_arr[j])

    axs[0][0].set_yticks([])  


    #####################################
    ## Rearrange the dataset into an array of coordinates
    for j in range(ngrid) : 
        for i in range(0, j+1) :
            ## Here i represents the x axis while j goes along the y axis
            
         
            x_min = fiducial_params[name_params[i]] - 4*np.sqrt(cov_matrix[i, i])
            x_max = fiducial_params[name_params[i]] + 4*np.sqrt(cov_matrix[i, i])
            axs[j][i].set_xlim([x_min, x_max])


            if i != j : 
                # Countour plot for the scatter
                sub_cov = np.zeros((2, 2))
                sub_cov[0, 0] = cov_matrix[i, i]
                sub_cov[0, 1] = cov_matrix[i, j]
                sub_cov[1, 0] = cov_matrix[j, i]
                sub_cov[1, 1] = cov_matrix[j, j]
                ellipse_x, ellipse_y = ellipse_from_covariance(sub_cov, [fiducial_params[name_params[i]], fiducial_params[name_params[j]]])
                axs[j][i].plot(ellipse_x, ellipse_y, linewidth=0.5, color='blue')
    
                y_min = fiducial_params[name_params[j]] - 4*np.sqrt(cov_matrix[j, j])
                y_max = fiducial_params[name_params[j]] + 4*np.sqrt(cov_matrix[j, j])
                axs[j][i].set_ylim([y_min, y_max])

                #axs[j][i].set_xlim([min_val_arr[i], max_val_arr[i]])
                #axs[j][i].set_ylim([min_val_arr[j], max_val_arr[j]])

                confidence_ellipse(sub_cov, fiducial_params[name_params[i]], fiducial_params[name_params[j]], axs[j][i],  n_std=2, facecolor='blue', alpha=0.3)
                confidence_ellipse(sub_cov, fiducial_params[name_params[i]], fiducial_params[name_params[j]], axs[j][i],  n_std=1, facecolor='blue', alpha=0.7)

            if i == j :
                sigma = np.sqrt(cov_matrix[i, i])
                mean_val = fiducial_params[name_params[i]]
                val_arr = np.linspace(mean_val-5*sigma, mean_val+5*sigma, 100)
                gaussian_approx = exp(-(val_arr - mean_val)**2/2./sigma**2)
                axs[i][i].plot(val_arr, gaussian_approx, color='blue')
                #axs[i][i].set_xlim([min_val_arr[i], max_val_arr[i]])
                axs[i][i].set_ylim([0, 1.2])
                axs[i][i].set_title(r'${} \pm {:.2}$'.format(mean_val, np.sqrt(cov_matrix[i, i])), fontsize=10)

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
                        axs[i][j].fill_between(k, f[iz] - 5*func_err[jf][iz], f[iz] + 5*func_err[jf][iz], 
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