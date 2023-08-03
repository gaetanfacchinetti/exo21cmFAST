##################################################################################
#
# Copyright (c) 2023, Gaétan Facchinetti
#
# 21cmCAST is free software: you can redistribute it and/or modify it 
# under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or any 
# later version. 21cmCAST is distributed in the hope that it will be useful, 
# but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU 
# General Public License along with 21cmCAST. 
# If not, see <https://www.gnu.org/licenses/>.
#
# -------------------------------------------------------------------------------
#
# Parts of this code have been copied and modified 
# from https://github.com/charlottenosam/21cmfish
# 
# - MIT License
# -
# - Copyright (c) 2019, Charlotte Mason
# - 
# - Permission is hereby granted, free of charge, to any person obtaining a copy
# - of this software and associated documentation files (the "Software"), to deal
# - in the Software without restriction, including without limitation the rights
# - to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# - copies of the Software, and to permit persons to whom the Software is
# - furnished to do so, subject to the following conditions:
# - 
# - The above copyright notice and this permission notice shall be included in all
# - copies or substantial portions of the Software.
# - 
# - THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# - IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# - FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# - AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# - LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# - OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# - SOFTWARE.
##################################################################################

import numpy as np
import powerbox.tools as pb_tools

from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import z_at_value

from astropy import units
from astropy import constants

from . import tools as tls


def deltak_zB(z, B) :
    
    # Definition of the 21 cm frequency (same as in 21cmSense)
    f21 = 1420.40575177 * units.MHz

    return 2*np.pi * f21 * cosmo.H(z) / constants.c / (1+z)**2 / B * 1000 * units.m / units.km


def generate_k_bins(z, k_min, k_max, B, logk):
    
    dk = deltak_zB(z, B) 
    _k_min = dk

    if _k_min < k_min :
        _k_min = k_min

    if logk is False:
        return np.arange(_k_min.value, k_max.value, dk.value) * k_min.unit
    else:
        ValueError("logarithmic k-bins not implemented yet")


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

    def generate_z_bins(z_min, z_max, B):
        fmax = f21/(1+z_min)
        fmin = f21/(1+z_max)
        f_bins    = np.arange(fmax.value, fmin.value, -B.value) * f21.unit
        f_centers = f_bins[:-1] - B/2
        z_bins    = (f21/f_bins).value - 1
        z_centers = (f21/f_centers).value -1
        return z_bins, z_centers

    # Get the redshift bin edges and centers
    z_bins, z_centers = generate_z_bins(z_min, z_max, B)
    
    # Get the k-bins edges
    k_bins = generate_k_bins(z_min, k_min, k_max, B, logk)

    return z_bins, z_centers, k_bins



def get_k_min_max(lightcone, n_chunks=24):
    """
    Get the minimum and maximum k in 1/Mpc to calculate powerspectra for
    given size of box and number of chunks
    """

    BOX_LEN = lightcone.user_params.pystruct['BOX_LEN']
    HII_DIM = lightcone.user_params.pystruct['HII_DIM']

    k_fundamental = 2*np.pi/BOX_LEN*max(1,len(lightcone.lightcone_distances)/n_chunks/HII_DIM) #either kpar or kperp sets the min
    k_max         = k_fundamental * HII_DIM
    Nk            = np.floor(HII_DIM/1).astype(int)
    return k_fundamental, k_max, Nk


def compute_power(box,
                   length,
                   n_psbins,
                   log_bins=True,
                   k_min=None,
                   k_max=None,
                   ignore_kperp_zero=True,
                   ignore_kpar_zero=False,
                   ignore_k_zero=False):
    """
    Calculate power spectrum for a redshift chunk
    TODO
    Parameters
    ----------
    box :
        lightcone brightness_temp chunk
    length :
        TODO
    n_psbins : int
        number of k bins
    Returns
    ----------
        k : 1/Mpc
        delta : mK^2
        err_delta : mK^2
    """
    # Determine the weighting function required from ignoring k's.
    k_weights = np.ones(box.shape, dtype=int)
    n0 = k_weights.shape[0]
    n1 = k_weights.shape[-1]

    if ignore_kperp_zero:
        k_weights[n0 // 2, n0 // 2, :] = 0
    if ignore_kpar_zero:
        k_weights[:, :, n1 // 2] = 0
    if ignore_k_zero:
        k_weights[n0 // 2, n0 // 2, n1 // 2] = 0

    # Define k bins
    if (k_min is None and k_max is None) or n_psbins is None:
        bins = n_psbins
    elif isinstance(n_psbins, int): 
        # In the case where n_psbins is just an integer
        if log_bins:
            bins = np.logspace(np.log10(k_min), np.log10(k_max), n_psbins)
        else:
            bins = np.linspace(k_min, k_max, n_psbins)
    else:
        bins = n_psbins

    res = pb_tools.get_power(
        box,
        boxlength=length,
        bins=bins,
        bin_ave=False,
        get_variance=True,
        log_bins=log_bins,
        k_weights=k_weights,
    )

    res = list(res)
    k = res[1]
    if log_bins:
        k = np.exp((np.log(k[1:]) + np.log(k[:-1])) / 2)
    else:
        k = (k[1:] + k[:-1]) / 2

    res[1] = k

    return res


def compute_powerspectra_1D(lightcone, nchunks=15,
                    chunk_indices=None,
                    n_psbins=None,
                    k_min=0.1,
                    k_max=1.0,
                    logk=True,
                    ignore_kperp_zero=True,
                    ignore_kpar_zero=False,
                    ignore_k_zero=False,
                    remove_nans=False,
                    vb=False):

    """
    Make power spectra for given number of equally spaced redshift chunks OR list of redshift chunk lightcone indices
    Output:
        k : 1/Mpc
        delta : mK^2
        err_delta : mK^2
    TODO this isn't using k_min, k_max...
    """
    data = []

    # Create lightcone redshift chunks
    # If chunk indices not given, divide lightcone into nchunks equally spaced redshift chunks
    if chunk_indices is None:
        #print(lightcone.n_slices, lightcone.brightness_temp.shape, round(lightcone.n_slices / nchunks))
        chunk_indices = list(range(0, lightcone.n_slices, int(lightcone.n_slices / nchunks)))
    else:
        nchunks = len(chunk_indices) - 1

    chunk_redshift = np.zeros(nchunks)
    z_centers      = np.zeros(nchunks)

    lc_redshifts = lightcone.lightcone_redshifts
    lc_distances = lightcone.lightcone_distances

    # Calculate PS in each redshift chunk
    for i in range(nchunks):
        
        if vb:
            print(f'Chunk {i}/{nchunks}...')
        
        start    = chunk_indices[i]
        end      = chunk_indices[i + 1]
        chunklen = (end - start) * lightcone.cell_size

        chunk_redshift[i] = np.median(lc_redshifts[start:end])


        #####

        index_center = int(lightcone.n_slices / nchunks)
           
        if index_center % 2 == 0:
            dist_center = 0.5 * ( lc_distances[start + int(0.5 * index_center)] + lc_distances[start + int(0.5 * index_center) - 1])
        else:
            dist_center = lc_distances[start + int(0.5 * index_center)]
            
        z_centers[i] = z_at_value(lightcone.cosmo_params.cosmo.comoving_distance, dist_center * units.Mpc)

        #####


        if chunklen == 0:
            print(f'Chunk size = 0 for z = {lc_redshifts[start]}-{lc_redshifts[end]}')
        else:
            power, k, variance = compute_power(
                    lightcone.brightness_temp[:, :, start:end],
                    (lightcone.user_params.BOX_LEN, lightcone.user_params.BOX_LEN, chunklen),
                    n_psbins,
                    log_bins=logk,
                    k_min=k_min,
                    k_max=k_max,
                    ignore_kperp_zero=ignore_kperp_zero,
                    ignore_kpar_zero=ignore_kpar_zero,
                    ignore_k_zero=ignore_k_zero,)

            if remove_nans:
                power, k, variance = power[~np.isnan(power)], k[~np.isnan(power)], variance[~np.isnan(power)]
            else:
                variance[np.isnan(power)] = np.inf

            data.append({"k": k, "delta": power * k ** 3 / (2 * np.pi ** 2), "err_delta": np.sqrt(variance) * k ** 3 / (2 * np.pi ** 2)})

    return z_centers, data


def export_powerspectra_1D_vs_k(path, z_centers, power_spectra, clean_existing_dir: bool = True) -> None:
    """ Export the power_spectra obtained from compute_power_spectra_1D """
    
    # Make the directories associated to path
    tls.make_directory(path + "/power_spectra_vs_k", clean_existing_dir)

    for iz, z in enumerate(z_centers):
        save_path_ps = path + '/power_spectra_vs_k/ps_z_' + "{0:.1f}".format(z) + '.txt' 

        with open(save_path_ps, 'w') as f:
            print("# Power spectrum at redshit z = " + str(z), file=f)
            print("# k [Mpc^{-1}] | Delta_{21}^2 [mK^2] | err_Delta [mK^2]", file=f)
            for ik, k in enumerate(power_spectra[iz]['k']): 
                print(str(k) + "\t" +  str(power_spectra[iz]['delta'][ik]) + "\t" +  str(power_spectra[iz]['err_delta'][ik]), file=f)

    save_path_redshifts = path + '/power_spectra_vs_k/redshift_chunks.txt'
    with open(save_path_redshifts, 'w') as f:
        print("# Redshift chunks at which the power spectrum is computed", file=f)
        for z in z_centers: 
            print(z, file=f)



def export_powerspectra_1D_vs_z(path, z_centers, power_spectra, clean_existing_dir:bool = True) -> None :
    """ Export the power_spectra obtained from compute_power_spectra_1D """
    
    # Make the directories associated to path
    tls.make_directory(path + "/power_spectra_vs_z", clean_existing_dir)

    for ik, k in enumerate(power_spectra[0]['k']):
        save_path_ps = path + '/power_spectra_vs_z/ps_k_' + "{0:.6f}".format(k) + '.txt' 

        with open(save_path_ps, 'w') as f:
            print("# Power spectrum at mode k = " + str(k), file=f)
            print("# z [Mpc^{-1}] | Delta_{21}^2 [mK^2] | err_Delta [mK^2]", file=f)
            for iz, z in enumerate(z_centers): 
                print(str(z) + "\t" +  str(power_spectra[iz]['delta'][ik]) + "\t" +  str(power_spectra[iz]['err_delta'][ik]), file=f)


    save_path_k = path + '/power_spectra_vs_z/k_chunks.txt'
    with open(save_path_k, 'w') as f:
        print("# k values at which the power spectrum is computed [Mpc^{-1}]", file=f)
        for k in power_spectra[0]['k']: 
            print(k, file=f)

