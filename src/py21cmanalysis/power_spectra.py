######################################################
# Power spectra from 21cmFish and 21cmmc
#
#
######################################################

# Define functions to calculate PS, following py21cmmc
import os 
import shutil

import numpy as np
import powerbox.tools as pb_tools
from astropy import cosmology
from astropy import units

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
    else:
        if log_bins:
            bins = np.logspace(np.log10(k_min), np.log10(k_max), n_psbins)
        else:
            bins = np.linspace(k_min, k_max, n_psbins)

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
                    n_psbins=40,
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
        chunk_indices = list(range(0,lightcone.n_slices,round(lightcone.n_slices / nchunks),))

        if len(chunk_indices) > nchunks:
            chunk_indices = chunk_indices[:-1]

        chunk_indices.append(lightcone.n_slices)

    else:
        nchunks = len(chunk_indices) - 1

    #chunk_redshift = np.zeros(nchunks)
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

        #chunk_redshift[i] = np.median(lc_redshifts[start:end])


        #####

        index_center = int(lightcone.brightness_temp.shape[0])
           
        if index_center % 2 == 0:
            dist_center = 0.5 * ( lc_distances[start + int(0.5 * index_center)] + lc_distances[start + int(0.5 * index_center) - 1])
        else:
            dist_center = lc_distances[start + int(0.5 * index_center)]
            
        z_centers[i] = cosmology.z_at_value(lightcone.cosmo_params.cosmo.comoving_distance, dist_center * units.Mpc)

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


def export_powerspectra_1D(path, z_centers, power_spectra) :
    """ Export the power_spectra obtained from compute_power_spectra_1D """
    
    # Make the directories associated to path
    make_directory(path)
    make_directory(path + "/power_spectra")

    for iz, z in enumerate(z_centers):
        save_path_ps = path + '/power_spectra/ps_z_' + "{0:.1f}".format(z) + '.txt' 

        with open(save_path_ps, 'w') as f:
            print("# Power spectrum at redshit z = " + str(z), file=f)
            print("# k [Mpc^{-1}] | Delta_{21}^2 [mK^2] | err_Delta [mK^2]", file=f)
            for ik, k in enumerate(power_spectra[iz]['k']): 
                print(str(k) + "\t" +  str(power_spectra[iz]['delta'][ik]) + "\t" +  str(power_spectra[iz]['err_delta'][ik]), file=f)

    save_path_redshifts = path + '/power_spectra/redshift_chunks.txt'
    with open(save_path_redshifts, 'w') as f:
        print("# Redshift chunks at which the power spectrum is computed", file=f)
        for z in z_centers: 
            print(z, file=f)



def make_directory(path):
    if not os.path.exists(path): 
        os.mkdir(path)
    else:
        clean_directory(path)
        print("Successfully cleaned the directory " + path)


def clean_directory(path):
    """ Clean the directory at the path: path """

    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)

            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

