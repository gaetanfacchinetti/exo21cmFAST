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


import numpy as np


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

