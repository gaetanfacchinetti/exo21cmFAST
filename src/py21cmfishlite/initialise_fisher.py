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

import configparser
import argparse
from . import input_output 

import numpy as np


def init_fisher(config_file: str, q_scale: float = 3.) -> None :

    """ 
    Initialise the runs for a fisher analysis according to 
    a fiducial model defined in config_file
    
    Params : 
    --------
    config_file : str
        Path to the config file representing the fiducial
    q_scale : float 
        Gives the points where to compute the derivative in pourcentage of the fiducial parameters
    
    """

    config = configparser.ConfigParser(delimiters=':')
    config.optionxform = str

    config.read(config_file)

    print(f'Calculating derivatives at {q_scale} percent from fiducial')

    name            = config.get('run', 'name')
    output_dir      = config.get('run', 'output_dir')
    cache_dir       = config.get('run', 'cache_dir')

    extra_params = {}
    extra_params['min_redshift']    = float(config.get('extra_params','min_redshift'))
    extra_params['max_redshift']    = float(config.get('extra_params','max_redshift'))
    extra_params['coarsen_factor']  = int(config.get('extra_params', 'coarsen_factor'))

    user_params       = input_output.read_config_params(config.items('user_params'))
    flag_options      = input_output.read_config_params(config.items('flag_options'))
    astro_params_fid  = input_output.read_config_params(config.items('astro_params'), int_type = False)
    astro_params_vary = input_output.read_config_params(config.items('astro_params_vary'))

    vary_array = np.array([-1, 1])
    astro_params_run_all = {}
    astro_params_run_all['fid'] = astro_params_fid

    for param, value in astro_params_vary.items(): 
        
        p_fid = astro_params_fid[param]

        # Make smaller for L_X
        if param == "L_X":
            q = 0.001*vary_array
        else:
            q = q_scale/100*vary_array

        if p_fid == 0.:
            p = q
        else:
            if value == 'linear' :
                p = p_fid - q*p_fid
            elif value == 'log':
                p = p_fid**(1-q)
            
        astro_params_run = astro_params_fid.copy()

        for i, pp in enumerate(p):
            astro_params_run[param] = pp
            if param == 'L_X': # change L_X and L_X_MINI at the same time
                astro_params_run['L_X_MINI'] = pp
            astro_params_run_all[f'{param}_{q[i]}'] = astro_params_run.copy()


    # Make the directory corresponding to the run
    output_run_dir = output_dir + "/" + name.upper() + "/"
    input_output.make_directory(output_run_dir, clean_existing_dir = True)
    input_output.make_directory(output_run_dir + "run_list/", clean_existing_dir = True)

    # Write down the separate config files
    irun = 0
    for key, astro_params in astro_params_run_all.items() : 
        input_output.write_config_params(output_run_dir + "/run_list/_run_" + str(irun) + ".config", name, cache_dir, extra_params, user_params, flag_options, astro_params, key)
        irun = irun + 1

    # Save the fiducial configuration somewhere
    with open(output_run_dir + "/fiducial_params.txt", 'w') as f:
        print("# Here we write down the fiductial parameters used to generate the run list", file = f)
        print(extra_params, file = f)
        print(user_params,  file = f)
        print(flag_options, file = f)
        print(astro_params, file = f)