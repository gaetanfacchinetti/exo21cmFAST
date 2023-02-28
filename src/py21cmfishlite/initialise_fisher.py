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

from py21cmfishlite import tools as p21fl_tools 
from py21cmanalysis import tools as p21a_tools

import numpy as np


def init_fisher_from_fiducial(config_file: str, q_scale: float = 3., clean_existing_dir: bool = False) -> None :

    """ 
    Initialise the runs for a fisher analysis according to 
    a fiducial model defined in config_file
    
    Params : 
    --------
    config_file : str
        Path to the config file representing the fiducial
    q_scale : float, optional
        Gives the points where to compute the derivative in pourcentage of the fiducial parameters
    erase_fir : bool, optional
        If True forces the creation of a folder 
    """

    config = configparser.ConfigParser(delimiters=':')
    config.optionxform = str

    config.read(config_file)

    print(f'Calculating derivatives at {q_scale} percent from fiducial')

    name            = config.get('run', 'name')
    output_dir      = config.get('run', 'output_dir')
    cache_dir       = config.get('run', 'cache_dir')

    extra_params = {}
    
    try: 
        extra_params['min_redshift']  = float(config.get('extra_params','min_redshift'))
    except configparser.NoOptionError: 
        print("Warning: min_redshift set to 5 by default")
        extra_params['min_redshift']  = 5

    try:
        extra_params['max_redshift']    = float(config.get('extra_params','max_redshift'))
    except configparser.NoOptionError:
        print("Warning: max_redshift set to 35 by default")
        extra_params['max_redshift']  = 35   

    try:
        extra_params['coarsen_factor']  = int(config.get('extra_params', 'coarsen_factor'))
    except configparser.NoOptionError:
        print("Warning: coarsen factor undifined")

    user_params       = p21fl_tools.read_config_params(config.items('user_params'))
    flag_options      = p21fl_tools.read_config_params(config.items('flag_options'))
    astro_params_fid  = p21fl_tools.read_config_params(config.items('astro_params'), int_type = False)
    astro_params_vary = p21fl_tools.read_config_params(config.items('astro_params_vary'), int_type = False)

    vary_array = np.array([-1, 1])
    astro_params_run_all = {}
    astro_params_run_all['FIDUCIAL'] = astro_params_fid

    # Make the directory to store the outputs and everything
    output_run_dir = output_dir + "/" + name.upper() + "/"
    existing_dir = p21a_tools.make_directory(output_run_dir, clean_existing_dir = clean_existing_dir)

    if existing_dir is True:
        print('WARNING: Cannot create a clean new folder because clean_existing_dir is False')
        return 

    for param, value in astro_params_vary.items(): 
        
        p_fid = astro_params_fid[param]


        if isinstance(value, float) and value > 0:
            q = value/100*vary_array
        else : 
            q = q_scale/100*vary_array

        if p_fid == 0.:
            p = q
        else:
            p = p_fid*(1+q)

            
        astro_params_run = astro_params_fid.copy()

        for i, pp in enumerate(p):
            astro_params_run[param] = pp
            if param == 'L_X': # change L_X and L_X_MINI at the same time
                astro_params_run['L_X_MINI'] = pp

            astro_params_run_all[f'{param}_{q[i]}'] = astro_params_run.copy()
        
    
    # Write down the separate config files
    irun = 0
    for key, astro_params in astro_params_run_all.items() : 
        p21fl_tools.write_config_params(output_run_dir + '/Config_' + key + ".config", name, cache_dir, extra_params, user_params, flag_options, astro_params, key)
        irun = irun + 1

    # Save the fiducial configuration somewhere
    # with open(output_run_dir + "/fiducial_params.txt", 'w') as f:
    #    print("# Here we write down the fiductial parameters used to generate the run list", file = f)
    #    print(extra_params, file = f)
    #    print(user_params,  file = f)
    #    print(flag_options, file = f)
    #    print(astro_params_fid, file = f)

    return 

