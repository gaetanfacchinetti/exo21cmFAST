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
import os

from py21cmfishlite import tools as p21fl_tools
import py21cmfast   as p21f

def run_lightcone_from_config(config_file: str, n_omp: int = None, random_seed: int = None) :
    """ ## Run a lightcone from a config file 

    Parameters
    ----------
    config_file: str
        path of the configuration file
    n_omp: int
        number of threads to use
    run_id : str, optional
        id of the run we are currently looking at
    random_seed : int
        random_seed with which to run 21cmFAST

    Returns
    ----------
    lightcone: Lightcone object (see 21cmFAST)
        lightcone of the run
    run_id: string
        identifier of the run
    """


    ####################### Getting the data ############################

    config = configparser.ConfigParser(delimiters=':')
    config.optionxform = str
    config.read(config_file)

    name            = config.get('run', 'name')
    run_id          = config.get('run', 'run_id') 
    cache_dir       = config.get('run', 'cache_dir')

    print("Treating config file :", config_file)

    try: 
        min_redshift  = float(config.get('extra_params','min_redshift'))
    except configparser.NoOptionError: 
        print("Warning: min_redshift set to 5 by default")
        min_redshift = 5

    try:
        max_redshift    = float(config.get('extra_params','max_redshift'))
    except configparser.NoOptionError:
        print("Warning: max_redshift set to 35 by default")
        max_redshift  = 35   

    try:
        coarsen_factor  = int(config.get('extra_params', 'coarsen_factor'))
    except configparser.NoOptionError:
        print("Warning: coarsen factor undifined")
        coarsen_factor = None 

    user_params     = p21fl_tools.read_config_params(config.items('user_params'))
    flag_options    = p21fl_tools.read_config_params(config.items('flag_options'))
    astro_params    = p21fl_tools.read_config_params(config.items('astro_params'), int_type=False)


    # manually set the number of threads
    if n_omp is not None: 
        user_params['N_THREADS'] = int(n_omp)

    seed_str : str = ''
    if random_seed is not None:
        seed_str = 'rs' + str(random_seed) + '_'

    cache_path = cache_dir + name.upper() + '/run_' + seed_str + run_id + '/'
    
    ####################### Running the lightcone ############################

    lightcone_quantities = ("brightness_temp", )
    global_quantities    = ("brightness_temp", "xH_box")

    try: 

        lightcone = p21f.run_lightcone(
                redshift             = min_redshift,
                max_redshift         = max_redshift, 
                user_params          = user_params,
                astro_params         = astro_params,
                flag_options         = flag_options,
                coarsen_factor       = coarsen_factor, 
                lightcone_quantities = lightcone_quantities,
                global_quantities    = global_quantities,
                verbose_ntbk         = False,
                direc                = cache_path, 
                random_seed          = random_seed,
            )
        
    except :

        lightcone = None
        run_id    = None

    # at the end, we clear the cache 
    p21f.cache_tools.clear_cache(direc=cache_path)
    # delete the directory once it has been emptied
    os.rmdir(cache_path) 

    return lightcone, run_id


    
 