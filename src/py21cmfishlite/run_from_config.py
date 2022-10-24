##################################################################################
# MIT Licence
# 
# Copyright (c) 2022, Gaétan Facchinetti
#
# This code has been taken and modified from 
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
import input_output 

import py21cmfast     as p21f
import py21cmanalysis as p21a

config = configparser.ConfigParser(delimiters=':')
config.optionxform = str

parser = argparse.ArgumentParser()
parser.add_argument("config_folder", type=str, help="Path to config file")
parser.add_argument("-id", "--id_config", type=float, help="id of the run we want to perform")
parser.add_argument("-nomp", "--n_omp", type=float, help="number of OMP threads available")
args = parser.parse_args()

config_folder = args.config_folder
config.read(config_folder)

n_omp = 1
if args.n_omp:
    n_omp  = args.n_omp

id_config = 0
if args.id_config: 
    id_config = args.id_config

name            = config.get('run', 'name')
output_dir      = config.get('run', 'output_dir')

extra_params = {}
min_redshift    = float(config.get('extra_params','min_redshift'))
max_redshift    = float(config.get('extra_params','max_redshift'))
coarsen_factor  = int(config.get('extra_params', 'coarsen_factor'))

user_params   = input_output.read_config_params(config.items('user_params'))
flag_options  = input_output.read_config_params(config.items('flag_options'))
astro_params  = input_output.read_config_params(config.items('astro_params'))


lightcone_quantities = ("brightness_temp", 'density')
global_quantities    = ("brightness_temp", 'density', 'xH_box')

lightcone, output_exotic_energy_injection = p21f.run_lightcone(
        redshift     = min_redshift,
        max_redshift = max_redshift, 
        user_params  = user_params,
        astro_params = astro_params,
        flag_options = flag_options,
        coarsen_factor       = coarsen_factor, 
        lightcone_quantities = lightcone_quantities,
        global_quantities    = global_quantities,
        verbose_ntbk = False,
        direc=cache_direc, 
    )

####################### ANALYSIS ############################

path_output = 
p21a.make_directory(path_output)
lightcone.save(fname = "lightcone.h5", direc = path_output)

# Export the data in human readable format
z_centers, power_spectra = p21a.compute_powerspectra_1D(lightcone=lightcone, nchunks=15, n_psbins=None, logk=True) 
p21a.export_global_quantities(path = path_output)
p21a.export_powerspectra_1D_vs_k(path=path_output, z_centers = z_centers, power_spectra = power_spectra)
p21a.export_powerspectra_1D_vs_z(path=path_output, z_centers = z_centers, power_spectra = power_spectra)