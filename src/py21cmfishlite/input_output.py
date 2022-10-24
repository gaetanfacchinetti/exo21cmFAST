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


import os

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
        print("cache_dir : " + cache_dir, file=f)
        print('', file=f)
        
        print("[extra_params]", file=f)
        
        for key, value in extra_params.items():
            print(key + " : " + str(value), file=f)

        print('', file=f)
        print("[user_options]", file=f)


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

import os 
import shutil


def make_directory(path: str, clean_existing_dir:bool = True):
    
    if not os.path.exists(path): 
        os.mkdir(path)
    else:
        if clean_existing_dir is True:
            clean_directory(path)


def clean_directory(path: str):
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

