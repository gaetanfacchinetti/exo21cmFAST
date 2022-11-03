from py21cmanalysis import tools as tls

def export_global_quantities(path: str, lightcone, clean_existing_dir: bool = True) -> None:
    """ 
    Export the global quantities in a human readable txt format
    
    Parameters
    ----------
        path: str
            path to the location of the export
        lightcone: Lightcone object (see 21cmFAST)
            lightcone of the run saved
        clean_existing_dir: bool (default True)
            erase preexisting folder at the same path
    """

    tls.make_directory(path + "/global_quantities", clean_existing_dir)

    lc_redshifts : list = lightcone.node_redshifts
    save_path_gq : str  = path + '/global_quantities/global_quantities.txt'
   
    ## Create a string of the keys
    str_keys: str  = ''
    data    : list = [None] * len(lightcone.global_quantities)
    for ikey, key in enumerate(lightcone.global_quantities):
        
        data[ikey] = lightcone.global_quantities[key]

        if ikey > 0 : 
            str_keys = str_keys + " | " +  str(key)
        else:
            str_keys = str(key)

    with open(save_path_gq, 'w') as f:
        print("# Global quantities evolution with the redshift ", file=f)
        print("# z | " + str_keys , file=f)

        for iz, z in enumerate(lc_redshifts):
            print(z, end='', file=f)
            for d in data: 
                print("\t" + str(d[iz]), end='', file=f)
            print('', file=f) # going back to a new line            
    
