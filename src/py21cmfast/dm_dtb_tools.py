###########################################################
# New in exo21cmFAST
# Gaetan Facchinetti: gaetan.facchinetti@ulb.be
#
# Manage the inputs/outputs and the database of files (related to DM energy injection)
# WARNING: Accessing/writting in the database is not optimised
###########################################################

import os
from pathlib import Path
from collections.abc import Callable
import shutil
import copy
from abc import abstractmethod

''' 
Ready for a new version:

def boost_from_file(model: str) -> Callable[[float], float]:
    
    """
    If we use a custom model that should start with custom
    The file should at least contain two columns: 
    The first one corresponding to (1+z) and the second one to the value of the boost
    """

    struct_data = np.loadtxt(ABSOLUTE_PATH + "/data/custom_boosts/" + model + ".txt")
    log_struct_interp = interp1d(np.log(struct_data[:,0]), np.log(struct_data[:,1]), bounds_error=False, fill_value=(np.nan, 0.))

    def func(rs):
        return np.exp(log_struct_interp(np.log(rs)))


    return func
'''

class model : 

    _attribute_list = ['bkr', 'process', 'mDM', 'primary', 'boost', 'fs_method', 'sigmav', 'lifetime', 'comment']

    _bkr_dflt = False
    _process_dflt = 'decay'
    _mDM_dflt = 1e+10
    _primary_dflt = 'e'
    _boost_dflt = 'none'
    _fs_method_dflt = 'no_He'
    _sigmav_dflt = 3e-26
    _lifetime_dflt = 0
    _comment_dflt = ''

    _allowed_boosts    = ['none', 'erfc', 'einasto_subs', 'einasto_no_subs', 'NFW_subs', 'NFW_no_subs', 'custom_NFW_no_subs_ST_21cmFAST_0', 'custom_NFW_no_subs_ST_21cmFAST_0']
    _allowed_primaries   = ['none', 'elec_delta', 'phot_delta', 'e_L', 'e_R', 'e', 'mu_L', 'mu_R', 'mu', 'tau_L', 'tau_R', 'tau', 'q', 'c', 'b', 't', 'gamma', 'g', 'W_L', 'W_T', 'W', 'Z_L', 'Z_T', 'Z', 'h']
    _allowed_fs_methods = ['none', 'He', 'no_He', 'He_recomb']
    _allowed_processes   = ['none', 'swave', 'decay']

    
    def __init__(self, bkr: bool = None, process: str = None, 
                mDM: float = None, primary: str = None, boost: str = None, fs_method: str = None, 
                sigmav: float = None, lifetime: float = None, comment: str = None, index: int = None) :
        
        self.index     = index
        self.bkr       = bkr       if (bkr is not None)       else self._bkr_dflt
        self.process   = process   if (process is not None)   else self._process_dflt
        self.mDM       = mDM       if (mDM is not None)       else self._mDM_dflt
        self.primary   = primary   if (primary is not None)   else self._primary_dflt
        self.boost     = boost     if (boost is not None)     else self._boost_dflt
        self.fs_method = fs_method if (fs_method is not None) else self._fs_method_dflt
        self.sigmav    = sigmav    if (sigmav is not None)    else self._sigmav_dflt
        self.lifetime  = lifetime  if (lifetime is not None)  else self._lifetime_dflt
        self.comment   = comment   if (comment is not None)   else self._comment_dflt

        if self.boost not in self._allowed_boosts : 
            raise ValueError('The boost', self.boost, 'is not in the list of allowed values:', self._allowed_boosts )
        if self.primary not in self._allowed_primaries : 
           raise ValueError('The primary', self.primary, 'is not in the list of allowed values:', self._allowed_primaries )
        if self.fs_method not in self._allowed_fs_methods : 
            raise ValueError('The fs_method', self.fs_method, 'is not in the list of allowed values:', self._allowed_fs_methods )
        if self.process not in self._allowed_processes : 
            raise ValueError('The process', self.process, 'is not in the list of allowed values:', self._allowed_processes )

    def __str__(self):
        string_to_print = "proc:" + self.process + ", bkr:" + str(self.bkr) + ", mDM:" + str("{:.2e}".format(self.mDM)) + ", prim:" + self.primary  + ", boost:" + self.boost + ", fs:" + self.fs_method + ", sigv:" + str("{:.3e}".format(self.sigmav)) + ", lftime:" + str("{:.3e}".format(self.lifetime)) + ", com:" + self.comment + "]"
        
        if self.index is not None :  
            string_to_print = "[index:" + str(self.index)  + ", " + string_to_print 
        else :
             string_to_print = "[" + string_to_print
        
        return string_to_print

    def __eq__(self, other):
        answer = True
        for attr in self._attribute_list:
            if getattr(self, attr) != getattr(other, attr):
                answer = False
        return answer

    def is_valid(self): 
        """
        This function checks if the model is a valid one
        """
        answer = False

        if (self.boost in self._allowed_boosts) \
            and (self.primary in self._allowed_primaries) \
            and (self.fs_method in self._allowed_fs_methods) \
            and (self.process in self._allowed_processes) \
            and self.mDM >= 0:
            answer=True

        return answer 

    def write_in_file(self, with_index = True):
        output_str =   str(self.bkr) +  "\t" + self.process + "\t" + str("{:.2e}".format(self.mDM)) +  "\t" + \
                self.primary +  "\t" + self.boost + "\t" + self.fs_method + "\t" + str("{:.2e}".format(self.sigmav)) +  "\t" + \
                str("{:.2e}".format(self.lifetime)) + "\t" + self.comment
        if with_index:
            output_str = str(self.index) +  "\t" + output_str
        
        return output_str

    def file_header(self, with_index = True):
        if with_index: 
            return "# index bkr process mDM [eV] primary boost fs_method sigmav lifetime comment"
        else: 
            return "# bkr process mDM [eV] primary boost fs_method sigmav lifetime comment"

class model_approx : 

    _attribute_list = ['process', 'mDM', 'approx_shape', 'approx_params', 'sigmav', 
                        'lifetime', 'fion_H_over_fheat', 'fion_He_over_fheat',
                        'fexc_over_fheat', 'force_init_cond', 'xe_init', 'Tm_init', 'comment']

    _process_dflt = 'decay'
    _mDM_dflt = 1e+10
    _approx_shape_dflt = 'schechter'
    _approx_params_dflt = [1., 1., 1.]
    _sigmav_dflt = 3e-26
    _lifetime_dflt = 0
    _fion_H_over_fheat_dflt  = -1
    _fion_He_over_fheat_dflt = -1
    _fexc_over_fheat_dflt    = -1
    _force_init_cond_dflt    = False
    _xe_init_dflt            = -1
    _Tm_init_dflt            = -1
    _comment_dflt = ''

    _allowed_shapes = ['constant', 'exponential', 'schechter']
    _allowed_processes   = ['none', 'swave', 'decay']

    
    def __init__(self, process: str = None, 
                mDM: float = None, approx_shape: str = None,
                approx_params: str = None, sigmav: float = None, lifetime: float = None, 
                fion_H_over_fheat: float = None, fion_He_over_fheat: float = None, fexc_over_fheat: float = None,
                force_init_cond: bool = None, xe_init:float = None, Tm_init:float = None, comment: str = None, index: int = None) :
        
        self.index              = index
        self.process            = process            if (process is not None)            else self._process_dflt
        self.mDM                = mDM                if (mDM is not None)                else self._mDM_dflt
        self.approx_shape       = approx_shape       if (approx_shape is not None)       else self._approx_shape_dflt 
        self.approx_params      = approx_params      if (approx_params is not None)      else self._approx_params_dflt 
        self.sigmav             = sigmav             if (sigmav is not None)             else self._sigmav_dflt
        self.lifetime           = lifetime           if (lifetime is not None)           else self._lifetime_dflt
        self.fion_H_over_fheat  = fion_H_over_fheat  if (fion_H_over_fheat is not None)  else self._fion_H_over_fheat_dflt
        self.fion_He_over_fheat = fion_He_over_fheat if (fion_He_over_fheat is not None) else self._fion_He_over_fheat_dflt
        self.fexc_over_fheat    = fexc_over_fheat    if (fexc_over_fheat is not None)    else self._fexc_over_fheat_dflt
        self.force_init_cond    = force_init_cond    if (force_init_cond is not None)    else self._force_init_cond_dflt
        self.xe_init            = xe_init            if (xe_init is not None)            else self._xe_init_dflt
        self.Tm_init            = Tm_init            if (Tm_init is not None)            else self._Tm_init_dflt
        self.comment            = comment            if (comment is not None)            else self._comment_dflt

        if self.process not in self._allowed_processes : 
            raise ValueError('The process', self.process, 'is not in the list of allowed values:', self._allowed_processes )

    def __str__(self):
        string_params = "("
        for ip, param in enumerate(self.approx_params):
            string_params += str("{:.2e}".format(param)) 
            if ip != len(self.approx_params)-1:
                string_params += ","
        string_params += ")"
        string_to_print = self.process + ", mDM:" + str("{:.2e}".format(self.mDM)) + ", shape:" + self.approx_shape  + \
            ", params:"  + string_params + \
            ", sigv:" + str("{:.3e}".format(self.sigmav)) + ", lftm:" + str("{:.3e}".format(self.lifetime)) + \
            ", fHifh:" + str("{:.2e}".format(self.fion_H_over_fheat)) + ", fHeifh:" + str("{:.2e}".format(self.fion_He_over_fheat)) + \
            ", fexfh:" + str("{:.2e}".format(self.fexc_over_fheat)) + \
            ", fini:" + str(self.force_init_cond) + \
            ", xei:" + str("{:.2e}".format(self.xe_init)) + \
            ", Tmi:" + str("{:.2e}".format(self.Tm_init))  + ", com:" + self.comment + "]"
        
        if self.index is not None :  
            string_to_print = "[ind:" + str(self.index)  + ", " + string_to_print 
        else :
             string_to_print = "[" + string_to_print

        return string_to_print

    def __eq__(self, other):
        answer = True
        for attr in self._attribute_list:
            if getattr(self, attr) != getattr(other, attr):
                answer = False
        return answer

    def is_valid(self): 

        """
        This function checks if the model is a valid one
        """
        answer = False

        if (self.process in self._allowed_processes) \
            and (self.approx_shape in self._allowed_shapes) \
            and self.mDM >= 0:
            answer=True

        return answer 

    def write_in_file(self, with_index = True): 

        string_params = "["
        for ip, param in enumerate(self.approx_params):
            string_params += str("{:.2e}".format(param)) 
            if ip != len(self.approx_params)-1:
                string_params += ":"
        string_params += "]"

        result = self.process + "\t" + \
                str("{:.2e}".format(self.mDM)) + "\t" + \
                self.approx_shape + "\t" + \
                string_params + "\t" + str("{:.2e}".format(self.sigmav)) + "\t" + \
                str("{:.2e}".format(self.lifetime)) + "\t" + \
                str(self.fion_H_over_fheat) + "\t" + \
                str(self.fion_He_over_fheat) + "\t" + \
                str(self.fexc_over_fheat) + "\t" + \
                str(self.force_init_cond) + "\t" + \
                str(self.xe_init) + "\t" + \
                str(self.Tm_init) + "\t" + \
                self.comment
        
        if with_index:
            result = str(self.index) + "\t" + result
        
        return result

    def file_header(self, with_index = True):
        if with_index:
            return "# index process mDM [eV] approx_shape approx_params sigmav lifetime fiH/fh fiHe/fh fexc/fh xe_init Tm_init [K] comment"
        else: 
            "# process mDM [eV] approx_shape approx_params sigmav lifetime fiH/fh fiHe/fh fexc/fh xe_init Tm_init [K] comment"

def parse_args(parser) :  

    ## Define all the arguments that can be parsed
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-r", "--remove", nargs='*', type=int, help="Remove the following entries in the database (integers)")
    group.add_argument("-sh", "--show", help="show the database", action="store_true")
    group.add_argument("-swave", "--DM_process_swave", help="set the DM process to swave", action="store_true")
    group.add_argument("-decay", "--DM_process_decay", help="set the DM process to decay", action="store_true")
    group.add_argument("-infile", "--input_file", help="Inpute file of the models to treat")

    parser.add_argument("-nobkr", "--no_backreaction", help="turn off backreaction", action="store_false")
    parser.add_argument("-m", "--DM_mass", nargs='*', type=float, help="DM mass in eV")
    parser.add_argument("-p", "--primary", nargs='*', help="primary particles")
    parser.add_argument("-b", "--boost", nargs='*', help="type of boost")
    parser.add_argument("-fs", "--fs_method", nargs='*', help="value of compute_fs_method")
    parser.add_argument("-c", "--comment", nargs='*', help="comment describing this particular run. By default there is no comment")
    parser.add_argument("-f", "--force_overwrite", help="force overwrite any existing entry in the database", action="store_true")
    parser.add_argument("-sigv", "--sigmav", nargs='*', type=float, help="cross-section in cm^3 s^{-1}. If none set to maximal value allowed by Planck18.")
    parser.add_argument("-lftm", "--lifetime", nargs='*', type=float, help="decay rate s^{-1}. If none set to maximal value allowed by Planck18.")
    parser.add_argument("-nomp", "--nthreads_omp", type=int, help="Number of threads used by the code")
    
    ## Special arguments when we use approximations
    parser.add_argument("-approx", "--approximate", help="Run an approximate energy injection", action="store_true")
    parser.add_argument("-shape", "--approximate_shape", nargs='*', help="Shape of fheat")
    parser.add_argument("-params", "--approximate_params", nargs='*', type=float, help="Parameters to feed the approximate shape of fheat")
    parser.add_argument("-fionH_fh", '--fion_H_over_fheat', nargs=1, type=float, help="Constant ratio of fion_H over f_heat")
    parser.add_argument("-fionHe_fh", '--fion_He_over_fheat', nargs=1, type=float, help="Constant ratio of fion_He over f_heat")
    parser.add_argument("-fexc_fh", '--fexc_over_fheat', nargs=1, type=float, help="Constant ratio of fexc over f_heat")
    parser.add_argument("-xe_init", '--xe_init', nargs=1, type=float, help="Initial value for the ionised fraction")
    parser.add_argument("-Tm_init", '--Tm_init', nargs=1, type=float, help="Initial value for the IGM temperature fraction [K]")
    parser.add_argument("-force_init", '--force_init_cond', help="Impose these new initial conditions", action="store_true")
    
    return parser.parse_args()



class DatabaseManager:
    
    def __init__(self, path:str = None, cache_path:str = None):
        
        self.path = path 
        self.cache_path = cache_path

        self.path_database_file = self.path + "/.database.txt"
        self.path_result_run = self.path + "/result_run_"
        self.path_brightness_temp = self.path + "/BrightnessTemp_"
        self.cache_path_folder = self.cache_path + "/_cache_"
        
        if not os.path.exists(self.path) : 
            os.mkdir(self.path)
        if not os.path.exists(self.cache_path):
            os.mkdir(self.cache_path)


    # Define here some abtsract methods
    @abstractmethod
    def define_model():
        pass
    
    @abstractmethod
    def read_database():
        pass

    @abstractmethod
    def search():
        pass


    # Define here the methods common to all databases

    def search_models(self, reference_model, models_arr: list) -> int:
        
        """
        Return the index that correspond to models models_arr (if they exist)
        """

        index = -1

        if models_arr == None : 
            models_dict, _ = self.read_database()

        i = 0
        found = False

        while found == False and i < len(models_arr) : 
            if reference_model == models_arr[i] : 
                index = models_arr[i].index
                found = True
            
            i=i+1

        return index

    
    def get_models_database(self, index: list or int, models_arr: list=None) : 

        """
        Return an array the models at position index (int or list)
        """

        if models_arr == None : 
            models_arr, _ = self.read_database()

        if not isinstance(index, list) :
            index = [index]

        models_out = []

        for ind in index: 
            if ind >= 0 and ind < len(models_arr) :  
                models_out.append(models_arr[ind])
            else : 
                models_out.append(None)

        return models_out


    
    def remove_entry_database(self, index, models_arr: list[model] = None) :

        if models_arr == None: 
            models_arr, _ = self.read_database()

        for i in range(index, len(models_arr)) : 
            if i == index : 

                if os.path.exists(self.path_result_run + str(models_arr[i].index) + ".txt") : 
                    os.remove(self.path_result_run + str(models_arr[i].index) + ".txt")
                else : 
                    print("The file:", self.path_result_run + str(models_arr[i].index) + ".txt", "does not exist in the first place.")
                    print("The database has missed a deletion ... entry only removed from the database file.")
                
                if os.path.exists(self.path_brightness_temp + str(models_arr[i].index)) : 
                    shutil.rmtree(self.path_brightness_temp + str(models_arr[i].index))

                if os.path.exists(self.cache_path_folder + str(models_arr[i].index)) : 
                    shutil.rmtree(self.cache_path_folder + str(models_arr[i].index))

                models_arr.pop(i)

            else : 
                if os.path.exists(self.path_result_run + str(models_arr[i-1].index) + ".txt") : 
                    os.rename(self.path_result_run + str(models_arr[i-1].index) + ".txt", self.path_result_run + str(models_arr[i-1].index-1) + ".txt")
                if os.path.exists(self.path_brightness_temp + str(models_arr[i-1].index)) : 
                    os.rename(self.path_brightness_temp + str(models_arr[i-1].index), self.path_brightness_temp + str(models_arr[i-1].index-1))
                if os.path.exists(self.cache_path_folder + str(models_arr[i-1].index)) : 
                    os.rename(self.cache_path_folder + str(models_arr[i-1].index), self.cache_path_folder + str(models_arr[i-1].index-1))
                
                models_arr[i-1].index = models_arr[i-1].index - 1
            

    def remove_entries_database(self, indices, force_deletion = False) : 

        """
        Remove several entries in the database
        """
        
        models_arr, _ = self.read_database()
        
     

        # If the input is not a list we make it a list of a single element
        if not isinstance(indices, list) :
            indices = [indices]

        if max(indices) >= len(models_arr)  :
            print("Error: cannot remove some of these items, they do not exits")
            exit(0)

        # This is to get the king of model we are dealing with before removing the models from models_arr
        # This is used in case models_arr ends up with no model inside, to still put the correct header
        # in the database file. This piece of code could be optimised
        model_type = copy.deepcopy(models_arr[0])

        if not force_deletion :
            print("Are you sure you want to remove these entries from the database (y/n):")
            for ind in indices: 
                if ind < len(models_arr)  :
                    print(models_arr[ind])
            
            if input() not in ['y', 'yes', 'Y', 'YES', 'Yes'] :
                print("Not removing anything")
                exit(0)

        j = 0
        for ind in indices :
            self.remove_entry_database(ind - j, models_arr) # remove a single entry
            j = j+1

        self.write_database_file(models_arr, model_type)



    def add_models_database(self, new_models: list, force_overwrite: bool=False, models_arr: list=None) : 

        """
        Return the same dictionnary as new_model but with the correct indices (according to what is already in the database).
        If the model has not been computed yet it is simply an index more
        If the model already exist we make sure the user wants to owerwrite it

        Write the new models in the database
        Create (if they do not already exist) files corresponding to these models
        """
        
        file_database_existed = True

        if models_arr == None : 
            models_arr, file_database_existed = self.read_database()
    

        # If the input is not a list of new models we make it a list of a single element
        if not isinstance(new_models, list) :
            new_models = [new_models]
        
        # Search if some of the models exist in the database already
        indices = [self.search_models(mod, models_arr) for mod in new_models]
        
        # Initialise the table of output indices
        output_indices = [-1 for ind in indices]

        # List of models that alreeady exist in the database
        existing_models = []

        # Loop on all the indices / models 
        for i in range(0, len(indices)) : 
            if indices[i] > -1 : 
                print("The model", new_models[i], "already exists (registered at index: " +  str(indices[i]) + ")")
                existing_models.append(i)

        if len(existing_models) > 0 :

            if force_overwrite == False : 
                print(" --------------------- ")
                print("Do you want to overwrite these models (y/n)")

                answer_str = input()

                if answer_str in ['y', 'yes', 'Y', 'YES', 'Yes'] :
                    for ex_mod in existing_models :
                        output_indices[ex_mod] =  indices[ex_mod]
                else :
                    print("If you do not want to overwrite you need to differentiate the runs/models by adding/changing a comment with the -c option")
                    exit(0)

            else : 
                print(" --------------------- ")
                print("These models are overwritten by force")
                for ex_mod in existing_models :
                    output_indices[ex_mod] = indices[ex_mod]
        

        # Put the correct indices to the models that are not overwriten
        if len(output_indices) > len(existing_models) : # Otherwise means they are not all overwritten

            if len(models_arr) > 0 : 
                mymax = max([mod.index for mod in models_arr]) # Get the maximum value of current indices in the database
            else :
                mymax = -1

            new_ind = 1

            for j in range(0, len(output_indices)) :
                if output_indices[j] == -1 :# Not overwritten
                    output_indices[j] = mymax + new_ind
                    new_ind = new_ind + 1

        filename = self.path_database_file

        # All the models we need to run with the correct index
        output_models_arr = []

        header_printed = False

        # We add the new models to the models_arr with the correct indices and update the database
        for i in range(0, len(new_models)) : 
            
            # Here we deepcopy to avoid copied by reference
            temp_model = copy.deepcopy(new_models[i]) 
            temp_model.index = output_indices[i]
            output_models_arr.append(temp_model)

            if output_indices[i] >= len(models_arr) :

                print_header = (not file_database_existed) and (len(models_arr) == 0) and (header_printed is False)
                header_printed = self.append_database_file(temp_model, print_header, header_printed)
                
                # Create a file with the correct message for all the models
                fle = Path(self.path + "/result_run_" + str(temp_model.index) + '.txt')
                fle.touch(exist_ok=True)
                f = open(fle)            
        
        return  output_models_arr

    
    def append_database_file(self, mod, print_header: bool = False, header_printed: bool = False ) :
        
        filename = self.path_database_file

        with open(filename, 'a') as f:

            if print_header:
                print("# Database for:", self.path, file = f)
                print(mod.file_header(), file = f)
                header_printed = True
            
            print(mod.write_in_file(), file=f)

        return header_printed


    def write_database_file(self, models_arr: list, model_type = None, path: str = None):

        # Write the entire database file
        filename = self.path_database_file

        with open(filename, 'w') as f:
            print("# Database for:", self.path, file=f)
            print(model_type.file_header(), file = f)
            
            for mod in models_arr :
                print(mod.write_in_file(), file=f)
    


    def print_f_vs_rs(self, evolve_data, inj_energy_smooth, input) :

        # Extract the values from the dictionnary evolve_data
        rs = evolve_data['rs']

        f_H_ion_low   = evolve_data['f']['low']['H ion']
        f_H_ion_high  = evolve_data['f']['high']['H ion']
        f_He_ion_low  = evolve_data['f']['low']['He ion']
        f_He_ion_high = evolve_data['f']['high']['He ion']
        f_exc_low     = evolve_data['f']['low']['exc']
        f_exc_high    = evolve_data['f']['high']['exc']
        f_heat_low    = evolve_data['f']['low']['heat']
        f_heat_high   = evolve_data['f']['high']['heat']
        f_cont_low    = evolve_data['f']['low']['cont']
        f_cont_high   = evolve_data['f']['high']['cont']
        
        f_H_ion  = f_H_ion_low + f_H_ion_high
        f_He_ion = f_He_ion_low + f_He_ion_high
        f_exc    = f_exc_low + f_exc_high
        f_heat   = f_heat_low + f_heat_high
        f_cont   = f_cont_low + f_cont_high

        x_arr = evolve_data['x']
        Tm    = evolve_data['Tm']
        
        x_HII  = x_arr[:, 0]
        x_HeII = x_arr[:, 1]

        filename = self.path_result_run + str(input.index) + ".txt"
        
        with open(filename, 'w') as f:

            print(input.file_header(with_index=False), file = f)
            print(input.write_in_file(with_index=False), file=f)
            print("# redshift (1+z) | f_H_ion, f_He_ion, f_exc, f_heat, f_cont, inj_energy_smooth [eV/s(/nb_baryons)] | x_HII | x_HeII | Tm [eV]", file=f)
            
            for i in range(0, len(rs)) :
                print(rs[i], "\t", f_H_ion[i], "\t", f_He_ion[i], "\t",
                    f_exc[i], "\t", f_heat[i], "\t", f_cont[i], "\t",
                    inj_energy_smooth[i], "\t", x_HII[i], "\t", x_HeII[i], "\t", Tm[i], file=f)

    

    def print_f_vs_rs_from21cmFAST(self, evolve_data, input) :

        # Extract the values from the dictionnary evolve_data
        z = evolve_data['z']
        f = evolve_data['f']
        T = evolve_data['Tm']
        x = evolve_data['x']

        #print(len(f), f[-1])
        #print(len(z), z)

        filename = self.path_result_run + str(input.index) + ".txt"
        
        with open(filename, 'w') as ff:
            
            print(input.file_header(with_index=False), file = ff)
            print(input.write_in_file(with_index=False), file=ff)
            print("# redshift z | f_H_ion, f_He_ion, f_exc, f_heat, f_cont, inj_energy_smooth [erg/s(/nb_baryons)] | x_HII | Tm [eV]", file=ff)
            
            for iz, zval in enumerate(z) :
                print(zval, "\t", f[iz]['f_H_ION'], "\t", f[iz]['f_He_ION'], "\t",
                    f[iz]['f_EXC'], "\t", f[iz]['f_HEAT'], "\t", f[iz]['f_CONT'], "\t",
                    f[iz]['Inj_ENERGY_SMOOTH'], "\t", x[iz], "\t", T[iz], file=ff)


    def remove_show_models(self, args):
        
        if args.remove is not None :
            self.remove_entries_database(args.remove)
            exit(0)

        if args.show is not None and args.show == True :
            models_arr, file_existed = self.read_database()
            if file_existed and len(models_arr) > 0:
                print("The database currently contains the following models:")
            elif file_existed and len(models_arr) == 0:
                print("The database currently contains no models.")
            elif not file_existed:
                print("No database file found.")
            for mod in models_arr:
                print(mod)
            exit(0)
        

    def common_parsed_args(self, args):
        
        force_overwrite = args.force_overwrite
        nomp            = args.nthreads_omp if (args.nthreads_omp is not None) else 1

        if args.input_file is not None : 
            return self.read_input_file(args.input_file), force_overwrite, nomp

        return force_overwrite, nomp


    def init_values_from_parser(self, parse_arr, value_arr): 

        for i in range(0, len(parse_arr)) :
            if parse_arr[i] != None : 
                value_arr[i] = parse_arr[i]

        # Check if the number of input argument is good
        first = True
        n = 1
        ind = 0
        for i in range(0, len(value_arr)) :
            if len(value_arr[i]) > 1 and first :
                n = len(value_arr[i])
                first = False
            if len(value_arr[i]) > 1 and not first :
                if len(value_arr[i]) != n :
                    print("Error in the input argument")
                    exit(0)

        # If we have the value of n we change everything to the same size
        for i in range(0, len(value_arr)) :
            if len(value_arr[i]) == 1:
                value_arr[i] = [value_arr[i][0] for j in range(0,n)]

        return value_arr, n


#######################################
## DATABASE Manager for DH type inputs
#######################################

class DHDatabase(DatabaseManager) : 

    def __init__(self, path:str = None, cache_path:str = None) :
        path += "/darkhistory"
        cache_path += "/darkhistory"
        super(DHDatabase, self).__init__(path, cache_path)

    def define_models(self, args) :

        self.remove_show_models(args)
        output_val =  self.common_parsed_args(args)

        ## Initialise the parameters
        swave = False
        decay = False
        mDM = [0]
        primary = ['e']
        boost = ['none']
        fs_method = ['no_He']
        comment = ['']
        sigmav = [0]
        lifetime = [0]

        swave = args.DM_process_swave
        decay = args.DM_process_decay
        bkr = args.no_backreaction

        
       
        parse_arr = [args.DM_mass, args.primary, args.boost, args.fs_method, args.sigmav, args.lifetime, args.comment]
        value_arr = [mDM, primary, boost, fs_method, sigmav, lifetime, comment]
        value_arr, n = self.init_values_from_parser(parse_arr, value_arr)
        
        process = ""

        if swave == True :
            process = ["swave" for i in range(0, n)]
            
            if any([val != 0 for val in value_arr[5]]): 
                print("WARNING: with swave setting the lifetime to 0")
            value_arr[5] = [0  for i in range(0, n)]
        
        if decay == True :
            process = ["decay" for i in range(0, n)]

            if any([val != 0 for val in value_arr[4]]): 
                print("WARNING: with decay setting sigmav to 0")
            value_arr[4] = [0 for i in range(0, n)]

            if any([val != 'none' for val in value_arr[2]]): 
                print("WARNING: with decay setting the boosts to 'none'")
            value_arr[2] = ['none'  for i in range(0, n)]  

        if decay == False and swave == False :
            if n > 1 :
                print("No reason to run the code more than one time with no DM injection")
                exit(0)
            process = ['none']
            bkr = False
            value_arr = [[0.], ['none'], ['none'], ['none'], [0.], [0.], value_arr[-1]]

        bkr = [bkr for i in range(0, n)]
        all_input = [bkr, process, *value_arr]

        input = []
        for i in range(0, len(all_input[0])) :
            
            mod = model() # Initialised with the default values
            
            for j in range(0, len(model._attribute_list)):
                setattr(mod, model._attribute_list[j],  all_input[j][i])
            
            if mod.is_valid():
                input.append(mod)
            else:
                print("Error: trying to run an invalid model")
                exit(0)

        print("The following models have been passed in input: ")
        for mod in input : 
            print(mod)

        return input, *output_val



    def read_database(self) : 
        
        """
        Read the different models registered in the database
        Convert them into a dictionnary 

        Returns the dictionnary created from the database file
        Argument: database_file_existed is set to False if the database file did not exist previously
        """

        filename = self.path_database_file
        database_file_existed = os.path.exists(filename)
        #fle = Path(filename)
        #fle.touch(exist_ok=True)

        if database_file_existed is False:
            return [], False


        with open(filename, 'r') as f:
            input_data = f.readlines()
        
        models = []
        for p in input_data:
            model_str = p.split()
            if len(model_str) > 0 and model_str[0] != '#' :
                models.append(p.split())
        
        models_arr = []
        for i in range(0, len(models)) :

            if len(models[i]) < 10 : 
                comment = ''
            else :
                comment = models[i][9]

            bkr = True
            if models[i][1] == 'False' : 
                bkr = False

            # append the new models to the list
            models_arr.append(model(bkr, models[i][2], float(models[i][3]), models[i][4], models[i][5], 
                                    models[i][6], float(models[i][7]), float(models[i][8]), 
                                    comment, int(models[i][0]) ))
        
        return models_arr, database_file_existed



    def search(self, process=None, bkr=None, mDM=None, primary=None, boost=None, fs_method=None, sigmav=None, lifetime=None, comment=None) -> list[int] : 
        
        """
        Search for the entry in the database with the correct behaviour
        Overwritten function search_in_database where we give the values instead of the dictionnary directly
        """

        models_arr, _ = self.read_database()


        indexlist = []
        keylist = []
        valuelist = []

        if process is not None:
            keylist.append('process')
            valuelist.append(process if isinstance(process, list) else [process])
        if bkr is not None:
            keylist.append('bkr')
            valuelist.append(bkr if isinstance(bkr, list) else [bkr])
        if mDM is not None: 
            keylist.append('mDM')
            valuelist.append(mDM if isinstance(mDM, list) else [mDM])
        if primary is not None:
            keylist.append('primary')
            valuelist.append(primary if isinstance(primary, list) else [primary])
        if boost is not None:
            keylist.append('boost')
            valuelist.append(boost if isinstance(boost, list) else [boost])
        if fs_method is not None:
            keylist.append('fs_method')
            valuelist.append(fs_method if isinstance(fs_method, list) else [fs_method])
        if sigmav is not None:
            keylist.append('sigmav')
            valuelist.append(sigmav if isinstance(sigmav, list) else [sigmav])
        if lifetime is not None:
            keylist.append('lifetime')
            valuelist.append(lifetime if isinstance(lifetime, list) else [lifetime])
        if comment is not None:
            keylist.append('comment')
            valuelist.append(comment if isinstance(comment, list) else [comment])

        for mod in models_arr:
            match = True
            for _ikey, _key in enumerate(keylist):
                if getattr(mod, _key) not in valuelist[_ikey]:
                    match = False
            if match == True:
                indexlist.append(mod.index)

        return indexlist


    def read_input_file(self, filename:str) -> list: 
        
        """
        Read the different models registered in an input file
        Convert them into a dictionnary 

        Returns the dictionnary created from the input file
        """

        if os.path.exists(filename):
            with open(filename, 'r') as f:
                input_data = f.readlines()
        else:
            print("Error input file does not exist")
            exit(0)
        
        models = []
        for p in input_data:
            model_str = p.split()
            if len(model_str) > 0 and p[0] != '#' :
                models.append(p.split())
        
        models_arr = []
        for i in range(0, len(models)) :

            if len(models[i]) < 9 : 
                comment = ''
            else :
                comment = models[i][8]

            bkr = True
            if models[i][0] == 'False' : 
                bkr = False

           # append the models to the list
            models_arr.append(model(bkr, models[i][2], float(models[i][3]), models[i][4], models[i][5], models[i][6], float(models[i][7]), float(models[i][8]), comment, None))

        return models_arr



#######################################
## DATABASE Manager for DH type inputs
#######################################

class ApproxDepDatabase(DatabaseManager) : 

    def __init__(self, path:str = None, cache_path:str = None) :
        path += "/approx"
        cache_path += "/approx"
        super(ApproxDepDatabase, self).__init__(path, cache_path)


    def define_models(self, args) :

        self.remove_show_models(args)
        output_val =  self.common_parsed_args(args)

        ## Initialise the parameters
        swave = False
        decay = False
        mDM = 0
        approx_shape = 'constant'
        approx_params = [1.]
        comment = ''
        sigmav = 0
        lifetime = 0
        fion_H_over_fheat  = -1.
        fion_He_over_fheat = -1.
        fexc_over_fheat    = -1.
        force_init_cond    = False
        xe_init            = -1.
        Tm_init            = -1. 

        swave = args.DM_process_swave
        decay = args.DM_process_decay
       
        parse_arr = [args.DM_mass, args.approximate_shape, args.approximate_params, args.sigmav, args.lifetime,
                     args.fion_H_over_fheat, args.fion_He_over_fheat, args.fexc_over_fheat, args.force_init_cond, 
                     args.xe_init, args.Tm_init, args.comment]
        value_arr = [mDM, approx_shape, approx_params, sigmav, lifetime, fion_H_over_fheat, 
                    fion_He_over_fheat, fexc_over_fheat, force_init_cond, xe_init, Tm_init, comment]


        for i in range(0, len(parse_arr)) :
            if parse_arr[i] != None : 
                if i != 2 and i!= 8: # Exceptions for params and force_init_cond arguments
                    value_arr[i] = parse_arr[i][0]
                else: 
                    value_arr[i] = parse_arr[i]

        if swave == True :
            process = "swave"
            if value_arr[4] != 0: 
                print("WARNING: with swave setting the lifetime to 0")
                value_arr[4] = 0
        
        if decay == True :
            process = "decay"
            if value_arr[3] != 0: 
                print("WARNING: with decay setting sigmav to 0")
                value_arr[3] = 0

        all_input = [process, *value_arr]


        input = []
            
        mod = model_approx() # Initialised with the default values
            
        for j in range(0, len(model_approx._attribute_list)):
            setattr(mod, model_approx._attribute_list[j],  all_input[j])
            
        if mod.is_valid():
            input.append(mod)
        else:
            print("Error: trying to run an invalid model")
            exit(0)

        print("The following model have been passed in input: ") 
        print(mod)

        return input, *output_val



    def read_database(self) : 
        
        """
        Read the different models registered in the database
        Convert them into a dictionnary 

        Returns the dictionnary created from the database file
        Argument: database_file_existed is set to False if the database file did not exist previously
        """

        filename = self.path_database_file
        database_file_existed = os.path.exists(filename)

        if database_file_existed is False:
            return [], False


        with open(filename, 'r') as f:
            input_data = f.readlines()
        
        models = []
        for p in input_data:
            model_str = p.split()
            if len(model_str) > 0 and model_str[0] != '#' :
                models.append(p.split())
        
        models_arr = []
        for i in range(0, len(models)) :

            if len(models[i]) < 14 : 
                comment = ''
            else :
                comment = models[i][13]

            force_init_cond = False
            if models[i][10] == 'True':
                force_init_cond = True

            params_str = models[i][4].rstrip(']').lstrip('[').split(":")
            params = [float(p_str) for p_str in params_str]

            # append the new models to the list
            models_arr.append(model_approx(models[i][1], float(models[i][2]), models[i][3], params, float(models[i][5]), 
                                    float(models[i][6]), float(models[i][7]), float(models[i][8]),  float(models[i][9]),
                                    force_init_cond, float(models[i][11]), float(models[i][12]),
                                    comment, int(models[i][0])))
        
        return models_arr, database_file_existed


    def search(self, process=None, mDM=None, approx_shape=None, approx_params=None, sigmav=None, lifetime=None, 
                fion_H_over_fheat=None, fion_He_over_fheat = None, fexc_over_feat = None, force_init_cond=None,
                xe_init = None, Tm_init = None, comment=None) -> list : 
        
        """
        Search for the entry in the database with the correct behaviour
        Overwritten function search_in_database where we give the values instead of the dictionnary directly
        """

        models_arr, _ = self.read_database()


        indexlist = []
        keylist = []
        valuelist = []


        inputs = [process, mDM, approx_shape, approx_params, sigmav, lifetime, fion_H_over_fheat, 
                fion_He_over_fheat, fexc_over_feat, force_init_cond, xe_init, Tm_init, comment]
        inputs_text = model_approx._attribute_list

        for ii, inp in enumerate(inputs):
            if inp is not None and inputs_text[ii] != 'approx_params':
                keylist.append(inputs_text[ii])
                valuelist.append(inp if isinstance(inp, list) else [inp])
            elif inp is not None and inputs_text[ii] == 'approx_params':
                keylist.append(inputs_text[ii])
                valuelist.append(inp if isinstance(inp[0], list) else [inp])
       
        for mod in models_arr:
            match = True
            for _ikey, _key in enumerate(keylist):
                if getattr(mod, _key) not in valuelist[_ikey]:
                    match = False
            if match == True:
                indexlist.append(mod.index)

        return indexlist


  

    