import os 
import shutil


def make_directory(path: str, clean_existing_dir:bool = True):
    
    if not os.path.exists(path): 
        os.mkdir(path)
    else:
        if clean_existing_dir is True:
            clean_directory(path)
        else:
            print("The directory "  + path + " already exists")


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

