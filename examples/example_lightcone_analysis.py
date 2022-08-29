############################################################################
#    Code to analyse the lightcone data 
#    
#    Gaetan Facchinetti from an original version of Quentin Decant 
#    Based on the 21cmFAST documentation
#    gaetan.facchinetti@ulb.be | quentin.decant@ulb.be
############################################################################

import os
from tkinter import E
import numpy as np
import csv
import matplotlib.pyplot as plt
from powerbox.tools import get_power
import shutil
from os import listdir
from os.path import isfile, join


# This function is called to do the whole Analysis
# nameRun = name of folder where we store everything
# lightcone = lightcone object on which we do the analysis
# nchunks = number of chunks used when computing the powerspectra
# plotsFixedZ = controls if we want to plot the powerspectra for fixed z but varying k 
def make_analysis(path, lightcone, n_psbins=50, nchunks = None):
    if nchunks == None:
        nchunks = len(lightcone.node_redshifts)
    if os.path.exists(path+"/Data"):
        clean_directory(path + "/Data")
    else:
        os.mkdir(path + "/Data")
    save_global_quantities(path, lightcone)
    compute_and_save_powerspectra(path, lightcone, n_psbins=n_psbins, nchunks=nchunks)
    

# Attention I modified this function
# Here I give the full path to the file
# Moreover I clean the directory if it already exists
def make_run_directory(path):
    if not os.path.exists(path): 
        os.mkdir(path)
        print("Successfully created the directory %s " % path)
    else:
        clean_directory(path)
        print("Successfully cleaned the directory %s " % path)
    

# Function copied from online
# Clean the directory nameRun
def clean_directory(nameRun):
    for filename in os.listdir(nameRun):
        file_path = os.path.join(nameRun, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


# This function is used to make the directories where we save the lightcones
# As well as saving the info in it.
# path = name of path of the main folder where we store everything
# lightcone = lightcone object on which we do the analysis
def make_and_save_lightcone_directory(path,lightcone):
        try:
            pathLightCone = path + "/Lightcone"
            if not os.path.exists(pathLightCone): 
                os.mkdir(pathLightCone)
            else:
                print("The lightcone folder already exists")
        except OSError:
            print("Creation of the directory %s failed" % pathLightCone)
        else:
            print("Successfully created the directory %s " % pathLightCone)
            save_input_data(pathLightCone, lightcone)
            lightcone.save(fname = "Lightcone.h5", direc = pathLightCone)
        return  pathLightCone

# This function is used to make the directory for the plots. As well as generating
# The plots of the global quantities, the powerspectra at fixed k and redshift
# path = name of path of the main folder where we store everything
# pathPowerspectraToPlot = name of path of the main folder where we store everything
# pathPowerspectraRedshift = name of path of the main folder where we store everything
# plotsFixedZ = controls if we want to plot the powerspectra for fixed z but varying k 
def make_plots_directory(path):
    try:
        os.mkdir(path + "/Plots")
        os.mkdir(path + "/Plots/Global_quantities")
        os.mkdir(path + "/Plots/Powerspectra")
    except OSError:
        print("Creation of the directory %s failed" % path + "/Plots")
        print("Creation of the directory %s failed" % path + "/Plots/Global_quantities")
        print("Creation of the directory %s failed" % path + "/Plots/Powerspectra")
    else:
        print("Successfully created the directory %s " % path + "/Plots")
        print("Successfully created the directory %s " % path + "/Plots/Global_quantities")
        print("Successfully created the directory %s " % path + "/Plots/Powerspectra")


#This function is used to save the input data we used to make the lightcone
def save_input_data(pathLightCone,lightcone):
    nameFile = "InputParametersLightcone.txt"
    parametersLightcone = lightcone._input_rep()
    tempParameterLightcone = parametersLightcone.split(")")
    with open(pathLightCone+"/"+nameFile,"w") as f:
        for item in tempParameterLightcone:
            f.write(item + ")" + "\n \n")
        f.close()
    
# This function is used to save the different global quantities in one file for different redshifts
def save_global_quantities(path,lightcone):
    lightconeRedshifts = np.array([lightcone.node_redshifts]).T 
    tempArray = lightconeRedshifts
    keyArray = np.array([['z']])
    for key in lightcone.global_quantities:
        keyArray = np.hstack((keyArray,[[str(key)]]))
        quantity = np.array([lightcone.global_quantities[key]]).T
        tempArray = np.hstack((tempArray,quantity))
    finalArray = np.vstack((keyArray,tempArray))
    np.savetxt(path + "/Data/global_quantities.csv", finalArray,  fmt="%s",delimiter=",")

# This function is used to compute and save the different powerspectra in one file for different redshifts.
# For the moment with take the number of n_chunks the same as the number of redshift nodes.
# Still not very clear to me how all these convolutioning of the data is done
def compute_and_save_powerspectra(path, lightcone, n_psbins=50, nchunks=10):
    powerspectras, redshift_indices = compute_powerspectra(lightcone, n_psbins=n_psbins, nchunks=nchunks)
    save_powerspectra(path, powerspectras, redshift_indices)
    powerspectraNoNaN = remove_nan_powerspectra(powerspectras)
    save_powerspectra(path, powerspectraNoNaN, redshift_indices, noNaN=True)

# This function is used to save the powerspectra. First it saves the original powerspectra,
# as well as constructing a table for the powerspectra at fixed k (he takes the closest one to the k1 k2 k3)
# given. The second part of the code saves the powerspectra for fixed redshift without the nan's.
def save_powerspectra(path,powerspectra,redshiftlist,noNaN=False):
    pathPowerspectra = []

    if noNaN == False:
        pathDataPS = path + "/Data/Powerspectra_Original" 
    else:
        pathDataPS = path + "/Data/Powerspectra_NoNan" 

    os.mkdir(pathDataPS)
    for i in range(len(redshiftlist)):
        keys = np.array([['k'],['delta']]).T
        data = np.hstack((np.array([powerspectra[i]['k']]).T,np.array([powerspectra[i]['delta']]).T))
        arrayToSave = np.vstack((keys,data))
        z = redshiftlist[i]
        nameRedshift = str(z).replace(".", "_")
        pathPowerspectrum = pathDataPS + "/PS_z_" + nameRedshift + ".csv"
        pathPowerspectra.append(pathPowerspectrum)
        np.savetxt(pathPowerspectrum, arrayToSave, fmt="%s",delimiter=",")


# This function makes plots of the global quantities. First we load in the data into a dictionary and then make the plots.
# The loading in is not very elegantly written, but it works..
def make_plots_global_quantities(path, plot=False):
    dataGlobalQuantities = {}
    with open(path + "/Data/global_quantities.csv", "r") as csvf:
        content = csv.reader(csvf, delimiter=",")
        i=0
        for row in content:
            for j in range(len(row)):
                if i==0: #Check that we are in the first row
                    keyrow = row
                    key = keyrow[j]
                    dataGlobalQuantities[key]=[]
                else:
                    dataGlobalQuantities[keyrow[j]].append(float(row[j]))
            i=1
        if plot == True: 
            for key in dataGlobalQuantities.keys():
                if key !="z":
                    fig = plt.figure()
                    ax= fig.gca()
                    ax.plot(np.array(dataGlobalQuantities["z"]), np.array(dataGlobalQuantities[key]))
                    ax.set_xlabel(r"$z$")
                    if key == "brightness_temp":
                        ax.set_ylabel(r"$\overline{\delta T_b}~\rm [mK]$")
                    if key == "xH_box":
                        ax.set_ylabel(r"$x_{\rm H}$")
                    if key == "density":
                        ax.set_ylabel(r"$\rho ~\rm units?$")
                    plt.minorticks_on()
                    fig.savefig(path + "/Plots/Global_quantities/" + key + ".pdf", bbox_inches ='tight')
                    plt.close(fig)

        z = np.array(dataGlobalQuantities["z"])
        dTb = np.array(dataGlobalQuantities["brightness_temp"])
        xH_box = np.array(dataGlobalQuantities["xH_box"])
        try:
            x_e_box = np.array(dataGlobalQuantities["x_e_box"])
        except :
            x_e_box = None

        try:
            Ts_box = np.array(dataGlobalQuantities["Ts_box"]) 
        except:
            Ts_box = None
        
        try:
            Tk_box = np.array(dataGlobalQuantities["Tk_box"])
        except:
            Tk_box = None

    return z, dTb, xH_box, x_e_box, Ts_box, Tk_box

# k in Mpc^{-1}
def make_plots_powerspectra(path, k_ref, plot=False): 
    
    k = [] 
    delta_k = []
    delta_z = []
    z_PS = []

    k_ideal = k_ref # Mpc^{-1}
    folder = path +  '/Data/Powerspectra_NoNan'
    filename = [folder + '/' + f for f in listdir(folder) if isfile(join(folder, f))]

    for fname in filename :
        file_data = open(fname)
        data = csv.reader(file_data)
        header = []
        header = next(data)
        rows = []
        for row in data:
            rows.append([float(r) for r in row])
        rows = np.transpose(rows)
        k.append(rows[0])
        delta_k.append(rows[1])

        #print("Here:",  float((((fname.split('/')[-1]).split('.'))[0].split('_'))[3]) + float((((fname.split('/')[-1]).split('.'))[0].split('_'))[]))
        temp = ((fname.split('/')[-1]).split('.'))[0].split('_')
        z_PS.append(float(temp[2]) + float(temp[3])*10**(-len(temp[3])))

        # For this filename we search the value of delta_z
        k_approx = min(k[-1], key=lambda x:abs(x-k_ideal))
        index_k = np.where(k[-1] == k_approx) 
        delta_z.append(delta_k[-1][index_k].tolist()[0]) 

    if plot == True:
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(z_PS, delta_z, label=r'$k={:2.2f}~\rm Mpc^{{-1}}$'.format(k_approx))
        ax.set_ylim(1e-2, 1e+4)
        #ax.set_title("Powerspectrum")
        ax.set_xlabel(r"$z$")
        ax.set_ylabel(r"$\overline{\delta T_{b}}^2 \Delta_{21}^2~{\rm [mK^2]}$")
        plt.minorticks_on()
        ax.set_yscale("log")
        ax.legend()
        fig.savefig(path + '/Plots/Powerspectra/PS_fixed_k.pdf', bbox_inches ='tight')
        plt.close(fig)
    
    ####
        

    return z_PS, delta_z, k_approx

##################################################################################################
##################################################################################################
####################### Functions for Computing powerspectra #####################################
##################################################################################################
##################################################################################################

#This function is used to compute the powerspectra
def compute_powerspectra(lightcone, n_psbins=50, nchunks=10):
    powerspectras ,redshift_indices = powerspectra(lightcone, n_psbins=n_psbins, nchunks=nchunks)
    return powerspectras, redshift_indices



#This function is taken from tutorial 21cmFAST
## WARNING: n_psbins cannnot be chosen too large otherwise during the average process 
##  it produces nan results and the average fluctuates a lot.
##  The higher the value of the HII_DIM (number of cells in the box) the higher we can consider n_psibin 
##  because it will produce more bins in the Fourier space and avoid averaging over nothing

def compute_power(
   box, #This defines which type of quantity we want the power specrum form
   length, # This gives the length of the box the input should be a 3D vector
   n_psbins, #Number of bins (in k) that is used for the computation
   log_bins=True,
   ignore_kperp_zero=True,
   ignore_kpar_zero=False,
   ignore_k_zero=False,
):
    # Determine the weighting function required from ignoring k's. 
    k_weights = np.ones(box.shape, dtype=np.int)
    n0 = k_weights.shape[0]
    n1 = k_weights.shape[-1]

    if ignore_kperp_zero:
        k_weights[n0 // 2, n0 // 2, :] = 0
    if ignore_kpar_zero:
        k_weights[:, :, n1 // 2] = 0
    if ignore_k_zero:
        k_weights[n0 // 2, n0 // 2, n1 // 2] = 0
    
    # Here we actually calculate the power with a built-in function. 
    # The params are quite self-explanatory
    res = get_power(
        box,
        boxlength=length,
        bins=n_psbins,
        bin_ave=False,
        get_variance=False,
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
    
#This function is taken from tutorial 21cmFAST
#With this function we are really going to calculate the binned Spectra (I think..)
def powerspectra(brightness_temp, n_psbins=50, nchunks=10, min_k=0.1, max_k=1.0, logk=True):
    data = []
    chunk_indices = list(range(0,brightness_temp.n_slices,round(brightness_temp.n_slices / nchunks),))
    redshift_indices = list(range(0,brightness_temp.n_slices,round(brightness_temp.n_slices / nchunks),))  #Quentin
    #n_slices returns the number of redshift slices in the lightcone
    #Basically this means that we bin the redshift slices in chunks. The amount controlled by n_chunks
    
    if len(chunk_indices) > nchunks:
        chunk_indices = chunk_indices[:-1]
    chunk_indices.append(brightness_temp.n_slices)

    redshift_indices = chunk_indices.copy() #Quentin
    redshifts_lightcone = brightness_temp.lightcone_redshifts 
    # We need to define this, because if we access it directly using the method commented below  
    # when trying to find the redshift indices it slows the code extremely down:
    # redshift_indices[i] = lightcone.lightcone_redshifts[(end-start)//2+start]

    for i in range(len(chunk_indices)-1):
        start = chunk_indices[i]
        end = chunk_indices[i + 1]
        chunklen = (end - start) * brightness_temp.cell_size #"""Cell size [Mpc] of the lightcone voxels."""
        redshift_indices[i] = redshifts_lightcone[(end-start)//2+start] #Quentin we take the middle redshift as the redshift of the chunk
        power, k = compute_power(
            brightness_temp.brightness_temp[:, :, start:end], (brightness_temp.lightcone_dimensions[0], brightness_temp.lightcone_dimensions[1], chunklen), #Quentin: made a change from BOX_length -> dimensions of lightcone
            n_psbins,
            log_bins=logk,
        )
        data.append({"k": k, "delta": power * k ** 3 / (2 * np.pi ** 2)})
    redshift_indices.pop() #We remove the last element Quentin
    return data , redshift_indices


# For some reason this function also changes the input list, eventough we make an explicit copy of it.
# Therefore, only use AFTER you have made a save of the original list with the NaN's.
def remove_nan_powerspectra(powerspectrum):
    powerspectrumNoNaN = powerspectrum.copy()
    for j in range(len(powerspectrumNoNaN)):
        listIndicesNan = []
        for i in range(len(powerspectrumNoNaN[j]["delta"])):
            if np.isnan(powerspectrumNoNaN[j]["delta"][i]):
                listIndicesNan.append(i)
        powerspectrumNoNaN[j]["delta"]=np.delete(powerspectrumNoNaN[j]["delta"],listIndicesNan)
        powerspectrumNoNaN[j]["k"]=np.delete(powerspectrumNoNaN[j]["k"],listIndicesNan)
    return powerspectrumNoNaN



