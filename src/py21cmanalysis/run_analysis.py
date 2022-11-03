import py21cmanalysis.power_spectra     as p21a_ps
import py21cmanalysis.global_quantities as p21a_gl
import py21cmanalysis.tools             as p21a_tools

def run_analysis(lightcone, path_output: str) -> None : 
    """ Save the important information in human readable format stored in a lightcone (and the lightcone) """

    p21a_tools.make_directory(path_output)
    lightcone.save(fname = "lightcone.h5", direc = path_output)

    # Export the data in human readable format
    z_centers, power_spectra = p21a_ps.compute_powerspectra_1D(lightcone=lightcone, nchunks=15, n_psbins=None, logk=True) 
    p21a_gl.export_global_quantities(path = path_output, lightcone = lightcone)
    p21a_ps.export_powerspectra_1D_vs_k(path=path_output, z_centers = z_centers, power_spectra = power_spectra)
    p21a_ps.export_powerspectra_1D_vs_z(path=path_output, z_centers = z_centers, power_spectra = power_spectra)
