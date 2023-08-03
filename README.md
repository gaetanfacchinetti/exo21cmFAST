# exo21cmFAST

**A semi-numerical cosmological simulation code for the radio 21-cm signal with dark matter energy injection.**

## Installing exo21cmFAST

We introduce several features to `21cmFAST <https://github.com/21cmfast/21cmFAST>`_ related to dark matter energy injection via s-wave annihilation or decay. The energy deposition can be evaluated in two ways, either using the `DarkHistory package <https://darkhistory.readthedocs.io/en/master/>`_ or with template functions. To use exo21cmFAST, download the repository and do a usual installation with::

    $ pip install -e .

inside the main folder. To use DarkHistory you need to download the package (see information on the DarkHistory documentation) and add these two lines to your PYTHONPATH::

    $ export PYTHONPATH=$PYTHONPATH:"folder containing DarkHistory"
    $ export PYTHONPATH=$PYTHONPATH:"folder containing DarkHistory"/DarkHistory


## What is new/different in exo21cmFAST ?

The new feature are implemented only for the run_lightcone function. In order to use the new features we provide new input parameters summarized below::

| User params:
| # General quantities
| "DM_MASS":         # (float) DM mass in eV
| "DM_PROCESS":      # (string) Energy injection process 'swave', 'decay', ... 
| "DM_SIGMAV":       # (float) Annihilation cross-section (in cm^3/s) | relevant only if DM_PROCESS = 'swave' 
| "DM_LIFETIME":     # (float) Lifetime | relevant only if DM_PROCESS = 'decay'
|
| # Specific to DarkHistory
| "DM_PRIMARY":       # (string) Primary particles (see list in user_params description)
| "DM_BOOST":         # (string) Annihilation boost | relevant only if DM_PROCESS = 'swave' 
| "DM_FS_METHOD":     # (string) Method to compute the energy deposition (see DarkHistory doc)
| "DM_BACKREACTION":  # (bool) Turns on backreaction
|
| # Specific to an approximative energy deposition
| "DM_FHEAT_APPROX_SHAPE":   # (string) Shape of the template for f_heat
| "DM_FHEAT_APPROX_PARAMS":  # (list of float)  Parameters to feed to the template of fheat 
| "DM_FION_H_OVER_FHEAT":    # (float) Ratio of f_ion_H over fheat  (if < 0 use values tabulated with DarkHistory)
| "DM_FION_HE_OVER_FHEAT":   # (float) Ratio of f_ion_He over fheat (if < 0 use values tabulated with DarkHistory)
| "DM_FEXC_OVER_FHEAT":      # (float) Ratio of fexc over fheat     (if < 0 use values tabulated with DarkHistory)
| 
| Flag options: 
| "USE_DM_ENERGY_INJECTION":  # (bool) Turn on DM energy injection
| "USE_EFFECTIVE_DEP_FUNCS":  # (bool) Treat the energy injection with approximate templates (instead of DarkHistory)
| "FORCE_INIT_COND":          # (bool) Can force the initial to be from a template or given by XION_AT_ZHEAT_MAX and TK_AT_ZHEAT_MAX in global_params (not used now)

The run_lighcone function also gets a new arguments called coarsen_factor (integer), which redifines the redshift steps to match with the table of DarkHistory. Note that if we use energy deposition through the templates this value can be arbitrary. Be default it is set to 16 to match with the nominal redshift step definition of 21cmFAST.

Alongisde these additions, we provide a built-in database manager for runs with DM energy injection (as we can play with many parameters and one may need to keep track of all the runs that are done). This database manager system is currently not optimised (probably better not to use it for more than a few thousand entries). An example of how it can be used is provided in examples/example_run_lightcone.py. From this script you can run a given model on command line. Set the strings <database_path> an <cache_path> (at the top of the file) to the location where you want to save the file. To see what it can do you can run::

    $ python example_run_lightcone.py --help

For instance, these two lines::

    $ python example_run_lightcone.py -decay -m 1e+8 -lftm 1e+26 -p elec_delta -nobkr
    $ python example_run_lightcone.py -approx -decay -shape schechter -params 0.1 2.0 0.4 -lftm 1e+26 -m 1e+8 -xe_init 1e-3 -Tm_init 120 -force_init 

perform the following two computations. The first one uses DarkHistory to fully compute the impact of decaying dark matter with mass 100 MeV and a lifetime of 1e+26 without including backreaction. The second one uses the shechter template with parameters (0.1, 2.0, 0.4) for decaying dark matter of the same mass and lifetime. In the second example we further fix the initial conditions to x_e = 1e-3 and Tm = 120 K at Z_HEAT_MAX. If -approx is not given in input it is also possible to launch several runs with a single command (by specifying two masses for instance, or two priamries). Examples of plots (and how to make them) are also avalailable in the example folder (example_plot_0.py and example_plot_1.py).
