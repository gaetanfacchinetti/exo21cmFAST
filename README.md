# exo21cmFAST

**A semi-numerical cosmological simulation code for the radio 21-cm signal including exotic sources of energy injection.**

In this code we introduce several features to [**21cmFAST**](https://github.com/21cmfast/21cmFAST)[^1][^2] related to dark matter energy injection via s-wave annihilation or decay. The energy deposition can be evaluated in two ways, either using the [**DarkHistory**](https://darkhistory.readthedocs.io/en/master/)[^3] package or with template functions. Future updates will include other sources of energy injection not necessarily related to dark matter.


## Installing exo21cmFAST

To install **exo21cmFAST**, download the repository, on your terminal run go to the main folder (containing the file `config.py`) and run

```bash
$ pip install -e .
```

To use **DarkHistory** (which is not necessary depending on what you want to do) you need to download the package (see information on the DarkHistory documentation) and add these two lines to your `PYTHONPATH`:

```bash
$ export PYTHONPATH=$PYTHONPATH:<folder containing DarkHistory>
$ export PYTHONPATH=$PYTHONPATH:<folder containing DarkHistory>/DarkHistory
```


## What is new/different in exo21cmFAST?

The new features are implemented in the `run_lightcone` function which accepts several new arguments. In addition, we provide new input parameters in the `astro_params` and `flag_options` arguments related to the exotic energy injection and its treatment. All new inputs are listed below (for more information go to `src/input.py` and `src/wrapper.py`).

### 1. Astro parameters
*These parameters are the *astrophysical* quantities which can be included in an inference analysis (a MCMC with [21CMMC]() or a Fisher forecast with [21cmCAST]())*
 - `DM_MASS`: (float) dark matter mass in eV
 - `DM_PROCESS`: (string) ...

### 2. Flag options

- `DM_BACKREACTION`: (bool) if `True` turns on exotic energy injection
- ... 


### 3. run_lightcone arguments

- `coarsen_factor`: (integer) redifines the redshift steps to match with the table of DarkHistory. Note that if we use energy deposition through the templates this value can be arbitrary. Be default it is set to 16 to match with the nominal redshift step definition of 21cmFAST.
- ... 

## Using exo21cmFAST




## Credits

If you use **exo21cmFAST** or parts of the new functionnalities not already present in 21cmFAST please cite 


[^1]: something here