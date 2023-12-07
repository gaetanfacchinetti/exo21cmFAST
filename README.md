# exo21cmFAST

**A semi-numerical cosmological simulation code for the radio 21-cm signal including non cold dark matter scenarios.**

In this code we introduce several features to [**21cmFAST**](https://github.com/21cmfast/21cmFAST)[^1][^2] related to non cold dark matter effects on the matter power spectrum and on structure formation.

In the future, this code should be merged with the main branch of **21cmFAST**.

## Installing the ncdm branch of exo21cmFAST

To install this version of **exo21cmFAST**, download that branch with
```
$ git clone https://github.com/gaetanfacchinetti/exo21cmFAST.git --branch ncdm --single-branch
```
On your terminal then go to the main folder (containing the file `config.py`) and run

```bash
$ pip install [-e] .
```


## The physics behind exo21cmFAST ncdm

The physics behind exotic energy injection and notations and introduced in this section.

### Matter power spectrum and transfer function

### Non linear effects on structure collapse

### Effective parametrisation
 

## What is new/different in exo21cmFAST?

The new features are implemented in the `run_lightcone` function which accepts several new arguments. In addition, we provide new input parameters in the `astro_params` and `flag_options` arguments related to the exotic energy injection and its treatment. All new inputs are listed below (for more information go to [`src/inputs.py`](https://github.com/gaetanfacchinetti/exo21cmFAST/blob/master/src/py21cmfast/inputs.py) and [`src/wrapper.py`](https://github.com/gaetanfacchinetti/exo21cmFAST/blob/master/src/py21cmfast/wrapper.py)).

### 1. Astro parameters
*These parameters are the *astrophysical* quantities which can be included in an inference analysis (a MCMC with [**21cmMC**](https://github.com/21cmfast/21CMMC) or a Fisher forecast with [**21cmCAST**](https://github.com/gaetanfacchinetti/21cmCAST))*

The numerical parameters related to the dark matter properties (the only necessary ones when using **DarkHistory**)
 - `DM_LOG10_MASS`: $\log_{10}(m_\chi / {\rm eV})$  mass of the dark matter particle
 - `DM_LOG10_SIGMAV`:  $\log_{10}(\left<\sigma v\right> /{\rm cm^3 / s^{-1}})$ annihilation cross section
 - `DM_LOG10_LIFETIME`: $\log_{10}(\tau / {\rm s})$ lifetime of decaying dark matter
 - (`DM_DECAY_RATE`: $\Gamma/{\rm s^{-1}} = {\rm s} / \tau$ decay rate of decaying dark matter). Need `DM_USE_DECAY_RATE` set to `True` in the Flag options to be used (otherwise the value of `DM_LOG10_LIFETIME` is used)

The numerical parameters related to the effective parametrisation

 - `DM_FHEAT_APPROX_PARAM_LOG10_F0`: $\log_{10}(f_0)$
 - `DM_FHEAT_APPROX_PARAM_A`: $a$
 - `DM_FHEAT_APPROX_PARAM_B`: $b$
 - `DM_LOG10_FION_H_OVER_FHEAT`: $\log_{10}(c_1) = \log_{10}(f_{\rm ion., HII} / f_{\rm heat})$
 - `DM_LOG10_FION_HE_OVER_FHEAT`: $\log_{10}(c_2) = \log_{10}(f_{\rm ion., HeII} / f_{\rm heat})$
 - `DM_LOG10_FEXC_OVER_FHEAT`:  $\log_{10}(c_3) = \log_{10}(f_{\rm exc.} / f_{\rm heat})$
 - `LOG10_TK_at_Z_HEAT_MAX$`: $T_{\rm k}(z_{\rm init})$ initial condition for the temperature
 - `LOG10_XION_at_Z_HEAT_MAX$`: $x_{\rm e}(z_{\rm init})$ initial condition for the electron fraction

### 2. Flag options

General flags
- `USE_DM_ENERGY_INJECTION`: *boolean* if `True` turns on exotic energy injection

Flags related to **DarkHistory**:
- `DM_PROCESS`: *string* $\in$ ['swave', 'decay', 'none'] specify the type of enerjy injection
- `DM_PRIMARY`: *string* $\in$ ['elec_delta', 'e', 'phot_delta', 'gamma', 'mu', ...] (see **DarkHistory** documentation for all the possibillities) primary particles in which dark matter decays or annihilates
- `DM_BOOST`: *string* $\in$ ['erfc', 'einasto_subs', 'einasto_no_subs', 'NFW_subs', 'NFW_no_subs'] preimplemented boost function $\mathcal{B}(z)$ for dark matter annihilation
- `DM_FS_METHOD`: *string* $\in$ ['He', 'no_He', 'He_recomb'] method to compute the deposition fractions (see **DarkHistory** documentation for more details)
- `DM_BACKREACTION`: *boolean* if `True` turns on exotic energy injection
- `DM_USE_DECAY_RATE`: *boolean* if `True` uses `DM_DECAY_RATE` instead of `DM_LOG10_LIFETIME` to evaluate the energy injection from dark matter decay
  
Flags related to the effective parametrisation:
- `USE_DM_EFFECTIVE_DEP_FUNCS`: *boolean* if `True` bypasses **DarkHistory** and uses the effective parametrisation for the deposition functions
- `DM_FHEAT_APPROX_SHAPE`: *integer* or *string* $\in$ [0: 'none', 1: 'constant', 2: 'exponential', 3: 'schechter'] functional form of the deposition fraction into heat $F(z)$
- `USE_DM_CUSTOM_F_RATIOS`: *boolean* if `True` uses `DM_LOG10_FION_H_OVER_FHEAT`, `DM_LOG10_FION_HE_OVER_FHEAT`, `DM_LOG10_FION_HE_OVER_FHEAT` to relate the other deposition fractions to $f_{\rm heat}$. If `False` uses predefined values obtained from scans of **DarkHistory** results
- `USE_CUSTOM_INIT_COND`: *boolean* forces initial conditions to `LOG10_TK_at_Z_HEAT_MAX` and `LOG10_XION_at_Z_HEAT_MAX`. If `USE_CUSTOM_INIT_COND` is `False` and `FORCE_DEFAULT_INIT_COND` is `False` the initial conditions are set either by vanilla RECFAST if `USE_DM_EFFECTIVE_DEP_FUNCS` is `False` or by the result of the **DarkHistory** run otherwise. Cannot be set to `True` if `FORCE_DEFAULT_INIT_COND` is `True` as well
- `FORCE_DEFAULT_INIT_COND`: *boolean*  forces the initial conditions to be that of vanilla RECFAST (even if exotic energy injection has happened before redshift $z_{\rm init}=$ `Z_HEAT_MAX`). If `USE_CUSTOM_INIT_COND` is `False` and `FORCE_DEFAULT_INIT_COND` is `False` the initial conditions are set either by vanilla RECFAST if `USE_DM_EFFECTIVE_DEP_FUNCS` is `False` or by the result of the **DarkHistory** run otherwise. Cannot be set to True if `USE_CUSTOM_INIT_COND` is `True` as well.


### 3. run_lightcone arguments

- `coarsen_factor`: *integer* redifines the redshift steps to match with the table of **DarkHistory**. Note that if we use energy deposition through the templates this value can be arbitrary. Be default it is set to 16 to match with the nominal redshift step definition of 21cmFAST.
- `verbose_ntbk`: *boolean* if `True` outputs more information during the run, which can be useful when running 21cmFAST on small boxed in a notebook.
- `output_exotic_data`: *boolean* if `True` gives a second output to the `run_lightcone()` function in the form of a dictionnary. This dictionnary contains the deposition fractions `'f'`, electron fraction `'x'`, gaz temperature `'Tm'` at every redshifts in `'z'`.
- `heating_rate_output`: *string* defines a file where to save the the heating rate due to exotic energy injection and astrophycial energy injection. If nothing specified, the heating rates are not saved.

## Using exo21cmFAST

Some examples are provided in `exo21cmFAST/examples` and play the role of small tutorials. In particular see [`example_notebook.py`](https://github.com/gaetanfacchinetti/exo21cmFAST/blob/master/examples/example_notebook.ipynb). 

The most simple code that can be run for a test assuming dark matter decyaing into $e^+e^-$ with a mass of 100 MeV, a lifetime $\tau = 10^{26}$ s, and using **DarkHistory** is

```python
import py21cmfast as p21f

lightcone = p21f.run_lightcone(
        redshift = 5,
        user_params = {"BOX_LEN": 250, "HII_DIM": 128},
        astro_params = {"DM_LOG10_MASS": 8.0, "DM_LOG10_LIFETIME": 26.0},
        flag_options = {
            "USE_DM_ENERGY_INJECTION" : True,
            "USE_TS_FLUCT"            : True, 
            "DM_PROCESS"              : 'decay',
            "DM_PRIMARY"              : 'elec_delta'    
        },
        direc='./cache', 
    )

lightcone.save(fname = "my_ligthcone.h5")
```

A more complete example is also provided in [`example_run.py`](https://github.com/gaetanfacchinetti/exo21cmFAST/blob/master/examples/example_run.py).

Once the lightcone is created, it can be analysed with different tools. See [**21cmCAST** documentation](https://github.com/gaetanfacchinetti/21cmCAST) for an example.

## Credits

If you use **exo21cmFAST** or parts of the new functionnalities not already present in 21cmFAST please cite 

- Gaetan Facchinetti, Laura Lopez-Honorez, Andrei Mesinger, Yuxiang Qin, *21cm signal sensitivity to dark matter
decay* (in prep.)


[^1]: Andrei Mesinger, Steven Furlanetto, and Renyue Cen, *21cmFAST: A Fast, Semi-Numerical Simulation of the High-Redshift 21-cm Signal* [[arXiv:1003.3878](https://arxiv.org/abs/1003.3878)]

[^2]: Andrei Mesinger and Steven Furlanetto, *Efficient Simulations of Early Structure Formation and Reionization* [[arXiv:0704.0946](https://arxiv.org/abs/0704.0946)]

[^3]: Hongwan Liu, Gregory W. Ridgway, Tracy R. Slatyer, *DarkHistory: A code package for calculating modified cosmic ionization and thermal histories with dark matter and other exotic energy injections* [[arXiv:1904.09296](https://arxiv.org/abs/1904.09296)]

[^4]: Yitian Sun, Tracy R. Slatyer, *Modeling early-universe energy injection with Dense Neural Networks* [[arXiv:2207.06425](https://arxiv.org/abs/2207.06425)]

[^5]: Laura Lopez-Honorez, Olga Mena, Ángeles Moliné, Sergio Palomares-Ruiz, Aaron C. Vincent, *The 21 cm signal and the interplay between dark matter annihilations and astrophysical processes* [[arXiv:1603.06795](https://arxiv.org/abs/1603.06795)]
