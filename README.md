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

## The physics behind exo21cmFAST

The physics behind exotic energy injection and notations and introduced in this section. For more information please read the companion paper[^4]. 

### Energy injection and deposition

Exotic energy is added to the simulation assuming a smooth homogeneous energy injection rate per number of baryon
$$\epsilon_{\rm inj.} \equiv \frac{1}{\overline{n_{\rm b}}}\left(\frac{{\rm d} E}{{\rm d} t {\rm d V}}\right)_{\rm inj.}$$
and several deposition fractions quantifying how much of the injected energy is deposited in the intergalactic mediuc at a given time. Here $\overline{n_{\rm b}}$ is the average number density of baryons. The deposition fractions are denoted $f_{\rm c}$ with c the corresponding deposition channel (*heating, ionization of hydrogen atoms, ionization of Helium atoms, and excitation*). They depend on the redshift and on $\{x_i\}_i$ the ionization levels of $i=$ neutral hydrogen HI, neutral helium HeI, or exited Helium HeII. The evolution equations of the intergalactic medium kinetic temperature ($T_{\rm k}$) and of the electron fraction $(x_{\rm e})$ as well as the Lyman-$\alpha$ flux $(J_\alpha)$ are then modified by respectively adding

$$ \left. \frac{\partial T_{\rm k}}{\partial t} \right|_{\rm exotic} = \frac{2}{3 k_{\rm B}(1+x_{\rm e})} \frac{1}{\overline{n_{\rm b}}} f_{\rm heat}(x_{i}, z) \epsilon_{\rm inj.}$$
$$ \left. \frac{\partial x_{\rm e}}{\partial t} \right|_{\rm exotic} =  \left[ \frac{\frak{f}_{\rm H}}{E_{\rm HI}} f_{\rm ion., HII}(x_{i}, z) + \frac{\frak{f}_{\rm He}}{E_{\rm HI}} f_{\rm ion., HeII}(x_{i}, z) \right] \frac{1}{\overline{n_{\rm b}}}\epsilon_{\rm inj.} $$
$$\left. J_\alpha \right|_{\rm exotic} = \frac{c}{4\pi H(z) h\nu_\alpha^2} f_{\rm exc.}(x_{i}, z) \epsilon_{\rm inj.}\, ,$$

to the *classical* sources. Here $H(z)$ is the Hubble rate, $h$ is the Planck constant, and $\nu_\alpha$ is the Lyman-$\alpha$ frequency. Moreover $\frak{f}_{\rm H} = \overline{n_{\rm H}}/\overline{n_{\rm b}}$ and $\frak{f}_{\rm He} = \overline{n_{\rm He}}/\overline{n_{\rm b}}$ are the hydrogen and helium number fractions. $E_{\rm HI}$ and $E_{\rm HeI}$ are the ionization energies of neutral hydrogen and helium. 

### The specific cases of dark matter decay and annihilation


In the case of dark matter decay the injected energy can be written in terms of the average dark matter energy density today $\overline{\rho_{\chi, 0}}$, the number density of baryons today $\overline{n_{\rm b, 0}}$ and the lifetime $\tau$ (or decay rate $\Gamma = 1/\tau$),
$$\epsilon_{\rm inj.} = \frac{\overline{\rho_{\rm \chi, 0}}c^2}{\overline{n_{\rm b, 0}}}\frac{1}{\tau}$$

In case of DM annihilation the relevant parameters are the annihilation cross-section $\left<\sigma v\right>$ and the dark matter mass $m_\chi$ and the smoothed energy injection rate reads
$$\epsilon_{\rm inj.} = (1+z)^3 \frac{\overline{\rho_{\rm \chi, 0}}^2c^4}{\overline{n_{\rm b, 0}}} \frac{\left<\sigma v\right>}{m_\chi}$$

**WARNING 1**: This is the smooth energy injection rate. The true energy injection rate depends on the average of the squared density $\overline{\rho_\chi(x, z)^2}$  and not on the square of the average density $\overline{\rho_{\chi}}^2$ (as used here). With the presence of dark matter halos, the two are not equal at small redshifts. The difference is accounted for with a boost function $\overline{\rho_\chi^2}(z) \simeq \overline{\rho_\chi}^2(1+\mathcal{B}(z))$ and included into the definition of the deposition functions $f_c$.

**WARNING 2**: Dark matter annihilation and decay produce Standard Model particles that shower to electrons and photons which, in turn, electromagnetically interact with the intergalactic medium. The deposion fractions then depend on the Standard Model *primary* species that are produced.


### Effective parametrisation

Based on observations of the evolution of the deposition fractions with the redshifts, one can often find $F$ a function, and  $c_1$, $c_2$, and $c_3$, three constants such that, in good approximation, 

$$f_{\rm heat}(z) \simeq F(z) $$
$$f_{\rm ion., HII}(z) \simeq c_1 f_{\rm heat}(z)$$
$$f_{\rm ion., HeII}(z) \simeq c_2 f_{\rm heat}(z)$$
$$f_{\rm exc.}(z) \simeq c_3 f_{\rm heat}(z) \, ,$$

(at least for dark matter decay and annihilation). Pre-implemented functions in the code are

$$F(z) = f_0$$
$$F(z) = f_0 e^{a(z-15)}$$
$$F(z) = f_0 e^{a(z-15)} \left(\frac{z}{15}\right)^b  \, , $$
with $f_0$, $a$, and $b$ three constants. Using this parametrisation bypasses **DarkHistory** for the computation of the deposition functions. However, when active, **DarkHistory** also determines the initial condition for 21cmFAST at around redshift 35. Therefore using the effective parametrisation also implies that initial conditions must be set by hand. 

## What is new/different in exo21cmFAST?

The new features are implemented in the `run_lightcone` function which accepts several new arguments. In addition, we provide new input parameters in the `astro_params` and `flag_options` arguments related to the exotic energy injection and its treatment. All new inputs are listed below (for more information go to `src/inputs.py` and `src/wrapper.py`).

### 1. Astro parameters
*These parameters are the *astrophysical* quantities which can be included in an inference analysis (a MCMC with [21CMMC]() or a Fisher forecast with [21cmCAST]())*

The numerical parameters related to the dark matter properties (the only necessary ones when using **DarkHistory**)
 - `DM_LOG10_MASS`: $\log_{10}(m_\chi / {\rm eV})$  mass of the dark matter particle
 - `DM_LOG10_SIGMAV`:  $\log_{10}(\left<\sigma v\right> /{\rm cm^3 / s^{-1}})$ annihilation cross section
 - `DM_LOG10_LIFETIME`: $\log_{10}(\tau / {\rm s})$ lifetime of decaying dark matter
 - (`DM_DECAY_RATE`: $\Gamma/{\rm s^{-1}} = {\rm s} / \tau$ decay rate of decaying dark matter)

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
- `DM_FHEAT_APPROX_SHAPE`: *integer* or *string* $\in$ [0: 'none', 1: 'constant', 2: 'exponential', 3: 'schechter'] form for the deposition fraction into heat $F(z)$
- `USE_DM_CUSTOM_F_RATIOS`: *boolean* if `True` uses `DM_LOG10_FION_H_OVER_FHEAT`, `DM_LOG10_FION_HE_OVER_FHEAT`, `DM_LOG10_FION_HE_OVER_FHEAT` to relate the other deposition fractions to $f_{\rm heat}$. If `False` uses predefined values obtained from scans of **DarkHistory** results
- `USE_CUSTOM_INIT_COND`: ...


### 3. run_lightcone arguments

- `coarsen_factor`: (integer) redifines the redshift steps to match with the table of DarkHistory. Note that if we use energy deposition through the templates this value can be arbitrary. Be default it is set to 16 to match with the nominal redshift step definition of 21cmFAST.
- ... 

## Using exo21cmFAST




## Credits

If you use **exo21cmFAST** or parts of the new functionnalities not already present in 21cmFAST please cite 


[^1]: something here