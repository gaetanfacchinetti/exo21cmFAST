"""The py21cmfast package."""
try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version

try:
    from ._version import version as __version__
except ModuleNotFoundError:  # pragma: no cover
    try:
        __version__ = version("21cmFAST")
    except PackageNotFoundError:
        # package is not installed
        __version__ = "unknown"

# This just ensures that the default directory for boxes is created.
from os import mkdir as _mkdir
from os import path

from . import cache_tools, inputs, outputs, plotting, wrapper
from ._cfg import config
from ._logging import configure_logging
from .cache_tools import query_cache
from .outputs import (
    Coeval,
    InitialConditions,
    IonizedBox,
    LightCone,
    PerturbedField,
    TsBox,
)
from .wrapper import (
    AstroParams,
    BrightnessTemp,
    CosmoParams,
    FlagOptions,
    HaloField,
    PerturbHaloField,
    UserParams,
    brightness_temperature,
    compute_luminosity_function,
    compute_tau,
    construct_fftw_wisdoms,
    determine_halo_list,
    get_all_fieldnames,
    global_params,
    initial_conditions,
    ionize_box,
    perturb_field,
    perturb_halo_list,
    run_coeval,
    run_lightcone,
    spin_temperature,
    matter_power_spectrum,
    transfer_function_nCDM,
    sigma_z0,
    dsigmasqdm_z0,
    dndm,
    mass_to_radius,
    radius_to_mass,
    f_gtr_mass,
    nion_conditional_m,
    growth_from_pmf,
    pmf_induced_matter_power_spectrum,
    init_TF_CLASS,
)

configure_logging()

try:
    _mkdir(path.expanduser(config["direc"]))
except FileExistsError:
    pass
