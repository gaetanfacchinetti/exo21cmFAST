######################################################################
## When this module is called we (re)define the current path correctly
## This way we do not have to worry too much about where are the files

import numpy as np

from config import load_data

import darkhistory.physics as phys

from darkhistory.spec import pppc
from darkhistory.spec.spectra import Spectra
from darkhistory.electrons import positronium as pos
from darkhistory.electrons.elec_cooling import get_elec_cooling_tf
from darkhistory.low_energy.lowE_deposition import compute_fs
from darkhistory.low_energy.lowE_electrons import make_interpolator
from darkhistory.history import tla

import DarkHistory.main as main


def evolve_for_21cmFAST(
    in_spec_elec=None, in_spec_phot=None,
    rate_func_N=None, rate_func_eng=None,
    DM_process=None, mDM=None, sigmav=None, lifetime=None, primary=None, struct_boost=None,
    start_rs=None, end_rs=4, init_cond=None, coarsen_factor=1,
    compute_fs_method='no_He', mxstep=1000, rtol=1e-4, cross_check=False,
    in_highengphot_specs=None, in_lowengphot_specs=None, in_lowengelec_specs=None, in_highengdep=None, 
    xHII_vs_rs=None, xHeII_vs_rs=None, Tm_vs_rs=None):

    """
    Main function computing histories and spectra.
    This function is a little bit different than main.evolve() because here we can save the output and start again the function from the output.
    Moreover we can also fix xe and Tm on the run

    Parameters
    -----------
    in_spec_elec : :class:`.Spectrum`, optional
        Spectrum per injection event into electrons. *in_spec_elec.rs*
        of the :class:`.Spectrum` must be the initial redshift.
    in_spec_phot : :class:`.Spectrum`, optional
        Spectrum per injection event into photons. *in_spec_phot.rs*
        of the :class:`.Spectrum` must be the initial redshift.
    rate_func_N : function, optional
        Function returning number of injection events per volume per time, with redshift :math:`(1+z)` as an input.
    rate_func_eng : function, optional
        Function returning energy injected per volume per time, with redshift :math:`(1+z)` as an input.
    DM_process : {'swave', 'decay'}, optional
        Dark matter process to use.
    sigmav : float, optional
        Thermally averaged cross section for dark matter annihilation.
    lifetime : float, optional
        Decay lifetime for dark matter decay.
    primary : string, optional
        Primary channel of annihilation/decay. See :func:`.get_pppc_spec` for complete list. Use *'elec_delta'* or *'phot_delta'* for delta function injections of a pair of photons/an electron-positron pair.
    struct_boost : function, optional
        Energy injection boost factor due to structure formation.
    start_rs : float, optional
        Starting redshift :math:`(1+z)` to evolve from. Default is :math:`(1+z)` = 3000. Specify only for use with *DM_process*. Otherwise, initialize *in_spec_elec.rs* and/or *in_spec_phot.rs* directly.
    end_rs : float, optional
        Final redshift :math:`(1+z)` to evolve to. Default is 1+z = 4.
    reion_switch : bool
        Reionization model included if *True*, default is *False*.
    helium_TLA : bool
        If *True*, the TLA is solved with helium. Default is *False*.
    reion_rs : float, optional
        Redshift :math:`(1+z)` at which reionization effects turn on.
    photoion_rate_func : tuple of functions, optional
        Functions take redshift :math:`1+z` as input, return the photoionization rate in s\ :sup:`-1` of HI, HeI and HeII respectively. If not specified, defaults to :func:`.photoion_rate`.
    photoheat_rate_func : tuple of functions, optional
        Functions take redshift :math:`1+z` as input, return the photoheating rate in s\ :sup:`-1` of HI, HeI and HeII respectively. If not specified, defaults to :func:`.photoheat_rate`.
    xe_reion_func : function, optional
        Specifies a fixed ionization history after reion_rs.
    init_cond : tuple of floats
        Specifies the initial (xH, xHe, Tm). Defaults to :func:`.Tm_std`, :func:`.xHII_std` and :func:`.xHeII_std` at the *start_rs*.
    coarsen_factor : int
        Coarsening to apply to the transfer function matrix. Default is 1.
    backreaction : bool
        If *False*, uses the baseline TLA solution to calculate :math:`f_c(z)`. Default is True.
    compute_fs_method : {'no_He', 'He_recomb', 'He'}

    mxstep : int, optional
        The maximum number of steps allowed for each integration point. See *scipy.integrate.odeint()* for more information. Default is *1000*.
    rtol : float, optional
        The relative error of the solution. See *scipy.integrate.odeint()* for more information. Default is *1e-4*.
    use_tqdm : bool, optional
        Uses tqdm if *True*. Default is *True*.
    cross_check : bool, optional
        If *True*, compare against 1604.02457 by using original MEDEA files, turning off partial binning, etc. Default is *False*.

    Examples
    --------

    1. *Dark matter annihilation* -- dark matter mass of 50 GeV, annihilation cross section :math:`2 \\times 10^{-26}` cm\ :sup:`3` s\ :sup:`-1`, annihilating to :math:`b \\bar{b}`, solved without backreaction, a coarsening factor of 32 and the default structure formation boost: ::

        import darkhistory.physics as phys

        out = evolve(
            DM_process='swave', mDM=50e9, sigmav=2e-26,
            primary='b', start_rs=3000.,
            backreaction=False,
            struct_boost=phys.struct_boost_func()
        )

    2. *Dark matter decay* -- dark matter mass of 100 GeV, decay lifetime :math:`3 \\times 10^{25}` s, decaying to a pair of :math:`e^+e^-`, solved with backreaction, a coarsening factor of 16: ::

        out = evolve(
            DM_process='decay', mDM=1e8, lifetime=3e25,
            primary='elec_delta', start_rs=3000.,
            backreaction=True
        )

    See Also
    ---------
    :func:`.get_pppc_spec`

    :func:`.struct_boost_func`

    :func:`.photoion_rate`, :func:`.photoheat_rate`

    :func:`.Tm_std`, :func:`.xHII_std` and :func:`.xHeII_std`


    """
    
    #########################################################################
    #########################################################################
    # Input                                                                 #
    #########################################################################
    #########################################################################


    #####################################
    # Initialization for DM_process     #
    #####################################

    # Load data.
    binning = load_data('binning')
    photeng = binning['phot']
    eleceng = binning['elec']

    dep_tf_data = load_data('dep_tf')

    highengphot_tf_interp = dep_tf_data['highengphot']
    lowengphot_tf_interp  = dep_tf_data['lowengphot']
    lowengelec_tf_interp  = dep_tf_data['lowengelec']
    highengdep_interp     = dep_tf_data['highengdep'] 

    ics_tf_data = load_data('ics_tf')

    ics_thomson_ref_tf  = ics_tf_data['thomson']
    ics_rel_ref_tf      = ics_tf_data['rel']
    engloss_ref_tf      = ics_tf_data['engloss']


    # Handle the case where a DM process is specified.
    if DM_process == 'swave':
        if sigmav is None or start_rs is None:
            raise ValueError('sigmav and start_rs must be specified.')
        
        # Get input spectra from PPPC.
        in_spec_elec = pppc.get_pppc_spec(mDM, eleceng, primary, 'elec')
        in_spec_phot = pppc.get_pppc_spec(mDM, photeng, primary, 'phot')
        # Initialize the input spectrum redshift.
        in_spec_elec.rs = start_rs
        in_spec_phot.rs = start_rs
        # Convert to type 'N'.
        in_spec_elec.switch_spec_type('N')
        in_spec_phot.switch_spec_type('N')

        # If struct_boost is none, just set to 1.
        if struct_boost is None:
            def struct_boost(rs):
                return 1.

        # Define the rate functions.
        def rate_func_N(rs):
            return (phys.inj_rate('swave', rs, mDM=mDM, sigmav=sigmav) * struct_boost(rs) / (2*mDM))
            
        def rate_func_eng(rs):
            return (phys.inj_rate('swave', rs, mDM=mDM, sigmav=sigmav) * struct_boost(rs))

    if DM_process == 'decay':
        if lifetime is None or start_rs is None:
            raise ValueError('lifetime and start_rs must be specified.')

        # The decay rate is insensitive to structure formation
        def struct_boost(rs):
            return 1
        
        # Get spectra from PPPC.
        in_spec_elec = pppc.get_pppc_spec(mDM, eleceng, primary, 'elec', decay=True)
        in_spec_phot = pppc.get_pppc_spec(mDM, photeng, primary, 'phot', decay=True)

        # Initialize the input spectrum redshift.
        in_spec_elec.rs = start_rs
        in_spec_phot.rs = start_rs
        # Convert to type 'N'.
        in_spec_elec.switch_spec_type('N')
        in_spec_phot.switch_spec_type('N')

        # Define the rate functions.
        def rate_func_N(rs):
            return (phys.inj_rate('decay', rs, mDM=mDM, lifetime=lifetime) / mDM)
       
        def rate_func_eng(rs):
            return phys.inj_rate('decay', rs, mDM=mDM, lifetime=lifetime)
    


    #####################################
    # Initialization                    #
    #####################################

    # Initialize start_rs for arbitrary injection.
    start_rs = in_spec_elec.rs

    # Initialize the initial x and Tm.
    xH_init  = init_cond[0] if (xHII_vs_rs is None)  else xHII_vs_rs(start_rs)
    xHe_init = init_cond[1] if (xHeII_vs_rs is None) else xHeII_vs_rs(start_rs)
    Tm_init  = init_cond[2] if (Tm_vs_rs is None)    else Tm_vs_rs(start_rs)


    # Initialize redshift/timestep related quantities.

    # Default step in the transfer function. Note highengphot_tf_interp.dlnz
    # contains 3 different regimes, and we start with the first.
    dlnz = highengphot_tf_interp.dlnz[-1]

    # The current redshift.
    rs   = start_rs

    # The timestep between evaluations of transfer functions, including
    # coarsening.
    dt   = dlnz * coarsen_factor / phys.hubble(rs)



    #####################################
    # Input Checks                      #
    #####################################

    if (
        not np.array_equal(in_spec_elec.eng, eleceng)
        or not np.array_equal(in_spec_phot.eng, photeng)
    ):
        raise ValueError('in_spec_elec and in_spec_phot must use config.photeng and config.eleceng respectively as abscissa.')

    if (
        highengphot_tf_interp.dlnz    != lowengphot_tf_interp.dlnz
        or highengphot_tf_interp.dlnz != lowengelec_tf_interp.dlnz
        or lowengphot_tf_interp.dlnz  != lowengelec_tf_interp.dlnz
    ):
        raise ValueError('TransferFuncInterp objects must all have the same dlnz.')

    if in_spec_elec.rs != in_spec_phot.rs:
        raise ValueError('Input spectra must have the same rs.')

    if cross_check:
        print('cross_check has been set to True -- No longer using all MEDEA files and no longer using partial-binning.')

    
    if ((in_highengphot_specs is not None and in_highengphot_specs[-1].rs != start_rs)
        or (in_lowengphot_specs is not None and in_lowengphot_specs[-1].rs != start_rs)
        or (in_lowengelec_specs is not None and  in_lowengelec_specs[-1].rs  != start_rs)
        ):
        print('WARNING: rs of input specs is:', in_highengphot_specs[-1].rs, 'while start_rs is:', start_rs)
        
        
        
    #####################################
    # Initialization (following)        #
    #####################################

    def norm_fac(rs):
        # Normalization to convert from per injection event to
        # per baryon per dlnz step.
        return rate_func_N(rs) * (dlnz * coarsen_factor / phys.hubble(rs) / (phys.nB * rs**3))

    def rate_func_eng_unclustered(rs):
        # The rate excluding structure formation for s-wave annihilation.
        # This is the correct normalization for f_c(z).
        if struct_boost is not None:
            return rate_func_eng(rs)/struct_boost(rs)
        else:
            return rate_func_eng(rs)


    # If there are no electrons, we get a speed up by ignoring them.
    elec_processes = False
    if in_spec_elec.totN() > 0:
        elec_processes = True

    if elec_processes:

        #####################################
        # High-Energy Electrons             #
        #####################################

        # Get the data necessary to compute the electron cooling results.
        # coll_ion_sec_elec_specs is \bar{N} for collisional ionization,
        # and coll_exc_sec_elec_specs \bar{N} for collisional excitation.
        # Heating and others are evaluated in get_elec_cooling_tf
        # itself.

        # Contains information that makes converting an energy loss spectrum
        # to a scattered electron spectrum fast.
        (coll_ion_sec_elec_specs, coll_exc_sec_elec_specs, ics_engloss_data) = main.get_elec_cooling_data(eleceng, photeng)

    #########################################################################
    #########################################################################
    # Pre-Loop Preliminaries                                                #
    #########################################################################
    #########################################################################

    # Initialize the arrays that will contain x and Tm results.
    x_arr  = np.array([[xH_init, xHe_init]])
    Tm_arr = np.array([Tm_init])


    # Initialize Spectra objects to contain all of the output spectra.
    # If we have no input we initialise them to an empy spectra object
    # Otherwise we use the input spectra
    out_highengphot_specs = Spectra([], spec_type='N') if (in_highengphot_specs is None) else in_highengphot_specs
    out_lowengphot_specs  = Spectra([], spec_type='N') if (in_lowengphot_specs is None)  else in_lowengphot_specs
    out_lowengelec_specs  = Spectra([], spec_type='N') if (in_lowengelec_specs is None)  else in_lowengelec_specs

    # Define these methods for speed.
    append_highengphot_spec = out_highengphot_specs.append
    append_lowengphot_spec  = out_lowengphot_specs.append
    append_lowengelec_spec  = out_lowengelec_specs.append

    # Initialize arrays to store f values.
    f_low  = np.empty((0,5))
    f_high = np.empty((0,5))

    # Initialize array to store high-energy energy deposition rate.
    highengdep_grid = np.empty((0,4))

    # Object to help us interpolate over MEDEA results.
    MEDEA_interp = make_interpolator(interp_type='2D', cross_check=cross_check)

    # Initialise the array of rs values that are treated here
    rs_arr = []

    #########################################################################
    #########################################################################
    # LOOP! LOOP! LOOP! LOOP!                                               #
    #########################################################################
    #########################################################################

    while rs > end_rs:

        #############################
        # First Step Special Cases  #
        #############################
        if rs == start_rs:
            # Initialize the electron and photon arrays.
            # These will carry the spectra produced by applying the
            # transfer function at rs to high-energy photons.
            if in_highengphot_specs is None:
                highengphot_spec_at_rs = in_spec_phot*0
            elif np.array_equal(in_highengphot_specs[-1].eng, in_spec_phot.eng):
                highengphot_spec_at_rs = in_highengphot_specs[-1]
            else:
                raise ValueError('Problem of compatibility with in_highengphot_specs')
                
            if in_lowengphot_specs is None:
                lowengphot_spec_at_rs = in_spec_phot*0
            elif np.array_equal(in_lowengphot_specs[-1].eng, in_spec_phot.eng):
                lowengphot_spec_at_rs = in_lowengphot_specs[-1]
            else:
                raise ValueError('Problem of compatibility with in_lowengphot_specs')
                
            if in_lowengelec_specs is None:
                lowengelec_spec_at_rs = in_spec_elec*0
            elif np.array_equal(in_lowengelec_specs[-1].eng, in_spec_elec.eng):
                lowengelec_spec_at_rs = in_lowengelec_specs[-1]
            else:
                raise ValueError('Problem of compatibility with in_lowengpelec_specs')
                

            if in_highengdep is None:
                highengdep_at_rs = np.zeros(4)
            elif len(in_highengdep) == 4:
                highengdep_at_rs = in_highengdep
            else:
                raise ValueError('Problem with in_highengdep, it should be an array of length 4')
                
                

        #####################################################################
        #####################################################################
        # Electron Cooling                                                  #
        #####################################################################
        #####################################################################

        # Get the transfer functions corresponding to electron cooling.
        # These are \bar{T}_\gamma, \bar{T}_e and \bar{R}_c.
        if elec_processes:

            # Give the value of the ionization fraction wether we give an input or not
            xHII_elec_cooling  = x_arr[-1, 0] if (xHII_vs_rs  is None) else xHII_vs_rs(rs)
            xHeII_elec_cooling = x_arr[-1, 1] if (xHeII_vs_rs is None) else xHeII_vs_rs(rs)


            (ics_sec_phot_tf, elec_processes_lowengelec_tf,
                deposited_ion_arr, deposited_exc_arr, deposited_heat_arr,
                continuum_loss, deposited_ICS_arr) = get_elec_cooling_tf(eleceng, photeng, rs,
                    xHII_elec_cooling, xHeII=xHeII_elec_cooling,
                    raw_thomson_tf=ics_thomson_ref_tf,
                    raw_rel_tf=ics_rel_ref_tf,
                    raw_engloss_tf=engloss_ref_tf,
                    coll_ion_sec_elec_specs=coll_ion_sec_elec_specs,
                    coll_exc_sec_elec_specs=coll_exc_sec_elec_specs,
                    ics_engloss_data=ics_engloss_data)

            # Apply the transfer function to the input electron spectrum.

            # Low energy electrons from electron cooling, per injection event.
            elec_processes_lowengelec_spec = (elec_processes_lowengelec_tf.sum_specs(in_spec_elec))

            #print(elec_processes_lowengelec_spec.eng)
            #print(lowengelec_spec_at_rs.eng)

            # Add this to lowengelec_at_rs.
            lowengelec_spec_at_rs += (elec_processes_lowengelec_spec*norm_fac(rs))

            # High-energy deposition into ionization,
            # *per baryon in this step*.
            # Gaetan: np.dot is the dot product of the two vectors
            deposited_ion = np.dot(deposited_ion_arr,  in_spec_elec.N*norm_fac(rs))
            
            # High-energy deposition into excitation,
            # *per baryon in this step*.
            deposited_exc = np.dot(deposited_exc_arr,  in_spec_elec.N*norm_fac(rs))
            
            # High-energy deposition into heating,
            # *per baryon in this step*.
            deposited_heat = np.dot(deposited_heat_arr, in_spec_elec.N*norm_fac(rs))
            
            # High-energy deposition numerical error,
            # *per baryon in this step*.
            deposited_ICS = np.dot(deposited_ICS_arr,  in_spec_elec.N*norm_fac(rs))


            #######################################
            # Photons from Injected Electrons     #
            #######################################

            # ICS secondary photon spectrum after electron cooling,
            # per injection event.
            ics_phot_spec = ics_sec_phot_tf.sum_specs(in_spec_elec)

            # Get the spectrum from positron annihilation, per injection event.
            # Only half of in_spec_elec is positrons!
            positronium_phot_spec = pos.weighted_photon_spec(photeng) * (in_spec_elec.totN()/2)
            positronium_phot_spec.switch_spec_type('N')

        # Add injected photons + photons from injected electrons
        # to the photon spectrum that got propagated forward.
        if elec_processes:
            highengphot_spec_at_rs += (in_spec_phot + ics_phot_spec + positronium_phot_spec) * norm_fac(rs)
        else:
            highengphot_spec_at_rs += in_spec_phot * norm_fac(rs)

        # Set the redshift correctly.
        highengphot_spec_at_rs.rs = rs

        #####################################################################
        #####################################################################
        # Save the Spectra!                                                 #
        #####################################################################
        #####################################################################

        # At this point, highengphot_at_rs, lowengphot_at_rs and
        # lowengelec_at_rs have been computed for this redshift.
        
        #append_highengphot_spec(highengphot_spec_at_rs)
        #append_lowengphot_spec(lowengphot_spec_at_rs)
        #append_lowengelec_spec(lowengelec_spec_at_rs)

        #####################################################################
        #####################################################################
        # Compute f_c(z)                                                    #
        #####################################################################
        #####################################################################
        if elec_processes:
            # High-energy deposition from input electrons.
            highengdep_at_rs += np.array([
                deposited_ion/dt,
                deposited_exc/dt,
                deposited_heat/dt,
                deposited_ICS/dt
            ])
            

        # Values of (xHI, xHeI, xHeII) to use for computing f.
        # Use baseline values if no backreaction.
        xHI_loc   = (1-x_arr[-1, 0])        if (xHII_vs_rs is None)  else (1-xHII_vs_rs(rs))
        xHeI_loc  = (phys.chi-x_arr[-1, 1]) if (xHeII_vs_rs is None) else (phys.chi -xHeII_vs_rs(rs))
        xHeII_loc = x_arr[-1, 1]            if (xHeII_vs_rs is None) else xHeII_vs_rs(rs)
        x_vec_for_f = np.array( [ xHI_loc, xHeI_loc, xHeII_loc] )

        #print(rs, dt, lowengelec_spec_at_rs.N[200:204], lowengphot_spec_at_rs.N[200:204])
        #print(x_vec_for_f)

        #if rs < 33 :
        #    print_spectrum_N(highengphot_spec_at_rs, 'high_phot_rs_' + "{:6.4f}".format(rs))
        #    print_spectrum_N(lowengelec_spec_at_rs, 'elec_rs_' + "{:6.4f}".format(rs))
        #    print_spectrum_N(lowengphot_spec_at_rs, 'phot_rs_' + "{:6.4f}".format(rs))
        #    print(rs, dt, rate_func_eng_unclustered(rs), x_vec_for_f, highengdep_at_rs, compute_fs_method, cross_check)


        f_raw = compute_fs(
            MEDEA_interp, lowengelec_spec_at_rs, lowengphot_spec_at_rs,
            x_vec_for_f, rate_func_eng_unclustered(rs), dt,
            highengdep_at_rs, method=compute_fs_method, cross_check=cross_check)


        # Save the f_c(z) values.
        f_low  = np.concatenate((f_low,  [f_raw[0]]))
        f_high = np.concatenate((f_high, [f_raw[1]]))


        # Save CMB upscattered rate and high-energy deposition rate.
        highengdep_grid = np.concatenate( (highengdep_grid, [highengdep_at_rs]) )


        #####################################################################
        #####################################################################
        # ********* AFTER THIS, COMPUTE QUANTITIES FOR NEXT STEP *********  #
        #####################################################################
        #####################################################################

        # Define the next redshift step.
        next_rs = np.exp(np.log(rs) - dlnz * coarsen_factor)

        #####################################################################
        #####################################################################
        # Photon Cooling Transfer Functions                                 #
        #####################################################################
        #####################################################################

        # Get the values of ionization fractions
        xHII_to_interp   = x_arr[-1, 0] if (xHII_vs_rs  is None) else xHII_vs_rs(rs)
        xHeII_to_interp = x_arr[-1, 1]  if (xHeII_vs_rs is None) else xHeII_vs_rs(rs)
   
        # Get the correct transfer function
        highengphot_tf, lowengphot_tf, lowengelec_tf, highengdep_arr = main.get_tf(rs, xHII_to_interp, xHeII_to_interp,dlnz, coarsen_factor=coarsen_factor)

        # Get the spectra for the next step by applying the
        # transfer functions.
        highengdep_at_rs = np.dot(np.swapaxes(highengdep_arr, 0, 1), highengphot_spec_at_rs.N)
        lowengphot_spec_at_rs  = lowengphot_tf.sum_specs(highengphot_spec_at_rs.N)
        lowengelec_spec_at_rs  = lowengelec_tf.sum_specs(highengphot_spec_at_rs.N)
        highengphot_spec_at_rs = highengphot_tf.sum_specs(highengphot_spec_at_rs.N) # Need to be modified at the end

        highengphot_spec_at_rs.rs = next_rs
        lowengphot_spec_at_rs.rs  = next_rs
        lowengelec_spec_at_rs.rs  = next_rs

        # We add the treated rs value to the end of the list
        rs_arr.append(rs)

        ## BE CAREFULL Here we do not save the spectra at redshift rs
        ## but rather the one that will be usefull for the next step of the loop
        append_highengphot_spec(highengphot_spec_at_rs)
        append_lowengphot_spec(lowengphot_spec_at_rs)
        append_lowengelec_spec(lowengelec_spec_at_rs)

        if next_rs > end_rs:

            # Set the values of Tm, xHII and xHeII for the next step
            # If we do not set input functions for the ionization fractions and temperature
            # then we reuse the same already used for the next step
            next_Tm    = Tm_arr[-1] if (Tm_vs_rs is None)      else Tm_vs_rs(next_rs)
            next_xHII  = x_arr[-1, 0] if (xHII_vs_rs is None)  else xHII_vs_rs(next_rs)
            next_xHeII = x_arr[-1, 1] if (xHeII_vs_rs is None) else xHeII_vs_rs(next_rs)
            
            Tm_arr = np.append(Tm_arr, next_Tm)
            x_arr = np.append(x_arr, [[next_xHII, next_xHeII]], axis=0)


        # Re-define existing variables.
        rs = next_rs
        dt = dlnz * coarsen_factor/phys.hubble(rs)

    #########################################################################
    #########################################################################
    # END OF LOOP! END OF LOOP!                                             #
    #########################################################################
    #########################################################################

    #f_to_return = (f_low, f_high)
    
    # Some processing to get the data into presentable shape.
    f_low_dict = {
        'H ion':  f_low[:,0],
        'He ion': f_low[:,1],
        'exc':    f_low[:,2],
        'heat':   f_low[:,3],
        'cont':   f_low[:,4]
    }
    f_high_dict = {
        'H ion':  f_high[:,0],
        'He ion': f_high[:,1],
        'exc':    f_high[:,2],
        'heat':   f_high[:,3],
        'cont':   f_high[:,4]
    }

    f = {
        'low': f_low_dict, 'high': f_high_dict
    }

    data = {
        'rs': rs_arr,
        'x': x_arr, 'Tm': Tm_arr,
        'highengphot': out_highengphot_specs,
        'lowengphot': out_lowengphot_specs,
        'lowengelec': out_lowengelec_specs,
        'highengdep': highengdep_at_rs, ## Here we need to know what was the last highenergy_dep
        'f': f
    }

    return data



def evolve_one_step(
    in_spec_elec=None, in_spec_phot=None,
    rate_func_N=None, rate_func_eng=None,
    DM_process=None, mDM=None, sigmav=None, lifetime=None, primary=None, struct_boost=None, init_cond=None, coarsen_factor=1,
    compute_fs_method='no_He', cross_check=False,
    in_highengphot_specs=None, in_lowengphot_specs=None, in_lowengelec_specs=None, in_highengdep=None, 
    xHII_vs_rs=None, xHeII_vs_rs=None, Tm_vs_rs=None):
    """
    Main function computing histories and spectra.
    This function is a little bit different than main.evolve() because here we can save the output and start again the function from the output.
    Moreover we can also fix xe and Tm on the run

    Parameters
    -----------
    in_spec_elec : :class:`.Spectrum`, optional
        Spectrum per injection event into electrons. *in_spec_elec.rs*
        of the :class:`.Spectrum` must be the initial redshift.
    in_spec_phot : :class:`.Spectrum`, optional
        Spectrum per injection event into photons. *in_spec_phot.rs*
        of the :class:`.Spectrum` must be the initial redshift.
    rate_func_N : function, optional
        Function returning number of injection events per volume per time, with redshift :math:`(1+z)` as an input.
    rate_func_eng : function, optional
        Function returning energy injected per volume per time, with redshift :math:`(1+z)` as an input.
    DM_process : {'swave', 'decay'}, optional
        Dark matter process to use.
    sigmav : float, optional
        Thermally averaged cross section for dark matter annihilation.
    lifetime : float, optional
        Decay lifetime for dark matter decay.
    primary : string, optional
        Primary channel of annihilation/decay. See :func:`.get_pppc_spec` for complete list. Use *'elec_delta'* or *'phot_delta'* for delta function injections of a pair of photons/an electron-positron pair.
    struct_boost : function, optional
        Energy injection boost factor due to structure formation.
    start_rs : float, optional
        Starting redshift :math:`(1+z)` to evolve from. Default is :math:`(1+z)` = 3000. Specify only for use with *DM_process*. Otherwise, initialize *in_spec_elec.rs* and/or *in_spec_phot.rs* directly.
    end_rs : float, optional
        Final redshift :math:`(1+z)` to evolve to. Default is 1+z = 4.
    reion_switch : bool
        Reionization model included if *True*, default is *False*.
    helium_TLA : bool
        If *True*, the TLA is solved with helium. Default is *False*.
    reion_rs : float, optional
        Redshift :math:`(1+z)` at which reionization effects turn on.
    photoion_rate_func : tuple of functions, optional
        Functions take redshift :math:`1+z` as input, return the photoionization rate in s\ :sup:`-1` of HI, HeI and HeII respectively. If not specified, defaults to :func:`.photoion_rate`.
    photoheat_rate_func : tuple of functions, optional
        Functions take redshift :math:`1+z` as input, return the photoheating rate in s\ :sup:`-1` of HI, HeI and HeII respectively. If not specified, defaults to :func:`.photoheat_rate`.
    xe_reion_func : function, optional
        Specifies a fixed ionization history after reion_rs.
    init_cond : tuple of floats
        Specifies the initial (xH, xHe, Tm). Defaults to :func:`.Tm_std`, :func:`.xHII_std` and :func:`.xHeII_std` at the *start_rs*.
    coarsen_factor : int
        Coarsening to apply to the transfer function matrix. Default is 1.
    backreaction : bool
        If *False*, uses the baseline TLA solution to calculate :math:`f_c(z)`. Default is True.
    compute_fs_method : {'no_He', 'He_recomb', 'He'}

    mxstep : int, optional
        The maximum number of steps allowed for each integration point. See *scipy.integrate.odeint()* for more information. Default is *1000*.
    rtol : float, optional
        The relative error of the solution. See *scipy.integrate.odeint()* for more information. Default is *1e-4*.
    use_tqdm : bool, optional
        Uses tqdm if *True*. Default is *True*.
    cross_check : bool, optional
        If *True*, compare against 1604.02457 by using original MEDEA files, turning off partial binning, etc. Default is *False*.

    Examples
    --------

    1. *Dark matter annihilation* -- dark matter mass of 50 GeV, annihilation cross section :math:`2 \\times 10^{-26}` cm\ :sup:`3` s\ :sup:`-1`, annihilating to :math:`b \\bar{b}`, solved without backreaction, a coarsening factor of 32 and the default structure formation boost: ::

        import darkhistory.physics as phys

        out = evolve(
            DM_process='swave', mDM=50e9, sigmav=2e-26,
            primary='b', start_rs=3000.,
            backreaction=False,
            struct_boost=phys.struct_boost_func()
        )

    2. *Dark matter decay* -- dark matter mass of 100 GeV, decay lifetime :math:`3 \\times 10^{25}` s, decaying to a pair of :math:`e^+e^-`, solved with backreaction, a coarsening factor of 16: ::

        out = evolve(
            DM_process='decay', mDM=1e8, lifetime=3e25,
            primary='elec_delta', start_rs=3000.,
            backreaction=True
        )

    See Also
    ---------
    :func:`.get_pppc_spec`

    :func:`.struct_boost_func`

    :func:`.photoion_rate`, :func:`.photoheat_rate`

    :func:`.Tm_std`, :func:`.xHII_std` and :func:`.xHeII_std`


    """
    
    #########################################################################
    #########################################################################
    # Input                                                                 #
    #########################################################################
    #########################################################################


    #####################################
    # Initialization for DM_process     #
    #####################################

    # Load data.
    binning = load_data('binning')
    photeng = binning['phot']
    eleceng = binning['elec']

    dep_tf_data = load_data('dep_tf')

    highengphot_tf_interp = dep_tf_data['highengphot']
    lowengphot_tf_interp  = dep_tf_data['lowengphot']
    lowengelec_tf_interp  = dep_tf_data['lowengelec']
    highengdep_interp     = dep_tf_data['highengdep'] 

    ics_tf_data = load_data('ics_tf')

    ics_thomson_ref_tf  = ics_tf_data['thomson']
    ics_rel_ref_tf      = ics_tf_data['rel']
    engloss_ref_tf      = ics_tf_data['engloss']


    start_rs = in_highengphot_specs[-1].rs

    #print("-> Here the redshift is =", start_rs-1)

    # Handle the case where a DM process is specified.
    if DM_process == 'swave':
        if sigmav is None or start_rs is None:
            raise ValueError('sigmav and start_rs must be specified.')
        
        # Get input spectra from PPPC.
        in_spec_elec = pppc.get_pppc_spec(mDM, eleceng, primary, 'elec')
        in_spec_phot = pppc.get_pppc_spec(mDM, photeng, primary, 'phot')
        # Initialize the input spectrum redshift.
        in_spec_elec.rs = start_rs
        in_spec_phot.rs = start_rs
        # Convert to type 'N'.
        in_spec_elec.switch_spec_type('N')
        in_spec_phot.switch_spec_type('N')

        # If struct_boost is none, just set to 1.
        if struct_boost is None:
            def struct_boost(rs):
                return 1.

        # Define the rate functions.
        def rate_func_N(rs):
            return (phys.inj_rate('swave', rs, mDM=mDM, sigmav=sigmav) * struct_boost(rs) / (2*mDM))
            
        def rate_func_eng(rs):
            return (phys.inj_rate('swave', rs, mDM=mDM, sigmav=sigmav) * struct_boost(rs))

    if DM_process == 'decay':
        if lifetime is None or start_rs is None:
            raise ValueError('lifetime and start_rs must be specified.')

        # The decay rate is insensitive to structure formation
        def struct_boost(rs):
            return 1
        
        # Get spectra from PPPC.
        in_spec_elec = pppc.get_pppc_spec(mDM, eleceng, primary, 'elec', decay=True)
        in_spec_phot = pppc.get_pppc_spec(mDM, photeng, primary, 'phot', decay=True)

        # Initialize the input spectrum redshift.
        in_spec_elec.rs = start_rs
        in_spec_phot.rs = start_rs

        # Convert to type 'N'.
        in_spec_elec.switch_spec_type('N')
        in_spec_phot.switch_spec_type('N')

        # Define the rate functions.
        def rate_func_N(rs):
            return (phys.inj_rate('decay', rs, mDM=mDM, lifetime=lifetime) / mDM)
       
        def rate_func_eng(rs):
            return phys.inj_rate('decay', rs, mDM=mDM, lifetime=lifetime)
    


    #####################################
    # Initialization                    #
    #####################################


    # Initialize the initial x and Tm.
    xH_init  = init_cond[0] if (xHII_vs_rs is None)  else xHII_vs_rs(start_rs)
    xHe_init = init_cond[1] if (xHeII_vs_rs is None) else xHeII_vs_rs(start_rs)
    Tm_init  = init_cond[2] if (Tm_vs_rs is None)    else Tm_vs_rs(start_rs)


    # Initialize redshift/timestep related quantities.

    # Default step in the transfer function. Note highengphot_tf_interp.dlnz
    # contains 3 different regimes, and we start with the first.
    dlnz = highengphot_tf_interp.dlnz[-1]

    # The current redshift.
    # Here both are undistinguishable
    rs   = start_rs

    # The timestep between evaluations of transfer functions, including
    # coarsening.
    dt   = dlnz * coarsen_factor / phys.hubble(rs)


    #####################################
    # Input Checks                      #
    #####################################

    if (
        not np.array_equal(in_spec_elec.eng, eleceng)
        or not np.array_equal(in_spec_phot.eng, photeng)
    ):
        raise ValueError('in_spec_elec and in_spec_phot must use config.photeng and config.eleceng respectively as abscissa.')

    if (
        highengphot_tf_interp.dlnz    != lowengphot_tf_interp.dlnz
        or highengphot_tf_interp.dlnz != lowengelec_tf_interp.dlnz
        or lowengphot_tf_interp.dlnz  != lowengelec_tf_interp.dlnz
    ):
        raise ValueError('TransferFuncInterp objects must all have the same dlnz.')

    if in_spec_elec.rs != in_spec_phot.rs:
        raise ValueError('Input spectra must have the same rs.')

    if cross_check:
        print('cross_check has been set to True -- No longer using all MEDEA files and no longer using partial-binning.')

    
    if ((in_highengphot_specs is not None and in_highengphot_specs[-1].rs != rs)
        or (in_lowengphot_specs is not None and in_lowengphot_specs[-1].rs != rs)
        or (in_lowengelec_specs is not None and  in_lowengelec_specs[-1].rs  != rs)
        ):
        print('WARNING: rs of input specs is:', in_highengphot_specs[-1].rs, 'while rs treated is:', rs)
        in_highengphot_specs[-1].rs = rs
        in_lowengelec_specs[-1].rs = rs
        in_lowengelec_specs[-1].rs = rs
        print("Given the good redshift")
        
        
        
    #####################################
    # Initialization (following)        #
    #####################################

    def norm_fac(rs):
        # Normalization to convert from per injection event to
        # per baryon per dlnz step.
        return rate_func_N(rs) * (dlnz * coarsen_factor / phys.hubble(rs) / (phys.nB * rs**3))

    def rate_func_eng_unclustered(rs):
        # The rate excluding structure formation for s-wave annihilation.
        # This is the correct normalization for f_c(z).
        if struct_boost is not None:
            return rate_func_eng(rs)/struct_boost(rs)
        else:
            return rate_func_eng(rs)


    # If there are no electrons, we get a speed up by ignoring them.
    elec_processes = False
    if in_spec_elec.totN() > 0:
        elec_processes = True

    if elec_processes:

        #####################################
        # High-Energy Electrons             #
        #####################################

        # Get the data necessary to compute the electron cooling results.
        # coll_ion_sec_elec_specs is \bar{N} for collisional ionization,
        # and coll_exc_sec_elec_specs \bar{N} for collisional excitation.
        # Heating and others are evaluated in get_elec_cooling_tf
        # itself.

        # Contains information that makes converting an energy loss spectrum
        # to a scattered electron spectrum fast.
        (coll_ion_sec_elec_specs, coll_exc_sec_elec_specs, ics_engloss_data) = main.get_elec_cooling_data(eleceng, photeng)

    #########################################################################
    #########################################################################
    # Pre-Loop Preliminaries                                                #
    #########################################################################
    #########################################################################

    # Initialize the arrays that will contain x and Tm results.
    x_arr  = np.array([[xH_init, xHe_init]])
    Tm_arr = np.array([Tm_init])


    # Initialize Spectra objects to contain all of the output spectra.
    # If we have no input we initialise them to an empy spectra object
    # Otherwise we use the input spectra
    out_highengphot_specs = Spectra([], spec_type='N') if (in_highengphot_specs is None) else in_highengphot_specs
    out_lowengphot_specs  = Spectra([], spec_type='N') if (in_lowengphot_specs is None)  else in_lowengphot_specs
    out_lowengelec_specs  = Spectra([], spec_type='N') if (in_lowengelec_specs is None)  else in_lowengelec_specs

    # Define these methods for speed.
    append_highengphot_spec = out_highengphot_specs.append
    append_lowengphot_spec  = out_lowengphot_specs.append
    append_lowengelec_spec  = out_lowengelec_specs.append

    # Initialize arrays to store f values.
    f_low  = np.empty((0,5))
    f_high = np.empty((0,5))

    # Initialize array to store high-energy energy deposition rate.
    highengdep_grid = np.empty((0,4))

    # Object to help us interpolate over MEDEA results.
    MEDEA_interp = make_interpolator(interp_type='2D', cross_check=cross_check)

    # Initialise the array of rs values that are treated here
    rs_arr = []

    #########################################################################
    #########################################################################
    # What should be in a loop                                              #
    #########################################################################
    #########################################################################


    # Initialize the electron and photon arrays.
    # These will carry the spectra produced by applying the
    # transfer function at rs to high-energy photons.
    if in_highengphot_specs is None:
        highengphot_spec_at_rs = in_spec_phot*0
    elif np.array_equal(in_highengphot_specs[-1].eng, in_spec_phot.eng):
        highengphot_spec_at_rs = in_highengphot_specs[-1]
    else:
        raise ValueError('Problem of compatibility with in_highengphot_specs')
        
    if in_lowengphot_specs is None:
        lowengphot_spec_at_rs = in_spec_phot*0
    elif np.array_equal(in_lowengphot_specs[-1].eng, in_spec_phot.eng):
        lowengphot_spec_at_rs = in_lowengphot_specs[-1]
    else:
        raise ValueError('Problem of compatibility with in_lowengphot_specs')
        
    if in_lowengelec_specs is None:
        lowengelec_spec_at_rs = in_spec_elec*0
    elif np.array_equal(in_lowengelec_specs[-1].eng, in_spec_elec.eng):
        lowengelec_spec_at_rs = in_lowengelec_specs[-1]
    else:
        raise ValueError('Problem of compatibility with in_lowengpelec_specs')
        

    if in_highengdep is None:
        highengdep_at_rs = np.zeros(4)
    elif len(in_highengdep) == 4:
        highengdep_at_rs = in_highengdep
    else:
        raise ValueError('Problem with in_highengdep, it should be an array of length 4')
        
        

    #####################################################################
    #####################################################################
    # Electron Cooling                                                  #
    #####################################################################
    #####################################################################

    # Get the transfer functions corresponding to electron cooling.
    # These are \bar{T}_\gamma, \bar{T}_e and \bar{R}_c.
    if elec_processes:

        # Give the value of the ionization fraction wether we give an input or not
        xHII_elec_cooling  = x_arr[-1, 0] if (xHII_vs_rs  is None) else xHII_vs_rs(rs)
        xHeII_elec_cooling = x_arr[-1, 1] if (xHeII_vs_rs is None) else xHeII_vs_rs(rs)


        (ics_sec_phot_tf, elec_processes_lowengelec_tf,
            deposited_ion_arr, deposited_exc_arr, deposited_heat_arr,
            continuum_loss, deposited_ICS_arr) = get_elec_cooling_tf(eleceng, photeng, rs,
                xHII_elec_cooling, xHeII=xHeII_elec_cooling,
                raw_thomson_tf=ics_thomson_ref_tf,
                raw_rel_tf=ics_rel_ref_tf,
                raw_engloss_tf=engloss_ref_tf,
                coll_ion_sec_elec_specs=coll_ion_sec_elec_specs,
                coll_exc_sec_elec_specs=coll_exc_sec_elec_specs,
                ics_engloss_data=ics_engloss_data)

        # Apply the transfer function to the input electron spectrum.

        # Low energy electrons from electron cooling, per injection event.
        elec_processes_lowengelec_spec = (elec_processes_lowengelec_tf.sum_specs(in_spec_elec))

        #print(elec_processes_lowengelec_spec.eng)
        #print(lowengelec_spec_at_rs.eng)

        # Add this to lowengelec_at_rs.
        lowengelec_spec_at_rs += (elec_processes_lowengelec_spec*norm_fac(rs))

        # High-energy deposition into ionization,
        # *per baryon in this step*.
        # Gaetan: np.dot is the dot product of the two vectors
        deposited_ion = np.dot(deposited_ion_arr,  in_spec_elec.N*norm_fac(rs))
        
        # High-energy deposition into excitation,
        # *per baryon in this step*.
        deposited_exc = np.dot(deposited_exc_arr,  in_spec_elec.N*norm_fac(rs))
        
        # High-energy deposition into heating,
        # *per baryon in this step*.
        deposited_heat = np.dot(deposited_heat_arr, in_spec_elec.N*norm_fac(rs))
        
        # High-energy deposition numerical error,
        # *per baryon in this step*.
        deposited_ICS = np.dot(deposited_ICS_arr,  in_spec_elec.N*norm_fac(rs))


        #######################################
        # Photons from Injected Electrons     #
        #######################################

        # ICS secondary photon spectrum after electron cooling,
        # per injection event.
        ics_phot_spec = ics_sec_phot_tf.sum_specs(in_spec_elec)

        # Get the spectrum from positron annihilation, per injection event.
        # Only half of in_spec_elec is positrons!
        positronium_phot_spec = pos.weighted_photon_spec(photeng) * (in_spec_elec.totN()/2)
        positronium_phot_spec.switch_spec_type('N')

    # Add injected photons + photons from injected electrons
    # to the photon spectrum that got propagated forward.
    if elec_processes:
        highengphot_spec_at_rs += (in_spec_phot + ics_phot_spec + positronium_phot_spec) * norm_fac(rs)
    else:
        highengphot_spec_at_rs += in_spec_phot * norm_fac(rs)

    # Set the redshift correctly.
    highengphot_spec_at_rs.rs = rs

    #####################################################################
    #####################################################################
    # Save the Spectra!                                                 #
    #####################################################################
    #####################################################################

    # At this point, highengphot_at_rs, lowengphot_at_rs and
    # lowengelec_at_rs have been computed for this redshift.
    
    #append_highengphot_spec(highengphot_spec_at_rs)
    #append_lowengphot_spec(lowengphot_spec_at_rs)
    #append_lowengelec_spec(lowengelec_spec_at_rs)

    #####################################################################
    #####################################################################
    # Compute f_c(z)                                                    #
    #####################################################################
    #####################################################################
    if elec_processes:
        # High-energy deposition from input electrons.
        highengdep_at_rs += np.array([
            deposited_ion/dt,
            deposited_exc/dt,
            deposited_heat/dt,
            deposited_ICS/dt
        ])

    # Values of (xHI, xHeI, xHeII) to use for computing f.
    # Use baseline values if no backreaction.
    xHI_loc   = (1-x_arr[-1, 0])        if (xHII_vs_rs is None)  else (1-xHII_vs_rs(rs))
    xHeI_loc  = (phys.chi-x_arr[-1, 1]) if (xHeII_vs_rs is None) else (phys.chi -xHeII_vs_rs(rs))
    xHeII_loc = x_arr[-1, 1]            if (xHeII_vs_rs is None) else xHeII_vs_rs(rs)
    x_vec_for_f = np.array( [ xHI_loc, xHeI_loc, xHeII_loc] )

    #print(rs, dt, lowengelec_spec_at_rs.N[200:204], lowengphot_spec_at_rs.N[200:204])
    #print(x_vec_for_f)

    #if rs < 33 :
    #    print_spectrum_N(highengphot_spec_at_rs, 'high_phot_rs_' + "{:6.4f}".format(rs))
    #    print_spectrum_N(lowengelec_spec_at_rs, 'elec_rs_' + "{:6.4f}".format(rs))
    #    print_spectrum_N(lowengphot_spec_at_rs, 'phot_rs_' + "{:6.4f}".format(rs))
    #    print(rs, dt, rate_func_eng_unclustered(rs), x_vec_for_f, highengdep_at_rs, compute_fs_method, cross_check)


    f_raw = compute_fs(
        MEDEA_interp, lowengelec_spec_at_rs, lowengphot_spec_at_rs,
        x_vec_for_f, rate_func_eng_unclustered(rs), dt,
        highengdep_at_rs, method=compute_fs_method, cross_check=cross_check)


    # Save the f_c(z) values.
    f_low  = np.concatenate((f_low,  [f_raw[0]]))
    f_high = np.concatenate((f_high, [f_raw[1]]))


    # Save CMB upscattered rate and high-energy deposition rate.
    highengdep_grid = np.concatenate( (highengdep_grid, [highengdep_at_rs]) )


    #####################################################################
    #####################################################################
    # ********* AFTER THIS, COMPUTE QUANTITIES FOR NEXT STEP *********  #
    #####################################################################
    #####################################################################

    # Define the next redshift step.
    next_rs = np.exp(np.log(rs) - dlnz * coarsen_factor)

    #####################################################################
    #####################################################################
    # Photon Cooling Transfer Functions                                 #
    #####################################################################
    #####################################################################

    # Get the values of ionization fractions
    xHII_to_interp   = x_arr[-1, 0] if (xHII_vs_rs  is None) else xHII_vs_rs(rs)
    xHeII_to_interp = x_arr[-1, 1] if (xHeII_vs_rs is None) else xHeII_vs_rs(rs)

    # Get the correct transfer function
    highengphot_tf, lowengphot_tf, lowengelec_tf, highengdep_arr, _ = main.get_tf(rs, xHII_to_interp, xHeII_to_interp,dlnz, coarsen_factor=coarsen_factor)

    # Get the spectra for the next step by applying the
    # transfer functions.
    highengdep_at_rs = np.dot(np.swapaxes(highengdep_arr, 0, 1), highengphot_spec_at_rs.N)
    lowengphot_spec_at_rs  = lowengphot_tf.sum_specs(highengphot_spec_at_rs.N)
    lowengelec_spec_at_rs  = lowengelec_tf.sum_specs(highengphot_spec_at_rs.N)
    highengphot_spec_at_rs = highengphot_tf.sum_specs(highengphot_spec_at_rs.N) # Need to be modified at the end

    highengphot_spec_at_rs.rs = next_rs
    lowengphot_spec_at_rs.rs  = next_rs
    lowengelec_spec_at_rs.rs  = next_rs

    # We add the treated rs value to the end of the list
    rs_arr.append(next_rs)

    ## BE CAREFULL Here we do not save the spectra at redshift rs
    ## but rather the one that will be usefull for the next step of the loop
    append_highengphot_spec(highengphot_spec_at_rs)
    append_lowengphot_spec(lowengphot_spec_at_rs)
    append_lowengelec_spec(lowengelec_spec_at_rs)


    # Set the values of Tm, xHII and xHeII for the next step
    # If we do not set input functions for the ionization fractions and temperature
    # then we reuse the same already used for the next step
    next_Tm    = Tm_arr[-1] if (Tm_vs_rs is None)      else Tm_vs_rs(next_rs)
    next_xHII  = x_arr[-1, 0] if (xHII_vs_rs is None)  else xHII_vs_rs(next_rs)
    next_xHeII = x_arr[-1, 1] if (xHeII_vs_rs is None) else xHeII_vs_rs(next_rs)
        
    Tm_arr = np.append(Tm_arr, next_Tm)
    x_arr = np.append(x_arr, [[next_xHII, next_xHeII]], axis=0)


    #########################################################################
    #########################################################################
    # What should be the end of the loop!                                   #
    #########################################################################
    #########################################################################

    #f_to_return = (f_low, f_high)
    
    # Some processing to get the data into presentable shape.
    f_low_dict = {
        'H ion':  f_low[:,0],
        'He ion': f_low[:,1],
        'exc':    f_low[:,2],
        'heat':   f_low[:,3],
        'cont':   f_low[:,4]
    }
    f_high_dict = {
        'H ion':  f_high[:,0],
        'He ion': f_high[:,1],
        'exc':    f_high[:,2],
        'heat':   f_high[:,3],
        'cont':   f_high[:,4]
    }

    f = {
        'low': f_low_dict, 'high': f_high_dict
    }

    data = {
        'rs': rs_arr,
        'x': x_arr, 'Tm': Tm_arr,
        'highengphot': out_highengphot_specs,
        'lowengphot': out_lowengphot_specs,
        'lowengelec': out_lowengelec_specs,
        'highengdep': highengdep_at_rs, ## Here we need to know what was the last highenergy_dep
        'f': f
    }

    return data


########################################################################################
## Copy of the evolve function from main
## We simply add extra output
########################################################################################
def evolve(
    in_spec_elec=None, in_spec_phot=None,
    rate_func_N=None, rate_func_eng=None,
    DM_process=None, mDM=None, sigmav=None, lifetime=None, primary=None,
    struct_boost=None,
    start_rs=None, end_rs=4, helium_TLA=False,
    reion_switch=False, reion_rs=None,
    photoion_rate_func=None, photoheat_rate_func=None, xe_reion_func=None,
    init_cond=None, coarsen_factor=1, backreaction=True, 
    compute_fs_method='no_He', mxstep=1000, rtol=1e-4,
    use_tqdm=True, cross_check=False, tf_mode='table', verbose= 0):
    """
    Main function computing histories and spectra. 

    Parameters
    -----------
    in_spec_elec : :class:`.Spectrum`, optional
        Spectrum per injection event into electrons. *in_spec_elec.rs*
        of the :class:`.Spectrum` must be the initial redshift. 
    in_spec_phot : :class:`.Spectrum`, optional
        Spectrum per injection event into photons. *in_spec_phot.rs* 
        of the :class:`.Spectrum` must be the initial redshift. 
    rate_func_N : function, optional
        Function returning number of injection events per volume per time, with redshift :math:`(1+z)` as an input.  
    rate_func_eng : function, optional
        Function returning energy injected per volume per time, with redshift :math:`(1+z)` as an input. 
    DM_process : {'swave', 'decay'}, optional
        Dark matter process to use. 
    sigmav : float, optional
        Thermally averaged cross section for dark matter annihilation. 
    lifetime : float, optional
        Decay lifetime for dark matter decay.
    primary : string, optional
        Primary channel of annihilation/decay. See :func:`.get_pppc_spec` for complete list. Use *'elec_delta'* or *'phot_delta'* for delta function injections of a pair of photons/an electron-positron pair. 
    struct_boost : function, optional
        Energy injection boost factor due to structure formation.
    start_rs : float, optional
        Starting redshift :math:`(1+z)` to evolve from. Default is :math:`(1+z)` = 3000. Specify only for use with *DM_process*. Otherwise, initialize *in_spec_elec.rs* and/or *in_spec_phot.rs* directly. 
    end_rs : float, optional
        Final redshift :math:`(1+z)` to evolve to. Default is 1+z = 4. 
    reion_switch : bool
        Reionization model included if *True*, default is *False*. 
    helium_TLA : bool
        If *True*, the TLA is solved with helium. Default is *False*.
    reion_rs : float, optional
        Redshift :math:`(1+z)` at which reionization effects turn on. 
    photoion_rate_func : tuple of functions, optional
        Functions take redshift :math:`1+z` as input, return the photoionization rate in s\ :sup:`-1` of HI, HeI and HeII respectively. If not specified, defaults to :func:`.photoion_rate`.
    photoheat_rate_func : tuple of functions, optional
        Functions take redshift :math:`1+z` as input, return the photoheating rate in s\ :sup:`-1` of HI, HeI and HeII respectively. If not specified, defaults to :func:`.photoheat_rate`.
    xe_reion_func : function, optional
        Specifies a fixed ionization history after reion_rs.
    init_cond : tuple of floats
        Specifies the initial (xH, xHe, Tm). Defaults to :func:`.Tm_std`, :func:`.xHII_std` and :func:`.xHeII_std` at the *start_rs*. 
    coarsen_factor : int
        Coarsening to apply to the transfer function matrix. Default is 1. 
    backreaction : bool
        If *False*, uses the baseline TLA solution to calculate :math:`f_c(z)`. Default is True.
    compute_fs_method : {'no_He', 'He_recomb', 'He'}

    mxstep : int, optional
        The maximum number of steps allowed for each integration point. See *scipy.integrate.odeint()* for more information. Default is *1000*. 
    rtol : float, optional
        The relative error of the solution. See *scipy.integrate.odeint()* for more information. Default is *1e-4*.
    use_tqdm : bool, optional
        Uses tqdm if *True*. Default is *True*. 
    cross_check : bool, optional
        If *True*, compare against 1604.02457 by using original MEDEA files, turning off partial binning, etc. Default is *False*.

    tf_mode : {'table', 'nn'}
        Specifies transfer function mode being used. Options: 'table': generate transfer functions from interpolating data tables; 'nn': use neural network to generate transfer functions with preset coarsen factor 12.
    verbose : {0, 1}
        Set verbosity. Tqdm not affected.

    Examples
    --------

    1. *Dark matter annihilation* -- dark matter mass of 50 GeV, annihilation cross section :math:`2 \\times 10^{-26}` cm\ :sup:`3` s\ :sup:`-1`, annihilating to :math:`b \\bar{b}`, solved without backreaction, a coarsening factor of 32 and the default structure formation boost: ::

        import darkhistory.physics as phys

        out = evolve(
            DM_process='swave', mDM=50e9, sigmav=2e-26, 
            primary='b', start_rs=3000., 
            backreaction=False,
            struct_boost=phys.struct_boost_func()
        )

    2. *Dark matter decay* -- dark matter mass of 100 GeV, decay lifetime :math:`3 \\times 10^{25}` s, decaying to a pair of :math:`e^+e^-`, solved with backreaction, a coarsening factor of 16: ::

        out = evolve(
            DM_process='decay', mDM=1e8, lifetime=3e25,
            primary='elec_delta', start_rs=3000.,
            backreaction=True
        ) 

    See Also
    ---------
    :func:`.get_pppc_spec`

    :func:`.struct_boost_func`

    :func:`.photoion_rate`, :func:`.photoheat_rate`

    :func:`.Tm_std`, :func:`.xHII_std` and :func:`.xHeII_std`


    """
    
    #########################################################################
    #########################################################################
    # Input                                                                 #
    #########################################################################
    #########################################################################


    #####################################
    # Initialization for DM_process     #
    #####################################

    # Load data.
    binning = load_data('binning')
    photeng = binning['phot']
    eleceng = binning['elec']

    dep_tf_data = load_data('dep_tf')

    highengphot_tf_interp = dep_tf_data['highengphot']
    lowengphot_tf_interp  = dep_tf_data['lowengphot']
    lowengelec_tf_interp  = dep_tf_data['lowengelec']
    highengdep_interp     = dep_tf_data['highengdep']

    ics_tf_data = load_data('ics_tf')

    ics_thomson_ref_tf  = ics_tf_data['thomson']
    ics_rel_ref_tf      = ics_tf_data['rel']
    engloss_ref_tf      = ics_tf_data['engloss']


    # Handle the case where a DM process is specified. 
    if DM_process == 'swave':
        if sigmav is None or start_rs is None:
            raise ValueError(
                'sigmav and start_rs must be specified.'
            )
        
        # Get input spectra from PPPC. 
        in_spec_elec = pppc.get_pppc_spec(mDM, eleceng, primary, 'elec')
        in_spec_phot = pppc.get_pppc_spec(mDM, photeng, primary, 'phot')
        # Initialize the input spectrum redshift. 
        in_spec_elec.rs = start_rs
        in_spec_phot.rs = start_rs
        # Convert to type 'N'. 
        in_spec_elec.switch_spec_type('N')
        in_spec_phot.switch_spec_type('N')

        # If struct_boost is none, just set to 1. 
        if struct_boost is None:
            def struct_boost(rs):
                return 1.

        # Define the rate functions. 
        def rate_func_N(rs):
            return (phys.inj_rate('swave', rs, mDM=mDM, sigmav=sigmav) * struct_boost(rs) / (2*mDM))
            
        def rate_func_eng(rs):
            return (phys.inj_rate('swave', rs, mDM=mDM, sigmav=sigmav) * struct_boost(rs))

    if DM_process == 'decay':
        if lifetime is None or start_rs is None:
            raise ValueError('lifetime and start_rs must be specified.')

        # The decay rate is insensitive to structure formation
        def struct_boost(rs):
            return 1
        
        # Get spectra from PPPC.
        in_spec_elec = pppc.get_pppc_spec(mDM, eleceng, primary, 'elec', decay=True)
        in_spec_phot = pppc.get_pppc_spec(mDM, photeng, primary, 'phot', decay=True)

        # Initialize the input spectrum redshift. 
        in_spec_elec.rs = start_rs
        in_spec_phot.rs = start_rs
        # Convert to type 'N'. 
        in_spec_elec.switch_spec_type('N')
        in_spec_phot.switch_spec_type('N')

        # Define the rate functions. 
        def rate_func_N(rs):
            return (phys.inj_rate('decay', rs, mDM=mDM, lifetime=lifetime) / mDM)
       
        def rate_func_eng(rs):
            return phys.inj_rate('decay', rs, mDM=mDM, lifetime=lifetime) 
    
    #####################################
    # Input Checks                      #
    #####################################

    if (
        not np.array_equal(in_spec_elec.eng, eleceng) 
        or not np.array_equal(in_spec_phot.eng, photeng)
    ):
        raise ValueError('in_spec_elec and in_spec_phot must use config.photeng and config.eleceng respectively as abscissa.')

    if (
        highengphot_tf_interp.dlnz    != lowengphot_tf_interp.dlnz
        or highengphot_tf_interp.dlnz != lowengelec_tf_interp.dlnz
        or lowengphot_tf_interp.dlnz  != lowengelec_tf_interp.dlnz
    ):
        raise ValueError('TransferFuncInterp objects must all have the same dlnz.')

    if in_spec_elec.rs != in_spec_phot.rs:
        raise ValueError('Input spectra must have the same rs.')

    if cross_check:
        print('cross_check has been set to True -- No longer using all MEDEA files and no longer using partial-binning.')

    #####################################
    # Initialization                    #
    #####################################

    # Initialize start_rs for arbitrary injection. 
    start_rs = in_spec_elec.rs

    # Initialize the initial x and Tm. 
    if init_cond is None:
        # Default to baseline
        xH_init  = phys.xHII_std(start_rs)
        xHe_init = phys.xHeII_std(start_rs)
        Tm_init  = phys.Tm_std(start_rs)
    else:
        # User-specified.
        xH_init  = init_cond[0]
        xHe_init = init_cond[1]
        Tm_init  = init_cond[2]

    # Initialize redshift/timestep related quantities. 

    if tf_mode == 'table':
        # Default step in the transfer function. Note highengphot_tf_interp.dlnz 
        # contains 3 different regimes, and we start with the first.
        dlnz = highengphot_tf_interp.dlnz[-1]
    else:
        # Default step for NN transfer functions.
        dlnz = 0.001

    # The current redshift. 
    rs   = start_rs

    # The timestep between evaluations of transfer functions, including 
    # coarsening. 
    dt   = dlnz * coarsen_factor / phys.hubble(rs)

    # tqdm set-up.
    if use_tqdm:
        from tqdm import tqdm_notebook as tqdm
        pbar = tqdm(
            total=np.ceil((np.log(rs) - np.log(end_rs))/dlnz/coarsen_factor)
        ) 

    def norm_fac(rs):
        # Normalization to convert from per injection event to 
        # per baryon per dlnz step. 
        return rate_func_N(rs) * (
            dlnz * coarsen_factor / phys.hubble(rs) / (phys.nB * rs**3)
        )

    def rate_func_eng_unclustered(rs):
        # The rate excluding structure formation for s-wave annihilation. 
        # This is the correct normalization for f_c(z). 
        if struct_boost is not None:
            return rate_func_eng(rs)/struct_boost(rs)
        else:
            return rate_func_eng(rs)


    # If there are no electrons, we get a speed up by ignoring them. 
    elec_processes = False
    if in_spec_elec.totN() > 0:
        elec_processes = True

    if elec_processes:

        #####################################
        # High-Energy Electrons             #
        #####################################

        # Get the data necessary to compute the electron cooling results. 
        # coll_ion_sec_elec_specs is \bar{N} for collisional ionization, 
        # and coll_exc_sec_elec_specs \bar{N} for collisional excitation. 
        # Heating and others are evaluated in get_elec_cooling_tf
        # itself.

        # Contains information that makes converting an energy loss spectrum 
        # to a scattered electron spectrum fast. 
        (coll_ion_sec_elec_specs, coll_exc_sec_elec_specs, ics_engloss_data) = main.get_elec_cooling_data(eleceng, photeng)

    #########################################################################
    #########################################################################
    # Pre-Loop Preliminaries                                                #
    #########################################################################
    #########################################################################

    # Initialize the arrays that will contain x and Tm results. 
    x_arr  = np.array([[xH_init, xHe_init]])
    Tm_arr = np.array([Tm_init])

    # Initialize Spectra objects to contain all of the output spectra.
    out_highengphot_specs = Spectra([], spec_type='N')
    out_lowengphot_specs  = Spectra([], spec_type='N')
    out_lowengelec_specs  = Spectra([], spec_type='N')

    # Define these methods for speed.
    append_highengphot_spec = out_highengphot_specs.append
    append_lowengphot_spec  = out_lowengphot_specs.append
    append_lowengelec_spec  = out_lowengelec_specs.append

    # Initialize arrays to store f values. 
    f_low  = np.empty((0,5))
    f_high = np.empty((0,5))

    # Initialize array to store high-energy energy deposition rate. 
    highengdep_grid = np.empty((0,4))


    # Object to help us interpolate over MEDEA results. 
    MEDEA_interp = make_interpolator(interp_type='2D', cross_check=cross_check)

    #########################################################################
    #########################################################################
    # LOOP! LOOP! LOOP! LOOP!                                               #
    #########################################################################
    #########################################################################

    while rs > end_rs:

        # Update tqdm. 
        if use_tqdm:
            pbar.update(1)

        #############################
        # First Step Special Cases  #
        #############################
        if rs == start_rs:
            # Initialize the electron and photon arrays. 
            # These will carry the spectra produced by applying the
            # transfer function at rs to high-energy photons.
            highengphot_spec_at_rs = in_spec_phot*0
            lowengphot_spec_at_rs  = in_spec_phot*0
            lowengelec_spec_at_rs  = in_spec_elec*0
            highengdep_at_rs       = np.zeros(4)


        #####################################################################
        #####################################################################
        # Electron Cooling                                                  #
        #####################################################################
        #####################################################################

        # Get the transfer functions corresponding to electron cooling. 
        # These are \bar{T}_\gamma, \bar{T}_e and \bar{R}_c. 
        if elec_processes:

            if backreaction:
                xHII_elec_cooling  = x_arr[-1, 0]
                xHeII_elec_cooling = x_arr[-1, 1]
            else:
                xHII_elec_cooling  = phys.xHII_std(rs)
                xHeII_elec_cooling = phys.xHeII_std(rs)

            (ics_sec_phot_tf, elec_processes_lowengelec_tf,
                deposited_ion_arr, deposited_exc_arr, deposited_heat_arr,
                continuum_loss, deposited_ICS_arr) = main.get_elec_cooling_tf(eleceng, photeng, rs,
                    xHII_elec_cooling, xHeII=xHeII_elec_cooling,
                    raw_thomson_tf=ics_thomson_ref_tf, 
                    raw_rel_tf=ics_rel_ref_tf, 
                    raw_engloss_tf=engloss_ref_tf,
                    coll_ion_sec_elec_specs=coll_ion_sec_elec_specs, 
                    coll_exc_sec_elec_specs=coll_exc_sec_elec_specs,
                    ics_engloss_data=ics_engloss_data)

            # Apply the transfer function to the input electron spectrum. 

            # Low energy electrons from electron cooling, per injection event.
            elec_processes_lowengelec_spec = (elec_processes_lowengelec_tf.sum_specs(in_spec_elec))

            # Add this to lowengelec_at_rs. 
            lowengelec_spec_at_rs += (elec_processes_lowengelec_spec*norm_fac(rs))

            # High-energy deposition into ionization, 
            # *per baryon in this step*.
            # Gaetan: np.dot is the dot product of the two vectors
            deposited_ion = np.dot(deposited_ion_arr,  in_spec_elec.N*norm_fac(rs))
            
            # High-energy deposition into excitation, 
            # *per baryon in this step*. 
            deposited_exc = np.dot(deposited_exc_arr,  in_spec_elec.N*norm_fac(rs))
            
            # High-energy deposition into heating, 
            # *per baryon in this step*. 
            deposited_heat = np.dot(deposited_heat_arr, in_spec_elec.N*norm_fac(rs))
            
            # High-energy deposition numerical error, 
            # *per baryon in this step*. 
            deposited_ICS = np.dot(deposited_ICS_arr,  in_spec_elec.N*norm_fac(rs))


            #######################################
            # Photons from Injected Electrons     #
            #######################################

            # ICS secondary photon spectrum after electron cooling, 
            # per injection event.
            ics_phot_spec = ics_sec_phot_tf.sum_specs(in_spec_elec)

            # Get the spectrum from positron annihilation, per injection event.
            # Only half of in_spec_elec is positrons!
            positronium_phot_spec = pos.weighted_photon_spec(photeng) * (in_spec_elec.totN()/2)
            positronium_phot_spec.switch_spec_type('N')

        # Add injected photons + photons from injected electrons
        # to the photon spectrum that got propagated forward. 
        if elec_processes:
            highengphot_spec_at_rs += (in_spec_phot + ics_phot_spec + positronium_phot_spec) * norm_fac(rs)
        else:
            highengphot_spec_at_rs += in_spec_phot * norm_fac(rs)

        # Set the redshift correctly. 
        highengphot_spec_at_rs.rs = rs

        #####################################################################
        #####################################################################
        # Save the Spectra!                                                 #
        #####################################################################
        #####################################################################

        ## Spectra are now saved later
        # At this point, highengphot_at_rs, lowengphot_at_rs and 
        # lowengelec_at_rs have been computed for this redshift. 
        #append_highengphot_spec(highengphot_spec_at_rs)
        #append_lowengphot_spec(lowengphot_spec_at_rs)
        #append_lowengelec_spec(lowengelec_spec_at_rs)

        #####################################################################
        #####################################################################
        # Compute f_c(z)                                                    #
        #####################################################################
        #####################################################################
        if elec_processes:
            # High-energy deposition from input electrons. 
            highengdep_at_rs += np.array([
                deposited_ion/dt,
                deposited_exc/dt,
                deposited_heat/dt,
                deposited_ICS/dt
            ])

        # Values of (xHI, xHeI, xHeII) to use for computing f.
        if backreaction:
            # Use the previous values with backreaction.
            x_vec_for_f = np.array(
                [1. - x_arr[-1, 0], phys.chi - x_arr[-1, 1], x_arr[-1, 1]]
            )
        else:
            # Use baseline values if no backreaction. 
            x_vec_for_f = np.array([
                    1. - phys.xHII_std(rs), 
                    phys.chi - phys.xHeII_std(rs), 
                    phys.xHeII_std(rs)
            ])

        f_raw = compute_fs(
            MEDEA_interp, lowengelec_spec_at_rs, lowengphot_spec_at_rs,
            x_vec_for_f, rate_func_eng_unclustered(rs), dt,
            highengdep_at_rs, method=compute_fs_method, cross_check=cross_check
        )

        # Save the f_c(z) values.
        f_low  = np.concatenate((f_low,  [f_raw[0]]))
        f_high = np.concatenate((f_high, [f_raw[1]]))

        # print(f_low, f_high)

        # Save CMB upscattered rate and high-energy deposition rate.
        highengdep_grid = np.concatenate(
            (highengdep_grid, [highengdep_at_rs])
        )

        # Compute f for TLA: sum of low and high. 
        f_H_ion = f_raw[0][0] + f_raw[1][0]
        f_exc   = f_raw[0][2] + f_raw[1][2]
        f_heat  = f_raw[0][3] + f_raw[1][3]

        if compute_fs_method == 'old':
            # The old method neglects helium.
            f_He_ion = 0. 
        else:
            f_He_ion = f_raw[0][1] + f_raw[1][1]
        

        #####################################################################
        #####################################################################
        # ********* AFTER THIS, COMPUTE QUANTITIES FOR NEXT STEP *********  #
        #####################################################################
        #####################################################################

        # Define the next redshift step. 
        next_rs = np.exp(np.log(rs) - dlnz * coarsen_factor)

        #####################################################################
        #####################################################################
        # TLA Integration                                                   #
        #####################################################################
        #####################################################################

        # Initial conditions for the TLA, (Tm, xHII, xHeII, xHeIII). 
        # This is simply the last set of these variables. 
        init_cond_TLA = np.array([Tm_arr[-1], x_arr[-1,0], x_arr[-1,1], 0])

        # Solve the TLA for x, Tm for the *next* step. 
        new_vals = tla.get_history(
            np.array([rs, next_rs]), init_cond=init_cond_TLA, 
            f_H_ion=f_H_ion, f_H_exc=f_exc, f_heating=f_heat,
            injection_rate=rate_func_eng_unclustered,
            reion_switch=reion_switch, reion_rs=reion_rs,
            photoion_rate_func=photoion_rate_func,
            photoheat_rate_func=photoheat_rate_func,
            xe_reion_func=xe_reion_func, helium_TLA=helium_TLA,
            f_He_ion=f_He_ion, mxstep=mxstep, rtol=rtol
        )

        #####################################################################
        #####################################################################
        # Photon Cooling Transfer Functions                                 #
        #####################################################################
        #####################################################################


        # Get the transfer functions for this step.
        if not backreaction:
            # Interpolate using the baseline solution.
            xHII_to_interp  = phys.xHII_std(rs)
            xHeII_to_interp = phys.xHeII_std(rs)
        else:
            # Interpolate using the current xHII, xHeII values.
            xHII_to_interp  = x_arr[-1,0]
            xHeII_to_interp = x_arr[-1,1]

        
        highengphot_tf, lowengphot_tf, lowengelec_tf, highengdep_arr, _ = (
            main.get_tf(
                rs, xHII_to_interp, xHeII_to_interp, 
                dlnz, coarsen_factor=coarsen_factor
            )
        )

        """
        # Get the spectra for the next step by applying the 
        # transfer functions. 
        highengdep_at_rs = np.dot( np.swapaxes(highengdep_arr, 0, 1), out_highengphot_specs[-1].N)
        highengphot_spec_at_rs = highengphot_tf.sum_specs(out_highengphot_specs[-1])
        lowengphot_spec_at_rs  = lowengphot_tf.sum_specs(out_highengphot_specs[-1])
        lowengelec_spec_at_rs  = lowengelec_tf.sum_specs(out_highengphot_specs[-1])


        highengphot_spec_at_rs.rs = next_rs
        lowengphot_spec_at_rs.rs  = next_rs
        lowengelec_spec_at_rs.rs  = next_rs
        """ 

        #######################################
        ## Gaetan
        ## In that modified version we save the spectra for next step, not the one we have just used
        ## Indeed this is what we will need in the following

        # Get the spectra for the next step by applying the
        # transfer functions.
        highengdep_at_rs = np.dot(np.swapaxes(highengdep_arr, 0, 1), highengphot_spec_at_rs.N)
        lowengphot_spec_at_rs  = lowengphot_tf.sum_specs(highengphot_spec_at_rs)
        lowengelec_spec_at_rs  = lowengelec_tf.sum_specs(highengphot_spec_at_rs)
        highengphot_spec_at_rs = highengphot_tf.sum_specs(highengphot_spec_at_rs) # Need to be modified at the end

        highengphot_spec_at_rs.rs = next_rs
        lowengphot_spec_at_rs.rs  = next_rs
        lowengelec_spec_at_rs.rs  = next_rs

        ## BE CAREFULL Here we do not save the spectra at redshift rs
        ## but rather the one that will be usefull for the next step of the loop
        append_highengphot_spec(highengphot_spec_at_rs)
        append_lowengphot_spec(lowengphot_spec_at_rs)
        append_lowengelec_spec(lowengelec_spec_at_rs)
        

        #if next_rs > end_rs:
            # Only save if next_rs < end_rs, since these are the x, Tm
            # values for the next redshift.

        # Save the x, Tm data for the next step in x_arr and Tm_arr.
        Tm_arr = np.append(Tm_arr, new_vals[-1, 0])

        if helium_TLA:
            # Append the calculated xHe to x_arr. 
            x_arr  = np.append(x_arr,  [[new_vals[-1,1], new_vals[-1,2]]], axis=0)
        else:
            # Append the baseline solution value. 
            x_arr  = np.append(x_arr,  [[new_vals[-1,1], phys.xHeII_std(next_rs)]], axis=0)

        eV_to_K = 11604.5250061657

        # Here we print some value in order to check where we are in the process
        # At redshift z=rs-1 we have this f_heat (computed here)
        print("| z (next): {:2.2f}".format(next_rs-1), "| x_HII: {:1.3e}".format(x_arr[-1, 0]), "| Tm: {:1.3e}".format(Tm_arr[-1]*eV_to_K), "K | f_heat: {:1.3e}".format(f_low[-1, 3] + f_high[-1, 3]))

        # Re-define existing variables. 
        rs = next_rs
        dt = dlnz * coarsen_factor/phys.hubble(rs)


    #########################################################################
    #########################################################################
    # END OF LOOP! END OF LOOP!                                             #
    #########################################################################
    #########################################################################


    if use_tqdm:
        pbar.close()

    
    # Some processing to get the data into presentable shape. 
    f_low_dict = {
        'H ion':  f_low[:,0],
        'He ion': f_low[:,1],
        'exc':    f_low[:,2],
        'heat':   f_low[:,3],
        'cont':   f_low[:,4]
    }
    f_high_dict = {
        'H ion':  f_high[:,0],
        'He ion': f_high[:,1],
        'exc':    f_high[:,2],
        'heat':   f_high[:,3],
        'cont':   f_high[:,4]
    }

    f = {
        'low': f_low_dict, 'high': f_high_dict
    }

    data = {
        'rs': out_highengphot_specs.rs,
        'x': x_arr, 'Tm': Tm_arr, 
        'highengphot': out_highengphot_specs,
        'lowengphot': out_lowengphot_specs, 
        'lowengelec': out_lowengelec_specs,
        'highengdep': highengdep_at_rs,
        'f': f
    }

  

    return data

