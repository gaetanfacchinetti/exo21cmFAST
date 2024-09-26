#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline2d.h>

/*** Some usefull math macros ***/
#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

static double mnarg1,mnarg2;
#define FMAX(a,b) (mnarg1=(a),mnarg2=(b),(mnarg1) > (mnarg2) ?\
(mnarg1) : (mnarg2))

static double mnarg1,mnarg2;
#define FMIN(a,b) (mnarg1=(a),mnarg2=(b),(mnarg1) < (mnarg2) ?\
(mnarg1) : (mnarg2))

#define ERFC_NPTS (int) 75
#define ERFC_PARAM_DELTA (float) 0.1
static double log_erfc_table[ERFC_NPTS], erfc_params[ERFC_NPTS];
static gsl_interp_accel *erfc_acc;
static gsl_spline *erfc_spline;

#define NGaussLegendre 40  //defines the number of points in the Gauss-Legendre quadrature integration

#define NMass 1200 // Gaetan: changed from 300 to 1200 in order to gain precision with sharp NCDM transfer functions

#define NSFR_high 200
#define NSFR_low 250
#define NGL_SFR 100 // 100
#define NMTURN 50//100
#define LOG10_MTURN_MAX ((double)(10))
#define LOG10_MTURN_MIN ((double)(5.-9e-8))

#define NR_END 1
#define FREE_ARG char*

#define MM 7
#define NSTACK 50

#define EPS2 3.0e-11

#define Luv_over_SFR (double)(1./1.15/1e-28)

//     Luv/SFR = 1 / 1.15 x 10^-28 [M_solar yr^-1/erg s^-1 Hz^-1]
//     G. Sun and S. R. Furlanetto (2016) MNRAS, 417, 33

#define delta_lnMhalo (double)(5e-6)
#define Mhalo_min (double)(1e6)
#define Mhalo_max (double)(1e16)

float calibrated_NF_min;

double *deltaz=NULL, *deltaz_smoothed=NULL, *NeutralFractions=NULL, *z_Q=NULL, *Q_value=NULL, *nf_vals=NULL, *z_vals=NULL;
int N_NFsamples,N_extrapolated, N_analytic, N_calibrated, N_deltaz;

bool initialised_ComputeLF = false;

gsl_interp_accel *LF_spline_acc;
gsl_spline *LF_spline;

gsl_interp_accel *deriv_spline_acc;
gsl_spline *deriv_spline;


gsl_spline2d *spline_log10_ps_v;
gsl_interp_accel *acc_nB, *acc_w;

struct CosmoParams *cosmo_params_ps;
struct UserParams *user_params_ps;
struct AstroParams *astro_params_ps;
struct FlagOptions *flag_options_ps;

//double sigma_norm, R, theta_cmb, omhh, z_equality, y_d, sound_horizon, alpha_nu, f_nu, f_baryon, beta_c, d2fact, DEL_CURR, SIG_CURR;
double sigma_norm, theta_cmb, omhh, z_equality, y_d, sound_horizon, alpha_nu, f_nu, f_baryon, beta_c, d2fact, DEL_CURR, SIG_CURR, GROWTH_FROM_PMF;

float MinMass, mass_bin_width, inv_mass_bin_width;

double sigmaparam_FgtrM_bias(float z, float sigsmallR, float del_bias, float sig_bias);

float *Mass_InterpTable=NULL, *Sigma_InterpTable=NULL, *dSigmadm_InterpTable=NULL;

float *log10_overdense_spline_SFR, *log10_Nion_spline, *Overdense_spline_SFR, *Nion_spline;
float *prev_log10_overdense_spline_SFR, *prev_log10_Nion_spline, *prev_Overdense_spline_SFR, *prev_Nion_spline;
float *Mturns, *Mturns_MINI;
float *log10_Nion_spline_MINI, *Nion_spline_MINI;
float *prev_log10_Nion_spline_MINI, *prev_Nion_spline_MINI;

float *xi_SFR,*wi_SFR, *xi_SFR_Xray, *wi_SFR_Xray;

float *overdense_high_table, *overdense_low_table, *log10_overdense_low_table;
float **log10_SFRD_z_low_table, **SFRD_z_high_table;
float **log10_SFRD_z_low_table_MINI, **SFRD_z_high_table_MINI;

double *lnMhalo_param, *Muv_param, *Mhalo_param;
double *log10phi, *M_uv_z, *M_h_z;
double *lnMhalo_param_MINI, *Muv_param_MINI, *Mhalo_param_MINI;
double *log10phi_MINI; *M_uv_z_MINI, *M_h_z_MINI;
double *deriv, *lnM_temp, *deriv_temp;

double *z_val=NULL, *z_X_val=NULL, *Nion_z_val=NULL, *SFRD_val=NULL;
double *Nion_z_val_MINI=NULL, *SFRD_val_MINI=NULL;


void initialiseSigmaMInterpTable(float M_Min, float M_Max);
double sigmaFromInterpolationTables(double M);
double dsigmasqdmFromInterpolationTables(double M);
void setFromInterpolationTables(double growthf, double M, double *sigma, double *dsigmadm);
void freeSigmaMInterpTable();
void initialiseGL_Nion(int n, float M_Min, float M_Max);
void initialiseGL_Nion_Xray(int n, float M_Min, float M_Max);

float Mass_limit (float logM, float PL, float FRAC);
void bisection(float *x, float xlow, float xup, int *iter);
float Mass_limit_bisection(float Mmin, float Mmax, float PL, float FRAC);

double sheth_delc(double del, double sig);
float dNdM_conditional(float growthf, float M1, float M2, float delta1, float delta2, float sigma2);
double dNion_ConditionallnM(double lnM, void *params);
double Nion_ConditionalM(double growthf, double M1, double M2, double sigma2, double delta1, double delta2, double MassTurnover, double Alpha_star, double Alpha_esc, double Fstar10, double Fesc10, double Mlim_Fstar, double Mlim_Fesc, bool FAST_FCOLL_TABLES);
double dNion_ConditionallnM_MINI(double lnM, void *params);
double Nion_ConditionalM_MINI(double growthf, double M1, double M2, double sigma2, double delta1, double delta2, double MassTurnover, double MassTurnover_upper, double Alpha_star, double Alpha_esc, double Fstar10, double Fesc10, double Mlim_Fstar, double Mlim_Fesc, bool FAST_FCOLL_TABLES);

float GaussLegendreQuad_Nion(int Type, int n, float growthf, float M2, float sigma2, float delta1, float delta2, float MassTurnover, float Alpha_star, float Alpha_esc, float Fstar10, float Fesc10, float Mlim_Fstar, float Mlim_Fesc, bool FAST_FCOLL_TABLES);
float GaussLegendreQuad_Nion_MINI(int Type, int n, float growthf, float M2, float sigma2, float delta1, float delta2, float MassTurnover, float MassTurnover_upper, float Alpha_star, float Alpha_esc, float Fstar7_MINI, float Fesc7_MINI, float Mlim_Fstar_MINI, float Mlim_Fesc_MINI, bool FAST_FCOLL_TABLES);


//JBM: Exact integral for power-law indices non zero (for zero it's erfc)
double Fcollapprox (double numin, double beta);



int n_redshifts_1DTable;
double zmin_1DTable, zmax_1DTable, zbin_width_1DTable;
double *FgtrM_1DTable_linear=NULL;

static gsl_interp_accel *Q_at_z_spline_acc=NULL;
static gsl_spline *Q_at_z_spline=NULL;
static gsl_interp_accel *z_at_Q_spline_acc=NULL;
static gsl_spline *z_at_Q_spline=NULL;
static double Zmin, Zmax, Qmin, Qmax;
void Q_at_z(double z, double *splined_value);
void z_at_Q(double Q, double *splined_value);

static gsl_interp_accel *deltaz_spline_for_photoncons_acc=NULL;
static gsl_spline *deltaz_spline_for_photoncons=NULL;

static gsl_interp_accel *NFHistory_spline_acc=NULL;
static gsl_spline *NFHistory_spline=NULL;
static gsl_interp_accel *z_NFHistory_spline_acc=NULL;
static gsl_spline *z_NFHistory_spline=NULL;

void initialise_NFHistory_spline(double *redshifts, double *NF_estimate, int NSpline);
void z_at_NFHist(double xHI_Hist, double *splined_value);
void NFHist_at_z(double z, double *splined_value);

//int nbin;
//double *z_Q, *Q_value, *Q_z, *z_value;

double FinalNF_Estimate, FirstNF_Estimate;

struct parameters_gsl_FgtrM_int_{
    double z_obs;
    double gf_obs;
};

struct parameters_gsl_SFR_General_int_{
    double z_obs;
    double gf_obs;
    double Mdrop;
    double Mdrop_upper;
    double pl_star;
    double pl_esc;
    double frac_star;
    double frac_esc;
    double LimitMass_Fstar;
    double LimitMass_Fesc;
};

struct parameters_gsl_SFR_con_int_{
    double gf_obs;
    double Mval;
    double sigma2;
    double delta1;
    double delta2;
    double Mdrop;
    double Mdrop_upper;
    double pl_star;
    double pl_esc;
    double frac_star;
    double frac_esc;
    double LimitMass_Fstar;
    double LimitMass_Fesc;
};

struct parameters_gsl_pmf_induced_power_int_{
    double nB;
    double v;
    double w;
};




unsigned long *lvector(long nl, long nh);
void free_lvector(unsigned long *v, long nl, long nh);

float *vector(long nl, long nh);
void free_vector(float *v, long nl, long nh);

void spline(float x[], float y[], int n, float yp1, float ypn, float y2[]);
void splint(float xa[], float ya[], float y2a[], int n, float x, float *y);

void gauleg(float x1, float x2, float x[], float w[], int n);

/*****     FUNCTION PROTOTYPES     *****/
void init_ps(); /* initialize global variables, MUST CALL THIS FIRST!!! */
void free_ps(); /* deallocates the gsl structures from init_ps */
double power_spectrum(double k); // Defines the matter power-spectrum
double power_spectrum_LCDM(double k); 

double transfer_function_PMF(double k);
double pmf_induced_power_spectrum(double k);
void interpolate_power_spectrum_from_pmf(bool free_tables);
double MB_PMF_TABLE(float z);

double window_function(double kR);
double dsigma_dlnk(double k, void *params);
double dsigma_dk_LCDM(double k, void *params);
double sigma_z0(double M); //calculates sigma at z=0 (no dicke)
double power_in_k(double k); /* Returns the value of the linear power spectrum density (i.e. <|delta_k|^2>/V) at a given k mode at z=0 */

double transfer_function(double k, bool with_pmf);
double transfer_function_LCDM(double k);
double transfer_function_EH(double k);
double transfer_function_BBKS(double k);
double transfer_function_EBW(double k);
double transfer_function_Peebles(double k);
double transfer_function_WF(double k);
double transfer_function_CLASS(double k);
double transfer_function_CLASS_LCDM(double k);

double transfer_function_abgd(double k, double alpha, double beta, double gamma, double delta);
double transfer_function_sharp(double k, double alpha, double delta);
double analytical_transfer_function_nCDM(double k);

void TFset_parameters();


double TF_CLASS(double k, int flag_dv); //transfer function of matter (flag_dv=0) and relative velocities (flag_dv=1) fluctuations from CLASS
double TF_CLASS_LCDM(double k, int flag_dv); //transfer function of matter (flag_dv=0) and relative velocities (flag_dv=1) fluctuations from CLASS for LCDM cosmology
double power_in_vcb(double k); /* Returns the value of the DM-b relative velocity power spectrum density (i.e. <|delta_k|^2>/V) at a given k mode at z=0 */


double FgtrM(double z, double M);
double FgtrM_wsigma(double z, double sig);
double FgtrM_st(double z, double M);
double FgtrM_Watson(double growthf, double M);
double FgtrM_Watson_z(double z, double growthf, double M);
double FgtrM_General(double z, double M);

float erfcc(float x);
double splined_erfc(double x);


void Broadcast_struct_global_PS(struct UserParams *user_params, struct CosmoParams *cosmo_params){

    cosmo_params_ps = cosmo_params;
    user_params_ps = user_params;
}

//
//
// Part 1. 
//
// ------------------------------------- //
// ------------------------------------- //
/* Power spectrum and transfer functions */
// ------------------------------------- //
// ------------------------------------- //

/* 
    power_spectrum(double k)

    primordial power spectrum up to a constant prefactor times k
    
    Params
    ------
    - k (double) mode in Mpc^{-1} 
    
*/
double primordial_power_spectrum(double k)
{
    double k0 = 0.05; // reference value of k0 in Mpc^{-1}
    double index = cosmo_params_ps->POWER_INDEX + 0.5 * cosmo_params_ps->ALPHA_S_PS * log(k/k0);
    return pow(k/k0, index) * k0; 
}


/* 
    power_spectrum(double k)

    matter power spectrum up to a constant prefactor times k
    returns P_{curvature}(k) * [T(k)]^2 where T is the transfer function
    
    Params
    ------
    - k (double) mode in Mpc^{-1} 
    
*/
double power_spectrum(double k)
{   
    return  primordial_power_spectrum(k) * pow(transfer_function(k, true), 2);
}

double power_spectrum_LCDM(double k)
{
    return  primordial_power_spectrum(k) * pow(transfer_function_LCDM(k), 2);
}


double transfer_function(double k, bool with_pmf)
{
    double T;

    if (user_params_ps->POWER_SPECTRUM == 5) // CLASS
        T = transfer_function_CLASS(k);
    else
    {
        T = transfer_function_LCDM(k) * analytical_transfer_function_nCDM(k);

        // with PMF we add the transfer function induced from PMF
        // condition put here as so far no implementation of PMF with CLASS
        if (user_params_ps->PMF_POWER_SPECTRUM && with_pmf)
            T = T * transfer_function_PMF(k);
    }

    return T;
}


/* 
    transfer_function_LCDM(double k)

    matter transfer function according to the chosen model by the user

    Params
    ------
    - k (double) mode in Mpc^{-1}
*/
double transfer_function_LCDM(double k)
{
    switch(user_params_ps->POWER_SPECTRUM)
    {
        case(0): 
            return transfer_function_EH(k);
            break;

        case(1): // BBKS
            return transfer_function_BBKS(k);
            break;
        
        case(2): // Efstathiou,G., Bond,J.R., and White,S.D.M., MNRAS,258,1P (1992)
            return transfer_function_EBW(k);
            break;

        case(3): // Peebles, pg. 626
            return transfer_function_Peebles(k);
            break;

        case(4): // White, SDM and Frenk, CS, 1991, 379, 52
            return transfer_function_WF(k);
            break;

        case(5): // output of CLASS
            return transfer_function_CLASS_LCDM(k);
            break;

        default:
            LOG_ERROR("No such power spectrum defined: %i. Output is bogus.", user_params_ps->POWER_SPECTRUM);
            Throw(ValueError);

    }

}

// FUNCTION transfer_function_EH is the power spectrum transfer function from Eisenstein & Hu ApJ, 1999, 511, 5
double transfer_function_EH(double k)
{
    double q, gamma_eff, q_eff, TF_m, q_nu;

    q = k*pow(theta_cmb,2)/omhh;
    gamma_eff = sqrt(alpha_nu) + (1.0-sqrt(alpha_nu))/(1.0+pow(0.43*k*sound_horizon, 4));
    q_eff = q/gamma_eff;
    TF_m = log(E+1.84*beta_c*sqrt(alpha_nu)*q_eff);
    TF_m /= TF_m + pow(q_eff,2) * (14.4 + 325.0/(1.0+60.5*pow(q_eff,1.11)));
    q_nu = 3.92*q/sqrt(f_nu/N_nu);
    TF_m *= 1.0 + (1.2*pow(f_nu,0.64)*pow(N_nu,0.3+0.6*f_nu)) / (pow(q_nu,-1.6)+pow(q_nu,0.8));

    return TF_m;
}

// Initialises global parameters to compute transfer_function_EH
void TFset_parameters()
{
    double z_drag, R_drag, R_equality, p_c, p_cb, f_c, f_cb, f_nub, k_equality;

    LOG_DEBUG("Setting Transfer Function parameters.");

    z_equality = 25000*omhh*pow(theta_cmb, -4) - 1.0;
    k_equality = 0.0746*omhh/(theta_cmb*theta_cmb);

    z_drag = 0.313*pow(omhh,-0.419) * (1 + 0.607*pow(omhh, 0.674));
    z_drag = 1 + z_drag*pow(cosmo_params_ps->OMb*cosmo_params_ps->hlittle*cosmo_params_ps->hlittle, 0.238*pow(omhh, 0.223));
    z_drag *= 1291 * pow(omhh, 0.251) / (1 + 0.659*pow(omhh, 0.828));

    y_d = (1 + z_equality) / (1.0 + z_drag);

    R_drag = 31.5 * cosmo_params_ps->OMb*cosmo_params_ps->hlittle*cosmo_params_ps->hlittle * pow(theta_cmb, -4) * 1000 / (1.0 + z_drag);
    R_equality = 31.5 * cosmo_params_ps->OMb*cosmo_params_ps->hlittle*cosmo_params_ps->hlittle * pow(theta_cmb, -4) * 1000 / (1.0 + z_equality);

    sound_horizon = 2.0/3.0/k_equality * sqrt(6.0/R_equality) *
    log( (sqrt(1+R_drag) + sqrt(R_drag+R_equality)) / (1.0 + sqrt(R_equality)) );

    p_c = -(5 - sqrt(1 + 24*(1 - f_nu-f_baryon)))/4.0;
    p_cb = -(5 - sqrt(1 + 24*(1 - f_nu)))/4.0;
    f_c = 1 - f_nu - f_baryon;
    f_cb = 1 - f_nu;
    f_nub = f_nu+f_baryon;

    alpha_nu = (f_c/f_cb) * (2*(p_c+p_cb)+5)/(4*p_cb+5.0);
    alpha_nu *= 1 - 0.553*f_nub+0.126*pow(f_nub,3);
    alpha_nu /= 1-0.193*sqrt(f_nu)+0.169*f_nu;
    alpha_nu *= pow(1+y_d, p_c-p_cb);
    alpha_nu *= 1+ (p_cb-p_c)/2.0 * (1.0+1.0/(4.0*p_c+3.0)/(4.0*p_cb+7.0))/(1.0+y_d);
    beta_c = 1.0/(1.0-0.949*f_nub);
}


// Transfer function from BBKS - k in Mpc^{-1}
double transfer_function_BBKS(double k)
{
    double gamma = cosmo_params_ps->OMm * cosmo_params_ps->hlittle * pow(E, -(cosmo_params_ps->OMb) - (cosmo_params_ps->OMb/cosmo_params_ps->OMm));
    double q = k / (cosmo_params_ps->hlittle*gamma);
    return sqrt( (log(1.0+2.34*q)/(2.34*q)) * pow( 1.0+3.89*q + pow(16.1*q, 2) + pow( 5.46*q, 3) + pow(6.71*q, 4), -0.25));
}

// Transfer function from Efstathiou,G., Bond,J.R., and White,S.D.M., MNRAS,258,1P (1992) - k in Mpc^{-1}
double transfer_function_EBW(double k)
{
    double gamma = 0.25;
    double aa = 6.4/(cosmo_params_ps->hlittle*gamma);
    double bb = 3.0/(cosmo_params_ps->hlittle*gamma);
    double cc = 1.7/(cosmo_params_ps->hlittle*gamma);

    return 1.0/pow(1.0 + pow( aa*k + pow(bb*k, 1.5) + pow(cc*k,2), 1.13), 1.0/1.13);    
}
// Transfer function from Peebles, pg. 626- k in Mpc^{-1}
double transfer_function_Peebles(double k)
{
    double gamma = cosmo_params_ps->OMm * cosmo_params_ps->hlittle * pow(E, -(cosmo_params_ps->OMb) - (cosmo_params_ps->OMb/cosmo_params_ps->OMm));
    double aa = 8.0 / (cosmo_params_ps->hlittle*gamma);
    double bb = 4.7 / pow(cosmo_params_ps->hlittle*gamma, 2);

    return 1.0 / (1.0 + aa*k + bb*k*k);
}
// Transfer function from White, SDM and Frenk, CS, 1991, 379, 52- k in Mpc^{-1}
double transfer_function_WF(double k)
{
    double gamma = cosmo_params_ps->OMm * cosmo_params_ps->hlittle * pow(E, -(cosmo_params_ps->OMb) - (cosmo_params_ps->OMb/cosmo_params_ps->OMm));
    double aa = 1.7/(cosmo_params_ps->hlittle*gamma);
    double bb = 9.0/pow(cosmo_params_ps->hlittle*gamma, 1.5);
    double cc = 1.0/pow(cosmo_params_ps->hlittle*gamma, 2);
    
    return  139.2838 / (1.0 + aa*k + bb*pow(k, 1.5) + cc*k*k);
}

// Transfer function from CLASS - k in Mpc^{-1}
double transfer_function_CLASS(double k)
{
    double T = TF_CLASS(k, 0);
    if(user_params_ps->USE_RELATIVE_VELOCITIES) // Add average relvel suppression
        T *= sqrt(1.0 - A_VCB_PM*exp( -pow(log(k/KP_VCB_PM),2.0)/(2.0*SIGMAK_VCB_PM*SIGMAK_VCB_PM))); //for v=vrms

    return T;
}

// Transfer function from CLASS - k in Mpc^{-1} for LCDM cosmology
double transfer_function_CLASS_LCDM(double k)
{
    double T = TF_CLASS_LCDM(k, 0);
    if(user_params_ps->USE_RELATIVE_VELOCITIES) // Add average relvel suppression
        T *= sqrt(1.0 - A_VCB_PM*exp( -pow(log(k/KP_VCB_PM),2.0)/(2.0*SIGMAK_VCB_PM*SIGMAK_VCB_PM))); //for v=vrms

    return T;
}


/*  
    transfer_function_abgd(double k, double alpha, double beta, double gamma, double delta)
    
    parametrises the transfer function in an effective way that can capture WDM, (H+C)DM or (W+C)DM scenarios 
    this parametrisation has been introduced in arXiv:2206.08188.
    the case of traditional WDM is recovered from delta = 0, beta = 2*nu gamma = -5/nu 
    with nu = 1.12 (arXiv:astro-ph/0501562) 

    Params
    ------
    - k (double) mode in Mpc^{-1} 
    - alpha (double) in Mpc (caracteristic scale)
    - beta (double)
    - gamma (double)
    - delta (double)
*/
double transfer_function_abgd(double k, double alpha, double beta, double gamma, double delta)
{
    return (1-delta) * pow(1 + pow(alpha * k, beta), gamma) + delta;
}

/*  
    transfer_function_sharp(double k, double R, double frac)
    
    parametrises a simple transfer function

    Params
    ------
    - k (double) mode in Mpc^{-1} 
    - R (double) in Mpc (caracteristic scale)
    - frac (fouble) percentage 
*/
double transfer_function_sharp(double k, double R, double frac)
{
    if (k < 1.0/R) 
        return 1.0; 
    else
        return frac;
}

/* 
    transfer_function_PMF(double k)

    paremtrises a transfer function for the (Primordial Magnetic Field)-induced matter power spectrum

    Here we assume that the effect of PMF can be caracterised by a transfer function
    that is redshift independant. In full generality, one should include redshift differences using
    the growth_from_pmf function. However growth_from_pmf and the standard growth function only differ
    at large redshifts (at most by 5% - 7% at redshift 35). We have checked, recalculing the excursion
    set theory expression with a moving barrier (which is then mass dependent) that this does not really
    change anything for the mass function on the redshift and mass ranges that matters for 21cm physics 

*/
double transfer_function_PMF(double k)
{
    //LOG_DEBUG("k = %e, pmf_ps = %e, primordial = %e, sigma_norm = %e, transfer_func = %e", k, pmf_induced_power_spectrum(k), primordial_power_spectrum(k), sigma_norm,  transfer_function(k, false));
    return sqrt(1.0 + pmf_induced_power_spectrum(k) / primordial_power_spectrum(k) / (TWOPI * PI) / (sigma_norm * sigma_norm)  * pow(transfer_function(k, false), -2));
    //return sqrt(pmf_induced_power_spectrum(k) / primordial_power_spectrum(k) / 2.0) / (PI * sigma_norm);
}

/*
    analytical_transfer_function_nCDM(double k)

    returns the transfer function (ratio of true nCDM power spectrum 
    to CDM power spectrum) for the different models chosen with the
    flag option ANALYTICAL_TF_NCDM (analytical formulas)

    Params
    ------
    - k (double) mode in Mpc^{-1}
*/
double analytical_transfer_function_nCDM(double k)
{  
    // CLASS output the total TF, no need to add an analytical nCDM
    // Transfer function (already computed by CLASS)
    if (user_params_ps->POWER_SPECTRUM == 5)
        return 1.0;

    if (user_params_ps->ANALYTICAL_TF_NCDM == 0) // classical LCDM model
        return 1.0;
    else if (user_params_ps->ANALYTICAL_TF_NCDM == 1) // vanilla warm dark matter
    {
        double m_wdm;

        if (user_params_ps->USE_INVERSE_PARAMS)
        {
            if (cosmo_params_ps->INVERSE_M_WDM == 0)
                return 1.0;
            
            m_wdm = 1.0/cosmo_params_ps->INVERSE_M_WDM;
        }
        else
            m_wdm = cosmo_params_ps->M_WDM;
        
        // cutoff parameter according to Q. Decant PhD thesis (in Mpc)
        double alpha = 0.049;
        if (m_wdm > 3.0)
            alpha = (m_wdm < 6.0) ? 0.045 : 0.043;

        alpha *= pow(omhh/0.25/m_wdm, 0.11) / m_wdm * 1.54518467138; // the last term is 0.7^(-1.22)

        // transfer function according to Viel,Lesgourgues, Haehnelt, Matarrese, Riotto arXiv:astro-ph/0501562
        return transfer_function_abgd(k, alpha, 2 * 1.12, -5.0 / 1.12, 0.0);

    }
    else if (user_params_ps->ANALYTICAL_TF_NCDM == 2) // alpha beta gamma delta parametrisation of cutoff
        return transfer_function_abgd(k, cosmo_params_ps->ALPHA_NCDM_TF, cosmo_params_ps->BETA_NCDM_TF, cosmo_params_ps->GAMMA_NCDM_TF, cosmo_params_ps->DELTA_NCDM_TF);
    else if (user_params_ps->ANALYTICAL_TF_NCDM == 3) // sharp transfer function
        return transfer_function_sharp(k, cosmo_params_ps->ALPHA_NCDM_TF, cosmo_params_ps->DELTA_NCDM_TF);
    else
    {
        LOG_ERROR("No such ANALYTICAL_TF_NCDM defined: %i. Output is bogus.", user_params_ps->ANALYTICAL_TF_NCDM);
        Throw(ValueError);
    }     
}



void interpolate_power_spectrum_from_pmf(bool free_tables)
{

    if (free_tables == false)
    { 

        LOG_DEBUG("INTERPOLATING PMF POWER SPECTRUM TABLES");

        double log10_w_array[300], nB_array[100];

        const size_t nx = sizeof(nB_array) / sizeof(double); /* x grid points */
        const size_t ny = sizeof(log10_w_array) / sizeof(double); /* y grid points */
        double *log10_ps_array = malloc(nx * ny * sizeof(double));

        gsl_set_error_handler_off();
     
        const gsl_interp2d_type *T = gsl_interp2d_bilinear;
        spline_log10_ps_v = gsl_spline2d_alloc(T, nx, ny);
        
        acc_nB = gsl_interp_accel_alloc();
        acc_w  = gsl_interp_accel_alloc();


        //LOG_DEBUG("We have %d, %d", nx, ny);

        FILE *F;
        char filename[500];

        sprintf(filename, "%s/%s", global_params.external_table_path, "dimensionless_power_spectrum_v_PMF.txt");

        if (!(F = fopen(filename, "r"))) {
                LOG_ERROR("Unable to open file: %s for reading.", filename);
                Throw(IOError);
            }

        // Passing the first two lines of comments
        fscanf(F, "%*[^\n]\n");
        fscanf(F, "%*[^\n]\n");
    
        int nscans, i, j;
        float temp_nB, temp_w, temp_log10_ps;

        for (int p = 0; p < (nx*ny); p++)
        {
            nscans = fscanf(F, "%e %e %e ", &temp_nB, &temp_w, &temp_log10_ps);
            if (nscans != 3) {
                LOG_ERROR("Reading PS_v for PMF failed: %d", nscans);
                Throw(IOError);
            }

            j = (int) p % ny; // index along w
            i = (int)((p - j)/(1.0 * ny)); // index along nB

            //LOG_DEBUG("Looping over the table: i = %d, j = %d", i, j);
            //LOG_DEBUG("We have read here, %e, %e, %e", temp_nB, log10(temp_w), temp_log10_ps);

            if (i == 0)
                log10_w_array[j] = log10(temp_w);

            if (j==0)
                nB_array[i] = temp_nB;

            gsl_spline2d_set(spline_log10_ps_v, log10_ps_array, i , j, temp_log10_ps);

            //LOG_DEBUG("We are here");

        }

        fclose(F);

        int gsl_status = gsl_spline2d_init(spline_log10_ps_v, nB_array, log10_w_array, log10_ps_array, nx, ny);
        GSL_ERROR(gsl_status);

        free(log10_ps_array);
        
        LOG_DEBUG("PMF POWER SPECTRUM TABLES INTERPOLATED");

    }
    else
    {
        gsl_spline2d_free(spline_log10_ps_v);
        gsl_interp_accel_free(acc_w);
        gsl_interp_accel_free(acc_nB);
    }

}


// ------------------------------------------------------ 
// Initialisation routines for Primordial Magnetic Fields

static const double *zB_growth=NULL, *MB=NULL;
static gsl_interp_accel *acc_MB = NULL;
static gsl_spline *spline_MB = NULL;
static TABLE_PMF_GROWTH_EVOL_NPTS;

void prepare_tables_PMF_growth_evolution(int table_length)
{
    zB_growth = malloc(table_length * sizeof(double));
    MB        = malloc(table_length * sizeof(double));
}


void init_spline_PMF_growth_evolution()
{
    // Set up spline table
    acc_MB   = gsl_interp_accel_alloc ();
    spline_MB  = gsl_spline_alloc (gsl_interp_cspline, TABLE_PMF_GROWTH_EVOL_NPTS);
    gsl_spline_init(spline_MB, zB_growth, MB, TABLE_PMF_GROWTH_EVOL_NPTS);
}


// Initialise the tables for the values of chiB from PMF
int InitPMFGrowthEvolutionTablesFromInput(float *z, float *input_MB, int length)
{
    const table_length = length;
    TABLE_PMF_GROWTH_EVOL_NPTS = table_length;

    prepare_tables_PMF_growth_evolution(TABLE_PMF_GROWTH_EVOL_NPTS);

    for (int i = 0; i < table_length; i++)
    {
        *((double *)zB_growth + i) = (double)z[i];
        *((double *)MB + i) = (double)input_MB[i];
    }
    
    init_spline_PMF_growth_evolution();

    return 1;
}

double MB_PMF_TABLE(float z)
{
    double ans;
    if (z > zB_growth[TABLE_PMF_GROWTH_EVOL_NPTS-1]) { // Called at z>500! Bail out
        LOG_ERROR("Called MB_PMF_TABLE with z=%f", z);
        Throw ValueError;
    }
    else { // Do spline
        ans = gsl_spline_eval (spline_MB, z, acc_MB);
    }
    return ans;
}


int destruct_pmf_growth()
{
    LOG_DEBUG("FREEING PMF TABLES");
    
    free_pointer((void**)&spline_MB, free_gsl_spline);
    free_pointer((void**)&acc_MB, free_gsl_interp_accel);
    free_pointer((void**)&zB_growth, free);

    LOG_DEBUG("PMF GROWTH TABLES FREED");
    return 1;
}




// ------------------------------------------------------ 



static const double *kclass=NULL, *Tmclass=NULL, *Tvclass_vcb=NULL, *kclass_LCDM=NULL, *Tmclass_LCDM=NULL, *Tvclass_vcb_LCDM=NULL;
static gsl_interp_accel *acc_density = NULL, *acc_vcb = NULL, *acc_density_LCDM = NULL, *acc_vcb_LCDM = NULL;
static gsl_spline *spline_density = NULL, *spline_vcb = NULL, *spline_density_LCDM = NULL, *spline_vcb_LCDM = NULL;
static TABLE_CLASS_LENGTH, TABLE_CLASS_LENGTH_LCDM;

int InitTFCLASS(struct UserParams *user_params, struct CosmoParams *cosmo_params, float *k, float *Tm, float *Tvcb,  float *k_LCDM, float *Tm_LCDM, float *Tvcb_LCDM, int length, int length_LCDM)
{  

    LOG_DEBUG("INITIALISING TF CLASS");

    if (user_params->USE_CLASS_TABLES)
    {
        length = CLASS_LENGTH;
        length_LCDM = CLASS_LENGTH;
    }

    const table_length = length;
    const table_length_LCDM = length_LCDM;
    TABLE_CLASS_LENGTH = table_length;
    TABLE_CLASS_LENGTH_LCDM = table_length_LCDM;

    kclass      = malloc(table_length * sizeof(double));
    Tmclass     = malloc(table_length * sizeof(double));
    Tvclass_vcb = malloc(table_length * sizeof(double));

    kclass_LCDM      = malloc(table_length_LCDM * sizeof(double));
    Tmclass_LCDM     = malloc(table_length_LCDM * sizeof(double));
    Tvclass_vcb_LCDM = malloc(table_length_LCDM * sizeof(double));

    float currk, currTm, currTv;
    int gsl_status, gsl_status_LCDM;


    if (!user_params->USE_CLASS_TABLES)
    {
        for (int i = 0; i < table_length; i++)
        {
            *((double *)kclass + i) = (double)k[i];
            *((double *)Tmclass + i) = (double)Tm[i];
            *((double *)Tvclass_vcb + i) = (double)Tvcb[i];
        }

        for (int i = 0; i < table_length_LCDM; i++)
        {
            *((double *)kclass_LCDM + i) = (double)k_LCDM[i];
            *((double *)Tmclass_LCDM + i) = (double)Tm_LCDM[i];
            *((double *)Tvclass_vcb_LCDM + i) = (double)Tvcb_LCDM[i];
        }
    }
    else
    {
        FILE *F;

        char filename[500];
        sprintf(filename,"%s/%s",global_params.external_table_path, CLASS_FILENAME);

        if (!(F = fopen(filename, "r"))) {
            LOG_ERROR("Unable to open file: %s for reading.", filename);
            Throw(IOError);
        }

        int nscans;
        for (int i = 0; i < table_length; i++) {
            nscans = fscanf(F, "%e %e %e ", &currk, &currTm, &currTv);
            if (nscans != 3) {
                LOG_ERROR("Reading CLASS Transfer Function failed.");
                Throw(IOError);
            }
            *((double *)kclass + i) = (double)currk;
            *((double *)Tmclass + i) = (double)currTm;
            *((double *)Tvclass_vcb + i) = (double) currTv;
            *((double *)kclass_LCDM + i) = (double)currk;
            *((double *)Tmclass_LCDM + i) = (double)currTm;
            *((double *)Tvclass_vcb_LCDM + i) = (double) currTv;
            if (i > 0 && kclass[i] <= kclass[i - 1]) {
                LOG_WARNING("Tk table not ordered");
                LOG_WARNING("k=%.1le kprev=%.1le", kclass[i], kclass[i - 1]);
            }
        }
        fclose(F);

        LOG_SUPER_DEBUG("Read CLASS Transfer file");
    }

    gsl_set_error_handler_off();
    // Set up spline table for densities
    acc_density        = gsl_interp_accel_alloc ();
    acc_density_LCDM   = gsl_interp_accel_alloc ();
    spline_density  = gsl_spline_alloc (gsl_interp_cspline, table_length);
    spline_density_LCDM  = gsl_spline_alloc (gsl_interp_cspline, table_length_LCDM);
    gsl_status = gsl_spline_init(spline_density, kclass, Tmclass, table_length);
    gsl_status_LCDM = gsl_spline_init(spline_density_LCDM, kclass_LCDM, Tmclass_LCDM, table_length_LCDM);
    GSL_ERROR(gsl_status);
    GSL_ERROR(gsl_status_LCDM);

    LOG_SUPER_DEBUG("Generated CLASS Density Spline.");

    //Set up spline table for velocities
    acc_vcb   = gsl_interp_accel_alloc ();
    acc_vcb_LCDM   = gsl_interp_accel_alloc ();
    spline_vcb  = gsl_spline_alloc (gsl_interp_cspline, table_length);
    spline_vcb_LCDM  = gsl_spline_alloc (gsl_interp_cspline, table_length_LCDM);
    gsl_status = gsl_spline_init(spline_vcb, kclass, Tvclass_vcb, table_length);
    gsl_status_LCDM = gsl_spline_init(spline_vcb_LCDM, kclass_LCDM, Tvclass_vcb_LCDM, table_length_LCDM);
    GSL_ERROR(gsl_status);
    GSL_ERROR(gsl_status_LCDM);

    LOG_SUPER_DEBUG("Generated CLASS velocity Spline.");

    return 1;

}


int free_TF_CLASS()
{

    LOG_DEBUG("FREEING TF CLASS POINTERS");

    free_pointer((void**)&spline_density, free_gsl_spline);
    free_pointer((void**)&spline_vcb, free_gsl_spline);
    free_pointer((void**)&spline_density_LCDM, free_gsl_spline);
    free_pointer((void**)&spline_vcb_LCDM, free_gsl_spline);
    
    free_pointer((void**)&acc_density, free_gsl_interp_accel);
    free_pointer((void**)&acc_vcb, free_gsl_interp_accel);
    free_pointer((void**)&acc_density_LCDM, free_gsl_interp_accel);
    free_pointer((void**)&acc_vcb_LCDM, free_gsl_interp_accel);

    free_pointer((void**)&kclass, free);
    free_pointer((void**)&Tmclass, free);
    free_pointer((void**)&Tvclass_vcb, free);
    free_pointer((void**)&kclass_LCDM, free);
    free_pointer((void**)&Tmclass_LCDM, free);
    free_pointer((void**)&Tvclass_vcb_LCDM, free);

    LOG_DEBUG("TF CLASS POINTERS FREED");

    return 1;
}


/*
  this function reads the z=0 matter (CDM+baryons)  and relative velocity transfer functions from CLASS (from a file)
  flag_int = 0 to initialize interpolator, flag_int = -1 to free memory, flag_int = else to interpolate.
  flag_dv = 0 to output density, flag_dv = 1 to output velocity.
  similar to built-in function "double T_RECFAST(float z, int flag)"
*/

double TF_CLASS(double k, int flag_dv)
{
    double ans;

    if (k > kclass[TABLE_CLASS_LENGTH-1]) { // k>kmax
        LOG_SUPER_DEBUG("Called TF_CLASS with k=%f, larger than kmax! Returning value at kmax = %f.", k, kclass[TABLE_CLASS_LENGTH-1]);
        if(flag_dv == 0){ // output is density
            return (Tmclass[TABLE_CLASS_LENGTH]/kclass[TABLE_CLASS_LENGTH-1]/kclass[TABLE_CLASS_LENGTH-1]);
        }
        else if(flag_dv == 1){ // output is rel velocity
            return (Tvclass_vcb[TABLE_CLASS_LENGTH]/kclass[TABLE_CLASS_LENGTH-1]/kclass[TABLE_CLASS_LENGTH-1]);
        }    //we just set it to the last value, since sometimes it wants large k for R<<cell_size, which does not matter much.
    }
    else { // Do spline
        if(flag_dv == 0){ // output is density
            ans = gsl_spline_eval (spline_density, k, acc_density);
        }
        else if(flag_dv == 1){ // output is relative velocity
            ans = gsl_spline_eval (spline_vcb, k, acc_vcb);
        }
        else{
            ans=0.0; //neither densities not velocities?
        }
    }

    return ans/k/k;
    //we have to divide by k^2 to agree with the old-fashioned convention.

}


/*
  this function reads the z=0 matter (CDM+baryons)  and relative velocity transfer functions from CLASS (from a file)
  flag_int = 0 to initialize interpolator, flag_int = -1 to free memory, flag_int = else to interpolate.
  flag_dv = 0 to output density, flag_dv = 1 to output velocity.
  similar to built-in function "double T_RECFAST(float z, int flag)"
*/

double TF_CLASS_LCDM(double k, int flag_dv)
{
    double ans;

    if (k > kclass_LCDM[TABLE_CLASS_LENGTH_LCDM-1]) { // k>kmax
        LOG_WARNING("Called TF_CLASS_LCDM with k=%f, larger than kmax! Returning value at kmax = %f.", k, kclass_LCDM[TABLE_CLASS_LENGTH_LCDM-1]);
        if(flag_dv == 0){ // output is density
            return (Tmclass_LCDM[TABLE_CLASS_LENGTH_LCDM]/kclass_LCDM[TABLE_CLASS_LENGTH_LCDM-1]/kclass_LCDM[TABLE_CLASS_LENGTH_LCDM-1]);
        }
        else if(flag_dv == 1){ // output is rel velocity
            return (Tvclass_vcb_LCDM[TABLE_CLASS_LENGTH_LCDM]/kclass_LCDM[TABLE_CLASS_LENGTH_LCDM-1]/kclass_LCDM[TABLE_CLASS_LENGTH_LCDM-1]);
        }    //we just set it to the last value, since sometimes it wants large k for R<<cell_size, which does not matter much.
    }
    else { // Do spline
        if(flag_dv == 0){ // output is density
            ans = gsl_spline_eval (spline_density_LCDM, k, acc_density_LCDM);
        }
        else if(flag_dv == 1){ // output is relative velocity
            ans = gsl_spline_eval (spline_vcb_LCDM, k, acc_vcb_LCDM);
        }
        else{
            ans=0.0; //neither densities not velocities?
        }
    }

    return ans/k/k;
    //we have to divide by k^2 to agree with the old-fashioned convention.

}


// Returns the value of the linear power spectrum DENSITY (i.e. <|delta_k|^2>/V)
// at a given k mode linearly extrapolated to z=0
double power_in_k(double k)
{
    //LOG_DEBUG("calling power_in_k = %e, %e, %e", sigma_norm, TWOPI*PI*sigma_norm*sigma_norm, power_spectrum(k));
    return power_spectrum(k)*TWOPI*PI*sigma_norm*sigma_norm;
}



/*
  Returns the value of the linear power spectrum of the DM-b relative velocity
  at kinematic decoupling (which we set at zkin=1010)
*/
double power_in_vcb(double k){

    double p, T, gamma, q, aa, bb, cc;

    //only works if using CLASS
    if (user_params_ps->POWER_SPECTRUM == 5){ // CLASS
        T = TF_CLASS(k, 1); //read from CLASS file. flag_int=1 since we have initialized before, flag_vcb=1 for velocity
        p = pow(k, cosmo_params_ps->POWER_INDEX) * T * T;
    }
    else{
        LOG_ERROR("Cannot get P_cb unless using CLASS: %i\n Set USE_RELATIVE_VELOCITIES 0 or use CLASS.\n", user_params_ps->POWER_SPECTRUM);
        Throw(ValueError);
    }

    return p*TWOPI*PI*sigma_norm*sigma_norm;
}


/* Below we compute all quantities related to the matter power spectrum induced by the presence of primordial magnetic fields */

double _int1_pmf_induced_power(double mu, void *params)
{   
    struct parameters_gsl_pmf_induced_power_int_ vals = *(struct parameters_gsl_pmf_induced_power_int_ *)params;

    double v = vals.v;
    double nB = vals.nB;
    double w = vals.w;

    //double y = sqrt(w*w + v*v - 2*v*w*mu);
    //return pow(y, nB) * exp(-2.0*y*y) * (x*x + (x*x - 2*x*x1*mu)*mu*mu);
    return pow(1.0 + v*v - 2.0*v*mu, nB/2.0) * (1.0 + mu*mu * (1.0 - 2*v*mu)) * exp(-4*w*w*v*(v-mu) - 2*w*w) * v;
}

double _int2_pmf_induced_power(double lnv, void *params)
{
    struct parameters_gsl_pmf_induced_power_int_ vals = *(struct parameters_gsl_pmf_induced_power_int_ *)params;

    double nB = vals.nB;
    double w = vals.w;
    double v = exp(lnv);

    struct parameters_gsl_pmf_induced_power_int_ parameters_gsl_pmf_1 = {.v = v, .nB = nB, .w  = w};
    
    gsl_function F;
    F.function = _int1_pmf_induced_power;
    F.params = &parameters_gsl_pmf_1;
    gsl_integration_workspace * w_gsl = gsl_integration_workspace_alloc (1000);
    double rel_tol = 1e-3;
    double result, error;
    double lower_limit = -1.0;
    double upper_limit = 1.0;
    int status;

    gsl_set_error_handler_off();

    status = gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol, 1000, GSL_INTEG_GAUSS61, w_gsl, &result, &error);

    if(status!=0) {
        LOG_ERROR("gsl integration error occured!");
        LOG_ERROR("function argument: lower_limit=%e upper_limit=%e rel_tol=%e result=%e error=%e",lower_limit, upper_limit,rel_tol,result,error);
        LOG_ERROR("data : nB = %e, w = k/kA = %e, v = k1/k = %e", nB, w, v);
        GSL_ERROR(status);
    }

    gsl_integration_workspace_free(w_gsl);

    return result * pow(v, nB+2.0);
    // last exp(lnx1) = x1 is here because we integrate over lnx1 and not x1)
}

/* 
    Returns the matter power spectrum enhencement induced by the presence of primordial magnetic fields
    Output is the dimensionful power spectrum in Mpc^3
*/
double pmf_induced_power_spectrum(double k)
{

    //LOG_DEBUG("Computing the PMF transfer function for k = %e Mpc^{-1}", k);

    double sB = cosmo_params_ps->PMF_SB;
    double sA = cosmo_params_ps->PMF_SIGMA_A;
    double nB  = cosmo_params_ps->PMF_NB;

    if (sB == 0)
        return 0.0;
    
    //double kA_approx = pow(sB0*sB0 / pow(2*PI, 3.0 + nB) / 4.2e+5, -1.0/(5.0 + nB));
    double kA_approx = TWOPI * pow(sB / sA, -2.0/(5.0 + nB));  // in Mpc^{-1}
    double amplitude = pow(TWOPI * sB, 2) / gsl_sf_gamma((nB+3.0)/2.0) * pow(2.0, (nB+3.0)/2.0); // in nG^2 Mpc^{3+nB}


    double dimensionless_power_spectrum_v = 0;
    double log10_w = log10(k/kA_approx);


    /*
    Condition imposed from physical considerations
    if kA ~ 10 - 10^3 Mpc^{-1} there is no need to
    look for the contribution of the PMF below 10^{-5} k_A
    this already corresponds to 10^{-2} Mpc^{-1} at least 
    where PMF are not expected to play a role
    */
    if (log10_w > -5)
    {

        // if in the parameter space where the table is defined use it otherwise the value is recomputed
        if (user_params_ps->USE_PMF_TABLES == true && log10_w < 1.0 && nB >= -3.0 && nB <= 0)  
            dimensionless_power_spectrum_v = pow(10, gsl_spline2d_eval(spline_log10_ps_v, nB, log10_w, acc_nB, acc_w));
        else
        {
            struct parameters_gsl_pmf_induced_power_int_ parameters_gsl_pmf_2 = {.nB = nB, .w  = k/kA_approx};

            gsl_function F;
            F.function = _int2_pmf_induced_power;
            F.params = &parameters_gsl_pmf_2;
            gsl_integration_workspace * w_gsl = gsl_integration_workspace_alloc(1000);
            double rel_tol  = 1e-3; //10.0 * FRACT_FLOAT_ERR;
            double result, error;
            double lower_limit = log(1e-5)-log(pow(10, log10_w));
            double upper_limit = 4.0;
            int status;

            gsl_set_error_handler_off();

            status = gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol, 1000, GSL_INTEG_GAUSS61, w_gsl, &result, &error);

            if(status!=0) {
                LOG_ERROR("gsl integration error occured!");
                LOG_ERROR("function argument: lower_limit=%e upper_limit=%e rel_tol=%e result=%e error=%e",lower_limit, upper_limit,rel_tol,result,error);
                LOG_ERROR("data : nB = %e, k = %e", nB, k);
                GSL_ERROR(status);
            }

            gsl_integration_workspace_free(w_gsl);

            /* Eq. 21 of Adi et al. 2023 [arXiv:2306.11319]
            the expression is devided by the prefactor alpha = f_b/MU_0/rhob_0, A_B^2 and k_A^(7+n_B)
            these prefactors are added below */
            dimensionless_power_spectrum_v =  pow(10, (7.0 + 2*nB)*log10_w) * result;
        }
    }

    double power_spectrum_v = pow(amplitude, 2) * pow(kA_approx, 7.0+2*nB) * dimensionless_power_spectrum_v / pow(4*PI, 2); // in nG^4 / Mpc
    
    double fb     = cosmo_params_ps->OMb / cosmo_params_ps->OMm;
    double rhob_0 = cosmo_params_ps->OMb  * RHOcrit; // in Msun / Mpc^3
    double rhom_0 = cosmo_params_ps->OMm  * RHOcrit; // in Msun / Mpc^3
    double sJ0    = 5.0 / 4.0 * sqrt(MU_0 * G / PI * rhob_0 * rhom_0) * Msun / CMperMPC / CMperMPC; // in nG
    double kJ     = 2.0 * PI * pow(sB / sJ0 , -2.0/(5.0+nB)); // in Mpc^{-1} 

    
    double alpha = fb / rhob_0 / MU_0 /Msun * CMperMPC; // in nG^{-2} Mpc^2 s^{-2}
    return alpha * alpha * power_spectrum_v * pow(GROWTH_FROM_PMF, 2) * pow(1.0  + pow(k / kJ, 2), -2);

}




//
//
// Part 2. 
//
// ------------------------------------- //
// ------------------------------------- //
/* Window function and smoothed variance */
// ------------------------------------- //
// ------------------------------------- //


/* 
    window function(double kR)

    returns the value of window function chosen according 
    to the flag_option PS_FILTER

    Params
    ------
    - kR (double) product of the mode and radius (dimensionless)
*/
double window_function(double kR)
{

    double w;

    if ( (user_params_ps->PS_FILTER == 0) || (sigma_norm < 0) ) // top hat
    { 
        if ( (kR) < 1.0e-4 ){ w = 1.0;} // w converges to 1 as (kR) -> 0
        else { w = 3.0 * (sin(kR)/pow(kR, 3) - cos(kR)/pow(kR, 2));}
    }
    else if (user_params_ps->PS_FILTER == 1)  // sharpK
    {
        kR <= 1 ? w = 1.0 : 0.0; // in practice we do not use that one but rather truncate the integrals

        if (kR == 1)
            w = 0.5;
    }
    else if (user_params_ps->PS_FILTER == 2)  // gaussian of width 1/R
    {
        w = pow(E, -kR*kR/2.0);
    }
    else 
    {
        LOG_ERROR("No such filter: %i. Output is bogus.", user_params_ps->PS_FILTER);
        Throw(ValueError);
    }

    return w;
}

/*
    dsigma_dlnk(double lnk, void *params)

    derivative with respect to ln(k) of the smoothed
    variance of the matter power spectrum    
*/
double dsigma_dlnk(double lnk, void *params){
   
    double k = exp(lnk);
    double p = power_spectrum(k);
    double Radius = *(double *)params;
    double kR = k*Radius;
    double w = window_function(kR);

    return k*k*p*w*w * k; 
}

/*
    dsigma_dk_LCDM(double lnk, void *params)

    derivative with respect to k of the smoothed
    variance of the LCDM matter power spectrum    
*/
double dsigma_dk_LCDM(double k, void *params){
   
    double p = power_spectrum_LCDM(k);
    double Radius = *(double *)params;
    double kR = k*Radius;
    double w = window_function(kR);

    return k*k*p*w*w; 
}

/*
    dsigma_dk_LCDM_Planck18(double lnk, void *params)

    derivative with respect to k of the smoothed
    variance of the LCDM matter power spectrum 
    as defined in Planck18 (used for normalisation)  
*/
double dsigma_dk_LCDM_Planck18(double k, void * params) {
    double ns = 0.9665;
    double Radius = *(double *)params;
    double kR = k*Radius;
    double w = window_function(kR);

    double k0 = 0.05; // reference value of k0 in Mpc^{-1}
    double index = cosmo_params_ps->POWER_INDEX;

    double p = pow(k/k0, index) * pow(transfer_function_LCDM(k), 2);

    return  k*k*p*w*w;
}


/*
    sigma_z0(double M)

    returns the standard deviation of the normalized, density excess (delta(x)) field,
    smoothed on the comoving scale of M (see filter definitions for M<->R conversion).
    The sigma is evaluated at z=0, with the time evolution contained in the dicke(z) factor,
    i.e. sigma(M,z) = sigma_z0(m) * dicke(z)

    normalized so that sigma_z0(M->8/h Mpc) = SIGMA8 in ../Parameter_files/COSMOLOGY.H
    NOTE: volume is normalized to = 1, so this is equvalent to the mass standard deviation

    Params
    ------
    - M (double) is in solar masses

    References: Padmanabhan, pg. 210, eq. 5.107
*/
double sigma_z0(double M){

    double result, error, lower_limit, upper_limit;
    gsl_function F;
    double rel_tol = FRACT_FLOAT_ERR*10; //<- relative tolerance
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);
    double kstart, kend;

    double Radius = MtoR(M);

    // now lets do the integral for sigma and scale it with sigma_norm
    /* 
        we establish a maximum k of KTOP_CLASS~1e3 Mpc-1 and 
        a minimum at kclass[0], ~1e-5 Mpc-1 since the CLASS 
        transfer function has a max! 
    */
    if(user_params_ps->POWER_SPECTRUM == 5)
    {
      kstart = fmax(1.0e-99/Radius, kclass[0]);
      kend = fmin(350.0/Radius, KTOP_CLASS);
    }
    else
    {
        kstart = 1.0e-99/Radius;
        kend = 350.0/Radius;
    }

    // decrease the required precision if including primordial magnetic fields
    if (user_params_ps->PMF_POWER_SPECTRUM)
        rel_tol = 1e-3;

    lower_limit = log(kstart);

    // for a sharp-k window function we truncate the intrgral
    if (user_params_ps->PS_FILTER == 1)  
        upper_limit = log(fmin(kend, 1.0/Radius));
    else
        upper_limit = log(kend);

    F.function = &dsigma_dlnk;
    F.params = &Radius;

    int status;

    gsl_set_error_handler_off();

    status = gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol, 1000, GSL_INTEG_GAUSS61, w, &result, &error);

    if(status!=0) 
    {
        LOG_ERROR("gsl integration error occured!");
        LOG_ERROR("(function argument): lower_limit=%e upper_limit=%e rel_tol=%e result=%e error=%e",lower_limit,upper_limit,rel_tol,result,error);
        LOG_ERROR("data: M=%e",M);
        GSL_ERROR(status);
    }

    gsl_integration_workspace_free(w);

    return sigma_norm * sqrt(result);
}


/*
 FUNCTION dsigmasqdm_z0(M)
 returns  d/dm (sigma^2) (see function sigma), in units of Msun^-1
 */
double dsigmasq_dm(double lnk, void *params)
{
    double dwdr, drdm, kR, w, Radius;
    double k = exp(lnk);

    Radius = *(double *)params;

    // now get the value of the window function
    kR = k * Radius;
    if (user_params_ps->PS_FILTER == 0){ // top hat
        if ( (kR) < 1.0e-4 ){ w = 1.0; }// w converges to 1 as (kR) -> 0
        else { w = 3.0 * (sin(kR)/pow(kR, 3) - cos(kR)/pow(kR, 2));}

        // now do d(w^2)/dm = 2 w dw/dr dr/dm
        if ( (kR) < 1.0e-10 ){  dwdr = 0;}
        else{ dwdr = 9*cos(kR)*k/pow(kR,3) + 3*sin(kR)*(1 - 3/(kR*kR))/(kR*Radius);}
        //3*k*( 3*cos(kR)/pow(kR,3) + sin(kR)*(-3*pow(kR, -4) + 1/(kR*kR)) );}
        //     dwdr = -1e8 * k / (R*1e3);
        drdm = 1.0 / (4.0*PI * cosmo_params_ps->OMm*RHOcrit * Radius*Radius);
    }
    else if (user_params_ps->PS_FILTER == 2){ // gaussian of width 1/R
        w = pow(E, -kR*kR/2.0);
        dwdr = - k*kR * w;
        drdm = 1.0 / (pow(2*PI, 1.5) * cosmo_params_ps->OMm*RHOcrit * 3*Radius*Radius);
    }
    else {
        LOG_ERROR("No such filter: %i. Output is bogus.", user_params_ps->PS_FILTER);
        Throw(ValueError);
    }

//    return k*k*p*2*w*dwdr*drdm * d2fact;
    //LOG_DEBUG("Here : k = %e, res = %e, w = %e", k, k*k*power_spectrum(k)*2*w*dwdr*drdm * k, w);
    return k*k*power_spectrum(k)*2*w*dwdr*drdm * k;
}

double dsigmasqdm_z0(double M){
    double result, error, lower_limit, upper_limit;
    gsl_function F;
    double rel_tol  = FRACT_FLOAT_ERR*10; //<- relative tolerance
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);
    double kstart, kend;


    double Radius = MtoR(M);

    // If the filter is sharp-k we do not need to integrate, the result is analytical
    if (user_params_ps->PS_FILTER == 1) 
        return -  sigma_norm * sigma_norm * pow(Radius, -3) * power_spectrum(1.0/Radius) / (3.0*M);


    // now lets do the integral for sigma and scale it with sigma_norm
    if(user_params_ps->POWER_SPECTRUM == 5){
      kstart = fmax(1.0e-10, kclass[0]);
      kend = fmin(350.0/Radius, KTOP_CLASS);
    }//we establish a maximum k of KTOP_CLASS~1e3 Mpc-1 and a minimum at kclass[0],~1e-5 Mpc-1 since the CLASS transfer function has a max!
    else{
      kstart = 1.0e-10;
      kend = 350.0/Radius;
    }

    if (kstart >= kend)
        return 0;

    lower_limit = log(kstart);
    upper_limit = log(kend);


    F.function = &dsigmasq_dm;
    F.params = &Radius;

    int status;

    gsl_set_error_handler_off();

    status = gsl_integration_qag(&F, lower_limit, upper_limit, 0, rel_tol, 1000, GSL_INTEG_GAUSS61, w, &result, &error);

    if(status!=0) {
        LOG_ERROR("gsl integration error occured!");
        LOG_ERROR("(function argument): lower_limit=%e upper_limit=%e rel_tol=%e result=%e error=%e",lower_limit,upper_limit,rel_tol,result,error);
        LOG_ERROR("data: M=%e",M);
        GSL_ERROR(status);
    }

    gsl_integration_workspace_free (w);

//    return sigma_norm * sigma_norm * result /d2fact;
    return sigma_norm * sigma_norm * result;
}








//
//
// Part 3. 
//
// ------------------------------------- //
// ------------------------------------- //
/* initialisation of the power spectrum  */
// ------------------------------------- //
// ------------------------------------- //


void init_ps(){
    double result, error, lower_limit, upper_limit;
    gsl_function F;
    double rel_tol  = FRACT_FLOAT_ERR*10; //<- relative tolerance
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);
    double kstart, kend, kmax;

    //we start the interpolator if using CLASS:
    //if (user_params_ps->POWER_SPECTRUM == 5){
    //    LOG_DEBUG("Setting CLASS Transfer Function inits.");
    //    InitTFCLASS()
    //}

    if (user_params_ps->PMF_POWER_SPECTRUM)
        interpolate_power_spectrum_from_pmf(false);

    omhh = cosmo_params_ps->OMm*cosmo_params_ps->hlittle*cosmo_params_ps->hlittle;
    theta_cmb = T_cmb / 2.7;

    // Translate Parameters into forms GLOBALVARIABLES form
    f_nu = global_params.OMn/cosmo_params_ps->OMm;
    f_baryon = cosmo_params_ps->OMb/cosmo_params_ps->OMm;
    if (f_nu < TINY) f_nu = 1e-10;
    if (f_baryon < TINY) f_baryon = 1e-10;

    TFset_parameters();

    sigma_norm = -1;
    GROWTH_FROM_PMF = 0;

    if (user_params_ps->PMF_POWER_SPECTRUM)
    {   
        GROWTH_FROM_PMF = MB_PMF_TABLE(global_params.Z_HEAT_MAX) / dicke(global_params.Z_HEAT_MAX);
        LOG_DEBUG("GROWTH_FROM_PMF: %e, (%e old value), unnormalised_D(0) = %e, MB = %e, D(z=35) = %e", GROWTH_FROM_PMF, growth_from_pmf(0.0), unnormalised_dicke(0, cosmo_params_ps->OMm, cosmo_params_ps->OMl), MB_PMF_TABLE(global_params.Z_HEAT_MAX), dicke(global_params.Z_HEAT_MAX));
    }

   
    if (user_params_ps->USE_SIGMA_8_NORM)
    {

        /* 
        Original implementation of the code
        Power spectrum normalised through 
        the variance to SIGMA_8
        */
            
        double Radius_8 = 8.0/cosmo_params_ps->hlittle;

        if(user_params_ps->POWER_SPECTRUM == 5){
            kstart = fmax(1.0e-99/Radius_8, kclass_LCDM[0]);
            kend = fmin(350.0/Radius_8, KTOP_CLASS);
            kmax = fmin(1e+7, KTOP_CLASS);
        }//we establish a maximum k of KTOP_CLASS~1e3 Mpc-1 and a minimum at kclass[0],~1e-5 Mpc-1 since the CLASS transfer function has a max!
        else{
            kstart = 1.0e-99/Radius_8;
            kend = 350.0/Radius_8;
            kmax = 1e+7;
        }

        lower_limit = kstart;
        upper_limit = kend;

        LOG_DEBUG("Initializing Power Spectrum with lower_limit=%e, upper_limit=%e, rel_tol=%e, radius_8=%g", lower_limit, upper_limit, rel_tol, Radius_8);

        F.function = &dsigma_dk_LCDM;        
        F.params = &Radius_8;

        int status;

        gsl_set_error_handler_off();

        status = gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol, 
                            1000, GSL_INTEG_GAUSS61, w, &result, &error);

        if(status!=0) {
            LOG_ERROR("gsl integration error occured!");
            LOG_ERROR("(function argument): lower_limit=%e upper_limit=%e rel_tol=%e result=%e error=%e",lower_limit,upper_limit,rel_tol,result,error);
            GSL_ERROR(status);
        }

        gsl_integration_workspace_free (w);

        sigma_norm = cosmo_params_ps->SIGMA_8/sqrt(result);

    } //takes care of volume factor


    if (!user_params_ps->USE_SIGMA_8_NORM && user_params_ps->POWER_SPECTRUM  != 5) // not CLASS
    {
        /* 
        Here the dimensiofull matter power spectrum is related to the dimensionless power spectrum through
        Pm(k) = 8 pi^2 /25 * (D0 / Omega_{m, 0} / H_0^2)^2 * T^2(k) * k * P_R(k) 
        Therefore one needs to incorporate the variations of the numerical prefactor with the cosomological parameters.
        */

        //double Ln_1010_As_Planck18 = 3.047;
        //double SIGMA_8_Planck18    = 0.8102;
        //double hlittle_Planck18    = 0.6766;
        //double ns_Planck18         = 0.9665;
        //double alpha_s_Planck      = 0.0;
        //double OMm_Planck18        = (0.02242 + 0.11933) / 0.6766 / 0.6766;
        //double OMl_Planck18        = 0.6889;
        //double D0_Planck18 = unnormalised_dicke(0, OMm_Planck18, OMl_Planck18);

        double D0 = unnormalised_dicke(0, cosmo_params_ps->OMm, cosmo_params_ps->OMl);
    
        //LOG_DEBUG("D0, D0_Planck = %e, %e", D0, D0_Planck18);

        //sigma_norm = SIGMA_8_Planck18/sqrt(result) * (D0 / D0_Planck18) * (OMm_Planck18/cosmo_params_ps->OMm) * pow(hlittle_Planck18/cosmo_params_ps->hlittle, 2) * exp((cosmo_params_ps->Ln_1010_As - Ln_1010_As_Planck18)/2.0);
        sigma_norm = sqrt(4.0 / 25.0 * 1e-10) * exp(cosmo_params_ps->Ln_1010_As/2.0) * D0 / cosmo_params_ps->OMm / pow(cosmo_params_ps->hlittle, 2) * pow(2997.92458, 2); // the 2 pi^2 prefactor will be added back when computing power_in_k
    }

    if (!user_params_ps->USE_SIGMA_8_NORM && user_params_ps->POWER_SPECTRUM == 5) // CLASS
    {
        /* 
        Here the dimensiofull matter power spectrum is related to the dimensionless power spectrum through
        Pm(k) = 2*pi^2 * T^2(k) * k * P_R(k) -- different definition of the transfer function
        The numerical "prefactor" (present in the case user_params_ps->POWER_SPECTRUM  != 5) is already included inside the transfer function.
        Therefore when the cosmological parameters are varied, the transfer function is varied accordingly including the variation of the "prefactor"
        No need to vary the prefactos again
        */

       sigma_norm = sqrt(1e-10) * exp(cosmo_params_ps->Ln_1010_As/2.0); // the 2 pi^2 prefactor will be added back when computing power_in_k
       //LOG_DEBUG("initialisation time sigma_norm = %e", sigma_norm);
    }


    LOG_DEBUG("Initialized Power Spectrum.");

}




//function to free arrays related to the power spectrum
void free_ps(){

    // we free the PS interpolator if using PMF
    if (user_params_ps->PMF_POWER_SPECTRUM)
        interpolate_power_spectrum_from_pmf(true);

  return;
}



//
//
// Part 4. 
//
// ---------------//
// -------------- //
/* mass functions */
// -------------- //
// -------------- //



/* sheth correction to delta crit */
double sheth_delc(double del, double sig){
    return sqrt(cosmo_params_ps->SHETH_q)*del*(1. + global_params.SHETH_b*pow(sig*sig/(cosmo_params_ps->SHETH_q*del*del), global_params.SHETH_c));
}


/*
 FUNCTION dNdM_st(z, M)
 Computes the Press_schechter mass function with Sheth-Torman correction for ellipsoidal collapse at
 redshift z, and dark matter halo mass M (in solar masses).

 Uses interpolated sigma and dsigmadm to be computed faster. Necessary for mass-dependent ionising efficiencies.

 The return value is the number density per unit mass of halos in the mass range M to M+dM in units of:
 comoving Mpc^-3 Msun^-1

 Reference: Sheth, Mo, Torman 2001
 */
double dNdM_st(double growthf, double M){

    double sigma, dsigmadm, nuhat, res;
    setFromInterpolationTables(growthf, M, &sigma, &dsigmadm);  
    nuhat = sqrt(cosmo_params_ps->SHETH_q) * Deltac / sigma;

    return (-(cosmo_params_ps->OMm)*RHOcrit/M) * (dsigmadm/sigma) * sqrt(2./PI)*cosmo_params_ps->SHETH_A * (1+ pow(nuhat, -2*cosmo_params_ps->SHETH_p)) * nuhat * pow(E, -nuhat*nuhat/2.0);

    //LOG_DEBUG("In the mass function: %e %e %e %e %e", M, res, sigma, dsigmadm, growthf);
    //LOG_DEBUG("M, dNdM, sigma, dsigmadm: %e, %e, %e, %e", M, res, sigma, dsigmadm);
    //return res;
}

/*
 FUNCTION dNdM_WatsonFOF(z, M)
 Computes the Press_schechter mass function with Warren et al. 2011 correction for ellipsoidal collapse at
 redshift z, and dark matter halo mass M (in solar masses).

 The Universial FOF function (Eq. 12) of Watson et al. 2013

 The return value is the number density per unit mass of halos in the mass range M to M+dM in units of:
 comoving Mpc^-3 Msun^-1

 Reference: Watson et al. 2013
 */
double dNdM_WatsonFOF(double growthf, double M){

    double sigma, dsigmadm, f_sigma;
    setFromInterpolationTables(growthf, M, &sigma, &dsigmadm);  
    f_sigma = Watson_A * ( pow( Watson_beta/sigma, Watson_alpha) + 1. ) * exp( - Watson_gamma/(sigma*sigma) );

    return (-(cosmo_params_ps->OMm)*RHOcrit/M) * (dsigmadm/sigma) * f_sigma;
}

/*
 FUNCTION dNdM_WatsonFOF_z(z, M)
 Computes the Press_schechter mass function with Warren et al. 2011 correction for ellipsoidal collapse at
 redshift z, and dark matter halo mass M (in solar masses).

 The Universial FOF function, with redshift evolution (Eq. 12 - 15) of Watson et al. 2013.

 The return value is the number density per unit mass of halos in the mass range M to M+dM in units of:
 comoving Mpc^-3 Msun^-1

 Reference: Watson et al. 2013
 */
double dNdM_WatsonFOF_z(double z, double growthf, double M){

    double sigma, dsigmadm, A_z, alpha_z, beta_z, Omega_m_z, f_sigma;
    setFromInterpolationTables(growthf, M, &sigma, &dsigmadm);  

    Omega_m_z = (cosmo_params_ps->OMm)*pow(1.+z,3.) / ( (cosmo_params_ps->OMl) + (cosmo_params_ps->OMm)*pow(1.+z,3.) + (global_params.OMr)*pow(1.+z,4.) );

    A_z = Omega_m_z * ( Watson_A_z_1 * pow(1. + z, Watson_A_z_2 ) + Watson_A_z_3 );
    alpha_z = Omega_m_z * ( Watson_alpha_z_1 * pow(1.+z, Watson_alpha_z_2 ) + Watson_alpha_z_3 );
    beta_z = Omega_m_z * ( Watson_beta_z_1 * pow(1.+z, Watson_beta_z_2 ) + Watson_beta_z_3 );

    f_sigma = A_z * ( pow(beta_z/sigma, alpha_z) + 1. ) * exp( - Watson_gamma_z/(sigma*sigma) );

    return (-(cosmo_params_ps->OMm)*RHOcrit/M) * (dsigmadm/sigma) * f_sigma;
}

/*
 FUNCTION dNdM(growthf, M)
 Computes the Press_schechter mass function at
 redshift z (using the growth factor), and dark matter halo mass M (in solar masses).

 Uses interpolated sigma and dsigmadm to be computed faster. Necessary for mass-dependent ionising efficiencies.

 The return value is the number density per unit mass of halos in the mass range M to M+dM in units of:
 comoving Mpc^-3 Msun^-1

 Reference: Padmanabhan, pg. 214
 */
double dNdM(double growthf, double M){
   
    double sigma, dsigmadm, res;
    setFromInterpolationTables(growthf, M, &sigma, &dsigmadm);  

    res = (-(cosmo_params_ps->OMm)*RHOcrit/M) * sqrt(2/PI) * (Deltac/(sigma*sigma)) * dsigmadm * pow(E, -(Deltac*Deltac)/(2*sigma*sigma));

    return res;
}


double sigmaFromInterpolationTables(double M)
{
    if(user_params_ps->USE_INTERPOLATION_TABLES)
    {
        int MassBin = (int)floor( (log(M) - MinMass )*inv_mass_bin_width );
        float MassBinLow = MinMass + mass_bin_width*(float)MassBin;

        return Sigma_InterpTable[MassBin] + ( log(M) - MassBinLow )*( Sigma_InterpTable[MassBin+1] - Sigma_InterpTable[MassBin] )*inv_mass_bin_width;
    }
    
    return sigma_z0(M);
}

double dsigmasqdmFromInterpolationTables(double M)
{
    if(user_params_ps->USE_INTERPOLATION_TABLES)
    {
        int MassBin = (int)floor( (log(M) - MinMass )*inv_mass_bin_width );
        float MassBinLow = MinMass + mass_bin_width*(float)MassBin;

        return -pow(10, dSigmadm_InterpTable[MassBin] + ( log(M) - MassBinLow )*( dSigmadm_InterpTable[MassBin+1] - dSigmadm_InterpTable[MassBin] )*inv_mass_bin_width);

    }

    return dsigmasqdm_z0(M);
}

void setFromInterpolationTables(double growthf, double M, double *sigma, double *dsigmadm)
{
    if(user_params_ps->USE_INTERPOLATION_TABLES)
    {
        int MassBin = (int)floor( (log(M) - MinMass )*inv_mass_bin_width );
        float MassBinLow = MinMass + mass_bin_width*(float)MassBin;

        *sigma = growthf * (Sigma_InterpTable[MassBin] + ( log(M) - MassBinLow )*( Sigma_InterpTable[MassBin+1] - Sigma_InterpTable[MassBin] )*inv_mass_bin_width) ;
        *dsigmadm = - (growthf*growthf/(2.* (*sigma))) * pow(10, dSigmadm_InterpTable[MassBin] + ( log(M) - MassBinLow )*( dSigmadm_InterpTable[MassBin+1] - dSigmadm_InterpTable[MassBin] )*inv_mass_bin_width);
    }
    else
    {
        *sigma =  growthf * sigma_z0(M);
        *dsigmadm = (growthf*growthf/(2.*(* sigma))) * dsigmasqdm_z0(M);
    }

}


/*
 FUNCTION FgtrM(z, M)
 Computes the fraction of mass contained in haloes with mass > M at redshift z
 */
double FgtrM(double z, double M){
    double del, sig;

    del = Deltac/dicke(z); //regular spherical collapse delta
    sig = sigma_z0(M);

    return splined_erfc(del / (sqrt(2)*sig));
}

/*
 FUNCTION FgtrM_wsigma(z, sigma_z0(M))
 Computes the fraction of mass contained in haloes with mass > M at redshift z.
 Requires sigma_z0(M) rather than M to make certain heating integrals faster
 */
double FgtrM_wsigma(double z, double sig){
    double del;

    del = Deltac/dicke(z); //regular spherical collapse delta

    return splined_erfc(del / (sqrt(2)*sig));
}


/*
 FUNCTION FgtrM_Watson(z, M)
 Computes the fraction of mass contained in haloes with mass > M at redshift z
 Uses Watson et al (2013) correction
 */
double dFdlnM_Watson_z (double lnM, void *params){
    struct parameters_gsl_FgtrM_int_ vals = *(struct parameters_gsl_FgtrM_int_ *)params;

    double M = exp(lnM);
    double z = vals.z_obs;
    double growthf = vals.gf_obs;

    return dNdM_WatsonFOF_z(z, growthf, M) * M * M;
}
double FgtrM_Watson_z(double z, double growthf, double M){
    double result, error, lower_limit, upper_limit;
    gsl_function F;
    double rel_tol  = 0.001; //<- relative tolerance
    gsl_integration_workspace * w
    = gsl_integration_workspace_alloc (1000);

    F.function = &dFdlnM_Watson_z;
    struct parameters_gsl_FgtrM_int_ parameters_gsl_FgtrM = {
        .z_obs = z,
        .gf_obs = growthf,
    };

    F.params = &parameters_gsl_FgtrM;
    lower_limit = log(M);
    upper_limit = log(fmax(global_params.M_MAX_INTEGRAL, M*100));

    int status;

    gsl_set_error_handler_off();

    status = gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol,
                         1000, GSL_INTEG_GAUSS61, w, &result, &error);

    if(status!=0) {
        LOG_ERROR("gsl integration error occured!");
        LOG_ERROR("(function argument): lower_limit=%e upper_limit=%e rel_tol=%e result=%e error=%e",lower_limit,upper_limit,rel_tol,result,error);
        LOG_ERROR("data: z=%e growthf=%e M=%e",z,growthf,M);
        GSL_ERROR(status);
    }

    gsl_integration_workspace_free (w);

    return result / (cosmo_params_ps->OMm*RHOcrit);
}


/*
 FUNCTION FgtrM_Watson(z, M)
 Computes the fraction of mass contained in haloes with mass > M at redshift z
 Uses Watson et al (2013) correction
 */
double dFdlnM_Watson (double lnM, void *params){
    double growthf = *(double *)params;
    double M = exp(lnM);
    return dNdM_WatsonFOF(growthf, M) * M * M;
}
double FgtrM_Watson(double growthf, double M){
    double result, error, lower_limit, upper_limit;
    gsl_function F;
    double rel_tol  = 0.001; //<- relative tolerance
    gsl_integration_workspace * w
    = gsl_integration_workspace_alloc (1000);

    F.function = &dFdlnM_Watson;
    F.params = &growthf;
    lower_limit = log(M);
    upper_limit = log(fmax(global_params.M_MAX_INTEGRAL, M*100));

    int status;

    gsl_set_error_handler_off();

    status = gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol,
                         1000, GSL_INTEG_GAUSS61, w, &result, &error);

    if(status!=0) {
        LOG_ERROR("gsl integration error occured!");
        LOG_ERROR("lower_limit=%e upper_limit=%e rel_tol=%e result=%e error=%e",lower_limit,upper_limit,rel_tol,result,error);
        LOG_ERROR("data: growthf=%e M=%e",growthf,M);
        GSL_ERROR(status);
    }

    gsl_integration_workspace_free (w);

    return result / (cosmo_params_ps->OMm*RHOcrit);
}

double dFdlnM_General(double lnM, void *params){
    struct parameters_gsl_FgtrM_int_ vals = *(struct parameters_gsl_FgtrM_int_ *)params;

    double M = exp(lnM);
    double z = vals.z_obs;
    double growthf = vals.gf_obs;

    double MassFunction;

    if(user_params_ps->HMF==0) {
        MassFunction = dNdM(growthf, M);
    }
    if(user_params_ps->HMF==1) {
        MassFunction = dNdM_st(growthf, M);
    }
    if(user_params_ps->HMF==2) {
        MassFunction = dNdM_WatsonFOF(growthf, M);
    }
    if(user_params_ps->HMF==3) {
        MassFunction = dNdM_WatsonFOF_z(z, growthf, M);
    }
    return MassFunction * M * M;
}

/*
 FUNCTION FgtrM_General(z, M)
 Computes the fraction of mass contained in haloes with mass > M at redshift z
 */
double FgtrM_General(double z, double M){

    double del, sig, growthf;
    int status;

    growthf = dicke(z);

    struct parameters_gsl_FgtrM_int_ parameters_gsl_FgtrM = {
        .z_obs = z,
        .gf_obs = growthf,
    };

    if(user_params_ps->HMF<4 && user_params_ps->HMF>-1) {

        double result, error, lower_limit, upper_limit;
        gsl_function F;
        double rel_tol  = 0.001; //<- relative tolerance
        gsl_integration_workspace * w 
        = gsl_integration_workspace_alloc (1000);

        F.function = &dFdlnM_General;
        F.params = &parameters_gsl_FgtrM;

        lower_limit = log(M);
        upper_limit = log(fmax(global_params.M_MAX_INTEGRAL, M*100));

        gsl_set_error_handler_off();

        status = gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol, 1000, GSL_INTEG_GAUSS61, w, &result, &error);

        if(status!=0) {
            LOG_ERROR("gsl integration error occured!");
            LOG_ERROR("lower_limit=%e upper_limit=%e rel_tol=%e result=%e error=%e",lower_limit,upper_limit,rel_tol,result,error);
            LOG_ERROR("data: z=%e growthf=%e M=%e",z,growthf,M);
            GSL_ERROR(status);
        }

        gsl_integration_workspace_free (w);

        return result / (cosmo_params_ps->OMm*RHOcrit);
    }
    else {
        LOG_ERROR("Incorrect HMF selected: %i (should be between 0 and 3).", user_params_ps->HMF);
        Throw(ValueError);
    }
}

double dNion_General(double lnM, void *params){
    struct parameters_gsl_SFR_General_int_ vals = *(struct parameters_gsl_SFR_General_int_ *)params;

    double M = exp(lnM);
    double z = vals.z_obs;
    double growthf = vals.gf_obs;
    double MassTurnover = vals.Mdrop;
    double Alpha_star = vals.pl_star;
    double Alpha_esc = vals.pl_esc;
    double Fstar10 = vals.frac_star;
    double Fesc10 = vals.frac_esc;
    double Mlim_Fstar = vals.LimitMass_Fstar;
    double Mlim_Fesc = vals.LimitMass_Fesc;

    double Fstar, Fesc, MassFunction;

    if (Alpha_star > 0. && M > Mlim_Fstar)
        Fstar = 1./Fstar10;
    else if (Alpha_star < 0. && M < Mlim_Fstar)
        Fstar = 1/Fstar10;
    else
        Fstar = pow(M/1e10,Alpha_star);

    if (Alpha_esc > 0. && M > Mlim_Fesc)
        Fesc = 1./Fesc10;
    else if (Alpha_esc < 0. && M < Mlim_Fesc)
        Fesc = 1./Fesc10;
    else
        Fesc = pow(M/1e10,Alpha_esc);

    if(user_params_ps->HMF==0) {
        MassFunction = dNdM(growthf, M);
    }
    if(user_params_ps->HMF==1) {
        MassFunction = dNdM_st(growthf,M);
    }
    if(user_params_ps->HMF==2) {
        MassFunction = dNdM_WatsonFOF(growthf, M);
    }
    if(user_params_ps->HMF==3) {
        MassFunction = dNdM_WatsonFOF_z(z, growthf, M);
    }

    //LOG_DEBUG("M, dNdM : %e, %e, %d", M, MassFunction, user_params_ps->HMF);

    return MassFunction * M * M * exp(-MassTurnover/M) * Fstar * Fesc;
}

double Nion_General(double z, double M_Min, double MassTurnover, double Alpha_star, double Alpha_esc, double Fstar10, double Fesc10, double Mlim_Fstar, double Mlim_Fesc){

    double growthf;

    growthf = dicke(z);

    double result, error, lower_limit, upper_limit;
    gsl_function F;
    double rel_tol = 0.001; //<- relative tolerance

    gsl_integration_workspace * w
    = gsl_integration_workspace_alloc (1000);

    struct parameters_gsl_SFR_General_int_ parameters_gsl_SFR = {
        .z_obs = z,
        .gf_obs = growthf,
        .Mdrop = MassTurnover,
        .pl_star = Alpha_star,
        .pl_esc = Alpha_esc,
        .frac_star = Fstar10,
        .frac_esc = Fesc10,
        .LimitMass_Fstar = Mlim_Fstar,
        .LimitMass_Fesc = Mlim_Fesc,
    };
    int status;

    if(user_params_ps->HMF<4 && user_params_ps->HMF>-1) {

        F.function = &dNion_General;
        F.params = &parameters_gsl_SFR;

        lower_limit = log(M_Min);
        upper_limit = log(fmax(global_params.M_MAX_INTEGRAL, M_Min*100));

        gsl_set_error_handler_off();

        status = gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol, 1000, GSL_INTEG_GAUSS61, w, &result, &error);

        if(status!=0) {
            LOG_ERROR("gsl integration error occured!");
            LOG_ERROR("(function argument): lower_limit=%e upper_limit=%e rel_tol=%e result=%e error=%e",lower_limit,upper_limit,rel_tol,result,error);
            LOG_ERROR("data: z=%e growthf=%e MassTurnover=%e Alpha_star=%e Alpha_esc=%e",z,growthf,MassTurnover,Alpha_star,Alpha_esc);
            LOG_ERROR("data: Fstar10=%e Fesc10=%e Mlim_Fstar=%e Mlim_Fesc=%e",Fstar10,Fesc10,Mlim_Fstar,Mlim_Fesc);
            LOG_ERROR("Function evaluated at lower-limit: %e",dNion_General(lower_limit,&parameters_gsl_SFR));
            LOG_ERROR("Function evaluated at upper-limit: %e",dNion_General(upper_limit,&parameters_gsl_SFR));
            LOG_ERROR("Mass Function Choice: %d",user_params_ps->HMF);
            LOG_ERROR("Mass Function at min: %e",dNdM_st(growthf, exp(lower_limit)));
            LOG_ERROR("Mass Function at max: %e",dNdM_st(growthf, exp(upper_limit)));
            GSL_ERROR(status);
        }
        gsl_integration_workspace_free (w);

        return result / ((cosmo_params_ps->OMm)*RHOcrit);
    }
    else {
        LOG_ERROR("Incorrect HMF selected: %i (should be between 0 and 3).", user_params_ps->HMF);
        Throw(ValueError);
    }
}

double dNion_General_MINI(double lnM, void *params){
    struct parameters_gsl_SFR_General_int_ vals = *(struct parameters_gsl_SFR_General_int_ *)params;

    double M = exp(lnM);
    double z = vals.z_obs;
    double growthf = vals.gf_obs;
    double MassTurnover = vals.Mdrop;
    double MassTurnover_upper = vals.Mdrop_upper;
    double Alpha_star = vals.pl_star;
    double Alpha_esc = vals.pl_esc;
    double Fstar7_MINI = vals.frac_star;
    double Fesc7_MINI = vals.frac_esc;
    double Mlim_Fstar = vals.LimitMass_Fstar;
    double Mlim_Fesc = vals.LimitMass_Fesc;

    double Fstar, Fesc, MassFunction;

    if (Alpha_star > 0. && M > Mlim_Fstar)
        Fstar = 1./Fstar7_MINI;
    else if (Alpha_star < 0. && M < Mlim_Fstar)
        Fstar = 1/Fstar7_MINI;
    else
        Fstar = pow(M/1e7,Alpha_star);

    if (Alpha_esc > 0. && M > Mlim_Fesc)
        Fesc = 1./Fesc7_MINI;
    else if (Alpha_esc < 0. && M < Mlim_Fesc)
        Fesc = 1./Fesc7_MINI;
    else
        Fesc = pow(M/1e7,Alpha_esc);

    if(user_params_ps->HMF==0) {
        MassFunction = dNdM(growthf, M);
    }
    if(user_params_ps->HMF==1) {
        MassFunction = dNdM_st(growthf,M);
    }
    if(user_params_ps->HMF==2) {
        MassFunction = dNdM_WatsonFOF(growthf, M);
    }
    if(user_params_ps->HMF==3) {
        MassFunction = dNdM_WatsonFOF_z(z, growthf, M);
    }

    return MassFunction * M * M * exp(-MassTurnover/M) * exp(-M/MassTurnover_upper) * Fstar * Fesc;
}

double Nion_General_MINI(double z, double M_Min, double MassTurnover, double MassTurnover_upper, double Alpha_star, double Alpha_esc, double Fstar7_MINI, double Fesc7_MINI, double Mlim_Fstar, double Mlim_Fesc){

    double growthf;
    int status;

    growthf = dicke(z);

    double result, error, lower_limit, upper_limit;
    gsl_function F;
    double rel_tol = 0.001; //<- relative tolerance

    gsl_integration_workspace * w
    = gsl_integration_workspace_alloc (1000);

    struct parameters_gsl_SFR_General_int_ parameters_gsl_SFR = {
        .z_obs = z,
        .gf_obs = growthf,
        .Mdrop = MassTurnover,
        .Mdrop_upper = MassTurnover_upper,
        .pl_star = Alpha_star,
        .pl_esc = Alpha_esc,
        .frac_star = Fstar7_MINI,
        .frac_esc = Fesc7_MINI,
        .LimitMass_Fstar = Mlim_Fstar,
        .LimitMass_Fesc = Mlim_Fesc,
    };

    if(user_params_ps->HMF<4 && user_params_ps->HMF>-1) {

        F.function = &dNion_General_MINI;
        F.params = &parameters_gsl_SFR;

        lower_limit = log(M_Min);
        upper_limit = log(fmax(global_params.M_MAX_INTEGRAL, M_Min*100));

        gsl_set_error_handler_off();

        status = gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol, 1000, GSL_INTEG_GAUSS61, w, &result, &error);

        if(status!=0) {
            LOG_ERROR("gsl integration error occurred!");
            LOG_ERROR("lower_limit=%e upper_limit=%e rel_tol=%e result=%e error=%e",lower_limit,upper_limit,rel_tol,result,error);
            LOG_ERROR("data: z=%e growthf=%e MassTurnover=%e MassTurnover_upper=%e",z,growthf,MassTurnover,MassTurnover_upper);
            LOG_ERROR("data: Alpha_star=%e Alpha_esc=%e Fstar7_MINI=%e Fesc7_MINI=%e Mlim_Fstar=%e Mlim_Fesc=%e",Alpha_star,Alpha_esc,Fstar7_MINI,Fesc7_MINI,Mlim_Fstar,Mlim_Fesc);
            GSL_ERROR(status);
        }

        gsl_integration_workspace_free (w);

        return result / ((cosmo_params_ps->OMm)*RHOcrit);
    }
    else {
        LOG_ERROR("Incorrect HMF selected: %i (should be between 0 and 3).", user_params_ps->HMF);
        Throw(ValueError);
    }
}




float erfcc(float x)
{
    double t,q,ans;

    q=fabs(x);
    t=1.0/(1.0+0.5*q);
    ans=t*exp(-q*q-1.2655122+t*(1.0000237+t*(0.374092+t*(0.0967842+
                                                         t*(-0.1862881+t*(0.2788681+t*(-1.13520398+t*(1.4885159+
                                                                                                      t*(-0.82215223+t*0.17087277)))))))));
    return x >= 0.0 ? ans : 2.0-ans;
}

double splined_erfc(double x){
    if (x < 0){
        return 1.0;
    }

    // TODO: This could be wrapped in a Try/Catch to try the fast way and if it doesn't
    // work, use the slow way.
    return erfcc(x); // the interpolation below doesn't seem to be stable in Ts.c
    if (x > ERFC_PARAM_DELTA*(ERFC_NPTS-1))
        return erfcc(x);
    else
        return exp(gsl_spline_eval(erfc_spline, x, erfc_acc));
}

void gauleg(float x1, float x2, float x[], float w[], int n)
//Given the lower and upper limits of integration x1 and x2, and given n, this routine returns arrays x[1..n] and w[1..n] of length n,
//containing the abscissas and weights of the Gauss- Legendre n-point quadrature formula.
{

    int m,j,i;
    double z1,z,xm,xl,pp,p3,p2,p1;

    m=(n+1)/2;
    xm=0.5*(x2+x1);
    xl=0.5*(x2-x1);
    for (i=1;i<=m;i++) {
        //High precision is a good idea for this routine.
        //The roots are symmetric in the interval, so we only have to find half of them.
        //Loop over the desired roots.

        z=cos(3.141592654*(i-0.25)/(n+0.5));

        //Starting with the above approximation to the ith root, we enter the main loop of refinement by Newton’s method.
        do {
            p1=1.0;
            p2=0.0;
            for (j=1;j<=n;j++) {
                //Loop up the recurrence relation to get the Legendre polynomial evaluated at z.
                p3=p2;
                p2=p1;
                p1=((2.0*j-1.0)*z*p2-(j-1.0)*p3)/j;
            }
            //p1 is now the desired Legendre polynomial. We next compute pp, its derivative, by a standard relation involving also p2,
            //the polynomial of one lower order.
            pp=n*(z*p1-p2)/(z*z-1.0);
            z1=z;
            z=z1-p1/pp;
        } while (fabs(z-z1) > EPS2);
        x[i]=xm-xl*z;
        x[n+1-i]=xm+xl*z;
        w[i]=2.0*xl/((1.0-z*z)*pp*pp);
        w[n+1-i]=w[i];
    }
}

void initialiseSigmaMInterpTable(float M_Min, float M_Max)
{
    int i;
    float Mass;

    if (Mass_InterpTable == NULL){
      Mass_InterpTable = calloc(NMass,sizeof(float));
      Sigma_InterpTable = calloc(NMass,sizeof(float));
      dSigmadm_InterpTable = calloc(NMass,sizeof(float));
    }

#pragma omp parallel shared(Mass_InterpTable,Sigma_InterpTable,dSigmadm_InterpTable) private(i) num_threads(user_params_ps->N_THREADS)
    {
#pragma omp for
        for(i=0;i<NMass;i++) {
            Mass_InterpTable[i] = log(M_Min) + (float)i/(NMass-1)*( log(M_Max) - log(M_Min) );
            Sigma_InterpTable[i] = sigma_z0(exp(Mass_InterpTable[i]));
            dSigmadm_InterpTable[i] = log10(-dsigmasqdm_z0(exp(Mass_InterpTable[i])));
        }
    }

    for(i=0;i<NMass;i++) {
        if(isfinite(Mass_InterpTable[i]) == 0 || isfinite(Sigma_InterpTable[i]) == 0 || isfinite(dSigmadm_InterpTable[i])==0) {
            LOG_ERROR("Detected either an infinite or NaN value in initialiseSigmaMInterpTable");
//            Throw(ParameterError);
            Throw(TableGenerationError);
        }
    }

    MinMass = log(M_Min);
    mass_bin_width = 1./(NMass-1)*( log(M_Max) - log(M_Min) );
    inv_mass_bin_width = 1./mass_bin_width;
}


void freeSigmaMInterpTable()
{
    
    free_pointer((void**)&Mass_InterpTable, free);
    free_pointer((void**)&Sigma_InterpTable, free);
    free_pointer((void**)&dSigmadm_InterpTable, free);
    //Mass_InterpTable = NULL;
}


void nrerror(char error_text[])
{
    LOG_ERROR("Numerical Recipes run-time error...");
    LOG_ERROR("%s",error_text);
    Throw(MemoryAllocError);
}

float *vector(long nl, long nh)
/* allocate a float vector with subscript range v[nl..nh] */
{
    float *v;
    v = (float *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(float)));
    if(!v) nrerror("allocation failure in vector()");
    return v - nl + NR_END;
}

void free_vector(float *v, long nl, long nh)
/* free a float vector allocated with vector() */
{
    free((FREE_ARG) (v+nl-NR_END));
}

void spline(float x[], float y[], int n, float yp1, float ypn, float y2[])
/*Given arrays x[1..n] and y[1..n] containing a tabulated function, i.e., yi = f(xi), with
 x1 <x2 < :: : < xN, and given values yp1 and ypn for the first derivative of the interpolating
 function at points 1 and n, respectively, this routine returns an array y2[1..n] that contains
 the second derivatives of the interpolating function at the tabulated points xi. If yp1 and/or
 ypn are equal to 1e30 or larger, the routine is signaled to set the corresponding boundary
 condition for a natural spline, with zero second derivative on that boundary.*/
{
    int i,k;
    float p,qn,sig,un,*u;
    int na,nb,check;
    u=vector(1,n-1);
    if (yp1 > 0.99e30)                     // The lower boundary condition is set either to be "natural"
        y2[1]=u[1]=0.0;
    else {                                 // or else to have a specified first derivative.
        y2[1] = -0.5;
        u[1]=(3.0/(x[2]-x[1]))*((y[2]-y[1])/(x[2]-x[1])-yp1);
    }
    for (i=2;i<=n-1;i++) {                              //This is the decomposition loop of the tridiagonal algorithm.
        sig=(x[i]-x[i-1])/(x[i+1]-x[i-1]);                //y2 and u are used for temporary
        na = 1;
        nb = 1;
        check = 0;
        while(((float)(x[i+na*1]-x[i-nb*1])==(float)0.0)) {
            check = check + 1;
            if(check%2==0) {
                na = na + 1;
            }
            else {
                nb = nb + 1;
            }
            sig=(x[i]-x[i-1])/(x[i+na*1]-x[i-nb*1]);
        }
        p=sig*y2[i-1]+2.0;                                //storage of the decomposed
        y2[i]=(sig-1.0)/p;                                //  factors.
        u[i]=(y[i+1]-y[i])/(x[i+1]-x[i]) - (y[i]-y[i-1])/(x[i]-x[i-1]);
        u[i]=(6.0*u[i]/(x[i+1]-x[i-1])-sig*u[i-1])/p;

        if(((float)(x[i+1]-x[i])==(float)0.0) || ((float)(x[i]-x[i-1])==(float)0.0)) {
            na = 0;
            nb = 0;
            check = 0;
            while((float)(x[i+na*1]-x[i-nb])==(float)(0.0) || ((float)(x[i+na]-x[i-nb*1])==(float)0.0)) {
                check = check + 1;
                if(check%2==0) {
                    na = na + 1;
                }
                else {
                    nb = nb + 1;
                }
            }
            u[i]=(y[i+1]-y[i])/(x[i+na*1]-x[i-nb]) - (y[i]-y[i-1])/(x[i+na]-x[i-nb*1]);

            u[i]=(6.0*u[i]/(x[i+na*1]-x[i-nb*1])-sig*u[i-1])/p;

        }
    }
    if (ypn > 0.99e30)                        //The upper boundary condition is set either to be "natural"
        qn=un=0.0;
    else {                                    //or else to have a specified first derivative.
        qn=0.5;
        un=(3.0/(x[n]-x[n-1]))*(ypn-(y[n]-y[n-1])/(x[n]-x[n-1]));
    }
    y2[n]=(un-qn*u[n-1])/(qn*y2[n-1]+1.0);

    for (k=n-1;k>=1;k--) {                      //This is the backsubstitution loop of the tridiagonal
        y2[k]=y2[k]*y2[k+1]+u[k];               //algorithm.
    }
    free_vector(u,1,n-1);
}


void splint(float xa[], float ya[], float y2a[], int n, float x, float *y)
/*Given the arrays xa[1..n] and ya[1..n], which tabulate a function (with the xai's in order),
 and given the array y2a[1..n], which is the output from spline above, and given a value of
 x, this routine returns a cubic-spline interpolated value y.*/
{
    void nrerror(char error_text[]);
    int klo,khi,k;
    float h,b,a;
    klo=1;                                                  // We will find the right place in the table by means of
    khi=n;                                                  //bisection. This is optimal if sequential calls to this
    while (khi-klo > 1) {                                   //routine are at random values of x. If sequential calls
        k=(khi+klo) >> 1;                                     //are in order, and closely spaced, one would do better
        if (xa[k] > x) khi=k;                                 //to store previous values of klo and khi and test if
        else klo=k;                                           //they remain appropriate on the next call.
    }                                                           // klo and khi now bracket the input value of x.
    h=xa[khi]-xa[klo];
    if (h == 0.0) nrerror("Bad xa input to routine splint");    //The xa's must be distinct.
    a=(xa[khi]-x)/h;
    b=(x-xa[klo])/h;                                            //Cubic spline polynomial is now evaluated.
    *y=a*ya[klo]+b*ya[khi]+((a*a*a-a)*y2a[klo]+(b*b*b-b)*y2a[khi])*(h*h)/6.0;
}

unsigned long *lvector(long nl, long nh)
/* allocate an unsigned long vector with subscript range v[nl..nh] */
{
    unsigned long *v;
    v = (unsigned long *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(long)));
    if(!v) nrerror("allocation failure in lvector()");
    return v - nl + NR_END;
}

void free_lvector(unsigned long *v, long nl, long nh)
/* free an unsigned long vector allocated with lvector() */
{
    free((FREE_ARG) (v+nl-NR_END));
}


/* dnbiasdM */
double dnbiasdM(double M, float z, double M_o, float del_o){
    double sigsq, del, sig_one, sig_o;

    if ((M_o-M) < TINY){
        LOG_ERROR("In function dnbiasdM: M must be less than M_o!\nAborting...\n");
        Throw(ValueError);
    }
    del = Deltac/dicke(z) - del_o;
    if (del < 0){
        LOG_ERROR(" In function dnbiasdM: del_o must be less than del_1 = del_crit/dicke(z)!\nAborting...\n");
        Throw(ValueError);
    }

    sig_o = sigma_z0(M_o);
    sig_one = sigma_z0(M);
    sigsq = sig_one*sig_one - sig_o*sig_o;
    return -(RHOcrit*cosmo_params_ps->OMm)/M /sqrt(2*PI) *del*pow(sigsq,-1.5)*pow(E, -0.5*del*del/sigsq)*dsigmasqdm_z0(M);
}

/*
 calculates the fraction of mass contained in haloes with mass > M at redshift z, in regions with a linear overdensity of del_bias, and standard deviation sig_bias
 */
double FgtrM_bias(double z, double M, double del_bias, double sig_bias){
    double del, sig, sigsmallR;

    sigsmallR = sigma_z0(M);

    if (!(sig_bias < sigsmallR)){ // biased region is smaller that halo!
//        fprintf(stderr, "FgtrM_bias: Biased region is smaller than halo!\nResult is bogus.\n");
//        return 0;
        return 0.000001;
    }

    del = Deltac/dicke(z) - del_bias;
    sig = sqrt(sigsmallR*sigsmallR - sig_bias*sig_bias);

    return splined_erfc(del / (sqrt(2)*sig));
}

/* Uses sigma parameters instead of Mass for scale */
double sigmaparam_FgtrM_bias(float z, float sigsmallR, float del_bias, float sig_bias){
    double del, sig;

    if (!(sig_bias < sigsmallR)){ // biased region is smaller that halo!
        //    fprintf(stderr, "local_FgtrM_bias: Biased region is smaller than halo!\nResult is bogus.\n");
        //    return 0;
        return 0.000001;
    }

    del = Deltac/dicke(z) - del_bias;
    sig = sqrt(sigsmallR*sigsmallR - sig_bias*sig_bias);

    return splined_erfc(del / (sqrt(2)*sig));
}

/* redshift derivative of the growth function at z */
double ddicke_dz(double z){
    float dz = 1e-10;
    double omegaM_z, ddickdz, dick_0, x, x_0, domegaMdz;

    return (dicke(z+dz)-dicke(z))/dz;
}





/* compute a mass limit where the stellar baryon fraction and the escape fraction exceed unity */
float Mass_limit (float logM, float PL, float FRAC) {
    return FRAC*pow(pow(10.,logM)/1e10,PL);
}
void bisection(float *x, float xlow, float xup, int *iter){
    *x=(xlow + xup)/2.;
    ++(*iter);
}

float Mass_limit_bisection(float Mmin, float Mmax, float PL, float FRAC){
    int i, iter, max_iter=200;
    float rel_tol=0.001;
    float logMlow, logMupper, x, x1;
    iter = 0;
    logMlow = log10(Mmin);
    logMupper = log10(Mmax);

    if (PL < 0.) {
        if (Mass_limit(logMlow,PL,FRAC) <= 1.) {
            return Mmin;
        }
    }
    else if (PL > 0.) {
        if (Mass_limit(logMupper,PL,FRAC) <= 1.) {
            return Mmax;
        }
    }
    else
        return 0;
    bisection(&x, logMlow, logMupper, &iter);
    do {
        if((Mass_limit(logMlow,PL,FRAC)-1.)*(Mass_limit(x,PL,FRAC)-1.) < 0.)
            logMupper = x;
        else
            logMlow = x;
        bisection(&x1, logMlow, logMupper, &iter);
        if(fabs(x1-x) < rel_tol) {
            return pow(10.,x1);
        }
        x = x1;
    }
    while(iter < max_iter);

    // Got to max_iter without finding a solution.
    LOG_ERROR("Failed to find a mass limit to regulate stellar fraction/escape fraction is between 0 and 1.");
    LOG_ERROR(" The solution does not converge or iterations are not sufficient.");
//    Throw(ParameterError);
    Throw(MassDepZetaError);

    return(0.0);
}

int initialise_ComputeLF(int nbins, struct UserParams *user_params, struct CosmoParams *cosmo_params, struct AstroParams *astro_params, struct FlagOptions *flag_options) {

    Broadcast_struct_global_PS(user_params,cosmo_params);
    Broadcast_struct_global_UF(user_params,cosmo_params);

    lnMhalo_param = calloc(nbins,sizeof(double));
    Muv_param = calloc(nbins,sizeof(double));
    Mhalo_param = calloc(nbins,sizeof(double));

    LF_spline_acc = gsl_interp_accel_alloc();
    LF_spline = gsl_spline_alloc(gsl_interp_cspline, nbins);

    init_ps();

    int status;
    Try initialiseSigmaMInterpTable(0.999*Mhalo_min,1.001*Mhalo_max);
    Catch(status) {
        LOG_ERROR("\t...called from initialise_ComputeLF");
        return(status);
    }

    initialised_ComputeLF = true;
    return(0);
}

void cleanup_ComputeLF(){
    free(lnMhalo_param);
    free(Muv_param);
    free(Mhalo_param);
    gsl_spline_free (LF_spline);
    gsl_interp_accel_free(LF_spline_acc);
    freeSigmaMInterpTable();
    free_TF_CLASS();
	initialised_ComputeLF = 0;
}

int ComputeLF(int nbins, struct UserParams *user_params, struct CosmoParams *cosmo_params, struct AstroParams *astro_params,
               struct FlagOptions *flag_options, int component, int NUM_OF_REDSHIFT_FOR_LF, float *z_LF, float *M_TURNs, double *M_uv_z, double *M_h_z, double *log10phi) {
    /*
        This is an API-level function and thus returns an int status.
    */
    int status;
    Try{ // This try block covers the whole function.
    // This NEEDS to be done every time, because the actual object passed in as
    // user_params, cosmo_params etc. can change on each call, freeing up the memory.
    initialise_ComputeLF(nbins, user_params,cosmo_params,astro_params,flag_options);

    int i,i_z;
    int i_unity, i_smth, mf, nbins_smth=7;
    double  dlnMhalo, lnMhalo_i, SFRparam, Muv_1, Muv_2, dMuvdMhalo;
    double Mhalo_i, lnMhalo_min, lnMhalo_max, lnMhalo_lo, lnMhalo_hi, dlnM, growthf;
    double f_duty_upper, Mcrit_atom;
    float Fstar, Fstar_temp;
    double dndm;
    int gsl_status;

    gsl_set_error_handler_off();
    if (astro_params->ALPHA_STAR < -0.5)
        LOG_WARNING(
            "ALPHA_STAR is %f, which is unphysical value given the observational LFs.\n"\
            "Also, when ALPHA_STAR < -.5, LFs may show a kink. It is recommended to set ALPHA_STAR > -0.5.",
            astro_params->ALPHA_STAR
        );

    mf = user_params_ps->HMF;

    lnMhalo_min = log(Mhalo_min*0.999);
    lnMhalo_max = log(Mhalo_max*1.001);
    dlnMhalo = (lnMhalo_max - lnMhalo_min)/(double)(nbins - 1);

    for (i_z=0; i_z<NUM_OF_REDSHIFT_FOR_LF; i_z++) {

        growthf = dicke(z_LF[i_z]);
        Mcrit_atom = atomic_cooling_threshold(z_LF[i_z]);

        i_unity = -1;
        for (i=0; i<nbins; i++) {
            // generate interpolation arrays
            lnMhalo_param[i] = lnMhalo_min + dlnMhalo*(double)i;
            Mhalo_i = exp(lnMhalo_param[i]);

            if (component == 1)
                Fstar = astro_params->F_STAR10*pow(Mhalo_i/1e10,astro_params->ALPHA_STAR);
            else
                Fstar = astro_params->F_STAR7_MINI*pow(Mhalo_i/1e7,astro_params->ALPHA_STAR_MINI);
            if (Fstar > 1.) Fstar = 1;

            if (i_unity < 0) { // Find the array number at which Fstar crosses unity.
                if (astro_params->ALPHA_STAR > 0.) {
                    if ( (1.- Fstar) < FRACT_FLOAT_ERR ) i_unity = i;
                }
                else if (astro_params->ALPHA_STAR < 0. && i < nbins-1) {
                    if (component == 1)
                        Fstar_temp = astro_params->F_STAR10*pow( exp(lnMhalo_min + dlnMhalo*(double)(i+1))/1e10,astro_params->ALPHA_STAR);
                    else
                        Fstar_temp = astro_params->F_STAR7_MINI*pow( exp(lnMhalo_min + dlnMhalo*(double)(i+1))/1e7,astro_params->ALPHA_STAR_MINI);
                    if (Fstar_temp < 1. && (1.- Fstar) < FRACT_FLOAT_ERR) i_unity = i;
                }
            }

            // parametrization of SFR
            SFRparam = Mhalo_i * cosmo_params->OMb/cosmo_params->OMm * (double)Fstar * (double)(hubble(z_LF[i_z])*SperYR/astro_params->t_STAR); // units of M_solar/year

            Muv_param[i] = 51.63 - 2.5*log10(SFRparam*Luv_over_SFR); // UV magnitude
            // except if Muv value is nan or inf, but avoid error put the value as 10.
            if ( isinf(Muv_param[i]) || isnan(Muv_param[i]) ) Muv_param[i] = 10.;

            M_uv_z[i + i_z*nbins] = Muv_param[i];
        }

        gsl_status = gsl_spline_init(LF_spline, lnMhalo_param, Muv_param, nbins);
        GSL_ERROR(gsl_status);

        lnMhalo_lo = log(Mhalo_min);
        lnMhalo_hi = log(Mhalo_max);
        dlnM = (lnMhalo_hi - lnMhalo_lo)/(double)(nbins - 1);

        // There is a kink on LFs at which Fstar crosses unity. This kink is a numerical artefact caused by the derivate of dMuvdMhalo.
        // Most of the cases the kink doesn't appear in magnitude ranges we are interested (e.g. -22 < Muv < -10). However, for some extreme
        // parameters, it appears. To avoid this kink, we use the interpolation of the derivate in the range where the kink appears.
        // 'i_unity' is the array number at which the kink appears. 'i_unity-3' and 'i_unity+12' are related to the range of interpolation,
        // which is an arbitrary choice.
        // NOTE: This method does NOT work in cases with ALPHA_STAR < -0.5. But, this parameter range is unphysical given that the
        //       observational LFs favour positive ALPHA_STAR in this model.
        // i_smth = 0: calculates LFs without interpolation.
        // i_smth = 1: calculates LFs using interpolation where Fstar crosses unity.
        if (i_unity-3 < 0) i_smth = 0;
        else if (i_unity+12 > nbins-1) i_smth = 0;
        else i_smth = 1;
        if (i_smth == 0) {
            for (i=0; i<nbins; i++) {
                // calculate luminosity function
                lnMhalo_i = lnMhalo_lo + dlnM*(double)i;
                Mhalo_param[i] = exp(lnMhalo_i);

                M_h_z[i + i_z*nbins] = Mhalo_param[i];

                Muv_1 = gsl_spline_eval(LF_spline, lnMhalo_i - delta_lnMhalo, LF_spline_acc);
                Muv_2 = gsl_spline_eval(LF_spline, lnMhalo_i + delta_lnMhalo, LF_spline_acc);

                dMuvdMhalo = (Muv_2 - Muv_1) / (2.*delta_lnMhalo * exp(lnMhalo_i));

                if (component == 1)
                    f_duty_upper = 1.;
                else
                    f_duty_upper = exp(-(Mhalo_param[i]/Mcrit_atom));
                if(mf==0) {
                    log10phi[i + i_z*nbins] = log10( dNdM(growthf, exp(lnMhalo_i)) * exp(-(M_TURNs[i_z]/Mhalo_param[i])) * f_duty_upper / fabs(dMuvdMhalo) );
                }
                else if(mf==1) {
                    log10phi[i + i_z*nbins] = log10( dNdM_st(growthf, exp(lnMhalo_i)) * exp(-(M_TURNs[i_z]/Mhalo_param[i])) * f_duty_upper / fabs(dMuvdMhalo) );
                }
                else if(mf==2) {
                    log10phi[i + i_z*nbins] = log10( dNdM_WatsonFOF(growthf, exp(lnMhalo_i)) * exp(-(M_TURNs[i_z]/Mhalo_param[i])) * f_duty_upper / fabs(dMuvdMhalo) );
                }
                else if(mf==3) {
                    log10phi[i + i_z*nbins] = log10( dNdM_WatsonFOF_z(z_LF[i_z], growthf, exp(lnMhalo_i)) * exp(-(M_TURNs[i_z]/Mhalo_param[i])) * f_duty_upper / fabs(dMuvdMhalo) );
                }
                else{
                    LOG_ERROR("HMF should be between 0-3, got %d", mf);
                    Throw(ValueError);
                }
                if (isinf(log10phi[i + i_z*nbins]) || isnan(log10phi[i + i_z*nbins]) || log10phi[i + i_z*nbins] < -30.)
                    log10phi[i + i_z*nbins] = -30.;
            }
        }
        else {
            lnM_temp = calloc(nbins_smth,sizeof(double));
            deriv_temp = calloc(nbins_smth,sizeof(double));
            deriv = calloc(nbins,sizeof(double));

            for (i=0; i<nbins; i++) {
                // calculate luminosity function
                lnMhalo_i = lnMhalo_lo + dlnM*(double)i;
                Mhalo_param[i] = exp(lnMhalo_i);

                M_h_z[i + i_z*nbins] = Mhalo_param[i];

                Muv_1 = gsl_spline_eval(LF_spline, lnMhalo_i - delta_lnMhalo, LF_spline_acc);
                Muv_2 = gsl_spline_eval(LF_spline, lnMhalo_i + delta_lnMhalo, LF_spline_acc);

                dMuvdMhalo = (Muv_2 - Muv_1) / (2.*delta_lnMhalo * exp(lnMhalo_i));
                deriv[i] = fabs(dMuvdMhalo);
            }

            deriv_spline_acc = gsl_interp_accel_alloc();
            deriv_spline = gsl_spline_alloc(gsl_interp_cspline, nbins_smth);

            // generate interpolation arrays to smooth discontinuity of the derivative causing a kink
            // Note that the number of array elements and the range of interpolation are made by arbitrary choices.
            lnM_temp[0] = lnMhalo_param[i_unity - 3];
            lnM_temp[1] = lnMhalo_param[i_unity - 2];
            lnM_temp[2] = lnMhalo_param[i_unity + 8];
            lnM_temp[3] = lnMhalo_param[i_unity + 9];
            lnM_temp[4] = lnMhalo_param[i_unity + 10];
            lnM_temp[5] = lnMhalo_param[i_unity + 11];
            lnM_temp[6] = lnMhalo_param[i_unity + 12];

            deriv_temp[0] = deriv[i_unity - 3];
            deriv_temp[1] = deriv[i_unity - 2];
            deriv_temp[2] = deriv[i_unity + 8];
            deriv_temp[3] = deriv[i_unity + 9];
            deriv_temp[4] = deriv[i_unity + 10];
            deriv_temp[5] = deriv[i_unity + 11];
            deriv_temp[6] = deriv[i_unity + 12];

            gsl_status = gsl_spline_init(deriv_spline, lnM_temp, deriv_temp, nbins_smth);
            GSL_ERROR(gsl_status);

            for (i=0;i<9;i++){
                deriv[i_unity + i - 1] = gsl_spline_eval(deriv_spline, lnMhalo_param[i_unity + i - 1], deriv_spline_acc);
            }
            for (i=0; i<nbins; i++) {
                if (component == 1)
                    f_duty_upper = 1.;
                else
                    f_duty_upper = exp(-(Mhalo_param[i]/Mcrit_atom));

                if(mf==0)
                    dndm = dNdM(growthf, Mhalo_param[i]);
                else if(mf==1)
                    dndm = dNdM_st(growthf, Mhalo_param[i]);
                else if(mf==2)
                    dndm = dNdM_WatsonFOF(growthf, Mhalo_param[i]);
                else if(mf==3)
                    dndm = dNdM_WatsonFOF_z(z_LF[i_z], growthf, Mhalo_param[i]);
                else{
                    LOG_ERROR("HMF should be between 0-3, got %d", mf);
                    Throw(ValueError);
                }
                log10phi[i + i_z*nbins] = log10(dndm * exp(-(M_TURNs[i_z]/Mhalo_param[i])) * f_duty_upper / deriv[i]);
                if (isinf(log10phi[i + i_z*nbins]) || isnan(log10phi[i + i_z*nbins]) || log10phi[i + i_z*nbins] < -30.)
                    log10phi[i + i_z*nbins] = -30.;
            }
        }
    }

	cleanup_ComputeLF();
    } // End try
    Catch(status){
        return status;
    }
    return(0);

}

void initialiseGL_Nion_Xray(int n, float M_Min, float M_Max){
    //calculates the weightings and the positions for Gauss-Legendre quadrature.
    gauleg(log(M_Min),log(M_Max),xi_SFR_Xray,wi_SFR_Xray,n);
}

float dNdM_conditional(float growthf, float M1, float M2, float delta1, float delta2, float sigma2){

    float sigma1, dsigmadm,dsigma_val;
    float MassBinLow;
    int MassBin;

    if(user_params_ps->USE_INTERPOLATION_TABLES) {
        MassBin = (int)floor( (M1 - MinMass )*inv_mass_bin_width );

        MassBinLow = MinMass + mass_bin_width*(float)MassBin;

        sigma1 = Sigma_InterpTable[MassBin] + ( M1 - MassBinLow )*( Sigma_InterpTable[MassBin+1] - Sigma_InterpTable[MassBin] )*inv_mass_bin_width;

        dsigma_val = dSigmadm_InterpTable[MassBin] + ( M1 - MassBinLow )*( dSigmadm_InterpTable[MassBin+1] - dSigmadm_InterpTable[MassBin] )*inv_mass_bin_width;
        dsigmadm = -pow(10.,dsigma_val);
    }
    else {
        sigma1 = sigma_z0(exp(M1));
        dsigmadm = dsigmasqdm_z0(exp(M1));
    }

    M1 = exp(M1);
    M2 = exp(M2);

    sigma1 = sigma1*sigma1;
    sigma2 = sigma2*sigma2;

    dsigmadm = dsigmadm/(2.0*sigma1); // This is actually sigma1^{2} as calculated above, however, it should just be sigma1. It cancels with the same factor below. Why I have decided to write it like that I don't know!

    if((sigma1 > sigma2)) {

        return -(( delta1 - delta2 )/growthf)*( 2.*sigma1*dsigmadm )*( exp( - ( delta1 - delta2 )*( delta1 - delta2 )/( 2.*growthf*growthf*( sigma1 - sigma2 ) ) ) )/(pow( sigma1 - sigma2, 1.5));
    }
    else if(sigma1==sigma2) {

        return -(( delta1 - delta2 )/growthf)*( 2.*sigma1*dsigmadm )*( exp( - ( delta1 - delta2 )*( delta1 - delta2 )/( 2.*growthf*growthf*( 1.e-6 ) ) ) )/(pow( 1.e-6, 1.5));

    }
    else {
        return 0.;
    }
}

void initialiseGL_Nion(int n, float M_Min, float M_Max){
    //calculates the weightings and the positions for Gauss-Legendre quadrature.
    gauleg(log(M_Min),log(M_Max),xi_SFR,wi_SFR,n);

}


double dNion_ConditionallnM_MINI(double lnM, void *params) {
    struct parameters_gsl_SFR_con_int_ vals = *(struct parameters_gsl_SFR_con_int_ *)params;
    double M = exp(lnM); // linear scale
    double growthf = vals.gf_obs;
    double M2 = vals.Mval; // natural log scale
    double sigma2 = vals.sigma2;
    double del1 = vals.delta1;
    double del2 = vals.delta2;
    double MassTurnover = vals.Mdrop;
    double MassTurnover_upper = vals.Mdrop_upper;
    double Alpha_star = vals.pl_star;
    double Alpha_esc = vals.pl_esc;
    double Fstar7_MINI = vals.frac_star;
    double Fesc7_MINI = vals.frac_esc;
    double Mlim_Fstar = vals.LimitMass_Fstar;
    double Mlim_Fesc = vals.LimitMass_Fesc;

    double Fstar,Fesc;

    if (Alpha_star > 0. && M > Mlim_Fstar)
        Fstar = 1./Fstar7_MINI;
    else if (Alpha_star < 0. && M < Mlim_Fstar)
        Fstar = 1./Fstar7_MINI;
    else
        Fstar = pow(M/1e7,Alpha_star);

    if (Alpha_esc > 0. && M > Mlim_Fesc)
        Fesc = 1./Fesc7_MINI;
    else if (Alpha_esc < 0. && M < Mlim_Fesc)
        Fesc = 1./Fesc7_MINI;
    else
        Fesc = pow(M/1e7,Alpha_esc);

    return M*exp(-MassTurnover/M)*exp(-M/MassTurnover_upper)*Fstar*Fesc*dNdM_conditional(growthf,log(M),M2,del1,del2,sigma2)/sqrt(2.*PI);
}

double dNion_ConditionallnM(double lnM, void *params) {
    struct parameters_gsl_SFR_con_int_ vals = *(struct parameters_gsl_SFR_con_int_ *)params; 
    double M = exp(lnM); // linear scale
    double growthf = vals.gf_obs;
    double M2 = vals.Mval; // natural log scale
    double sigma2 = vals.sigma2;
    double del1 = vals.delta1;
    double del2 = vals.delta2;
    double MassTurnover = vals.Mdrop;
    double Alpha_star = vals.pl_star;
    double Alpha_esc = vals.pl_esc;
    double Fstar10 = vals.frac_star;
    double Fesc10 = vals.frac_esc;
    double Mlim_Fstar = vals.LimitMass_Fstar;
    double Mlim_Fesc = vals.LimitMass_Fesc;

    double Fstar,Fesc;

    if (Alpha_star > 0. && M > Mlim_Fstar)
        Fstar = 1./Fstar10;
    else if (Alpha_star < 0. && M < Mlim_Fstar)
        Fstar = 1./Fstar10;
    else
        Fstar = pow(M/1e10,Alpha_star);

    if (Alpha_esc > 0. && M > Mlim_Fesc)
        Fesc = 1./Fesc10;
    else if (Alpha_esc < 0. && M < Mlim_Fesc)
        Fesc = 1./Fesc10;
    else
        Fesc = pow(M/1e10,Alpha_esc);


    //LOG_DEBUG("values : M = %e, exp(-Mturn/M) = %e, Fstar = %e, Fesc = %e, dNdM = %e", M, exp(-MassTurnover/M), Fstar, Fesc, dNdM_conditional(growthf,log(M),M2,del1,del2,sigma2));

    return M*exp(-MassTurnover/M)*Fstar*Fesc*dNdM_conditional(growthf,log(M),M2,del1,del2,sigma2)/sqrt(2.*PI);
}


double Nion_ConditionalM_MINI(double growthf, double M1, double M2, double sigma2, double delta1, double delta2, double MassTurnover, double MassTurnover_upper, double Alpha_star, double Alpha_esc, double Fstar10, double Fesc10, double Mlim_Fstar, double Mlim_Fesc, bool FAST_FCOLL_TABLES) {


  if (FAST_FCOLL_TABLES) { //JBM: Fast tables. Assume sharp Mturn, not exponential cutoff.

      return GaussLegendreQuad_Nion_MINI(0, 0, (float) growthf, (float) M2, (float) sigma2, (float) delta1, (float) delta2, (float) MassTurnover, (float) MassTurnover_upper, (float) Alpha_star, (float) Alpha_esc, (float) Fstar10, (float) Fesc10, (float) Mlim_Fstar, (float) Mlim_Fesc, FAST_FCOLL_TABLES);
  }
  else{ //standard old code
    double result, error, lower_limit, upper_limit;
    gsl_function F;
    double rel_tol = 0.01; //<- relative tolerance
    gsl_integration_workspace * w
    = gsl_integration_workspace_alloc (1000);

    struct parameters_gsl_SFR_con_int_ parameters_gsl_SFR_con = {
        .gf_obs = growthf,
        .Mval = M2,
        .sigma2 = sigma2,
        .delta1 = delta1,
        .delta2 = delta2,
        .Mdrop = MassTurnover,
        .Mdrop_upper = MassTurnover_upper,
        .pl_star = Alpha_star,
        .pl_esc = Alpha_esc,
        .frac_star = Fstar10,
        .frac_esc = Fesc10,
        .LimitMass_Fstar = Mlim_Fstar,
        .LimitMass_Fesc = Mlim_Fesc
    };
    int status;

    F.function = &dNion_ConditionallnM_MINI;
    F.params = &parameters_gsl_SFR_con;
    lower_limit = M1;
    upper_limit = M2;

    gsl_set_error_handler_off();

    status = gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol,
                         1000, GSL_INTEG_GAUSS61, w, &result, &error);

    if(status!=0) {
        LOG_ERROR("gsl integration error occured!");
        LOG_ERROR("(function argument): lower_limit=%e upper_limit=%e rel_tol=%e result=%e error=%e",lower_limit,upper_limit,rel_tol,result,error);
        LOG_ERROR("data: growthf=%e M2=%e sigma2=%e delta1=%e delta2=%e MassTurnover=%e",growthf,M2,sigma2,delta1,delta2,MassTurnover);
        LOG_ERROR("data: MassTurnover_upper=%e Alpha_star=%e Alpha_esc=%e Fstar10=%e Fesc10=%e Mlim_Fstar=%e Mlim_Fesc=%e",MassTurnover_upper,Alpha_star,Alpha_esc,Fstar10,Fesc10,Mlim_Fstar,Mlim_Fesc);
        GSL_ERROR(status);
    }

    gsl_integration_workspace_free (w);

    if(delta2 > delta1) {
        result = 1.;
        return result;
    }
    else {
        return result;
    }
  }


}




double Nion_ConditionalM(double growthf, double M1, double M2, double sigma2, double delta1, double delta2, double MassTurnover, double Alpha_star, double Alpha_esc, double Fstar10, double Fesc10, double Mlim_Fstar, double Mlim_Fesc, bool FAST_FCOLL_TABLES) {


  if (FAST_FCOLL_TABLES && global_params.USE_FAST_ATOMIC) { //JBM: Fast tables. Assume sharp Mturn, not exponential cutoff.

    return GaussLegendreQuad_Nion(0, 0, (float) growthf, (float) M2, (float) sigma2, (float) delta1, (float) delta2, (float) MassTurnover, (float) Alpha_star, (float) Alpha_esc, (float) Fstar10, (float) Fesc10, (float) Mlim_Fstar, (float) Mlim_Fesc, FAST_FCOLL_TABLES);

  }
  else{ //standard

    if(delta2 > delta1) {
        return 1.;
    }

    double result, error, lower_limit, upper_limit;
    gsl_function F;
    double rel_tol = 0.01; //<- relative tolerance
    gsl_integration_workspace * w 
    = gsl_integration_workspace_alloc (1000);

    struct parameters_gsl_SFR_con_int_ parameters_gsl_SFR_con = {
        .gf_obs = growthf,
        .Mval = M2,
        .sigma2 = sigma2,
        .delta1 = delta1,
        .delta2 = delta2,
        .Mdrop = MassTurnover,
        .pl_star = Alpha_star,
        .pl_esc = Alpha_esc,
        .frac_star = Fstar10,
        .frac_esc = Fesc10,
        .LimitMass_Fstar = Mlim_Fstar,
        .LimitMass_Fesc = Mlim_Fesc
    };

    F.function = &dNion_ConditionallnM;
    F.params = &parameters_gsl_SFR_con;
    lower_limit = M1;
    upper_limit = M2;

    // Gaetan suspsecting a big problem here
    // Limits are M1 and M2 but the integrand takes ln(M) as input 

    int status;

    gsl_set_error_handler_off();

    status = gsl_integration_qag (&F, lower_limit, upper_limit, 0, rel_tol, 
                         1000, GSL_INTEG_GAUSS61, w, &result, &error);

    if(status!=0) {
        LOG_ERROR("gsl integration error occured!");
        LOG_ERROR("(function argument): lower_limit=%e upper_limit=%e rel_tol=%e result=%e error=%e",lower_limit,upper_limit,rel_tol,result,error);
        LOG_ERROR("data: growthf=%e M1=%e M2=%e sigma2=%e delta1=%e delta2=%e",growthf,M1,M2,sigma2,delta1,delta2);
        LOG_ERROR("data: MassTurnover=%e Alpha_star=%e Alpha_esc=%e Fstar10=%e Fesc10=%e Mlim_Fstar=%e Mlim_Fesc=%e",MassTurnover,Alpha_star,Alpha_esc,Fstar10,Fesc10,Mlim_Fstar,Mlim_Fesc);
        GSL_ERROR(status);
    }

    gsl_integration_workspace_free (w);

    return result;

  }

}


float Nion_ConditionallnM_GL_MINI(float lnM, struct parameters_gsl_SFR_con_int_ parameters_gsl_SFR_con){
    float M = exp(lnM);
    float growthf = parameters_gsl_SFR_con.gf_obs;
    float M2 = parameters_gsl_SFR_con.Mval;
    float sigma2 = parameters_gsl_SFR_con.sigma2;
    float del1 = parameters_gsl_SFR_con.delta1;
    float del2 = parameters_gsl_SFR_con.delta2;
    float MassTurnover = parameters_gsl_SFR_con.Mdrop;
    float MassTurnover_upper = parameters_gsl_SFR_con.Mdrop_upper;
    float Alpha_star = parameters_gsl_SFR_con.pl_star;
    float Alpha_esc = parameters_gsl_SFR_con.pl_esc;
    float Fstar7_MINI = parameters_gsl_SFR_con.frac_star;
    float Fesc7_MINI = parameters_gsl_SFR_con.frac_esc;
    float Mlim_Fstar = parameters_gsl_SFR_con.LimitMass_Fstar;
    float Mlim_Fesc = parameters_gsl_SFR_con.LimitMass_Fesc;

    float Fstar,Fesc;

    if (Alpha_star > 0. && M > Mlim_Fstar)
        Fstar = 1./Fstar7_MINI;
    else if (Alpha_star < 0. && M < Mlim_Fstar)
        Fstar = 1./Fstar7_MINI;
    else
        Fstar = pow(M/1e7,Alpha_star);

    if (Alpha_esc > 0. && M > Mlim_Fesc)
        Fesc = 1./Fesc7_MINI;
    else if (Alpha_esc < 0. && M < Mlim_Fesc)
        Fesc = 1./Fesc7_MINI;
    else
        Fesc = pow(M/1e7,Alpha_esc);

    return M*exp(-MassTurnover/M)*exp(-M/MassTurnover_upper)*Fstar*Fesc*dNdM_conditional(growthf,log(M),M2,del1,del2,sigma2)/sqrt(2.*PI);
}

float Nion_ConditionallnM_GL(float lnM, struct parameters_gsl_SFR_con_int_ parameters_gsl_SFR_con){
    float M = exp(lnM);
    float growthf = parameters_gsl_SFR_con.gf_obs;
    float M2 = parameters_gsl_SFR_con.Mval;
    float sigma2 = parameters_gsl_SFR_con.sigma2;
    float del1 = parameters_gsl_SFR_con.delta1;
    float del2 = parameters_gsl_SFR_con.delta2;
    float MassTurnover = parameters_gsl_SFR_con.Mdrop;
    float Alpha_star = parameters_gsl_SFR_con.pl_star;
    float Alpha_esc = parameters_gsl_SFR_con.pl_esc;
    float Fstar10 = parameters_gsl_SFR_con.frac_star;
    float Fesc10 = parameters_gsl_SFR_con.frac_esc;
    float Mlim_Fstar = parameters_gsl_SFR_con.LimitMass_Fstar;
    float Mlim_Fesc = parameters_gsl_SFR_con.LimitMass_Fesc;

    float Fstar,Fesc;

    if (Alpha_star > 0. && M > Mlim_Fstar)
        Fstar = 1./Fstar10;
    else if (Alpha_star < 0. && M < Mlim_Fstar)
        Fstar = 1./Fstar10;
    else
        Fstar = pow(M/1e10,Alpha_star);

    if (Alpha_esc > 0. && M > Mlim_Fesc)
        Fesc = 1./Fesc10;
    else if (Alpha_esc < 0. && M < Mlim_Fesc)
        Fesc = 1./Fesc10;
    else
        Fesc = pow(M/1e10,Alpha_esc);

    return M*exp(-MassTurnover/M)*Fstar*Fesc*dNdM_conditional(growthf,log(M),M2,del1,del2,sigma2)/sqrt(2.*PI);

}



//JBM: Same as above but for minihaloes. Has two cutoffs, lower and upper.
float GaussLegendreQuad_Nion_MINI(int Type, int n, float growthf, float M2, float sigma2, float delta1, float delta2, float MassTurnover, float MassTurnover_upper, float Alpha_star, float Alpha_esc, float Fstar7_MINI, float Fesc7_MINI, float Mlim_Fstar_MINI, float Mlim_Fesc_MINI, bool FAST_FCOLL_TABLES) {

    double result, nu_lower_limit, nu_higher_limit, nupivot;
    int i;

    double integrand, x;
    integrand = 0.;

    struct parameters_gsl_SFR_con_int_ parameters_gsl_SFR_con = {
        .gf_obs = growthf,
        .Mval = M2,
        .sigma2 = sigma2,
        .delta1 = delta1,
        .delta2 = delta2,
        .Mdrop = MassTurnover,
        .Mdrop_upper = MassTurnover_upper,
        .pl_star = Alpha_star,
        .pl_esc = Alpha_esc,
        .frac_star = Fstar7_MINI,
        .frac_esc = Fesc7_MINI,
        .LimitMass_Fstar = Mlim_Fstar_MINI,
        .LimitMass_Fesc = Mlim_Fesc_MINI
    };




    if(delta2 > delta1*0.9999) {
        result = 1.;
        return result;
    }


    if(FAST_FCOLL_TABLES){ //JBM: Fast tables. Assume sharp Mturn, not exponential cutoff.


      if(MassTurnover_upper <= MassTurnover){
        return 1e-40; //in sharp cut it's zero
      }

      double delta_arg = pow( (delta1 - delta2)/growthf , 2.);


      double LogMass=log(MassTurnover);
      int MassBin = (int)floor( (LogMass - MinMass )*inv_mass_bin_width );
      double MassBinLow = MinMass + mass_bin_width*(double)MassBin;
      double sigmaM1 = Sigma_InterpTable[MassBin] + ( LogMass - MassBinLow )*( Sigma_InterpTable[MassBin+1] - Sigma_InterpTable[MassBin] )*inv_mass_bin_width;
      nu_lower_limit = delta_arg/(sigmaM1 * sigmaM1 - sigma2 * sigma2);



      LogMass = log(MassTurnover_upper);
      MassBin = (int)floor( (LogMass - MinMass )*inv_mass_bin_width );
      MassBinLow = MinMass + mass_bin_width*(double)MassBin;
      double sigmaM2 = Sigma_InterpTable[MassBin] + ( LogMass - MassBinLow )*( Sigma_InterpTable[MassBin+1] - Sigma_InterpTable[MassBin] )*inv_mass_bin_width;
      nu_higher_limit = delta_arg/(sigmaM2*sigmaM2-sigma2*sigma2);


      //note we keep nupivot1 just in case very negative delta makes it reach that nu
      LogMass = log(MPIVOT1); //jbm could be done outside and it'd be even faster
      int MassBinpivot = (int)floor( (LogMass - MinMass )*inv_mass_bin_width );
      double MassBinLowpivot = MinMass + mass_bin_width*(double)MassBinpivot;
      double sigmapivot1 = Sigma_InterpTable[MassBinpivot] + ( LogMass - MassBinLowpivot )*( Sigma_InterpTable[MassBinpivot+1] - Sigma_InterpTable[MassBinpivot] )*inv_mass_bin_width;
      double nupivot1 = delta_arg/(sigmapivot1*sigmapivot1); //note, it does not have the sigma2 on purpose.

      LogMass = log(MPIVOT2); //jbm could be done outside and it'd be even faster
      MassBinpivot = (int)floor( (LogMass - MinMass )*inv_mass_bin_width );
      MassBinLowpivot = MinMass + mass_bin_width*(double)MassBinpivot;
      double sigmapivot2 = Sigma_InterpTable[MassBinpivot] + ( LogMass - MassBinLowpivot )*( Sigma_InterpTable[MassBinpivot+1] - Sigma_InterpTable[MassBinpivot] )*inv_mass_bin_width;
      double nupivot2 = delta_arg/(sigmapivot2*sigmapivot2);


      double beta1 = (Alpha_star+Alpha_esc) * AINDEX1 * (0.5); //exponent for Fcollapprox for nu>nupivot1 (large M)
      double beta2 = (Alpha_star+Alpha_esc) * AINDEX2 * (0.5); //exponent for Fcollapprox for nupivot1>nu>nupivot2 (small M)
      double beta3 = (Alpha_star+Alpha_esc) * AINDEX3 * (0.5); //exponent for Fcollapprox for nu<nupivot2 (smallest M)
      //beta2 fixed by continuity.


      // // 3PLs
      double fcollres=0.0;
      double fcollres_high=0.0; //for the higher threshold to subtract


      // re-written for further speedups
      if (nu_higher_limit <= nupivot2){ //if both are below pivot2 don't bother adding and subtracting the high contribution
      fcollres=(Fcollapprox(nu_lower_limit,beta3))*pow(nupivot2,-beta3);
      fcollres_high=(Fcollapprox(nu_higher_limit,beta3))*pow(nupivot2,-beta3);
      }
      else {
      fcollres_high=(Fcollapprox(nu_higher_limit,beta2))*pow(nupivot1,-beta2);
      if (nu_lower_limit > nupivot2){
        fcollres=(Fcollapprox(nu_lower_limit,beta2))*pow(nupivot1,-beta2);
      }
      else {
        fcollres=(Fcollapprox(nupivot2,beta2))*pow(nupivot1,-beta2);
        fcollres+=(Fcollapprox(nu_lower_limit,beta3)-Fcollapprox(nupivot2,beta3) )*pow(nupivot2,-beta3);
      }

      }

      if (fcollres < fcollres_high){
      return 1e-40;
      }
      return (fcollres-fcollres_high);
    }
    else{
      for(i=1; i<(n+1); i++){
          if(Type==1) {
              x = xi_SFR_Xray[i];
              integrand += wi_SFR_Xray[i]*Nion_ConditionallnM_GL_MINI(x,parameters_gsl_SFR_con);
          }
          if(Type==0) {
              x = xi_SFR[i];
              integrand += wi_SFR[i]*Nion_ConditionallnM_GL_MINI(x,parameters_gsl_SFR_con);
          }
      }
      return integrand;
    }
}
//JBM: Added the approximation if user_params->FAST_FCOLL_TABLES==True
float GaussLegendreQuad_Nion(int Type, int n, float growthf, float M2, float sigma2, float delta1, float delta2, float MassTurnover, float Alpha_star, float Alpha_esc, float Fstar10, float Fesc10, float Mlim_Fstar, float Mlim_Fesc, bool FAST_FCOLL_TABLES) {
    //Performs the Gauss-Legendre quadrature.
    int i;

    double result, nu_lower_limit, nupivot;

    if(delta2 > delta1*0.9999) {
      result = 1.;
      return result;
    }


    double integrand, x;
    integrand = 0.;

    struct parameters_gsl_SFR_con_int_ parameters_gsl_SFR_con = {
        .gf_obs = growthf,
        .Mval = M2,
        .sigma2 = sigma2,
        .delta1 = delta1,
        .delta2 = delta2,
        .Mdrop = MassTurnover,
        .pl_star = Alpha_star,
        .pl_esc = Alpha_esc,
        .frac_star = Fstar10,
        .frac_esc = Fesc10,
        .LimitMass_Fstar = Mlim_Fstar,
        .LimitMass_Fesc = Mlim_Fesc
    };

  if (FAST_FCOLL_TABLES && global_params.USE_FAST_ATOMIC){ //JBM: Fast tables. Assume sharp Mturn, not exponential cutoff.


      double delta_arg = pow( (delta1 - delta2)/growthf , 2.0);

      double LogMass=log(MassTurnover);
      int MassBin = (int)floor( (LogMass - MinMass )*inv_mass_bin_width );
      double MassBinLow = MinMass + mass_bin_width*(double)MassBin;
      double sigmaM1 = Sigma_InterpTable[MassBin] + ( LogMass - MassBinLow )*( Sigma_InterpTable[MassBin+1] - Sigma_InterpTable[MassBin] )*inv_mass_bin_width;
      nu_lower_limit = delta_arg/(sigmaM1*sigmaM1-sigma2*sigma2);


      LogMass = log(MPIVOT1); //jbm could be done outside and it'd be even faster
      int MassBinpivot = (int)floor( (LogMass - MinMass )*inv_mass_bin_width );
      double MassBinLowpivot = MinMass + mass_bin_width*(double)MassBinpivot;
      double sigmapivot1 = Sigma_InterpTable[MassBinpivot] + ( LogMass - MassBinLowpivot )*( Sigma_InterpTable[MassBinpivot+1] - Sigma_InterpTable[MassBinpivot] )*inv_mass_bin_width;
      double nupivot1 = delta_arg/(sigmapivot1*sigmapivot1); //note, it does not have the sigma2 on purpose.

      LogMass = log(MPIVOT2); //jbm could be done outside and it'd be even faster
      MassBinpivot = (int)floor( (LogMass - MinMass )*inv_mass_bin_width );
      MassBinLowpivot = MinMass + mass_bin_width*(double)MassBinpivot;
      double sigmapivot2 = Sigma_InterpTable[MassBinpivot] + ( LogMass - MassBinLowpivot )*( Sigma_InterpTable[MassBinpivot+1] - Sigma_InterpTable[MassBinpivot] )*inv_mass_bin_width;
      double nupivot2 = delta_arg/(sigmapivot2*sigmapivot2);


      double beta1 = (Alpha_star+Alpha_esc) * AINDEX1 * (0.5); //exponent for Fcollapprox for nu>nupivot1 (large M)
      double beta2 = (Alpha_star+Alpha_esc) * AINDEX2 * (0.5); //exponent for Fcollapprox for nupivot2<nu<nupivot1 (small M)
      double beta3 = (Alpha_star+Alpha_esc) * AINDEX3 * (0.5); //exponent for Fcollapprox for nu<nupivot2 (smallest M)
    //beta2 fixed by continuity.


      double nucrit_sigma2 = delta_arg*pow(sigma2+1e-10,-2.0); //above this nu sigma2>sigma1, so HMF=0. eps added to avoid infinities




    // // 3PLs
      double fcollres=0.0;
      if(nu_lower_limit >= nucrit_sigma2){ //fully in the flat part of sigma(nu), M^alpha is nu-independent.
        return 1e-40;
      }
      else{ //we subtract the contribution from high nu, since the HMF is set to 0 if sigma2>sigma1
        fcollres -= Fcollapprox(nucrit_sigma2,beta1)*pow(nupivot1,-beta1);
      }

      if(nu_lower_limit >= nupivot1){
        fcollres+=Fcollapprox(nu_lower_limit,beta1)*pow(nupivot1,-beta1);
      }
      else{
        fcollres+=Fcollapprox(nupivot1,beta1)*pow(nupivot1,-beta1);
        if (nu_lower_limit > nupivot2){
          fcollres+=(Fcollapprox(nu_lower_limit,beta2)-Fcollapprox(nupivot1,beta2))*pow(nupivot1,-beta2);
        }
        else {
        fcollres+=(Fcollapprox(nupivot2,beta2)-Fcollapprox(nupivot1,beta2) )*pow(nupivot1,-beta2);
        fcollres+=(Fcollapprox(nu_lower_limit,beta3)-Fcollapprox(nupivot2,beta3) )*pow(nupivot2,-beta3);
        }
      }
      if (fcollres<=0.0){
        LOG_DEBUG("Negative fcoll? fc=%.1le Mt=%.1le \n",fcollres, MassTurnover);
        fcollres=1e-40;
      }
      return fcollres;


    }
    else{
        for(i=1; i<(n+1); i++){
            if(Type==1) {
                x = xi_SFR_Xray[i];
                integrand += wi_SFR_Xray[i]*Nion_ConditionallnM_GL(x,parameters_gsl_SFR_con);
            }
            if(Type==0) {
                x = xi_SFR[i];
                integrand += wi_SFR[i]*Nion_ConditionallnM_GL(x,parameters_gsl_SFR_con);
            }
        }
        return integrand;
    }
}



//JBM: Integral of a power-law times exponential for EPS: \int dnu nu^beta * exp(-nu/2)/sqrt(nu) from numin to infty.
double Fcollapprox (double numin, double beta){
//nu is deltacrit^2/sigma^2, corrected by delta(R) and sigma(R)
  double gg = gsl_sf_gamma_inc(0.5+beta,0.5*numin);
  return gg*pow(2,0.5+beta)*pow(2.0*PI,-0.5);
}

void initialise_Nion_General_spline(float z, float min_density, float max_density, float Mmax, float MassTurnover, float Alpha_star, float Alpha_esc, float Fstar10, float Fesc10, float Mlim_Fstar, float Mlim_Fesc, bool FAST_FCOLL_TABLES){


    float Mmin = MassTurnover/50.;
    double overdense_val, growthf, sigma2;
    double overdense_large_high = Deltac, overdense_large_low = global_params.CRIT_DENS_TRANSITION*0.999;
    double overdense_small_high, overdense_small_low;
    int i;

    float ln_10;

    if(max_density > global_params.CRIT_DENS_TRANSITION*1.001) {
        overdense_small_high = global_params.CRIT_DENS_TRANSITION*1.001;
    }
    else {
        overdense_small_high = max_density;
    }
    overdense_small_low = min_density;

    ln_10 = log(10);

    float MassBinLow;
    int MassBin;

    growthf = dicke(z);

    Mmin = log(Mmin);
    Mmax = log(Mmax);

    MassBin = (int)floor( ( Mmax - MinMass )*inv_mass_bin_width );

    MassBinLow = MinMass + mass_bin_width*(float)MassBin;

    sigma2 = Sigma_InterpTable[MassBin] + ( Mmax - MassBinLow )*( Sigma_InterpTable[MassBin+1] - Sigma_InterpTable[MassBin] )*inv_mass_bin_width;

#pragma omp parallel shared(log10_overdense_spline_SFR,log10_Nion_spline,overdense_small_low,overdense_small_high,growthf,Mmax,sigma2,MassTurnover,Alpha_star,Alpha_esc,Fstar10,Fesc10,Mlim_Fstar,Mlim_Fesc) private(i,overdense_val) num_threads(user_params_ps->N_THREADS)
    {
#pragma omp for
        for (i=0; i<NSFR_low; i++){
            overdense_val = log10(1. + overdense_small_low) + (double)i/((double)NSFR_low-1.)*(log10(1.+overdense_small_high)-log10(1.+overdense_small_low));

            log10_overdense_spline_SFR[i] = overdense_val;
            log10_Nion_spline[i] = GaussLegendreQuad_Nion(0,NGL_SFR,growthf,Mmax,sigma2,Deltac,pow(10.,overdense_val)-1.,MassTurnover,Alpha_star,Alpha_esc,Fstar10,Fesc10,Mlim_Fstar,Mlim_Fesc, FAST_FCOLL_TABLES);
            if(fabs(log10_Nion_spline[i]) < 1e-38) {
                log10_Nion_spline[i] = 1e-38;
            }
            log10_Nion_spline[i] = log10(log10_Nion_spline[i]);

            if(log10_Nion_spline[i] < -40.){
                log10_Nion_spline[i] = -40.;
            }

            log10_Nion_spline[i] *= ln_10;

        }
    }

    for (i=0; i<NSFR_low; i++){
        if(!isfinite(log10_Nion_spline[i])) {
            LOG_ERROR("Detected either an infinite or NaN value in log10_Nion_spline");
//            Throw(ParameterError);
            Throw(TableGenerationError);
        }
    }

#pragma omp parallel shared(Overdense_spline_SFR,Nion_spline,overdense_large_low,overdense_large_high,growthf,Mmin,Mmax,sigma2,MassTurnover,Alpha_star,Alpha_esc,Fstar10,Fesc10,Mlim_Fstar,Mlim_Fesc) private(i) num_threads(user_params_ps->N_THREADS)
    {
#pragma omp for
        for(i=0;i<NSFR_high;i++) {
            Overdense_spline_SFR[i] = overdense_large_low + (float)i/((float)NSFR_high-1.)*(overdense_large_high - overdense_large_low);
            Nion_spline[i] = Nion_ConditionalM(growthf,Mmin,Mmax,sigma2,Deltac,Overdense_spline_SFR[i],MassTurnover,Alpha_star,Alpha_esc,Fstar10,Fesc10,Mlim_Fstar,Mlim_Fesc, FAST_FCOLL_TABLES);

            if(Nion_spline[i]<0.) {
                Nion_spline[i]=pow(10.,-40.0);
            }
        }
    }

    for(i=0;i<NSFR_high;i++) {
        if(!isfinite(Nion_spline[i])) {
            LOG_ERROR("Detected either an infinite or NaN value in log10_Nion_spline");
//            Throw(ParameterError);
            Throw(TableGenerationError);
        }
    }
}

void initialise_Nion_General_spline_MINI(float z, float Mcrit_atom, float min_density, float max_density, float Mmax, float Mmin, float log10Mturn_min, float log10Mturn_max, float log10Mturn_min_MINI, float log10Mturn_max_MINI, float Alpha_star, float Alpha_star_mini, float Alpha_esc, float Fstar10, float Fesc10, float Mlim_Fstar, float Mlim_Fesc, float Fstar7_MINI, float Fesc7_MINI, float Mlim_Fstar_MINI, float Mlim_Fesc_MINI, bool FAST_FCOLL_TABLES){

    double growthf, sigma2;
    double overdense_large_high = Deltac, overdense_large_low = global_params.CRIT_DENS_TRANSITION*0.999;
    double overdense_small_high, overdense_small_low;
    int i,j;

    float ln_10;

    if(max_density > global_params.CRIT_DENS_TRANSITION*1.001) {
        overdense_small_high = global_params.CRIT_DENS_TRANSITION*1.001;
    }
    else {
        overdense_small_high = max_density;
    }
    overdense_small_low = min_density;

    ln_10 = log(10);

    float MassBinLow;
    int MassBin;

    growthf = dicke(z);

    Mmin = log(Mmin);
    Mmax = log(Mmax);

    MassBin = (int)floor( ( Mmax - MinMass )*inv_mass_bin_width );

    MassBinLow = MinMass + mass_bin_width*(float)MassBin;

    sigma2 = Sigma_InterpTable[MassBin] + ( Mmax - MassBinLow )*( Sigma_InterpTable[MassBin+1] - Sigma_InterpTable[MassBin] )*inv_mass_bin_width;

    for (i=0; i<NSFR_low; i++){
        log10_overdense_spline_SFR[i] = log10(1. + overdense_small_low) + (double)i/((double)NSFR_low-1.)*(log10(1.+overdense_small_high)-log10(1.+overdense_small_low));
    }
    for (i=0;i<NSFR_high;i++) {
        Overdense_spline_SFR[i] = overdense_large_low + (float)i/((float)NSFR_high-1.)*(overdense_large_high - overdense_large_low);
    }
    for (i=0;i<NMTURN;i++){
        Mturns[i] = pow(10., log10Mturn_min + (float)i/((float)NMTURN-1.)*(log10Mturn_max-log10Mturn_min));
        Mturns_MINI[i] = pow(10., log10Mturn_min_MINI + (float)i/((float)NMTURN-1.)*(log10Mturn_max_MINI-log10Mturn_min_MINI));
    }

#pragma omp parallel shared(log10_Nion_spline,growthf,Mmax,sigma2,log10_overdense_spline_SFR,Mturns,Mturns_MINI,\
                            Alpha_star,Alpha_star_mini,Alpha_esc,Fstar10,Fesc10,Mlim_Fstar,Mlim_Fesc,ln_10,log10_Nion_spline_MINI,Mcrit_atom,\
                            Fstar7_MINI,Fesc7_MINI,Mlim_Fstar_MINI,Mlim_Fesc_MINI) \
                    private(i,j) num_threads(user_params_ps->N_THREADS)
    {
#pragma omp for
        for (i=0; i<NSFR_low; i++){
            for (j=0; j<NMTURN; j++){
                log10_Nion_spline[i+j*NSFR_low] = log10(GaussLegendreQuad_Nion(0,NGL_SFR,growthf,Mmax,sigma2,Deltac,\
                                                        pow(10.,log10_overdense_spline_SFR[i])-1.,Mturns[j],Alpha_star,\
                                                                Alpha_esc,Fstar10,Fesc10,Mlim_Fstar,Mlim_Fesc, FAST_FCOLL_TABLES));



                if(log10_Nion_spline[i+j*NSFR_low] < -40.){
                    log10_Nion_spline[i+j*NSFR_low] = -40.;
                }

                log10_Nion_spline[i+j*NSFR_low] *= ln_10;

                log10_Nion_spline_MINI[i+j*NSFR_low] = log10(GaussLegendreQuad_Nion_MINI(0,NGL_SFR,growthf,Mmax,sigma2,Deltac,\
                                                        pow(10.,log10_overdense_spline_SFR[i])-1.,Mturns_MINI[j],Mcrit_atom,\
                                                                Alpha_star_mini,Alpha_esc,Fstar7_MINI,Fesc7_MINI,Mlim_Fstar_MINI,Mlim_Fesc_MINI, FAST_FCOLL_TABLES));

                if(log10_Nion_spline_MINI[i+j*NSFR_low] < -40.){
                    log10_Nion_spline_MINI[i+j*NSFR_low] = -40.;
                }

                log10_Nion_spline_MINI[i+j*NSFR_low] *= ln_10;
            }
        }
    }

    for (i=0; i<NSFR_low; i++){
        for (j=0; j<NMTURN; j++){
            if(isfinite(log10_Nion_spline[i+j*NSFR_low])==0) {
                LOG_ERROR("Detected either an infinite or NaN value in log10_Nion_spline");
//                Throw(ParameterError);
                Throw(TableGenerationError);
            }

            if(isfinite(log10_Nion_spline_MINI[i+j*NSFR_low])==0) {
                LOG_ERROR("Detected either an infinite or NaN value in log10_Nion_spline_MINI");
//                Throw(ParameterError);
                Throw(TableGenerationError);
            }
        }
    }


#pragma omp parallel shared(Nion_spline,growthf,Mmin,Mmax,sigma2,Overdense_spline_SFR,Mturns,Alpha_star,Alpha_star_mini,\
                            Alpha_esc,Fstar10,Fesc10,Mlim_Fstar,Mlim_Fesc,Nion_spline_MINI,Mturns_MINI,Mcrit_atom,\
                            Fstar7_MINI,Fesc7_MINI,Mlim_Fstar_MINI,Mlim_Fesc_MINI) \
                    private(i,j) num_threads(user_params_ps->N_THREADS)
    {
#pragma omp for
        for(i=0;i<NSFR_high;i++) {
            for (j=0; j<NMTURN; j++){
                Nion_spline[i+j*NSFR_high] = Nion_ConditionalM(
                    growthf,Mmin,Mmax,sigma2,Deltac,Overdense_spline_SFR[i],
                    Mturns[j],Alpha_star,Alpha_esc,Fstar10,Fesc10,Mlim_Fstar,Mlim_Fesc, FAST_FCOLL_TABLES
                );

                if(Nion_spline[i+j*NSFR_high]<0.) {
                    Nion_spline[i+j*NSFR_high]=pow(10.,-40.0);
                }

                Nion_spline_MINI[i+j*NSFR_high] = Nion_ConditionalM_MINI(
                    growthf,Mmin,Mmax,sigma2,Deltac,Overdense_spline_SFR[i],
                    Mturns_MINI[j],Mcrit_atom,Alpha_star_mini,Alpha_esc,Fstar7_MINI,Fesc7_MINI,
                    Mlim_Fstar_MINI,Mlim_Fesc_MINI, FAST_FCOLL_TABLES
                );


                if(Nion_spline_MINI[i+j*NSFR_high]<0.) {
                    Nion_spline_MINI[i+j*NSFR_high]=pow(10.,-40.0);
                }
            }
        }
    }

    for(i=0;i<NSFR_high;i++) {
        for (j=0; j<NMTURN; j++){
            if(isfinite(Nion_spline[i+j*NSFR_high])==0) {
                LOG_ERROR("Detected either an infinite or NaN value in Nion_spline");
//                Throw(ParameterError);
                Throw(TableGenerationError);
            }

            if(isfinite(Nion_spline_MINI[i+j*NSFR_high])==0) {
               LOG_ERROR("Detected either an infinite or NaN value in Nion_spline_MINI");
//                Throw(ParameterError);
                Throw(TableGenerationError);
            }
        }
    }
}

void initialise_Nion_General_spline_MINI_prev(float z, float Mcrit_atom, float min_density, float max_density, float Mmax, float Mmin, float log10Mturn_min, float log10Mturn_max, float log10Mturn_min_MINI, float log10Mturn_max_MINI, float Alpha_star, float Alpha_star_mini, float Alpha_esc, float Fstar10, float Fesc10, float Mlim_Fstar, float Mlim_Fesc, float Fstar7_MINI, float Fesc7_MINI, float Mlim_Fstar_MINI, float Mlim_Fesc_MINI, bool FAST_FCOLL_TABLES){

    double growthf, sigma2;
    double overdense_large_high = Deltac, overdense_large_low = global_params.CRIT_DENS_TRANSITION*0.999;
    double overdense_small_high, overdense_small_low;
    int i,j;

    float ln_10;

    if(max_density > global_params.CRIT_DENS_TRANSITION*1.001) {
        overdense_small_high = global_params.CRIT_DENS_TRANSITION*1.001;
    }
    else {
        overdense_small_high = max_density;
    }
    overdense_small_low = min_density;

    ln_10 = log(10);

    float MassBinLow;
    int MassBin;

    growthf = dicke(z);

    Mmin = log(Mmin);
    Mmax = log(Mmax);

    MassBin = (int)floor( ( Mmax - MinMass )*inv_mass_bin_width );

    MassBinLow = MinMass + mass_bin_width*(float)MassBin;

    sigma2 = Sigma_InterpTable[MassBin] + ( Mmax - MassBinLow )*( Sigma_InterpTable[MassBin+1] - Sigma_InterpTable[MassBin] )*inv_mass_bin_width;

    for (i=0; i<NSFR_low; i++){
        prev_log10_overdense_spline_SFR[i] = log10(1. + overdense_small_low) + (double)i/((double)NSFR_low-1.)*(log10(1.+overdense_small_high)-log10(1.+overdense_small_low));
    }
    for (i=0;i<NSFR_high;i++) {
        prev_Overdense_spline_SFR[i] = overdense_large_low + (float)i/((float)NSFR_high-1.)*(overdense_large_high - overdense_large_low);
    }
    for (i=0;i<NMTURN;i++){
        Mturns[i] = pow(10., log10Mturn_min + (float)i/((float)NMTURN-1.)*(log10Mturn_max-log10Mturn_min));
        Mturns_MINI[i] = pow(10., log10Mturn_min_MINI + (float)i/((float)NMTURN-1.)*(log10Mturn_max_MINI-log10Mturn_min_MINI));
    }

#pragma omp parallel shared(prev_log10_Nion_spline,growthf,Mmax,sigma2,prev_log10_overdense_spline_SFR,Mturns,Alpha_star,Alpha_star_mini,\
                            Alpha_esc,Fstar10,Fesc10,Mlim_Fstar,Mlim_Fesc,prev_log10_Nion_spline_MINI,Mturns_MINI,Mcrit_atom,\
                            Fstar7_MINI,Fesc7_MINI,Mlim_Fstar_MINI,Mlim_Fesc_MINI) \
                    private(i,j) num_threads(user_params_ps->N_THREADS)
    {
#pragma omp for
        for (i=0; i<NSFR_low; i++){
            for (j=0; j<NMTURN; j++){
                prev_log10_Nion_spline[i+j*NSFR_low] = log10(GaussLegendreQuad_Nion(0,NGL_SFR,growthf,Mmax,sigma2,Deltac,\
                                                            pow(10.,prev_log10_overdense_spline_SFR[i])-1.,Mturns[j],\
                                                            Alpha_star,Alpha_esc,Fstar10,Fesc10,Mlim_Fstar,Mlim_Fesc, FAST_FCOLL_TABLES));

                if(prev_log10_Nion_spline[i+j*NSFR_low] < -40.){
                    prev_log10_Nion_spline[i+j*NSFR_low] = -40.;
                }

                prev_log10_Nion_spline[i+j*NSFR_low] *= ln_10;

                prev_log10_Nion_spline_MINI[i+j*NSFR_low] = log10(GaussLegendreQuad_Nion_MINI(0,NGL_SFR,growthf,Mmax,sigma2,Deltac,\
                                                            pow(10.,prev_log10_overdense_spline_SFR[i])-1.,Mturns_MINI[j],Mcrit_atom,\
                                                            Alpha_star_mini,Alpha_esc,Fstar7_MINI,Fesc7_MINI,Mlim_Fstar_MINI,Mlim_Fesc_MINI, FAST_FCOLL_TABLES));

                if(prev_log10_Nion_spline_MINI[i+j*NSFR_low] < -40.){
                    prev_log10_Nion_spline_MINI[i+j*NSFR_low] = -40.;
                }

                prev_log10_Nion_spline_MINI[i+j*NSFR_low] *= ln_10;
            }
        }
    }

    for (i=0; i<NSFR_low; i++){
        for (j=0; j<NMTURN; j++){
            if(isfinite(prev_log10_Nion_spline[i+j*NSFR_low])==0) {
                LOG_ERROR("Detected either an infinite or NaN value in prev_log10_Nion_spline");
//                Throw(ParameterError);
                Throw(TableGenerationError);
            }

            if(isfinite(prev_log10_Nion_spline_MINI[i+j*NSFR_low])==0) {
                LOG_ERROR("Detected either an infinite or NaN value in prev_log10_Nion_spline_MINI");
//                Throw(ParameterError);
                Throw(TableGenerationError);
            }
        }
    }


#pragma omp parallel shared(prev_Nion_spline,growthf,Mmin,Mmax,sigma2,prev_Overdense_spline_SFR,Mturns,\
                            Alpha_star,Alpha_star_mini,Alpha_esc,Fstar10,Fesc10,Mlim_Fstar,Mlim_Fesc,prev_Nion_spline_MINI,Mturns_MINI,\
                            Mcrit_atom,Fstar7_MINI,Fesc7_MINI,Mlim_Fstar_MINI,Mlim_Fesc_MINI) \
                    private(i,j) num_threads(user_params_ps->N_THREADS)
    {
#pragma omp for
        for(i=0;i<NSFR_high;i++) {
            for (j=0; j<NMTURN; j++){

                prev_Nion_spline[i+j*NSFR_high] = Nion_ConditionalM(growthf,Mmin,Mmax,sigma2,Deltac,prev_Overdense_spline_SFR[i],\
                                                                    Mturns[j],Alpha_star,Alpha_esc,Fstar10,Fesc10,Mlim_Fstar,Mlim_Fesc, FAST_FCOLL_TABLES);

                if(prev_Nion_spline[i+j*NSFR_high]<0.) {
                    prev_Nion_spline[i+j*NSFR_high]=pow(10.,-40.0);
                }


                prev_Nion_spline_MINI[i+j*NSFR_high] = Nion_ConditionalM_MINI(growthf,Mmin,Mmax,sigma2,Deltac,\
                                                                    prev_Overdense_spline_SFR[i],Mturns_MINI[j],Mcrit_atom,Alpha_star_mini,\
                                                                    Alpha_esc,Fstar7_MINI,Fesc7_MINI,Mlim_Fstar_MINI,Mlim_Fesc_MINI, FAST_FCOLL_TABLES);

                if(prev_Nion_spline_MINI[i+j*NSFR_high]<0.) {
                    prev_Nion_spline_MINI[i+j*NSFR_high]=pow(10.,-40.0);
                }


            }
        }
    }

    for(i=0;i<NSFR_high;i++) {
        for (j=0; j<NMTURN; j++){
            if(isfinite(prev_Nion_spline[i+j*NSFR_high])==0) {
                LOG_ERROR("Detected either an infinite or NaN value in prev_Nion_spline");
//                Throw(ParameterError);
                Throw(TableGenerationError);
            }

            if(isfinite(prev_Nion_spline_MINI[i+j*NSFR_high])==0) {
                LOG_ERROR("Detected either an infinite or NaN value in prev_Nion_spline_MINI");
//                Throw(ParameterError);
                Throw(TableGenerationError);
            }
        }
    }
}

void initialise_Nion_Ts_spline(
    int Nbin, float zmin, float zmax, float MassTurn, float Alpha_star, float Alpha_esc,
    float Fstar10, float Fesc10
){
    int i;
    float Mmin = MassTurn/50., Mmax = global_params.M_MAX_INTEGRAL;
    float Mlim_Fstar, Mlim_Fesc;

    if (z_val == NULL){
      z_val = calloc(Nbin,sizeof(double));
      Nion_z_val = calloc(Nbin,sizeof(double));
    }

    Mlim_Fstar = Mass_limit_bisection(Mmin, Mmax, Alpha_star, Fstar10);
    Mlim_Fesc = Mass_limit_bisection(Mmin, Mmax, Alpha_esc, Fesc10);

#pragma omp parallel shared(z_val,Nion_z_val,zmin,zmax, MassTurn, Alpha_star, Alpha_esc, Fstar10, Fesc10, Mlim_Fstar, Mlim_Fesc) private(i) num_threads(user_params_ps->N_THREADS)
    {
#pragma omp for
        for (i=0; i<Nbin; i++){
            z_val[i] = zmin + (double)i/((double)Nbin-1.)*(zmax - zmin);
            Nion_z_val[i] = Nion_General(z_val[i], Mmin, MassTurn, Alpha_star, Alpha_esc, Fstar10, Fesc10, Mlim_Fstar, Mlim_Fesc);
        }
    }

    for (i=0; i<Nbin; i++){
        if(isfinite(Nion_z_val[i])==0) {
            LOG_ERROR("Detected either an infinite or NaN value in Nion_z_val");
//            Throw(ParameterError);
            Throw(TableGenerationError);
        }
    }
}

void initialise_Nion_Ts_spline_MINI(
    int Nbin, float zmin, float zmax, float Alpha_star, float Alpha_star_mini, float Alpha_esc, float Fstar10,
    float Fesc10, float Fstar7_MINI, float Fesc7_MINI
){
    int i,j;
    float Mmin = global_params.M_MIN_INTEGRAL, Mmax = global_params.M_MAX_INTEGRAL;
    float Mlim_Fstar, Mlim_Fesc, Mlim_Fstar_MINI, Mlim_Fesc_MINI, Mcrit_atom_val;

    if (z_val == NULL){
      z_val = calloc(Nbin,sizeof(double));
      Nion_z_val = calloc(Nbin,sizeof(double));
      Nion_z_val_MINI = calloc(Nbin*NMTURN,sizeof(double));
    }

    Mlim_Fstar = Mass_limit_bisection(Mmin, Mmax, Alpha_star, Fstar10);
    Mlim_Fesc = Mass_limit_bisection(Mmin, Mmax, Alpha_esc, Fesc10);
    Mlim_Fstar_MINI = Mass_limit_bisection(Mmin, Mmax, Alpha_star_mini, Fstar7_MINI * pow(1e3, Alpha_star_mini));
    Mlim_Fesc_MINI = Mass_limit_bisection(Mmin, Mmax, Alpha_esc, Fesc7_MINI * pow(1e3, Alpha_esc));
    float MassTurnover[NMTURN];
    for (i=0;i<NMTURN;i++){
        MassTurnover[i] = pow(10., LOG10_MTURN_MIN + (float)i/((float)NMTURN-1.)*(LOG10_MTURN_MAX-LOG10_MTURN_MIN));
    }

#pragma omp parallel shared(z_val,Nion_z_val,Nbin,zmin,zmax,Mmin,Alpha_star,Alpha_star_mini,Alpha_esc,Fstar10,Fesc10,Mlim_Fstar,Mlim_Fesc,\
                            Nion_z_val_MINI,MassTurnover,Fstar7_MINI, Fesc7_MINI, Mlim_Fstar_MINI, Mlim_Fesc_MINI) \
                    private(i,j,Mcrit_atom_val) num_threads(user_params_ps->N_THREADS)
    {
#pragma omp for
        for (i=0; i<Nbin; i++){
            z_val[i] = zmin + (double)i/((double)Nbin-1.)*(zmax - zmin);
            Mcrit_atom_val = atomic_cooling_threshold(z_val[i]);
            Nion_z_val[i] = Nion_General(z_val[i], Mmin, Mcrit_atom_val, Alpha_star, Alpha_esc, Fstar10, Fesc10, Mlim_Fstar, Mlim_Fesc);

            for (j=0; j<NMTURN; j++){
                Nion_z_val_MINI[i+j*Nbin] = Nion_General_MINI(z_val[i], Mmin, MassTurnover[j], Mcrit_atom_val, Alpha_star_mini, Alpha_esc, Fstar7_MINI, Fesc7_MINI, Mlim_Fstar_MINI, Mlim_Fesc_MINI);
            }
        }
    }

    for (i=0; i<Nbin; i++){
        if(isfinite(Nion_z_val[i])==0) {
            i = Nbin;
            LOG_ERROR("Detected either an infinite or NaN value in Nion_z_val");
//            Throw(ParameterError);
            Throw(TableGenerationError);
        }

        for (j=0; j<NMTURN; j++){
            if(isfinite(Nion_z_val_MINI[i+j*Nbin])==0){
                j = NMTURN;
                LOG_ERROR("Detected either an infinite or NaN value in Nion_z_val_MINI");
//                Throw(ParameterError);
                Throw(TableGenerationError);
            }
        }
    }
}


void initialise_SFRD_spline(int Nbin, float zmin, float zmax, float MassTurn, float Alpha_star, float Fstar10){
    int i;
    float Mmin = MassTurn/50., Mmax = global_params.M_MAX_INTEGRAL;
    float Mlim_Fstar;

    if (z_X_val == NULL){
      z_X_val = calloc(Nbin,sizeof(double));
      SFRD_val = calloc(Nbin,sizeof(double));
    }

    Mlim_Fstar = Mass_limit_bisection(Mmin, Mmax, Alpha_star, Fstar10);

#pragma omp parallel shared(z_X_val,SFRD_val,zmin,zmax, MassTurn, Alpha_star, Fstar10, Mlim_Fstar) private(i) num_threads(user_params_ps->N_THREADS)
    {
#pragma omp for
        for (i=0; i<Nbin; i++){
            z_X_val[i] = zmin + (double)i/((double)Nbin-1.)*(zmax - zmin);
            SFRD_val[i] = Nion_General(z_X_val[i], Mmin, MassTurn, Alpha_star, 0., Fstar10, 1.,Mlim_Fstar,0.);
        }
    }

    for (i=0; i<Nbin; i++){
        if(isfinite(SFRD_val[i])==0) {
            LOG_ERROR("Detected either an infinite or NaN value in SFRD_val");
//            Throw(ParameterError);
            Throw(TableGenerationError);
        }
    }
}

void initialise_SFRD_spline_MINI(int Nbin, float zmin, float zmax, float Alpha_star, float Alpha_star_mini, float Fstar10, float Fstar7_MINI){
    int i,j;
    float Mmin = global_params.M_MIN_INTEGRAL, Mmax = global_params.M_MAX_INTEGRAL;
    float Mlim_Fstar, Mlim_Fstar_MINI, Mcrit_atom_val;

    if (z_X_val == NULL){
      z_X_val = calloc(Nbin,sizeof(double));
      SFRD_val = calloc(Nbin,sizeof(double));
      SFRD_val_MINI = calloc(Nbin*NMTURN,sizeof(double));
    }

    Mlim_Fstar = Mass_limit_bisection(Mmin, Mmax, Alpha_star, Fstar10);
    Mlim_Fstar_MINI = Mass_limit_bisection(Mmin, Mmax, Alpha_star_mini, Fstar7_MINI * pow(1e3, Alpha_star_mini));

    float MassTurnover[NMTURN];
    for (i=0;i<NMTURN;i++){
        MassTurnover[i] = pow(10., LOG10_MTURN_MIN + (float)i/((float)NMTURN-1.)*(LOG10_MTURN_MAX-LOG10_MTURN_MIN));
    }

#pragma omp parallel shared(z_X_val,zmin,zmax,Nbin,SFRD_val,Mmin, Alpha_star,Alpha_star_mini,Fstar10,Mlim_Fstar,\
                            SFRD_val_MINI,MassTurnover,Fstar7_MINI,Mlim_Fstar_MINI) \
                    private(i,j,Mcrit_atom_val) num_threads(user_params_ps->N_THREADS)
    {
#pragma omp for
        for (i=0; i<Nbin; i++){
            z_X_val[i] = zmin + (double)i/((double)Nbin-1.)*(zmax - zmin);
            Mcrit_atom_val = atomic_cooling_threshold(z_X_val[i]);
            SFRD_val[i] = Nion_General(z_X_val[i], Mmin, Mcrit_atom_val, Alpha_star, 0., Fstar10, 1.,Mlim_Fstar,0.);

            for (j=0; j<NMTURN; j++){
                SFRD_val_MINI[i+j*Nbin] = Nion_General_MINI(z_X_val[i], Mmin, MassTurnover[j], Mcrit_atom_val, Alpha_star_mini, 0., Fstar7_MINI, 1.,Mlim_Fstar_MINI,0.);
            }
        }
    }

    for (i=0; i<Nbin; i++){
        if(isfinite(SFRD_val[i])==0) {
            i = Nbin;
            LOG_ERROR("Detected either an infinite or NaN value in SFRD_val");
//            Throw(ParameterError);
            Throw(TableGenerationError);
        }

        for (j=0; j<NMTURN; j++){
            if(isfinite(SFRD_val_MINI[i+j*Nbin])==0) {
                j = NMTURN;
                LOG_ERROR("Detected either an infinite or NaN value in SFRD_val_MINI");
//                Throw(ParameterError);
                Throw(TableGenerationError);
            }
        }
    }
}

void initialise_SFRD_Conditional_table(
    int Nfilter, float min_density[], float max_density[], float growthf[], float R[],
    float MassTurnover, float Alpha_star, float Fstar10, bool FAST_FCOLL_TABLES
){

    double overdense_val;
    double overdense_large_high = Deltac, overdense_large_low = global_params.CRIT_DENS_TRANSITION;
    double overdense_small_high, overdense_small_low;

    float Mmin,Mmax,Mlim_Fstar,sigma2;
    int i,j,k,i_tot;

    float ln_10;

    ln_10 = log(10);

    Mmin = MassTurnover/50.;
    Mmax = RtoM(R[Nfilter-1]);
    Mlim_Fstar = Mass_limit_bisection(Mmin, Mmax, Alpha_star, Fstar10);

    Mmin = log(Mmin);

    for (i=0; i<NSFR_high;i++) {
        overdense_high_table[i] = overdense_large_low + (float)i/((float)NSFR_high-1.)*(overdense_large_high - overdense_large_low);
    }

    float MassBinLow;
    int MassBin;

    for (j=0; j < Nfilter; j++) {

        Mmax = RtoM(R[j]);

        initialiseGL_Nion_Xray(NGL_SFR, MassTurnover/50., Mmax);

        Mmax = log(Mmax);
        MassBin = (int)floor( ( Mmax - MinMass )*inv_mass_bin_width );

        MassBinLow = MinMass + mass_bin_width*(float)MassBin;

        sigma2 = Sigma_InterpTable[MassBin] + ( Mmax - MassBinLow )*( Sigma_InterpTable[MassBin+1] - Sigma_InterpTable[MassBin] )*inv_mass_bin_width;

        if(min_density[j]*growthf[j] < -1.) {
            overdense_small_low = -1. + global_params.MIN_DENSITY_LOW_LIMIT;
        }
        else {
            overdense_small_low = min_density[j]*growthf[j];
        }
        overdense_small_high = max_density[j]*growthf[j];
        if(overdense_small_high > global_params.CRIT_DENS_TRANSITION) {
            overdense_small_high = global_params.CRIT_DENS_TRANSITION;
        }

        for (i=0; i<NSFR_low; i++) {
            overdense_val = log10(1. + overdense_small_low) + (float)i/((float)NSFR_low-1.)*(log10(1.+overdense_small_high)-log10(1.+overdense_small_low));
            overdense_low_table[i] = pow(10.,overdense_val);
        }

#pragma omp parallel shared(log10_SFRD_z_low_table,growthf,Mmax,sigma2,overdense_low_table,MassTurnover,Alpha_star,Fstar10,Mlim_Fstar) private(i) num_threads(user_params_ps->N_THREADS)
        {
#pragma omp for
            for (i=0; i<NSFR_low; i++){

                log10_SFRD_z_low_table[j][i] = GaussLegendreQuad_Nion(1,NGL_SFR,growthf[j],Mmax,sigma2,Deltac,overdense_low_table[i]-1.,MassTurnover,Alpha_star,0.,Fstar10,1.,Mlim_Fstar,0., FAST_FCOLL_TABLES);
                if(fabs(log10_SFRD_z_low_table[j][i]) < 1e-38) {
                    log10_SFRD_z_low_table[j][i] = 1e-38;
                }
                log10_SFRD_z_low_table[j][i] = log10(log10_SFRD_z_low_table[j][i]);

                log10_SFRD_z_low_table[j][i] += 10.0;
                log10_SFRD_z_low_table[j][i] *= ln_10;
            }
        }

        for (i=0; i<NSFR_low; i++){
            if(isfinite(log10_SFRD_z_low_table[j][i])==0) {
                LOG_ERROR("Detected either an infinite or NaN value in log10_SFRD_z_low_table");
//                Throw(ParameterError);
                Throw(TableGenerationError);
            }
        }

#pragma omp parallel shared(SFRD_z_high_table,growthf,Mmin,Mmax,sigma2,overdense_high_table,MassTurnover,Alpha_star,Fstar10,Mlim_Fstar) private(i) num_threads(user_params_ps->N_THREADS)
        {
#pragma omp for
            for(i=0;i<NSFR_high;i++) {

                SFRD_z_high_table[j][i] = Nion_ConditionalM(growthf[j],Mmin,Mmax,sigma2,Deltac,overdense_high_table[i],MassTurnover,Alpha_star,0.,Fstar10,1.,Mlim_Fstar,0., FAST_FCOLL_TABLES);
                SFRD_z_high_table[j][i] *= pow(10., 10.0);

            }
        }

        for(i=0;i<NSFR_high;i++) {
            if(isfinite(SFRD_z_high_table[j][i])==0) {
                LOG_ERROR("Detected either an infinite or NaN value in SFRD_z_high_table");
//                Throw(ParameterError);
                Throw(TableGenerationError);
            }
        }

    }
}

void initialise_SFRD_Conditional_table_MINI(
    int Nfilter, float min_density[], float max_density[], float growthf[], float R[],
    float Mcrit_atom[], float Alpha_star, float Alpha_star_mini, float Fstar10, float Fstar7_MINI, bool FAST_FCOLL_TABLES
){

    double overdense_val;
    double overdense_large_high = Deltac, overdense_large_low = global_params.CRIT_DENS_TRANSITION;
    double overdense_small_high, overdense_small_low;

    float Mmin,Mmax,Mlim_Fstar,sigma2,Mlim_Fstar_MINI;
    int i,j,k,i_tot;

    float ln_10;

    ln_10 = log(10);

    Mmin = global_params.M_MIN_INTEGRAL;
    Mmax = RtoM(R[Nfilter-1]);
    Mlim_Fstar = Mass_limit_bisection(Mmin, Mmax, Alpha_star, Fstar10);
    Mlim_Fstar_MINI = Mass_limit_bisection(Mmin, Mmax, Alpha_star_mini, Fstar7_MINI * pow(1e3, Alpha_star_mini));

    float MassTurnover[NMTURN];
    for (i=0;i<NMTURN;i++){
        MassTurnover[i] = pow(10., LOG10_MTURN_MIN + (float)i/((float)NMTURN-1.)*(LOG10_MTURN_MAX-LOG10_MTURN_MIN));
    }

    Mmin = log(Mmin);

    for (i=0; i<NSFR_high;i++) {
        overdense_high_table[i] = overdense_large_low + (float)i/((float)NSFR_high-1.)*(overdense_large_high - overdense_large_low);
    }

    float MassBinLow;
    int MassBin;

    for (j=0; j < Nfilter; j++) {

        Mmax = RtoM(R[j]);

        initialiseGL_Nion_Xray(NGL_SFR, global_params.M_MIN_INTEGRAL, Mmax);

        Mmax = log(Mmax);
        MassBin = (int)floor( ( Mmax - MinMass )*inv_mass_bin_width );

        MassBinLow = MinMass + mass_bin_width*(float)MassBin;

        sigma2 = Sigma_InterpTable[MassBin] + ( Mmax - MassBinLow )*( Sigma_InterpTable[MassBin+1] - Sigma_InterpTable[MassBin] )*inv_mass_bin_width;

        if(min_density[j]*growthf[j] < -1.) {
            overdense_small_low = -1. + global_params.MIN_DENSITY_LOW_LIMIT;
        }
        else {
            overdense_small_low = min_density[j]*growthf[j];
        }
        overdense_small_high = max_density[j]*growthf[j];
        if(overdense_small_high > global_params.CRIT_DENS_TRANSITION) {
            overdense_small_high = global_params.CRIT_DENS_TRANSITION;
        }

        for (i=0; i<NSFR_low; i++) {
            overdense_val = log10(1. + overdense_small_low) + (float)i/((float)NSFR_low-1.)*(log10(1.+overdense_small_high)-log10(1.+overdense_small_low));
            overdense_low_table[i] = pow(10.,overdense_val);
        }

#pragma omp parallel shared(log10_SFRD_z_low_table,growthf,Mmax,sigma2,overdense_low_table,Mcrit_atom,Alpha_star,Alpha_star_mini,Fstar10,Mlim_Fstar,\
                            log10_SFRD_z_low_table_MINI,MassTurnover,Fstar7_MINI,Mlim_Fstar_MINI,ln_10) \
                    private(i,k) num_threads(user_params_ps->N_THREADS)
        {
#pragma omp for
            for (i=0; i<NSFR_low; i++){
                log10_SFRD_z_low_table[j][i] = log10(GaussLegendreQuad_Nion(1,NGL_SFR,growthf[j],Mmax,sigma2,Deltac,overdense_low_table[i]-1.,Mcrit_atom[j],Alpha_star,0.,Fstar10,1.,Mlim_Fstar,0., FAST_FCOLL_TABLES));
                if(log10_SFRD_z_low_table[j][i] < -50.){
                    log10_SFRD_z_low_table[j][i] = -50.;
                }

                log10_SFRD_z_low_table[j][i] += 10.0;
                log10_SFRD_z_low_table[j][i] *= ln_10;

                for (k=0; k<NMTURN; k++){
                    log10_SFRD_z_low_table_MINI[j][i+k*NSFR_low] = log10(GaussLegendreQuad_Nion_MINI(1,NGL_SFR,growthf[j],Mmax,sigma2,Deltac,overdense_low_table[i]-1.,MassTurnover[k], Mcrit_atom[j],Alpha_star_mini,0.,Fstar7_MINI,1.,Mlim_Fstar_MINI, 0., FAST_FCOLL_TABLES));
                    if(log10_SFRD_z_low_table_MINI[j][i+k*NSFR_low] < -50.){
                        log10_SFRD_z_low_table_MINI[j][i+k*NSFR_low] = -50.;
                    }

                    log10_SFRD_z_low_table_MINI[j][i+k*NSFR_low] += 10.0;
                    log10_SFRD_z_low_table_MINI[j][i+k*NSFR_low] *= ln_10;
                }
            }
        }

        for (i=0; i<NSFR_low; i++){
            if(isfinite(log10_SFRD_z_low_table[j][i])==0) {
                LOG_ERROR("Detected either an infinite or NaN value in log10_SFRD_z_low_table");
//                Throw(ParameterError);
                Throw(TableGenerationError);
            }

            for (k=0; k<NMTURN; k++){
                if(isfinite(log10_SFRD_z_low_table_MINI[j][i+k*NSFR_low])==0) {
                    LOG_ERROR("Detected either an infinite or NaN value in log10_SFRD_z_low_table_MINI");
//                    Throw(ParameterError);
                    Throw(TableGenerationError);
                }
            }
        }

#pragma omp parallel shared(SFRD_z_high_table,growthf,Mmin,Mmax,sigma2,overdense_high_table,Mcrit_atom,Alpha_star,Alpha_star_mini,Fstar10,\
                            Mlim_Fstar,SFRD_z_high_table_MINI,MassTurnover,Fstar7_MINI,Mlim_Fstar_MINI) \
                    private(i,k) num_threads(user_params_ps->N_THREADS)
        {
#pragma omp for
            for(i=0;i<NSFR_high;i++) {

                SFRD_z_high_table[j][i] = Nion_ConditionalM(growthf[j],Mmin,Mmax,sigma2,Deltac,overdense_high_table[i],\
                                                            Mcrit_atom[j],Alpha_star,0.,Fstar10,1.,Mlim_Fstar,0., FAST_FCOLL_TABLES);
                if (SFRD_z_high_table[j][i] < 1e-50){
                    SFRD_z_high_table[j][i] = 1e-50;
                }

                SFRD_z_high_table[j][i] *= pow(10., 10.0);

                for (k=0; k<NMTURN; k++){
                    SFRD_z_high_table_MINI[j][i+k*NSFR_high] = Nion_ConditionalM_MINI(growthf[j],Mmin,Mmax,sigma2,Deltac,\
                                                                    overdense_high_table[i],MassTurnover[k],Mcrit_atom[j],\
                                                                    Alpha_star_mini,0.,Fstar7_MINI,1.,Mlim_Fstar_MINI, 0., FAST_FCOLL_TABLES);

                    if (SFRD_z_high_table_MINI[j][i+k*NSFR_high] < 1e-50){
                        SFRD_z_high_table_MINI[j][i+k*NSFR_high] = 1e-50;
                    }
                }
            }
        }

        for(i=0;i<NSFR_high;i++) {
            if(isfinite(SFRD_z_high_table[j][i])==0) {
                LOG_ERROR("Detected either an infinite or NaN value in SFRD_z_high_table");
//                Throw(ParameterError);
                Throw(TableGenerationError);
            }

            for (k=0; k<NMTURN; k++){
                if(isfinite(SFRD_z_high_table_MINI[j][i+k*NSFR_high])==0) {
                    LOG_ERROR("Detected either an infinite or NaN value in SFRD_z_high_table_MINI");
//                    Throw(ParameterError);
                    Throw(TableGenerationError);
                }
            }
        }
    }
}

// The volume filling factor at a given redshift, Q(z), or find redshift at a given Q, z(Q).
//
// The evolution of Q can be written as
// dQ/dt = n_{ion}/dt - Q/t_{rec},
// where n_{ion} is the number of ionizing photons per baryon. The averaged recombination time is given by
// t_{rec} ~ 0.93 Gyr * (C_{HII}/3)^-1 * (T_0/2e4 K)^0.7 * ((1+z)/7)^-3.
// We assume the clumping factor of C_{HII}=3 and the IGM temperature of T_0 = 2e4 K, following
// Section 2.1 of Kuhlen & Faucher-Gigue`re (2012) MNRAS, 423, 862 and references therein.
// 1) initialise interpolation table
// -> initialise_Q_value_spline(NoRec, M_TURN, ALPHA_STAR, ALPHA_ESC, F_STAR10, F_ESC10)
// NoRec = 0: Compute dQ/dt with the recombination time.
// NoRec = 1: Ignore recombination.
// 2) find Q value at a given z -> Q_at_z(z, &(Q))
// or find z at a given Q -> z_at_Q(Q, &(z)).
// 3) free memory allocation -> free_Q_value()

//   Set up interpolation table for the volume filling factor, Q, at a given redshift z and redshift at a given Q.
int InitialisePhotonCons(struct UserParams *user_params, struct CosmoParams *cosmo_params,
                         struct AstroParams *astro_params, struct FlagOptions *flag_options)
{

    /*
        This is an API-level function for initialising the photon conservation.
    */

    int status;
    Try{  // this try wraps the whole function.
    Broadcast_struct_global_PS(user_params,cosmo_params);
    Broadcast_struct_global_UF(user_params,cosmo_params);
    init_ps();
    //     To solve differentail equation, uses Euler's method.
    //     NOTE:
    //     (1) With the fiducial parameter set,
    //	    when the Q value is < 0.9, the difference is less than 5% compared with accurate calculation.
    //	    When Q ~ 0.98, the difference is ~25%. To increase accuracy one can reduce the step size 'da', but it will increase computing time.
    //     (2) With the fiducial parameter set,
    //     the difference for the redshift where the reionization end (Q = 1) is ~0.2 % compared with accurate calculation.
    float ION_EFF_FACTOR,M_MIN,M_MIN_z0,M_MIN_z1,Mlim_Fstar, Mlim_Fesc;
    double a_start = 0.03, a_end = 1./(1. + global_params.PhotonConsEndCalibz); // Scale factors of 0.03 and 0.17 correspond to redshifts of ~32 and ~5.0, respectively.
    double C_HII = 3., T_0 = 2e4;
    double reduce_ratio = 1.003;
    double Q0,Q1,Nion0,Nion1,Trec,da,a,z0,z1,zi,dadt,ans,delta_a,zi_prev,Q1_prev;
    double *z_arr,*Q_arr;
    int Nmax = 2000; // This is the number of step, enough with 'da = 2e-3'. If 'da' is reduced, this number should be checked.
    int cnt, nbin, i, istart;
    int fail_condition, not_mono_increasing, num_fails;
    int gsl_status;

    z_arr = calloc(Nmax,sizeof(double));
    Q_arr = calloc(Nmax,sizeof(double));

    //set the minimum source mass
    if (flag_options->USE_MASS_DEPENDENT_ZETA) {
        ION_EFF_FACTOR = global_params.Pop2_ion * astro_params->F_STAR10 * astro_params->F_ESC10;

        M_MIN = astro_params->M_TURN/50.;
        Mlim_Fstar = Mass_limit_bisection(M_MIN, global_params.M_MAX_INTEGRAL, astro_params->ALPHA_STAR, astro_params->F_STAR10);
        Mlim_Fesc = Mass_limit_bisection(M_MIN, global_params.M_MAX_INTEGRAL, astro_params->ALPHA_ESC, astro_params->F_ESC10);
        if(user_params->FAST_FCOLL_TABLES){
          initialiseSigmaMInterpTable(fmin(MMIN_FAST,M_MIN), 1.0e+21);
        }
        else{
          initialiseSigmaMInterpTable(M_MIN, 1.0e+21);
        }
    }
    else {
        ION_EFF_FACTOR = astro_params->HII_EFF_FACTOR;
    }

    fail_condition = 1;
    num_fails = 0;

    // We are going to come up with the analytic curve for the photon non conservation correction
    // This can be somewhat numerically unstable and as such we increase the sampling until it works
    // If it fails to produce a monotonically increasing curve (for Q as a function of z) after 10 attempts we crash out
    while(fail_condition!=0) {

        //LOG_DEBUG("ENTERING THIS LOOP");

        a = a_start;
        if(num_fails < 3) {
            da = 3e-3 - ((double)num_fails)*(1e-3);
       	}
        else {
            da = 1e-3 - ((double)num_fails - 2.)*(1e-4);
       	}
        delta_a = 1e-7;

        zi_prev = Q1_prev = 0.;
        not_mono_increasing = 0;

        if(num_fails>0) {
            for(i=0;i<Nmax;i++) {
                z_arr[i] = 0.;
                Q_arr[i] = 0.;
            }
        }

        cnt = 0;
        Q0 = 0.;

        while (a < a_end) {

            zi = 1./a - 1.;
            z0 = 1./(a+delta_a) - 1.;
            z1 = 1./(a-delta_a) - 1.;

            // Ionizing emissivity (num of photons per baryon)
            if (flag_options->USE_MASS_DEPENDENT_ZETA) {
                Nion0 = ION_EFF_FACTOR*Nion_General(z0, 1.01*astro_params->M_TURN/50., astro_params->M_TURN, astro_params->ALPHA_STAR,
                                                astro_params->ALPHA_ESC, astro_params->F_STAR10, astro_params->F_ESC10,
                                                Mlim_Fstar, Mlim_Fesc);
                Nion1 = ION_EFF_FACTOR*Nion_General(z1, 1.01*astro_params->M_TURN/50., astro_params->M_TURN, astro_params->ALPHA_STAR,
                                                astro_params->ALPHA_ESC, astro_params->F_STAR10, astro_params->F_ESC10,
                                                Mlim_Fstar, Mlim_Fesc);

                if(Nion0 != Nion0 || Nion1 != Nion1)
                {
                    LOG_ERROR("Nion_General outputs a nan result");
                    return(0);
                }
            }
            else {
                //set the minimum source mass
                if (astro_params->ION_Tvir_MIN < 9.99999e3) { // neutral IGM
                    M_MIN_z0 = (float)TtoM(z0, astro_params->ION_Tvir_MIN, 1.22);
                    M_MIN_z1 = (float)TtoM(z1, astro_params->ION_Tvir_MIN, 1.22);
                }
                else { // ionized IGM
                    M_MIN_z0 = (float)TtoM(z0, astro_params->ION_Tvir_MIN, 0.6);
                    M_MIN_z1 = (float)TtoM(z1, astro_params->ION_Tvir_MIN, 0.6);
                }

                if(M_MIN_z0 < M_MIN_z1) {
                  if(user_params->FAST_FCOLL_TABLES){
                    initialiseSigmaMInterpTable(fmin(MMIN_FAST,M_MIN_z0), 1.0e+21);
                  }
                  else{
                    initialiseSigmaMInterpTable(M_MIN_z0, 1.0e+21);
                  }
                }
                else {
                  if(user_params->FAST_FCOLL_TABLES){
                    initialiseSigmaMInterpTable(fmin(MMIN_FAST,M_MIN_z1), 1.0e+21);
                  }
                  else{
                    initialiseSigmaMInterpTable(M_MIN_z1,1.0e+21);
                  }
                }

                Nion0 = ION_EFF_FACTOR*FgtrM_General(z0,M_MIN_z0);
                Nion1 = ION_EFF_FACTOR*FgtrM_General(z1,M_MIN_z1);
                freeSigmaMInterpTable();
            }

            // With scale factor a, the above equation is written as dQ/da = n_{ion}/da - Q/t_{rec}*(dt/da)
            if (!global_params.RecombPhotonCons) {
                Q1 = Q0 + ((Nion0-Nion1)/2/delta_a)*da; // No Recombination
            }
            else {
                dadt = Ho*sqrt(cosmo_params_ps->OMm/a + global_params.OMr/a/a + cosmo_params_ps->OMl*a*a); // da/dt = Ho*a*sqrt(OMm/a^3 + OMr/a^4 + OMl)
                Trec = 0.93 * 1e9 * SperYR * pow(C_HII/3.,-1) * pow(T_0/2e4,0.7) * pow((1.+zi)/7.,-3);
                Q1 = Q0 + ((Nion0-Nion1)/2./delta_a - Q0/Trec/dadt)*da;
            }


            //LOG_DEBUG("Ok : %e, %e %e", zi, Q1_prev, Q1);

            // Curve is no longer monotonically increasing, we are going to have to exit and start again
            if(Q1 < Q1_prev) {
                not_mono_increasing = 1;
                break;
            }

            zi_prev = zi;
            Q1_prev = Q1;

            z_arr[cnt] = zi;
            Q_arr[cnt] = Q1;

            cnt = cnt + 1;
            if (Q1 >= 1.0) break; // if fully ionized, stop here.
            // As the Q value increases, the bin size decreases gradually because more accurate calculation is required.
            if (da < 7e-5) da = 7e-5; // set minimum bin size.
            else da = pow(da,reduce_ratio);
            Q0 = Q1;
            a = a + da;
        }

        // A check to see if we ended up with a monotonically increasing function
        if(not_mono_increasing==0) {
            fail_condition = 0;
        }
        else {
            num_fails += 1;
            if(num_fails>10) {
                LOG_ERROR("Failed too many times.");
//                Throw ParameterError;
                Throw(PhotonConsError);
            }
        }

    }
    cnt = cnt - 1;
    istart = 0;
    for (i=1;i<cnt;i++){
        if (Q_arr[i-1] == 0. && Q_arr[i] != 0.) istart = i-1;
    }
    nbin = cnt - istart;

    N_analytic = nbin;

    // initialise interploation Q as a function of z
    z_Q = calloc(nbin,sizeof(double));
    Q_value = calloc(nbin,sizeof(double));

    Q_at_z_spline_acc = gsl_interp_accel_alloc ();
    Q_at_z_spline = gsl_spline_alloc (gsl_interp_cspline, nbin);

    for (i=0; i<nbin; i++){
        z_Q[i] = z_arr[cnt-i];
        Q_value[i] = Q_arr[cnt-i];
        //LOG_DEBUG("z_arr, Q: %e, %e, %e", z_Q[i], Q_value[i], istart);
    }

    gsl_set_error_handler_off();
    gsl_status = gsl_spline_init(Q_at_z_spline, z_Q, Q_value, nbin);
    GSL_ERROR(gsl_status);

    Zmin = z_Q[0];
    Zmax = z_Q[nbin-1];
    Qmin = Q_value[nbin-1];
    Qmax = Q_value[0];

    // initialise interpolation z as a function of Q
    double *Q_z = calloc(nbin,sizeof(double));
    double *z_value = calloc(nbin,sizeof(double));

    z_at_Q_spline_acc = gsl_interp_accel_alloc ();
    z_at_Q_spline = gsl_spline_alloc (gsl_interp_linear, nbin);
    for (i=0; i<nbin; i++){
        Q_z[i] = Q_value[nbin-1-i];
        z_value[i] = z_Q[nbin-1-i];
    }

    

    gsl_status = gsl_spline_init(z_at_Q_spline, Q_z, z_value, nbin);
    GSL_ERROR(gsl_status);

    free(z_arr);
    free(Q_arr);

    if (flag_options->USE_MASS_DEPENDENT_ZETA) {
      freeSigmaMInterpTable();
    }

    LOG_DEBUG("Initialised PhotonCons.");
    } // End of try
    Catch(status){
        return status;
    }

    return(0);
}

// Function to construct the spline for the calibration curve of the photon non-conservation
int PhotonCons_Calibration(double *z_estimate, double *xH_estimate, int NSpline){
    int status;
    Try{
        if(xH_estimate[NSpline-1] > 0.0 && xH_estimate[NSpline-2] > 0.0 && xH_estimate[NSpline-3] > 0.0 && xH_estimate[0] <= global_params.PhotonConsStart) {
            initialise_NFHistory_spline(z_estimate,xH_estimate,NSpline);
        }
    }
    Catch(status){
        return status;
    }
    return(0);
}

// Function callable from Python to know at which redshift to start sampling the calibration curve (to minimise function calls)
int ComputeZstart_PhotonCons(double *zstart) {
    int status;
    double temp;

    Try{
        if((1.-global_params.PhotonConsStart) > Qmax) {
            // It is possible that reionisation never even starts
            // Just need to arbitrarily set a high redshift to perform the algorithm
            temp = 20.;
        }
        else {
            z_at_Q(1. - global_params.PhotonConsStart,&(temp));
        // Multiply the result by 10 per-cent to fix instances when this isn't high enough
            temp *= 1.1;
        }
    }
    Catch(status){
        return(status); // Use the status to determine if something went wrong.
    }

    *zstart = temp;
    return(0);
}


void determine_deltaz_for_photoncons() {

    int i, j, increasing_val, counter, smoothing_int;
    double temp;
    float z_cal, z_analytic, NF_sample, returned_value, NF_sample_min, gradient_analytic, z_analytic_at_endpoint, const_offset, z_analytic_2, smoothing_width;
    float bin_width, delta_NF, val1, val2, extrapolated_value;

    LOG_DEBUG("Determining deltaz for photon cons.");

    // Number of points for determine the delta z correction of the photon non-conservation
    N_NFsamples = 100;
    // Determine the change in neutral fraction to calculate the gradient for the linear extrapolation of the photon non-conservation correction
    delta_NF = 0.025;
    // A width (in neutral fraction data points) in which point we average over to try and avoid sharp features in the correction (removes some kinks)
    // Effectively acts as filtering step
    smoothing_width = 35.;


    // The photon non-conservation correction has a threshold (in terms of neutral fraction; global_params.PhotonConsEnd) for which we switch
    // from using the exact correction between the calibrated (21cmFAST all flag options off) to analytic expression to some extrapolation.
    // This threshold is required due to the behaviour of 21cmFAST at very low neutral fractions, which cause extreme behaviour with recombinations on

    // A lot of the steps and choices are not completely rubust, just chosed to smooth/average the data to have smoother resultant reionisation histories

    // Determine the number of extrapolated points required, if required at all.
    if(calibrated_NF_min < global_params.PhotonConsEnd) {
        // We require extrapolation, set minimum point to the threshold, and extrapolate beyond.
        NF_sample_min = global_params.PhotonConsEnd;

        // Determine the number of extrapolation points (to better smooth the correction) between the threshod (global_params.PhotonConsEnd) and a
        // point close to zero neutral fraction (set by global_params.PhotonConsAsymptoteTo)
        // Choice is to get the delta neutral fraction between extrapolated points to be similar to the cadence in the exact correction
        if(calibrated_NF_min > global_params.PhotonConsAsymptoteTo) {
            N_extrapolated = ((float)N_NFsamples - 1.)*(NF_sample_min - calibrated_NF_min)/( global_params.PhotonConsStart - NF_sample_min );
        }
        else {
            N_extrapolated = ((float)N_NFsamples - 1.)*(NF_sample_min - global_params.PhotonConsAsymptoteTo)/( global_params.PhotonConsStart - NF_sample_min );
        }
        N_extrapolated = (int)floor( N_extrapolated ) - 1; // Minus one as the zero point is added below
    }
    else {
        // No extrapolation required, neutral fraction never reaches zero
        NF_sample_min = calibrated_NF_min;

        N_extrapolated = 0;
    }

    // Determine the bin width for the sampling of the neutral fraction for the correction
    bin_width = ( global_params.PhotonConsStart - NF_sample_min )/((float)N_NFsamples - 1.);

    // allocate memory for arrays required to determine the photon non-conservation correction
    deltaz = calloc(N_NFsamples + N_extrapolated + 1,sizeof(double));
    deltaz_smoothed = calloc(N_NFsamples + N_extrapolated + 1,sizeof(double));
    NeutralFractions = calloc(N_NFsamples + N_extrapolated + 1,sizeof(double));

    // Go through and fill the data points (neutral fraction and corresponding delta z between the calibrated and analytic curves).
    for(i=0;i<N_NFsamples;i++) {

        NF_sample = NF_sample_min + bin_width*(float)i;

        // Determine redshift given a neutral fraction for the calibration curve
        z_at_NFHist(NF_sample,&(temp));

        z_cal = temp;

        // Determine redshift given a neutral fraction for the analytic curve
        z_at_Q(1. - NF_sample,&(temp));

        z_analytic = temp;

        deltaz[i+1+N_extrapolated] = fabs( z_cal - z_analytic );
        NeutralFractions[i+1+N_extrapolated] = NF_sample;
    }

    // Determining the end-point (lowest neutral fraction) for the photon non-conservation correction
    if(calibrated_NF_min >= global_params.PhotonConsEnd) {

        increasing_val = 0;
        counter = 0;

        // Check if all the values of delta z are increasing
        for(i=0;i<(N_NFsamples-1);i++) {
            if(deltaz[i+1+N_extrapolated] >= deltaz[i+N_extrapolated]) {
                counter += 1;
            }
        }
        // If all the values of delta z are increasing, then some of the smoothing of the correction done below cannot be performed
        if(counter==(N_NFsamples-1)) {
            increasing_val = 1;
        }

        // Since we never have reionisation, need to set an appropriate end-point for the correction
        // Take some fraction of the previous point to determine the end-point
        NeutralFractions[0] = 0.999*NF_sample_min;
        if(increasing_val) {
            // Values of delta z are always increasing with decreasing neutral fraction thus make the last point slightly larger
            deltaz[0] = 1.001*deltaz[1];
        }
        else {
            // Values of delta z are always decreasing with decreasing neutral fraction thus make the last point slightly smaller
            deltaz[0] = 0.999*deltaz[1];
        }
    }
    else {

        // Ok, we are going to be extrapolating the photon non-conservation (delta z) beyond the threshold
        // Construct a linear curve for the analytic function to extrapolate to the new endpoint
        // The choice for doing so is to ensure the corrected reionisation history is mostly smooth, and doesn't
        // artificially result in kinks due to switching between how the delta z should be calculated

        z_at_Q(1. - (NeutralFractions[1+N_extrapolated] + delta_NF),&(temp));
        z_analytic = temp;

        z_at_Q(1. - NeutralFractions[1+N_extrapolated],&(temp));
        z_analytic_2 = temp;

        // determine the linear curve
        // Multiplitcation by 1.1 is arbitrary but effectively smooths out most kinks observed in the resultant corrected reionisation histories
        gradient_analytic = 1.1*( delta_NF )/( z_analytic - z_analytic_2 );
        const_offset = ( NeutralFractions[1+N_extrapolated] + delta_NF ) - gradient_analytic * z_analytic;

        // determine the extrapolation end point
        if(calibrated_NF_min > global_params.PhotonConsAsymptoteTo) {
            extrapolated_value = calibrated_NF_min;
        }
        else {
            extrapolated_value = global_params.PhotonConsAsymptoteTo;
        }

        // calculate the delta z for the extrapolated end point
        z_at_NFHist(extrapolated_value,&(temp));
        z_cal = temp;

        z_analytic_at_endpoint = ( extrapolated_value - const_offset )/gradient_analytic ;

        deltaz[0] = fabs( z_cal - z_analytic_at_endpoint );
        NeutralFractions[0] = extrapolated_value;

        // If performing extrapolation, add in all the extrapolated points between the end-point and the threshold to end the correction (global_params.PhotonConsEnd)
        for(i=0;i<N_extrapolated;i++) {
            if(calibrated_NF_min > global_params.PhotonConsAsymptoteTo) {
                NeutralFractions[i+1] = calibrated_NF_min + (NF_sample_min - calibrated_NF_min)*(float)(i+1)/((float)N_extrapolated + 1.);
            }
            else {
                NeutralFractions[i+1] = global_params.PhotonConsAsymptoteTo + (NF_sample_min - global_params.PhotonConsAsymptoteTo)*(float)(i+1)/((float)N_extrapolated + 1.);
            }

            deltaz[i+1] = deltaz[0] + ( deltaz[1+N_extrapolated] - deltaz[0] )*(float)(i+1)/((float)N_extrapolated + 1.);
        }
    }

    // We have added the extrapolated values, now check if they are all increasing or not (again, to determine whether or not to try and smooth the corrected curve
    increasing_val = 0;
    counter = 0;

    for(i=0;i<(N_NFsamples-1);i++) {
        if(deltaz[i+1+N_extrapolated] >= deltaz[i+N_extrapolated]) {
            counter += 1;
        }
    }
    if(counter==(N_NFsamples-1)) {
        increasing_val = 1;
    }

    // For some models, the resultant delta z for extremely high neutral fractions ( > 0.95) seem to oscillate or sometimes drop in value.
    // This goes through and checks if this occurs, and tries to smooth this out
    // This doesn't occur very often, but can cause an artificial drop in the reionisation history (neutral fraction value) connecting the
    // values before/after the photon non-conservation correction starts.
    for(i=0;i<(N_NFsamples+N_extrapolated);i++) {

        val1 = deltaz[i];
        val2 = deltaz[i+1];

        counter = 0;

        // Check if we have a neutral fraction above 0.95, that the values are decreasing (val2 < val1), that we haven't sampled too many points (counter)
        // and that the NF_sample_min is less than around 0.8. That is, if a reasonable fraction of the reionisation history is sampled.
        while( NeutralFractions[i+1] > 0.95 && val2 < val1 && NF_sample_min < 0.8 && counter < 100) {

            NF_sample = global_params.PhotonConsStart - 0.001*(counter+1);

            // Determine redshift given a neutral fraction for the calibration curve
            z_at_NFHist(NF_sample,&(temp));
            z_cal = temp;

            // Determine redshift given a neutral fraction for the analytic curve
            z_at_Q(1. - NF_sample,&(temp));
            z_analytic = temp;

            // Determine the delta z
            val2 = fabs( z_cal - z_analytic );
            deltaz[i+1] = val2;
            counter += 1;

            // If after 100 samplings we couldn't get the value to increase (like it should), just modify it from the previous point.
            if(counter==100) {
                deltaz[i+1] = deltaz[i] * 1.01;
            }

        }
    }

    // Store the data in its intermediate state before averaging
    for(i=0;i<(N_NFsamples+N_extrapolated+1);i++) {
        deltaz_smoothed[i] = deltaz[i];
    }

    // If we are not increasing for all values, we can smooth out some features in delta z when connecting the extrapolated delta z values
    // compared to those from the exact correction (i.e. when we cross the threshold).
    if(!increasing_val) {

        for(i=0;i<(N_NFsamples+N_extrapolated);i++) {

            val1 = deltaz[0];
            val2 = deltaz[i+1];

            counter = 0;
            // Try and find a point which can be used to smooth out any dip in delta z as a function of neutral fraction.
            // It can be flat, then drop, then increase. This smooths over this drop (removes a kink in the resultant reionisation history).
            // Choice of 75 is somewhat arbitrary
            while(val2 < val1 && (counter < 75 || (1+(i+1)+counter) > (N_NFsamples+N_extrapolated))) {
                counter += 1;
                val2 = deltaz[i+1+counter];

                deltaz_smoothed[i+1] = ( val1 + deltaz[1+(i+1)+counter] )/2.;
            }
            if(counter==75 || (1+(i+1)+counter) > (N_NFsamples+N_extrapolated)) {
                deltaz_smoothed[i+1] = deltaz[i+1];
            }
        }
    }

    // Here we effectively filter over the delta z as a function of neutral fraction to try and minimise any possible kinks etc. in the functional curve.
    for(i=0;i<(N_NFsamples+N_extrapolated+1);i++) {

        // We are at the end-points, cannot smooth
        if(i==0 || i==(N_NFsamples+N_extrapolated)) {
            deltaz[i] = deltaz_smoothed[i];
        }
        else {

            deltaz[i] = 0.;

            // We are symmetrically smoothing, making sure we have the same number of data points either side of the point we are filtering over
            // This determins the filter width when close to the edge of the data ranges
            if( (i - (int)floor(smoothing_width/2.) ) < 0) {
                smoothing_int = 2*( i ) + (int)((int)smoothing_width%2);
            }
            else if( (i - (int)floor(smoothing_width/2.) + ((int)smoothing_width - 1) ) > (N_NFsamples + N_extrapolated) ) {
                smoothing_int = ((int)smoothing_width - 1) - 2*((i - (int)floor(smoothing_width/2.) + ((int)smoothing_width - 1) ) - (N_NFsamples + N_extrapolated)  ) + (int)((int)smoothing_width%2);
            }
            else {
                smoothing_int = (int)smoothing_width;
            }

            // Average (filter) over the delta z values to smooth the result
            counter = 0;
            for(j=0;j<(int)smoothing_width;j++) {
                if(((i - (int)floor((float)smoothing_int/2.) + j)>=0) && ((i - (int)floor((float)smoothing_int/2.) + j) <= (N_NFsamples + N_extrapolated + 1)) && counter < smoothing_int ) {

                    deltaz[i] += deltaz_smoothed[i - (int)floor((float)smoothing_int/2.) + j];
                    counter += 1;

                }
            }
            deltaz[i] /= (float)counter;
        }

    }

    N_deltaz = N_NFsamples + N_extrapolated + 1;

    // Now, we can construct the spline of the photon non-conservation correction (delta z as a function of neutral fraction)
    deltaz_spline_for_photoncons_acc = gsl_interp_accel_alloc ();
    deltaz_spline_for_photoncons = gsl_spline_alloc (gsl_interp_linear, N_NFsamples + N_extrapolated + 1);

    gsl_set_error_handler_off();
    int gsl_status;
    gsl_status = gsl_spline_init(deltaz_spline_for_photoncons, NeutralFractions, deltaz, N_NFsamples + N_extrapolated + 1);
    GSL_ERROR(gsl_status);

}


float adjust_redshifts_for_photoncons(
    struct AstroParams *astro_params, struct FlagOptions *flag_options, float *redshift,
    float *stored_redshift, float *absolute_delta_z
) {

    int i, new_counter;
    double temp;
    float required_NF, adjusted_redshift, future_z, gradient_extrapolation, const_extrapolation, temp_redshift, check_required_NF;

    LOG_DEBUG("Adjusting redshifts for photon cons.");

    if(*redshift < global_params.PhotonConsEndCalibz) {
        LOG_ERROR(
            "You have passed a redshift (z = %f) that is lower than the enpoint of the photon non-conservation correction "\
            "(global_params.PhotonConsEndCalibz = %f). If this behaviour is desired then set global_params.PhotonConsEndCalibz "\
            "to a value lower than z = %f.",*redshift,global_params.PhotonConsEndCalibz,*redshift
                  );
//        Throw(ParameterError);
        Throw(PhotonConsError);
    }

    // Determine the neutral fraction (filling factor) of the analytic calibration expression given the current sampled redshift
    Q_at_z(*redshift, &(temp));
    required_NF = 1.0 - (float)temp;

    // Find which redshift we need to sample in order for the calibration reionisation history to match the analytic expression
    if(required_NF > global_params.PhotonConsStart) {
        // We haven't started ionising yet, so keep redshifts the same
        adjusted_redshift = *redshift;

        *absolute_delta_z = 0.;
    }
    else if(required_NF<=global_params.PhotonConsEnd) {
        // We have gone beyond the threshold for the end of the photon non-conservation correction
        // Deemed to be roughly where the calibration curve starts to approach the analytic expression

        if(FirstNF_Estimate <= 0. && required_NF <= 0.0) {
            // Reionisation has already happened well before the calibration
            adjusted_redshift = *redshift;
        }
        else {
            // Initialise the photon non-conservation correction curve
            // It is possible that for certain parameter choices that we can get here without initialisation happening.
            // Thus check and initialise if not already done so
            if(!photon_cons_allocated) {
                determine_deltaz_for_photoncons();
                photon_cons_allocated = true;
            }

            // We have crossed the NF threshold for the photon conservation correction so now set to the delta z at the threshold
            if(required_NF < global_params.PhotonConsAsymptoteTo) {

                // This counts the number of times we have exceeded the extrapolated point and attempts to modify the delta z
                // to try and make the function a little smoother
                *absolute_delta_z = gsl_spline_eval(deltaz_spline_for_photoncons, global_params.PhotonConsAsymptoteTo, deltaz_spline_for_photoncons_acc);

                new_counter = 0;
                temp_redshift = *redshift;
                check_required_NF = required_NF;

                // Ok, find when in the past we exceeded the asymptote threshold value using the global_params.ZPRIME_STEP_FACTOR
                // In doing it this way, co-eval boxes will be the same as lightcone boxes with regard to redshift sampling
                while( check_required_NF < global_params.PhotonConsAsymptoteTo ) {

                    temp_redshift = ((1. + temp_redshift)*global_params.ZPRIME_STEP_FACTOR - 1.);

                    Q_at_z(temp_redshift, &(temp));
                    check_required_NF = 1.0 - (float)temp;

                    new_counter += 1;
                }

                // Now adjust the final delta_z by some amount to smooth if over successive steps
                if(deltaz[1] > deltaz[0]) {
                    *absolute_delta_z = pow( 0.96 , (new_counter - 1) + 1. ) * ( *absolute_delta_z );
                }
                else {
                    *absolute_delta_z = pow( 1.04 , (new_counter - 1) + 1. ) * ( *absolute_delta_z );
                }

                // Check if we go into the future (z < 0) and avoid it
                adjusted_redshift = (*redshift) - (*absolute_delta_z);
                if(adjusted_redshift < 0.0) {
                    adjusted_redshift = 0.0;
                }

            }
            else {
                *absolute_delta_z = gsl_spline_eval(deltaz_spline_for_photoncons, required_NF, deltaz_spline_for_photoncons_acc);
                adjusted_redshift = (*redshift) - (*absolute_delta_z);
            }
        }
    }
    else {
        // Initialise the photon non-conservation correction curve
        if(!photon_cons_allocated) {
            determine_deltaz_for_photoncons();
            photon_cons_allocated = true;
        }

        // We have exceeded even the end-point of the extrapolation
        // Just smooth ever subsequent point
        // Note that this is deliberately tailored to light-cone quantites, but will still work with co-eval cubes
        // Though might produce some very minor discrepancies when comparing outputs.
        if(required_NF < NeutralFractions[0]) {

            new_counter = 0;
            temp_redshift = *redshift;
            check_required_NF = required_NF;

            // Ok, find when in the past we exceeded the asymptote threshold value using the global_params.ZPRIME_STEP_FACTOR
            // In doing it this way, co-eval boxes will be the same as lightcone boxes with regard to redshift sampling
            while( check_required_NF < NeutralFractions[0] ) {

                temp_redshift = ((1. + temp_redshift)*global_params.ZPRIME_STEP_FACTOR - 1.);

                Q_at_z(temp_redshift, &(temp));
                check_required_NF = 1.0 - (float)temp;

                new_counter += 1;
            }
            if(new_counter > 5) {
                LOG_WARNING(
                    "The photon non-conservation correction has employed an extrapolation for\n"\
                    "more than 5 consecutive snapshots. This can be unstable, thus please check "\
                    "resultant history. Parameters are:\n"
                );
                #if LOG_LEVEL >= LOG_WARNING
                    writeAstroParams(flag_options, astro_params);
                #endif
            }

            // Now adjust the final delta_z by some amount to smooth if over successive steps
            if(deltaz[1] > deltaz[0]) {
                *absolute_delta_z = pow( 0.998 , (new_counter - 1) + 1. ) * ( *absolute_delta_z );
            }
            else {
                *absolute_delta_z = pow( 1.002 , (new_counter - 1) + 1. ) * ( *absolute_delta_z );
            }

            // Check if we go into the future (z < 0) and avoid it
            adjusted_redshift = (*redshift) - (*absolute_delta_z);
            if(adjusted_redshift < 0.0) {
                adjusted_redshift = 0.0;
            }
        }
        else {
            // Find the corresponding redshift for the calibration curve given the required neutral fraction (filling factor) from the analytic expression
            *absolute_delta_z = gsl_spline_eval(deltaz_spline_for_photoncons, (double)required_NF, deltaz_spline_for_photoncons_acc);
            adjusted_redshift = (*redshift) - (*absolute_delta_z);
        }
    }

    // keep the original sampled redshift
    *stored_redshift = *redshift;

    // This redshift snapshot now uses the modified redshift following the photon non-conservation correction
    *redshift = adjusted_redshift;
}

void Q_at_z(double z, double *splined_value){
    float returned_value;

    if (z >= Zmax) {
        *splined_value = 0.;
    }
    else if (z <= Zmin) {
        *splined_value = 1.;
    }
    else {
        returned_value = gsl_spline_eval(Q_at_z_spline, z, Q_at_z_spline_acc);
        *splined_value = returned_value;
    }
}

void z_at_Q(double Q, double *splined_value){
    float returned_value;

    if (Q < Qmin) {
        LOG_ERROR("The minimum value of Q is %.4e and you passed Q = %.4e", Qmin, Q);
//        Throw(ParameterError);
        Throw(PhotonConsError);
    }
    else if (Q > Qmax) {
        LOG_ERROR("The maximum value of Q is %.4e. Reionization ends at ~%.4f.",Qmax,Zmin);
        LOG_ERROR("This error can occur if global_params.PhotonConsEndCalibz is close to "\
                  "the final sampled redshift. One can consider a lower value for "\
                  "global_params.PhotonConsEndCalibz to mitigate this");
//        Throw(ParameterError);
        Throw(PhotonConsError);
    }
    else {
        returned_value = gsl_spline_eval(z_at_Q_spline, Q, z_at_Q_spline_acc);
        *splined_value = returned_value;
    }
}

void free_Q_value() {

    free_pointer((void**)&Q_at_z_spline, free_gsl_spline);
    free_pointer((void**)&z_at_Q_spline, free_gsl_spline);

    free_pointer((void**)&Q_at_z_spline_acc, gsl_interp_accel_free);
    free_pointer((void**)&z_at_Q_spline_acc, gsl_interp_accel_free);

    /*
    gsl_spline_free (Q_at_z_spline);
    gsl_interp_accel_free (Q_at_z_spline_acc);
    gsl_spline_free (z_at_Q_spline);
    gsl_interp_accel_free (z_at_Q_spline_acc);
    */
}

void initialise_NFHistory_spline(double *redshifts, double *NF_estimate, int NSpline){

    int i, counter, start_index, found_start_index;

    // This takes in the data for the calibration curve for the photon non-conservation correction

    counter = 0;
    start_index = 0;
    found_start_index = 0;

    FinalNF_Estimate = NF_estimate[0];
    FirstNF_Estimate = NF_estimate[NSpline-1];

    // Determine the point in the data where its no longer zero (basically to avoid too many zeros in the spline)
    for(i=0;i<NSpline-1;i++) {
        if(NF_estimate[i+1] > NF_estimate[i]) {
            if(found_start_index == 0) {
                start_index = i;
                found_start_index = 1;
            }
        }
        counter += 1;
    }
    counter = counter - start_index;

    N_calibrated = (counter+1);

    // Store the data points for determining the photon non-conservation correction
    nf_vals = calloc((counter+1),sizeof(double));
    z_vals = calloc((counter+1),sizeof(double));

    calibrated_NF_min = 1.;

    // Store the data, and determine the end point of the input data for estimating the extrapolated results
    for(i=0;i<(counter+1);i++) {
        nf_vals[i] = NF_estimate[start_index+i];
        z_vals[i] = redshifts[start_index+i];
        // At the extreme high redshift end, there can be numerical issues with the solution of the analytic expression
        if(i>0) {
            while(nf_vals[i] <= nf_vals[i-1]) {
                nf_vals[i] += 0.000001;
            }
        }

        if(nf_vals[i] < calibrated_NF_min) {
            calibrated_NF_min = nf_vals[i];
        }
    }

    NFHistory_spline_acc = gsl_interp_accel_alloc ();
//    NFHistory_spline = gsl_spline_alloc (gsl_interp_cspline, (counter+1));
    NFHistory_spline = gsl_spline_alloc (gsl_interp_linear, (counter+1));

    gsl_set_error_handler_off();
    int gsl_status;
    gsl_status = gsl_spline_init(NFHistory_spline, nf_vals, z_vals, (counter+1));
    GSL_ERROR(gsl_status);

    z_NFHistory_spline_acc = gsl_interp_accel_alloc ();
//    z_NFHistory_spline = gsl_spline_alloc (gsl_interp_cspline, (counter+1));
    z_NFHistory_spline = gsl_spline_alloc (gsl_interp_linear, (counter+1));

    gsl_status = gsl_spline_init(z_NFHistory_spline, z_vals, nf_vals, (counter+1));
    GSL_ERROR(gsl_status);
}


void z_at_NFHist(double xHI_Hist, double *splined_value){
    float returned_value;

    returned_value = gsl_spline_eval(NFHistory_spline, xHI_Hist, NFHistory_spline_acc);
    *splined_value = returned_value;
}

void NFHist_at_z(double z, double *splined_value){
    float returned_value;

    returned_value = gsl_spline_eval(z_NFHistory_spline, z, NFHistory_spline_acc);
    *splined_value = returned_value;
}

int ObtainPhotonConsData(
    double *z_at_Q_data, double *Q_data, int *Ndata_analytic, double *z_cal_data,
    double *nf_cal_data, int *Ndata_calibration,
    double *PhotonCons_NFdata, double *PhotonCons_deltaz, int *Ndata_PhotonCons) {

    int i;

    *Ndata_analytic = N_analytic;
    *Ndata_calibration = N_calibrated;
    *Ndata_PhotonCons = N_deltaz;

    for(i=0;i<N_analytic;i++) {
        z_at_Q_data[i] = z_Q[i];
        Q_data[i] = Q_value[i];
    }

    for(i=0;i<N_calibrated;i++) {
        z_cal_data[i] = z_vals[i];
        nf_cal_data[i] = nf_vals[i];
    }

    for(i=0;i<N_deltaz;i++) {
        PhotonCons_NFdata[i] = NeutralFractions[i];
        PhotonCons_deltaz[i] = deltaz[i];
    }

    return(0);
}


int ObtainQanalyticVsZ(double *z_at_Q_data, double *Q_data, int *Ndata_analytic)
{

    int i;
    *Ndata_analytic = N_analytic;

    for(i=0;i<N_analytic;i++) {
        z_at_Q_data[i] = z_Q[i];
        Q_data[i] = Q_value[i];
    }

    return(0);
}


void FreePhotonConsMemory() {

    LOG_DEBUG("FREEING SOME PHOTON CONS. MEMORY");

    free_pointer((void**)&deltaz, free);
    free_pointer((void**)&deltaz_smoothed, free);
    free_pointer((void**)&NeutralFractions, free);
    free_pointer((void**)&z_Q, free);
    free_pointer((void**)&Q_value, free);
    free_pointer((void**)&nf_vals, free);
    free_pointer((void**)&z_vals, free);

    free_Q_value();

    free_pointer((void**)&NFHistory_spline, free_gsl_spline);
    free_pointer((void**)&z_NFHistory_spline, free_gsl_spline);
    free_pointer((void**)&deltaz_spline_for_photoncons, free_gsl_spline);

    free_pointer((void**)&NFHistory_spline_acc, gsl_interp_accel_free);
    free_pointer((void**)&z_NFHistory_spline_acc, gsl_interp_accel_free);
    free_pointer((void**)&deltaz_spline_for_photoncons_acc, gsl_interp_accel_free);

    /*
    gsl_spline_free (NFHistory_spline);
    gsl_interp_accel_free (NFHistory_spline_acc);
    gsl_spline_free (z_NFHistory_spline);
    gsl_interp_accel_free (z_NFHistory_spline_acc);
    gsl_spline_free (deltaz_spline_for_photoncons);
    gsl_interp_accel_free (deltaz_spline_for_photoncons_acc);
    */

    LOG_DEBUG("PHOTON CONS. MEMORY FREED");

    photon_cons_allocated = false;
}

void FreeTsInterpolationTables(){//struct FlagOptions *flag_options) {
    
    LOG_DEBUG("FREEING SOME INTERPOLATION TABLE MEMORY");
	
    freeSigmaMInterpTable();

    LOG_DEBUG("SIGMA INTERP TABLE FREED");

    free_pointer((void**)&z_val, free); //z_val = NULL;
    free_pointer((void**)&Nion_z_val, free);
    free_pointer((void**)&z_X_val, free);// z_X_val = NULL;
    free_pointer((void**)&SFRD_val, free);
    free_pointer((void**)&Nion_z_val_MINI, free);
    free_pointer((void**)&SFRD_val_MINI, free);
    free_pointer((void**)&FgtrM_1DTable_linear, free);
    
    /*
    if (flag_options->USE_MASS_DEPENDENT_ZETA) {
        free_pointer((void**)&z_val, free); //z_val = NULL;
        free_pointer((void**)&Nion_z_val, free);
        free_pointer((void**)&z_X_val, free);// z_X_val = NULL;
        free_pointer((void**)&SFRD_val, free);
        if (flag_options->USE_MINI_HALOS){
            free_pointer((void**)Nion_z_val_MINI, free);
            free_pointer((void**)&SFRD_val_MINI, free);
        }
    }
    else{
        free_pointer((void**)&FgtrM_1DTable_linear, free);
    }
    */

    LOG_DEBUG("INTERPOLATION TABLE MEMORY FREED");
	interpolation_tables_allocated = false;
}


// ----------------------------------------------- //
// ----------------------------------------------- //
/* Function called by the python wrapper interface */
// ----------------------------------------------- //
// ----------------------------------------------- //

/*
    ComputeTransferFunctionNCDM(..., float *k, int length)

    this function is called in the python wrapper analytical_transfer_function_nCDM()

    returns a pointer to an array (to be freed in the python wrapper)
    needs the length of the input array to allocate the output array
*/
float* ComputeTransferFunctionNCDM(struct UserParams *user_params, struct CosmoParams *cosmo_params, 
                        struct AstroParams *astro_params, struct FlagOptions *flag_options, float *k, int length) 
{

    Broadcast_struct_global_PS(user_params,cosmo_params);
    Broadcast_struct_global_UF(user_params,cosmo_params);
    init_ps();

    float* result = malloc(length * sizeof(float));

    for (int i = 0; i < length; i++) 
        result[i] = (float) analytical_transfer_function_nCDM(k[i]);

    free_ps();
    free_TF_CLASS();

    return result;
}

/*
    ComputeTransferFunction(..., float *k, int length)

    this function is called in the python wrapper transfer_function()

    returns a pointer to an array (to be freed in the python wrapper)
    needs the length of the input array to allocate the output array
*/
float* ComputeTransferFunction(struct UserParams *user_params, struct CosmoParams *cosmo_params, 
                        struct AstroParams *astro_params, struct FlagOptions *flag_options, float *k, int length) 
{

    Broadcast_struct_global_PS(user_params,cosmo_params);
    Broadcast_struct_global_UF(user_params,cosmo_params);
    init_ps();

    float* result = malloc(length * sizeof(float));

    for (int i = 0; i < length; i++) 
        result[i] = (float) transfer_function(k[i], true);

    free_ps();
    free_TF_CLASS();

    return result;
}


/*
    ComputeTransferFunctionLCDM(..., float *k, int length)

    this function is called in the python wrapper transfer_function_LCDM()

    returns a pointer to an array (to be freed in the python wrapper)
    needs the length of the input array to allocate the output array
*/
float* ComputeTransferFunctionLCDM(struct UserParams *user_params, struct CosmoParams *cosmo_params, 
                        struct AstroParams *astro_params, struct FlagOptions *flag_options, float *k, int length) 
{

    Broadcast_struct_global_PS(user_params,cosmo_params);
    Broadcast_struct_global_UF(user_params,cosmo_params);
    init_ps();

    float* result = malloc(length * sizeof(float));

    for (int i = 0; i < length; i++) 
        result[i] = (float) transfer_function_LCDM(k[i]);

    free_ps();
    free_TF_CLASS();

    return result;
}




/*
    ComputeMatterPowerSpectrum(..., float *k, int length)

    from the power_spectrum function evaluates the matter power spectrum
    this function is called in the python wrapper matter_power_spectrum()

    returns a pointer to an array (to be freed in the python wrapper)
    needs the length of the input array to allocate the output array
*/
float* ComputeMatterPowerSpectrum(struct UserParams *user_params, struct CosmoParams *cosmo_params, 
                        struct AstroParams *astro_params, struct FlagOptions *flag_options, float *k, int length) 
{

    Broadcast_struct_global_PS(user_params,cosmo_params);
    Broadcast_struct_global_UF(user_params,cosmo_params);
    init_ps();

    float* result = malloc(length * sizeof(float));

    // Uncomment if you want to evaluate the power spectrum amplitude in the form As * D^2(0) where D is the growth function
    // float k0 = 0.05; // reference value of k0 in Mpc^{-1}
    // float AsD2 = 25.0 / 4.0 * 1e+28 * pow(sigma_norm * omhh / pow(C, 2), 2) * pow(k0, cosmo_params_ps->POWER_INDEX - 1.0);
    
    for (int i = 0; i < length; i++) 
        result[i] = (float) power_in_k(k[i]);

    free_ps();
    free_TF_CLASS();

    return result;
}


/*
    ComputePowerInVcb(..., float *k, int length)

    from the power_spectrum function evaluates the matter power spectrum
    this function is called in the python wrapper matter_power_spectrum()

    returns a pointer to an array (to be freed in the python wrapper)
    needs the length of the input array to allocate the output array
*/
float* ComputePowerInVcb(struct UserParams *user_params, struct CosmoParams *cosmo_params, 
                        struct AstroParams *astro_params, struct FlagOptions *flag_options, float *k, int length) 
{

    Broadcast_struct_global_PS(user_params,cosmo_params);
    Broadcast_struct_global_UF(user_params,cosmo_params);
    init_ps();

    float* result = malloc(length * sizeof(float));

    for (int i = 0; i < length; i++) 
        result[i] = (float) power_in_vcb(k[i]);

    free_ps();
    free_TF_CLASS();

    return result;
}


float* ComputePMFInducedMatterPowerSpectrum(struct UserParams *user_params, struct CosmoParams *cosmo_params, 
                        struct AstroParams *astro_params, struct FlagOptions *flag_options, float *k, int length) 
{

    Broadcast_struct_global_PS(user_params,cosmo_params);
    Broadcast_struct_global_UF(user_params,cosmo_params);
    init_ps();

    interpolate_power_spectrum_from_pmf(false);

    float* result = malloc(length * sizeof(float));

    for (int i = 0; i < length; i++) 
        result[i] = (float) pmf_induced_power_spectrum(k[i]);

    free_ps();
    free_TF_CLASS();

    return result;
}


float* ComputeSigmaZ0(struct UserParams *user_params, struct CosmoParams *cosmo_params, 
                        struct AstroParams *astro_params, struct FlagOptions *flag_options, float *mass, int length) 
{

    Broadcast_struct_global_PS(user_params,cosmo_params);
    Broadcast_struct_global_UF(user_params,cosmo_params);
    init_ps();

    if (user_params_ps->USE_INTERPOLATION_TABLES)
        initialiseSigmaMInterpTable(astro_params->M_TURN/50., 1.0e+21);
    
    float* result = malloc(length * sizeof(float));

    for (int i = 0; i < length; i++) 
        result[i] = (float) sigmaFromInterpolationTables(mass[i]);

    if (user_params_ps->USE_INTERPOLATION_TABLES)
        freeSigmaMInterpTable();

    free_ps();
    free_TF_CLASS();

    return result;
}


float *ComputeDSigmaSqDmDk(struct UserParams *user_params, struct CosmoParams *cosmo_params, 
                        struct AstroParams *astro_params, struct FlagOptions *flag_options, float *k, float *params, int length)
{
    Broadcast_struct_global_PS(user_params,cosmo_params);
    Broadcast_struct_global_UF(user_params,cosmo_params);
    init_ps();

    double radius = MtoR((double) params[0]);

    float* result = malloc(length * sizeof(float));

    for (int i = 0; i < length; i++) 
        result[i] = (float) dsigmasq_dm(log(k[i]), &radius);
    
    free_ps();
    free_TF_CLASS();

    return result;
}



float* ComputeDSigmaSqDmZ0(struct UserParams *user_params, struct CosmoParams *cosmo_params, 
                        struct AstroParams *astro_params, struct FlagOptions *flag_options, float *mass, int length) 
{

    Broadcast_struct_global_PS(user_params,cosmo_params);
    Broadcast_struct_global_UF(user_params,cosmo_params);
    init_ps();

    if (user_params_ps->USE_INTERPOLATION_TABLES)
        initialiseSigmaMInterpTable(astro_params->M_TURN/50., 1.0e+21);

    float* result = malloc(length * sizeof(float));

    for (int i = 0; i < length; i++) 
        result[i] = (float) dsigmasqdmFromInterpolationTables(mass[i]);

    if (user_params_ps->USE_INTERPOLATION_TABLES)
        freeSigmaMInterpTable();
    
    free_ps();
    free_TF_CLASS();

    return result;
}


float* ComputeDNDM(struct UserParams *user_params, struct CosmoParams *cosmo_params, 
                        struct AstroParams *astro_params, struct FlagOptions *flag_options, 
                        float *mass, float *z, int length) 
{

    Broadcast_struct_global_PS(user_params,cosmo_params);
    Broadcast_struct_global_UF(user_params,cosmo_params);
    init_ps();

    if (user_params_ps->USE_INTERPOLATION_TABLES)
        initialiseSigmaMInterpTable(astro_params->M_TURN/50., 1.0e+21);

    float* result = malloc(length * sizeof(float));

    double growthf = dicke(z[0]);

    for (int i = 0; i < length; i++) 
    {
        if (user_params_ps->HMF == 0) // Press-Schechter
            result[i] = (float) dNdM(growthf, mass[i]);
        else if (user_params_ps->HMF == 1) // Seth Tormen
            result[i] = (float) dNdM_st(growthf, mass[i]);
        else if (user_params_ps->HMF == 2) // WatsonFOF
            result[i] = (float) dNdM_WatsonFOF(growthf, mass[i]);
        else if (user_params_ps->HMF == 3) // WatsonFOF-z
            result[i] = (float) dNdM_WatsonFOF_z(z[0], growthf, mass[i]);
        else
        {
            LOG_ERROR("Incorrect HMF selected: %i (should be between 0 and 3).", user_params_ps->HMF);
            Throw(ValueError);
        }
    }

    if (user_params_ps->USE_INTERPOLATION_TABLES)
        freeSigmaMInterpTable();
    
    free_ps();
    free_TF_CLASS();

    return result;
}


float* ComputeDNDMConditionnal(struct UserParams *user_params, struct CosmoParams *cosmo_params, 
                        struct AstroParams *astro_params, struct FlagOptions *flag_options, 
                        float *ln_mass, float *params, int length) 
{

    Broadcast_struct_global_PS(user_params,cosmo_params);
    Broadcast_struct_global_UF(user_params,cosmo_params);
    init_ps();

    if (user_params_ps->USE_INTERPOLATION_TABLES)
        initialiseSigmaMInterpTable(astro_params->M_TURN/50., 1.0e+21);

    float* result = malloc(length * sizeof(float));

    double growthf   = (double) params[0];
    double m2        = (double) params[1];
    double delta1    = (double) params[2];
    double delta2    = (double) params[3];
    double sigma2    = (double) params[4];

    for (int i = 0; i < length; i++) 
        result[i] = (float) dNdM_conditional(growthf, (double) ln_mass[i], m2, delta1, delta2, sigma2);
    
    LOG_DEBUG("Here growthf = %e, m2 = %e, delta1 = %e, delta2 = %e, sigma2 = %e", growthf, m2, delta1, delta2, sigma2);
    
    if (user_params_ps->USE_INTERPOLATION_TABLES)
        freeSigmaMInterpTable();
    
    free_ps();
    free_TF_CLASS();

    return result;
}



float* ComputeFgtrMGeneral(struct UserParams *user_params, struct CosmoParams *cosmo_params, 
                        struct AstroParams *astro_params, struct FlagOptions *flag_options, 
                        float *mass, float *z, int length) 
{

    Broadcast_struct_global_PS(user_params,cosmo_params);
    Broadcast_struct_global_UF(user_params,cosmo_params);
    init_ps();

    if (user_params_ps->USE_INTERPOLATION_TABLES)
        initialiseSigmaMInterpTable(astro_params->M_TURN/50., 1.0e+21);

    float* result = malloc(length * sizeof(float));

    for (int i = 0; i < length; i++) 
        result[i] = (float) FgtrM_General(z[0], mass[i]);

    if (user_params_ps->USE_INTERPOLATION_TABLES)
        freeSigmaMInterpTable();
    
    free_ps();
    free_TF_CLASS();

    return result;
}


float* ComputeNionConditionalM(struct UserParams *user_params, struct CosmoParams *cosmo_params, 
                        struct AstroParams *astro_params, struct FlagOptions *flag_options, 
                        float *mass, float *params, int length) 
{

    Broadcast_struct_global_PS(user_params,cosmo_params);
    Broadcast_struct_global_UF(user_params,cosmo_params);
    init_ps();

    if (user_params_ps->USE_INTERPOLATION_TABLES)
        initialiseSigmaMInterpTable(astro_params->M_TURN/50., 5.0e+20);

    float* result = malloc(length * sizeof(float));

    double growthf   = (double) params[0];
    double m2        = (double) params[1];
    double sigma2    = (double) params[2];
    double delta1    = (double) params[3];
    double delta2    = (double) params[4];
    double m_turn    = (double) params[5];
    double alpha_s   = (double) params[6];
    double alpha_e   = (double) params[7];
    double f_star_10 = (double) params[8];
    double f_esc_10  = (double) params[9];
    double mlim_fs   = (double) params[10];
    double mlim_fe   = (double) params[11];

    for (int i = 0; i < length; i++) 
        result[i] = (float) Nion_ConditionalM(growthf, (double) mass[i], m2, sigma2, delta1, delta2, m_turn, alpha_s, alpha_e, f_star_10, f_esc_10, mlim_fs, mlim_fe, user_params_ps->FAST_FCOLL_TABLES);

    // freeing the interpolation tables
    if (user_params_ps->USE_INTERPOLATION_TABLES)
        freeSigmaMInterpTable();

    free_ps();
    free_TF_CLASS();

    return result;
}


float* ComputeDNionConditionalLnM(struct UserParams *user_params, struct CosmoParams *cosmo_params, 
                        struct AstroParams *astro_params, struct FlagOptions *flag_options, 
                        float *ln_mass, float *params, int length) 
{

    Broadcast_struct_global_PS(user_params,cosmo_params);
    Broadcast_struct_global_UF(user_params,cosmo_params);
    init_ps();

    if (user_params_ps->USE_INTERPOLATION_TABLES)
        initialiseSigmaMInterpTable(astro_params->M_TURN/50., 5.0e+20);

    float* result = malloc(length * sizeof(float));

    struct parameters_gsl_SFR_con_int_ parameters_gsl_SFR_con = {
        .gf_obs = params[0],
        .Mval = params[1],
        .sigma2 = params[2],
        .delta1 = params[3],
        .delta2 = params[4],
        .Mdrop = params[5],
        .pl_star = params[6],
        .pl_esc = params[7],
        .frac_star = params[8],
        .frac_esc = params[9],
        .LimitMass_Fstar = params[10],
        .LimitMass_Fesc = params[11]
    };

    for (int i = 0; i < length; i++) 
        result[i] = (float) dNion_ConditionallnM((double) ln_mass[i], &parameters_gsl_SFR_con);

    // freeing the interpolation tables
    if (user_params_ps->USE_INTERPOLATION_TABLES)
        freeSigmaMInterpTable();

    free_ps();
    free_TF_CLASS();

    return result;
}


float* ComputeNionGeneral(struct UserParams *user_params, struct CosmoParams *cosmo_params, 
                        struct AstroParams *astro_params, struct FlagOptions *flag_options, 
                        float *z, float *params, int length) 
{

    Broadcast_struct_global_PS(user_params,cosmo_params);
    Broadcast_struct_global_UF(user_params,cosmo_params);
    init_ps();

    if (user_params_ps->USE_INTERPOLATION_TABLES)
        initialiseSigmaMInterpTable(astro_params->M_TURN/50., 5.0e+20);

    float* result = malloc(length * sizeof(float));

    double m_min       = (double) params[0];
    double m_turn      = (double) params[1];
    double alpha_star  = (double) params[2];
    double alpha_esc   = (double) params[3];
    double f_star10    = (double) params[4];
    double f_esc10     = (double) params[5];
    double mlim_fstar  = (double) params[6];
    double mlim_fesc   = (double) params[7];

    LOG_DEBUG("global_params : PhotonConsEndCalibz = %e", global_params.PhotonConsEndCalibz);

    for (int i = 0; i < length; i++) 
        result[i] = (float) Nion_General((double) z[i], m_min, m_turn, alpha_star, alpha_esc, f_star10, f_esc10, mlim_fstar, mlim_fesc);

    // freeing the interpolation tables
    if (user_params_ps->USE_INTERPOLATION_TABLES)
        freeSigmaMInterpTable();

    free_ps();
    free_TF_CLASS();

    return result;
}
