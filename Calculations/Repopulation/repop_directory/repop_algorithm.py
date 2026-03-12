import os
import sys
import yaml
import psutil
import inspect
import numpy as np
import time

from scipy.optimize import newton
from scipy.interpolate import UnivariateSpline
from scipy.integrate import simpson

# from astropy import units as u


from shvf_functions import SHVF_model, SHVF_model_integral


def read_config_file(ConfigFile):
    with open(ConfigFile, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml


def ff(c):
    return np.log(1. + c) - c / (1. + c)


class repop_algorithm:
    def __init__(self, sim_type, res_string, path_input):
        self.input_dict = read_config_file(path_input)

        self.cosmo_G = self.input_dict['cosmo_constants']['G']
        self.cosmo_H_0 = self.input_dict['cosmo_constants']['H_0']
        self.cosmo_rho_crit = self.input_dict['cosmo_constants']['rho_crit']

        self.host_R_vir = self.input_dict['host']['R_vir']
        self.host_rho_0 = float(self.input_dict['host']['rho_0'])
        self.host_r_s = self.input_dict['host']['r_s']

        cv_cts = self.input_dict['Cv']
        srd_cts = self.input_dict['SRD']
        self.SHVF_cts = self.input_dict['SHVF']

        repopulations = self.input_dict['repopulations']

        self.repop_its = repopulations['its']
        self.num_subs_max = int(float(repopulations['num_subs_max']))
        self.repop_print_freq = repopulations['print_freq']
        self.repop_num_brightest = int(repopulations['num_brightest'])
        self.repop_inc_factor = repopulations['inc_factor']

        self.SHVF_cts_RangeMin = self.SHVF_cts['RangeMin']
        self.SHVF_cts_RangeMax = self.SHVF_cts['RangeMax']

        if sim_type == 'dmo':
            self.SHVF_bb = self.SHVF_cts['dmo']['bb']
            self.SHVF_mm = self.SHVF_cts['dmo']['mm']

            self.Cv_bb = cv_cts['dmo']['bb']
            self.Cv_mm = cv_cts['dmo']['mm']
            self.Cv_sigma = cv_cts['dmo']['sigma']

            self.srd_args_repop = srd_cts['dmo'][res_string]['args']
            self.srd_args_visible = srd_cts['dmo']['fragile']['args']
            self.srd_last_sub = np.asarray(
                srd_cts['dmo'][res_string]['last_subhalo'])

        if sim_type == 'hydro':
            self.SHVF_bb = self.SHVF_cts['hydro']['bb']
            self.SHVF_mm = self.SHVF_cts['hydro']['mm']

            self.Cv_bb = cv_cts['hydro']['bb']
            self.Cv_mm = cv_cts['hydro']['mm']
            self.Cv_sigma = cv_cts['hydro']['sigma']

            self.srd_args_repop = srd_cts['hydro'][res_string]['args']
            self.srd_args_visible = srd_cts['hydro']['fragile']['args']
            self.srd_last_sub = srd_cts['hydro'][res_string]['last_subhalo']

        print(self.SHVF_cts_RangeMin,
              self.SHVF_cts_RangeMax)
        print('    Max. number of repop subhalos: %i' %
              SHVF_model_integral(
                  Vmax_min=self.SHVF_cts_RangeMin,
                  Vmax_max=self.SHVF_cts_RangeMax,
                  SHVF_model_int='SHVF_Grand2012',
                  SHVF_params_int=[self.SHVF_bb, self.SHVF_mm],
                  verbose_int=True))



    def run(self, path_output):

        result = self.fun2()

        yaml.dump(data_dict, file_inputs,
                  default_flow_style=False, allow_unicode=True)

print(os.getcwd())
model = repop_algorithm('dmo', 'resilient',
                        '../input_files/input_paper2024.yml')
model.run('outputs')
'''

"""
        Initialize dataset for N objects.
        :param N: Number of objects
        :param config: Optional dict with raw data or parameters
        """
        self.N = N
        # Initialize raw parameters (e.g., raw concentration)
        self.params = {}
        self.cache = {}  # cache for computed parameters
        self.units = {}  # store units info for each parameter

        if config:
            for key, value in config.items():
                # Assume raw data are numpy arrays
                self.params[key] = value
        # To be filled with raw data (e.g., concentrations)

    def set_parameter(self, name, array, unit=None):
        """Set raw parameter data with optional unit."""
        self.params[name] = array
        if unit:
            self.units[name] = unit

    def get_parameter(self, name):
        """
        Retrieve parameter, computing if necessary.
        Uses cache to avoid recomputation.
        """
        if name in self.cache:
            return self.cache[name]
        elif name in self.params:
            # Raw parameter, no computation needed
            param_data = self.params[name]
            # Attach unit if known
            if name in self.units:
                param_data = param_data * self.units[name]
            self.cache[name] = param_data
            return param_data
        else:
            # Need to compute parameter
            compute_func = getattr(self, f"compute_{name}", None)
            if compute_func:
                data = compute_func()
                self.cache[name] = data
                return data
            else:
                raise ValueError(
                    f"Parameter '{name}' not found and no compute method defined.")

    # Example: compute concentration if not provided
    def compute_concentration(self):
        # If raw data exists, process it; else, generate default
        if 'concentration' in self.params:
            data = self.params['concentration']
        else:
            # Generate some default data for illustration
            data = np.ones(self.N)
        # Attach units if known, or set defaults
        if 'concentration' not in self.units:
            self.units['concentration'] = u.cm3
        return data

    # Example: custom parametrization dependent on concentration
    def compute_Jfactor(self):
        # Depends on concentration
        conc = self.get_parameter('concentration')
        # Vectorized calculation
        J = conc ** 2  # placeholder formula
        if 'Jfactor' not in self.units:
            self.units['Jfactor'] = u.cm3 ** 2
        return J

    # You can add more compute functions here

    def calculate_all(self):
        """
        Compute all parameters that depend on the raw data.
        """
        for param in self.get_all_required_params():
            self.get_parameter(param)

    def get_all_required_params(self):
        """Return list of all parameters needing calculation."""
        return ['concentration', 'Jfactor']

    def sort_by_parameter(self, param_name):
        """
        Return indices to sort objects based on a parameter.
        """
        param_data = self.get_parameter(
            param_name).value  # get raw numpy array
        return np.argsort(param_data)

    def get_sorted_parameters(self, param_name):
        indices = self.sort_by_parameter(param_name)
        sorted_params = {}
        for key in self.params:
            sorted_params[key] = self.get_parameter(key)[indices]
        # Also include computed parameters if needed
        for key in self.cache:
            # cache contains computed params
            # retrieve and sort as well
            pass
        return sorted_params


# Usage example:

# Initialize dataset for 10 million objects
N_objects = 10_000_000
dataset = LargeObjectDataset(N_objects)

# Set raw parameter with units
raw_conc = np.random.rand(N_objects) * 1e-3  # example raw data
dataset.set_parameter('concentration', raw_conc, unit=u.cm ** 3)

# Calculate derived parameters
dataset.calculate_all()

# Retrieve a parameter
J = dataset.get_parameter('Jfactor')

# Sort objects by Jfactor
indices = dataset.sort_by_parameter('Jfactor')

# Get sorted data for all parameters
sorted_params = dataset.get_sorted_parameters('Jfactor')


def memory_usage_psutil():
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(10 ** 6)
    return mem





# ----------- CONCENTRATIONS ----------------------
def Cv_Grand2012(Vmax, Cv_bb, Cv_mm):
    """
    Calculate the concentration of a subhalo population.
    Based on Grand 2012.07846.

    :param Vmax: float or array-like [km/s]
        Maximum radial velocity of a bound particle in the subhalo.

    :return: float or array-like
        Concentrations of a subhalo population.
    """
    # Concentration based on Grand 2012.07846.
    return (10 ** Cv_bb
            * Vmax ** Cv_mm)


def Cv_Mol2021_redshift0(V, c0=1.75e5, c1=-0.90368, c2=0.2749, c3=-0.028):
    # Median subhalo concentration depending on its Vmax and
    # its redshift (here z=0).
    # Moline et al. 2110.02097
    #
    # V - max radial velocity of a bound particle in the subhalo [km/s]
    ci = [c0, c1, c2, c3]
    return ci[0] * (1 + (sum([ci[i + 1] * np.log10(V) ** (i + 1)
                              for i in range(3)])))


def Moline21_normalization(V, c0):
    return Cv_Mol2021_redshift0(V, c0, c1=-0.90368,
                                c2=0.2749, c3=-0.028)


def C_Scatt(C, Cv_sigma):
    """
    Create a scatter in the concentration parameter of the
    repopulated population.
    Scatter in logarithmic scale, following a Gaussian distribution.

    :param C: float or array-like
        Concentration of a subhalo (according to the concentration
        law).
    :return: float or array-like
        Subhalos with scattered concentrations.
    """
    scatter = np.random.normal(loc=0, scale=Cv_sigma, size=C.size)
    return C * 10 ** scatter


# ----------- J-FACTORS --------------------------------
def J_abs_vel(V, D_earth, C,
              cosmo_G=input_dict['cosmo_constants']['G'],
              cosmo_H_0=input_dict['cosmo_constants']['H_0'],
              change_units=True):
    """
    J-factor enclosing whole subhalo as a function of the
    subhalo Vmax.

    :param V: float or array-like  [km/s]
        Maximum circular velocity inside a subhalo.
    :param D_earth: float or array-like [kpc]
        Distance between the subhalo and the Earth.
    :param C: float or array-like
        Subhalo concentration.
    :param change_units: Bool
        Change the output units of the Jfactors.

    :return: float or array-like
        Jfactor of a whole subhalo.
        Units in which it can be returned:
        -> [Msun**2 / kpc**5] with change_units=False
        -> [GeV**2 / cm**5] with change_units=True
    """
    yy = (2.163 ** 3. / D_earth ** 2.
          / (np.log(1. + 2.163) - 2.163 / (1. + 2.163)) ** 2
          * cosmo_H_0 / 12 / np.pi / float(cosmo_G) ** 2
          * np.sqrt(C / 2) * V ** 3
          * 1e-3)

    if change_units:
        yy *= 4.446e6  # GeV ^ 2 cm ^ -5 Msun ^ -2 kpc ^ 5
    return yy


def Js_vel(V, D_earth, C,
           cosmo_G,
           cosmo_H_0, change_units=True):
    """
    Jfactor enclosing the subhalo up to rs as a function of Vmax.

    :param V: float or array-like  [km/s]
        Maximum circular velocity inside a subhalo.
    :param D_earth: float or array-like [kpc]
        Distance between the subhalo and the Earth.
    :param C: float or array-like
        Subhalo concentration.
    :param change_units: Bool
        Change the output units of the Jfactors.

    :return: float or array-like
        Jfactor of a subhalo up to rs.
        Units in which it can be returned:
        -> [Msun**2 / kpc**5] with change_units=False
        -> [GeV**2 / cm**5] with change_units=True
    """
    return J_abs_vel(V, D_earth, C,
                     cosmo_G=cosmo_G,
                     cosmo_H_0=cosmo_H_0,
                     change_units=change_units) * 7 / 8


def J03_vel(V, D_earth, C,
            cosmo_G,
            cosmo_H_0, change_units=True):
    """
    Jfactor enclosing the subhalo up to 0.3 degrees as a
    function of Vmax.

    :param V: float or array-like  [km/s]
        Maximum circular velocity inside a subhalo.
    :param D_earth: float or array-like [kpc]
        Distance between the subhalo and the Earth.
    :param C: float or array-like
        Subhalo concentration.
    :param change_units: Bool
        Change the output units of the Jfactors.

    :return: float or array-like
        Jfactor of a subhalo up to 0.3 degrees.
        Units in which it can be returned:
        -> [Msun**2 / kpc**5] with change_units=False
        -> [GeV**2 / cm**5] with change_units=True
    """
    return (J_abs_vel(V, D_earth, C,
                      cosmo_G=cosmo_G,
                      cosmo_H_0=cosmo_H_0,
                      change_units=change_units)
            * (1 - 1 / (1 + 2.163 * D_earth * np.tan(0.15 * np.pi / 180.)
                        / R_max(V, C, cosmo_H_0)) ** 3))


# ----------- REPOPULATION ----------------
def R_max(V, C,
          cosmo_H_0):
    """
    Calculate Rmax of a subhalo.

    :param V: float or array-like [km/s]
        Maximum circular velocity inside a subhalo.
    :param C: float or array-like
        Subhalo concentration.

    :return: float or array-like [kpc]
        Rmax of the subhalo given by the inputs.
    """
    return V / cosmo_H_0 * np.sqrt(2. / C) * 1e3


def R_s(V, C, cosmo_H_0):
    """
    Calculate scale radius (R_s) of a subhalo following the NFW
    analytical expression for a subhalo density profile.

    :param V: float or array-like [km/s]
        Maximum circular velocity inside a subhalo.
    :param C: float or array-like
        Subhalo concentration.

    :return: float or array-like [kpc]
        R_s of the subhalo given by the inputs.
    """
    return R_max(V, C, cosmo_H_0) / 2.163


def R_t(V, C, DistGC,
        cosmo_H_0, cosmo_G,
        host_rho_0, host_r_s,
        singular_case=True):
    """
    Calculation of tidal radius (R_t) of a subhalo, following the
    NFW analytical expression for a subhalo density profile.

    Definition of R_t: 1603.04057 King radius pg 14

    :param V: float or array-like [km/s]
        Maximum circular velocity inside a subhalo.
    :param C: float or array-like
        Subhalo concentration.
    :param DistGC: float or array-like [kpc]
        Distance from the center of the subhalo to the
        Galactic Center (GC).

    :return: float or array-like [kpc]
        Tidal radius of the subhalo given by the inputs.
    """
    Rmax = R_max(V, C, cosmo_H_0)

    if singular_case:
        c200 = C200_from_Cv_float(C)
    else:
        c200 = C200_from_Cv_array(C)

    M = mass_from_Vmax(V, Rmax, c200, cosmo_G)

    return (((M / (3 * Mhost_encapsulated(
        DistGC, host_rho_0, host_r_s))) ** (1. / 3))
            * DistGC)


def Mhost_encapsulated(R, host_rho_0, host_r_s):
    """
    Host mass encapsulated up to a certain radius. We are following
    a NFW density profile for the host.

    :param R: float or array-like [kpc]
        Radius up to which we integrate the density profile.

    :return: float or array-like [Msun]
        Host mass encapsulated up to R.
    """
    return (4 * np.pi * host_rho_0
            * host_r_s ** 3
            * (np.log((host_r_s + R) / host_r_s)
               - R / (host_r_s + R)))


def N_subs_resilient(DistGC, args):
    return args * np.ones_like(DistGC)


def N_subs_fragile(DistGC, args, srd_last_sub):
    return (args[1] * np.exp(args[0] / DistGC * args[2])
            * (DistGC >= srd_last_sub))


def mass_from_Vmax(Vmax, Rmax, c200,
                   cosmo_G):
    """
    Mass from a subhalo assuming a NFW profile.
    Theoretical steps in Moline16.

    :param Vmax: float or array-like [km/s]
        Maximum radial velocity of a bound particle in the subhalo.
    :param Rmax: float or array-like [kpc]
        Radius at which Vmax happens (from the subhalo center).
    :param c200: float or array-like
        Concentration of the subhalo in terms of mass.
    :return: float or array-like [Msun]
        Mass from the subhalo assuming a NFW profile.
    """
    return (Vmax ** 2 * Rmax / float(cosmo_G)
            * ff(c200) / (np.log(1. + 2.163) - 2.163 / (1. + 2.163)))


def def_Cv(c200, Cv):
    """
    Formula to find c200 knowing Cv to input in the Newton
    root-finding method.

    :param c200: float or array-like
        c200 of subhalo (concentration definition)
    :param Cv: float or array-like
        Cv of subhalo (concentration definition)

    :return: float or array-like
        The output will be 0 when you find the c200 for a
        specific Cv.
    """
    return (200 * (np.log(1. + 2.163) - 2.163 / (1. + 2.163))
            / ff(c200) * (c200 / 2.163) ** 3 - Cv)


def newton2(fun, x0, args):
    """
    Newton method to find the root of a function.

    :param fun: function
        Function that you want the root of.
    :param x0: float
        Initial guess for the root.
    :param args: float or array-like
        Additional parameters that the function might need.

    :return: float
        Root of the function.
    """
    x = x0
    for i in range(100):
        x = x - fun(x, args) * 0.02 / (fun(x + 0.01, args)
                                       - fun(x - 0.01, args))
    return x


# @jit()
def C200_from_Cv_array(Cv):
    """
    Function to find c200 knowing Cv.

    :param Cv: float or array-like
        Cv of subhalo (concentration definition)

    :return: float or array-like
        c200 of subhalo (concentration definition)
    """
    C200_med = []
    for i in Cv:
        C200_med.append(newton2(def_Cv, 40.0, i))

    return np.array(C200_med)


# @njit()
def C200_from_Cv_float(Cv):
    """
    Function to find c200 knowing Cv.

    :param Cv: float or array-like
        Cv of subhalo (concentration definition)

    :return: float or array-like
        c200 of subhalo (concentration definition)
    """
    C200_med = newton2(def_Cv, 40.0, Cv)

    return C200_med


# @jit(forceobj=True)
def montecarlo_algorithm(x_min, x_max, pdf, num_subhalos,
                         sim_type, res_string,
                         cosmo_G,
                         cosmo_H_0,
                         cosmo_rho_crit,

                         host_R_vir,
                         host_rho_0,
                         host_r_s,

                         pathname,
                         repop_its,
                         repop_print_freq,
                         repop_inc_factor,

                         SHVF_cts_RangeMin,
                         SHVF_cts_RangeMax,
                         SHVF_bb,
                         SHVF_mm,

                         Cv_bb,
                         Cv_mm,
                         Cv_sigma,

                         srd_args_repop,
                         srd_args_visible,
                         srd_last_sub):
    """
    Montecarlo sample algorithm. It populates a number of objects
    with a probability distribution defined by the pdf function.
    Calculates the cdf and relates it to the distribution of
    parameters.

    :param x_min: float
        Minimum value the function can intake.
    :param x_max: float
        Maximum value the function can intake.
    :param pdf: function
        Function that we use a probability distribution.
    :param num_subhalos: int
        Number of objects we want.

    :return: float or array-like
        Population following the probability distribution.
    """
    x = np.geomspace(x_min, x_max, num=2000)
    y = pdf(x,
            sim_type, res_string,
            cosmo_G,
            cosmo_H_0,
            cosmo_rho_crit,

            host_R_vir,
            host_rho_0,
            host_r_s,

            pathname,
            repop_its,
            repop_print_freq,
            repop_inc_factor,

            SHVF_cts_RangeMin,
            SHVF_cts_RangeMax,
            SHVF_bb,
            SHVF_mm,

            Cv_bb,
            Cv_mm,
            Cv_sigma,

            srd_args_repop,
            srd_args_visible,
            srd_last_sub)

    cumul = [simpson(y=y[:i], x=x[:i]) for i in range(1, len(x))]
    cumul /= cumul[-1]

    x_mean = (x[1:] + x[:-1]) / 2.

    x_min = ((np.array(cumul) - 1e-8) < 0).argmin() - 1

    spline = UnivariateSpline(cumul[x_min:], x_mean[x_min:],
                              s=0, k=1, ext=0)

    return spline(np.random.random(num_subhalos))


def calculate_characteristics_subhalo(
        Vmax, Distgc,
        cosmo_G,
        cosmo_H_0,
        Cv_bb,
        Cv_mm,
        Cv_sigma):
    # Random distribution of subhalos around the celestial sphere
    num_subs = len(Vmax)
    repop_theta = np.random.uniform(0, 2 * np.pi, num_subs)  # el que va a 2pi
    repop_phi = np.arccos(
        2 * np.random.uniform(0, 1, num_subs) - 1)  # el que va a pi

    # Positions of the subhalos
    repop_Xs = (Distgc * np.cos(repop_theta)
                * np.sin(repop_phi))
    repop_Ys = (Distgc * np.sin(repop_theta)
                * np.sin(repop_phi))
    repop_Zs = Distgc * np.cos(repop_phi)

    repop_DistEarth = ((repop_Xs - 8.5) ** 2 + repop_Ys ** 2
                       + repop_Zs ** 2) ** 0.5

    # repop_C = Cv_Grand2012(Vmax, Cv_bb, Cv_mm)
    repop_C = Moline21_normalization(Vmax, c0=Cv_bb)
    repop_C = C_Scatt(repop_C, Cv_sigma)

    repop_Js = Js_vel(Vmax, repop_DistEarth, repop_C,
                      cosmo_G=cosmo_G,
                      cosmo_H_0=cosmo_H_0)
    repop_J03 = J03_vel(Vmax, repop_DistEarth, repop_C,
                        cosmo_G=cosmo_G,
                        cosmo_H_0=cosmo_H_0)

    # if res_string == 'fragile' and (repop_num_brightest > 100):
    # roche = (R_t(Vmax, repop_C, Distgc,
    #              cosmo_H_0, cosmo_G, host_rho_0, host_r_s,
    #              singular_case=False)
    #          < R_s(Vmax, repop_C, cosmo_H_0))
    #
    # repop_Js[roche] = 0.
    # repop_J03[roche] = 0.

    # Angular size of subhalos (up to R_s)
    repop_Theta = 180 / np.pi * np.arctan(
        R_s(Vmax, repop_C, cosmo_H_0) / repop_DistEarth)

    return np.column_stack((repop_Js, repop_J03, Distgc, repop_DistEarth,
                            Vmax, repop_Theta, repop_C))


def xx(mmax, mmin, SHVF_bb, SHVF_mm, root):
    return SHVF_Grand2012_int(mmin, mmax, SHVF_bb, SHVF_mm) - root


def xxx(mmax, params):
    return (SHVF_Grand2012_int(params[0], mmax,
                               params[1], params[2]) - params[3])


def interior_loop_singularbrightest(
        num_subs_max,
        sim_type, res_string,
        cosmo_G,
        cosmo_H_0,
        cosmo_rho_crit,

        host_R_vir,
        host_rho_0,
        host_r_s,

        pathname,
        repop_its,
        repop_print_freq,
        repop_inc_factor,
        repop_num_brightest,

        SHVF_cts_RangeMin,
        SHVF_cts_RangeMax,
        SHVF_bb,
        SHVF_mm,

        Cv_bb,
        Cv_mm,
        Cv_sigma,

        srd_args_repop,
        srd_args_visible,
        srd_last_sub,
        Vmax_completion):
    # We have 6 variables we want to save in our files,
    # change this number if necessary
    # (output from 'calculate_characteristics_subhalo()')
    full_repop = True
    if not full_repop:
        brightest_Js = np.zeros((2 * repop_num_brightest, 6))
        brightest_J03 = np.zeros((2 * repop_num_brightest, 6))
    else:
        # In case some of the specific ranges gives a number bigger
        # than the general Vmax range, allow for a few more slots
        # (they get erased before shipping out)
        brightest_Js = np.zeros((repop_num_brightest+5, 6))
        brightest_J03 = np.zeros((repop_num_brightest+5, 6))
        nn = 0

    # We calculate our subhalo population in bins to save memory
    m_min = SHVF_cts_RangeMin

    while m_min < SHVF_cts_RangeMax:

        if SHVF_Grand2012_int(m_min, m_min * repop_inc_factor,
                              SHVF_bb, SHVF_mm) > num_subs_max:

            m_max = newton(xx, m_min,
                           args=[m_min, SHVF_bb, SHVF_mm, num_subs_max])
            new_mmin = m_max

        else:
            m_max = np.minimum(m_min * repop_inc_factor,
                               SHVF_cts_RangeMax)
            new_mmin = m_min * repop_inc_factor

            if (SHVF_Grand2012_int(
                    m_max, np.minimum(
                        m_max * repop_inc_factor,
                        SHVF_cts_RangeMax),
                    SHVF_bb, SHVF_mm) < 1.) and (
                    m_max < SHVF_cts_RangeMax
            ):
                m_max = SHVF_cts_RangeMax
                new_mmin = SHVF_cts_RangeMax

        min_distGC = 1e-3

        repop_Vmax = montecarlo_algorithm(
            m_min, m_max,
            SHVF_Grand2012,
            num_subhalos=SHVF_Grand2012_int(m_min, m_max, SHVF_bb, SHVF_mm),
            sim_type=sim_type,
            res_string=res_string,

            cosmo_G=cosmo_G,
            cosmo_H_0=cosmo_H_0,
            cosmo_rho_crit=cosmo_rho_crit,

            host_R_vir=host_R_vir,
            host_rho_0=host_rho_0,
            host_r_s=host_r_s,

            pathname=pathname,
            repop_its=repop_its,
            repop_print_freq=repop_print_freq,
            repop_inc_factor=repop_inc_factor,

            SHVF_cts_RangeMin=SHVF_cts_RangeMin,
            SHVF_cts_RangeMax=SHVF_cts_RangeMax,
            SHVF_bb=SHVF_bb,
            SHVF_mm=SHVF_mm,

            Cv_bb=Cv_bb,
            Cv_mm=Cv_mm,
            Cv_sigma=Cv_sigma,

            srd_args_repop=srd_args_repop,
            srd_args_visible=srd_args_visible,
            srd_last_sub=srd_last_sub)

        repop_DistGC = montecarlo_algorithm(
            min_distGC, host_R_vir,
            Nr_Ntot_repop,
            num_subhalos=SHVF_Grand2012_int(m_min, m_max, SHVF_bb, SHVF_mm),
            sim_type=sim_type,
            res_string=res_string,

            cosmo_G=cosmo_G,
            cosmo_H_0=cosmo_H_0,
            cosmo_rho_crit=cosmo_rho_crit,

            host_R_vir=host_R_vir,
            host_rho_0=host_rho_0,
            host_r_s=host_r_s,

            pathname=pathname,
            repop_its=repop_its,
            repop_print_freq=repop_print_freq,
            repop_inc_factor=repop_inc_factor,

            SHVF_cts_RangeMin=SHVF_cts_RangeMin,
            SHVF_cts_RangeMax=SHVF_cts_RangeMax,
            SHVF_bb=SHVF_bb,
            SHVF_mm=SHVF_mm,

            Cv_bb=Cv_bb,
            Cv_mm=Cv_mm,
            Cv_sigma=Cv_sigma,

            srd_args_repop=srd_args_repop,
            srd_args_visible=srd_args_visible,
            srd_last_sub=srd_last_sub)

        new_data = calculate_characteristics_subhalo(
            repop_Vmax, repop_DistGC,

            cosmo_G=cosmo_G,
            cosmo_H_0=cosmo_H_0,

            Cv_bb=Cv_bb,
            Cv_mm=Cv_mm,
            Cv_sigma=Cv_sigma)

        if not full_repop:
            for new_sub in range(repop_num_brightest):

                bright_Js = np.argmax(new_data[:, 0])

                while (R_t(new_data[bright_Js, 4],
                           new_data[bright_Js, 6],
                           new_data[bright_Js, 2],
                           cosmo_H_0, cosmo_G,
                           host_rho_0, host_r_s,
                           singular_case=True)
                       < R_s(new_data[bright_Js, 4],
                             new_data[bright_Js, 6],
                             cosmo_H_0)) \
                        and (new_data[bright_Js, 0] > 1.):
                    print('broken Js')
                    print(new_data[bright_Js, :])

                    new_data[bright_Js, 0] = 0.
                    bright_Js = np.argmax(new_data[:, 0])

                brightest_Js[
                repop_num_brightest + new_sub, :] = new_data[
                    bright_Js, [0, 2, 3, 4, 5, 6]]
                new_data[bright_Js, 0] = 0.

            for new_sub in range(repop_num_brightest):

                bright_J03 = np.argmax(new_data[:, 1])

                while (R_t(new_data[bright_J03, 4],
                           new_data[bright_J03, 6],
                           new_data[bright_J03, 2],
                           cosmo_H_0, cosmo_G,
                           host_rho_0, host_r_s,
                           singular_case=True)
                       < R_s(new_data[bright_J03, 4],
                             new_data[bright_J03, 6],
                             cosmo_H_0)) \
                        and (new_data[bright_J03, 1] > 1.):
                    print('broken J03')
                    print(new_data[bright_J03, :])

                    new_data[bright_J03, 1] = 0.
                    bright_J03 = np.argmax(new_data[:, 1])

                brightest_J03[
                repop_num_brightest + new_sub, :] = new_data[bright_J03, 1:]
                new_data[bright_J03, 1] = 0.

            # if sum(new_data[:, 0]) > 1.:
            #     for new_sub in range(repop_num_brightest):
            #
            #         while sum(new_data[:, 0]) > 1.:
            #
            #             bright_Js = np.argmax(new_data[:, 0])
            #
            #             brightest_Js[
            #             repop_num_brightest + new_sub, :] = new_data[
            #                 bright_Js, [0, 2, 3, 4, 5, 6]]
            #             new_data[bright_Js, 0] = 0.

            # while (R_t(new_data[bright_Js, 4],
            #                new_data[bright_Js, 6],
            #                new_data[bright_Js, 2],
            #                cosmo_H_0, cosmo_G,
            #                host_rho_0, host_r_s,
            #                singular_case=True)
            #            < R_s(new_data[bright_Js, 4],
            #                  new_data[bright_Js, 6],
            #                  cosmo_H_0)):
            #         print('subhalo broken (Js)')
            # progress = open(pathname + '/progress_' +
            #                 sim_type + '_'
            #                 + str(res_string)
            #                 + '_results.txt', 'a')
            # progress.write('subhalo broken (Js)'
            #                + str(new_data[bright_Js, :])
            #                + str(R_t(new_data[bright_Js, 4],
            #                          new_data[bright_Js, 6],
            #                          new_data[bright_Js, 2],
            #                          cosmo_H_0, cosmo_G,
            #                          host_rho_0, host_r_s,
            #                          singular_case=True))
            #                + '  '
            #                + str(R_s(new_data[bright_Js, 4],
            #                          new_data[bright_Js, 6],
            #                          cosmo_H_0))
            #                + '\n')
            # progress.write(bright_Js + '\n')
            # new_data[bright_Js, 0] = 0.
            # bright_Js = np.argmax(new_data[:, 0])
            # progress.write('Js  ' + bright_Js + '\n')
            # progress.close()

            # if sum(new_data[:, 1]) > 1.:
            #     for new_sub in range(repop_num_brightest):
            #         while sum(new_data[:, 1]) > 1.:
            #             bright_J03 = np.argmax(new_data[:, 1])
            #
            #             brightest_J03[
            #             repop_num_brightest + new_sub, :] = new_data[bright_J03, 1:]
            #             new_data[bright_J03, 1] = 0.

            # while (R_t(new_data[bright_J03, 4],
            #            new_data[bright_J03, 6],
            #            new_data[bright_J03, 2],
            #            cosmo_H_0, cosmo_G,
            #            host_rho_0, host_r_s,
            #            singular_case=True)
            #        < R_s(new_data[bright_J03, 4],
            #              new_data[bright_J03, 6],
            #              cosmo_H_0)):
            #     print('subhalo broken (J03)')
            # progress = open(pathname + '/progress_' +
            #                 sim_type + '_'
            #                 + str(res_string)
            #                 + '_results.txt', 'a')
            # progress.write('subhalo broken (J03)'
            #                + str(new_data[bright_J03, :])
            #                + str(R_t(new_data[bright_J03, 4],
            #                          new_data[bright_J03, 6],
            #                          new_data[bright_J03, 2],
            #                          cosmo_H_0, cosmo_G,
            #                          host_rho_0, host_r_s,
            #                          singular_case=True))
            #                + '  '
            #                + str(R_s(new_data[bright_J03, 4],
            #                          new_data[bright_J03, 6],
            #                          cosmo_H_0))
            #                + '\n')
            # progress.write(bright_J03 + '\n')
            # new_data[bright_J03, 1] = 0.
            # bright_J03 = np.argmax(new_data[:, 1])
            # progress.write('J03 ' + bright_J03 + '\n')
            # progress.close()

            # We take the brightest subhalos only
            brightest_Js = brightest_Js[np.argsort(brightest_Js[:, 0])[::-1],
                           :]
            brightest_J03 = brightest_J03[
                            np.argsort(brightest_J03[:, 0])[::-1], :]

        else:
            aa = SHVF_Grand2012_int(m_min, m_max, SHVF_bb, SHVF_mm)
            brightest_Js[nn:nn + aa, :] = new_data[:, [0, 2, 3, 4, 5, 6]]
            brightest_J03[nn:nn + aa, :] = new_data[:, 1:]
            nn += aa

        m_min = new_mmin

    if not full_repop:
        return (brightest_Js[:repop_num_brightest, :],
                brightest_J03[:repop_num_brightest, :])

    else:
        brightest_Js[:, 0] *= (
                R_t(brightest_Js[:, 3], brightest_Js[:, 5],
                    brightest_Js[:, 1],
                    cosmo_H_0, cosmo_G,
                    host_rho_0, host_r_s,
                    singular_case=False)
                > R_s(brightest_Js[:, 3],
                      brightest_Js[:, 5],
                      cosmo_H_0))

        brightest_Js = brightest_Js[:nn, :]
        brightest_J03 = brightest_J03[:nn, :]

        brightest_Js = brightest_Js[
                       np.argsort(brightest_Js[:, 0])[::-1], :]
        brightest_J03 = brightest_J03[
                        np.argsort(brightest_J03[:, 0])[::-1], :]

        return (brightest_Js, brightest_J03)


# #@jit(forceobj=True)
# def interior_loop_manybrigthest(
#         num_subs_max,
#         sim_type, res_string,
#         cosmo_G,
#         cosmo_H_0,
#         cosmo_rho_crit,
#
#         host_R_vir,
#         host_rho_0,
#         host_r_s,
#
#         pathname,
#         repop_its,
#         repop_print_freq,
#         repop_inc_factor,
#         repop_num_brightest,
#
#         SHVF_cts_RangeMin,
#         SHVF_cts_RangeMax,
#         SHVF_bb,
#         SHVF_mm,
#
#         Cv_bb,
#         Cv_mm,
#         Cv_sigma,
#
#         srd_args_repop,
#         srd_args_visible,
#         srd_last_sub,
#         Vmax_completion):
#     # We have 6 variables we want to save in our files,
#     # change this number if necessary
#     # (output from 'calculate_characteristics_subhalo()')
#     brightest_Js = np.zeros((repop_num_brightest, 6))
#     brightest_J03 = np.zeros((repop_num_brightest, 6))
#
#     # We calculate our subhalo population in bins to save memory
#     m_min = SHVF_cts_RangeMin
#
#     while m_min < SHVF_cts_RangeMax:
#
#         if SHVF_Grand2012_int(m_min, m_min * repop_inc_factor,
#                               SHVF_bb, SHVF_mm) > num_subs_max:
#
#             m_max = newton(xx, m_min,
#                            args=[m_min, SHVF_bb, SHVF_mm, num_subs_max])
#             new_mmin = m_max
#
#         else:
#             m_max = np.minimum(m_min * repop_inc_factor,
#                                SHVF_cts_RangeMax)
#             new_mmin = m_min * repop_inc_factor
#
#         repop_Vmax = montecarlo_algorithm(
#             m_min, m_max,
#             SHVF_Grand2012,
#             num_subhalos=SHVF_Grand2012_int(m_min, m_max, SHVF_bb, SHVF_mm),
#             sim_type=sim_type,
#             res_string=res_string,
#
#             cosmo_G=cosmo_G,
#             cosmo_H_0=cosmo_H_0,
#             cosmo_rho_crit=cosmo_rho_crit,
#
#             host_R_vir=host_R_vir,
#             host_rho_0=host_rho_0,
#             host_r_s=host_r_s,
#
#             pathname=pathname,
#             repop_its=repop_its,
#             repop_print_freq=repop_print_freq,
#             repop_inc_factor=repop_inc_factor,
#
#             SHVF_cts_RangeMin=SHVF_cts_RangeMin,
#             SHVF_cts_RangeMax=SHVF_cts_RangeMax,
#             SHVF_bb=SHVF_bb,
#             SHVF_mm=SHVF_mm,
#
#             Cv_bb=Cv_bb,
#             Cv_mm=Cv_mm,
#             Cv_sigma=Cv_sigma,
#
#             srd_args_repop=srd_args_repop,
#             srd_args_visible=srd_args_visible,
#             srd_last_sub=srd_last_sub)
#
#         repop_DistGC_lower = montecarlo_algorithm(
#             1e-3, host_R_vir,
#             Nr_Ntot_repop,
#             num_subhalos=SHVF_Grand2012_int(m_min, m_max, SHVF_bb, SHVF_mm),
#             sim_type=sim_type,
#             res_string=res_string,
#
#             cosmo_G=cosmo_G,
#             cosmo_H_0=cosmo_H_0,
#             cosmo_rho_crit=cosmo_rho_crit,
#
#             host_R_vir=host_R_vir,
#             host_rho_0=host_rho_0,
#             host_r_s=host_r_s,
#
#             pathname=pathname,
#             repop_its=repop_its,
#             repop_print_freq=repop_print_freq,
#             repop_inc_factor=repop_inc_factor,
#
#             SHVF_cts_RangeMin=SHVF_cts_RangeMin,
#             SHVF_cts_RangeMax=SHVF_cts_RangeMax,
#             SHVF_bb=SHVF_bb,
#             SHVF_mm=SHVF_mm,
#
#             Cv_bb=Cv_bb,
#             Cv_mm=Cv_mm,
#             Cv_sigma=Cv_sigma,
#
#             srd_args_repop=srd_args_repop,
#             srd_args_visible=srd_args_visible,
#             srd_last_sub=srd_last_sub)
#
#         repop_DistGC_upper = montecarlo_algorithm(
#             1e-3, host_R_vir,
#             Nr_Ntot_visible,
#             num_subhalos=SHVF_Grand2012_int(m_min, m_max, SHVF_bb, SHVF_mm),
#             sim_type=sim_type,
#             res_string=res_string,
#
#             cosmo_G=cosmo_G,
#             cosmo_H_0=cosmo_H_0,
#             cosmo_rho_crit=cosmo_rho_crit,
#
#             host_R_vir=host_R_vir,
#             host_rho_0=host_rho_0,
#             host_r_s=host_r_s,
#
#             pathname=pathname,
#             repop_its=repop_its,
#             repop_print_freq=repop_print_freq,
#             repop_inc_factor=repop_inc_factor,
#
#             SHVF_cts_RangeMin=SHVF_cts_RangeMin,
#             SHVF_cts_RangeMax=SHVF_cts_RangeMax,
#             SHVF_bb=SHVF_bb,
#             SHVF_mm=SHVF_mm,
#
#             Cv_bb=Cv_bb,
#             Cv_mm=Cv_mm,
#             Cv_sigma=Cv_sigma,
#
#             srd_args_repop=srd_args_repop,
#             srd_args_visible=srd_args_visible,
#             srd_last_sub=srd_last_sub)
#
#         repop_DistGC = (repop_DistGC_lower * (repop_Vmax <= Vmax_completion)
#                         + repop_DistGC_upper * (repop_Vmax > Vmax_completion))
#
#         new_data = calculate_characteristics_subhalo(
#             repop_Vmax, repop_DistGC,
#             sim_type=sim_type,
#             res_string=res_string,
#
#             cosmo_G=cosmo_G,
#             cosmo_H_0=cosmo_H_0,
#             cosmo_rho_crit=cosmo_rho_crit,
#
#             host_R_vir=host_R_vir,
#             host_rho_0=host_rho_0,
#             host_r_s=host_r_s,
#
#             pathname=pathname,
#             repop_its=repop_its,
#             repop_print_freq=repop_print_freq,
#             repop_inc_factor=repop_inc_factor,
#             repop_num_brightest=repop_num_brightest,
#
#             SHVF_cts_RangeMin=SHVF_cts_RangeMin,
#             SHVF_cts_RangeMax=SHVF_cts_RangeMax,
#             SHVF_bb=SHVF_bb,
#             SHVF_mm=SHVF_mm,
#
#             Cv_bb=Cv_bb,
#             Cv_mm=Cv_mm,
#             Cv_sigma=Cv_sigma,
#
#             srd_args_repop=srd_args_repop,
#             srd_args_visible=srd_args_visible,
#             srd_last_sub=srd_last_sub)
#
#         # We take the brightest subhalos only
#         brightest_Js = np.append(brightest_Js,
#                                  new_data[:, [0, 2, 3, 4, 5, 6]],
#                                  axis=0)
#
#         brightest_Js = brightest_Js[np.argsort(brightest_Js[:, 0])[::-1], :]
#         brightest_Js = brightest_Js[:repop_num_brightest, :]
#
#         # Same for J03
#         brightest_J03 = np.append(brightest_J03, new_data[:, 1:], axis=0)
#
#         brightest_J03 = brightest_J03[np.argsort(
#             brightest_J03[:, 0])[::-1], :]
#         brightest_J03 = brightest_J03[:repop_num_brightest, :]
#
#         m_min = new_mmin
#
#     return brightest_Js, brightest_J03


def repopulation_bin_by_bin(num_subs_max, sim_type, res_string,
                            cosmo_G,
                            cosmo_H_0,
                            cosmo_rho_crit,

                            host_R_vir,
                            host_rho_0,
                            host_r_s,

                            pathname,
                            repop_its,
                            repop_print_freq,
                            repop_inc_factor,
                            repop_num_brightest,

                            SHVF_cts_RangeMin,
                            SHVF_cts_RangeMax,
                            SHVF_bb,
                            SHVF_mm,

                            Cv_bb,
                            Cv_mm,
                            Cv_sigma,

                            srd_args_repop,
                            srd_args_visible,
                            srd_last_sub,
                            Vmax_completion):
    headerS = (('#\n# Vmin: [' + str(SHVF_cts_RangeMin) + ', '
                + str(SHVF_cts_RangeMax) + '], '
                + str(res_string) + '; '
                + str(repop_its)
                + ' iterations, ' + str(repop_num_brightest)
                + ' brightest\n# Read the individual iterations with: '
                  'np.loadtxt().reshape('
                + str(repop_its) + ', '
                + str(repop_num_brightest) + ', '
                + str(6) + ')\n'
                           '# Js (<r_s) (GeV^2 cm^-5)'
                           '       Dgc (kpc)'
                           '           D_Earth (kpc)'
                           '               Vmax (km/s)'
                           '                ang size (deg)'
                           '               Cv \n#\n'))

    header03 = (('#\n# Vmin: [' + str(SHVF_cts_RangeMin) + ', '
                 + str(SHVF_cts_RangeMax) + '], resilient: '
                 + str(res_string) + '; '
                 + str(repop_its) +
                 ' iterations, ' + str(repop_num_brightest)
                 + ' brightest\n# Read the individual iterations with: '
                   'np.loadtxt().reshape('
                 + str(repop_its) + ', '
                 + str(repop_num_brightest) + ', '
                 + str(6) + ')\n'
                            '# J03 (<0.3deg) (GeV^2 cm^-5)'
                            '       Dgc (kpc)'
                            '           D_Earth (kpc)'
                            '               Vmax (km/s)'
                            '               ang size (deg)'
                            '               Cv \n#\n'))

    file_Js = open(pathname + '/Js_' + sim_type + '_'
                   + str(res_string) + '_results.txt', 'w')
    file_J03 = open(pathname + '/J03_' + sim_type + '_'
                    + str(res_string) + '_results.txt', 'w')

    file_Js.write(headerS)
    file_J03.write(header03)

    for it in range(repop_its):

        if it % repop_print_freq == 0:
            print('    %s %s %s: it %d \n' % (
                time.strftime(" %Y-%m-%d %H:%M:%S", time.gmtime()),
                sim_type, res_string, it))
            progress = open(pathname + '/progress_' +
                            sim_type + '_'
                            + str(res_string)
                            + '_results.txt', 'a')
            progress.write(str(sim_type)
                           + ', res: ' + str(res_string)
                           + ', iteration ' + str(it))
            progress.write('        %.3f  %s\n' %
                           (memory_usage_psutil(),
                            time.strftime(" %Y-%m-%d %H:%M:%S",
                                          time.gmtime())))
            progress.close()

        # if repop_num_brightest < 100:
        brightest_Js, brightest_J03 = interior_loop_singularbrightest(
            num_subs_max=num_subs_max,
            sim_type=sim_type,
            res_string=res_string,

            cosmo_G=cosmo_G,
            cosmo_H_0=cosmo_H_0,
            cosmo_rho_crit=cosmo_rho_crit,

            host_R_vir=host_R_vir,
            host_rho_0=host_rho_0,
            host_r_s=host_r_s,

            pathname=pathname,
            repop_its=repop_its,
            repop_print_freq=repop_print_freq,
            repop_inc_factor=repop_inc_factor,
            repop_num_brightest=repop_num_brightest,

            SHVF_cts_RangeMin=SHVF_cts_RangeMin,
            SHVF_cts_RangeMax=SHVF_cts_RangeMax,
            SHVF_bb=SHVF_bb,
            SHVF_mm=SHVF_mm,

            Cv_bb=Cv_bb,
            Cv_mm=Cv_mm,
            Cv_sigma=Cv_sigma,

            srd_args_repop=srd_args_repop,
            srd_args_visible=srd_args_visible,
            srd_last_sub=srd_last_sub,
            Vmax_completion=Vmax_completion)
        # else:
        #     brightest_Js, brightest_J03 = interior_loop_manybrigthest(
        #         num_subs_max=num_subs_max,
        #         sim_type=sim_type,
        #         res_string=res_string,
        #
        #         cosmo_G=cosmo_G,
        #         cosmo_H_0=cosmo_H_0,
        #         cosmo_rho_crit=cosmo_rho_crit,
        #
        #         host_R_vir=host_R_vir,
        #         host_rho_0=host_rho_0,
        #         host_r_s=host_r_s,
        #
        #         pathname=pathname,
        #         repop_its=repop_its,
        #         repop_print_freq=repop_print_freq,
        #         repop_inc_factor=repop_inc_factor,
        #         repop_num_brightest=repop_num_brightest,
        #
        #         SHVF_cts_RangeMin=SHVF_cts_RangeMin,
        #         SHVF_cts_RangeMax=SHVF_cts_RangeMax,
        #         SHVF_bb=SHVF_bb,
        #         SHVF_mm=SHVF_mm,
        #
        #         Cv_bb=Cv_bb,
        #         Cv_mm=Cv_mm,
        #         Cv_sigma=Cv_sigma,
        #
        #         srd_args_repop=srd_args_repop,
        #         srd_args_visible=srd_args_visible,
        #         srd_last_sub=srd_last_sub,
        #         Vmax_completion=Vmax_completion)

        np.savetxt(file_Js, brightest_Js)
        np.savetxt(file_J03, brightest_J03)

    print('End of repop loop: %.3f  %s\n' %
          (memory_usage_psutil(),
           time.strftime(" %Y-%m-%d %H:%M:%S",
                         time.gmtime())))
    file_Js.close()
    file_J03.close()

    return


def main(inputs):
    sim_type = inputs[0]
    res_string = inputs[1]
    path_input = inputs[2]
    path_output = inputs[3]
    print(sim_type, res_string)
    print(path_output)

    # Calculations ----------------#

    input_dict = read_config_file(path_input)

    cosmo_G = input_dict['cosmo_constants']['G']
    cosmo_H_0 = input_dict['cosmo_constants']['H_0']
    cosmo_rho_crit = input_dict['cosmo_constants']['rho_crit']

    host_R_vir = input_dict['host']['R_vir']
    host_rho_0 = float(input_dict['host']['rho_0'])
    host_r_s = input_dict['host']['r_s']

    cv_cts = input_dict['Cv']
    srd_cts = input_dict['SRD']
    SHVF_cts = input_dict['SHVF']

    repopulations = input_dict['repopulations']

    # Save input data in a file in the outputs directory
    file_inputs = open(path_output + '/input_data.yml', 'w')
    input_dict['SRD']['formula']['resilient'] = inspect.getsource(
        N_subs_resilient)
    input_dict['SRD']['formula']['fragile'] = inspect.getsource(
        N_subs_fragile)
    yaml.dump(input_dict, file_inputs,
              default_flow_style=False, allow_unicode=True)
    file_inputs.close()

    repop_its = repopulations['its']
    num_subs_max = int(float(repopulations['num_subs_max']))
    repop_print_freq = repopulations['print_freq']
    repop_num_brightest = int(repopulations['num_brightest'])
    repop_inc_factor = repopulations['inc_factor']

    SHVF_cts_RangeMin = SHVF_cts['RangeMin']
    SHVF_cts_RangeMax = SHVF_cts['RangeMax']

    Vmax_completion = SHVF_cts['Vmax_completion']

    if sim_type == 'dmo':
        SHVF_bb = SHVF_cts['dmo']['bb']
        SHVF_mm = SHVF_cts['dmo']['mm']

        Cv_bb = cv_cts['dmo']['bb']
        Cv_mm = cv_cts['dmo']['mm']
        Cv_sigma = cv_cts['dmo']['sigma']

        srd_args_repop = srd_cts['dmo'][res_string]['args']
        srd_args_visible = srd_cts['dmo']['fragile']['args']
        srd_last_sub = np.asarray(srd_cts['dmo'][res_string]['last_subhalo'])

    if sim_type == 'hydro':
        SHVF_bb = SHVF_cts['hydro']['bb']
        SHVF_mm = SHVF_cts['hydro']['mm']

        Cv_bb = cv_cts['hydro']['bb']
        Cv_mm = cv_cts['hydro']['mm']
        Cv_sigma = cv_cts['hydro']['sigma']

        srd_args_repop = srd_cts['hydro'][res_string]['args']
        srd_args_visible = srd_cts['hydro']['fragile']['args']
        srd_last_sub = srd_cts['hydro'][res_string]['last_subhalo']

    print(SHVF_cts_RangeMin,
          SHVF_cts_RangeMax)
    print('    Max. number of repop subhalos: %i' %
          SHVF_Grand2012_int(
              SHVF_cts_RangeMin,
              SHVF_cts_RangeMax,
              SHVF_bb,
              SHVF_mm)
          )
    repopulation_bin_by_bin(
        num_subs_max=num_subs_max,
        sim_type=sim_type,
        res_string=res_string,

        cosmo_G=cosmo_G,
        cosmo_H_0=cosmo_H_0,
        cosmo_rho_crit=cosmo_rho_crit,

        host_R_vir=host_R_vir,
        host_rho_0=host_rho_0,
        host_r_s=host_r_s,

        pathname=path_output,
        repop_its=repop_its,
        repop_print_freq=repop_print_freq,
        repop_inc_factor=repop_inc_factor,
        repop_num_brightest=repop_num_brightest,

        SHVF_cts_RangeMin=SHVF_cts_RangeMin,
        SHVF_cts_RangeMax=SHVF_cts_RangeMax,
        SHVF_bb=SHVF_bb,
        SHVF_mm=SHVF_mm,

        Cv_bb=Cv_bb,
        Cv_mm=Cv_mm,
        Cv_sigma=Cv_sigma,

        srd_args_repop=srd_args_repop,
        srd_args_visible=srd_args_visible,
        srd_last_sub=srd_last_sub,
        Vmax_completion=Vmax_completion
    )


if __name__ == "__main__":
    print(time.strftime("%d-%m-%Y %H:%M:%S", time.gmtime()))
    main([sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]])
'''

