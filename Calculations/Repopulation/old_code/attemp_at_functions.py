import os
import math
import yaml
import psutil
import numpy as np
import random as rdm
import time

from numba import njit, jit
from numba.typed import List
from numba.core.errors import NumbaDeprecationWarning, \
    NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


def memory_usage_psutil():
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(10 ** 6)
    return mem


def read_config_file(ConfigFile):
    with open(ConfigFile, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml


# Calculations ----------------#

data_dict = read_config_file('input_files/data_newResSRD.yml')

cosmo_G = data_dict['cosmo_constants']['G']
cosmo_H_0 = data_dict['cosmo_constants']['H_0']
cosmo_rho_crit = data_dict['cosmo_constants']['rho_crit']

host_R_vir = data_dict['host']['R_vir']
host_rho_0 = float(data_dict['host']['rho_0'])
host_r_s = data_dict['host']['r_s']

cv_cts = data_dict['Cv']
srd_cts = data_dict['SRD']
SHVF_cts = data_dict['SHVF']

repopulations = data_dict['repopulations']

path_name = str('outputs/' + repopulations['id']
                + time.strftime(" %Y-%m-%d %H:%M:%S",
                                time.gmtime()))

if not os.path.exists(path_name):
    os.makedirs(path_name)

# Save input data in a file in the outputs directory
file_inputs = open(path_name + '/input_data.yml', 'w')
yaml.dump(data_dict, file_inputs,
          default_flow_style=False, allow_unicode=True)
file_inputs.close()

print(path_name)


def repopulation(sim_types, res_frag):
    print(SHVF_cts_RangeMin,
          SHVF_cts_RangeMax)
    print('    Max. number of repop subhalos: %i' %
          SHVF_Grand2012_int(
              SHVF_cts_RangeMin,
              SHVF_cts_RangeMax)
          )

    computing_bin_by_bin(sim_types, res_frag)

    print(path_name)
    return


@njit()
def ff(c):
    return np.log(1. + c) - c / (1. + c)


# SHVF --------------------------------------
@njit
def SHVF_Grand2012(V, sim_types, res_frag):
    """
    SubHalo Velocity Function (SHVF) - number of subhalos as a
    function of Vmax. Power law formula.
    Definition taken from Grand 2012.07846.

    :param V: float or array-like [km/s]
        Maximum radial velocity of a bound particle in the subhalo.

    :return: float or array-like
        Number of subhalos defined by the Vmax input.
    """
    return (10 ** SHVF_bb
            * V ** SHVF_mm)


@njit
def SHVF_Grand2012_int(V1, V2):
    """
    Integration of the SHVF defined above.
    (numba needed an analytical expression.)

    :param V1: float or array-like [km/s]
        Minimum limit of integration of the SHVF.
    :param V2: float or array-like [km/s]
        Maximum limit of integration of the SHVF.

    :return: float or array-like
        Integrated SHVF.
    """
    return int(np.round(10 ** SHVF_bb
                        / (SHVF_mm + 1) *
                        (V2 ** (SHVF_mm + 1)
                         - V1 ** (SHVF_mm + 1))))


# ----------- CONCENTRATIONS ----------------------
@njit
def Cv_Grand2012(Vmax):
    """
    Calculate the scatter of a subhalo population.
    Based on Grand 2012.07846.

    :param Vmax: float or array-like [km/s]
        Maximum radial velocity of a bound particle in the subhalo.

    :return: float or array-like
        Concentrations of a subhalo population.
    """
    # Concentration based on Grand 2012.07846.
    return (10 ** Cv_bb
            * Vmax ** Cv_mm)


@jit
def C_Scatt(C):
    """
    Create a scatter in the concentration parameter of the
    repopulated population.
    Scatter in logarithmic scale, following a Gauss distribution.

    :param C: float or array-like
        Concentration of a subhalo (according to the concentration
        law).
    :return: float or array-like
        Subhalos with scattered concentrations.
    """
    return np.array([np.random.lognormal(np.log(C[i]), Cv_sigma)
                     for i in range(C.size)])


# ----------- J-FACTORS --------------------------------
@njit
def J_abs_vel(V, D_earth, C, change_units=True):
    """
    Jfactor enclosing whole subhalo as a function of the
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
    yy = (2.163 ** 3. / D_earth ** 2. / ff(2.163) ** 2
          * cosmo_H_0 / 12 / np.pi / float(cosmo_G) ** 2
          * np.sqrt(C / 2) * V ** 3
          * 1e-3)

    if change_units:
        yy *= 4.446e6  # GeV ^ 2 cm ^ -5 Msun ^ -2 kpc ^ 5
    return yy


@njit
def Js_vel(V, D_earth, C, change_units=True):
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
    return J_abs_vel(V, D_earth, C, change_units=change_units) * 7 / 8


@njit
def J03_vel(V, D_earth, C, change_units=True):
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
    return (J_abs_vel(V, D_earth, C, change_units=change_units)
            * (1 - 1 / (1 + 2.163 * D_earth * np.tan(0.15 * np.pi / 180.)
                        / R_max(V, C)) ** 3))


# ----------- REPOPULATION ----------------
@njit
def R_max(V, C):
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


@njit
def R_s(V, C):
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
    return R_max(V, C) / 2.163


@njit
def R_t(V, C, DistGC):
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
    Rmax = R_max(V, C)
    c200 = C200_from_Cv(C)
    M = mass_from_Vmax(V, Rmax, c200)

    return (((M / (3 * Mhost_encapsulated(DistGC))) ** (1. / 3))
            * DistGC)


@njit
def Mhost_encapsulated(R):
    """
    Host mass encapsulated up to a certain radius. We are following
    a NFW density profile for the host.
    
    :param R: float or array-like [kpc]
        Radius up to which we integrate the density profile.
        
    :return: float or array-like [Msun]
        Host mass encapsulated up to R.
    """
    return (4 * math.pi * host_rho_0
            * host_r_s ** 3
            * (np.log((host_r_s + R) / host_r_s)
               - R / (host_r_s + R)))


@njit()
def N_subs_resilient(DistGC, args):
    return DistGC ** args[0] * 10 ** args[1]


@njit()
def N_subs_fragile(DistGC, args):
    return (10 ** ((DistGC / args[0]) ** args[1]
                   * np.exp(-args[2] * (DistGC - args[0]) / args[0])))


@njit
def Nr_Ntot(DistGC, sim_types, res_frag):
    """
    Proportion of subhalo numbers at a certain distance of the GC.

    :param DistGC: float or array-like [kpc]
        Distance from the center of the subhalo to the
        Galactic Center (GC).

    :return: float or array-like
        Number of subhalos at a certain distance of the GC.
    """
    if res_frag == 'resilient':
        if sim_types == 'dmo':
            return N_subs_resilient(DistGC,
                                    [0.54343928, -2.17672138])
        if sim_types == 'hydro':
            return N_subs_resilient(DistGC,
                                    [0.97254648, -3.08584275])

    else:
        if sim_types == 'dmo':
            return (N_subs_fragile(DistGC,
                                   [1011.38716, 0.4037927, 2.35522213])
                    * (DistGC >= srd_last_sub))
        if sim_types == 'hydro':
            return (N_subs_fragile(DistGC,
                                   [666.49179, 0.75291017, 2.90546523])
                    * (DistGC >= srd_last_sub))


@njit
def mass_from_Vmax(Vmax, Rmax, c200):
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
            * ff(c200) / ff(2.163))


@njit
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
    return 200 * ff(2.163) / ff(c200) * (c200 / 2.163) ** 3 - Cv


@njit
def newton(fun, x0, args):
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


@njit
def C200_from_Cv(Cv):
    """
    Function to find c200 knowing Cv.

    :param Cv: float or array-like
        Cv of subhalo (concentration definition)

    :return: float or array-like
        c200 of subhalo (concentration definition)
    """
    if np.shape(Cv) == ():
        C200_med = newton(def_Cv, 40.0, Cv[0])
        C200_med = np.array([C200_med])
    else:
        num = np.shape(Cv)[0]
        C200_med = np.zeros(num)

        for i in range(num):
            # opt.root(def_Cv, 40, args=Cv[i])['x'][0]
            C200_med[i] = newton(def_Cv, 40.0, Cv[i])

    return C200_med


@njit()
def rej_samp(x_min, x_max, pdf, num_subhalos, sim_types, res_frag):
    """
    Rejection sample algorithm. It populates a number of objects
    with a probability distribution defined by the pdf function.

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
    results = np.zeros(num_subhalos)
    i = 0

    # Compute the maximum of the function
    fn_max = np.max(np.array([pdf(i, sim_types, res_frag)
                              for i in np.linspace(x_min, x_max, 1000)]))

    while i < num_subhalos:
        x = rdm.uniform(x_min, x_max)
        h = rdm.uniform(0, fn_max)

        if h < pdf(x, sim_types, res_frag):
            results[i] = x
            i += 1

    return results


@njit
def calculate_characteristics_subhalo(Vmax, Distgc, num_subs):
    # Random distribution of subhalos around the celestial sphere
    repop_theta = 2 * math.pi * np.random.random(num_subs)
    repop_phi = math.pi * np.random.random(num_subs)

    # Positions of th subhalos
    repop_Xs = (Distgc * np.cos(repop_theta)
                * np.sin(repop_phi))
    repop_Ys = (Distgc * np.sin(repop_theta)
                * np.sin(repop_phi))
    repop_Zs = Distgc * np.cos(repop_phi)

    repop_DistEarth = ((repop_Xs - 8.5) ** 2 + repop_Ys ** 2
                       + repop_Zs ** 2) ** 0.5

    repop_C = Cv_Grand2012(Vmax)
    repop_C = C_Scatt(repop_C)

    repop_Js = Js_vel(Vmax, repop_DistEarth, repop_C)
    repop_J03 = J03_vel(Vmax, repop_DistEarth, repop_C)

    if resilient is False:
        roche = (R_t(Vmax, repop_C, Distgc)
                 > R_s(Vmax, repop_C))

        repop_Js *= roche
        repop_J03 *= roche

    # Angular size of subhalos (up to R_s)
    repop_Theta = 180 / np.pi * np.arctan(
        R_s(Vmax, repop_C) / repop_DistEarth)

    return np.column_stack((repop_Js, repop_J03, Distgc, repop_DistEarth,
                            Vmax, repop_Theta, repop_C))


def computing_bin_by_bin(sim_types, res_frag):
    headerS = (('#\n# Vmin: [' + str(SHVF_cts_RangeMin) + ', '
                + str(SHVF_cts_RangeMax) + '], resilient: '
                + str(resilient) + '; '
                + str(repop_its) +
                (' iterations \n# Js (<r_s) (GeV^2 cm^-5)'
                 '       Dgc (kpc)'
                 '           D_Earth (kpc)'
                 '               Vmax (km/s)'
                 '                ang size (deg)'
                 '               Cv \n#\n')))

    header03 = (('#\n# Vmin: [' + str(SHVF_cts_RangeMin) + ', '
                 + str(SHVF_cts_RangeMax) + '], resilient: '
                 + str(resilient) + '; '
                 + str(repop_its) +
                 (' iterations \n# J03 (<0.3deg) (GeV^2 cm^-5)'
                  '       Dgc (kpc)'
                  '           D_Earth (kpc)'
                  '               Vmax (km/s)'
                  '               ang size (deg)'
                  '               Cv \n#\n')))

    file_Js = open(path_name + '/Js_' + sim_type + '_Res'
                   + str(resilient) + '_results.txt', 'w')
    file_J03 = open(path_name + '/J03_' + sim_type + '_Res'
                    + str(resilient) + '_results.txt', 'w')

    file_Js.write(headerS)
    file_J03.write(header03)

    for it in range(repop_its):

        # We have 6 variables we want to save in our files,
        # change this number if necessary
        brightest_Js = np.zeros([6, repop_num_brightest])
        brightest_J03 = np.zeros([6, repop_num_brightest])

        if it % repop_print_freq == 0:
            print('    ', sim_type, ', res: ', resilient,
                  ', iteration ', it)
            print('        %.3f' % memory_usage_psutil())
            progress = open(path_name + '/progress' +
                            sim_type + '_Res'
                            + str(resilient)
                            + '_results.txt', 'a')
            progress.write(str(sim_type)
                           + ', res: ' + str(resilient)
                           + ', iteration ' + str(it))
            progress.write('        %.3f  %s\n' %
                           (memory_usage_psutil(),
                            time.strftime(" %Y-%m-%d %H:%M:%S",
                                          time.gmtime())))
            progress.close()

        # We calculate our subhalo population one by one in order to
        # save memory
        m_min = SHVF_cts_RangeMin
        ceil = int(np.ceil(np.log(SHVF_cts_RangeMax
                                  / SHVF_cts_RangeMin)
                           / np.log(repop_inc_factor)))

        for num_bins in range(ceil):
            m_max = np.minimum(m_min * repop_inc_factor,
                               SHVF_cts_RangeMax)

            num_subhalos = SHVF_Grand2012_int(m_min, m_max)

            repop_Vmax = rej_samp(m_min, m_max,
                                  SHVF_Grand2012,
                                  num_subhalos=num_subhalos,
                                  sim_types=sim_types,
                                  res_frag=res_frag)
            repop_DistGC = rej_samp(0, host_R_vir,
                                    Nr_Ntot,
                                    num_subhalos=num_subhalos,
                                    sim_types=sim_types,
                                    res_frag=res_frag)

            new_data = calculate_characteristics_subhalo(
                repop_Vmax, repop_DistGC, num_subhalos)

            bright_Js = np.where(
                np.max(new_data[:, 0]) == new_data[:, 0])[0][0]

            if new_data[bright_Js, 0] > brightest_Js[0, 0]:
                brightest_Js[0, 0] = new_data[bright_Js, 0]
                brightest_Js[1:, 0] = new_data[bright_Js, 2:]

            bright_J03 = np.where(
                np.max(new_data[:, 0]) == new_data[:, 0])[0][0]
            if new_data[bright_J03, 1] > brightest_J03[0, 0]:
                brightest_J03[0, 0] = new_data[bright_J03, 1]
                brightest_J03[1:, 0] = new_data[bright_J03, 2:]

            if it == 0:
                print('   ', m_min, m_max, num_subhalos)
                print('        %.3f' % memory_usage_psutil())
                progress = open(path_name + '/progress' +
                                sim_type + '_Res'
                                + str(resilient)
                                + '_results.txt', 'a')
                progress.write('   %.3f - %.3f %s'
                               % (m_min, m_max, num_subhalos))
                progress.write('        %.3f  %s\n' %
                               (memory_usage_psutil(),
                                time.strftime(" %Y-%m-%d %H:%M:%S",
                                              time.gmtime())))
                progress.close()

            m_min *= repop_inc_factor

        np.savetxt(file_Js, brightest_Js.T)
        np.savetxt(file_J03, brightest_J03.T)

    print('        %.3f' % memory_usage_psutil())
    file_Js.close()
    file_J03.close()

    return


sim_type = None
resilient = None

for each in data_dict['repopulations']['type']:
    print(data_dict['repopulations']['type'][each])
    sim_type = data_dict['repopulations']['type'][each][0]
    resilient = data_dict['repopulations']['type'][each][1]

    print(sim_type, resilient)

    repop_its = repopulations['its']
    repop_id = repopulations['id']
    repop_print_freq = repopulations['print_freq']
    repop_num_brightest = repopulations['num_brightest']
    repop_inc_factor = repopulations['inc_factor']

    SHVF_cts_RangeMin = SHVF_cts['RangeMin']
    SHVF_cts_RangeMax = SHVF_cts['RangeMax']

    if resilient:
        res_string = 'resilient'
    else:
        res_string = 'fragile'

    if sim_type == 'dmo':
        SHVF_bb = SHVF_cts['dmo']['bb']
        SHVF_mm = SHVF_cts['dmo']['mm']

        Cv_bb = cv_cts['dmo']['bb']
        Cv_mm = cv_cts['dmo']['mm']
        Cv_sigma = cv_cts['dmo']['sigma']

        srd_args = srd_cts['dmo'][res_string]['args']
        srd_args = List(srd_args)
        srd_last_sub = srd_cts['dmo'][res_string]['last_subhalo']

    if sim_type == 'hydro':
        SHVF_bb = SHVF_cts['hydro']['bb']
        SHVF_mm = SHVF_cts['hydro']['mm']

        Cv_bb = cv_cts['hydro']['bb']
        Cv_mm = cv_cts['hydro']['mm']
        Cv_sigma = cv_cts['hydro']['sigma']

        srd_args = srd_cts['hydro'][res_string]['args']
        srd_last_sub = srd_cts['hydro'][res_string]['last_subhalo']

    repopulation(sim_type, res_string)

print(time.strftime(" %d-%m-%Y %H:%M:%S", time.gmtime()))
