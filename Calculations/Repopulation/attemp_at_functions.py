import os
import sys
import math
import yaml
import psutil
import inspect
import numpy as np
import time

from scipy.integrate import simpson
from scipy.optimize import newton
from scipy.interpolate import UnivariateSpline


def main(inputs):
    sim_type = inputs[0]
    res_string = inputs[1]
    path_input = inputs[2]
    path_output = inputs[3]
    print(sim_type, res_string)
    print(path_input, path_output)

    # Calculations ----------------#

    with open(path_input, 'r') as stream:
        try:
            data_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    cosmo_G = data_dict['cosmo_constants']['G']
    cosmo_H_0 = data_dict['cosmo_constants']['H_0']
    cosmo_rho_crit = data_dict['cosmo_constants']['rho_crit']

    host_R_vir = data_dict['host']['R_vir']
    host_rho_0 = float(data_dict['host']['rho_0'])
    host_r_s = data_dict['host']['r_s']

    cv_cts = data_dict['Cv']
    srd_cts = data_dict['SRD']
    SHVF_cts = data_dict['SHVF']

    repop_its = data_dict['repopulations']['its']
    num_subs_max = int(float(data_dict['repopulations']['num_subs_max']))
    repop_print_freq = data_dict['repopulations']['print_freq']
    repop_num_brightest = int(data_dict['repopulations']['num_brightest'])
    repop_inc_factor = data_dict['repopulations']['inc_factor']

    SHVF_cts_RangeMin = SHVF_cts['RangeMin']
    SHVF_cts_RangeMax = SHVF_cts['RangeMax']

    Vmax_completion = SHVF_cts['Vmax_completion']

    if sim_type == 'dmo':
        SHVF_bb = SHVF_cts['dmo']['bb']
        SHVF_mm = SHVF_cts['dmo']['mm']

        Cv_norm = cv_cts['dmo']['bb']
        Cv_sigma = cv_cts['dmo']['sigma']

        srd_args_repop = srd_cts['dmo'][res_string]['args']
        srd_args_visible = srd_cts['dmo']['fragile']['args']
        srd_last_sub = srd_cts['dmo'][res_string]['last_subhalo']

    if sim_type == 'hydro':
        SHVF_bb = SHVF_cts['hydro']['bb']
        SHVF_mm = SHVF_cts['hydro']['mm']

        Cv_norm = cv_cts['hydro']['bb']
        Cv_sigma = cv_cts['hydro']['sigma']

        srd_args_repop = srd_cts['hydro'][res_string]['args']
        srd_args_visible = srd_cts['hydro']['fragile']['args']
        srd_last_sub = srd_cts['hydro'][res_string]['last_subhalo']

    print(SHVF_cts_RangeMin, SHVF_cts_RangeMax)

    def memory_usage_psutil():
        # return the memory usage in MB
        process = psutil.Process(os.getpid())
        mem = process.memory_info()[0] / float(10 ** 6)
        return mem

    def ff(c):
        return np.log(1. + c) - c / (1. + c)

    # SHVF --------------------------------------
    def SHVF_Grand2012(V, SHVF_bb=SHVF_bb, SHVF_mm=SHVF_mm):
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

    def SHVF_Grand2012_int(V1, V2, SHVF_bb=SHVF_bb, SHVF_mm=SHVF_mm):
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
        return int(np.ceil(10 ** SHVF_bb / (SHVF_mm + 1) *
                           (V2 ** (SHVF_mm + 1) - V1 ** (SHVF_mm + 1))))

    # ----------- CONCENTRATIONS ----------------------
    def Cv_Mol2021_redshift0(V, c0=1.75e5, c1=-0.90368, c2=0.2749, c3=-0.028):
        # Median subhalo concentration depending on its Vmax and
        # its redshift (here z=0).
        # Moline et al. 2110.02097
        #
        # V - max radial velocity of a bound particle in the subhalo [km/s]
        ci = [c0, c1, c2, c3]
        return ci[0] * (1 + (sum([ci[i + 1] * np.log10(V) ** (i + 1)
                                  for i in range(3)])))

    def Moline21_normalization(V, c0=Cv_norm):
        return Cv_Mol2021_redshift0(V, c0, c1=-0.90368,
                                    c2=0.2749, c3=-0.028)

    def C_Scatt(C, Cv_sigma=Cv_sigma):
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
        scatter = np.random.normal(loc=1, scale=Cv_sigma, size=C.size)
        return C * (10 ** scatter - 1.)
        # return np.random.lognormal(
        #     np.log(C) + Cv_sigma[0], Cv_sigma[1], C.size)

    # ----------- J-FACTORS --------------------------------
    def J_abs_vel(V, D_earth, C, change_units=True,
                  cosmo_G=cosmo_G, cosmo_H_0=cosmo_H_0):
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
        yy = (2.163 ** 3. / D_earth ** 2.
              / (np.log(1. + 2.163) - 2.163 / (1. + 2.163)) ** 2
              * cosmo_H_0 / 12 / np.pi / float(cosmo_G) ** 2
              * np.sqrt(C / 2) * V ** 3
              * 1e-3)

        if change_units:
            yy *= 4.446e6  # GeV ^ 2 cm ^ -5 Msun ^ -2 kpc ^ 5
        return yy

    # @njit
    def Js_vel(V, D_earth, C, change_units=True,
               cosmo_G=cosmo_G, cosmo_H_0=cosmo_H_0):
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

    # @njit
    def J03_vel(V, D_earth, C, change_units=True,
                cosmo_G=cosmo_G, cosmo_H_0=cosmo_H_0):
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
        return (J_abs_vel(V, D_earth, C, change_units=change_units,
                          cosmo_G=cosmo_G, cosmo_H_0=cosmo_H_0)
                * (1 - 1 / (1 + 2.163 * D_earth
                            * np.tan(0.15 * np.pi / 180.)
                            / R_max(V, C, cosmo_H_0)) ** 3))

    # ----------- REPOPULATION ----------------
    def R_max(V, C, cosmo_H_0=cosmo_H_0):
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

    def R_s(V, C, cosmo_H_0=cosmo_H_0):
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

    def R_t(V, C, DistGC, singular_case=True,
            cosmo_H_0=cosmo_H_0, cosmo_G=cosmo_G,
            host_rho_0=host_rho_0, host_r_s=host_r_s):
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

    def Mhost_encapsulated(R, host_rho_0=host_rho_0, host_r_s=host_r_s):
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
                * (np.log((host_r_s + R) / host_r_s) - R / (host_r_s + R)))

    def N_subs_resilient(DistGC, args):
        return args

    def N_subs_fragile(DistGC, args):
        return args[1] * np.exp(args[0] / DistGC)

    def Nr_Ntot_visible(DistGC,
                        srd_args_visible=srd_args_visible,
                        srd_last_sub=srd_last_sub):
        return (N_subs_fragile(DistGC, srd_args_visible)
                * (DistGC >= srd_last_sub))

    def Nr_Ntot_repop(DistGC,
                      res_string=res_string,
                      srd_args_repop=srd_args_repop,
                      srd_last_sub=srd_last_sub):
        """
        Proportion of subhalo numbers at a certain distance of the GC.

        :param DistGC: float or array-like [kpc]
            Distance from the center of the subhalo to the
            Galactic Center (GC).

        :return: float or array-like
            Number of subhalos at a certain distance of the GC.
        """
        if res_string == 'resilient':
            return (N_subs_resilient(DistGC, srd_args_repop)
                    * (DistGC >= srd_last_sub))

        else:
            return (N_subs_fragile(DistGC, srd_args_repop)
                    * (DistGC >= srd_last_sub))

    def mass_from_Vmax(Vmax, Rmax, c200,
                       cosmo_G=cosmo_G):
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
        return (Vmax ** 2 * Rmax / cosmo_G
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
            C200_med.append(newton(def_Cv, 40.0, i))

        return np.array(C200_med)

    def C200_from_Cv_float(Cv):
        """
        Function to find c200 knowing Cv.

        :param Cv: float or array-like
            Cv of subhalo (concentration definition)

        :return: float or array-like
            c200 of subhalo (concentration definition)
        """
        C200_med = newton(def_Cv, 40.0, Cv)

        return C200_med

    def montecarlo_algorithm(x_min, x_max, pdf, num_subhalos):
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
        x = np.logspace(np.log10(x_min), np.log10(x_max),
                        num=2000)

        y = pdf(x)

        cumul = [simpson(y=y[:i], x=x[:i]) for i in range(1, len(x))]
        cumul /= cumul[-1]

        x_mean = (x[1:] + x[:-1]) / 2.

        x_min = ((np.array(cumul) - 1e-8) < 0).argmin() - 1

        spline = UnivariateSpline(cumul[x_min:], x_mean[x_min:],
                                  s=0, k=1, ext=0)

        return spline(np.random.random(num_subhalos))

    def calculate_characteristics_subhalo(Vmax, Distgc):
        # Random distribution of subhalos around the celestial sphere
        num_subs = len(Vmax)
        repop_theta = 2 * math.pi * np.random.random(num_subs)
        repop_phi = math.pi * np.random.random(num_subs)

        # Positions of the subhalos
        repop_Xs = (Distgc * np.cos(repop_theta)
                    * np.sin(repop_phi))
        repop_Ys = (Distgc * np.sin(repop_theta)
                    * np.sin(repop_phi))
        repop_Zs = Distgc * np.cos(repop_phi)

        repop_DistEarth = ((repop_Xs - 8.5) ** 2 + repop_Ys ** 2
                           + repop_Zs ** 2) ** 0.5

        # repop_C = Cv_Grand2012(Vmax, Cv_norm, Cv_mm)
        repop_C = Moline21_normalization(Vmax, c0=Cv_norm)
        repop_C = C_Scatt(repop_C, Cv_sigma)

        repop_Js = Js_vel(Vmax, repop_DistEarth, repop_C)
        repop_J03 = J03_vel(Vmax, repop_DistEarth, repop_C)

        if res_string == 'fragile' and (repop_num_brightest > 100):
            roche = (R_t(Vmax, repop_C, Distgc,
                         singular_case=False)
                     < R_s(Vmax, repop_C))[0]

            repop_Js[roche] = 0.
            repop_J03[roche] = 0.

        # Angular size of subhalos (up to R_s)
        repop_Theta = 180 / np.pi * np.arctan(
            R_s(Vmax, repop_C) / repop_DistEarth)

        return np.column_stack((repop_Js, repop_J03, Distgc, repop_DistEarth,
                                Vmax, repop_Theta, repop_C))

    def xx(mmax, mmin, SHVF_bb, SHVF_mm, root):
        return SHVF_Grand2012_int(mmin, mmax, SHVF_bb, SHVF_mm) - root

    def interior_loop_singularbrightest():
        # We have 6 variables we want to save in our files,
        # change this number if necessary
        # (output from 'calculate_characteristics_subhalo()')
        brightest_Js = np.zeros((2 * repop_num_brightest, 6))
        brightest_J03 = np.zeros((2 * repop_num_brightest, 6))

        # We calculate our subhalo population in bins to save memory
        m_min = SHVF_cts_RangeMin

        while m_min < SHVF_cts_RangeMax:

            if SHVF_Grand2012_int(m_min, SHVF_cts_RangeMax,
                                  SHVF_bb, SHVF_mm) > num_subs_max:

                m_max = newton(xx, m_min,
                               args=[m_min, SHVF_bb, SHVF_mm, num_subs_max])
                new_mmin = m_max

            else:
                m_max = np.minimum(m_min * repop_inc_factor, SHVF_cts_RangeMax)
                new_mmin = m_min * repop_inc_factor

            repop_Vmax = montecarlo_algorithm(
                m_min, m_max,
                SHVF_Grand2012,
                num_subhalos=SHVF_Grand2012_int(m_min, m_max))

            repop_DistGC_lower = montecarlo_algorithm(
                1e-3, host_R_vir,
                Nr_Ntot_repop,
                num_subhalos=SHVF_Grand2012_int(m_min, m_max))

            repop_DistGC_upper = montecarlo_algorithm(
                1e-3, host_R_vir,
                Nr_Ntot_visible,
                num_subhalos=SHVF_Grand2012_int(m_min, m_max))

            repop_DistGC = (
                    repop_DistGC_lower * (repop_Vmax <= Vmax_completion)
                    + repop_DistGC_upper * (repop_Vmax > Vmax_completion))

            new_data = calculate_characteristics_subhalo(
                repop_Vmax, repop_DistGC)

            for new_sub in range(repop_num_brightest):

                bright_Js = np.argmax(new_data[:, 0])
                bright_J03 = np.argmax(new_data[:, 1])

                if res_string == 'fragile':
                    while (R_t(new_data[bright_Js, 4],
                               new_data[bright_Js, 6],
                               new_data[bright_Js, 2],
                               singular_case=True)
                           < R_s(new_data[bright_Js, 4],
                                 new_data[bright_Js, 6])):
                        print('subhalo broken (Js)')
                        progress = open(path_output + '/progress_' +
                                        sim_type + '_'
                                        + str(res_string)
                                        + '_results.txt', 'a')
                        progress.write('subhalo broken (Js)'
                                       + str(new_data[bright_Js, :])
                                       + str(R_t(new_data[bright_Js, 4],
                                                 new_data[bright_Js, 6],
                                                 new_data[bright_Js, 2],
                                                 singular_case=True))
                                       + str(R_s(new_data[bright_Js, 4],
                                                 new_data[bright_Js, 6]))
                                       + '\n')
                        # progress.write(bright_Js + '\n')
                        new_data[bright_Js, 0] = 0.
                        bright_Js = np.argmax(new_data[:, 0])
                        # progress.write(bright_Js + '\n')
                        progress.close()

                    while (R_t(new_data[bright_J03, 4],
                               new_data[bright_J03, 6],
                               new_data[bright_J03, 2],
                               singular_case=True)
                           < R_s(new_data[bright_J03, 4],
                                 new_data[bright_J03, 6])):
                        print('subhalo broken (J03)')
                        progress = open(path_output + '/progress_' +
                                        sim_type + '_'
                                        + str(res_string)
                                        + '_results.txt', 'a')
                        progress.write('subhalo broken (J03)'
                                       + str(new_data[bright_J03, :])
                                       + str(R_t(new_data[bright_J03, 4],
                                                 new_data[bright_J03, 6],
                                                 new_data[bright_J03, 2],
                                                 singular_case=True))
                                       + str(R_s(new_data[bright_J03, 4],
                                                 new_data[bright_J03, 6]))
                                       + '\n')
                        # progress.write(bright_J03 + '\n')
                        new_data[bright_J03, 1] = 0.
                        bright_J03 = np.argmax(new_data[:, 1])
                        # progress.write(bright_J03 + '\n')
                        progress.close()

                brightest_Js[
                repop_num_brightest + new_sub, :] = new_data[
                    bright_Js, [0, 2, 3, 4, 5, 6]]
                new_data[bright_Js, 0] = 0.

                brightest_J03[
                repop_num_brightest + new_sub, :] = new_data[bright_J03, 1:]
                new_data[bright_J03, 1] = 0.

            # We take the brightest subhalos only
            brightest_Js = brightest_Js[np.argsort(brightest_Js[:, 0])[::-1],
                           :]
            brightest_J03 = brightest_J03[
                            np.argsort(brightest_J03[:, 0])[::-1], :]

            m_min = new_mmin

        return (brightest_Js[:repop_num_brightest, :],
                brightest_J03[:repop_num_brightest, :])

    def interior_loop_manybrigthest():
        # We have 6 variables we want to save in our files,
        # change this number if necessary
        # (output from 'calculate_characteristics_subhalo()')
        brightest_Js = np.zeros((repop_num_brightest, 6))
        brightest_J03 = np.zeros((repop_num_brightest, 6))

        # We calculate our subhalo population in bins to save memory
        m_min = SHVF_cts_RangeMin

        while m_min < SHVF_cts_RangeMax:

            if SHVF_Grand2012_int(m_min, m_min * repop_inc_factor,
                                  SHVF_bb, SHVF_mm) > num_subs_max:

                m_max = newton(xx, m_min,
                               args=[m_min, SHVF_bb, SHVF_mm, num_subs_max])
                new_mmin = m_max

            else:
                m_max = np.minimum(m_min * repop_inc_factor, SHVF_cts_RangeMax)
                new_mmin = m_min * repop_inc_factor

            repop_Vmax = montecarlo_algorithm(
                m_min, m_max,
                SHVF_Grand2012,
                num_subhalos=SHVF_Grand2012_int(m_min, m_max, SHVF_bb,
                                                SHVF_mm))

            repop_DistGC_lower = montecarlo_algorithm(
                1e-3, host_R_vir,
                Nr_Ntot_repop,
                num_subhalos=SHVF_Grand2012_int(m_min, m_max, SHVF_bb,
                                                SHVF_mm))

            repop_DistGC_upper = montecarlo_algorithm(
                1e-3, host_R_vir,
                Nr_Ntot_visible,
                num_subhalos=SHVF_Grand2012_int(m_min, m_max, SHVF_bb,
                                                SHVF_mm))

            repop_DistGC = (
                        repop_DistGC_lower * (repop_Vmax <= Vmax_completion)
                        + repop_DistGC_upper * (repop_Vmax > Vmax_completion))

            new_data = calculate_characteristics_subhalo(
                repop_Vmax, repop_DistGC)

            # We take the brightest subhalos only
            brightest_Js = np.append(brightest_Js,
                                     new_data[:, [0, 2, 3, 4, 5, 6]],
                                     axis=0)

            brightest_Js = brightest_Js[np.argsort(brightest_Js[:, 0])[::-1],
                           :]
            brightest_Js = brightest_Js[:repop_num_brightest, :]

            # Same for J03
            brightest_J03 = np.append(brightest_J03, new_data[:, 1:], axis=0)

            brightest_J03 = brightest_J03[np.argsort(
                brightest_J03[:, 0])[::-1], :]
            brightest_J03 = brightest_J03[:repop_num_brightest, :]

            m_min = new_mmin

        return brightest_Js, brightest_J03

    def repopulation_bin_by_bin():
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

        file_Js = open(path_output + '/Js_' + sim_type + '_'
                       + str(res_string) + '_results.txt', 'w')
        file_J03 = open(path_output + '/J03_' + sim_type + '_'
                        + str(res_string) + '_results.txt', 'w')

        file_Js.write(headerS)
        file_J03.write(header03)

        for it in range(repop_its):

            if it % repop_print_freq == 0:
                print('    %s %s %s: it %d \n' % (
                    time.strftime(" %Y-%m-%d %H:%M:%S", time.gmtime()),
                    sim_type, res_string, it))
                progress = open(path_output + '/progress_' +
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

            if repop_num_brightest < 100:
                brightest_Js, brightest_J03 = interior_loop_singularbrightest()
            else:
                brightest_Js, brightest_J03 = interior_loop_manybrigthest()

            np.savetxt(file_Js, brightest_Js)
            np.savetxt(file_J03, brightest_J03)

        print('End of repop loop: %.3f  %s\n' %
              (memory_usage_psutil(),
               time.strftime(" %Y-%m-%d %H:%M:%S",
                             time.gmtime())))
        file_Js.close()
        file_J03.close()

        return

    print()
    print('    Max. number of repop subhalos: %i'
          % SHVF_Grand2012_int(SHVF_cts_RangeMin, SHVF_cts_RangeMax))

    repopulation_bin_by_bin()

    # Save input data in a file in the outputs directory
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    file_inputs = open(path_output + '/input_data.yml', 'w')
    data_dict['SRD']['formula']['resilient'] = inspect.getsource(
        N_subs_resilient)
    data_dict['SRD']['formula']['fragile'] = inspect.getsource(
        N_subs_fragile)
    yaml.dump(data_dict, file_inputs,
              default_flow_style=False, allow_unicode=True)
    file_inputs.close()


if __name__ == "__main__":
    print(time.strftime("%d-%m-%Y %H:%M:%S", time.gmtime()))
    main([sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]])
