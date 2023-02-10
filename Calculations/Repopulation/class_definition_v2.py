import os
import math
import yaml
import numpy as np
import random as rdm
import time

from scipy import integrate


def memory_usage_psutil():
    # return the memory usage in MB
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(10 ** 6)
    return mem


class Jfact_calculation(object):

    def __init__(self, data_dict):

        self._cosmo = data_dict['cosmo_constants']
        self._host_cts = data_dict['host']

        self._cv_cts = data_dict['Cv']
        self._srd_cts = data_dict['SRD']
        self._SHVF_cts = data_dict['SHVF']

        self._repopulations = data_dict['repopulations']

        self._sim_type = None
        self._resilient = None

        self.path_name = str('outputs/' + self._repopulations['id']
                             + time.strftime(" %Y-%m-%d %H:%M:%S",
                                             time.gmtime()))

        if not os.path.exists(self.path_name):
            os.makedirs(self.path_name)

        # Save input data in a file in the outputs directory
        file_inputs = open(self.path_name + '/input_data.yml', 'w')
        yaml.dump(data_dict, file_inputs,
                  default_flow_style=False, allow_unicode=True)
        file_inputs.close()

        print(self.path_name)
        return

    def repopulation(self, new_sim_types, type_loop):
        self._sim_type = new_sim_types[0]
        self._resilient = new_sim_types[1]
        print(self._SHVF_cts['RangeMin'],
              self._SHVF_cts['RangeMax'])
        print('    Max. number of repop subhalos: %i' %
              np.round(integrate.quad(
                  self.SHVF_Grand2012,
                  self._SHVF_cts['RangeMin'],
                  self._SHVF_cts['RangeMax'])[0])
              )

        if type_loop == 'all_at_once':
            self.computing()

        elif type_loop == 'one_by_one':
            self.computing_one_by_one()

        elif type_loop == 'bin_by_bin':
            self.computing_bin_by_bin()

        else:
            print('No method for compution input: repopulation cancelled.')

        print(self.path_name)
        return

    @staticmethod
    def ff(c):
        return np.log(1. + c) - c / (1. + c)

    # SHVF --------------------------------------
    def SHVF_Grand2012(self, V):
        """
        SubHalo Velocity Function (SHVF) - number of subhalos as a
        function of Vmax. Power law formula.
        Definition taken from Grand 2012.07846.

        :param V: float or array-like [km/s]
            Maximum radial velocity of a bound particle in the subhalo.

        :return: float or array-like
            Number of subhalos defined by the Vmax input.
        """
        return (10 ** self._SHVF_cts[self._sim_type]['bb']
                * V ** self._SHVF_cts[self._sim_type]['mm'])

    def SHVF_Grand2012_int(self, V1, V2):
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
        return np.round(10 ** self._SHVF_cts[self._sim_type]['bb']
                        / (self._SHVF_cts[self._sim_type]['mm'] + 1) *
                        (V2 ** (self._SHVF_cts[self._sim_type]['mm'] + 1)
                         - V1 ** (self._SHVF_cts[self._sim_type]['mm'] + 1)))

    # ----------- CONCENTRATIONS ----------------------

    def Cv_Grand2012(self, Vmax):
        """
        Calculate the scatter of a subhalo population.
        Based on Grand 2012.07846.

        :param Vmax: float or array-like [km/s]
            Maximum radial velocity of a bound particle in the subhalo.

        :return: float or array-like
            Concentrations of a subhalo population.
        """
        # Concentration based on Grand 2012.07846.
        return (10 ** self._cv_cts[self._sim_type]['bb']
                * Vmax ** self._cv_cts[self._sim_type]['mm'])

    def C_Scatt(self, C):
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
        if np.size(C) == 1:
            return np.random.lognormal(
                np.log(C), self._cv_cts[self._sim_type]['sigma'])

        else:
            return np.array([np.random.lognormal(
                np.log(C[i]), self._cv_cts[self._sim_type]['sigma'])
                for i in range(np.size(C))])

    # ----------- J-FACTORS --------------------------------

    def J_abs_vel(self, V, D_earth, C, change_units=True):
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
        yy = (2.163 ** 3. / D_earth ** 2. / self.ff(2.163) ** 2
              * self._cosmo['H_0'] / 12 / np.pi / float(self._cosmo['G']) ** 2
              * np.sqrt(C / 2) * V ** 3
              * 1e-3)

        if change_units:
            yy *= 4.446e6  # GeV ^ 2 cm ^ -5 Msun ^ -2 kpc ^ 5
        return yy

    def Js_vel(self, V, D_earth, C, change_units=True):
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
        return self.J_abs_vel(V, D_earth, C, change_units=change_units) * 7 / 8

    def J03_vel(self, V, D_earth, C, change_units=True):
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
        return (self.J_abs_vel(V, D_earth, C, change_units=change_units)
                * (1 - 1 / (1 + 2.163 * D_earth * np.tan(0.15 * np.pi / 180.)
                            / self.R_max(V, C)) ** 3))

    # ----------- REPOPULATION ----------------

    def R_max(self, V, C):
        """
        Calculate Rmax of a subhalo.

        :param V: float or array-like [km/s]
            Maximum circular velocity inside a subhalo.
        :param C: float or array-like
            Subhalo concentration.

        :return: float or array-like [kpc]
            Rmax of the subhalo given by the inputs.
        """
        return V / self._cosmo['H_0'] * np.sqrt(2. / C) * 1e3

    def R_s(self, V, C):
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
        return self.R_max(V, C) / 2.163

    def R_t(self, V, C, DistGC):
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
        Rmax = self.R_max(V, C)
        c200 = self.C200_from_Cv(C)
        M = self.mass_from_Vmax(V, Rmax, c200)

        return (((M / (3 * self.Mhost_encapsulated(DistGC))) ** (1. / 3))
                * DistGC)

    def Mhost_encapsulated(self, R):
        """
        Host mass encapsulated up to a certain radius. We are following
        a NFW density profile for the host.
        
        :param R: float or array-like [kpc]
            Radius up to which we integrate the density profile.
            
        :return: float or array-like [Msun]
            Host mass encapsulated up to R.
        """
        return (4 * math.pi * float(self._host_cts['rho_0'])
                * self._host_cts['r_s'] ** 3
                * (np.log((self._host_cts['r_s'] + R) / self._host_cts['r_s'])
                   - R / (self._host_cts['r_s'] + R)))

    def N_Dgc_Cosmic(self, str_function, DistGC, args):
        """
        Number of subhalos (NOT DM DENSITY) at a distance to the GC.
        The inputs are being used, even if the function does
        not recognize them.
        them.

        :param str_function: str
            Formula of the SRD.
        :param DistGC: float or array-like [kpc]
            Distance from the center of the subhalo to the
            Galactic Center (GC).
        :param args: float or array-like
            Arguments passed to the SRD formula.

        :return:
            eval() formula of the SRD.
        """
        return eval(str_function)  # )

    def Nr_Ntot(self, DistGC):
        """
        Proportion of subhalo numbers at a certain distance of the GC.

        :param DistGC: float or array-like [kpc]
            Distance from the center of the subhalo to the
            Galactic Center (GC).

        :return: float or array-like
            Number of subhalos at a certain distance of the GC.
        """
        if self._resilient:
            return self.N_Dgc_Cosmic(self._srd_cts['formula']['resilient'],
                                     DistGC,
                                     self._srd_cts[self._sim_type][
                                         'resilient']['args'])
        else:
            return (self.N_Dgc_Cosmic(self._srd_cts['formula']['fragile'],
                                      DistGC,
                                      self._srd_cts[self._sim_type][
                                          'fragile']['args'])
                    * (DistGC >= self._srd_cts[self._sim_type]['fragile'][
                        'last_subhalo']))

    def mass_from_Vmax(self, Vmax, Rmax, c200):
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
        return (Vmax ** 2 * Rmax / float(self._cosmo['G'])
                * self.ff(c200) / self.ff(2.163))

    def def_Cv(self, c200, Cv):
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
        return 200 * self.ff(2.163) / self.ff(c200) * (c200 / 2.163) ** 3 - Cv

    @staticmethod
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
            x -= fun(x, args) * 0.02 / (fun(x + 0.01, args)
                                        - fun(x - 0.01, args))
        return x

    def C200_from_Cv(self, Cv):
        """
        Function to find c200 knowing Cv.

        :param Cv: float or array-like
            Cv of subhalo (concentration definition)

        :return: float or array-like
            c200 of subhalo (concentration definition)
        """
        if np.shape(Cv) == ():
            C200_med = self.newton(self.def_Cv, 40.0, Cv)
            C200_med = np.array([C200_med])
        else:
            num = np.shape(Cv)[0]
            C200_med = np.zeros(num)

            for i in range(num):
                # opt.root(def_Cv, 40, args=Cv[i])['x'][0]
                C200_med[i] = self.newton(self.def_Cv, 40.0, Cv[i])

        return C200_med

    def rej_samp(self, x_min, x_max, pdf, num_subhalos):
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
        fn_max = np.max(pdf(np.linspace(x_min, x_max, 1000)))

        while i < num_subhalos:
            x = rdm.uniform(x_min, x_max)
            h = rdm.uniform(0, fn_max)

            if h < pdf(x):
                results[i] = x
                i += 1

        return results

    def expected_number(self, M1, M2):
        """
        Expected number of subhalos between two values.
        Rounded to nearest int.

        :param M1: float
            Minimum value to integrate.
        :param M2: float
            Maximum value to integrate.

        :return: int
            Expected number of subhalos between two values.
        """
        # return integrate.quad(mass_pdf, M1, M2)[0])
        return self.SHVF_Grand2012_int(M1, M2)

    def make_local_data(self, M1, M2, mass_pdf, rad_pdf):
        """
        Calculate populations of subhalos in a certain Vmax range.
        Vmax and distance to the GC of the subhalos are returned.

        :param M1: float
            Minimum value to populate.
        :param M2: float
            Maximum value to integrate.
        :param mass_pdf: function
            Function to define dependency on Vmax (SHVF).
        :param rad_pdf: function
            Function to define dependency on distance to GC (SRD).

        :return: two floats or array-likes
            Populations: Vmax and distances to GC.
        """

        num_subhalos = self.expected_number(M1, M2)

        samp_ms = self.rej_samp(M1, M2, mass_pdf, num_subhalos=num_subhalos)
        samp_rs = self.rej_samp(0, self._host_cts['R_vir'], rad_pdf,
                                num_subhalos=num_subhalos)

        return np.array(samp_ms), np.array(samp_rs)

    def local_population(self, m_min, m_max, mass_pdf, rad_pdf, inc_factor=2):
        """
        Populate a host with subhalos. Dependency on Vmax and distance
        to GC.

        :param m_min: float
            Minimum value to populate.
        :param m_max: float
            Maximum value to integrate.
        :param mass_pdf: function
            Function to define dependency on Vmax (SHVF).
        :param rad_pdf: function
            Function to define dependency on distance to GC (SRD).
        :param inc_factor: float
            Factor to increase the steps at which the individual
            populations are calculated.

        :return: two floats or array-likes
            Populations: Vmax and distances to GC.
        """
        m = m_min
        ms, rs = [], []
        while m * inc_factor < m_max:
            partial_ms, partial_rs = self.make_local_data(m, m * inc_factor,
                                                          mass_pdf, rad_pdf)
            ms.extend(partial_ms)
            rs.extend(partial_rs)
            m *= inc_factor

        partial_ms, partial_rs = self.make_local_data(m, m_max,
                                                      mass_pdf, rad_pdf)
        ms.extend(partial_ms)
        rs.extend(partial_rs)
        return np.array(ms), np.array(rs)

    def computing(self):

        headerS = (('#\n# Vmin: [' + str(self._SHVF_cts['RangeMin']) + ', '
                    + str(self._SHVF_cts['RangeMax']) + '], resilient: '
                    + str(self._resilient) + '; '
                    + str(self._repopulations['its']) +
                    (' iterations \n# Js (<r_s) (GeV^2 cm^-5)'
                     '       Dgc (kpc)           D_Earth (kpc)'
                     '               Vmax (km/s)                ang size (deg)'
                     '               Cv \n#\n')))

        header03 = (('#\n# Vmin: [' + str(self._SHVF_cts['RangeMin']) + ', '
                     + str(self._SHVF_cts['RangeMax']) + '], resilient: '
                     + str(self._resilient) + '; '
                     + str(self._repopulations['its']) +
                     (' iterations \n# J03 (<0.3deg) (GeV^2 cm^-5)'
                      '       Dgc (kpc)           D_Earth (kpc)'
                      '               Vmax (km/s)               ang size (deg)'
                      '               Cv \n#\n')))

        file_Js = open(self.path_name + '/Js_' + self._sim_type + '_Res'
                       + str(self._resilient) + '_results.txt', 'w')
        file_J03 = open(self.path_name + '/J03_' + self._sim_type + '_Res'
                        + str(self._resilient) + '_results.txt', 'w')

        # if self._resilient is False:
        #     fileRoche = open(self.path_name + '/RocheMuerte_'
        #                      + self._sim_type + '.txt', 'w')

        file_Js.write(headerS)
        file_J03.write(header03)

        for it in range(self._repopulations['its']):

            if it % self._repopulations['print_freq'] == 0:
                print('    ', self._sim_type, ', res: ', self._resilient,
                      ', iteration ', it)

            # Repopulation happens, obtain subhalo masses and distances to GC
            repop_Vmax, repop_DistShGc = self.local_population(
                self._SHVF_cts['RangeMin'], self._SHVF_cts['RangeMax'],
                self.SHVF_Grand2012, self.Nr_Ntot,
                inc_factor=self._repopulations['inc_factor'])

            # Random distribution of subhalos around the celestial sphere
            repop_Theta = 2 * math.pi * np.random.random(len(repop_Vmax))
            repop_phis = math.pi * np.random.random(len(repop_Vmax))

            # Positions of th subhalos
            repop_Xs = (repop_DistShGc * np.cos(repop_Theta)
                        * np.sin(repop_phis))
            repop_Ys = (repop_DistShGc * np.sin(repop_Theta)
                        * np.sin(repop_phis))
            repop_Zs = repop_DistShGc * np.cos(repop_phis)

            repop_DistShEarth = ((repop_Xs - 8.5) ** 2 + repop_Ys ** 2
                                 + repop_Zs ** 2) ** 0.5

            if it % self._repopulations['print_freq'] == 0:
                print('        %.3f' % memory_usage_psutil())

            del (repop_Xs, repop_Ys, repop_Zs, repop_Theta, repop_phis)

            repop_C = self.Cv_Grand2012(repop_Vmax)
            repop_C = self.C_Scatt(repop_C)

            repop_Js = self.Js_vel(repop_Vmax, repop_DistShEarth, repop_C)
            repop_J03 = self.J03_vel(repop_Vmax, repop_DistShEarth, repop_C)

            # Angle of subhalos
            repop_Theta = 180 / np.pi * np.arctan(
                self.R_s(repop_Vmax, repop_C) / repop_DistShEarth)

            # if self._resilient is False:
            #     Roche_cut = (self.R_t(repop_Vmax, repop_C, repop_DistShGc)
            #                  < self.R_s(repop_Vmax, repop_C))
            #     np.savetxt(fileRoche,
            #                np.column_stack((repop_J03[Roche_cut],
            #                                 repop_Vmax[Roche_cut],
            #                                 repop_DistShGc[Roche_cut])))
            #     fileRoche.write('\n')

            if it % self._repopulations['print_freq'] == 0:
                print('        %.3f' % memory_usage_psutil())
            for num in range(self._repopulations['num_brightest']):

                bright_Js = np.where(np.max(repop_Js) == repop_Js)[0][0]

                if self._resilient is False:

                    while (self.R_t(
                            repop_Vmax[bright_Js],
                            repop_C[bright_Js],
                            repop_DistShGc[bright_Js])
                           < self.R_s(repop_Vmax[bright_Js],
                                      repop_C[bright_Js])):
                        repop_Js[bright_Js] = 0.
                        bright_Js = np.where(
                            np.max(repop_Js) == repop_Js)[0][0]

                np.savetxt(file_Js,
                           np.column_stack((repop_Js[bright_Js],
                                            repop_DistShGc[bright_Js],
                                            repop_DistShEarth[bright_Js],
                                            repop_Vmax[bright_Js],
                                            repop_Theta[bright_Js],
                                            repop_C[bright_Js])))
                repop_Js[bright_Js] = 0.
            if it % self._repopulations['print_freq'] == 0:
                print('        %.3f' % memory_usage_psutil())
            for num in range(self._repopulations['num_brightest']):

                bright_J03 = np.where(np.max(repop_J03) == repop_J03)[0][0]

                if self._resilient is False:
                    while (self.R_t(
                            repop_Vmax[bright_J03],
                            repop_C[bright_J03],
                            repop_DistShGc[bright_J03])
                           < self.R_s(
                                repop_Vmax[bright_J03],
                                repop_C[bright_J03])):
                        repop_J03[bright_J03] = 0.
                        bright_J03 = np.where(
                            np.max(repop_J03) == repop_J03)[0][0]

                np.savetxt(file_J03,
                           np.column_stack((repop_J03[bright_J03],
                                            repop_DistShGc[bright_J03],
                                            repop_DistShEarth[bright_J03],
                                            repop_Vmax[bright_J03],
                                            repop_Theta[bright_J03],
                                            repop_C[bright_J03])))
                repop_J03[bright_J03] = 0.
        print('        %.3f' % memory_usage_psutil())
        file_Js.close()
        file_J03.close()

        # if self._resilient is False:
        #     fileRoche.close()
        return

    def calculate_characteristics_subhalo(self, Vmax, Distgc, num_subs):

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

        repop_C = self.Cv_Grand2012(Vmax)
        repop_C = self.C_Scatt(repop_C)

        repop_Js = self.Js_vel(Vmax, repop_DistEarth, repop_C)
        repop_J03 = self.J03_vel(Vmax, repop_DistEarth, repop_C)

        if self._resilient is False:
            roche = (self.R_t(Vmax, repop_C, Distgc)
                     > self.R_s(Vmax, repop_C))

            repop_Js *= roche
            repop_J03 *= roche

        # Angular size of subhalos (up to R_s)
        repop_Theta = 180 / np.pi * np.arctan(
            self.R_s(Vmax, repop_C) / repop_DistEarth)

        return np.column_stack((repop_Js, repop_J03, Distgc, repop_DistEarth,
                                Vmax, repop_Theta, repop_C))

    def computing_one_by_one(self):

        headerS = (('#\n# Vmin: [' + str(self._SHVF_cts['RangeMin']) + ', '
                    + str(self._SHVF_cts['RangeMax']) + '], resilient: '
                    + str(self._resilient) + '; '
                    + str(self._repopulations['its']) +
                    (' iterations \n# Js (<r_s) (GeV^2 cm^-5)'
                     '       Dgc (kpc)           D_Earth (kpc)'
                     '               Vmax (km/s)                ang size (deg)'
                     '               Cv \n#\n')))

        header03 = (('#\n# Vmin: [' + str(self._SHVF_cts['RangeMin']) + ', '
                     + str(self._SHVF_cts['RangeMax']) + '], resilient: '
                     + str(self._resilient) + '; '
                     + str(self._repopulations['its']) +
                     (' iterations \n# J03 (<0.3deg) (GeV^2 cm^-5)'
                      '       Dgc (kpc)           D_Earth (kpc)'
                      '               Vmax (km/s)               ang size (deg)'
                      '               Cv \n#\n')))

        file_Js = open(self.path_name + '/Js_' + self._sim_type + '_Res'
                       + str(self._resilient) + '_results.txt', 'w')
        file_J03 = open(self.path_name + '/J03_' + self._sim_type + '_Res'
                        + str(self._resilient) + '_results.txt', 'w')

        file_Js.write(headerS)
        file_J03.write(header03)

        for it in range(self._repopulations['its']):

            # We have 6 variables we want to save in our files,
            # change this number if necessary
            brightest_Js = np.zeros([6, self._repopulations['num_brightest']])
            brightest_J03 = np.zeros([6, self._repopulations['num_brightest']])

            if it % self._repopulations['print_freq'] == 0:
                print('    ', self._sim_type, ', res: ', self._resilient,
                      ', iteration ', it)
                print('        %.3f' % memory_usage_psutil())

            # We calculate our subhalo population one by one in order to
            # save memory
            m_min = self._SHVF_cts['RangeMin']
            ceil = int(np.ceil(np.log(self._SHVF_cts['RangeMax']
                                      / self._SHVF_cts['RangeMin'])
                               / np.log(self._repopulations['inc_factor'])))

            for num_bins in range(ceil):
                m_max = np.minimum(m_min * self._repopulations['inc_factor'],
                                   self._SHVF_cts['RangeMax'])

                num_subhalos = int(np.round(integrate.quad(
                    self.SHVF_Grand2012, m_min, m_max)[0]))

                if it == 0:
                    print('   ', m_min, m_max, num_subhalos, int(num_subhalos))

                for i in range(int(num_subhalos)):
                    repop_Vmax = self.rej_samp(m_min, m_max,
                                               self.SHVF_Grand2012,
                                               num_subhalos=1)[0]
                    repop_DistGC = self.rej_samp(0, self._host_cts['R_vir'],
                                                 self.Nr_Ntot,
                                                 num_subhalos=1)[0]

                    new_data = self.calculate_characteristics_subhalo(
                        repop_Vmax, repop_DistGC, 1)

                    if new_data[0, 0] > brightest_Js[0, 0]:
                        brightest_Js[0, 0] = new_data[0, 0]
                        brightest_Js[1:, 0] = new_data[0, 2:]

                    if new_data[0, 1] > brightest_J03[0, 0]:
                        brightest_J03[0, 0] = new_data[0, 1]
                        brightest_J03[1:, 0] = new_data[0, 2:]

                if it == 0:
                    print('   ', m_min, m_max, num_subhalos,
                          int(num_subhalos))
                    print('        %.3f' % memory_usage_psutil())

                m_min *= self._repopulations['inc_factor']

            np.savetxt(file_Js, brightest_Js.T)
            np.savetxt(file_J03, brightest_J03.T)

        print('        %.3f' % memory_usage_psutil())
        file_Js.close()
        file_J03.close()

        return

    def computing_bin_by_bin(self):

        headerS = (('#\n# Vmin: [' + str(self._SHVF_cts['RangeMin']) + ', '
                    + str(self._SHVF_cts['RangeMax']) + '], resilient: '
                    + str(self._resilient) + '; '
                    + str(self._repopulations['its']) +
                    (' iterations \n# Js (<r_s) (GeV^2 cm^-5)'
                     '       Dgc (kpc)           D_Earth (kpc)'
                     '               Vmax (km/s)                ang size (deg)'
                     '               Cv \n#\n')))

        header03 = (('#\n# Vmin: [' + str(self._SHVF_cts['RangeMin']) + ', '
                     + str(self._SHVF_cts['RangeMax']) + '], resilient: '
                     + str(self._resilient) + '; '
                     + str(self._repopulations['its']) +
                     (' iterations \n# J03 (<0.3deg) (GeV^2 cm^-5)'
                      '       Dgc (kpc)           D_Earth (kpc)'
                      '               Vmax (km/s)               ang size (deg)'
                      '               Cv \n#\n')))

        file_Js = open(self.path_name + '/Js_' + self._sim_type + '_Res'
                       + str(self._resilient) + '_results.txt', 'w')
        file_J03 = open(self.path_name + '/J03_' + self._sim_type + '_Res'
                        + str(self._resilient) + '_results.txt', 'w')

        file_Js.write(headerS)
        file_J03.write(header03)

        for it in range(self._repopulations['its']):

            # We have 6 variables we want to save in our files,
            # change this number if necessary
            brightest_Js = np.zeros([6, self._repopulations['num_brightest']])
            brightest_J03 = np.zeros([6, self._repopulations['num_brightest']])

            if it % self._repopulations['print_freq'] == 0:
                print('    ', self._sim_type, ', res: ', self._resilient,
                      ', iteration ', it)
                print('        %.3f' % memory_usage_psutil())
                progress = open(self.path_name + '/progress' +
                                self._sim_type + '_Res'
                                + str(self._resilient)
                                + '_results.txt', 'a')
                progress.write(str(self._sim_type)
                               + ', res: ' + str(self._resilient)
                               + ', iteration ' + str(it))
                progress.write('        %.3f  %s\n' %
                               (memory_usage_psutil(),
                                time.strftime(" %Y-%m-%d %H:%M:%S",
                                              time.gmtime())))
                progress.close()

            # We calculate our subhalo population one by one in order to
            # save memory
            m_min = self._SHVF_cts['RangeMin']
            ceil = int(np.ceil(np.log(self._SHVF_cts['RangeMax']
                                      / self._SHVF_cts['RangeMin'])
                               / np.log(self._repopulations['inc_factor'])))

            for num_bins in range(ceil):
                m_max = np.minimum(m_min * self._repopulations['inc_factor'],
                                   self._SHVF_cts['RangeMax'])

                num_subhalos = int(np.round(integrate.quad(
                    self.SHVF_Grand2012, m_min, m_max)[0]))

                repop_Vmax = self.rej_samp(m_min, m_max,
                                           self.SHVF_Grand2012,
                                           num_subhalos=num_subhalos)
                repop_DistGC = self.rej_samp(0, self._host_cts['R_vir'],
                                             self.Nr_Ntot,
                                             num_subhalos=num_subhalos)

                new_data = self.calculate_characteristics_subhalo(
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
                    progress = open(self.path_name + '/progress' +
                                    self._sim_type + '_Res'
                                    + str(self._resilient)
                                    + '_results.txt', 'a')
                    progress.write('   %.3f - %.3f %s'
                                   % (m_min, m_max, num_subhalos))
                    progress.write('        %.3f  %s\n' %
                                   (memory_usage_psutil(),
                                    time.strftime(" %Y-%m-%d %H:%M:%S",
                                                  time.gmtime())))
                    progress.close()

                m_min *= self._repopulations['inc_factor']

            np.savetxt(file_Js, brightest_Js.T)
            np.savetxt(file_J03, brightest_J03.T)

        print('        %.3f' % memory_usage_psutil())
        file_Js.close()
        file_J03.close()

        return
