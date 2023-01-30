#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 18:21:13 2023

@author: saraporras
"""

import os
import math
import numpy as np
import random as rdm
import time

from optparse import OptionParser

from numba import njit, jit
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


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

        self.path_name = str(self._repopulations['id'] + time.strftime(" %d-%m-%Y %H:%M:%S", time.gmtime()))

        if not os.path.exists(self.path_name):
            os.makedirs(self.path_name)

    def repopulation(self, new_sim_types):
        self._sim_type = new_sim_types[0]
        self._resilient = new_sim_types[1]
        self.cosa_computar()
        return

    @staticmethod
    def ff(c):
        return np.log(1. + c) - c / (1. + c)

    # SHVF ----------------------------------------------------------------------------------------------------------
    def SHVF_Grand2012(self, V):
        # SubHalo Velocity Function - number of subhs defined by their Vmax.
        # Grand 2012.07846
        #
        # V - max radial velocity of a bound particle in the subhalo [km/s]
        return 10 ** self._SHVF_cts[self._sim_type]['bb'] * V ** self._SHVF_cts[self._sim_type]['mm']

    def SHVF_Grand2012_int(self, V1, V2):
        # SubHalo Velocity Function - number of subhs defined by their Vmax. Integration of SHVF.
        # Grand 2012.07846
        #
        # V1, V2 - limits of integration. Max radial velocity of a bound particle in the subhalo [km/s]
        return (10 ** self._SHVF_cts[self._sim_type]['bb'] / (self._SHVF_cts[self._sim_type]['mm'] + 1) *
                (V2 ** (self._SHVF_cts[self._sim_type]['mm'] + 1) - V1 ** (self._SHVF_cts[self._sim_type]['mm'] + 1)))

    # ----------- CONCENTRATIONS --------------------------------------------

    def Cv_Grand2012(self, Vmax):
        # Concentration based on Grand 2012.07846. DMO simulation.
        return 10 ** self._cv_cts[self._sim_type]['bb'] * Vmax ** self._cv_cts[self._sim_type]['mm']

    def C_Scatt(self, C):
        # Create a scatter in the concentrations from relation Rmax-Vmax in DMo sim
        return np.array([np.random.lognormal(np.log(C[i]), self._cv_cts[self._sim_type]['sigma'])
                         for i in range(C.size)])

    # ----------- J-FACTORS ------------------------------------------------

    def J_abs_vel(self, V, D_earth, C):
        # Jfactor enclosing whole subhalo as a function of the Vmax of the subhalo
        #
        # V       - maximum velocity inside the subhalos [km/s]
        # D_Earth - distancia centro subhalo-Tierra [kpc]
        # C       - concentracion del subhalo
        return (2.163 ** 3. / D_earth ** 2. / self.ff(2.163) ** 2
                * self._cosmo['H_0'] / 12 / np.pi / float(self._cosmo['G']) ** 2
                * np.sqrt(C / 2) * V ** 3)

    def Js_vel(self, V, D_earth, C):
        # Jfactor enclosing the subhalo up to rs as a function of the Vmax of the subhalo
        #
        # V       - maximum velocity inside the subhalos [km/s]
        # D_Earth - distancia centro subhalo-Tierra [kpc]
        # C       - concentracion del subhalo
        return (2.163 ** 3. / D_earth ** 2. / self.ff(2.163) ** 2
                * self._cosmo['H_0'] / 12 / np.pi / float(self._cosmo['G']) ** 2
                * np.sqrt(C / 2) * V ** 3 * 7 / 8)

    def J03_vel(self, V, D_earth, C):
        # Jfactor enclosing the subhalo up to 0.3 degrees as a function of the Vmax of the subhalo
        #
        # V       - maximum velocity inside the subhalos [km/s]
        # D_Earth - distancia centro subhalo-Tierra [kpc]
        # C       - concentracion del subhalo
        return (2.163 ** 3. / D_earth ** 2. / self.ff(2.163) ** 2
                * self._cosmo['H_0'] / 12 / np.pi / float(self._cosmo['G']) ** 2
                * np.sqrt(C / 2) * V ** 3 *
                (1 - 1 / (1 + 2.163 * D_earth * np.tan(0.15 * np.pi / 180.) / self.R_max(V, C)) ** 3))

    # ----------- REPOPULATION -----------------------------------------

    def R_max(self, V, C):
        return 0.5 * V / self._cosmo['H_0'] * np.sqrt(C) * 1e3

    def R_s(self, V, C):
        return self.R_max(V, C) / 2.163

    def R_t(self, V, C, DistGC):
        # 1603.04057 King radius pg 14
        # Para Roche al menos
        Rmax = self.R_max(V, C)
        c200 = self.C200_from_Cv(C)
        M = self.mass_from_Vmax(V, Rmax, c200)
        return ((M / (3 * self.M_Cont(DistGC))) ** (1. / 3)) * DistGC

    def M_Cont(self, R):
        # 1603.04057 pg 14 la explican: masa del halo hasta ese radio
        # Para Roche al menos
        return 4 * math.pi * float(self._host_cts['rho_0']) * self._host_cts['r_s'] ** 3 * self.NFW_int(R)

    def NFW_int(self, R):  # NOTE: proportionality constants excluded
        return np.log((self._host_cts['r_s'] + R) / self._host_cts['r_s']) - R / (self._host_cts['r_s'] + R)

    def N_Dgc_Cosmic(self, R, R0, aa, bb):
        # Number of subhalos (NOT DM DENSITY) at a distance to the GC R.
        return (R / R0) ** aa * np.exp(-bb * (R - R0) / R0)

    def Nr_Ntot(self, dist_gc):
        # Number of subhalos at a certain distance of the GC.
        if self._resilient:
            args = self._srd_cts[self._sim_type]['resilient']['args']
            yy = self.N_Dgc_Cosmic(dist_gc, args[0], args[1], args[2])
            return 10 ** yy
        else:
            args = self._srd_cts[self._sim_type]['fragile']['args']
            yy = self.N_Dgc_Cosmic(dist_gc, args[0], args[1], args[2])
            return 10 ** yy * (dist_gc >= 6.6176)

    def mass_from_Vmax(self, Vmax, Rmax, c200):
        # From Moline16
        return Vmax ** 2 * Rmax * 3.086e16 * 1e9 / 6.6741e-11 / 1.989e30 * self.ff(c200) / self.ff(2.163)

    def def_Cv(self, c200, Cv):
        return 200 * self.ff(2.163) / self.ff(c200) * (c200 / 2.163) ** 3 - Cv

    def newton(self, fun, x0, args):
        x = x0
        for i in range(100):
            x -= 0  # fun(x,args[0])*0.02/(fun(x+0.01,args[0]) - fun(x-0.01,args[0]))
        return x

    def C200_from_Cv(self, Cv):
        if np.shape(Cv) == ():
            C200_med = self.newton(self.def_Cv, 40.0, Cv)
            C200_med = np.array([C200_med])
        else:
            num = np.shape(Cv)[0]
            C200_med = np.zeros(num)

            for i in range(num):
                C200_med[i] = self.newton(self.def_Cv, 40.0, Cv[i])  # opt.root(def_Cv, 40, args=Cv[i])['x'][0]

        return C200_med

    def rej_samp(self, xmin, xmax, pdf, num_subhalos, mass=1):
        results = []

        # Compute the maximum of the function
        fn_max = np.max(pdf(np.linspace(xmin, xmax, 1000)))

        while len(results) < num_subhalos:
            x = rdm.uniform(xmin, xmax)
            h = rdm.uniform(0, fn_max)

            if h < pdf(x):
                results.append(x)

        return results

    def exp_num_sh(self, M1, M2, Rcut, mass_pdf, dist_cut=True):
        # Expected number of subhalos between M1 and M2, R1 and R2
        # (rounded down to nearest int)
        if dist_cut:
            #        if printt:
            #            print ('     Dist frac:', dist_frac(Rcut))
            return 1  # math.floor(dist_frac(Rcut)*integrate.quad(mass_pdf, M1, M2)[0])
        else:
            #        if printt:
            #            print ('     WARNING: no detectability distance cut applied')
            return np.floor(self.SHVF_Grand2012_int(M1, M2))  # integrate.quad(mass_pdf, M1, M2)[0])

    def make_local_data(self, M1, M2, mass_pdf, rad_pdf, dist_cut=True):
        Rcut = 10  # R_Cut(M2)
        if Rcut > 8.5:
            dist_cut = False
        num_subhalos = self.exp_num_sh(M1, M2, Rcut, mass_pdf, dist_cut=dist_cut)
        samp_ms = self.rej_samp(M1, M2, mass_pdf, num_subhalos=num_subhalos)
        #    if printt:
        #        print ('     Total subhalos generated in this range: ', num_subhalos)
        #        print ('     Cutting at: %.2f kpc' % Rcut)

        # if Rcut < 8.5:
        if dist_cut:
            samp_rs = self.rej_samp(8.5 - Rcut, 8.5 + Rcut, rad_pdf, num_subhalos=num_subhalos, mass=0)
        else:
            samp_rs = self.rej_samp(0, self._host_cts['R_vir'], rad_pdf, num_subhalos=num_subhalos, mass=0)
        return np.array(samp_ms), np.array(samp_rs)

    def local_population(self, mmin, mmax, mass_pdf, rad_pdf, inc_factor=2, dist_cut=True):
        m = mmin
        ms, rs = [], []
        while (m * inc_factor < mmax):
            #        if printt:
            #            print()
            #            print ('Populating subhalos between %.2f - %.2f' %(m, m*inc_factor))
            partial_ms, partial_rs = self.make_local_data(m, m * inc_factor, mass_pdf, rad_pdf, dist_cut=dist_cut)
            ms.extend(partial_ms)
            rs.extend(partial_rs)
            m *= inc_factor

        #    if printt:
        #        print()
        #        print ('Populating subhalos between %.2f - %.2f' %(m, mmax))
        partial_ms, partial_rs = self.make_local_data(m, mmax, mass_pdf, rad_pdf, dist_cut=dist_cut)
        ms.extend(partial_ms)
        rs.extend(partial_rs)
        return np.array(ms), np.array(rs)

    def cosa_computar(self):
        printt = True

        headerS = (('#\n# Resilient: ' + str(self._resilient) + '; ' + str(self._repopulations['its']) +
                    (' iterations \n# Js (<r_s) (GeV^2 cm^-5)'
                     '       Dgc (kpc)      D_Earth (kpc)'
                     '                   Vmax (km/s)                   ang size (deg)'
                     '               Cv \n#\n')))

        header03 = (('#\n# Resilient: ' + str(self._resilient) + '; ' + str(self._repopulations['its']) +
                     (' iterations \n# J03 (<0.3deg) (GeV^2 cm^-5)'
                      '       Dgc (kpc)      D_Earth (kpc)'
                      '                   Vmax (km/s)                   ang size (deg)'
                      '               Cv \n#\n')))

        file_Js = open(self.path_name + '/Js_' + self._sim_type + '_Res' + str(self._resilient) + '_results.txt', 'w')
        file_J03 = open(self.path_name + '/J03_' + self._sim_type + '_Res' + str(self._resilient) + '_results.txt', 'w')
        fileRoche = open(self.path_name + '/RocheMuerte_' + self._sim_type + '.txt', 'w')

        file_Js.write(headerS)
        file_J03.write(header03)

        for it in range(self._repopulations['its']):

            if it % self._repopulations['print_freq'] == 0:
                print(self._sim_type, ', iteration ', it)

            # Repopulation happens, you firstly obtain masses and distances to GC
            repop_Vmax, repop_DistShGc = self.local_population(self._SHVF_cts['RangeMin'], self._SHVF_cts['RangeMax'],
                                                               self.SHVF_Grand2012, self.Nr_Ntot,
                                                               dist_cut=False, inc_factor=2)

            # Random distribution of subhalos around the celestial sphere
            repop_Theta = 2 * math.pi * np.random.random(len(repop_Vmax))
            repop_phis = math.pi * np.random.random(len(repop_Vmax))

            # Positions of th subhalos
            repop_Xs, repop_Ys, repop_Zs = (repop_DistShGc * np.cos(repop_Theta) * np.sin(repop_phis),
                                            repop_DistShGc * np.sin(repop_Theta) * np.sin(repop_phis),
                                            repop_DistShGc * np.cos(repop_phis))

            repop_DistShEarth = ((repop_Xs - 8.5) ** 2 + repop_Ys ** 2 + repop_Zs ** 2) ** (.5)

            del (repop_Xs, repop_Ys, repop_Zs, repop_Theta, repop_phis)

            if printt:
                print()
                print('Calculating concentrations....')

            repop_C = self.Cv_Grand2012(repop_Vmax)
            repop_C = self.C_Scatt(repop_C)

            if printt:
                print()
                print('Calculating J-factors....')

            repop_Js = ((3.241 * 10 ** (-22)) ** 5 * (1.12 * 10 ** 57) ** 2 *
                        self.Js_vel(repop_Vmax, repop_DistShEarth, repop_C))

            repop_J03 = ((3.241 * 10 ** (-22)) ** 5 * (1.12 * 10 ** 57) ** 2 *
                         self.J03_vel(repop_Vmax, repop_DistShEarth, repop_C))

            # Angle of subhalos
            repop_Theta = 180 / np.pi * np.arctan(self.R_s(repop_Vmax, repop_C) / repop_DistShEarth)

            Roche_cut = (self.R_t(repop_Vmax, repop_C, repop_DistShGc) < self.R_s(repop_Vmax, repop_C))

            np.savetxt(fileRoche, np.column_stack((repop_J03[Roche_cut],
                                                   repop_Vmax[Roche_cut],
                                                   repop_DistShGc[Roche_cut])))
            fileRoche.write('\n')

            for num in range(self._repopulations['num_brightest']):

                bright_Js = np.where(np.max(repop_Js) == repop_Js)[0][0]

                if self._resilient == False:
                    # print(num, 'aaaa')
                    while (self.R_t(repop_Vmax[bright_Js], repop_C[bright_Js], repop_DistShGc[bright_Js])
                           > self.R_s(repop_Vmax[bright_Js], repop_C[bright_Js])):
                        repop_Js[bright_Js] = 0.
                        bright_Js = np.where(np.max(repop_Js) == repop_Js)[0][0]
                        # print('Roche')

                np.savetxt(file_Js, np.column_stack((repop_Js[bright_Js],
                                                     repop_DistShGc[bright_Js],
                                                     repop_DistShEarth[bright_Js],
                                                     repop_Vmax[bright_Js],
                                                     repop_Theta[bright_Js],
                                                     repop_C[bright_Js])))
                repop_Js[bright_Js] = 0.

            for num in range(self._repopulations['num_brightest']):

                bright_J03 = np.where(np.max(repop_J03) == repop_J03)[0][0]

                if self._resilient == False:
                    while (self.R_t(repop_Vmax[bright_J03], repop_C[bright_J03], repop_DistShGc[bright_J03])
                           > self.R_s(repop_Vmax[bright_J03], repop_C[bright_J03])):
                        repop_J03[bright_J03] = 0.
                        bright_J03 = np.where(np.max(repop_J03) == repop_J03)[0][0]
                #                print('Roche')

                np.savetxt(file_J03, np.column_stack((repop_J03[bright_J03],
                                                      repop_DistShGc[bright_J03],
                                                      repop_DistShEarth[bright_J03],
                                                      repop_Vmax[bright_J03],
                                                      repop_Theta[bright_J03],
                                                      repop_C[bright_J03])))
                repop_J03[bright_J03] = 0.

        file_Js.close()
        file_J03.close()
        fileRoche.close()
