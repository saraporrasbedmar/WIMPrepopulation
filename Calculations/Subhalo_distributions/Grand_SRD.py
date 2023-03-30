#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 20:37:40 2022

@author: saraporras
"""

import numpy as np
import matplotlib.pyplot as plt

import scipy.optimize as opt
from scipy import integrate
from matplotlib.lines import Line2D

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rc('font', size=20)
plt.rc('axes', titlesize=16)
plt.rc('axes', labelsize=16)
plt.rc('xtick', labelsize=21)
plt.rc('ytick', labelsize=22)
plt.rc('legend', fontsize=20)
plt.rc('figure', titlesize=17)
plt.rc('xtick', top=True, direction='in')
plt.rc('ytick', right=True, direction='in')
plt.rc('xtick.major', size=7, width=1.5, top=True)
plt.rc('ytick.major', size=7, width=1.5, right=True)
plt.rc('xtick.minor', size=4, width=1)
plt.rc('ytick.minor', size=4, width=1)

#        Rmax[kpc]        Vmax[km/s]      Radius[Mpc]
Grand_dmo = np.loadtxt('../Data_subhalos_simulations/RmaxVmaxRadDMO0_1.txt')
Grand_hydro = np.loadtxt('../Data_subhalos_simulations/RmaxVmaxRadFP0_1.txt')

# #        Rmax        Vmax      Radius
# Grand_dmo = np.loadtxt('../../RmaxVmaxRadDMO0_1.txt')
# Grand_hydro = np.loadtxt('../../RmaxVmaxRadFP0_1.txt')

# Grand_hydro = Grand_hydro[Grand_hydro[:,1]>1e-4, :]
Grand_hydro = Grand_hydro[Grand_hydro[:, 1] > np.min(Grand_dmo[:, 1]), :]

# Grand_dmo[:,2] *= 1e3
# Grand_hydro[:,2] *= 1e3

Grand_dmo = Grand_dmo[np.argsort(Grand_dmo[:, 2])]
Grand_hydro = Grand_hydro[np.argsort(Grand_hydro[:, 2])]

# Los mayores Dist_Gc son 264 kpc, pero entonces cuál es R200?
R200 = 0.265  # kpc
R_max = 0.220


# %% SRD

def encontrar_SRD_sinVol(data):
    n_final = []
    a = 0
    for delta in range(len(bins) - 2):
        X1limit = np.where(data[:, 2] >= bins[delta])[0][0]
        X2limit = np.where(data[:, 2] > bins[delta + 1])[0][0]

        y = X2limit - X1limit
        #        vol = 4/3 * np.pi * (bins[delta+1]**3 - bins[delta]**3)# * 1e9

        n_final.append(y)  # /vol)
        a += y
    #        print(delta, X2limit, X1limit)

    y = len(data) - X2limit
    n_final.append(y)  # /(4/3 * np.pi * (bins[-1]**3 - bins[-2]**3)))

    a += y
    print(a)

    return n_final


def encontrar_SRD(data):
    n_final = []
    a = 0
    for delta in range(len(bins) - 2):
        X1limit = np.where(data[:, 2] >= bins[delta])[0][0]
        X2limit = np.where(data[:, 2] > bins[delta + 1])[0][0]

        y = X2limit - X1limit
        vol = 4 / 3 * np.pi * (
                bins[delta + 1] ** 3 - bins[delta] ** 3)  # * 1e9

        n_final.append(y / vol)
        a += y
    #        print(delta, X2limit, X1limit)

    y = len(data) - X2limit
    n_final.append(y / (4 / 3 * np.pi * (bins[-1] ** 3 - bins[-2] ** 3)))

    a += y
    print(a)

    return n_final


def N_Dgc_Cosmic(R, R0, aa, bb):
    # Number of subhalos (NOT DM DENSITY) at a distance to the GC R.
    return (R / R0) ** aa * np.exp(-bb * (R - R0) / R0)


def resilient(R, yy_args):
    yy = ((R <= yy_args[0][0] * yy_args[0][1] / yy_args[0][2]) *
          N_Dgc_Cosmic(yy_args[0][0] * yy_args[0][1] / yy_args[0][2],
                       yy_args[0][0], yy_args[0][1], yy_args[0][2])
          + (R > yy_args[0][0] * yy_args[0][1] / yy_args[0][2]) *
          N_Dgc_Cosmic(R, yy_args[0][0], yy_args[0][1], yy_args[0][2]))

    return yy


def fragile(R, yy_args, x_cut):
    yy = (N_Dgc_Cosmic(R, yy_args[0][0], yy_args[0][1], yy_args[0][2])
          * (R >= x_cut))

    return yy


# ---------------------------------------------------------------------------------------------------


plt.close('all')
plt.figure(figsize=(12, 7))
num_bins = 15

bins = np.linspace(0, R200, num=num_bins)
x_med = (bins[:-1] + bins[1:]) / 2.

srd_dmo = np.array(encontrar_SRD(Grand_dmo)) / len(Grand_dmo)
srd_hydro = np.array(encontrar_SRD(Grand_hydro)) / len(Grand_hydro)

plt.plot(x_med * 1e3, srd_dmo, '-', color='k')
plt.plot(x_med * 1e3, srd_dmo, label='Data', color='k', marker='.', ms=10)
plt.plot(x_med * 1e3, srd_hydro, '-', color='limegreen')
plt.plot(x_med * 1e3, srd_hydro, color='limegreen', marker='.', ms=10)

xxx = np.linspace(3e-3, R200 * 1e3, num=100)

vol = 4 / 3 * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)

x_med_kpc = x_med * 1e3

# plt.plot(x_med_kpc, 10**(-1.22080523 + x_med_kpc*0.00149078) / vol, '-',
#          color='grey', alpha=0.7,
#          label='res_maximum lin')
# plt.plot(x_med_kpc, 10**(-1.37456315 + x_med_kpc*0.00266445) / vol, '-',
#          color='limegreen', alpha=0.7)
#
# plt.plot(x_med_kpc, 10**(-2.17672138) * x_med_kpc**0.54343928 / vol, '--',
#          color='grey', alpha=0.7,
#          label='res_maximum log')
# plt.plot(x_med_kpc, 10**(-3.08584275) * x_med_kpc**0.97254648 / vol, '--',
#          color='limegreen', alpha=0.7)


plt.plot(x_med_kpc, (x_med_kpc ** 0.54343928
                     * 10 ** -2.17672138) / vol,
         '--', marker='+', ms=10,
         color='grey', alpha=0.7, label='res_maximum N/Ntot')
plt.plot(x_med_kpc, (x_med_kpc ** 0.97254648
                     * 10 ** -3.08584275) / vol,
         '--', marker='+', ms=10, color='limegreen', alpha=0.7)

xmin = 2
xmax = 11
log_tend_dmo = np.polyfit(np.log10(x_med[xmin:xmax] * 1e3),
                          srd_dmo[xmin:xmax], 1)
print('max res dmo log')
print(log_tend_dmo)

xmin = 5
xmax = 11
log_tend_hydro = np.polyfit(np.log10(x_med[xmin:xmax] * 1e3),
                            srd_hydro[xmin:xmax], 1)
print('max res hydro log')
print(log_tend_hydro)

plt.plot(xxx, np.log10(xxx) * log_tend_dmo[0] + log_tend_dmo[1], '-',
         color='grey', alpha=0.7, label='res_maximum density')
plt.plot(xxx, np.log10(xxx) * log_tend_hydro[0] + log_tend_hydro[1], '-',
         color='limegreen', alpha=0.7)

xxx = np.linspace(3e-3, R200, num=3000)
X_max = np.where(x_med >= R_max)[0][0]

yy_dmo = opt.curve_fit(N_Dgc_Cosmic, x_med[:X_max], srd_dmo[:X_max],
                       p0=[0.5, 0.5, 5])
peak_dmo = yy_dmo[0][0] * 1e3 * yy_dmo[0][1] / yy_dmo[0][2]
plt.plot(xxx * 1e3, resilient(xxx, yy_dmo), 'k', linestyle='dashdot',
         alpha=0.7, label='Resilient')
plt.plot(xxx * 1e3, fragile(xxx, yy_dmo, np.min(Grand_dmo[:, 2])), 'k',
         alpha=0.7, label='Fragile', linestyle=':')
# plt.plot(x_med*1e3, resilient(x_med, yy_dmo), 'xk', ms=10, alpha=0.7)
plt.axvline(peak_dmo, alpha=0.5, color='k')
print('DMO:  ', yy_dmo[0])

yy_hyd = opt.curve_fit(N_Dgc_Cosmic, x_med[:X_max], srd_hydro[:X_max],
                       p0=[0.5, 0.5, 5])
peak_hyd = yy_hyd[0][0] * 1e3 * yy_hyd[0][1] / yy_hyd[0][2]
plt.plot(xxx * 1e3, resilient(xxx, yy_hyd), linestyle='dashdot', alpha=0.7,
         color='limegreen')
plt.plot(xxx * 1e3, fragile(xxx, yy_hyd, np.min(Grand_hydro[:, 2])), alpha=0.7,
         color='limegreen', ls=':')
plt.axvline(peak_hyd, alpha=0.5, color='limegreen')
# plt.plot(x_med*1e3, resilient(x_med, yy_hyd), 'x', ms=10, alpha=0.7, color='limegreen')
print('Hydro:', yy_hyd[0])

plt.axvline(R_max * 1e3, alpha=0.7, linestyle='--')  # , label='220 kpc')
plt.annotate(r'R$_\mathrm{vir}$', (170, 32), color='b', rotation=45, alpha=0.7)

plt.axvline(8.5, linestyle='--', alpha=1, color='Sandybrown')
plt.annotate('Earth', (8.6, 35), color='Sandybrown', rotation=45)

plt.ylabel(r'n(r) = $\frac{N(r)}{N_{Tot}\,Volumen}$ [Mpc$^{-3}$]', size=24)
plt.xlabel('r [kpc]', size=26)

legend_elements = [Line2D([0], [0], marker='o', color='w', label='Frag',
                          markerfacecolor='k', markersize=8),
                   Line2D([0], [0], marker='o', color='w', label='Res',
                          markerfacecolor='limegreen', markersize=8)]
legend1 = plt.legend(legend_elements, ['DMO', 'Hydro'], loc=8,
                     handletextpad=0.2)  # ,handlelength=1)
leg = plt.legend(framealpha=1, loc=1)
plt.gca().add_artist(legend1)

plt.xscale('log')
plt.yscale('log')
plt.xlim(6, 300)

# plt.ylim(0, 60)


# ------------------------ N(r)/Ntot figure ------------------------------

srd_dmo_sinVol = (np.array(encontrar_SRD_sinVol(Grand_dmo))
                  / len(Grand_dmo))
srd_hydro_sinVol = (np.array(encontrar_SRD_sinVol(Grand_hydro))
                    / len(Grand_hydro))

x_kpc = x_med * 1e3

plt.figure(2, figsize=(10, 8))

plt.plot(x_kpc, srd_dmo_sinVol, '-', color='k')
plt.plot(x_kpc, srd_dmo_sinVol, color='k', marker='.', ms=10, label='DMO')

plt.plot(x_kpc, srd_hydro_sinVol, '-', color='g')
plt.plot(x_kpc, srd_hydro_sinVol, color='g', marker='.', ms=10, label='Hydro')

xmin = 6
xmax = 11
linear_tend_dmo = np.polyfit(np.log10(x_kpc[xmin:xmax]),
                             np.log10(srd_dmo_sinVol[xmin:xmax]), 1)
print('max res dmo log-log')
print(linear_tend_dmo)

xmin = 6
xmax = 11
linear_tend_hydro = np.polyfit(np.log10(x_kpc[xmin:xmax]),
                               np.log10(srd_hydro_sinVol[xmin:xmax]), 1)
print('max res hydro log-log')
print(linear_tend_hydro)

plt.plot(x_kpc, x_kpc ** linear_tend_dmo[0] * 10 ** linear_tend_dmo[1],
         '--', color='grey', alpha=0.7, label='res_maximum N/Ntot')
plt.plot(x_kpc, x_kpc ** linear_tend_hydro[0] * 10 ** linear_tend_hydro[1],
         '--', color='limegreen', alpha=0.7)

vol = 4 / 3 * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
plt.plot(x_kpc, (np.log10(x_kpc) * log_tend_dmo[0] + log_tend_dmo[1]) * vol,
         '-', marker='+', ms=10,
         color='grey', alpha=0.7, label='res_maximum density')
plt.plot(x_kpc,
         (np.log10(x_kpc) * log_tend_hydro[0] + log_tend_hydro[1]) * vol,
         '-', marker='+', ms=10,
         color='limegreen', alpha=0.7)

# def N_Dgc_Cosmic_slide(R, R0, aa, bb):
#     # Number of subhalos (NOT DM DENSITY) at a distance to the GC R.
#     return 10**((R / R0) ** aa * np.exp(-bb * (R - R0) / R0))
# cts_dmo = opt.curve_fit(N_Dgc_Cosmic_slide, x_kpc, srd_dmo_sinVol,
#                         p0=[450, 0.5, 1e-5])
# print(cts_dmo)


# plt.plot(x_kpc, N_Dgc_Cosmic_slide(x_kpc,
#                              cts_dmo[0][0], cts_dmo[0][1], cts_dmo[0][2]))

# xmin = 6
# xmax = 11
# linear_tend_dmo = np.polyfit((x_kpc[xmin:xmax]),
#                              np.log10(srd_dmo_sinVol[xmin:xmax]), 1)
# print('max res dmo lin-log')
# print(linear_tend_dmo)
#
# xmin = 6
# xmax = 11
# linear_tend_hydro = np.polyfit((x_kpc[xmin:xmax]),
#                                np.log10(srd_hydro_sinVol[xmin:xmax]), 1)
# print('max res hydro lin-log')
# print(linear_tend_hydro)
#
# plt.plot(x_kpc, 10 ** (x_kpc * linear_tend_dmo[0] + linear_tend_dmo[1]),
#          '-', color='grey', alpha=0.7, label='res_maximum lineal')
# plt.plot(x_kpc, 10 ** (x_kpc * linear_tend_hydro[0] + linear_tend_hydro[1]),
#          '-', color='limegreen', alpha=0.7)
#


# plt.axvline(peak_dmo, alpha=0.5, color='k')
# plt.axvline(peak_hyd, alpha=0.5, color='g')
plt.axvline(R_max * 1e3, alpha=0.5, label='220 kpc')

plt.ylabel(r'n(r) = $\frac{N(r)}{N_{Tot}}$')
plt.xlabel('r [kpc]', size=24)

plt.legend(framealpha=1)

plt.xscale('log')
plt.yscale('log')

print()
print('Place of [Mpc] and number of subs below the resilient-disrupted cut:')
print('%.5f ----- %r' % (yy_dmo[0][0] * yy_dmo[0][1] / yy_dmo[0][2],
                         np.shape(
                             Grand_dmo[Grand_dmo[:, 2]
                                       <= peak_dmo / 1e3, :])[0]))
print('%.5f ----- %r' % (yy_hyd[0][0] * yy_hyd[0][1] / yy_hyd[0][2],
                         np.shape(
                             Grand_hydro[Grand_hydro[:, 2]
                                         <= peak_hyd / 1e3, :])[0]))

# FIGURE: ATTEMPT AT HAVING A MONTECARLO ALGORITHM TO POPULATE THE HOST
'''
plt.figure()

plt.plot(x_kpc, srd_dmo_sinVol, '-', color='k')
plt.plot(x_kpc, srd_dmo_sinVol, color='k', marker='.', ms=10, label='DMO')

plt.plot(x_kpc, srd_hydro_sinVol, '-', color='g')
plt.plot(x_kpc, srd_hydro_sinVol, color='g', marker='.', ms=10, label='Hydro')

plt.plot(xxx * 1e3,
         fragile(xxx * 1e3, [[1011.38716, 0.4037927, 2.35522213]], 13.6))
plt.plot(xxx * 1e3,
         fragile(xxx * 1e3, [[666.49179, 0.75291017, 2.90546523]], 6.61))

plt.axvline(8.5, linestyle='--', alpha=1, color='Sandybrown')
plt.annotate('Earth', (8.6, 35), color='Sandybrown', rotation=45)

plt.ylabel(r'n(r) = $\frac{N(r)}{N_{Tot}\,Volumen}$ [Mpc$^{-3}$]', size=24)
plt.xlabel('r [kpc]', size=26)

# legend_elements = [Line2D([0], [0], marker='o', color='w', label='Frag',
#                           markerfacecolor='k', markersize=8),
#                    Line2D([0], [0], marker='o', color='w', label='Res',
#                           markerfacecolor='limegreen', markersize=8)]
# legend1 = plt.legend(legend_elements, ['DMO', 'Hydro'], loc=8,
#                      handletextpad=0.2)  # ,handlelength=1)
# leg = plt.legend(framealpha=1, loc=1)
# plt.gca().add_artist(legend1)

plt.xscale('log')
plt.yscale('log')
# plt.xlim(6, 300)
'''

plt.figure(3, figsize=(10, 8))

plt.plot(x_kpc, srd_dmo_sinVol, '-', color='k')
plt.plot(x_kpc, srd_dmo_sinVol, color='k', marker='.', ms=10, label='DMO')

plt.plot(x_kpc, srd_hydro_sinVol, '-', color='g')
plt.plot(x_kpc, srd_hydro_sinVol, color='g', marker='.', ms=10, label='Hydro')

xmin = 6
xmax = 11
linear_tend_dmo = np.polyfit(np.log10(x_kpc[xmin:xmax]),
                             np.log10(srd_dmo_sinVol[xmin:xmax]), 1)
print('max res dmo log-log')
print(linear_tend_dmo)

xmin = 6
xmax = 11
linear_tend_hydro = np.polyfit(np.log10(x_kpc[xmin:xmax]),
                               np.log10(srd_hydro_sinVol[xmin:xmax]), 1)
print('max res hydro log-log')
print(linear_tend_hydro)

plt.plot(x_kpc, x_kpc ** linear_tend_dmo[0] * 10 ** linear_tend_dmo[1],
         '--', color='grey', alpha=0.7, label='res_maximum N/Ntot')
plt.plot(x_kpc, x_kpc ** linear_tend_hydro[0] * 10 ** linear_tend_hydro[1],
         '--', color='limegreen', alpha=0.7)

vol = 4 / 3 * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
plt.plot(x_kpc, (np.log10(x_kpc) * log_tend_dmo[0] + log_tend_dmo[1]) * vol,
         '-', marker='+', ms=10,
         color='grey', alpha=0.7, label='res_maximum density')
plt.plot(x_kpc,
         (np.log10(x_kpc) * log_tend_hydro[0] + log_tend_hydro[1]) * vol,
         '-', marker='+', ms=10,
         color='limegreen', alpha=0.7)


def N_Dgc_Cosmic_slide(R, R0, aa, bb):
    # Number of subhalos (NOT DM DENSITY) at a distance to the GC R.
    return R0 + aa * R ** bb


# cts_dmo = opt.curve_fit(N_Dgc_Cosmic_slide, x_kpc, np.log10(srd_dmo_sinVol),
#                         p0=[-5, 0.5, 0.3])
# print(cts_dmo)
#
# plt.plot(x_kpc, N_Dgc_Cosmic_slide(x_kpc,
#                                    cts_dmo[0][0], cts_dmo[0][1],
#                                    cts_dmo[0][2]))

plt.axvline(R_max * 1e3, alpha=0.5, label='220 kpc')

plt.ylabel(r'n(r) = $\frac{N(r)}{N_{Tot}}$')
plt.xlabel('r [kpc]', size=24)

plt.legend(framealpha=1)

# plt.xscale('log')
plt.yscale('log')

plt.show()
