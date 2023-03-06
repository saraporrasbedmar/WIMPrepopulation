#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 10:29:55 2022

@author: saraporras
"""

import numpy as np
import matplotlib.pyplot as plt

import scipy.special as spice
import scipy.optimize as opt

from scipy.optimize import curve_fit
from scipy.stats import gmean

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rc('font', size=20)
plt.rc('axes', titlesize=18)
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=24)
plt.rc('ytick', labelsize=24)
plt.rc('legend', fontsize=24)
plt.rc('figure', titlesize=17)
plt.rc('xtick', top=True, direction='in')
plt.rc('ytick', right=True, direction='in')
plt.rc('xtick.major', size=7, width=1.5, top=True)
plt.rc('ytick.major', size=7, width=1.5, right=True)
plt.rc('xtick.minor', size=4, width=1)
plt.rc('ytick.minor', size=4, width=1)

#        Rmax        Vmax      Radius
Grand_dmo = np.loadtxt('../../RmaxVmaxRadDMO0_1.txt')
Grand_hydro = np.loadtxt('../../RmaxVmaxRadFP0_1.txt')

Grand_hydro = Grand_hydro[Grand_hydro[:, 1] > 1e-4, :]

Grand_dmo = Grand_dmo[Grand_dmo[:, 0] > 2 * 0.184, :]
Grand_hydro = Grand_hydro[Grand_hydro[:, 0] > 2 * 0.184, :]

Grand_dmo[:, 2] *= 1e3
Grand_hydro[:, 2] *= 1e3

Grand_dmo = Grand_dmo[np.argsort(Grand_dmo[:, 1])]
Grand_hydro = Grand_hydro[np.argsort(Grand_hydro[:, 1])]

Grand_dmo_raw = Grand_dmo.copy()
Grand_hydro_raw = Grand_hydro.copy()

# Los mayores Dist_Gc son 264 kpc, pero entonces cuál es R200?
R200 = 263  # kpc

plt.close('all')


# %%  CALCULATE THE RMAX-VMAX RELATION

def Cv_Mol2021(V, x, ci=[1.12e5, -0.9512, -0.5538, -0.3221, -1.7828]):
    # Median subhalo concentration depending on its Vmax and distance to the host center
    # Moline et al. 2110.02097
    #
    # V - max radial velocity of a bound particle in the subhalo [km/s]
    # x - distance from the subh to the halo center [kpc]
    return (ci[0] * (1 + (
        sum([(ci[i + 1] * np.log10(V)) ** (i + 1) for i in range(3)])))
            * (1 + ci[4] * np.log10(x)))


def Cv_Mol2021_redshift0(V, c0=1.75e5, c1=-0.90368, c2=0.2749, c3=-0.028):
    # Median subhalo concentration depending on its Vmax and its redshift (here z=0)
    # Moline et al. 2110.02097
    #
    # V - max radial velocity of a bound particle in the subhalo [km/s]
    ci = [c0, c1, c2, c3]
    return ci[0] * (1 + (sum([ci[i + 1] * np.log10(V) ** (i + 1)
                              for i in range(3)])))


def Cv_Grand(Vmax, mm, bb):
    return 2. * (10 ** (-bb) * Vmax ** (1. - mm) / H0) ** 2


def Cv_Grand_points(xx, yy):
    return 2. * (xx / yy / H0) ** 2


H0 = 70. / 1e3


def calculate_med(dataset, num_interv, perc_low=20, perc_high=80, nmax=60):
    x_approx = np.logspace(np.log10(dataset[0, 1]), np.log10(nmax),
                           num=num_interv)
    # print(x_approx)

    ymed = []
    yp20 = []
    yp80 = []
    # print(dataset[:, 1])

    for i in range(num_interv - 1):
        # print(len(np.where(dataset[:, 1] > x_approx[i])[0]),
        #       len(np.where(dataset[:, 1] > x_approx[i + 1])[0]))
        if len(np.where(dataset[:, 1] > x_approx[i + 1])[0]) == 0 \
                or len(np.where(dataset[:, 1] > x_approx[i])[0]) == 0 \
                or (len(np.where(dataset[:, 1] > x_approx[i])[0])
                    == len(np.where(dataset[:, 1] > x_approx[i + 1])[0])):
            ymed.append(0)
            yp20.append(0)
            yp80.append(0)
        else:
            xmin = np.where(dataset[:, 1] > x_approx[i])[0][0]
            xmax = np.where(dataset[:, 1] > x_approx[i + 1])[0][0]

            # print(dataset[:, 0][xmin:xmax])

            ymed.append(np.median(dataset[:, 0][xmin:xmax]))
            yp20.append(np.nanpercentile(dataset[:, 0][xmin:xmax], perc_low))
            yp80.append(np.nanpercentile(dataset[:, 0][xmin:xmax], perc_high))

    x_approx = (x_approx[:-1] + x_approx[1:]) / 2.
    y_ceros = np.array(ymed) > 0.
    x_approx = x_approx[y_ceros]
    ymed = np.array(ymed)[y_ceros]
    yp20 = np.array(yp20)[y_ceros]
    yp80 = np.array(yp80)[y_ceros]

    return x_approx, ymed, yp20, yp80


def calculate_mean(dataset, num_interv, perc_low=20, perc_high=80, nmax=60):
    x_approx = np.logspace(np.log10(dataset[0, 1]), np.log10(nmax),
                           num=num_interv)
    # x_approx = np.logspace(np.log10(0.95), np.log10(np.min([60, np.max(dataset[:, 1])])), num=num_interv)

    ymed = []

    for i in range(num_interv - 1):
        if len(np.where(dataset[:, 1] > x_approx[i + 1])[0]) == 0 \
                or len(np.where(dataset[:, 1] > x_approx[i])[0]) == 0 \
                or (len(np.where(dataset[:, 1] > x_approx[i])[0])
                    == len(np.where(dataset[:, 1] > x_approx[i + 1])[0])):
            ymed.append(0)
        else:
            xmin = np.where(dataset[:, 1] > x_approx[i])[0][0]
            xmax = np.where(dataset[:, 1] > x_approx[i + 1])[0][0]

            ymed.append(np.mean(dataset[:, 0][xmin:xmax]))

    x_approx = (x_approx[:-1] + x_approx[1:]) / 2.
    y_ceros = np.array(ymed) > 0.
    x_approx = x_approx[y_ceros]
    ymed = np.array(ymed)[y_ceros]
    return x_approx, ymed


def calculate_gmean(dataset, num_interv, nmax=60):
    x_approx = np.logspace(np.log10(dataset[0, 1]), np.log10(nmax),
                           num=num_interv)

    ymed = []
    for i in range(num_interv - 1):
        # print(len(np.where(dataset[:, 1] > x_approx[i])[0]),
        #       len(np.where(dataset[:, 1] > x_approx[i + 1])[0]))
        if len(np.where(dataset[:, 1] > x_approx[i + 1])[0]) == 0 \
                or len(np.where(dataset[:, 1] > x_approx[i])[0]) == 0 \
                or (len(np.where(dataset[:, 1] > x_approx[i])[0])
                    == len(np.where(dataset[:, 1] > x_approx[i + 1])[0])):
            ymed.append(0)
        else:
            xmin = np.where(dataset[:, 1] > x_approx[i])[0][0]
            xmax = np.where(dataset[:, 1] > x_approx[i + 1])[0][0]

            ymed.append(gmean(dataset[:, 0][xmin:xmax]))

    x_approx = (x_approx[:-1] + x_approx[1:]) / 2.
    y_ceros = np.array(ymed) > 0.
    x_approx = x_approx[y_ceros]
    ymed = np.array(ymed)[y_ceros]

    return x_approx, ymed


def fits2(data, xxx, limit):
    Xlimit = np.where(xxx >= limit)[0][0]
    return np.polyfit(np.log10(xxx[Xlimit:]), np.log10(data[Xlimit:]), 1)


def cumulative(sigma, x1, x0, perc):
    return 0.5 * (1 + spice.erf((x1 - x0) / sigma / 2 ** 0.5)) - perc


def mass_from_Vmax(Vmax, Rmax, c200):
    # From Moline16
    return Vmax ** 2 * Rmax * 3.086e16 * 1e9 / 6.6741e-11 / 1.989e30 * ff(
        c200) / ff(2.163)


def def_Cv(c200, Cv):
    return 200 * ff(2.163) / ff(c200) * (c200 / 2.163) ** 3 - Cv


def ff(x):
    return (np.log(1 + x) - 1 / (1 + x))


def power_law_from_median(median, perct20, perct80, xx, limit):
    fitsM, fitsB = fits2(median, xx, limit)
    print('m and b')
    print("%.3f  %.3f" % (fitsM, fitsB))
    print()

    exponents = []
    exponents.append(fitsB)

    yy = perct20 / xx ** fitsM
    Xlimit = np.where(xx >= limit)[0][0]
    bb = np.polyfit(np.log10(xx[Xlimit:]), np.log10(yy[Xlimit:]), 0)
    exponents.append(bb[0])
    print('Low percentile:  b=%.3f' % bb[0])

    yy = perct80 / xx ** fitsM
    Xlimit = np.where(xx >= limit)[0][0]
    bb = np.polyfit(np.log10(xx[Xlimit:]), np.log10(yy[Xlimit:]), 0)
    exponents.append(bb[0])
    print('High percentile: b=%.3f' % bb[0])

    return fitsM, exponents


def sigma_from_2080(exponents):
    bb20 = opt.root(cumulative, 0.25, args=(exponents[0], exponents[1], 0.2))
    print(bb20['x'], 10 ** (bb20['x'][0]))

    bb80 = opt.root(cumulative, 0.1, args=(exponents[0], exponents[2], 0.8))
    print(bb80['x'], 10 ** (bb80['x'][0]))

    # xmax-xmed = n.sigma in a normal distribution
    print('n from sigmas: %.3f and %.3f' % (
        (-exponents[0] + exponents[1]) / bb20['x'],
        (-exponents[2] + exponents[0]) / bb80['x']))

    return [bb20['x'], bb80['x']]


def C200_from_Cv(yy_med, y20, y80, xx):
    median = Cv_Grand_points(xx, yy_med)
    perct20 = Cv_Grand_points(xx, y20)
    perct80 = Cv_Grand_points(xx, y80)

    C200_med = []
    C200_20 = []
    C200_80 = []

    for i in range(len(median)):
        C200_med.append(opt.root(def_Cv, 40, args=median[i])['x'][0])
        C200_20.append(opt.root(def_Cv, 40, args=perct20[i])['x'][0])
        C200_80.append(opt.root(def_Cv, 40, args=perct80[i])['x'][0])

    return np.array(C200_med), np.array(C200_20), np.array(C200_80)


def Cm_Mol16(M, x, ci=[19.9, -0.195, 0.089, 0.089, -0.54]):
    # Median subhalo concentration depending on its mass and distance
    # to the host center
    # Moline et al. 1603.04057
    #
    # M - tidal mass of the subhalo [Msun]
    # x - distance from the subh to the halo center [kpc]
    return ci[0] * (
            1 + (sum([(ci[i + 1] * np.log10(M * 0.7 / 10 ** 8)) ** (i + 1)
                      for i in range(3)]))) * (
                   1 + ci[4] * np.log10(x / R200))


# %% CALCULATE THE MEDIAN VALUES OF THE DISTRIBUTION Rmax-Vmax

number_bins_dmo = 18
number_bins_hydro = 18

# Limit to the power law tendency
hydroLimit = 10
dmoLimit = 10

cv_dmo_cloud = Cv_Grand_points(Grand_dmo_raw[:, 1], Grand_dmo_raw[:, 0])
cv_hydro_cloud = Cv_Grand_points(Grand_hydro_raw[:, 1], Grand_hydro_raw[:, 0])

Grand_dmo[:, 0] = cv_dmo_cloud
Grand_hydro[:, 0] = cv_hydro_cloud

vv_medians_dmo, cv_dmo_median, cv_dmo_min, cv_dmo_max = calculate_med(
    Grand_dmo, number_bins_dmo, perc_low=16, perc_high=84)
vv_medians_hydro, cv_hydro_median, cv_hydro_min, cv_hydro_max = calculate_med(
    Grand_hydro, number_bins_hydro, perc_low=16, perc_high=84)

vv_means_dmo, cv_dmo_mean = calculate_mean(
    Grand_dmo, number_bins_dmo, perc_low=16, perc_high=84)
vv_means_hydro, cv_hydro_mean = calculate_mean(
    Grand_hydro, number_bins_hydro, perc_low=16, perc_high=84)

vv_gmeans_dmo, cv_dmo_gmean = calculate_gmean(
    Grand_dmo, number_bins_dmo)
vv_gmeans_hydro, cv_hydro_gmean = calculate_gmean(
    Grand_hydro, number_bins_hydro)

plt.close('all')

plt.figure()
plt.ylim(1e3, 2e5)

plt.scatter(Grand_dmo[:, 1], cv_dmo_cloud, color='k', s=3, label='DMO')
plt.scatter(Grand_hydro[:, 1], cv_hydro_cloud, color='limegreen',
            label='Hydro', alpha=0.5, s=3)

plt.plot(vv_medians_dmo, cv_dmo_median, '-k')
plt.plot(vv_medians_hydro, cv_hydro_median, '-g')

plt.plot(vv_medians_dmo, cv_dmo_max, '--k')
plt.plot(vv_medians_hydro, cv_hydro_max, '--g')

plt.plot(vv_medians_dmo, cv_dmo_min, '--k')
plt.plot(vv_medians_hydro, cv_hydro_min, '--g')

plt.xscale('log')
plt.yscale('log')
plt.legend()

plt.xlabel(r'$V_\mathrm{max}$ [km s$^{-1}$]', size=28)
plt.ylabel(r'c$_\mathrm{V}$', size=28)

# FIGURE 2
plt.subplots(12, figsize=(14, 7))
ax1 = plt.subplot(121)

plt.plot(vv_medians_dmo, cv_dmo_median, '.k', ms=13, label='median')
plt.plot(vv_medians_hydro, cv_hydro_median, '.g', ms=13)

plt.plot(vv_means_dmo, cv_dmo_mean, 'xk', ms=13, label='mean')
plt.plot(vv_means_hydro, cv_hydro_mean, 'xg', ms=13)

plt.plot(vv_gmeans_dmo, cv_dmo_gmean, '+k', ms=13, label='gmean')
plt.plot(vv_gmeans_hydro, cv_hydro_gmean, '+g', ms=13)

plt.plot(vv_medians_dmo, cv_dmo_max, '--k')
plt.plot(vv_medians_hydro, cv_hydro_max, '--g')

plt.plot(vv_medians_dmo, cv_dmo_min, '--k')
plt.plot(vv_medians_hydro, cv_hydro_min, '--g')

plt.ylabel(r'c$_\mathrm{V}$', size=28)

plt.xscale('log')
plt.yscale('log')
plt.legend()

plt.xlabel(r'$V_\mathrm{max}$ [km s$^{-1}$]', size=28)

plt.subplot(122, sharex=ax1, sharey=ax1)

print('Median--------------')
print('DMO')
fitsM_dmo, exps_dmo = power_law_from_median(cv_dmo_median, cv_dmo_min,
                                            cv_dmo_max, vv_medians_dmo,
                                            dmoLimit)
print('Scatter: %.3f and %.3f' % (
    exps_dmo[0] - exps_dmo[1], exps_dmo[2] - exps_dmo[0]))

print()
print('Hydro')
fitsM_hydro, exps_hydro = power_law_from_median(cv_hydro_median,
                                                cv_hydro_min, cv_hydro_max,
                                                vv_medians_hydro, hydroLimit)
print('Scatter: %.3f and %.3f' % (exps_hydro[0] - exps_hydro[1],
                                  exps_hydro[2] - exps_hydro[0]))
print()
print()

# print('Mean--------------')
# print('DMO')
# fitsM_dmo_mean, exps_dmo_mean = power_law_from_median(
#     cv_dmo_mean, cv_dmo_min, cv_dmo_max, vv_means_dmo, dmoLimit)
# print('Scatter: %.3f and %.3f' % (exps_dmo_mean[0] - exps_dmo_mean[1],
#                                   exps_dmo_mean[2] - exps_dmo_mean[0]))
#
# print()
# print('Hydro')
# fitsM_hydro_mean, exps_hydro_mean = power_law_from_median(
#     cv_hydro_mean, cv_hydro_min, cv_hydro_max, vv_means_hydro, hydroLimit)
# print('Scatter: %.3f and %.3f' % (exps_hydro_mean[0] - exps_hydro_mean[1],
#                                   exps_hydro_mean[2] - exps_hydro_mean[0]))
#
# print()
# print()

print('Gmean--------------')
print('DMO')
fitsM_dmo_gmean, exps_dmo_gmean = power_law_from_median(
    cv_dmo_gmean, cv_dmo_min, cv_dmo_max, vv_gmeans_dmo, dmoLimit)
print('Scatter: %.3f and %.3f' % (exps_dmo_gmean[0] - exps_dmo_gmean[1],
                                  exps_dmo_gmean[2] - exps_dmo_gmean[0]))

print()
print('Hydro')
fitsM_hydro_gmean, exps_hydro_gmean = power_law_from_median(
    cv_hydro_gmean, cv_hydro_min, cv_hydro_max, vv_gmeans_hydro, hydroLimit)
print('Scatter: %.3f and %.3f' % (exps_hydro_gmean[0] - exps_hydro_gmean[1],
                                  exps_hydro_gmean[2] - exps_hydro_gmean[0]))

xx_plot = np.logspace(np.log10(vv_medians_dmo[0]), np.log10(60))

plt.plot(xx_plot, 10 ** exps_dmo[0] * xx_plot ** fitsM_dmo, '-k', linewidth=3,
         label='median')
plt.plot(xx_plot, 10 ** exps_hydro[0] * xx_plot ** fitsM_hydro, '-g',
         linewidth=3)

# plt.plot(xx_plot, 10 ** exps_dmo_mean[0] * xx_plot ** fitsM_dmo_mean, '--k',
#          linewidth=3, label='mean')
# plt.plot(xx_plot, 10 ** exps_hydro_mean[0] * xx_plot ** fitsM_hydro_mean,
#          '--g',
#          linewidth=3)

plt.plot(xx_plot, 10 ** exps_dmo_gmean[0] * xx_plot ** fitsM_dmo_gmean, ':k',
         linewidth=3, label='gmean')
plt.plot(xx_plot, 10 ** exps_hydro_gmean[0] * xx_plot ** fitsM_hydro_gmean,
         ':g',
         linewidth=3)

plt.plot(vv_medians_dmo, cv_dmo_median, '.k', ms=13)
plt.plot(vv_medians_hydro, cv_hydro_median, '.g', ms=13)

# plt.plot(vv_means_dmo, cv_dmo_mean, 'xk', ms=13)
# plt.plot(vv_means_hydro, cv_hydro_mean, 'xg', ms=13)

plt.axvline(dmoLimit, linestyle='-', color='k', alpha=0.3, linewidth=2)
plt.axvline(hydroLimit, linestyle='-', color='g', alpha=0.3, linewidth=2)

plt.plot(xx_plot, Cv_Mol2021_redshift0(xx_plot), color='red',
         label='Moliné+21', alpha=0.5, linewidth=3)


# fit to moline21
def log10MOline21(V, c0, c1, c2, c3):
    return np.log10(Cv_Mol2021_redshift0(V, c0, c1, c2, c3))


moline_fits = curve_fit(log10MOline21,
                        xdata=vv_medians_dmo[vv_medians_dmo > dmoLimit],
                        ydata=np.log10(
                            cv_dmo_median[vv_medians_dmo > dmoLimit]),
                        p0=[1.75e5, -0.90368, 0.2749, -0.028],
                        bounds=([-np.inf, -5, -5, -5],
                                [+np.inf, 5, 5, 5]))

print(moline_fits)

plt.plot(xx_plot, Cv_Mol2021_redshift0(V=xx_plot,
                                       c0=moline_fits[0][0],
                                       c1=moline_fits[0][1],
                                       c2=moline_fits[0][2],
                                       c3=moline_fits[0][3]),
         color='k',
         label='Moliné+21 fit', alpha=0.5, linewidth=3)

moline_fits = curve_fit(log10MOline21,
                        xdata=vv_medians_hydro[vv_medians_hydro > hydroLimit],
                        ydata=np.log10(
                            cv_hydro_median[vv_medians_hydro > hydroLimit]),
                        p0=[1.75e5, -0.90368, 0.2749, -0.028],
                        bounds=([-np.inf, -np.inf, -np.inf, -np.inf],
                                [5e5, +np.inf, +np.inf, +np.inf]))

print(moline_fits)

plt.plot(xx_plot, Cv_Mol2021_redshift0(V=xx_plot,
                                       c0=moline_fits[0][0],
                                       c1=moline_fits[0][1],
                                       c2=moline_fits[0][2],
                                       c3=moline_fits[0][3]),
         color='g',
         alpha=0.5, linewidth=3)

plt.xscale('log')
plt.yscale('log')
plt.legend()

plt.xlabel(r'$V_\mathrm{max}$ [km s$^{-1}$]', size=28)

# plt.figure()
#
# xmin = np.where(Grand_dmo_raw[:,1]>=10)[0][0]
# xmax = np.where(Grand_dmo_raw[:,1]>20)[0][0]
#
# median_dmo = np.median(Grand_dmo_raw[xmin:xmax, 0])
# mean_dmo = np.mean(Grand_dmo_raw[xmin:xmax, 0])
# gmean_dmo = gmean(Grand_dmo_raw[xmin:xmax, 0])
#
# plt.hist(Grand_dmo_raw[xmin:xmax, 0], log=True, color='k', alpha=0.5,
#          bins=np.logspace(np.log10(np.min(Grand_dmo_raw[xmin:xmax, 0])),
#                           np.log10(np.max(Grand_dmo_raw[xmin:xmax, 0])), num=15))
#
# plt.axvline(median_dmo, linestyle='-', color='k', label='median')
# plt.axvline(mean_dmo, linestyle='--', color='k', label='mean')
# plt.axvline(gmean_dmo, linestyle=':', color='k', label='gmean')
#
# plt.xscale('log')
# #---------------------
# xmin = np.where(Grand_hydro_raw[:,1]>=10)[0][0]
# xmax = np.where(Grand_hydro_raw[:,1]>20)[0][0]
#
# median_dmo = np.median(Grand_hydro_raw[xmin:xmax, 0])
# mean_dmo = np.mean(Grand_hydro_raw[xmin:xmax, 0])
# gmean_dmo = gmean(Grand_hydro_raw[xmin:xmax, 0])
#
# plt.hist(Grand_hydro_raw[xmin:xmax, 0], log=True, color='limegreen', alpha=0.5,
#          bins=np.logspace(np.log10(np.min(Grand_hydro_raw[xmin:xmax, 0])), np.log10(np.max(Grand_hydro_raw[xmin:xmax, 0])), num=15))
#
# plt.axvline(median_dmo, linestyle='-', color='limegreen')
# plt.axvline(mean_dmo, linestyle='--', color='limegreen')
# plt.axvline(gmean_dmo, linestyle=':', color='limegreen')
#
# plt.legend()


# SEPARATE Cv IN RADIAL DISTANCE (TWO BINS ONLY, IN 0.3 R200)
print()
print('Radial subplots')

plt.subplots(2, 2)
for i in range(1, 3):
    if i == 1:
        Grand_dmo = Grand_dmo_raw[Grand_dmo_raw[:, 2] < 0.3 * 220, :]
        Grand_dmo[:, 0] = Cv_Grand_points(Grand_dmo[:, 1], Grand_dmo[:, 0])
        Grand_hydro = Grand_hydro_raw[Grand_hydro_raw[:, 2] < 0.3 * 220, :]
        Grand_hydro[:, 0] = Cv_Grand_points(Grand_hydro[:, 1],
                                            Grand_hydro[:, 0])

    if i == 2:
        Grand_dmo = Grand_dmo_raw[Grand_dmo_raw[:, 2] > 0.3 * 220, :]
        Grand_dmo[:, 0] = Cv_Grand_points(Grand_dmo[:, 1], Grand_dmo[:, 0])
        Grand_hydro = Grand_hydro_raw[Grand_hydro_raw[:, 2] > 0.3 * 220, :]
        Grand_hydro[:, 0] = Cv_Grand_points(Grand_hydro[:, 1],
                                            Grand_hydro[:, 0])

    ax1 = plt.subplot(2, 2, i)
    plt.plot(Grand_dmo[:, 1], Grand_dmo[:, 0], '.k', label='DMO', markersize=3)
    plt.plot(Grand_hydro[:, 1], Grand_hydro[:, 0], '.', label='Hydro',
             color='limegreen', markersize=3)
    plt.axvline(6.4, color='k', alpha=0.5, linestyle='--')

    plt.xscale('log')
    plt.yscale('log')

    plt.ylim(5e3, 3e5)

    number_bins_dmo = 18
    number_bins_hydro = 18

    xx_dmo, yy_dmo, y20_dmo, y80_dmo = calculate_med(
        Grand_dmo, number_bins_dmo, perc_low=16, perc_high=84)
    xx_hydro, yy_hydro, y20_hydro, y80_hydro = calculate_med(
        Grand_hydro, number_bins_hydro, perc_low=16, perc_high=84)

    xx_dmo_mean, yy_dmo_mean = calculate_gmean(Grand_dmo, number_bins_dmo)
    xx_hydro_mean, yy_hydro_mean = calculate_gmean(
        Grand_hydro, number_bins_hydro)

    # Limit to the power law tendency
    hydroLimit = 10
    dmoLimit = 10

    # ------- Calculations of power laws assuming m from median value -----------------------
    print()

    print('DMO')
    fitsM_dmo, exps_dmo = power_law_from_median(yy_dmo, y20_dmo, y80_dmo,
                                                xx_dmo, dmoLimit)
    print('Scatter: %.3f and %.3f' % (
        exps_dmo[0] - exps_dmo[1], exps_dmo[2] - exps_dmo[0]))
    print()
    print()

    print('Hydro')
    fitsM_hydro, exps_hydro = power_law_from_median(yy_hydro, y20_hydro,
                                                    y80_hydro, xx_hydro,
                                                    hydroLimit)
    print('Scatter: %.3f and %.3f' % (
        exps_hydro[0] - exps_hydro[1], exps_hydro[2] - exps_hydro[0]))

    print()
    print('Mean')
    print('DMO')
    fitsM_dmo_mean, exps_dmo_mean = power_law_from_median(yy_dmo_mean, y20_dmo,
                                                          y80_dmo, xx_dmo_mean,
                                                          dmoLimit)
    print('Scatter: %.3f and %.3f' % (
        exps_dmo_mean[0] - exps_dmo_mean[1],
        exps_dmo_mean[2] - exps_dmo_mean[0]))
    print()
    print()
    print('Hydro')
    fitsM_hydro_mean, exps_hydro_mean = power_law_from_median(yy_hydro_mean,
                                                              y20_hydro,
                                                              y80_hydro,
                                                              xx_hydro_mean,
                                                              hydroLimit)
    print('Scatter: %.3f and %.3f' % (exps_hydro_mean[0] - exps_hydro_mean[1],
                                      exps_hydro_mean[2] - exps_hydro_mean[0]))

    plt.subplot(2, 2, i + 2, sharex=ax1, sharey=ax1)

    # xx_plot = np.logspace(np.log10(xx_dmo[0]), np.log10(xx_dmo[-1]))
    xx_plot = np.logspace(np.log10(xx_dmo[0]), np.log10(60))

    '''
    plt.fill_between(xx_plot, 10**exps_dmo[1]*xx_plot**fitsM_dmo,
                              10**exps_dmo[2]*xx_plot**fitsM_dmo,
                     color='grey', alpha=0.3, label=r'1 $\sigma$')

    plt.fill_between(xx_plot, 10**exps_hydro[1]*xx_plot**fitsM_hydro,
                              10**exps_hydro[2]*xx_plot**fitsM_hydro,
                     color='limegreen', alpha=0.4)
    '''
    plt.plot(xx_plot, Cv_Mol2021_redshift0(xx_plot), color='red',
             label='Moliné+21', alpha=0.5, linewidth=3)

    plt.plot(xx_plot, 10 ** exps_dmo[0] * xx_plot ** fitsM_dmo, '-k',
             linewidth=3)
    plt.plot(xx_plot, 10 ** exps_hydro[0] * xx_plot ** fitsM_hydro, '-g',
             linewidth=3)

    plt.plot(xx_plot, 10 ** exps_dmo_mean[0] * xx_plot ** fitsM_dmo_mean,
             '--k', linewidth=3)
    plt.plot(xx_plot, 10 ** exps_hydro_mean[0] * xx_plot ** fitsM_hydro_mean,
             '--g', linewidth=3)

    plt.plot(xx_dmo, yy_dmo, '.k', label='DMO', ms=13)
    plt.plot(xx_hydro, yy_hydro, '.g', label='Hydro', ms=13)

    plt.plot(xx_dmo_mean, yy_dmo_mean, 'xk', ms=13)
    plt.plot(xx_hydro_mean, yy_hydro_mean, 'xg', ms=13)

    plt.axvline(dmoLimit, linestyle='-', color='k', alpha=0.3, linewidth=2)
    plt.axvline(hydroLimit, linestyle='-', color='g', alpha=0.3, linewidth=2)

    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel(r'$V_\mathrm{max}$ [km s$^{-1}$]', size=28)

    plt.annotate('- median \n -- gmean', (20, 6500))

    # plt.xlim(1.3, 53)

plt.subplot(221)
plt.title('Subs < 0.3 * 220')
plt.ylabel(r'$R_\mathrm{max}$ [kpc]', size=22)

plt.subplot(222)
plt.title('Subs > 0.3 * 220')

plt.subplot(223)
plt.ylabel(r'c$_\mathrm{V}$', size=28)
#
#
#
# #%%
#
# print()
# print("COMPARISON BETWEEN Cv AND C200 (assuming NFW)")
#
#
#
# C200_dmo,   C20_dmo,   C80_dmo   = C200_from_Cv(yy_dmo,   y20_dmo,   y80_dmo,   xx_dmo)
# C200_hydro, C20_hydro, C80_hydro = C200_from_Cv(yy_hydro, y20_hydro, y80_hydro, xx_hydro)
#
#
# masses_dmo   = mass_from_Vmax(xx_dmo,   yy_dmo,   C200_dmo)
# masses_hydro = mass_from_Vmax(xx_hydro, yy_hydro, C200_hydro)
#
# dmoLimit_mass = 1.5e8
# hydroLimit_mass = 1.5e8
#
# print('DMO')
# fitsM_dmo_mass,   exps_dmo_mass   = power_law_from_median(C200_dmo,   C20_dmo,   C80_dmo,   masses_dmo,   dmoLimit_mass)
# print('Scatter: %.3f and %.3f' %(exps_dmo_mass[0]-exps_dmo_mass[1], exps_dmo_mass[2]-exps_dmo_mass[0]))
# print()
# print('Hydro')
# fitsM_hydro_mass, exps_hydro_mass = power_law_from_median(C200_hydro, C20_hydro, C80_hydro, masses_hydro, hydroLimit_mass)
# print('Scatter: %.3f and %.3f' %(exps_hydro_mass[0]-exps_hydro_mass[1], exps_hydro_mass[2]-exps_hydro_mass[0]))
#
# #print()
# #print()
# #print('Dmo')
# #bb_dmo   = sigma_from_2080(exps_dmo)
# #print()
# #print('Hydro')
# #bb_hydro = sigma_from_2080(exps_hydro)
#
#
# #%%
# #---------------------------------------------------------------------------------------------------
#
# plt.figure(figsize=(9,9))
#
# xx_plot = np.logspace(np.log10(masses_dmo[0]), np.log10(masses_dmo[-1]))
#
# colors= ['wheat', 'orange', 'darkorange']
# col=0
# for dist in [0.1, 0.3, 1.]:
#     orig = Cm_Mol16(xx_plot, R200*dist)
#     plt.plot(xx_plot, orig, '-', color=colors[col], label=(r'Mol+17 x$_\mathrm{sub}$ = ' + str(dist)))
#     col += 1
#
# plt.plot(masses_dmo,   C200_dmo,   '.', color='k', label='DMO')
# plt.plot(masses_hydro, C200_hydro, '.', color='g', label='Hydro')
#
# #plt.axvline(LIMIT, color=colour, alpha=0.5)
#
#
# plt.plot(xx_plot, 10**exps_dmo_mass  [0]*xx_plot**fitsM_dmo_mass,   '-k')
# plt.plot(xx_plot, 10**exps_hydro_mass[0]*xx_plot**fitsM_hydro_mass, '-g')
#
# plt.fill_between(xx_plot, 10**exps_dmo_mass[1]*xx_plot**fitsM_dmo_mass,
#                           10**exps_dmo_mass[2]*xx_plot**fitsM_dmo_mass,
#                  color='k', alpha=0.3, label=r'1 $\sigma$')
#
# plt.fill_between(xx_plot, 10**exps_hydro_mass[1]*xx_plot**fitsM_hydro_mass,
#                           10**exps_hydro_mass[2]*xx_plot**fitsM_hydro_mass,
#                  color='g', alpha=0.3)
#
#
#
# plt.ylabel(r'$c_{200}$', size=22)
# plt.xlabel(r'$\mathrm{M}$ ($\mathrm{M}_{\mathrm{sun}}$)', size=20)
#
#
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
#
#
# plt.figure()
# plt.plot(xx_dmo, masses_dmo, '.k', label='DMO')
# plt.plot(xx_hydro, masses_hydro, '.g', label='Hydro')
#
# Rmax = 56.7728 # from VLII website
# Vmax = 201.033 # from VLII website
# R_vir = 402. # from Pieri
# r_s_MW = 21.
#
# print('MW mass from NFW transformation: %.2e Msun' %mass_from_Vmax(Vmax, Rmax, R_vir/r_s_MW))
#
# plt.plot(Vmax, mass_from_Vmax(Vmax, Rmax, R_vir/r_s_MW), '+r', label='MW')
#
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
#
# plt.ylabel(r'$\mathrm{M}$ ($\mathrm{M}_{\mathrm{sun}}$)', size=20)
# plt.xlabel(r'V$_{max}$ [km/s]', size=22)
#
#
# #%%
#
# xxx = np.logspace(0, np.log10(60))
# #plt.close('all')
# plt.figure()
#
# plt.plot(xxx, 10**(exps_dmo[0]-exps_hydro[0]) * xxx**(fitsM_dmo-fitsM_hydro))
#
#
# plt.xscale('log')
# #plt.legend()
#
# plt.xlabel(r'V$_{max}$ [km/s]', size=22)
# plt.ylabel(r'$\frac{c_{V, DMO}}{c_{V, Hydro}}$', size=26)
plt.show()
