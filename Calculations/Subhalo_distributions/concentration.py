#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 10:29:55 2022

@author: saraporras
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import matplotlib.patches as mpatches

from scipy.optimize import curve_fit

all_size = 24
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['axes.labelsize'] = all_size
plt.rcParams['lines.markersize'] = 10
plt.rc('font', size=all_size)
plt.rc('axes', titlesize=all_size)
plt.rc('axes', labelsize=all_size)
plt.rc('xtick', labelsize=all_size)
plt.rc('ytick', labelsize=all_size)
plt.rc('legend', fontsize=20)
plt.rc('figure', titlesize=all_size)
plt.rc('xtick', top=True, direction='in')
plt.rc('ytick', right=True, direction='in')
plt.rc('xtick.major', size=10, width=2, top=False, pad=10)
plt.rc('ytick.major', size=10, width=2, right=True, pad=10)
plt.rc('xtick.minor', size=7, width=1.5, top=False)
plt.rc('ytick.minor', size=7, width=1.5)
#
#        Rmax[kpc]        Vmax[km/s]      Radius[Mpc]
try:
    Grand_dmo = np.loadtxt(
        '../Data_subhalos_simulations/RmaxVmaxRadDMO0_1.txt')
    Grand_hydro = np.loadtxt(
        '../Data_subhalos_simulations/RmaxVmaxRadFP0_1.txt')

    data_release_dmo = np.loadtxt(
        '../Data_subhalos_simulations/dmo_table.txt')
    data_release_hydro = np.loadtxt(
        '../Data_subhalos_simulations/hydro_table.txt')

except:
    Grand_dmo = np.loadtxt('../../RmaxVmaxRadDMO0_1.txt')
    Grand_hydro = np.loadtxt('../../RmaxVmaxRadFP0_1.txt')

Grand_hydro = Grand_hydro[Grand_hydro[:, 1] > 1e-4, :]

Grand_dmo = Grand_dmo[Grand_dmo[:, 0] > 2 * 0.184, :]
Grand_hydro = Grand_hydro[Grand_hydro[:, 0] > 2 * 0.184, :]

Grand_dmo[:, 2] *= 1e3
Grand_hydro[:, 2] *= 1e3

Grand_dmo = Grand_dmo[np.argsort(Grand_dmo[:, 1])]
Grand_hydro = Grand_hydro[np.argsort(Grand_hydro[:, 1])]

data_release_dmo = data_release_dmo[np.argsort(data_release_dmo[:, 1])]
data_release_hydro = data_release_hydro[np.argsort(data_release_hydro[:, 1])]

Grand_dmo_raw = Grand_dmo.copy()
Grand_hydro_raw = Grand_hydro.copy()


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


def Moline21_normalization(V, c0):
    return (Cv_Mol2021_redshift0(V, c0, c1=-0.90368,
                                 c2=0.2749, c3=-0.028))

def Moline21_normalization2(V, c0, c1):
    return (Cv_Mol2021_redshift0(V, c0, c1,
                                 c2=0., c3=0.0))


def Cv_Grand(Vmax, mm, bb):
    return 2. * (10 ** (-bb) * Vmax ** (1. - mm) / H0) ** 2


def Cv_Grand_points(xx, yy):
    return 2. * (xx / yy / H0) ** 2


H0 = 70. / 1e3


def calculate_med(dataset, num_interv, perc_low=20, perc_high=80, nmax=60):
    x_approx = np.geomspace(dataset[0, 1], nmax, num=num_interv)

    ymed = []
    yp20 = []
    yp80 = []
    std = []
    # print(dataset[:, 1])

    for i in range(num_interv - 1):
        try:
            xmin = np.where(dataset[:, 1] > x_approx[i])[0][0]
            xmax = np.where(dataset[:, 1] > x_approx[i + 1])[0][0]

            ymed.append(np.median(dataset[:, 0][xmin:xmax]))
            aaa = np.std(np.log10(dataset[:, 0][xmin:xmax]))

            std.append(aaa)
            yp20.append(np.nanpercentile(dataset[:, 0][xmin:xmax], perc_low))
            yp80.append(np.nanpercentile(dataset[:, 0][xmin:xmax], perc_high))

        except:
            ymed.append(0)
            yp20.append(0)
            yp80.append(0)
            std.append(0)

    x_approx = (x_approx[:-1] + x_approx[1:]) / 2.
    y_ceros = np.array(ymed) > 0.
    x_approx = x_approx[y_ceros]
    ymed = np.array(ymed)[y_ceros]
    yp20 = np.array(yp20)[y_ceros]
    yp80 = np.array(yp80)[y_ceros]
    std = np.array(std)[y_ceros]

    return x_approx, ymed, yp20, yp80, std


# %% CALCULATE THE MEDIAN VALUES OF THE DISTRIBUTION Rmax-Vmax

number_bins_dmo = 20
number_bins_hydro = 25

# Limit to the power law tendency
hydroLimit = 11
dmoLimit = 11

cv_dmo_cloud = Cv_Grand_points(Grand_dmo_raw[:, 1], Grand_dmo_raw[:, 0])
cv_hydro_cloud = Cv_Grand_points(Grand_hydro_raw[:, 1], Grand_hydro_raw[:, 0])

cv_dmo_cloud_release = Cv_Grand_points(data_release_dmo[:, 1],
                                       data_release_dmo[:, 0])
cv_hydro_cloud_release = Cv_Grand_points(data_release_hydro[:, 1],
                                         data_release_hydro[:, 0])

data_release_dmo[:, 0] = cv_dmo_cloud_release
data_release_hydro[:, 0] = cv_hydro_cloud_release

Grand_dmo[:, 0] = cv_dmo_cloud
Grand_hydro[:, 0] = cv_hydro_cloud

vv_medians_dmo, cv_dmo_median, cv_dmo_min, cv_dmo_max, _ = calculate_med(
    Grand_dmo, number_bins_dmo, perc_low=16, perc_high=84)
vv_medians_hydro, cv_hydro_median, cv_hydro_min, cv_hydro_max, _ = calculate_med(
    Grand_hydro, number_bins_hydro, perc_low=16, perc_high=84)

vv_medians_dmo_release, cv_dmo_median_release, _, _, _ = calculate_med(
    data_release_dmo, number_bins_dmo, perc_low=16, perc_high=84, nmax=150)
vv_medians_hydro_release, cv_hydro_median_release, _, _, _ = calculate_med(
    data_release_hydro, number_bins_hydro, perc_low=16, perc_high=84, nmax=150)

xx_plot = np.geomspace(1., 100)

# FIGURE start ---------------------------------------------------------
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(20, 8))
plt.subplot(121)
plt.plot(vv_medians_dmo_release, cv_dmo_median_release,
         ms=10, marker='+', markeredgewidth=2, ls='',
         alpha=1, color='purple', zorder=15, label='Release data')
plt.plot(vv_medians_hydro_release, cv_hydro_median_release,
         ms=10, marker='+', markeredgewidth=2, ls='',
         alpha=1, color='orange', zorder=15)
plt.plot(vv_medians_dmo, cv_dmo_median, '.k', ms=13, zorder=10,
         label='2 years old')
plt.plot(vv_medians_hydro, cv_hydro_median, '.',
         color='green', ms=13, zorder=10)

plt.plot(xx_plot, Cv_Mol2021_redshift0(xx_plot), color='red',
         label='Moliné+21', alpha=0.6, linewidth=3, zorder=9)


# DMO fit -----------------------------------------------------
moline_fits_dmo = curve_fit(
    Moline21_normalization,
    xdata=vv_medians_dmo[vv_medians_dmo > dmoLimit],
    ydata=cv_dmo_median[vv_medians_dmo > dmoLimit],
    p0=[1.e4],
    # bounds=([-np.inf],
    #         [5e5])
)

print(moline_fits_dmo)

plt.plot(xx_plot,
         Moline21_normalization(V=xx_plot,
                                c0=moline_fits_dmo[0][0]),
         color='k',
         alpha=1,
         linewidth=3,
         zorder=5, label='Fit to 2 year')
print(moline_fits_dmo[0][0] + moline_fits_dmo[1][0] ** 0.5,
      moline_fits_dmo[0][0]
      - moline_fits_dmo[1][0] ** 0.5
      )

# Release ---
moline_fits_dmo_release = curve_fit(
    Moline21_normalization2,
    xdata=vv_medians_dmo_release[vv_medians_dmo_release > dmoLimit],
    ydata=cv_dmo_median_release[vv_medians_dmo_release > dmoLimit],
    p0=[1.e4, -0.5],
    # bounds=([-np.inf],
    #         [5e5])
)

print(moline_fits_dmo_release)

plt.plot(xx_plot,
         Cv_Mol2021_redshift0(V=xx_plot,
                              c0=moline_fits_dmo_release[0][0],
                              c1=moline_fits_dmo_release[0][1],
                              c2=0.,
                              c3=0.),
         color='purple',
         alpha=1,
         linewidth=3, ls='--',
         zorder=5, label='Fit to release')
print(moline_fits_dmo_release[0][0] + moline_fits_dmo_release[1][0] ** 0.5,
      moline_fits_dmo_release[0][0]
      - moline_fits_dmo_release[1][0] ** 0.5
      )


# Hydro fit -----------------------------------------------------------
moline_fits_hydro = curve_fit(
    Moline21_normalization,
    xdata=vv_medians_hydro[vv_medians_hydro > hydroLimit],
    ydata=cv_hydro_median[vv_medians_hydro > hydroLimit],
    p0=[1.e4],
    # bounds=([-np.inf],
    #         [5e5])
)
print(moline_fits_hydro[0][0] + moline_fits_hydro[1][0] ** 0.5,
      moline_fits_hydro[0][0]
      - moline_fits_hydro[1][0] ** 0.5
      )
print(moline_fits_hydro)

plt.plot(xx_plot,
         Moline21_normalization(V=xx_plot, c0=moline_fits_hydro[0][0]),
         color='green',
         alpha=1,
         linewidth=3,
         zorder=5)

# Release ---
try:
    moline_fits_hydro_release = curve_fit(
        Moline21_normalization2,
        xdata=vv_medians_hydro_release[vv_medians_hydro_release > hydroLimit],
        ydata=cv_hydro_median_release[vv_medians_hydro_release > hydroLimit],
        p0=[1.e4, -0.5],
        # bounds=([-np.inf],
        #         [5e5])
    )
    print(moline_fits_hydro_release[0][0]
          + moline_fits_hydro_release[1][0] ** 0.5,
          moline_fits_hydro_release[0][0]
          - moline_fits_hydro_release[1][0] ** 0.5
          )
    print(moline_fits_hydro_release)

    plt.plot(xx_plot,
             Cv_Mol2021_redshift0(V=xx_plot,
                                  c0=moline_fits_hydro_release[0][0],
                                  c1=moline_fits_hydro_release[0][1],
                                  c2=0.,
                                  c3=0.),
             color='orange',
             alpha=1,
             linewidth=3, ls='--',
             zorder=5)
except:
    print('Hydro release did not found good parameters')

plt.axvline(dmoLimit, linestyle='-.', color='k', alpha=0.3,
            linewidth=2, label='Fit limit')

plt.xscale('log')
plt.yscale('log')

handles = (mpatches.Patch(color='k', label='DMO', alpha=0.8),
           mpatches.Patch(color='limegreen', label='Hydro', alpha=0.8)
           )

legend11 = plt.legend(handles=handles,
                      loc=1)  # , bbox_to_anchor=(1.001, 0.99))

plt.ylabel(r'c$_\mathrm{V}$', size=28)
plt.xlabel(r'$V_\mathrm{max}$ [km s$^{-1}$]', size=28)


plt.subplot(122) # -------------------------------------------------

xx2_plot = np.geomspace(0.01, 6, num=100)

num = 25

vv_over_dmo = Grand_dmo[Grand_dmo[:, 1] > dmoLimit, 1]
cc_over_dmo = Grand_dmo[Grand_dmo[:, 1] > dmoLimit, 0]

n_dmo, bins_dmo, _ = plt.hist(cc_over_dmo / Moline21_normalization(
    V=vv_over_dmo, c0=moline_fits_dmo[0][0]),
                              bins=np.geomspace(0.05, 6, num=num),
                              density=True,
                              color='grey', alpha=0.7
                              )

num = 25
vv_over_hydro = Grand_hydro[Grand_hydro[:, 1] > hydroLimit, 1]
cc_over_hydro = Grand_hydro[Grand_hydro[:, 1] > hydroLimit, 0]

n_hydro, bins_hydro, _ = plt.hist(cc_over_hydro / Moline21_normalization(
    V=vv_over_hydro, c0=moline_fits_hydro[0][0]),
                                  bins=np.geomspace(0.05, 6, num=num),
                                  density=True,
                                  color='limegreen', alpha=0.7
                                  )


plt.xlabel(r'$\frac{c_\mathrm{V, data}}{c_\mathrm{V, Moliné21}(V)}$',
           fontsize=34)
plt.ylabel('Normalized density')


def lognormal_fit(xx, mean, sigma):
    return (1 / (sigma * xx * np.sqrt(2. * np.pi))
            * np.exp(-0.5 * ((np.log(xx) - mean) / sigma) ** 2.))


fit_lognormal_dmo = curve_fit(
    lognormal_fit,
    xdata=(bins_dmo[1:] + bins_dmo[:-1]) / 2.,
    ydata=n_dmo)

print('Sigma for the lognormal distribution, dmo: ',
      np.log10(np.exp(fit_lognormal_dmo[0][1])))

plt.plot(xx2_plot,
         lognormal_fit(xx2_plot, fit_lognormal_dmo[0][0],
                       fit_lognormal_dmo[0][1]),
         color='k', lw=2)

fit_lognormal_hydro = curve_fit(
    lognormal_fit,
    xdata=(bins_hydro[1:] + bins_hydro[:-1]) / 2.,
    ydata=n_hydro)

print('Sigma for the lognormal distribution, hydro: ',
      np.log10(np.exp(fit_lognormal_hydro[0][1])))

plt.plot(xx2_plot,
         lognormal_fit(xx2_plot, fit_lognormal_hydro[0][0],
                       fit_lognormal_hydro[0][1]),
         color='green', lw=2)

plt.xscale('log')
lg1 = plt.legend(handles=handles, loc=1)
lg2 = plt.legend([plt.Line2D([], [],
                             linestyle='-', lw=2,
                             color='k')],
                 ['Lognormal fit'],
                 loc=2, framealpha=1)
ax2.add_artist(lg1)
ax2.add_artist(lg2)

# -----------------------------------------------------------
def gaussian_not(xx, sigma, x0, aa):
    return aa / ((2. * np.pi) ** 0.5 * sigma) * np.exp(
        -0.5 * ((xx - x0) / sigma) ** 2.)


fit_normal_dmo = curve_fit(
    gaussian_not,
    xdata=np.log10((bins_dmo[1:] + bins_dmo[:-1]) / 2.),
    ydata=n_dmo)

fit_normal_hydro = curve_fit(
    gaussian_not,
    xdata=np.log10((bins_hydro[1:] + bins_hydro[:-1]) / 2.),
    ydata=n_hydro)


ax.fill_between(xx_plot,
                Moline21_normalization(
                    V=xx_plot,
                    c0=moline_fits_dmo[0][0])
                * 10 ** -fit_normal_dmo[0][0],
                Moline21_normalization(
                    V=xx_plot,
                    c0=moline_fits_dmo[0][0]
                       * 10 ** fit_normal_dmo[0][0]),
                color='grey', alpha=0.5, zorder=3
                )
ax.fill_between(xx_plot,
                Moline21_normalization(
                    V=xx_plot,
                    c0=moline_fits_hydro[0][0])
                * 10 ** -fit_normal_hydro[0][
                    0],
                Moline21_normalization(
                    V=xx_plot,
                    c0=moline_fits_hydro[0][0])
                * 10 ** fit_normal_hydro[0][0],
                color='limegreen', alpha=0.5, zorder=2
                )


legend_types = ax.legend(fontsize=18, loc=4, framealpha=1)
ax.add_artist(legend11)
ax.add_artist(legend_types)


fig.savefig('outputs/cv_median.pdf', bbox_inches='tight')
fig.savefig('outputs/cv_median.png', bbox_inches='tight')
plt.show()
