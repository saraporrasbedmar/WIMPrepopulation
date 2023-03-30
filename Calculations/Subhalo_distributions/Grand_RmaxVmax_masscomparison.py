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
    x_approx = np.logspace(np.log10(np.nanmin(dataset[:, 1])), np.log10(nmax),
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

    x_approx = 10 ** ((np.log10(x_approx[:-1]) + np.log10(x_approx[1:]))
                      / 2.)
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

    x_approx = 10 ** ((np.log10(x_approx[:-1]) + np.log10(x_approx[1:]))
                      / 2.)
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

    x_approx = 10 ** ((np.log10(x_approx[:-1]) + np.log10(x_approx[1:]))
                      / 2.)
    y_ceros = np.array(ymed) > 0.
    x_approx = x_approx[y_ceros]
    ymed = np.array(ymed)[y_ceros]

    return x_approx, ymed


def fits2(data, xxx, limit):
    Xlimit = np.where(xxx >= limit)[0][0]
    return np.polyfit(np.log10(xxx[Xlimit:]), np.log10(data[Xlimit:]), 1)


def cumulative(sigma, x1, x0, perc):
    return 0.5 * (1 + spice.erf((x1 - x0) / sigma / 2 ** 0.5)) - perc


def mass_from_Vmax(Vmax, Rmax, c200,
                   cosmo_G=4.297e-6):
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


def C200_from_Cv(cv):
    C200_med = []

    for i in cv:
        C200_med.append(opt.root(def_Cv, 40, args=i)['x'][0])

    return np.array(C200_med)


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

vv_gmeans_dmo, cv_dmo_gmean = calculate_gmean(
    Grand_dmo, number_bins_dmo)
vv_gmeans_hydro, cv_hydro_gmean = calculate_gmean(
    Grand_hydro, number_bins_hydro)

print("COMPARISON BETWEEN Cv AND C200 (assuming NFW)")

C200_dmo = C200_from_Cv(cv_dmo_cloud)
C200_hydro = C200_from_Cv(cv_hydro_cloud)

masses_dmo = mass_from_Vmax(Grand_dmo_raw[:, 1], Grand_dmo_raw[:, 0],
                            C200_dmo)
masses_hydro = mass_from_Vmax(Grand_hydro_raw[:, 1], Grand_hydro_raw[:, 0],
                              C200_hydro)

plt.subplots(12)

plt.subplot(121)
plt.scatter(cv_dmo_cloud, C200_dmo, color='k', alpha=0.5)
plt.scatter(cv_hydro_cloud, C200_hydro, color='g', alpha=0.5)

plt.xlabel('Cv')
plt.ylabel('C200')

plt.xscale('log')

plt.subplot(122)
plt.scatter(cv_dmo_cloud, masses_dmo, color='k', alpha=0.5)
plt.scatter(cv_hydro_cloud, masses_hydro, color='g', alpha=0.5)

plt.xlabel('Cv')
plt.ylabel('Mass')

plt.xscale('log')
plt.yscale('log')

# Calculate median ang gmean values for the population

bins_mass_dmo = 15
bins_mass_hydro = 15


plt.subplots(12)
ax1 = plt.subplot(121)
plt.scatter(masses_dmo, C200_dmo, color='k', alpha=0.5)
plt.scatter(masses_hydro, C200_hydro, color='limegreen', alpha=0.5)

x_dmo_median, c200_dmo_median, c200_dmo_p20, c200_dmo_p80 = \
    calculate_med(
        np.column_stack((C200_dmo, masses_dmo)),
        bins_mass_dmo,
        nmax=2e11
    )
print(x_dmo_median)
x_hydro_median, c200_hydro_median, c200_hydro_p20, c200_hydro_p80 = \
    calculate_med(
        np.column_stack((C200_hydro, masses_hydro)),
        bins_mass_hydro,
        nmax=2e11#np.nanmax(masses_hydro)
    )

plt.plot(x_dmo_median, c200_dmo_median, '-k')
plt.plot(x_dmo_median, c200_dmo_p20, '--k')
plt.plot(x_dmo_median, c200_dmo_p80, '--k')

plt.plot(x_hydro_median, c200_hydro_median, '-g')
plt.plot(x_hydro_median, c200_hydro_p20, '--g')
plt.plot(x_hydro_median, c200_hydro_p80, '--g')

plt.ylabel('C200')
plt.xlabel('Mass')

plt.xscale('log')
plt.yscale('log')


dmoLimit_mass = 1.5e8
hydroLimit_mass = 1.5e8

print('DMO')
fitsM_dmo_mass, exps_dmo_mass = power_law_from_median(
    c200_dmo_median, c200_dmo_p20, c200_dmo_p80, x_dmo_median,
    dmoLimit_mass)
print('Scatter: %.3f and %.3f' % (
    exps_dmo_mass[0] - exps_dmo_mass[1], exps_dmo_mass[2] - exps_dmo_mass[0]))
print()
print('Hydro')
fitsM_hydro_mass, exps_hydro_mass = power_law_from_median(
    c200_hydro_median, c200_hydro_p20, c200_hydro_p80, x_hydro_median,
    hydroLimit_mass)
print('Scatter: %.3f and %.3f' % (exps_hydro_mass[0] - exps_hydro_mass[1],
                                  exps_hydro_mass[2] - exps_hydro_mass[0]))


plt.subplot(122, sharex=ax1, sharey=ax1)
xx_plot = np.logspace(np.log10(masses_dmo[0]), np.log10(masses_dmo[-1]))


plt.plot(x_dmo_median, c200_dmo_median, '-k')
plt.plot(x_dmo_median, c200_dmo_p20, '--k')
plt.plot(x_dmo_median, c200_dmo_p80, '--k')

plt.plot(x_hydro_median, c200_hydro_median, '-g')
plt.plot(x_hydro_median, c200_hydro_p20, '--g')
plt.plot(x_hydro_median, c200_hydro_p80, '--g')

plt.plot(xx_plot, 10 ** exps_dmo_mass[0] * xx_plot ** fitsM_dmo_mass,
         'k', linestyle='dotted')
plt.plot(xx_plot, 10 ** exps_hydro_mass[0] * xx_plot ** fitsM_hydro_mass,
         'g', linestyle='dotted')

plt.fill_between(xx_plot, 10 ** exps_dmo_mass[1] * xx_plot ** fitsM_dmo_mass,
                 10 ** exps_dmo_mass[2] * xx_plot ** fitsM_dmo_mass,
                 color='k', alpha=0.3, label=r'1 $\sigma$')

plt.fill_between(xx_plot,
                 10 ** exps_hydro_mass[1] * xx_plot ** fitsM_hydro_mass,
                 10 ** exps_hydro_mass[2] * xx_plot ** fitsM_hydro_mass,
                 color='g', alpha=0.3)


plt.axvline(dmoLimit_mass, linestyle='-', color='k',
            alpha=0.3, linewidth=2)
plt.axvline(hydroLimit_mass, linestyle='-', color='g',
            alpha=0.3, linewidth=2)

plt.ylabel('C200')
plt.xlabel('Mass')

plt.xscale('log')
plt.yscale('log')

# %%
# ---------------------------------------------------------------------------------------------------

plt.figure(figsize=(9, 9))

xx_plot = np.logspace(np.log10(masses_dmo[0]), np.log10(masses_dmo[-1]))

colors = ['wheat', 'orange', 'darkorange']
col = 0
for dist in [0.1, 0.3, 1.]:
    orig = Cm_Mol16(xx_plot, R200 * dist)
    plt.plot(xx_plot, orig, '-', color=colors[col],
             label=(r'Mol+17 x$_\mathrm{sub}$ = ' + str(dist)))
    col += 1

plt.plot(masses_dmo, C200_dmo, '.', color='k', label='DMO')
plt.plot(masses_hydro, C200_hydro, '.', color='g', label='Hydro')

# plt.axvline(LIMIT, color=colour, alpha=0.5)


plt.plot(xx_plot, 10 ** exps_dmo_mass[0] * xx_plot ** fitsM_dmo_mass, '-k')
plt.plot(xx_plot, 10 ** exps_hydro_mass[0] * xx_plot ** fitsM_hydro_mass, '-g')

plt.fill_between(xx_plot, 10 ** exps_dmo_mass[1] * xx_plot ** fitsM_dmo_mass,
                 10 ** exps_dmo_mass[2] * xx_plot ** fitsM_dmo_mass,
                 color='k', alpha=0.3, label=r'1 $\sigma$')

plt.fill_between(xx_plot,
                 10 ** exps_hydro_mass[1] * xx_plot ** fitsM_hydro_mass,
                 10 ** exps_hydro_mass[2] * xx_plot ** fitsM_hydro_mass,
                 color='g', alpha=0.3)

plt.axvline(dmoLimit_mass, linestyle='-', color='k',
            alpha=0.3, linewidth=2)
plt.axvline(hydroLimit_mass, linestyle='-', color='g',
            alpha=0.3, linewidth=2)

plt.ylabel(r'$c_{200}$', size=22)
plt.xlabel(r'$\mathrm{M}$ ($\mathrm{M}_{\mathrm{sun}}$)', size=20)

plt.xscale('log')
plt.yscale('log')
plt.legend()

plt.figure()
plt.plot(Grand_dmo_raw[:, 1], masses_dmo, '.k', label='DMO')
plt.plot(Grand_hydro_raw[:, 1], masses_hydro, '.g', label='Hydro')

Rmax = 56.7728  # from VLII website
Vmax = 201.033  # from VLII website
R_vir = 402.  # from Pieri
r_s_MW = 21.

print('MW mass from NFW transformation: %.2e Msun' % mass_from_Vmax(Vmax, Rmax,
                                                                    R_vir / r_s_MW))

plt.plot(Vmax, mass_from_Vmax(Vmax, Rmax, R_vir / r_s_MW), '+r', label='MW')

plt.xscale('log')
plt.yscale('log')
plt.legend()

plt.ylabel(r'$\mathrm{M}$ ($\mathrm{M}_{\mathrm{sun}}$)', size=20)
plt.xlabel(r'V$_{max}$ [km/s]', size=22)

# %%

xxx = np.logspace(0, np.log10(60))
# plt.close('all')
# plt.figure()
#
# plt.plot(xxx, 10 ** (exps_dmo[0] - exps_hydro[0]) * xxx ** (
#         fitsM_dmo - fitsM_hydro))
#
# plt.xscale('log')
# # plt.legend()
#
# plt.xlabel(r'V$_{max}$ [km/s]', size=22)
# plt.ylabel(r'$\frac{c_{V, DMO}}{c_{V, Hydro}}$', size=26)
plt.show()
