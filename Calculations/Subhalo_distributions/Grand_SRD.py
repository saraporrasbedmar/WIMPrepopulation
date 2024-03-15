#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 20:37:40 2022

@author: saraporras
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

import scipy.optimize as opt
from scipy import integrate
from scipy.interpolate import UnivariateSpline

all_size = 26
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['axes.labelsize'] = all_size
plt.rcParams['lines.markersize'] = 10
plt.rc('font', size=all_size)
plt.rc('axes', titlesize=all_size)
plt.rc('axes', labelsize=all_size)
plt.rc('xtick', labelsize=all_size)
plt.rc('ytick', labelsize=all_size)
plt.rc('legend', fontsize=22)
plt.rc('figure', titlesize=all_size)
plt.rc('xtick', top=True, direction='in')
plt.rc('ytick', right=True, direction='in')
plt.rc('xtick.major', size=10, width=2, top=True, pad=10)
plt.rc('ytick.major', size=10, width=2, right=True, pad=10)
plt.rc('xtick.minor', size=7, width=1.5)
plt.rc('ytick.minor', size=7, width=1.5)

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

# Grand_hydro = Grand_hydro[Grand_hydro[:,1]>1e-4, :]
Grand_hydro = Grand_hydro[Grand_hydro[:, 1] > np.min(Grand_dmo[:, 1]), :]

# Grand_dmo[:,2] *= 1e3
# Grand_hydro[:,2] *= 1e3

Grand_dmo = Grand_dmo[np.argsort(Grand_dmo[:, 2])]
Grand_hydro = Grand_hydro[np.argsort(Grand_hydro[:, 2])]

data_release_dmo = data_release_dmo[np.argsort(data_release_dmo[:, 2])]
data_release_hydro = data_release_hydro[np.argsort(data_release_hydro[:, 2])]


data_release_dmo[:, 2] = data_release_dmo[:, 2]*1e-3
data_release_hydro[:, 2] = data_release_hydro[:, 2]*1e-3

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

        n_final.append(y)
        a += y

    y = len(data) - X2limit
    n_final.append(y)

    a += y
    print(a)

    return n_final


def encontrar_SRD(data):
    n_final = []
    a = 0
    for delta in range(len(bins) - 2):
        # print(delta, bins[delta])
        X1limit = np.where(data[:, 2] >= bins[delta])[0][0]
        X2limit = np.where(data[:, 2] > bins[delta + 1])[0][0]

        y = X2limit - X1limit
        vol = 4 / 3 * np.pi * (
                bins[delta + 1] ** 3 - bins[delta] ** 3)  # * 1e9

        n_final.append(y / vol)
        a += y

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
print('Density figure')
'''
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

plt.plot(x_med_kpc, (0.0711) / vol,
         '--', marker='*', ms=10,
         color='grey', alpha=0.7, label='res_maximum N/Ntot')
plt.plot(x_med_kpc, (0.08394) / vol,
         '--', marker='*', ms=10, color='limegreen', alpha=0.7)

xmin = 2
xmax = 11
log_tend_dmo = np.polyfit(np.log10(x_med[xmin:xmax] * 1e3),
                          srd_dmo[xmin:xmax], 1)
print('max res dmo log-log:   ', log_tend_dmo)

xmin = 5
xmax = 11
log_tend_hydro = np.polyfit(np.log10(x_med[xmin:xmax] * 1e3),
                            srd_hydro[xmin:xmax], 1)
print('max res hydro log-log: ', log_tend_hydro)

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
print('DMO frag:  ', yy_dmo[0])

yy_hyd = opt.curve_fit(N_Dgc_Cosmic, x_med[:X_max], srd_hydro[:X_max],
                       p0=[0.5, 0.5, 5])
peak_hyd = yy_hyd[0][0] * 1e3 * yy_hyd[0][1] / yy_hyd[0][2]
plt.plot(xxx * 1e3, resilient(xxx, yy_hyd), linestyle='dashdot', alpha=0.7,
         color='limegreen')
plt.plot(xxx * 1e3, fragile(xxx, yy_hyd, np.min(Grand_hydro[:, 2])), alpha=0.7,
         color='limegreen', ls=':')
plt.axvline(peak_hyd, alpha=0.5, color='limegreen')
# plt.plot(x_med*1e3, resilient(x_med, yy_hyd), 'x', ms=10, alpha=0.7,
# color='limegreen')
print('Hydro frag:', yy_hyd[0])

print('first subhalos dmo', np.min(Grand_dmo[:, 2]))
print('first subhalos hydro', np.min(Grand_hydro[:, 2]))

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
print()
print('N/Ntot figures')

srd_dmo_sinVol = (np.array(encontrar_SRD_sinVol(Grand_dmo))
                  / len(Grand_dmo))
srd_hydro_sinVol = (np.array(encontrar_SRD_sinVol(Grand_hydro))
                    / len(Grand_hydro))

x_kpc = x_med * 1e3
xxx = np.geomspace(4, 260, num=300)

fig = plt.figure(2, figsize=(11, 10))
ax1 = fig.gca()

# plt.plot(x_kpc, srd_dmo_sinVol, '-', color='k')
plt.plot(x_kpc, srd_dmo_sinVol, color='k', marker='.', ms=15,
         label='Data', linestyle='')

# plt.plot(x_kpc, srd_hydro_sinVol, '-', color='g')
plt.plot(x_kpc, srd_hydro_sinVol, color='limegreen', marker='.', ms=15,
         linestyle='')

xmin = 6
xmax = 11
linear_tend_dmo = np.polyfit(np.log10(x_kpc[xmin:xmax]),
                             np.log10(srd_dmo_sinVol[xmin:xmax]), 1)
print('max res dmo log-log:   ', linear_tend_dmo)

xmin = 6
xmax = 11
linear_tend_hydro = np.polyfit(np.log10(x_kpc[xmin:xmax]),
                               np.log10(srd_hydro_sinVol[xmin:xmax]), 1)
print('max res hydro log-log: ', linear_tend_hydro)
plt.plot(xxx, xxx ** linear_tend_dmo[0] * 10 ** linear_tend_dmo[1],
         linestyle='dotted', color='k', alpha=1, label='Resilient fit',
         lw=2)
plt.plot(xxx, xxx ** linear_tend_hydro[0] * 10 ** linear_tend_hydro[1],
         linestyle='dotted', color='limegreen', alpha=1, lw=2)

# vol = 4 / 3 * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
# plt.plot(x_kpc, (np.log10(x_kpc) * log_tend_dmo[0] + log_tend_dmo[1]) * vol,
#          '-', marker='+', ms=10,
#          color='grey', alpha=0.7, label='res_maximum density')
# plt.plot(x_kpc,
#          (np.log10(x_kpc) * log_tend_hydro[0] + log_tend_hydro[1]) * vol,
#          '-', marker='+', ms=10,
#          color='limegreen', alpha=0.7)

xxx = np.geomspace(4, 220, num=300)
x_xpc_long = np.append(x_kpc[:X_max], 220.)
srd_dmo_sinVol_long = np.append(srd_dmo_sinVol[:X_max],
                                srd_dmo_sinVol[:X_max][-1])
srd_hydro_sinVol_long = np.append(srd_hydro_sinVol[:X_max],
                                srd_hydro_sinVol[:X_max][-1])
spline_dmo1 = UnivariateSpline(np.log10(x_xpc_long),
                               np.log10(srd_dmo_sinVol_long),
                               k=1, s=0, ext=0)
plt.plot(xxx, 10 ** spline_dmo1(np.log10(xxx))
         * (xxx > np.min(Grand_dmo[:, 2]*1e3)),
         '--', color='k', label='Fragile fit', lw=2)

spline_hydro1 = UnivariateSpline(np.log10(x_xpc_long),
                                 np.log10(srd_hydro_sinVol_long),
                                 k=1, s=0, ext=0)
plt.plot(xxx, 10 ** spline_hydro1(np.log10(xxx))
         * (xxx > np.min(Grand_hydro[:, 2]*1e3)),
         '--', color='limegreen', lw=2)

plt.axvline(np.min(Grand_dmo[:, 2])*1e3, alpha=0.8, color='k',
            label='Inner subhalo', linestyle='-', lw=1)
plt.axvline(np.min(Grand_hydro[:, 2])*1e3, alpha=0.8,
            color='limegreen', linestyle='-', lw=1)


plt.axvline(R_max * 1e3, alpha=0.7, linestyle='-', lw=2)  # , label='220 kpc')
plt.annotate(r'R$_\mathrm{vir}$', (160, 0.04), color='b', rotation=45,
             alpha=1)

plt.axvline(8.5, linestyle='-', alpha=1, color='Sandybrown', lw=2)
plt.annotate('Earth', (8.6, 0.1), color='Sandybrown', rotation=45)

plt.ylabel(r'n(r) = $\frac{N(r)}{N_{Tot}}$', size=26)
plt.xlabel('r [kpc]', size=26)


legend11 = plt.legend(loc=4, framealpha=1)

colors = ['k', 'limegreen']
legend33 = plt.legend([plt.Line2D([], [],
                                  linestyle='', marker='o',
                                  color=colors[i])
                       for i in range(2)],
                      ['DMO', 'Hydro'],
                      loc=1, title='Simulation', framealpha=1,
                      bbox_to_anchor=(0.99, 0.6)
                      )

ax1.add_artist(legend11)
ax1.add_artist(legend33)

# plt.legend(framealpha=1)

plt.xscale('log')
plt.yscale('log')

plt.savefig('outputs/srd_num.png', bbox_inches='tight')
plt.savefig('outputs/srd_num.pdf', bbox_inches='tight')

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

# ----------------------- FIG 3 -----------------------------------------
fig = plt.figure(3, figsize=(13, 10))
ax1 = plt.gca()

plt.plot(x_kpc, srd_dmo_sinVol, color='k', marker='o', linestyle='-',
          label='Data')
plt.plot(x_kpc, srd_hydro_sinVol, color='g', marker='o', linestyle='-')

xmin = 6
xmax = 11
linear_tend_dmo = np.polyfit(np.log10(x_kpc[xmin:xmax]),
                             np.log10(srd_dmo_sinVol[xmin:xmax]), 1)

xmin = 6
xmax = 11
linear_tend_hydro = np.polyfit(np.log10(x_kpc[xmin:xmax]),
                               np.log10(srd_hydro_sinVol[xmin:xmax]), 1)

# plt.plot(x_kpc, x_kpc ** linear_tend_dmo[0] * 10 ** linear_tend_dmo[1],
#          '--', color='grey', alpha=0.7, label='res_maximum N/Ntot')
# plt.plot(x_kpc, x_kpc ** linear_tend_hydro[0] * 10 ** linear_tend_hydro[1],
#          '--', color='limegreen', alpha=0.7)

vol = 4 / 3 * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
# plt.plot(x_kpc, (np.log10(x_kpc) * log_tend_dmo[0] + log_tend_dmo[1]) * vol,
#          '-', marker='+', ms=10,
#          color='grey', alpha=0.7, label='res_maximum density')
# plt.plot(x_kpc,
#          (np.log10(x_kpc) * log_tend_hydro[0] + log_tend_hydro[1]) * vol,
#          '-', marker='+', ms=10,
#          color='limegreen', alpha=0.7)


def func_radialexpcutoff(R, a, b):
    return b * np.exp(a / R)


xxx = np.linspace(3e-3, R200 * 1e3, num=200)
p0 = [-4, 1.4]
cts_dmo = opt.curve_fit(func_radialexpcutoff, x_kpc[:X_max],
                        (srd_dmo_sinVol[:X_max]),
                        p0=p0)
plt.plot(xxx, func_radialexpcutoff(xxx,
                                   cts_dmo[0][0], cts_dmo[0][1]),
         '--k')
print('Attempt at exp of dmo: ', cts_dmo[0])

p0 = [-4, 1.4]
cts_hydro = opt.curve_fit(func_radialexpcutoff, x_kpc[:X_max],
                          (srd_hydro_sinVol[:X_max]),
                          p0=p0)
plt.plot(xxx, func_radialexpcutoff(xxx,
                                   cts_hydro[0][0], cts_hydro[0][1]),
         '--g')
print('Attempt at exp of hydro: ', cts_hydro[0])
print(x_kpc[:X_max])
print(np.log10(srd_dmo_sinVol[:X_max]))

x_xpc_long = np.append(x_kpc[:X_max], 220.)
srd_dmo_sinVol_long = np.append(srd_dmo_sinVol[:X_max],
                                srd_dmo_sinVol[:X_max][-1])
srd_hydro_sinVol_long = np.append(srd_hydro_sinVol[:X_max],
                                srd_hydro_sinVol[:X_max][-1])
print(x_xpc_long)
print(srd_dmo_sinVol_long)
print(srd_hydro_sinVol_long)

spline_dmo1 = UnivariateSpline(np.log10(x_xpc_long),
                               np.log10(srd_dmo_sinVol_long),
                               k=1, s=0, ext=0)
plt.plot(xxx, 10 ** spline_dmo1(np.log10(xxx)),
         color='orange',
         linestyle='dotted')

spline_hydro1 = UnivariateSpline(np.log10(x_xpc_long),
                                 np.log10(srd_hydro_sinVol_long),
                                 k=1, s=0, ext=0)
plt.plot(xxx, 10 ** spline_hydro1(np.log10(xxx)),
         color='orange', linestyle='dotted')

spline_dmo1 = UnivariateSpline(np.log10(x_xpc_long),
                               np.log10(srd_dmo_sinVol_long),
                               k=3, s=0, ext=0)
plt.plot(xxx, 10 ** spline_dmo1(np.log10(xxx)),
         color='r', linestyle='dotted')

spline_hydro1 = UnivariateSpline(np.log10(x_xpc_long),
                                 np.log10(srd_hydro_sinVol_long),
                                 k=3, s=0, ext=0)
plt.plot(xxx, 10 ** spline_hydro1(np.log10(xxx)),
         color='r', linestyle='dotted')


def N_Dgc_Cosmic_slide(R, R0, aa, bb):
    # Number of subhalos (NOT DM DENSITY) at a distance to the GC R.
    return R0 + aa * R ** bb


p0 = [-0.5, 0.5, 0.055]
# plt.plot(x_kpc[:X_max], N_Dgc_Cosmic_slide(x_kpc,
#                                          p0[0], p0[1], p0[2])[:X_max])

cts_dmo = opt.curve_fit(N_Dgc_Cosmic_slide, x_kpc[:X_max],
                        (srd_dmo_sinVol[:X_max]),
                        p0=p0)
print('Attempt at sqrt of dmo: ', cts_dmo[0])
# plt.plot(xxx, N_Dgc_Cosmic_slide(xxx,
#                                    cts_dmo[0][0], cts_dmo[0][1],
#                                    cts_dmo[0][2]),
#          'r')


plt.axvline(R_max * 1e3, alpha=1, label='220 kpc')

plt.axvline(6.61, alpha=1, color='k', linestyle='dotted',
            label='Last subhalo', lw=2)
plt.axvline(13.6, alpha=1, color='limegreen', linestyle='dotted', lw=2)

plt.axvline(8.5, alpha=1, color='orange', lw=2, label='Earth')

plt.ylabel(r'n(r) = $\frac{N(r)}{N_{Tot}}$')
plt.xlabel('r [kpc]', size=24)

linestyles = ['dotted', 'dotted', '--']
colors = ['r', 'orange', 'k']
markers = [None, None, None]
legend22 = plt.legend([plt.Line2D([], [],
                                  linestyle=linestyles[i],
                                  color=colors[i],
                                  marker=markers[i], ms=14)
                       for i in range(3)],
                      ['Spline k=3', 'Spline k=1', 'Exponential function'],
                      loc=4, title='Functions', framealpha=1)

legend11 = plt.legend(loc=8, framealpha=1)

colors = ['k', 'g']
legend33 = plt.legend([plt.Line2D([], [],
                                  linestyle='-',
                                  color=colors[i])
                       for i in range(2)],
                      ['DMO', 'Hydro'],
                      loc=5, title='Colors', framealpha=1)


ax1.add_artist(legend22)
ax1.add_artist(legend11)
ax1.add_artist(legend33)

# plt.xscale('log')
plt.yscale('log')

plt.ylim(5e-5, 0.5)


# --------------------------- FIG 4 ------------------------------------
plt.figure(4, figsize=(10, 8))

plt.plot(x_kpc, np.log10(srd_dmo_sinVol), '-', color='k')
plt.plot(x_kpc, np.log10(srd_dmo_sinVol), color='k', marker='.', ms=10,
         label='DMO')

# plt.plot(x_kpc, np.log10(srd_hydro_sinVol), '-', color='g')
# plt.plot(x_kpc, np.log10(srd_hydro_sinVol), color='g', marker='.', ms=10,
#          label='Hydro')

xmin = 6
xmax = 11
linear_tend_dmo = np.polyfit(np.log10(x_kpc[xmin:xmax]),
                             np.log10(srd_dmo_sinVol[xmin:xmax]), 1)

xmin = 6
xmax = 11
linear_tend_hydro = np.polyfit(np.log10(x_kpc[xmin:xmax]),
                               np.log10(srd_hydro_sinVol[xmin:xmax]), 1)

plt.plot(x_kpc, np.log10(x_kpc ** linear_tend_dmo[0] * 10 **
                         linear_tend_dmo[1]),
         '--', color='grey', alpha=0.7, label='res_maximum N/Ntot')
# plt.plot(x_kpc, np.log10(x_kpc ** linear_tend_hydro[0] * 10 **
#                          linear_tend_hydro[1]),
#          '--', color='limegreen', alpha=0.7)

vol = 4 / 3 * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
plt.plot(x_kpc, np.log10((np.log10(x_kpc) * log_tend_dmo[0] + log_tend_dmo[
    1]) * vol),
         '-', marker='+', ms=10,
         color='grey', alpha=0.7, label='res_maximum density')
# plt.plot(x_kpc,
#          np.log10((np.log10(x_kpc) * log_tend_hydro[0] + log_tend_hydro[1])
#                   * vol),
#          '-', marker='+', ms=10,
#          color='limegreen', alpha=0.7)

plt.axvline(peak_dmo, alpha=0.5, color='k')


# plt.axvline(peak_hyd, alpha=0.5, color='limegreen')


def N_Dgc_Cosmic_slide(R, R0, aa, bb):
    # Number of subhalos (NOT DM DENSITY) at a distance to the GC R.
    return R0 + aa * R ** bb


xxx = np.linspace(3e-3, R200 * 1e3, num=200)

p0 = cts_dmo[0]
plt.plot(xxx, np.log10(N_Dgc_Cosmic_slide(xxx,
                                          p0[0], p0[1], p0[2])))

p0 = [-4, 1.4, 0.15]
plt.plot(xxx, (N_Dgc_Cosmic_slide(xxx,
                                  p0[0], p0[1], p0[2])), 'k')


def func_radialexpcutoff(R, a, b):
    return b * np.exp(a / R)


p0 = [-4, 1.4]
cts_dmo = opt.curve_fit(func_radialexpcutoff, x_kpc[:X_max],
                        (srd_dmo_sinVol[:X_max]),
                        p0=p0)
print('Attempt at sqrt of dmo: ', cts_dmo[0])

plt.plot(xxx, np.log10(func_radialexpcutoff(xxx,
                                   cts_dmo[0][0], cts_dmo[0][1])),
         'r')
#
# cts_dmo = opt.curve_fit(N_Dgc_Cosmic_slide, x_kpc[:X_max],
#                         np.log10(srd_dmo_sinVol[:X_max]),
#                         p0=p0)
# print('Attempt at sqrt of dmo: ', cts_dmo[0])


# plt.plot(xxx, N_Dgc_Cosmic_slide(xxx,
#                                    cts_dmo[0][0], cts_dmo[0][1],
#                                    cts_dmo[0][2]),
#          'r')

plt.axvline(R_max * 1e3, alpha=0.5, label='220 kpc')

plt.ylabel(r'log10(n(r) = $\frac{N(r)}{N_{Tot}}$)')
plt.xlabel('r [kpc]', size=24)

plt.legend(framealpha=1)

# plt.xscale('log')
# plt.yscale('log')


# NEW POSSIBILITY LALALALA

plt.figure(figsize=(12, 7))
plt.title('DMO')
num_bins = 15
print(bins)
print(x_med)

bins = np.linspace(0, R200, num=num_bins)
x_med = (bins[:-1] + bins[1:]) / 2.

v_cut = [2., 12.]
v_cut = np.linspace(2, 10, num=25)
cm_subsection = np.linspace(0, 1, 25)
from matplotlib import cm
colors = [ cm.jet(x) for x in cm_subsection ]

for ni, ii in enumerate(v_cut):
    print('v_cut: ', ii, ni)
    print((1))
    Grand_dmo_over = Grand_dmo[Grand_dmo[:, 1] > ii, :]
    Grand_dmo_below = Grand_dmo[Grand_dmo[:, 1] <= ii, :]
    Grand_hydro_over = Grand_hydro[Grand_hydro[:, 1] > ii, :]
    Grand_hydro_below = Grand_hydro[Grand_hydro[:, 1] <= ii, :]

    srd_dmo_over = (np.array(encontrar_SRD(Grand_dmo_over))
                    / len(Grand_dmo_over))
    srd_dmo_below = (np.array(encontrar_SRD(Grand_dmo_below))
                     / len(Grand_dmo_below))

    srd_hydro_over = (np.array(encontrar_SRD(Grand_hydro_over))
                      / len(Grand_hydro_over))
    srd_hydro_below = (np.array(encontrar_SRD(Grand_hydro_below))
                       / len(Grand_hydro_below))
    # (abs(ni * (0.1 - 1.) / (len(v_cut))))
    plt.plot(x_med * 1e3, srd_dmo_over, '-', color=colors[ni], ms=10,
             alpha=(1), label='%.1f' % ii)

    # plt.plot(x_med * 1e3, srd_hydro_over, '-', color=colors[ni], ms=10,
    #          alpha=(1), label='%.1f' % ii)

    plt.plot(x_med * 1e3, srd_dmo_below, '--', color=colors[ni], ms=10,
             alpha=(1))#, marker='x')

    # plt.plot(x_med * 1e3, srd_hydro_below, '--', color=colors[ni], ms=10,
    #          alpha=(1))#, marker='x')

srd_dmo = (np.array(encontrar_SRD(Grand_dmo))
                    / len(Grand_dmo))

srd_hydro = (np.array(encontrar_SRD(Grand_hydro))
                      / len(Grand_hydro))
plt.plot(x_med * 1e3, srd_dmo, '-', color='k', ms=10, linewidth=5,
         label='total')
# plt.plot(x_med * 1e3, srd_hydro, '--', color='k', ms=10)

plt.axvline(R_max * 1e3, alpha=0.7, linestyle='--')  # , label='220 kpc')
plt.annotate(r'R$_\mathrm{vir}$', (170, 1), color='b', rotation=45, alpha=0.7)

plt.axvline(8.5, linestyle='--', alpha=1, color='Sandybrown')
plt.annotate('Earth', (8.6, 1), color='Sandybrown', rotation=45)

plt.ylabel(r'n(r) = $\frac{N(r)}{N_{Tot}\,Volumen}$ [Mpc$^{-3}$]', size=24)
plt.xlabel('r [kpc]', size=26)

legend_elements = [Line2D([0], [0], color='k',
                           markersize=8),
                   Line2D([0], [0], color='k', ls='--',
                          markersize=8)]
legend1 = plt.legend(legend_elements, ['Over', 'Below'], loc=8,
                     handletextpad=0.2)  # ,handlelength=1)
leg = plt.legend(framealpha=1, loc=1, fontsize=10)
plt.gca().add_artist(legend1)

plt.xscale('log')
plt.yscale('log')
plt.xlim(6, 300)

# plt.ylim(0, 60)
'''
# ------------------------ N(r)/Ntot figure ------------------------------
print()
print('N/Ntot figures')

plt.figure(figsize=(8, 8))
ax1 = plt.gca()
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
# plt.subplot(121)

v_cut = [8.0]  # np.linspace(6, 10, num=10)
cm_subsection = np.linspace(0, 1, 25)
from matplotlib import cm

colors = [cm.jet(x) for x in cm_subsection]

num_bins = 15
bins = np.linspace(0, R200, num=num_bins)
bins_mean = (bins[:-1] + bins[1:]) / 2. * 1e3

xxx = np.linspace(3e-3, R_max+0.02, num=200) * 1e3

for ni, ii in enumerate(v_cut):
    print('v_cut: ', ii, ni)
    Grand_dmo_over = Grand_dmo[Grand_dmo[:, 1] >= ii, :]
    Grand_hydro_over = Grand_hydro[Grand_hydro[:, 1] >= ii, :]
    release_dmo_over = data_release_dmo[data_release_dmo[:, 1] >= ii, :]
    release_hydro_over = data_release_hydro[data_release_hydro[:, 1] >= ii, :]

    srd_dmo_over = (np.array(encontrar_SRD_sinVol(Grand_dmo_over))
                    / len(Grand_dmo_over))
    srd_hydro_over = (np.array(encontrar_SRD_sinVol(Grand_hydro_over))
                      / len(Grand_hydro_over))

    srd_dmo_over_release = (np.array(encontrar_SRD_sinVol(release_dmo_over))
                    / len(release_dmo_over))
    srd_hydro_over_release = (np.array(encontrar_SRD_sinVol(release_hydro_over))
                      / len(release_hydro_over))
    print(release_dmo_over)
    print(srd_dmo_over_release)

    plt.plot(bins_mean, srd_dmo_over, ls='', color='k',
             ms=15, marker='.',
             alpha=1, zorder=15, label='2 years old data')

    plt.plot(bins_mean, srd_hydro_over, ls='', color='forestgreen',
             ms=15, marker='.',
             alpha=1, zorder=15)


    plt.plot(bins_mean, srd_dmo_over_release, ls='', color='k',
             ms=15, marker='+', markeredgewidth=3,
             alpha=1, zorder=15, label='Release data')

    plt.plot(bins_mean, srd_hydro_over_release, ls='', color='forestgreen',
             ms=15, marker='+', markeredgewidth=3,
             alpha=1, zorder=15)


def funct_ale(Dgc, a, b):
    return b * np.exp(a / Dgc)


cts_dmo = opt.curve_fit(funct_ale, xdata=bins_mean,
                        ydata=srd_dmo_over,
                        p0=[-20, 0.1])
print('Funct Ale: ', cts_dmo[0])

cts_hydro = opt.curve_fit(funct_ale, xdata=bins_mean,
                        ydata=srd_hydro_over,
                        p0=[-20, 0.1])
print('Funct Ale: ', cts_hydro[0])


plt.plot(xxx, funct_ale(xxx, cts_dmo[0][0], cts_dmo[0][1]),
         'k', linestyle='--', lw=2, alpha=0.7,
         label='Fragile fit', zorder=5)
plt.plot(xxx, funct_ale(xxx, cts_hydro[0][0], cts_hydro[0][1]),
         'limegreen', linestyle='--', lw=2, zorder=5)

plt.plot(xxx, np.ones(len(xxx)) * cts_dmo[0][1],
         marker=None, linestyle='dotted', color='k', lw=4, alpha=0.7,
         label='Resilient fit')
plt.plot(xxx, np.ones(len(xxx)) * cts_hydro[0][1],
         marker=None, linestyle='dotted', color='limegreen', lw=4)



plt.axvline(R_max * 1e3, alpha=0.7, linestyle='-.')
plt.annotate(r'R$_\mathrm{vir}$', (190, 2e-2), color='b', rotation=45,
             alpha=0.7, zorder=0)

plt.axvline(8.5, linestyle='-.', alpha=1, color='Sandybrown')
plt.annotate('Earth', (10, 0.15), color='Saddlebrown', rotation=0.,
             fontsize=18, zorder=10)

plt.axvline(6.61, alpha=0.5, color='k', linestyle='-',
            label='Last subhalo', lw=2)
plt.axvline(13.6, alpha=0.5, color='limegreen', linestyle='-', lw=2)

plt.ylabel(r'$N(D_\mathrm{GC}) \, / \, N_{Total}$')
plt.xlabel(r'D$_\mathrm{GC}$ [kpc]', size=24)

plt.xscale('linear')
plt.yscale('log')

plt.ylim(1e-3, 2e-1)

# plt.legend(framealpha=1, fontsize=10, loc=4)
handles = (mpatches.Patch(color='k', label='DMO', alpha=0.8),
           mpatches.Patch(color='limegreen', label='Hydro', alpha=0.8)
           )

legend_colors = plt.legend(handles=handles, bbox_to_anchor=(0.13, 0.2),
                           fontsize=20)


legend11 = plt.legend(loc=4, framealpha=1)

# colors = ['k', 'limegreen']
# legend33 = plt.legend([plt.Line2D([], [],
#                                   linestyle='', marker='o',
#                                   color=colors[i])
#                        for i in range(2)],
#                       ['DMO', 'Hydro'],
#                       loc=8, title='Simulation', framealpha=1,
#                       # bbox_to_anchor=(0.99, 0.6)
#                       )

ax1.add_artist(legend11)
ax1.add_artist(legend_colors)
plt.savefig('outputs/srd_compar_lin.png', bbox_inches='tight')
plt.savefig('outputs/srd_compar_lin.pdf', bbox_inches='tight')
plt.show()
# -------------------------------------------------------------------------
print('Density figure')

# plt.figure(figsize=(10, 8))
plt.subplot(122)
srd_dmo = np.array(encontrar_SRD(Grand_dmo_over)) / len(Grand_dmo_over)
srd_hydro = np.array(encontrar_SRD(Grand_hydro_over)) / len(Grand_hydro_over)

plt.plot(bins_mean, srd_dmo, '.-', color='k', ms=10, label='Data')
plt.plot(bins_mean, srd_hydro, '.-', color='limegreen', ms=10)

xxx = np.linspace(3e-3, R200 * 1e3, num=100)

vol = 4 / 3 * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)

print((bins[:-1] - bins[1:]))
print(cts_hydro[0][1] / vol/(cts_dmo[0][1] / vol))
plt.plot(bins_mean, cts_dmo[0][1] / vol,# * (bins[1:] - bins[:-1])*1e3,
         '--', marker='*', ms=10,
         color='grey', alpha=1, linestyle='dotted', label='resilient possible')
plt.plot(bins_mean, cts_hydro[0][1] / vol,
         '--', marker='*', linestyle='dotted', ms=10, color='limegreen', alpha=1)

plt.plot(bins_mean, funct_ale(bins_mean, cts_dmo[0][0], cts_dmo[0][1]) / vol,
         '--', marker='+', ms=10,
         color='grey', alpha=1, label='exp cutoff')
plt.plot(bins_mean, funct_ale(bins_mean, cts_hydro[0][0], cts_hydro[0][1]) / vol,
         '--', marker='+', ms=10, color='limegreen', alpha=1)

print(cts_dmo, cts_hydro)


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
# plt.xlim(6, 300)

# plt.ylim(0, 60)

plt.savefig('outputs/srd_compar_log.png', bbox_inches='tight')
plt.savefig('outputs/srd_compar_log.pdf', bbox_inches='tight')
plt.show()
