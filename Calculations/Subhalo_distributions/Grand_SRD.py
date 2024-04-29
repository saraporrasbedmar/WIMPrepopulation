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

data_release_dmo = np.loadtxt(
    '../Data_subhalo_simulations/dmo_table.txt', skiprows=3)
data_release_hydro = np.loadtxt(
    '../Data_subhalo_simulations/hydro_table.txt', skiprows=3)

data_release_dmo = data_release_dmo[
                   data_release_dmo[:, 0] > 0.184, :]
data_release_hydro = data_release_hydro[
                     data_release_hydro[:, 0] > 0.184, :]

data_release_dmo = data_release_dmo[np.argsort(data_release_dmo[:, 2])]
data_release_hydro = data_release_hydro[np.argsort(data_release_hydro[:, 2])]
print(data_release_dmo[0, 2], data_release_hydro[0, 2])
print(data_release_dmo[-1, 2], data_release_hydro[-1, 2])

R_vir = 211.834
R_last = 251.400


# %% SRD

def encontrar_SRD_sinVol(data):
    n_final = []

    for delta in range(len(bins) - 1):
        interval = ((data[:, 2]/data[:, 5] >= bins[delta])
                    * (data[:, 2]/data[:, 5] <= bins[delta + 1]))
        n_final.append(sum(interval))
    print(sum(n_final))
    return np.array(n_final)


def encontrar_SRD(data):
    n_final = []
    a = 0
    for delta in range(len(bins) - 1):
        interval = ((data[:, 2] >= bins[delta])
                    * (data[:, 2] <= bins[delta + 1]))

        y = sum(interval)

        vol = 4 / 3 * np.pi * (
                bins[delta + 1] ** 3 - bins[delta] ** 3) * 1e9

        n_final.append(y / vol)
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


# ------------------------ N(r)/Ntot figure ------------------------------
print()
print('N/Ntot figures')

plt.figure(figsize=(10, 8))
ax1 = plt.gca()

v_cut = [8.]  # np.linspace(1., 8., num=10)
cm_subsection = np.linspace(0, 1, 25)
from matplotlib import cm

colors = [cm.jet(x) for x in cm_subsection]

num_bins = 15
bins = np.linspace(0, 1., num=num_bins)
bins_mean = (bins[:-1] + bins[1:]) / 2.

xxx = np.linspace(0., 1.2, num=200)

for ni, ii in enumerate(v_cut):
    print('v_cut: ', ii, ni)
    release_dmo_over = data_release_dmo[data_release_dmo[:, 1] >= ii, :]
    release_hydro_over = data_release_hydro[data_release_hydro[:, 1] >= ii, :]

    srd_dmo_over_release = (encontrar_SRD_sinVol(release_dmo_over)
                            / len(release_dmo_over))
    srd_hydro_over_release = (encontrar_SRD_sinVol(release_hydro_over)
                              / len(release_hydro_over))
    print(release_dmo_over)
    print(srd_dmo_over_release)

    plt.plot(bins_mean, srd_dmo_over_release, ls='',
             color='k',
             ms=15, marker='.', markeredgewidth=3,
             alpha=1, zorder=15,
             label='Data'
             # label=ii
             )

    plt.plot(bins_mean, srd_hydro_over_release, ls='',
             color='#00CC00',
             ms=15, marker='.', markeredgewidth=3,
             alpha=1, zorder=15)


def funct_ale(Dgc, a, b):
    return b * np.exp(a / Dgc)


# --------------
print('Release')
cts_dmo = opt.curve_fit(funct_ale, xdata=bins_mean,
                        ydata=srd_dmo_over_release,
                        p0=[-20, 0.1])
print('Funct Ale: ', cts_dmo[0])

cts_hydro = opt.curve_fit(funct_ale, xdata=bins_mean,
                          ydata=srd_hydro_over_release,
                          p0=[0, 0.1])
print('Funct Ale: ', cts_hydro[0])

plt.plot(xxx, funct_ale(xxx, cts_dmo[0][0], cts_dmo[0][1]),
         'dimgray', linestyle='--', lw=3, alpha=0.7,
         label='Fragile fit', zorder=5)
plt.plot(xxx, funct_ale(xxx, cts_hydro[0][0], cts_hydro[0][1]),
         '#00FF00', linestyle='--', lw=3, zorder=5)

plt.plot(xxx, np.ones(len(xxx)) * cts_dmo[0][1],
         'dimgray', linestyle='dotted', lw=4, alpha=0.7,
         label='Resilient fit', zorder=5)
plt.plot(xxx, np.ones(len(xxx)) * cts_hydro[0][1],
         '#00FF00', linestyle='dotted', lw=4, zorder=5)
# ----------------

plt.axvline(1., alpha=0.7, linestyle='-.',
            lw=3, color='royalblue')
plt.annotate(r'R$_\mathrm{vir}$', (1, 2e-2), color='b', rotation=45,
             alpha=0.7, zorder=0)

plt.axvline(8.5/220., linestyle='-.', alpha=1, color='Sandybrown', lw=3)
plt.annotate('Earth', (0.086, 0.117), color='Saddlebrown', rotation=0.,
             fontsize=20, zorder=10)

plt.axvline(data_release_dmo[0, 2]/data_release_dmo[0, 5],
            alpha=0.5, color='k', linestyle='-',
            lw=3, label='Last subhalo')
plt.axvline(data_release_hydro[0, 2]/data_release_hydro[0, 5],
            alpha=0.5, color='limegreen',
            linestyle='-', lw=3)

plt.ylabel(r'$N(D_\mathrm{GC}) \, / \, N_{Total}$')
plt.xlabel(r'D$_\mathrm{GC} \, / \, R_{vir}$ ', size=24)
# plt.xlabel(r'D$_\mathrm{GC}$ [kpc]', size=24)

plt.xscale('linear')
plt.yscale('log')

plt.xlim(0, 1.15)
plt.ylim(1e-3, 2e-1)

# plt.legend(framealpha=1, fontsize=10, loc=4)
handles = (mpatches.Patch(color='k', label='DMO', alpha=0.8),
           mpatches.Patch(color='limegreen', label='Hydro', alpha=0.8)
           )

legend_colors = plt.legend(handles=handles, bbox_to_anchor=(0.13, 0.2),
                           fontsize=20)

legend11 = plt.legend(loc=4, framealpha=1)

ax1.add_artist(legend11)
ax1.add_artist(legend_colors)
plt.savefig('outputs/srd_compar_lin.png', bbox_inches='tight')
plt.savefig('outputs/srd_compar_lin.pdf', bbox_inches='tight')

# -------------------------------------------------------------------------
print('Density figure')

plt.figure(figsize=(10, 8))
# plt.subplot(122)

xxx = np.linspace(3e-3, R_vir * 1e3, num=100)

vol = 4 / 3 * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)

print((bins[:-1] - bins[1:]))
print(cts_hydro[0][1] / vol / (cts_dmo[0][1] / vol))
plt.plot(bins_mean, cts_dmo[0][1] / vol,  # * (bins[1:] - bins[:-1])*1e3,
         '--', marker='*', ms=10,
         color='grey', alpha=1, linestyle='dotted', label='resilient possible')
plt.plot(bins_mean, cts_hydro[0][1] / vol,
         '--', marker='*', linestyle='dotted', ms=10, color='limegreen',
         alpha=1)

plt.plot(bins_mean, funct_ale(bins_mean, cts_dmo[0][0], cts_dmo[0][1]) / vol,
         '--', marker='+', ms=10,
         color='grey', alpha=1, label='exp cutoff')
plt.plot(bins_mean,
         funct_ale(bins_mean, cts_hydro[0][0], cts_hydro[0][1]) / vol,
         '--', marker='+', ms=10, color='limegreen', alpha=1)

print(cts_dmo, cts_hydro)

plt.axvline(R_vir * 1e3, alpha=0.7, linestyle='--')  # , label='220 kpc')
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

plt.savefig('outputs/srd_compar_den.png', bbox_inches='tight')
plt.savefig('outputs/srd_compar_den.pdf', bbox_inches='tight')
plt.show()
