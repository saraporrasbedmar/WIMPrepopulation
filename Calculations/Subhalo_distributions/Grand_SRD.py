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
    std_fin = []

    for delta in range(len(bins) - 1):
        aaa = []
        for halo in np.unique(data[:, 6]):
            data_ind = data[data[:, 6] == halo, :]
            interval = ((data_ind[:, 2] / data_ind[:, 5] >= bins[delta])
                        * (data_ind[:, 2] / data_ind[:, 5] <= bins[delta + 1]))
            aaa.append(sum(interval))
        if delta == 0:
            print(aaa, np.nanmean(aaa), np.std(aaa))
        n_final.append(np.nanmean(aaa))
        std_fin.append(np.std(aaa))
    return np.array(n_final), np.array(std_fin)


def encontrar_SRD(data):
    n_final = []
    std_fin = []

    for delta in range(len(bins) - 1):

        aaa = []
        vol = []
        for halo in np.unique(data[:, 6]):
            data_ind = data[data[:, 6] == halo, :]
            interval = ((data_ind[:, 2] / data_ind[:, 5] >= bins[delta])
                        * (data_ind[:, 2] / data_ind[:, 5] <= bins[delta + 1]))
            aaa.append(sum(interval))
            vol.append(4 / 3 * np.pi * (
                    bins[delta + 1] ** 3 - bins[delta] ** 3)
                       #* data_ind[0, 5]**3.
                       # * 1e-9
                       )  # * 1e9
            if delta == 0:
                print(int(halo), np.min(data_ind[:, 2]))

        aaa = np.array(aaa)
        vol = np.array(vol)
        y = np.nanmean(aaa / vol)  # / len(data_ind))
        std = np.std(aaa / vol)
        print(delta, y, std, aaa/vol, vol)

        n_final.append(y)
        std_fin.append(std)
    print(n_final)
    print(std_fin)
    return np.array(n_final), np.array(std_fin)


# ------------------------ N(r)/Ntot figure ------------------------------
print()
print('N/Ntot figures')

# plt.figure(figsize=(10, 8))
# ax1 = plt.gca()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
plt.subplots_adjust(wspace=0.27)
plt.subplot(121)

v_cut = [8.]  #
v_cut = np.linspace(1., 8., num=10)
cm_subsection = np.linspace(0, 1, 10)
from matplotlib import cm

colors = [cm.jet(x) for x in cm_subsection]

num_bins = 15
bins = np.linspace(0, 1., num=num_bins)
bins_mean = (bins[:-1] + bins[1:]) / 2.
volume = 4 / 3 * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
print('bins: ', bins*220)

xxx = np.linspace(0., 1., num=200)

for ni, ii in enumerate(v_cut):
    print('v_cut: ', ii, ni)
    release_dmo_over = data_release_dmo[data_release_dmo[:, 1] >= ii, :]
    release_hydro_over = data_release_hydro[data_release_hydro[:, 1] >= ii, :]

    srd_dmo_over_release, std_dmo_num = (np.array(encontrar_SRD_sinVol(
        release_dmo_over))
                                         / len(release_dmo_over)
                                         )
    srd_hydro_over_release, std_hydro_num = (np.array(encontrar_SRD_sinVol(
        release_hydro_over))
        / len(release_hydro_over)
    )
    print('dmo')
    srddensity_dmo_over_release, std_dmo_den = (encontrar_SRD(
        release_dmo_over))
    print('\nhydro')
    srddensity_hydro_over_release, std_hydro_den = (encontrar_SRD(
        release_hydro_over))
    print(release_dmo_over)
    print(srd_dmo_over_release)
    print('errorbars density')
    print(std_dmo_den)
    print(std_hydro_den)

    ax1.errorbar(bins_mean, srd_dmo_over_release,
                 yerr=std_dmo_num,
                 ls='',
                 # color='k',
                 c=colors[ni],
                 ms=15, marker='.', markeredgewidth=2,
                 alpha=1, zorder=15,
                 label='Data',
                 capsize=5
                 # label=ii
                 )

    ax1.errorbar(bins_mean, srd_hydro_over_release,
                 yerr=std_hydro_num,
                 ls='',
                 # color='#00CC00',
                 c=colors[ni],
                 ms=15, marker='.', markeredgewidth=2,
                 alpha=1, zorder=15,
                 capsize=5)

    ax2.errorbar(bins_mean, srddensity_dmo_over_release,
                 yerr=std_dmo_den,
                 ls='',
                 color='k',
                 ms=15, marker='.', markeredgewidth=2,
                 alpha=1, zorder=15,
                 label='Data',
                 capsize=5
                 # label=ii
                 )

    ax2.errorbar(bins_mean, srddensity_hydro_over_release,
                 yerr=std_hydro_den,
                 ls='',
                 color='#00CC00',
                 ms=15, marker='.', #markeredgewidth=3,
                 alpha=1, zorder=15,
                 capthick=2.,
                 capsize=5)


def funct_ale(Dgc, a, b):
    return b * np.exp(a / Dgc)


# --------------
print('Release')
cts_dmo = opt.curve_fit(funct_ale, xdata=bins_mean,
                        ydata=srd_dmo_over_release,
                        sigma=std_dmo_num,
                        p0=[-0.5, 200])
print('Funct Ale: ', cts_dmo[0])
print(np.diag(cts_dmo[1]) ** 0.5)

cts_hydro = opt.curve_fit(funct_ale, xdata=bins_mean,
                          ydata=srd_hydro_over_release,
                          sigma=std_hydro_num,
                          p0=[0, 200])
print('Funct Ale: ', cts_hydro[0])
print(np.diag(cts_hydro[1]) ** 0.5)

plt.plot(xxx, funct_ale(xxx, cts_dmo[0][0], cts_dmo[0][1]),
         'dimgray', linestyle='--', lw=3, alpha=0.7,
         label='Fragile fit', zorder=5)
plt.plot(xxx, funct_ale(xxx, cts_hydro[0][0], cts_hydro[0][1]),
         '#00FF00', linestyle='--', lw=3, zorder=5)

plt.plot(xxx, np.ones(len(xxx))
         * funct_ale(1., cts_dmo[0][0], cts_dmo[0][1]),
         'dimgray', linestyle='dotted', lw=4, alpha=0.7,
         label='Resilient fit', zorder=5)
plt.plot(xxx, np.ones(len(xxx))
         * funct_ale(1., cts_hydro[0][0], cts_hydro[0][1]),
         '#00FF00', linestyle='dotted', lw=4, zorder=5)
print('resilient values: ',
      np.max(srd_dmo_over_release), np.max(srd_hydro_over_release))
# ----------------

# plt.axvline(1., alpha=0.7, linestyle='-.',
#             lw=3, color='royalblue')
# plt.annotate(r'R$_\mathrm{vir}$', (0.9, 2e-2), color='b', rotation=45,
#              alpha=0.7, zorder=0)
# plt.annotate(r'R$_\mathrm{vir}$', (0.9, 25), color='b', rotation=45,
#              alpha=0.7, zorder=0)

plt.axvline(8.5 / 220., linestyle='-.', alpha=1, color='Sandybrown', lw=3)
# plt.annotate('Earth', (0.05, 0.130), color='Saddlebrown', rotation=0.,
#              fontsize=20, zorder=10)
plt.annotate('Earth', (0.05, 35), color='Saddlebrown', rotation=0.,
             fontsize=20, zorder=10)

plt.axvline(data_release_dmo[0, 2] / data_release_dmo[0, 5],
            alpha=0.5, color='k', linestyle='-',
            lw=3, label='Last subhalo')
plt.axvline(data_release_hydro[0, 2] / data_release_hydro[0, 5],
            alpha=0.5, color='limegreen',
            linestyle='-', lw=3)

# plt.ylabel(r'$N(D_\mathrm{GC}) \, / \, N_\mathrm{Total}$')
plt.ylabel(r'$N(D_\mathrm{GC})$')
plt.xlabel(r'D$_\mathrm{GC} \, / \, R_\mathrm{vir}$ ', size=26)
# plt.xlabel(r'D$_\mathrm{GC}$ [kpc]', size=24)

plt.xscale('linear')
plt.yscale('log')

plt.xlim(0, 1.)
# plt.ylim(1e-3, 2e-1)
plt.ylim(0.1, 50)

# plt.legend(framealpha=1, fontsize=10, loc=4)
handles = (mpatches.Patch(color='k', label='DMO', alpha=0.8),
           mpatches.Patch(color='limegreen', label='Hydro', alpha=0.8)
           )

legend_colors = plt.legend(handles=handles, loc=5,
                           # bbox_to_anchor=(0.13, 0.2),
                           fontsize=20)

legend11 = plt.legend(loc=4, framealpha=1)

ax1.add_artist(legend11)
ax1.add_artist(legend_colors)
plt.savefig('outputs/srd_compar_lin_after.png', bbox_inches='tight')
plt.savefig('outputs/srd_compar_lin_after.pdf', bbox_inches='tight')
# plt.show()
# -------------------------------------------------------------------------
print('Density figure')

# plt.figure(figsize=(10, 8))
plt.subplot(122)

xxx = np.linspace(3e-3, R_vir * 1e3, num=100)
volume_220 = volume #* 0.220**3.

print((bins[:-1] - bins[1:]))
# print(cts_hydro[0][1] / volume / (cts_dmo[0][1] / volume))

plt.plot(bins_mean, funct_ale(bins_mean, cts_dmo[0][0], cts_dmo[0][1])
         / volume_220
         # / len(release_dmo_over)
         ,
         '--', marker='', ms=20, lw=3,
         color='grey', alpha=1, label='Fragile fit')
plt.plot(bins_mean,
         funct_ale(bins_mean, cts_hydro[0][0], cts_hydro[0][1])
         / volume_220
         # / len(release_hydro_over)
         ,
         '--', marker='', ms=20, color='limegreen', alpha=1, lw=3)

plt.plot(bins_mean, funct_ale(1., cts_dmo[0][0], cts_dmo[0][1]) / volume_220
         # / len(release_dmo_over)
         ,  # * (bins[1:] - bins[
         # :-1])*1e3,
         marker='', ms=10, lw=3,
         color='grey', alpha=1, linestyle='dotted', label='Resilient fit')
plt.plot(bins_mean, funct_ale(1., cts_hydro[0][0], cts_hydro[0][1]) / volume_220
         # / len(release_hydro_over)
         ,
         marker='', linestyle='dotted', ms=10, color='limegreen',
         alpha=1, lw=3)
# print(cts_dmo, cts_hydro)


plt.axvline(8.5 / 220., linestyle='-.', alpha=1, color='Sandybrown', lw=3)
# plt.annotate('Earth', (0.05, 0.130), color='Saddlebrown', rotation=0.,
#              fontsize=20, zorder=10)
plt.annotate('Earth', (0.05, 1.7e4), color='Saddlebrown', rotation=0.,
             fontsize=20, zorder=10)

plt.axvline(data_release_dmo[0, 2] / data_release_dmo[0, 5],
            alpha=0.5, color='k', linestyle='-',
            lw=3, label='Last subhalo')
plt.axvline(data_release_hydro[0, 2] / data_release_hydro[0, 5],
            alpha=0.5, color='limegreen',
            linestyle='-', lw=3)

# plt.ylabel(r'$\frac{N(D_\mathrm{GC})}{N_{Tot}\,Volume}$ [Mpc$^{-3}$]',
#            size=24)
plt.ylabel(r'$\frac{N(D_\mathrm{GC})}{Volume}$ [Mpc$^{-3}$]',
           size=24)
plt.xlabel(r'D$_\mathrm{GC} \, / \, R_\mathrm{vir}$ ', size=26)

legend_elements = [Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='k', markersize=8),
                   Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='limegreen', markersize=8)]
legend1 = plt.legend(handles, ['DMO', 'Hydro'], loc=1,
                           bbox_to_anchor=(0.995, 0.67),
                           fontsize=20)
leg = plt.legend(framealpha=1, loc=1)
plt.gca().add_artist(legend1)

# plt.xscale('log')
plt.yscale('log')
plt.xlim(0., 1.)

# plt.ylim(0, 60)

plt.savefig('outputs/srd_compar_den.png', bbox_inches='tight')
plt.savefig('outputs/srd_compar_den.pdf', bbox_inches='tight')
plt.show()
