#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 20:34:39 2022

@author: saraporras
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colorbar as colorbarr
import matplotlib.patches as mpatches

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

data_release_dmo = np.loadtxt(
    '../Data_subhalo_simulations/dmo_table.txt', skiprows=3)
data_release_hydro = np.loadtxt(
    '../Data_subhalo_simulations/hydro_table.txt', skiprows=3)


data_release_dmo = data_release_dmo[
                   data_release_dmo[:, 0] > 1e-4, :]

data_release_dmo = data_release_dmo[
                   data_release_dmo[:, 0] > 0.184, :]
data_release_hydro = data_release_hydro[
                     data_release_hydro[:, 0] > 0.184, :]

data_release_dmo = data_release_dmo[np.argsort(data_release_dmo[:, 1])]
data_release_hydro = data_release_hydro[np.argsort(data_release_hydro[:, 1])]

x_cumul = np.geomspace(1., 120., num=25)


def calcular_dNdV(Vmax):
    Vmax_cumul = np.zeros(len(x_cumul) - 1)
    num_cumul = np.zeros(len(x_cumul) - 1)

    for radius in range(len(Vmax_cumul)):
        aa = Vmax >= x_cumul[radius]
        bb = Vmax < x_cumul[radius + 1]

        Vmax_cumul[radius] = sum(aa * bb) / (
                x_cumul[radius + 1] - x_cumul[radius])
        num_cumul[radius] = sum(aa * bb)

    return Vmax_cumul, num_cumul


def calcular_VdNdV(Vmax):
    Vmax_cumul = np.zeros(len(x_cumul) - 1)
    num_cumul = np.zeros(len(x_cumul) - 1)

    for radius in range(len(Vmax_cumul)):
        aa = Vmax >= x_cumul[radius]
        bb = Vmax < x_cumul[radius + 1]

        Vmax_cumul[radius] = (sum(aa * bb) / (
                x_cumul[radius + 1] - x_cumul[radius])
                              * (x_cumul[radius] + x_cumul[radius + 1]) / 2.)
        num_cumul[radius] = sum(aa * bb)

    return Vmax_cumul, num_cumul


Vmax_cumul_dmo_release, num_dmo = calcular_dNdV(data_release_dmo[:, 1])
Vmax_cumul_hydro_release, num_hydro = calcular_dNdV(data_release_hydro[:, 1])


x_cumul = (x_cumul[:-1] + x_cumul[1:]) / 2.


def find_PowerLaw(xx, yy, lim_inf, lim_sup):
    X1limit = np.where(xx >= lim_inf)[0][0]
    try:
        X2limit = np.where(xx >= lim_sup)[0][0]
    except:
        X2limit = len(xx)
    xx_copy = np.log10(xx[X1limit:X2limit])
    yy_copy = np.log10(yy[X1limit:X2limit])

    xx_copy = xx_copy[np.isfinite(yy_copy)]
    yy_copy = yy_copy[np.isfinite(yy_copy)]

    fits, cov_matrix = np.polyfit(xx_copy, yy_copy, 1, cov=True, full=False)
    perr = np.sqrt(np.diag(cov_matrix))
    # print(perr)
    #
    # print('%r: %.2f pm %.2f ---- %.2f pm %.2f'
    #       % (label, fits[0],
    #          cov_matrix[0, 0] ** 0.5, fits[1],
    #          cov_matrix[1, 1] ** 0.5))

    return fits[0], fits[1], perr[0], perr[1]


fig, ax = plt.subplots(figsize=(9, 8))
print(num_dmo)
print(num_hydro)
limit_infG = 8
limit_supG = 120
fitsM_DMO_release, fitsB_DMO_release, _, _ = find_PowerLaw(
    x_cumul[num_dmo > 10],
    Vmax_cumul_dmo_release[num_dmo > 10] / 6.,
    limit_infG, limit_supG)

xxx = np.logspace(np.log10(2), np.log10(92), 100)
fitsM_Hydro_release, fitsB_Hydro_release, _, _ = find_PowerLaw(
    x_cumul[num_hydro > 10],
    Vmax_cumul_hydro_release[num_hydro > 10] / 6.,
    limit_infG, limit_supG)

plt.axvline(x_cumul[np.where(num_dmo < 10)[0][1]] * 0.9,
            color='k',
            alpha=1,
            linewidth=2, ls='-.',
            zorder=0, label='Fit limits')

plt.axvline(x_cumul[np.where(num_hydro < 10)[0][1]] * 0.9,
            color='green',
            alpha=1,
            linewidth=2, ls='-.',
            zorder=0)

plt.plot(x_cumul, Vmax_cumul_dmo_release / 6.,
         linestyle='', ms=10, marker='.', markeredgewidth=2,
         color='k', zorder=10)
plt.plot(xxx, 10 ** fitsB_DMO_release * xxx ** fitsM_DMO_release,
                 color='dimgray', alpha=1,
                 linestyle='-', lw=2.5, label='Power-law fit')

plt.plot(x_cumul, Vmax_cumul_hydro_release / 6.,
         linestyle='',
         ms=10, marker='.', markeredgewidth=2,
         color='green', zorder=10)
plt.plot(xxx, 10 ** fitsB_Hydro_release * xxx ** fitsM_Hydro_release,
                 color='limegreen', alpha=1,
                 linestyle='-', lw=2.5)

print('Release')
print(fitsM_DMO_release, fitsB_DMO_release)
print(fitsM_Hydro_release, fitsB_Hydro_release)

plt.xscale('log')
plt.yscale('log')

plt.xlabel(r'$V_{\mathrm{max}}$ [km s$^{-1}$]', size=24)
plt.ylabel(r'$\frac{dN(V_{\mathrm{max}})}{dV_{\mathrm{max}}}$', size=27)


plt.axvline(limit_infG, linestyle='-.', color='k', alpha=0.3,
            linewidth=2)

handles = (mpatches.Patch(color='k', label='DMO', alpha=0.8),
           mpatches.Patch(color='limegreen', label='Hydro', alpha=0.8)
           )

legend11 = plt.legend(handles=handles,
                      loc=1, framealpha=1)

legend22 = plt.legend(loc=3, framealpha=1)

ax.add_artist(legend11)
ax.add_artist(legend22)

plt.savefig('outputs/shvf.pdf', bbox_inches='tight')
plt.savefig('outputs/shvf.png', bbox_inches='tight')
plt.show()
# %% STUDY THE LIMITS OF THE VMAX IN THE FIT

print('Study')

limit_infG = np.linspace(6, 10, num=20)
limit_supG = np.linspace(20, 70, num=20)

fitsM_DMO = np.zeros((len(limit_infG), len(limit_supG)))
fitsB_DMO = np.zeros((len(limit_infG), len(limit_supG)))

fitsM_Hydro = np.zeros((len(limit_infG), len(limit_supG)))
fitsB_Hydro = np.zeros((len(limit_infG), len(limit_supG)))

fitsM_DMO_std = np.zeros((len(limit_infG), len(limit_supG)))
fitsB_DMO_std = np.zeros((len(limit_infG), len(limit_supG)))

fitsM_Hydro_std = np.zeros((len(limit_infG), len(limit_supG)))
fitsB_Hydro_std = np.zeros((len(limit_infG), len(limit_supG)))

for ninf, inferior in enumerate(limit_infG):
    for nsup, superior in enumerate(limit_supG):
        fitsM_DMO[ninf, nsup], fitsB_DMO[ninf, nsup], fitsM_DMO_std[
            ninf, nsup], fitsB_DMO_std[ninf, nsup] = \
            find_PowerLaw(x_cumul, Vmax_cumul_dmo / 6., inferior, superior,
                          plot=False)
        fitsM_Hydro[ninf, nsup], fitsB_Hydro[ninf, nsup], fitsM_Hydro_std[
            ninf, nsup], fitsB_Hydro_std[ninf, nsup] = \
            find_PowerLaw(x_cumul, Vmax_cumul_hydro / 6., inferior, superior,
                          plot=False)

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
plt.suptitle('Pendiente de SHVF')

vmin = np.min((np.min(fitsM_DMO), np.min(fitsM_Hydro)))
vmax = np.max((np.max(fitsM_DMO), np.max(fitsM_Hydro)))

plt.subplot(211)
plt.title('dmo')

aaa = plt.imshow(fitsM_DMO.T,
                 extent=[limit_infG[0], limit_infG[-1], limit_supG[0],
                         limit_supG[-1]], aspect='auto',
                 vmin=vmin, vmax=vmax)

plt.subplot(212)
plt.title('hydro')
plt.xlabel('Limite inferior', size=20)
aaa = plt.imshow(fitsM_Hydro.T,
                 extent=[limit_infG[0], limit_infG[-1], limit_supG[0],
                         limit_supG[-1]], aspect='auto',
                 vmin=vmin, vmax=vmax)

cax, kw = colorbarr.make_axes([ax for ax in axes.flat])
c2 = plt.colorbar(aaa, cax=cax, **kw)

fig.text(0.06, 0.5, 'Limite superior', ha='center', va='center',
         rotation='vertical')

# -------------------------------------------------

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
plt.suptitle('Pendiente de SHVF entre -4.1 y -3.9')

vmin = np.min((np.min(fitsM_DMO), np.min(fitsM_Hydro)))
vmax = np.max((np.max(fitsM_DMO), np.max(fitsM_Hydro)))

plt.subplot(211)
plt.title('dmo')

mask = (fitsM_DMO > -4.1) * (fitsM_DMO < -3.9)
aaa = plt.imshow((fitsM_DMO * mask).T,
                 extent=[limit_infG[0], limit_infG[-1], limit_supG[0],
                         limit_supG[-1]], aspect='auto',
                 vmin=vmin, vmax=vmax)

plt.subplot(212)
plt.title('hydro')
plt.xlabel('Limite inferior', size=20)
mask = (fitsM_Hydro > -4.1) * (fitsM_Hydro < -3.9)
aaa = plt.imshow((fitsM_Hydro * mask).T,
                 extent=[limit_infG[0], limit_infG[-1], limit_supG[0],
                         limit_supG[-1]], aspect='auto',
                 vmin=vmin, vmax=vmax)

cax, kw = colorbarr.make_axes([ax for ax in axes.flat])
c2 = plt.colorbar(aaa, cax=cax, **kw)

fig.text(0.06, 0.5, 'Limite superior', ha='center', va='center',
         rotation='vertical')

# -------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(16, 8))
plt.suptitle('Pendiente de SHVF')

vmin = -4.3  # np.min((np.min(fitsM_DMO), np.min(fitsM_Hydro)))
vmax = -3.7  # np.max((np.max(fitsM_DMO), np.max(fitsM_Hydro)))

plt.subplot(221)
plt.title('dmo slope')

aaa = plt.imshow(fitsM_DMO.T,
                 extent=[limit_infG[0], limit_infG[-1], limit_supG[0],
                         limit_supG[-1]], aspect='auto',
                 vmin=vmin, vmax=vmax)

plt.subplot(223)
plt.title('hydro slope')
plt.xlabel('Limite inferior', size=20)
aaa = plt.imshow(fitsM_Hydro.T,
                 extent=[limit_infG[0], limit_infG[-1], limit_supG[0],
                         limit_supG[-1]], aspect='auto',
                 vmin=vmin, vmax=vmax)

cax, kw = colorbarr.make_axes([ax for ax in axes.flat])
c2 = plt.colorbar(aaa, cax=cax, **kw)
c2.set_label(r'Slope', fontsize=20)

plt.subplot(222)
plt.title('dmo std')

aaa = plt.imshow(fitsM_DMO_std.T,
                 extent=[limit_infG[0], limit_infG[-1], limit_supG[0],
                         limit_supG[-1]], aspect='auto', cmap='Greys')
#                 vmin=vmin, vmax=vmax)


plt.subplot(224)
plt.title('hydro std')
plt.xlabel('Limite inferior', size=20)
aaa = plt.imshow(fitsM_Hydro_std.T,
                 extent=[limit_infG[0], limit_infG[-1], limit_supG[0],
                         limit_supG[-1]], aspect='auto', cmap='Greys')
#                 vmin=vmin, vmax=vmax)


cax, kw = colorbarr.make_axes([ax for ax in axes.flat])
c2 = plt.colorbar(aaa, cax=cax, **kw)
c2.set_label(r'std', fontsize=20)

fig.text(0.06, 0.5, 'Limite superior', ha='center', va='center',
         rotation='vertical')

# ---------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(16, 8))
plt.suptitle('Pendiente de SHVF entre -4.1 y -3.9')

vmin = -4.3  # np.min((np.min(fitsM_DMO), np.min(fitsM_Hydro)))
vmax = -3.7  # np.max((np.max(fitsM_DMO), np.max(fitsM_Hydro)))

plt.subplot(221)
plt.title('dmo slope')

mask = (fitsM_DMO > -4.1) * (fitsM_DMO < -3.9)
aaa = plt.imshow((fitsM_DMO * mask).T,
                 extent=[limit_infG[0], limit_infG[-1], limit_supG[0],
                         limit_supG[-1]], aspect='auto',
                 vmin=vmin, vmax=vmax)

plt.subplot(222)
plt.title('dmo std')
vmaxx = np.max(fitsM_DMO_std)
fitsM_DMO_std[~mask] = 1
aaa = plt.imshow(fitsM_DMO_std.T,
                 extent=[limit_infG[0], limit_infG[-1], limit_supG[0],
                         limit_supG[-1]],
                 aspect='auto', cmap='Greys', vmax=vmaxx)

plt.subplot(223)
plt.title('hydro slope')
plt.xlabel('Limite inferior', size=20)
mask = (fitsM_Hydro > -4.1) * (fitsM_Hydro < -3.9)
aaa = plt.imshow((fitsM_Hydro * mask).T,
                 extent=[limit_infG[0], limit_infG[-1], limit_supG[0],
                         limit_supG[-1]], aspect='auto',
                 vmin=vmin, vmax=vmax)

cax, kw = colorbarr.make_axes([ax for ax in axes.flat])
c2 = plt.colorbar(aaa, cax=cax, **kw)
c2.set_label(r'Slope', fontsize=20)

plt.subplot(224)
plt.title('hydro std')
plt.xlabel('Limite inferior', size=20)
vmaxx = np.max(fitsM_Hydro_std)
fitsM_Hydro_std[~mask] = 1
aaa = plt.imshow(fitsM_Hydro_std.T,
                 extent=[limit_infG[0], limit_infG[-1], limit_supG[0],
                         limit_supG[-1]],
                 aspect='auto', cmap='Greys', vmax=vmaxx)

cax, kw = colorbarr.make_axes([ax for ax in axes.flat])
c2 = plt.colorbar(aaa, cax=cax, **kw)
c2.set_label(r'std', fontsize=20)

fig.text(0.06, 0.5, 'Limite superior', ha='center', va='center',
         rotation='vertical')

# ----------------------------------------------------------------------
# %% STUDY THE LIMITS OF THE VMAX IN THE FIT

number_bins = np.arange(15, 35)

numbers = 3
limit_infG = np.linspace(6, 10, num=numbers)
limit_supG = np.linspace(30, 70, num=numbers)

plt.figure(figsize=(10, 13))

ax1 = plt.subplot(222)
ax4 = plt.subplot(224)

red_array = np.linspace(0, 1, num=numbers)
blue_array = np.linspace(0, 1, num=numbers)

for ni, i in enumerate(red_array):
    for nj, j in enumerate(blue_array):
        ax1.plot(limit_infG[ni], limit_supG[nj], '.',
                 color=(i, 0, j), markersize=40)
        ax4.plot(limit_infG[ni], limit_supG[nj], '.',
                 color=(0, i, j), markersize=40)

ax2 = plt.subplot(221)
plt.ylabel('DMO')
ax2.axhline(4, color='grey')

ax3 = plt.subplot(223)
plt.ylabel('Hydro')
ax3.axhline(4, color='grey')
plt.xlabel('Number of bins')

for nbin, bin_i in enumerate(number_bins):
    if bin_i % 10:
        print(bin_i)
    x_cumul = np.logspace(np.log10(Grand_hydro[0, 1]),
                          np.log10(Grand_hydro[-1, 1]), num=bin_i)

    Vmax_cumul_dmo = calcular_dNdV(Grand_dmo[:, 1])
    Vmax_cumul_hydro = calcular_dNdV(Grand_hydro[:, 1])

    for ninf, inferior in enumerate(limit_infG):
        for nsup, superior in enumerate(limit_supG):
            fitsM_DMO, fitsB_DMO, fitsM_DMO_std, fitsB_DMO_std = \
                find_PowerLaw(x_cumul, Vmax_cumul_dmo / 6., inferior, superior,
                              plot=False)
            fitsM_Hydro, fitsB_Hydro, fitsM_Hydro_std, fitsB_Hydro_std = \
                find_PowerLaw(x_cumul, Vmax_cumul_hydro / 6., inferior,
                              superior,
                              plot=False)

            ax3.errorbar(x=bin_i + 0.45 * np.random.random(1)
                           * (-1) ** (np.random.random() < 0.5),
                         y=-fitsM_Hydro,
                         # yerr=fitsM_Hydro_std,
                         color=(red_array[ninf], 0, blue_array[nsup]),
                         linestyle='', marker='x', capsize=6, markersize=6)
            ax2.errorbar(x=bin_i + 0.45 * np.random.random(1)
                           * (-1) ** (np.random.random() < 0.5),
                         y=-fitsM_DMO,
                         # yerr=fitsM_DMO_std,
                         color=(red_array[ninf], 0, blue_array[nsup]),
                         linestyle='', marker='x',
                         capsize=6, markersize=6)

            ax3.errorbar(x=bin_i + 0.45 * np.random.random(1)
                           * (-1) ** (np.random.random() < 0.5),
                         y=fitsB_Hydro,
                         # yerr=fitsB_Hydro_std,
                         color=(0, red_array[ninf], blue_array[nsup]),
                         linestyle='', marker='o', capsize=6, markersize=3)
            ax2.errorbar(x=bin_i + 0.45 * np.random.random(1)
                           * (-1) ** (np.random.random() < 0.5),
                         y=fitsB_DMO,
                         # yerr=fitsB_DMO_std,
                         color=(0, red_array[ninf], blue_array[nsup]),
                         linestyle='', marker='o', capsize=6, markersize=3)

plt.show()
