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
plt.rc('legend', fontsize=16)
plt.rc('figure', titlesize=all_size)
plt.rc('xtick', top=True, direction='in')
plt.rc('ytick', right=True, direction='in')
plt.rc('xtick.major', size=10, width=2, top=False, pad=10)
plt.rc('ytick.major', size=10, width=2, right=True, pad=10)
plt.rc('xtick.minor', size=7, width=1.5, top=False)
plt.rc('ytick.minor', size=7, width=1.5)

#        Rmax[kpc]        Vmax[km/s]      Radius[Mpc]
try:
    Grand_dmo = np.loadtxt(
        '../Data_subhalos_simulations/RmaxVmaxRadDMO0_1.txt')
    Grand_hydro = np.loadtxt(
        '../Data_subhalos_simulations/RmaxVmaxRadFP0_1.txt')

except:
    Grand_dmo = np.loadtxt('../../RmaxVmaxRadDMO0_1.txt')
    Grand_hydro = np.loadtxt('../../RmaxVmaxRadFP0_1.txt')

Grand_hydro = Grand_hydro[Grand_hydro[:, 1] > 1e-4, :]
Grand_dmo[:, 2] *= 1e3
Grand_hydro[:, 2] *= 1e3

Grand_dmo = Grand_dmo[np.argsort(Grand_dmo[:, 1])]
Grand_hydro = Grand_hydro[np.argsort(Grand_hydro[:, 1])]

# Los mayores Dist_Gc son 264 kpc, pero entonces cuál es R200?
R200 = 263  # kpc

paper2012_dmo = np.loadtxt('dmo_2012.txt')
paper2012_hydro = np.loadtxt('hydro_2012.txt')

# datos_repop_hydro = np.loadtxt(
# '/home/saraporras/Desktop/TFM/Repopulation_codes/Hydro_complete.txt')
# datos_repop_dmo   = np.loadtxt(
# '/home/saraporras/Desktop/TFM/Repopulation_codes/DMO_complete.txt')

# vl2_data = np.loadtxt(
# '/home/saraporras/Desktop/TFM/Repopulation_codes/newVLtable.txt')
plt.close('all')

# %%

x_cumul = np.logspace(np.log10(Grand_hydro[0, 1]),
                      np.log10(Grand_hydro[-1, 1]), num=26)


def calcular_dNdV(Vmax):
    Vmax_cumul = np.zeros(len(x_cumul) - 1)

    for radius in range(len(Vmax_cumul)):
        aa = Vmax >= x_cumul[radius]
        bb = Vmax < x_cumul[radius + 1]

        Vmax_cumul[radius] = sum(aa * bb) / (
                x_cumul[radius + 1] - x_cumul[radius])

    return Vmax_cumul

def calcular_VdNdV(Vmax):
    Vmax_cumul = np.zeros(len(x_cumul) - 1)

    for radius in range(len(Vmax_cumul)):
        aa = Vmax >= x_cumul[radius]
        bb = Vmax < x_cumul[radius + 1]

        Vmax_cumul[radius] = (sum(aa * bb) / (
                x_cumul[radius + 1] - x_cumul[radius])
                              * (x_cumul[radius] + x_cumul[radius+1]) / 2.)

    return Vmax_cumul


Vmax_cumul_dmo = calcular_dNdV(Grand_dmo[:, 1])
Vmax_cumul_hydro = calcular_dNdV(Grand_hydro[:, 1])

Vmax_paper_dmo = calcular_VdNdV(Grand_dmo[:, 1])
Vmax_paper_hydro = calcular_VdNdV(Grand_hydro[:, 1])


x_cumul = (x_cumul[:-1] + x_cumul[1:]) / 2.
# Vhydro_repop = calcular_dNdV(datos_repop_hydro[:,3])
# Vdmo_repop   = calcular_dNdV(datos_repop_dmo[:,3])

# vl2_dndv = calcular_dNdV(vl2_data[:,3])



def find_PowerLaw(xx, yy, lim_inf, lim_sup, plot=True, color='k', label='',
                  style=''):
    X1limit = np.where(xx >= lim_inf)[0][0]
    X2limit = np.where(xx >= lim_sup)[0][0]

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

    if plot:
        plt.plot(xx, yy, label=label, color=color, linestyle=style)
        plt.plot(xx, yy, '.', color=color, zorder=10)

        xxx = np.logspace(np.log10(2), np.log10(xx[-1]), 100)
        plt.plot(xxx, 10 ** fits[1] * xxx ** fits[0],
                 color=color, alpha=0.7,
                 linestyle='-', lw=2)

    return fits[0], fits[1], perr[0], perr[1]


plt.figure()
plt.plot(paper2012_dmo[:, 0], paper2012_dmo[:, 1], color='k')
plt.plot(paper2012_hydro[:, 0], paper2012_hydro[:, 1], color='limegreen')

fits_dmo_paper = np.polyfit(x=np.log10(paper2012_dmo[:, 0]),
                            y=np.log10(paper2012_dmo[:, 1]),
                            deg=1)
print(fits_dmo_paper)

fits_hydro_paper = np.polyfit(x=np.log10(paper2012_hydro[:, 0]),
                            y=np.log10(paper2012_hydro[:, 1]),
                            deg=1)
print(fits_hydro_paper)

limit_infG = 8
limit_supG = 70
fitsM_DMO, fitsB_DMO, _, _ = find_PowerLaw(x_cumul, Vmax_paper_dmo / 6.,
                                           limit_infG, limit_supG,
                                           label='DMO')

fitsM_Hydro, fitsB_Hydro, _, _ = find_PowerLaw(x_cumul, Vmax_paper_hydro / 6.,
                                               limit_infG, limit_supG,
                                               color='limegreen', label='Hyd')

print(fitsM_DMO, fitsB_DMO)
print(fitsM_Hydro, fitsB_Hydro)

plt.xscale('log')
plt.yscale('log')

plt.show()

# %%
# plt.close('all')
fig = plt.figure(5)
print('Fig 5')

limit_infG = 8
limit_supG = 70

# NEW DATA CALCULUS dV/dN (dividing by 6) ----------------------

fitsM_DMO, fitsB_DMO, _, _ = find_PowerLaw(x_cumul, Vmax_cumul_dmo / 6.,
                                           limit_infG, limit_supG,
                                           label='DMO')

fitsM_Hydro, fitsB_Hydro, _, _ = find_PowerLaw(x_cumul, Vmax_cumul_hydro / 6.,
                                               limit_infG, limit_supG,
                                               color='limegreen', label='Hyd')


# COMPARISON MOLINE21 --------------------------------------------

def SHVF_Mol2021(V, inputs=[3.91, 9.72, 0.57, 0.92]):
    # SubHalo Velocity Function - number of subhs defined by their Vmax
    # Moline et al. 2110.02097
    #
    # V - max radial velocity of a bound particle in the subhalo [km/s]
    return ((V / Vmax / inputs[2]) ** (-inputs[0])
            * np.exp(-(V / Vmax / inputs[3]) ** inputs[1]))


Vmax_sub = 10
Vmax = 201.033  # from VLII website

# x = np.logspace(np.log10(3), np.log10(80))
# plt.plot(x, SHVF_Mol2021(x), label='Moline+21', color='r')

# bb = np.logspace(np.log10(limit_inf), np.log10(limit_sup))
# plt.plot(x, dNdV, '--', color='b', alpha=0.8)
# plt.plot(bb, -fitsM * bb**(fitsM-1) * 10**fitsB, '-', color='b',
# label='Hydro repop')


#  FIGURE DEFINITIONS ---------------------------------------------

# plt.plot(x_cumul, Vhydro_repop)
# plt.plot(x_cumul, Vdmo_repop)

plt.xscale('log')
plt.yscale('log')


plt.xlabel(r'$V_{\mathrm{max}}$ [km s$^{-1}$]', size=24)
plt.ylabel(r'$\frac{dN(V_{\mathrm{max}})}{dV_{\mathrm{max}}}$', size=27)

# fig.set_xticklabels( )

plt.legend()
plt.axvline(limit_infG, linestyle='-.', color='k', alpha=0.3,
            linewidth=2, label='Fit limits')
plt.axvline(limit_supG, linestyle='-.', color='k', alpha=0.3,
            linewidth=2)

handles = (mpatches.Patch(color='k', label='DMO', alpha=0.8),
           mpatches.Patch(color='limegreen', label='Hydro', alpha=0.8)
           )

legend11 = plt.legend(handles=handles,
                      loc=1)  # , bbox_to_anchor=(1.001, 0.99))

plt.savefig('outputs/shvf.pdf', bbox_inches='tight')
plt.savefig('outputs/shvf.png', bbox_inches='tight')

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
