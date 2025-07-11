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
plt.rc('ytick.major', size=10, width=2, right=True, pad=5)
plt.rc('xtick.minor', size=7, width=1.5)
plt.rc('ytick.minor', size=7, width=1.5)

data_release_dmo = np.loadtxt(
    '../Data_subhalo_simulations/dmo_table.txt', skiprows=3)
data_release_hydro = np.loadtxt(
    '../Data_subhalo_simulations/hydro_table.txt', skiprows=3)

# data_release_dmo = np.loadtxt(
#     '../Data_subhalo_simulations/data_dmo_level4.txt')
# data_release_hydro = np.loadtxt(
#     '../Data_subhalo_simulations/data_hydro_level4.txt')

data_release_dmo = data_release_dmo[
                   data_release_dmo[:, 0] > 0.184, :]
data_release_hydro = data_release_hydro[
                     data_release_hydro[:, 0] > 0.184, :]

data_release_dmo = data_release_dmo[np.argsort(data_release_dmo[:, 2])]
data_release_hydro = data_release_hydro[np.argsort(data_release_hydro[:, 2])]
print(data_release_dmo[0, 2] * 220/data_release_dmo[0, 5],
      data_release_hydro[0, 2] * 220/data_release_dmo[0, 5])
print(data_release_dmo[-1, 2], data_release_hydro[-1, 2])

print()
print(sum(data_release_dmo[:, 1] > 20), sum(data_release_hydro[:, 1] > 20))

R_vir = 211.834
R_last = 251.400

unique_halos = np.unique(data_release_hydro[:, 6])

# %% SRD

def encontrar_SRD_sinVol(data, bins):
    n_final = []
    std_fin = []

    for delta in range(len(bins) - 1):
        aaa = []
        for halo in unique_halos:
            data_ind = data[data[:, 6] == halo, :]
            interval = ((data_ind[:, 2] / data_ind[:, 5] >= bins[delta])
                        * (data_ind[:, 2] / data_ind[:, 5] <= bins[delta + 1]))
            aaa.append(sum(interval))

        # if delta == 0 or delta == 1 or delta == 2:
        #     print(aaa, np.nanmean(aaa), np.std(aaa))
        n_final.append(np.nanmean(aaa))
        std_fin.append(np.std(aaa))
    return np.array(n_final), np.array(std_fin)


def encontrar_SRD(data, bins):
    n_final = []
    std_fin = []

    for delta in range(len(bins) - 1):

        aaa = []
        vol = []
        for halo in unique_halos:
            data_ind = data[data[:, 6] == halo, :]
            interval = ((data_ind[:, 2] / data_ind[:, 5] >= bins[delta])
                        * (data_ind[:, 2] / data_ind[:, 5] <= bins[delta + 1]))
            aaa.append(sum(interval))
            vol.append(4 / 3 * np.pi * (
                    bins[delta + 1] ** 3 - bins[delta] ** 3)
                       * data_ind[0, 5] ** 3.
                       # * 1e-9
                       )
            if delta == 0:
                print(int(halo), np.min(data_ind[:, 2]))

        aaa = np.array(aaa)
        vol = np.array(vol)
        y = np.nanmean(aaa / vol)  # / len(data_ind))
        std = np.nanstd(aaa / vol)

        n_final.append(y)
        std_fin.append(std)
    return np.array(n_final), np.array(std_fin)


# ------------------------ N(r)/Ntot figure ------------------------------
print()
print('N/Ntot figures')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
plt.subplots_adjust(wspace=0.27)
plt.subplot(121)
plt.title('dmo')
lineas_verticales_dmo = [0., 8.22558240752492,
                     11.515815370534886, 16.12214151874884,
                     22.570998126248377, 31.599397376747724,
                     44.239156327446814, 61.93481885842554,
                     86.70874640179575]
lineas_verticales_hydro = [8.279260254013769, 11.590964355619276,
                           16.227350097866985, 22.718290137013778,
                           31.805606191819287, 44.527848668547,
                           62.3389881359658, 87.27458339035212]
# v_cut = [0.0, 9., 13., 18., 25., 36., 50., 70., 90.]
# v_cut = [0., 3., 4., 5., 6., 7., 8., 9., 13.]
v_cut = np.arange(3., 10., 5.)
# v_cut = lineas_verticales_dmo

from matplotlib import cm

num_bins = 15
bins = np.linspace(0, 1., num=num_bins)
bins_mean = (bins[:-1] + bins[1:]) / 2.
# bins = np.geomspace(1e-2, 1., num=num_bins)
# bins_mean = np.sqrt(bins[:-1] * bins[1:])
# plt.xscale('log')
volume = 4 / 3 * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
print('bins: ', bins * 220)

xxx = np.linspace(0., 1., num=200)

for ni, ii in enumerate(v_cut):
    print('v_cut: ', ii, ni)
    release_dmo_over = data_release_dmo[data_release_dmo[:, 1] >= ii, :]
    release_hydro_over = data_release_hydro[data_release_hydro[:, 1] >= ii, :]
    print(
          release_hydro_over[0, 2] * 220/data_release_dmo[0, 5])
    print(release_dmo_over[0, 2] * 220/data_release_dmo[0, 5],)
    srd_dmo_over_release, std_dmo_num = (
        np.array(encontrar_SRD_sinVol(release_dmo_over, bins))
        / len(release_dmo_over)
    )
    # srd_dmo_over_release[srd_dmo_over_release==0] = 0.01
    srd_hydro_over_release, std_hydro_num = (
        np.array(encontrar_SRD_sinVol(release_hydro_over, bins))
        / len(release_hydro_over)
    )
    # srd_hydro_over_release[srd_hydro_over_release==0] = 0.01

    # print('under stuff')

    srdnum_dmo_under, _ = np.array(encontrar_SRD_sinVol(
        data_release_dmo[data_release_dmo[:, 1] < ii, :], bins))
    srdnum_hydro_under, _ = np.array(encontrar_SRD_sinVol(
        data_release_hydro[data_release_hydro[:, 1] < ii, :], bins))

    ax1.plot(bins_mean, srd_dmo_over_release,
             c=cm.CMRmap(ni / float(len(v_cut))))
    ax1.plot(bins_mean, srdnum_dmo_under/sum(data_release_dmo[:, 1] < ii),
             c=cm.CMRmap(ni / float(len(v_cut))), ls='--')

    ax2.plot(bins_mean, srd_hydro_over_release,
             c=cm.CMRmap(ni / float(len(v_cut))), label='%.1f' % ii)
    ax2.plot(bins_mean, srdnum_hydro_under
              /sum(data_release_hydro[:, 1] < ii),
             c=cm.CMRmap(ni / float(len(v_cut))),
             ls='--')

plt.ylabel(r'$N(D_\mathrm{GC})$')
plt.ylabel(r'$N(D_\mathrm{GC})/N_\mathrm{total}$')
plt.xlabel(r'D$_\mathrm{GC} \, / \, R_\mathrm{vir}$ ', size=26)

# plt.xscale('linear')
plt.yscale('log')

plt.xlim(0, 1.)
plt.ylim(0.001, 0.03)

plt.subplot(122)
plt.xlim(0, 1.)
plt.ylim(0.001, 0.03)
plt.title('MHD')
plt.xlabel(r'D$_\mathrm{GC} \, / \, R_\mathrm{vir}$ ', size=26)

legend11 = plt.legend(loc=2, framealpha=1,
                      bbox_to_anchor=(1.04, 1), fontsize=12)
handles = (Line2D([0], [0], color='k', ls='-', label='Over'),
           Line2D([0], [0], color='k', ls='--', label='Below')
           )
legend22 = plt.legend(
    handles=handles,
    loc=3, framealpha=1,
    bbox_to_anchor=(1.04, 0), fontsize=12)
ax2.add_artist(legend11)
ax2.add_artist(legend22)

plt.yscale('log')

plt.savefig('outputs/srd_compar_Vmax_cut.png')

# plt.show()
# ------------------------ N(r)/Ntot figure ------------------------------
print()
print('N/Ntot figures')
vv_array = [0., 1., 5., 6., 7., 8., 9., 10., 13.]
vv_array = np.arange(0., 12., 0.5)
vv_array = [8.]
alpha_dmo = []
beta_dmo = []
alpha_hydro = []
beta_hydro = []

for aa in vv_array:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    plt.subplots_adjust(wspace=0.27)
    plt.subplot(121)

    v_cut = [aa]  #
    # v_cut = np.linspace(1., 8., num=10)

    xxx = np.linspace(0., 1., num=200)

    for ni, ii in enumerate(v_cut):
        print('v_cut: ', ii, ni)
        release_dmo_over = data_release_dmo[data_release_dmo[:, 1] >= ii, :]
        release_hydro_over = data_release_hydro[data_release_hydro[:, 1] >= ii, :]

        minnDgc = np.argmin(data_release_dmo[:, 2])

        bins_dmo = np.linspace(
            data_release_dmo[minnDgc, 2] / data_release_dmo[minnDgc, 5],
            1., num=num_bins)
        bins_mean_dmo = (bins_dmo[:-1] + bins_dmo[1:]) / 2.
        volume_dmo = 4 / 3 * np.pi * (bins_dmo[1:] ** 3 - bins_dmo[:-1] ** 3)
        srd_dmo_over_release, std_dmo_num = (np.array(encontrar_SRD_sinVol(
            release_dmo_over, bins_dmo))
            # / len(release_dmo_over)
        )

        print('dmo')
        srddensity_dmo_over_release, std_dmo_den = (encontrar_SRD(
            release_dmo_over, bins_dmo))

        bins_hydro = np.linspace(
            data_release_hydro[minnDgc, 2] / data_release_hydro[minnDgc, 5],
            1., num=num_bins)
        bins_mean_hydro = (bins_hydro[:-1] + bins_hydro[1:]) / 2.
        volume_hydro = 4 / 3 * np.pi * (bins_hydro[1:] ** 3 - bins_hydro[:-1] ** 3)
        srd_hydro_over_release, std_hydro_num = (np.array(encontrar_SRD_sinVol(
            release_hydro_over, bins_hydro))
            # / len(release_hydro_over)
        )

        print('\nhydro')
        srddensity_hydro_over_release, std_hydro_den = (encontrar_SRD(
            release_hydro_over, bins_hydro))

        ax1.errorbar(bins_mean_dmo, srd_dmo_over_release,
                     yerr=std_dmo_num,
                     ls='',
                     c='k',
                     ms=15, marker='.', markeredgewidth=2,
                     alpha=1, zorder=15,
                     label='Auriga',
                     capsize=5
                     # label=ii
                     )

        ax1.errorbar(bins_mean_hydro, srd_hydro_over_release,
                     yerr=std_hydro_num,
                     ls='',
                     c='g',
                     ms=15, marker='.', markeredgewidth=2,
                     alpha=1, zorder=15,
                     capsize=5)

        ax2.errorbar(bins_mean_dmo, srddensity_dmo_over_release,
                     yerr=std_dmo_den,
                     ls='',
                     color='k',
                     ms=15, marker='.', markeredgewidth=2,
                     alpha=1, zorder=15,
                     label='Auriga',
                     capsize=5
                     # label=ii
                     )

        ax2.errorbar(bins_mean_hydro, srddensity_hydro_over_release,
                     yerr=std_hydro_den,
                     ls='',
                     color='g',
                     ms=15, marker='.',  # markeredgewidth=3,
                     alpha=1, zorder=15,
                     capthick=2.,
                     capsize=5)


    def funct_ale(Dgc, a, b):
        return b * np.exp(a / Dgc)


    # --------------
    print('Release')
    cts_dmo = opt.curve_fit(funct_ale, xdata=bins_mean_dmo,
                            ydata=srd_dmo_over_release,
                            sigma=std_dmo_num,
                            p0=[-0.5, 200])
    print('Funct Ale: ', cts_dmo[0])
    bbb = np.diag(cts_dmo[1]) ** 0.5
    alpha_dmo.append([cts_dmo[0][0], bbb[0]])
    beta_dmo.append([cts_dmo[0][1], bbb[1]])

    cts_hydro = opt.curve_fit(funct_ale, xdata=bins_mean_hydro,
                              ydata=srd_hydro_over_release,
                              sigma=std_hydro_num,
                              p0=[0, 200])
    print('Funct Ale: ', cts_hydro[0])
    bbb = np.diag(cts_hydro[1]) ** 0.5
    alpha_hydro.append([cts_hydro[0][0], bbb[0]])
    beta_hydro.append([cts_hydro[0][1], bbb[1]])

    plt.plot(xxx, funct_ale(xxx, cts_dmo[0][0], cts_dmo[0][1]),
             'dimgray', linestyle='--', lw=3, alpha=0.7,
             label='Fragile', zorder=5)
    plt.plot(xxx, funct_ale(xxx, cts_hydro[0][0], cts_hydro[0][1]),
             'limegreen', linestyle='--', lw=3, zorder=5)

    plt.plot(xxx, np.ones(len(xxx))
             * funct_ale(1., cts_dmo[0][0], cts_dmo[0][1]),
             'dimgray', linestyle='dotted', lw=4, alpha=0.7,
             label='Resilient', zorder=5)
    plt.plot(xxx, np.ones(len(xxx))
             * funct_ale(1., cts_hydro[0][0], cts_hydro[0][1]),
             'limegreen', linestyle='dotted', lw=4, zorder=5)
    print('resilient values: ',
          funct_ale(1., cts_dmo[0][0], cts_dmo[0][1]),
          funct_ale(1., cts_hydro[0][0], cts_hydro[0][1]))

    print('resilient errors: ',
          funct_ale(1., cts_dmo[0][0], cts_dmo[0][1]),
          funct_ale(1., cts_hydro[0][0], cts_hydro[0][1]))

    # ----------------
    plt.axvline(8.5 / 220., linestyle='-.', alpha=1, color='Sandybrown', lw=3)
    # plt.annotate('Earth', (0.05, 0.130), color='Saddlebrown', rotation=0.,
    #              fontsize=20, zorder=10)
    plt.annotate(r'$R_\oplus$', (0.05, 39),  color='chocolate',
                 rotation=0., weight='bold',
                 fontsize=20, zorder=10)

    # plt.axvline(data_release_dmo[0, 2] / data_release_dmo[0, 5],
    #             alpha=0.5, color='k', linestyle='-',
    #             lw=3, label='Last subhalo')
    # plt.axvline(data_release_hydro[0, 2] / data_release_hydro[0, 5],
    #             alpha=0.5, color='limegreen',
    #             linestyle='-', lw=3)

    # plt.ylabel(r'$N(D_\mathrm{GC}) \, / \, N_\mathrm{Total}$')
    plt.ylabel(r'$N(D_\mathrm{GC})$')
    plt.xlabel(r'D$_\mathrm{GC} \, / \, R_\mathrm{vir}$ ', size=30)
    # plt.xlabel(r'D$_\mathrm{GC}$ [kpc]', size=24)

    plt.xscale('linear')
    plt.yscale('log')

    plt.xlim(0, 1.)
    # plt.ylim(1, 300)
    plt.ylim(0.5, 50)

    # plt.legend(framealpha=1, fontsize=10, loc=4)
    handles = (mpatches.Patch(color='k', label='DMO', alpha=0.8),
               mpatches.Patch(color='limegreen', label='MHD', alpha=0.8)
               )

    legend_colors = plt.legend(handles=handles, loc=8,
                               # bbox_to_anchor=(0.99, 0.26),
                               fontsize=20)

    legend11 = plt.legend(loc=4, bbox_to_anchor=(0.99, 0.01),
                          framealpha=1)

    ax1.add_artist(legend11)
    ax1.add_artist(legend_colors)
    plt.savefig('outputs/srd_compar_lin_after.png', bbox_inches='tight')
    plt.savefig('outputs/srd_compar_lin_after.pdf', bbox_inches='tight')
    # plt.show()
    # -------------------------------------------------------------------------
    print('Density figure')

    # plt.figure(figsize=(10, 8))
    plt.subplot(122)


    mean_r_dmo = sum([
        i * sum(data_release_dmo[:, 5] == i)
        for i in np.unique(data_release_dmo[:, 5])]) / len(data_release_dmo)

    mean_r_hyd = sum([
        i * sum(data_release_hydro[:, 5] == i)
        for i in np.unique(data_release_hydro[:, 5])]) / len(data_release_hydro)

    print(mean_r_dmo, mean_r_hyd)

    volume_220_dmo = volume_dmo * mean_r_dmo ** 3.
    volume_220_hydro = volume_hydro * mean_r_hyd ** 3.

    # print((bins[:-1] - bins[1:]))
    # print(cts_hydro[0][1] / volume / (cts_dmo[0][1] / volume))

    aaa = UnivariateSpline(
        x=bins_mean_dmo,
        y=funct_ale(bins_mean_dmo, cts_dmo[0][0], cts_dmo[0][1])
          / volume_220_dmo, k=1, s=0)
    new_x = np.concatenate([[bins_dmo[0]], bins_mean_dmo, [bins_dmo[-1]]])
    plt.plot(new_x, aaa(new_x),
             '--', marker='', ms=20, lw=3,
             color='dimgray', alpha=0.7, label='Fragile')

    aaa = UnivariateSpline(
        x=bins_mean_hydro,
        y=funct_ale(bins_mean_hydro, cts_hydro[0][0], cts_hydro[0][1])
          / volume_220_hydro, k=1, s=0)
    new_x = np.concatenate([[bins_hydro[0]], bins_mean_hydro, [bins_hydro[-1]]])
    plt.plot(new_x, aaa(new_x),
             '--', marker='', ms=20, color='limegreen', alpha=1, lw=3)

    aaa = UnivariateSpline(
        x=bins_mean_dmo,
        y=np.log10(funct_ale(1., cts_dmo[0][0], cts_dmo[0][1])
          / volume_220_dmo), k=1, s=0)
    new_x = np.concatenate([[bins_dmo[0]], bins_mean_dmo, [bins_dmo[-1]]])
    plt.plot(new_x, 10**aaa(new_x),
             marker='', ms=10, lw=3,
             color='dimgray', alpha=0.7, linestyle='dotted',
             label='Resilient')

    aaa = UnivariateSpline(
        x=bins_mean_hydro,
        y=np.log10(funct_ale(1., cts_hydro[0][0], cts_hydro[0][1])
          / volume_220_hydro), k=1, s=0)
    new_x = np.concatenate(
        [[bins_hydro[0]], bins_mean_hydro, [bins_hydro[-1]]])
    plt.plot(new_x, 10**aaa(new_x),
             marker='', linestyle='dotted', ms=10, color='limegreen',
             alpha=1, lw=3)
    # print(cts_dmo, cts_hydro)


    plt.axvline(8.5 / 220., linestyle='-.', alpha=1, color='Sandybrown', lw=3)
    # plt.annotate('Earth', (0.05, 0.130), color='Saddlebrown', rotation=0.,
    #              fontsize=20, zorder=10)
    plt.annotate(r'$R_\oplus$', (0.05, 1.47e-3), color='chocolate',
                 rotation=0., weight='bold',
                 fontsize=20, zorder=10)

    # plt.axvline(data_release_dmo[0, 2] / data_release_dmo[0, 5],
    #             alpha=0.5, color='k', linestyle='-',
    #             lw=3, label='Last subhalo')
    # plt.axvline(data_release_hydro[0, 2] / data_release_hydro[0, 5],
    #             alpha=0.5, color='limegreen',
    #             linestyle='-', lw=3)

    # plt.ylabel(r'$\frac{N(D_\mathrm{GC})}{\mathrm{Unit\,\,volume}}$',
    #            size=24)
    plt.ylabel(r'$\frac{N(D_\mathrm{GC})}{Volume}$'
               r' $\left[\mathrm{kpc}^{-3}\right]$',
               size=24)
    plt.xlabel(r'D$_\mathrm{GC} \, / \, R_\mathrm{vir}$ ', size=26)

    legend_elements = [Line2D([0], [0], marker='o', color='w',
                              markerfacecolor='k', markersize=8),
                       Line2D([0], [0], marker='o', color='w',
                              markerfacecolor='limegreen', markersize=8)]
    legend1 = plt.legend(handles, ['DMO', 'MHD'], loc=7,
                         # bbox_to_anchor=(0.99, 0.74),
                         fontsize=20)
    leg = plt.legend(framealpha=1, loc=1,
                     bbox_to_anchor=(0.99, 0.99))
    plt.gca().add_artist(legend1)

    # plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0., 1.)

    # plt.ylim(0, 60)

    plt.savefig('outputs/srd_compar_den'+ str(aa) + '.png',
                bbox_inches='tight')
    plt.savefig('outputs/srd_compar_den'+ str(aa) + '.pdf',
                bbox_inches='tight')

# plt.close('all')
alpha_dmo = np.array(alpha_dmo)
beta_dmo = np.array(beta_dmo)
alpha_hydro = np.array(alpha_hydro)
beta_hydro = np.array(beta_hydro)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
plt.suptitle(r'$y = b \cdot e^{a / Dgc}$')
ax1.errorbar(vv_array, alpha_dmo[:, 0], yerr=alpha_dmo[:, 1], c='k',
             ls='',
             ms=15, marker='.', markeredgewidth=2,
             capsize=5
             )
ax1.errorbar(vv_array, alpha_hydro[:, 0], yerr=alpha_hydro[:, 1], c='g',
             ls='',
             ms=15, marker='.', markeredgewidth=2,
             capsize=5
             )

ax1.axhline(-0.15, alpha=0.5, zorder=0, color='k')
ax1.axhline(-0.27, alpha=0.5, zorder=0, color='g')

ax2.errorbar(vv_array, beta_dmo[:, 0], yerr=beta_dmo[:, 1], c='k',
             ls='',
             ms=15, marker='.', markeredgewidth=2,
             capsize=5
             )
ax2.errorbar(vv_array, beta_hydro[:, 0], yerr=beta_hydro[:, 1], c='g',
             ls='',
             ms=15, marker='.', markeredgewidth=2,
             capsize=5
             )

ax1.set_ylabel('a')
ax2.set_ylabel('b')

ax1.set_xlabel(r'$V_{\mathrm{max}}$ [km s$^{-1}$]')
ax2.set_xlabel(r'$V_{\mathrm{max}}$ [km s$^{-1}$]')

# plt.show()
