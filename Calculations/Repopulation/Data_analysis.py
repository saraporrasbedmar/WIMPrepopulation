#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 10:02:09 2022

@author: saraporras
"""
import os
import numpy as np
import matplotlib.colorbar as colorbarr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.colors as mcb
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator, LogLocator

import Calculations.Repopulation.attemp_at_functions2 as funct_repop

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rc('font', size=20)
plt.rc('axes', titlesize=16)
plt.rc('axes', labelsize=16)
plt.rc('xtick', labelsize=22)
plt.rc('ytick', labelsize=22)
plt.rc('legend', fontsize=18)
plt.rc('figure', titlesize=17)
plt.rc('xtick', top=True, direction='in')
plt.rc('ytick', right=True, direction='in')
plt.rc('xtick.major', size=7, width=1.5, top=True, pad=7)
plt.rc('ytick.major', size=7, width=1.5, right=True, pad=7)
plt.rc('xtick.minor', size=4, width=1)
plt.rc('ytick.minor', size=4, width=1)

# path_name = '/home/porrassa/Downloads' \
#             '/2024_resilient_const_SHVFnorm_to120_roche_changeSRD'
# '/final_2024_120max'
# path_name = '/home/saraporras/Desktop/WIMPSproject/compiled_results'\
#             '/final_2024_8max'

path_name = ('/home/saraporras/Desktop/WIMPSproject/'#Calculations/'
             # 'Repopulation/outputs/2024_resilient_const_to120_rint/'
            # 'Physnet_outputs_repops/2024'
             'compiled_results'
             '/2024_resilient_const_to120_rint')
            # 'final_2024_8max'
#
print(os.getcwd())
print(os.listdir(path_name))
final_size = (500, 1, 6)
plot_res = True
plot_frag = True

if plot_res:
    end_str = '_res'

if plot_frag:
    end_str = '_frag'

if plot_res and plot_frag:
    end_str = '_both'
'''
datos_Js_frag_hyd = np.loadtxt(path_name +
                               '/Js_hydro_fragile_results.txt'
                               ).reshape(10, 2198, 6)
datos_Js_frag_dmo = np.loadtxt(path_name +
                               '/Js_dmo_fragile_results.txt')
datos_Js_frag_hyd = datos_Js_frag_hyd[:, 0, :]
datos_Js_frag_dmo = datos_Js_frag_dmo[:, 0, :]
# datos_Js_frag_hyd = datos_Js_frag_hyd[datos_Js_frag_hyd[:, 0] != 0, :]
# datos_Js_frag_dmo = datos_Js_frag_dmo[datos_Js_frag_dmo[:, 0] != 0, :]
# datos_Js_resi_hyd = datos_Js_resi_hyd.reshape(final_size)[:, 0, :]
# datos_Js_resi_dmo = datos_Js_resi_dmo.reshape(final_size)[:, 0, :]

# datos_Js_frag_hyd = datos_Js_frag_hyd[:, 0, :]
# where0 = datos_Js_frag_hyd[:, 0] < 1e-43
# print('datos_Js_frag_hyd', sum(where0))
# datos_Js_frag_hyd = datos_Js_frag_hyd[~where0, :]

# datos_Js_frag_dmo = datos_Js_frag_dmo[:, 0, :]
# where0 = datos_Js_frag_dmo[:, 0] < 1e-43
# print('datos_Js_frag_dmo', sum(where0))
# datos_Js_frag_dmo = datos_Js_frag_dmo[~where0, :]

datos_J03_frag_hyd = np.loadtxt(path_name +
                                '/J03_hydro_fragile_results.txt').reshape(
    10, 2198, 6)  # .reshape(25, 310, 6)
datos_J03_frag_dmo = np.loadtxt(path_name +
                                '/J03_dmo_fragile_results.txt'
                                ).reshape(10, 6629, 6)  # .reshape(25, 1034, 6)
datos_J03_frag_hyd = datos_J03_frag_hyd[:, 0, :]
datos_J03_frag_dmo = datos_J03_frag_dmo[:, 0, :]
# datos_J03_frag_hyd = datos_J03_frag_hyd[datos_J03_frag_hyd[:, 0] != 0, :]
# datos_J03_frag_dmo = datos_J03_frag_dmo[datos_J03_frag_dmo[:, 0] != 0, :]
# datos_J03_frag_hyd = datos_J03_frag_hyd[:, 0, :]
# where0 = datos_J03_frag_hyd[:, 0] < 1e-43
# print('datos_J03_frag_hyd', sum(where0))
# datos_J03_frag_hyd = datos_J03_frag_hyd[~where0, :]
#
# datos_J03_frag_dmo = datos_J03_frag_dmo[:, 0, :]
# where0 = datos_J03_frag_dmo[:, 0] < 1e-43
# print('datos_J03_frag_dmo', sum(where0))
# datos_J03_frag_dmo = datos_J03_frag_dmo[~where0, :]

# datos_Js_frag_hyd = np.loadtxt(path_name + '/Js_hydro_fragile_results.txt')
# datos_Js_frag_dmo = np.loadtxt(path_name + '/Js_dmo_fragile_results.txt')
# datos_Js_frag_hyd = datos_Js_frag_hyd.reshape(
#     (int(datos_Js_frag_hyd.size/6), 1, 6))[:, 0, :]
# datos_Js_frag_dmo = datos_Js_frag_dmo.reshape(
#     (int(datos_Js_frag_dmo.size/6), 1, 6))[:, 0, :]

# datos_J03_frag_hyd = np.loadtxt(path_name + '/J03_hydro_fragile_results.txt')
# datos_J03_frag_dmo = np.loadtxt(path_name + '/J03_dmo_fragile_results.txt')
# datos_J03_frag_hyd = datos_J03_frag_hyd.reshape(
#     (int(datos_J03_frag_hyd.size/6), 1, 6))[:, 0, :]
# datos_J03_frag_dmo = datos_J03_frag_dmo.reshape(
#     (int(datos_J03_frag_dmo.size/6), 1, 6))[:, 0, :]



datos_Js_resi_hyd = np.loadtxt(path_name +
                               '/Js_hydro_resilient_results.txt').reshape(
    10, 2198, 6)#.reshape(25, 310, 6)
datos_Js_resi_dmo = np.loadtxt(path_name +
                               '/Js_dmo_resilient_results.txt'
                               ).reshape(10, 6629, 6)#.reshape(25, 1034, 6)
datos_Js_resi_hyd = datos_Js_resi_hyd[:, 0, :]
datos_Js_resi_dmo = datos_Js_resi_dmo[:, 0, :]
# datos_Js_resi_hyd = datos_Js_resi_hyd[datos_Js_resi_hyd[:, 0] != 0, :]
# datos_Js_resi_dmo = datos_Js_resi_dmo[datos_Js_resi_dmo[:, 0] != 0, :]
# datos_Js_resi_hyd = datos_Js_resi_hyd.reshape(final_size)[:, 0, :]
# datos_Js_resi_dmo = datos_Js_resi_dmo.reshape(final_size)[:, 0, :]

# datos_Js_resi_hyd = datos_Js_resi_hyd[:, 0, :]
# where0 = datos_Js_resi_hyd[:, 0] < 1e-43
# print('datos_Js_resi_hyd', sum(where0))
# datos_Js_resi_hyd = datos_Js_resi_hyd[~where0, :]
#
# datos_Js_resi_dmo = datos_Js_resi_dmo[:, 0, :]
# where0 = datos_Js_resi_dmo[:, 0] < 1e-43
# print('datos_Js_resi_dmo', sum(where0))
# datos_Js_resi_dmo = datos_Js_resi_dmo[~where0, :]

datos_J03_resi_hyd = np.loadtxt(path_name +
                                '/J03_hydro_resilient_results.txt').reshape(
    10, 2198, 6)#.reshape(25, 310, 6)
datos_J03_resi_dmo = np.loadtxt(path_name +
                                '/J03_dmo_resilient_results.txt').reshape(
    10, 6629, 6)#.reshape(25, 1034, 6)
datos_J03_resi_hyd = datos_J03_resi_hyd[:, 0, :]
datos_J03_resi_dmo = datos_J03_resi_dmo[:, 0, :]
# datos_J03_resi_hyd = datos_J03_resi_hyd[datos_J03_resi_hyd[:, 0] != 0, :]
# datos_J03_resi_dmo = datos_J03_resi_dmo[datos_J03_resi_dmo[:, 0] != 0, :]
# datos_J03_resi_hyd = datos_J03_resi_hyd.reshape(final_size)[:, 0, :]
# datos_J03_resi_dmo = datos_J03_resi_dmo.reshape(final_size)[:, 0, :]

# datos_J03_resi_hyd = datos_J03_resi_hyd[:, 0, :]
# where0 = datos_J03_resi_hyd[:, 0] < 1e-43
# print('datos_J03_resi_hyd', sum(where0))
# datos_J03_resi_hyd = datos_J03_resi_hyd[~where0, :]

# datos_J03_resi_dmo = datos_J03_resi_dmo[:, 0, :]
# where0 = datos_J03_resi_dmo[:, 0] < 1e-43
# print('datos_J03_resi_dmo', sum(where0))
# datos_J03_resi_dmo = datos_J03_resi_dmo[~where0, :]
'''

path_name_res = ('/home/saraporras/Desktop/WIMPSproject/'#Calculations/'
             # 'Repopulation/outputs/2024_resilient_const_to120_rint/'
            # 'Physnet_outputs_repops/2024/'
                 'compiled_results/'
                 '2024_resilient_const_to120_rint_SHVFnorm')

final_size = (500, 100, 6)

datos_Js_resi_hyd = np.ones((1, 6))
datos_Js_resi_dmo = np.ones((1, 6))

datos_J03_resi_hyd = np.ones((1, 6))
datos_J03_resi_dmo = np.ones((1, 6))

datos_Js_frag_hyd = np.ones((1, 6))
datos_Js_frag_dmo = np.ones((1, 6))

datos_J03_frag_hyd = np.ones((1, 6))
datos_J03_frag_dmo = np.ones((1, 6))

for i in range(1, 6, 1):
    print(i, np.shape(datos_Js_resi_dmo),
          np.shape(np.loadtxt(path_name_res + '/' + str(i) +
                              '/Js_dmo_resilient_results.txt')))
    datos_Js_resi_dmo = np.concatenate((
        datos_Js_resi_dmo,
        np.loadtxt(path_name_res + '/' + str(i) +
                   '/Js_dmo_resilient_results.txt')))
    datos_Js_resi_hyd = np.concatenate((
        datos_Js_resi_hyd,
        np.loadtxt(path_name_res + '/' + str(i) +
                   '/Js_hydro_resilient_results.txt')))
    datos_J03_resi_dmo = np.concatenate((
        datos_J03_resi_dmo,
        np.loadtxt(path_name_res + '/' + str(i) +
                   '/J03_dmo_resilient_results.txt')))
    datos_J03_resi_hyd = np.concatenate((
        datos_J03_resi_hyd,
        np.loadtxt(path_name_res + '/' + str(i) +
                   '/J03_hydro_resilient_results.txt')))

    datos_Js_frag_dmo = np.concatenate((datos_Js_frag_dmo,
        np.loadtxt(path_name + '/' + str(i) +
                                 '/Js_dmo_fragile_results.txt')))
    datos_Js_frag_hyd = np.concatenate((datos_Js_frag_hyd,
        np.loadtxt(path_name + '/' + str(i) +
                                 '/Js_hydro_fragile_results.txt')))
    datos_J03_frag_dmo = np.concatenate((datos_J03_frag_dmo,
        np.loadtxt(path_name + '/' + str(i) +
                                 '/J03_dmo_fragile_results.txt')))
    datos_J03_frag_hyd = np.concatenate((datos_J03_frag_hyd,
        np.loadtxt(path_name + '/' + str(i) +
                                 '/J03_hydro_fragile_results.txt')))

datos_Js_resi_hyd = datos_Js_resi_hyd[1:, :]
datos_Js_resi_dmo = datos_Js_resi_dmo[1:, :]
datos_J03_resi_hyd = datos_J03_resi_hyd[1:, :]
datos_J03_resi_dmo = datos_J03_resi_dmo[1:, :]

datos_Js_frag_hyd = datos_Js_frag_hyd[1:, :]
datos_Js_frag_dmo = datos_Js_frag_dmo[1:, :]
datos_J03_frag_hyd = datos_J03_frag_hyd[1:, :]
datos_J03_frag_dmo = datos_J03_frag_dmo[1:, :]
#


constraints_bb_2204 = np.loadtxt('../Constraints_2204/Limit_bb.txt')
constraints_tau_2204 = np.loadtxt('../Constraints_2204/Limit_tau.txt')
sigmav_bb_2204 = np.loadtxt('../Constraints_2204/sigmav_bb.txt')
sigmav_tau_2204 = np.loadtxt('../Constraints_2204/sigmav_tau.txt')

sigmav_bb_2204 = sigmav_bb_2204[sigmav_bb_2204[:, 0].argsort()[::], :]
sigmav_tau_2204 = sigmav_tau_2204[sigmav_tau_2204[:, 0].argsort()[::], :]

J03_min95_2204 = 18.9208  # From digitalizing
Js_min95_2204 = 19.4642  # From digitalizing

path_name_res = path_name_res + '/figures'

# path_name = '/home/porrassa/Desktop/WIMPS_project/' \
#             'Physnet_outputs_repops' \
#             '/2024/'
# path_name = '/home/saraporras/Desktop/WIMPSproject/WIMPrepopulation/' \
#             'Calculations/Repopulation/outputs/2024_multidark'

def minnmaxxS(i):
    minn = np.min((
        np.min(np.log10(datos_Js_frag_dmo[:, i])),
        np.min(np.log10(datos_Js_frag_hyd[:, i])),
        np.min(np.log10(datos_Js_resi_dmo[:, i])),
        np.min(np.log10(datos_Js_resi_hyd[:, i]))
        ))

    maxx = np.max((
        np.max(np.log10(datos_Js_frag_dmo[:, i])),
        np.max(np.log10(datos_Js_frag_hyd[:, i])),
        np.max(np.log10(datos_Js_resi_dmo[:, i])),
        np.max(np.log10(datos_Js_resi_hyd[:, i])),
    ))

    return minn, maxx


def minnmaxx03(i):
    minn = np.min((
        np.min(np.log10(datos_J03_frag_dmo[:, i])),
        np.min(np.log10(datos_J03_frag_hyd[:, i])),
        np.min(np.log10(datos_J03_resi_dmo[:, i])),
        np.min(np.log10(datos_J03_resi_hyd[:, i]))))

    maxx = np.max((
        np.max(np.log10(datos_J03_frag_dmo[:, i])),
        np.max(np.log10(datos_J03_frag_hyd[:, i])),
        np.max(np.log10(datos_J03_resi_dmo[:, i])),
        np.max(np.log10(datos_J03_resi_hyd[:, i])),))

    return minn, maxx


def perc_total(i, number):
    data = (
        np.concatenate((
            datos_Js_frag_dmo[:, i], datos_Js_frag_hyd[:, i],
            datos_Js_resi_dmo[:, i], datos_Js_resi_hyd[:, i],
            datos_J03_frag_dmo[:, i], datos_J03_frag_hyd[:, i],
            datos_J03_resi_dmo[:, i], datos_J03_resi_hyd[:, i]),
            axis=None))
    return np.percentile(data, number)


darkgreen = (0.024, 0.278, 0.047)
cmap = cm.viridis
colormapp = 'viridis'

# ----------------------- Pop close to GC -------------------------------

high_pop_res_dmo = datos_Js_resi_dmo[datos_Js_resi_dmo[:, 1] < 15, :]
high_pop_res_dmo = high_pop_res_dmo[high_pop_res_dmo[:, 3] > 60, :]
# datos_Js_resi_dmo = high_pop_res_dmo

high_pop_res_hydro = datos_Js_resi_hyd[datos_Js_resi_hyd[:, 1] < 15, :]
high_pop_res_hydro = high_pop_res_hydro[high_pop_res_hydro[:, 3] > 60, :]
# datos_Js_resi_hyd = high_pop_res_hydro

high_pop_res_dmo = datos_J03_resi_dmo[datos_J03_resi_dmo[:, 1] < 15, :]
high_pop_res_dmo = high_pop_res_dmo[high_pop_res_dmo[:, 3] > 60, :]
# datos_J03_resi_dmo = high_pop_res_dmo

high_pop_res_hydro = datos_J03_resi_hyd[datos_J03_resi_hyd[:, 1] < 15, :]
high_pop_res_hydro = high_pop_res_hydro[high_pop_res_hydro[:, 3] > 60, :]
# datos_J03_resi_hyd = high_pop_res_hydro

high_pop_frag_dmo = datos_Js_frag_dmo[datos_Js_frag_dmo[:, 1] < 15, :]
high_pop_frag_dmo = high_pop_frag_dmo[high_pop_frag_dmo[:, 3] > 60, :]
# datos_Js_frag_dmo = high_pop_frag_dmo

high_pop_frag_hydro = datos_Js_frag_hyd[datos_Js_frag_hyd[:, 1] < 15, :]
high_pop_frag_hydro = high_pop_frag_hydro[high_pop_frag_hydro[:, 3] > 60, :]
# datos_Js_frag_hyd = high_pop_frag_hydro

# ----------------------- Vmax - Jss (z==DistEarth) 2x2 -----------------

fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,
                         figsize=(12, 9))

plt.subplots_adjust(wspace=0, hspace=0)

legend_elements = [
    Line2D([0], [0], marker='o', color='k', label='Frag',
                          markerfacecolor='w', ls='',
                          markersize=8, mew=2.5),
    Line2D([0], [0], marker='P', color='w', label='Res',
                          markerfacecolor='k', markersize=12)]

x_col = 3
y_col = 0
z_col = 2

x_label = r'$V_\mathrm{max}$ $\left[\mathrm{km}\,\,\mathrm{s}^{-1}\right]$'
y1_label = r'fragile'
y2_label = r'resilient'

plt.text(-0.3, 0., r'log$_{10}$ ($J_\mathrm{s}$ [GeV$^2$ cm$^{-5}$])',
         horizontalalignment='center',
 verticalalignment='center', transform=axes[0, 0].transAxes,
         rotation=90)

# vminn = 1e-3
# vmaxx = np.log10(150)  # 10 ** 1.5
vminn = np.log10(perc_total(z_col, 5))
vmaxx = np.log10(perc_total(z_col, 95))
print(vminn, vmaxx)
norm = mcb.Normalize(vminn, vmaxx)

minns, maxxs = minnmaxxS(y_col)
minns = minns - 0.2
maxxs = maxxs + 0.2


minnx = 0.1  #perc_total(x_col, 0) * 0.7
maxxx = perc_total(x_col, 100) * 1.3

plt.subplot(221)

plt.title('DMO', fontsize=20)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
# plt.yscale('log')

# plt.text(1., 1.01, 'DMO', horizontalalignment='center',
#  verticalalignment='bottom', transform=axes[0, 0].transAxes)

plt.scatter(datos_Js_frag_dmo[:, x_col],
            np.log10(datos_Js_frag_dmo[:, y_col]),
            c=np.log10(datos_Js_frag_dmo[:, z_col]),
            lw=0, marker='P', s=75,
            cmap=colormapp, vmin=vminn, vmax=vmaxx)

plt.ylabel(y1_label, fontsize=20, labelpad=10)

plt.legend(handles=legend_elements, handletextpad=0.2,
           handlelength=1, loc=2)

plt.subplot(223)

# plt.tick_params('y', labelleft=False)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.ylabel(y2_label, size=20, labelpad=10)
plt.xlabel(x_label, size=20)

# plt.yscale('log')
# plt.axvspan(0.935 * maxxx, maxxx, color='k')

plt.scatter(datos_Js_resi_dmo[:, x_col],
            np.log10(datos_Js_resi_dmo[:, y_col]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_resi_dmo[:, z_col]))))


plt.subplot(222)

plt.title('Hydro', fontsize=20)
# plt.text(1., 1.01, 'Hydro', horizontalalignment='center',
#  verticalalignment='bottom', transform=axes[0, 2].transAxes)

plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
# plt.yscale('log')
plt.tick_params('y', labelleft=False)
# plt.axvspan(minnx, minnx * 1.04, color='k')

plt.scatter(datos_Js_frag_hyd[:, x_col],
            np.log10(datos_Js_frag_hyd[:, y_col]),
            c=np.log10(datos_Js_frag_hyd[:, z_col]),
            lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

plt.subplot(224)

plt.tick_params('y', labelleft=False)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
# plt.yscale('log')
plt.xlabel(x_label, size=20)

plt.scatter(datos_Js_resi_hyd[:, x_col],
            np.log10(datos_Js_resi_hyd[:, y_col]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_resi_hyd[:, z_col]))),
            label='Resilient')


# plt.subplot(2, 4, 5)
# axes[1, 0].set_xticks([0.1, 1, 10], labels=('aaaaaa', '1', '10'))
    # a.set_xticklabels(['0.01', '0.1', '1', '10'])
# axes[1, 0].set_xticks([1,4,5])
# axes[1, 0].set_xticklabels([1,4,5], fontsize=12)

# for ii in range(1, 5):
#     plt.subplot(2, 4, ii)
plt.yticks((19, 20, 21, 22, 23))

plt.subplot(2, 2, 3)
plt.xticks([0.1, 1, 10, 100], labels=('0.1', '1', '10', ''))

plt.subplot(2, 2, 4)
plt.xticks([0.1, 1, 10, 100], labels=('0.1', '1', '10', '100'))


cax, kw = colorbarr.make_axes([ax for ax in axes.flat])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
c2 = plt.colorbar(sm, cax=cax, extend='both', spacing='proportional')
# c2.set_label(r'log$_{10}$(D$_\mathrm{Earth}$ [kpc])', fontsize=20)
c2.set_label(r'D$_\mathrm{Earth}$ [kpc]', fontsize=20)
c2.ax.tick_params(axis='y', direction='out')
# c2.set_tickparams(direction='out')
yticks = c2.get_ticks()
print(yticks)
c2.set_ticklabels([str(10 ** i)[:4] for i in yticks])
# c2.set_ticks(np.log10([2., 5,  10., 20., 50., 100.]),
#              labels=['2', '5', '10', '20', '50', '100'])

plt.savefig(path_name_res + '/VmaxJs_Js' + end_str + '.png',
            bbox_inches='tight')
plt.savefig(path_name_res + '/VmaxJs_Js' + end_str + '.pdf',
            bbox_inches='tight')
plt.show()
# ----------------------- Vmax - J (z==DistEarth) 2x4 -----------------

fig, axes = plt.subplots(nrows=2, ncols=4, sharex=True,  # sharey=True,
                         figsize=(15, 6))

plt.subplots_adjust(wspace=0, hspace=0)

legend_elements = [
    Line2D([0], [0], marker='o', color='k', label='Frag',
                          markerfacecolor='w', ls='',
                          markersize=8, mew=2.5),
    Line2D([0], [0], marker='P', color='w', label='Res',
                          markerfacecolor='k', markersize=12)]
vminn = 1e-3
vmaxx = np.log10(150)  # 10 ** 1.5
vminn = np.log10(perc_total(2, 5))
vmaxx = np.log10(perc_total(2, 95))
print(vminn, vmaxx)
norm = mcb.Normalize(vminn, vmaxx)

minn, maxx = minnmaxxS(0)

minnx = 0.1  # perc_total(3, 0)
maxxx = 200.  # perc_total(3, 100)

plt.subplot(241)

plt.ylim(minn - 0.2, maxx + 0.2)
plt.xlim(minnx, maxxx)
plt.xscale('log')

plt.text(1., 1.01, 'DMO', horizontalalignment='center',
 verticalalignment='bottom', transform=axes[0, 0].transAxes)

plt.scatter(datos_Js_frag_dmo[:, 3], np.log10(datos_Js_frag_dmo[:, 0]),
            c=np.log10(datos_Js_frag_dmo[:, 2]), lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

plt.ylabel(r'log$_{10}$ (J$_\mathrm{S}$ [GeV$^2$ cm$^{-5}$])', fontsize=16)

plt.legend(handles=legend_elements, handletextpad=0.2,
           handlelength=1, loc=2)

plt.subplot(242)

plt.tick_params('y', labelleft=False)
plt.ylim(minn - 0.2, maxx + 0.2)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.axvspan(187, 200, color='k')

plt.scatter(datos_Js_resi_dmo[:, 3], np.log10(datos_Js_resi_dmo[:, 0]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_resi_dmo[:, 2]))),
            label='Resilient')


plt.subplot(243)

plt.text(1., 1.01, 'Hydro', horizontalalignment='center',
 verticalalignment='bottom', transform=axes[0, 2].transAxes)

plt.ylim(minn - 0.2, maxx + 0.2)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.tick_params('y', labelleft=False)
plt.axvspan(0.1, 0.104, color='k')

plt.scatter(datos_Js_frag_hyd[:, 3], np.log10(datos_Js_frag_hyd[:, 0]),
            c=np.log10(datos_Js_frag_hyd[:, 2]), lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

plt.subplot(244)

plt.tick_params('y', labelleft=False)
plt.ylim(minn - 0.2, maxx + 0.2)
plt.xlim(minnx, maxxx)
plt.xscale('log')

plt.scatter(datos_Js_resi_hyd[:, 3], np.log10(datos_Js_resi_hyd[:, 0]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_resi_hyd[:, 2]))),
            label='Resilient')


minn, maxx = minnmaxx03(0)

plt.subplot(245)

plt.ylim(minn - 0.2, maxx + 0.2)
plt.xlim(minnx, maxxx)
plt.xscale('log')

plt.ylabel(r'log$_{10}$ (J$_{03}$ [GeV$^2$ cm$^{-5}$])', fontsize=16)
plt.xlabel(r'$V_\mathrm{max}$ $\left[\mathrm{km}\,\,\mathrm{s}^{-1}\right]$',
           size=20)
plt.scatter(datos_J03_frag_dmo[:, 3], np.log10(datos_J03_frag_dmo[:, 0]),
            c=np.log10(datos_J03_frag_dmo[:, 2]), lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

plt.subplot(246)

plt.tick_params('y', labelleft=False)
plt.ylim(minn - 0.2, maxx + 0.2)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.axvspan(187, 200, color='k')

plt.xlabel(r'$V_\mathrm{max}$ $\left[\mathrm{km}\,\,\mathrm{s}^{-1}\right]$',
           size=20)

plt.scatter(datos_J03_resi_dmo[:, 3], np.log10(datos_J03_resi_dmo[:, 0]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_resi_dmo[:, 2]))),
            label='Resilient')


plt.subplot(247)

plt.tick_params('y', labelleft=False)
plt.ylim(minn - 0.2, maxx + 0.2)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.axvspan(0.1, 0.104, color='k')

plt.xlabel(r'$V_\mathrm{max}$ $\left[\mathrm{km}\,\,\mathrm{s}^{-1}\right]$',
           size=20)

plt.scatter(datos_J03_frag_hyd[:, 3], np.log10(datos_J03_frag_hyd[:, 0]),
                 c=np.log10(datos_J03_frag_hyd[:, 2]), lw=0, marker='P', s=75,
                 cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

plt.subplot(248)

plt.tick_params('y', labelleft=False)
plt.ylim(minn - 0.2, maxx + 0.2)
plt.xlim(minnx, maxxx)
plt.xscale('log')

plt.xlabel(r'$V_\mathrm{max}$ $\left[\mathrm{km}\,\,\mathrm{s}^{-1}\right]$',
           size=20)

plt.scatter(datos_J03_resi_hyd[:, 3], np.log10(datos_J03_resi_hyd[:, 0]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_resi_hyd[:, 2]))),
            label='Resilient')

# plt.subplot(2, 4, 5)
# axes[1, 0].set_xticks([0.1, 1, 10], labels=('aaaa', '1', '10'))
    # a.set_xticklabels(['0.01', '0.1', '1', '10'])
axes[1, 0].set_xticks([1,4,5])
axes[1, 0].set_xticklabels([1,4,5], fontsize=12)

# for ii in range(1, 5):
#     plt.subplot(2, 4, ii)
#     plt.yticks((19, 20, 21, 22, 23), labels=('19', '', '21', '', '23'))

# plt.subplot(2, 4, 8)
axes[1, 3].set_xticks([0.1, 1, 10, 100], labels=('0.1', '1', '10', '100'))


cax, kw = colorbarr.make_axes([ax for ax in axes.flat])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
c2 = plt.colorbar(sm, cax=cax, extend='both')
# c2.set_label(r'log$_{10}$(D$_\mathrm{Earth}$ [kpc])', fontsize=20)
c2.set_label(r'D$_\mathrm{Earth}$ [kpc]', fontsize=20)
yticks = c2.get_ticks()
print(yticks)
# c2.set_ticklabels([str(10 ** i)[:4] for i in yticks])
c2.set_ticks(np.log10([2., 5,  10., 20., 50., 100.]),
             labels=['2', '5', '10', '20', '50', '100'])

plt.savefig(path_name_res + '/VmaxJs_full' + end_str + '.png',
            bbox_inches='tight')
plt.savefig(path_name_res + '/VmaxJs_full' + end_str + '.pdf',
            bbox_inches='tight')
# plt.show()
# ----------------------- Vmax - Ang size (z==DistEarth) 2x4 -----------------

fig, axes = plt.subplots(nrows=2, ncols=4, sharex=True,  # sharey=True,
                         figsize=(15, 6))

plt.subplots_adjust(wspace=0, hspace=0)

legend_elements = [
    Line2D([0], [0], marker='o', color='k', label='Frag',
                          markerfacecolor='w', ls='',
                          markersize=8, mew=2.5),
    Line2D([0], [0], marker='P', color='w', label='Res',
                          markerfacecolor='k', markersize=12)]

x_col = 3
y_col = 4
z_col = 2

x_label = r'$V_\mathrm{max}$ $\left[\mathrm{km}\,\,\mathrm{s}^{-1}\right]$'
y1_label = r'$\theta\,\,\left[\mathrm{deg}\right]$'
y2_label = r'$\theta\,\,\left[\mathrm{deg}\right]$'

# vminn = 1e-3
# vmaxx = np.log10(150)  # 10 ** 1.5
vminn = np.log10(perc_total(z_col, 5))
vmaxx = np.log10(perc_total(z_col, 95))
print(vminn, vmaxx)
norm = mcb.Normalize(vminn, vmaxx)

minns, maxxs = minnmaxxS(y_col)
minns = 10**minns * 0.8
maxxs = 10**maxxs * 1.2

minn03, maxx03 = minnmaxx03(y_col)
minn03 = 10**minn03 * 0.8
maxx03 = 10**maxx03 * 1.2

minnx = perc_total(x_col, 0) * 0.7
maxxx = perc_total(x_col, 100) * 1.3

plt.subplot(241)

plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.yscale('log')

plt.text(1., 1.01, 'DMO', horizontalalignment='center',
 verticalalignment='bottom', transform=axes[0, 0].transAxes)

plt.scatter(datos_Js_frag_dmo[:, x_col],
            (datos_Js_frag_dmo[:, y_col]),
            c=np.log10(datos_Js_frag_dmo[:, z_col]),
            lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

plt.ylabel(y1_label, fontsize=16)

plt.legend(handles=legend_elements, handletextpad=0.2,
           handlelength=1, loc=2)

plt.subplot(242)

plt.tick_params('y', labelleft=False)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.yscale('log')
plt.axvspan(0.935 * maxxx, maxxx, color='k')

plt.scatter(datos_Js_resi_dmo[:, x_col],
            (datos_Js_resi_dmo[:, y_col]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_resi_dmo[:, z_col]))),
            label='Resilient')


plt.subplot(243)

plt.text(1., 1.01, 'Hydro', horizontalalignment='center',
 verticalalignment='bottom', transform=axes[0, 2].transAxes)

plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.yscale('log')
plt.tick_params('y', labelleft=False)
plt.axvspan(minnx, minnx * 1.04, color='k')

plt.scatter(datos_Js_frag_hyd[:, x_col],
            (datos_Js_frag_hyd[:, y_col]),
            c=np.log10(datos_Js_frag_hyd[:, z_col]),
            lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

plt.subplot(244)

plt.tick_params('y', labelleft=False)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.yscale('log')

plt.scatter(datos_Js_resi_hyd[:, x_col],
            (datos_Js_resi_hyd[:, y_col]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_resi_hyd[:, z_col]))),
            label='Resilient')


plt.subplot(245)

plt.ylim(minn03, maxx03)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.yscale('log')

plt.ylabel(y2_label, fontsize=16)
plt.xlabel(x_label, size=20)

plt.scatter(datos_J03_frag_dmo[:, x_col],
            (datos_J03_frag_dmo[:, y_col]),
            c=np.log10(datos_J03_frag_dmo[:, z_col]),
            lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

plt.subplot(246)

plt.tick_params('y', labelleft=False)
plt.ylim(minn03, maxx03)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.yscale('log')
plt.axvspan(0.935 * maxxx, maxxx, color='k')

plt.xlabel(x_label, size=20)

plt.scatter(datos_J03_resi_dmo[:, x_col],
            (datos_J03_resi_dmo[:, y_col]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_resi_dmo[:, z_col]))),
            label='Resilient')


plt.subplot(247)

plt.tick_params('y', labelleft=False)
plt.ylim(minn03, maxx03)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.yscale('log')
plt.axvspan(minnx, minnx * 1.04, color='k')

plt.xlabel(x_label, size=20)

plt.scatter(datos_J03_frag_hyd[:, x_col],
            (datos_J03_frag_hyd[:, y_col]),
            c=np.log10(datos_J03_frag_hyd[:, z_col]),
            lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

plt.subplot(248)

plt.tick_params('y', labelleft=False)
plt.ylim(minn03, maxx03)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.yscale('log')

plt.xlabel(x_label, size=20)

plt.scatter(datos_J03_resi_hyd[:, x_col],
            (datos_J03_resi_hyd[:, y_col]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_resi_hyd[:, z_col]))),
            label='Resilient')

# plt.subplot(2, 4, 5)
# axes[1, 0].set_xticks([0.1, 1, 10], labels=('aaaa', '1', '10'))
    # a.set_xticklabels(['0.01', '0.1', '1', '10'])
# axes[1, 0].set_xticks([1,4,5])
# axes[1, 0].set_xticklabels([1,4,5], fontsize=12)

# for ii in range(1, 5):
#     plt.subplot(2, 4, ii)
#     plt.yticks((19, 20, 21, 22, 23), labels=('19', '', '21', '', '23'))

# plt.subplot(2, 4, 8)
# axes[1, 3].set_xticks([0.1, 1, 10, 100], labels=('0.1', '1', '10', '100'))


cax, kw = colorbarr.make_axes([ax for ax in axes.flat])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
c2 = plt.colorbar(sm, cax=cax, extend='both')
# c2.set_label(r'log$_{10}$(D$_\mathrm{Earth}$ [kpc])', fontsize=20)
c2.set_label(r'D$_\mathrm{Earth}$ [kpc]', fontsize=20)
yticks = c2.get_ticks()
print(yticks)
c2.set_ticklabels([str(10 ** i)[:4] for i in yticks])
# c2.set_ticks(np.log10([2., 5,  10., 20., 50., 100.]),
#              labels=['2', '5', '10', '20', '50', '100'])

plt.savefig(path_name_res + '/VmaxAng_full' + end_str + '.png',
            bbox_inches='tight')
plt.savefig(path_name_res + '/VmaxAng_full' + end_str + '.pdf',
            bbox_inches='tight')

# ----------------------- Ang size - J (z==Vmax) 2x4 -----------------

fig, axes = plt.subplots(nrows=2, ncols=4, sharex=True,  # sharey=True,
                         figsize=(15, 6))

plt.subplots_adjust(wspace=0, hspace=0)

legend_elements = [
    Line2D([0], [0], marker='o', color='k', label='Frag',
                          markerfacecolor='w', ls='',
                          markersize=8, mew=2.5),
    Line2D([0], [0], marker='P', color='w', label='Res',
                          markerfacecolor='k', markersize=12)]

x_col = 4
y_col = 0
z_col = 3

x_label = r'$\theta\,\,\left[\mathrm{deg}\right]$'
y1_label = r'log$_{10}$ (J$_\mathrm{S}$ [GeV$^2$ cm$^{-5}$])'
y2_label = r'log$_{10}$ (J$_{03}$ [GeV$^2$ cm$^{-5}$])'

# vminn = 1e-3
# vmaxx = np.log10(150)  # 10 ** 1.5
vminn = np.log10(perc_total(z_col, 5))
vmaxx = np.log10(perc_total(z_col, 95))
print(vminn, vmaxx)
norm = mcb.Normalize(vminn, vmaxx)

minns, maxxs = minnmaxxS(y_col)
minns -= 0.2
maxxs += 0.2

minn03, maxx03 = minnmaxx03(y_col)
minn03 -= 0.2
maxx03 += 0.2

minnx = perc_total(x_col, 0) * 0.7
maxxx = perc_total(x_col, 100) * 1.3

plt.subplot(241)

plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')

plt.text(1., 1.01, 'DMO', horizontalalignment='center',
 verticalalignment='bottom', transform=axes[0, 0].transAxes)

plt.scatter(datos_Js_frag_dmo[:, x_col],
            np.log10(datos_Js_frag_dmo[:, y_col]),
            c=np.log10(datos_Js_frag_dmo[:, z_col]),
            lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

plt.ylabel(y1_label, fontsize=16)

plt.legend(handles=legend_elements, handletextpad=0.2,
           handlelength=1, loc=2)

plt.subplot(242)

plt.tick_params('y', labelleft=False)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.axvspan(0.935 * maxxx, maxxx, color='k')

plt.scatter(datos_Js_resi_dmo[:, x_col],
            np.log10(datos_Js_resi_dmo[:, y_col]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_resi_dmo[:, z_col]))),
            label='Resilient')


plt.subplot(243)

plt.text(1., 1.01, 'Hydro', horizontalalignment='center',
 verticalalignment='bottom', transform=axes[0, 2].transAxes)

plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.tick_params('y', labelleft=False)
plt.axvspan(minnx, minnx * 1.04, color='k')

plt.scatter(datos_Js_frag_hyd[:, x_col],
            np.log10(datos_Js_frag_hyd[:, y_col]),
            c=np.log10(datos_Js_frag_hyd[:, z_col]),
            lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

plt.subplot(244)

plt.tick_params('y', labelleft=False)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')

plt.scatter(datos_Js_resi_hyd[:, x_col],
            np.log10(datos_Js_resi_hyd[:, y_col]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_resi_hyd[:, z_col]))),
            label='Resilient')


plt.subplot(245)

plt.ylim(minn03, maxx03)
plt.xlim(minnx, maxxx)
plt.xscale('log')

plt.ylabel(y2_label, fontsize=16)
plt.xlabel(x_label, size=20)

plt.scatter(datos_J03_frag_dmo[:, x_col],
            np.log10(datos_J03_frag_dmo[:, y_col]),
            c=np.log10(datos_J03_frag_dmo[:, z_col]),
            lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

plt.subplot(246)

plt.tick_params('y', labelleft=False)
plt.ylim(minn03, maxx03)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.axvspan(0.935 * maxxx, maxxx, color='k')

plt.xlabel(x_label, size=20)

plt.scatter(datos_J03_resi_dmo[:, x_col],
            np.log10(datos_J03_resi_dmo[:, y_col]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_resi_dmo[:, z_col]))),
            label='Resilient')


plt.subplot(247)

plt.tick_params('y', labelleft=False)
plt.ylim(minn03, maxx03)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.axvspan(minnx, minnx * 1.04, color='k')

plt.xlabel(x_label, size=20)

plt.scatter(datos_J03_frag_hyd[:, x_col],
            np.log10(datos_J03_frag_hyd[:, y_col]),
            c=np.log10(datos_J03_frag_hyd[:, z_col]),
            lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

plt.subplot(248)

plt.tick_params('y', labelleft=False)
plt.ylim(minn03, maxx03)
plt.xlim(minnx, maxxx)
plt.xscale('log')

plt.xlabel(x_label, size=20)

plt.scatter(datos_J03_resi_hyd[:, x_col],
            np.log10(datos_J03_resi_hyd[:, y_col]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_resi_hyd[:, z_col]))),
            label='Resilient')

# plt.subplot(2, 4, 5)
# axes[1, 0].set_xticks([0.1, 1, 10], labels=('aaaa', '1', '10'))
    # a.set_xticklabels(['0.01', '0.1', '1', '10'])
# axes[1, 0].set_xticks([1,4,5])
# axes[1, 0].set_xticklabels([1,4,5], fontsize=12)

for ii in range(1, 5):
    plt.subplot(2, 4, ii)
    plt.yticks((19, 20, 21, 22, 23), labels=('19', '', '21', '', '23'))

# plt.subplot(2, 4, 8)
# axes[1, 3].set_xticks([0.1, 1, 10, 100], labels=('0.1', '1', '10', '100'))


cax, kw = colorbarr.make_axes([ax for ax in axes.flat])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
c2 = plt.colorbar(sm, cax=cax, extend='both')
# c2.set_label(r'log$_{10}$(D$_\mathrm{Earth}$ [kpc])', fontsize=20)
c2.set_label(r'$V_\mathrm{max}$ [km s$^{-1}$]', fontsize=20)
yticks = c2.get_ticks()
print(yticks)
c2.set_ticklabels([str(10 ** i)[:4] for i in yticks])
# c2.set_ticks(np.log10([2., 5,  10., 20., 50., 100.]),
#              labels=['2', '5', '10', '20', '50', '100'])

plt.savefig(path_name_res + '/AngJs_full' + end_str + '.png',
            bbox_inches='tight')
plt.savefig(path_name_res + '/AngJs_full' + end_str + '.pdf',
            bbox_inches='tight')
# plt.show()
# ----------------------- Dgc - Vmax (z==Js) 1x2 --------------------------

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
plt.subplots_adjust(wspace=0, hspace=0)

vminn = 1e-3
vmaxx = np.log10(150)  # 10 ** 1.5
vminn = np.log10(perc_total(3, 5))
vmaxx = np.log10(perc_total(3, 95))
print(vminn, vmaxx)
norm = mcb.Normalize(vminn, vmaxx)

plt.subplot(121)

plt.text(x=0.04, y=22.5, s='DMO', size=18)
plt.xscale('log')
# plt.yscale('log')

minn, maxx = minnmaxxS(0)
plt.ylim((minn - 0.1), (maxx + 0.1))

minnx = perc_total(2, 0)
maxxx = perc_total(2, 100)
plt.xlim((minnx - 0.1), (maxxx + 0.1))

plt.scatter(datos_Js_frag_dmo[:, 2], np.log10(datos_Js_frag_dmo[:, 0]),
            c=np.log10(datos_Js_frag_dmo[:, 3]), lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

# plt.xlabel(r'$V_\mathrm{max}$ [km s$^{-1}$]', size=22)
plt.xlabel(r'D$_\mathrm{Earth}$ [kpc]', size=22)

plt.ylabel(r'log$_{10}$ (J$_\mathrm{S}$ [GeV$^2$ cm$^{-5}$])', fontsize=18)

# plt.legend(handles=legend_elements, handletextpad=0.2,
#            handlelength=1, loc=4, title=r'J$_\mathrm{S}$')

plt.subplot(122)
plt.text(x=0.04, y=22.5, s='Hydro', size=18)
plt.xscale('log')
# plt.yscale('log')


plt.tick_params('y', labelleft=False)

plt.scatter(datos_Js_frag_hyd[:, 2], np.log10(datos_Js_frag_hyd[:, 0]),
            c=np.log10(datos_Js_frag_hyd[:, 3]), lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

# plt.ylim((minn - 0.1), (maxx + 0.1))
# plt.legend(handles=legend_elements, handletextpad=0.2, handlelength=1,
#            loc=4, title=r'J$_\mathrm{S}$')


plt.xlabel(r'D$_\mathrm{Earth}$ [kpc]', size=22)
# plt.xlabel(r'$V_\mathrm{max}$ [km s$^{-1}$]', size=22)


cax, kw = colorbarr.make_axes([ax for ax in axes.flat])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
c2 = plt.colorbar(sm, cax=cax, extend='both')
# c2.set_label(r'D$_\mathrm{Earth}$ [kpc]', fontsize=20)
c2.set_label(r'$V_\mathrm{max}$ [km s$^{-1}$]', fontsize=20)

yticks = c2.get_ticks()
c2.set_ticklabels([str(10 ** i)[:4] for i in yticks])


plt.savefig(path_name_res + '/PresAle.png', bbox_inches='tight')
plt.savefig(path_name_res + '/PresAle.pdf', bbox_inches='tight')
# plt.show()

# ----------------------- Dgc - Vmax (z==Js) 2x2 --------------------------

fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True,  # sharey=True,
                         figsize=(12, 9))

plt.subplots_adjust(wspace=0, hspace=0)

legend_elements = [
    Line2D([0], [0], marker='o', color='k', label='Frag',
                          markerfacecolor='w', ls='',
                          markersize=8, mew=2.5),
    Line2D([0], [0], marker='P', color='w', label='Res',
                          markerfacecolor='k', markersize=12)]
cmap = cm.viridis
colormapp = 'viridis'
vminn = 1e-3
vmaxx = np.log10(150)  # 10 ** 1.5
vminn = np.log10(perc_total(0, 5))
vmaxx = np.log10(perc_total(0, 95))
print(vminn, vmaxx)
norm = mcb.Normalize(vminn, vmaxx)

lineas_verticales_dmo = [0.7803158130029454, 1.0924421382041234,
                         1.5294189934857727, 2.1411865908800816,
                         2.997661227232114, 4.196725718124959,
                         5.875416005374943, 8.22558240752492,
                         11.515815370534886, 16.12214151874884,
                         22.570998126248377, 31.599397376747724,
                         44.239156327446814, 61.93481885842554,
                         86.70874640179575]
lineas_verticales_hydro = [0.5610056681443348, 0.7854079354020688,
                           1.0995711095628962, 1.5393995533880545,
                           2.1551593747432762, 3.0172231246405867,
                           4.224112374496821, 5.913757324295549,
                           8.279260254013769, 11.590964355619276,
                           16.227350097866985, 22.718290137013778,
                           31.805606191819287, 44.527848668547,
                           62.3389881359658, 87.27458339035212]

# Dmin_dmo = [5.083123646, 10.3282713, 10.591064, 10.5910649,
#             13.1990738, 25.7525612, 25.7525612, 147.357412]
# Dmin_hydro = [14.3401868724, 14.3401868724, 14.3401868724,
#               18.999690481501997, 33.4699020, 75.58802, 82.87051104994488,
#               82.87051104]
# Dmin_dmo =
# Dmin_hydro =
# lineas_verticales_dmo = [8.2255, 11.5158, 16.122,  22.570]
# lineas_verticales_hydro =  [8.27926, 11.590, 16.227,  22.71]

lineas_verticales_dmo = [2.89725799, 3.766435389, 4.896366, 6.365275]
lineas_verticales_hydro = [2.70786, 3.5202, 4.5762955, 7.73393]

# input_data = funct_repop.read_config_file(
#     '/home/porrassa/Desktop/WIMPS_project/Physnet_outputs_repops/2024/'
#     '2024_resilient_const_SHVFnorm_to120_roche_changeSRD/5/input_data.yml'
# )
plt.subplot(221)

plt.title('DMO', size=18)
plt.xscale('log')
plt.yscale('log')

minn, maxx = minnmaxxS(3)
plt.ylim(10 ** (minn - 0.1), 10 ** (maxx + 0.1))

minnx = perc_total(1, 0)
maxxx = perc_total(1, 100)
plt.xlim(minnx * 0.8, maxxx * 1.2)

if plot_res:
    plt.scatter(datos_Js_resi_dmo[:, 1], datos_Js_resi_dmo[:, 3],
                c='none', lw=2, marker='o',
                edgecolors=cmap(norm(np.log10(datos_Js_resi_dmo[:, 0]))),
                label='Resilient')
if plot_frag:
    plt.scatter(datos_Js_frag_dmo[:, 1], (datos_Js_frag_dmo[:, 3]),
                c=np.log10(datos_Js_frag_dmo[:, 0]), lw=0, marker='P', s=75,
                cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

# for i in lineas_verticales_dmo:
#     plt.axhline(i, alpha=0.5, color='grey')
# for i in Dmin_dmo:
#     plt.axvline(i, alpha=0.5, color='grey')
plt.ylabel(r'$V_\mathrm{max}$ [km s$^{-1}$]', size=22)

plt.legend(handles=legend_elements, handletextpad=0.2,
           handlelength=1, loc=4, title=r'J$_\mathrm{S}$')
# plt.show()
plt.subplot(222)
plt.title('Hydro', size=18)
plt.xscale('log')
plt.yscale('log')

plt.tick_params('y', labelleft=False)

if plot_res:
    plt.scatter(datos_Js_resi_hyd[:, 1], (datos_Js_resi_hyd[:, 3]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_resi_hyd[:, 0]))),
            label='Resilient')

if plot_frag:
    plt.scatter(datos_Js_frag_hyd[:, 1], (datos_Js_frag_hyd[:, 3]),
            c=np.log10(datos_Js_frag_hyd[:, 0]), lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)


# for i in lineas_verticales_hydro:
#     plt.axhline(i, alpha=0.5, color='grey')
# for i in Dmin_hydro:
#     plt.axvline(i, alpha=0.5, color='grey')
plt.ylim(10 ** (minn - 0.1), 10 ** (maxx + 0.1))

plt.legend(handles=legend_elements, handletextpad=0.2, handlelength=1,
           loc=4, title=r'J$_\mathrm{S}$')

plt.subplot(223)
minn, maxx = minnmaxx03(3)
plt.ylim(10 ** (minn - 0.1), 10 ** (maxx + 0.1))

plt.yticks((19, 20))

plt.xscale('log')
plt.yscale('log')

if plot_res:
    plt.scatter(datos_J03_resi_dmo[:, 1], datos_J03_resi_dmo[:, 3],
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_resi_dmo[:, 0]))),
            label='Resilient')

if plot_frag:
    plt.scatter(datos_J03_frag_dmo[:, 1], (datos_J03_frag_dmo[:, 3]),
            c=np.log10(datos_J03_frag_dmo[:, 0]), lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)


# for i in lineas_verticales_dmo:
#     plt.axhline(i, alpha=0.5, color='grey')
# for i in Dmin_dmo:
#     plt.axvline(i, alpha=0.5, color='grey')
plt.ylabel(r'$V_\mathrm{max}$ [km s$^{-1}$]', size=22)
plt.xlabel(r'log$_{10}$ (D$_\mathrm{GC}$ [kpc])', fontsize=18)

plt.legend(handles=legend_elements, handletextpad=0.2, handlelength=1,
           loc=4, framealpha=0.9, title=r'J$_\mathrm{03}$')

plt.subplot(224)

plt.xscale('log')
plt.yscale('log')

if plot_res:
    plt.scatter(datos_J03_resi_hyd[:, 1], (datos_J03_resi_hyd[:, 3]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_resi_hyd[:, 0]))),
            label='Resilient')

if plot_frag:
    plt.scatter(datos_J03_frag_hyd[:, 1], (datos_J03_frag_hyd[:, 3]),
                 c=np.log10(datos_J03_frag_hyd[:, 0]), lw=0, marker='P', s=75,
                 cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)


# for i in lineas_verticales_hydro:
#     plt.axhline(i, alpha=0.5, color='grey')
# for i in Dmin_hydro:
#     plt.axvline(i, alpha=0.5, color='grey')
plt.ylim(10 ** (minn - 0.1), 10 ** (maxx + 0.1))

plt.tick_params('y', labelleft=False)
plt.xlabel(r'log$_{10}$ (D$_\mathrm{GC}$ [kpc])', fontsize=18)

plt.legend(handles=legend_elements, handletextpad=0.2, handlelength=1,
           loc=2, title=r'J$_\mathrm{03}$')

cax, kw = colorbarr.make_axes([ax for ax in axes.flat])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
c2 = plt.colorbar(sm, cax=cax, extend='both')
# c2.set_label(r'log$_{10}$(D$_\mathrm{Earth}$ [kpc])', fontsize=20)
c2.set_label(r'$log_{10}(J-factor)$', fontsize=20)
# yticks = c2.get_ticks()
# c2.set_ticklabels([str(10**i)[:4] for i in yticks])

# plt.show()
plt.savefig(path_name_res + '/DgcVmax' + end_str + '.png',
            bbox_inches='tight')
plt.savefig(path_name_res + '/DgcVmax' + end_str + '.pdf',
            bbox_inches='tight')
# plt.show()
# ----------------------- DEarth - J (z==Vmax) 2x4 -----------------

fig, axes = plt.subplots(nrows=2, ncols=4, sharex=True,  # sharey=True,
                         figsize=(15, 6))

plt.subplots_adjust(wspace=0, hspace=0)

legend_elements = [
    Line2D([0], [0], marker='o', color='k', label='Frag',
                          markerfacecolor='w', ls='',
                          markersize=8, mew=2.5),
    Line2D([0], [0], marker='P', color='w', label='Res',
                          markerfacecolor='k', markersize=12)]

x_col = 2
y_col = 0
z_col = 3

x_label = r'D$_\mathrm{Earth}$ [kpc]'
y1_label = r'log$_{10}$ (J$_\mathrm{S}$ [GeV$^2$ cm$^{-5}$])'
y2_label = r'log$_{10}$ (J$_{03}$ [GeV$^2$ cm$^{-5}$])'

# vminn = 1e-3
# vmaxx = np.log10(150)  # 10 ** 1.5
vminn = np.log10(perc_total(z_col, 5))
vmaxx = np.log10(perc_total(z_col, 95))
print(vminn, vmaxx)
norm = mcb.Normalize(vminn, vmaxx)

minns, maxxs = minnmaxxS(y_col)
minns -= 0.2
maxxs += 0.2

minn03, maxx03 = minnmaxx03(y_col)
minn03 -= 0.2
maxx03 += 0.2

minnx = perc_total(x_col, 0) * 0.7
maxxx = perc_total(x_col, 100) * 1.3

plt.subplot(241)

plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')

plt.text(1., 1.01, 'DMO', horizontalalignment='center',
 verticalalignment='bottom', transform=axes[0, 0].transAxes)

plt.scatter(datos_Js_frag_dmo[:, x_col],
            np.log10(datos_Js_frag_dmo[:, y_col]),
            c=np.log10(datos_Js_frag_dmo[:, z_col]),
            lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

plt.ylabel(y1_label, fontsize=16)

plt.legend(handles=legend_elements, handletextpad=0.2,
           handlelength=1, loc=2)

plt.subplot(242)

plt.tick_params('y', labelleft=False)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.axvspan(0.935 * maxxx, maxxx, color='k')

plt.scatter(datos_Js_resi_dmo[:, x_col],
            np.log10(datos_Js_resi_dmo[:, y_col]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_resi_dmo[:, z_col]))),
            label='Resilient')


plt.subplot(243)

plt.text(1., 1.01, 'Hydro', horizontalalignment='center',
 verticalalignment='bottom', transform=axes[0, 2].transAxes)

plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.tick_params('y', labelleft=False)
plt.axvspan(minnx, minnx * 1.04, color='k')

plt.scatter(datos_Js_frag_hyd[:, x_col],
            np.log10(datos_Js_frag_hyd[:, y_col]),
            c=np.log10(datos_Js_frag_hyd[:, z_col]),
            lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

plt.subplot(244)

plt.tick_params('y', labelleft=False)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')

plt.scatter(datos_Js_resi_hyd[:, x_col],
            np.log10(datos_Js_resi_hyd[:, y_col]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_resi_hyd[:, z_col]))),
            label='Resilient')


plt.subplot(245)

plt.ylim(minn03, maxx03)
plt.xlim(minnx, maxxx)
plt.xscale('log')

plt.ylabel(y2_label, fontsize=16)
plt.xlabel(x_label, size=20)

plt.scatter(datos_J03_frag_dmo[:, x_col],
            np.log10(datos_J03_frag_dmo[:, y_col]),
            c=np.log10(datos_J03_frag_dmo[:, z_col]),
            lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

plt.subplot(246)

plt.tick_params('y', labelleft=False)
plt.ylim(minn03, maxx03)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.axvspan(0.935 * maxxx, maxxx, color='k')

plt.xlabel(x_label, size=20)

plt.scatter(datos_J03_resi_dmo[:, x_col],
            np.log10(datos_J03_resi_dmo[:, y_col]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_resi_dmo[:, z_col]))),
            label='Resilient')


plt.subplot(247)

plt.tick_params('y', labelleft=False)
plt.ylim(minn03, maxx03)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.axvspan(minnx, minnx * 1.04, color='k')

plt.xlabel(x_label, size=20)

plt.scatter(datos_J03_frag_hyd[:, x_col],
            np.log10(datos_J03_frag_hyd[:, y_col]),
            c=np.log10(datos_J03_frag_hyd[:, z_col]),
            lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

plt.subplot(248)

plt.tick_params('y', labelleft=False)
plt.ylim(minn03, maxx03)
plt.xlim(minnx, maxxx)
plt.xscale('log')

plt.xlabel(x_label, size=20)

plt.scatter(datos_J03_resi_hyd[:, x_col],
            np.log10(datos_J03_resi_hyd[:, y_col]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_resi_hyd[:, z_col]))),
            label='Resilient')

# plt.subplot(2, 4, 5)
# axes[1, 0].set_xticks([0.1, 1, 10], labels=('aaaa', '1', '10'))
    # a.set_xticklabels(['0.01', '0.1', '1', '10'])
# axes[1, 0].set_xticks([1,4,5])
# axes[1, 0].set_xticklabels([1,4,5], fontsize=12)

for ii in range(1, 5):
    plt.subplot(2, 4, ii)
    plt.yticks((19, 20, 21, 22, 23), labels=('19', '', '21', '', '23'))

# plt.subplot(2, 4, 8)
# axes[1, 3].set_xticks([0.1, 1, 10, 100], labels=('0.1', '1', '10', '100'))


cax, kw = colorbarr.make_axes([ax for ax in axes.flat])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
c2 = plt.colorbar(sm, cax=cax, extend='both')
# c2.set_label(r'log$_{10}$(D$_\mathrm{Earth}$ [kpc])', fontsize=20)
c2.set_label(r'$V_\mathrm{max}$ [km s$^{-1}$]', fontsize=20)
yticks = c2.get_ticks()
print(yticks)
c2.set_ticklabels([str(10 ** i)[:4] for i in yticks])
# c2.set_ticks(np.log10([2., 5,  10., 20., 50., 100.]),
#              labels=['2', '5', '10', '20', '50', '100'])

plt.savefig(path_name_res + '/DEarthJs_full' + end_str + '.png',
            bbox_inches='tight')
plt.savefig(path_name_res + '/DEarthJs_full' + end_str + '.pdf',
            bbox_inches='tight')
plt.show()
# ------------------------ DEarthJs -------------------------------------------
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True,  # sharey=True,
                         figsize=(12, 9))

plt.subplots_adjust(wspace=0, hspace=0)

legend_elements = [
    Line2D([0], [0], marker='o', color='k', label='Frag',
                          markerfacecolor='w', ls='',
                          markersize=8, mew=2.5),
    Line2D([0], [0], marker='P', color='w', label='Res',
                          markerfacecolor='k', markersize=12)]

cmap = cm.viridis
colormapp = 'viridis'
vminn = np.log10(perc_total(3, 5))  # np.log10(3)
vmaxx = np.log10(perc_total(3, 95))  # np.log10(100)
norm = mcb.Normalize(vminn, vmaxx)
print(10 ** vminn, 10 ** vmaxx)

plt.subplot(221)
minn, maxx = minnmaxxS(0)
plt.ylim(minn - 0.1, maxx + 0.1)

minnx = perc_total(2, 0)
maxxx = perc_total(2, 100)
plt.xlim(minnx * 0.8, maxxx * 1.2)

plt.title('DMO', size=18)
plt.xscale('log')

if plot_res:
    plt.scatter(datos_Js_resi_dmo[:, 2], np.log10(datos_Js_resi_dmo[:, 0]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_resi_dmo[:, 3]))),
            label='Resilient')

if plot_frag:
    plt.scatter(datos_Js_frag_dmo[:, 2], np.log10(datos_Js_frag_dmo[:, 0]),
            c=np.log10(datos_Js_frag_dmo[:, 3]),
            lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

plt.ylabel(r'log$_{10}$ (J$_\mathrm{S}$ [GeV$^2$ cm$^{-5}$])', fontsize=18)
plt.legend(handles=legend_elements, handletextpad=0.2,
           handlelength=1, loc=2, title=r'J$_\mathrm{S}$')

plt.subplot(222)
plt.title('Hydro', size=18)
plt.xscale('log')

if plot_res:
    plt.scatter(datos_Js_resi_hyd[:, 2], np.log10(datos_Js_resi_hyd[:, 0]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_resi_hyd[:, 3]))),
            label='Resilient')

if plot_frag:
    plt.scatter(datos_Js_frag_hyd[:, 2], np.log10(datos_Js_frag_hyd[:, 0]),
            c=np.log10(datos_Js_frag_hyd[:, 3]),
            lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

plt.tick_params('y', labelleft=False)
plt.ylim(minn - 0.1, maxx + 0.1)

plt.legend(handles=legend_elements, handletextpad=0.2, handlelength=1,
           loc=2, title=r'J$_\mathrm{S}$')

plt.subplot(223)

if plot_res:
    plt.scatter(datos_J03_resi_dmo[:, 2], np.log10(datos_J03_resi_dmo[:, 0]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_resi_dmo[:, 3]))),
            label='Resilient')

if plot_frag:
    plt.scatter(datos_J03_frag_dmo[:, 2], np.log10(datos_J03_frag_dmo[:, 0]),
            c=np.log10(datos_J03_frag_dmo[:, 3]),
            lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

minn, maxx = minnmaxx03(0)
plt.ylim(minn - 0.1, maxx + 0.1)

print('minn - 0.1, maxx + 0.1', minn - 0.1, maxx + 0.1)
plt.xscale('log')

plt.xlabel(r'log$_{10}$ (D$_\mathrm{Earth}$ [kpc])', fontsize=18)
plt.ylabel(r'log$_{10}$ (J$_{03}$ [GeV$^2$ cm$^{-5}$])', fontsize=18)
plt.legend(handles=legend_elements, handletextpad=0.2, handlelength=1,
           loc=2, framealpha=0.9, title=r'J$_\mathrm{03}$')

plt.subplot(224)

if plot_res:
    plt.scatter(datos_J03_resi_hyd[:, 2], np.log10(datos_J03_resi_hyd[:, 0]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_resi_hyd[:, 3]))),
            label='Resilient')

if plot_frag:
    plt.scatter(datos_J03_frag_hyd[:, 2], np.log10(datos_J03_frag_hyd[:, 0]),
                 c=np.log10(datos_J03_frag_hyd[:, 3]),
                 lw=0, marker='P', s=75,
                 cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)


plt.xlabel(r'log$_{10}$ (D$_\mathrm{Earth}$ [kpc])', fontsize=18)
plt.legend(handles=legend_elements, handletextpad=0.2, handlelength=1,
           loc=2, title=r'J$_\mathrm{03}$')

plt.tick_params('y', labelleft=False)
plt.ylim(minn - 0.1, maxx + 0.1)
print('minn - 0.1, maxx + 0.1', minn - 0.1, maxx + 0.1)

plt.xscale('log')

cax, kw = colorbarr.make_axes([ax for ax in axes.flat])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
c2 = plt.colorbar(sm, cax=cax, extend='both')
# c2.set_label(r'log$_{10}$(V$_\mathrm{max}$ [km/s])', fontsize=20)
c2.set_label(r'V$_\mathrm{max}$ [km/s])', fontsize=20)
yticks = c2.get_ticks()
c2.set_ticklabels([str(10 ** i)[:4] for i in yticks])

plt.savefig(path_name_res + '/DEarthJs' + end_str + '.png',
            bbox_inches='tight')
plt.savefig(path_name_res + '/DEarthJs' + end_str + '.pdf',
            bbox_inches='tight')
# plt.show()
# ---------------- Dgc_Dearth -------------------------------------------------
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True,  # sharey=True,
                         figsize=(12, 9))

legend_elements = [
    Line2D([0], [0], marker='o', color='k', label='Frag',
                          markerfacecolor='w', ls='',
                          markersize=8, mew=2.5),
    Line2D([0], [0], marker='P', color='w', label='Res',
                          markerfacecolor='k', markersize=12)]
vminn = 19
vmaxx = 23
vminn = np.log10(perc_total(0, 5))
vmaxx = np.log10(perc_total(0, 95))
norm = mcb.Normalize(vminn, vmaxx)

plt.subplot(221)

minn, maxx = minnmaxxS(2)
plt.ylim(minn - 0.1, maxx + 0.1)

minnx = perc_total(1, 0)
maxxx = perc_total(1, 100)
plt.xlim(minnx * 0.8, maxxx * 1.2)

plt.title('DMO', size=18)

if plot_res:
    plt.scatter(datos_Js_resi_dmo[:, 1], np.log10(datos_Js_resi_dmo[:, 3]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_resi_dmo[:, 0]))),
            label='Resilient')

if plot_frag:
    plt.scatter(datos_Js_frag_dmo[:, 1], np.log10(datos_Js_frag_dmo[:, 3]),
            c=np.log10(datos_Js_frag_dmo[:, 0]),
            lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

plt.ylabel(r'log$_{10}$ (J$_\mathrm{S}$ [GeV$^2$ cm$^{-5}$])', fontsize=18)

plt.legend(handles=legend_elements, handletextpad=0.2,
           handlelength=1, loc=2, title=r'J$_\mathrm{S}$')

plt.xscale('log')
# plt.yscale('log')

plt.subplot(222)
plt.title('Hydro', size=18)

if plot_res:
    plt.scatter(datos_Js_resi_hyd[:, 2], np.log10(datos_Js_resi_hyd[:, 0]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_resi_hyd[:, 3]))),
            label='Resilient')

if plot_frag:
    plt.scatter(datos_Js_frag_hyd[:, 2], np.log10(datos_Js_frag_hyd[:, 0]),
            c=np.log10(datos_Js_frag_hyd[:, 3]),
            lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

plt.xscale('log')

# plt.yticks((18, 19, 20, 21, 22), labels=('', '', '', '', ''))
plt.ylim(minn - 0.1, maxx + 0.1)

plt.legend(handles=legend_elements, handletextpad=0.2, handlelength=1,
           loc=2, title=r'J$_\mathrm{S}$')

plt.subplot(223)
minn, maxx = minnmaxx03(0)
plt.ylim(minn - 0.1, maxx + 0.1)
# plt.yticks((19, 20))

plt.xscale('log')

plt.scatter(datos_J03_resi_dmo[:, 2], np.log10(datos_J03_resi_dmo[:, 0]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_resi_dmo[:, 3]))),
            label='Resilient')

if plot_frag:
    plt.scatter(datos_J03_frag_dmo[:, 2], np.log10(datos_J03_frag_dmo[:, 0]),
            c=np.log10(datos_J03_frag_dmo[:, 3]),
            lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)


plt.xlabel(r'log$_{10}$ (D$_\mathrm{Earth}$ [kpc])', fontsize=18)
plt.ylabel(r'log$_{10}$ (J$_{03}$ [GeV$^2$ cm$^{-5}$])', fontsize=18)
plt.legend(handles=legend_elements, handletextpad=0.2, handlelength=1,
           loc=2, framealpha=0.9, title=r'J$_\mathrm{03}$')

plt.subplot(224)

plt.ylim(minn - 0.1, maxx + 0.1)

if plot_res:
    plt.scatter(datos_J03_resi_hyd[:, 2], np.log10(datos_J03_resi_hyd[:, 0]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_resi_hyd[:, 3]))),
            label='Resilient')

if plot_frag:
    plt.scatter(datos_J03_frag_hyd[:, 2], np.log10(datos_J03_frag_hyd[:, 0]),
            c=np.log10(datos_J03_frag_hyd[:, 3]),
            lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

plt.xscale('log')
# plt.yticks((19, 20), labels=('', ''))

plt.xlabel(r'log$_{10}$ (D$_\mathrm{Earth}$ [kpc])', fontsize=18)
plt.legend(handles=legend_elements, handletextpad=0.2, handlelength=1,
           loc=2, title=r'J$_\mathrm{03}$')

cax, kw = colorbarr.make_axes([ax for ax in axes.flat])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
c2 = plt.colorbar(sm, cax=cax, extend='both')
# c2.set_label(r'log$_{10}$(V$_\mathrm{max}$ [km/s])', fontsize=20)
c2.set_label(r'V$_\mathrm{max}$ [km/s]', fontsize=20)
yticks = c2.get_ticks()
c2.set_ticklabels([str(10 ** i)[:4] for i in yticks])

plt.savefig(path_name_res + '/Dgc_Dearth' + end_str + '.png',
            bbox_inches='tight')
plt.savefig(path_name_res + '/Dgc_Dearth' + end_str + '.pdf',
            bbox_inches='tight')

# ------------------------ DgcJs ----------------------------------------------
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True,  # sharey=True,
                         figsize=(12, 9))

plt.subplots_adjust(wspace=0, hspace=0)

legend_elements = [
    Line2D([0], [0], marker='o', color='k', label='Frag',
                          markerfacecolor='w', ls='',
                          markersize=8, mew=2.5),
    Line2D([0], [0], marker='P', color='w', label='Res',
                          markerfacecolor='k', markersize=12)]
vminn = 1.
vmaxx = np.log10(100)  # 10 ** 1.5
vminn = np.log10(perc_total(3, 5))
vmaxx = np.log10(perc_total(3, 95))
print('Vmax perc', 10 ** vminn, 10 ** vmaxx)
norm = mcb.Normalize(vminn, vmaxx)

plt.subplot(221)

minn, maxx = minnmaxxS(0)
plt.ylim(minn - 0.1, maxx + 0.1)

minnx = perc_total(1, 0)
maxxx = perc_total(1, 100)
plt.xlim(minnx * 0.8, maxxx * 1.2)

plt.title('DMO', size=18)
plt.xscale('log')

if plot_res:
    plt.scatter(datos_Js_resi_dmo[:, 1], np.log10(datos_Js_resi_dmo[:, 0]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_resi_dmo[:, 3]))),
            label='Resilient')

if plot_frag:
    plt.scatter(datos_Js_frag_dmo[:, 1], np.log10(datos_Js_frag_dmo[:, 0]),
            c=np.log10(datos_Js_frag_dmo[:, 3]),
            lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

plt.ylabel(r'log$_{10}$ (J$_\mathrm{S}$ [GeV$^2$ cm$^{-5}$])', fontsize=18)
plt.legend(handles=legend_elements, handletextpad=0.2,
           handlelength=1, loc=2, title=r'J$_\mathrm{S}$')

plt.subplot(222)
plt.title('Hydro', size=18)

if plot_res:
    plt.scatter(datos_Js_resi_hyd[:, 1], np.log10(datos_Js_resi_hyd[:, 0]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_resi_hyd[:, 3]))),
            label='Resilient')

if plot_frag:
    plt.scatter(datos_Js_frag_hyd[:, 1], np.log10(datos_Js_frag_hyd[:, 0]),
c=np.log10(datos_Js_frag_hyd[:, 3]),
lw=0, marker='P', s=75,
cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

plt.xscale('log')

ytickss = plt.yticks()
plt.yticks(ytickss[0], ['' for i in range(len(ytickss[0]))])
plt.ylim(minn - 0.1, maxx + 0.1)

plt.legend(handles=legend_elements, handletextpad=0.2, handlelength=1,
           loc=2, title=r'J$_\mathrm{S}$')

plt.subplot(223)

if plot_res:
    plt.scatter(datos_J03_resi_dmo[:, 1], np.log10(datos_J03_resi_dmo[:, 0]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_resi_dmo[:, 3]))),
            label='Resilient')
if plot_frag:
    plt.scatter(datos_J03_frag_dmo[:, 1], np.log10(datos_J03_frag_dmo[:, 0]),
            c=np.log10(datos_J03_frag_dmo[:, 3]),
            lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

minn, maxx = minnmaxx03(0)
plt.ylim(minn - 0.1, maxx + 0.1)
# plt.yticks((19, 20))

plt.xscale('log')

plt.xlabel(r'log$_{10}$ (D$_\mathrm{GC}$ [kpc])', fontsize=18)
plt.ylabel(r'log$_{10}$ (J$_{03}$ [GeV$^2$ cm$^{-5}$])', fontsize=18)

plt.legend(handles=legend_elements, handletextpad=0.2, handlelength=1,
           loc=2, framealpha=0.9, title=r'J$_\mathrm{03}$')

plt.subplot(224)

plt.xscale('log')

if plot_res:
    plt.scatter(datos_J03_resi_hyd[:, 1], np.log10(datos_J03_resi_hyd[:, 0]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_resi_hyd[:, 3]))),
            label='Resilient')

if plot_frag:
    plt.scatter(datos_J03_frag_hyd[:, 1], np.log10(datos_J03_frag_hyd[:, 0]),
                 c=np.log10(datos_J03_frag_hyd[:, 3]),
                 lw=0, marker='P', s=75,
                 cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

ytickss = plt.yticks()
plt.yticks(ytickss[0], ['' for i in range(len(ytickss[0]))])
plt.ylim(minn - 0.1, maxx + 0.1)

plt.xlabel(r'log$_{10}$ (D$_\mathrm{GC}$ [kpc])', fontsize=18)

plt.legend(handles=legend_elements, handletextpad=0.2, handlelength=1,
           loc=2, title=r'J$_\mathrm{03}$')

cax, kw = colorbarr.make_axes([ax for ax in axes.flat])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
c2 = plt.colorbar(sm, cax=cax, extend='both')
# c2.set_label(r'log$_{10}$(V$_\mathrm{max}$ [km/s])', fontsize=20)
c2.set_label(r'V$_\mathrm{max}$ [km/s]', fontsize=20)
yticks = c2.get_ticks()
c2.set_ticklabels([str(10 ** i)[:4] for i in yticks])

plt.savefig(path_name_res + '/DgcJs' + end_str + '.png',
            bbox_inches='tight')
plt.savefig(path_name_res + '/DgcJs' + end_str + '.pdf',
            bbox_inches='tight')

# ---------------------- VmaxDEarth -------------------------------------------
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True,  # sharey=True,
                         figsize=(12, 9))

plt.subplots_adjust(wspace=0, hspace=0)

legend_elements = [
    Line2D([0], [0], marker='o', color='k', label='Frag',
                          markerfacecolor='w', ls='',
                          markersize=8, mew=2.5),
    Line2D([0], [0], marker='P', color='w', label='Res',
                          markerfacecolor='k', markersize=12)]
vminn = 0.1
vmaxx = 50
vminn = np.log10(perc_total(4, 5))
vmaxx = np.log10(perc_total(4, 95))
print(vminn, vmaxx)
norm = mcb.Normalize(vminn, vmaxx)

print(10 ** minnmaxxS(4)[0], 10 ** minnmaxxS(4)[1],
      10 ** minnmaxx03(4)[0], 10 ** minnmaxx03(4)[1])

plt.subplot(221)

minn, maxx = minnmaxxS(2)
plt.ylim(minn - 0.1, maxx + 0.1)

minnx = perc_total(3, 0)
maxxx = perc_total(3, 100)
plt.xlim(minnx * 0.8, maxxx * 1.2)

plt.title('DMO', size=18)
plt.xscale('log')

if plot_res:
    plt.scatter(datos_Js_resi_dmo[:, 3], np.log10(datos_Js_resi_dmo[:, 2]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_resi_dmo[:, 4]))),
            label='Resilient')

if plot_frag:
    plt.scatter(datos_Js_frag_dmo[:, 3], np.log10(datos_Js_frag_dmo[:, 2]),
            c=np.log10(datos_Js_frag_dmo[:, 4]),
            lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

plt.ylabel(r'log$_{10}$ (D$_\mathrm{Earth}$ [kpc])', fontsize=18)

plt.legend(handles=legend_elements, handletextpad=0.2,
           handlelength=1, loc=2, title=r'J$_\mathrm{S}$')

plt.subplot(222)
plt.title('Hydro', size=18)
plt.xscale('log')

if plot_res:
    plt.scatter(datos_Js_resi_hyd[:, 3], np.log10(datos_Js_resi_hyd[:, 2]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_resi_hyd[:, 4]))),
            label='Resilient')

if plot_frag:
    plt.scatter(datos_Js_frag_hyd[:, 3], np.log10(datos_Js_frag_hyd[:, 2]),
            c=np.log10(datos_Js_frag_hyd[:, 4]),
            lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

plt.legend(handles=legend_elements, handletextpad=0.2, handlelength=1,
           loc=2, title=r'J$_\mathrm{S}$')

ytickss = plt.yticks()
plt.yticks(ytickss[0], ['' for i in range(len(ytickss[0]))])
plt.ylim(minn - 0.1, maxx + 0.1)

plt.subplot(223)
minn, maxx = minnmaxx03(2)
plt.ylim(minn - 0.1, maxx + 0.1)
# plt.yticks((19, 20))

plt.xscale('log')

if plot_res:
    plt.scatter(datos_J03_resi_dmo[:, 3], np.log10(datos_J03_resi_dmo[:, 2]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_resi_dmo[:, 4]))),
            label='Resilient')

if plot_frag:
    plt.scatter(datos_J03_frag_dmo[:, 3], np.log10(datos_J03_frag_dmo[:, 2]),
            c=np.log10(datos_J03_frag_dmo[:, 4]),
            lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

plt.ylabel(r'log$_{10}$ (D$_\mathrm{Earth}$ [kpc])', fontsize=18)
plt.xlabel(r'$V_\mathrm{max}$ [km s$^{-1}$]', size=22)

plt.legend(handles=legend_elements, handletextpad=0.2, handlelength=1,
           loc=2, framealpha=0.9, title=r'J$_\mathrm{03}$')

plt.subplot(224)

plt.xscale('log')

if plot_res:
    plt.scatter(datos_J03_resi_hyd[:, 3], np.log10(datos_J03_resi_hyd[:, 2]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_resi_hyd[:, 4]))),
            label='Resilient')
if plot_frag:
    plt.scatter(datos_J03_frag_hyd[:, 3], np.log10(datos_J03_frag_hyd[:, 2]),
            c=np.log10(datos_J03_frag_hyd[:, 4]),
            lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

ytickss = plt.yticks()
plt.yticks(ytickss[0], ['' for i in range(len(ytickss[0]))])
plt.ylim(minn - 0.1, maxx + 0.1)
plt.xlabel(r'$V_\mathrm{max}$ [km s$^{-1}$]', size=22)

plt.legend(handles=legend_elements, handletextpad=0.2, handlelength=1,
           loc=2, title=r'J$_\mathrm{03}$')

cax, kw = colorbarr.make_axes([ax for ax in axes.flat])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
c2 = plt.colorbar(sm, cax=cax, extend='both')
c2.set_label(r'Angular size', fontsize=20)
yticks = c2.get_ticks()
c2.set_ticklabels([str(10 ** i)[:4] for i in yticks])

plt.savefig(path_name_res + '/VmaxDEarth' + end_str + '.png',
            bbox_inches='tight')
plt.savefig(path_name_res + '/VmaxDEarth' + end_str + '.pdf',
            bbox_inches='tight')

# -------------------- J_hist -------------------------------------------------
fig, _ = plt.subplots(2, 2, figsize=(12, 9))

plt.subplots_adjust(wspace=0, hspace=0)

minS, maxS = minnmaxxS(0)
min03, max03 = minnmaxx03(0)
minn = np.min((minS, min03))
maxx = np.max((maxS, max03))

bines = np.linspace(minn, maxx, 40)
# bines = np.linspace(19, 23, 40)

ax1 = plt.subplot(221)

plt.title('DMO', size=18)
print(np.shape(datos_Js_frag_dmo))
plt.hist(np.log10(datos_Js_frag_dmo[:, 0]), log=False,
         label=r'Frag', color='teal', alpha=0.6,
         bins=bines)

plt.hist(np.log10(datos_Js_resi_dmo[:, 0]), log=False,
         label=r'Res', color='k', alpha=0.5,
         bins=bines)

Js95_frag_dmo = np.log10(np.percentile(datos_Js_frag_dmo[:, 0], 5))
Js95_resi_dmo = np.log10(np.percentile(datos_Js_resi_dmo[:, 0], 5))
print('Js, DMO')
print(Js95_resi_dmo, Js95_frag_dmo, Js95_resi_dmo - Js95_frag_dmo)

plt.axvline(Js95_frag_dmo, color='teal')  # , alpha=0.6)
plt.axvline(Js95_resi_dmo, color='k')  # , alpha=0.5)

plt.xlim(minn, maxx)
plt.ylim(bottom=0.9, top=120)

plt.yscale('log')

ax1.tick_params(labelbottom=False)

# plt.annotate(r'J$_S$ 95%', (Js95_frag_dmo, 20), rotation=90, color='k')
# plt.annotate(r'J$_{03}$ 95%', (J0395_frag_dmo, 20),
# rotation=90, color='g', horizontalalignment='right')
plt.legend(title=r'J$_\mathrm{S}$')

ax2 = plt.subplot(222, sharex=ax1, sharey=ax1)
plt.title('Hydro', size=18)
plt.yscale('log')
plt.hist(np.log10(datos_Js_frag_hyd[:, 0]), log=False,
         label=r'Frag', color='yellowgreen', alpha=0.6,
         bins=bines)
plt.hist(np.log10(datos_Js_resi_hyd[:, 0]), log=False,
         label=r'Res', color=darkgreen, alpha=0.6,
         bins=bines)

Js95_frag_hyd = np.log10(np.percentile(datos_Js_frag_hyd[:, 0], 5))
Js95_resi_hyd = np.log10(np.percentile(datos_Js_resi_hyd[:, 0], 5))
print('Js, Hydro')
print(Js95_resi_hyd, Js95_frag_hyd, Js95_resi_hyd - Js95_frag_hyd)

plt.axvline(Js95_frag_hyd, color='yellowgreen')  # , alpha=0.6)
plt.axvline(Js95_resi_hyd, color=darkgreen)  # , alpha=0.6)

# plt.annotate(r'J$_S$ 95%', (Js95_frag_hyd, 20), rotation=90, color='k')
# plt.annotate(r'J$_{03}$ 95%', (Js95_resi_hyd, 20),
# rotation=90, color='g', horizontalalignment='right')

ax2.tick_params(labelleft=False)

plt.legend(title=r'J$_\mathrm{S}$')

plt.subplot(223, sharex=ax1, sharey=ax1)

plt.hist(np.log10(datos_J03_frag_dmo[:, 0]), log=False,
         label=r'Frag', color='teal', alpha=0.6,
         bins=bines)
plt.hist(np.log10(datos_J03_resi_dmo[:, 0]), log=False,
         label=r'Res', color='k', alpha=0.5,
         bins=bines)

J0395_frag_dmo = np.log10(np.percentile(datos_J03_frag_dmo[:, 0], 5))
J0395_resi_dmo = np.log10(np.percentile(datos_J03_resi_dmo[:, 0], 5))
print('J03, DMO')
print(J0395_frag_dmo, J0395_resi_dmo, J0395_frag_dmo - J0395_resi_dmo)
plt.yscale('log')

plt.axvline(J0395_frag_dmo, color='teal')  # , alpha=0.6)
plt.axvline(J0395_resi_dmo, color='k')  # , alpha=0.5)

# plt.annotate(r'J$_S$ 95%', (Js95_resi_dmo, 20), rotation=90, color='k')
# plt.annotate(r'J$_{03}$ 95%', (J0395_resi_dmo, 20),
# rotation=90, color='g', horizontalalignment='right')

plt.xlabel(r'log$_{10}$ (J$_\mathrm{factor}$ [GeV$^2$ cm$^{-5}$])',
           fontsize=20)
plt.legend(title=r'J$_{03}$')

ax4 = plt.subplot(224, sharex=ax1, sharey=ax1)

plt.hist(np.log10(datos_J03_frag_hyd[:, 0]), log=False,
         label=r'Frag', color='yellowgreen', alpha=0.6,
         bins=bines)
plt.hist(np.log10(datos_J03_resi_hyd[:, 0]), log=False,
         label=r'Res', color=darkgreen, alpha=0.6,
         bins=bines)

J0395_frag_hyd = np.log10(np.percentile(datos_J03_frag_hyd[:, 0], 5))
J0395_resi_hyd = np.log10(np.percentile(datos_J03_resi_hyd[:, 0], 5))
print('J03, Hydro')
print(J0395_resi_hyd, J0395_frag_hyd, J0395_resi_hyd - J0395_frag_hyd)

plt.axvline(J0395_frag_hyd, color='yellowgreen')  # , alpha=0.6)
plt.axvline(J0395_resi_hyd, color=darkgreen)  # , alpha=0.6)

# plt.annotate(r'J$_S$ 95%', (Js95_resi_hyd, 20), rotation=90, color='k')
# plt.annotate(r'J$_{03}$ 95%', (J0395_resi_hyd, 20),
# rotation=90, color='g', horizontalalignment='right')

plt.yscale('log')
plt.xlabel(r'log$_{10}$ (J$_\mathrm{factor}$ [GeV$^2$ cm$^{-5}$])',
           fontsize=20)
plt.legend(title=r'J$_{03}$')

fig.text(0.06, 0.5, 'Number of repops', ha='center',
         va='center', rotation='vertical')

ax4.tick_params(labelleft=False)

plt.savefig(path_name_res + '/J_hist.png', bbox_inches='tight')
plt.savefig(path_name_res + '/J_hist.pdf', bbox_inches='tight')

# -------------- Vmax_hist ----------------------------------------------------

fig, _ = plt.subplots(2, 2, figsize=(12, 9))

plt.subplots_adjust(wspace=0, hspace=0)

minS, maxS = minnmaxxS(3)
min03, max03 = minnmaxx03(3)
minn = 10 ** np.min((minS, min03))
maxx = 10 ** np.max((maxS, max03))

bines = np.linspace(minn, maxx, 20)

ax1 = plt.subplot(221)

plt.title('DMO', size=18)
plt.hist(datos_Js_frag_dmo[:, 3], log=False,
         label=r'Frag', color='teal', alpha=0.6,
         bins=bines)
plt.hist(datos_Js_resi_dmo[:, 3], log=False,
         label=r'Res', color='k', alpha=0.5,
         bins=bines)

plt.xlim(minn, maxx)
plt.ylim(bottom=0.9, top=240)

plt.yscale('log')

ax1.tick_params(labelbottom=False)
plt.legend(title=r'J$_\mathrm{S}$')

ax2 = plt.subplot(222, sharex=ax1, sharey=ax1)
plt.title('Hydro', size=18)
plt.yscale('log')
plt.hist(datos_Js_frag_hyd[:, 3], log=False,
         label=r'Frag', color='yellowgreen', alpha=0.6,
         bins=bines)
plt.hist(datos_Js_resi_hyd[:, 3], log=False,
         label=r'Res', color=darkgreen, alpha=0.6,
         bins=bines)

ax2.tick_params(labelleft=False)

plt.legend(title=r'J$_\mathrm{S}$')

plt.subplot(223, sharex=ax1, sharey=ax1)
plt.hist(datos_J03_frag_dmo[:, 3], log=False,
         label=r'Frag', color='teal', alpha=0.6,
         bins=bines)
plt.hist(datos_J03_resi_dmo[:, 3], log=False,
         label=r'Res', color='k', alpha=0.5,
         bins=bines)

plt.xlabel(r'$V_\mathrm{max}$ [km s$^{-1}$]', size=20)
plt.legend(title=r'J$_{03}$')

ax4 = plt.subplot(224, sharex=ax1, sharey=ax1)
plt.hist(datos_J03_frag_hyd[:, 3], log=False,
         label=r'Frag', color='yellowgreen', alpha=0.6,
         bins=bines)
plt.hist(datos_J03_resi_hyd[:, 3], log=False,
         label=r'Res', color=darkgreen, alpha=0.6,
         bins=bines)

plt.yscale('log')
plt.xlabel(r'$V_\mathrm{max}$ [km s$^{-1}$]', size=20)

plt.legend(title=r'J$_{03}$')

fig.text(0.06, 0.5, 'Number of repops', ha='center',
         va='center', rotation='vertical')

ax4.tick_params(labelleft=False)

# plt.savefig(path_name_res + '/Vmax_hist_geom.png', bbox_inches='tight')
# plt.savefig(path_name_res + '/Vmax_hist_geom.pdf', bbox_inches='tight')
plt.savefig(path_name_res + '/Vmax_hist_linear.png', bbox_inches='tight')
plt.savefig(path_name_res + '/Vmax_hist_linear.pdf', bbox_inches='tight')

# ----------------------- Vmax - J (z==DistEarth) 2x2 --------------------------

fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True,  # sharey=True,
                         figsize=(12, 9))

plt.subplots_adjust(wspace=0, hspace=0)

legend_elements = [
    Line2D([0], [0], marker='o', color='k', label='Frag',
                          markerfacecolor='w', ls='',
                          markersize=8, mew=2.5),
    Line2D([0], [0], marker='P', color='w', label='Res',
                          markerfacecolor='k', markersize=12)]
vminn = 1e-3
vmaxx = np.log10(150)  # 10 ** 1.5
vminn = np.log10(perc_total(2, 5))
vmaxx = np.log10(perc_total(2, 95))
print(vminn, vmaxx)
norm = mcb.Normalize(vminn, vmaxx)

lineas_verticales_dmo = [0.7803158130029454, 1.0924421382041234,
                         1.5294189934857727, 2.1411865908800816,
                         2.997661227232114, 4.196725718124959,
                         5.875416005374943, 8.22558240752492,
                         11.515815370534886, 16.12214151874884,
                         22.570998126248377, 31.599397376747724,
                         44.239156327446814, 61.93481885842554,
                         86.70874640179575]
lineas_verticales_hydro = [0.5610056681443348, 0.7854079354020688,
                           1.0995711095628962, 1.5393995533880545,
                           2.1551593747432762, 3.0172231246405867,
                           4.224112374496821, 5.913757324295549,
                           8.279260254013769, 11.590964355619276,
                           16.227350097866985, 22.718290137013778,
                           31.805606191819287, 44.527848668547,
                           62.3389881359658, 87.27458339035212]

plt.subplot(221)

minn, maxx = minnmaxxS(0)
plt.ylim(minn - 0.1, maxx + 0.1)

minnx = perc_total(3, 0)
maxxx = perc_total(3, 100)
plt.xlim(0.1, maxxx * 1.3)

plt.title('DMO', size=18)
plt.xscale('log')

if plot_res:
    plt.scatter(datos_Js_resi_dmo[:, 3], np.log10(datos_Js_resi_dmo[:, 0]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_resi_dmo[:, 2]))),
            label='Resilient')
if plot_frag:
    plt.scatter(datos_Js_frag_dmo[:, 3], np.log10(datos_Js_frag_dmo[:, 0]),
            c=np.log10(datos_Js_frag_dmo[:, 2]), lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

# for i in lineas_verticales_dmo:
#     plt.axvline(i, alpha=0.5, color='grey')

plt.ylabel(r'log$_{10}$ (J$_\mathrm{S}$ [GeV$^2$ cm$^{-5}$])', fontsize=18)

plt.legend(handles=legend_elements, handletextpad=0.2,
           handlelength=1, loc=2, title=r'J$_\mathrm{S}$')

plt.subplot(222)
plt.title('Hydro', size=18)
plt.xscale('log')

plt.tick_params('y', labelleft=False)

if plot_res:
    plt.scatter(datos_Js_resi_hyd[:, 3], np.log10(datos_Js_resi_hyd[:, 0]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_resi_hyd[:, 2]))),
            label='Resilient')
if plot_frag:
    plt.scatter(datos_Js_frag_hyd[:, 3], np.log10(datos_Js_frag_hyd[:, 0]),
            c=np.log10(datos_Js_frag_hyd[:, 2]), lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)


# for i in lineas_verticales_hydro:
#     plt.axvline(i, alpha=0.5, color='grey')
plt.ylim(minn - 0.1, maxx + 0.1)
plt.legend(handles=legend_elements, handletextpad=0.2, handlelength=1,
           loc=2, title=r'J$_\mathrm{S}$')

plt.subplot(223)
minn, maxx = minnmaxx03(0)
plt.ylim(minn - 0.1, maxx + 0.1)
plt.yticks((19, 20, 21))

plt.xscale('log')

if plot_res:
    plt.scatter(datos_J03_resi_dmo[:, 3], np.log10(datos_J03_resi_dmo[:, 0]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_resi_dmo[:, 2]))),
            label='Resilient')
if plot_frag:
    plt.scatter(datos_J03_frag_dmo[:, 3], np.log10(datos_J03_frag_dmo[:, 0]),
            c=np.log10(datos_J03_frag_dmo[:, 2]), lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)


# for i in lineas_verticales_dmo:
#     plt.axvline(i, alpha=0.5, color='grey')
plt.ylabel(r'log$_{10}$ (J$_{03}$ [GeV$^2$ cm$^{-5}$])', fontsize=18)
plt.xlabel(r'$V_\mathrm{max}$ [km s$^{-1}$]', size=22)

plt.legend(handles=legend_elements, handletextpad=0.2, handlelength=1,
           loc=2, framealpha=0.9, title=r'J$_\mathrm{03}$')

plt.subplot(224)

plt.xscale('log')

if plot_res:
    plt.scatter(datos_J03_resi_hyd[:, 3], np.log10(datos_J03_resi_hyd[:, 0]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_resi_hyd[:, 2]))),
            label='Resilient')
if plot_frag:
    plt.scatter(datos_J03_frag_hyd[:, 3], np.log10(datos_J03_frag_hyd[:, 0]),
                 c=np.log10(datos_J03_frag_hyd[:, 2]), lw=0, marker='P', s=75,
                 cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)


# for i in lineas_verticales_hydro:
#     plt.axvline(i, alpha=0.5, color='grey')
plt.ylim(minn - 0.1, maxx + 0.1)
plt.tick_params('y', labelleft=False)

# plt.xlim(0.09, 130)

# plt.gca().xaxis.set_major_locator(LogLocator(numticks=4))

plt.xlabel(r'$V_\mathrm{max}$ [km s$^{-1}$]', size=22)

plt.legend(handles=legend_elements, handletextpad=0.2, handlelength=1,
           loc=2, title=r'J$_\mathrm{03}$')

cax, kw = colorbarr.make_axes([ax for ax in axes.flat])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
c2 = plt.colorbar(sm, cax=cax, extend='both')
# c2.set_label(r'log$_{10}$(D$_\mathrm{Earth}$ [kpc])', fontsize=20)
c2.set_label(r'D$_\mathrm{Earth}$ [kpc]', fontsize=20)
yticks = c2.get_ticks()
c2.set_ticklabels([str(10 ** i)[:4] for i in yticks])

# plt.show()
plt.savefig(path_name_res + '/VmaxJs' + end_str + '.png',
            bbox_inches='tight')
plt.savefig(path_name_res + '/VmaxJs' + end_str + '.pdf',
            bbox_inches='tight')

# ------------------- Cross sections ------------------------------------------
plt.subplots(1, 1, figsize=(6, 6))

plt.subplots_adjust(wspace=0, hspace=0)

ax1 = plt.subplot(111)

plt.plot(constraints_bb_2204[:, 0], constraints_bb_2204[:, 1], '-',
         label='CB+22', alpha=1., color='b', lw=2)

plt.plot(constraints_bb_2204[:, 0],
         constraints_bb_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_frag_dmo),
         '--', c='k', label='DMO', alpha=1, lw=2.5)
plt.plot(constraints_bb_2204[:, 0],
         constraints_bb_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_frag_hyd),
         '--', label='Hydro', color='limegreen', lw=2.5)

plt.plot(constraints_bb_2204[:, 0],
         constraints_bb_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_resi_dmo),
         ':', c='k', alpha=1, lw=2.5)
plt.plot(constraints_bb_2204[:, 0],
         constraints_bb_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_resi_hyd),
         ':', color='limegreen', alpha=1, lw=2.5)

plt.plot(sigmav_bb_2204[:, 0], sigmav_bb_2204[:, 1], '-',
         c='grey', lw=2, zorder=0)

plt.xlim(sigmav_bb_2204[0, 0], sigmav_bb_2204[-1, 0])

plt.annotate(r'$b\bar{b}$', (2000, 5e-22), color='k')
plt.annotate(r'<$\sigma\nu$>$_\mathrm{th}$', (1000, 3e-26))

plt.xscale('log')
plt.yscale('log')

plt.xlabel('m$_{\chi}$ [GeV]', size=20)
plt.ylabel(r'<$\sigma\nu$> [cm$^3$ s$^{-1}$]', size=20)

legend_elements = [Line2D([0], [0], color='k', label='Frag',
                          linestyle='--', lw=2.5),
                   Line2D([0], [0], color='k', label='Res',
                          linestyle=':', lw=2.5)]

legend1 = plt.legend(legend_elements, ['Frag', 'Res'], loc=6)


legend_elements = [Line2D([0], [0], color='b',
                          linestyle='-', lw=2),
                   mpatches.Patch(color='k', alpha=0.8),
                   mpatches.Patch(color='limegreen', alpha=0.8)]
leg = plt.legend(legend_elements, ['CB+22', 'DMO', 'Hydro'], loc=2)
plt.gca().add_artist(legend1)

t1, t2, t3 = leg.get_texts()
# here we create the distinct instance
t1._fontproperties = t2._fontproperties.copy()
t3.set_size(16)

# ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)
#
# plt.plot(constraints_tau_2204[:, 0],
#          constraints_tau_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_frag_dmo),
#          '-k', label='DMO', lw=2)
# plt.plot(constraints_tau_2204[:, 0],
#          constraints_tau_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_frag_hyd),
#          '-', label='Hydro', color='limegreen', lw=2)
#
# plt.plot(constraints_tau_2204[:, 0],
#          constraints_tau_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_resi_dmo),
#          '--k', lw=2)
# plt.plot(constraints_tau_2204[:, 0],
#          constraints_tau_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_resi_hyd),
#          '--', color='limegreen', lw=2)
#
# plt.plot(sigmav_tau_2204[1:, 0], sigmav_tau_2204[1:, 1], ':k')
# plt.plot(constraints_tau_2204[:, 0], constraints_tau_2204[:, 1], '-.',
#          label='CB+22', alpha=1., color='b', lw=2)
#
# plt.annotate(r'$\tau^+\tau^-$', (1100, 5e-22), color='k')
# plt.annotate(r'<$\sigma\nu$>$_\mathrm{th}$', (1000, 3e-26))
#
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(sigmav_tau_2204[0, 0], sigmav_tau_2204[-1, 0])
#
# plt.xlabel('m$_{\chi}$ [GeV]', size=20)
# ax2.tick_params(labelleft=False)
# # legend1 = plt.legend(legend_elements, ['Frag', 'Res'], loc=9)
# # leg = plt.legend(loc=2)
# # plt.gca().add_artist(legend1)
# # t1, t2, t3 = leg.get_texts()
# # # here we create the distinct instance
# # t1._fontproperties = t2._fontproperties.copy()
# # t3.set_size(16)
#
# print()
# print('J03')
# print('%.2f' % J03_min95_2204)
# print('%.2f  %.2f  %.2f  %.2f' % (J0395_resi_dmo, J0395_resi_hyd,
#                                   J0395_frag_dmo, J0395_frag_hyd))
#
# print()
# print('Js')
# print('%.2f' % Js_min95_2204)
# print('%.2f  %.2f  %.2f  %.2f' % (Js95_resi_dmo, Js95_resi_hyd,
#                                   Js95_frag_dmo, Js95_frag_hyd))

plt.savefig(path_name_res + '/Cross.png', bbox_inches='tight')
plt.savefig(path_name_res + '/Cross.pdf', bbox_inches='tight')

# ---------------- Dgc_hist ---------------------------------------------------
fig, _ = plt.subplots(2, 2, figsize=(12, 9))

plt.subplots_adjust(wspace=0, hspace=0)
column = 1
xxlabel = r'D$_\mathrm{GC}$ [kpc]'

minS, maxS = minnmaxxS(1)
min03, max03 = minnmaxx03(1)
minn = np.min((minS, min03))
maxx = np.max((maxS, max03))

# bines = np.linspace(minn, maxx, 25)
bines = np.logspace(minn * 1.001, maxx * 1.001, 30)
locc = 2

ax1 = plt.subplot(221)

plt.title('DMO', size=18)
if plot_frag:
    plt.hist((datos_Js_frag_dmo[:, column]), log=False,
         label=r'Frag', color='teal', alpha=0.6,
         bins=bines)
if plot_res:
    plt.hist((datos_Js_resi_dmo[:, column]), log=False,
         label=r'Res', color='k', alpha=0.5,
         bins=bines)

plt.axvline(8.5, color='Sandybrown', alpha=1, linestyle='--')
plt.legend(title=r'J$_\mathrm{S}$', loc=locc)

plt.xscale('log')
plt.yscale('log')

plt.xlim(bines[0], bines[-1])

ax2 = plt.subplot(222, sharex=ax1, sharey=ax1)
plt.title('Hydro', size=18)
plt.yscale('log')
if plot_frag:
    plt.hist((datos_Js_frag_hyd[:, column]), log=False,
         label=r'Frag', color='yellowgreen', alpha=0.6,
         bins=bines)
if plot_res:
    plt.hist((datos_Js_resi_hyd[:, column]), log=False,
         label=r'Res', color=darkgreen, alpha=0.6,
         bins=bines)

plt.axvline(8.5, color='orange', alpha=1, linestyle='--')
plt.legend(title=r'J$_\mathrm{S}$', loc=locc)
ax2.tick_params(labelleft=False)

plt.subplot(223, sharex=ax1, sharey=ax1)
if plot_frag:
    plt.hist((datos_J03_frag_dmo[:, column]), log=False,
         label=r'Frag', color='teal', alpha=0.6,
         bins=bines)
if plot_res:
    plt.hist((datos_J03_resi_dmo[:, column]), log=False,
         label=r'Res', color='k', alpha=0.5,
         bins=bines)

plt.axvline(8.5, color='Sandybrown', alpha=1, linestyle='--')

plt.yscale('log')
plt.xlabel(xxlabel, fontsize=20)
plt.legend(title=r'J$_{03}$', loc=locc)

ax4 = plt.subplot(224, sharex=ax1, sharey=ax1)
if plot_frag:
    plt.hist((datos_J03_frag_hyd[:, column]), log=False,
         label=r'Frag', color='yellowgreen', alpha=0.6,
         bins=bines)
if plot_res:
    plt.hist((datos_J03_resi_hyd[:, column]), log=False,
         label=r'Res', color=darkgreen, alpha=0.6,
         bins=bines)

plt.axvline(8.5, color='orange', alpha=1, linestyle='--')

plt.yscale('log')
plt.xlabel(xxlabel, fontsize=20)
plt.legend(title=r'J$_{03}$', loc=locc)

fig.text(0.06, 0.5, 'Number of repops', ha='center',
         va='center', rotation='vertical')
ax4.tick_params(labelleft=False)

# %%

minn = np.min((np.min(datos_Js_frag_dmo[:, column]),
               np.min(datos_J03_frag_dmo[:, column]),
               np.min(datos_Js_frag_hyd[:, column]),
               np.min(datos_J03_frag_hyd[:, column]),
               np.min(datos_Js_resi_dmo[:, column]),
               np.min(datos_J03_resi_dmo[:, column]),
               np.min(datos_Js_resi_hyd[:, column]),
               np.min(datos_J03_resi_hyd[:, column])))

num_min = 2.
column = 3

print(np.shape(np.where(datos_Js_frag_dmo[:, column] < num_min))[1],
      np.shape(np.where(datos_Js_resi_dmo[:, column] < num_min))[1], '---',
      np.shape(np.where(datos_Js_frag_hyd[:, column] < num_min))[1],
      np.shape(np.where(datos_Js_resi_hyd[:, column] < num_min))[1])

print(np.shape(np.where(datos_J03_frag_dmo[:, column] < num_min))[1],
      np.shape(np.where(datos_J03_resi_dmo[:, column] < num_min))[1], '---',
      np.shape(np.where(datos_J03_frag_hyd[:, column] < num_min))[1],
      np.shape(np.where(datos_J03_resi_hyd[:, column] < num_min))[1])

plt.savefig(path_name_res + '/Dgc_hist.png', bbox_inches='tight')
plt.savefig(path_name_res + '/Dgc_hist.pdf', bbox_inches='tight')

import Calculations.Repopulation.attemp_at_functions2 as funct_repop
from scipy.optimize import newton

SHVF_cts_RangeMax = 120.0
SHVF_cts_RangeMin = 0.1
SHVF_bb = 5.68
SHVF_mm = -3.92
num_subs_max = 5e5
repop_inc_factor = 1.4
m_min = SHVF_cts_RangeMin
m_max = 0.
print(funct_repop.SHVF_Grand2012_int(
    SHVF_cts_RangeMin, SHVF_cts_RangeMax,
    SHVF_bb, SHVF_mm))
print(funct_repop.SHVF_Grand2012_int(
    SHVF_cts_RangeMin, SHVF_cts_RangeMax,
    5.77, SHVF_mm))
mean_dmo = []
mean_hydro = []
while m_max < SHVF_cts_RangeMax:
    if funct_repop.SHVF_Grand2012_int(m_min, SHVF_cts_RangeMax,
                                      SHVF_bb, SHVF_mm) > num_subs_max:
        m_max = newton(funct_repop.xx, m_min,
                       args=[m_min, SHVF_bb, SHVF_mm, num_subs_max])
        new_mmin = m_max
    else:
        m_max = np.minimum(m_min * repop_inc_factor,
                           SHVF_cts_RangeMax)
        new_mmin = m_min * repop_inc_factor
        if m_max > 100:
            m_max = SHVF_cts_RangeMax
        print(m_min, m_max, (m_min * m_max) ** 0.5)
        print(funct_repop.SHVF_Grand2012_int(m_min, m_max,
                                             SHVF_bb, SHVF_mm))
        mean_dmo.append((m_min * m_max) ** 0.5)
    m_min = new_mmin
SHVF_bb = 5.3
SHVF_mm = -4.08
m_min = SHVF_cts_RangeMin
m_max = 0.
while m_max < SHVF_cts_RangeMax:
    if funct_repop.SHVF_Grand2012_int(m_min, SHVF_cts_RangeMax,
                                      SHVF_bb, SHVF_mm) > num_subs_max:
        m_max = newton(funct_repop.xx, m_min,
                       args=[m_min, SHVF_bb, SHVF_mm, num_subs_max])
        new_mmin = m_max
    else:
        m_max = np.minimum(m_min * repop_inc_factor,
                           SHVF_cts_RangeMax)
        new_mmin = m_min * repop_inc_factor
        if m_max > 100:
            m_max = SHVF_cts_RangeMax
        print(m_min, m_max, (m_min * m_max) ** 0.5)
        print(funct_repop.SHVF_Grand2012_int(m_min, m_max,
                                             SHVF_bb, SHVF_mm))
        mean_hydro.append((m_min * m_max) ** 0.5)
    m_min = new_mmin

plt.figure()
plt.plot(mean_dmo, np.ones(len(mean_dmo)), c='k', marker='.', ls='')
plt.plot(mean_hydro, 0.9 * np.ones(len(mean_hydro)), c='green', marker='x',
         ls='')
for i in [0.0, 9.0, 13.0, 18.0, 25.0, 36.0, 50.0, 70.0, 90.0]:
    plt.axvline(i, alpha=0.5, color='grey')
plt.ylim(0.7, 1.3)
plt.xscale('log')

plt.show()
