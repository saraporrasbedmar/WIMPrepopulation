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
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator

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


path_name = '/home/porrassa/Desktop/WIMPS_project/Physnet_outputs_repops/' \
            '2024/outputs/final_2024_8max'

final_size = (500, 1, 6)

datos_Js_frag_hyd = np.loadtxt(path_name + '/Js_hydro_fragile_results.txt')
datos_Js_frag_dmo = np.loadtxt(path_name + '/Js_dmo_fragile_results.txt')
datos_Js_frag_hyd = datos_Js_frag_hyd.reshape(
    (int(datos_Js_frag_hyd.size/6), 1, 6))[:, 0, :]
datos_Js_frag_dmo = datos_Js_frag_dmo.reshape(
    (int(datos_Js_frag_dmo.size/6), 1, 6))[:, 0, :]
print(datos_Js_frag_hyd)
datos_J03_frag_hyd = np.loadtxt(path_name + '/J03_hydro_fragile_results.txt')
datos_J03_frag_dmo = np.loadtxt(path_name + '/J03_dmo_fragile_results.txt')
datos_J03_frag_hyd = datos_J03_frag_hyd.reshape(
    (int(datos_J03_frag_hyd.size/6), 1, 6))[:, 0, :]
datos_J03_frag_dmo = datos_J03_frag_dmo.reshape(
    (int(datos_J03_frag_dmo.size/6), 1, 6))[:, 0, :]

datos_Js_resi_hyd = np.loadtxt(path_name + '/Js_hydro_resilient_results.txt')
datos_Js_resi_dmo = np.loadtxt(path_name + '/Js_dmo_resilient_results.txt')
datos_Js_resi_hyd = datos_Js_resi_hyd.reshape(final_size)[:, 0, :]
datos_Js_resi_dmo = datos_Js_resi_dmo.reshape(final_size)[:, 0, :]

datos_J03_resi_hyd = np.loadtxt(path_name + '/J03_hydro_resilient_results.txt')
datos_J03_resi_dmo = np.loadtxt(path_name + '/J03_dmo_resilient_results.txt')

datos_J03_resi_hyd = datos_J03_resi_hyd.reshape(final_size)[:, 0, :]
datos_J03_resi_dmo = datos_J03_resi_dmo.reshape(final_size)[:, 0, :]

constraints_bb_2204 = np.loadtxt('../Constraints_2204/Limit_bb.txt')
constraints_tau_2204 = np.loadtxt('../Constraints_2204/Limit_tau.txt')
sigmav_bb_2204 = np.loadtxt('../Constraints_2204/sigmav_bb.txt')
sigmav_tau_2204 = np.loadtxt('../Constraints_2204/sigmav_tau.txt')

sigmav_bb_2204 = sigmav_bb_2204[sigmav_bb_2204[:, 0].argsort()[::], :]
sigmav_tau_2204 = sigmav_tau_2204[sigmav_tau_2204[:, 0].argsort()[::], :]

J03_min95_2204 = 18.9208  # From digitalizing
Js_min95_2204 = 19.4642  # From digitalizing

plt.close('all')

darkgreen = (0.024, 0.278, 0.047)


fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True,  # sharey=True,
                         figsize=(12, 9))

plt.subplots_adjust(wspace=0, hspace=0)

colormapp = 'rainbow'
legend_elements = [Line2D([0], [0], marker='P', color='w', label='Frag',
                          markerfacecolor='k', markersize=12),
                   Line2D([0], [0], marker='o', color='w', label='Res',
                          markerfacecolor='k', markersize=8)]
vminn = 1e-3
vmaxx = np.log10(100)  #10 ** 1.5

plt.subplot(221)

minn = np.min((np.min(np.log10(datos_Js_frag_dmo[:, 3])),
               np.min(np.log10(datos_Js_frag_hyd[:, 3])),
               np.min(np.log10(datos_Js_resi_dmo[:, 3])),
               np.min(np.log10(datos_Js_resi_hyd[:, 3]))))

maxx = np.max((np.max(np.log10(datos_Js_frag_dmo[:, 3])),
               np.max(np.log10(datos_Js_frag_hyd[:, 3])),
               np.max(np.log10(datos_Js_resi_dmo[:, 3])),
               np.max(np.log10(datos_Js_resi_hyd[:, 3])),))
# plt.ylim(minn - 0.1, maxx + 0.1)

plt.title('DMO', size=18)
plt.xscale('log')

plt.scatter(datos_Js_resi_dmo[:, 3], np.log10(datos_Js_resi_dmo[:, 2]),
            # c=np.log10(datos_Js_resi_dmo[:, 2]),
            lw=0, marker='o',
            cmap=colormapp, label='Resilient', vmin=vminn, vmax=vmaxx)

plt.scatter(datos_Js_frag_dmo[:, 3], np.log10(datos_Js_frag_dmo[:, 2]),
            # c=np.log10(datos_Js_frag_dmo[:, 2]),
            lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

plt.ylabel(r'log$_{10}$ (D$_\mathrm{Earth}$ [kpc])', fontsize=18)
plt.legend(handles=legend_elements, handletextpad=0.2,
           handlelength=1, loc=2, title=r'J$_\mathrm{S}$')

plt.subplot(222)
plt.title('Hydro', size=18)
plt.xscale('log')

# plt.yticks((18, 19, 20, 21, 22), labels=('', '', '', '', ''))

plt.scatter(datos_Js_resi_hyd[:, 3], np.log10(datos_Js_resi_hyd[:, 2]),
            # c=np.log10(datos_Js_resi_hyd[:, 2]),
            lw=0, marker='o',
            cmap=colormapp, label='Resilient', vmin=vminn, vmax=vmaxx)

plt.scatter(datos_Js_frag_hyd[:, 3], np.log10(datos_Js_frag_hyd[:, 2]),
            # c=np.log10(datos_Js_frag_hyd[:, 2]),
            lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

# plt.ylim(minn - 0.1, maxx + 0.1)
plt.legend(handles=legend_elements, handletextpad=0.2, handlelength=1,
           loc=2, title=r'J$_\mathrm{S}$')

plt.subplot(223)
minn = np.min((np.min(np.log10(datos_J03_frag_dmo[:, 3])),
               np.min(np.log10(datos_J03_frag_hyd[:, 3])),
               np.min(np.log10(datos_J03_resi_dmo[:, 3])),
               np.min(np.log10(datos_J03_resi_hyd[:, 3]))))

maxx = np.max((np.max(np.log10(datos_J03_frag_dmo[:, 3])),
               np.max(np.log10(datos_J03_frag_hyd[:, 3])),
               np.max(np.log10(datos_J03_resi_dmo[:, 3])),
               np.max(np.log10(datos_J03_resi_hyd[:, 3])),))
plt.xscale('log')

plt.scatter(datos_J03_resi_dmo[:, 3], np.log10(datos_J03_resi_dmo[:, 2]),
            # c=np.log10(datos_J03_resi_dmo[:, 2]),
            lw=0, marker='o',
            cmap=colormapp, label='Resilient', vmin=vminn, vmax=vmaxx)

plt.scatter(datos_J03_frag_dmo[:, 3], np.log10(datos_J03_frag_dmo[:, 2]),
            # c=np.log10(datos_J03_frag_dmo[:, 2]),
            lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

# plt.ylim(minn - 0.1, maxx + 0.1)
print('minn - 0.1, maxx + 0.1', minn - 0.1, maxx + 0.1)
plt.ylabel(r'log$_{10}$ (D$_\mathrm{Earth}$ [kpc])', fontsize=18)
plt.xlabel(r'$V_\mathrm{max}$ [km s$^{-1}$]', size=22)
plt.legend(handles=legend_elements, handletextpad=0.2, handlelength=1,
           loc=2, framealpha=0.9, title=r'J$_\mathrm{03}$')
# plt.yticks((19, 20))

plt.subplot(224)

plt.xscale('log')

plt.scatter(datos_J03_resi_hyd[:, 3], np.log10(datos_J03_resi_hyd[:, 2]),
            # c=np.log10(datos_J03_resi_hyd[:, 2]),
            lw=0, marker='o',
            cmap=colormapp, label='Resilient', vmin=vminn, vmax=vmaxx)

im = plt.scatter(datos_J03_frag_hyd[:, 3], np.log10(datos_J03_frag_hyd[:, 2]),
                 # c=np.log10(datos_J03_frag_hyd[:, 2]),
                 lw=0, marker='P', s=75,
                 cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

# plt.yticks((19, 20), labels=('', ''))
plt.xlabel(r'$V_\mathrm{max}$ [km s$^{-1}$]', size=22)
plt.legend(handles=legend_elements, handletextpad=0.2, handlelength=1,
           loc=2, title=r'J$_\mathrm{03}$')
# plt.ylim(minn - 0.1, maxx + 0.1)
print('minn - 0.1, maxx + 0.1', minn - 0.1, maxx + 0.1)

# cax, kw = colorbarr.make_axes([ax for ax in axes.flat])
# c2 = plt.colorbar(im, cax=cax, **kw)
# c2.set_label(r'log$_{10}$(D$_\mathrm{Earth}$ [kpc])', fontsize=20)

plt.savefig(path_name + '/VmaxDEarth.png', bbox_inches='tight')
plt.savefig(path_name + '/VmaxDEarth.pdf', bbox_inches='tight')
plt.show()

plt.scatter()

fig, _ = plt.subplots(2, 2, figsize=(12, 9))

plt.subplots_adjust(wspace=0, hspace=0)

minn = np.min((np.min(np.log10(datos_Js_frag_dmo[:, 0])),
               np.min(np.log10(datos_J03_frag_dmo[:, 0])),
               np.min(np.log10(datos_Js_frag_hyd[:, 0])),
               np.min(np.log10(datos_J03_frag_hyd[:, 0])),
               np.min(np.log10(datos_Js_resi_dmo[:, 0])),
               np.min(np.log10(datos_J03_resi_dmo[:, 0])),
               np.min(np.log10(datos_Js_resi_hyd[:, 0])),
               np.min(np.log10(datos_J03_resi_hyd[:, 0]))))

maxx = np.max((np.max(np.log10(datos_Js_frag_dmo[:, 0])),
               np.max(np.log10(datos_J03_frag_dmo[:, 0])),
               np.max(np.log10(datos_Js_frag_hyd[:, 0])),
               np.max(np.log10(datos_J03_frag_hyd[:, 0])),
               np.max(np.log10(datos_Js_resi_dmo[:, 0])),
               np.max(np.log10(datos_J03_resi_dmo[:, 0])),
               np.max(np.log10(datos_Js_resi_hyd[:, 0])),
               np.max(np.log10(datos_J03_resi_hyd[:, 0]))))

bines = np.linspace(minn, maxx, 40)

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
print(Js95_resi_dmo, Js95_frag_dmo)

plt.axvline(Js95_frag_dmo, color='teal')  # , alpha=0.6)
plt.axvline(Js95_resi_dmo, color='k')  # , alpha=0.5)

plt.xlim(minn, maxx)
plt.ylim(0.9, 150)

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
print(Js95_resi_hyd, Js95_frag_hyd)

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
print(J0395_frag_dmo, J0395_resi_dmo)
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
print(J0395_resi_hyd, J0395_resi_hyd)

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



plt.savefig(path_name + '/J_hist.png', bbox_inches='tight')
plt.savefig(path_name + '/J_hist.pdf', bbox_inches='tight')
# %%


### Vmax - J (z==DistEarth) 2x2 --------------------------

fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True,  # sharey=True,
                         figsize=(12, 9))

plt.subplots_adjust(wspace=0, hspace=0)

colormapp = 'rainbow'
legend_elements = [Line2D([0], [0], marker='P', color='w', label='Frag',
                          markerfacecolor='k', markersize=12),
                   Line2D([0], [0], marker='o', color='w', label='Res',
                          markerfacecolor='k', markersize=8)]
vminn = 1e-3
vmaxx = np.log10(100)  #10 ** 1.5

plt.subplot(221)

minn = np.min((np.min(np.log10(datos_Js_frag_dmo[:, 0])),
               np.min(np.log10(datos_Js_frag_hyd[:, 0])),
               np.min(np.log10(datos_Js_resi_dmo[:, 0])),
               np.min(np.log10(datos_Js_resi_hyd[:, 0]))))

maxx = np.max((np.max(np.log10(datos_Js_frag_dmo[:, 0])),
               np.max(np.log10(datos_Js_frag_hyd[:, 0])),
               np.max(np.log10(datos_Js_resi_dmo[:, 0])),
               np.max(np.log10(datos_Js_resi_hyd[:, 0])),))
plt.ylim(minn - 0.1, maxx + 0.1)

plt.title('DMO', size=18)
plt.xscale('log')

plt.scatter(datos_Js_resi_dmo[:, 3], np.log10(datos_Js_resi_dmo[:, 0]),
            c=np.log10(datos_Js_resi_dmo[:, 2]), lw=0, marker='o',
            cmap=colormapp, label='Resilient', vmin=vminn, vmax=vmaxx)

plt.scatter(datos_Js_frag_dmo[:, 3], np.log10(datos_Js_frag_dmo[:, 0]),
            c=np.log10(datos_Js_frag_dmo[:, 2]), lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

plt.ylabel(r'log$_{10}$ (J$_\mathrm{S}$ [GeV$^2$ cm$^{-5}$])', fontsize=18)
plt.legend(handles=legend_elements, handletextpad=0.2,
           handlelength=1, loc=2, title=r'J$_\mathrm{S}$')

plt.subplot(222)
plt.title('Hydro', size=18)
plt.xscale('log')

plt.yticks((18, 19, 20, 21, 22), labels=('', '', '', '', ''))

plt.scatter(datos_Js_resi_hyd[:, 3], np.log10(datos_Js_resi_hyd[:, 0]),
            c=np.log10(datos_Js_resi_hyd[:, 2]), lw=0, marker='o',
            cmap=colormapp, label='Resilient', vmin=vminn, vmax=vmaxx)

plt.scatter(datos_Js_frag_hyd[:, 3], np.log10(datos_Js_frag_hyd[:, 0]),
            c=np.log10(datos_Js_frag_hyd[:, 2]), lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

plt.ylim(minn - 0.1, maxx + 0.1)
plt.legend(handles=legend_elements, handletextpad=0.2, handlelength=1,
           loc=2, title=r'J$_\mathrm{S}$')

plt.subplot(223)
minn = np.min((np.min(np.log10(datos_J03_frag_dmo[:, 0])),
               np.min(np.log10(datos_J03_frag_hyd[:, 0])),
               np.min(np.log10(datos_J03_resi_dmo[:, 0])),
               np.min(np.log10(datos_J03_resi_hyd[:, 0]))))

maxx = np.max((np.max(np.log10(datos_J03_frag_dmo[:, 0])),
               np.max(np.log10(datos_J03_frag_hyd[:, 0])),
               np.max(np.log10(datos_J03_resi_dmo[:, 0])),
               np.max(np.log10(datos_J03_resi_hyd[:, 0])),))
plt.xscale('log')

plt.scatter(datos_J03_resi_dmo[:, 3], np.log10(datos_J03_resi_dmo[:, 0]),
            c=np.log10(datos_J03_resi_dmo[:, 2]), lw=0, marker='o',
            cmap=colormapp, label='Resilient', vmin=vminn, vmax=vmaxx)

plt.scatter(datos_J03_frag_dmo[:, 3], np.log10(datos_J03_frag_dmo[:, 0]),
            c=np.log10(datos_J03_frag_dmo[:, 2]), lw=0, marker='P', s=75,
            cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

plt.ylim(minn - 0.1, maxx + 0.1)
print('minn - 0.1, maxx + 0.1', minn - 0.1, maxx + 0.1)
plt.ylabel(r'log$_{10}$ (J$_{03}$ [GeV$^2$ cm$^{-5}$])', fontsize=18)
plt.xlabel(r'$V_\mathrm{max}$ [km s$^{-1}$]', size=22)
plt.legend(handles=legend_elements, handletextpad=0.2, handlelength=1,
           loc=2, framealpha=0.9, title=r'J$_\mathrm{03}$')
plt.yticks((19, 20))

plt.subplot(224)

plt.xscale('log')

plt.scatter(datos_J03_resi_hyd[:, 3], np.log10(datos_J03_resi_hyd[:, 0]),
            c=np.log10(datos_J03_resi_hyd[:, 2]), lw=0, marker='o',
            cmap=colormapp, label='Resilient', vmin=vminn, vmax=vmaxx)

im = plt.scatter(datos_J03_frag_hyd[:, 3], np.log10(datos_J03_frag_hyd[:, 0]),
                 c=np.log10(datos_J03_frag_hyd[:, 2]), lw=0, marker='P', s=75,
                 cmap=colormapp, label='Fragile', vmin=vminn, vmax=vmaxx)

plt.yticks((19, 20), labels=('', ''))
plt.xlabel(r'$V_\mathrm{max}$ [km s$^{-1}$]', size=22)
plt.legend(handles=legend_elements, handletextpad=0.2, handlelength=1,
           loc=2, title=r'J$_\mathrm{03}$')
plt.ylim(minn - 0.1, maxx + 0.1)
print('minn - 0.1, maxx + 0.1', minn - 0.1, maxx + 0.1)

cax, kw = colorbarr.make_axes([ax for ax in axes.flat])
c2 = plt.colorbar(im, cax=cax, **kw)
c2.set_label(r'log$_{10}$(D$_\mathrm{Earth}$ [kpc])', fontsize=20)

plt.savefig(path_name + '/VmaxJs.png', bbox_inches='tight')
plt.savefig(path_name + '/VmaxJs.pdf', bbox_inches='tight')


# %%

plt.subplots(1, 2, figsize=(14, 6))

plt.subplots_adjust(wspace=0, hspace=0)

ax1 = plt.subplot(121)

plt.plot(constraints_bb_2204[:, 0],
         constraints_bb_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_frag_dmo),
         '-k', label='DMO', alpha=0.8)
plt.plot(constraints_bb_2204[:, 0],
         constraints_bb_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_frag_hyd),
         '-', label='Hydro', color='limegreen')

plt.plot(constraints_bb_2204[:, 0],
         constraints_bb_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_resi_dmo),
         '--k', alpha=0.8)
plt.plot(constraints_bb_2204[:, 0],
         constraints_bb_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_resi_hyd),
         '--', color='limegreen')

plt.plot(sigmav_bb_2204[:, 0], sigmav_bb_2204[:, 1], ':k')
plt.plot(constraints_bb_2204[:, 0], constraints_bb_2204[:, 1], '--',
         label='CB+22', alpha=0.5, color='grey')

plt.annotate(r'$b\bar{b}$', (2000, 1e-20), color='k')
plt.annotate(r'<$\sigma\nu$>$_\mathrm{th}$', (1000, 3e-26))

plt.xscale('log')
plt.yscale('log')

plt.xlabel('m$_{\chi}$ [GeV]', size=20)
plt.ylabel(r'<$\sigma\nu$> [cm$^3$ s$^{-1}$]', size=20)

legend_elements = [Line2D([0], [0], color='k', label='Frag', linestyle='-'),
                   Line2D([0], [0], color='k', label='Res', linestyle='--')]

legend1 = plt.legend(legend_elements, ['Frag', 'Res'], loc=9)
leg = plt.legend(loc=2)
plt.gca().add_artist(legend1)

t1, t2, t3 = leg.get_texts()
# here we create the distinct instance
t1._fontproperties = t2._fontproperties.copy()
t3.set_size(16)

ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)

plt.plot(constraints_tau_2204[:, 0],
         constraints_tau_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_frag_dmo),
         '-k', label='DMO', alpha=0.8)
plt.plot(constraints_tau_2204[:, 0],
         constraints_tau_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_frag_hyd),
         '-', label='Hydro', color='limegreen')

plt.plot(constraints_tau_2204[:, 0],
         constraints_tau_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_resi_dmo),
         '--k', alpha=0.8)
plt.plot(constraints_tau_2204[:, 0],
         constraints_tau_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_resi_hyd),
         '--', color='limegreen')

plt.plot(sigmav_tau_2204[1:, 0], sigmav_tau_2204[1:, 1], ':k')
plt.plot(constraints_tau_2204[:, 0], constraints_tau_2204[:, 1], '--',
         label='CB+22', alpha=0.5, color='grey')

plt.annotate(r'$\tau^+\tau^-$', (1100, 1e-20), color='k')
plt.annotate(r'<$\sigma\nu$>$_\mathrm{th}$', (1000, 3e-26))

plt.xscale('log')
plt.yscale('log')
plt.xlim(sigmav_tau_2204[0, 0], sigmav_tau_2204[-1, 0])

plt.xlabel('m$_{\chi}$ [GeV]', size=20)
ax2.tick_params(labelleft=False)
# legend1 = plt.legend(legend_elements, ['Frag', 'Res'], loc=9)
# leg = plt.legend(loc=2)
# plt.gca().add_artist(legend1)
# t1, t2, t3 = leg.get_texts()
# # here we create the distinct instance
# t1._fontproperties = t2._fontproperties.copy()
# t3.set_size(16)

print()
print('J03')
print('%.2f' % J03_min95_2204)
print('%.2f  %.2f  %.2f  %.2f' % (J0395_resi_dmo, J0395_resi_hyd,
                                  J0395_frag_dmo, J0395_frag_hyd))

print()
print('Js')
print('%.2f' % Js_min95_2204)
print('%.2f  %.2f  %.2f  %.2f' % (Js95_resi_dmo, Js95_resi_hyd,
                                  Js95_frag_dmo, Js95_frag_hyd))


plt.savefig(path_name + '/Cross.png', bbox_inches='tight')
plt.savefig(path_name + '/Cross.pdf', bbox_inches='tight')



fig, _ = plt.subplots(2, 2, figsize=(12, 9))

plt.subplots_adjust(wspace=0, hspace=0)
column = 1
xxlabel = r'D$_\mathrm{GC}$ [kpc]'

minn = np.min((np.min(datos_Js_frag_dmo[:, column]),
               np.min(datos_J03_frag_dmo[:, column]),
               np.min(datos_Js_frag_hyd[:, column]),
               np.min(datos_J03_frag_hyd[:, column]),
               np.min(datos_Js_resi_dmo[:, column]),
               np.min(datos_J03_resi_dmo[:, column]),
               np.min(datos_Js_resi_hyd[:, column]),
               np.min(datos_J03_resi_hyd[:, column])))

maxx = np.max((np.max(datos_Js_frag_dmo[:, column]),
               np.max(datos_J03_frag_dmo[:, column]),
               np.max(datos_Js_frag_hyd[:, column]),
               np.max(datos_J03_frag_hyd[:, column]),
               np.max(datos_Js_resi_dmo[:, column]),
               np.max(datos_J03_resi_dmo[:, column]),
               np.max(datos_Js_resi_hyd[:, column]),
               np.max(datos_J03_resi_hyd[:, column])))

# bines = np.linspace(minn, maxx, 25)
bines = np.logspace(np.log10(minn), np.log10(maxx), 30)
locc = 2

ax1 = plt.subplot(221)
plt.xlim(0.5, 250)
plt.yscale('log')

plt.title('DMO', size=18)

plt.hist((datos_Js_frag_dmo[:, column]), log=False,
         label=r'Frag', color='teal', alpha=0.6,
         bins=bines)
plt.hist((datos_Js_resi_dmo[:, column]), log=False,
         label=r'Res', color='k', alpha=0.5,
         bins=bines)

plt.axvline(8.5, color='Sandybrown', alpha=1, linestyle='--')
plt.legend(title=r'J$_\mathrm{S}$', loc=locc)
plt.xscale('log')

ax2 = plt.subplot(222, sharex=ax1, sharey=ax1)
plt.title('Hydro', size=18)
plt.yscale('log')

plt.hist((datos_Js_frag_hyd[:, column]), log=False,
         label=r'Frag', color='yellowgreen', alpha=0.6,
         bins=bines)
plt.hist((datos_Js_resi_hyd[:, column]), log=False,
         label=r'Res', color=darkgreen, alpha=0.6,
         bins=bines)

plt.axvline(8.5, color='orange', alpha=1, linestyle='--')
plt.legend(title=r'J$_\mathrm{S}$', loc=locc)
ax2.tick_params(labelleft=False)

plt.subplot(223, sharex=ax1, sharey=ax1)

plt.hist((datos_J03_frag_dmo[:, column]), log=False,
         label=r'Frag', color='teal', alpha=0.6,
         bins=bines)
plt.hist((datos_J03_resi_dmo[:, column]), log=False,
         label=r'Res', color='k', alpha=0.5,
         bins=bines)

plt.axvline(8.5, color='Sandybrown', alpha=1, linestyle='--')

plt.yscale('log')
plt.xlabel(xxlabel, fontsize=20)
plt.legend(title=r'J$_{03}$', loc=locc)

ax4 = plt.subplot(224, sharex=ax1, sharey=ax1)

plt.hist((datos_J03_frag_hyd[:, column]), log=False,
         label=r'Frag', color='yellowgreen', alpha=0.6,
         bins=bines)
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

plt.savefig(path_name + '/Dgc_hist.png', bbox_inches='tight')
plt.savefig(path_name + '/Dgc_hist.pdf', bbox_inches='tight')

plt.show()
