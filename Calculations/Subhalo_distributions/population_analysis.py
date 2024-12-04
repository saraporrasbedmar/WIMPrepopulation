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
from matplotlib.ticker import MaxNLocator

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

print(os.getcwd())

path_name = 'outputs'

data_release_dmo = np.loadtxt(
    '../Data_subhalo_simulations/dmo_table.txt', skiprows=3)
data_release_hydro = np.loadtxt(
    '../Data_subhalo_simulations/hydro_table.txt', skiprows=3)

data_release_dmo = data_release_dmo[
                   data_release_dmo[:, 0] > 0.184, :]
data_release_hydro = data_release_hydro[
                     data_release_hydro[:, 0] > 0.184, :]

data_dict = funct_repop.read_config_file(
    '../Repopulation/input_files/input_paper2024_SHVFnorm.yml')
cv_cts = data_dict['Cv']
srd_cts = data_dict['SRD']
SHVF_cts = data_dict['SHVF']

repopulations = data_dict['repopulations']

res_string = 'fragile'

dmo = funct_repop.calculate_characteristics_subhalo(
    Vmax=data_release_dmo[:, 1],
    Distgc=data_release_dmo[:, 2],
    sim_type='dmo',
    res_string=res_string,
    cosmo_G = data_dict['cosmo_constants']['G'],
    cosmo_H_0 = data_dict['cosmo_constants']['H_0'],
    cosmo_rho_crit = data_dict['cosmo_constants']['rho_crit'],

    host_R_vir = data_dict['host']['R_vir'],
    host_rho_0 = float(data_dict['host']['rho_0']),
    host_r_s = data_dict['host']['r_s'],

    pathname='',
    repop_its = repopulations['its'],
    repop_print_freq = repopulations['print_freq'],
    repop_num_brightest = int(repopulations['num_brightest']),
    repop_inc_factor = repopulations['inc_factor'],

    SHVF_cts_RangeMin = SHVF_cts['RangeMin'],
    SHVF_cts_RangeMax = SHVF_cts['RangeMax'],

    SHVF_bb = SHVF_cts['dmo']['bb'],
    SHVF_mm = SHVF_cts['dmo']['mm'],

    Cv_bb = cv_cts['dmo']['bb'],
    Cv_mm = cv_cts['dmo']['mm'],
    Cv_sigma = cv_cts['dmo']['sigma'],

    srd_args_repop = srd_cts['dmo'][res_string]['args'],
    srd_args_visible = srd_cts['dmo']['fragile']['args'],
    srd_last_sub = srd_cts['dmo'][res_string]['last_subhalo']
)

datos_Js_dmo = dmo[:, [0, 2, 3, 4, 5, 6]]
datos_J03_dmo = dmo[:, 1:]

hydro = funct_repop.calculate_characteristics_subhalo(
    Vmax=data_release_hydro[:, 1],
    Distgc=data_release_hydro[:, 2],
    sim_type='hydro',
    res_string=res_string,
    cosmo_G=data_dict['cosmo_constants']['G'],
    cosmo_H_0=data_dict['cosmo_constants']['H_0'],
    cosmo_rho_crit=data_dict['cosmo_constants']['rho_crit'],

    host_R_vir=data_dict['host']['R_vir'],
    host_rho_0=float(data_dict['host']['rho_0']),
    host_r_s=data_dict['host']['r_s'],

    pathname='',
    repop_its=repopulations['its'],
    repop_print_freq=repopulations['print_freq'],
    repop_num_brightest=int(repopulations['num_brightest']),
    repop_inc_factor=repopulations['inc_factor'],

    SHVF_cts_RangeMin=SHVF_cts['RangeMin'],
    SHVF_cts_RangeMax=SHVF_cts['RangeMax'],

    SHVF_bb=SHVF_cts['hydro']['bb'],
    SHVF_mm=SHVF_cts['hydro']['mm'],

    Cv_bb=cv_cts['hydro']['bb'],
    Cv_mm=cv_cts['hydro']['mm'],
    Cv_sigma=cv_cts['hydro']['sigma'],

    srd_args_repop=srd_cts['hydro'][res_string]['args'],
    srd_args_visible=srd_cts['hydro']['fragile']['args'],
    srd_last_sub=srd_cts['hydro'][res_string]['last_subhalo']
)
datos_Js_hyd = hydro[:, [0, 2, 3, 4, 5, 6]]
datos_J03_hyd = hydro[:, 1:]

J03_min95_2204 = 18.9208  # From digitalizing
Js_min95_2204 = 19.4642  # From digitalizing


def minnmaxxS(i):
    minn = np.min((np.min(np.log10(datos_Js_dmo[:, i])),
                   np.min(np.log10(datos_Js_hyd[:, i]))))

    maxx = np.max((np.max(np.log10(datos_Js_dmo[:, i])),
                   np.max(np.log10(datos_Js_hyd[:, i])),))

    return minn, maxx
def minnmaxx03(i):
    minn = np.min((np.min(np.log10(datos_J03_dmo[:, i])),
                   np.min(np.log10(datos_J03_hyd[:, i]))))

    maxx = np.max((np.max(np.log10(datos_J03_dmo[:, i])),
                   np.max(np.log10(datos_J03_hyd[:, i])),))

    return minn, maxx

def perc_total(i, number):
    data = (
        np.concatenate((datos_Js_dmo[:, i], datos_Js_hyd[:, i],
                        datos_J03_dmo[:, i], datos_J03_hyd[:, i]),
                       axis=None))
    return np.percentile(data, number)

darkgreen = (0.024, 0.278, 0.047)

# ------------------------ DEarthJs -------------------------------------------
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True,  # sharey=True,
                         figsize=(12, 9))

plt.subplots_adjust(wspace=0, hspace=0)

cmap = cm.viridis
colormapp = 'viridis'
vminn = np.log10(perc_total(3, 2))  # np.log10(3)
vmaxx = np.log10(perc_total(3, 98))  # np.log10(100)
norm = mcb.Normalize(vminn, vmaxx)
print(vminn, vmaxx)

plt.subplot(221)
minn, maxx = minnmaxxS(0)
plt.ylim(minn - 0.1, maxx + 0.1)

plt.title('DMO', size=18)
plt.xscale('log')

plt.scatter(datos_Js_dmo[:, 2], np.log10(datos_Js_dmo[:, 0]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_dmo[:, 3]))),
            label='Resilient')

plt.ylabel(r'log$_{10}$ (J$_\mathrm{S}$ [GeV$^2$ cm$^{-5}$])', fontsize=18)


plt.subplot(222)
plt.title('Hydro', size=18)
plt.xscale('log')

plt.scatter(datos_Js_hyd[:, 2], np.log10(datos_Js_hyd[:, 0]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_hyd[:, 3]))),
            label='Resilient')

# plt.yticks((18, 19, 20, 21, 22), labels=('', '', '', '', ''))
plt.ylim(minn - 0.1, maxx + 0.1)


plt.subplot(223)

plt.scatter(datos_J03_dmo[:, 2], np.log10(datos_J03_dmo[:, 0]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_dmo[:, 3]))),
            label='Resilient')

minn, maxx = minnmaxx03(0)
plt.ylim(minn - 0.1, maxx + 0.1)
# plt.yticks((19, 20))
print('minn - 0.1, maxx + 0.1', minn - 0.1, maxx + 0.1)
plt.xscale('log')

plt.xlabel(r'log$_{10}$ (D$_\mathrm{Earth}$ [kpc])', fontsize=18)
plt.ylabel(r'log$_{10}$ (J$_{03}$ [GeV$^2$ cm$^{-5}$])', fontsize=18)


plt.subplot(224)

plt.scatter(datos_J03_hyd[:, 2], np.log10(datos_J03_hyd[:, 0]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_hyd[:, 3]))),
            label='Resilient')

plt.xlabel(r'log$_{10}$ (D$_\mathrm{Earth}$ [kpc])', fontsize=18)

plt.ylim(minn - 0.1, maxx + 0.1)
print('minn - 0.1, maxx + 0.1', minn - 0.1, maxx + 0.1)
# plt.yticks((19, 20), labels=('', ''))

plt.xscale('log')


cax, kw = colorbarr.make_axes([ax for ax in axes.flat])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
c2 = plt.colorbar(sm, cax=cax,)
# c2.set_label(r'log$_{10}$(V$_\mathrm{max}$ [km/s])', fontsize=20)
c2.set_label(r'V$_\mathrm{max}$ [km/s])', fontsize=20)
yticks = c2.get_ticks()
c2.set_ticks(yticks)
# c2.set_ticklabels([str(10**i)[:4] for i in yticks])
c2.set_ticklabels(['%.2f' % 10**i for i in yticks])

plt.savefig(path_name + '/DEarthJs.png', bbox_inches='tight')
plt.savefig(path_name + '/DEarthJs.pdf', bbox_inches='tight')

# ---------------- Dgc_Dearth -------------------------------------------------
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True,  # sharey=True,
                         figsize=(12, 9))

plt.subplots_adjust(wspace=0, hspace=0)

vminn = 19
vmaxx = 23
vminn = np.log10(perc_total(0, 5))
vmaxx = np.log10(perc_total(0, 95))
print(vminn, vmaxx)
norm = mcb.Normalize(vminn, vmaxx)

plt.subplot(221)

minn, maxx = minnmaxxS(2)
plt.ylim(minn - 0.1, maxx + 0.1)

plt.title('DMO', size=18)

plt.scatter(datos_Js_dmo[:, 1], np.log10(datos_Js_dmo[:, 3]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_dmo[:, 0]))),
            label='Resilient')


plt.ylabel(r'log$_{10}$ (J$_\mathrm{S}$ [GeV$^2$ cm$^{-5}$])', fontsize=18)


plt.xscale('log')
# plt.yscale('log')

plt.subplot(222)
plt.title('Hydro', size=18)

plt.scatter(datos_Js_hyd[:, 2], np.log10(datos_Js_hyd[:, 0]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_hyd[:, 3]))),
            label='Resilient')


plt.xscale('log')

# plt.yticks((18, 19, 20, 21, 22), labels=('', '', '', '', ''))
plt.ylim(minn - 0.1, maxx + 0.1)


plt.subplot(223)
minn , maxx = minnmaxx03(0)
plt.ylim(minn - 0.1, maxx + 0.1)
# plt.yticks((19, 20))

plt.xscale('log')

plt.scatter(datos_J03_dmo[:, 2], np.log10(datos_J03_dmo[:, 0]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_dmo[:, 3]))),
            label='Resilient')


plt.xlabel(r'log$_{10}$ (D$_\mathrm{Earth}$ [kpc])', fontsize=18)
plt.ylabel(r'log$_{10}$ (J$_{03}$ [GeV$^2$ cm$^{-5}$])', fontsize=18)


plt.subplot(224)

plt.ylim(minn - 0.1, maxx + 0.1)

plt.scatter(datos_J03_hyd[:, 2], np.log10(datos_J03_hyd[:, 0]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_hyd[:, 3]))),
            label='Resilient')


plt.xscale('log')
# plt.yticks((19, 20), labels=('', ''))

plt.xlabel(r'log$_{10}$ (D$_\mathrm{Earth}$ [kpc])', fontsize=18)


cax, kw = colorbarr.make_axes([ax for ax in axes.flat])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
c2 = plt.colorbar(sm, cax=cax,)
# c2.set_label(r'log$_{10}$(V$_\mathrm{max}$ [km/s])', fontsize=20)
c2.set_label(r'V$_\mathrm{max}$ [km/s]', fontsize=20)
yticks = c2.get_ticks()
c2.set_ticks(yticks)
# c2.set_ticklabels([str(10**i)[:4] for i in yticks])
c2.set_ticklabels(['%.2f' % 10**i for i in yticks])

plt.savefig(path_name + '/Dgc_Dearth.png', bbox_inches='tight')
plt.savefig(path_name + '/Dgc_Dearth.pdf', bbox_inches='tight')

# ------------------------ DgcJs ----------------------------------------------
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True,  # sharey=True,
                         figsize=(12, 9))

plt.subplots_adjust(wspace=0, hspace=0)

vminn = 1.
vmaxx = np.log10(100)  # 10 ** 1.5
vminn = np.log10(perc_total(3, 5))
vmaxx = np.log10(perc_total(3, 95))
print('Vmax perc', 10**vminn, 10**vmaxx)
norm = mcb.Normalize(vminn, vmaxx)

plt.subplot(221)

minn, maxx = minnmaxxS(0)
plt.ylim(minn - 0.1, maxx + 0.1)

plt.title('DMO', size=18)
plt.xscale('log')

plt.scatter(datos_Js_dmo[:, 1], np.log10(datos_Js_dmo[:, 0]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_dmo[:, 3]))),
            label='Resilient')

plt.ylabel(r'log$_{10}$ (J$_\mathrm{S}$ [GeV$^2$ cm$^{-5}$])', fontsize=18)

plt.subplot(222)
plt.title('Hydro', size=18)

plt.scatter(datos_Js_hyd[:, 1], np.log10(datos_Js_hyd[:, 0]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_hyd[:, 3]))),
            label='Resilient')

plt.xscale('log')

# plt.yticks((18, 19, 20, 21, 22), labels=('', '', '', '', ''))
plt.ylim(minn - 0.1, maxx + 0.1)

plt.subplot(223)

plt.scatter(datos_J03_dmo[:, 1], np.log10(datos_J03_dmo[:, 0]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_dmo[:, 3]))),
            label='Resilient')

minn, maxx = minnmaxx03(0)
plt.ylim(minn - 0.1, maxx + 0.1)
# plt.yticks((19, 20))

plt.xscale('log')

plt.xlabel(r'log$_{10}$ (D$_\mathrm{GC}$ [kpc])', fontsize=18)
plt.ylabel(r'log$_{10}$ (J$_{03}$ [GeV$^2$ cm$^{-5}$])', fontsize=18)


plt.subplot(224)

plt.xscale('log')

plt.scatter(datos_J03_hyd[:, 1], np.log10(datos_J03_hyd[:, 0]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_hyd[:, 3]))),
            label='Resilient')

# plt.yticks((19, 20), labels=('', ''))
plt.xlabel(r'log$_{10}$ (D$_\mathrm{GC}$ [kpc])', fontsize=18)

plt.ylim(minn - 0.1, maxx + 0.1)

cax, kw = colorbarr.make_axes([ax for ax in axes.flat])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
c2 = plt.colorbar(sm, cax=cax,)
# c2.set_label(r'log$_{10}$(V$_\mathrm{max}$ [km/s])', fontsize=20)
c2.set_label(r'V$_\mathrm{max}$ [km/s]', fontsize=20)
yticks = c2.get_ticks()
c2.set_ticks(yticks)
# c2.set_ticklabels([str(10**i)[:4] for i in yticks])
c2.set_ticklabels(['%.2f' % 10**i for i in yticks])

plt.savefig(path_name + '/DgcJs.png', bbox_inches='tight')
plt.savefig(path_name + '/DgcJs.pdf', bbox_inches='tight')


# ---------------------- VmaxDEarth -------------------------------------------
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True,  # sharey=True,
                         figsize=(12, 9))

plt.subplots_adjust(wspace=0, hspace=0)

vminn = 0.1
vmaxx = 50
vminn = np.log10(perc_total(4, 5))
vmaxx = np.log10(perc_total(4, 95))
print(vminn, vmaxx)
norm = mcb.Normalize(vminn, vmaxx)

print(10**minnmaxxS(4)[0], 10**minnmaxxS(4)[1],
      10**minnmaxx03(4)[0], 10**minnmaxx03(4)[1])

plt.subplot(221)

minn, maxx = minnmaxxS(2)
plt.ylim(minn - 0.1, maxx + 0.1)

plt.title('DMO', size=18)
plt.xscale('log')

plt.scatter(datos_Js_dmo[:, 3], np.log10(datos_Js_dmo[:, 2]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_dmo[:, 4]))),
            label='Resilient')


plt.ylabel(r'log$_{10}$ (D$_\mathrm{Earth}$ [kpc])', fontsize=18)


plt.subplot(222)
plt.title('Hydro', size=18)
plt.xscale('log')

plt.ylim(minn - 0.1, maxx + 0.1)
# plt.yticks((18, 19, 20, 21, 22), labels=('', '', '', '', ''))

plt.scatter(datos_Js_hyd[:, 3], np.log10(datos_Js_hyd[:, 2]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_hyd[:, 4]))),
            label='Resilient')


plt.subplot(223)
minn, maxx = minnmaxx03(2)
plt.ylim(minn - 0.1, maxx + 0.1)
# plt.yticks((19, 20))

plt.xscale('log')

plt.scatter(datos_J03_dmo[:, 3], np.log10(datos_J03_dmo[:, 2]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_dmo[:, 4]))),
            label='Resilient')

plt.ylabel(r'log$_{10}$ (D$_\mathrm{Earth}$ [kpc])', fontsize=18)
plt.xlabel(r'$V_\mathrm{max}$ [km s$^{-1}$]', size=22)


plt.subplot(224)

plt.xscale('log')

plt.scatter(datos_J03_hyd[:, 3], np.log10(datos_J03_hyd[:, 2]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_hyd[:, 4]))),
            label='Resilient')


plt.ylim(minn - 0.1, maxx + 0.1)
# plt.yticks((19, 20), labels=('', ''))

plt.xlabel(r'$V_\mathrm{max}$ [km s$^{-1}$]', size=22)


cax, kw = colorbarr.make_axes([ax for ax in axes.flat])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
c2 = plt.colorbar(sm, cax=cax,)
c2.set_label(r'Angular size', fontsize=20)
yticks = c2.get_ticks()
c2.set_ticks(yticks)
# c2.set_ticklabels([str(10**i)[:4] for i in yticks])
c2.set_ticklabels(['%.2f' % 10**i for i in yticks])

plt.savefig(path_name + '/VmaxDEarth.png', bbox_inches='tight')
plt.savefig(path_name + '/VmaxDEarth.pdf', bbox_inches='tight')

# -------------------- J_hist -------------------------------------------------
fig, _ = plt.subplots(2, 2, figsize=(12, 9))

plt.subplots_adjust(wspace=0, hspace=0)

minS, maxS = minnmaxxS(0)
min03, max03 = minnmaxx03(0)
minn = np.min((minS, min03))
maxx = np.max((maxS, max03))

bines = np.linspace(minn, maxx, 40)

ax1 = plt.subplot(221)

plt.title('DMO', size=18)

plt.hist(np.log10(datos_Js_dmo[:, 0]), log=False,
         label=r'Res', color='k', alpha=0.5,
         bins=bines)

Js95_resi_dmo = np.log10(np.percentile(datos_Js_dmo[:, 0], 5))
print('Js, DMO')
print(Js95_resi_dmo)

plt.axvline(Js95_resi_dmo, color='k')  # , alpha=0.5)

plt.xlim(minn, maxx)
plt.ylim(bottom=0.9, top=4000)

plt.yscale('log')

ax1.tick_params(labelbottom=False)

# plt.annotate(r'J$_S$ 95%', (Js95_frag_dmo, 20), rotation=90, color='k')
# plt.annotate(r'J$_{03}$ 95%', (J0395_frag_dmo, 20),
# rotation=90, color='g', horizontalalignment='right')
plt.legend(title=r'J$_\mathrm{S}$')

ax2 = plt.subplot(222, sharex=ax1, sharey=ax1)
plt.title('Hydro', size=18)
plt.yscale('log')

plt.hist(np.log10(datos_Js_hyd[:, 0]), log=False,
         label=r'Res', color=darkgreen, alpha=0.6,
         bins=bines)

Js95_resi_hyd = np.log10(np.percentile(datos_Js_hyd[:, 0], 5))
print('Js, Hydro')
print(Js95_resi_hyd)

plt.axvline(Js95_resi_hyd, color=darkgreen)  # , alpha=0.6)

# plt.annotate(r'J$_S$ 95%', (Js95_frag_hyd, 20), rotation=90, color='k')
# plt.annotate(r'J$_{03}$ 95%', (Js95_resi_hyd, 20),
# rotation=90, color='g', horizontalalignment='right')

ax2.tick_params(labelleft=False)

plt.legend(title=r'J$_\mathrm{S}$')

plt.subplot(223, sharex=ax1, sharey=ax1)

plt.hist(np.log10(datos_J03_dmo[:, 0]), log=False,
         label=r'Res', color='k', alpha=0.5,
         bins=bines)

J0395_resi_dmo = np.log10(np.percentile(datos_J03_dmo[:, 0], 5))
print('J03, DMO')
print(J0395_resi_dmo)
plt.yscale('log')

plt.axvline(J0395_resi_dmo, color='k')  # , alpha=0.5)

# plt.annotate(r'J$_S$ 95%', (Js95_resi_dmo, 20), rotation=90, color='k')
# plt.annotate(r'J$_{03}$ 95%', (J0395_resi_dmo, 20),
# rotation=90, color='g', horizontalalignment='right')

plt.xlabel(r'log$_{10}$ (J$_\mathrm{factor}$ [GeV$^2$ cm$^{-5}$])',
           fontsize=20)
plt.legend(title=r'J$_{03}$')

ax4 = plt.subplot(224, sharex=ax1, sharey=ax1)

plt.hist(np.log10(datos_J03_hyd[:, 0]), log=False,
         label=r'Res', color=darkgreen, alpha=0.6,
         bins=bines)

J0395_resi_hyd = np.log10(np.percentile(datos_J03_hyd[:, 0], 5))
print('J03, Hydro')
print(J0395_resi_hyd)

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

# -------------- Vmax_hist ----------------------------------------------------

fig, _ = plt.subplots(2, 2, figsize=(12, 9))

plt.subplots_adjust(wspace=0, hspace=0)

minS, maxS = minnmaxxS(3)
min03, max03 = minnmaxx03(3)
minn = 10**np.min((minS, min03))
maxx = 10**np.max((maxS, max03))

bines = np.geomspace(minn, maxx, 30)

ax1 = plt.subplot(221)

plt.title('DMO', size=18)

plt.hist(datos_Js_dmo[:, 3], log=False,
         label=r'Res', color='k', alpha=0.5,
         bins=bines)

plt.xlim(minn, maxx)
plt.ylim(bottom=0.9, top=6000)

plt.yscale('log')

plt.xscale('log')

ax1.tick_params(labelbottom=False)
plt.legend(title=r'J$_\mathrm{S}$')

ax2 = plt.subplot(222, sharex=ax1, sharey=ax1)
plt.title('Hydro', size=18)
plt.yscale('log')

plt.hist(datos_Js_hyd[:, 3], log=False,
         label=r'Res', color=darkgreen, alpha=0.6,
         bins=bines)

ax2.tick_params(labelleft=False)

plt.legend(title=r'J$_\mathrm{S}$')

plt.subplot(223, sharex=ax1, sharey=ax1)

plt.hist(datos_J03_dmo[:, 3], log=False,
         label=r'Res', color='k', alpha=0.5,
         bins=bines)

plt.xlabel(r'$V_\mathrm{max}$ [km s$^{-1}$]', size=20)
plt.legend(title=r'J$_{03}$')

ax4 = plt.subplot(224, sharex=ax1, sharey=ax1)

plt.hist(datos_J03_hyd[:, 3], log=False,
         label=r'Res', color=darkgreen, alpha=0.6,
         bins=bines)

plt.yscale('log')
plt.xlabel(r'$V_\mathrm{max}$ [km s$^{-1}$]', size=20)

plt.legend(title=r'J$_{03}$')

fig.text(0.06, 0.5, 'Number of repops', ha='center',
         va='center', rotation='vertical')

ax4.tick_params(labelleft=False)

plt.savefig(path_name + '/Vmax_hist_geom.png', bbox_inches='tight')
plt.savefig(path_name + '/Vmax_hist_geom.pdf', bbox_inches='tight')

#----------------------- Vmax - J (z==DistEarth) 2x2 --------------------------

fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True,  # sharey=True,
                         figsize=(12, 9))

plt.subplots_adjust(wspace=0, hspace=0)

vminn = 1e-3
vmaxx = np.log10(150)  # 10 ** 1.5
vminn = np.log10(perc_total(2, 5))
vmaxx = np.log10(perc_total(2, 95))
print(vminn, vmaxx)
norm = mcb.Normalize(vminn, vmaxx)

plt.subplot(221)

minn, maxx = minnmaxxS(0)
plt.ylim(minn - 0.1, maxx + 0.1)

plt.title('DMO', size=18)
plt.xscale('log')

plt.scatter(datos_Js_dmo[:, 3], np.log10(datos_Js_dmo[:, 0]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_dmo[:, 2]))),
            label='Resilient')


plt.ylabel(r'log$_{10}$ (J$_\mathrm{S}$ [GeV$^2$ cm$^{-5}$])', fontsize=18)


plt.subplot(222)
plt.title('Hydro', size=18)
plt.xscale('log')

plt.yticks((18, 19, 20, 21, 22), labels=('', '', '', '', ''))

plt.scatter(datos_Js_hyd[:, 3], np.log10(datos_Js_hyd[:, 0]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_hyd[:, 2]))),
            label='Resilient')

plt.ylim(minn - 0.1, maxx + 0.1)

plt.subplot(223)
minn, maxx = minnmaxx03(0)
plt.ylim(minn - 0.1, maxx + 0.1)
plt.yticks((19, 20))

plt.xscale('log')

plt.scatter(datos_J03_dmo[:, 3], np.log10(datos_J03_dmo[:, 0]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_dmo[:, 2]))),
            label='Resilient')


plt.ylabel(r'log$_{10}$ (J$_{03}$ [GeV$^2$ cm$^{-5}$])', fontsize=18)
plt.xlabel(r'$V_\mathrm{max}$ [km s$^{-1}$]', size=22)

plt.subplot(224)

plt.xscale('log')

plt.scatter(datos_J03_hyd[:, 3], np.log10(datos_J03_hyd[:, 0]),
            c='none', lw=2, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_hyd[:, 2]))),
            label='Resilient')


plt.ylim(minn - 0.1, maxx + 0.1)
plt.yticks((19, 20), labels=('', ''))

plt.xlabel(r'$V_\mathrm{max}$ [km s$^{-1}$]', size=22)


cax, kw = colorbarr.make_axes([ax for ax in axes.flat])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
c2 = plt.colorbar(sm, cax=cax,)
# c2.set_label(r'log$_{10}$(D$_\mathrm{Earth}$ [kpc])', fontsize=20)
yticks = c2.get_ticks()
c2.set_label(r'D$_\mathrm{Earth}$ [kpc]', fontsize=20)
c2.set_ticks(yticks)
# c2.set_ticklabels([str(10**i)[:4] for i in yticks])
c2.set_ticklabels(['%4.2f' % 10**i for i in yticks])

plt.savefig(path_name + '/VmaxJs.png', bbox_inches='tight')
plt.savefig(path_name + '/VmaxJs.pdf', bbox_inches='tight')


# ---------------- Dgc_hist ---------------------------------------------------
fig, _ = plt.subplots(2, 2, figsize=(12, 9))

plt.subplots_adjust(wspace=0, hspace=0)
column = 1
xxlabel = r'D$_\mathrm{GC}$ [kpc]'

minS, maxS = minnmaxxS(1)
min03, max03 = minnmaxx03(1)
minn = 10**np.min((minS, min03))
maxx = 10**np.max((maxS, max03))

# bines = np.linspace(minn, maxx, 25)
bines = np.logspace(np.log10(minn), np.log10(maxx), 30)
locc = 2

ax1 = plt.subplot(221)
plt.xlim(0.5, 250)
plt.yscale('log')

plt.title('DMO', size=18)

plt.hist((datos_Js_dmo[:, column]), log=False,
         label=r'Res', color='k', alpha=0.5,
         bins=bines)

plt.axvline(8.5, color='Sandybrown', alpha=1, linestyle='--')
plt.legend(title=r'J$_\mathrm{S}$', loc=locc)
plt.xscale('log')

plt.xlim(minn * 0.9, maxx * 1.1)

ax2 = plt.subplot(222, sharex=ax1, sharey=ax1)
plt.title('Hydro', size=18)
plt.yscale('log')

plt.hist((datos_Js_hyd[:, column]), log=False,
         label=r'Res', color=darkgreen, alpha=0.6,
         bins=bines)

plt.axvline(8.5, color='orange', alpha=1, linestyle='--')
plt.legend(title=r'J$_\mathrm{S}$', loc=locc)
ax2.tick_params(labelleft=False)

plt.subplot(223, sharex=ax1, sharey=ax1)

plt.hist((datos_J03_dmo[:, column]), log=False,
         label=r'Res', color='k', alpha=0.5,
         bins=bines)

plt.axvline(8.5, color='Sandybrown', alpha=1, linestyle='--')

plt.yscale('log')
plt.xlabel(xxlabel, fontsize=20)
plt.legend(title=r'J$_{03}$', loc=locc)

ax4 = plt.subplot(224, sharex=ax1, sharey=ax1)

plt.hist((datos_J03_hyd[:, column]), log=False,
         label=r'Res', color=darkgreen, alpha=0.6,
         bins=bines)

plt.axvline(8.5, color='orange', alpha=1, linestyle='--')

plt.yscale('log')
plt.xlabel(xxlabel, fontsize=20)
plt.legend(title=r'J$_{03}$', loc=locc)

fig.text(0.06, 0.5, 'Number of repops', ha='center',
         va='center', rotation='vertical')
ax4.tick_params(labelleft=False)


plt.savefig(path_name + '/Dgc_hist.png', bbox_inches='tight')
plt.savefig(path_name + '/Dgc_hist.pdf', bbox_inches='tight')

plt.show()
