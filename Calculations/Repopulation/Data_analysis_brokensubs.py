import os

import matplotlib.transforms as trans
import numpy as np
import matplotlib.colorbar as colorbarr
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, newton
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.colors as mcb
import matplotlib.patches as mpatches

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


constraints_bb_2204 = np.loadtxt('../Constraints_2204/Limit_bb.txt')
constraints_tau_2204 = np.loadtxt('../Constraints_2204/Limit_tau.txt')
sigmav_bb_2204 = np.loadtxt('../Constraints_2204/sigmav_bb.txt')
sigmav_tau_2204 = np.loadtxt('../Constraints_2204/sigmav_tau.txt')

sigmav_bb_2204 = sigmav_bb_2204[sigmav_bb_2204[:, 0].argsort()[::], :]
sigmav_tau_2204 = sigmav_tau_2204[sigmav_tau_2204[:, 0].argsort()[::], :]

J03_min95_2204 = 18.9208  # From digitalizing
Js_min95_2204 = 19.4642  # From digitalizing


# def minnmaxxS(i):
#     minn = np.min((
#         np.min(np.log10(datos_Js_frag_dmo[:, i])),
#         np.min(np.log10(datos_Js_frag_hyd[:, i])),
#         np.min(np.log10(datos_Js_resi_dmo[:, i])),
#         np.min(np.log10(datos_Js_resi_hyd[:, i]))
#     ))
#
#     maxx = np.max((
#         np.max(np.log10(datos_Js_frag_dmo[:, i])),
#         np.max(np.log10(datos_Js_frag_hyd[:, i])),
#         np.max(np.log10(datos_Js_resi_dmo[:, i])),
#         np.max(np.log10(datos_Js_resi_hyd[:, i])),
#     ))
#
#     return minn, maxx


darkgreen = (0.024, 0.278, 0.047)
cmap = cm.viridis
colormapp = 'viridis'

def pow_law_old(x, j0, m):
    return 10**j0 * x**m
def pow_law(x, j0, m):
    yy = j0 + m * np.log10(x)
    return yy

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# try:
#     path_name_res = ('/home/porrassa/Desktop/WIMPS_project/'
#                  'Physnet_outputs_repops/2025/'
#                  '/2025_to120_angles')
# except FileNotFoundError:
#     path_name_res = ('/home/saraporras/Desktop/WIMPSproject/'
#                      'compiled_results/'
#                      '2024_resilient_const_to120_rint_SHVFnorm')

path_name_res = '/home/porrassa/Desktop/WIMPS_project/Calculations/' \
                'Repopulation/outputs/test_2025_angles_res/'


path_out = path_name_res + '/figures'
if not os.path.exists(path_name_res):
    os.makedirs(path_name_res)

end_str = '_to120'

datos = np.loadtxt('/home/porrassa/Downloads/'
                   'broken_subh_120_resilient_500reps.txt')

# datos_Js_hyd = np.ones((1, 6))
# datos_Js_dmo = np.ones((1, 6))

print('datos_Js_resi_dmo')
rr_ss = funct_repop.R_s(
    V=datos[:, 4], C=datos[:, 6], cosmo_H_0=67.7)
print(sum(rr_ss > datos[:, 3]))
aaa = rr_ss > datos[:, 3]
datos_Js_resi_dmo = datos [aaa, :]

# print('datos_Js_resi_dmo')
# rr_ss = funct_repop.R_s(
#     V=datos_Js_resi_dmo[:, 3], C=datos_Js_resi_dmo[:, 5], cosmo_H_0=67.7)
# print(sum(rr_ss > datos_Js_resi_dmo[:, 2]))
# aaa = rr_ss > datos_Js_resi_dmo[:, 2]
# datos_Js_resi_dmo = datos_Js_resi_dmo [~aaa, :]
# print(sum(datos_Js_resi_dmo[:, 3]<=1.), np.shape(datos_Js_resi_dmo))
#
# print('datos_Js_resi_hyd')
# rr_ss = funct_repop.R_s(
#     V=datos_Js_resi_hyd[:, 3], C=datos_Js_resi_hyd[:, 5], cosmo_H_0=67.7)
# print(sum(rr_ss > datos_Js_resi_hyd[:, 2]))
# aaa = rr_ss > datos_Js_resi_hyd[:, 2]
# datos_Js_resi_hyd = datos_Js_resi_hyd [~aaa, :]
# print(sum(datos_Js_resi_hyd[:, 3]<=1.), np.shape(datos_Js_resi_hyd))

# ----------------------- Ang size - J (z==Vmax) 2x2 -----------------
def Cv_Mol2021_redshift0(V, c0):
    # Median subhalo concentration depending on its Vmax
    # and its redshift (here z=0)
    # Moline et al. 2110.02097
    #
    # V - max radial velocity of a bound particle in the subhalo [km/s]
    c1 = -0.90368
    c2 = 0.2749
    c3 = -0.028
    ci = [c0, c1, c2, c3]
    return ci[0] * (1 + (sum([ci[i + 1] * np.log10(V) ** (i + 1)
                              for i in range(3)])))


# ----------------------- Ang size - J (z==Vmax) 2x2 -----------------
xx_plot = np.geomspace(0.1, 120)
# fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True,  # sharey=True,
#                          figsize=(7, 9))
fig, axes = plt.subplots(figsize=(8, 8))

plt.subplots_adjust(wspace=0, hspace=0)

legend_elements = [
    Line2D([0], [0], marker='o', color='k', label='Frag',
           markerfacecolor='w', ls='',
           markersize=8, mew=2.5),
    Line2D([0], [0], marker='P', color='w', label='Res',
           markerfacecolor='k', markersize=12)]

x_col = 2
y_col = 4
z_col = 6

x_label = r'$\theta_\mathrm{S}\,\,\left[\mathrm{deg}\right]$'
y1_label = r'log$_{10}$ (J$_\mathrm{S}$ [GeV$^2$ cm$^{-5}$])'
y2_label = r'log$_{10}$ (J$_{03}$ [GeV$^2$ cm$^{-5}$])'

# plt.text(-0.25, 0., r'log$_{10}$ ($J_\mathrm{s}$ [GeV$^2$ cm$^{-5}$])',
#          horizontalalignment='center',
#          verticalalignment='center', transform=axes[0, 0].transAxes,
#          rotation=90)


# vminn = 1e-3
# vmaxx = np.log10(150)  # 10 ** 1.5
vminn = np.log10(np.percentile(datos[:, z_col], 5))  # perc_total(z_col, 5))
vmaxx = np.log10(np.percentile(datos[:, z_col], 95))
print(vminn, vmaxx, 10 ** vminn, 10 ** vmaxx)
norm = mcb.Normalize(vminn, vmaxx)

# minns, maxxs = minnmaxxS(y_col)
minns = 18.2
maxxs = 23.35

minnx = 0.1  # perc_total(x_col, 0) * 0.99
maxxx = 100  # perc_total(x_col, 100) * 1.3

plt.title('DMO', fontsize=20)
plt.ylim(0, 120)
# plt.xlim(minnx, maxxx)
plt.xscale('log')
# plt.yscale('log')
# plt.tick_params('x', labelbottom=False)

plt.xlabel('Dgc (kpc)')
plt.ylabel('Vmax (km/s)')

plt.scatter(datos_Js_resi_dmo[:, x_col],
            datos_Js_resi_dmo[:, y_col],
            c='k', lw=1, marker='x',)

plt.scatter(datos[:, x_col],
            datos[:, y_col],
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(datos[:, z_col]))))


plt.axvline(8.5, linestyle='-', alpha=1, color='Sandybrown', lw=1)
plt.axhline(8.5, linestyle='-', alpha=1, color='Sandybrown', lw=1)

# plt.yticks((19, 20, 21, 22, 23), labels=('19', '', '21', '', '23'))

# aa = plt.subplot(2, 2, 3)
# aa.set_xticks([0.1, 1, 10, 100], labels=('0.1', '1', '10', ''))
#
# aa = plt.subplot(2, 2, 4)
# aa.set_xticks([0.1, 1, 10, 100], labels=('0.1', '1', '10', '100'))

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
c2 = plt.colorbar(sm,# ax=axes,
                  extend='both', spacing='proportional',
                  # location='bottom'
                  )
c2.set_label(r'log10(Cv)', fontsize=20)
yticks = c2.get_ticks()
print(yticks)
c2.ax.tick_params(axis='x', direction='out')
# c2.set_ticklabels([str(10 ** i)[:4] for i in yticks])
# c2.set_ticks(np.log10([5., 10., 20, 50, 80.]),
#              labels=['5', '10', '20', '50', '80'])

# plt.savefig(path_out + '/AngJss_full' + end_str + '.png',
#             bbox_inches='tight')
# plt.savefig(path_out + '/AngJss_full' + end_str + '.pdf',
#             bbox_inches='tight')

## ---------------------------------------------------------------------

# datos_Js_resi_hyd = np.loadtxt(path_name_res + 'Js_hydro_fragile_results.txt')
# datos_Js_resi_dmo = np.loadtxt(path_name_res + 'Js_dmo_fragile_results.txt')

datos_Js_resi_hyd = np.loadtxt('/home/porrassa/Downloads/'
                   'broken_subh_120_resilient_hydro_500reps.txt')
datos_Js_resi_dmo = np.loadtxt('/home/porrassa/Downloads/'
                   'broken_subh_120_resilient_dmo_500reps.txt')

print(np.shape(datos_Js_resi_dmo), np.shape(datos_Js_resi_hyd))

print(np.shape(datos_Js_resi_dmo)[0]/500,
      np.shape(datos_Js_resi_hyd)[0]/500)

# datos_Js_resi_hyd = datos_Js_resi_hyd[datos_Js_resi_hyd[:, 3] > 10., :]
# datos_Js_resi_dmo = datos_Js_resi_dmo[datos_Js_resi_dmo[:, 3] > 10., :]
print(np.shape(datos_Js_resi_dmo), np.shape(datos_Js_resi_hyd))

broken_dmo = datos_Js_resi_dmo[datos_Js_resi_dmo[:, 0] < 1, :]
broken_mhd = datos_Js_resi_hyd[datos_Js_resi_hyd[:, 0] < 1, :]

print('datos_Js_resi_dmo')
rr_ss = funct_repop.R_s(
    V=datos_Js_resi_dmo[:, 3], C=datos_Js_resi_dmo[:, 5], cosmo_H_0=67.7)
print(sum(rr_ss > datos_Js_resi_dmo[:, 2]))
aaa = rr_ss > datos_Js_resi_dmo[:, 2]
datos_Js_resi_dmo_engEarth = datos_Js_resi_dmo [aaa, :]

print('datos_Js_resi_hyd')
rr_ss = funct_repop.R_s(
    V=datos_Js_resi_hyd[:, 3], C=datos_Js_resi_hyd[:, 5], cosmo_H_0=67.7)
print(sum(rr_ss > datos_Js_resi_hyd[:, 2]))
aaa = rr_ss > datos_Js_resi_hyd[:, 2]
datos_Js_resi_hyd_engEarth = datos_Js_resi_hyd [aaa, :]

def perc_total(i, number):
    data = (
        np.concatenate((
            datos_Js_resi_dmo[:, i], datos_Js_resi_hyd[:, i]),
            axis=None))
    return np.percentile(np.log10(data), number)


data_dmo_auriga = np.loadtxt(
    '../Data_subhalo_simulations/dmo_table.txt', skiprows=3)
data_hydro_auriga = np.loadtxt(
    '../Data_subhalo_simulations/hydro_table.txt', skiprows=3)


plt.plot(data_dmo_auriga[:, 3], data_dmo_auriga[:, 1], marker='.', ls='',
           color='k', alpha=0.7)
plt.plot(data_hydro_auriga[:, 3], data_hydro_auriga[:, 1], marker='.', ls='',
           color='green', alpha=0.7)

# ----------------------- Dgc - Vmax (z==Cv) -------------------------------
xx_plot = np.geomspace(0.1, 120)
# fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True,  # sharey=True,
#                          figsize=(7, 9))
fig, axes = plt.subplots(1, 2, figsize=(14, 8))

plt.subplots_adjust(wspace=0, hspace=0)

legend_elements = [
    Line2D([0], [0], marker='o', color='k', label='Frag',
           markerfacecolor='w', ls='',
           markersize=8, mew=2.5),
    Line2D([0], [0], marker='P', color='w', label='Res',
           markerfacecolor='k', markersize=12)]

x_col = 1
y_col = 3
z_col = 5

vminn = perc_total(z_col, 5)
vmaxx = perc_total(z_col, 95)
print(vminn, vmaxx, 10 ** vminn, 10 ** vmaxx)
norm = mcb.Normalize(vminn, vmaxx)

# minns, maxxs = minnmaxxS(y_col)
minns = 8
maxxs = 120

minnx = 0.1  # perc_total(x_col, 0) * 0.99
maxxx = 100  # perc_total(x_col, 100) * 1.3

plt.subplot(121)
plt.title('DMO', fontsize=20)
plt.ylim(minns, maxxs)
# plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.yscale('log')
# plt.tick_params('x', labelbottom=False)

plt.xlabel('Dgc (kpc)')
plt.ylabel('Vmax (km/s)')

plt.scatter(broken_dmo[:, x_col],
            broken_dmo[:, y_col],
            c='k', lw=1, marker='x',)

plt.scatter(datos_Js_resi_dmo[:, x_col],
            datos_Js_resi_dmo[:, y_col],
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_resi_dmo[:, z_col]))))


plt.axhline(30, linestyle='-', alpha=1, color='r', lw=1)
plt.axvline(8.5, linestyle='-', alpha=1, color='Sandybrown', lw=1)

plt.subplot(122)
plt.title('MHD', fontsize=20)
plt.ylim(minns, maxxs)
# plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.yscale('log')
plt.tick_params('y', labelleft=False)

plt.xlabel('Dgc (kpc)')
# plt.ylabel('Vmax (km/s)')

plt.scatter(broken_mhd[:, x_col],
            broken_mhd[:, y_col],
            c='k', lw=1, marker='x',)

plt.scatter(datos_Js_resi_hyd[:, x_col],
            datos_Js_resi_hyd[:, y_col],
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_resi_hyd[:, z_col]))))


plt.axhline(30, linestyle='-', alpha=1, color='r', lw=1)
plt.axvline(8.5, linestyle='-', alpha=1, color='Sandybrown', lw=1)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
c2 = plt.colorbar(sm, ax=axes,
                  extend='both', spacing='proportional',
                  location='bottom'
                  )
c2.set_label(r'log10(Cv)', fontsize=20)
yticks = c2.get_ticks()
print(yticks)
c2.ax.tick_params(axis='x', direction='out')

# plt.savefig(path_out + '/AngJss_full' + end_str + '.png',
#             bbox_inches='tight')
# plt.savefig(path_out + '/AngJss_full' + end_str + '.pdf',
#             bbox_inches='tight')
# plt.show()
# ------------------ Mass - Vmax (z==Cv) ------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

plt.subplots_adjust(wspace=0, hspace=0)

legend_elements = [
    Line2D([0], [0], marker='o', color='k', label='Frag',
           markerfacecolor='w', ls='',
           markersize=8, mew=2.5),
    Line2D([0], [0], marker='P', color='w', label='Res',
           markerfacecolor='k', markersize=12)]

x_col = 1
y_col = 3
z_col = 5

vminn = perc_total(z_col, 5)
vmaxx = perc_total(z_col, 95)
print(vminn, vmaxx, 10 ** vminn, 10 ** vmaxx)
norm = mcb.Normalize(vminn, vmaxx)

# minns, maxxs = minnmaxxS(y_col)
minns = 0.08
maxxs = 130

minnx = 0.1  # perc_total(x_col, 0) * 0.99
maxxx = 100  # perc_total(x_col, 100) * 1.3

mm_plot = np.geomspace(1e1, 1e11)

plt.subplot(121)
plt.title('DMO', fontsize=20)
plt.ylim(minns, maxxs)
# plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.yscale('log')
# plt.tick_params('x', labelbottom=False)

plt.xlabel('Mass (Msun)')
plt.ylabel('Vmax (km/s)')



massdmo = funct_repop.mass_from_Vmax(
    Vmax=datos_Js_resi_dmo[:, 3],
    Rmax=funct_repop.R_max(datos_Js_resi_dmo[:, 3],
                           datos_Js_resi_dmo[:, 5],
                           67.7),
    c200=funct_repop.C200_from_Cv_array(datos_Js_resi_dmo[:, 5]),
    cosmo_G=4.297e-06
)

def power_law(xx, v0, mm):
    return v0 + mm * xx

aaa = curve_fit(power_law,
                ydata=np.log10(datos_Js_resi_dmo[:, y_col]),
                xdata=np.log10(massdmo),
                p0=(7, 0.5))

print('\nData dmo 1 repop')
print(aaa)
plt.plot(mm_plot, 10**power_law(np.log10(mm_plot), aaa[0][0], aaa[0][1]))
print(10**((np.log10(0.1) - aaa[0][0]) / aaa[0][1]),
      10**((np.log10(1) - aaa[0][0]) / aaa[0][1]),
      10**((np.log10(10) - aaa[0][0]) / aaa[0][1]))

plt.scatter(datos_Js_resi_dmo_engEarth[:, x_col],
            datos_Js_resi_dmo_engEarth[:, y_col],
            c='k', lw=1, marker='x',)

plt.scatter(massdmo,
            datos_Js_resi_dmo[:, y_col],
            c='none', lw=1, marker='o',
            edgecolors='blue', alpha=0.5
            # edgecolors=cmap(norm(np.log10(datos_Js_resi_dmo[:, z_col])))
            )

plt.plot(10**data_dmo_auriga[:, 3], data_dmo_auriga[:, 1],
         marker='+', ls='',
           color='k', alpha=0.7, zorder=0)

aaa = curve_fit(power_law,
                ydata=np.log10(data_dmo_auriga[:, 1]),
                xdata=(data_dmo_auriga[:, 3]),
                p0=(7, 0.5))

print('\nData dmo auriga')
print(aaa)
plt.plot(mm_plot, 10**power_law(np.log10(mm_plot), aaa[0][0], aaa[0][1]),
         zorder=10, color='grey')
print(10**((np.log10(0.1) - aaa[0][0]) / aaa[0][1]),
      10**((np.log10(1) - aaa[0][0]) / aaa[0][1]),
      10**((np.log10(10) - aaa[0][0]) / aaa[0][1]))

plt.axhline(0.1, linestyle='-', alpha=1, color='Sandybrown', lw=1)
plt.axhline(1, linestyle='-', alpha=1, color='fuchsia', lw=1)
plt.axhline(10, linestyle='-', alpha=1, color='r', lw=1)

handles = (mpatches.Patch(color='k', label='Auriga', alpha=1),
           mpatches.Patch(color='blue', label='1 repop', alpha=1)
           )

legend11 = plt.legend(handles=handles, handlelength=0.9,
                      loc=2, framealpha=1
                      )

plt.xticks(np.geomspace(10, 1e11, num=11))

plt.subplot(122)
plt.title('MHD', fontsize=20)
plt.ylim(minns, maxxs)
# plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.yscale('log')
plt.tick_params('y', labelleft=False)

plt.xlabel('Mass (Msun)')

masshyo = funct_repop.mass_from_Vmax(
    Vmax=datos_Js_resi_hyd[:, 3],
    Rmax=funct_repop.R_max(datos_Js_resi_hyd[:, 3],
                           datos_Js_resi_hyd[:, 5],
                           67.7),
    c200=funct_repop.C200_from_Cv_array(datos_Js_resi_hyd[:, 5]),
    cosmo_G=4.297e-06
)
aaa = curve_fit(power_law,
                ydata=np.log10(datos_Js_resi_hyd[:, y_col]),
                xdata=np.log10(masshyo),
                p0=(-1., 0.5))

print('Hydro 1 repop')
print(aaa)
plt.plot(mm_plot, 10**power_law(np.log10(mm_plot), aaa[0][0], aaa[0][1]),
         c='brown')
print(10**((np.log10(0.1) - aaa[0][0]) / aaa[0][1]),
      10**((np.log10(1) - aaa[0][0]) / aaa[0][1]),
      10**((np.log10(10) - aaa[0][0]) / aaa[0][1]))

plt.scatter(masshyo,
            datos_Js_resi_hyd[:, y_col],
            c='none', lw=1, marker='o',
            edgecolors='orange', alpha=0.5
            # edgecolors=cmap(norm(np.log10(datos_Js_resi_hyd[:, z_col])))
            )

plt.plot(10**data_hydro_auriga[:, 3], data_hydro_auriga[:, 1],
         marker='+', ls='', color='green', alpha=0.7, zorder=0)
aaa = curve_fit(power_law,
                ydata=np.log10(data_hydro_auriga[:, 1]),
                xdata=(data_hydro_auriga[:, 3]),
                p0=(7, 0.5))

print('\nData hydro auriga')
print(aaa)
plt.plot(mm_plot, 10**power_law(np.log10(mm_plot), aaa[0][0], aaa[0][1]),
         zorder=10, color='limegreen')
print(10**((np.log10(0.1) - aaa[0][0]) / aaa[0][1]),
      10**((np.log10(1) - aaa[0][0]) / aaa[0][1]),
      10**((np.log10(10) - aaa[0][0]) / aaa[0][1]))

plt.axhline(0.1, linestyle='-', alpha=1, color='Sandybrown', lw=1)
plt.axhline(1, linestyle='-', alpha=1, color='fuchsia', lw=1)
plt.axhline(10, linestyle='-', alpha=1, color='r', lw=1)

handles = (mpatches.Patch(color='green', label='Auriga', alpha=1),
           mpatches.Patch(color='orange', label='1 repop', alpha=1)
           )

legend11 = plt.legend(handles=handles, handlelength=0.9,
                      loc=2, framealpha=1
                      )

plt.xticks(np.geomspace(10, 1e11, num=11))

# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# c2 = plt.colorbar(sm, ax=axes,
#                   extend='both', spacing='proportional',
#                   location='bottom'
#                   )
# c2.set_label(r'log10(Cv)', fontsize=20)
# yticks = c2.get_ticks()
# print(yticks)
# c2.ax.tick_params(axis='x', direction='out')

# plt.savefig(path_out + '/massVmax_1kms' + end_str + '.png',
#             bbox_inches='tight')
# plt.savefig(path_out + '/massVmax_1kms' + end_str + '.pdf',
#             bbox_inches='tight')


# -----------------------------------------------------------------------
plt.figure()

data_dmo_auriga = np.loadtxt(
    '../Data_subhalo_simulations/dmo_table.txt', skiprows=3)
data_hydro_auriga = np.loadtxt(
    '../Data_subhalo_simulations/hydro_table.txt', skiprows=3)


plt.plot(data_dmo_auriga[:, 3], data_dmo_auriga[:, 1], marker='.', ls='',
           color='k', alpha=0.7)
plt.plot(data_hydro_auriga[:, 3], data_hydro_auriga[:, 1], marker='.', ls='',
           color='green', alpha=0.7)

dmo_1repop_resilient = ([-1.41533305,  0.30017893],
              [[ 6.52995982e-07, -1.24255765e-07],
               [-1.24255765e-07,  2.38468146e-08]])

hydro_1repop_resilient = ([-1.37849246,  0.28964565],
                [[ 2.08162682e-06, -3.93735005e-07],
                 [-3.93735005e-07,  7.50595187e-08]])

dmo_1repop_fragile = ([-1.41424174,  0.29998525],
              [[ 8.27998008e-07, -1.57559874e-07],
               [-1.57559874e-07,  3.02394416e-08]])

hydro_1repop_fragile = ([-1.37708588,  0.28942203],
                [[ 3.36584890e-06, -6.36824522e-07],
                 [-6.36824522e-07,  1.21437127e-07]])

dmo_auriga = ([-1.8367626 ,  0.37006702],
              [[ 3.03227778e-05, -4.59489685e-06],
               [-4.59489685e-06,  7.00644753e-07]])
hydro_auriga = ([-1.84582279,  0.36581809],
                [[ 5.99520326e-05, -9.18097702e-06],
                 [-9.18097702e-06,  1.41505553e-06]])

plt.plot(mm_plot, 10**power_law(
    np.log10(mm_plot), dmo_1repop_resilient[0][0], dmo_1repop_resilient[0][1]),
         c='k', ls='--', label='1 repop resilient')

plt.plot(mm_plot, 10**power_law(
    np.log10(mm_plot), hydro_1repop_resilient[0][0], hydro_1repop_resilient[0][1]),
         c='limegreen', ls='--')

plt.plot(mm_plot, 10**power_law(
    np.log10(mm_plot), dmo_1repop_fragile[0][0], dmo_1repop_fragile[0][1]),
         c='k', ls='-.', label='1 repop fragile')

plt.plot(mm_plot, 10**power_law(
    np.log10(mm_plot), hydro_1repop_fragile[0][0], hydro_1repop_fragile[0][1]),
         c='limegreen', ls='-.')

plt.plot(mm_plot, 10**power_law(
    np.log10(mm_plot), dmo_auriga[0][0], dmo_auriga[0][1]),
         c='k', ls='-', label='Auriga')

plt.plot(mm_plot, 10**power_law(
    np.log10(mm_plot), hydro_auriga[0][0], hydro_auriga[0][1]),
         c='limegreen', ls='-')

plt.xscale('log')
plt.yscale('log')

plt.show()
