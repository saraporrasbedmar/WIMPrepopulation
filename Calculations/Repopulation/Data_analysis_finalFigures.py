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

to8 = True


# path_name = ('/home/porrassa/Desktop/WIMPS_project/'
#                  'Physnet_outputs_repops/2024/'
#                  '/compiled_results_allRoche/'
#                  'final_2024_8max'
#                  )
# path_name_res = ('/home/porrassa/Desktop/WIMPS_project/'
#                      'Physnet_outputs_repops/2024/'
#                      '/compiled_results_allRoche/'
#                      'final_2024_8max')

# path_name = ('/home/porrassa/Desktop/WIMPS_project/'
#                  'Physnet_outputs_repops/2025'
#                  '/2025_to8_angles'
#                  )
# path_name_res = ('/home/porrassa/Desktop/WIMPS_project/'
#                  'Physnet_outputs_repops/2025'
#                  '/2025_to8_angles')

print(os.getcwd())
# print(os.listdir(path_name))
final_size = (500, 1, 6)
plot_res = True
plot_frag = True

end_str = '_to8'
'''
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

# datos_Js_frag_hyd = np.loadtxt(path_name +
#                                '/Js_hydro_fragile_results.txt'
#                                )
# datos_Js_frag_dmo = np.loadtxt(path_name +
#                                '/Js_dmo_fragile_results.txt')
#
# datos_J03_frag_hyd = np.loadtxt(path_name +
#                                 '/J03_hydro_fragile_results.txt')
# datos_J03_frag_dmo = np.loadtxt(path_name +
#                                 '/J03_dmo_fragile_results.txt'
#                                 )
#
# datos_Js_resi_hyd = np.loadtxt(path_name +
#                                '/Js_hydro_resilient_results.txt')
# datos_Js_resi_dmo = np.loadtxt(path_name +
#                                '/Js_dmo_resilient_results.txt'
#                                )
#
# datos_J03_resi_hyd = np.loadtxt(path_name +
#                                 '/J03_hydro_resilient_results.txt')
# datos_J03_resi_dmo = np.loadtxt(path_name +
#                                 '/J03_dmo_resilient_results.txt')


rr_ss = funct_repop.R_s(
    V=datos_Js_frag_dmo[:, 3], C=datos_Js_frag_dmo[:, 5], cosmo_H_0=67.7)
print(sum(rr_ss > datos_Js_frag_dmo[:, 2]))
print('datos_Js_frag_dmo')
print(sum(datos_Js_frag_dmo[:, 3]<1.), np.shape(datos_Js_frag_dmo))

rr_ss = funct_repop.R_s(
    V=datos_Js_frag_hyd[:, 3], C=datos_Js_frag_hyd[:, 5], cosmo_H_0=67.7)
print(sum(rr_ss > datos_Js_frag_hyd[:, 2]))
print('datos_Js_frag_hyd')
print(sum(datos_Js_frag_hyd[:, 3]<1.), np.shape(datos_Js_frag_hyd))

rr_ss = funct_repop.R_s(
    V=datos_Js_resi_dmo[:, 3], C=datos_Js_resi_dmo[:, 5], cosmo_H_0=67.7)
print(sum(rr_ss > datos_Js_resi_dmo[:, 2]))
print('datos_Js_resi_dmo')
print(sum(datos_Js_resi_dmo[:, 3]<1.), np.shape(datos_Js_resi_dmo))

rr_ss = funct_repop.R_s(
    V=datos_Js_resi_hyd[:, 3], C=datos_Js_resi_hyd[:, 5], cosmo_H_0=67.7)
aaa = rr_ss > datos_Js_resi_hyd[:, 2]
print(sum(rr_ss > datos_Js_resi_hyd[:, 2]))
print(datos_Js_resi_hyd[aaa, :])
datos_Js_resi_hyd = datos_Js_resi_hyd [~aaa, :]
print('datos_Js_resi_hyd')
print(sum(datos_Js_resi_hyd[:, 3]<1.), np.shape(datos_Js_resi_hyd))


rr_ss = funct_repop.R_s(
    V=datos_J03_frag_dmo[:, 3], C=datos_J03_frag_dmo[:, 5], cosmo_H_0=67.7)
print(sum(rr_ss > datos_J03_frag_dmo[:, 2]))
print('datos_J03_frag_dmo')
print(sum(datos_J03_frag_dmo[:, 3]<1.), np.shape(datos_J03_frag_dmo))

rr_ss = funct_repop.R_s(
    V=datos_J03_frag_hyd[:, 3], C=datos_J03_frag_hyd[:, 5], cosmo_H_0=67.7)
print(sum(rr_ss > datos_J03_frag_hyd[:, 2]))
print('datos_J03_frag_hyd')
print(sum(datos_J03_frag_hyd[:, 3]<1.), np.shape(datos_J03_frag_hyd))

rr_ss = funct_repop.R_s(
    V=datos_J03_resi_dmo[:, 3], C=datos_J03_resi_dmo[:, 5], cosmo_H_0=67.7)
print(sum(rr_ss > datos_J03_resi_dmo[:, 2]))
print('datos_J03_resi_dmo')
print(sum(datos_J03_resi_dmo[:, 3]<1.), np.shape(datos_J03_resi_dmo))

rr_ss = funct_repop.R_s(
    V=datos_J03_resi_hyd[:, 3], C=datos_J03_resi_hyd[:, 5], cosmo_H_0=67.7)
print(sum(rr_ss > datos_J03_resi_hyd[:, 2]))
aaa = rr_ss > datos_J03_resi_hyd[:, 2]
datos_J03_resi_hyd = datos_J03_resi_hyd [~aaa, :]
print('datos_J03_resi_hyd')
print(sum(datos_J03_resi_hyd[:, 3]<1.), np.shape(datos_J03_resi_hyd))

'''
constraints_bb_2204 = np.loadtxt('../Constraints_2204/Limit_bb.txt')
constraints_tau_2204 = np.loadtxt('../Constraints_2204/Limit_tau.txt')
sigmav_bb_2204 = np.loadtxt('../Constraints_2204/sigmav_bb.txt')
sigmav_tau_2204 = np.loadtxt('../Constraints_2204/sigmav_tau.txt')

sigmav_bb_2204 = sigmav_bb_2204[sigmav_bb_2204[:, 0].argsort()[::], :]
sigmav_tau_2204 = sigmav_tau_2204[sigmav_tau_2204[:, 0].argsort()[::], :]

J03_min95_2204 = 18.9208  # From digitalizing
Js_min95_2204 = 19.4642  # From digitalizing


darkgreen = (0.024, 0.278, 0.047)
cmap = cm.viridis
colormapp = 'viridis'

'''
path_name_res = path_name_res + '/figures'
if not os.path.exists(path_name_res):
    os.makedirs(path_name_res)


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


def pow_law_old(x, j0, m):
    return 10**j0 * x**m
def pow_law(x, j0, m):
    yy = j0 + m * np.log10(x)
    return yy

# ------------------------ Vmax ang size -----------------------------

fig, axes = plt.subplots(nrows=2, ncols=2,  # sharey=True,  # sharey=True,
                         figsize=(7, 9))

plt.subplots_adjust(wspace=0, hspace=0)

x_col = 3
y_col = 4
z_col = 2

x_label = r'$V_\mathrm{max}$ $\left[\mathrm{km}\,\,\mathrm{s}^{-1}\right]$'

plt.text(-0.25, 0., r'$\theta\,\,\left[\mathrm{deg}\right]$',
         horizontalalignment='center',
         verticalalignment='center', transform=axes[0, 0].transAxes,
         rotation=90)
plt.text(1., 0.92, r'Fragile',
         horizontalalignment='center',
         verticalalignment='top', transform=axes[0, 0].transAxes,
         backgroundcolor='w')
plt.text(1., 0.18, r'Resilient',
         horizontalalignment='center',
         verticalalignment='top', transform=axes[1, 0].transAxes,
         backgroundcolor='w')

# vminn = 1e-3
# vmaxx = np.log10(150)  # 10 ** 1.5
vminn = np.log10(perc_total(z_col, 5))
vmaxx = np.log10(perc_total(z_col, 95))
print(vminn, vmaxx, 10 ** vminn, 10 ** vmaxx)
norm = mcb.Normalize(vminn, vmaxx)

minns, maxxs = minnmaxx03(y_col)
minns = 0.02
maxxs = 100

minnx = 0.09  # perc_total(x_col, 0) * 0.99
maxxx = perc_total(x_col, 100) * 1.5

plt.subplot(221)
plt.title('DMO', fontsize=20)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.yscale('log')
plt.tick_params('x', labelbottom=False)

plt.scatter(datos_J03_frag_dmo[:, x_col],
            (datos_J03_frag_dmo[:, y_col]),
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_frag_dmo[:, z_col]))))

plt.subplot(223)

plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(x_label, size=20)

plt.scatter(datos_J03_resi_dmo[:, x_col],
            (datos_J03_resi_dmo[:, y_col]),
            c=np.log10(datos_J03_resi_dmo[:, z_col]),
            marker='+', s=100, linewidths=1,
            cmap=colormapp, vmin=vminn, vmax=vmaxx)

plt.subplot(222)

plt.title('MHD', fontsize=20)

plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.yscale('log')
plt.tick_params('y', labelleft=False)
plt.tick_params('x', labelbottom=False)

plt.scatter(datos_J03_frag_hyd[:, x_col],
            (datos_J03_frag_hyd[:, y_col]),
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_frag_hyd[:, z_col]))))

plt.subplot(224)

plt.tick_params('y', labelleft=False)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(x_label, size=20)

plt.scatter(datos_J03_resi_hyd[:, x_col],
            (datos_J03_resi_hyd[:, y_col]),
            c=np.log10(datos_J03_resi_hyd[:, z_col]),
            marker='+', s=100, linewidths=1,
            cmap=colormapp, vmin=vminn, vmax=vmaxx)

aa = plt.subplot(2, 2, 1)
aa.set_yticks([0.1, 1, 10, 100], labels=('0.1', '1', '10', '100'))

aa = plt.subplot(2, 2, 2)
aa.set_yticks([0.1, 1, 10, 100], labels=[])

aa = plt.subplot(2, 2, 3)
aa.set_xticks([0.1, 1, 10], labels=('0.1', '1', ''))
aa.set_yticks([0.1, 1, 10, 100], labels=('0.1', '1', '10', '100'))

aa = plt.subplot(2, 2, 4)
aa.set_xticks([0.1, 1, 10], labels=('0.1', '1', '10'))
aa.set_yticks([0.1, 1, 10, 100], labels=[])

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
c2 = plt.colorbar(sm, ax=axes,
                  extend='both', spacing='proportional',
                  location='bottom')
c2.ax.tick_params(axis='x', direction='out')
c2.set_label(r'D$_\mathrm{Earth}$ [kpc]', fontsize=20)
yticks = c2.get_ticks()
# c2.set_ticklabels([str(10 ** i)[:4] for i in yticks])
c2.set_ticks(np.log10([0.5, 1., 2, 5., 10., 25]),
             labels=['0.5', '1', '2', '5', '10', '25'])

plt.savefig(path_name_res + '/VmaxAng' + end_str + '.png',
            bbox_inches='tight')
plt.savefig(path_name_res + '/VmaxAng' + end_str + '.pdf',
            bbox_inches='tight')
plt.show()

# ----------------------- Ang size - J (z==Vmax) 2x4 -----------------

xx_plot = np.geomspace(0.01, 100)

fig, axes = plt.subplots(nrows=2, ncols=4, sharey=True,
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

x_label = r'$\theta_\mathrm{S}\,\,\left[\mathrm{deg}\right]$'
y1_label = r'log$_{10}$ (J$_\mathrm{factor}$ [GeV$^2$ cm$^{-5}$])'
# y2_label = r'log$_{10}$ (J$_{03}$ [GeV$^2$ cm$^{-5}$])'

# vminn = 1e-3
# vmaxx = np.log10(150)  # 10 ** 1.5
vminn = 0  # np.log10(perc_total(z_col, 5))
vmaxx = np.log10(perc_total(z_col, 95))
print(vminn, vmaxx)
norm = mcb.Normalize(vminn, vmaxx)

minns, maxxs = minnmaxxS(y_col)
minns = 17.5
maxxs = 22.5

minnx = perc_total(x_col, 0) * 0.7
maxxx = 92  # perc_total(x_col, 100) * 1.3

plt.text(1., 0.92, r'Fragile, J$_\mathrm{s}$',
         horizontalalignment='center',
         verticalalignment='top', transform=axes[0, 0].transAxes,
         backgroundcolor='w')
plt.text(1., 0.18, r'Resilient, J$_\mathrm{s}$',
         horizontalalignment='center',
         verticalalignment='top', transform=axes[1, 0].transAxes,
         backgroundcolor='w')
plt.text(1., 0.92, r'Fragile, J$_\mathrm{03}$',
         horizontalalignment='center',
         verticalalignment='top', transform=axes[0, 2].transAxes,
         backgroundcolor='w')
plt.text(0.96, 0.18, r'Resilient, J$_\mathrm{03}$',
         horizontalalignment='center',
         verticalalignment='top', transform=axes[1, 2].transAxes,
         backgroundcolor='w')

plt.text(-0.27, 0., y1_label,
         horizontalalignment='center',
         verticalalignment='center', transform=axes[0, 0].transAxes,
         rotation=90)

plt.subplot(241)

plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')

plt.title('DMO', size=20)

plt.scatter(datos_Js_frag_dmo[:, x_col],
            np.log10(datos_Js_frag_dmo[:, y_col]),
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_frag_dmo[:, z_col]))))

# aaa, cov = curve_fit(pow_law,
#                      xdata=datos_Js_frag_dmo[:, x_col],
#                      ydata=datos_Js_frag_dmo[:, y_col],
#                      p0=(18, 0.6)
# )
aaa, cov = curve_fit(pow_law,
                     xdata=datos_Js_frag_dmo[:, x_col],
                     ydata=np.log10(datos_Js_frag_dmo[:, y_col]),
                     p0=(18, 0.6)
)
print('datos_Js_frag_dmo')
for i in range(len(aaa)):
    print(aaa[i], np.sqrt(np.diag(cov))[i])

plt.plot(xx_plot, (pow_law(xx_plot, j0=aaa[0], m=aaa[1])),
         zorder=0, ls=':', c='gray', alpha=0.5)


plt.subplot(242)

plt.title('MHD', size=20)
plt.tick_params('y', labelleft=False)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.axvspan(0.935 * maxxx, maxxx, color='k')

plt.scatter(datos_Js_frag_hyd[:, x_col],
            np.log10(datos_Js_frag_hyd[:, y_col]),
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_frag_hyd[:, z_col]))))

# aaa, cov = curve_fit(pow_law,
#                      xdata=datos_Js_frag_hyd[:, x_col],
#                      ydata=datos_Js_frag_hyd[:, y_col],
#                      p0=(18, 0.6)
# )
aaa, cov = curve_fit(pow_law,
                     xdata=datos_Js_frag_hyd[:, x_col],
                     ydata=np.log10(datos_Js_frag_hyd[:, y_col]),
                     p0=(18, 0.6)
)
print('datos_Js_frag_hyd')
for i in range(len(aaa)):
    print(aaa[i], np.sqrt(np.diag(cov))[i])

plt.plot(xx_plot, (pow_law(xx_plot, j0=aaa[0], m=aaa[1])),
         zorder=0, ls=':', c='gray', alpha=0.5)

plt.subplot(243)


# plt.text(1., 1.15, r'J$_\mathrm{03}$', horizontalalignment='center',
#          verticalalignment='bottom', transform=axes[0, 2].transAxes,
#          size=22)
plt.title('DMO', size=20)

plt.axvline(0.15, c='grey', alpha=0.5, zorder=0, ls='-')

plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.tick_params('y', labelleft=False)
plt.axvspan(minnx, minnx * 1.04, color='k')

plt.scatter(datos_J03_frag_dmo[:, x_col],
            np.log10(datos_J03_frag_dmo[:, y_col]),
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_frag_dmo[:, z_col]))))

# aaa, cov = curve_fit(pow_law,
#                      xdata=datos_J03_frag_dmo[:, x_col],
#                      ydata=datos_J03_frag_dmo[:, y_col],
#                      p0=(18, 0.6)
# )
aaa, cov = curve_fit(pow_law,
                     xdata=datos_J03_frag_dmo[:, x_col],
                     ydata=np.log10(datos_J03_frag_dmo[:, y_col]),
                     p0=(18, 0.6)
)
print('datos_J03_frag_dmo')
for i in range(len(aaa)):
    print(aaa[i], np.sqrt(np.diag(cov))[i])

plt.subplot(244)
plt.title('MHD', size=20)
plt.tick_params('y', labelleft=False)
plt.tick_params('x', labelbottom=False)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')

plt.axvline(0.15, c='grey', alpha=0.5, zorder=0, ls='-')

plt.scatter(datos_J03_frag_hyd[:, x_col],
            np.log10(datos_J03_frag_hyd[:, y_col]),
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_frag_hyd[:, z_col]))))

# aaa, cov = curve_fit(pow_law,
#                      xdata=datos_J03_frag_hyd[:, x_col],
#                      ydata=datos_J03_frag_hyd[:, y_col],
#                      p0=(18, 0.6)
# )
aaa, cov = curve_fit(pow_law,
                     xdata=datos_J03_frag_hyd[:, x_col],
                     ydata=np.log10(datos_J03_frag_hyd[:, y_col]),
                     p0=(18, 0.6)
)
print('datos_J03_frag_hyd')
for i in range(len(aaa)):
    print(aaa[i], np.sqrt(np.diag(cov))[i])

print('percentage of extended subhalos')
print(sum(datos_J03_frag_hyd[:, x_col]<0.15), np.shape(datos_J03_frag_hyd[:, x_col]))
plt.subplot(245)

plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')

# plt.ylabel(y2_label, fontsize=16)
plt.xlabel(x_label, size=20)

plt.scatter(datos_Js_resi_dmo[:, x_col],
            np.log10(datos_Js_resi_dmo[:, y_col]),
            c=np.log10(datos_Js_resi_dmo[:, z_col]),
            marker='+', s=100, linewidths=1,
            cmap=colormapp, vmin=vminn, vmax=vmaxx)

# aaa, cov = curve_fit(pow_law,
#                      xdata=datos_Js_resi_dmo[:, x_col],
#                      ydata=datos_Js_resi_dmo[:, y_col],
#                      p0=(18, 0.6)
# )
aaa, cov = curve_fit(pow_law,
                     xdata=datos_Js_resi_dmo[:, x_col],
                     ydata=np.log10(datos_Js_resi_dmo[:, y_col]),
                     p0=(18, 0.6)
)
print('datos_Js_resi_dmo')
for i in range(len(aaa)):
    print(aaa[i], np.sqrt(np.diag(cov))[i])

plt.plot(xx_plot, (pow_law(xx_plot, j0=aaa[0], m=aaa[1])),
         zorder=0, ls=':', c='gray', alpha=0.5)

plt.subplot(246)

plt.tick_params('y', labelleft=False)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.axvspan(0.935 * maxxx, maxxx, color='k')

plt.xlabel(x_label, size=20)

plt.scatter(datos_Js_resi_hyd[:, x_col],
            np.log10(datos_Js_resi_hyd[:, y_col]),
            c=np.log10(datos_Js_resi_hyd[:, z_col]),
            marker='+', s=100, linewidths=1,
            cmap=colormapp, vmin=vminn, vmax=vmaxx)

ddd = np.argsort(datos_Js_resi_hyd[:, y_col])
datos_Js_resi_hyd = datos_Js_resi_hyd[ddd, :]
# aaa, cov = curve_fit(pow_law,
#                      xdata=datos_Js_resi_hyd[:-2, x_col],
#                      ydata=datos_Js_resi_hyd[:-2, y_col],
#                      p0=(19.4, 1.1)
# )
aaa, cov = curve_fit(pow_law,
                     xdata=datos_Js_resi_hyd[:-2, x_col],
                     ydata=np.log10(datos_Js_resi_hyd[:-2, y_col]),
                     p0=(19.4, 1.1)
)
print('datos_Js_resi_hyd')
for i in range(len(aaa)):
    print(aaa[i], np.sqrt(np.diag(cov))[i])
print(sum((datos_Js_resi_hyd[:, y_col]
           -pow_law(datos_Js_resi_hyd[:, x_col],
                    j0=aaa[0], m=aaa[1]))**2.))
print(sum((datos_Js_resi_hyd[:, y_col]
           -pow_law(datos_Js_resi_hyd[:, x_col],
                    j0=aaa[0]+0.1, m=aaa[1]))**2.))

plt.plot(xx_plot, (pow_law(xx_plot, j0=aaa[0], m=aaa[1])),
         zorder=0, ls=':', c='gray', alpha=0.5)

plt.subplot(247)

plt.tick_params('y', labelleft=False)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.axvspan(minnx, minnx * 1.04, color='k')

plt.axvline(0.15, c='grey', alpha=0.5, zorder=0, ls='-')

plt.xlabel(x_label, size=20)

plt.scatter(datos_J03_resi_dmo[:, x_col],
            np.log10(datos_J03_resi_dmo[:, y_col]),
            c=np.log10(datos_J03_resi_dmo[:, z_col]),
            marker='+', s=100, linewidths=1,
            cmap=colormapp, vmin=vminn, vmax=vmaxx)

# aaa, cov = curve_fit(pow_law,
#                      xdata=datos_J03_resi_dmo[:, x_col],
#                      ydata=datos_J03_resi_dmo[:, y_col],
#                      p0=(18, 0.6)
# )
aaa, cov = curve_fit(pow_law,
                     xdata=datos_J03_resi_dmo[:, x_col],
                     ydata=np.log10(datos_J03_resi_dmo[:, y_col]),
                     p0=(18, 0.6)
)
print('datos_J03_resi_dmo')
for i in range(len(aaa)):
    print(aaa[i], np.sqrt(np.diag(cov))[i])

plt.subplot(248)

plt.tick_params('y', labelleft=False)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')

plt.xlabel(x_label, size=20)

plt.axvline(0.15, c='grey', alpha=0.5, zorder=0, ls='-')

plt.scatter(datos_J03_resi_hyd[:, x_col],
            np.log10(datos_J03_resi_hyd[:, y_col]),
            c=np.log10(datos_J03_resi_hyd[:, z_col]),
            marker='+', s=100, linewidths=1,
            cmap=colormapp, vmin=vminn, vmax=vmaxx)

# aaa, cov = curve_fit(pow_law,
#                      xdata=datos_J03_resi_hyd[:, x_col],
#                      ydata=datos_J03_resi_hyd[:, y_col],
#                      p0=(18, 0.6)
# )
aaa, cov = curve_fit(pow_law,
                     xdata=datos_J03_resi_hyd[:, x_col],
                     ydata=np.log10(datos_J03_resi_hyd[:, y_col]),
                     p0=(18, 0.6)
)
print('datos_J03_resi_hyd')
for i in range(len(aaa)):
    print(aaa[i], np.sqrt(np.diag(cov))[i])

for ii in range(5, 9):
    plt.subplot(2, 4, ii)
    plt.xticks([0.1, 1, 10], labels=('0.1', '1', '10'))

for ii in range(1, 5):
    plt.subplot(2, 4, ii)
    plt.yticks((18, 19, 20, 21, 22),
               labels=('18', '', '20', '', '22'))

cax, kw = colorbarr.make_axes([ax for ax in axes.flat])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
c2 = plt.colorbar(sm, cax=cax, extend='both')
c2.set_label(r'$V_\mathrm{max}$ [km s$^{-1}$]', fontsize=20)
yticks = c2.get_ticks()
c2.ax.tick_params(axis='y', direction='out')
print(yticks)
# c2.set_ticklabels([str(10 ** i)[:4] for i in yticks])
c2.set_ticks(np.log10([1., 2., 4, 6]),
             labels=['1', '2', '4', '6'])

plt.savefig(path_name_res + '/AngJs_full' + end_str + '.png',
            bbox_inches='tight')
plt.savefig(path_name_res + '/AngJs_full' + end_str + '.pdf',
            bbox_inches='tight')


fig, ax = plt.subplots(1, 2, figsize=(14, 5))
plt.subplot(121)
data_subplot = datos_Js_frag_dmo
plt.xscale('log')
plt.xlabel(x_label, size=20)
plt.ylabel(r'log$_{10}$ (J$_\mathrm{S}$ [GeV$^2$ cm$^{-5}$])', size=20)

plt.title(r'DMO', size=20, pad=15, fontweight='bold')

plt.scatter(data_subplot[:, x_col],
            np.log10(data_subplot[:, y_col]),
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(data_subplot[:, z_col]))))

aaa, cov = curve_fit(pow_law,
                     xdata=data_subplot[:, x_col],
                     ydata=np.log10(data_subplot[:, y_col]),
                     p0=(18, 0.6)
)
for i in range(len(aaa)):
    print(aaa[i], np.sqrt(np.diag(cov))[i])

plt.plot(xx_plot, (pow_law(xx_plot, j0=aaa[0], m=aaa[1])),
         zorder=0, ls=':', c='gray', alpha=0.5)

plt.xticks([0.1, 1, 10], labels=('0.1', '1', '10'))

plt.xlim(np.min(data_subplot[:, x_col])*0.8,
         np.max(data_subplot[:, x_col])*1.2)
plt.ylim(np.min(np.log10(data_subplot[:, y_col]))-0.2,
         np.max(np.log10(data_subplot[:, y_col]))+0.2)

plt.subplot(122)
data_subplot = datos_Js_frag_hyd
plt.xscale('log')
plt.xlabel(x_label, size=20)

plt.title(r'MHD', size=20, pad=15, fontweight='bold')

plt.scatter(data_subplot[:, x_col],
            np.log10(data_subplot[:, y_col]),
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(data_subplot[:, z_col]))))

plt.xticks([0.1, 1, 10], labels=('0.1', '1', '10'))
plt.xlim(np.min(data_subplot[:, x_col])*0.8,
         np.max(data_subplot[:, x_col])*1.2)
plt.ylim(np.min(np.log10(data_subplot[:, y_col]))-0.2,
         np.max(np.log10(data_subplot[:, y_col]))+0.2)

aaa, cov = curve_fit(pow_law,
                     xdata=data_subplot[:, x_col],
                     ydata=np.log10(data_subplot[:, y_col]),
                     p0=(18, 0.6)
)
print(aaa, np.sqrt(np.diag(cov)))
plt.plot(xx_plot, (pow_law(xx_plot, j0=aaa[0], m=aaa[1])),
         zorder=0, ls=':', c='gray', alpha=0.5)

cax, kw = colorbarr.make_axes([axi for axi in ax.flat])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
c2 = plt.colorbar(sm, cax=cax, extend='both')
c2.set_label(r'$V_\mathrm{max}$ [km s$^{-1}$]', fontsize=20)
yticks = c2.get_ticks()
c2.ax.tick_params(axis='y', direction='out')
c2.set_ticks(np.log10([1., 2., 4, 6]),
             labels=['1', '2', '4', '6'])

print('%.3f\n%.3f' % (aaa[0], aaa[1]))

plt.savefig(path_name_res + '/datos_Js_frag' + end_str + '.png',
            bbox_inches='tight', dpi=400)
plt.savefig(path_name_res + '/datos_Js_frag' + end_str + '.pdf',
            bbox_inches='tight')

# ----------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
plt.subplot(121)
plt.xscale('log')
data_subplot = datos_J03_frag_dmo
plt.xlabel(x_label, size=20)
plt.ylabel(r'log$_{10}$ (J$_\mathrm{03}$ [GeV$^2$ cm$^{-5}$])', size=20)

plt.title(r'DMO', size=20, pad=15, fontweight='bold')

plt.axvline(0.15, c='grey', alpha=0.5, zorder=0, ls='-')

plt.scatter(datos_J03_frag_dmo[:, x_col],
            np.log10(datos_J03_frag_dmo[:, y_col]),
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_frag_dmo[:, z_col]))))

aaa, cov = curve_fit(pow_law,
                     xdata=data_subplot[:, x_col],
                     ydata=np.log10(data_subplot[:, y_col]),
                     p0=(18, 0.6)
)
for i in range(len(aaa)):
    print(aaa[i], np.sqrt(np.diag(cov))[i])

plt.plot(xx_plot, (pow_law(xx_plot, j0=aaa[0], m=aaa[1])),
         zorder=0, ls=':', c='gray', alpha=0.5)

plt.xticks([0.1, 1, 10], labels=('0.1', '1', '10'))
plt.xlim(np.min(data_subplot[:, x_col])*0.8,
         np.max(data_subplot[:, x_col])*1.2)
plt.ylim(np.min(np.log10(data_subplot[:, y_col]))-0.2,
         np.max(np.log10(data_subplot[:, y_col]))+0.2)


plt.subplot(122)
data_subplot = datos_J03_frag_hyd
plt.xscale('log')
plt.xlabel(x_label, size=20)
plt.title(r'MHD', size=20, pad=15, fontweight='bold')

plt.axvline(0.15, c='grey', alpha=0.5, zorder=0, ls='-')

plt.scatter(datos_J03_frag_hyd[:, x_col],
            np.log10(datos_J03_frag_hyd[:, y_col]),
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_frag_hyd[:, z_col]))))

plt.xticks([0.1, 1, 10], labels=('0.1', '1', '10'))
plt.xlim(np.min(data_subplot[:, x_col])*0.8,
         np.max(data_subplot[:, x_col])*1.2)
plt.ylim(np.min(np.log10(data_subplot[:, y_col]))-0.2,
         np.max(np.log10(data_subplot[:, y_col]))+0.2)

aaa, cov = curve_fit(pow_law,
                     xdata=data_subplot[:, x_col],
                     ydata=np.log10(data_subplot[:, y_col]),
                     p0=(18, 0.6)
)
print(aaa, np.sqrt(np.diag(cov)))
plt.plot(xx_plot, (pow_law(xx_plot, j0=aaa[0], m=aaa[1])),
         zorder=0, ls=':', c='gray', alpha=0.5)

cax, kw = colorbarr.make_axes([axi for axi in ax.flat])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
c2 = plt.colorbar(sm, cax=cax, extend='both')
c2.set_label(r'$V_\mathrm{max}$ [km s$^{-1}$]', fontsize=20)
yticks = c2.get_ticks()
c2.ax.tick_params(axis='y', direction='out')
c2.set_ticks(np.log10([1., 2., 4, 6]),
             labels=['1', '2', '4', '6'])

plt.savefig(path_name_res + '/datos_J03_frag' + end_str + '.png',
            bbox_inches='tight', dpi=400)
plt.savefig(path_name_res + '/datos_J03_frag' + end_str + '.pdf',
            bbox_inches='tight')

# ----------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
plt.subplot(121)
data_subplot = datos_Js_resi_dmo
plt.xscale('log')
plt.title(r'DMO', size=20, pad=15, fontweight='bold')
plt.xlabel(x_label, size=20)
plt.ylabel(r'log$_{10}$ (J$_\mathrm{S}$ [GeV$^2$ cm$^{-5}$])', size=20)

plt.scatter(datos_Js_resi_dmo[:, x_col],
            np.log10(datos_Js_resi_dmo[:, y_col]),
            c=np.log10(datos_Js_resi_dmo[:, z_col]),
            marker='+', s=100, linewidths=1,
            cmap=colormapp, vmin=vminn, vmax=vmaxx)

aaa, cov = curve_fit(pow_law,
                     xdata=data_subplot[:, x_col],
                     ydata=np.log10(data_subplot[:, y_col]),
                     p0=(18, 0.6)
)
for i in range(len(aaa)):
    print(aaa[i], np.sqrt(np.diag(cov))[i])

plt.plot(xx_plot, (pow_law(xx_plot, j0=aaa[0], m=aaa[1])),
         zorder=0, ls=':', c='gray', alpha=0.5)

plt.xticks([0.1, 1, 10], labels=('0.1', '1', '10'))

plt.xlim(np.min(data_subplot[:, x_col])*0.8,
         np.max(data_subplot[:, x_col])*1.2)
plt.ylim(np.min(np.log10(data_subplot[:, y_col]))-0.2,
         np.max(np.log10(data_subplot[:, y_col]))+0.2)

plt.subplot(122)
data_subplot = datos_Js_resi_hyd
plt.xscale('log')
plt.xlabel(x_label, size=20)

plt.title(r'MHD', size=20, pad=15, fontweight='bold')

plt.scatter(datos_Js_resi_hyd[:, x_col],
            np.log10(datos_Js_resi_hyd[:, y_col]),
            c=np.log10(datos_Js_resi_hyd[:, z_col]),
            marker='+', s=100, linewidths=1,
            cmap=colormapp, vmin=vminn, vmax=vmaxx)

plt.xticks([0.1, 1, 10], labels=('0.1', '1', '10'))
plt.xlim(np.min(data_subplot[:, x_col])*0.8,
         np.max(data_subplot[:, x_col])*1.2)
plt.ylim(np.min(np.log10(data_subplot[:, y_col]))-0.2,
         np.max(np.log10(data_subplot[:, y_col]))+0.2)

ddd = np.argsort(datos_Js_resi_hyd[:, y_col])
datos_Js_resi_hyd = datos_Js_resi_hyd[ddd, :]
aaa, cov = curve_fit(pow_law,
                     xdata=data_subplot[:, x_col],
                     ydata=np.log10(data_subplot[:, y_col]),
                     p0=(18, 0.6)
)
print(aaa, np.sqrt(np.diag(cov)))
plt.plot(xx_plot, (pow_law(xx_plot, j0=aaa[0], m=aaa[1])),
         zorder=0, ls=':', c='gray', alpha=0.5)

cax, kw = colorbarr.make_axes([axi for axi in ax.flat])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
c2 = plt.colorbar(sm, cax=cax, extend='both')
c2.set_label(r'$V_\mathrm{max}$ [km s$^{-1}$]', fontsize=20)
yticks = c2.get_ticks()
c2.ax.tick_params(axis='y', direction='out')
c2.set_ticks(np.log10([1., 2., 4, 6]),
             labels=['1', '2', '4', '6'])

print('%.3f\n%.3f' % (aaa[0], aaa[1]))

plt.savefig(path_name_res + '/datos_Js_res' + end_str + '.png',
            bbox_inches='tight', dpi=400)
plt.savefig(path_name_res + '/datos_Js_res' + end_str + '.pdf',
            bbox_inches='tight')
# plt.show()
# ----------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
plt.subplot(121)
data_subplot = datos_J03_resi_dmo
plt.xscale('log')
plt.xlabel(x_label, size=20)
plt.ylabel(r'log$_{10}$ (J$_\mathrm{03}$ [GeV$^2$ cm$^{-5}$])', size=20)

plt.title(r'DMO', size=20, pad=15, fontweight='bold')

plt.axvline(0.15, c='grey', alpha=0.5, zorder=0, ls='-')
plt.scatter(datos_J03_resi_dmo[:, x_col],
            np.log10(datos_J03_resi_dmo[:, y_col]),
            c=np.log10(datos_J03_resi_dmo[:, z_col]),
            marker='+', s=100, linewidths=1,
            cmap=colormapp, vmin=vminn, vmax=vmaxx)

aaa, cov = curve_fit(pow_law,
                     xdata=data_subplot[:, x_col],
                     ydata=np.log10(data_subplot[:, y_col]),
                     p0=(18, 0.6)
)
for i in range(len(aaa)):
    print(aaa[i], np.sqrt(np.diag(cov))[i])

plt.plot(xx_plot, (pow_law(xx_plot, j0=aaa[0], m=aaa[1])),
         zorder=0, ls=':', c='gray', alpha=0.5)

plt.xticks([0.1, 1, 10], labels=('0.1', '1', '10'))

plt.xlim(np.min(data_subplot[:, x_col])*0.8,
         np.max(data_subplot[:, x_col])*1.2)
plt.ylim(np.min(np.log10(data_subplot[:, y_col]))-0.2,
         np.max(np.log10(data_subplot[:, y_col]))+0.2)

plt.subplot(122)
data_subplot = datos_J03_resi_hyd
plt.xscale('log')
plt.xlabel(x_label, size=20)

plt.axvline(0.15, c='grey', alpha=0.5, zorder=0, ls='-')
plt.title(r'MHD', size=20, pad=15, fontweight='bold')


plt.scatter(datos_J03_resi_hyd[:, x_col],
            np.log10(datos_J03_resi_hyd[:, y_col]),
            c=np.log10(datos_J03_resi_hyd[:, z_col]),
            marker='+', s=100, linewidths=1,
            cmap=colormapp, vmin=vminn, vmax=vmaxx)
plt.xticks([0.1, 1, 10], labels=('0.1', '1', '10'))
plt.xlim(np.min(data_subplot[:, x_col])*0.8,
         np.max(data_subplot[:, x_col])*1.2)
plt.ylim(np.min(np.log10(data_subplot[:, y_col]))-0.2,
         np.max(np.log10(data_subplot[:, y_col]))+0.2)

aaa, cov = curve_fit(pow_law,
                     xdata=data_subplot[:, x_col],
                     ydata=np.log10(data_subplot[:, y_col]),
                     p0=(18, 0.6)
)
print(aaa, np.sqrt(np.diag(cov)))
plt.plot(xx_plot, (pow_law(xx_plot, j0=aaa[0], m=aaa[1])),
         zorder=0, ls=':', c='gray', alpha=0.5)

cax, kw = colorbarr.make_axes([axi for axi in ax.flat])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
c2 = plt.colorbar(sm, cax=cax, extend='both')
c2.set_label(r'$V_\mathrm{max}$ [km s$^{-1}$]', fontsize=20)
yticks = c2.get_ticks()
c2.ax.tick_params(axis='y', direction='out')
c2.set_ticks(np.log10([1., 2., 4, 6]),
             labels=['1', '2', '4', '6'])

print('%.3f\n%.3f' % (aaa[0], aaa[1]))

plt.savefig(path_name_res + '/datos_J03_resi' + end_str + '.png',
            bbox_inches='tight', dpi=400)
plt.savefig(path_name_res + '/datos_J03_resi' + end_str + '.pdf',
            bbox_inches='tight')

plt.show()
# ----------------------- Solid ang size - J (z==Vmax) 2x2 -------------

fig, axes = plt.subplots(nrows=2, ncols=2, sharey=True,  # sharey=True,
                         figsize=(7, 9))

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

x_label = r'$\Omega\,\,\left[\mathrm{sr}\right]$'
y1_label = r'log$_{10}$ (J$_\mathrm{S}$ [GeV$^2$ cm$^{-5}$])'
y2_label = r'log$_{10}$ (J$_{03}$ [GeV$^2$ cm$^{-5}$])'

plt.text(-0.25, 0., r'log$_{10}$ ($J_\mathrm{s}$ [GeV$^2$ cm$^{-5}$])',
         horizontalalignment='center',
         verticalalignment='center', transform=axes[0, 0].transAxes,
         rotation=90)
plt.text(1., 0.92, r'Fragile',
         horizontalalignment='center',
         verticalalignment='top', transform=axes[0, 0].transAxes,
         backgroundcolor='w')
plt.text(1., 0.18, r'Resilient',
         horizontalalignment='center',
         verticalalignment='top', transform=axes[1, 0].transAxes,
         backgroundcolor='w')

# vminn = 1e-3
# vmaxx = np.log10(150)  # 10 ** 1.5
vminn = np.log10(5)  # perc_total(z_col, 5))
vmaxx = np.log10(perc_total(z_col, 95))
print(vminn, vmaxx, 10 ** vminn, 10 ** vmaxx)
norm = mcb.Normalize(vminn, vmaxx)

minns, maxxs = minnmaxxS(y_col)
minns = 18.2
maxxs = 23.6

minnx = 1e-5  # perc_total(x_col, 0) * 0.99
maxxx = 10  # perc_total(x_col, 100) * 1.3

plt.subplot(221)
plt.title('DMO', fontsize=20)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.tick_params('x', labelbottom=False)
# plt.figure()
# print('we start debug')
# for i in range(len(datos_Js_frag_dmo[:, 0])):
#     if (datos_Js_frag_dmo[i, 2] < funct_repop.R_s(
#                     V=datos_Js_frag_dmo[i, 3],
#                     C=datos_Js_frag_dmo[i, 5],
#                     cosmo_H_0=67.7
#                 )):
#         print(i, datos_Js_frag_dmo[i, :])
# plt.plot(2*np.pi * (1 - np.cos(datos_Js_frag_dmo[:, 4])), marker='+')
# plt.plot(funct_repop.R_s(
#                     V=datos_Js_frag_dmo[:, 3],
#                     C=datos_Js_frag_dmo[:, 5],
#                     cosmo_H_0=67.7
#                 ), marker='x')
# plt.show()

rr_ss = funct_repop.R_s(
                    V=datos_Js_frag_dmo[:, 3],
                    C=datos_Js_frag_dmo[:, 5],
                    cosmo_H_0=67.7
                )
real_theta = np.arcsin(rr_ss/datos_Js_frag_dmo[:, 2])
# np.nanmin((
#     np.pi/2. * np.ones_like(datos_Js_frag_dmo[:, 0]),
#     np.arcsin(rr_ss/datos_Js_frag_dmo[:, 2])), axis=0)

plt.scatter(2*np.pi * (1 - np.cos(real_theta)),
            np.log10(datos_Js_frag_dmo[:, y_col]),
            marker='+', linewidths=1,)
plt.scatter(2*np.pi *
            (1 - np.sqrt(datos_Js_frag_dmo[:, 2]**2. - rr_ss**2.)
             /datos_Js_frag_dmo[:, 2]),
            np.log10(datos_Js_frag_dmo[:, y_col]),
            marker='s', c='none', lw=1, edgecolors='orange', s=120)

plt.scatter(2*np.pi * (1 - np.cos(np.pi/180*datos_Js_frag_dmo[:, 4])),
            np.log10(datos_Js_frag_dmo[:, y_col]),
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_frag_dmo[:, z_col]))))

plt.subplot(223)

plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.xlabel(x_label, size=20)
rr_ss = funct_repop.R_s(
                    V=datos_Js_frag_dmo[:, 3],
                    C=datos_Js_frag_dmo[:, 5],
                    cosmo_H_0=67.7
                )
real_theta = np.arcsin(rr_ss/datos_Js_frag_dmo[:, 2])

plt.scatter(2*np.pi * (1 - np.cos(real_theta)),
            np.log10(datos_Js_frag_dmo[:, y_col]),
            marker='+', linewidths=1,)
plt.scatter(2*np.pi *
            (1 - np.sqrt(datos_Js_frag_dmo[:, 2]**2. - rr_ss**2.)
             /datos_Js_frag_dmo[:, 2]),
            np.log10(datos_Js_frag_dmo[:, y_col]),
            marker='s', c='none', lw=1, edgecolors='orange', s=120)

plt.scatter(2*np.pi *
            (1 - np.sqrt(
                datos_Js_resi_dmo[:, 2]**2.
                - funct_repop.R_s(
                    V=datos_Js_resi_dmo[:, 3],
                    C=datos_Js_resi_dmo[:, 5],
                    cosmo_H_0=67.7
                )**2.
            )/datos_Js_resi_dmo[:, 2]),
            np.log10(datos_Js_resi_dmo[:, y_col]),
            c=np.log10(datos_Js_resi_dmo[:, z_col]),
            marker='+', s=100, linewidths=1,
            cmap=colormapp, vmin=vminn, vmax=vmaxx)

plt.subplot(222)

plt.title('MHD', fontsize=20)

plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.tick_params('y', labelleft=False)
plt.tick_params('x', labelbottom=False)

plt.scatter(2*np.pi *
            (1 - np.sqrt(
                datos_Js_frag_hyd[:, 2]**2.
                - funct_repop.R_s(
                    V=datos_Js_frag_hyd[:, 3],
                    C=datos_Js_frag_hyd[:, 5],
                    cosmo_H_0=67.7
                )**2.
            )/datos_Js_frag_hyd[:, 2]),
            np.log10(datos_Js_frag_hyd[:, y_col]),
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_frag_hyd[:, z_col]))))

plt.subplot(224)

plt.tick_params('y', labelleft=False)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.xlabel(x_label, size=20)

plt.scatter(2*np.pi *
            (1 - np.sqrt(
                datos_Js_resi_hyd[:, 2]**2.
                - funct_repop.R_s(
                    V=datos_Js_resi_hyd[:, 3],
                    C=datos_Js_resi_hyd[:, 5],
                    cosmo_H_0=67.7
                )**2.
            )/datos_Js_resi_hyd[:, 2]),
            np.log10(datos_Js_resi_hyd[:, y_col]),
            c=np.log10(datos_Js_resi_hyd[:, z_col]),
            marker='+', s=100, linewidths=1,
            cmap=colormapp, vmin=vminn, vmax=vmaxx)

plt.yticks((19, 20, 21, 22, 23), labels=('19', '', '21', '', '23'))

aa = plt.subplot(2, 2, 3)
aa.set_xticks([1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1., 10],
              labels=(r'10$^{-5}$', '', r'10$^{-3}$', '', '0.1', '1', '')
              )
aa = plt.subplot(2, 2, 1)
aa.set_xticks([1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1., 10])
aa = plt.subplot(2, 2, 2)
aa.set_xticks([1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1., 10])
#
aa = plt.subplot(2, 2, 4)
aa.set_xticks([1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1., 10],
              labels=(r'10$^{-5}$', '', r'10$^{-3}$', '', '0.1', '1', '10')
              )

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
c2 = plt.colorbar(sm, ax=axes,
                  extend='both', spacing='proportional',
                  location='bottom')
c2.set_label(r'$V_\mathrm{max}$ [km s$^{-1}$]', fontsize=20)
yticks = c2.get_ticks()
print(yticks)
c2.ax.tick_params(axis='x', direction='out')
c2.set_ticklabels([str(10 ** i)[:4] for i in yticks])
c2.set_ticks(np.log10([5., 10., 20, 50, 90.]),
             labels=['5', '10', '20', '50', '90'])

plt.savefig(path_name_res + '/solidAngJss_full' + end_str + '.png',
            bbox_inches='tight')
plt.savefig(path_name_res + '/solidAngJss_full' + end_str + '.pdf',
            bbox_inches='tight')
# plt.show()

# -------------------- J_hist -------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(7, 6))

plt.subplots_adjust(wspace=0, hspace=0)

minS, maxS = minnmaxxS(0)
min03, max03 = minnmaxx03(0)
minn = np.min((minS, min03))
maxx = np.max((maxS, max03))

bines = np.linspace(minn, maxx, 40)
# bines = np.linspace(19, 23, 40)

ax1 = plt.subplot(221)

plt.title('DMO', size=18)
plt.tick_params('x', labelbottom=False)
print(np.shape(datos_Js_frag_dmo))
plt.hist(np.log10(datos_Js_frag_dmo[:, 0]), log=False,
         label=r'Frag', color='teal', alpha=0.6, edgecolor='k',
         bins=bines, hatch='//', lw=0, histtype='stepfilled')

plt.hist(np.log10(datos_Js_resi_dmo[:, 0]), log=False,
         label=r'Res', color='k', alpha=0.5,
         bins=bines, histtype='stepfilled')

Js95_frag_dmo = np.log10(np.percentile(datos_Js_frag_dmo[:, 0], 5))
Js95_resi_dmo = np.log10(np.percentile(datos_Js_resi_dmo[:, 0], 5))
print('Js, DMO')
print(Js95_resi_dmo, Js95_frag_dmo, Js95_resi_dmo - Js95_frag_dmo)

plt.axvline(Js95_frag_dmo, color='#004F7D', lw=1.5, ls='--')  # , alpha=0.6)
plt.axvline(Js95_resi_dmo, color='k', lw=1.5, ls='--')  # , alpha=0.5)

plt.xlim(18., maxx)
plt.ylim(bottom=0., top=130)

plt.xticks((18, 19, 20, 21, 22), labels=('', '', '', '', ''))
# plt.yscale('log')

ax1.tick_params(labelbottom=False)

# plt.annotate(r'J$_S$ 95%', (Js95_frag_dmo, 20), rotation=90, color='k')
# plt.annotate(r'J$_{03}$ 95%', (J0395_frag_dmo, 20),
# rotation=90, color='g', horizontalalignment='right')
plt.legend(title=r'J$_\mathrm{S}$', handlelength=0.9,
           handletextpad=0.5, fontsize=16)
ax2 = plt.subplot(222)
plt.title('MHD', size=18)
plt.tick_params('x', labelbottom=False)
# plt.yscale('log')
plt.hist(np.log10(datos_Js_frag_hyd[:, 0]), log=False,
         label=r'Frag', facecolor='yellowgreen', alpha=0.6,
         bins=bines, hatch='//', edgecolor='k',
         lw=0, histtype='stepfilled')
plt.hist(np.log10(datos_Js_resi_hyd[:, 0]), log=False,
         label=r'Res', color=darkgreen, alpha=0.6,
         bins=bines, histtype='stepfilled')

Js95_frag_hyd = np.log10(np.percentile(datos_Js_frag_hyd[:, 0], 5))
Js95_resi_hyd = np.log10(np.percentile(datos_Js_resi_hyd[:, 0], 5))
print('Js, MHD')
print(Js95_resi_hyd, Js95_frag_hyd, Js95_resi_hyd - Js95_frag_hyd)

plt.xlim(18., maxx)
plt.ylim(bottom=0., top=130)

plt.xticks((18, 19, 20, 21, 22), labels=('', '', '', '', ''))

plt.axvline(Js95_frag_hyd, color='#006E0B', lw=1.5, ls='--')  # , alpha=0.6)
plt.axvline(Js95_resi_hyd, color='k', lw=1.5, ls='--')  # , alpha=0.6)

# plt.annotate(r'J$_S$ 95%', (Js95_frag_hyd, 20), rotation=90, color='k')
# plt.annotate(r'J$_{03}$ 95%', (Js95_resi_hyd, 20),
# rotation=90, color='g', horizontalalignment='right')

ax2.tick_params(labelleft=False)

plt.legend(title=r'J$_\mathrm{S}$', handlelength=0.9,
           handletextpad=0.5, fontsize=16)

plt.subplot(223)

plt.hist(np.log10(datos_J03_frag_dmo[:, 0]), log=False,
         label=r'Frag', color='teal', alpha=0.6,
         bins=bines, hatch='//', edgecolor='k',
         lw=0, histtype='stepfilled')
plt.hist(np.log10(datos_J03_resi_dmo[:, 0]), log=False,
         label=r'Res', color='k', alpha=0.5,
         bins=bines, histtype='stepfilled')

J0395_frag_dmo = np.log10(np.percentile(datos_J03_frag_dmo[:, 0], 5))
J0395_resi_dmo = np.log10(np.percentile(datos_J03_resi_dmo[:, 0], 5))
print('J03, DMO')
print(J0395_frag_dmo, J0395_resi_dmo, J0395_frag_dmo - J0395_resi_dmo)
# plt.yscale('log')

plt.xlim(18., maxx)
plt.ylim(bottom=0., top=130)

plt.axvline(J0395_frag_dmo, color='#004F7D', lw=1.5, ls='--')  # , alpha=0.6)
plt.axvline(J0395_resi_dmo, color='k', lw=1.5, ls='--')  # , alpha=0.5)

# plt.annotate(r'J$_S$ 95%', (Js95_resi_dmo, 20), rotation=90, color='k')
# plt.annotate(r'J$_{03}$ 95%', (J0395_resi_dmo, 20),
# rotation=90, color='g', horizontalalignment='right')

plt.xticks((18, 19, 20, 21, 22))
# plt.xlabel(r'log$_{10}$ (J$_\mathrm{factor}$ [GeV$^2$ cm$^{-5}$])',
#            fontsize=20)
plt.legend(title=r'J$_{03}$', handlelength=0.9,
           handletextpad=0.5, fontsize=16)

ax4 = plt.subplot(224)

plt.hist(np.log10(datos_J03_frag_hyd[:, 0]), log=False,
         label=r'Frag', color='yellowgreen', alpha=0.6,
         bins=bines, hatch='//', edgecolor='k',
         lw=0, histtype='stepfilled')
plt.hist(np.log10(datos_J03_resi_hyd[:, 0]), log=False,
         label=r'Res', color=darkgreen, alpha=0.6,
         bins=bines, histtype='stepfilled')

J0395_frag_hyd = np.log10(np.percentile(datos_J03_frag_hyd[:, 0], 5))
J0395_resi_hyd = np.log10(np.percentile(datos_J03_resi_hyd[:, 0], 5))
print('J03, MHD')
print(J0395_resi_hyd, J0395_frag_hyd, J0395_resi_hyd - J0395_frag_hyd)

plt.axvline(J0395_frag_hyd, color='#006E0B', lw=1.5, ls='--')  # , alpha=0.6)
plt.axvline(J0395_resi_hyd, color='k', lw=1.5, ls='--')  # , alpha=0.6)

plt.xlim(18., maxx)
plt.ylim(bottom=0., top=130)
# plt.annotate(r'J$_S$ 95%', (Js95_resi_hyd, 20), rotation=90, color='k')
# plt.annotate(r'J$_{03}$ 95%', (J0395_resi_hyd, 20),
# rotation=90, color='g', horizontalalignment='right')

# # plt.yscale('log')
# plt.xlabel(r'log$_{10}$ (J$_\mathrm{factor}$ [GeV$^2$ cm$^{-5}$])',
#            fontsize=20)
plt.legend(title=r'J$_{03}$', handlelength=0.9,
           handletextpad=0.5, fontsize=16)

fig.text(0.03, 0.5, 'Number of repopulations',
         ha='center',
         va='center', rotation='vertical')

fig.text(0.5, 0.01,
         'log$_{10}$ (J$_\mathrm{factor}$ [GeV$^2$ cm$^{-5}$])',
         ha='center',
         va='center')

ax4.tick_params(labelleft=False)

plt.xticks((18, 19, 20, 21, 22), labels=('', 19, 20, 21, 22))

plt.savefig(path_name_res + '/J_hist' + end_str + '.png',
            bbox_inches='tight')
plt.savefig(path_name_res + '/J_hist' + end_str + '.pdf',
            bbox_inches='tight')
plt.show()
# ------------------- Cross sections ------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

J0395_frag_dmo = np.log10(np.percentile(datos_J03_frag_dmo[:, 0], 5))
J0395_resi_dmo = np.log10(np.percentile(datos_J03_resi_dmo[:, 0], 5))
print('Numbers for cross sections')
print('J03, DMO')
print(J0395_frag_dmo, J0395_resi_dmo, J0395_frag_dmo - J0395_resi_dmo)

J0395_frag_hyd = np.log10(np.percentile(datos_J03_frag_hyd[:, 0], 5))
J0395_resi_hyd = np.log10(np.percentile(datos_J03_resi_hyd[:, 0], 5))
print('J03, MHD')
print(J0395_resi_hyd, J0395_frag_hyd, J0395_resi_hyd - J0395_frag_hyd)

plt.subplots_adjust(wspace=0, hspace=0)

ax1 = plt.subplot(121)


plt.plot(constraints_bb_2204[:, 0],
         constraints_bb_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_frag_hyd),
         '-', label='MHD', color='limegreen', lw=2.5)
plt.plot(constraints_bb_2204[:, 0],
         constraints_bb_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_frag_dmo),
         '-', c='k', label='DMO', alpha=1, lw=2.5)

plt.plot(constraints_bb_2204[:, 0],
         constraints_bb_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_resi_dmo),
         ':', c='k', alpha=1, lw=2.5)
plt.plot(constraints_bb_2204[:, 0],
         constraints_bb_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_resi_hyd),
         ':', color='limegreen', alpha=1, lw=2.5)

plt.plot(constraints_bb_2204[:, 0], constraints_bb_2204[:, 1], '--',
         label='CB22', alpha=1., color='#6CCAFF', lw=2)
plt.plot(constraints_bb_2204[:, 0], 2*constraints_bb_2204[:, 1], '--',
         label='CB22', alpha=1., color='#6CCAFF', lw=2)

plt.plot(sigmav_bb_2204[:, 0], sigmav_bb_2204[:, 1], '-.',
         c='silver', lw=2, zorder=0)

plt.xlim(sigmav_bb_2204[0, 0], sigmav_bb_2204[-1, 0])

plt.text(0.7, 0.85, r'$b\bar{b}$', transform=ax1.transAxes,
         horizontalalignment='center', size=30)

plt.annotate(r'$\langle\sigma\nu\rangle_\mathrm{th}$', (1000, 3e-26))

plt.xscale('log')
plt.yscale('log')

plt.xlabel('m$_{\chi}$ [GeV]', size=20)
plt.ylabel(r'$\langle\sigma\nu\rangle$ [cm$^3$ s$^{-1}$]', size=20)

# legend1 = plt.legend(legend_elements, ['Fragile', 'Resilient'], loc=2,
#                      bbox_to_anchor=(0.1, 0.6))

legend_elements = [mpatches.Patch(color='limegreen', alpha=0.8),
                   mpatches.Patch(color='k', alpha=0.8),
                   Line2D([0], [0], color='#6CCAFF',
                          linestyle='--', lw=2)]
leg = plt.legend(legend_elements, ['MHD', 'DMO', 'CB22'], loc=2)

bapad = plt.rcParams['legend.borderaxespad']
fontsize = plt.rcParams['font.size']
axline = plt.rcParams['axes.linewidth']
pad_pixels = (bapad*fontsize + axline) / 72. * fig.dpi
inv = axes[0].transAxes.inverted()

# Inverse transform two points on the display and find the relative distance
pad_axes = inv.transform((pad_pixels, 0)) - inv.transform((0, 0))
pad_xaxis = pad_axes[0]
# Find how may pixels there are on the x-axis
x_pixels = (axes[0].transAxes.transform((1, 0))
            - axes[0].transAxes.transform((0, 0)))
# Compute the ratio between the pixel offset and the total amount of pixels
pad_xaxis = (pad_pixels - 2.5)/x_pixels[0]

legend_elements = [Line2D([0], [0], color='k', label='Frag',
                          linestyle='-', lw=2.5),
                   Line2D([0], [0], color='k', label='Res',
                          linestyle=':', lw=2.5)]
legend1 = plt.legend(legend_elements, ['Fragile', 'Resilient'],
                     loc=(pad_xaxis, 0.53))

plt.gca().add_artist(leg)
plt.gca().add_artist(legend1)

t1, t2, t3 = leg.get_texts()
# here we create the distinct instance
t1._fontproperties = t2._fontproperties.copy()
t3.set_size(16)

ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)

plt.plot(constraints_tau_2204[:, 0],
         constraints_tau_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_frag_dmo),
         '-', c='k', label='DMO', lw=2.5)
plt.plot(constraints_tau_2204[:, 0],
         constraints_tau_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_frag_hyd),
         '-', label='MHD', color='limegreen', lw=2.5)

plt.plot(constraints_tau_2204[:, 0],
         constraints_tau_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_resi_dmo),
         ':', c='k', lw=2.5)
plt.plot(constraints_tau_2204[:, 0],
         constraints_tau_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_resi_hyd),
         ':', color='limegreen', lw=2.5)

plt.plot(sigmav_tau_2204[1:, 0], sigmav_tau_2204[1:, 1],
         '-.', c='silver', lw=2)
plt.plot(constraints_tau_2204[:, 0], constraints_tau_2204[:, 1], '--',
         label='CB22', alpha=1., color='#6CCAFF', lw=2)

plt.text(0.7, 0.85, r'$\tau^+\tau^-$', transform=ax2.transAxes, size=30,
         horizontalalignment='center',
         )
plt.annotate(r'$\langle\sigma\nu\rangle_\mathrm{th}$', (1000, 3e-26))

plt.xscale('log')
plt.yscale('log')
plt.xlim(sigmav_tau_2204[0, 0], sigmav_tau_2204[-1, 0])
plt.ylim(1e-26, 1e-20)

ax2.tick_params(labelleft=False)
plt.xlabel('m$_{\chi}$ [GeV]', size=20)
plt.yticks(10**np.array([-26., -25, -24, -23, -22, -21, -20]),
               # labels=('', '19', '', '21', '', '23')
           )

# ------------------- Cross sections ------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
J0395_frag_dmo = np.log10(np.percentile(datos_J03_frag_dmo[:, 0], 5))
J0395_resi_dmo = np.log10(np.percentile(datos_J03_resi_dmo[:, 0], 5))
print('Numbers for cross sections')
print('J03, DMO')
print(J0395_frag_dmo, J0395_resi_dmo, J0395_frag_dmo - J0395_resi_dmo)
J0395_frag_hyd = np.log10(np.percentile(datos_J03_frag_hyd[:, 0], 5))
J0395_resi_hyd = np.log10(np.percentile(datos_J03_resi_hyd[:, 0], 5))
print('J03, MHD')
print(J0395_resi_hyd, J0395_frag_hyd, J0395_resi_hyd - J0395_frag_hyd)

plt.subplots_adjust(wspace=0, hspace=0)

ax1 = plt.subplot(121)

plt.plot(constraints_bb_2204[:, 0],
         constraints_bb_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_frag_hyd),
         '-', label='MHD', color='limegreen', lw=2.5)
plt.plot(constraints_bb_2204[:, 0],
         constraints_bb_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_frag_dmo),
         '-', c='k', label='DMO', alpha=1, lw=2.5)
plt.plot(constraints_bb_2204[:, 0],
         constraints_bb_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_resi_dmo),
         ':', c='k', alpha=1, lw=2.5)
plt.plot(constraints_bb_2204[:, 0],
         constraints_bb_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_resi_hyd),
         ':', color='limegreen', alpha=1, lw=2.5)

plt.plot(constraints_bb_2204[:, 0], constraints_bb_2204[:, 1], '-',
         label='CB22', alpha=1., color='dodgerblue', lw=1.)

plt.plot(sigmav_bb_2204[:, 0], sigmav_bb_2204[:, 1], '-.',
         c='silver', lw=2, zorder=0)

plt.xlim(sigmav_bb_2204[0, 0], sigmav_bb_2204[-1, 0])
plt.text(0.7, 0.85, r'$b\bar{b}$', transform=ax1.transAxes,
         horizontalalignment='center', size=30)
plt.annotate(r'$\langle\sigma\nu\rangle_\mathrm{th}$', (1000, 3e-26))
plt.xscale('log')
plt.yscale('log')
plt.xlabel('m$_{\chi}$ [GeV]', size=20)
plt.ylabel(r'$\langle\sigma\nu\rangle$ [cm$^3$ s$^{-1}$]', size=20)
# legend1 = plt.legend(legend_elements, ['Fragile', 'Resilient'], loc=2,
#                      bbox_to_anchor=(0.1, 0.6))
legend_elements = [Line2D([0], [0], color='k', label='Frag',
                          linestyle='-', lw=2.5),
                   Line2D([0], [0], color='k', label='Res',
                          linestyle=':', lw=2.5)]
legend1 = plt.legend(legend_elements, ['Fragile', 'Resilient'],
                     loc=2)
bapad = plt.rcParams['legend.borderaxespad']
fontsize = plt.rcParams['font.size']
axline = plt.rcParams['axes.linewidth']
pad_pixels = (bapad*fontsize + axline) / 72. * fig.dpi
inv = axes[0].transAxes.inverted()
# Inverse transform two points on the display and find the relative distance
pad_axes = inv.transform((pad_pixels, 0)) - inv.transform((0, 0))
pad_xaxis = pad_axes[0]
# Find how may pixels there are on the x-axis
x_pixels = (axes[0].transAxes.transform((1, 0))
            - axes[0].transAxes.transform((0, 0)))
# Compute the ratio between the pixel offset and the total amount of pixels
pad_xaxis = (pad_pixels - 2.5)/x_pixels[0]
legend_elements = [mpatches.Patch(color='limegreen', alpha=0.8),
                   mpatches.Patch(color='k', alpha=0.8),
                   Line2D([0], [0], color='dodgerblue',
                          linestyle='-', lw=1.)]
leg = plt.legend(legend_elements, ['MHD', 'DMO', 'CB22'],
                     loc=(pad_xaxis, 0.45), handlelength=0.95,)
plt.gca().add_artist(leg)
plt.gca().add_artist(legend1)
t1, t2, t3 = leg.get_texts()
# here we create the distinct instance
t1._fontproperties = t2._fontproperties.copy()
t3.set_size(16)

ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)

plt.plot(constraints_tau_2204[:, 0],
         constraints_tau_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_frag_dmo),
         '-', c='k', label='DMO', lw=2.5)
plt.plot(constraints_tau_2204[:, 0],
         constraints_tau_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_frag_hyd),
         '-', label='MHD', color='limegreen', lw=2.5)
plt.plot(constraints_tau_2204[:, 0],
         constraints_tau_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_resi_dmo),
         ':', c='k', lw=2.5)
plt.plot(constraints_tau_2204[:, 0],
         constraints_tau_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_resi_hyd),
         ':', color='limegreen', lw=2.5)

plt.plot(sigmav_tau_2204[1:, 0], sigmav_tau_2204[1:, 1],
         '-.', c='silver', lw=2)
plt.plot(constraints_tau_2204[:, 0], constraints_tau_2204[:, 1], '-',
         label='CB22', alpha=1., color='dodgerblue', lw=1)

plt.text(0.7, 0.85, r'$\tau^+\tau^-$', transform=ax2.transAxes, size=30,
         horizontalalignment='center',
         )
plt.annotate(r'$\langle\sigma\nu\rangle_\mathrm{th}$', (1000, 3e-26))
plt.xscale('log')
plt.yscale('log')
plt.xlim(sigmav_tau_2204[0, 0], sigmav_tau_2204[-1, 0])
plt.ylim(1e-26, 1e-20)
ax2.tick_params(labelleft=False)
plt.xlabel('m$_{\chi}$ [GeV]', size=20)
plt.yticks(10**np.array([-26., -25, -24, -23, -22, -21, -20]),
               labels=(r'10$^{-26}$', r'10$^{-25}$',
                       r'10$^{-24}$', r'10$^{-23}$',
                       r'10$^{-22}$', r'10$^{-21}$',
                       r'10$^{-20}$')
           )
plt.savefig(path_name_res + '/Cross.png', bbox_inches='tight')
plt.savefig(path_name_res + '/Cross.pdf', bbox_inches='tight')

# plt.show()
'''
# ------------------- Cross sections vertical -------------------------------
fig, axes = plt.subplots(2, 1, figsize=(6, 9))
J0395_frag_dmo = 18.696 #np.log10(np.percentile(datos_J03_frag_dmo[:, 0], 5))
J0395_resi_dmo = 19.346 #np.log10(np.percentile(datos_J03_resi_dmo[:, 0], 5))
print('Numbers for cross sections')
print('J03, DMO')
print(J0395_frag_dmo, J0395_resi_dmo, J0395_frag_dmo - J0395_resi_dmo)
J0395_frag_hyd = 18.1513 #np.log10(np.percentile(datos_J03_frag_hyd[:, 0], 5))
J0395_resi_hyd = 19.1586 #np.log10(np.percentile(datos_J03_resi_hyd[:, 0], 5))
print('J03, MHD')
print(J0395_resi_hyd, J0395_frag_hyd, J0395_resi_hyd - J0395_frag_hyd)

plt.subplots_adjust(wspace=0, hspace=0)

fig.text(-0.07, 0.5, r'$\langle\sigma\nu\rangle$ [cm$^3$ s$^{-1}$]',
         size=24,
         ha='center',
         va='center', rotation='vertical')

ax1 = plt.subplot(211)

plt.plot(constraints_bb_2204[:, 0],
         constraints_bb_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_frag_hyd),
         '-', label='MHD', color='limegreen', lw=2.5)
plt.plot(constraints_bb_2204[:, 0],
         constraints_bb_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_frag_dmo),
         '-', c='k', label='DMO', alpha=1, lw=2.5)
plt.plot(constraints_bb_2204[:, 0],
         constraints_bb_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_resi_dmo),
         ':', c='k', alpha=1, lw=2.5)
plt.plot(constraints_bb_2204[:, 0],
         constraints_bb_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_resi_hyd),
         ':', color='limegreen', alpha=1, lw=2.5)

plt.plot(constraints_bb_2204[:, 0], constraints_bb_2204[:, 1], '--',
         label='CB22', alpha=1., color='dodgerblue', lw=1.)

plt.plot(sigmav_bb_2204[:, 0], sigmav_bb_2204[:, 1], '-.',
         c='silver', lw=2, zorder=0)

plt.xlim(5., 1e4)
plt.ylim(1e-26, 1e-20)

plt.text(0.15, 0.47, r'$b\bar{b}$', transform=ax1.transAxes,
         horizontalalignment='center', size=24)

plt.annotate(r'$\langle\sigma\nu\rangle_\mathrm{th}$', (1000, 3.5e-26))
plt.xscale('log')
plt.yscale('log')

# legend1 = plt.legend(legend_elements, ['Fragile', 'Resilient'], loc=2,
#                      bbox_to_anchor=(0.1, 0.6))
bapad = plt.rcParams['legend.borderaxespad']
fontsize = plt.rcParams['font.size']
axline = plt.rcParams['axes.linewidth']
pad_pixels = (bapad*fontsize + axline) / 72. * fig.dpi
inv = axes[0].transAxes.inverted()
# Inverse transform two points on the display and find the relative distance
pad_axes = inv.transform((pad_pixels, 0)) - inv.transform((0, 0))
pad_xaxis = pad_axes[0]
# Find how may pixels there are on the x-axis
x_pixels = (axes[0].transAxes.transform((1, 0))
            - axes[0].transAxes.transform((0, 0)))
# Compute the ratio between the pixel offset and the total amount of pixels
pad_xaxis = (pad_pixels - 2.5)/x_pixels[0]

legend_elements = [Line2D([0], [0], color='k', label='Frag',
                          linestyle='-', lw=2.5),
                   Line2D([0], [0], color='k', label='Res',
                          linestyle=':', lw=2.5)]
legend1 = plt.legend(legend_elements, ['Fragile', 'Resilient'],
                     # loc=(pad_xaxis, 0.37),
                     loc=9
                     )

legend_elements = [mpatches.Patch(color='limegreen', alpha=0.8),
                   mpatches.Patch(color='k', alpha=0.8),
                   Line2D([0], [0], color='dodgerblue',
                          linestyle='--', lw=1.)]

leg = plt.legend(legend_elements, ['MHD', 'DMO', 'CB22'],
                     loc=2, handlelength=0.95,)

plt.gca().add_artist(leg)
plt.gca().add_artist(legend1)
t1, t2, t3 = leg.get_texts()
# here we create the distinct instance
t1._fontproperties = t2._fontproperties.copy()
t3.set_size(16)
ax1.tick_params(labelbottom=False)

plt.yticks(10**np.array([-26., -25, -24, -23, -22, -21, -20]),
               labels=(r'10$^{-26}$', r'10$^{-25}$',
                       r'10$^{-24}$', r'10$^{-23}$',
                       r'10$^{-22}$', r'10$^{-21}$',
                       r'10$^{-20}$')
           )

ax2 = plt.subplot(212)

plt.plot(constraints_tau_2204[:, 0],
         constraints_tau_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_frag_dmo),
         '-', c='k', label='DMO', lw=2.5)
plt.plot(constraints_tau_2204[:, 0],
         constraints_tau_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_frag_hyd),
         '-', label='MHD', color='limegreen', lw=2.5)
plt.plot(constraints_tau_2204[:, 0],
         constraints_tau_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_resi_dmo),
         ':', c='k', lw=2.5)
plt.plot(constraints_tau_2204[:, 0],
         constraints_tau_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_resi_hyd),
         ':', color='limegreen', lw=2.5)

plt.plot(sigmav_tau_2204[1:, 0], sigmav_tau_2204[1:, 1],
         '-.', c='silver', lw=2, zorder=0)
plt.plot(constraints_tau_2204[:, 0], constraints_tau_2204[:, 1], '--',
         label='CB22', alpha=1., color='dodgerblue', lw=1)

# plt.text(0.45, 0.85, r'$\tau^+\tau^-$', transform=ax2.transAxes, size=26,
#          horizontalalignment='center',
#          )
plt.text(0.16, 0.7, r'$\tau^+\tau^-$', transform=ax2.transAxes, size=24,
         horizontalalignment='center',
         )
plt.annotate(r'$\langle\sigma\nu\rangle_\mathrm{th}$', (1000, 3.5e-26))
plt.xscale('log')
plt.yscale('log')
plt.xlim(5., 1e4)
plt.ylim(1e-26, 1e-20)
# ax2.tick_params(labelleft=False)
plt.xlabel('m$_{\chi}$ [GeV]', size=22)
plt.yticks(10**np.array([-26., -25, -24, -23, -22, -21, -20]),
               labels=(r'10$^{-26}$', r'10$^{-25}$',
                       r'10$^{-24}$', r'10$^{-23}$',
                       r'10$^{-22}$', r'10$^{-21}$',
                       '')
           )
plt.savefig('Crossvertical.png', bbox_inches='tight')
plt.savefig('Crossvertical.pdf', bbox_inches='tight')
aaa
plt.show()
'''
# ------------------- Cross sections ------------------------------------------
fig, ax1 = plt.subplots(figsize=(6, 6))

J0395_frag_dmo = np.log10(np.percentile(datos_J03_frag_dmo[:, 0], 5))
J0395_resi_dmo = np.log10(np.percentile(datos_J03_resi_dmo[:, 0], 5))
print('J03, DMO')
print(J0395_frag_dmo, J0395_resi_dmo, J0395_frag_dmo - J0395_resi_dmo)

J0395_frag_hyd = np.log10(np.percentile(datos_J03_frag_hyd[:, 0], 5))
J0395_resi_hyd = np.log10(np.percentile(datos_J03_resi_hyd[:, 0], 5))
print('J03, Hydro')
print(J0395_resi_hyd, J0395_frag_hyd, J0395_resi_hyd - J0395_frag_hyd)


plt.plot(constraints_bb_2204[:, 0],
         constraints_bb_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_frag_hyd),
         '-', label='MHD', color='limegreen', lw=2.5)
plt.plot(constraints_bb_2204[:, 0],
         constraints_bb_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_frag_dmo),
         '-', c='k', label='DMO', alpha=1, lw=2.5)

plt.plot(constraints_bb_2204[:, 0],
         constraints_bb_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_resi_dmo),
         ':', c='k', alpha=1, lw=2.5)
plt.plot(constraints_bb_2204[:, 0],
         constraints_bb_2204[:, 1] * 10 ** (J03_min95_2204 - J0395_resi_hyd),
         ':', color='limegreen', alpha=1, lw=2.5)

plt.plot(constraints_bb_2204[:, 0], constraints_bb_2204[:, 1], '--',
         label='CB22', alpha=1., color='#6CCAFF', lw=2)

plt.plot(sigmav_bb_2204[:, 0], sigmav_bb_2204[:, 1], '-.',
         c='silver', lw=2, zorder=0)

plt.xlim(sigmav_bb_2204[0, 0], sigmav_bb_2204[-1, 0])

plt.text(0.7, 0.85, r'$b\bar{b}$', transform=ax1.transAxes,
         horizontalalignment='center', size=30)

plt.annotate(r'<$\sigma\nu$>$_\mathrm{th}$', (1000, 3e-26))

plt.xscale('log')
plt.yscale('log')

plt.xlabel('m$_{\chi}$ [GeV]', size=20)
plt.ylabel(r'<$\sigma\nu$> [cm$^3$ s$^{-1}$]', size=20)


legend_elements = [mpatches.Patch(color='limegreen', alpha=0.8),
                   mpatches.Patch(color='k', alpha=0.8),
                   Line2D([0], [0], color='#6CCAFF',
                          linestyle='--', lw=2)]
leg = plt.legend(legend_elements, ['MHD', 'DMO', 'CB22'], loc=2)

bapad = plt.rcParams['legend.borderaxespad']
fontsize = plt.rcParams['font.size']
axline = plt.rcParams['axes.linewidth']
pad_pixels = (bapad*fontsize + axline) / 72. * fig.dpi
inv = axes[0].transAxes.inverted()

# Inverse transform two points on the display and find the relative distance
pad_axes = inv.transform((pad_pixels, 0)) - inv.transform((0, 0))
pad_xaxis = pad_axes[0]
# Find how may pixels there are on the x-axis
x_pixels = (axes[0].transAxes.transform((1, 0))
            - axes[0].transAxes.transform((0, 0)))
# Compute the ratio between the pixel offset and the total amount of pixels
pad_xaxis = (pad_pixels - 2.5)/x_pixels[0]

legend_elements = [Line2D([0], [0], color='k', label='Frag',
                          linestyle='-', lw=2.5),
                   Line2D([0], [0], color='k', label='Res',
                          linestyle=':', lw=2.5)]
legend1 = plt.legend(legend_elements, ['Fragile', 'Resilient'],
                     loc=(pad_xaxis, 0.53))

plt.gca().add_artist(leg)
plt.gca().add_artist(legend1)

t1, t2, t3 = leg.get_texts()
# here we create the distinct instance
t1._fontproperties = t2._fontproperties.copy()
t3.set_size(16)

# ax2.tick_params(labelleft=False)
plt.xlabel(r'm$_{\chi}$ $\left[\mathrm{GeV} \right]$', size=20)
plt.yticks(10**np.array([-26., -25, -24, -23, -22, -21, -20]),
               # labels=('', '19', '', '21', '', '23')
           )

plt.savefig(path_name_res + '/Cross_only1.png', bbox_inches='tight')
plt.savefig(path_name_res + '/Cross_only1.pdf', bbox_inches='tight')
# plt.show()

# -------------- Vmax_hist ----------------------------------------------------

fig, _ = plt.subplots(1, 2, figsize=(7, 3.5))

plt.subplots_adjust(wspace=0, hspace=0)

minS, maxS = minnmaxxS(3)
bines = np.linspace(10 ** minS, 10 ** maxS, 20)

ax1 = plt.subplot(121)

plt.hist(datos_Js_frag_dmo[:, 3], log=False,
               color='teal', alpha=0.6, label=r'Frag',
               bins=bines, hatch='//', edgecolor='k', lw=0,
         histtype='stepfilled')

plt.hist(datos_Js_resi_dmo[:, 3], log=False,
         label=r'Res', color='k', alpha=0.5,
         bins=bines, histtype='stepfilled')

plt.xlim(10 ** minS, 10 ** maxS)
plt.ylim(bottom=0.9, top=100)

plt.yscale('log')

# plt.xticks([1, 10, 30, 50, 70, 100])
plt.legend(handlelength=0.9, title='DMO')

plt.xlabel(r'$V_\mathrm{max}$ [km s$^{-1}$]', size=20)

ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)

plt.hist(datos_Js_frag_hyd[:, 3], log=False,
               color='yellowgreen', alpha=0.6, label=r'Frag',
               bins=bines, hatch='//', edgecolor='k', lw=0,
         histtype='stepfilled')

plt.hist(datos_Js_resi_hyd[:, 3], log=False,
         label=r'Res', color=darkgreen, alpha=0.6,
         bins=bines, histtype='stepfilled')

# plt.xticks([1, 10, 30, 50, 70, 100])
ax2.tick_params(labelleft=False)

plt.legend(handlelength=0.9, title='MHD', loc=2)

plt.yscale('log')
plt.xlabel(r'$V_\mathrm{max}$ [km s$^{-1}$]', size=20)

plt.savefig(path_name_res + '/Vmax_hist_linear' + end_str + '.png',
 bbox_inches='tight')
plt.savefig(path_name_res + '/Vmax_hist_linear' + end_str + '.pdf',
bbox_inches='tight')

fig, _ = plt.subplots(1, 2, figsize=(8, 4))

plt.subplots_adjust(wspace=0, hspace=0)

minS, maxS = minnmaxxS(3)
bines = np.geomspace(10 ** minS, 10 ** maxS, 20)

ax1 = plt.subplot(121)

plt.hist(datos_Js_frag_dmo[:, 3], log=False,
               color='teal', alpha=0.6, label=r'Frag',
               bins=bines, hatch='//', edgecolor='k', lw=0,
         histtype='stepfilled')

plt.hist(datos_Js_resi_dmo[:, 3], log=False,
         label=r'Res', color='k', alpha=0.5,
         bins=bines, histtype='stepfilled')

plt.xlim(0.1, 8)
plt.ylim(bottom=0.9, top=300)

plt.yscale('log')
plt.xscale('log')

plt.legend(handlelength=0.9, title='DMO', loc=2)

plt.xticks([0.1, 1, 8], labels=('0.1', '1', '8'))
plt.yticks([1, 10, 100], labels=('1', '10', '100'))

plt.xlabel(r'$V_\mathrm{max}$ [km s$^{-1}$]', size=20)

ax2 = plt.subplot(122)

plt.hist(datos_Js_frag_hyd[:, 3], log=False,
               color='yellowgreen', alpha=0.6, label=r'Frag',
               bins=bines, hatch='//', edgecolor='k', lw=0,
         histtype='stepfilled')

plt.hist(datos_Js_resi_hyd[:, 3], log=False,
         label=r'Res', color=darkgreen, alpha=0.6,
         bins=bines, histtype='stepfilled')

ax2.tick_params(labelleft=False)

plt.xlim(0.1, 8)
plt.ylim(bottom=0.9, top=300)

plt.yscale('log')
plt.xscale('log')

plt.legend(handlelength=0.9, title='MHD', loc=2)

plt.xticks([0.1, 1, 8], labels=('', '1', '8'))

plt.yscale('log')
plt.xlabel(r'$V_\mathrm{max}$ [km s$^{-1}$]', size=20)
plt.savefig(path_name_res + '/Vmax_hist_geom' + end_str + '.png',
bbox_inches='tight')
plt.savefig(path_name_res + '/Vmax_hist_geom' + end_str + '.pdf',
 bbox_inches='tight')
# plt.show()
# ----------------------- Vmax - J (z==DistEarth) 2x4 -----------------

fig, axes = plt.subplots(nrows=2, ncols=4, sharey=True,
                         figsize=(15, 6))

plt.subplots_adjust(wspace=0, hspace=0)

x_col = 3
y_col = 0
z_col = 2

x_label = r'$V_\mathrm{max}$ [km s$^{-1}$]'
y1_label = r'log$_{10}$ (J$_\mathrm{S}$ [GeV$^2$ cm$^{-5}$])'
y2_label = r'log$_{10}$ (J$_{03}$ [GeV$^2$ cm$^{-5}$])'

# vminn = 1e-3
# vmaxx = np.log10(150)  # 10 ** 1.5
vminn = np.log10(perc_total(z_col, 5))
vmaxx = np.log10(perc_total(z_col, 95))
print(vminn, vmaxx)
norm = mcb.Normalize(vminn, vmaxx)

minns, maxxs = minnmaxxS(y_col)
minns = 17.5
maxxs = 23.

minnx = perc_total(x_col, 0) * 0.7
maxxx = 12  # perc_total(x_col, 100) * 1.3

plt.subplot(241)

plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.tick_params('x', labelbottom=False)

plt.text(1., 1.01, 'DMO', horizontalalignment='center',
         verticalalignment='bottom', transform=axes[0, 0].transAxes)

plt.scatter(datos_Js_frag_dmo[:, x_col],
            np.log10(datos_Js_frag_dmo[:, y_col]),
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_frag_dmo[:, z_col]))))

plt.ylabel(y1_label, fontsize=16)

plt.subplot(242)

plt.tick_params('y', labelleft=False)
plt.tick_params('x', labelbottom=False)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.axvspan(0.935 * maxxx, maxxx, color='k')

plt.scatter(datos_Js_resi_dmo[:, x_col],
            np.log10(datos_Js_resi_dmo[:, y_col]),
            c=np.log10(datos_Js_resi_dmo[:, z_col]),
            marker='+', s=100, linewidths=1,
            cmap=colormapp, vmin=vminn, vmax=vmaxx)

plt.subplot(243)

plt.text(1., 1.01, 'MHD', horizontalalignment='center',
         verticalalignment='bottom', transform=axes[0, 2].transAxes)

plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.tick_params('y', labelleft=False)
plt.tick_params('x', labelbottom=False)
plt.axvspan(minnx, minnx * 1.04, color='k')

plt.scatter(datos_Js_frag_hyd[:, x_col],
            np.log10(datos_Js_frag_hyd[:, y_col]),
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_frag_hyd[:, z_col]))))

plt.subplot(244)

plt.tick_params('y', labelleft=False)
plt.tick_params('x', labelbottom=False)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')

plt.scatter(datos_Js_resi_hyd[:, x_col],
            np.log10(datos_Js_resi_hyd[:, y_col]),
            c=np.log10(datos_Js_resi_hyd[:, z_col]),
            marker='+', s=100, linewidths=1,
            cmap=colormapp, vmin=vminn, vmax=vmaxx)

plt.subplot(245)

plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')

plt.ylabel(y2_label, fontsize=16)
plt.xlabel(x_label, size=20)

plt.scatter(datos_J03_frag_dmo[:, x_col],
            np.log10(datos_J03_frag_dmo[:, y_col]),
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_frag_dmo[:, z_col]))))

plt.subplot(246)

plt.tick_params('y', labelleft=False)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.axvspan(0.935 * maxxx, maxxx, color='k')

plt.xlabel(x_label, size=20)

plt.scatter(datos_J03_resi_dmo[:, x_col],
            np.log10(datos_J03_resi_dmo[:, y_col]),
            c=np.log10(datos_J03_resi_dmo[:, z_col]),
            marker='+', s=100, linewidths=1,
            cmap=colormapp, vmin=vminn, vmax=vmaxx)

plt.subplot(247)

plt.tick_params('y', labelleft=False)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.axvspan(minnx, minnx * 1.04, color='k')

plt.xlabel(x_label, size=20)

plt.scatter(datos_J03_frag_hyd[:, x_col],
            np.log10(datos_J03_frag_hyd[:, y_col]),
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_frag_hyd[:, z_col]))))

plt.subplot(248)

plt.tick_params('y', labelleft=False)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')

plt.xlabel(x_label, size=20)

plt.scatter(datos_J03_resi_hyd[:, x_col],
            np.log10(datos_J03_resi_hyd[:, y_col]),
            c=np.log10(datos_J03_resi_hyd[:, z_col]),
            marker='+', s=100, linewidths=1,
            cmap=colormapp, vmin=vminn, vmax=vmaxx)

for ii in range(5, 8):
    plt.subplot(2, 4, ii)
    plt.xticks([0.1, 1, 10], labels=('0.1', '1', ''))

plt.subplot(2, 4, 8)
plt.xticks([0.1, 1, 10], labels=('0.1', '1', '10'))

for ii in range(1, 5):
    plt.subplot(2, 4, ii)
    plt.yticks((18, 19, 20, 21, 22, 23),
               labels=('18', '', '20', '', '22', ''))

cax, kw = colorbarr.make_axes([ax for ax in axes.flat])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
c2 = plt.colorbar(sm, cax=cax, extend='both')
# c2.set_label(r'log$_{10}$(D$_\mathrm{Earth}$ [kpc])', fontsize=20)
c2.set_label(r'D$_\mathrm{Earth}$ [kpc]', fontsize=20)
yticks = c2.get_ticks()
print(yticks)
# c2.set_ticklabels([str(10 ** i)[:4] for i in yticks])
c2.set_ticks(np.log10([0.5, 1., 2, 5., 10., 30]),
             labels=['0.5', '1', '2', '5', '10', '30'])

plt.savefig(path_name_res + '/VmaxJs_full_old' + end_str + '.png',
            bbox_inches='tight')
plt.savefig(path_name_res + '/VmaxJs_full_old' + end_str + '.pdf',
            bbox_inches='tight')

# ----------------------- Vmax - J (z==DistEarth) 2x4 -----------------

fig, axes = plt.subplots(nrows=2, ncols=4, sharey=True,
                         figsize=(15, 6))

plt.subplots_adjust(wspace=0, hspace=0)

x_col = 3
y_col = 0
z_col = 2

x_label = r'$V_\mathrm{max}$ [km s$^{-1}$]'
y1_label = r'log$_{10}$ (J$_\mathrm{factor}$ [GeV$^2$ cm$^{-5}$])'
y2_label = r'log$_{10}$ (J$_{03}$ [GeV$^2$ cm$^{-5}$])'

plt.text(1., 0.92, r'Fragile, J$_\mathrm{s}$',
         horizontalalignment='center',
         verticalalignment='top', transform=axes[0, 0].transAxes,
         backgroundcolor='w')
plt.text(1., 0.19, r'Resilient, J$_\mathrm{s}$',
         horizontalalignment='center',
         verticalalignment='top', transform=axes[1, 0].transAxes,
         backgroundcolor='w')
plt.text(1., 0.92, r'Fragile, J$_\mathrm{03}$',
         horizontalalignment='center',
         verticalalignment='top', transform=axes[0, 2].transAxes,
         backgroundcolor='w')
plt.text(0.96, 0.19, r'Resilient, J$_\mathrm{03}$',
         horizontalalignment='center',
         verticalalignment='top', transform=axes[1, 2].transAxes,
         backgroundcolor='w')

plt.text(-0.27, 0., y1_label,
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
minns = 17.5
maxxs = 23.

minnx = 0.08 #perc_total(x_col, 0) * 0.7
maxxx = 12  # perc_total(x_col, 100) * 1.3

plt.subplot(241)

plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')

plt.title('DMO', size=20)

plt.scatter(datos_Js_frag_dmo[:, x_col],
            np.log10(datos_Js_frag_dmo[:, y_col]),
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_frag_dmo[:, z_col]))))


# plt.text(1., 1.15, r'J$_\mathrm{s}$', horizontalalignment='center',
#          verticalalignment='bottom', transform=axes[0, 0].transAxes,
#          size=22)


plt.subplot(242)

plt.title('MHD', size=20)
plt.tick_params('y', labelleft=False)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.axvspan(0.935 * maxxx, maxxx, color='k')

plt.scatter(datos_Js_frag_hyd[:, x_col],
            np.log10(datos_Js_frag_hyd[:, y_col]),
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_frag_hyd[:, z_col]))))

plt.subplot(243)


# plt.text(1., 1.15, r'J$_\mathrm{03}$', horizontalalignment='center',
#          verticalalignment='bottom', transform=axes[0, 2].transAxes,
#          size=22)
plt.title('DMO', size=20)

plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.tick_params('y', labelleft=False)
plt.axvspan(minnx, minnx * 1.04, color='k')

plt.scatter(datos_J03_frag_dmo[:, x_col],
            np.log10(datos_J03_frag_dmo[:, y_col]),
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_frag_dmo[:, z_col]))))

plt.subplot(244)
plt.title('MHD', size=20)
plt.tick_params('y', labelleft=False)
plt.tick_params('x', labelbottom=False)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')

plt.scatter(datos_J03_frag_hyd[:, x_col],
            np.log10(datos_J03_frag_hyd[:, y_col]),
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_frag_hyd[:, z_col]))))

plt.subplot(245)

plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')

# plt.ylabel(y2_label, fontsize=16)
plt.xlabel(x_label, size=20)

plt.scatter(datos_Js_resi_dmo[:, x_col],
            np.log10(datos_Js_resi_dmo[:, y_col]),
            c=np.log10(datos_Js_resi_dmo[:, z_col]),
            marker='+', s=100, linewidths=1,
            cmap=colormapp, vmin=vminn, vmax=vmaxx)

plt.subplot(246)

plt.tick_params('y', labelleft=False)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.axvspan(0.935 * maxxx, maxxx, color='k')

plt.xlabel(x_label, size=20)

plt.scatter(datos_Js_resi_hyd[:, x_col],
            np.log10(datos_Js_resi_hyd[:, y_col]),
            c=np.log10(datos_Js_resi_hyd[:, z_col]),
            marker='+', s=100, linewidths=1,
            cmap=colormapp, vmin=vminn, vmax=vmaxx)

plt.subplot(247)

plt.tick_params('y', labelleft=False)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.axvspan(minnx, minnx * 1.04, color='k')

plt.xlabel(x_label, size=20)

plt.scatter(datos_J03_resi_dmo[:, x_col],
            np.log10(datos_J03_resi_dmo[:, y_col]),
            c=np.log10(datos_J03_resi_dmo[:, z_col]),
            marker='+', s=100, linewidths=1,
            cmap=colormapp, vmin=vminn, vmax=vmaxx)

plt.subplot(248)

plt.tick_params('y', labelleft=False)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')

plt.xlabel(x_label, size=20)

plt.scatter(datos_J03_resi_hyd[:, x_col],
            np.log10(datos_J03_resi_hyd[:, y_col]),
            c=np.log10(datos_J03_resi_hyd[:, z_col]),
            marker='+', s=100, linewidths=1,
            cmap=colormapp, vmin=vminn, vmax=vmaxx)

plt.xlabel(x_label, size=20)

for ii in range(1, 5):
    plt.subplot(2, 4, ii)
    plt.xticks([0.1, 1, 10], labels=('', '', ''))

for ii in range(5, 8):
    plt.subplot(2, 4, ii)
    plt.xticks([0.1, 1, 10], labels=('0.1', '1', ''))

plt.subplot(2, 4, 8)
plt.xticks([0.1, 1, 10], labels=('0.1', '1', '10'))

for ii in range(1, 5):
    plt.subplot(2, 4, ii)
    plt.yticks((18, 19, 20, 21, 22, 23),
               labels=('18', '', '20', '', '22', ''))

cax, kw = colorbarr.make_axes([ax for ax in axes.flat])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
c2 = plt.colorbar(sm, cax=cax, extend='both')
# c2.set_label(r'log$_{10}$(D$_\mathrm{Earth}$ [kpc])', fontsize=20)
c2.set_label(r'D$_\mathrm{Earth}$ [kpc]', fontsize=20)
yticks = c2.get_ticks()
print(yticks)
# c2.set_ticklabels([str(10 ** i)[:4] for i in yticks])
c2.set_ticks(np.log10([0.5, 1., 2, 5., 10., 25]),
             labels=['0.5', '1', '2', '5', '10', '25'])

plt.savefig(path_name_res + '/VmaxJs_full' + end_str + '.png',
            bbox_inches='tight')
plt.savefig(path_name_res + '/VmaxJs_full' + end_str + '.pdf',
            bbox_inches='tight')
plt.show()
# ------------------------ DgcJs ----------------------------------------------

fig, axes = plt.subplots(nrows=2, ncols=2, sharey=True,  # sharey=True,
                         figsize=(7, 9))

plt.subplots_adjust(wspace=0, hspace=0)

legend_elements = [
    Line2D([0], [0], marker='o', color='k', label='Frag',
           markerfacecolor='w', ls='',
           markersize=8, mew=2.5),
    Line2D([0], [0], marker='P', color='w', label='Res',
           markerfacecolor='k', markersize=12)]

x_col = 1
y_col = 0
z_col = 3

x_label = r'D$_\mathrm{GC}$ [kpc]'

plt.text(-0.25, 0., r'log$_{10}$ ($J_\mathrm{s}$ [GeV$^2$ cm$^{-5}$])',
         horizontalalignment='center',
         verticalalignment='center', transform=axes[0, 0].transAxes,
         rotation=90)
plt.text(1., 0.92, r'Fragile',
         horizontalalignment='center',
         verticalalignment='top', transform=axes[0, 0].transAxes,
         backgroundcolor='w')
plt.text(1., 0.18, r'Resilient',
         horizontalalignment='center',
         verticalalignment='top', transform=axes[1, 0].transAxes,
         backgroundcolor='w')

# vminn = 1e-3
# vmaxx = np.log10(150)  # 10 ** 1.5
vminn = 0 #np.log10(perc_total(z_col, 5))
vmaxx = np.log10(perc_total(z_col, 95))
print(vminn, vmaxx, 10 ** vminn, 10 ** vmaxx)
norm = mcb.Normalize(vminn, vmaxx)

minns, maxxs = minnmaxxS(y_col)
minns = 17.5
maxxs = 23.1

minnx = 0.1  # perc_total(x_col, 0) * 0.99
maxxx = perc_total(x_col, 100) * 1.5

plt.subplot(221)
plt.title('DMO', fontsize=20)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.tick_params('x', labelbottom=False)

plt.scatter(datos_Js_frag_dmo[:, x_col],
            np.log10(datos_Js_frag_dmo[:, y_col]),
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_frag_dmo[:, z_col]))))

plt.subplot(223)

plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.xlabel(x_label, size=20)

plt.scatter(datos_Js_resi_dmo[:, x_col],
            np.log10(datos_Js_resi_dmo[:, y_col]),
            c=np.log10(datos_Js_resi_dmo[:, z_col]),
            marker='+', s=100, linewidths=1,
            cmap=colormapp, vmin=vminn, vmax=vmaxx)

plt.subplot(222)

plt.title('MHD', fontsize=20)

plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.tick_params('y', labelleft=False)
plt.tick_params('x', labelbottom=False)

plt.scatter(datos_Js_frag_hyd[:, x_col],
            np.log10(datos_Js_frag_hyd[:, y_col]),
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_frag_hyd[:, z_col]))))

plt.subplot(224)

plt.tick_params('y', labelleft=False)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.xlabel(x_label, size=20)

plt.scatter(datos_Js_resi_hyd[:, x_col],
            np.log10(datos_Js_resi_hyd[:, y_col]),
            c=np.log10(datos_Js_resi_hyd[:, z_col]),
            marker='+', s=100, linewidths=1,
            cmap=colormapp, vmin=vminn, vmax=vmaxx)

plt.yticks((19, 20, 21, 22, 23), labels=('19', '', '21', '', '23'))
#
aa = plt.subplot(2, 2, 3)
aa.set_xticks([0.1, 1, 10, 100], labels=('0.1', '1', '10', ''))
#
aa = plt.subplot(2, 2, 4)
aa.set_xticks([0.1, 1, 10, 100], labels=('0.1', '1', '10', '100'))


sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
c2 = plt.colorbar(sm, ax=axes,
                  extend='both', spacing='proportional',
                  location='bottom')
c2.ax.tick_params(axis='x', direction='out')
c2.set_label(r'$V_\mathrm{max}$ [km/s]', fontsize=20)
yticks = c2.get_ticks()
# c2.set_ticklabels([str(10 ** i)[:4] for i in yticks])
c2.set_ticks(np.log10([1., 2., 4, 6]),
             labels=['1', '2', '4', '6'])

plt.savefig(path_name_res + '/DgcJs' + end_str + '.png',
            bbox_inches='tight')
plt.savefig(path_name_res + '/DgcJs' + end_str + '.pdf',
            bbox_inches='tight')

plt.show()
'''
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
try:
    path_name = ('/home/porrassa/Desktop/WIMPS_project/'
                 'Physnet_outputs_repops/2025/'
                 '/2025_to120_angles'
                 )

    path_name_res = ('/home/porrassa/Desktop/WIMPS_project/'
                 'Physnet_outputs_repops/2025/'
                 '/2025_to120_angles')
except FileNotFoundError:
    path_name = ('/home/saraporras/Desktop/WIMPSproject/'
                 'compiled_results'
                 '/2024_resilient_const_to120_rint')

    path_name_res = ('/home/saraporras/Desktop/WIMPSproject/'
                     'compiled_results/'
                     '2024_resilient_const_to120_rint_SHVFnorm')



print(os.getcwd())
print(os.listdir(path_name))
final_size = (500, 1, 6)
plot_res = True
plot_frag = True


end_str = '_to120'
# if plot_res:
#     end_str = '_res'
#
# if plot_frag:
#     end_str = '_frag'
#
# if plot_res and plot_frag:
#     end_str = '_both'

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

print('datos_Js_frag_dmo')
rr_ss = funct_repop.R_s(
    V=datos_Js_frag_dmo[:, 3], C=datos_Js_frag_dmo[:, 5], cosmo_H_0=67.7)
print(sum(rr_ss > datos_Js_frag_dmo[:, 2]))
aaa = rr_ss > datos_Js_frag_dmo[:, 2]
datos_Js_frag_dmo = datos_Js_frag_dmo [~aaa, :]
print(sum(datos_Js_frag_dmo[:, 3]<=1.), np.shape(datos_Js_frag_dmo))

print('datos_Js_frag_hyd')
rr_ss = funct_repop.R_s(
    V=datos_Js_frag_hyd[:, 3], C=datos_Js_frag_hyd[:, 5], cosmo_H_0=67.7)
print(sum(rr_ss > datos_Js_frag_hyd[:, 2]))
aaa = rr_ss > datos_Js_frag_hyd[:, 2]
datos_Js_frag_hyd = datos_Js_frag_hyd [~aaa, :]
print(sum(datos_Js_frag_hyd[:, 3]<=1.), np.shape(datos_Js_frag_hyd))

print('datos_Js_resi_dmo')
rr_ss = funct_repop.R_s(
    V=datos_Js_resi_dmo[:, 3], C=datos_Js_resi_dmo[:, 5], cosmo_H_0=67.7)
print(sum(rr_ss > datos_Js_resi_dmo[:, 2]))
aaa = rr_ss > datos_Js_resi_dmo[:, 2]
datos_Js_resi_dmo = datos_Js_resi_dmo [~aaa, :]
print(sum(datos_Js_resi_dmo[:, 3]<=1.), np.shape(datos_Js_resi_dmo))

print('datos_Js_resi_hyd')
rr_ss = funct_repop.R_s(
    V=datos_Js_resi_hyd[:, 3], C=datos_Js_resi_hyd[:, 5], cosmo_H_0=67.7)
print(sum(rr_ss > datos_Js_resi_hyd[:, 2]))
aaa = rr_ss > datos_Js_resi_hyd[:, 2]
datos_Js_resi_hyd = datos_Js_resi_hyd [~aaa, :]
print(sum(datos_Js_resi_hyd[:, 3]<=1.), np.shape(datos_Js_resi_hyd))




rr_ss = funct_repop.R_s(
    V=datos_J03_frag_dmo[:, 3], C=datos_J03_frag_dmo[:, 5], cosmo_H_0=67.7)
print(sum(rr_ss > datos_J03_frag_dmo[:, 2]))
aaa = rr_ss > datos_J03_frag_dmo[:, 2]
datos_J03_frag_dmo = datos_J03_frag_dmo [~aaa, :]
print(sum(datos_J03_frag_dmo[:, 3]<=1.), np.shape(datos_J03_frag_dmo))


rr_ss = funct_repop.R_s(
    V=datos_J03_frag_hyd[:, 3], C=datos_J03_frag_hyd[:, 5], cosmo_H_0=67.7)
print(sum(rr_ss > datos_J03_frag_hyd[:, 2]))
aaa = rr_ss > datos_J03_frag_hyd[:, 2]
datos_J03_frag_hyd = datos_J03_frag_hyd [~aaa, :]
print(sum(datos_J03_frag_hyd[:, 3]<=1.), np.shape(datos_J03_frag_hyd))

rr_ss = funct_repop.R_s(
    V=datos_J03_resi_dmo[:, 3], C=datos_J03_resi_dmo[:, 5], cosmo_H_0=67.7)
print(sum(rr_ss > datos_J03_resi_dmo[:, 2]))
aaa = rr_ss > datos_J03_resi_dmo[:, 2]
datos_J03_resi_dmo = datos_J03_resi_dmo [~aaa, :]
print(sum(datos_J03_resi_dmo[:, 3]<=1.), np.shape(datos_J03_resi_dmo))


rr_ss = funct_repop.R_s(
    V=datos_J03_resi_hyd[:, 3], C=datos_J03_resi_hyd[:, 5], cosmo_H_0=67.7)
print(sum(rr_ss > datos_J03_resi_hyd[:, 2]))
aaa = rr_ss > datos_J03_resi_hyd[:, 2]
datos_J03_resi_hyd = datos_J03_resi_hyd [~aaa, :]
print(sum(datos_J03_resi_hyd[:, 3]<=1.), np.shape(datos_J03_resi_hyd))

path_name_res = path_name_res + '/figures'
if not os.path.exists(path_name_res):
    os.makedirs(path_name_res)

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




fig, axes = plt.subplots(nrows=2, ncols=2, sharey=True,  # sharey=True,
                         figsize=(7, 9))

xx_plot = np.geomspace(0.1, 120)
plt.subplots_adjust(wspace=0, hspace=0)

legend_elements = [
    Line2D([0], [0], marker='o', color='k', label='Frag',
           markerfacecolor='w', ls='',
           markersize=8, mew=2.5),
    Line2D([0], [0], marker='P', color='w', label='Res',
           markerfacecolor='k', markersize=12)]

x_col = 3
y_col = 5
z_col = 0

x_label = r'$V_\mathrm{max}$ [km/s]'
# y1_label = r'log$_{10}$ (J$_\mathrm{S}$ [GeV$^2$ cm$^{-5}$])'
# y2_label = r'log$_{10}$ (J$_{03}$ [GeV$^2$ cm$^{-5}$])'

plt.text(-0.25, 0., r'log$_{10}$ ($c_\mathrm{V}$)',
         horizontalalignment='center',
         verticalalignment='center', transform=axes[0, 0].transAxes,
         rotation=90)
plt.text(1., 0.95, r'Fragile',
         horizontalalignment='center',
         verticalalignment='top', transform=axes[0, 0].transAxes,
         backgroundcolor='w')
plt.text(1., 0.95, r'Resilient',
         horizontalalignment='center',
         verticalalignment='top', transform=axes[1, 0].transAxes,
         backgroundcolor='w')

# vminn = 1e-3
# vmaxx = np.log10(150)  # 10 ** 1.5
vminn = np.log10(perc_total(z_col, 5))
vmaxx = np.log10(perc_total(z_col, 95))
print(vminn, vmaxx, 10 ** vminn, 10 ** vmaxx)
norm = mcb.Normalize(vminn, vmaxx)

minns, maxxs = minnmaxxS(y_col)
# minns = 18.2
# maxxs = 23.

minnx = 0.1  # perc_total(x_col, 0) * 0.99
maxxx = 120  # perc_total(x_col, 100) * 1.3

plt.subplot(221)
plt.title('DMO', fontsize=20)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.tick_params('x', labelbottom=False)

aaa, aaa_cov = curve_fit(
        Cv_Mol2021_redshift0,
        xdata=datos_Js_frag_hyd[:, x_col],
        ydata=datos_Js_frag_hyd[:, y_col],
        p0=[10**4],
    )
print(aaa)
print(aaa_cov)
plt.plot(xx_plot, np.log10(Cv_Mol2021_redshift0(xx_plot, aaa[0])),
         zorder=0, ls=':', c='gray', alpha=0.5)
plt.plot(xx_plot, np.log10(Cv_Mol2021_redshift0(xx_plot, 283862 )),
         zorder=0, ls='--', c='red', alpha=0.5)

plt.scatter(datos_Js_frag_dmo[:, x_col],
            np.log10(datos_Js_frag_dmo[:, y_col]),
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_frag_dmo[:, z_col]))))

plt.subplot(223)

plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.xlabel(x_label, size=20)

plt.scatter(datos_Js_resi_dmo[:, x_col],
            np.log10(datos_Js_resi_dmo[:, y_col]),
            c=np.log10(datos_Js_resi_dmo[:, z_col]),
            marker='+', s=100, linewidths=1,
            cmap=colormapp, vmin=vminn, vmax=vmaxx)

aaa, aaa_cov = curve_fit(
        Cv_Mol2021_redshift0,
        xdata=datos_Js_resi_dmo[:, x_col],
        ydata=datos_Js_resi_dmo[:, y_col],
        p0=[10**4],
        # sigma=yerr_dmo[xx_pos:]
    )
print(aaa)
print(aaa_cov)
plt.plot(xx_plot, np.log10(Cv_Mol2021_redshift0(xx_plot, aaa[0])),
         zorder=0, ls=':', c='gray', alpha=0.5)
plt.plot(xx_plot, np.log10(Cv_Mol2021_redshift0(xx_plot, 283862 )),
         zorder=0, ls='--', c='red', alpha=0.5)

plt.subplot(222)

plt.title('MHD', fontsize=20)

plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.tick_params('y', labelleft=False)
plt.tick_params('x', labelbottom=False)

plt.scatter(datos_Js_frag_hyd[:, x_col],
            np.log10(datos_Js_frag_hyd[:, y_col]),
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_frag_hyd[:, z_col]))))

aaa, aaa_cov = curve_fit(
        Cv_Mol2021_redshift0,
        xdata=datos_Js_frag_hyd[:, x_col],
        ydata=datos_Js_frag_hyd[:, y_col],
        p0=[10**4],
    )
print(aaa)
print(aaa_cov)
plt.plot(xx_plot, np.log10(Cv_Mol2021_redshift0(xx_plot, aaa[0])),
         zorder=0, ls=':', c='gray', alpha=0.5)
plt.plot(xx_plot, np.log10(Cv_Mol2021_redshift0(xx_plot, 192879)),
         zorder=0, ls='--', c='red', alpha=0.5)



plt.subplot(224)

plt.tick_params('y', labelleft=False)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.xlabel(x_label, size=20)

plt.scatter(datos_Js_resi_hyd[:, x_col],
            np.log10(datos_Js_resi_hyd[:, y_col]),
            c=np.log10(datos_Js_resi_hyd[:, z_col]),
            marker='+', s=100, linewidths=1,
            cmap=colormapp, vmin=vminn, vmax=vmaxx)

aaa, aaa_cov = curve_fit(
        Cv_Mol2021_redshift0,
        xdata=datos_Js_resi_hyd[:, x_col],
        ydata=datos_Js_resi_hyd[:, y_col],
        p0=[10**4],
    )
print(aaa)
print(aaa_cov)
plt.plot(xx_plot, np.log10(Cv_Mol2021_redshift0(xx_plot, aaa[0])),
         zorder=0, ls=':', c='gray', alpha=0.5)
plt.plot(xx_plot, np.log10(Cv_Mol2021_redshift0(xx_plot, 192879)),
         zorder=0, ls='--', c='red', alpha=0.5)


# plt.yticks((19, 20, 21, 22, 23), labels=('19', '', '21', '', '23'))

aa = plt.subplot(2, 2, 3)
aa.set_xticks([0.1, 1, 10, 100], labels=('0.1', '1', '10', ''))

aa = plt.subplot(2, 2, 4)
aa.set_xticks([0.1, 1, 10, 100], labels=('0.1', '1', '10', '100'))

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
c2 = plt.colorbar(sm, ax=axes,
                  extend='both', spacing='proportional',
                  location='bottom'
                  )
c2.set_label(r'log10(J-factor)', fontsize=20)
yticks = c2.get_ticks()
print(yticks)
c2.ax.tick_params(axis='x', direction='out')
# c2.set_ticklabels([str(10 ** i)[:4] for i in yticks])
# c2.set_ticks(np.log10([5., 10., 20, 50, 80.]),
#              labels=['5', '10', '20', '50', '80'])

plt.savefig(path_name_res + '/VmaxCv' + end_str + '.png',
            bbox_inches='tight')
plt.savefig(path_name_res + '/VmaxCv' + end_str + '.pdf',
            bbox_inches='tight')

# ----------------------- DistEarth - Jss (z==Vmax) 2x2 ----------------

fig, axes = plt.subplots(nrows=2, ncols=2, sharey=True,  # , sharex=True,
                         figsize=(7, 9))

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
y1_label = r'fragile'
y2_label = r'resilient'

plt.text(-0.25, 0., r'log$_{10}$ ($J_\mathrm{s}$ [GeV$^2$ cm$^{-5}$])',
         horizontalalignment='center',
         verticalalignment='center', transform=axes[0, 0].transAxes,
         rotation=90)
plt.text(1., 0.94, r'Fragile',
         horizontalalignment='center',
         verticalalignment='top', transform=axes[0, 0].transAxes,
         backgroundcolor='w')
plt.text(1., 0.17, r'Resilient',
         horizontalalignment='center',
         verticalalignment='top', transform=axes[1, 0].transAxes,
         backgroundcolor='w')

vminn = np.log10(1.)  # np.log10(perc_total(z_col, 5))
vmaxx = np.log10(perc_total(z_col, 95))
print(vminn, vmaxx, 10 ** vminn, 10 ** vmaxx)
print(perc_total(z_col, 0))
# vminn = np.log10(0.1)
# vmaxx = np.log10(200)
norm = mcb.Normalize(vminn, vmaxx)

minns, maxxs = minnmaxxS(y_col)
minns = minns - 0.3
maxxs = maxxs + 0.3

minnx = 0.1  # perc_total(x_col, 0) * 0.7
maxxx = perc_total(x_col, 100) * 1.4

plt.subplot(221)

plt.title('DMO', fontsize=20)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')

plt.scatter(datos_Js_frag_dmo[:, x_col],
            np.log10(datos_Js_frag_dmo[:, y_col]),
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_frag_dmo[:, z_col]))))

plt.subplot(223)

plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.xlabel(x_label, size=20)

plt.scatter(datos_Js_resi_dmo[:, x_col],
            np.log10(datos_Js_resi_dmo[:, y_col]),
            c=np.log10(datos_Js_resi_dmo[:, z_col]),
            marker='+', s=100, linewidths=1,
            cmap=colormapp, vmin=vminn, vmax=vmaxx)

plt.subplot(222)

plt.title('MHD', fontsize=20)

plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.tick_params('y', labelleft=False)

plt.scatter(datos_Js_frag_hyd[:, x_col],
            np.log10(datos_Js_frag_hyd[:, y_col]),
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_frag_hyd[:, z_col]))))

plt.subplot(224)

plt.tick_params('y', labelleft=False)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.xlabel(x_label, size=20)

plt.scatter(datos_Js_resi_hyd[:, x_col],
            np.log10(datos_Js_resi_hyd[:, y_col]),
            c=np.log10(datos_Js_resi_hyd[:, z_col]),
            marker='+', s=100, linewidths=1,
            cmap=colormapp, vmin=vminn, vmax=vmaxx)

axes[0][0].set_xticklabels([])
axes[0][1].set_xticklabels([])

plt.yticks((19, 20, 21, 22, 23), labels=('19', '', '21', '', '23'))

aa = plt.subplot(2, 2, 3)
aa.set_xticks([0.1, 1, 10, 100], labels=('0.1', '1', '10', ''))

aa = plt.subplot(2, 2, 4)
aa.set_xticks([0.1, 1, 10, 100], labels=('0.1', '1', '10', '100'))

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
c2 = plt.colorbar(sm, ax=axes,
                  extend='both', spacing='proportional',
                  location='bottom')
c2.set_label(r'$V_\mathrm{max}$ $\left[\mathrm{km}\,\,\mathrm{s}^{-1}\right]$',
             fontsize=20)
c2.ax.tick_params(axis='x', direction='out')
yticks = c2.get_ticks()
print(yticks)
# c2.set_ticklabels([str(10 ** i)[:4] for i in yticks])
c2.set_ticks(np.log10([1., 2, 5., 10., 20., 50, 90.]),
             labels=[1, 2, 5, 10, 20, 50, 90]
             )

plt.savefig(path_name_res + '/DEarthJs_Vmax' + end_str + '.png',
            bbox_inches='tight')
plt.savefig(path_name_res + '/DEarthJs_Vmax' + end_str + '.pdf',
            bbox_inches='tight')

# plt.show()
# ----------------------- Vmax - Jss (z==DistEarth) 2x2 ----------------

fig, axes = plt.subplots(nrows=2, ncols=2, sharey=True,  # , sharex=True,
                         figsize=(7, 9))

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

plt.text(-0.25, 0., r'log$_{10}$ ($J_\mathrm{s}$ [GeV$^2$ cm$^{-5}$])',
         horizontalalignment='center',
         verticalalignment='center', transform=axes[0, 0].transAxes,
         rotation=90)
plt.text(1., 0.95, r'Fragile',
         horizontalalignment='center',
         verticalalignment='top', transform=axes[0, 0].transAxes,
         backgroundcolor='w')
plt.text(1., 0.17, r'Resilient',
         horizontalalignment='center',
         verticalalignment='top', transform=axes[1, 0].transAxes,
         backgroundcolor='w')

vminn = np.log10(perc_total(z_col, 0))
vmaxx = np.log10(perc_total(z_col, 100))
print(vminn, vmaxx, 10 ** vminn, 10 ** vmaxx)
vminn = np.log10(0.0999)
if to8:
    vmaxx = np.log10(perc_total(z_col, 100))
else:
    vmaxx = np.log10(200)
norm = mcb.Normalize(vminn, vmaxx)

minns, maxxs = minnmaxxS(y_col)
minns = minns - 0.3
maxxs = maxxs + 0.3

print('vmax Jmax', minns, maxxs)

minnx = 0.1  # perc_total(x_col, 0) * 0.7
maxxx = perc_total(x_col, 100) * 1.4

plt.subplot(221)

plt.title('DMO', fontsize=20)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')

plt.scatter(datos_Js_frag_dmo[:, x_col],
            np.log10(datos_Js_frag_dmo[:, y_col]),
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_frag_dmo[:, z_col]))))

plt.subplot(223)

plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.xlabel(x_label, size=20)

plt.scatter(datos_Js_resi_dmo[:, x_col],
            np.log10(datos_Js_resi_dmo[:, y_col]),
            c=np.log10(datos_Js_resi_dmo[:, z_col]),
            marker='+', s=100, linewidths=1,
            cmap=colormapp, vmin=vminn, vmax=vmaxx)

plt.subplot(222)

plt.title('MHD', fontsize=20)

plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.tick_params('y', labelleft=False)

plt.scatter(datos_Js_frag_hyd[:, x_col],
            np.log10(datos_Js_frag_hyd[:, y_col]),
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_frag_hyd[:, z_col]))))

plt.subplot(224)

plt.tick_params('y', labelleft=False)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.xlabel(x_label, size=20)

plt.scatter(datos_Js_resi_hyd[:, x_col],
            np.log10(datos_Js_resi_hyd[:, y_col]),
            c=np.log10(datos_Js_resi_hyd[:, z_col]),
            marker='+', s=100, linewidths=1,
            cmap=colormapp, vmin=vminn, vmax=vmaxx)

axes[0][0].set_xticklabels([])
axes[0][1].set_xticklabels([])

plt.yticks((19, 20, 21, 22, 23), labels=('19', '', '21', '', '23'))

aa = plt.subplot(2, 2, 3)
aa.set_xticks([0.1, 1, 10, 100], labels=('0.1', '1', '10', ''))

aa = plt.subplot(2, 2, 4)
aa.set_xticks([0.1, 1, 10, 100], labels=('0.1', '1', '10', '100'))

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
c2 = plt.colorbar(sm, ax=axes,
                  extend='both', spacing='proportional',
                  location='bottom')
c2.set_label(r'D$_\mathrm{Earth}$ [kpc]', fontsize=20)
c2.ax.tick_params(axis='x', direction='out')
yticks = c2.get_ticks()
print(yticks)
c2.set_ticklabels([str(10 ** i)[:4] for i in yticks])
c2.set_ticks(np.log10([0.1, 1., 10., 100.]),
             labels=['0.1', '1', '10', '100'])

plt.savefig(path_name_res + '/VmaxJs_Js' + end_str + '.png',
            bbox_inches='tight')
plt.savefig(path_name_res + '/VmaxJs_Js' + end_str + '.pdf',
            bbox_inches='tight')
'''
# ----------------------- Ang size - J (z==Vmax) 2x2 -----------------
xx_plot = np.geomspace(0.1, 120)
fig, axes = plt.subplots(nrows=2, ncols=2, sharey=True,  # sharey=True,
                         figsize=(7, 9))

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

x_label = r'$\theta_\mathrm{S}\,\,\left[\mathrm{deg}\right]$'
y1_label = r'log$_{10}$ (J$_\mathrm{S}$ [GeV$^2$ cm$^{-5}$])'
y2_label = r'log$_{10}$ (J$_{03}$ [GeV$^2$ cm$^{-5}$])'

plt.text(-0.25, 0., r'log$_{10}$ ($J_\mathrm{s}$ [GeV$^2$ cm$^{-5}$])',
         horizontalalignment='center',
         verticalalignment='center', transform=axes[0, 0].transAxes,
         rotation=90)
plt.text(1., 0.95, r'Fragile',
         horizontalalignment='center',
         verticalalignment='top', transform=axes[0, 0].transAxes,
         backgroundcolor='w')
plt.text(1., 0.17, r'Resilient',
         horizontalalignment='center',
         verticalalignment='top', transform=axes[1, 0].transAxes,
         backgroundcolor='w')

# vminn = 1e-3
# vmaxx = np.log10(150)  # 10 ** 1.5
vminn = np.log10(5)  # perc_total(z_col, 5))
vmaxx = np.log10(perc_total(z_col, 95))
print(vminn, vmaxx, 10 ** vminn, 10 ** vmaxx)
norm = mcb.Normalize(vminn, vmaxx)

minns, maxxs = minnmaxxS(y_col)
minns = 18.2
maxxs = 23.35

minnx = 0.1  # perc_total(x_col, 0) * 0.99
maxxx = 100  # perc_total(x_col, 100) * 1.3

plt.subplot(221)
plt.title('DMO', fontsize=20)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.tick_params('x', labelbottom=False)

xx_bins = np.geomspace(0.1, 100, num=15)
xx_bins_mean = np.sqrt(xx_bins[1:] * xx_bins[:-1])

nn_bins = []
std_bins = []
number_bins = []
for ii in range(len(xx_bins_mean)):
    aaa_range = ((datos_Js_frag_dmo[:, x_col] > xx_bins[ii])
                 * (datos_Js_frag_dmo[:, x_col] <= xx_bins[ii+1]))
    nn_bins.append(np.mean(np.log10(datos_Js_frag_dmo[aaa_range, y_col])))
    std_bins.append(np.std(np.log10(datos_Js_frag_dmo[aaa_range, y_col])))
    number_bins.append(sum(aaa_range))
    plt.text(x=xx_bins_mean[ii]*0.9, y=22.5 + 0.3*(-1)**(ii%2),
             s=sum(aaa_range), fontsize=10)
nn_bins = np.array(nn_bins)
std_bins = np.array(std_bins)
number_bins = np.array(number_bins)
plt.errorbar(xx_bins_mean, nn_bins,
            yerr=std_bins,
             ls='',
            c='k', zorder=100)
print(nn_bins)
aaa_range = np.isnan(nn_bins)
aaa, cov = curve_fit(pow_law,
                     xdata=xx_bins_mean[~aaa_range],
                     ydata=nn_bins[~aaa_range],
                     sigma=std_bins[~aaa_range],
                     p0=(18, 0.6)
)
print('datos_Js_frag_dmo')
for i in range(len(aaa)):
    print(aaa[i], np.sqrt(np.diag(cov))[i])

plt.plot(xx_plot, (pow_law(xx_plot, j0=aaa[0], m=aaa[1])),
         zorder=1000, ls=':', c='r', alpha=1)

aaa, cov = curve_fit(pow_law,
                     xdata=xx_bins_mean[~aaa_range],
                     ydata=nn_bins[~aaa_range],
                     p0=(18, 0.6)
)
print('datos_Js_frag_dmo')
for i in range(len(aaa)):
    print(aaa[i], np.sqrt(np.diag(cov))[i])

plt.plot(xx_plot, (pow_law(xx_plot, j0=aaa[0], m=aaa[1])),
         zorder=1000, ls=':', c='fuchsia', alpha=1)

aaa, cov = curve_fit(pow_law,
                     xdata=xx_bins_mean[~aaa_range],
                     ydata=nn_bins[~aaa_range],
                     sigma=1/np.sqrt(number_bins)[~aaa_range],
                     p0=(18, 0.6)
)
print('datos_Js_frag_dmo')
for i in range(len(aaa)):
    print(aaa[i], np.sqrt(np.diag(cov))[i])

plt.plot(xx_plot, (pow_law(xx_plot, j0=aaa[0], m=aaa[1])),
         zorder=1000, ls=':', c='black', alpha=1)

aaa, cov = curve_fit(pow_law,
                     xdata=datos_Js_frag_dmo[:, x_col],
                     ydata=np.log10(datos_Js_frag_dmo[:, y_col]),
                     p0=(18, 0.6)
)
print('datos_Js_frag_dmo')
for i in range(len(aaa)):
    print(aaa[i], np.sqrt(np.diag(cov))[i])

plt.plot(xx_plot, (pow_law(xx_plot, j0=aaa[0], m=aaa[1])),
         zorder=1000, ls=':', c='gray', alpha=1)


plt.scatter(datos_Js_frag_dmo[:, x_col],
            np.log10(datos_Js_frag_dmo[:, y_col]),
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_frag_dmo[:, z_col]))))


plt.subplot(223)

plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.xlabel(x_label, size=20)

plt.scatter(datos_Js_resi_dmo[:, x_col],
            np.log10(datos_Js_resi_dmo[:, y_col]),
            c=np.log10(datos_Js_resi_dmo[:, z_col]),
            marker='+', s=100, linewidths=1,
            cmap=colormapp, vmin=vminn, vmax=vmaxx)

ddd = np.argsort(datos_Js_resi_dmo[:, y_col])
datos_Js_resi_dmo = datos_Js_resi_dmo[ddd, :]

aaa, cov = curve_fit(pow_law,
                     xdata=datos_Js_resi_dmo[:-1, x_col],
                     ydata=np.log10(datos_Js_resi_dmo[:-1, y_col]),
                     p0=(20.12, 1.06),
                     # bounds=[[19., 0.1], [21., 1.2]]
)
print('datos_Js_resi_dmo')
for i in range(len(aaa)):
    print(aaa[i], np.sqrt(np.diag(cov))[i])

plt.plot(xx_plot, (pow_law(xx_plot, j0=aaa[0], m=aaa[1])),
         zorder=0, ls=':', c='gray', alpha=0.5)


plt.subplot(222)

plt.title('MHD', fontsize=20)

plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.tick_params('y', labelleft=False)
plt.tick_params('x', labelbottom=False)

plt.scatter(datos_Js_frag_hyd[:, x_col],
            np.log10(datos_Js_frag_hyd[:, y_col]),
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_frag_hyd[:, z_col]))))

aaa, cov = curve_fit(pow_law,
                     xdata=datos_Js_frag_hyd[:, x_col],
                     ydata=np.log10(datos_Js_frag_hyd[:, y_col]),
                     p0=(18, 0.6)
)
print('datos_Js_frag_hyd')
for i in range(len(aaa)):
    print(aaa[i], np.sqrt(np.diag(cov))[i])

plt.plot(xx_plot, (pow_law(xx_plot, j0=aaa[0], m=aaa[1])),
         zorder=0, ls=':', c='gray', alpha=0.5)


plt.subplot(224)

plt.tick_params('y', labelleft=False)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.xlabel(x_label, size=20)

plt.scatter(datos_Js_resi_hyd[:, x_col],
            np.log10(datos_Js_resi_hyd[:, y_col]),
            c=np.log10(datos_Js_resi_hyd[:, z_col]),
            marker='+', s=100, linewidths=1,
            cmap=colormapp, vmin=vminn, vmax=vmaxx)

aaa, cov = curve_fit(pow_law,
                     xdata=datos_Js_resi_hyd[:, x_col],
                     ydata=np.log10(datos_Js_resi_hyd[:, y_col]),
                     p0=(18, 0.6)
)
print('datos_Js_resi_hyd')
for i in range(len(aaa)):
    print(aaa[i], np.sqrt(np.diag(cov))[i])

plt.plot(xx_plot, (pow_law(xx_plot, j0=aaa[0], m=aaa[1])),
         zorder=0, ls=':', c='gray', alpha=0.5)


plt.yticks((19, 20, 21, 22, 23), labels=('19', '', '21', '', '23'))

aa = plt.subplot(2, 2, 3)
aa.set_xticks([0.1, 1, 10, 100], labels=('0.1', '1', '10', ''))

aa = plt.subplot(2, 2, 4)
aa.set_xticks([0.1, 1, 10, 100], labels=('0.1', '1', '10', '100'))

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
c2 = plt.colorbar(sm, ax=axes,
                  extend='both', spacing='proportional',
                  location='bottom'
                  )
c2.set_label(r'$V_\mathrm{max}$ [km s$^{-1}$]', fontsize=20)
yticks = c2.get_ticks()
print(yticks)
c2.ax.tick_params(axis='x', direction='out')
c2.set_ticklabels([str(10 ** i)[:4] for i in yticks])
c2.set_ticks(np.log10([5., 10., 20, 50, 80.]),
             labels=['5', '10', '20', '50', '80'])

plt.savefig(path_name_res + '/AngJss_full' + end_str + '.png',
            bbox_inches='tight')
plt.savefig(path_name_res + '/AngJss_full' + end_str + '.pdf',
            bbox_inches='tight')
plt.show()

# ----------------------- Ang size - J (z==Vmax) 2x2 -----------------

fig, axes = plt.subplots(nrows=2, ncols=2, sharey=True,  # sharey=True,
                         figsize=(7, 9))

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

x_label = r'$\Omega\,\,\left[\mathrm{sr}\right]$'
y1_label = r'log$_{10}$ (J$_\mathrm{S}$ [GeV$^2$ cm$^{-5}$])'
y2_label = r'log$_{10}$ (J$_{03}$ [GeV$^2$ cm$^{-5}$])'

plt.text(-0.25, 0., r'log$_{10}$ ($J_\mathrm{s}$ [GeV$^2$ cm$^{-5}$])',
         horizontalalignment='center',
         verticalalignment='center', transform=axes[0, 0].transAxes,
         rotation=90)
plt.text(1., 0.92, r'Fragile',
         horizontalalignment='center',
         verticalalignment='top', transform=axes[0, 0].transAxes,
         backgroundcolor='w')
plt.text(1., 0.18, r'Resilient',
         horizontalalignment='center',
         verticalalignment='top', transform=axes[1, 0].transAxes,
         backgroundcolor='w')

# vminn = 1e-3
# vmaxx = np.log10(150)  # 10 ** 1.5
vminn = np.log10(5)  # perc_total(z_col, 5))
vmaxx = np.log10(perc_total(z_col, 95))
print(vminn, vmaxx, 10 ** vminn, 10 ** vmaxx)
norm = mcb.Normalize(vminn, vmaxx)

minns, maxxs = minnmaxxS(y_col)
minns = 18.2
maxxs = 23.6

minnx = 1e-5  # perc_total(x_col, 0) * 0.99
maxxx = 10  # perc_total(x_col, 100) * 1.3

plt.subplot(221)
plt.title('DMO', fontsize=20)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.tick_params('x', labelbottom=False)
# plt.figure()
# print('we start debug')
# for i in range(len(datos_Js_frag_dmo[:, 0])):
#     if (datos_Js_frag_dmo[i, 2] < funct_repop.R_s(
#                     V=datos_Js_frag_dmo[i, 3],
#                     C=datos_Js_frag_dmo[i, 5],
#                     cosmo_H_0=67.7
#                 )):
#         print(i, datos_Js_frag_dmo[i, :])
# plt.plot(datos_Js_frag_dmo[:, 2], marker='.')
# plt.plot(2*np.pi * (1 - np.cos(datos_Js_frag_dmo[:, 2])), marker='+')
# plt.plot(funct_repop.R_s(
#                     V=datos_Js_frag_dmo[:, 3],
#                     C=datos_Js_frag_dmo[:, 5],
#                     cosmo_H_0=67.7
#                 ), marker='x')
# plt.show()

rr_ss = funct_repop.R_s(
                    V=datos_Js_frag_dmo[:, 3],
                    C=datos_Js_frag_dmo[:, 5],
                    cosmo_H_0=67.7
                )
real_theta = np.arcsin(rr_ss/datos_Js_frag_dmo[:, 2])
# np.nanmin((
#     np.pi/2. * np.ones_like(datos_Js_frag_dmo[:, 0]),
#     np.arcsin(rr_ss/datos_Js_frag_dmo[:, 2])), axis=0)

plt.scatter(2*np.pi * (1 - np.cos(real_theta)),
            np.log10(datos_Js_frag_dmo[:, y_col]),
            marker='+', linewidths=1,)
plt.scatter(2*np.pi *
            (1 - np.sqrt(datos_Js_frag_dmo[:, 2]**2. - rr_ss**2.)
             /datos_Js_frag_dmo[:, 2]),
            np.log10(datos_Js_frag_dmo[:, y_col]),
            marker='s', c='none', lw=1, edgecolors='orange')

plt.scatter(2*np.pi * (1 - np.cos(np.pi/180*datos_Js_frag_dmo[:, 4])),
            np.log10(datos_Js_frag_dmo[:, y_col]),
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_frag_dmo[:, z_col]))))

plt.subplot(223)

plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.xlabel(x_label, size=20)

plt.scatter(2*np.pi *
            (1 - np.sqrt(
                datos_Js_resi_dmo[:, 2]**2.
                - funct_repop.R_s(
                    V=datos_Js_resi_dmo[:, 3],
                    C=datos_Js_resi_dmo[:, 5],
                    cosmo_H_0=67.7
                )**2.
            )/datos_Js_resi_dmo[:, 2]),
            np.log10(datos_Js_resi_dmo[:, y_col]),
            c=np.log10(datos_Js_resi_dmo[:, z_col]),
            marker='+', s=100, linewidths=1,
            cmap=colormapp, vmin=vminn, vmax=vmaxx)

plt.subplot(222)

plt.title('MHD', fontsize=20)

plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.tick_params('y', labelleft=False)
plt.tick_params('x', labelbottom=False)

plt.scatter(2*np.pi *
            (1 - np.sqrt(
                datos_Js_frag_hyd[:, 2]**2.
                - funct_repop.R_s(
                    V=datos_Js_frag_hyd[:, 3],
                    C=datos_Js_frag_hyd[:, 5],
                    cosmo_H_0=67.7
                )**2.
            )/datos_Js_frag_hyd[:, 2]),
            np.log10(datos_Js_frag_hyd[:, y_col]),
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_frag_hyd[:, z_col]))))

plt.subplot(224)

plt.tick_params('y', labelleft=False)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.xlabel(x_label, size=20)

plt.scatter(2*np.pi *
            (1 - np.sqrt(
                datos_Js_resi_hyd[:, 2]**2.
                - funct_repop.R_s(
                    V=datos_Js_resi_hyd[:, 3],
                    C=datos_Js_resi_hyd[:, 5],
                    cosmo_H_0=67.7
                )**2.
            )/datos_Js_resi_hyd[:, 2]),
            np.log10(datos_Js_resi_hyd[:, y_col]),
            c=np.log10(datos_Js_resi_hyd[:, z_col]),
            marker='+', s=100, linewidths=1,
            cmap=colormapp, vmin=vminn, vmax=vmaxx)

plt.yticks((19, 20, 21, 22, 23), labels=('19', '', '21', '', '23'))

aa = plt.subplot(2, 2, 3)
aa.set_xticks([1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1., 10],
              labels=(r'10$^{-5}$', '', r'10$^{-3}$', '', '0.1', '1', '')
              )
aa = plt.subplot(2, 2, 1)
aa.set_xticks([1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1., 10])
aa = plt.subplot(2, 2, 2)
aa.set_xticks([1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1., 10])
#
aa = plt.subplot(2, 2, 4)
aa.set_xticks([1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1., 10],
              labels=(r'10$^{-5}$', '', r'10$^{-3}$', '', '0.1', '1', '10')
              )

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
c2 = plt.colorbar(sm, ax=axes,
                  extend='both', spacing='proportional',
                  location='bottom')
c2.set_label(r'$V_\mathrm{max}$ [km s$^{-1}$]', fontsize=20)
yticks = c2.get_ticks()
print(yticks)
c2.ax.tick_params(axis='x', direction='out')
c2.set_ticklabels([str(10 ** i)[:4] for i in yticks])
c2.set_ticks(np.log10([5., 10., 20, 50, 90.]),
             labels=['5', '10', '20', '50', '90'])

plt.savefig(path_name_res + '/solidAngJss_full' + end_str + '.png',
            bbox_inches='tight')
plt.savefig(path_name_res + '/solidAngJss_full' + end_str + '.pdf',
            bbox_inches='tight')
# plt.show()

# ------------------------ DgcJs ----------------------------------------------

fig, axes = plt.subplots(nrows=2, ncols=2, sharey=True,  # sharey=True,
                         figsize=(7, 9))

plt.subplots_adjust(wspace=0, hspace=0)

legend_elements = [
    Line2D([0], [0], marker='o', color='k', label='Frag',
           markerfacecolor='w', ls='',
           markersize=8, mew=2.5),
    Line2D([0], [0], marker='P', color='w', label='Res',
           markerfacecolor='k', markersize=12)]

x_col = 1
y_col = 0
z_col = 3

x_label = r'D$_\mathrm{GC}$ [kpc]'

plt.text(-0.25, 0., r'log$_{10}$ ($J_\mathrm{s}$ [GeV$^2$ cm$^{-5}$])',
         horizontalalignment='center',
         verticalalignment='center', transform=axes[0, 0].transAxes,
         rotation=90)
plt.text(1., 0.94, r'Fragile',
         horizontalalignment='center',
         verticalalignment='top', transform=axes[0, 0].transAxes,
         backgroundcolor='w')
plt.text(1., 0.18, r'Resilient',
         horizontalalignment='center',
         verticalalignment='top', transform=axes[1, 0].transAxes,
         backgroundcolor='w')

# vminn = 1e-3
# vmaxx = np.log10(150)  # 10 ** 1.5
vminn = np.log10(5)  # perc_total(z_col, 5))
vmaxx = np.log10(perc_total(z_col, 95))
print(vminn, vmaxx, 10 ** vminn, 10 ** vmaxx)
norm = mcb.Normalize(vminn, vmaxx)

minns, maxxs = minnmaxxS(y_col)
minns = 18.2
maxxs = 23.

minnx = 0.1  # perc_total(x_col, 0) * 0.99
maxxx = perc_total(x_col, 100) * 1.5

plt.subplot(221)
plt.title('DMO', fontsize=20)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.tick_params('x', labelbottom=False)

plt.scatter(datos_Js_frag_dmo[:, x_col],
            np.log10(datos_Js_frag_dmo[:, y_col]),
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_frag_dmo[:, z_col]))))

plt.subplot(223)

plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.xlabel(x_label, size=20)

plt.scatter(datos_Js_resi_dmo[:, x_col],
            np.log10(datos_Js_resi_dmo[:, y_col]),
            c=np.log10(datos_Js_resi_dmo[:, z_col]),
            marker='+', s=100, linewidths=1,
            cmap=colormapp, vmin=vminn, vmax=vmaxx)

plt.subplot(222)

plt.title('MHD', fontsize=20)

plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.tick_params('y', labelleft=False)
plt.tick_params('x', labelbottom=False)

plt.scatter(datos_Js_frag_hyd[:, x_col],
            np.log10(datos_Js_frag_hyd[:, y_col]),
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_frag_hyd[:, z_col]))))

plt.subplot(224)

plt.tick_params('y', labelleft=False)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.xlabel(x_label, size=20)

plt.scatter(datos_Js_resi_hyd[:, x_col],
            np.log10(datos_Js_resi_hyd[:, y_col]),
            c=np.log10(datos_Js_resi_hyd[:, z_col]),
            marker='+', s=100, linewidths=1,
            cmap=colormapp, vmin=vminn, vmax=vmaxx)

plt.yticks((19, 20, 21, 22, 23), labels=('19', '', '21', '', '23'))

aa = plt.subplot(2, 2, 3)
aa.set_xticks([0.1, 1, 10, 100], labels=('0.1', '1', '10', '100'))

aa = plt.subplot(2, 2, 4)
aa.set_xticks([0.1, 1, 10, 100], labels=('0.1', '1', '10', '100'))

dx = 4 / 72.
offset = trans.ScaledTranslation(dx, 0, fig.dpi_scale_trans)
for nn, label in enumerate(axes[1][0].xaxis.get_majorticklabels()):
    if nn == 3:
        label.set_transform(label.get_transform() - offset)

for nn, label in enumerate(axes[1][1].xaxis.get_majorticklabels()):
    if nn == 0:
        label.set_transform(label.get_transform() + offset)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
c2 = plt.colorbar(sm, ax=axes,
                  extend='both', spacing='proportional',
                  location='bottom')
c2.ax.tick_params(axis='x', direction='out')
c2.set_label(r'$V_\mathrm{max}$ [km/s]', fontsize=20)
yticks = c2.get_ticks()
# c2.set_ticklabels([str(10 ** i)[:4] for i in yticks])
c2.set_ticks(np.log10([5., 10., 20, 50, 80.]),
             labels=['5', '10', '20', '50', '80'])

plt.savefig(path_name_res + '/DgcJs' + end_str + '.png',
            bbox_inches='tight')
plt.savefig(path_name_res + '/DgcJs' + end_str + '.pdf',
            bbox_inches='tight')

# -------------------- J_hist -------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(4.5, 6.5))

plt.subplots_adjust(wspace=0, hspace=0)

minS, maxS = minnmaxxS(0)

bines = np.linspace(minS, maxS, 30)
# bines = np.linspace(19, 23, 40)

ax1 = plt.subplot(211)

plt.hist(np.log10(datos_Js_frag_dmo[:, 0]), log=False,
               color='teal', alpha=0.6,
               bins=bines, label=r'Frag',
         hatch='//', edgecolor='k',
         lw=0, histtype='stepfilled'
         )

plt.hist(np.log10(datos_Js_resi_dmo[:, 0]), log=False,
         label=r'Res', color='k', alpha=0.5,
         bins=bines, histtype='stepfilled')

Js95_frag_dmo = np.log10(np.percentile(datos_Js_frag_dmo[:, 0], 5))
Js95_resi_dmo = np.log10(np.percentile(datos_Js_resi_dmo[:, 0], 5))
print('Js, DMO')
print(Js95_resi_dmo, Js95_frag_dmo, Js95_resi_dmo - Js95_frag_dmo)

plt.axvline(Js95_frag_dmo, color='#004F7D', lw=1.5, ls='--')  # , alpha=0.6)
plt.axvline(Js95_resi_dmo, color='k', lw=1.5, ls='--')  # , alpha=0.5)

# plt.ylabel('Number of repopulations', size=20)
plt.xlim(minS, maxS)
plt.ylim(bottom=0., top=100)

plt.xticks((19, 20, 21, 22, 23), labels=('19', '20', '21', '22', ''))
# plt.yscale('log')

plt.legend(handlelength=0.9, title='DMO', title_fontsize=18,
           handletextpad=0.5, fontsize=16,
           loc=1)

ax2 = plt.subplot(212)

aaa = plt.hist(np.log10(datos_Js_frag_hyd[:, 0]), log=False,
               color='yellowgreen', alpha=0.6,
               bins=bines, label=r'Frag',
               hatch='//', edgecolor='k',
               lw=0, histtype='stepfilled'
               )
plt.hist(np.log10(datos_Js_resi_hyd[:, 0]), log=False,
         label=r'Res', color=darkgreen, alpha=0.6,
         bins=bines, histtype='stepfilled')

Js95_frag_hyd = np.log10(np.percentile(datos_Js_frag_hyd[:, 0], 5))
Js95_resi_hyd = np.log10(np.percentile(datos_Js_resi_hyd[:, 0], 5))
print('Js, Hydro')
print(Js95_resi_hyd, Js95_frag_hyd, Js95_resi_hyd - Js95_frag_hyd)

plt.axvline(Js95_frag_hyd, color='#006E0B', lw=1.5, ls='--')  # , alpha=0.6)
plt.axvline(Js95_resi_hyd, color='k', lw=1.5, ls='--')  # , alpha=0.6)

plt.xlim(minS, maxS)
plt.ylim(bottom=0., top=100)

plt.xlabel(r'log$_{10}$ (J$_\mathrm{S}$ [GeV$^2$ cm$^{-5}$])',
           fontsize=20)
plt.legend(handlelength=0.9, title='MHD', title_fontsize=18,
           handletextpad=0.5, fontsize=16)

# fig.text(0.5, 0.001,
#          'log$_{10}$ (J$_\mathrm{S}$ [GeV$^2$ cm$^{-5}$])',
#          ha='center',
#          va='center')

fig.text(0.001, 0.5, 'Number of repopulations',
         ha='center',
         va='center', rotation='vertical')

plt.xticks((19, 20, 21, 22, 23))
plt.yticks((0, 25, 50, 75, 100), labels=('0', '25', '50', '75', ''))

# dx = 10 / 72.
# offset = trans.ScaledTranslation(0, dx, fig.dpi_scale_trans)
# for nn, label in enumerate(ax1.yaxis.get_majorticklabels()):
#     if nn == 0:
#         label.set_transform(label.get_transform() + offset)
# for nn, label in enumerate(ax2.yaxis.get_majorticklabels()):
#     if nn == 4:
#         label.set_transform(label.get_transform() - offset)

plt.savefig(path_name_res + '/J_hist_vert' + end_str + '.png',
            bbox_inches='tight')
plt.savefig(path_name_res + '/J_hist_vert' + end_str + '.pdf',
            bbox_inches='tight')
plt.show()
# -------------------- J_hist -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(8.5, 4))
plt.subplots_adjust(wspace=0, hspace=0)
minS, maxS = minnmaxxS(0)
bines = np.linspace(minS, maxS, 30)
# bines = np.linspace(19, 23, 40)
ax1 = plt.subplot(121)
plt.hist(np.log10(datos_Js_frag_dmo[:, 0]), log=False,
               color='teal', alpha=0.6,
               bins=bines, label=r'Frag',
         hatch='//', edgecolor='k',
         lw=0, histtype='stepfilled'
         )
plt.hist(np.log10(datos_Js_resi_dmo[:, 0]), log=False,
         label=r'Res', color='k', alpha=0.5,
         bins=bines, histtype='stepfilled')
Js95_frag_dmo = np.log10(np.percentile(datos_Js_frag_dmo[:, 0], 5))
Js95_resi_dmo = np.log10(np.percentile(datos_Js_resi_dmo[:, 0], 5))
print('Js, DMO')
print(Js95_resi_dmo, Js95_frag_dmo, Js95_resi_dmo - Js95_frag_dmo)
plt.axvline(Js95_frag_dmo, color='#004F7D', lw=1.5, ls='--')  # , alpha=0.6)
plt.axvline(Js95_resi_dmo, color='k', lw=1.5, ls='--')  # , alpha=0.5)
plt.ylabel('Number of repopulations', size=20)
plt.xlim(minS, maxS)
plt.ylim(bottom=0., top=85)
# plt.yscale('log')
# plt.xlabel(r'log$_{10}$ (J$_\mathrm{S}$ [GeV$^2$ cm$^{-5}$])',
#            fontsize=20)
plt.legend(handlelength=0.9, title='DMO', title_fontsize=18,
           handletextpad=0.5, fontsize=16,
           loc=2, framealpha=1)
ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)
aaa = plt.hist(np.log10(datos_Js_frag_hyd[:, 0]), log=False,
               color='yellowgreen', alpha=0.6,
               bins=bines, label=r'Frag',
               hatch='//', edgecolor='k',
               lw=0, histtype='stepfilled'
               )
plt.hist(np.log10(datos_Js_resi_hyd[:, 0]), log=False,
         label=r'Res', color=darkgreen, alpha=0.6,
         bins=bines, histtype='stepfilled')
Js95_frag_hyd = np.log10(np.percentile(datos_Js_frag_hyd[:, 0], 5))
Js95_resi_hyd = np.log10(np.percentile(datos_Js_resi_hyd[:, 0], 5))
print('Js, Hydro')
print(Js95_resi_hyd, Js95_frag_hyd, Js95_resi_hyd - Js95_frag_hyd)
plt.axvline(Js95_frag_hyd, color='#006E0B', lw=1.5, ls='--')  # , alpha=0.6)
plt.axvline(Js95_resi_hyd, color='k', lw=1.5, ls='--')  # , alpha=0.6)
ax2.tick_params(labelleft=False)
plt.legend(handlelength=0.9, title='MHD', title_fontsize=18,
           handletextpad=0.5, fontsize=16)
fig.text(0.5, -0.03,
         'log$_{10}$ (J$_\mathrm{S}$ [GeV$^2$ cm$^{-5}$])',
         ha='center',
         va='center')
plt.xticks((19, 20, 21, 22))
plt.savefig(path_name_res + '/J_hist' + end_str + '.png',
            bbox_inches='tight')
plt.savefig(path_name_res + '/J_hist' + end_str + '.pdf',
            bbox_inches='tight')
# plt.show()
# -------------- Vmax_hist ----------------------------------------------------

fig, _ = plt.subplots(1, 2, figsize=(10, 5))

plt.subplots_adjust(wspace=0, hspace=0)

minS, maxS = minnmaxxS(3)
bines = np.linspace(10 ** minS, 10 ** maxS, 20)
bines = np.linspace(0., 120., 20)

ax1 = plt.subplot(121)

plt.hist(datos_Js_frag_dmo[:, 3], log=False,
               color='teal', alpha=0.6, label=r'Frag',
               bins=bines, hatch='//', edgecolor='k', lw=0,
         histtype='stepfilled')

plt.hist(datos_Js_resi_dmo[:, 3], log=False,
         label=r'Res', color='k', alpha=0.5,
         bins=bines, histtype='stepfilled')

plt.xlim(0, 120)
plt.ylim(bottom=0.9, top=110)

plt.yscale('log')

plt.xticks([0, 20, 40, 60, 80, 100])
plt.legend(handlelength=0.9, title='DMO')

plt.xlabel(r'$V_\mathrm{max}$ [km s$^{-1}$]', size=20)

ax2 = plt.subplot(122, sharey=ax1)

plt.hist(datos_Js_frag_hyd[:, 3], log=False,
               color='yellowgreen', alpha=0.6, label=r'Frag',
               bins=bines, hatch='//', edgecolor='k', lw=0,
         histtype='stepfilled')

plt.hist(datos_Js_resi_hyd[:, 3], log=False,
         label=r'Res', color=darkgreen, alpha=0.6,
         bins=bines, histtype='stepfilled')

plt.xlim(0, 120)
plt.xticks([0, 20, 40, 60, 80, 100, 120])
ax2.tick_params(labelleft=False)

plt.legend(handlelength=0.9, title='MHD')

plt.yscale('log')
plt.xlabel(r'$V_\mathrm{max}$ [km s$^{-1}$]', size=20)

plt.savefig(path_name_res + '/Vmax_hist_linear' + end_str + '.png',
            bbox_inches='tight')
plt.savefig(path_name_res + '/Vmax_hist_linear' + end_str + '.pdf',
            bbox_inches='tight')

fig, _ = plt.subplots(1, 2, figsize=(8, 4))

plt.subplots_adjust(wspace=0, hspace=0)

minS, maxS = minnmaxxS(3)
bines = np.geomspace(10 ** minS, 10 ** maxS, 20)

ax1 = plt.subplot(121)

plt.hist(datos_Js_frag_dmo[:, 3], log=False,
               color='teal', alpha=0.6, label=r'Frag',
               bins=bines, hatch='//', edgecolor='k', lw=0,
         histtype='stepfilled')

plt.hist(datos_Js_resi_dmo[:, 3], log=False,
         label=r'Res', color='k', alpha=0.5,
         bins=bines, histtype='stepfilled')


plt.xlim(0.1, 120)
plt.ylim(bottom=0.9, top=120)

plt.yscale('log')
plt.xscale('log')

plt.legend(handlelength=0.9, title='DMO', loc=2)

plt.xticks([0.1, 1, 10, 100], labels=('0.1', '1', '10', '100'))
plt.yticks([1, 10, 100], labels=('1', '10', '100'))

plt.legend(handlelength=0.9, title='DMO', loc=2)

plt.xlabel(r'$V_\mathrm{max}$ [km s$^{-1}$]', size=20)

ax2 = plt.subplot(122)

plt.hist(datos_Js_frag_hyd[:, 3], log=False,
               color='yellowgreen', alpha=0.6, label=r'Frag',
               bins=bines, hatch='//', edgecolor='k', lw=0,
         histtype='stepfilled')

plt.hist(datos_Js_resi_hyd[:, 3], log=False,
         label=r'Res', color=darkgreen, alpha=0.6,
         bins=bines, histtype='stepfilled')

ax2.tick_params(labelleft=False)

plt.xscale('log')
plt.yscale('log')

plt.xlim(0.1, 120)
plt.ylim(bottom=0.9, top=120)

plt.xticks([0.1, 1, 10, 100], labels=('', '1', '10', '100'))

plt.legend(handlelength=0.9, title='MHD', loc=2)

plt.xlabel(r'$V_\mathrm{max}$ [km s$^{-1}$]', size=20)

plt.savefig(path_name_res + '/Vmax_hist_geom' + end_str + '.png',
            bbox_inches='tight')
plt.savefig(path_name_res + '/Vmax_hist_geom' + end_str + '.pdf',
            bbox_inches='tight')

# ----------------------- Ang size - J (z==Vmax) 2x4 -----------------

fig, axes = plt.subplots(nrows=2, ncols=4, sharey=True,
                         figsize=(15, 6))

plt.subplots_adjust(wspace=0, hspace=0)

legend_elements = [
    Line2D([0], [0], marker='o', color='k', label='Frag',
           markerfacecolor='w', ls='',
           markersize=8, mew=2.5),
    Line2D([0], [0], marker='P', color='w', label='Res',
           markerfacecolor='k', markersize=12)]

x_col = 3
y_col = 5
z_col = 0

x_label = r'$V_\mathrm{max}$ [km s$^{-1}$]'
y1_label = 'Cv'  # r'log$_{10}$ (J$_\mathrm{S}$ [GeV$^2$ cm$^{-5}$])'
y2_label = 'Cv'  # r'log$_{10}$ (J$_{03}$ [GeV$^2$ cm$^{-5}$])'

# vminn = 1e-3
# vmaxx = np.log10(150)  # 10 ** 1.5
vminn = np.log10(perc_total(z_col, 5))
vmaxx = np.log10(perc_total(z_col, 95))
print(vminn, vmaxx)
norm = mcb.Normalize(vminn, vmaxx)

minns, maxxs = minnmaxxS(y_col)
# minns = 17.5
# maxxs = 23.1

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
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_frag_dmo[:, z_col]))))

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
            c=np.log10(datos_Js_resi_dmo[:, z_col]),
            marker='+', s=100, linewidths=1,
            cmap=colormapp, vmin=vminn, vmax=vmaxx)

plt.subplot(243)

plt.text(1., 1.01, 'MHD', horizontalalignment='center',
         verticalalignment='bottom', transform=axes[0, 2].transAxes)

plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.tick_params('y', labelleft=False)
plt.axvspan(minnx, minnx * 1.04, color='k')

plt.scatter(datos_Js_frag_hyd[:, x_col],
            np.log10(datos_Js_frag_hyd[:, y_col]),
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(datos_Js_frag_hyd[:, z_col]))))

plt.subplot(244)

plt.tick_params('y', labelleft=False)
plt.tick_params('x', labelbottom=False)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')

plt.scatter(datos_Js_resi_hyd[:, x_col],
            np.log10(datos_Js_resi_hyd[:, y_col]),
            c=np.log10(datos_Js_resi_hyd[:, z_col]),
            marker='+', s=100, linewidths=1,
            cmap=colormapp, vmin=vminn, vmax=vmaxx)

plt.subplot(245)

plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')

plt.ylabel(y2_label, fontsize=16)
plt.xlabel(x_label, size=20)

plt.scatter(datos_J03_frag_dmo[:, x_col],
            np.log10(datos_J03_frag_dmo[:, y_col]),
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_frag_dmo[:, z_col]))))

plt.subplot(246)

plt.tick_params('y', labelleft=False)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.axvspan(0.935 * maxxx, maxxx, color='k')

plt.xlabel(x_label, size=20)

plt.scatter(datos_J03_resi_dmo[:, x_col],
            np.log10(datos_J03_resi_dmo[:, y_col]),
            c=np.log10(datos_J03_resi_dmo[:, z_col]),
            marker='+', s=100, linewidths=1,
            cmap=colormapp, vmin=vminn, vmax=vmaxx)

plt.subplot(247)

plt.tick_params('y', labelleft=False)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')
plt.axvspan(minnx, minnx * 1.04, color='k')

plt.xlabel(x_label, size=20)

plt.scatter(datos_J03_frag_hyd[:, x_col],
            np.log10(datos_J03_frag_hyd[:, y_col]),
            c='none', lw=1, marker='o',
            edgecolors=cmap(norm(np.log10(datos_J03_frag_hyd[:, z_col]))))

plt.subplot(248)

plt.tick_params('y', labelleft=False)
plt.ylim(minns, maxxs)
plt.xlim(minnx, maxxx)
plt.xscale('log')

plt.xlabel(x_label, size=20)

plt.scatter(datos_J03_resi_hyd[:, x_col],
            np.log10(datos_J03_resi_hyd[:, y_col]),
            c=np.log10(datos_J03_resi_hyd[:, z_col]),
            marker='+', s=100, linewidths=1,
            cmap=colormapp, vmin=vminn, vmax=vmaxx)

for ii in range(5, 9):
    plt.subplot(2, 4, ii)
    plt.xticks([0.1, 1, 10], labels=('0.1', '1', '10'))
#
# for ii in range(1, 5):
#     plt.subplot(2, 4, ii)
#     plt.yticks((18, 19, 20, 21, 22, 23),
#                labels=('', '19', '', '21', '', '23'))

cax, kw = colorbarr.make_axes([ax for ax in axes.flat])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
c2 = plt.colorbar(sm, cax=cax, extend='both')
c2.set_label(r'J-factor', fontsize=20)
yticks = c2.get_ticks()
c2.ax.tick_params(axis='y', direction='out')
print(yticks)
# c2.set_ticklabels([str(10 ** i)[:4] for i in yticks])
# c2.set_ticks(np.log10([1., 2., 4, 6]),
#              labels=['1', '2', '4', '6'])

plt.savefig(path_name_res + '/VmaxCv_full' + end_str + '.png',
            bbox_inches='tight')
plt.savefig(path_name_res + '/VmaxCv_full' + end_str + '.pdf',
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
if to8:
    vmaxx = np.log10(perc_total(2, 100))
else:
    vmaxx = np.log10(200)
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

plt.text(1., 1.01, 'MHD', horizontalalignment='center',
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

if not to8:
    axes[1, 0].set_xticks([1, 4, 5])
    axes[1, 0].set_xticklabels([1, 4, 5], fontsize=12)

    axes[1, 3].set_xticks([0.1, 1, 10, 100], labels=('0.1', '1', '10', '100'))

cax, kw = colorbarr.make_axes([ax for ax in axes.flat])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
c2 = plt.colorbar(sm, cax=cax, extend='both')
# c2.set_label(r'log$_{10}$(D$_\mathrm{Earth}$ [kpc])', fontsize=20)
c2.set_label(r'D$_\mathrm{Earth}$ [kpc]', fontsize=20)
yticks = c2.get_ticks()
print(yticks)
# c2.set_ticklabels([str(10 ** i)[:4] for i in yticks])
c2.set_ticks(np.log10([2., 5, 10., 20., 50., 100.]),
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
minns = 10 ** minns * 0.8
maxxs = 10 ** maxxs * 1.2

minn03, maxx03 = minnmaxx03(y_col)
minn03 = 10 ** minn03 * 0.8
maxx03 = 10 ** maxx03 * 1.2

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

plt.text(1., 1.01, 'MHD', horizontalalignment='center',
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
plt.text(x=0.04, y=22.5, s='MHD', size=18)
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
plt.title('MHD', size=18)
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

plt.text(1., 1.01, 'MHD', horizontalalignment='center',
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
# plt.show()
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
plt.title('MHD', size=18)
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
# c2.set_label(r'log$_{10}$($V_\mathrm{max}$ [km/s])', fontsize=20)
c2.set_label(r'$V_\mathrm{max}$ [km/s])', fontsize=20)
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
plt.title('MHD', size=18)

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
# c2.set_label(r'log$_{10}$($V_\mathrm{max}$ [km/s])', fontsize=20)
c2.set_label(r'$V_\mathrm{max}$ [km/s]', fontsize=20)
yticks = c2.get_ticks()
c2.set_ticklabels([str(10 ** i)[:4] for i in yticks])

plt.savefig(path_name_res + '/Dgc_Dearth' + end_str + '.png',
            bbox_inches='tight')
plt.savefig(path_name_res + '/Dgc_Dearth' + end_str + '.pdf',
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
plt.title('MHD', size=18)
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
plt.title('MHD', size=18)
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

# ---------------- Dgc_hist ---------------------------------------------------
fig, _ = plt.subplots(2, 2, figsize=(10, 8))

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
aaaa = plt.legend(title=r'J$_\mathrm{S}$', loc=locc, handlelength=0.9)

plt.xscale('log')
plt.yscale('log')

plt.xlim(bines[0], bines[-1])

ax2 = plt.subplot(222, sharex=ax1, sharey=ax1)
plt.title('MHD', size=18)
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
plt.legend(title=r'J$_\mathrm{S}$', loc=locc, handlelength=0.9)
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
plt.legend(title=r'J$_{03}$', loc=locc, handlelength=0.9)

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
plt.legend(title=r'J$_{03}$', loc=locc, handlelength=0.9)

fig.text(0.05, 0.5, 'Number of repopulations', ha='center',
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
'''
