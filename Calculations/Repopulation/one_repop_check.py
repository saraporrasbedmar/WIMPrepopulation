import os

import scipy.optimize
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

import attemp_at_functions2 as funct_repop

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

#        Rmax[kpc]        Vmax[km/s]      Radius[Mpc]
Grand_dmo = np.loadtxt(
    '../Data_subhalo_simulations/dmo_table.txt', skiprows=3)
Grand_hydro = np.loadtxt(
    '../Data_subhalo_simulations/hydro_table.txt', skiprows=3)

Grand_hydro = Grand_hydro[Grand_hydro[:, 1] > np.min(Grand_dmo[:, 1]), :]


def read_config_file(ConfigFile):
    with open(ConfigFile, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml


path_outputs = 'outputs/' \
               'test_srds_and_Cv'
path_outputs = 'outputs/test1repop'

rerun_sims = True
# rerun_sims = False

if rerun_sims:

    if not os.path.exists(path_outputs):
        os.makedirs(path_outputs)

    path_input = 'input_files/input_1_repopulation.yml'

    input_data = read_config_file(path_input)
    print('dmo num subs over completion: ',
          funct_repop.SHVF_Grand2012_int(input_data['SHVF']['Vmax_completion'],
                                         input_data['SHVF']['RangeMax'],
                                         input_data['SHVF']['dmo']['bb'],
                                         input_data['SHVF']['dmo']['mm']))
    print('hydro num subs over completion: ',
          funct_repop.SHVF_Grand2012_int(input_data['SHVF']['Vmax_completion'],
                                         input_data['SHVF']['RangeMax'],
                                         input_data['SHVF']['hydro']['bb'],
                                         input_data['SHVF']['hydro']['mm']))

    print(funct_repop.SHVF_Grand2012_int(
        input_data['SHVF']['RangeMin'], input_data['SHVF']['RangeMax'],
        input_data['SHVF']['dmo']['bb'], input_data['SHVF']['dmo']['mm']))

    input_data['repopulations']['num_brightest'] = \
        funct_repop.SHVF_Grand2012_int(
            input_data['SHVF']['RangeMin'],
            input_data['SHVF']['RangeMax'],
            input_data['SHVF']['dmo']['bb'],
            input_data['SHVF']['dmo']['mm'])

    with open(path_input, 'w') as f:
        yaml.dump(input_data, f)

    funct_repop.main(['dmo', 'resilient', path_input, path_outputs])
    funct_repop.main(['dmo', 'fragile', path_input, path_outputs])

    input_data['repopulations']['num_brightest'] = \
        funct_repop.SHVF_Grand2012_int(
            input_data['SHVF']['RangeMin'],
            input_data['SHVF']['RangeMax'],
            input_data['SHVF']['hydro']['bb'],
            input_data['SHVF']['hydro']['mm'])

    with open(path_input, 'w') as f:
        yaml.dump(input_data, f)

    funct_repop.main(['hydro', 'resilient', path_input, path_outputs])
    funct_repop.main(['hydro', 'fragile', path_input, path_outputs])

datos_frag_hyd = np.loadtxt(path_outputs + '/Js_hydro_fragile_results.txt')
datos_frag_dmo = np.loadtxt(path_outputs + '/Js_dmo_fragile_results.txt')

datos_resi_hyd = np.loadtxt(path_outputs + '/Js_hydro_resilient_results.txt')
datos_resi_dmo = np.loadtxt(path_outputs + '/Js_dmo_resilient_results.txt')

input_data = read_config_file(path_outputs + '/input_data.yml')

# SHVF -----------------------------------------------------------------
x_cumul = np.geomspace(input_data['SHVF']['RangeMin'],
                       input_data['SHVF']['RangeMax'],
                       num=26)


def calcular_dNdV(Vmax):
    Vmax_cumul = np.zeros(len(x_cumul) - 1)

    for radius in range(len(Vmax_cumul)):
        aa = Vmax >= x_cumul[radius]
        bb = Vmax < x_cumul[radius + 1]

        Vmax_cumul[radius] = sum(aa * bb) / (
                x_cumul[radius + 1] - x_cumul[radius])

    return Vmax_cumul


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

    if plot:
        plt.plot(xx, yy, label=label, color=color, linestyle=style)
        plt.plot(xx, yy, '.', color=color, zorder=10)

        xxx = np.logspace(np.log10(2), np.log10(xx[-1]), 100)
        plt.plot(xxx, 10 ** fits[1] * xxx ** fits[0],
                 color=color, alpha=0.7,
                 linestyle='-', lw=2)

    return fits[0], fits[1], perr[0], perr[1]


Vmax_cumul_dmo_res = calcular_dNdV(datos_resi_dmo[:, 3])
Vmax_cumul_dmo_frag = calcular_dNdV(datos_frag_dmo[:, 3])
Vmax_cumul_hydro_res = calcular_dNdV(datos_resi_hyd[:, 3])
Vmax_cumul_hydro_frag = calcular_dNdV(datos_frag_hyd[:, 3])

x_cumul = (x_cumul[:-1] + x_cumul[1:]) / 2.

plt.figure(figsize=(10, 10))

plt.plot(x_cumul, Vmax_cumul_dmo_res, label='DMO res')
plt.plot(x_cumul, Vmax_cumul_dmo_frag, label='DMO frag')
plt.plot(x_cumul, Vmax_cumul_hydro_res, label='Hydro res')
plt.plot(x_cumul, Vmax_cumul_hydro_frag, label='Hydro frag')

fitsM_DMO, fitsB_DMO, _, _ = find_PowerLaw(
    x_cumul, Vmax_cumul_dmo_res,
    lim_inf=1.5, lim_sup=10., plot=True, label='DMO fit')
print(fitsM_DMO, fitsB_DMO)

fitsM_hydro, fitsB_hydro, _, _ = find_PowerLaw(
    x_cumul, Vmax_cumul_hydro_res, plot=True,
    lim_inf=1.5, lim_sup=10., color='limegreen', label='Hydro fit')
print(fitsM_hydro, fitsB_hydro)

plt.xscale('log')
plt.yscale('log')

plt.legend()

plt.savefig(path_outputs + '/SHVF.png', bbox_inches='tight')
plt.savefig(path_outputs + '/SHVF.pdf', bbox_inches='tight')


# plt.show()

# SRD ------------------------------------------------------------------

def encontrar_SRD_sinVol(data, Rvir=220):
    n_final = []
    for delta in range(len(bins) - 1):
        interval = (data / Rvir >= bins[delta]) \
                   * (data / Rvir <= bins[delta + 1])
        # interval = ((data[:, 2] / data[:, 5] >= bins[delta])
        #                 * (data[:, 2] / data[:, 5] <= bins[delta + 1]))
        n_final.append(sum(interval))
    return np.array(n_final)


def encontrar_SRD(data):
    n_final = []
    for delta in range(len(bins) - 1):
        interval = (data >= bins[delta]) * (data <= bins[delta + 1])
        vol = 4 / 3 * np.pi * (
                bins[delta + 1] ** 3 - bins[delta] ** 3) / R_vir ** 3.
        n_final.append(sum(interval) / vol)
    return np.array(n_final)


R_vir = input_data['host']['R_vir']

num_bins = 50
bins_kpc = np.linspace(1e-2, R_vir + 1., num=num_bins)
x_med_kpc = (bins_kpc[:-1] + bins_kpc[1:]) / 2.
bins = bins_kpc/R_vir

print('Density figure')
'''
fig = plt.figure(figsize=(12, 10))
ax1 = fig.gca()

# --- Resilient all of it ---
plt.plot(x_med_kpc, (encontrar_SRD(datos_resi_dmo[:, 1])
                     / len(datos_resi_dmo[:, 1])
                     ),
         color='k', marker='+', linestyle='-', ms=14)
plt.plot(x_med_kpc, (encontrar_SRD(datos_resi_hyd[:, 1])
                     / len(datos_resi_hyd[:, 1])
                     ),
         color='g', marker='+', linestyle='-', ms=14)

# --- Resilient over Vcompletion ---
data_used_dmo = datos_resi_dmo[
    datos_resi_dmo[:, 3] > input_data['SHVF']['Vmax_completion'], 1]
data_used_hyd = datos_resi_hyd[
    datos_resi_hyd[:, 3] > input_data['SHVF']['Vmax_completion'], 1]
srd_dmo_frag_sinVol = (encontrar_SRD(data_used_dmo)
                       / len(data_used_dmo)
                       )
srd_hydro_frag_sinVol = (encontrar_SRD(data_used_hyd)
                         / len(data_used_hyd)
                         )

plt.plot(x_med_kpc, srd_dmo_frag_sinVol,
         color='k', linestyle='--')
plt.plot(x_med_kpc, srd_hydro_frag_sinVol,
         color='g', linestyle='--')


# --- Resilient under Vcompletion ---
data_used_dmo = datos_resi_dmo[
    datos_resi_dmo[:, 3] < input_data['SHVF']['Vmax_completion'], 1]
data_used_hyd = datos_resi_hyd[
    datos_resi_hyd[:, 3] < input_data['SHVF']['Vmax_completion'], 1]

srd_dmo_frag_sinVol = (encontrar_SRD(data_used_dmo)
                       / len(data_used_dmo)
                       )
srd_hydro_frag_sinVol = (encontrar_SRD(data_used_hyd)
                         / len(data_used_hyd)
                         )

plt.plot(x_med_kpc, srd_dmo_frag_sinVol,
         color='k', linestyle='dotted')
plt.plot(x_med_kpc, srd_hydro_frag_sinVol,
         color='g', linestyle='dotted')

# --- Fragile all of it ---
plt.plot(x_med_kpc, (encontrar_SRD(datos_frag_dmo[:, 1])
                     / len(datos_frag_dmo[:, 1])
                     ),
         color='k', marker='.', linestyle='-', ms=10, label='DMO')
plt.plot(x_med_kpc, (encontrar_SRD(datos_frag_hyd[:, 1])
                     / len(datos_frag_hyd[:, 1])
                     ),
         color='g', marker='.', linestyle='-', ms=10, label='Hydro')


# --- Figure information ---
plt.axvline(R_vir, alpha=0.7, linestyle='--')  # , label='220 kpc')
plt.annotate(r'R$_\mathrm{vir}$', (170, 32), color='b',
             rotation=45, alpha=0.7)

plt.axvline(8.5, linestyle='--', alpha=1, color='Sandybrown')
plt.annotate('Earth', (8.6, 35), color='Sandybrown', rotation=45)

plt.ylabel(r'n(r) = $\frac{N(r)}{N_{Tot}\,Volumen}$ [Mpc$^{-3}$]',
           size=24)
plt.xlabel('r [kpc]', size=26)

linestyles = ['dotted', '--', '-']
markers = [None, None, '+']
legend22 = plt.legend([plt.Line2D([], [],
                                  linestyle=linestyles[i],
                                  color='k',
                                  marker=markers[i], ms=14)
                       for i in range(3)],
                      ['Under completion', 'Over completion', 'Total'],
                      loc=1, title='Resilient', framealpha=1)

colors = ['k', 'g']
legend11 = plt.legend([plt.Line2D([], [],
                                  linestyle='-',
                                  color=colors[i])
                       for i in range(2)],
                      ['DMO', 'Hydro'],
                      loc=5, title='Colors', framealpha=1)

legend33 = plt.legend([plt.Line2D([], [],
                                  linestyle='-',
                                  color='k',
                                  marker='o', ms=10)
                       for i in range(1)],
                      ['Total'],
                      loc=9, title='Fragile', framealpha=1)

ax1.add_artist(legend22)
ax1.add_artist(legend11)
ax1.add_artist(legend33)

plt.xscale('log')
plt.yscale('log')

plt.savefig(path_outputs + '/srd_density.png', bbox_inches='tight')
plt.savefig(path_outputs + '/srd_density.pdf', bbox_inches='tight')
'''
# ------------ N(r)/Ntot figure -------------------------------
print()
print('N/Ntot figures')

fig = plt.figure(figsize=(10, 8))
ax1 = fig.gca()

# --- Resilient all of it ---
plt.plot(x_med_kpc/R_vir, (encontrar_SRD_sinVol(datos_resi_dmo[:, 1])
                     # / len(datos_resi_dmo[:, 1])
                     ),
         color='k', marker='+', linestyle='-', ms=10)
plt.plot(x_med_kpc/R_vir, (encontrar_SRD_sinVol(datos_resi_hyd[:, 1])
                     # / len(datos_resi_hyd[:, 1])
                     ),
         color='g', marker='+', linestyle='-', ms=10)

# --- Resilient over Vcompletion ---
data_used_dmo = datos_resi_dmo[
    datos_resi_dmo[:, 3] > input_data['SHVF']['Vmax_completion'], 1]
data_used_hyd = datos_resi_hyd[
    datos_resi_hyd[:, 3] > input_data['SHVF']['Vmax_completion'], 1]
srd_dmo_frag_sinVol = (encontrar_SRD_sinVol(data_used_dmo)
                       # / len(data_used_dmo)
                       )
srd_hydro_frag_sinVol = (encontrar_SRD_sinVol(data_used_hyd)
                         # / len(data_used_hyd)
                         )

plt.plot(x_med_kpc/R_vir, srd_dmo_frag_sinVol,
         color='k', marker='^', linestyle='--', ms=10)
plt.plot(x_med_kpc/R_vir, srd_hydro_frag_sinVol,
         color='g', marker='^', linestyle='--', ms=10)
def N_subs_fragile(DistGC, args0, args1):
    return args1 * np.exp(args0 / DistGC)


arr_take = np.where(srd_dmo_frag_sinVol != 0.)
aaa = curve_fit(f=N_subs_fragile,
                xdata=x_med_kpc[arr_take]/R_vir,
                ydata=srd_dmo_frag_sinVol[arr_take],
                      # / len(datos_frag_dmo[:, 1]),
                p0=[-0.15, 1000])


arr_take = np.where(srd_hydro_frag_sinVol != 0.)
bbb = curve_fit(f=N_subs_fragile,
                xdata=x_med_kpc[arr_take]/R_vir,
                ydata=srd_hydro_frag_sinVol[arr_take],
                      # / len(datos_frag_hyd[:, 1]),
                p0=[-0.25, 1000])
print('fit resilient ove Vcompletion')
print(aaa)
print(bbb)
print()

xx_plot = np.linspace(1e-6, 1., num=100)
plt.plot(xx_plot, N_subs_fragile(xx_plot, aaa[0][0], aaa[0][1]),
         c='k')
plt.plot(xx_plot, N_subs_fragile(xx_plot, bbb[0][0], bbb[0][1]),
         color='g')

# --- Resilient under Vcompletion ---
data_used_dmo = datos_resi_dmo[
    datos_resi_dmo[:, 3] < input_data['SHVF']['Vmax_completion'], 1]
data_used_hyd = datos_resi_hyd[
    datos_resi_hyd[:, 3] < input_data['SHVF']['Vmax_completion'], 1]

srd_dmo_frag_sinVol = (encontrar_SRD_sinVol(data_used_dmo)
                       # / len(data_used_dmo)
                       )
srd_hydro_frag_sinVol = (encontrar_SRD_sinVol(data_used_hyd)
                         # / len(data_used_hyd)
                         )
plt.plot(x_med_kpc/R_vir, srd_dmo_frag_sinVol,
         color='k', marker='x', linestyle='-', ms=10, label='DMO')
plt.plot(x_med_kpc/R_vir, srd_hydro_frag_sinVol,
         color='g', marker='x', linestyle='-', ms=10, label='Hydro')


# --- Fragile all of it ---
plt.plot(x_med_kpc/R_vir, (encontrar_SRD_sinVol(datos_frag_dmo[:, 1])
                     # / len(datos_frag_dmo[:, 1])
                     ),
         color='k', marker='.', linestyle='-', ms=10, label='DMO')
plt.plot(x_med_kpc/R_vir, (encontrar_SRD_sinVol(datos_frag_hyd[:, 1])
                     # / len(datos_frag_hyd[:, 1])
                     ),
         color='g', marker='.', linestyle='-', ms=10, label='Hydro')


def N_subs_fragile(DistGC, args0, args1):
    return args1 * np.exp(args0 / DistGC)


all = encontrar_SRD_sinVol(datos_frag_dmo[:, 1])
arr_take = np.where(all != 0.)
aaa = curve_fit(f=N_subs_fragile,
                xdata=x_med_kpc[arr_take]/R_vir,
                ydata=all[arr_take],
                      # / len(datos_frag_dmo[:, 1]),
                p0=[-0.15, 1000])

all = encontrar_SRD_sinVol(datos_frag_hyd[:, 1])
arr_take = np.where(all != 0.)
bbb = curve_fit(f=N_subs_fragile,
                xdata=x_med_kpc[arr_take]/R_vir,
                ydata=all[arr_take],
                      # / len(datos_frag_hyd[:, 1]),
                p0=[-0.25, 1000])
print('Fit fragile complete')
print(aaa)
print(bbb)
print()

plt.plot(x_med_kpc/R_vir, encontrar_SRD_sinVol(datos_frag_dmo[:, 1]),
         c='r', ls='', marker='>')
plt.plot(x_med_kpc/R_vir, encontrar_SRD_sinVol(datos_frag_hyd[:, 1]),
         color='orange', ls='', marker='>')

xx_plot = np.linspace(1e-6, 1., num=100)
plt.plot(xx_plot, N_subs_fragile(xx_plot, aaa[0][0], aaa[0][1]),
         c='r')
plt.plot(xx_plot, N_subs_fragile(xx_plot, bbb[0][0], bbb[0][1]),
         color='orange')

# plt.show()
# Original Auriga simulations --------------------------------------
def encontrar_SRD_sinVol_auriga(data, bins):
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


release_dmo_over = np.array(Grand_dmo[Grand_dmo[:, 1]
                                      >= input_data['SHVF']['Vmax_completion'],
                            :])
release_hydro_over = np.array(Grand_hydro[Grand_hydro[:, 1]
                                          >= input_data['SHVF'][
                                              'Vmax_completion'], :])

minnDgc = np.argmin(Grand_dmo[:, 2])

bins = np.linspace(
    Grand_dmo[minnDgc, 2] / Grand_dmo[minnDgc, 5],
    1., num=15)
bins_mean_dmo = (bins[:-1] + bins[1:]) / 2.
# volume_dmo = 4 / 3 * np.pi * (bins_dmo[1:] ** 3 - bins_dmo[:-1] ** 3)
srd_dmo_over_release, std_dmo_aur = (np.array(encontrar_SRD_sinVol_auriga(
    release_dmo_over, bins=bins))
                                     # / float(len(release_dmo_over))
                                     )

minnDgc = np.argmin(Grand_hydro[:, 2])
bins = np.linspace(
    Grand_hydro[minnDgc, 2] / R_vir,
    1., num=15)
bins_mean_hydro = (bins[:-1] + bins[1:])/ 2.
# volume_hydro = 4 / 3 * np.pi * (bins_hydro[1:] ** 3 - bins_hydro[:-1] ** 3)
srd_hydro_over_release, std_hyd_aur = (np.array(encontrar_SRD_sinVol_auriga(
    release_hydro_over, bins=bins))
                                       # / float(len(release_hydro_over))
                                       )

plt.errorbar(bins_mean_dmo, srd_dmo_over_release,
             yerr=std_dmo_aur,
             ls='',
             c='k',
             ms=15, marker='*',
             alpha=1, zorder=15,
             label='Data',
             # label=ii
             )

plt.errorbar(bins_mean_hydro, srd_hydro_over_release,
             yerr=std_hyd_aur,
             ls='',
             c='#00CC00',
             ms=15, marker='*',
             alpha=1, zorder=15)

xx_plot = np.linspace(1e-6, 1., num=100)
# plt.plot(
#     xx_plot,
#     funct_repop.N_subs_fragile(
#         xx_plot, args=input_data['SRD']['dmo']['fragile']['args']),
#     c='k', ls='dotted')
# plt.plot(
#     xx_plot,
#     funct_repop.N_subs_fragile(
#         xx_plot,
#         args=[-0.29716816, 16.55605584]),
#         # args=input_data['SRD']['hydro']['fragile']['args']),
#     c='green', ls='dotted')
# print(input_data['SRD']['dmo']['fragile']['args'], input_data['SRD']['hydro'][
#     'fragile']['args'])
print('Release')


def N_subs_fragile(DistGC, args0, args1):
    return args1 * np.exp(args0 / DistGC)


cts_dmo = curve_fit(N_subs_fragile, xdata=bins_mean_dmo,
                    ydata=srd_dmo_over_release,
                    sigma=std_dmo_aur,
                    p0=[-0.151, 30])
print('Funct Ale: ', cts_dmo[0])
print(np.diag(cts_dmo[1]) ** 0.5)

cts_hydro = curve_fit(N_subs_fragile, xdata=bins_mean_hydro,
                      ydata=srd_hydro_over_release,
                      sigma=std_hyd_aur,
                      p0=[-0.3, 15])
print('Funct Ale: ', cts_hydro[0])
print(np.diag(cts_hydro[1]) ** 0.5)

plt.plot(xx_plot, N_subs_fragile(xx_plot, cts_dmo[0][0], cts_dmo[0][1]),
         'dimgray', linestyle='--', lw=3, alpha=0.7,
         label='Fragile fit', zorder=5)
plt.plot(xx_plot, N_subs_fragile(xx_plot, cts_hydro[0][0], cts_hydro[0][1]),
         '#00FF00', linestyle='--', lw=3, zorder=5)

# --- Figure information ---
# plt.axvline(1., alpha=0.7, linestyle='--')  # , label='220 kpc')
# plt.annotate(r'R$_\mathrm{vir}$', (203, 2e-2), color='b', rotation=45,
#              alpha=0.7)

plt.axvline(8.5/R_vir, linestyle='--', alpha=1, color='Sandybrown')
plt.annotate('Earth', (8.6, 0.07), color='Sandybrown', rotation=45)

plt.ylabel(r'n(r) = $\frac{N(r)}{N_{Tot}}$')
plt.xlabel('r [kpc]', size=24)

plt.xlim(0, 1)

linestyles = ['dotted', '--', '-']
markers = [None, None, '+']
legend22 = plt.legend([plt.Line2D([], [],
                                  linestyle=linestyles[i],
                                  color='k',
                                  marker=markers[i], ms=14)
                       for i in range(3)],
                      ['Under completion', 'Over completion', 'Total'],
                      loc=4, title='Resilient', framealpha=1)

colors = ['k', 'g']
legend11 = plt.legend([plt.Line2D([], [],
                                  linestyle='-',
                                  color=colors[i])
                       for i in range(2)],
                      ['DMO', 'Hydro'],
                      loc=5, title='Colors', framealpha=1)

legend33 = plt.legend([plt.Line2D([], [],
                                  linestyle='-',
                                  color='k',
                                  marker='o', ms=10)
                       for i in range(1)],
                      ['Total'],
                      loc=8, title='Fragile', framealpha=1)

ax1.add_artist(legend22)
ax1.add_artist(legend11)
ax1.add_artist(legend33)

# plt.xscale('log')
plt.yscale('log')

plt.savefig(path_outputs + '/srd_number.png', bbox_inches='tight')
plt.savefig(path_outputs + '/srd_number.pdf', bbox_inches='tight')
# plt.show()
# Cv -------------------------------------------------------------------

plt.figure(figsize=(12, 10))


def Cv_Mol2021_redshift0(V, c0=1.75e5, c1=-0.90368,
                         c2=0.2749, c3=-0.028):
    # Median subhalo concentration depending on its Vmax
    # and its redshift (here z=0)
    # Moline et al. 2110.02097
    #
    # V - max radial velocity of a bound particle in the subhalo [km/s]
    ci = [c0, c1, c2, c3]
    return ci[0] * (1 + (sum([ci[i + 1] * np.log10(V) ** (i + 1)
                              for i in range(3)])))


plt.plot(datos_resi_dmo[:, 3], datos_resi_dmo[:, 5], '.', alpha=0.5)
plt.plot(datos_frag_dmo[:, 3], datos_frag_dmo[:, 5], '.', alpha=0.5)
plt.plot(datos_resi_hyd[:, 3], datos_resi_hyd[:, 5], '.', alpha=0.5)
plt.plot(datos_frag_hyd[:, 3], datos_frag_hyd[:, 5], '.', alpha=0.5)

plt.plot(Grand_dmo[:, 1],
         2. * (Grand_dmo[:, 1] / Grand_dmo[:, 0]
               / input_data['cosmo_constants']['H_0'] * 1e3) ** 2.,
         '+', alpha=0.5)
plt.plot(Grand_hydro[:, 1],
         2. * (Grand_hydro[:, 1] / Grand_hydro[:, 0]
               / input_data['cosmo_constants']['H_0'] * 1e3) ** 2.,
         '+', alpha=0.5)

x_vmax = np.geomspace(input_data['SHVF']['RangeMin'],
                      input_data['SHVF']['RangeMax'],
                      num=26)

# plt.plot(
#     x_vmax,
#     funct_repop.Cv_Grand2012(x_vmax,
#                              input_data['Cv']['dmo']['bb'],
#                              input_data['Cv']['dmo']['mm']))
# plt.plot(
#     x_vmax,
#     funct_repop.Cv_Grand2012(x_vmax,
#                              input_data['Cv']['hydro']['bb'],
#                              input_data['Cv']['hydro']['mm']))

plt.plot(
    x_vmax,
    Cv_Mol2021_redshift0(x_vmax))
plt.plot(
    x_vmax,
    funct_repop.Moline21_normalization(x_vmax,
                                       c0=input_data['Cv']['dmo']['bb']))
plt.plot(
    x_vmax,
    funct_repop.Moline21_normalization(x_vmax,
                                       c0=input_data['Cv']['hydro']['bb']))

plt.xlabel(r'$V_\mathrm{max}$ [km s$^{-1}$]', size=28)
plt.ylabel(r'c$_\mathrm{V}$', size=28)

plt.xscale('log')
plt.yscale('log')

plt.savefig(path_outputs + '/Cv.png', bbox_inches='tight')
plt.savefig(path_outputs + '/Cv.pdf', bbox_inches='tight')

# Cv sigmas  -------------------------------------------------------
plt.figure(figsize=(12, 10))
print('\nSigmas of concentrations')


def gaussian(xx, sigma, x0):
    return 1 / ((2. * np.pi) ** 0.5 * sigma) * np.exp(
        -0.5 * ((xx - x0) / sigma) ** 2.)


# DMO
fraction_dmo = (datos_resi_dmo[:, 5]
                / funct_repop.Moline21_normalization(
            datos_resi_dmo[:, 3], c0=input_data['Cv']['dmo']['bb']))

per50_dmo = np.nanpercentile(fraction_dmo, 50)
per16_dmo = np.nanpercentile(fraction_dmo, 16)
per84_dmo = np.nanpercentile(fraction_dmo, 84)
print(per16_dmo, per50_dmo, per84_dmo)

# Hydro
fraction_hydro = (datos_resi_hyd[:, 5]
                  / funct_repop.Moline21_normalization(
            datos_resi_hyd[:, 3], c0=input_data['Cv']['hydro']['bb']))

per50_hydro = np.nanpercentile(fraction_hydro, 50)
per16_hydro = np.nanpercentile(fraction_hydro, 16)
per84_hydro = np.nanpercentile(fraction_hydro, 84)
print(per16_hydro, per50_hydro, per84_hydro)

# Histograms
num = 40
array_bins = np.geomspace(min(min(fraction_dmo), min(fraction_hydro)),
                          max(min(fraction_dmo), max(fraction_hydro)),
                          num=num)
n_dmo, bins_dmo, _ = plt.hist(fraction_dmo,
                              alpha=0.5, color='k', density=True,
                              bins=array_bins)
n_hydro, bins_hydro, _ = plt.hist(fraction_hydro,
                                  alpha=0.5, color='green', density=True,
                                  bins=array_bins)
print(n_dmo)
for_fits_dmo = n_dmo

xx3_plot = np.log10(np.geomspace(5e-2, 6))


def lognormal_fit(xx, mean, sigma):
    return (1 / (sigma * xx * np.sqrt(2. * np.pi))
            * np.exp(-0.5 * ((np.log(xx) - mean) / sigma) ** 2.))


fit_lognormal_dmo = curve_fit(
    lognormal_fit,
    xdata=(bins_dmo[1:] + bins_dmo[:-1]) / 2.,
    ydata=n_dmo)
print(fit_lognormal_dmo)
print('Sigma for the lognormal distribution, dmo: ',
      np.log10(np.exp(fit_lognormal_dmo[0][1])))

plt.plot(xx3_plot,
         lognormal_fit(xx3_plot, fit_lognormal_dmo[0][0],
                       fit_lognormal_dmo[0][1]))

fit_lognormal_hydro = curve_fit(
    lognormal_fit,
    xdata=(bins_hydro[1:] + bins_hydro[:-1]) / 2.,
    ydata=n_hydro)
print(fit_lognormal_hydro)
print('Sigma for the lognormal distribution, hydro: ',
      np.log10(np.exp(fit_lognormal_hydro[0][1])))

plt.plot(xx3_plot,
         lognormal_fit(xx3_plot, fit_lognormal_hydro[0][0],
                       fit_lognormal_hydro[0][1]))


def gaussian_not(xx, sigma, x0, aa):
    return aa / ((2. * np.pi) ** 0.5 * sigma) * np.exp(
        -0.5 * ((xx - x0) / sigma) ** 2.)


fit_normal_dmo = curve_fit(
    gaussian_not,
    xdata=np.log10((bins_dmo[1:] + bins_dmo[:-1]) / 2.),
    ydata=n_dmo)
print(fit_normal_dmo)

plt.plot(10 ** xx3_plot,
         gaussian_not(xx3_plot,
                      fit_normal_dmo[0][0],
                      fit_normal_dmo[0][1],
                      fit_normal_dmo[0][2]))

fit_normal_hydro = curve_fit(
    gaussian_not,
    xdata=np.log10((bins_hydro[1:] + bins_hydro[:-1]) / 2.),
    ydata=n_hydro)
print(fit_normal_hydro)

plt.plot(10 ** xx3_plot,
         gaussian_not(xx3_plot,
                      fit_normal_hydro[0][0],
                      fit_normal_hydro[0][1],
                      fit_normal_hydro[0][2]))

plt.xlabel(r'$V_\mathrm{max}$ [km s$^{-1}$]', size=28)
plt.ylabel(r'c$_\mathrm{V}$', size=28)

plt.xscale('log')
plt.yscale('log')

plt.savefig(path_outputs + '/Cv_hist.png', bbox_inches='tight')
plt.savefig(path_outputs + '/Cv_hist.pdf', bbox_inches='tight')

plt.show()
