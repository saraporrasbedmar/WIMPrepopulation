import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
try:
    Grand_dmo = np.loadtxt(
        '../Data_subhalos_simulations/RmaxVmaxRadDMO0_1.txt')
    Grand_hydro = np.loadtxt(
        '../Data_subhalos_simulations/RmaxVmaxRadFP0_1.txt')

except:
    Grand_dmo = np.loadtxt('../../RmaxVmaxRadDMO0_1.txt')
    Grand_hydro = np.loadtxt('../../RmaxVmaxRadFP0_1.txt')

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

    funct_repop.main_repop(['dmo', 'resilient', path_input, path_outputs])
    funct_repop.main_repop(['dmo', 'fragile', path_input, path_outputs])

    input_data['repopulations']['num_brightest'] = \
        funct_repop.SHVF_Grand2012_int(
            input_data['SHVF']['RangeMin'],
            input_data['SHVF']['RangeMax'],
            input_data['SHVF']['hydro']['bb'],
            input_data['SHVF']['hydro']['mm'])

    with open(path_input, 'w') as f:
        yaml.dump(input_data, f)

    funct_repop.main_repop(['hydro', 'resilient', path_input, path_outputs])
    funct_repop.main_repop(['hydro', 'fragile', path_input, path_outputs])

input_data = read_config_file(path_outputs + '/input_data.yml')

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

plt.xscale('log')
plt.yscale('log')

plt.savefig(path_outputs + '/SHVF.png', bbox_inches='tight')
plt.savefig(path_outputs + '/SHVF.pdf', bbox_inches='tight')

# SRD ------------------------------------------------------------------

def encontrar_SRD_sinVol(data):
    data = np.sort(data)
    n_final = []
    a = 0
    for delta in range(len(bins) - 2):
        # print(delta, bins[delta])
        X1limit = np.where(data >= bins[delta])[0][0]
        X2limit = np.where(data > bins[delta + 1])[0][0]

        y = X2limit - X1limit

        n_final.append(y)
        a += y

    y = len(data) - X2limit
    n_final.append(y)

    a += y
    # print(a)

    return np.array(n_final)


def encontrar_SRD(data):
    data = np.sort(data)
    n_final = []
    a = 0
    for delta in range(len(bins) - 2):
        # print(delta, bins[delta])
        X1limit = np.where(data >= bins[delta])[0][0]
        X2limit = np.where(data > bins[delta + 1])[0][0]

        y = X2limit - X1limit
        vol = 4 / 3 * np.pi * (
                bins[delta + 1] ** 3 - bins[delta] ** 3) / 1e9

        n_final.append(y / vol)
        a += y

    y = len(data) - X2limit
    n_final.append(
        y / (4 / 3 * np.pi * (bins[-1] ** 3 - bins[-2] ** 3) / 1e9))

    a += y
    # print(a)

    return np.array(n_final)


R200 = input_data['host']['R_vir'] + 1.  # input_data['host']['R_vir']
R_max = input_data['host']['R_vir']

num_bins = 25
bins = np.linspace(1e-1, R200, num=num_bins)
x_med_kpc = (bins[:-1] + bins[1:]) / 2.

print('Density figure')

fig = plt.figure(figsize=(12, 10))
ax1 = fig.gca()

# --- Resilient all of it ---
plt.plot(x_med_kpc, (encontrar_SRD(datos_resi_dmo[:, 1])
                     / len(datos_resi_dmo[:, 1])),
         color='k', marker='+', linestyle='-', ms=14)
plt.plot(x_med_kpc, (encontrar_SRD(datos_resi_hyd[:, 1])
                     / len(datos_resi_hyd[:, 1])),
         color='g', marker='+', linestyle='-', ms=14)

# --- Resilient over Vcompletion ---
# data_used_dmo = datos_resi_dmo[
#     datos_resi_dmo[:, 3] > input_data['SHVF']['Vmax_completion'], 1]
# data_used_hyd = datos_resi_hyd[
#     datos_resi_hyd[:, 3] > input_data['SHVF']['Vmax_completion'], 1]
# srd_dmo_frag_sinVol = (encontrar_SRD(data_used_dmo)
#                        / len(data_used_dmo))
# srd_hydro_frag_sinVol = (encontrar_SRD(data_used_hyd)
#                          / len(data_used_hyd))
#
# plt.plot(x_med_kpc, srd_dmo_frag_sinVol,
#          color='k', linestyle='--')
# plt.plot(x_med_kpc, srd_hydro_frag_sinVol,
#          color='g', linestyle='--')


# --- Resilient under Vcompletion ---
data_used_dmo = datos_resi_dmo[
    datos_resi_dmo[:, 3] < input_data['SHVF']['Vmax_completion'], 1]
data_used_hyd = datos_resi_hyd[
    datos_resi_hyd[:, 3] < input_data['SHVF']['Vmax_completion'], 1]

srd_dmo_frag_sinVol = (encontrar_SRD(data_used_dmo)
                       / len(data_used_dmo))
srd_hydro_frag_sinVol = (encontrar_SRD(data_used_hyd)
                         / len(data_used_hyd))

plt.plot(x_med_kpc, srd_dmo_frag_sinVol,
         color='k', linestyle='dotted')
plt.plot(x_med_kpc, srd_hydro_frag_sinVol,
         color='g', linestyle='dotted')

# --- Fragile all of it ---
plt.plot(x_med_kpc, (encontrar_SRD(datos_frag_dmo[:, 1])
                     / len(datos_frag_dmo[:, 1])),
         color='k', marker='.', linestyle='-', ms=10, label='DMO')
plt.plot(x_med_kpc, (encontrar_SRD(datos_frag_hyd[:, 1])
                     / len(datos_frag_hyd[:, 1])),
         color='g', marker='.', linestyle='-', ms=10, label='Hydro')

# --- Figure information ---
plt.axvline(R_max, alpha=0.7, linestyle='--')  # , label='220 kpc')
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

# ------------ N(r)/Ntot figure -------------------------------
print()
print('N/Ntot figures')

fig = plt.figure(figsize=(10, 8))
ax1 = fig.gca()

# --- Resilient all of it ---
plt.plot(x_med_kpc, (encontrar_SRD_sinVol(datos_resi_dmo[:, 1])
                     / len(datos_resi_dmo[:, 1])),
         color='k', marker='+', linestyle='-', ms=10)
plt.plot(x_med_kpc, (encontrar_SRD_sinVol(datos_resi_hyd[:, 1])
                     / len(datos_resi_hyd[:, 1])),
         color='g', marker='+', linestyle='-', ms=10)

# --- Resilient over Vcompletion ---
# data_used_dmo = datos_resi_dmo[
#     datos_resi_dmo[:, 3] > input_data['SHVF']['Vmax_completion'], 1]
# data_used_hyd = datos_resi_hyd[
#     datos_resi_hyd[:, 3] > input_data['SHVF']['Vmax_completion'], 1]
# srd_dmo_frag_sinVol = (encontrar_SRD_sinVol(data_used_dmo)
#                        / len(data_used_dmo))
# srd_hydro_frag_sinVol = (encontrar_SRD_sinVol(data_used_hyd)
#                          / len(data_used_hyd))

# plt.plot(x_med_kpc, srd_dmo_frag_sinVol,
#          color='k', marker='x', linestyle='--', ms=10)
# plt.plot(x_med_kpc, srd_hydro_frag_sinVol,
#          color='g', marker='x', linestyle='--', ms=10)


# --- Resilient under Vcompletion ---
data_used_dmo = datos_resi_dmo[
    datos_resi_dmo[:, 3] < input_data['SHVF']['Vmax_completion'], 1]
data_used_hyd = datos_resi_hyd[
    datos_resi_hyd[:, 3] < input_data['SHVF']['Vmax_completion'], 1]

srd_dmo_frag_sinVol = (encontrar_SRD_sinVol(data_used_dmo)
                       / len(data_used_dmo))
srd_hydro_frag_sinVol = (encontrar_SRD_sinVol(data_used_hyd)
                         / len(data_used_hyd))

plt.plot(x_med_kpc, srd_dmo_frag_sinVol,
         color='k', marker='x', linestyle='dotted', ms=10)
plt.plot(x_med_kpc, srd_hydro_frag_sinVol,
         color='g', marker='x', linestyle='dotted', ms=10)

aaa = np.polyfit(np.log10(x_med_kpc[:-1]),
                 np.log10(srd_dmo_frag_sinVol[:-1]), 1)
bbb = np.polyfit(np.log10(x_med_kpc[:-1]),
                 np.log10(srd_hydro_frag_sinVol[:-1]), 1)
print(aaa)
print(bbb)

# plt.plot(x_med_kpc, 10**aaa[1]*x_med_kpc**aaa[0], 'r')
# plt.plot(x_med_kpc, 10**bbb[1]*x_med_kpc**bbb[0], color='orange')
# plt.plot(x_med_kpc, 10**-2.17672138*x_med_kpc**0.54343928,
#          'r', linestyle='--')
# plt.plot(x_med_kpc, 10**-3.08584275*x_med_kpc**0.97254648,
#          color='orange', linestyle='--')

# --- Fragile all of it ---
plt.plot(x_med_kpc, (encontrar_SRD_sinVol(datos_frag_dmo[:, 1])
                     / len(datos_frag_dmo[:, 1])),
         color='k', marker='.', linestyle='-', ms=10, label='DMO')
plt.plot(x_med_kpc, (encontrar_SRD_sinVol(datos_frag_hyd[:, 1])
                     / len(datos_frag_hyd[:, 1])),
         color='g', marker='.', linestyle='-', ms=10, label='Hydro')

# --- Figure information ---
plt.axvline(R_max, alpha=0.7, linestyle='--')  # , label='220 kpc')
plt.annotate(r'R$_\mathrm{vir}$', (203, 2e-2), color='b', rotation=45,
             alpha=0.7)

plt.axvline(8.5, linestyle='--', alpha=1, color='Sandybrown')
plt.annotate('Earth', (8.6, 0.07), color='Sandybrown', rotation=45)

plt.ylabel(r'n(r) = $\frac{N(r)}{N_{Tot}}$')
plt.xlabel('r [kpc]', size=24)

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

# Cv -------------------------------------------------------------------

plt.figure(figsize=(12, 10))

def Cv_Mol2021_redshift0(V, c0=1.75e5, c1=-0.90368, c2=0.2749, c3=-0.028):
    # Median subhalo concentration depending on its Vmax and its redshift (here z=0)
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

plt.show()
