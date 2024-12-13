import os

import scipy.optimize
import yaml
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

import attemp_at_functions22 as funct_repop

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
path_outputs = 'outputs/test1repop_resilient_manycuts'

rerun_sims = True
# rerun_sims = False

if rerun_sims:

    if not os.path.exists(path_outputs):
        os.makedirs(path_outputs)

    path_input = 'input_files/input_paper2024_SHVFnorm.yml'

    input_data = read_config_file(path_input)
    print('dmo num subs over completion: ',
          funct_repop.SHVF_Grand2012_int(input_data['SHVF'][
                                             'Vmax_completion'][1],
                                         input_data['SHVF']['RangeMax'],
                                         input_data['SHVF']['dmo']['bb'],
                                         input_data['SHVF']['dmo']['mm']))
    print('hydro num subs over completion: ',
          funct_repop.SHVF_Grand2012_int(input_data['SHVF'][
                                             'Vmax_completion'][1],
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

# path_outputs = 'outputs/test1repop_resilient_highNormSHVF'
datos_resi_dmo = np.loadtxt(path_outputs + '/Js_dmo_resilient_results.txt')
datos_resi_hyd = np.loadtxt(path_outputs + '/Js_hydro_resilient_results.txt')

datos_frag_hyd = np.loadtxt(path_outputs + '/Js_hydro_fragile_results.txt')
datos_frag_dmo = np.loadtxt(path_outputs + '/Js_dmo_fragile_results.txt')

input_data = read_config_file(path_outputs + '/input_data.yml')
num_its = input_data['repopulations']['its']
# path_outputs = 'outputs/test1repop_resilient_highNormSHVF'

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


def find_PowerLaw(xx, yy, lim_inf, lim_sup):
    X1limit = np.where(xx >= lim_inf)[0][0]
    X2limit = np.where(xx >= lim_sup)[0][0]

    xx_copy = np.log10(xx[X1limit:X2limit])
    yy_copy = np.log10(yy[X1limit:X2limit])

    xx_copy = xx_copy[np.isfinite(yy_copy)]
    yy_copy = yy_copy[np.isfinite(yy_copy)]

    fits, cov_matrix = np.polyfit(xx_copy, yy_copy, 1, cov=True, full=False)
    perr = np.sqrt(np.diag(cov_matrix))

    return fits[0], fits[1], perr[0], perr[1]


Vmax_cumul_dmo_res = calcular_dNdV(datos_resi_dmo[:, 3]) / num_its
Vmax_cumul_dmo_frag = calcular_dNdV(datos_frag_dmo[:, 3]) / num_its
Vmax_cumul_hydro_res = calcular_dNdV(datos_resi_hyd[:, 3]) / num_its
Vmax_cumul_hydro_frag = calcular_dNdV(datos_frag_hyd[:, 3]) / num_its

x_cumul = (x_cumul[:-1] + x_cumul[1:]) / 2.

plt.figure(figsize=(10, 10))

xx_plot = np.logspace(np.log10(2), np.log10(120), 100)

plt.scatter(x_cumul, Vmax_cumul_dmo_res, c='k', marker='+', s=12**2)
plt.scatter(x_cumul, Vmax_cumul_dmo_frag, c='k')
plt.scatter(x_cumul, Vmax_cumul_hydro_res, c='g', marker='+', s=12**2)
plt.scatter(x_cumul, Vmax_cumul_hydro_frag, c='g')

fitsM_DMO, fitsB_DMO, _, _ = find_PowerLaw(
    x_cumul, Vmax_cumul_dmo_res, lim_inf=1.5, lim_sup=10.)
print(fitsM_DMO, fitsB_DMO)
plt.plot(xx_plot, 10 ** fitsB_DMO * xx_plot ** fitsM_DMO,
         color='k', alpha=0.7, linestyle='-', lw=2, label='DMO resilient')

fitsM_hydro, fitsB_hydro, _, _ = find_PowerLaw(
    x_cumul, Vmax_cumul_hydro_res, lim_inf=1.5, lim_sup=10.)
print(fitsM_hydro, fitsB_hydro)
plt.plot(xx_plot, 10 ** fitsB_hydro * xx_plot ** fitsM_hydro,
         color='g', alpha=0.7, linestyle='-', lw=2, label='Hydro resilient')

# Fragile
fitsM_DMO, fitsB_DMO, _, _ = find_PowerLaw(
    x_cumul, Vmax_cumul_dmo_frag, lim_inf=1.5, lim_sup=10.)
print(fitsM_DMO, fitsB_DMO)
plt.plot(xx_plot, 10 ** fitsB_DMO * xx_plot ** fitsM_DMO,
         color='k', alpha=0.7, linestyle='--', lw=2, label='DMO fragile')

fitsM_hydro, fitsB_hydro, _, _ = find_PowerLaw(
    x_cumul, Vmax_cumul_hydro_frag, lim_inf=1.5, lim_sup=10.)
print(fitsM_hydro, fitsB_hydro)
plt.plot(xx_plot, 10 ** fitsB_hydro * xx_plot ** fitsM_hydro,
         color='g', alpha=0.7, linestyle='--', lw=2, label='Hydro fragile')

plt.xscale('log')
plt.yscale('log')

plt.legend()

plt.xlabel(r'$V_{\mathrm{max}}$ [km s$^{-1}$]', size=24)
plt.ylabel(r'$\frac{dN(V_{\mathrm{max}})}{dV_{\mathrm{max}}}$', size=27)

plt.savefig(path_outputs + '/SHVF.png', bbox_inches='tight')
plt.savefig(path_outputs + '/SHVF.pdf', bbox_inches='tight')


# plt.show()

# SRD ------------------------------------------------------------------

R_vir = input_data['host']['R_vir']

num_bins = 15
bins_kpc = np.linspace(0., R_vir + 1., num=num_bins)
x_med_kpc = (bins_kpc[:-1] + bins_kpc[1:]) / 2.
bins_repop = bins_kpc / R_vir

print('Density figure')

def encontrar_SRD_sinVol(data, Rvir=220, bins=bins_repop):
    n_final = []
    for delta in range(len(bins) - 1):
        interval = ((data / Rvir >= bins[delta])
                    * (data / Rvir < bins[delta + 1]))
        # interval = ((data[:, 2] / data[:, 5] >= bins[delta])
        #                 * (data[:, 2] / data[:, 5] <= bins[delta + 1]))
        n_final.append(sum(interval))
    return np.array(n_final)


def encontrar_SRD(data):
    n_final = []
    for delta in range(len(bins_repop) - 1):
        interval = (data >= bins_repop[delta]) * (data <= bins_repop[delta + 1])
        vol = 4 / 3 * np.pi * (
                bins_repop[delta + 1] ** 3 - bins_repop[delta] ** 3) / R_vir ** 3.
        n_final.append(sum(interval) / vol)
    return np.array(n_final)

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
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 10),
                       sharey=True,
                       sharex=True
                       )
plt.subplots_adjust(wspace=0.27)
plt.subplot(121)
plt.title('dmo')

# v_cut = [2., 10., 20., 30., 40., 50.]
v_cut = input_data['SHVF']['Vmax_completion']


num_bins = 15
bins_repop = np.linspace(0, 1., num=num_bins)
bins_mean = (bins_repop[:-1] + bins_repop[1:]) / 2.
volume = 4 / 3 * np.pi * (bins_repop[1:] ** 3 - bins_repop[:-1] ** 3)
print('bins: ', bins_repop * 220)

unique_halos = np.unique(Grand_dmo[:, 6])
print(unique_halos)

xxx = np.linspace(0., 1., num=200)
def encontrar_SRD_sinVol_grand(data, bins):
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


for ni, ii in enumerate(v_cut):
    print('v_cut: ', ii, ni)
    release_dmo_over = Grand_dmo[Grand_dmo[:, 1] >= ii, :]
    release_hydro_over = Grand_hydro[Grand_hydro[:, 1] >= ii, :]
    print(np.min(release_dmo_over[:, 2]),
          np.min(release_hydro_over[:, 2]))
    srd_dmo_over_release, std_dmo_num = (
        np.array(encontrar_SRD_sinVol_grand(release_dmo_over, bins_repop))
        # / len(release_dmo_over)
    )
    srd_dmo_over_release[srd_dmo_over_release==0] = 0.01
    srd_hydro_over_release, std_hydro_num = (
        np.array(encontrar_SRD_sinVol_grand(release_hydro_over, bins_repop))
        # / len(release_hydro_over)
    )
    srd_hydro_over_release[srd_hydro_over_release==0] = 0.01

    print('under stuff')

    srdnum_dmo_under, _ = np.array(encontrar_SRD_sinVol_grand(
        Grand_dmo[Grand_dmo[:, 1] < ii, :], bins_repop))
    srdnum_hydro_under, _ = np.array(encontrar_SRD_sinVol_grand(
        Grand_hydro[Grand_hydro[:, 1] < ii, :], bins_repop))

    ax0.plot(bins_mean, srd_dmo_over_release,
             c=cm.CMRmap(ni / float(len(v_cut))))

    ax1.plot(bins_mean, srd_hydro_over_release,
             c=cm.CMRmap(ni / float(len(v_cut))), label='%.1f' % ii)

    # --- Resilient over Vcompletion ---
    data_used_dmo = datos_resi_dmo[datos_resi_dmo[:, 3] > ii, 1]
    data_used_hyd = datos_resi_hyd[datos_resi_hyd[:, 3] > ii, 1]
    srd_dmo_resi_sinVol = (encontrar_SRD_sinVol(
        data_used_dmo, bins=bins_repop)
                           # / len(data_used_dmo)
                           )
    srd_dmo_resi_sinVol[srd_dmo_resi_sinVol == 0] = 0.01
    srd_hydro_resi_sinVol = (encontrar_SRD_sinVol(
        data_used_hyd, bins=bins_repop)
                             # / len(data_used_hyd)
                             )
    srd_hydro_resi_sinVol[srd_hydro_resi_sinVol == 0] = 0.01

    ax0.plot(x_med_kpc / R_vir, srd_dmo_resi_sinVol,
             color=cm.CMRmap(ni / float(len(v_cut))),  linestyle=':')
    ax1.plot(x_med_kpc / R_vir, srd_hydro_resi_sinVol,
             color=cm.CMRmap(ni / float(len(v_cut))),  linestyle=':')


plt.subplot(122)
for ni, i in enumerate(input_data['SRD']['hydro']['resilient']['last_subhalo']):
    plt.axvline(float(i)/220, c=cm.CMRmap((ni) / float(len(v_cut))),
                alpha=0.5, zorder=0)

plt.subplot(121)
for ni, i in enumerate(input_data['SRD']['dmo']['resilient']['last_subhalo']):
    plt.axvline(float(i)/220, c=cm.CMRmap((ni)  / float(len(v_cut))),
                alpha=0.5, zorder=0)


plt.ylabel(r'Over - $N(D_\mathrm{GC})$')

plt.xscale('linear')
plt.yscale('log')

plt.xlim(0, 1.)
plt.ylim(0.009, 630)

plt.subplot(122)
plt.ylim(0.009, 630)
plt.title('hydro')

legend11 = plt.legend(loc=2, framealpha=1,
                      bbox_to_anchor=(1.04, 1), fontsize=12)
handles = (Line2D([0], [0], color='k', ls='-', label='Over'),
           Line2D([0], [0], color='k', ls='--', label='Below')
           )
legend22 = plt.legend(
    handles=handles,
    loc=3, framealpha=1,
    bbox_to_anchor=(1.04, 0), fontsize=12)
ax1.add_artist(legend11)
ax1.add_artist(legend22)

plt.xlim(0., 1.)

plt.subplot(121)
plt.xlabel(r'D$_\mathrm{GC} \, / \, R_\mathrm{vir}$ ', size=26)
plt.subplot(122)
plt.xlabel(r'D$_\mathrm{GC} \, / \, R_\mathrm{vir}$ ', size=26)

# ------------ N(r)/Ntot figure -------------------------------
print()
print('N/Ntot figures')

vmax_completion = 20.

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


def N_subs_fragile(DistGC, args0, args1):
    return args1 * np.exp(args0 / DistGC)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
plt.subplot(121)
plt.title('fragile')

# --- Fragile all of it ---
plt.plot(x_med_kpc / R_vir,
         encontrar_SRD_sinVol(datos_frag_dmo[:, 1])/ num_its,
         color='navy', marker='.', ls='', ms=15, label='DMO')
plt.plot(x_med_kpc / R_vir,
         encontrar_SRD_sinVol(datos_frag_hyd[:, 1])/ num_its,
         color='peru', marker='.', linestyle='', ms=15, label='Hydro')

all = encontrar_SRD_sinVol(datos_frag_dmo[:, 1])/ num_its
arr_take = np.where(all != 0.)
aaa = curve_fit(f=N_subs_fragile,
                xdata=x_med_kpc[arr_take] / R_vir,
                ydata=all[arr_take],
                p0=[-0.15, 1000])

all = encontrar_SRD_sinVol(datos_frag_hyd[:, 1])/ num_its
arr_take = np.where(all != 0.)
bbb = curve_fit(f=N_subs_fragile,
                xdata=x_med_kpc[arr_take] / R_vir,
                ydata=all[arr_take],
                p0=[-0.25, 1000])
print('Fit fragile complete')
print(aaa)
print(bbb)
print()

xx_plot = np.linspace(1e-6, 1., num=100)
plt.plot(xx_plot, N_subs_fragile(xx_plot, aaa[0][0], aaa[0][1]),
         c='navy', ls='--', lw=4)
plt.plot(xx_plot, N_subs_fragile(xx_plot, bbb[0][0], bbb[0][1]),
         color='peru', ls='--', lw=4)

# --- Fragile over Vcompletion -------------------------------
all = encontrar_SRD_sinVol(
    datos_frag_dmo[datos_frag_dmo[:, 3] > vmax_completion, 1]) / num_its
plt.plot(x_med_kpc / R_vir, all,
         color='k', marker='P', linestyle='', ms=12, label='DMO')

arr_take = np.where(all != 0.)
aaa = curve_fit(f=N_subs_fragile,
                xdata=x_med_kpc[arr_take] / R_vir,
                ydata=all[arr_take],
                p0=[-0.15, 1000])

all = encontrar_SRD_sinVol(
    datos_frag_hyd[datos_frag_hyd[:, 3] > vmax_completion, 1]) / num_its
plt.plot(x_med_kpc / R_vir, all,
         color='g', marker='P', linestyle='', ms=12, label='Hydro')
arr_take = np.where(all != 0.)
bbb = curve_fit(f=N_subs_fragile,
                xdata=x_med_kpc[arr_take] / R_vir,
                ydata=all[arr_take],
                p0=[-0.25, 1000])
print('Fit fragile complete')
print(aaa)
print(bbb)
print()

xx_plot = np.linspace(1e-6, 1., num=100)
plt.plot(xx_plot, N_subs_fragile(xx_plot, aaa[0][0], aaa[0][1]),
         c='k', ls=':', lw=4)
plt.plot(xx_plot, N_subs_fragile(xx_plot, bbb[0][0], bbb[0][1]),
         color='g', ls=':', lw=4)

# Original Auriga simulations --------------------------------------


plt.axvline(float(input_data['SRD']['dmo']['fragile']['last_subhalo'])/R_vir,
            c='k', lw=3)
plt.axvline(float(input_data['SRD']['hydro']['fragile'][
                      'last_subhalo'])/R_vir,
            c='g', lw=3)

release_dmo_over = np.array(Grand_dmo[
                            Grand_dmo[:, 1] >= vmax_completion, :])
release_hydro_over = np.array(Grand_hydro[
                              Grand_hydro[:, 1] >= vmax_completion, :])

minnDgc = np.argmin(Grand_dmo[:, 2])

bins_repop = np.linspace(
    Grand_dmo[minnDgc, 2] / Grand_dmo[minnDgc, 5], 1., num=15)
bins_mean_dmo = (bins_repop[:-1] + bins_repop[1:]) / 2.

srd_dmo_over_release, std_dmo_aur = (np.array(encontrar_SRD_sinVol_auriga(
    release_dmo_over, bins=bins_repop))
    # / float(len(release_dmo_over))
)

minnDgc = np.argmin(Grand_hydro[:, 2])
bins_repop = np.linspace(
    Grand_hydro[minnDgc, 2] / R_vir,
    1., num=15)
bins_mean_hydro = (bins_repop[:-1] + bins_repop[1:]) / 2.
# volume_hydro = 4 / 3 * np.pi * (bins_hydro[1:] ** 3 - bins_hydro[:-1] ** 3)
srd_hydro_over_release, std_hyd_aur = (np.array(encontrar_SRD_sinVol_auriga(
    release_hydro_over, bins=bins_repop))
    # / float(len(release_hydro_over))
)

plt.errorbar(bins_mean_dmo, srd_dmo_over_release,
             yerr=std_dmo_aur,
             ls='',
             c='grey',
             ms=12, marker='*',
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

print('Release')

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
         'grey', linestyle='-', lw=3, alpha=0.7,
         label='Fragile fit', zorder=5)
plt.plot(xx_plot, N_subs_fragile(xx_plot, cts_hydro[0][0], cts_hydro[0][1]),
         'limegreen', linestyle='-', lw=3, zorder=5)

# --- Figure information ---

plt.axvline(8.5 / R_vir, linestyle='--', alpha=1, color='Sandybrown')
plt.annotate('Earth', (8.6, 0.07), color='Sandybrown', rotation=45)

plt.ylabel(r'N(r)')
plt.xlabel('r [kpc]', size=24)

plt.xlim(0, 1)

linestyles = ['-', '--', ':']
markers = ['*', '.', 'P']
legend22 = plt.legend([plt.Line2D([], [],
                                  linestyle=linestyles[i], lw=2,
                                  color='grey',
                                  markerfacecolor='k', markeredgecolor='k',
                                  marker=markers[i], ms=12)
                       for i in range(3)],
                      ['Data Auriga', 'Repop total',
                       r'Repop V$_{max}> 8$ km/s'],
                      loc=4, framealpha=1)

handles = (mpatches.Patch(color='k', label='DMO', alpha=0.8),
           mpatches.Patch(color='limegreen', label='Hydro', alpha=0.8)
           )

legend_colors = plt.legend(handles=handles, loc=2,
                           # bbox_to_anchor=(0.13, 0.2),
                           fontsize=20)

ax1.add_artist(legend22)
ax1.add_artist(legend_colors)

# plt.yscale('log')
plt.ylim(-5, 45)

# --------------------------------------------------------------------
plt.subplot(122)
plt.title('resilient')

plt.errorbar(bins_mean_dmo, srd_dmo_over_release,
             yerr=std_dmo_aur,
             ls='',
             c='grey',
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

plt.plot(xx_plot, np.ones(len(xx_plot))
         * N_subs_fragile(1., cts_dmo[0][0], cts_dmo[0][1]),
         'dimgray', linestyle='-', lw=3, alpha=0.7,
         label='Resilient fit', zorder=5)
plt.plot(xx_plot, np.ones(len(xx_plot))
         * N_subs_fragile(1., cts_hydro[0][0], cts_hydro[0][1]),
         'limegreen', linestyle='-', lw=3, zorder=5)

print('Resilient inputs: dmo %.2f and hydro %.2f'
      % (N_subs_fragile(1., cts_dmo[0][0], cts_dmo[0][1]),
         N_subs_fragile(1., cts_hydro[0][0], cts_hydro[0][1])))

# --- Resilient all of it ---
num_bins = 15
bins_kpc = np.linspace(0., R_vir + 1., num=num_bins)
x_med_kpc = (bins_kpc[:-1] + bins_kpc[1:]) / 2.
bins_repop = bins_kpc / R_vir

plt.plot(x_med_kpc / R_vir, 
         encontrar_SRD_sinVol(datos_resi_dmo[:, 1])/ num_its,
         color='navy', marker='.', ls='', ms=15, label='DMO')
plt.plot(x_med_kpc / R_vir, 
         encontrar_SRD_sinVol(datos_resi_hyd[:, 1])/ num_its,
         color='peru', marker='.', linestyle='', ms=15, label='Hydro')


# --- Resilient over Vcompletion -------------------------------

all = encontrar_SRD_sinVol(
    datos_resi_dmo[datos_resi_dmo[:, 3] > vmax_completion, 1])/ num_its
plt.plot(x_med_kpc / R_vir, all,
         color='k', marker='P', linestyle='', ms=12, label='DMO')

const_dmo = np.polyfit(x_med_kpc / R_vir, all, 0)[0]
plt.plot(xx_plot, np.ones(len(xx_plot))*const_dmo,
         c='dimgray', ls=':', lw=3, alpha=0.7)

all = encontrar_SRD_sinVol(
    datos_resi_hyd[datos_resi_hyd[:, 3] > vmax_completion, 1])/ num_its
plt.plot(x_med_kpc / R_vir, all,
         color='g', marker='P', linestyle='', ms=12, label='Hydro')

const_hydro = np.polyfit(x_med_kpc / R_vir, all, 0)[0]
plt.plot(xx_plot, np.ones(len(xx_plot))*const_hydro,
         c='limegreen', lw=3, ls=':')

print('Resilient fits:   dmo %.2f and hydro %.2f'
      % (const_dmo, const_hydro))

print('Ratio: dmo %.2f and hydro %.2f'
      % (N_subs_fragile(1., cts_dmo[0][0], cts_dmo[0][1])/const_dmo,
         N_subs_fragile(1., cts_hydro[0][0], cts_hydro[0][1])/const_hydro))


# --- Figure information ---
plt.axvline(8.5 / R_vir, linestyle='--', alpha=1, color='Sandybrown')
plt.annotate('Earth', (8.6, 0.07), color='Sandybrown', rotation=45)

plt.ylabel(r'N(r)')
plt.xlabel('r [kpc]', size=24)

plt.xlim(0, 1)
plt.ylim(-5, 45)

linestyles = ['-', '', ':']
markers = ['*', '.', 'P']
legend22 = plt.legend([plt.Line2D([], [],
                                  linestyle=linestyles[i], lw=2,
                                  color='grey',
                                  markerfacecolor='k', markeredgecolor='k',
                                  marker=markers[i], ms=12)
                       for i in range(3)],
                      ['Data Auriga', 'Repop total',
                       r'Repop V$_{max}> 8$ km/s'],
                      loc=4, framealpha=1)

handles = (mpatches.Patch(color='k', label='DMO', alpha=0.8),
           mpatches.Patch(color='limegreen', label='Hydro', alpha=0.8)
           )

legend_colors = plt.legend(handles=handles, loc=2,
                           # bbox_to_anchor=(0.13, 0.2),
                           fontsize=20)

ax2.add_artist(legend22)
ax2.add_artist(legend_colors)

# plt.yscale('log')
# plt.ylim(0.8, 1000)

plt.savefig(path_outputs + '/srd_number.png', bbox_inches='tight')
plt.savefig(path_outputs + '/srd_number.pdf', bbox_inches='tight')

print()
print('N/Ntot figures')

# -----------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
plt.subplots_adjust(wspace=0.27)
plt.subplot(121)
plt.title('dmo')


num_bins = 15
bins = np.linspace(0, 1., num=num_bins)
bins_mean = (bins[:-1] + bins[1:]) / 2.
volume = 4 / 3 * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
print('bins: ', bins * 220)

xxx = np.linspace(0., 1., num=200)

v_cut = np.copy(input_data['SHVF']['Vmax_completion'])
v_cut = np.append(v_cut, 120.)
print(input_data['SHVF']['Vmax_completion'], v_cut)
from matplotlib import cm

for ni, ii in enumerate(input_data['SHVF']['Vmax_completion']):
    print('v_cut: ', ii, ni)
    aaa = ((datos_resi_dmo[:, 3] >= ii)
           * (datos_resi_dmo[:, 3] < v_cut[ni+1]))
    release_dmo_over = datos_resi_dmo[aaa, 1]
    print(np.shape(release_dmo_over), release_dmo_over)

    srd_dmo_over_release = (
        np.array(encontrar_SRD_sinVol(release_dmo_over, bins=bins))
        / len(release_dmo_over)
    )
    srd_dmo_over_release[srd_dmo_over_release==0] = 0.01


    aaa = ((datos_resi_hyd[:, 3] >= ii)
           * (datos_resi_hyd[:, 3] < v_cut[ni+1]))
    release_hydro_over = datos_resi_hyd[aaa, 1]
    srd_hydro_over_release = (
        np.array(encontrar_SRD_sinVol(release_hydro_over, bins=bins))
        / len(release_hydro_over)
    )
    srd_hydro_over_release[srd_hydro_over_release==0] = 0.01

    ax1.plot(bins_mean, srd_dmo_over_release,
             c=cm.CMRmap(ni / float(len(v_cut))))

    ax2.plot(bins_mean, srd_hydro_over_release,
             c=cm.CMRmap(ni / float(len(v_cut))), label='%.1f' % ii)

plt.ylabel(r'$N(D_\mathrm{GC})$')
plt.xlabel(r'D$_\mathrm{GC} \, / \, R_\mathrm{vir}$ ', size=26)

plt.xscale('linear')
plt.yscale('log')

plt.xlim(0, 1.)
plt.ylim(0.009, 630)

plt.subplot(122)
plt.ylim(0.009, 630)
plt.title('hydro')
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
plt.xlim(0., 1.)

plt.show()
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
# plt.plot(datos_frag_dmo[:, 3], datos_frag_dmo[:, 5], '.', alpha=0.5)
plt.plot(datos_resi_hyd[:, 3], datos_resi_hyd[:, 5], '.', alpha=0.5)
# plt.plot(datos_frag_hyd[:, 3], datos_frag_hyd[:, 5], '.', alpha=0.5)

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
