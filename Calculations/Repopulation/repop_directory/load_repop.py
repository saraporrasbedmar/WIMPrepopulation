import os
import yaml
import h5py
import psutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import curve_fit

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
plt.rc('legend', fontsize=20)
plt.rc('figure', titlesize=all_size)
plt.rc('xtick', top=True, direction='in')
plt.rc('ytick', right=True, direction='in')
plt.rc('xtick.major', size=10, width=2, top=False, pad=10)
plt.rc('ytick.major', size=10, width=2, right=True, pad=5)
plt.rc('xtick.minor', size=7, width=1.5, top=False)
plt.rc('ytick.minor', size=7, width=1.5)

def memory_usage_psutil():
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(10 ** 6)
    print('Memory used: %.2f MB' % mem)
    return mem


def read_config_file(ConfigFile):
    with open(ConfigFile, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml


input_directory = '../outputs/test_2026/test_2026-03-20 11:47:30/'

data_dmo_frg = h5py.File(
    input_directory + 'fullrepop_dmo_fragile.h5', 'r')
data_dmo_res = h5py.File(
    input_directory + 'fullrepop_dmo_resilient.h5', 'r')
data_mhd_frg = h5py.File(
    input_directory + 'fullrepop_mhd_fragile.h5', 'r')
data_mhd_res = h5py.File(
    input_directory + 'fullrepop_mhd_resilient.h5', 'r')

input_data = read_config_file(
    input_directory + 'input_data.yml')


memory_usage_psutil()

# ------ SHVF ----------------------------------------------------------

x_cumul = np.geomspace(input_data['repopulations']['RangeMin'],
                       input_data['repopulations']['RangeMax'],
                       num=26)

x_cumul_mean = np.sqrt(x_cumul[:-1] * x_cumul[1:])


def calculate_dNdV(Vmax):
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

    fits, cov_matrix = np.polyfit(
        xx_copy, yy_copy, 1, cov=True, full=False)
    perr = np.sqrt(np.diag(cov_matrix))

    return fits[0], fits[1], perr[0], perr[1]



print('----- SHVF -----')
plt.figure(figsize=(10, 10))

xx_plot = np.logspace(
    np.log10(input_data['repopulations']['RangeMin']),
    np.log10(input_data['repopulations']['RangeMax']), 100)

dNdV_dict_dmo_res = []
dNdV_dict_dmo_frg = []
dNdV_dict_mhd_res = []
dNdV_dict_mhd_frg = []

for i in range(input_data['repopulations']['its']):

    aaa = calculate_dNdV(
        data_dmo_res['iteration_' + str(i)]['Vmax'])
    plt.scatter(x_cumul_mean, aaa,
                c='k', marker='+', s=12**2)
    fitsM_DMO, fitsB_DMO, _, _ = find_PowerLaw(
        x_cumul_mean, aaa,
        lim_inf=input_data['repopulations']['RangeMin'],
        lim_sup=10.)
    dNdV_dict_dmo_res.append([fitsM_DMO, fitsB_DMO])
    plt.plot(xx_plot, 10 ** fitsB_DMO * xx_plot ** fitsM_DMO,
             color='k', alpha=0.7, linestyle='-', lw=2)

    aaa = calculate_dNdV(
        data_dmo_frg['iteration_' + str(i)]['Vmax'])
    plt.scatter(x_cumul_mean, aaa,
                c='b', marker='+', s=12**2)
    fitsM_DMO, fitsB_DMO, _, _ = find_PowerLaw(
        x_cumul_mean, aaa,
        lim_inf=input_data['repopulations']['RangeMin'],
        lim_sup=10.)
    dNdV_dict_dmo_frg.append([fitsM_DMO, fitsB_DMO])
    plt.plot(xx_plot, 10 ** fitsB_DMO * xx_plot ** fitsM_DMO,
             color='b', alpha=0.7, linestyle='-', lw=2)

    aaa = calculate_dNdV(
        data_mhd_res['iteration_' + str(i)]['Vmax'])
    plt.scatter(x_cumul_mean, aaa,
                c='green', marker='+', s=12**2)
    fitsM_DMO, fitsB_DMO, _, _ = find_PowerLaw(
        x_cumul_mean, aaa,
        lim_inf=input_data['repopulations']['RangeMin'],
        lim_sup=10.)
    dNdV_dict_mhd_res.append([fitsM_DMO, fitsB_DMO])
    plt.plot(xx_plot, 10 ** fitsB_DMO * xx_plot ** fitsM_DMO,
             color='green', alpha=0.7, linestyle='-', lw=2)

    aaa = calculate_dNdV(
        data_mhd_frg['iteration_' + str(i)]['Vmax'])
    plt.scatter(x_cumul_mean, aaa,
                c='orange', marker='+', s=12**2)
    fitsM_DMO, fitsB_DMO, _, _ = find_PowerLaw(
        x_cumul_mean, aaa,
        lim_inf=input_data['repopulations']['RangeMin'],
        lim_sup=10.)
    dNdV_dict_mhd_frg.append([fitsM_DMO, fitsB_DMO])
    plt.plot(xx_plot, 10 ** fitsB_DMO * xx_plot ** fitsM_DMO,
             color='orange', alpha=0.7, linestyle='-', lw=2)

dNdV_dict_dmo_res = np.array(dNdV_dict_dmo_res)
print('dNdV_dict_dmo_res')
print('%.2f pm %.2f  --  %.2f pm %.2f' % (
    np.mean(dNdV_dict_dmo_res[:, 0]), np.std(dNdV_dict_dmo_res[:, 0]),
    np.mean(dNdV_dict_dmo_res[:, 1]), np.std(dNdV_dict_dmo_res[:, 1])))
print()

dNdV_dict_dmo_frg = np.array(dNdV_dict_dmo_frg)
print('dNdV_dict_dmo_frg')
print('%.2f pm %.2f  --  %.2f pm %.2f' % (
    np.mean(dNdV_dict_dmo_frg[:, 0]), np.std(dNdV_dict_dmo_frg[:, 0]),
    np.mean(dNdV_dict_dmo_frg[:, 1]), np.std(dNdV_dict_dmo_frg[:, 1])))
print()

dNdV_dict_mhd_res = np.array(dNdV_dict_mhd_res)
print('dNdV_dict_mhd_res')
print('%.2f pm %.2f  --  %.2f pm %.2f' % (
    np.mean(dNdV_dict_mhd_res[:, 0]), np.std(dNdV_dict_mhd_res[:, 0]),
    np.mean(dNdV_dict_mhd_res[:, 1]), np.std(dNdV_dict_mhd_res[:, 1])))
print()

dNdV_dict_mhd_frg = np.array(dNdV_dict_mhd_frg)
print('dNdV_dict_mhd_frg')
print('%.2f pm %.2f  --  %.2f pm %.2f' % (
    np.mean(dNdV_dict_mhd_frg[:, 0]), np.std(dNdV_dict_mhd_frg[:, 0]),
    np.mean(dNdV_dict_mhd_frg[:, 1]), np.std(dNdV_dict_mhd_frg[:, 1])))
print()

plt.xscale('log')
plt.yscale('log')


memory_usage_psutil()


fig, ax = plt.subplots()
xx_plot = np.geomspace(0.1, 120, 50)

for i in range(input_data['repopulations']['its']):
    hist_dmo, bins_dmo = np.histogram(
        data_dmo_res['iteration_' + str(i)]['Vmax'], bins=xx_plot)
    plt.stairs(hist_dmo, edges=bins_dmo, color='grey', alpha=0.5,
               fill=True, label='Repopulation')


plt.axvline(7.4, c='k', label=r'$V_\mathrm{cut}$', ls=':', lw=2)
plt.axvline(5., c='forestgreen', ls=':', lw=2)

plt.xscale('log')
plt.yscale('log')

plt.xlim(1., 120.)

handles = (mpatches.Patch(color='k', label='DMO', alpha=0.8),
           mpatches.Patch(color='limegreen', label='MHD', alpha=0.8)
           )

legend11 = plt.legend(handles=handles, handlelength=0.9,
                      loc=7, framealpha=1,
                      bbox_to_anchor=(0.999, 0.5))

legend22 = plt.legend(loc=1, framealpha=1)

ax.add_artist(legend11)
ax.add_artist(legend22)

plt.xticks([1., 10, 100],
              labels=('1', '10', '100')
              )

plt.xlabel(r'$V_{\mathrm{max}}$ [km s$^{-1}$]', size=24)
plt.ylabel(r'Number of subhalos', size=24)


# ------- SRD ----------------------------------------------------------
print('----- SRD -----')
fig, ax = plt.subplots(figsize=(7, 5))

R_vir = float(input_data['host']['R_vir'])
num_bins = 15
xx_plot = np.linspace(0, 1, num_bins)


def N_subs_fragile(DistGC, args0, args1):
    return args1 * np.exp(args0 / DistGC)


fits_srd = {}
for i in input_data['configurations']:
    fits_srd[i] = []

for i in range(input_data['repopulations']['its']):
    hist_dmo, bins_dmo = np.histogram(
        data_dmo_res['iteration_' + str(i)]['Distgc'][:]/R_vir,
        bins=xx_plot, density=False)
    plt.stairs(hist_dmo/sum(hist_dmo), edges=bins_dmo, color='grey',
                alpha=0.5,
                fill=False)
    fits, _ = np.polyfit(
        (bins_dmo[1:] + bins_dmo[:-1])/2.,
        hist_dmo/sum(hist_dmo),
        1, cov=True, full=False)
    fits_srd['dmo_resilient'].append(fits)


    hist_dmo, bins_dmo = np.histogram(
        data_mhd_res['iteration_' + str(i)]['Distgc'][:]/R_vir,
        bins=xx_plot, density=False)
    plt.stairs(hist_dmo/sum(hist_dmo), edges=bins_dmo, color='limegreen',
               alpha=0.5,
               fill=False,)
    fits, _ = np.polyfit(
        (bins_dmo[1:] + bins_dmo[:-1])/2.,
        hist_dmo/sum(hist_dmo), 1, cov=True, full=False)
    fits_srd['mhd_resilient'].append(fits)


    hist_dmo, bins_dmo = np.histogram(
        data_dmo_frg['iteration_' + str(i)]['Distgc'][:]/R_vir,
        bins=xx_plot, density=False)
    plt.stairs(hist_dmo/sum(hist_dmo), edges=bins_dmo, color='b',
               alpha=0.5,
               fill=False)
    fits, _ = curve_fit(
        N_subs_fragile,
        xdata=(bins_dmo[1:] + bins_dmo[:-1])/2.,
        ydata=hist_dmo/sum(hist_dmo),
        p0=[-0.15, 1]
    )
    fits_srd['dmo_fragile'].append(fits)


    hist_dmo, bins_dmo = np.histogram(
        data_mhd_frg['iteration_' + str(i)]['Distgc'][:]/R_vir,
        bins=xx_plot, density=False)
    plt.stairs(hist_dmo/sum(hist_dmo), edges=bins_dmo, color='orange',
               alpha=0.5,
               fill=False,)
    fits, _ = curve_fit(
        N_subs_fragile,
        xdata=(bins_dmo[1:] + bins_dmo[:-1])/2.,
        ydata=hist_dmo/sum(hist_dmo),
        p0=[-0.15, 1]
    )
    fits_srd['mhd_fragile'].append(fits)

for i in input_data['configurations']:
    fits_srd[i] = np.array(fits_srd[i])

print('dmo_res')
print('%.4f pm %.4f  --  %.2f pm %.2f' % (
    np.mean(fits_srd['dmo_resilient'][:, 0]),
    np.std(fits_srd['dmo_resilient'][:, 0]),
    np.mean(fits_srd['dmo_resilient'][:, 1]),
    np.std(fits_srd['dmo_resilient'][:, 1])))
print()

print('mhd_res')
print('%.4f pm %.4f  --  %.2f pm %.2f' % (
    np.mean(fits_srd['mhd_resilient'][:, 0]),
    np.std(fits_srd['mhd_resilient'][:, 0]),
    np.mean(fits_srd['mhd_resilient'][:, 1]),
    np.std(fits_srd['mhd_resilient'][:, 1])))
print()

print('dmo_frag')
print('%.2f pm %.2f  --  %.2f pm %.2f' % (
    np.mean(fits_srd['dmo_fragile'][:, 0]),
    np.std(fits_srd['dmo_fragile'][:, 0]),
    np.mean(fits_srd['dmo_fragile'][:, 1]),
    np.std(fits_srd['dmo_fragile'][:, 1])))
print()

print('mhd_frg')
print('%.2f pm %.2f  --  %.2f pm %.2f' % (
    np.mean(fits_srd['mhd_fragile'][:, 0]),
    np.std(fits_srd['mhd_fragile'][:, 0]),
    np.mean(fits_srd['mhd_fragile'][:, 1]),
    np.std(fits_srd['mhd_fragile'][:, 1])))
print()


# plt.xscale('log')
# plt.yscale('log')

plt.xlim(0., 1.)
plt.ylim(0., 0.12)

handles = (mpatches.Patch(color='k', label='DMO', alpha=0.8),
           mpatches.Patch(color='limegreen', label='MHD', alpha=0.8)
           )

legend11 = plt.legend(handles=handles, handlelength=0.9,
                      loc=2, framealpha=1,
                      # bbox_to_anchor=(0.001, 0.64)
                      )

legend22 = plt.legend(loc=4, framealpha=1)

ax.add_artist(legend11)
ax.add_artist(legend22)

# plt.xticks([1., 10, 100],
#               labels=('1', '10', '100')
#               )

plt.xlabel(r'D$_\mathrm{GC} \, / \, R_\mathrm{vir}$ ', size=24)
plt.ylabel(r'Number of subhalos', size=24)


plt.show()
