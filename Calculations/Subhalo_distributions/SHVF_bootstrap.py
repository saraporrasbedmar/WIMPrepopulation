import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colorbar as colorbarr
import matplotlib.patches as mpatches

from scipy.integrate import simpson
from scipy.optimize import curve_fit

all_size = 24
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
plt.rc('ytick.major', size=10, width=2, right=True, pad=10)
plt.rc('xtick.minor', size=7, width=1.5, top=False)
plt.rc('ytick.minor', size=7, width=1.5)


data_release_dmo = np.loadtxt(
    '../Data_subhalo_simulations/dmo_table.txt', skiprows=3)
data_release_hydro = np.loadtxt(
    '../Data_subhalo_simulations/hydro_table.txt', skiprows=3)

data_release_dmo = data_release_dmo[
                   data_release_dmo[:, 0] > 0.184, :]
data_release_hydro = data_release_hydro[
                     data_release_hydro[:, 0] > 0.184, :]

data_release_dmo = data_release_dmo[np.argsort(data_release_dmo[:, 1])]
data_release_hydro = data_release_hydro[np.argsort(data_release_hydro[:, 1])]

x_cumul = np.geomspace(1., 120., num=25)
x_mean = (x_cumul[:-1] + x_cumul[1:]) / 2.

rng = np.random.default_rng()

print(sum(data_release_dmo[:, 1]<54), sum(data_release_dmo[:, 1]>54),
sum((data_release_dmo[:, 1]<54)*(data_release_dmo[:, 1]>7.4)))
print(sum(data_release_hydro[:, 1]<54),sum(data_release_hydro[:, 1]>54),
sum((data_release_hydro[:, 1]<54)*(data_release_hydro[:, 1]>5.)))

def calcular_dNdV(Vmax):
    Vmax_cumul = np.zeros(len(x_cumul) - 1)
    num_cumul = np.zeros(len(x_cumul) - 1)

    Vmax = Vmax[:, 1]

    for radius in range(len(Vmax_cumul)):
        aa = Vmax >= x_cumul[radius]
        bb = Vmax < x_cumul[radius + 1]

        Vmax_cumul[radius] = sum(aa * bb) / (
                x_cumul[radius + 1] - x_cumul[radius])
        num_cumul[radius] = sum(aa * bb)

    return Vmax_cumul/6., num_cumul

if 1 == 0:

    num_subs_dmo = np.shape(data_release_dmo)[0]
    num_subs_hydro = np.shape(data_release_hydro)[0]

    mm_dmo = []
    bb_dmo = []
    mm_hyd = []
    bb_hyd = []


    for i in range(1000):
        if i%25 == 0:
            print(i)
        positions = rng.integers(num_subs_dmo, size=num_subs_dmo)
        data_dmo = data_release_dmo[positions, :]
        Vmax_cumul_dmo_release, num_dmo = calcular_dNdV(data_dmo)

        positions = rng.integers(num_subs_hydro, size=num_subs_hydro)
        data_hydro = data_release_hydro[positions, :]
        Vmax_cumul_hydro_release, num_hydro = calcular_dNdV(data_hydro)


        # Fit to the power laws
        limit_inf = rng.random(1) * 2. + 7.4
        limit_sup = -rng.random(1) * 34 + 54.

        true_values = ((x_mean > limit_inf) * (x_mean < limit_sup))
        true_values = (true_values * (Vmax_cumul_dmo_release > 0.))
        fits, cov_matrix = np.polyfit(
            np.log10(x_mean[true_values]),
            np.log10(Vmax_cumul_dmo_release[true_values]),
            deg=1, cov=True, full=False)
        # print(fits, cov_matrix)
        mm_dmo.append(fits[0])
        bb_dmo.append(fits[1])

        limit_inf = rng.random(1) * 3. + 5.
        limit_sup = -rng.random(1) * 34 + 54.

        true_values = ((x_mean > limit_inf) * (x_mean < limit_sup))
        true_values = (true_values * (Vmax_cumul_hydro_release > 0.))
        fits, cov_matrix = np.polyfit(
            np.log10(x_mean[true_values]),
            np.log10(Vmax_cumul_hydro_release[true_values]),
            deg=1, cov=True, full=False)
        mm_hyd.append(fits[0])
        bb_hyd.append(fits[1])
        # print(fits, type(fits[0]))
        if np.isnan(fits[0]):
            print('aaaa')
            plt.figure()
            plt.plot(x_mean, Vmax_cumul_hydro_release,
                     marker='.')
            plt.axvline(limit_inf)
            plt.axvline(limit_sup)

    np.savetxt('outputs/data_shvf.txt',
               np.column_stack((mm_dmo, bb_dmo, mm_hyd, bb_hyd)),
               header='mm_dmo, bb_dmo, mm_hyd, bb_hyd')

data = np.loadtxt('outputs/data_shvf.txt')
mm_dmo = data[:, 0]
bb_dmo = data[:, 1]
mm_hyd = data[:, 2]
bb_hyd = data[:, 3]

plt.subplots(1, 2, figsize=(20, 8))
num_bins_bb = np.linspace(4.4, 7, num=30)
num_bins_mm = np.linspace(-5., -3.2, num=20)

plt.subplot(121)
plt.suptitle(r'$log_{10}\left(\frac{dN(V_{\mathrm{max}})}'
             r'{dV_{\mathrm{max}}}\right)'
             r' = V_0 + m * V$')
plt.hist(mm_dmo, log=False,
         label=r'DMO', color='k', alpha=0.6,
         bins=num_bins_mm)
plt.hist(mm_hyd, log=False,
         label=r'Hydro', color='limegreen', alpha=0.5,
         bins=num_bins_mm)

plt.axvline(np.nanmean(mm_dmo), c='k', lw=2)
plt.axvline(np.nanmean(mm_dmo) + np.nanstd(mm_dmo),
            c='k', ls='--', lw=1.5)
plt.axvline(np.nanmean(mm_dmo) - np.nanstd(mm_dmo),
            c='k', ls='--', lw=1.5)

plt.axvline(np.nanmean(mm_hyd), c='limegreen', lw=2)
plt.axvline(np.nanmean(mm_hyd) + np.nanstd(mm_hyd),
            c='limegreen', ls='--', lw=1.5)
plt.axvline(np.nanmean(mm_hyd) - np.nanstd(mm_hyd),
            c='limegreen', ls='--', lw=1.5)

plt.xlabel('m')

print(np.nanmean(mm_dmo), np.nanstd(mm_dmo))
print(np.nanmean(mm_hyd), np.nanstd(mm_hyd))


plt.subplot(122)
plt.hist(bb_dmo, log=False,
         label=r'DMO', color='k', alpha=0.6,
         bins=num_bins_bb)
plt.hist(bb_hyd, log=False,
         label=r'Hydro', color='limegreen', alpha=0.5,
         bins=num_bins_bb)

plt.axvline(np.nanmean(bb_dmo), c='k', lw=2)
plt.axvline(np.nanmean(bb_dmo) + np.nanstd(bb_dmo),
            c='k', ls='--', lw=1.5)
plt.axvline(np.nanmean(bb_dmo) - np.nanstd(bb_dmo),
            c='k', ls='--', lw=1.5)

plt.axvline(np.nanmean(bb_hyd), c='limegreen', lw=2)
plt.axvline(np.nanmean(bb_hyd) + np.nanstd(bb_hyd),
            c='limegreen', ls='--', lw=1.5)
plt.axvline(np.nanmean(bb_hyd) - np.nanstd(bb_hyd),
            c='limegreen', ls='--', lw=1.5)
print(np.nanmean(bb_dmo), np.nanstd(bb_dmo))
print(np.nanmean(bb_hyd), np.nanstd(bb_hyd))

plt.xlabel(r'$V_0$')

plt.savefig('outputs/SHVF_bootstrap_hist.png', bbox_inches='tight')


# -- SHVF figure -----------------------------------------------
fig, ax = plt.subplots(figsize=(9, 8))

Vmax_cumul_dmo_release, num_dmo = calcular_dNdV(data_release_dmo)
Vmax_cumul_hydro_release, num_hydro = calcular_dNdV(data_release_hydro)

plt.axvline(7.4,
            color='k',
            alpha=1,
            linewidth=2, ls='-.',
            zorder=0, label='Fit limits')

plt.axvline(5.,
            color='limegreen',
            alpha=1,
            linewidth=2, ls='-.',
            zorder=0)


xxx = np.logspace(np.log10(2), np.log10(92), 100)
plt.plot(x_mean, Vmax_cumul_dmo_release,
         linestyle='', ms=10, marker='.', markeredgewidth=2,
         color='k', zorder=10)
plt.plot(xxx, 10 ** np.nanmean(bb_dmo) * xxx ** np.nanmean(mm_dmo),
                 color='dimgray', alpha=1,
                 linestyle='-', lw=2.5, label='Power-law fit')

plt.plot(x_mean, Vmax_cumul_hydro_release,
         linestyle='',
         ms=10, marker='.', markeredgewidth=2,
         color='#00CC00', zorder=10)
plt.plot(xxx, 10 ** np.nanmean(bb_hyd) * xxx ** np.nanmean(mm_hyd),
                 color='#00FF00', alpha=1,
                 linestyle='-', lw=2.5)



plt.xscale('log')
plt.yscale('log')

plt.xlabel(r'$V_{\mathrm{max}}$ [km s$^{-1}$]', size=24)
plt.ylabel(r'$\frac{dN(V_{\mathrm{max}})}{dV_{\mathrm{max}}}$', size=27)


plt.axvline(54, linestyle='-.', color='r', alpha=0.5,
            linewidth=2)
plt.annotate(r'$V_\mathrm{'
             r'cut}$', (60, 55), color='r',
             rotation=0., alpha=0.8,
             fontsize=20, zorder=10)

handles = (mpatches.Patch(color='k', label='DMO', alpha=0.8),
           mpatches.Patch(color='limegreen', label='Hydro', alpha=0.8)
           )

legend11 = plt.legend(handles=handles,
                      loc=1, framealpha=1)

legend22 = plt.legend(loc=3, framealpha=1)

ax.add_artist(legend11)
ax.add_artist(legend22)

plt.savefig('outputs/shvf.pdf', bbox_inches='tight')
plt.savefig('outputs/shvf.png', bbox_inches='tight')

plt.show()
