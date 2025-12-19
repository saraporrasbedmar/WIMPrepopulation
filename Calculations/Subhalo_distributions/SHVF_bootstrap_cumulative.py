import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colorbar as colorbarr
import matplotlib.patches as mpatches

from scipy.integrate import simpson
from scipy.optimize import curve_fit

from iminuit import Minuit
from iminuit.cost import LeastSquares

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
plt.rc('ytick.major', size=10, width=2, right=True, pad=10)
plt.rc('xtick.minor', size=7, width=1.5, top=False)
plt.rc('ytick.minor', size=7, width=1.5)

data_release_dmo = np.loadtxt(
    '../Data_subhalo_simulations/dmo_table.txt', skiprows=3)
data_release_hydro = np.loadtxt(
    '../Data_subhalo_simulations/hydro_table.txt', skiprows=3)

# data_release_dmo = np.loadtxt(
#     '../Data_subhalo_simulations/data_dmo_level4.txt')
# data_release_hydro = np.loadtxt(
#     '../Data_subhalo_simulations/data_hydro_level4.txt')

data_release_dmo = data_release_dmo[
                   data_release_dmo[:, 0] > 0.184, :]
data_release_hydro = data_release_hydro[
                     data_release_hydro[:, 0] > 0.184, :]

data_release_dmo = data_release_dmo[np.argsort(data_release_dmo[:, 1])]
data_release_hydro = data_release_hydro[np.argsort(data_release_hydro[:, 1])]
unique_halos = np.unique(data_release_hydro[:, 6])

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

        Vmax_cumul[radius] = sum(aa)
        num_cumul[radius] = sum(aa)

    return Vmax_cumul/6., num_cumul


Vmax_cumul_dmo_original, num_dmo = calcular_dNdV(data_release_dmo)
vmax_dmo = x_cumul[np.argwhere(num_dmo >= 10.)[-1][0] + 1]

Vmax_cumul_hydro_original, num_hydro = calcular_dNdV(data_release_hydro)
vmax_hydro = x_cumul[np.argwhere(num_hydro >= 10.)[-1][0] + 1]


def powerlaw(vv_array, V0, alpha):
    return 10**V0 * vv_array ** alpha

def linear_funct(vv_array, V0, alpha):
    return V0 + vv_array * alpha

# repop = True
repop = False

if repop:

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
        limit_inf_dmo = rng.random(1) * 2. + 6.
        # limit_inf_dmo = rng.random(1) * 2. + 8.
        limit_sup_dmo = -rng.random(1) * 20. + vmax_dmo

        true_values = ((x_mean > limit_inf_dmo) * (x_mean < limit_sup_dmo))
        true_values = (true_values * (Vmax_cumul_dmo_release > 0.))
        true_values = (true_values * num_dmo >= 10)


        combined_likelihood = LeastSquares(
            x_mean[true_values], Vmax_cumul_dmo_release[true_values],
            Vmax_cumul_dmo_release[true_values]*0.001,
            powerlaw)

        m_best_fit = Minuit(combined_likelihood, V0=5., alpha=-4.)
        m_best_fit.migrad()
        m_best_fit.hesse()



        mm_dmo.append(m_best_fit.values[1])
        bb_dmo.append(m_best_fit.values[0])


        limit_inf_hyd = rng.random(1) * 3. + 5.
        limit_sup_hyd = -rng.random(1) *10. + 30.

        true_values = ((x_mean > limit_inf_hyd) * (x_mean < limit_sup_hyd))
        true_values = (true_values * (Vmax_cumul_hydro_release > 0.))
        true_values = (true_values * (num_hydro >= 10))

        combined_likelihood = LeastSquares(
            x_mean[true_values], Vmax_cumul_hydro_release[true_values],
            Vmax_cumul_hydro_release[true_values] * 0.001,
            powerlaw)

        m_best_fit = Minuit(combined_likelihood, V0=5., alpha=-4.)
        m_best_fit.migrad()
        m_best_fit.hesse()

        mm_hyd.append(m_best_fit.values[1])
        bb_hyd.append(m_best_fit.values[0])

    np.savetxt('outputs/data_shvf_cumulative.txt',
               np.column_stack((mm_dmo, bb_dmo, mm_hyd, bb_hyd)),
               header='mm_dmo, bb_dmo, mm_hyd, bb_hyd')

data = np.loadtxt('outputs/data_shvf_cumulative.txt')
mm_dmo = data[:, 0]
bb_dmo = data[:, 1]
mm_hyd = data[:, 2]
bb_hyd = data[:, 3]

plt.subplots(1, 2, figsize=(20, 8))

print('min and max values')
print(np.nanmax(mm_dmo), np.nanmin(mm_dmo),
      np.nanmax(mm_hyd), np.nanmin(mm_hyd))

num_bins_bb = np.linspace(
    min(np.nanmin(bb_dmo), np.nanmin(bb_hyd)),
    max(np.nanmax(bb_dmo), np.nanmax(bb_hyd)),
    num=20)

num_bins_mm = np.linspace(
    min(np.nanmin(mm_dmo), np.nanmin(mm_hyd)),
    max(np.nanmax(mm_dmo), np.nanmax(mm_hyd)),
    num=20)

plt.subplot(121)
plt.suptitle(r'$log_{10}\left(\frac{dN(V_{\mathrm{max}})}'
             r'{dV_{\mathrm{max}}}\right)'
             r' = V_0 + m * V$')
plt.hist(mm_dmo, log=False,
         label=r'DMO', color='k', alpha=0.6,
         bins=num_bins_mm)
plt.hist(mm_hyd, log=False,
         label=r'MHD', color='limegreen', alpha=0.5,
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

print('means')
print(np.nanmean(mm_dmo), np.nanstd(mm_dmo))
print(np.nanmean(mm_hyd), np.nanstd(mm_hyd))


plt.subplot(122)
plt.hist(bb_dmo, log=False,
         label=r'DMO', color='k', alpha=0.6,
         bins=num_bins_bb)
plt.hist(bb_hyd, log=False,
         label=r'MHD', color='limegreen', alpha=0.5,
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

plt.savefig('outputs/SHVF_bootstrap_hist_cumulative.png',
            bbox_inches='tight')


# -- SHVF figure -----------------------------------------------
fig, ax = plt.subplots(figsize=(8, 7))

Vmax_cumul_dmo_release, num_dmo = calcular_dNdV(data_release_dmo)
Vmax_cumul_hydro_release, num_hydro = calcular_dNdV(data_release_hydro)


plt.plot(x_mean, Vmax_cumul_dmo_release,
         linestyle='', ms=10, marker='.', markeredgewidth=2,
         color='k', zorder=10, label='Auriga data')
plt.axvline(7.4,
            color='k',
            alpha=1,
            linewidth=2.5, ls=':',
            zorder=0, label='Completion velocity')
xxx = np.geomspace(1., 120, num=100)
plt.plot(xxx, 10 ** np.nanmean(bb_dmo) * xxx ** np.nanmean(mm_dmo),
                 color='dimgray', alpha=1,
                 linestyle='--', lw=2.5)
xxx = np.geomspace(7.4, 120, num=100)
plt.plot(xxx, 10 ** np.nanmean(bb_dmo) * xxx ** np.nanmean(mm_dmo),
                 color='dimgray', alpha=1,
                 linestyle='-', lw=2.5, label='Power-law fit')

plt.plot(x_mean, Vmax_cumul_hydro_release,
         linestyle='',
         ms=10, marker='.', markeredgewidth=2,
         color='#00CC00', zorder=10)
plt.axvline(5.,
            color='limegreen',
            alpha=1,
            linewidth=2.5, ls=':',
            zorder=0)
xxx = np.geomspace(1., 120, num=100)
plt.plot(xxx, 10 ** np.nanmean(bb_hyd) * xxx ** np.nanmean(mm_hyd),
                 color='#00FF00', alpha=1,
                 linestyle='--', lw=2.5)
xxx = np.geomspace(5., 120, num=100)
plt.plot(xxx, 10 ** np.nanmean(bb_hyd) * xxx ** np.nanmean(mm_hyd),
                 color='#00FF00', alpha=1,
                 linestyle='-', lw=2.5)

plt.plot(xxx, 0.038*(xxx/201.)**-2.97, c='b', label='VLII paper Ale')

print(np.log10(0.038*201.**2.97))
print(np.log10((0.038 + 0.006)*201.**2.97)-np.log10(0.038*201.**2.97))


data_grand21_shvf = np.loadtxt(
    '../Data_subhalo_simulations/grand21_level3_shvf.txt')
plt.scatter(data_grand21_shvf[:, 0], data_grand21_shvf[:, 1],
         )

plt.xscale('log')
plt.yscale('log')

plt.xlabel(r'$V_{\mathrm{max}}$ [km s$^{-1}$]', size=24)
plt.ylabel(r'$\frac{dN(V_{\mathrm{max}})}{dV_{\mathrm{max}}}$', size=27)


# plt.axvline(54, linestyle='-.', color='r', alpha=0.5,
#             linewidth=2)
# plt.annotate(r'$V_\mathrm{'
#              r'cut}$', (60, 55), color='r',
#              rotation=0., alpha=0.8,
#              fontsize=20, zorder=10)

handles = (mpatches.Patch(color='k', label='DMO', alpha=0.8),
           mpatches.Patch(color='limegreen', label='MHD', alpha=0.8)
           )

legend11 = plt.legend(handles=handles, handlelength=0.9,
                      loc=1, framealpha=1)

legend22 = plt.legend(loc=3, framealpha=1)

ax.add_artist(legend11)
ax.add_artist(legend22)

plt.ylim(0.005, 1e4)

ax.set_xticks([1., 10, 100],
              labels=('1', '10', '100')
              )

plt.savefig('outputs/shvf_cumulative.pdf', bbox_inches='tight')
plt.savefig('outputs/shvf_cumulative.png', bbox_inches='tight')

plt.show()
