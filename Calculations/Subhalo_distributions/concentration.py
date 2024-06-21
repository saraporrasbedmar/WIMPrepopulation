import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import matplotlib.patches as mpatches

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

data_release_dmo = data_release_dmo[
    np.argsort(data_release_dmo[:, 1])]
data_release_hydro = data_release_hydro[
    np.argsort(data_release_hydro[:, 1])]


# %%  CALCULATE THE RMAX-VMAX RELATION

def Cv_Mol2021_redshift0(V, c0=1.75e5, c1=-0.90368, c2=0.2749, c3=-0.028):
    # Median subhalo concentration depending on its Vmax
    # and its redshift (here z=0)
    # Moline et al. 2110.02097
    #
    # V - max radial velocity of a bound particle in the subhalo [km/s]
    ci = [c0, c1, c2, c3]
    return ci[0] * (1 + (sum([ci[i + 1] * np.log10(V) ** (i + 1)
                              for i in range(3)])))


def Moline21_normalization(V, c0):
    return (Cv_Mol2021_redshift0(V, c0, c1=-0.90368,
                                 c2=0.2749, c3=-0.028))


def Moline21_norm_log(V, c0):
    return np.log10(Cv_Mol2021_redshift0(V, c0, c1=-0.90368,
                                         c2=0.2749, c3=-0.028))


def Moline21_normalization2(V, c0, c1):
    return (Cv_Mol2021_redshift0(V, c0, c1,
                                 c2=0., c3=0.0))


def Cv_Grand_points(xx, yy):
    return 2. * (xx / yy / H0) ** 2


H0 = 67.77 / 1e3


def calculate_med(xx, yy, num_interv, nmax=60):
    x_approx = np.geomspace(1., nmax, num=num_interv)

    ymed = np.zeros(len(x_approx) - 1)
    num_subs = np.zeros(len(x_approx) - 1)

    for i in range(num_interv - 1):
        interval = ((xx > x_approx[i]) * (xx <= x_approx[i + 1]))
        num_subs[i] = sum(interval)
        ymed[i] = np.median(yy[interval])
        # ymed[i] = np.exp(np.log(yy[interval]).mean())

    x_approx = (x_approx[:-1] + x_approx[1:]) / 2.
    y_ceros = ymed > 0.
    x_approx = x_approx[y_ceros]
    ymed = ymed[y_ceros]
    num_subs = num_subs[y_ceros]

    return x_approx, ymed, num_subs


def calculate_med_numconst(xx, yy, nmax=10):
    x_min = len(yy) % nmax

    ymed = []
    xmed = []
    xerrors = []
    yerrors = []

    for i in range(int(len(yy) / nmax)):
        vv_min = x_min + i * nmax
        vv_max = x_min + (i + 1) * nmax
        log10_ymean = np.log10(yy[vv_min:vv_max]).mean()
        ymed.append(10 ** log10_ymean)
        xmed.append(10 ** (np.log10(xx[vv_min:vv_max]).mean()))
        xerrors.append([xmed[-1] - xx[vv_min], xx[vv_max - 1] - xmed[-1]])
        yerrors.append(
            [ymed[-1]
             - 10 ** (log10_ymean - np.log10(yy[vv_min:vv_max]).std()),
             10 ** (np.log10(yy[vv_min:vv_max]).std() + log10_ymean)
             - ymed[-1]])
        if i == 1384:
            print(i, yy[vv_min:vv_max], np.log10(yy[vv_min:vv_max]))
            print(10 ** (np.log10(yy[vv_min:vv_max]).std() + log10_ymean),
                  10 ** (np.log10(yy[vv_min:vv_max]).std() + log10_ymean) -
                  ymed[-1])

    return np.array(xmed), np.array(ymed), np.array(xerrors), np.array(
        yerrors).T


def calculate_mean_std(xx, yy, num_interv, nmax=60):
    x_approx = np.geomspace(1., nmax, num=num_interv)

    ymed = np.zeros(len(x_approx) - 1)
    std = np.zeros((len(x_approx) - 1))
    # std = np.zeros((len(x_approx) - 1, 2))
    num_subs = np.zeros(len(x_approx) - 1)

    for i in range(num_interv - 1):
        interval = ((xx > x_approx[i]) * (xx <= x_approx[i + 1]))
        num_subs[i] = sum(interval)
        ymed[i] = np.mean(np.log10(yy[interval]))
        # std[i] = 10**np.std(np.log10(yy[interval]))

        # std[i, 1] = (10**(np.std(np.log10(yy[interval]))
        #       + np.mean(np.log10(yy[interval])))
        #       - ymed[i])
        #
        # std[i, 0] = (-10**(-np.std(np.log10(yy[interval]))
        #       + np.mean(np.log10(yy[interval])))
        #       + ymed[i])

        std[i] = np.std(np.log10(yy[interval]))

    x_approx = (x_approx[:-1] + x_approx[1:]) / 2.
    y_ceros = ymed > 0.
    x_approx = x_approx[y_ceros]
    ymed = ymed[y_ceros]
    std = std[y_ceros]
    num_subs = num_subs[y_ceros]

    return x_approx, ymed, num_subs, std


# %% CALCULATE THE MEDIAN VALUES OF THE DISTRIBUTION Rmax-Vmax

number_bins_dmo = 26
number_bins_hydro = 26

# Limit to the power law tendency
hydroLimit = 15
dmoLimit = 15

cv_dmo_cloud_release = Cv_Grand_points(data_release_dmo[:, 1],
                                       data_release_dmo[:, 0])
cv_hydro_cloud_release = Cv_Grand_points(data_release_hydro[:, 1],
                                         data_release_hydro[:, 0])

# vv_medians_dmo_release, cv_dmo_median_release,\
# num_dmo = calculate_med(
#     data_release_dmo[:, 1], cv_dmo_cloud_release,
#     number_bins_dmo, nmax=120)
# vv_medians_hydro_release, cv_hydro_median_release, \
# num_hydro = calculate_med(
#     data_release_hydro[:, 1], cv_hydro_cloud_release,
#     number_bins_hydro, nmax=120)


vv_medians_dmo_release, cv_dmo_median_release, xerr_dmo, yerr_dmo = calculate_med_numconst(
    data_release_dmo[:, 1], cv_dmo_cloud_release, nmax=10)
vv_medians_hydro_release, cv_hydro_median_release, xerr_hydro, yerr_hydro = calculate_med_numconst(
    data_release_hydro[:, 1], cv_hydro_cloud_release, nmax=10)

fig0, ax0 = plt.subplots()
plt.errorbar(vv_medians_dmo_release, cv_dmo_median_release,
             xerr=xerr_dmo.T, yerr=yerr_dmo,
             color='k', ls='', lw=3, zorder=10,
             capsize=5)
plt.errorbar(vv_medians_hydro_release, cv_hydro_median_release,
             xerr=xerr_hydro.T, yerr=yerr_hydro,
             color='g', ls='', lw=3, zorder=10,
                 capsize=5)

plt.scatter(data_release_dmo[:, 1],
            cv_dmo_cloud_release,
            s=10, marker='x',
            alpha=0.5, color='k', zorder=1, label='Data'
            )
plt.scatter(data_release_hydro[:, 1],
            cv_hydro_cloud_release,
            s=10, marker='x',
            alpha=0.5, color='limegreen', zorder=1
            )

plt.xscale('log')
plt.yscale('log')
plt.show()

plt.subplots(1, 2, figsize=(24, 8))
plt.subplot(121)
plt.title('Subhalos over Vlimit')
data_dmo = np.log10(cv_dmo_cloud_release[data_release_dmo[:, 1] > dmoLimit])
data_hydro = np.log10(
    cv_hydro_cloud_release[data_release_hydro[:, 1] > hydroLimit])
n_dmo, bins_dmo, _ = plt.hist(
    data_dmo,
    # bins=np.geomspace(0.01, 10, num=40),
    # density=True,
    color='grey', alpha=0.7,
    bins=20
)
n_hydro, bins_hydro, _ = plt.hist(
    data_hydro,
    # bins=np.geomspace(0.01, 10, num=40),
    # density=True,
    color='limegreen', alpha=0.7,
    bins=20
)

plt.axvline(np.median(data_dmo)
            , color='k', ls='--', lw=4, label='Median')
plt.axvline(np.percentile(data_dmo, 75)
            , color='k', ls='--', lw=2, label='25 and 75 perc')
plt.axvline(np.percentile(data_dmo, 25)
            , color='k', ls='--', lw=2, zorder=200)

plt.axvline(np.median(data_hydro)
            , color='lime', ls='--', lw=4)
plt.axvline(np.percentile(data_hydro, 75)
            , color='lime', ls='--', lw=2)
plt.axvline(np.percentile(data_hydro, 25)
            , color='lime', ls='--', lw=2)

plt.axvline(np.mean(data_dmo)
            , color='k', ls='-', lw=3, label='Mean')
plt.axvline(np.mean(data_dmo) + np.std(data_dmo)
            , color='k', ls='-', lw=2, label='1 sigma from mean')
plt.axvline(np.mean(data_dmo) - np.std(data_dmo)
            , color='k', ls='-', lw=2)

plt.axvline(np.mean(data_hydro)
            , color='lime', ls='-', lw=4)
plt.axvline(np.mean(data_hydro) + np.std(data_hydro)
            , color='lime', ls='-', lw=2)
plt.axvline(np.mean(data_hydro) - np.std(data_hydro)
            , color='lime', ls='-', lw=2)
plt.legend()
plt.xlabel('log10(c_V)')

plt.subplot(122)
dmo_inlims = ((data_release_dmo[:, 1] > dmoLimit)
    # * (data_release_dmo[:, 1] < 38.5)
    )
hydro_inlims = ((data_release_hydro[:, 1] > hydroLimit)
    # * (data_release_hydro[:, 1] < 26.6)
    )
plt.scatter(data_release_dmo[:, 1][dmo_inlims],
            data_dmo,
            s=10, marker='x',
            alpha=1, color='k', zorder=14, label='Data'
            )
plt.scatter(data_release_hydro[:, 1][hydro_inlims],
            data_hydro,
            s=10, marker='x',
            alpha=1, color='limegreen', zorder=14
            )

plt.axhline(np.median(data_dmo)
            , color='k', ls='--', lw=4, label='Median')
plt.axhline(np.percentile(data_dmo, 75)
            , color='k', ls='--', lw=2, label='25 and 75 perc')
plt.axhline(np.percentile(data_dmo, 25)
            , color='k', ls='--', lw=2, zorder=200)

plt.axhline(np.median(data_hydro)
            , color='lime', ls='--', lw=4)
plt.axhline(np.percentile(data_hydro, 75)
            , color='lime', ls='--', lw=2)
plt.axhline(np.percentile(data_hydro, 25)
            , color='lime', ls='--', lw=2)

plt.axhline(np.mean(data_dmo)
            , color='k', ls='-', lw=3, label='Mean')
plt.axhline(np.mean(data_dmo) + np.std(data_dmo)
            , color='k', ls='-', lw=2, label='1 sigma from mean')
plt.axhline(np.mean(data_dmo) - np.std(data_dmo)
            , color='k', ls='-', lw=2)

plt.axhline(np.mean(data_hydro)
            , color='lime', ls='-', lw=4)
plt.axhline(np.mean(data_hydro) + np.std(data_hydro)
            , color='lime', ls='-', lw=2)
plt.axhline(np.mean(data_hydro) - np.std(data_hydro)
            , color='lime', ls='-', lw=2)
plt.legend()
plt.xscale('log')
plt.ylabel('log10(c_V)')

plt.xlabel(r'$V_\mathrm{max}$ [km s$^{-1}$]', size=28)
plt.savefig('outputs/cv_hist.png', bbox_inches='tight')
# plt.show()

fig, ax0 = plt.subplots(figsize=(8, 8))

vv_means_dmo_release, cv_dmo_mean_release, \
num_dmo, std_dmo = calculate_mean_std(
    data_release_dmo[:, 1], cv_dmo_cloud_release,
    number_bins_dmo, nmax=120)
vv_means_hydro_release, cv_hydro_mean_release, \
num_hydro, std_hydro = calculate_mean_std(
    data_release_hydro[:, 1], cv_hydro_cloud_release,
    number_bins_hydro, nmax=120)

plt.errorbar(vv_means_dmo_release, cv_dmo_mean_release,
             yerr=std_dmo,
             ms=10, marker='.', markeredgewidth=2, ls='', mec='k',
             alpha=1, color='w', zorder=15, label='Data medians',
             ecolor='k')
plt.errorbar(vv_means_hydro_release, cv_hydro_mean_release,
             yerr=std_hydro,
             ms=10, marker='.', markeredgewidth=2, ls='', mec='green',
             alpha=1, color='w', zorder=15, ecolor='green')

dmo_inlims = ((data_release_dmo[:, 1] > dmoLimit)
    # * (data_release_dmo[:, 1] < 38.5)
    )
hydro_inlims = ((data_release_hydro[:, 1] > hydroLimit)
    # * (data_release_hydro[:, 1] < 26.6)
    )
plt.scatter(data_release_dmo[:, 1][dmo_inlims],
            np.log10(cv_dmo_cloud_release[dmo_inlims]),
            s=10, marker='x',
            alpha=1, color='k', zorder=14, label='Data'
            )
plt.scatter(data_release_hydro[:, 1][hydro_inlims],
            np.log10(cv_hydro_cloud_release[hydro_inlims]),
            s=10, marker='x',
            alpha=1, color='limegreen', zorder=14
            )

xx_plot = np.geomspace(1., 120)
plt.plot(xx_plot, np.log10(Cv_Mol2021_redshift0(xx_plot)), color='red',
         label='Moliné+21', alpha=0.6, linewidth=3, zorder=9)

# print((std_dmo[0, :] + std_dmo[1, :])/2.,
#       len((std_dmo[0, :] + std_dmo[1, :])/2.),)
true_array_dmo = ((vv_means_dmo_release > dmoLimit)
    # * (num_dmo >= 10)
    )
moline_fits_dmo_release, cov_dmo = curve_fit(
    Moline21_norm_log,
    xdata=vv_means_dmo_release[true_array_dmo],
    ydata=cv_dmo_mean_release[true_array_dmo],
    p0=[4.],
    sigma=std_dmo[true_array_dmo],
    # sigma=((std_dmo[0, :] + std_dmo[1, :])/2.)[true_array_dmo],
    # bounds=([0.],
    #         [5e5])
)
print(std_dmo)
print(moline_fits_dmo_release)
print(cov_dmo, np.sqrt(cov_dmo))

plt.plot(xx_plot,
         Moline21_norm_log(V=xx_plot,
                           c0=moline_fits_dmo_release[0],
                           # c1=moline_fits_dmo_release[1],
                           # c2=moline_fits_dmo_release[2],
                           # c3=moline_fits_dmo_release[3]
                           ),
         color='k',
         alpha=1,
         linewidth=3, ls='--',
         zorder=5, label='Fits to Moliné+21')

ax0.fill_between(
    xx_plot,
    Moline21_norm_log(
        V=xx_plot,
        c0=moline_fits_dmo_release[0] - cov_dmo[0][0] ** 0.5),
    Moline21_norm_log(
        V=xx_plot,
        c0=moline_fits_dmo_release[0] + cov_dmo[0][0] ** 0.5),
    color='grey', alpha=0.5, zorder=3
)

dmo_inlims = ((data_release_dmo[:, 1] > dmoLimit)
    # * (data_release_dmo[:, 1] < 38.5)
    )
hydro_inlims = ((data_release_hydro[:, 1] > hydroLimit)
    # * (data_release_hydro[:, 1] < 26.6)
    )
moline_fits_dmo_release, cov_dmo = curve_fit(
    Moline21_norm_log,
    xdata=data_release_dmo[:, 1][dmo_inlims],
    ydata=np.log10(cv_dmo_cloud_release[dmo_inlims]),
    p0=[4.],
    # sigma=std_dmo[true_array_dmo],
    # sigma=((std_dmo[0, :] + std_dmo[1, :])/2.)[true_array_dmo],
    # bounds=([0.],
    #         [5e5])
)
print('Fit to the cloud')
print(moline_fits_dmo_release)
print(cov_dmo, np.sqrt(cov_dmo))

plt.plot(xx_plot,
         Moline21_norm_log(V=xx_plot,
                           c0=moline_fits_dmo_release[0],
                           # c1=moline_fits_dmo_release[1],
                           # c2=moline_fits_dmo_release[2],
                           # c3=moline_fits_dmo_release[3]
                           ),
         color='limegreen',
         alpha=1,
         linewidth=3, ls='--',
         zorder=5, label='Fits to Moliné+21')

ax0.fill_between(
    xx_plot,
    Moline21_norm_log(
        V=xx_plot,
        c0=moline_fits_dmo_release[0] - cov_dmo[0][0] ** 0.5),
    Moline21_norm_log(
        V=xx_plot,
        c0=moline_fits_dmo_release[0] + cov_dmo[0][0] ** 0.5),
    color='limegreen', alpha=0.5, zorder=3
)
# ax0.fill_between(
#     xx_plot,
#     Moline21_normalization(
#         V=xx_plot,
#         c0=moline_fits_hydro_release[0])
#     * 10**(fit_normalfree_hydro[0][0] - fit_normalfree_hydro[0][1]),
#     Moline21_normalization(
#         V=xx_plot,
#         c0=moline_fits_hydro_release[0])
#     * 10**(fit_normalfree_hydro[0][0] + fit_normalfree_hydro[0][1]),
#     color='#00FF00', alpha=0.4, zorder=2
# )
ax0.axvline(dmoLimit,
            linestyle='-.', color='k', alpha=0.3,
            linewidth=2, label='Fit limits')

# plt.yscale('log')
plt.xscale('log')

# plt.show()
# FIGURE start ---------------------------------------------------------
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(20, 8))
plt.subplot(121)

plt.plot(vv_medians_dmo_release, cv_dmo_median_release,
         ms=10, marker='.', markeredgewidth=2, ls='', mec='k',
         alpha=1, color='w', zorder=15, label='Data medians')
plt.plot(vv_medians_hydro_release, cv_hydro_median_release,
         ms=10, marker='.', markeredgewidth=2, ls='', mec='green',
         alpha=1, color='w', zorder=15)

dmo_inlims = ((data_release_dmo[:, 1] > dmoLimit)
              * (data_release_dmo[:, 1] < 38.5)
              )
hydro_inlims = ((data_release_hydro[:, 1] > hydroLimit)
                * (data_release_hydro[:, 1] < 26.6)
                )
plt.scatter(data_release_dmo[:, 1][dmo_inlims],
            cv_dmo_cloud_release[dmo_inlims],
            s=10, marker='x',
            alpha=1, color='k', zorder=14, label='Data'
            )
plt.scatter(data_release_hydro[:, 1][hydro_inlims],
            cv_hydro_cloud_release[hydro_inlims],
            s=10, marker='x',
            alpha=1, color='limegreen', zorder=14
            )

xx_plot = np.geomspace(1., 120)
plt.plot(xx_plot, Cv_Mol2021_redshift0(xx_plot), color='red',
         label='Moliné+21', alpha=0.6, linewidth=3, zorder=9)

# DMO fit -----------------------------------------------------

true_array_dmo = ((vv_medians_dmo_release > dmoLimit)
    # * (num_dmo >= 10)
    )
print('DMO data with V>Vmin and #subs in bin > 10: ', sum(true_array_dmo))

moline_fits_dmo_release, cov_dmo = curve_fit(
    Moline21_normalization,
    # xdata=vv_medians_dmo_release[true_array_dmo],
    # ydata=cv_dmo_median_release[true_array_dmo],
    xdata=data_release_dmo[:, 1][dmo_inlims],
    ydata=cv_dmo_cloud_release[dmo_inlims],
    # p0=[1.e4, -0.90368, 0.2749, -0.028],
    # bounds=([0., -5., -5., -5.],
    #         [5e5, 5., 5., 5.])
)

print(moline_fits_dmo_release)
print(cov_dmo, np.sqrt(cov_dmo))

plt.plot(xx_plot,
         Cv_Mol2021_redshift0(V=xx_plot,
                              c0=moline_fits_dmo_release[0],
                              # c1=moline_fits_dmo_release[1],
                              # c2=moline_fits_dmo_release[2],
                              # c3=moline_fits_dmo_release[3]
                              ),
         color='k',
         alpha=1,
         linewidth=3, ls='--',
         zorder=5, label='Fits to Moliné+21')

ax.axvline(dmoLimit,
           linestyle='-.', color='k', alpha=0.3,
           linewidth=2, label='Fit limits')

# Hydro fit -----------------------------------------------------------
print()
true_array_hydro = ((vv_medians_hydro_release > hydroLimit)
    # * (num_hydro >= 10)
    )
print('Hydro data with V>Vmin and #subs in bin > 10: ',
      sum(true_array_hydro))

try:
    moline_fits_hydro_release, cov_hydro = curve_fit(
        Moline21_normalization,
        xdata=vv_medians_hydro_release[true_array_hydro],
        ydata=cv_hydro_median_release[true_array_hydro],
        p0=[1.e4],
        # bounds=([-np.inf],
        #         [5e5])
    )
    print(moline_fits_hydro_release)
    print(cov_hydro, np.sqrt(cov_hydro))

    plt.plot(xx_plot,
             Cv_Mol2021_redshift0(V=xx_plot,
                                  c0=moline_fits_hydro_release[0]),
             color='#00FF00',
             alpha=1,
             linewidth=3, ls='--',
             zorder=5)
except:
    print('Hydro release did not found good parameters')

plt.xscale('log')
plt.yscale('log')

handles = (mpatches.Patch(color='k', label='DMO', alpha=0.8),
           mpatches.Patch(color='limegreen', label='Hydro', alpha=0.8)
           )

legend11 = plt.legend(handles=handles,
                      loc=3, bbox_to_anchor=(0.17, 0.))

plt.ylabel(r'c$_\mathrm{V}$', size=28)
plt.xlabel(r'$V_\mathrm{max}$ [km s$^{-1}$]', size=28)
plt.show()
plt.subplot(122)  # -------------------------------------------------
print()

limit_dmo_max = 10 ** (
        (np.log10(vv_medians_dmo_release[
                      np.where(num_dmo < 10)[0][1]])
         + np.log10(vv_medians_dmo_release[
                        np.where(num_dmo < 10)[0][1] - 1])
         ) / 2.)
true_array_dmo = (
        (data_release_dmo[:, 1] > dmoLimit)
        * (data_release_dmo[:, 1]
           < limit_dmo_max))

vv_over_dmo = data_release_dmo[true_array_dmo, 1]
cc_over_dmo = cv_dmo_cloud_release[true_array_dmo]
print('number over dmo', len(cc_over_dmo))
ax.axvline(limit_dmo_max,
           color='k',
           alpha=1,
           linewidth=2, ls='-.',
           zorder=0)
n_dmo, bins_dmo, _ = plt.hist(
    cc_over_dmo / Cv_Mol2021_redshift0(V=vv_over_dmo,
                                       c0=moline_fits_dmo_release[0],
                                       # c1=moline_fits_dmo_release[1],
                                       # c2=moline_fits_dmo_release[2],
                                       # c3=moline_fits_dmo_release[3]
                                       ),
    bins=np.geomspace(0.01, 10, num=40),
    density=True,
    color='grey', alpha=0.7
)

# n_dmo, bins_dmo, _ = plt.hist(
#     cc_over_dmo / Moline21_normalization(
#         V=vv_over_dmo,
#         # c0=1.75e5,
#         c0=moline_fits_dmo_release[0]
#     ),
#     bins=np.geomspace(0.01, 10, num=40),
#     density=True,
#     color='grey', alpha=0.7
# )


limit_hydro_max = 10 ** (
        (np.log10(vv_medians_hydro_release[
                      np.where(num_hydro < 10)[0][1]])
         + np.log10(vv_medians_hydro_release[
                        np.where(num_hydro < 10)[0][1] - 1])
         ) / 2.)
true_array_hydro = ((data_release_hydro[:, 1] > hydroLimit)
                    * (data_release_hydro[:, 1] < limit_hydro_max))

vv_over_hydro = data_release_hydro[true_array_hydro, 1]
cc_over_hydro = cv_hydro_cloud_release[true_array_hydro]
print('number over hydro', len(cc_over_hydro))
print(min(cc_over_dmo / Moline21_normalization(
    V=vv_over_dmo, c0=moline_fits_dmo_release[0])),
      max(cc_over_dmo / Moline21_normalization(
          V=vv_over_dmo, c0=moline_fits_dmo_release[0])))

n_hydro, bins_hydro, _ = plt.hist(
    cc_over_hydro / Moline21_normalization(
        V=vv_over_hydro,
        # c0=1.75e5,
        c0=moline_fits_hydro_release[0]
    )
    ,
    bins=np.geomspace(0.01, 10, num=40),
    density=True,
    color='limegreen', alpha=0.7
)
ax.axvline(limit_hydro_max,
           color='limegreen',
           alpha=1,
           linewidth=2, ls='-.',
           zorder=0)

print(np.max(cc_over_dmo / Moline21_normalization(
    V=vv_over_dmo, c0=moline_fits_dmo_release[0])),
      np.max(cc_over_hydro / Moline21_normalization(
          V=vv_over_hydro, c0=moline_fits_hydro_release[0])))
print(np.min(cc_over_dmo / Moline21_normalization(
    V=vv_over_dmo, c0=moline_fits_dmo_release[0])),
      np.min(cc_over_hydro / Moline21_normalization(
          V=vv_over_hydro, c0=moline_fits_hydro_release[0])))

plt.xlabel(r'$\frac{c_\mathrm{V, data}}{c_\mathrm{V, Moliné21}(V)}$',
           fontsize=34)
plt.ylabel('Normalized density')


def lognormal_fit(xx, mean, sigma):
    return (1 / (sigma * xx * np.sqrt(2. * np.pi))
            * np.exp(-0.5 * ((np.log(xx) - mean) / sigma) ** 2.))


def normal_fit(xx, mean, sigma):
    return (1 / (sigma * np.sqrt(2. * np.pi))
            * np.exp(-0.5 * ((xx - mean) / sigma) ** 2.))


def normal_fitfree(xx, mean, sigma, aa):
    return aa * (1 / (sigma * np.sqrt(2. * np.pi))
                 * np.exp(-0.5 * ((xx - mean) / sigma) ** 2.))


fit_lognormal_dmo = curve_fit(
    lognormal_fit,
    xdata=(bins_dmo[1:] + bins_dmo[:-1]) / 2.,
    ydata=n_dmo)
fit_normal_dmo = curve_fit(
    normal_fit,
    xdata=np.log10((bins_dmo[1:] + bins_dmo[:-1]) / 2.),
    ydata=n_dmo)
fit_normalfree_dmo = curve_fit(
    normal_fitfree,
    xdata=np.log10((bins_dmo[1:] + bins_dmo[:-1]) / 2.),
    ydata=n_dmo)

print('Sigma for the lognormal distribution, dmo: ',
      np.log10(np.exp(fit_lognormal_dmo[0][1])))
print()
print(fit_lognormal_dmo)
print('Fit normal dmo')
print(fit_normal_dmo)
print('fit_normalfree_dmo')
print(fit_normalfree_dmo)

xx2_plot = np.geomspace(0.01, 10, num=100)

plt.plot(xx2_plot,
         lognormal_fit(xx2_plot, fit_lognormal_dmo[0][0],
                       fit_lognormal_dmo[0][1]),
         color='k', lw=2, label='Lognormal fit')
# plt.plot(xx2_plot,
#          normal_fit(np.log10(xx2_plot), fit_normal_dmo[0][0],
#                        fit_normal_dmo[0][1]),
#          color='k', lw=2, ls='--')
# plt.plot(xx2_plot,
#          normal_fitfree(np.log10(xx2_plot), fit_normalfree_dmo[0][0],
#                        fit_normalfree_dmo[0][1],
#                        fit_normalfree_dmo[0][2]),
#          color='k', lw=2, marker='+', ls='')

fit_lognormal_hydro = curve_fit(
    lognormal_fit,
    xdata=(bins_hydro[1:] + bins_hydro[:-1]) / 2.,
    ydata=n_hydro)

fit_normalfree_hydro = curve_fit(
    normal_fitfree,
    xdata=np.log10((bins_hydro[1:] + bins_hydro[:-1]) / 2.),
    ydata=n_hydro)

print('Sigma for the lognormal distribution, hydro: ',
      np.log10(np.exp(fit_lognormal_hydro[0][1])),
      np.log10(np.exp(fit_lognormal_hydro[0][0])),
      (np.exp(fit_lognormal_hydro[0][0])))
print(fit_lognormal_hydro)
print(fit_normalfree_hydro[0], 10 ** fit_normalfree_hydro[0][0])

plt.plot(xx2_plot,
         lognormal_fit(xx2_plot, fit_lognormal_hydro[0][0],
                       fit_lognormal_hydro[0][1]),
         color='green', lw=2)
# plt.plot(xx2_plot,
#          normal_fitfree(np.log10(xx2_plot), fit_normalfree_hydro[0][0],
#                        fit_normalfree_hydro[0][1],
#                        fit_normalfree_hydro[0][2]),
#          color='g', lw=2, marker='+', ls='')
# plt.plot(np.log10(xx2_plot),
#          lognormal_fit(xx2_plot, fit_lognormal_hydro[0][0],
#                        fit_lognormal_hydro[0][1]),
#          color='green', lw=2)
# plt.plot(np.log10(xx2_plot),
#          normal_fitfree(np.log10(xx2_plot), fit_normalfree_hydro[0][0],
#                        fit_normalfree_hydro[0][1],
#                        fit_normalfree_hydro[0][2]),
#          color='g', lw=2, marker='+', ls='')
print()

plt.axvline(10 ** fit_normalfree_dmo[0][0]
            , color='k', ls='dotted', lw=4, label='Mean value from fit')
plt.axvline(10 ** (fit_normalfree_dmo[0][0] + fit_normalfree_dmo[0][1])
            , color='k', ls='--', lw=2, label='1sigma spread from fit')
plt.axvline(10 ** (fit_normalfree_dmo[0][0] - fit_normalfree_dmo[0][1])
            , color='k', ls='--', lw=2)

plt.axvline(10 ** fit_normalfree_hydro[0][0]
            , color='green', ls='dotted', lw=4)
plt.axvline(10 ** (fit_normalfree_hydro[0][0] + fit_normalfree_hydro[0][1])
            , color='green', ls='--', lw=2)
plt.axvline(10 ** (fit_normalfree_hydro[0][0] - fit_normalfree_hydro[0][1])
            , color='green', ls='--', lw=2)

plt.xscale('log')
lg1 = plt.legend(handles=handles, loc=1)
lg2 = plt.legend(
    loc=2, framealpha=1, fontsize=16)
ax2.add_artist(lg1)
ax2.add_artist(lg2)

plt.xlim(1e-2, 1e1)

# -----------------------------------------------------------

ax.plot(xx_plot, Moline21_normalization(
    xx_plot,
    c0=moline_fits_dmo_release[0] * 10 ** fit_normalfree_dmo[0][0]),
        c='k', linestyle='dotted', zorder=10,
        label='Mean of histogram (fit)', lw=2.5)
ax.plot(xx_plot, Moline21_normalization(
    xx_plot,
    c0=moline_fits_hydro_release[0] * 10 ** fit_normalfree_hydro[0][0]),
        c='green', linestyle='dotted', zorder=10, lw=2.5)

ax.fill_between(
    xx_plot,
    Moline21_normalization(
        V=xx_plot,
        c0=moline_fits_dmo_release[0])
    * 10 ** (fit_normalfree_dmo[0][0] - fit_normalfree_dmo[0][1]),
    Moline21_normalization(
        V=xx_plot,
        c0=moline_fits_dmo_release[0])
    * 10 ** (fit_normalfree_dmo[0][0] + fit_normalfree_dmo[0][1]),
    color='grey', alpha=0.5, zorder=3
)
ax.fill_between(
    xx_plot,
    Moline21_normalization(
        V=xx_plot,
        c0=moline_fits_hydro_release[0])
    * 10 ** (fit_normalfree_hydro[0][0] - fit_normalfree_hydro[0][1]),
    Moline21_normalization(
        V=xx_plot,
        c0=moline_fits_hydro_release[0])
    * 10 ** (fit_normalfree_hydro[0][0] + fit_normalfree_hydro[0][1]),
    color='#00FF00', alpha=0.4, zorder=2
)

# ax.fill_between(
#     xx_plot,
#     Moline21_normalization(
#         V=xx_plot,
#         c0=moline_fits_dmo_release[0])
#     * 10**(fit_normalfree_dmo[0][0] - fit_normalfree_dmo[0][1]),
#     Moline21_normalization(
#         V=xx_plot,
#         c0=moline_fits_dmo_release[0])
#     * 10**(fit_normalfree_dmo[0][0] + fit_normalfree_dmo[0][1]),
#     color='grey', alpha=0.5, zorder=3
# )
# ax.fill_between(
#     xx_plot,
#     Moline21_normalization(
#         V=xx_plot,
#         c0=moline_fits_hydro_release[0])
#     * 10**(fit_normalfree_hydro[0][0] - fit_normalfree_hydro[0][1]),
#     Moline21_normalization(
#         V=xx_plot,
#         c0=moline_fits_hydro_release[0])
#     * 10**(fit_normalfree_hydro[0][0] + fit_normalfree_hydro[0][1]),
#     color='#00FF00', alpha=0.4, zorder=2
# )
# ax.fill_between(
#     xx_plot,
#     Moline21_normalization(
#         V=xx_plot,
#         c0=moline_fits_dmo_release[0] - cov_dmo[0][0]**0.5
#     ),
#     Moline21_normalization(
#         V=xx_plot,
#         c0=moline_fits_dmo_release[0] + cov_dmo[0][0]**0.5),
#     color='fuchsia', alpha=0.5, zorder=30
# )
# ax.fill_between(
#     xx_plot,
#     Moline21_normalization(
#         V=xx_plot,
#         c0=moline_fits_hydro_release[0] - cov_hydro[0][0]**0.5),
#     Moline21_normalization(
#         V=xx_plot,
#         c0=moline_fits_hydro_release[0] + cov_hydro[0][0]**0.5),
#     color='orange', alpha=0.4, zorder=20
# )

legend_types = ax.legend(fontsize=18, loc=4, framealpha=1)
ax.add_artist(legend11)
ax.add_artist(legend_types)

fig.savefig('outputs/cv_median.pdf', bbox_inches='tight')
fig.savefig('outputs/cv_median.png', bbox_inches='tight')
plt.show()
