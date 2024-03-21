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
    # Median subhalo concentration depending on its Vmax and its redshift (here z=0)
    # Moline et al. 2110.02097
    #
    # V - max radial velocity of a bound particle in the subhalo [km/s]
    ci = [c0, c1, c2, c3]
    return ci[0] * (1 + (sum([ci[i + 1] * np.log10(V) ** (i + 1)
                              for i in range(3)])))


def Moline21_normalization(V, c0):
    return (Cv_Mol2021_redshift0(V, c0, c1=-0.90368,
                                 c2=0.2749, c3=-0.028))


def Moline21_normalization2(V, c0, c1):
    return (Cv_Mol2021_redshift0(V, c0, c1,
                                 c2=0., c3=0.0))


def Cv_Grand_points(xx, yy):
    return 2. * (xx / yy / H0) ** 2


H0 = 70. / 1e3


def calculate_med(xx, yy, num_interv, nmax=60):
    x_approx = np.geomspace(xx[0], nmax, num=num_interv)

    ymed = []
    num_subs = []

    for i in range(num_interv - 1):
        try:
            xmin = np.where(xx > x_approx[i])[0][0]
            xmax = np.where(xx > x_approx[i + 1])[0][0]

            num_subs.append(xmax - xmin)
            ymed.append(np.median(yy[xmin:xmax]))

        except:
            ymed.append(0)
            num_subs.append(0)

    x_approx = (x_approx[:-1] + x_approx[1:]) / 2.
    y_ceros = np.array(ymed) > 0.
    x_approx = x_approx[y_ceros]
    ymed = np.array(ymed)[y_ceros]
    num_subs = np.array(num_subs)[y_ceros]

    return x_approx, ymed, num_subs


# %% CALCULATE THE MEDIAN VALUES OF THE DISTRIBUTION Rmax-Vmax

number_bins_dmo = 25
number_bins_hydro = 25

# Limit to the power law tendency
hydroLimit = 11
dmoLimit = 11

cv_dmo_cloud_release = Cv_Grand_points(data_release_dmo[:, 1],
                                       data_release_dmo[:, 0])
cv_hydro_cloud_release = Cv_Grand_points(data_release_hydro[:, 1],
                                         data_release_hydro[:, 0])

vv_medians_dmo_release, cv_dmo_median_release, num_dmo = calculate_med(
    data_release_dmo[:, 1], cv_dmo_cloud_release,
    number_bins_dmo, nmax=120)
vv_medians_hydro_release, cv_hydro_median_release, num_hydro = calculate_med(
    data_release_hydro[:, 1], cv_hydro_cloud_release,
    number_bins_hydro, nmax=120)

xx_plot = np.geomspace(1., 100)

# FIGURE start ---------------------------------------------------------
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(20, 8))
plt.subplot(121)
plt.plot(vv_medians_dmo_release, cv_dmo_median_release,
         ms=10, marker='+', markeredgewidth=2, ls='',
         alpha=1, color='k', zorder=15, label='Release data')
plt.plot(vv_medians_hydro_release, cv_hydro_median_release,
         ms=10, marker='+', markeredgewidth=2, ls='',
         alpha=1, color='limegreen', zorder=15)

plt.plot(xx_plot, Cv_Mol2021_redshift0(xx_plot), color='red',
         label='Moliné+21', alpha=0.6, linewidth=3, zorder=9)

# DMO fit -----------------------------------------------------

true_array = ((vv_medians_dmo_release > dmoLimit)
              * (num_dmo > 10))
print('DMO data with V>Vmin and #subs in bin > 10: ', sum(true_array))

moline_fits_dmo_release = curve_fit(
    Moline21_normalization,
    xdata=vv_medians_dmo_release[true_array],
    ydata=cv_dmo_median_release[true_array],
    p0=[1.e4],
    # bounds=([-np.inf],
    #         [5e5])
)

print(moline_fits_dmo_release)

plt.plot(xx_plot,
         Cv_Mol2021_redshift0(V=xx_plot,
                              c0=moline_fits_dmo_release[0][0]),
         color='k',
         alpha=1,
         linewidth=3, ls='--',
         zorder=5, label='Fit to release')
print(moline_fits_dmo_release[0][0] + moline_fits_dmo_release[1][0] ** 0.5,
      moline_fits_dmo_release[0][0]
      - moline_fits_dmo_release[1][0] ** 0.5
      )
plt.axvline(vv_medians_dmo_release[
                np.where(true_array[:-1] > true_array[1:])],
            color='k',
            alpha=1,
            linewidth=2, ls='-.',
            zorder=0)

# Hydro fit -----------------------------------------------------------
print()
true_array = ((vv_medians_hydro_release > hydroLimit)
              * (num_hydro > 10))
print('Hydro data with V>Vmin and #subs in bin > 10: ', sum(true_array))
try:
    moline_fits_hydro_release = curve_fit(
        Moline21_normalization,
        xdata=vv_medians_hydro_release[true_array],
        ydata=cv_hydro_median_release[true_array],
        p0=[1.e4],
        # bounds=([-np.inf],
        #         [5e5])
    )
    print(moline_fits_hydro_release[0][0]
          + moline_fits_hydro_release[1][0] ** 0.5,
          moline_fits_hydro_release[0][0]
          - moline_fits_hydro_release[1][0] ** 0.5
          )
    print(moline_fits_hydro_release)

    plt.plot(xx_plot,
             Cv_Mol2021_redshift0(V=xx_plot,
                                  c0=moline_fits_hydro_release[0][0]),
             color='green',
             alpha=1,
             linewidth=3, ls='--',
             zorder=5)
    plt.axvline(vv_medians_hydro_release[
                    np.where(true_array[:-1] > true_array[1:])],
                color='limegreen',
                alpha=1,
                linewidth=2, ls='-.',
                zorder=0)
except:
    print('Hydro release did not found good parameters')

plt.axvline(dmoLimit, linestyle='-.', color='k', alpha=0.3,
            linewidth=2, label='Fit limit')

plt.xscale('log')
plt.yscale('log')

handles = (mpatches.Patch(color='k', label='DMO', alpha=0.8),
           mpatches.Patch(color='limegreen', label='Hydro', alpha=0.8)
           )

legend11 = plt.legend(handles=handles,
                      loc=1)  # , bbox_to_anchor=(1.001, 0.99))

plt.ylabel(r'c$_\mathrm{V}$', size=28)
plt.xlabel(r'$V_\mathrm{max}$ [km s$^{-1}$]', size=28)

plt.subplot(122)  # -------------------------------------------------

xx2_plot = np.geomspace(0.01, 6, num=100)

num = 25

vv_over_dmo = data_release_dmo[data_release_dmo[:, 1] > dmoLimit, 1]
cc_over_dmo = cv_dmo_cloud_release[data_release_dmo[:, 1] > dmoLimit]

n_dmo, bins_dmo, _ = plt.hist(
    cc_over_dmo / Moline21_normalization(
        V=vv_over_dmo, c0=moline_fits_dmo_release[0][0]),
    bins=np.geomspace(0.05, 6, num=num),
    density=True,
    color='grey', alpha=0.7
)

num = 25
vv_over_hydro = data_release_hydro[data_release_hydro[:, 1] > hydroLimit, 1]
cc_over_hydro = cv_hydro_cloud_release[data_release_hydro[:, 1] > hydroLimit]

n_hydro, bins_hydro, _ = plt.hist(cc_over_hydro / Moline21_normalization(
    V=vv_over_hydro, c0=moline_fits_hydro_release[0][0]),
                                  bins=np.geomspace(0.05, 6, num=num),
                                  density=True,
                                  color='limegreen', alpha=0.7
                                  )

plt.xlabel(r'$\frac{c_\mathrm{V, data}}{c_\mathrm{V, Moliné21}(V)}$',
           fontsize=34)
plt.ylabel('Normalized density')


def lognormal_fit(xx, mean, sigma):
    return (1 / (sigma * xx * np.sqrt(2. * np.pi))
            * np.exp(-0.5 * ((np.log(xx) - mean) / sigma) ** 2.))


fit_lognormal_dmo = curve_fit(
    lognormal_fit,
    xdata=(bins_dmo[1:] + bins_dmo[:-1]) / 2.,
    ydata=n_dmo)

print('Sigma for the lognormal distribution, dmo: ',
      np.log10(np.exp(fit_lognormal_dmo[0][1])))

plt.plot(xx2_plot,
         lognormal_fit(xx2_plot, fit_lognormal_dmo[0][0],
                       fit_lognormal_dmo[0][1]),
         color='k', lw=2)

fit_lognormal_hydro = curve_fit(
    lognormal_fit,
    xdata=(bins_hydro[1:] + bins_hydro[:-1]) / 2.,
    ydata=n_hydro)

print('Sigma for the lognormal distribution, hydro: ',
      np.log10(np.exp(fit_lognormal_hydro[0][1])))

plt.plot(xx2_plot,
         lognormal_fit(xx2_plot, fit_lognormal_hydro[0][0],
                       fit_lognormal_hydro[0][1]),
         color='green', lw=2)

plt.xscale('log')
lg1 = plt.legend(handles=handles, loc=1)
lg2 = plt.legend([plt.Line2D([], [],
                             linestyle='-', lw=2,
                             color='k')],
                 ['Lognormal fit'],
                 loc=2, framealpha=1)
ax2.add_artist(lg1)
ax2.add_artist(lg2)


# -----------------------------------------------------------
def gaussian_not(xx, sigma, x0, aa):
    return aa / ((2. * np.pi) ** 0.5 * sigma) * np.exp(
        -0.5 * ((xx - x0) / sigma) ** 2.)


fit_normal_dmo = curve_fit(
    gaussian_not,
    xdata=np.log10((bins_dmo[1:] + bins_dmo[:-1]) / 2.),
    ydata=n_dmo)

fit_normal_hydro = curve_fit(
    gaussian_not,
    xdata=np.log10((bins_hydro[1:] + bins_hydro[:-1]) / 2.),
    ydata=n_hydro)

ax.fill_between(xx_plot,
                Moline21_normalization(
                    V=xx_plot,
                    c0=moline_fits_dmo_release[0][0])
                * 10 ** -fit_normal_dmo[0][0],
                Moline21_normalization(
                    V=xx_plot,
                    c0=moline_fits_dmo_release[0][0]
                       * 10 ** fit_normal_dmo[0][0]),
                color='grey', alpha=0.5, zorder=3
                )
ax.fill_between(xx_plot,
                Moline21_normalization(
                    V=xx_plot,
                    c0=moline_fits_hydro_release[0][0])
                * 10 ** -fit_normal_hydro[0][
                    0],
                Moline21_normalization(
                    V=xx_plot,
                    c0=moline_fits_hydro_release[0][0])
                * 10 ** fit_normal_hydro[0][0],
                color='limegreen', alpha=0.5, zorder=2
                )

legend_types = ax.legend(fontsize=18, loc=4, framealpha=1)
ax.add_artist(legend11)
ax.add_artist(legend_types)

fig.savefig('outputs/cv_median.pdf', bbox_inches='tight')
fig.savefig('outputs/cv_median.png', bbox_inches='tight')
plt.show()
