import os
import yaml
import scipy
import numpy as np
import matplotlib.colorbar as colorbarr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.colors as mcb
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator

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

def read_config_file(ConfigFile):
    with open(ConfigFile, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml

data_dict = read_config_file('input_files/input_paper2024_SHVFnorm.yml')

cosmo_G = data_dict['cosmo_constants']['G']
cosmo_H_0 = data_dict['cosmo_constants']['H_0']
cosmo_rho_crit = data_dict['cosmo_constants']['rho_crit']

host_R_vir = data_dict['host']['R_vir']
host_rho_0 = float(data_dict['host']['rho_0'])
host_r_s = 12. #data_dict['host']['r_s']

def find_roche(dist, vmax, c0=1.75e5):
    c_mean = funct_repop.Cv_Mol2021_redshift0(vmax, c0=c0)
    return (funct_repop.R_t(vmax, c_mean, dist,
                    cosmo_H_0, cosmo_G,
                    host_rho_0, host_r_s,
                    singular_case=True)
            - funct_repop.R_s(vmax, c_mean, cosmo_H_0))


def r_t_other(mass, rr, r_s=host_r_s, rho_0=host_rho_0):
    aaa = 4. * np.pi * rho_0 * r_s ** 3.
    print(aaa)
    aaa *= (2 * r_s + 3. * rr) / (r_s + rr) ** 2.
    print(aaa)
    aaa /= np.log(4. * np.pi * rho_0 * r_s ** 3.
                  * (np.log((r_s + rr) / r_s) - rr / (r_s + rr)))
    print(aaa)

    aaa = 2. - aaa
    print(aaa)

    aaa = rr * (mass / aaa
                / funct_repop.Mhost_encapsulated(rr, rho_0, r_s)) ** (1 / 3.)
    print(aaa)
    return aaa


plt.figure(figsize=(10, 8))

# vmax_array = np.geomspace(0.1, 120., num=10)
vmax_array = [0.1, 1., 10., 120.]
vv = float(len(vmax_array))
dist_gc = np.geomspace(1e-3, 250., num=100)
c0_arr = [1e4, 1.75e5, 1e6]

markers = ['.', 'x', '*']
lines = ['-', '--', ':']
print('rtt')
print(r_t_other(1e7, dist_gc))
for nj, j in enumerate(c0_arr):
    for ni, i in enumerate(vmax_array):
        c0 = j
        c_mean = funct_repop.Cv_Mol2021_redshift0(i, c0=c0)
        r_s = funct_repop.R_s(i, c_mean, cosmo_H_0)

        r_t = funct_repop.R_t(i, c_mean, dist_gc,
            cosmo_H_0, cosmo_G,
            host_rho_0, host_r_s,
            singular_case=True)

        plt.plot(dist_gc, r_t_other(1e7, dist_gc))

        if nj == 0:
            cut = np.argmin(abs(r_t-r_s))
            plt.plot(dist_gc[cut:], r_t[cut:], c=cm.viridis(ni / vv),
                 label='%.1f' % i, ls=lines[nj], lw=2)
            plt.plot(dist_gc[:cut], r_t[:cut], c=cm.viridis(ni / vv),
                 ls=lines[nj], alpha=0.3)
        else:
            cut = np.argmin(abs(r_t-r_s))
            plt.plot(dist_gc[cut:], r_t[cut:], c=cm.viridis(ni / vv),
                     ls=lines[nj], lw=2)
            plt.plot(dist_gc[:cut], r_t[:cut], c=cm.viridis(ni / vv),
                     ls=lines[nj], alpha=0.3)
        plt.scatter(scipy.optimize.newton(
            find_roche, 10, args=[i, c0]),
        r_s, c=cm.viridis(ni / vv), marker=markers[nj],
        s=500)

plt.grid(which='both')

plt.xscale('log')
plt.yscale('log')

# plt.xlim(0, 250)

aaa = plt.legend(title=r'$V_\mathrm{max}$ [km s$^{-1}$]')
bbb = plt.legend([Line2D([], [], c='k', ls=lines[i],
                         marker=markers[i], ms=15)
                         for i in range(len(markers))],
                 ['%.2e' %i for i in c0_arr], loc=9,
                 title=r'$c_0$')

plt.gca().add_artist(aaa)
plt.gca().add_artist(bbb)

plt.ylabel(r'R$_t$ [kpc]', fontsize=20)
plt.xlabel(r'D$_\mathrm{GC}$ [kpc]', fontsize=20)

plt.show()