import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colorbar as colorbarr
import matplotlib.patches as mpatches


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
plt.rc('legend', fontsize=16)
plt.rc('figure', titlesize=all_size)
plt.rc('xtick', top=True, direction='in')
plt.rc('ytick', right=True, direction='in')
plt.rc('xtick.major', size=10, width=2, top=False, pad=10)
plt.rc('ytick.major', size=10, width=2, right=True, pad=10)
plt.rc('xtick.minor', size=7, width=1.5, top=False)
plt.rc('ytick.minor', size=7, width=1.5)


data_dmo = np.loadtxt(
    '../Data_subhalo_simulations/dmo_table.txt', skiprows=3)
data_hydro = np.loadtxt(
    '../Data_subhalo_simulations/hydro_table.txt', skiprows=3)

plt.subplots(2, 1)
plt.subplot(211)
plt.loglog(data_dmo[:, 3], data_dmo[:, 1], marker='.', ls='',
           color='k', alpha=0.7)
plt.loglog(data_hydro[:, 3], data_hydro[:, 1], marker='.', ls='',
           color='green', alpha=0.7)
plt.ylabel(r'$V_\mathrm{max}$ [km s$^{-1}$]')

plt.subplot(212)
plt.loglog(data_dmo[:, 3], data_dmo[:, 0], marker='.', ls='',
           color='k', alpha=0.7)
plt.loglog(data_hydro[:, 3], data_hydro[:, 0], marker='.', ls='',
           color='green', alpha=0.7)
plt.xlabel('Log10(Mass [Msun])')
plt.ylabel(r'$R_\mathrm{max}$ [kpc]')
plt.show()
