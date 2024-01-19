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

#        Rmax[kpc]        Vmax[km/s]      Radius[Mpc]
try:
    Grand_dmo = np.loadtxt(
        '../Data_subhalos_simulations/RmaxVmaxRadDMO0_1.txt')
    Grand_hydro = np.loadtxt(
        '../Data_subhalos_simulations/RmaxVmaxRadFP0_1.txt')

except:
    Grand_dmo = np.loadtxt('../../RmaxVmaxRadDMO0_1.txt')
    Grand_hydro = np.loadtxt('../../RmaxVmaxRadFP0_1.txt')

Grand_hydro = Grand_hydro[Grand_hydro[:, 1] > 1e-4, :]

Grand_dmo[:, 2] *= 1e3
Grand_hydro[:, 2] *= 1e3

Grand_dmo = Grand_dmo[np.argsort(Grand_dmo[:, 1])]
Grand_hydro = Grand_hydro[np.argsort(Grand_hydro[:, 1])]

fig, ax = plt.subplots(figsize=(10, 8))

plt.scatter(Grand_dmo[:, 1], Grand_dmo[:, 0],
            color='k', s=10, zorder=10)
plt.scatter(Grand_hydro[:, 1], Grand_hydro[:, 0],
            color='limegreen', s=10, zorder=10)


plt.axhline(0.184, linestyle='-.', color='k', alpha=0.6,
            linewidth=2, zorder=0)
plt.text(x=30, y=0.20, s='softening length', color='k', alpha=0.8,
         fontsize=18)


# Arrows and text
plt.axvline(7.5, linestyle='dotted', color='k', alpha=0.8,
            linewidth=2)

plt.arrow(x=6.5, y=11, dx=-2.5, dy=0, width=0.7, head_length=0.3,
          facecolor='w')
plt.text(x=6.5, y=14, s='Dark satellites\n(DM)', fontsize=18,
         horizontalalignment='right')

plt.arrow(x=8.7, y=11, dx=5, dy=0, width=0.7, head_length=1,
          facecolor='w')
plt.text(x=8.7, y=14, s='Dwarfs\n(DM + baryons)', fontsize=18,
         horizontalalignment='left')


plt.xscale('log')
plt.yscale('log')

handles = (mpatches.Patch(color='k', label='DMO', alpha=0.8),
           mpatches.Patch(color='limegreen', label='Hydro', alpha=0.8)
           )

legend11 = plt.legend(handles=handles, loc=2)

ax.add_artist(legend11)

plt.ylabel(r'$R_\mathrm{max}$ [kpc]')
plt.xlabel(r'$V_\mathrm{max}$ [km s$^{-1}$]')

fig.savefig('outputs/raw.pdf', bbox_inches='tight')
fig.savefig('outputs/raw.png', bbox_inches='tight')
# plt.show()