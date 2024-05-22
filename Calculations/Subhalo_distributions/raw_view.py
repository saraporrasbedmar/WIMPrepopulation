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


data_release_dmo = np.loadtxt(
    '../Data_subhalo_simulations/dmo_table.txt', skiprows=3)
data_release_hydro = np.loadtxt(
    '../Data_subhalo_simulations/hydro_table.txt', skiprows=3)
print(np.shape(data_release_dmo), np.shape(data_release_hydro))

fig, ax = plt.subplots(figsize=(10, 8))


plt.scatter(data_release_dmo[:, 1], data_release_dmo[:, 0],
            color='k', s=20, zorder=10, marker='.', alpha=0.7)
plt.scatter(data_release_hydro[:, 1], data_release_hydro[:, 0],
            color='limegreen', s=20, zorder=10, marker='.', alpha=0.7)


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


# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 8))


plt.hist2d(data_release_dmo[:, 1], data_release_dmo[:, 0],
            bins=[np.geomspace(0.9, 120, 60),
                  np.geomspace(0.09, 30, 60)],
            cmap='Greys', zorder=20, alpha=0.8,
           norm='log')
plt.hist2d(data_release_hydro[:, 1], data_release_hydro[:, 0],
            bins=[np.geomspace(0.9, 120, 60),
                  np.geomspace(0.09, 30, 60)],
            cmap='Greens',
            zorder=10, alpha=0.8,
           norm='log')


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

fig.savefig('outputs/raw_hist.pdf', bbox_inches='tight')
fig.savefig('outputs/raw_hist.png', bbox_inches='tight')
plt.show()