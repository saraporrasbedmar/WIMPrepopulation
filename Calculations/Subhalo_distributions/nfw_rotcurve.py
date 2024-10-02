import numpy as np
import matplotlib.pyplot as plt

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
plt.rc('legend', fontsize=22)
plt.rc('figure', titlesize=all_size)
plt.rc('xtick', top=True, direction='in')
plt.rc('ytick', right=True, direction='in')
plt.rc('xtick.major', size=10, width=2, top=True, pad=10)
plt.rc('ytick.major', size=10, width=2, right=True, pad=10)
plt.rc('xtick.minor', size=7, width=1.5)
plt.rc('ytick.minor', size=7, width=1.5)


plt.figure(figsize=(10, 8))

xx = np.linspace(0, 1., num=600)

V200 = 100
cc = 20


yy = V200 * np.sqrt(1/xx
                    * (np.log(1+cc*xx)-cc*xx/(1+cc*xx))
                    /(np.log(1+cc)-cc/(1+cc)))
plt.plot(xx, yy, lw=2)

plt.xlabel(r'radius/r$_\mathrm{vir}$')
plt.ylabel(r'V$_\mathrm{c}$ [km/s]')

ymax = np.nanmax(yy)
plt.axvline(xx[np.argmax(yy[1:])], c='k', ls='--', lw=2)
plt.axhline(ymax, c='k', ls='dotted', lw=2)

plt.text(x=0.3, y=144.5, s=r'V$_\mathrm{max}$')
plt.text(x=0.12, y=130, s=r'R$_\mathrm{max}$')

plt.xlim(0, 0.5)
plt.ylim(120, 150)

plt.show()