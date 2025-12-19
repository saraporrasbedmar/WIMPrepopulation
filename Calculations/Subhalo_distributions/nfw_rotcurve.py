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


def ff(c):
    return np.log(1. + c) - c / (1. + c)

def cv(c200):
    return 200. * (c200/2.163)**3. / ff(c200) * ff(2.163)

def vcirc_v(xx_kpc, vmax, rmax):
    return vmax * np.sqrt(rmax / ff(2.163)
                          * ff(xx_kpc * 2.163 / rmax) / xx_kpc)

def rmax_from_cv(vmax, cv, h_0):
    return vmax / h_0 / np.sqrt(cv/2.)

'''
plt.figure(figsize=(10, 8))
xx = np.linspace(0, 1., num=1000)

V200 = 100
cc = 20
r_s = 20.
R_vir = 220.
H_0 = 66.7 * 1e-3
print(cv(cc))

yy = V200 * np.sqrt(1/xx
                    * (np.log(1+cc*xx)-cc*xx/(1+cc*xx))
                    /(np.log(1+cc)-cc/(1+cc)))
plt.plot(xx, yy, lw=2)



plt.loglog(xx, vcirc_v(xx*R_vir, 144.,
                       rmax_from_cv(144., 3.5e4, h_0=H_0)),
           ls='--')

plt.xlabel(r'radius/r$_\mathrm{vir}$')
plt.ylabel(r'V$_\mathrm{c}$ [km/s]')

ymax = np.nanmax(yy)
plt.axvline(xx[np.argmax(yy[1:])], c='k', ls='--', lw=2)
plt.axhline(ymax, c='k', ls='dotted', lw=2)

plt.text(x=0.3, y=144.5, s=r'V$_\mathrm{max}$')
plt.text(x=0.12, y=137, s=r'R$_\mathrm{max}$')

plt.xlim(0, 0.5)
plt.ylim(120, 150)
plt.savefig('curve1.png', bbox_inches='tight')
plt.savefig('curve1.pdf', bbox_inches='tight')


cc = 15

print(cv(cc))

yy = V200 * np.sqrt(1/xx
                    * (np.log(1+cc*xx)-cc*xx/(1+cc*xx))
                    /(np.log(1+cc)-cc/(1+cc)))
plt.plot(xx, yy, lw=2)

plt.text(x=0.23, y=137, s=r'$c_\mathrm{V}=3.5 \times 10^4$', c='b')
plt.text(x=0.115, y=128, s=r'$c_\mathrm{V}=1.7 \times 10^4$', c='orange')


plt.savefig('curve2.png', bbox_inches='tight')
plt.savefig('curve2.pdf', bbox_inches='tight')
'''
# ----------------------------------------------------------------------

plt.figure(figsize=(10, 8))
xx = np.linspace(0, 1., num=5000)
V200 = 140
c_v = 3.5e4
R_vir = 220.
H_0 = 66.7 * 1e-3

r_max = rmax_from_cv(V200, c_v, h_0=H_0)
yy = vcirc_v(xx*R_vir, V200, r_max)
plt.plot(xx, yy, ls='-', c='b', lw=2)

plt.xlabel(r'r/$R_\mathrm{vir}$')
plt.ylabel(r'V$_\mathrm{c}$ [km/s]')

plt.axhline(V200, c='k', ls='dotted', lw=1.5)
plt.text(x=0.35, y=141, s=r'V$_\mathrm{max}$')

plt.axvline(r_max/R_vir, c='blue', ls='--', lw=1.5)
# plt.text(x=0.078, y=145, s=r'R$_\mathrm{max}$', color='blue')

plt.xlim(0, 0.4)
plt.ylim(100, 150)

plt.savefig('curve11.png', bbox_inches='tight')
plt.savefig('curve11.pdf', bbox_inches='tight')


plt.text(x=0.265, y=123, s=r'$c_\mathrm{V, 1'
                           r'}=3.5 \times 10^4$', c='b')

rmax2 = rmax_from_cv(V200, c_v/10., h_0=H_0)

yy = vcirc_v(xx*R_vir, V200, rmax2)
plt.plot(xx, yy, c='orange', lw=2)
plt.text(x=0.235, y=134, s=r'$c_\mathrm{V, 2}=3.5 \times 10^3$', c='orange')

plt.axvline(rmax2/R_vir, c='orange', ls='--', lw=1.5)
plt.text(x=0.235, y=145, s=r'R$_\mathrm{max, 2}$', color='orange')

plt.text(x=0.078, y=145, s=r'R$_\mathrm{max, 1}$', color='blue')


plt.savefig('curve22.png', bbox_inches='tight')
plt.savefig('curve22.pdf', bbox_inches='tight')

plt.show()