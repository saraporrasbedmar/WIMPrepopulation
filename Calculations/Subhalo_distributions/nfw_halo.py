import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from scipy.optimize import root

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
plt.rc('ytick.major', size=10, width=2, right=True, pad=5)
plt.rc('xtick.minor', size=7, width=1.5, top=False)
plt.rc('ytick.minor', size=7, width=1.5)

def ff(x):
    return np.log(1. + x) - x/(1. + x)

def nfw_rho_profile(xx, rs, rho0):
    return rho0 / (xx/rs) / (1. + xx/rs)**2.

def mass_inside(xx, rs, rho0):
    return (4.*np.pi*rho0 * rs**3.
            * (np.log(1. + xx/rs) - xx/(rs + xx)))

def vcirc(xx, rs, rho0, G):
    return np.sqrt(G * mass_inside(xx, rs, rho0)/xx)

def funct_mk501_inside(xx, N0, gamma1, gamma2, Ebreak, fi):
    return (N0
            * xx ** gamma1
            * (1. + (xx / Ebreak) ** fi) ** (-(gamma1 - gamma2) / fi)
            )

r_s = 20.
rho_0 = 9.04e6

def find_rvir(xx):
    return (mass_inside(xx, r_s, rho_0)/(4./3.*np.pi*xx**3.)
            - 200*135.73)

r_200_numerical = root(find_rvir, x0=200.)
print(r_200_numerical)
r_200_numerical = r_200_numerical['x'][0]
print(mass_inside(r_200_numerical, rs=r_s, rho0=rho_0)*1e-12)

dgc_kpc = np.geomspace(1e-2, 400, num=1000)


number_params = 3

fig, (ax1, ax2, ax3) = plt.subplots(number_params, 1, figsize=(12, 8))

plt.subplots_adjust(wspace=0, hspace=0)
x_min = 1e-3
x_max = 1.3
# -----------------------------------------------------------------

plt.subplot(number_params, 1, 1)
plt.loglog(dgc_kpc / r_200_numerical,
           nfw_rho_profile(xx=dgc_kpc, rs=r_s, rho0=rho_0),
           lw=3, c='lime',
           path_effects=[pe.Stroke(linewidth=4, foreground='k'),
                         pe.Normal()]
           )

plt.axvline(r_s / r_200_numerical, ls='--', c='grey', lw=3)
plt.text(x=r_s / r_200_numerical, y=3e9,
         s=r'$r_\mathrm{s}$', ha='center')

plt.axvline(r_200_numerical / r_200_numerical, ls='-', c='k', lw=3)
plt.axvspan(1, 2, color=(235/255, 235/255, 235/255))

plt.text(x=r_200_numerical / r_200_numerical, y=3e9,
         s=r'$R_\mathrm{200}$', ha='center')

plt.text(x=0.0033, y=1e7, s=r'$\log_{10} \rho \propto -1$', fontsize=18)
plt.text(x=0.25, y=8e5, s=r'$\log_{10} \rho \propto -3$', fontsize=18)

plt.ylabel(r'$\rho$ ($\mathrm{M}_\odot / \mathrm{kpc}^3$)')

plt.xlim(x_min, x_max)
plt.tick_params('x', labelbottom=False)

xx = np.geomspace(1e-3, 0.17*r_200_numerical)
yy = rho_0 / (xx/r_s)
plt.plot(xx / r_200_numerical, yy, c='k', ls=':', lw=3)

xx = np.geomspace(0.045*r_200_numerical, 300)
yy = rho_0 / (xx/r_s)**3.
plt.plot(xx / r_200_numerical, yy, c='k', ls=':', lw=3)


plt.ylim(1e3, 1e9)

plt.yticks(10**np.array([3, 5, 7, 9]),
               # labels=(r'10$^3$', '',
               #         r'10$^7$', ''),
           fontsize=22
           )
print('how much bigger rs', 1/np.tan(0.15*np.pi/180.))
# -----------------------------------------------------------------
plt.subplot(number_params, 1, 2)
plt.loglog(dgc_kpc / r_200_numerical,
           mass_inside(xx=dgc_kpc, rs=r_s, rho0=rho_0),
           lw=3,c='lime',
           path_effects=[pe.Stroke(linewidth=4, foreground='k'),
                         pe.Normal()])
plt.ylabel('M(<r) ($\mathrm{M}_\odot$)')

plt.axvline(r_s/ r_200_numerical, ls='--', c='grey', lw=3)
# plt.text(x=14, y=1e9, s=r'$r_\mathrm{s}$')
plt.axvspan(1, 2, color=(235/255, 235/255, 235/255))

plt.axvline(r_200_numerical/ r_200_numerical, ls='-', c='k', lw=3)

plt.axhline(mass_inside(xx=r_200_numerical, rs=r_s, rho0=rho_0),
            ls='-.', c='fuchsia', lw=3,
           path_effects=[pe.Stroke(linewidth=4, foreground='k'),
                         pe.Normal()])
plt.text(x=4e-3, y=2e11, s=r'$M_\mathrm{200}$', color='fuchsia')

plt.tick_params('x', labelbottom=False)

plt.yticks(10**np.array([8, 9, 10, 11, 12]),
               labels=('', r'10$^{9}$', '',
                       r'10$^{11}$', ''),
           fontsize=22
           )

plt.xlim(x_min, x_max)
plt.ylim(bottom=5e7)
# -----------------------------------------------------------------
plt.subplot(number_params, 1, 3)
aaa = vcirc(xx=dgc_kpc, rs=r_s, rho0=rho_0, G=4.297e-6)
plt.plot(dgc_kpc / r_200_numerical,
           aaa, lw=3, c='lime',
           path_effects=[pe.Stroke(linewidth=4, foreground='k'),
                         pe.Normal()])
plt.ylabel(r'$V_\mathrm{circ}$ (km$\,/\,$s)')

plt.xlabel(r'r / $R_\mathrm{200}$')

plt.axvline(r_s/ r_200_numerical, ls='--', c='grey', lw=3)
# plt.text(x=14, y=35., s=r'$r_\mathrm{s}$')

plt.axvline(r_200_numerical/ r_200_numerical, ls='-', c='k', lw=3)
plt.axvspan(1, 2, color=(235/255, 235/255, 235/255))

array_max = np.argmax(aaa)
vmax = np.max(aaa)
rmax = dgc_kpc[np.argmax(aaa)]

plt.axvline(rmax/ r_200_numerical, ls='--', c='orange', lw=3,
           path_effects=[pe.Stroke(linewidth=4, foreground='k'),
                         pe.Normal()])
plt.text(x=0.2, y=80., s=r'$R_\mathrm{max}$', color='darkorange')

plt.axhline(vmax, ls='--', c='coral', lw=3,
           path_effects=[pe.Stroke(linewidth=4, foreground='k'),
                         pe.Normal()])
plt.text(x=4e-3, y=140, s=r'$V_\mathrm{max}$', color='red')

print(rmax, vmax)
print(rmax/r_s)
plt.xlim(x_min, x_max)
plt.ylim(30, 250)

plt.yscale('log')
plt.xscale('log')

# plt.yticks((30, 40, 50, 60, 70, 80, 90, 100, 200),
#                labels=('30', '', '50', '', '', '', '', '100', '200'),
#            fontsize=22
#            )
# plt.yticks((40, 60, 70, 80, 90),
#                # labels=('30', '', '50', '', '', '', '', '100', '200'),
#            fontsize=22, which='minor'
#            )

ax3.set_xticks([1e-3, 1e-2, 0.1, 1])
ax3.set_xticklabels([r'10$^{-3}$', r'10$^{-2}$', '0.1', '1'])

ax3.set_yticks([30, 100, 200])
ax3.set_yticklabels(['30', '100', '200'],
                    fontsize=22)
import matplotlib.ticker as ticker
# Set minor ticks
ax3.yaxis.set_minor_locator(ticker.FixedLocator([40, 60, 70, 80, 90]))
ax3.tick_params(axis='y', which='minor', labelleft=False)


# Optionally, customize minor tick appearance
# ax3.tick_params(axis='y', which='minor', direction='in', length=4, width=1)

plt.yticks(fontsize=22)



def find_rvir(xx):
    return (mass_inside(xx, r_s, rho_0)/(4./3.*np.pi*xx**3.)
            - 100*135.73)

aaa = root(find_rvir, x0=200.)
print(aaa)
print(aaa['x'][0])
print(mass_inside(aaa['x'][0], rs=20., rho0=9.04e6)*1e-12)



def find_rvir(xx):
    return (3./200.*9.04e6/135.73 - xx**3./ff(xx))

aaa = root(find_rvir, x0=10.)
print(aaa)
print(aaa['x'][0])
print(aaa['x'][0]*20.)
print(mass_inside(aaa['x'][0]*20., rs=20., rho0=9.04e6)*1e-12)
print(mass_inside(220., rs=20., rho0=9.04e6)*1e-12)

print(200./3. * (220./12.)**3./ff(220./12.) * 135.73)

plt.savefig('outputs/nfw_profile.png',
            bbox_inches='tight')
plt.savefig('outputs/nfw_profile.pdf',
            bbox_inches='tight')
plt.show()