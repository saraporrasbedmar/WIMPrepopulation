import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colorbar as colorbarr
import matplotlib.patches as mpatches
import scipy.optimize as sciopt
import scipy.stats as stats
from iminuit import Minuit
from iminuit.cost import LeastSquares

from scipy.integrate import simpson

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

dgc_kpc = np.geomspace(1e-3, 220, num=500)
r_s = 2.
rho_0 = 9.04e6

number_params = 3

plt.subplots(number_params, 1, figsize=(8, 16))
x_min = 1e-4
x_max = 1e3
# -----------------------------------------------------------------
plt.subplot(number_params, 1, 1)
plt.loglog(dgc_kpc, nfw_rho_profile(xx=dgc_kpc, rs=r_s, rho0=rho_0))

plt.axvline(r_s, ls=':', c='k')

plt.ylabel('NFW density')

plt.xlim(x_min, x_max)

for i in [0.1, 1., 10, 50, r_s/np.tan(0.15*np.pi/180.)]:
    plt.axvline(i * np.tan(0.15*np.pi/180.))
    plt.text(s=i, x=i * np.tan(0.15*np.pi/180.)*1.1, y=1e5)
print('how much bigger rs', 1/np.tan(0.15*np.pi/180.))
# -----------------------------------------------------------------
plt.subplot(number_params, 1, 2)
plt.loglog(dgc_kpc, mass_inside(xx=dgc_kpc, rs=r_s, rho0=rho_0))
plt.ylabel('Mass inside r')
plt.axvline(r_s, ls=':', c='k')

plt.xlim(x_min, x_max)
# -----------------------------------------------------------------
plt.subplot(number_params, 1, 3)
aaa = vcirc(xx=dgc_kpc, rs=r_s, rho0=rho_0,
                          G=6.61e-11)
plt.loglog(dgc_kpc, aaa)
plt.ylabel('Vcirc')


plt.axvline(r_s, ls=':', c='k')

array_max = np.argmax(aaa)
vmax = np.max(aaa)
rmax = dgc_kpc[np.argmax(aaa)]
plt.axvline(rmax)
plt.axhline(vmax)
print(rmax, vmax)
print(rmax/r_s)
plt.xlim(x_min, x_max)
#
# plt.subplot(number_params, 1, 1)
# plt.loglog(dgc_kpc, nfw_profile(xx=dgc_kpc, rs=21., rho0=rho_0))
# plt.ylabel('NFW density')


plt.show()