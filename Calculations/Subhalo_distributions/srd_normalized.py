import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.interpolate import UnivariateSpline
from numpy.random import random


def N_subs_resilient(DistGC, args):
    return DistGC ** args[0] * 10 ** args[1]


def N_subs_fragile(DistGC, args):
    return (((DistGC / args[0]) ** args[1]
             * np.exp(-args[2] * (DistGC - args[0]) / args[0])))


def Nr_Ntot(DistGC,
            sim_type, res_string,
            srd_last_sub):
    """
    Proportion of subhalo numbers at a certain distance of the GC.

    :param DistGC: float or array-like [kpc]
        Distance from the center of the subhalo to the
        Galactic Center (GC).

    :return: float or array-like
        Number of subhalos at a certain distance of the GC.
    """
    if res_string == 'resilient':
        if sim_type == 'dmo':
            return N_subs_resilient(DistGC,
                                    [0.54343928, -2.17672138])
        if sim_type == 'hydro':
            return N_subs_resilient(DistGC,
                                    [0.97254648, -3.08584275])

    else:
        if sim_type == 'dmo':
            return (10 ** N_subs_fragile(DistGC,
                                         [1011.38716, 0.4037927, 2.35522213])
                    * (DistGC >= srd_last_sub))
        if sim_type == 'hydro':
            return (10 ** N_subs_fragile(DistGC,
                                         [666.49179, 0.75291017, 2.90546523])
                    * (DistGC >= srd_last_sub))


plt.figure()
xxx = np.logspace(-7, np.log10(220), num=10000)
y_dmo_res = Nr_Ntot(xxx, 'dmo', 'resilient', 13.1)
y_dmo_frag = Nr_Ntot(xxx, 'dmo', 'aa', 13.1)
y_hydro_res = Nr_Ntot(xxx, 'hydro', 'resilient', 6.61)
y_hydro_frag = Nr_Ntot(xxx, 'hydro', 'aa', 6.61)


def cumulative(y, x):
    cumul = [simpson(y[:i], x=x[:i]) for i in range(1, len(x))]
    cumul /= cumul[-1]

    x_mean = (x[1:] + x[:-1])/2.

    x_min = ((np.array(cumul) - 1e-5) < 0).argmin() - 1

    return UnivariateSpline(cumul[x_min:], x_mean[x_min:],
                            s=0, k=1, ext=0)


x_muchos = random(1000000)
# print(cumulative(y_dmo_res, xxx)(0))
# print(cumulative(y_dmo_frag, xxx)(0))
# print(cumulative(y_hydro_res, xxx)(0))
# print(cumulative(y_hydro_frag, xxx)(0))
plt.hist(cumulative(y_dmo_res, xxx)(x_muchos),
         bins=40, color='k', alpha=0.5, density=True,
         histtype = 'step', fill = None)
plt.hist(cumulative(y_dmo_frag, xxx)(x_muchos),
         bins=40, color='gray', alpha=0.5, density=True,
         histtype = 'step', fill = None)
plt.hist(cumulative(y_hydro_res, xxx)(x_muchos),
         bins=40, color='g', alpha=0.5, density=True,
         histtype = 'step', fill = None)
plt.hist(cumulative(y_hydro_frag, xxx)(x_muchos),
         bins=40, color='limegreen', alpha=0.5, density=True,
         histtype = 'step', fill = None)
# plt.xscale('log')
plt.yscale('log')
# plt.xlim(6, 300)




plt.show()
