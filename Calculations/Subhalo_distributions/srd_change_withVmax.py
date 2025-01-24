import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

import scipy.optimize as opt
from scipy import integrate
from scipy.interpolate import UnivariateSpline

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

def funct_ale(Dgc, a, b):
    return b * np.exp(a / Dgc)


plt.figure()

xx = np.linspace(0., 1., num=200)
num = 5

aaa = np.geomspace(1e-3, 0.27, num=num)
aaa[0] = 0
print(aaa)

for ni, ii in enumerate(aaa):
    plt.plot(xx, funct_ale(xx, -ii, 1.),
             color=cm.CMRmap(ni / num), label='%.3f ' %ii)
plt.legend()


aaa = np.geomspace(1e-3, 0.151, num=num)
aaa[0] = 0
print(aaa)
for ni, ii in enumerate(aaa):
    plt.plot(xx, funct_ale(xx, -ii, 1.),
             color=cm.CMRmap(ni / num), label='%.3f ' %ii,
             ls='--')
plt.legend()
# plt.yscale('log')

plt.axvline(8.5/220)
plt.show()
