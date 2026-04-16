import os
import time

import numpy as np
from astropy import units as u
from astropy import constants as c
import matplotlib.pyplot as plt

from scipy.optimize import newton
from scipy.integrate import simpson, cumulative_trapezoid, quad, trapezoid

from repop_algorithm import RepopAlgorithm, read_config_file


input_file = read_config_file('input_paper_example.yml')


outtime = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


def save_example_with_callable(Vmax, params):
    return (0.5 * (u.km/ u.s) + 0.01 * Vmax)**params


input_file['repopulations']['params_to_save']['ex_with_callable'] = {
    'formula': save_example_with_callable,
    'params': 10, 'variables': 'Vmax'}


# Careful with this, because this technically works, but the SRD
# does NOT depend on Vmax. Rather, Vmax is taken as the variable
# necessary to calculate the probability distribution function of the
# SRD. Similar to the SHVF inputs. The rest of the functions do
# introduce the variables.
def srd_example(Vmax, params):
    return (0.5 + 0.01 * Vmax**params)*(Vmax > 150)

# input_file['configurations']['dmo_resilient']['SRD'] = {
#     'formula': srd_example, 'params': 1e2}

def aaa(Vmax, params):
    return 1e5 + Vmax.value + params

# TODO: esto falla con draco
# input_file['configurations']['dmo_resilient']['Cv'] = {
#     'formula': aaa, 'params': 12, 'variables': 'Vmax'}



def rho_VL(D_GC):

    a = 0.8220569767611305
    b = 8.410128564347877
    try:
        r0 = 1036.2376643526568 * u.kpc
        return (D_GC/r0)**a*np.exp(-b*(D_GC - r0)/r0) #cosmic
    except:
        r0 = 1036.2376643526568
        return (D_GC/r0)**a*np.exp(-b*(D_GC - r0)/r0) #cosmic


input_file['configurations']['dmo_ale']['SRD'] = {
    'formula': rho_VL, 'variables': ['D_GC'],
    'params': None }

raaange = [0.5, 0.6, 0.72, 0.864, 1.0368, 1.24416, 1.4929919999999999,
           1.7915903999999998, 2.1499084799999997, 2.5798901759999997,
           3.0958682111999996, 3.7150418534399994, 4., 400.]

model = RepopAlgorithm(input_file)
model.configuration = 'dmo_fragile'

plt.figure()
xx = np.geomspace(0.1, 10) * u.km / u.s
yy = 8226.1 * xx ** 3.72
plt.plot(xx, yy)

cv_mean = model.calculate_formula(
    xx,
    model.input_dict['configurations'][
        model.configuration]['Cv']['formula'],
    {'c0': model.input_dict['configurations'][
        model.configuration]['Cv']['params']['c0'],
     'sigma_scatter': 0.}
)
c200 = model.C200_from_Cv(cv_mean)
# print((Vmax_max * u.km / u.s / (
#             self.input_dict['cosmo_constants']['H_0']
#             * np.sqrt(2. * cv_mean))).to(u.kpc))
R_max=(xx / (
            model.input_dict['cosmo_constants']['H_0']
            * np.sqrt(2. * cv_mean))).to(u.kpc)
def ff(c):
    return np.log(1. + c) - c / (1. + c)

M = (xx ** 2 * R_max / (4.297e-06 * u.Unit('kpc * km2 / (Msun * s2)'))
                * ff(c200)/ ff(2.163)).to(u.Msun)

plt.loglog(xx, M)
plt.xscale('log')

# plt.show()

for i in range(len(raaange) - 1):
    print(raaange[i], raaange[i + 1])

    print(model.SHVF_integral(
        Vmax_min=raaange[i],
        Vmax_max=raaange[i + 1],
        force_no_fraction=False
    ))
    print()
model.run('../outputs/test_2026/test_' + outtime,
          configuration='dmo_fragile'
          )
plt.show()

from scipy.integrate import simpson, cumtrapz

def xx(mmin, mmax, root):
    mmin = max(mmin, 1e-20)
    vmax_array = np.geomspace(mmin, mmax, num=150)

    yy = 10 ** 5.78 * vmax_array ** -3.92
    print(mmin, mmax, simpson(
        y=yy * np.log(10) * vmax_array, x=np.log10(vmax_array)))

    return (int(np.rint(simpson(
        y=yy * np.log(10) * vmax_array, x=np.log10(vmax_array))))
            - root)

# aa = newton(xx, 1., args=[120., int(2e5)])
# print(aa)
# print(xx(aa, 120., 0))


# def SHVF_integraltraaa(Vmax_min, Vmax_max,
#                   formula=None, params=None):
#     if formula is None:
#         formula = input_file['configurations']['mhd_fragile']['SHVF']['formula']
#     if params is None:
#         params = input_file['configurations'][
#             'mhd_fragile']['SHVF']['params']
#
#     vmax_array = np.geomspace(Vmax_min, Vmax_max, num=2000)
#
#     yy = model.calculate_formula(
#         vmax_array, formula=formula, params=params)
#
#     return int(np.rint(trapezoid(
#         y=yy * np.log(10) * vmax_array, x=np.log10(vmax_array))))
# def SHVF_integralsimson(Vmax_min, Vmax_max,
#                   formula=None, params=None):
#     if formula is None:
#         formula = input_file['configurations']['mhd_fragile']['SHVF']['formula']
#     if params is None:
#         params = input_file['configurations'][
#             'mhd_fragile']['SHVF']['params']
#
#     vmax_array = np.geomspace(Vmax_min, Vmax_max, num=2000)
#
#     yy = model.calculate_formula(
#         vmax_array, formula=formula, params=params)
#
#     return int(np.rint(simpson(
#         y=yy * np.log(10) * vmax_array, x=np.log10(vmax_array))))
# def SHVF_integralquad(Vmax_min, Vmax_max,
#                            formula=None, params=None):
#         if formula is None:
#             formula = input_file['configurations'][
#                 'mhd_fragile']['SHVF']['formula']
#         if params is None:
#             params = input_file['configurations'][
#                 'mhd_fragile']['SHVF']['params']
#
#         return ((quad(
#             model.calculate_formula,
#             a=Vmax_min, b=Vmax_max,
#             args=(formula, params))[0]))
#
# print((SHVF_integraltraaa(0.1, 120.)))
# print((SHVF_integralsimson(0.1, 120.)))
# print((SHVF_integralquad(0.1, 120.)))

# model.run('../outputs/test_2026/test_' + outtime,
#           # configuration='mhd_fragile'
#           )
