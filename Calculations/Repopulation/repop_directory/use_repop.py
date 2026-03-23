import os
import time

import numpy as np
from astropy import units as u
from astropy import constants as c

from scipy.optimize import newton
from scipy.integrate import simpson, cumulative_trapezoid, quad, trapezoid

from repop_algorithm import RepopAlgorithm, read_config_file


input_file = read_config_file('input_paper_example.yml')


outtime = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


def save_example_with_callable(Vmax, params):
    return (0.5 * (u.km/ u.s) + 0.01 * Vmax)**params


input_file['repopulations']['params_to_save'][
    'example_with_callable'] = {
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


model = RepopAlgorithm(input_file)

model.run('../outputs/test_2026/test_' + outtime,
          # configuration='mhd_fragile'
          )


### some tests ----------------------------------------
# def xx(mmin, mmax, root):
#     mmin = max(mmin, 1e-20)
#     vmax_array = np.geomspace(mmin, mmax, num=150)
#
#     yy = 10 ** 5.78 * vmax_array ** -3.92
#     print(mmin, mmax, simpson(
#         y=yy * np.log(10) * vmax_array, x=np.log10(vmax_array)))
#
#     return (int(np.rint(simpson(
#         y=yy * np.log(10) * vmax_array, x=np.log10(vmax_array))))
#             - root)
#
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
