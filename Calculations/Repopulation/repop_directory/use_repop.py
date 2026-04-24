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


# def save_example_with_callable(Vmax, params):
#     return (0.5 * (u.km/ u.s) + 0.01 * Vmax)**params


# input_file['repopulations']['params_to_save']['ex_with_callable'] = {
#     'formula': save_example_with_callable,
#     'params': 10, 'variables': 'Vmax'}


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

def functionfff(r):
    aa = model.M_encapsulated(
        radius=r*u.kpc,
        rho_0=model.input_dict['host']['rho_0'],
        r_s=model.input_dict['host']['r_s'],
        density_profile=model.input_dict['host']['density_profile'])
    return ((aa/(4./3.*np.pi*(r*u.kpc)**3.)
       ).to(u.Unit('Msun / kpc3')).value - 200*135.73)


rvir = newton(functionfff, x0=235)
mass = model.M_encapsulated(
        radius=rvir*u.kpc,
        rho_0=model.input_dict['host']['rho_0'],
        r_s=model.input_dict['host']['r_s'],
        density_profile=model.input_dict['host']['density_profile'])
print('rvir', rvir)
print('mass', mass)
# print(aa)
# print((aa/(4./3.*np.pi*model.input_dict['host']['R_vir']**3.)
#        ).to(u.Unit('Msun / kpc3')))

print(model.J_general(
            D_Earth=200 * u.kpc,
            density_profile='NFW',
            calculate_from='mass_Cmass', integrate_up_to=0.03,
            Mass=mass, Cmass=rvir*u.kpc/model.input_dict['host']['r_s'],
            ))

print(model.J_general(
            D_Earth=200 * u.kpc,
            density_profile='NFW',
            calculate_from='rho0_rS', integrate_up_to=0.03,
            rho_0=model.input_dict['host']['rho_0'],
            r_s=model.input_dict['host']['r_s'],
            ))
rmax = 2.16257584237*model.input_dict['host']['r_s']
Vmax=np.sqrt(model.M_encapsulated(
        radius=rmax,
        rho_0=model.input_dict['host']['rho_0'],
        r_s=model.input_dict['host']['r_s'],
        density_profile=model.input_dict['host']['density_profile'])
             * model.input_dict['cosmo_constants']['G']/rmax)
cv = (2. * (Vmax/rmax/model.input_dict['cosmo_constants']['H_0'])**2.).to(1)
print(Vmax, cv)
print(model.J_general(
            D_Earth=200 * u.kpc,
            density_profile='NFW',
            calculate_from='Vmax_Cv', integrate_up_to=0.03,
    Vmax=Vmax,
    Cv=cv
            ))

model.run('../outputs/test_2026/test_' + outtime,
          configuration='dmo_fragile'
          )
model.configuration = 'mhd_resilient'


print(model.RmaxoverrS())

plt.figure()
xx = np.geomspace(0.1, 10) * u.km / u.s
yy = 8226.1 * xx ** 3.72
plt.loglog(xx, yy)

print(model.configuration)

cv_mean = model.calculate_formula(
    xx,
    model.input_dict['configurations'][
        model.configuration]['Cv']['formula'],
    {'c0': model.input_dict['configurations'][
        model.configuration]['Cv']['params']['c0'],
     'sigma_scatter': 0.}
)
print(cv_mean[0])
c200 = model.C200_from_Cv(cv_mean)
R_max=(xx / (
            model.input_dict['cosmo_constants']['H_0']
            * np.sqrt(2. * cv_mean))).to(u.kpc)
def ff(c):
    return np.log(1. + c) - c / (1. + c)

M = (xx ** 2 * R_max / (4.297e-06 * u.Unit('kpc * km2 / (Msun * s2)'))
                * model.ff(c200)/ ff(2.163)).to(u.Msun)

plt.loglog(xx, M)
plt.xscale('log')

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
