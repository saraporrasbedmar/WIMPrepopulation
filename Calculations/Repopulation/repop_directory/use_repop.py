import os
import time

import numpy
from astropy import units as u
from astropy import constants as c

from repop_algorithm import RepopAlgorithm, read_config_file


input_file = read_config_file('input_paper_example.yml')


outtime = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


def save_example_with_callable(Vmax, params):
    return (0.5 * (u.km/ u.s) + 0.01 * Vmax)**params


input_file['repopulations']['columns_to_save']['ex_with_callable'] = {
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


input_file['configurations']['dmo_resilient']['Cv'] = {
    'formula': aaa, 'params': 12, 'variables': 'Vmax'}


model = RepopAlgorithm(input_file)
model.run('../outputs/test_2026/test_' + outtime,
          configuration='mhd_fragile'
          )
