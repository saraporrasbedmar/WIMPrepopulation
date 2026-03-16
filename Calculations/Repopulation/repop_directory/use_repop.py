import os
import time

from repop_algorithm import RepopAlgorithm, read_config_file


input_file = read_config_file('../input_files/input_paper2024.yml')


outtime = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

def JJcosas(Vmax, params):
    return (0.5 + 0.01 * Vmax**params)*(Vmax > 150)


input_file['repopulations']['columns_to_save']['JJcosas'] = {
    'formula': JJcosas, 'params': 10, 'variables': 'Vmax'}


# Careful with this, because this technically works, but the SRD
# does NOT depend on Vmax. Rather, Vmax is taken as the variable
# necessary to calculate the probability distribution function of the
# SRD. Similar to the SHVF inputs. The rest of the functions do
# introduce the variables.
input_file['SRD']['dmo']['resilient'] = {
    'formula': JJcosas, 'params': 1e2}

# print(input_file['repopulations']['columns_to_save'])

model = RepopAlgorithm('mhd', 'resilient', input_file)
model.run('../outputs/test_2026/test_' + outtime)