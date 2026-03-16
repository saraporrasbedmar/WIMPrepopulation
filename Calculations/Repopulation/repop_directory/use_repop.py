import os
import time

from repop_algorithm import repop_algorithm, read_config_file


input_file = read_config_file('../input_files/input_paper2024.yml')


outtime = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

def JJcosas(Vmax, params):
    return Vmax**2. + params

input_file['repopulations']['columns_to_save']['JJcosas'] = {
    'formula': JJcosas, 'params': 10, 'variables': 'Vmax'}

# print(input_file['repopulations']['columns_to_save'])

model = repop_algorithm('dmo', 'resilient', input_file)
model.run('../outputs/test_2026/test_' + outtime)