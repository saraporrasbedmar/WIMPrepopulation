# IMPORTS --------------------------------------------#
import yaml
import time
from class_definition import Jfact_calculation


# Configuration file reading and data input/output ---------#
def read_config_file(ConfigFile):
    with open(ConfigFile, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml


# Calculations ----------------#

config_data = read_config_file('data_newResSRD.yml')

cosa = Jfact_calculation(data_dict=config_data)

print(time.strftime(" %d-%m-%Y %H:%M:%S", time.gmtime()))

for each in config_data['repopulations']['type']:
    init_process_time = time.process_time()
    cosa.repopulation(config_data['repopulations']['type'][each])
    end = time.process_time()
    print(end - init_process_time)
