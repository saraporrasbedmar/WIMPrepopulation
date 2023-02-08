# IMPORTS --------------------------------------------#
import yaml
import time
from class_definition_v2 import Jfact_calculation
from multiprocessing import Pool


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
config_data['SHVF']['RangeMin'] = 1.
config_data['repopulations']['print_freq'] = 25

cosa = Jfact_calculation(data_dict=config_data)

print(time.strftime(" %d-%m-%Y %H:%M:%S", time.gmtime()))
print(config_data['repopulations']['type'])

def try_Vminmin(Vmin):
    init_process_time = time.process_time()
    config_data_process = config_data
    config_data_process['SHVF']['RangeMin'] = Vmin
    config_data_process['repopulations']['id'] = str(Vmin) + ' as Vmin'
    config_data_process['repopulations']['its'] = 1
    print()
    print(config_data_process['repopulations']['id'])
    print(config_data_process['SHVF']['RangeMin'])
    cosa = Jfact_calculation(data_dict=config_data_process)
    for each in config_data_process['repopulations']['type']:
        cosa.repopulation(config_data_process['repopulations']['type'][each])
    end = time.process_time()
    print('%.2f s' % (end - init_process_time))


def one_Vmin(data):
    init_process_time = time.process_time()
    print(data)
    cosa.repopulation(config_data['repopulations']['type'][data])
    end = time.process_time()
    print('%.2f s' % (end - init_process_time))


#for each in config_data['repopulations']['type']:
#    one_Vmin(each)
if __name__ == '__main__':

    p = Pool(4, None)
    # p.map(try_Vminmin, [5., 3., 2., 1., 0.7, 0.6])
    p.map(one_Vmin, config_data['repopulations']['type'])
    p.close()
    p.join()
