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

config_data = read_config_file('input_files/data_newResSRD.yml')
config_data['SHVF']['RangeMin'] = 6.
config_data['repopulations']['print_freq'] = 25
config_data['repopulations']['its'] = 100
cosa = Jfact_calculation(data_dict=config_data)

print(time.strftime(" %d-%m-%Y %H:%M:%S", time.gmtime()))
print(config_data['repopulations']['type'])


def one_Vmin(data):
    init_process_time = time.process_time()
    print(data)
    cosa.repopulation(config_data['repopulations']['type'][data],
                      type_loop='one_by_one')  # 'all_at_once',
                                               # 'one_by_one',
                                               # 'bin_by_bin'
    end = time.process_time()
    print('%.2f s' % (end - init_process_time))


for each in config_data['repopulations']['type']:
    one_Vmin(each)
# if __name__ == '__main__':
#
#     p = Pool(4, None)
#     p.map(one_Vmin, config_data['repopulations']['type'])
#     p.close()
#     p.join()
