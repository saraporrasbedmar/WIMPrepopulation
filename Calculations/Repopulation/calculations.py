#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 18:34:06 2023

@author: saraporras
"""

# IMPORTS --------------------------------------------#
import yaml
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

config_data = read_config_file('data.yml')

cosa = Jfact_calculation(data_dict=config_data)

for each in config_data['repopulations']['type']:
    cosa.repopulation(config_data['repopulations']['type'][each])
