from parameterset.parameter import *

def get_parameters_from_config(config):

    # The list of parameters as classes
    parameter_list = []

    # Loop over all the parameters in the configuration
    for parameter_config in config['parameters']:

        if parameter_config['type'] not in registry.keys():
            raise Exception('Unrecognized parameter type: '+str(parameter_config['type']))

        parameter_list.append(registry[parameter_config['type']](parameter_config))

    return parameter_list