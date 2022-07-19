import numpy as np
from parameterset.parameter import ParameterInterface
from parameterset.utils import get_parameters_from_config
from typing import Union, dict, List

class ParameterSet(object):

    def __init__(self, parameter_list: Union[List[ParameterInterface], dict]):
        
        if type(parameter_list) is dict:
            self.parameter_list = get_parameters_from_config(parameter_list)
        else:
            self.parameter_list = parameter_list
        
        # The categorical parameter classes
        # These are noted so that the column names can have Name$Category label
        self.categorical_parameter_classes = ['Categorical',
                                              'WeightedCategorical']

        # Define the column names
        self.column_names = []
        for parameter in self.parameter_list:
            if parameter.__name__ in self.categorical_parameter_classes:
                for category in parameter.categories:
                    self.column_names.append(parameter.name + '$' + category)
            else:
                self.column_names.append(parameter.name)

    def sample(self, N):
        
        # Initialize the parameter sample container
        sample_container = []
        for parameter in self.parameter_list:
            # Sample the individual parameter
            sample_container.append(parameter.sample(N))
        
        return np.concatenate(sample_container, axis=1)


