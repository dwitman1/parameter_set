import numpy as np
from scipy.stats import truncnorm

registry = {}

class ParameterInterface(object):

    def __init__(self, config):
        self.config = config
    
    def __init_subclass__(cls):
        registry[cls.__name__] = cls
    
    def sample(self, N):
        raise Exception('Sample method must be implemented')


class Uniform(ParameterInterface):

    def __init__(self, config):
        
        # The name of the class
        self.__name__ = 'Uniform'

        # Set the low/hig parameters
        self.low = config['low']
        self.high = config['high']
    
    def sample(self, N):
        return np.random.uniform(self.low, self.high, size=(N,))

class Normal(ParameterInterface):

    def __init__(self, config):
        
        # The name of the class
        self.__name__ = 'Normal'
        
        # Set the mean and standard deviation
        self.mean = config['mean']
        self.std = config['std']
    
    def sample(self, N):
        return np.random.normal(self.mean, self.std, size=(N,))

class TruncNormal(ParameterInterface):

    def __init__(self, config):

        # The name of the class
        self.__name__ = 'TruncNormal'

        # The parameters
        self.mean = config['mean']
        self.std = config['std']
        self.low = config['low']
        self.high = config['high']
    
    def sample(self, N):
        return truncnorm.rvs((self.low - self.mean) / self.std,
                             (self.high - self.mean) / self.std,
                             loc=self.mean, scale=self.std, size=(N,))

class Discrete(ParameterInterface):

    def __init__(self, config):

        # The name of the class
        self.__name__ = 'Discrete'

        # The parameters
        self.samples = config['samples']
    
    def sample(self, N):
        return np.random.choice(self.samples, size=(N,))

class WeightedDiscrete(ParameterInterface):

    def __init__(self, config):

        # The name of the class
        self.__name__ = 'WeightedDiscrete'

        # The parameters
        self.samples = config['samples']
        self.p = config['p']
    
    def sample(self, N):
        return np.random.choice(self.samples, size=(N,), p=self.p)

class Categorical(ParameterInterface):

    def __init__(self, config):

        # The name of the class
        self.__name__ = 'Categorical'

        # The parameters
        self.categories = config['categories']
        self.n_class = len(self.categories)
    
    def sample(self, N):
        return np.eye(self.n_class)[np.random.choice(self.n_class, N)]

class WeightedCategorical(ParameterInterface):

    def __init__(self, config):

        # The name of the class
        self.__name__ = 'WeightedCategorical'

        # The parameters
        self.categories = config['categories']
        self.p = config['p']
        self.n_class = len(self.categories)
    
    def sample(self, N):
        return np.eye(self.n_class)[np.random.choice(self.n_class, N, p=self.p)]