import numpy as np
from ruamel.yaml import YAML
from parameterset.parameterset import ParameterSet
import matplotlib.pyplot as plt
import argparse


def main(config_file):
    
    yaml=YAML(typ='safe')   # default, if not specfied, is 'rt' (round-trip)
    with open(config_file, 'rb') as f:
        config = yaml.load(f)
    
    # Create a parameter set from the configuration
    param_set =  ParameterSet(config)

    # Sample the parameter set
    x = param_set.sample(1000)

    print(np.mean(x,axis=0))

    # plot comparisons
    n_var = x.shape[1]
    fig, axs = plt.subplots(n_var, n_var)

    for col in range(n_var):
        for row in range(n_var):
            ax = axs[row, col]
            ax.plot(x[:,col], x[:,row],'.')

    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    # Training parameters
    parser.add_argument('--config', type=str, default='sample_config/test_all.yml',
                        help='Parameter yaml configuration file', required=False)

    args = vars(parser.parse_args())

    print(args)

    main(args['config'])
