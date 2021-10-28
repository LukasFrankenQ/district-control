import os
import sys
import pypsa
import pandas as pd

sys.path.append(os.path.join(os.getcwd(), 'src', 'utils'))
from network_utils import show_results
from mpc import Controller
# from prophet import Prophet




def main():

    controls = {'generators': ['wind', 'gas_market'],
                'loads': ['house 1 heat', 'house 2 elec'],
                'links': ['generator', 'boiler']}
    controller = Controller(controls)






if __name__ == '__main__':
    main()