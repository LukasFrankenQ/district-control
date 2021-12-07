import sys
import os
import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
plt.style.use('bmh')

from prophet import Prophet


class Controller:
    """
    Helper class to move a pypsa.Network instance in time:
    The main task of the controller is to extract the first time step
    proposed by a lopf optimized network ... (see mpc_step for a better overview)

    
    Attributes
    ----------
    pypsa_components : list of str
        lists components inherent to pypsa networks: generators, loads, links, stores, storages
    control_config : dict of str of str
        list of network components that are subject to MPC
        have to be unique names among all network components, i.e. a generator can not 
        have the same name as a load
        e.g. controls = ['wind turbine', 'solar panel', 'stes', 'house 1 heat', ...]
    horizon : int
        number of snapshots considered in every lopf step
    addresses: list of tuples
        stores addresses to access proposed control from network easily
    backend : TBD
        physics-based estimator that connects power flow to target quantities
        such as temperature
    curr_t : int
        current time step of control; used to assign received control suggestions
        to the controls_t dataframe
    state : dict 
        dictionary to describe current system state. NOTE: This is not a 
        system state x as referred to in the control theory and does
        not abide by the mathematics associated with that formalism
    op_cost : float
        stores the operational cost of the accumulated actual control
    steps : int
        stores the total number of steps conducted
    u : np.ndarray 
        array of current control inputs extracted from the first timestep of 
        received lopf run. Used to inter


    Methods
    ----------
    mpc_step(network)
        central method of class: roles a network that was lopf-optimized
        forward in time: Extracts the control nearest in the future and 
        sets them up in the network as initial condition for the next time step.
        at the same time, the method stores the extracted control in a dataframe, 
        stores the associated cost and number of steps conducted
    get_addresses(network, controls)
        used to set up tuples that are used to extract control suggestions from 
        the lopf optimized network
    get_control(network, address)
        extracts control suggestions from an address-tuple
    """

    pypsa_components = ['generators', 'loads', 'links', 'stores', 'lines']

    def __init__(self, network, total_snapshots, config, horizon):
        """
        Initiates class by first creating a dataframe for all objects subject to control

        Parameters
        ----------
        network: pypsa.Network
            instance subject to control

        config : TBD

        horizon : int
            length of rolling horizon

        Returns:
        ----------
        -

        """

        self.config = config
        self.total_snapshots = total_snapshots
        self.control_names = list(config)
        self.controls_t = pd.DataFrame(columns=list(config))
        self.costs_t = pd.DataFrame(columns=list(config))
        
        self.addresses = self.get_addresses(network, config)

        self.horizon = horizon
        self.prophets = {}

        # for comp, prophets in config.items():
        for (comp, kind, name, idx) in self.addresses:

            self.prophets[(comp, kind, name, idx)] = \
                    Prophet(total_snapshots, horizon, **config[name][idx])

        self.curr_t = 0
        self.op_cost = 0.


    def mpc_step(self, network, snapshots, state):
        """
        Main method of Controller; 

        Roles time forward and sets up constraints for the next lopf 
        These constraints are defined by two aspects: 

            First, the results of the previous lopf, making the outcome of the 
            previous optimization the initial conditions of the next timestep

            Secondly, we call prophet and obtain the most recent data and prediction
            on renewable generation, demand and market prices 
        
            Executes the following:

            0) Obtains data from prophets

            1) Extracts the next control operation proposed by lopf
            2) Sets up the resulting control as initial conditions for next optimization
            
            3) Obtains new predictions from prophet
            4) Compares constraints posed by control and by predictions and sets up
               overall constrains that conform with both

            5) Error management stage

        Parameters
        ----------
        network : pypsa.Network
            subject to control
        snapshots : pd.Series
            snapshots for next lopf
        state : dict
            contains current system state
        """

        network.set_snapshots(snapshots)

        # obtain current predictions
        for address in self.addresses:
            comp, kind, name, idx = address

            # obtain predicted time series
            time_series = self.prophets[address].predict(**state)

            # put that series as constraint into the model
            getattr(getattr(network, comp), kind)[name] = time_series 




        self.show_current_ts(network)

        # conduct lopf 
        print('snapshots')
        print(network.snapshots)
        network.lopf(solver_name='gurobi')








        '''
        # obtain next mpc step from lopf network
        curr_control = {address[1]: self.get_control(network, address) for address in self.addresses}
        curr_costs = {address[1]: self.get_cost(network, address) for address in self.addresses}
        
        # store control and marginal cost at current snapshot
        self.controls_t = self.controls_t.append(curr_control, ignore_index=True)
        self.costs_t = self.costs_t.append(curr_costs, ignore_index=True)


        # set constraints for next lopf

        print('Before setting constraints:')
        self.show_controllables(network)
        for (key, item), address in zip(curr_control.items(), self.addresses):

            self.set_constraint(network, address, item)
        
        print('After setting constraints:')
        self.show_controllables(network)
        '''


    def show_current_ts(self, network):
        '''
        Helper function to show the current time series in the network

        Parameters
        ----------
        network : pypsa.Network
            network under investigation

        Returns
        ----------
        -

        '''
        have_shown = set()
        for address in self.addresses:
            comp, _, _, _ = address
            if not comp in have_shown:

                have_shown.add(comp)
                print('{}:'.format(comp))
                print(getattr(network, comp)) 
                print('------------')


    def get_addresses(self, network, config):   
        '''
        creates component-name pairs that make is easy to access the control
        suggested by lopf
        Also sets values as the desired constant value if prophet

        Parameters
        ----------
        network : pypsa.Network
            Network instance to be controlled
        controls : dict of list of dicts
            see tbd for documentation on this object

        Returns
        ----------
        addresses : list of tuples
            list of tuples (component, quantity, name, mode) such 
            that network.component_t[quantity][name] accesses the
            desired time series.
        '''

        addresses = []

        for component, (name, prophets) in product(Controller.pypsa_components, config.items()):

            for idx, prophet in enumerate(prophets): 

                if prophet['mode'] == 'read' or prophet['mode'] == 'predict':
                    if name in getattr(network, component).index:
                        addresses.append((component+'_t', prophet['kind'], name, idx))


                elif prophet['mode'] == 'fix':
                    if name in getattr(network, component).index:

                        comp_df = getattr(network, component)
                        comp_df.at[name, prophet['kind']] = prophet['value']
                        setattr(network, component, comp_df)

        return addresses


    def get_control(self, network, address):
        '''
        takes a network that has undergone lopf optimization 
        and an address and returns the proposed time series associated with that
        address

        note that time step 0 of lopf was entirely defined by previous boundary
        conditions: The time step of interest has index 1

        Parameters
        ----------
        network : pypsa.Network
            network under investigation
        address : tuple
            pair of component and name that refers to a controlled object in the network

        Returns
        ----------
        control : float
            first entry of the pd.Series found
        '''

        component, name = address
        assert hasattr(network, component+'_t'), "Network has not been lopf optimized over multiple snapshots"
        
        time_series = getattr(network, component+'_t')

        if component == 'stores':
            control = time_series.e.at[1, name]

        elif component == 'links':
            control = time_series.p0.at[1, name]

        else:
            control = time_series.p.at[1, name]

        return control


    def get_cost(self, network, address):
        '''
        takes a network that has undergone lopf optimization 
        and an address and returns the marginal cost of usage at relevant snapshots

        Parameters
        ----------
        network : pypsa.Network
            network under investigation
        address : tuple
            pair of component and name that refers to a controlled object in the network

        Returns
        ----------
        cost : float
            marginal cost at currently optimized snapshot
        '''

        component, name = address

        # check for time-depentend marginal cost
        if name in getattr(getattr(network, component+"_t"), 'marginal_cost').columns:
            marginal_cost = getattr(getattr(network, component+'_t'), 'marginal_cost')
            cost = marginal_cost.at[1, name]

        else:
            marginal_cost = getattr(getattr(network, component), 'marginal_cost')
            cost = marginal_cost[name]

        return cost


    def show_controllables(self, network):
        '''
        Prints time series, i.e. controls of current network

        Parameters
        ----------
        network : pypsa.Network
            network under investigation

        Returns
        ----------
        -
        '''

        components = set([address[0] for address in self.addresses])

        for component in components:
            
            print('For {}:'.format(component))
            if component == 'stores':
                lower = getattr(network, component+'_t').e_min_pu 
                upper = getattr(network, component+'_t').e_max_pu 

            else:
                lower = getattr(network, component+'_t').p_min_pu 
                upper = getattr(network, component+'_t').p_max_pu 

            for name in ['upper', 'lower']:
                if not eval(name).empty:
                    print('{}: {}'.format(name, eval(name)))
                else:                 
                    print('{} is empty'.format(name))


def make_small_network():
    '''
    creates 3 bus network:
    house (predicted demand)
    wind farm (predicted supply)
    plant (fixed high price)
    '''

    network = pypsa.Network()
    network.add('Bus', 'bus pv', carrier='elec')
    network.add('Bus', 'bus plant', carrier='elec')
    network.add('Bus', 'bus house', carrier='elec')

    network.add('Load', 'house', bus='bus house', carrier='elec')
    network.add('Generator', 'pv', bus='bus pv', carrier='elec')
    network.add('Generator', 'plant', bus='bus plant', carrier='elec')

    network.add('Link', 'pv link', bus0='bus pv', bus1='bus house', carrier='elec',
                efficiency=1.)
    network.add('Link', 'plant link', bus0='bus plant', bus1='bus house', carrier='elec',
                efficiency=1.)

    return network



if __name__ == '__main__':
    print(os.getcwd())
    sys.path.append(os.path.join(os.getcwd(), 'src', 'utils'))

    from network_helper import make_simple_lopf
    from network_utils import show_results

    network = make_small_network()


    # network.lopf(solver_name='gurobi')

    # print('[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[')    


    total_snapshots = pd.date_range('2020-01-01', '2020-02-01', freq='30min')
    t_steps = 25
    horizon = 13
    total_snapshots = total_snapshots[:t_steps]


    print(os.getcwd())
    data_path = os.path.join(os.getcwd(), 'data', 'dummy')

    '''
    information on components of network and the time series subject to (predicted)
    constraints. Data format:
    dict of sets of dict

    This should be their content:
    outer dict:
        each key refers to a component in the network (key must be unique and refer to name given in pypsa network)
    set:
        set of dicts, each addressing contraints on a quantity during the lopf optimization
    inner dict:
        mandatory keys:
            'kind': quantity during the lopf optimization (p_max_pu, marginal_cost, p_min_pu, p_set etc...)
            'mode': if 'fix' set to constant value during lopf
                    if 'read' quantity is time series and read from data with optionally superimposed with noise 
                    if 'predict' quantity is time series and predicted by ml model
            'data': if mode is 'read': pd.Series or pd.DataFrame or path to csv with data to be read 
                    if mode is 'predict': pd.Series or pd.DataFrame or path to csv with features for model
            'model': (for mode predict only) model object        TBD
            'value': (for mode fix only) constant value to be set
        
        optional keys:
            'noise_scale': standard deviation of gaussian noise induced per step (default=0.05)



    '''

    pd.set_option('display.max_columns', None)
    prophets_config = {
            'pv': [
                    {
                     'kind': 'p_max_pu', 
                     'mode': 'read', 
                     'data': os.path.join(data_path, 'supply.csv')
                     },
                  ],
            'house': [
                    {
                     'kind': 'p_set', 
                     'mode': 'read', 
                     'data': os.path.join(data_path, 'demand.csv')
                    },
                  ],
            'plant': [
                    {
                     'kind': 'marginal_cost',
                     'mode': 'fix',
                     'value': 1. 
                    }
                  ]
            }


    mpc = Controller(network, total_snapshots, prophets_config, horizon)

    for t in range(t_steps - horizon):

        snapshots = total_snapshots[t:t+horizon]

        state = {'t': t}
        # snapshots = total_snapshots[step:step+horizon]
        mpc.mpc_step(network, snapshots, state)




        break

    '''

    network.set_snapshots(snapshots)



    controls = ['wind', 'gas', 'Boiler', 'HP',
                'Thermal Storage', 'Charge Storage',
                'Discharge Storage']

    mpc = Controller(network, controls)
    mpc.mpc_step(network)

    print('Its a done job init!')

    fig, ax = plt.subplots(1, 1)
    plt.show()
    '''