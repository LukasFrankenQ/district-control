
import sys
import os
import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
plt.style.use('bmh')


class Controller:
    """
    Helper class to move a pypsa.Network instance in time:
    The main task of the controller is to extract the first time step
    proposed by a lopf optimized network ... (see mpc_step for a better overview)

    
    Attributes
    ----------
    pypsa_components : list of str
        lists components inherent to pypsa networks: generators, loads, links, stores, storages
    control_names : list of str
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

    pypsa_components = ['generators', 'loads', 'links', 'stores']

    def __init__(self, network, controls):
        """
        Initiates class by first creating a dataframe for all objects subject to control

        Parameters
        ----------
        network: pypsa.Network
            instance subject to control
        controls : list of str 
            unique names of network parts that are subject to control. See class attributes for further
            instructions on formatting

        Returns:
        ----------
        -

        """
        self.control_names = controls
        self.controls_t = pd.DataFrame(columns=controls)
        self.costs_t = pd.DataFrame(columns=controls)
        self.addresses = self.get_addresses(network, controls) 
        self.horizon = len(network.snapshots)

        self.curr_t = 0
        self.op_cost = 0.


    def mpc_step(self, network):
        """
        Main method of Controller; Executes the following:
            1) Extracts the next control operation proposed by lopf
            2) TBD: Compares the proposed control to constraints posed
               prediction errors 
            3) roles the lopf optimization forward: by setting up
               initial conditions for the next lopf optimization

        Parameters
        ----------
        network : pypsa.Network
            subject to control
        """

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


    def set_constraint(self, network, address, value):
        """
        sets up upper and lower bounds in the first time step of the 
        next lopf optimization 

        Parameters
        ----------
        network : pypsa.Network
            network under investigation
        address : tuple
            stores (component, name) for easy access to relevant quantities in 
            network's dataframes
        value : 0 <= float <= 1
            value at which that component should start from in next lopf

        Returns
        ----------
        -
        
        """

        component, name = address

        upper = pd.Series([value] + np.ones(self.horizon-1).tolist())
        lower = pd.Series([value] + np.zeros(self.horizon-1).tolist())

        if component in {'generators', 'loads', 'links'}:
            getattr(getattr(network, component+'_t'), 'p_min_pu')[name] = lower
            getattr(getattr(network, component+'_t'), 'p_max_pu')[name] = upper

        elif component == {'stores'}:
            getattr(getattr(network, component+'_t'), 'e_min_pu')[name] = lower
            getattr(getattr(network, component+'_t'), 'e_max_pu')[name] = upper


    def get_addresses(self, network, controls):   
        '''
        creates component-name pairs that make is easy to access the control
        suggested by lopf

        Parameters
        ----------
        network : pypsa.Network
            Network instance to be controlled
        controls : list of str
            names of network components to be controlled

        Returns
        ----------
        addresses : list of tuples
            list of tuples (component, name) such that network.component.name accesses
            the obtained control
        '''

        addresses = []

        for component, name in product(Controller.pypsa_components, controls):

            if name in getattr(network, component).index:
                addresses.append((component, name))

        # make sure all controls are found and no controls 
        assert len(addresses) == len(controls), 'Assignment between controls and addresses not 1-to-1!'

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

            for ts, name in zip([upper, lower], ['upper', 'lower']):
                if not ts.empty:
                    print('{}: {}'.format(name, ts))
                else:                 
                    print('{} is empty'.format(name))


if __name__ == '__main__':
    print(os.getcwd())
    sys.path.append(os.path.join(os.getcwd(), 'src', 'utils'))

    from network_helper import make_simple_lopf
    from network_utils import show_results

    network = make_simple_lopf()
    controls = ['wind', 'gas', 'Boiler', 'HP',
                'Thermal Storage', 'Charge Storage',
                'Discharge Storage']

    mpc = Controller(network, controls)
    mpc.mpc_step(network)

    print('Its a done job init!')

    fig, ax = plt.subplots(1, 1)
    plt.show()