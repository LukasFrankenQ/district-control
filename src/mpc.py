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

    def __init__(self, network, total_snapshots, config, horizon, init_values=None, solver_name='gurobi'):
        """
        Initiates class by first creating a dataframe for all objects subject to control

        Parameters
        ----------
        network : pypsa.Network
            instance subject to control
        total_snapshots : pd.Series
            all snapshots that are considered in the whole optimization (the horizon is a 
            slice of this); Defines the data that has to be set up for the prophets
        config : dict of sets of dicts
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
        horizon : int
            number of snapshots considered in a single lopf (length of rolling horizon)
        init_values : dict
            power (for generators, loads, stores and storages) at the start of the mpc
            (assumed to represent p for the former two and energy for the latter two)

        Returns:
        ----------
        -

        """
        self.solver_name = solver_name

        self.config = config
        self.total_snapshots = total_snapshots

        self.with_init = True if init_values is not None else False

        init_time = total_snapshots[0]

        # make dataframe of all components that can create cost
        cost_components = [comp for comp in self.pypsa_components 
                           if comp != 'lines' and comp != 'loads' and comp != 'links']
        cost_names = []
        for comp in cost_components:
                cost_names.extend(getattr(network, comp).index)

        control_names = []
        for comp in self.pypsa_components:
                control_names.extend(getattr(network, comp).index)

        self.costs_t = pd.DataFrame(columns=cost_names)
        self.controls_t = pd.DataFrame(columns=control_names)        


        if init_values is not None:
            self.controls_t.loc[init_time] = init_values
            self.controls_t.loc[init_time] = self.controls_t.loc[init_time].fillna(0)
        else:   
            self.controls_t.loc[init_time] = {}
            self.controls_t.loc[init_time] = self.controls_t.loc[init_time].fillna(0)


        self.addresses = self.get_addresses(network, config)

        self.horizon = horizon
        self.prophets = {}

        # for comp, prophets in config.items():
        for (comp, kind, name, idx) in self.addresses:

            self.prophets[(comp, kind, name, idx)] = \
                    Prophet(total_snapshots, horizon, **config[name][idx])

        self.curr_t = 0
        self.op_cost = 0.


    def mpc_step(self, network_func, snapshots, plot_constraints=False,
                        ax=None):
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
        network_func : function 
            function that builds the network
        snapshots : pd.Series
            snapshots for next lopf
        state : dict
            contains current system state
        plot_constraints : bool
            call plotting fct if True
        """
        
        network = network_func()
        network.set_snapshots(snapshots)
        curr_time = snapshots[0]


        # set up max and min for next lopf if this is not the 
        # first iteration or there are initial conditions
        # if not network.generators_t.p.empty:

        #     for comp in Controller.pypsa_components:
        #         self.set_control(network, comp)
        # for (comp, kind, name, idx) in self.addresses:
        self.set_control(network, curr_time)

        # obtain current predictions
        for address in self.addresses:
            comp, kind, name, idx = address

            # obtain predicted time series
            time_series = self.prophets[address].predict(curr_time)
            
            if ax is not None:
                time_series.plot(ax=ax, linewidth=0.5)

            # put that series as constraint into the model
            getattr(getattr(network, comp), kind)[name].loc[time_series.index] = time_series 

            # extract time steps that are considered during the 


        if plot_constraints: self.plot_constraints(network)

        network.lopf(solver_name=self.solver_name)

        # obtain control for next time step
        fix_time = snapshots[1]

        self.controls_t.loc[fix_time] = {}
        self.costs_t.loc[fix_time] = {}

        for comp in self.pypsa_components:
            self.controls_t.loc[fix_time] = self.controls_t.loc[fix_time].fillna(
                                            self.get_control(network, comp, fix_time))

            # obtain cost of control for storages, generators, links, lines
            if not comp == 'loads' and not comp == 'lines' and not comp == 'links':
        
                self.costs_t.loc[fix_time] = self.costs_t.loc[fix_time].fillna(
                                                self.get_cost(network, comp, fix_time))


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
            list of tuples (component, kind, name, index) such 
            that network.component_t[kind][name] accesses the
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


    def set_control(self, network, time):
        '''
        takes a network that has undergone lopf optimization 
        and sets the (p/e)_(min & max)_pu in the first time step to 
        the p/e value at that value. 
        Serves as constraint for the next time step

        Parameters
        ----------
        network : pypsa.Network
            network under investigation
        time : pd.TimeStamp
            time at which contraints should be set

        Returns
        ----------
        -

        '''

        # generators:
        gens = network.generators.index

        df = self.controls_t[gens].loc[:time]

        lower = df.append(pd.DataFrame(0., index=network.snapshots[1:], columns=gens))
        upper = df.append(pd.DataFrame(1., index=network.snapshots[1:], columns=gens))

        network.generators_t['p_min_pu'] = lower
        network.generators_t['p_max_pu'] = upper 

        # loads:
        loads = network.loads.index

        df = self.controls_t[loads].loc[:time]

        df = df.append(pd.DataFrame(index=network.snapshots[1:], columns=loads))

        network.loads_t['p_set'] = df


        '''
        for (comp, kind, name, idx) in self.addresses:
        
            if 'loads' in comp:
                continue

            # print('current uple:')
            # print(comp, kind, name, idx)
            # print('and time')
            # print(time)
            
            carrier = 'p'

            # print(self.controls_t)

            if comp in ['stores', 'storages']:
                carrier = 'e'            
        
            val = self.controls_t.loc[time][name]
            
            lower = pd.Series(np.r_[val, np.zeros(self.horizon)], index=network.snapshots)
            upper = pd.Series(np.r_[val, np.ones(self.horizon)], index=network.snapshots)

            getattr(network, comp)[carrier+"_min_pu"][name] = lower
            getattr(network, comp)[carrier+"_max_pu"][name] = upper 


            # lower = df.iloc[:1].append(pd.DataFrame(0., index=df.index[1:], columns=df.columns))
            # upper = df.iloc[:1].append(pd.DataFrame(1., index=df.index[1:], columns=df.columns))

            # getattr(network, comp+"_t")[carrier+'_min_pu'] = lower
            # getattr(network, comp+"_t")[carrier+'_max_pu'] = upper 
        '''


    def get_control(self, network, comp, time):
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
        comp : str 
            attribute of pypsa containing time series of power
        time : pd.TimeStamp
            time from which control should be taken

        Returns
        ----------
        control : float
            first entry of the pd.Series found
        '''


        assert hasattr(network, comp+'_t'), f"Network has not been lopf optimized; No time series at {comp}"
        
        time_series = getattr(network, comp+'_t')

        if comp == 'stores':
            control = time_series.e.loc[time]

        elif comp == 'links' or comp == 'lines':
            control = time_series.p0.loc[time]

        else:
            control = time_series.p.loc[time]

        return control


    def get_cost(self, network, comp, time):
        '''
        takes a network that has undergone lopf optimization 
        and an address and returns the marginal cost of usage at relevant snapshots

        Parameters
        ----------
        network : pypsa.Network
            network under investigation
        comp : str 
            attribute of pypsa.Network that contains time series of power/energy
        time : pd.TimeStamp
            time from which cost should be taken

        Returns
        ----------
        cost : float
            marginal cost at currently optimized snapshot
        '''

        # get static cost first, replace if time dependent cost exists
        cost = getattr(network, comp).marginal_cost

        # check for dynamic cost
        dynamic_cost = getattr(network, comp+'_t').marginal_cost
        for name, _ in cost.iteritems():
            if name in dynamic_cost.columns:
                cost[name] = dynamic_cost[name].loc[time]

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


    def plot_constraints(self, network):
        '''
        Creates a plot of all time series in the system

        Parameters
        ----------
        network : pypsa.Network
            we want to plot quantities of this network
        
        Returns
        ----------
        -

        '''

        fig, ax = plt.subplots(1, 1, figsize=(16, 7))

        plot_df = pd.DataFrame(index=network.snapshots, columns=[])

        # gather time dependent quantitites
        for comp in Controller.pypsa_components:

            for kind, df in getattr(network, comp+'_t').items():
                if not df.empty:
                    for col in df.columns:                    
                        plot_df[col+': '+kind] = df[col]

        # add constant values
        of_interest = ['p_min_pu', 'p_max_pu', 'marginal_cost']

        for comp in Controller.pypsa_components:
            if not getattr(network, comp).empty:
                df = getattr(network, comp)

                for col in [entry for entry in of_interest if entry in df.columns]:
                    for name, row in df.iterrows():
                        if not name+': '+col in plot_df.columns:
                            plot_df[name+': '+col] = np.ones(len(plot_df)) * row[col] + \
                                                    np.random.normal(scale=0.01)

        plot_df.plot(ax=ax, linewidth=2., linestyle='-')

        plt.show()





    




def make_small_network():
    '''
    creates 3 bus network:
    house (predicted demand)
    wind farm (predicted supply)
    plant (fixed high price)
    '''

    network = pypsa.Network()

    network.add('Bus', 'bus0')
    network.add('Bus', 'bus1')
    network.add('Bus', 'bus2')

    # network.add('Load', 'house', bus='bus0', p_nom=1, p_set=pd.Series(0.5*np.ones(len(snapshots))))
    
    # pv_cost = pd.Series(0.01 * np.ones(len(snapshots))) 
    # pv_cost.index = snapshots

    network.add('Load', 'house', bus='bus0', p_set=0.5)
    network.add('Generator', 'pv', bus='bus1', p_nom=1.,
                        ramp_limit_up=0.2, ramp_limit_down=0.2, marginal_cost=0.)
    network.add('Generator', 'plant', bus='bus2', p_nom=1., marginal_cost=1.,
                        ramp_limit_up=0.2, ramp_limit_down=0.2)

    network.add('Link', 'pv link', bus0='bus1', bus1='bus0',
                efficiency=1., p_nom=1)
    network.add('Link', 'plant link', bus0='bus2', bus1='bus0',
                efficiency=1., p_nom=1)

    return network



if __name__ == '__main__':
    print(os.getcwd())
    sys.path.append(os.path.join(os.getcwd(), 'src', 'utils'))

    from network_helper import make_simple_lopf
    from network_utils import show_results

    total_snapshots = pd.date_range('2020-01-01', '2020-02-01', freq='30min')
    t_steps = 48
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
                     'noise_scale': 0.02,
                     'data': os.path.join(data_path, 'supply.csv')
                     },
                  ],
            'house': [
                    {
                     'kind': 'p_set', 
                     'mode': 'read',
                     'noise_scale': 0.005, 
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

    init_values = {'pv': 0., 'plant': 0.5, 'house': 0.5}

    mpc = Controller(make_small_network(), total_snapshots, prophets_config, horizon, init_values=init_values, solver_name='glpk')

    fig, axs = plt.subplots(2, 1, figsize=(16,8))

    for _, prophet in mpc.prophets.items():
        prophet.data.plot(ax=axs[0])

    for time in range(t_steps - horizon - 1):
        
        print(f'Conducting step for time {time}.')

        snapshots = total_snapshots[time:time+horizon+1]
        mpc.mpc_step(make_small_network, snapshots, plot_constraints=False, 
                    ax=axs[0])


    mpc.controls_t[init_values].plot(ax=axs[0], linestyle=':')
    mpc.costs_t.plot(ax=axs[1])
    
    for ax, ylabel in zip(axs, ['power flow', 'marginal costs']):
        ax.set_ylabel(ylabel)

    plt.show()


