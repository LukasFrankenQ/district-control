import pypsa
import numpy as np
import pandas as pd


def make_simple_lopf():

    days = 10
    network = setup_network(days)
    availability, prices = get_suppliers(network)
    Lh = 0.9 # load
    omega_demand = 2*np.pi / 24.

    demand = Lh * 0.5 * (1 + np.sin(omega_demand * network.snapshots.to_series()))

    # 1) Set up only boiler and heat load
    network = setup_network(days)

    # add buses
    buses = ['Gas Bus', "Load Bus", 'Wind Bus', "Store Bus"]
    carriers = ['Gas', 'Heat', 'AC', 'Heat']

    for bus, carrier in zip(buses, carriers):
        network.add("Bus", bus, carrier=carrier)

    # set up load
    network.add('Load', "load", bus='Load Bus', p_set=demand)

    # add wind power with low cost but only partial availability
    network.add("Generator",
                "wind",
                bus="Wind Bus",
                p_nom_extendable=True,
                p_nom_max=1.,
                marginal_cost=prices['wind'],
                p_max_pu=availability['wind'],
                capital_cost=1.
                )

    # set up gas
    network.add("Generator", 
                'gas', 
                bus='Gas Bus',
                p_nom_extendable=True,
                capital_cost=1.,
                marginal_cost=prices['gas'],
                p_max_pu=availability['gas'],
                )

    # add boiler to the system as link
    network.add("Link",
                'Boiler',
                bus0='Gas Bus', 
                bus1="Load Bus",
                p_nom_extendable=True,
                capital_cost=0.5,
                efficiency=1.
                )

    # add heat pump to the system
    network.add("Link", 
                "HP",
                bus0='Wind Bus',
                bus1='Load Bus',
                p_nom_extendable=True,
                capital_cost=0.5,
                efficiency=1.
                )

    # adding thermal storage
    network.add("Bus",
                "Storage Bus", 
                carrier="Heat")      

    network.add("Store",
                "Thermal Storage",
                bus='Storage Bus',
                capital_cost=0.5,
                e_nom_extendable=True,
                e_nom_max=1)

    # link to store heat using a heat pump and wind power
    network.add("Link",
                "Charge Storage",
                bus0='Wind Bus',
                bus1='Storage Bus',
                capital_cost=0,
                p_nom_extendable=True,
                efficiency=1.)

    network.add("Link",
                "Discharge Storage",
                bus0='Storage Bus',
                bus1='Load Bus',
                capital_cost=0,
                marginal_cost=0.1 * np.ones(len(network.snapshots.to_series())),
                p_nom_extendable=True,
                efficiency=1.)

    network.lopf(solver_name='gurobi')

    return network




def get_suppliers(network, plot=True):

    # set system parameters
    Lh = 0.9 # load
    Cw = 0.5 # wind capacity
    gas_price = 1.
    omega_wind = 2*np.pi / 48.

    time_steps = len(network.snapshots)

    availability = pd.DataFrame({
                        "wind": Cw * 0.5 * (1 + np.cos(omega_wind * network.snapshots.to_series())),
                        "gas": Lh * pd.Series(np.ones(time_steps))
                        })

    prices = pd.DataFrame({
                        "wind": pd.Series(np.zeros(time_steps)),
                        "gas": gas_price * pd.Series(np.ones(time_steps))
                        })
    
    return availability, prices


def setup_network(days):
    network = pypsa.Network()
    network.set_snapshots(np.arange(0, days*24))
    return network

