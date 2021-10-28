import pypsa
import numpy as np
import inspect
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("bmh")


def show_results(*args, network=None):
    '''
    shows overview of network time-series that resulted from
    a lopf optimization 
    ''' 

    assert network is not None, "Please pass network as kwarg!"

    print(f"Objective: {network.objective}")

    components = ['generators', 'stores', 'storages', 'loads', 'links', 'lines']

    print("\n Getting optimal capacities")
    for component in components:
        print_capacities(network, component)
    
    print("\n Plotting production and dispatch curves")
    for component in components:
        plot_time_series(network, component)



    # hacky method to get argument names
    frame = inspect.currentframe()
    frame = inspect.getouterframes(frame)[1]
    string = inspect.getframeinfo(frame[0]).code_context[0].strip()
    hold_args = string[string.find('(')+1:-1].split(',')

    names = []
    for i in hold_args:
        if i.find("=") != -1:
            names.append(i.split('=')[1].strip())

        else:
            names.append(i)

    print('For system placed in conditions: ')

    for arg, name in zip(args, names):
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.set_title(name)
        arg.plot(ax=ax)
        plt.show()


def print_capacities(network, component):
    if hasattr(network, component):
        comp = getattr(network, component)
        if hasattr(comp, 'p_nom_opt'): print(f"Capacity of {component}: \n {comp.p_nom_opt}")
    
    else: print(f"No {component} in the system!")


def plot_time_series(network, component):

    if hasattr(network, component+'_t'):
        comp = getattr(network, component+'_t')

        for unit in ['p', 'e']:

            if hasattr(comp, unit):
                fig, ax = plt.subplots(1, 1, figsize=(12, 6))
                quantity = getattr(comp, unit)
                print(f'{unit} {component}:')
                quantity.plot(ax=ax, label=component+" "+unit)
                plt.show()