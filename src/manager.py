import numpy as np
from geopy.geocoders import Nominatim
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('bmh')
import geopy
import sys
import os

from prophet import Prophet


class Manager:
    """
    Class to manage time series predictions and setting up the 
    respective predictors (referred to as prophets).
    Has prophets as attributes which are responsible to return time
    series of a time series that is relevant

    Attributes
    ----------
    time : pd.Series
        pandas date_range of time frame of interest
    lon : float
        longitude of place of interest
    lat : float
        latitude of place of interest
    configs : dict of dicts
        has keys:
            'name' which can have item:
                'pv' for prediction of solar power generation
                'wind' for prediction of wind power generation
                'demand' for prediction of energy demand
                'market' for prediction of market prices
            'mode'
                'simulate'  returns time series from database but 
                            superimposed with noise (for testing purposes)
                            noise in increases in magnitude in later parts of
                            the returned series
                'predict' returns time series from machine learning model
                          based on real data in current time step
            'source' file to source of data or model
    'prophets' : dict
        keeps a dictionary of the prophets and their respective tasks


    Methods
    ----------
    setup_wind_prophet(config)
        sets up predictor of wind generation
        based on config (please find method for more details)

    setup_pv_prophet(config)
        sets up predictor of pv generation
        based on config (please find method for more details)

    setup_demand_prophet(config)
        sets up predictor of demand generation
        based on config (please find method for more details)
    
    setup_market_prophet(config)
        sets up predictor of demand generation
        based on config (please find method for more details)
    """

    def __init__(self, when_where, configs):
        """
        sets prophets that are needed to satisfy the
        responsibilities

        Parameters
        ----------
        when_where : dict
            holds spatio-temporal information of area of interest
        configs : dict of dicts 
            dictionary of config dictionaries, each referring a required prophet
            each config should contain:
                'name': [can be 'wind', 'pv', 'house' or ...]
                'mode': [can be 'predict' or 'simulate' ...]
                'source': path to source file. if mode is 'predict' this
                          will be understood as path to model. if mode is
                          'simulate' this will be understood as path to data'
        """
        self.configs = configs 

        assert ['start', 'end', 'time_step', 'city'] == list(when_where.keys()), \
                'when_where dict must have keys start, end, time_step and city \
                not {}. \n Also: please use fractions of a full hour as \
                    format for time_step.'.format(when_where.keys())

        self.time = pd.date_range(when_where['start'], 
                                  when_where['end'], 
                                  freq=str(int(60*when_where['time_step']))+'min')
        print('Received snapshots:\n {}'.format(self.time))
        
        city = Nominatim(user_agent='dummy').geocode(when_where['city'])
        print('Working with {}'.format(city.address))
        print('With coordinates {}, {}'.format(city.latitude, city.longitude))
        self.lon = city.longitude
        self.lat = city.latitude

        for key, config in configs.items():
            assert 'name' in list(config.keys()) and \
                    'mode' in list(config.keys()) and \
                    'source' in list(config.keys()), 'config {} must contain keys \
                    name, mode and source, but contains {}'.format(key, config.keys())

            setup_method = getattr(self, 'setup_'+config['name']+'_prophet')
            self.configs[key]['prophet'] = setup_method(config)






    def setup_wind_prophet(self, model_path=None, edinburgh=True):
        '''
        Based on self.mode prepares a model that 


        '''

        import atlite
        pass


        

if __name__ == "__main__":
    when_where = {'start': '2018-01-01',
                  'end': '2019-01-01',
                  'time_step': 0.5,
                  'city': 'edinburgh'} 

    print(list(when_where.keys()))
    responsibilities = ['wind', 'pv', 'demand']
    manager = Manager(when_where, responsibilities, )







'''
import atlite
import geopandas as gpd
import logging

cutout = atlite.Cutout(path='western_europe-2011-01.nc',
                       module='era5',                           
                       x=slice(-13.6913, 1.7712),
                       y=slice(49.9096, 60.8479),
                       time='2011-01'
                       )
cutout.prepare()

'''