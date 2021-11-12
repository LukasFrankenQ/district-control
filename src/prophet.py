import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('bmh')
import sys
import os

from baseprophet import BaseProphet


class Prophet:
    """
    Class to manage and return time series in accordance with current system
    conditions. Can

    Attributes
    ----------
    responsibilities : list of str
        list can contain
            'generation' for prediction of renewable generation
        
            'demand' for prediction of energy demand

            'market' for prediction of market prices
    'mode' : str
        determines source of returned time series
            'adversarial' returns time series from database but 
                          superimposed with noise (for testing purposes)
                          noise in increases in magnitude in later parts of
                          the returned series
            'predict' returns time series from machine learning model
                      based on real data in current time step

    Methods
    ----------
    setup_renewable_prophet(config)
        sets up predictor of renewable generation
        based on config (please find method for more details)

    setup_demand_prophet(config)
        sets up predictor of demand generation
        based on config (please find method for more details)
    
    setup_market_prophet(config)
        sets up predictor of demand generation
        based on config (please find method for more details)
    """

    def __init__(self, responsibilities, mode='adversarial'):
        """
        sets prophets that are needed to satisfy the
        responsibilities

        Parameters
        ----------
        responsibilities : list of str
            list of required models
        mode : str
            'adversarial'->returns synthetically noised time series
            'predict'->returns time series from ML model

        """
        self.responsibilities = responsibilities
        self.mode = mode


    # @mode.setter
    def mode(self, mode):
        '''
        Changes current mode of prophet
        
        Parameters
        ----------
        mode : str
            See class documentation

        Returns
        ----------
        -
        '''
        self.mode = mode


    def setup_renewable_prophet(self, model_path=None, edinburgh=True):
        '''
        Based on self.mode prepares a model that 


        '''

        import atlite
        pass
    










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