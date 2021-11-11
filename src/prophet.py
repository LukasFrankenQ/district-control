import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('bmh')
import sys
import os


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
    



class BaseProphet:
    """
    class that manages predcitions based on its mode either 
    
    a database and returs time series 
    from the database superimposed with noise
    (if mode is 'adversarial')
    or
    a machine learning model and returns predictions 
    made by the model
    (if mode is 'predict')

    even though two modes exist, from the outside, the class has the 
    exact same functionalities independent of the mode
    

    Attributes
    ----------
    mode : str
        'adversarial' for data based time series with noise
        'predict' for machine learning based time series

    data : pd.Series
        full time series available from which a small exempt is
        taken at every call

    noise_scale : float 
        (relevant only for mode = 'adversarial')
        determines how much fluctuation the returned time series is subject to
        noise is sampled from a gaussian with mean=0 and std=noise_scale 
        noise increases linearly with index of time series
        i.e. Let x_t be the t-th entry of time series of quantity x.
             entry x_0 will be free of noise, entry x_t will be subject
             to gaussian noise with standard deviation t*noise_scale

    steps : int
        length of returned time series

    model : tbd
        machine learning model that returns time series

    Methods
    ----------
    predict(**args)
        returns next time series
    
    setup(mode, data=None, model=None)
        based on mode sets up data for mode='adversarial' and model for 
        mode='predict'

    """
    def __init__(self, mode, data=None, model=None):
        """
        Parameters
        ----------
        mode : str
            either 'adversarial' for data-based prophet or 
            'predict' for machine learning model based return
        data : str or pd.Series
            necessary if mode is 'adversarial'
        model : str or tbd model type 
            necessary if mode is 'predict'


        Returns
        ----------
        -

        inits the class
        Also sets up the data or model that should be returned
        """

        self.mode = mode

        if mode == 'adversarial': 
            assert data is not None, f'for chosen mode adversarial, \
                                       kwarg data has to provide either path or pd.Series'
        if mode == 'predict': 
            assert model is not None, f'for chosen mode predict, \
                                       kwarg model has to provide either path or model type (tbd)'
        assert mode == 'predict' or mode == 'adversarial', f'Please choose mode \
                                        aversarial or predict instead of {mode}.'

        if mode == 'adversarial':
            self.setup_adversarial(data) 
             

        elif mode == 'predict':
            self.setup_predict(model)
            raise NotImplementedError('implement me!')


    def setup_adversarial(self, data):
        """
        Either from file path or from pd.Series, sets up full 


        """



if __name__ == '__main__':

    # for testing purposes
    # create artificial time series
    df = pd.DataFrame({'a': np.sin(np.linspace(0, 20, 1000))})
    series = pd.Series(np.sin(np.linspace(0, 20, 1000)))
    filename = 'test.csv'
    series.to_csv(filename)

    # test for pass of filename
    prophet = BaseProphet('adversarialll', data='filename')




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