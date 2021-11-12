import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('bmh')
import sys
import os


class Prophet:
    """
    class that makes predcitions based on its mode either 
    
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
    num_steps : int
        length of returned time series
    model : tbd
        machine learning model that returns time series

    Methods
    ----------
    predict(**args)
        returns next time series
    
    setup_adversarial(data)
        obtains self.data from which chunks will be returned

    setup_prediction(model)
        sets up saved model for prediction

    """
    def __init__(self, mode, data=None, model=None, num_steps=10, 
                       noise_scale=0.05):
        """
        inits the class
        Also sets up the data or model that should be returned

        Parameters
        ----------
        mode : str
            either 'adversarial' for data-based prophet or 
            'predict' for machine learning model based return
        data : str or pd.Series
            necessary if mode is 'adversarial'
        model : str or tbd model type 
            necessary if mode is 'predict'
        num_steps : int
            length of time series that will be returned in predict call
        noise_scale : float
            std of gaussian noise that is added every time step

        Returns
        ----------
        -
        """

        self.mode = mode
        self.num_steps = num_steps
        self.noise_scale = noise_scale

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
            self.setup_ml(model)
            raise NotImplementedError('implement me!')


    def setup_adversarial(self, data):
        """
        Either from file path or from pd.Series, sets up full available time series 

        Parameters
        ----------
        data : str or Series
            full data from which smaller exempts will be drawn at each 'prediction'
        
        Returns
        ----------
        -

        """
        if isinstance(data, str):
            print('Set up adversarial prophet from file {}!'.format(data)) 
            data = pd.read_csv(data)
            self.data = data[data.columns[-1]]

        elif isinstance(data, pd.Series):
            print('Set up adversarial prophet from pd.Series!')
            self.data = data

        elif isinstance(data, pd.DataFrame):
            print('Set up adversarial prophet from pd.DataFrame!')
            self.data = data[data.columns[-1]]


    def setup_ml(self, model):
        """
        Either from file or from object sets up machine learning model

        Parameters
        ----------
        model : str or tbd

        Returns
        ----------
        -
        """
        raise NotImplementedError('implement me!')

    
    def predict(self, **state):
        """
        Returns prediction based on current state of the system

        Parameters
        ----------
        state : dict
            reflects current system state
        
        Returns
        ----------
        y : pd.Series
            vector of predictions
        """
        if self.mode == 'adversarial':

            assert 't' in state, 'Passed systems state for prediction must contain t. \
                                  Currently contains {}'.format(state)

            t = state['t']

            # create cumulative random noise
            noise = np.random.normal(scale=self.noise_scale, size=self.num_steps)
            noise = np.array([noise[:i+1].sum() for i in range(self.num_steps)])

            return pd.Series(self.data[t:t+self.num_steps].to_numpy() + noise)
            


if __name__ == '__main__':
    # for testing purposes
    # create artificial time series
    df = pd.DataFrame({'a': np.sin(np.linspace(0, 20, 1000))})
    series = pd.Series(np.sin(np.linspace(0, 20, 1000)))
    filename = 'test.csv'
    series.to_csv(filename)

    n = 20
    steps = 3
    # test for pass of filename
    prophet = Prophet('adversarial', data=series, noise_scale=0.05, num_steps=n)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    prophet.data.plot(ax=ax, linewidth=1.)

    for i in range(steps):
        ts = prophet.predict(t=i)
        x = np.linspace(i, i+n, n)
        ax.plot(x, ts, linewidth=0.7)
        # ts.plot(ax=ax, linewidth=0.7)

    ax.set_xlim(-1, steps+n+1)
    plt.show()