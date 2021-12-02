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
    (if mode is 'simulate')
    or
    a machine learning model and returns predictions 
    made by the model
    (if mode is 'predict')

    even though two modes exist, from the outside, the class has the 
    exact same functionalities
    
    Attributes
    ----------
    mode : str
        'simulate' for data based time series with noise
        'predict' for machine learning based time series
    data : pd.Series
        full time series available from which a small exempt is
        taken at every call
    noise_scale : float 
        (relevant only for mode == 'simulate')
        determines how much fluctuation the returned time series is subject to
        noise is sampled from a gaussian with mean=0 and std=noise_scale 
        noise increases linearly with index of time series
        i.e. Let x_t be the t-th entry of time series of quantity x.
             entry x_0 will be free of noise, entry x_t will be subject
             to gaussian noise with standard deviation t*noise_scale
    horizon : int
        length of returned time series
    model : tbd
        machine learning model that returns time series

    Methods
    ----------
    predict(**args)
        returns next time series
    
    setup_simulation(data)
        obtains self.data from which chunks will be returned

    setup_prediction(model)
        sets up saved model for prediction

    """
    def __init__(self, mode, snapshots, data=None, model=None, horizon=10, 
                       noise_scale=0.05):
        """
        inits the class
        Also sets up the data or model that should be returned

        Parameters
        ----------
        mode : str
            either 'simulate' for data-based prophet or 
            'predict' for machine learning model based return
        snapshots : pd.DatatimeIndex
            points in time for which predictions (simulations) will be made
        data : str or pd.Series
            required if mode is 'simulate'
        model : str or tbd model type 
            necessary if mode is 'predict'
        horizon : int
            length of time series that will be returned in predict call
        noise_scale : float
            std of gaussian noise that is added every time step

        Returns
        ----------
        -
        """

        self.mode = mode
        self.snapshots = snapshots
        self.horizon = horizon 
        self.noise_scale = noise_scale

        if mode == 'simulate': 
            assert data is not None, f'for chosen mode simulate, \
                                       kwarg data has to provide either path, pd.Series or pd.DataFrame'
        if mode == 'predict': 
            assert model is not None, f'for chosen mode predict, \
                                       kwarg model has to provide either path or model type (tbd)'
        assert mode == 'predict' or mode == 'simulate', f'Please choose mode \
                                        aversarial or predict instead of {mode}.'

        if mode == 'simulate':
            self.setup_simulation(data) 

        elif mode == 'predict':
            self.setup_ml(model)
            raise NotImplementedError('implement me!')


    def setup_simulation(self, data):
        """
        Either from file path or from pd.Series, sets up full available time series 

        Parameters
        ----------
        data : str or Series
            full data from which smaller exempts will be drawn at each call
        
        Returns
        ----------
        -

        """
        if isinstance(data, str):
            print('Set up simulating prophet from file {}!'.format(data)) 
            data = pd.read_csv(data, parse_dates=True, index_col=0)
            self.data = data[data.columns[-1]]

        elif isinstance(data, pd.Series):
            print('Set up simulating prophet from pd.Series!')
            self.data = data

        elif isinstance(data, pd.DataFrame):
            print('Set up simulating prophet from pd.DataFrame!')
            self.data = data[data.columns[-1]]

        assert isinstance(data.index, pd.DatetimeIndex) is True, f'\
                    Index of received data {self.data} of is not of type pd.DatetimeIndex!'

        print('Working with data: ', self.data)


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
        if self.mode == 'simulate':

            assert 't' in state, 'Passed systems state for prediction must contain t. \
                                  Currently contains {}'.format(state)

            t = state['t']

            # create cumulative random noise
            noise = np.random.normal(scale=self.noise_scale, size=self.horizon)
            noise = pd.Series([noise[:i+1].sum() for i in range(self.horizon)])

            if isinstance(t, int) is not True:
                t = self.data.index.get_loc(t)
            
            snippet = self.data.iloc[t:t+self.horizon]
            noise.index = self.data.index[t:t+horizon]

            print(f'bare cutout {snippet}')
            print(f'bare noise {noise}')
            print(f'together {snippet + noise}')

            return snippet + noise


if __name__ == '__main__':
    # for testing purposes
    # create artificial time series
    data = os.path.join(os.getcwd(), 'data', 'dummy', 'demand.csv')

    snapshots = pd.date_range('2020-01-01', '2020-01-02', freq='30min')

    horizon = 10
    # test for pass of filename
    prophet = Prophet('simulate', snapshots, data=data, noise_scale=0.05, horizon=horizon)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    prophet.data.plot(ax=ax, linewidth=1.)


    for i in range(10, 10+4):
        prediction = prophet.predict(t=i)
        prediction.plot(ax=ax, linewidth=0.7)
        # ts.plot(ax=ax, linewidth=0.7)

    # ax.set_xlim(-1, horizon+n+1)
    plt.show()