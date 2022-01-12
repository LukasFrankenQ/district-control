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
    (if mode is 'read')
    or
    a machine learning model and returns predictions 
    made by the model
    (if mode is 'predict')

    even though two modes exist, from the outside, the class has the 
    exact same functionalities
    
    Attributes
    ----------
    snapshots : pd.DatatimeIndex
        points in time for which predictions (simulations) will be made
    horizon : int
        length of returned time series
    mode : str
        'read' for data based time series with noise
        'predict' for machine learning based time series
    noise_scale : float 
        (relevant only for mode == 'read')
        determines how much fluctuation the returned time series is subject to
        noise is sampled from a gaussian with mean=0 and std=noise_scale 
        noise increases linearly with index of time series
        i.e. Let x_t be the t-th entry of time series of quantity x.
             entry x_0 will be free of noise, entry x_t will be subject
             to gaussian noise with standard deviation t*noise_scale
    data : pd.Series
        Object from which time series of length horizon are read of mode is 'read'
        Object from which features are fed into the model if mode is 'predict' 
    model : tbd
        prediction model
        None if mode is 'read'

    Methods
    ----------
    predict(**state)
        returns next time series
    
    setup_reader(data)
        obtains self.data from which chunks will be returned

    setup_prediction(model)
        sets up saved model for prediction

    """
    def __init__(self, snapshots, horizon, mode=None, data=None, model=None, 
                        noise_scale=0.05, **kwargs):
        """
        inits the class
        Also sets up the data or model that should be returned

        Parameters
        ----------
        snapshots : pd.DatatimeIndex
            points in time for which predictions (simulations) will be made
        horizon : int
            length of time series that will be returned in predict call
        mode : str
            either 'read' for data-based prophet or 
            'predict' for machine learning model based return
        data : str or pd.Series or pd.DataFrame
            interpreted as data source if mode is 'read'
            interpreted as data for features if mode is 'predict' 
        noise_scale : float
            std of gaussian noise that is added every time step

        Returns
        ----------
        -
        """

        self.snapshots = snapshots
        self.horizon = horizon 
        self.mode = mode
        self.noise_scale = noise_scale
        self.data = data  
        self.model = None

        if mode == 'read': 
            assert data is not None, f'for chosen mode simulate, \
                                       kwarg data has to provide either path, pd.Series or pd.DataFrame'
        if mode == 'predict': 
            assert data is not None, f'for chosen mode predict, \
                                       kwarg model has to provide either path to or the features themselves'
        assert mode == 'predict' or mode == 'read', f'Please choose mode \
                                        read or predict instead of {mode}.'

        if mode == 'read':
            self.setup_reader(data) 

        elif mode == 'predict':
            self.setup_ml(data)
            raise NotImplementedError('implement me!')


    def setup_reader(self, data):
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
            print('Set up reading prophet from file {}!'.format(data)) 
            data = pd.read_csv(data, parse_dates=True, index_col=0)
            self.data = data[data.columns[-1]]

        elif isinstance(data, pd.Series):
            print('Set up reading prophet from pd.Series!')
            self.data = data

        elif isinstance(data, pd.DataFrame):
            print('Set up reading prophet from pd.DataFrame!')
            self.data = data[data.columns[-1]]

        assert isinstance(data.index, pd.DatetimeIndex) is True, f'\
                    Index of received data {self.data} of is not of type pd.DatetimeIndex!'


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

    
    def predict(self, time):
        """
        Returns prediction based on current state of the system

        Parameters
        ----------
        time : pd.Timestamp 
            reflects current system time
            Note that for both mode 'read' and 'predict' this is the
            only required information as it points to the respective data
        
        Returns
        ----------
        y : pd.Series
            vector of predictions. Index is in datetime format
        """
        if self.mode == 'read':

            # create cumulative random noise
            noise = np.random.normal(scale=self.noise_scale, size=self.horizon-1)
            noise = pd.Series([noise[:i+1].sum() for i in range(self.horizon)])

            # format time to timestamp
            if isinstance(time, int) is not True:
                assert time in self.data.index, f'Got timestamp {time} which is not in index of data {self.data}'
                time = self.data.index.get_loc(time)
            
            # cut out data within horizon
            snippet = self.data.iloc[time+1:time+self.horizon+1]
            # make consistent index
            noise.index = self.data.index[time+1:time+self.horizon+1]

            return snippet + noise

        elif self.mode == 'predict':

            raise NotImplementedError('Implement prediction from model')



if __name__ == '__main__':
    # for testing purposes
    # create artificial time series
    snapshots = pd.date_range('2020-01-01', '2020-01-02', freq='30min')
    horizon = 10

    data_path = os.path.join(os.getcwd(), 'data', 'dummy')
    demand_path = os.path.join(data_path, 'demand.csv')
    supply_path = os.path.join(data_path, 'supply.csv')

    demand_config = {'mode': 'read', 'source': demand_path, 'noise_scale': 0.05}
    supply_config = {'mode': 'read', 'source': supply_path, 'noise_scale': 0.2}
    names = ['demand', 'supply']

    configs = {name: config for name, config in zip(names, [demand_config, supply_config])}

    prophets = {key: Prophet(snapshots, horizon, **config) for key, config in configs.items()}


    # initial points of returns
    anchors = snapshots[:-horizon+1]

    # test for pass of filename
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for key, prophet in prophets.items():

        prophet.data.plot(ax=ax, linewidth=1.)

        for anchor in anchors:
            prediction = prophet.predict(t=anchor)
            prediction.plot(ax=ax, linewidth=0.7)

    plt.show()