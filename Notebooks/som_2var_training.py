# OLD AND OUTDATED - DON'T USE!!!!!!


import xarray as xr
import numpy as np
import pprint

from minisom import MiniSom
from sklearn.preprocessing import RobustScaler


def read_and_transform(config, long_fmt='360'):
    """`Reads in the GEFS filename and subsets the data based on the configuration 
    dictionary. This function subsets by coordinates and variable name.
    
    Parameters:
        config: dict
            Dictionary containing lat / lon bounds, filename, and variable name.
        long_fmt: str
            Format of longitude degrees. Default 360.
    
    Returns:
        data_in: xarray dataset
            xarray representation of the GEFS dataset.
        npy: numpy ndarray
            numpy representation of the GEFS dataset.
    """

    v1, v2 = config['var1'], config['var2']
    data_in = xr.open_dataset(config['filename'])[[v1, v2]]
    
    if long_fmt == '360':
        # data already uses 0..360 longitudes
        data_in = data_in.sel(lon=slice(config['wlon'], config['elon']),
                              lat=slice(config['nlat'], config['slat']))
    else:
        # convert from -180..180 to 0..360, then sort so slicing works
        data_in['lon'] = (data_in['lon'] + 360) % 360
        data_in = data_in.sortby('lon')
        data_in = data_in.sel(lon=slice(config['wlon'], config['elon']),
                              lat=slice(config['nlat'], config['slat']))
        
    data_in = data_in.squeeze()

    a1 = data_in[v1].values  # shape: (time, lat, lon, ...) -> iterates along first axis
    a2 = data_in[v2].values

    # Per-timestep flatten (your original style) and side-by-side concat
    npy = np.array([
        np.concatenate([t1.flatten(), t2.flatten()])
        for t1, t2 in zip(a1, a2)
    ])
    
    return data_in, npy

def build_scaler(config):
    """`Reads in the GEFS data based on the filename and configuration
    and then preprocesses the data by applying a percentile scaler to
    normalize the data. This should help address outlier impacts
    regardless of the variable used.
    
    Parameters:
        config: dict
            Dictionary containing lat / lon bounds and variable name.
            
    Return:
        training_ds: xarray dataset
            xarray representation of the GEFS data.
        training_npy: numpy ndarray
            numpy representation of GEFS data.
        scale: RobustScaler
            trained scaler used for training and prediction.
    """
    
    #read in the file and get a flattened representation
    training_ds, training_npy = read_and_transform(config)
    
    #define the scaler that transforms the data
    scale = RobustScaler()

    #fit the scaler that transforms the data
    scale.fit(training_npy)

    return training_ds, training_npy, scale

def train_som(preferences):
    """Trains the SOM by reading in the data based on 
    the preferences and performs preprocessing tasks
    and provides training output messages.
    
    Parameters:
        filename: str
            Absolute or relative location of a GEFS file.
        config: dict
            Dictionary containing lat / lon bounds and variable name.
            
    Returns:
        som: MiniSom model
            Trained SOM model.
        scaler: RobustScaler
            The trained scaler used in training.
        post_preferences: dict
            Updated preferences that include trained SOM config data
            and training data.
        ds: xarray dataset
            The xarray representation of the GEFS data.
    """

    ds, npy, scaler = build_scaler(preferences)

    scaled_npy = scaler.transform(npy)

    preferences['som_config']['input_len'] = 2 * ds.sizes['lon'] * ds.sizes['lat']
    preferences['som_train']['data'] = scaled_npy
    
    print("current model configuration")
    pprint.pprint(preferences['som_config'])
    print("current training configuration")
    pprint.pprint(preferences['som_train'])
    
    som = MiniSom(**preferences['som_config'])
    som.train(**preferences['som_train'])

    return som, scaler, preferences, ds, scaled_npy
