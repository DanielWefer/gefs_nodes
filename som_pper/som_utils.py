import xarray as xr
import numpy as np
import pprint

from minisom import MiniSom
from sklearn.preprocessing import RobustScaler

def _preprocess_var(arr, var_name):
    """
    Apply variable-specific preprocessing.
    """
    if 'cape' in var_name.lower():
        arr = np.clip(arr, 0, None)
        arr = np.log1p(arr)
    return arr


def read_and_transform(config, long_fmt='360'):
    """
    Reads and subsets GEFS data for an arbitrary number of variables.
    
    Parameters:
        config: dict
            Must contain 'vars' (list of strings), 'filename', and spatial bounds.
        long_fmt: str
            Format of longitude degrees. Default 360.
    """
    
    variables = config['vars']
    data_in = xr.open_dataset(config['filename'])[variables]
    
    if long_fmt == '360':
        data_in = data_in.sel(
            longitude=slice(config['wlon'], config['elon']),
            latitude=slice(config['nlat'], config['slat'])
        )
    else:
        raise ValueError("-180 to 180 not implemented yet, use 0-360")
        
    data_in = data_in.squeeze()

    arrays = [_preprocess_var(data_in[v].values, v) for v in variables]

    processed_list = []
    for i in range(len(data_in.time)):
        timestep_data = np.concatenate([a[i].flatten() for a in arrays])
        processed_list.append(timestep_data)
        
    npy = np.array(processed_list)
    
    return data_in, npy

def build_scaler(config):
    # This function remains largely the same, but 'config' now supports multiple vars
    training_ds, training_npy = read_and_transform(config)
    
    scale = RobustScaler()
    scale.fit(training_npy)

    return training_ds, training_npy, scale

def train_som(preferences):
    ds, npy, scaler = build_scaler(preferences)

    scaled_npy = scaler.transform(npy)

    # Calculate input_len dynamically: (Number of Vars) * (Lat) * (Lon)
    num_vars = len(preferences['vars'])
    preferences['som_config']['input_len'] = num_vars * ds.sizes['longitude'] * ds.sizes['latitude']
    preferences['som_train']['data'] = scaled_npy
    
    print("current model configuration")
    pprint.pprint(preferences['som_config'])
    print("current training configuration")
    pprint.pprint(preferences['som_train'])
    
    som = MiniSom(**preferences['som_config'])
    som.train(**preferences['som_train'])

    return som, scaler, preferences, ds



def preprocess_single_sample(filepath, config, long_fmt='360', time_index=0):
    """
    Read, subset, and preprocess a single sample from a new file
    so it matches the training feature layout.

    Parameters
    ----------
    filepath : str
        Path to the new dataset, e.g. './Datasets/era5.nc'
    config : dict
        Must contain:
            - 'vars'
            - 'wlon', 'elon', 'nlat', 'slat'
    long_fmt : str
        Longitude format. Default is '360'
    time_index : int
        Which time index to classify if multiple times are present

    Returns
    -------
    data_in : xarray.Dataset
        Subset dataset
    x_raw : np.ndarray
        Shape (1, n_features), not yet scaled
    """
    variables = config['vars']
    data_in = xr.open_dataset(filepath)[variables]

    if long_fmt == '360':
        data_in = data_in.sel(
            longitude=slice(config['wlon'], config['elon']),
            latitude=slice(config['nlat'], config['slat'])
        )
    else:
        raise ValueError("-180 to 180 not implemented yet, use 0-360")

    data_in = data_in.squeeze()

    arrays = [_preprocess_var(data_in[v].values, v) for v in variables]

    if 'time' in data_in.dims:
        x = np.concatenate([a[time_index].flatten() for a in arrays])
    else:
        x = np.concatenate([a.flatten() for a in arrays])

    x_raw = x[np.newaxis, :]

    return data_in, x_raw


def classify_single_sample(filepath, config, som, scaler, long_fmt='360', time_index=0):
    """
    Preprocess, scale, and classify a single sample onto an already-trained SOM.

    Parameters
    ----------
    filepath : str
        Path to new dataset
    config : dict
        Same spatial/variable config used for training
    som : MiniSom
        Already-trained MiniSom object
    scaler : RobustScaler
        Already-fitted scaler from training
    long_fmt : str
        Longitude format. Default is '360'
    time_index : int
        Time index to classify if multiple times are present

    Returns
    -------
    winner : tuple
        Winning node as (row, col)
    flat_node : int
        Flattened node number
    dist_to_bmu : float
        Euclidean distance to BMU
    x_scaled : np.ndarray
        Scaled sample, shape (n_features,)
    data_in : xarray.Dataset
        Subset dataset used for classification
    """
    data_in, x_raw = preprocess_single_sample(
        filepath=filepath,
        config=config,
        long_fmt=long_fmt,
        time_index=time_index
    )

    x_scaled = scaler.transform(x_raw)[0]

    winner = som.winner(x_scaled)

    ncols = som.get_weights().shape[1]
    flat_node = winner[0] * ncols + winner[1]

    bmu_weight = som.get_weights()[winner]
    dist_to_bmu = np.linalg.norm(x_scaled - bmu_weight)

    return winner, flat_node, dist_to_bmu, x_scaled, data_in