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