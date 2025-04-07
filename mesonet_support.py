'''
Advanced Machine Learning
Mesonet PDF support code


Andrew Justin, Andrew Fagg, 2025-03

'''


import numpy as np


import tensorflow as tf

# THIS IS REALLY IMPORTANT
import tf_keras as keras
from tf_keras.models import Sequential
import tensorflow_probability as tfp

# Sub namespaces that are useful later
# Tensorflow Distributions
tfd = tfp.distributions
# Probability Layers 
tfpl = tfp.layers

from tf_keras.layers import Layer, Concatenate
from tf_keras.layers import Dense, BatchNormalization, Dropout
from tf_keras import Input, Model
from matplotlib import colors
from tf_keras.utils import plot_model

import pandas as pd

def get_mesonet_folds(dataset_fname:str='/home/fagg/datasets/mesonet/allData1994_2000.csv',
                      ntrain_folds: int = 6, 
                      nvalid_folds: int = 1, 
                      ntest_folds: int = 1,
                      rotation: int = 0)->(np.array, np.array, int, np.array, np.array, int, np.array, np.array, int):
    """
    Split mesonet data into training, validation, and test folds.

    Parameters
    ----------
    dataset_fname: str: full path to the Mesonet data set
    ntrain_folds: int, default = 6
        Number of training folds.
    nvalid_folds: int, default = 1
        Number of validation folds.
    ntest_folds: int, default = 1
        Number of testing folds.

    Return: numpy arrays: ins_training, outs_training, ins_validation, outs_validation, ins_testing, outs_testing
    """

    # load in the mesonet dataframe & clean missing values
    df = pd.read_csv(dataset_fname, na_values=[-999, -998, -997, -996]).fillna(0)

    # Identify unique station IDs and split into training, validation and test folds
    STIDs = np.unique(df['STID'])

    # Split into folds
    nfolds = ntrain_folds + nvalid_folds + ntest_folds
    # Must be evenly divisible
    assert (STIDs.shape[0] % nfolds) == 0, "The number of stations (%d) must be divisible by the total number of folds (%d)"%(STIDs.shape[0], nfolds)
    
    folds = np.array(np.split(STIDs, nfolds))

    # Collect station IDs into folds
    train_STIDs = folds[(np.arange(ntrain_folds)+rotation)%nfolds].flatten()
    valid_STIDs = folds[(np.arange(nvalid_folds)+ntrain_folds+rotation)%nfolds].flatten()
    test_STIDs = folds[(np.arange(ntest_folds)+ntrain_folds+nvalid_folds+rotation)%nfolds].flatten()

    # Fold sizes
    train_nstations = len(train_STIDs)
    valid_nstations = len(valid_STIDs)
    test_nstations = len(test_STIDs)

    # Extract data from full data set by station
    train_dataset = df[df['STID'].isin(train_STIDs)]
    valid_dataset = df[df['STID'].isin(valid_STIDs)]
    test_dataset = df[df['STID'].isin(test_STIDs)]

    # precip is removed from the datasets with the .pop() command
    train_labels = train_dataset.pop('RAIN').to_numpy()[:, np.newaxis]
    valid_labels = valid_dataset.pop('RAIN').to_numpy()[:, np.newaxis]
    test_labels = test_dataset.pop('RAIN').to_numpy()[:, np.newaxis]
    
    # remove year, month, day, STID from the datasets
    train_dataset = train_dataset.drop(['YEAR', 'MONTH', 'DAY', 'STID'], axis=1).to_numpy()
    valid_dataset = valid_dataset.drop(['YEAR', 'MONTH', 'DAY', 'STID'], axis=1).to_numpy()
    test_dataset = test_dataset.drop(['YEAR', 'MONTH', 'DAY', 'STID'], axis=1).to_numpy()
    
    return train_dataset, train_labels, train_nstations, valid_dataset, valid_labels, valid_nstations, test_dataset, test_labels, test_nstations

def extract_station_timeseries(ins:np.array, outs:np.array, nstations:int, station_index:int)->(np.array, np.array):
    '''
    Extract the station data and precipitation labels for a single station from a dataset that contains
    multiple stations.

    :param ins: Numpy array of model inputs for a full data set (training, validation or testing)
    :param outs: Numpy array of the corresponding desired outputs for a full data set
    :param nstations: Number of stations in the data set
    :param station_index: Station to extract (0 ... nstations-1)
    :return: Numpy arrays for the model inputs/desired outputs for the selected station

    We are taking advantage of the fact that the samples in the original data set are sorted by time and then station
    '''
    assert ((station_index >= 0) and (station_index < nstations)), "station_index must fall within [0 ... nstations-1]"
    return ins[station_index::nstations,:], outs[station_index::nstations]

class SinhArcsinh():
    '''
    Support for creating a Keras Layer that implements the Sinh-ArcSinh distribution
    
    '''
    @classmethod
    def num_params(cls):
        '''
        :return: The number of parameter types expected.  In this case, mu, std, skewness, tailweight

        Note: each parameter type can be multi-dimensional for multi-dimensional events
        '''
        return 4

    @classmethod 
    def create_layer(cls):
        '''
        Create a Keras 3 layer that implements a Sinh-Archsinh distribution.
        This layer takes as input 4 TF tensors & returns a TF distribution

        :return: Keras 3 layer
        '''
        return tfpl.DistributionLambda(SinhArcsinh._make_sinharcsinh)
    
    @classmethod
    def _make_sinharcsinh(cls, params):
        '''
        :param params: 4 TF Tensors, corresponding to mu, std, skewness, tailweight
        
        :return: A SinhArcsinh distribution object.
        
        NOTEs: 
        - We assume that params[1] > 0 and params[3] > 0
        - This method is only called when the wrapping lambda distribution is
            called (i.e., when the params are proper TF Tensors)
        '''
        return tfd.SinhArcsinh(
            loc=params[0],
            scale=params[1],
            skewness=params[2],
            tailweight=params[3], 
        )

    # Custom loss function

    @classmethod
    def mdn_loss(cls, y, dist):
        '''
        Compute negative log likelihood of the desired output.  Used for loss
        when compiling a model
        
        :param y: True value (from the training set)
        :param dist: A TF Probability Distribution
        :return: The negative likelihood of each true value
        '''
        return -dist.log_prob(y)
        