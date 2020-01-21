from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
import pathlib
from tensorflow import keras
import os
from keras.callbacks import ModelCheckpoint
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn import datasets
from sklearn.datasets import dump_svmlight_file
from sklearn.externals import joblib
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import argparse
import logging
from keras.layers import Input, Dense    #using set model component
from keras.models import Model    #using set model 
from keras.utils import plot_model    #show model structure
from keras import layers as Layer
import keras 
from pandas import DataFrame as df
from keras.callbacks import EarlyStopping
import gzip
import pickle
from keras import backend as K
import tensorflow as tf
from keras.layers import Dense, Dropout, Input, Activation, BatchNormalization
import nni
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
from tensorflow.keras.callbacks import Callback
from keras.optimizers import Adam

_logger = logging.getLogger('healthy vs unhealthy')
_logger.setLevel(logging.INFO)


#들어가는 feature 갯수 = 10783
#맞추고 싶은 것은 환자의 상태 0 = healthy; 1 = unhealthy

def load_data():
   
    dataframe = pd.read_csv("./combined_metatdata.csv")
    value = dataframe.drop(["SampleID", "Sample", "Glu_status", "Stat"],axis=1).values
    interest = dataframe.Stat.values #맞추고 싶은 것은 환자의 상태 0 = healthy; 1 = unhealthy
    X = np.array(value)
    min_max_scaler = MinMaxScaler()
    X_MinMax = min_max_scaler.fit_transform(X)
    train_x, test_x, train_y, test_y = train_test_split(X_MinMax, interest, test_size=0.2, random_state=10, stratify=interest)

    return train_x, train_y, test_x, test_y


class dnnmodel(Model):
    def __init__(self, hidden_size):
        super().__init__()
        
        #처음 들어가는 feature이 107843개인데 여기에 자동으로 들어감??
        #input layer따로 만들어줘야되는거 아니야???
        self.fc1 = Dense(units=hidden_size, activation = 'relu')
        self.fc2 = Dense(units=hidden_size, activation = 'relu')
        self.fc3 = Dense(units=hidden_size, activation = 'relu')
        self.fc4 = Dense(units=hidden_size, activation = 'relu')
        self.fc5 = Dense(units=1, activation='sigmoid')
    
    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return self.fc5(x)

class ReportIntermediates(Callback): 
    def on_epoch_end(self, epoch, logs=None):
        """Reports intermediate accuracy to NNI framework"""
        # TensorFlow 2.0 API reference claims the key is `val_acc`, but in fact it's `val_accuracy`
        if 'val_acc' in logs:
            nni.report_intermediate_result(logs['val_acc'])
        else:
            nni.report_intermediate_result(logs['val_accuracy'])

def main(params):
    """
    Main program:
      - Build network
      - Prepare dataset
      - Train the model
      - Report accuracy to tuner
    """
    model = dnnmodel(
        hidden_size=params['hidden_size']
    )
    optimizer = Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    _logger.info('Model built')

    train_x, train_y, test_x, test_y = load_data()
    _logger.info('Data loaded')

    model.fit(
        train_x,
        train_y,
        batch_size=params['batch_size'],
        epochs=100,
        verbose=0,
        callbacks=[ReportIntermediates()],
        validation_data=(test_x, test_y)
    )
    _logger.info('Training completed')

    loss, accuracy = model.evaluate(test_x, test_y, verbose=0)
    nni.report_final_result(accuracy)  # send final accuracy to NNI tuner and web UI
    _logger.info('Final accuracy reported: %s', accuracy)    

if __name__ == '__main__':
    params = nni.get_next_parameter()
    _logger.info('Hyper-parameters: %s', params)
    main(params)