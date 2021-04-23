# load dependacies
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from globalvar import *


class LotteryLSTM:
    
    def __init__(self, DataLoader, hid_dim = 128):
        
        self.train_X = DataLoader.train_X
        self.test_X = DataLoader.test_X
        self.train_Y = DataLoader.train_Y
        self.test_Y = DataLoader.test_Y
        
        self.model = Sequential()
        self.model.add(LSTM(hid_dim, input_shape=(self.train_X.shape[1], self.train_X.shape[2])))
        self.model.add(Dense(ENTIRE_NUMBER))
        
        self.model.compile(loss='mse', optimizer='adam')
        
        
    def training(self, num_epoch, num_batch):
        
        # no validation currently.
        history = self.model.fit(self.train_X, self.train_Y, epochs = num_epoch, batch_size = num_batch,
                                 verbose=2, shuffle=False)
        
        return history
    
    def predict_lottery_numbers(self):
        '''
        greed assignment is used
        '''
        
        yhat = self.model.predict(self.test_X) # [1x45] dim
        yhat_assigned = np.argsort(-yhat[:])
        
        prediction_number_set = yhat_assigned[:,:6]        
        print("predicted set of lottery numbers :", prediction_number_set)
        
        return prediction_number_set 
        
        
        
    def evaluate(self, prediction_number_set):
        
        gth = np.argsort(-self.test_Y[:])
        gth = gth[:,:TOT_NUMBER_OF_GTH]   # considering bonus number, entire number is 7, not 6
        count = 0
        for i in prediction_number_set[0]:
            if i in gth:
                count += 1
        
        score = ( count / 6 ) * 100
        
        print('-----------evaluation ----------')
        print('predicted:', prediction_number_set)
        print('actual numbers were:', gth)
        print('model accuracy score is {}%'.format(score))
        


