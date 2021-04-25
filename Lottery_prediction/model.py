# load dependacies
import pandas as pd
import numpy as np
import random as random
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
        self.n_hours = DataLoader.window_prev
        
        self.model = Sequential()
        self.model.add(LSTM(hid_dim, input_shape=(self.train_X.shape[1], self.train_X.shape[2])))
        self.model.add(Dense(ENTIRE_NUMBER))
        
        self.model.compile(loss='mse', optimizer='adam')
        
        
    def training(self, num_epoch, num_batch):
        
        # no validation currently.
        history = self.model.fit(self.train_X, self.train_Y, epochs = num_epoch, batch_size = num_batch,
                                 verbose=2, shuffle=False)
        
        return history
    
    def predict_lottery_numbers(self, mode2, trial):
        
        yhat = self.model.predict(self.test_X) # [1x45] dim

        overall_prediction = []
        
        print('-----------Start lottery prediction ----------')
        
        for t in range(1, trial+1):
        
            if mode2 == "greed":
                '''
                greed assignment is used
                '''                 
                yhat_assigned = np.argsort(-yhat[:])                
                prediction_number_set = yhat_assigned[0][:6]        
                
                
            else:
                '''
                sampling from y_hat pdf
                '''
                prediction_number_set = []
                pdf = list(yhat[0]) # use the output as prob. desity dist.
                for _ in range(6):
                    selected = random.choices(np.arange(1, ENTIRE_NUMBER+1), pdf)
                    prediction_number_set.append(selected[0])
                    
            print("predicted set of lottery numbers at {}th trial :".format(t), prediction_number_set)
            
            overall_prediction.append(prediction_number_set)
        
        return overall_prediction
        
        
        
    def evaluate(self, overall_prediction):
        
        gth = np.argsort(-self.test_Y[:])
        gth = gth[:,:TOT_NUMBER_OF_GTH]   # considering bonus number, entire number is 7, not 6
        
        print('-----------evaluation ----------')
        print('Lottery Winning numbers :', gth[0])
        trial = 1
        for pred_set in overall_prediction:
            count = 0
            for i in pred_set:
                if i in gth:
                    count += 1          
                       
            print('{}th predicted:'.format(trial), pred_set)            
            print('{}th trial: {} out of 6 is correct !!'.format(trial, count))
            trial += 1 
        
        
        


