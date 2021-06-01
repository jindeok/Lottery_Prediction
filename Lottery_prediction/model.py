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
    
    def __init__(self, DataLoader, verb, hid_dim = 128):
        
        self.train_X = DataLoader.train_X
        self.test_X = DataLoader.test_X
        self.train_Y = DataLoader.train_Y
        self.test_Y = DataLoader.test_Y
        self.n_hours = DataLoader.window_prev
        
        self.model = Sequential()
        self.model.add(LSTM(hid_dim, input_shape=(self.train_X.shape[1], self.train_X.shape[2])))
        self.model.add(Dense(ENTIRE_NUMBER))
        
        self.model.compile(loss='mse', optimizer='adam')
        
        self.verb = verb
        
        
    def training(self, num_epoch, num_batch):
        
        # no validation currently.
        history = self.model.fit(self.train_X, self.train_Y, epochs = num_epoch, batch_size = num_batch,
                                 verbose=2, shuffle=False)
        
        return history
    
    def predict_lottery_numbers(self, mode2, trial):
        
        yhat = self.model.predict(self.test_X) # [1x45] dim
       # print(yhat)

        prediction_number_set = []
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
                
                pdf = list(yhat[0]) # use the output as prob. desity dist.
                pdf = [0 if i < 0 else i for i in pdf]
                pdf_norm = [float(i)/sum(pdf) for i in pdf] #normalize for make it as a pdf form.
                selected = np.random.choice(ENTIRE_NUMBER, size=6, replace=False, p=pdf_norm)
                prediction_number_set.append(selected)
                if self.verb == 'verbose':
                    print("predicted set of lottery numbers at {}th trial :".format(t), selected + 1)

        #for greed
        yhat_assigned = np.argsort(-yhat[:])                
        greed_pred = yhat_assigned[0][:6] + 1
        print("greed assignment result :" ,greed_pred)
        
        return prediction_number_set
        
    def predict_randomely(self, trial):
        
        prediction_number_set = []
        for t in range(1,trial+1):
            
            selected = np.random.choice(ENTIRE_NUMBER, size=6, replace=False)
            prediction_number_set.append(selected)
        
        return prediction_number_set
        
    def evaluate(self, overall_prediction):
        
        gth = np.argsort(-self.test_Y[:])
        gth = gth[:,:TOT_NUMBER_OF_GTH]   # considering bonus number, entire number is 7, not 6
        
        print('-----------evaluation ----------')
        print('Lottery Winning numbers :', gth[0])
        trial = 1
        all_count = 0
        for pred_set in overall_prediction:
            count = 0
            for i in pred_set:
                if i in gth:
                    count += 1          
                    all_count += 1
            if self.verb == 'verbose':                       
                print('{}th predicted:'.format(trial), pred_set)            
                print('{}th trial: {} out of 6 is correct !!'.format(trial, count))
            
            trial += 1 
            
        print('Precision:{}%'.format(100*all_count/(6*(trial-1))))
        
