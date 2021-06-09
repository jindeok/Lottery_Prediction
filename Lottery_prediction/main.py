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
from data import *
from model import *

import os
import argparse


def parse_args():
    '''
    Parses the arguments.
    '''
    parser = argparse.ArgumentParser(description="Run Lottery prediction..")
    
    parser.add_argument('--window', type = int, default = 1 , help = "time stamps")
    parser.add_argument('--data_dir', nargs='?', default='dataset/lottery_history.csv')
    
    parser.add_argument('--mode', nargs='?', default='back-test', help = "back-test or predict")
    parser.add_argument('--mode2', nargs='?', default='sampling', help = "greed or sampling")
    parser.add_argument('--verb', nargs='?', default='not_verbose', help = "verbose or not_verb")
    
    parser.add_argument('--trial', type = int, default='10000', help = "how much trials to generate")
    parser.add_argument('--training_length', type=float, default = 0.9)   
    
    return parser.parse_args()

args = parse_args()

if __name__ == '__main__':
    dataset = DataLoader(args.data_dir, args.training_length, args.window, args.mode)
    LotteryLSTM = LotteryLSTM(dataset, hid_dim = 32, verb = args.verb)
    LotteryLSTM.training(num_epoch = 500, num_batch = 24)
    prediction_number_set = LotteryLSTM.predict_lottery_numbers(args.mode2, args.trial)
    random_pred_set = LotteryLSTM.predict_randomely(args.trial)
    
    if args.mode == 'back-test':
        LotteryLSTM.evaluate(prediction_number_set)
        print("---------Random baseline-------------")
        LotteryLSTM.evaluate(random_pred_set)
    
    
    # dataset.test_X = np.array([[[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #      0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0.,
    #      0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]]])
