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
    
    parser.add_argument('--window', type = int, default = 5 , help = "time stamps")
    parser.add_argument('--data_dir', nargs='?', default='dataset/lottery_history.csv')
    parser.add_argument('--mode', nargs='?', default='eval_model', help = "eval_model or predict")
    parser.add_argument('--mode2', nargs='?', default='sampling', help = "greed or sampling")
    parser.add_argument('--trial', type = int, default='5', help = "how much number to generate")
    parser.add_argument('--training_length', type=float, default = 0.6)   
    
    return parser.parse_args()

args = parse_args()

if __name__ == '__main__':
    
    dataset = DataLoader(args.data_dir, args.training_length, args.window, args.mode)
    LotteryLSTM = LotteryLSTM(dataset, hid_dim = 64)
    LotteryLSTM.training(num_epoch = 500, num_batch = 24)
    prediction_number_set = LotteryLSTM.predict_lottery_numbers(args.mode2, args.trial)
    
    if args.mode == 'eval_model':
        LotteryLSTM.evaluate(prediction_number_set)
        
    
    

