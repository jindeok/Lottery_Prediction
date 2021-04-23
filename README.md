# Lottery_Prediction
LSTM based lottery forecast deep learning model
based on lottery data acquisited from 동행복권 Wep: https://dhlottery.co.kr/common.do?method=main

# Dependancy
any version of (if not too old) ..

* tensorflow
* keras
* sklearn
* numpy, pandas




# Running

run main.py in the Lottery_Prediction folder.
dataset version of date is [21.04.23]
you can update dataset from the Wep:  https://dhlottery.co.kr/common.do?method=main

***some comments on arguments:***

training_length - there are ~980 lottery case in total. you can decide to what extend to use as for training length.
(e.g. 0.5 training_lengh uses 485 lottery cases are used for training and infer 486 th as a test set.)
