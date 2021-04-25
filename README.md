# Lottery_Prediction
LSTM based lottery forecast deep learning model.
Lottery dataset is collected from official korea lottery Web: https://dhlottery.co.kr/common.do?method=main

# Dependancy
any version of {} might works ..

* tensorflow
* keras
* sklearn
* numpy, pandas




# Running

run main.py in the Lottery_Prediction folder
or in terminal,
``python main.py --data_dir 'dataset/lottery_history.csv' --mode 'eval_model' --trial 5 --training_lengt 0.90 ''  
dataset version of date is [21.04.23]
you can update dataset from the Web:  https://dhlottery.co.kr/common.do?method=main

***some comments on the arguments:***

training_length - there are ~980 lottery case in total. you can decide to what extend to use as for training length.
(e.g. 0.5 training_lengh uses 485 lottery cases are used for training and infer 486 th as a test set.)

