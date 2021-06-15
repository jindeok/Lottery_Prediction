# Lottery_Prediction
LSTM based lottery forecast deep learning model.

The dataset is collected from official korea lottery Web: https://dhlottery.co.kr/common.do?method=main

# Dependancy
any version of {} might works.
{
* tensorflow
* keras
* sklearn
* numpy, pandas
}



# Running

run ``main.py`` script

or in your prompt, type

``python main.py --data_dir 'dataset/lottery_history.csv' --mode 'eval_model' --trial 5 --training_lengt 0.90 ``  

dataset version is on the date: [21.04.23]

you can easily update dataset from the Web:  https://dhlottery.co.kr/common.do?method=main

***some comments on the arguments:***

training_length : there are ~980 lottery cases in total. you can decide to what extent to use as for training length.
(e.g. 0.5 training_lengh uses 485 lottery cases are used for training and infer 486 th as a test set.)

