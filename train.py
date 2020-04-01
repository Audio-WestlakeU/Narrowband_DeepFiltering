import numpy as np
import os,fnmatch

from keras.models import Model 
from keras.layers import Dense,Activation,Input,Lambda
from keras.layers import Bidirectional,LSTM,TimeDistributed,concatenate
from keras.callbacks import ModelCheckpoint

from BatchGenerator import DataGenerator
from complex_linear_filtering import complex_linear_filtering

# please set datapath to the directory of batch data
dataPath = '/scratch/mensa/xiali/CHiME3/data/audio/16kHz/NBDF/'
train_batch = dataPath+'train_batch/'
validation_batch = dataPath+'validation_batch/'

# directory for saving trained models
modelPath = 'models/'
if not os.path.isdir(modelPath):
  os.makedirs(modelPath)

# time_steps is 192 by default, but it can be set to a value < 192, DataGenerator will exctract a subsegment     
time_steps = 192 

# batch_size could be <= 512
batch_size = 512

# the number of batches in train_batch/validation_batch, or use only a part of batches  
# by setting a smaller number 
batchFiles = fnmatch.filter(os.listdir(train_batch),'batch*.npz')
n_batch_train = len(batchFiles)
batchFiles = fnmatch.filter(os.listdir(validation_batch),'batch*.npz')
n_batch_validation = len(batchFiles)

if __name__ == '__main__':

  for target in ['mrm','cirm','cc','sf']:             # target can be: 'mrm', 'cirm', 'cc' and 'sf'
    rnnnet = 'blstm'                                  # rnnnet can be: 'lstm', 'blstm'

    # channels used for train, the reference channel, i.e. 3 (CHiME mic #6), has to be included.
    # could be a subset of [0,1,2,3], e.g. [0,3], [1,3], [2,3], [0,1,3], [0,2,3], [1,2,3]
    channels = [0,1,2,3]    
    chime_channels = ''
    for ch in channels:
      chime_channels += str(ch+3)
    n_channels = len(channels)

    # parameters for DataGenerator 
    params = {'channels':channels,
		  'time_steps': time_steps,  
		  'batch_size': batch_size,
		  'target': target,
          'shuffle': True}

    # data generator
    training_generator = DataGenerator(train_batch,n_batch_train,**params)
    validation_generator = DataGenerator(validation_batch,n_batch_validation,**params)

    ######################### lstm network #########################
    lstm1_output_size = 256 
    lstm2_output_size = 128 
    model_input = Input(shape=(time_steps,n_channels*2))
    if rnnnet == 'lstm':
      lstm1 = LSTM(lstm1_output_size,return_sequences=True)(model_input)
      lstm2 = LSTM(lstm2_output_size,return_sequences=True)(lstm1)
    elif rnnnet == 'blstm':
      lstm1 = Bidirectional(LSTM(lstm1_output_size,return_sequences=True))(model_input)
      lstm2 = Bidirectional(LSTM(lstm2_output_size,return_sequences=True))(lstm1)
    if target == 'mrm':
      model_output = TimeDistributed(Dense(1,activation='sigmoid'))(lstm2)
    elif (target == 'cirm') or (target == 'cc'):
      model_output = TimeDistributed(Dense(2))(lstm2)
    elif target == 'sf':
      dense = TimeDistributed(Dense(n_channels*2,activation='tanh'))(lstm2)
      input_filter_concat = concatenate([model_input,dense])
      model_output = TimeDistributed(Lambda(complex_linear_filtering))(input_filter_concat)
    model = Model(inputs=model_input,outputs=model_output)
    model.compile(loss='mean_squared_error',optimizer='adam',metrics=['mse'])
    model.summary()


    ############################ Train ##############################
    # save models for all epochs
    filepath = modelPath+rnnnet+'-'+target+'-'+chime_channels+'chs-{epoch:02d}-{val_loss:.4f}.hdf5'
    check_point = ModelCheckpoint(filepath, save_best_only=False)
    callbacks_list = [check_point]
   
    # set epochs based on preliminary experiments
    epochs = 5
    if rnnnet is 'blstm':
      if (target is 'cc') or (target is 'sf'):
        epochs = 10      

    # train
    print('{} {} {} Train...'.format(rnnnet,target,str(channels)+'chs'))
    model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=epochs,
                    callbacks=callbacks_list,
                    use_multiprocessing=True,
                    workers=4)




