import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    # 'Generates data for Keras'
    def __init__(self, data_path,n_batch,channels=[0,1,2,3], batch_size=512, time_steps=192, target='sf', shuffle=True):
    #    'Initialization'
        self.data_path = data_path   
        self.n_batch = n_batch		
        self.channels = channels
        self.n_channels = len(channels)
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.shuffle = shuffle
        self.target = target
        self.on_epoch_end()

    def __len__(self):
    #    'Denotes the number of batches per epoch'       
        return self.n_batch

    def __getitem__(self, index):
    #    'Generate one batch of data'
        # Generate indexes of the batch
           
        batch_path = self.data_path+'batch'+str(self.indexes[index])+'.npz'
        batch_data = np.load(batch_path)

        X = np.empty((self.batch_size,self.time_steps,self.n_channels*2))		
        for ch in range(self.n_channels):
          X[:,:,ch*2:(ch+1)*2] = batch_data['X'][:self.batch_size,:self.time_steps,self.channels[ch]*2:(self.channels[ch]+1)*2]		
        
        if self.target == 'mrm':
          y = batch_data['mrm'][:self.batch_size,:self.time_steps].reshape(self.batch_size,self.time_steps,1)
        elif self.target == 'cirm':
          y = batch_data['cirm'][:self.batch_size,:self.time_steps,:]
        elif (self.target == 'sf') or (self.target == 'cc'):
          y = batch_data['cln'][:self.batch_size,:self.time_steps,:]		

        return X, y

    def on_epoch_end(self):
    #    'Updates indexes after each epoch'
        self.indexes = np.arange(self.n_batch)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
              
 
