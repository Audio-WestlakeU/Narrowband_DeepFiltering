import numpy as np
import scipy.signal as signal
import scipy.io.wavfile 
import os,fnmatch

from keras.models import Model 
from keras.layers import Dense,Activation,Input,Lambda
from keras.layers import Bidirectional,LSTM,TimeDistributed,concatenate

from complex_linear_filtering import complex_linear_filtering

# please set datapath to the directory of test wav data
dataPath = '/scratch/mensa/xiali/CHiME3/data/audio/16kHz/NBDF/'
testPath = dataPath+'test_mixed_wav/'

modelPath = 'models/'

#import pdb; pdb.set_trace()

# STFT parameters, should be identical to the ones used for train
ft_len = 512
ft_overlap = ft_len//2
fre_num = ft_len//2+1
win = 'hann'

amp = np.iinfo(np.int16).max

# test setup
channels = [0,1,2,3]
ref_channel = 3
chime_channels = ''
for ch in channels:
  chime_channels += str(ch+3)
n_channels = len(channels)

for target in ['mrm','cirm','cc','sf']:            
  rnnnet = 'blstm'      

  # choose model 
  bestepoch = '05'
  if rnnnet is 'blstm':
    if (target is 'cc') or (target is 'sf'):
      bestepoch = '10'   
  modelname = fnmatch.filter(os.listdir(modelPath),rnnnet+'-'+target+'-'+chime_channels+'chs-'+bestepoch+'*'+'.hdf5')
 
  # directory for enhanced signals
  outPath = dataPath+rnnnet+'-'+target+'-'+chime_channels+'chs-enhanced/'

  ######################### lstm network #########################
  lstm1_output_size = 256 
  lstm2_output_size = 128 
  model_input = Input(shape=(None,n_channels*2))
  if rnnnet is 'lstm':
    lstm1 = LSTM(lstm1_output_size,return_sequences=True)(model_input)
    lstm2 = LSTM(lstm2_output_size,return_sequences=True)(lstm1)
  elif rnnnet is 'blstm':
    lstm1 = Bidirectional(LSTM(lstm1_output_size,return_sequences=True))(model_input)
    lstm2 = Bidirectional(LSTM(lstm2_output_size,return_sequences=True))(lstm1)
  if target is 'mrm':
    model_output = TimeDistributed(Dense(1,activation='sigmoid'))(lstm2)
  elif (target is 'cirm') or (target is 'cc'):
    model_output = TimeDistributed(Dense(2))(lstm2)
  elif target is 'sf':
    dense = TimeDistributed(Dense(n_channels*2,activation='tanh'))(lstm2)
    input_filter_concat = concatenate([model_input,dense])
    model_output = TimeDistributed(Lambda(complex_linear_filtering))(input_filter_concat)
  model = Model(inputs=model_input,outputs=model_output)
  model.load_weights(modelPath+modelname[0])
  
  ####################### speech enhancement #####################
  envirs = ['bus','caf','ped','str']     
  SNR = [-4,0,4,8]   
  for envir in envirs:
   for snr in SNR:
    print("Processing {}, snr {}dB".format(envir,str(snr)))
    wavPath = testPath+envir+'/snr'+str(snr)+'/'
    wavFiles = fnmatch.filter(os.listdir(wavPath),'*_ms.wav')
    outDir = outPath+envir+'/snr'+str(snr)+'/'
    if not os.path.isdir(outDir):
      os.makedirs(outDir)

    for wavIndx in range(len(wavFiles)):
      # read wav, apply stft, sequence normalization
      rate,s = scipy.io.wavfile.read(wavPath+wavFiles[wavIndx])        
      if len(s.shape) == 2:
        if s.shape[0] > s.shape[1]:
          s = np.transpose(s) 
      f, t, S = signal.stft(s,window=win,nperseg=ft_len,noverlap=ft_overlap)
      Sref = S[ref_channel,:,:]
      mu = np.abs(Sref).mean(axis=1)
      fra_num = S.shape[2]
      X = np.empty((fre_num,fra_num,n_channels*2))
      for ch in range(n_channels):
        X[:,:,2*ch] = np.real(S[channels[ch],:,:])
        X[:,:,2*ch+1] = np.imag(S[channels[ch],:,:]) 
      X = X/mu.reshape(fre_num,1,1)

      # prediction: directly input the whole utterance to the network 
      y = model.predict_on_batch(X)

      # compute STFT of enhanced signal based on network prediction
      if target is 'mrm':
        Y = Sref*y.reshape(fre_num,fra_num)
      elif target is 'cirm':
        lim = 9.99
        y = lim*(y>=lim)-lim*(y<=-lim)+y*(np.abs(y)<lim)
        y = -10*np.log((10-y)/(10+y))          
        Yr = y[:,:,0]*np.real(Sref)-y[:,:,1]*np.imag(Sref)
        Yi = y[:,:,1]*np.real(Sref)+y[:,:,0]*np.imag(Sref)
        Y = Yr+Yi*1j
      elif (target is 'cc') or (target is 'sf'):
        y = y*mu.reshape(fre_num,1,1)
        Y = y[:,:,0]+y[:,:,1]*1j
      
      # istft
      t,enhanced = signal.istft(Y,window=win,nperseg=ft_len,noverlap=ft_overlap,input_onesided=True)      
      enhanced = np.int16(amp*enhanced/np.max(np.abs(enhanced)))

      # wav write
      outname = outDir+wavFiles[wavIndx][:12]+'.wav'
      scipy.io.wavfile.write(outname,rate,enhanced) 
      

    


