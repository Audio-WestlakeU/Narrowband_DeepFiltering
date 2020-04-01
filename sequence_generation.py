##################################################################
#Extract and shuffle STFT frequency-wise sequences, then store as .npz files batch by batch. 
#Do this, to read in one (mini)batch during training, only one .npz file (with shuffled order) is loaded, which makes training faster. 
#One problem of this setup is that the sequences in one batch are fixed for all epochs, which however is not a critical issue.
##################################################################

import numpy as np
import scipy.signal as signal
import scipy.io.wavfile 
import os,fnmatch

# please set datapath to the directory of mixed wav data
dataPath = '/scratch/mensa/xiali/CHiME3/data/audio/16kHz/NBDF/'     
datasets = ['train','validation']

# channels correspond to the one set in mix_bthplusbackground.m
channels = [0,1,2,3]     # CHiME mics #3, #4, #5 and #6
ref_channel = 3           
n_channels = len(channels)

# parameters, adjustable
ft_len = 512
ft_overlap = 256
time_steps = 192
batch_size = 512
fre_num = ft_len//2+1
step_inc = time_steps//2	  

# One needs to first collect the training sequences, and then shuffle them. 
# However, due to the memory limit, it is difficult to collect all the training sequences at a time, thence they are processed block by block.
block_size = 0.5*1e6     

for dataset in datasets:  
  print("Processing {} data ...".format(dataset))

  wavPath = dataPath+dataset+'_mixed_wav/'
  batchPath = dataPath+dataset+'_batch/'
  if not os.path.isdir(batchPath):
    os.makedirs(batchPath)

  wavFiles = fnmatch.filter(os.listdir(wavPath),'*.wav')
  shuWavIndx = list(range(len(wavFiles)))
  np.random.shuffle(shuWavIndx)
   
  wavIndx = 0    
  batchIndx = 0
  while wavIndx<len(wavFiles):       
      nb_sequence = np.empty((int(block_size),time_steps,(n_channels+1)*2))	
      seqIndx = 0

      # Collect sequences of one block  
      while wavIndx<len(wavFiles):
        rate,s = scipy.io.wavfile.read(wavPath+wavFiles[shuWavIndx[wavIndx]])         
        if len(s.shape) == 2:
          if s.shape[0] > s.shape[1]:
            s = np.transpose(s)
        f, t, S = signal.stft(s,nperseg=ft_len,noverlap=ft_overlap)        
        S = np.transpose(S,(1,2,0))
        fra_num = S.shape[1]
        if seqIndx+len(range(0,fra_num-time_steps,step_inc))*fre_num>block_size:
          break
        for fra in range(0,fra_num-time_steps,step_inc):
          nb_sequence[seqIndx:seqIndx+fre_num,:,0:(n_channels+1)*2:2] = np.real(S[:,fra:fra+time_steps,:])
          nb_sequence[seqIndx:seqIndx+fre_num,:,1:(n_channels+1)*2:2] = np.imag(S[:,fra:fra+time_steps,:])				
          seqIndx += fre_num
        wavIndx += 1

      # Shuffle sequences and extract batch
      shuSeqIndx = list(range(seqIndx))
      np.random.shuffle(shuSeqIndx)
      for i in range(0,seqIndx-batch_size,batch_size): 
        batch = np.empty((batch_size,time_steps,(n_channels+1)*2))
        for j in range(batch_size):
          batch[j,] = nb_sequence[shuSeqIndx[i+j],]	

        Xr = batch[:,:,ref_channel*2:(ref_channel+1)*2]
        cln = batch[:,:,-2:]

		##### compute training targets, i.e. mrm, cirm and cc #### and normalize sequence #########	
        mrm = np.sqrt(np.square(cln).sum(axis=2))/np.sqrt(np.square(Xr).sum(axis=2))
        mrm = (mrm>=1)+((mrm<1)*mrm)

        Y2 = np.square(Xr).sum(axis=2)
        M = np.empty((batch_size,time_steps,2))
        M[:,:,0] = (Xr*cln).sum(axis=2)/Y2
        M[:,:,1] = (Xr[:,:,0]*cln[:,:,1]-Xr[:,:,1]*cln[:,:,0])/Y2
        M = -100*(M<=-100)+M*(M>-100)
        cirm = 10*(1-np.exp(-0.1*M))/(1+np.exp(-0.1*M))  

        mu = np.sqrt(np.square(Xr).sum(axis=2)).mean(axis=1)
        X = batch[:,:,:n_channels*2]/mu.reshape(batch_size,1,1) 
        cln = cln/mu.reshape(batch_size,1,1)       
		
        # save one batch        
        np.savez(batchPath+'batch'+str(batchIndx)+'.npz',X=np.float32(X),cln=np.float32(cln),mrm=np.float32(mrm),cirm=np.float32(cirm))
        batchIndx += 1
      del nb_sequence
      print("{}/{} wav files have been processed".format(wavIndx,len(wavFiles)))

  print("Number of batchs for {}: {}".format(dataset,batchIndx)) 


        
  



