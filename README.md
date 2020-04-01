This repository provides code for the TASLP submission:

- Xiaofei Li and Radu Horaud. Narrow-band deep filtering for multichannel speech enhancement. https://hal.inria.fr/hal-02378413/file/multichannel_lstm.pdf

and for

- Xiaofei Li and Radu Horaud. Multichannel Speech Enhancement Based on Time-frequency Masking Using Subband Long Short-Term Memory. WASPAA 2019.

For more details, please refer to: https://team.inria.fr/perception/research/mse-lstm/

Preparation:

	The codes are developed using MATLAB R2017a and Python3. The following dataset and tools should be first prepared:

	1. CHiME3 dataset. Actually only the multichannel real recordings are used.

	2. tensorflow, Keras, SciPy

	3. To evaluate the speech enhancement performance, please download and setup the matlab toolkits for PESQ, STOI, SDR and (normalized) SRMR.

Usage:

	1. Mixed data generation for train/validation/test

	move mix_bthplusbackground.m into CHiME3/tools/simulation/, and run it.

	2. Extract subband sequences for train/validation

	run sequence_generation.py

	3. Train

	run train.py

	4. Speech enhancement on test data using trained models

	run prediction.py

	5. Speech enhancement performance evaluation

	run evaluation.m


Note:

	1. For each of the above steps, please set the dataPath in the script. All the data are stored in such dataPath. About 100 GB of disk space is required. 

	2. In ref_models/, model weights for four networks trained by the author are provided.

	3. In ref_se_result/, performance scores obtained by the author are provided. Please refer to evaluation.m for the data structure.

 
Author: Xiaofei Li, Westlake University, China and INRIA Grenoble Rhone-Alpes, France  
 
