import numpy as np
import librosa
from Augmentation import add_noise, shift, stretch
import os
import pandas as pd

def InstantiateAttributes(dir_,path_patient_disease_list):
	#'''
	#    Extract feature from the Sound data. We extracted Mel-frequency cepstral coefficients( spectral 
	#    features ), from the audio data. Augmentation of sound data by adding Noise, streaching and shifting 
	#    is also implemented here. 40 features are extracted from each audio data and used to train the model.
	#    Args:
	#        dir_: Input directory to the Sound input file.
	#    Returns:
	#        X_: Array of features extracted from the sound file.
	#        y_: Array of target Labels.
	#'''
    X_=[]
    y_=[]
    COPD=[]
    copd_count=0
    
    patient_disease_list = pd.read_csv(path_patient_disease_list, sep= ";")

    for soundDir in (os.listdir(dir_)):
        if soundDir[-3:]=='wav'and soundDir[:3]!='103'and soundDir[:3]!='108'and soundDir[:3]!='115':           #Do not use "Asthma" and "LRTI" since there are very few instances of those.
        	##data_x, sampling_rate = librosa.load(dir_+soundDir,res_type='kaiser_fast')
            ##mfccs = np.mean(librosa.feature.mfcc(y=data_x, sr=sampling_rate, n_mfcc=40).T,axis=0) 
            ##X_.append(mfccs)
            ##y_.append(list(data[data['patient_id']==int(soundDir[:3])]['disease'])[0])
            ####
            
            ##https://www.statology.org/pandas-select-rows-based-on-column-values/
            #slask4 = df.loc[df['b'] == 7]
            #print(slask4)
            #print("=======")
            #print(slask4['a'])
            ##https://stackoverflow.com/questions/31536835/extract-value-from-single-row-of-pandas-dataframe
            #current_val = slask4['a'].values[0]
            #print("a_val: ",current_val)
            #new_val = current_val +3
            #print("new_val: ",new_val)

            #p = list(data[data['patient_id']==int(soundDir[:3])]['disease'])[0]
            current_row = patient_disease_list.loc[patient_disease_list['patient_id'] == int(soundDir[:3])]
            current_disease = current_row['disease'].values[0]
            #print("Current disease: ",current_disease)
            if (current_disease=='COPD'):
                if (soundDir[:3] in COPD) and copd_count<2:
                    data_x, sampling_rate = librosa.load(dir_+soundDir,res_type='kaiser_fast')
                    # 40 features are extracted from each audio data and used to train the model.
                    mfccs = np.mean(librosa.feature.mfcc(y=data_x, sr=sampling_rate, n_mfcc=40).T,axis=0) 
                    COPD.append(soundDir[:3])
                    copd_count+=1
                    X_.append(mfccs)
                    #y_.append(list(data[data['patient_id']==int(soundDir[:3])]['disease'])[0])
                    y_.append(current_disease)
                if (soundDir[:3] not in COPD):
                    data_x, sampling_rate = librosa.load(dir_+soundDir,res_type='kaiser_fast')
                    mfccs = np.mean(librosa.feature.mfcc(y=data_x, sr=sampling_rate, n_mfcc=40).T,axis=0) 
                    COPD.append(soundDir[:3])
                    copd_count=0
                    X_.append(mfccs)
                    #y_.append(list(data[data['patient_id']==int(soundDir[:3])]['disease'])[0])
                    y_.append(current_disease)
                
            if (current_disease!='COPD'):
                data_x, sampling_rate = librosa.load(dir_+soundDir,res_type='kaiser_fast')
                mfccs = np.mean(librosa.feature.mfcc(y=data_x, sr=sampling_rate, n_mfcc=40).T,axis=0) 
                X_.append(mfccs)
                #y_.append(list(data[data['patient_id']==int(soundDir[:3])]['disease'])[0])
                y_.append(current_disease)
            
                data_noise = add_noise(data_x,0.005)
                mfccs_noise = np.mean(librosa.feature.mfcc(y=data_noise, sr=sampling_rate, n_mfcc=40).T,axis=0) 
                X_.append(mfccs_noise)
                y_.append(current_disease)

                data_shift = shift(data_x,1600)
                mfccs_shift = np.mean(librosa.feature.mfcc(y=data_shift, sr=sampling_rate, n_mfcc=40).T,axis=0) 
                X_.append(mfccs_shift)
                y_.append(current_disease)

                data_stretch = stretch(data_x,1.2)
                mfccs_stretch = np.mean(librosa.feature.mfcc(y=data_stretch, sr=sampling_rate, n_mfcc=40).T,axis=0) 
                X_.append(mfccs_stretch)
                y_.append(current_disease)
                
                data_stretch_2 = stretch(data_x,0.8)
                mfccs_stretch_2 = np.mean(librosa.feature.mfcc(y=data_stretch_2, sr=sampling_rate, n_mfcc=40).T,axis=0) 
                X_.append(mfccs_stretch_2)
                y_.append(current_disease)

    X_ = np.array(X_)
    y_ = np.array(y_)
    
    return X_,y_