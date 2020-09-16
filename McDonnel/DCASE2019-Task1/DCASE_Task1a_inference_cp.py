# select a GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#%%

#imports
import numpy as np
import h5py
import scipy.io
from sklearn.metrics import confusion_matrix
import pandas as pd
from DCASE_plots import plot_confusion_matrix

from pathlib import Path


import librosa
import soundfile as sound
import keras
import tensorflow
print("Librosa version = ",librosa.__version__)
print("Pysoundfile version = ",sound.__version__)
print("keras version = ",keras.__version__)
print("tensorflow version = ",tensorflow.__version__)
#%%
# DataSource = ''  # original DCASE data.
DataSource = 'filtered_'  # filtered DCASE data.

model_version = 'DCASE_filtered_1a_Task_development_1_13-09-2020_22:03:36'

num_audio_channels = 2

# Change paths or file names if needed!
p = Path('.')
saved_vectors_path = p.resolve() / 'saved_vectors'
data_details = p, model_version, DataSource

#%%

#Task 1a dev validation set
ThisPath = '../../data/' + DataSource + 'TAU-urban-acoustic-scenes-2019-development/'
File = ThisPath + 'evaluation_setup/fold1_evaluate.csv'
sr = 48000
SampleDuration = 10
NumFreqBins = 128
NumFFTPoints = 2048
HopLength = int(NumFFTPoints/2)
NumTimeBins = int(np.ceil(SampleDuration*sr/HopLength))

#%%

#load filenames and labels
dev_test_df = pd.read_csv(File,sep='\t', encoding='ASCII')
wavpaths_val = dev_test_df['filename'].tolist()
ClassNames = np.unique(dev_test_df['scene_label'])
y_val_labels =  dev_test_df['scene_label'].astype('category').cat.codes.values

#%%

#swap codes for 2 and 1 to match the DCASE ordering of classes
a1=np.where(y_val_labels==2)
a2=np.where(y_val_labels==3)
y_val_labels.setflags(write=1)
y_val_labels[a1] = 3
y_val_labels[a2] = 2

#%%

# load wav files and get log-mel spectrograms, deltas, and delta-deltas

# If the data has already analyzed and saved, then load it. Else analyze and save it.

LM_val_name = saved_vectors_path / (DataSource + 'LM_val.npy')

if not saved_vectors_path.exists():
    saved_vectors_path.mkdir()

if LM_val_name.exists():
    print('loading', LM_val_name)
    LM_val = np.load(LM_val_name)
    print('the files have been loaded!')

else:
    print('Anlyzing new data..')
    def deltas(X_in):
        X_out = (X_in[:, :, 2:, :] - X_in[:, :, :-2, :]) / 10.0
        X_out = X_out[:, :, 1:-1, :] + (X_in[:, :, 4:, :] - X_in[:, :, :-4, :]) / 5.0
        return X_out



    LM_val = np.zeros((len(wavpaths_val), NumFreqBins, NumTimeBins, num_audio_channels), 'float32')
    for i in range(len(wavpaths_val)):
        stereo, fs = sound.read(ThisPath + wavpaths_val[i], stop=SampleDuration * sr)
        for channel in range(num_audio_channels):
            if len(stereo.shape) == 1:
                stereo = np.expand_dims(stereo, -1)
            LM_val[i, :, :, channel] = librosa.feature.melspectrogram(stereo[:, channel],
                                                                      sr=sr,
                                                                      n_fft=NumFFTPoints,
                                                                      hop_length=HopLength,
                                                                      n_mels=NumFreqBins,
                                                                      fmin=0.0,
                                                                      fmax=sr / 2,
                                                                      htk=True,
                                                                      norm=None)

    LM_val = np.log(LM_val + 1e-8)
    LM_deltas_val = deltas(LM_val)
    LM_deltas_deltas_val = deltas(LM_deltas_val)
    LM_val = np.concatenate((LM_val[:, :, 4:-4, :], LM_deltas_val[:, :, 2:-2, :], LM_deltas_deltas_val), axis=-1)

    print('saving the analysis')

    np.save(LM_val_name, LM_val)

#%%

#load and run the model
best_model = keras.models.load_model('./models/' + model_version + '.h5')
y_pred_val = np.argmax(best_model.predict(LM_val),axis=1)

#%%

#get metrics
Overall_accuracy = np.sum(y_pred_val==y_val_labels)/LM_val.shape[0]
print("overall accuracy: ", Overall_accuracy)

plot_confusion_matrix(y_val_labels, y_pred_val, ClassNames,normalize=True,title=None, data_details=data_details)

conf_matrix = confusion_matrix(y_val_labels,y_pred_val)
conf_mat_norm_recall = conf_matrix.astype('float32')/conf_matrix.sum(axis=1)[:,np.newaxis]
conf_mat_norm_precision = conf_matrix.astype('float32')/conf_matrix.sum(axis=0)[:,np.newaxis]
recall_by_class = np.diagonal(conf_mat_norm_recall)
precision_by_class = np.diagonal(conf_mat_norm_precision)
mean_recall = np.mean(recall_by_class)
mean_precision = np.mean(precision_by_class)

print("per-class accuracy (recall): ",recall_by_class)
print("per-class precision: ",precision_by_class)
print("mean per-class recall: ",mean_recall)
print("mean per-class precision: ",mean_precision)


