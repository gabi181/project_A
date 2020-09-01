# %%

# select a GPU
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# %%
import tensorflow as tf
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.2)
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
from keras import backend as K
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
# %%

# imports
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import librosa
import soundfile as sound

import keras
import tensorflow
from keras.optimizers import SGD

from DCASE2019_network import model_resnet
from DCASE_training_functions import LR_WarmRestart, MixupGenerator

print("Librosa version = ", librosa.__version__)
print("Pysoundfile version = ", sound.__version__)
print("keras version = ", keras.__version__)
print("tensorflow version = ", tensorflow.__version__)

# %%

WhichTask = '1a'
# WhichTask = '1b'
# WhichTask = '1c'

if WhichTask == '1a':
    ThisPath = '../../data/filtered_TAU-urban-acoustic-scenes-2019-development/'
    TrainFile = ThisPath + 'evaluation_setup/fold1_train.csv'
    ValFile = ThisPath + 'evaluation_setup/fold1_evaluate.csv'
    sr = 48000
    num_audio_channels = 2
elif WhichTask == '1b':
    ThisPath = '../Task1b/'
    TrainFile = ThisPath + 'evaluation_setup/fold1_train.csv'
    ValFile = ThisPath + 'evaluation_setup/fold1_evaluate.csv'
    sr = 44100
    num_audio_channels = 1
elif WhichTask == '1c':
    ThisPath = '../Task1c/'
    TrainFile = ThisPath + 'evaluation_setup/fold1_train.csv'
    sr = 44100
    num_audio_channels = 1

SampleDuration = 10

# log-mel spectrogram parameters
NumFreqBins = 128
NumFFTPoints = 2048
HopLength = int(NumFFTPoints / 2)
NumTimeBins = int(np.ceil(SampleDuration * sr / HopLength))

# training parameters
max_lr = 0.1
batch_size = 1  # 32
num_epochs = 5   # 510
mixup_alpha = 0.4
crop_length = 400

# %%

# load filenames and labels
dev_train_df = pd.read_csv(TrainFile, sep='\t', encoding='ASCII')
dev_val_df = pd.read_csv(ValFile, sep='\t', encoding='ASCII')
wavpaths_train = dev_train_df['filename'].tolist()
wavpaths_val = dev_val_df['filename'].tolist()
y_train_labels = dev_train_df['scene_label'].astype('category').cat.codes.values
y_val_labels = dev_val_df['scene_label'].astype('category').cat.codes.values

ClassNames = np.unique(dev_train_df['scene_label'])
NumClasses = len(ClassNames)

y_train = keras.utils.to_categorical(y_train_labels, NumClasses)
y_val = keras.utils.to_categorical(y_val_labels, NumClasses)


# %%

# load wav files and get log-mel spectrograms, deltas, and delta-deltas
def deltas(X_in):
    X_out = (X_in[:, :, 2:, :] - X_in[:, :, :-2, :]) / 10.0
    X_out = X_out[:, :, 1:-1, :] + (X_in[:, :, 4:, :] - X_in[:, :, :-4, :]) / 5.0
    return X_out


LM_train = np.zeros((len(wavpaths_train), NumFreqBins, NumTimeBins, num_audio_channels), 'float32')
for i in range(len(wavpaths_train)):
    stereo, fs = sound.read(ThisPath + wavpaths_train[i], stop=SampleDuration * sr)
    for channel in range(num_audio_channels):
        if len(stereo.shape) == 1:
            stereo = np.expand_dims(stereo, -1)
        LM_train[i, :, :, channel] = librosa.feature.melspectrogram(stereo[:, channel],
                                                                    sr=sr,
                                                                    n_fft=NumFFTPoints,
                                                                    hop_length=HopLength,
                                                                    n_mels=NumFreqBins,
                                                                    fmin=0.0,
                                                                    fmax=sr / 2,
                                                                    htk=True,
                                                                    norm=None)

LM_train = np.log(LM_train + 1e-8)
LM_deltas_train = deltas(LM_train)
LM_deltas_deltas_train = deltas(LM_deltas_train)
LM_train = np.concatenate((LM_train[:, :, 4:-4, :], LM_deltas_train[:, :, 2:-2, :], LM_deltas_deltas_train), axis=-1)

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

# %%

# create and compile the model
model = model_resnet(NumClasses,
                     input_shape=[NumFreqBins, None, 3 * num_audio_channels],
                     num_filters=24,
                     wd=1e-3)
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=max_lr, decay=0, momentum=0.9, nesterov=False),
              metrics=['accuracy'])

model.summary()

# %%

# set learning rate schedule
lr_scheduler = LR_WarmRestart(nbatch=np.ceil(LM_train.shape[0] / batch_size), Tmult=2,
                              initial_lr=max_lr, min_lr=max_lr * 1e-4,
                              epochs_restart=[3.0, 7.0, 15.0, 31.0, 63.0, 127.0, 255.0, 511.0])
callbacks = [lr_scheduler]

# create data generator
TrainDataGen = MixupGenerator(LM_train,
                              y_train,
                              batch_size=batch_size,
                              alpha=mixup_alpha,
                              crop_length=crop_length)()

# train the model
history = model.fit_generator(TrainDataGen,
                              validation_data=(LM_val, y_val),
                              epochs=num_epochs,
                              verbose=1,
                              workers=4,
                              max_queue_size=100,
                              callbacks=callbacks,
                              steps_per_epoch=np.ceil(LM_train.shape[0] / batch_size)
                              )

# %%

model.save('DCASE_' + WhichTask + '_Task_development_1.h5')
