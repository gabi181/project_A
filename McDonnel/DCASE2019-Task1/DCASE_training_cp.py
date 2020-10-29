# %%

# select a GPU
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# %%
# import tensorflow as tf
# gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.2)
# # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
# from keras import backend as K
# import tensorflow as tf
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.2
# session = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(session)

import tensorflow as tf
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
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
from DCASE_plots import accuracy_plot

from time import gmtime, strftime
from datetime import datetime
import pytz
from pathlib import Path

print("Librosa version = ", librosa.__version__)
print("Pysoundfile version = ", sound.__version__)
print("keras version = ", keras.__version__)
print("tensorflow version = ", tensorflow.__version__)

# %%

# preprocess = ''  # original DCASE data.
# preprocess = 'filtered'  # filtered DCASE data.
preprocess = 'cut_length_3'  # cut DCASE data in length 1 sec
# preprocess = 'decimate_3'
# preprocess = 'placed_speaker_prop_2'
# preprocess = 'filtered_speaker_2_cut_3_dec_3_mono'
# preprocess = 'filtered_speaker_1-5_cut_3_dec_3'
# preprocess = 'filtered_speaker_2_cut_3_dec_3'
# preprocess = 'filtered_speaker_2-5_cut_3_dec_3'

length_of_segment = 3

for preprocess in ['cut_length_3']:

    # data_source = 'DCASE'
    # data_source = 'rafael'
    data_source = 'airport_str_traf'

    # csv_train = 'fold1_train.csv'
    # csv_train = 'in-air_out-str_traf_train.csv'
    csv_train = 'fold1_train_80-20.csv'
    # csv_val = 'fold1_evaluate.csv'
    # csv_val = 'in-air_out-str_traf_evaluate.csv'
    csv_val = 'fold1_test_80-20.csv'

    dec_factor = 1

    ThisPath = '../../data/' + preprocess + '_' + data_source + '/'
    # ThisPathVal = '../../data/' + 'cut_length_3_rafael_16667/'
    TrainFile = ThisPath + 'evaluation_setup/' + csv_train
    ValFile = ThisPath + 'evaluation_setup/' + csv_val
    if data_source in ['rafael']:
        sr = 16667
    elif data_source in ['DCASE', 'airport_str_traf']:
        sr = int(48000 / dec_factor)
    num_audio_channels = 2  #2

    SampleDuration = length_of_segment  # 10
    length_of_segment = length_of_segment + 1
    # log-mel spectrogram parameters
    NumFreqBins = 128  # effects on mel freq resolution.
    NumFFTPoints = 2048  # length of the window that ffted. (bigger window better freq resolution worse time resolution)
    HopLength = int(NumFFTPoints / 2)  # the gap between start of two adjacent windows
    NumTimeBins = int(np.ceil(SampleDuration * sr / HopLength))  # size of time dimension.
    # NumTimeBinsVal = int(np.ceil(SampleDuration * srVal / HopLength))  # size of time dimension.

    # training parameters
    max_lr = 0.1
    batch_size = 32  # filtered+normal = 8, cut_length_1 = 32, decimated = 16
    num_epochs = 510
    mixup_alpha = 0.4
    crop_length = 90  # cut_length_1 = 30, normal+filtered = 400, dec_3 = 100

#    # %%

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


#    # %%

    # load wav files and get log-mel spectrograms, deltas, and delta-deltas

    # If the data has already analyzed and saved, then load it. Else analyze and save it.

    p = Path('.')
    saved_vectors_path = p.resolve() / 'saved_vectors'

    # Change paths or file names if needed!

    LM_train_npy = saved_vectors_path / (preprocess + '_' + data_source + '_LM_train.npy')
    LM_val_npy = saved_vectors_path / (preprocess + '_' + data_source + '_LM_val.npy')
    y_train_npy = saved_vectors_path / (preprocess + '_' + data_source + '_' + str(NumClasses) + '_y_train.npy')
    y_val_npy = saved_vectors_path / (preprocess + '_' + data_source + '_' + str(NumClasses) + '_y_val.npy')

    if not saved_vectors_path.exists():
        saved_vectors_path.mkdir()

    if LM_train_npy.exists() and LM_val_npy.exists() and y_train_npy.exists() and y_val_npy.exists():
        print('loading', LM_train_npy)
        LM_train = np.load(LM_train_npy)
        print('loading', LM_val_npy)
        LM_val = np.load(LM_val_npy)
        y_train = np.load(y_train_npy)
        y_val = np.load(y_val_npy)
        print('the files have been loaded!')

    else:
        print('Anlyzing new data..')
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

        print('saving the analysis')

        np.save(LM_train_npy, LM_train)
        np.save(LM_val_npy, LM_val)
        np.save(y_train_npy, y_train)
        np.save(y_val_npy, y_val)

#    # %%

    # create and compile the model
    model = model_resnet(NumClasses,
                         input_shape=[NumFreqBins, None, 3 * num_audio_channels],
                         num_filters=24,
                         wd=1e-3)
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=max_lr, decay=0, momentum=0.9, nesterov=False),
                  metrics=['accuracy'])

    model.summary()

#    # %%

    # set learning rate schedule
    lr_scheduler = LR_WarmRestart(nbatch=np.ceil(LM_train.shape[0] / batch_size), Tmult=2,
                                  initial_lr=max_lr, min_lr=max_lr * 1e-4,
                                  epochs_restart=[3.0, 7.0, 15.0, 31.0, 63.0, 127.0, 255.0])

    callbacks = [lr_scheduler]
#    # %%
    # create data generator
    TrainDataGen = MixupGenerator(LM_train,
                                  y_train,
                                  batch_size=batch_size,
                                  alpha=mixup_alpha,
                                  crop_length=crop_length)()
#    # %%
    # train the model

    history = model.fit(TrainDataGen,
                        validation_data=(LM_val, y_val),
                        epochs=num_epochs,
                        verbose=1,
                        workers=4,
                        max_queue_size=100,
                        callbacks=callbacks,
                        steps_per_epoch=np.ceil(LM_train.shape[0] / batch_size)
                        )


    tz = pytz.timezone('Asia/Jerusalem')
    israel_datetime = datetime.now(tz)

    model_name = preprocess + '_' + data_source + '_' + str(NumClasses) + '_' + 'class' + '_' + israel_datetime.strftime("%d-%m_%H-%M")
    model.save('./models/' + model_name + '.h5')

    model_doc_file = open('./models/Readme.txt', 'a')
    model_doc_file.write('\n' + model_name + '\n' +
                         'max_lr= ' + str(max_lr) +
                         '\tbatch_size= ' + str(batch_size) +
                         '\tnum_epochs= ' + str(num_epochs) +
                         '\tcrop_length= ' + str(crop_length) + '\n')
    model_doc_file.close()

    path = p.resolve() / 'accuracy_fig'
    data_details = path, model_name
    accuracy_plot(history, data_details)