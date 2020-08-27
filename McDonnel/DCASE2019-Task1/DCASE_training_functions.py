import keras
from keras import backend as K
import numpy as np
import threading
import soundfile as sound
import librosa

#for implementing warm restarts in learning rate
class LR_WarmRestart(keras.callbacks.Callback):
    
    def __init__(self,nbatch,initial_lr,min_lr,epochs_restart,Tmult):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.epochs_restart = epochs_restart
        self.nbatch = nbatch
        self.currentEP=0
        self.startEP=0
        self.Tmult=Tmult
        
    def on_epoch_begin(self, epoch, logs={}):
        if epoch+1<self.epochs_restart[0]:
            self.currentEP = epoch
        else:
            self.currentEP = epoch+1
            
        if np.isin(self.currentEP,self.epochs_restart):
            self.startEP=self.currentEP
            self.Tmult=2*self.Tmult
        
    def on_epoch_end(self, epochs, logs={}):
        lr = K.get_value(self.model.optimizer.lr)
        print ('\nLearningRate:{:.6f}'.format(lr))
    
    def on_batch_begin(self, batch, logs={}):
        pts = self.currentEP + batch/self.nbatch - self.startEP
        decay = 1+np.cos(pts/self.Tmult*np.pi)
        lr = self.min_lr+0.5*(self.initial_lr-self.min_lr)*decay
        K.set_value(self.model.optimizer.lr,lr)

        
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()
        
        
def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


"""     
This function extracts delta features from audio input.
"""

def deltas(X_in):
    X_out = (X_in[:, :, 2:, :] - X_in[:, :, :-2, :]) / 10.0
    X_out = X_out[:, :, 1:-1, :] + (X_in[:, :, 4:, :] - X_in[:, :, :-4, :]) / 5.0
    return X_out

def concatenated_LM_delta_deltadelta(wavpaths, indecies ,audio_params):
    NumFreqBins, NumTimeBins, num_audio_channels, SampleDuration, sr, NumFFTPoints, HopLength = audio_params
    LM = np.zeros((len(wavpaths), NumFreqBins, NumTimeBins, num_audio_channels), 'float32')
    for i in indecies:
        stereo, fs = sound.read(wavpaths[i], stop=SampleDuration * sr)
        for channel in range(num_audio_channels):
            if len(stereo.shape) == 1:
                stereo = np.expand_dims(stereo, -1)
            LM[i, :, :, channel] = librosa.feature.melspectrogram(stereo[:, channel],
                                                                        sr=sr,
                                                                        n_fft=NumFFTPoints,
                                                                        hop_length=HopLength,
                                                                        n_mels=NumFreqBins,
                                                                        fmin=0.0,
                                                                        fmax=sr / 2,
                                                                        htk=True,
                                                                        norm=None)

    LM = np.log(LM + 1e-8)
    LM_deltas = deltas(LM)
    LM_deltas_deltas = deltas(LM_deltas)
    LM = np.concatenate((LM[:, :, 4:-4, :], LM_deltas[:, :, 2:-2, :], LM_deltas_deltas), axis=-1)
    return LM


class MixupGenerator():
    def __init__(self, X_paths, y, audio_params, batch_size=32, alpha=0.2, shuffle=True, crop_length=400): #datagen=None):
        self.X_paths = X_paths
        self.y = y
        self.audio_params = audio_params
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_paths)
        self.lock = threading.Lock()
        self.NewLength = crop_length
        self.swap_inds = [1,0,3,2,5,4]
        
    def __iter__(self):
        return self
    
    @threadsafe_generator
    def __call__(self):
        with self.lock:
            while True:
                indexes = self.__get_exploration_order()
                itr_num = int(len(indexes) // (self.batch_size * 2))

                for i in range(itr_num):
                    batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                    X, y = self.__data_generation(batch_ids)

                    yield X, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        # _, h, w, c = self.X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)


        X1 = concatenated_LM_delta_deltadelta(self.X_paths, batch_ids[:self.batch_size], self.audio_params)
        X2 = concatenated_LM_delta_deltadelta(self.X_paths, batch_ids[self.batch_size:], self.audio_params)

        for j in range(X1.shape[0]):
            StartLoc1 = np.random.randint(0,X1.shape[2]-self.NewLength)
            StartLoc2 = np.random.randint(0,X2.shape[2]-self.NewLength)

            X1[j,:,0:self.NewLength,:] = X1[j,:,StartLoc1:StartLoc1+self.NewLength,:]
            X2[j,:,0:self.NewLength,:] = X2[j,:,StartLoc2:StartLoc2+self.NewLength,:]
            
            if X1.shape[-1]==6:
                #randomly swap left and right channels 
                if np.random.randint(2) == 1:
                    X1[j,:,:,:] = X1[j:j+1,:,:,self.swap_inds]
                if np.random.randint(2) == 1:
                    X2[j,:,:,:] = X2[j:j+1,:,:,self.swap_inds]
            
            
        X1 = X1[:,:,0:self.NewLength,:]
        X2 = X2[:,:,0:self.NewLength,:]
        
        X = X1 * X_l + X2 * (1.0 - X_l)

        if isinstance(self.y, list):
            y_app = []

            for y_ in self.y:
                y1 = y_[batch_ids[:self.batch_size]]
                y2 = y_[batch_ids[self.batch_size:]]
                y_app.append(y1 * y_l + y2 * (1.0 - y_l))
        else:
            y1 = self.y[batch_ids[:self.batch_size]]
            y2 = self.y[batch_ids[self.batch_size:]]
            y_app = y1 * y_l + y2 * (1.0 - y_l)

        return X, y_app

