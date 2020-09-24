"""
This function filters wav files in the two way radio filter that has been given from Rafael.
The function saves the filtered file, in a subdirectory named filtered_audio in the parent directory.
Inputs:
p - path (from pathlib.Path module) to wav file.
show (Boolean) - prints plots of the original data in time and frequency plane, filter and filtered data.
sound (Boolean) - plays the original file and the filtered one.
"""


def filter_wav_file(data, fs, show=False):

    import matplotlib.pyplot as plt
    from scipy.fftpack import rfft, irfft
    import numpy as np

    # Read, plot, and listen the original file.
    if show:
        plt.plot(data.T[0])
        plt.show()

    F_data = rfft(data.T)  # calculate fourier transform (complex numbers list)

    if show:
        plt.plot(abs(F_data[0]))
        plt.xlim(0, 96000)
        plt.show()

    # Building the filter, and mirror it such that the middle is the axis.

    N = len(F_data[0])

    radio_filter = create_two_way_radio_filter(f_length=N, data_sec=len(data.T[0])/fs, show=show)
    # Filter the fft of the original file.

    channel1 = (F_data[0]*radio_filter)
    channel2 = (F_data[1]*radio_filter)

    # Build 2 channels signal.

    res = np.concatenate((channel1, channel2))
    res = res.reshape(2, -1)

    if show:
        plt.plot(abs(res[0]), 'r')
        plt.xlim(0, 48000)
        plt.show()

    filtered_data = irfft(res)

    if show:
        plt.plot(filtered_data[0], 'r')
        plt.show()


    return filtered_data.T



"""
 In this function, we are creating a filter that mimic the effect that
 RAFAEL's two way radio does on audio.
 To asses the filter function the next stages has been done:
 1. few chirps was recorded on the two way radio. (a perfect chirp should look
 like a rectangle.)
 2. the recordings had been demodulated.
 3. FFT
 4. plot
 the function had been assesed from the plots.
 inputs:
 f_length - the length of the frequency axis.
 data_sec - scales the filter from HZs to the samples scale.
 show - show=1 plot the filter. default value is 0.
 output:
 Two way radio filter in signal's samples scale.
 """


def create_two_way_radio_filter(f_length=8500, data_sec=10, show=False):

    import numpy as np
    import matplotlib.pyplot as plt

    F = np.array(range(f_length))
    window = np.zeros(f_length)

    zer_1 = int(150 * data_sec)
    lin_up = int(750 * data_sec)
    const = int(2750 * data_sec)
    lin_down = int(3500 * data_sec)
    # zer_2 = end

    m_lin_up = 1 / (lin_up - zer_1)
    b_lin_up = 1 - m_lin_up * lin_up

    m_lin_down = -1 / (lin_down - const)
    b_lin_down = 1 - m_lin_down * const

    window[zer_1:lin_up] = m_lin_up*F[zer_1:lin_up] + b_lin_up
    window[lin_up:const] = 1
    window[const:lin_down] = m_lin_down*F[const:lin_down] + b_lin_down

    if show:
        plt.plot(window)
        plt.xlim(0, 48000)  # normal hearing frequencies are up to 8,000 Hz.
        plt.show()

    return window


"""
This function cut src_file to segments with length (seconds) of length_of_segments.
The remainder is thrown.
The function creates 10 files from the original file data.
"""


def cut_n_write_wav_file(data, fs, src_file, dest_dir, length_of_segment):
    import soundfile as sf
    from pathlib import Path
    import numpy as np

    time_len = len(data) / fs
    num_of_segments = int(np.floor(time_len / length_of_segment))

    for seg in range(num_of_segments):
        cut_data = data[seg * length_of_segment * fs:(seg + 1) * length_of_segment * fs]
        sf.write(dest_dir / (src_file.stem + '_' + str(seg) + src_file.suffix), cut_data, fs)


"""
This function read, decimate in factor 3 and save a wav file.
"""
#%%


def decimate_wav_file(data, fs, dec_factor):
    from scipy import signal

    dec_data = signal.decimate(data, dec_factor, axis=0)
    dec_fs = int(fs/dec_factor)
    return dec_data, dec_fs


#%%
"""
This function place a speaker into audio file, the proportion is determined by speaker_proportion.
"""
def place_speaker(src_data, speaker, speaker_proportion):

    mean_speaker = abs(speaker).mean()
    mean_src = abs(src_data).mean()
    norm = speaker_proportion * mean_src / mean_speaker
    if speaker.shape[0] > src_data.shape[0]:
        speaker = speaker[:src_data.shape[0], :]
    else:
        src_data = src_data[:speaker.shape[0], :]
    new_data = norm * speaker + src_data

    return new_data


