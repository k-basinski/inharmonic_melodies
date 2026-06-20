from os import listdir, system
from os.path import isdir

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from librosa.onset import onset_detect
from scipy import signal




def cut_samples(signal, sample_start, sample_stop):
    return signal[sample_start:sample_stop]


def empty_output_dir(dir_path):
    dir_contents = listdir(dir_path)
    for f in dir_contents:
        system(f'rm {dir_path}/{f}')

def create_dir(dir_path):
    if not isdir(dir_path):
        system(f'mkdir {dir_path}')

def rectified_first_derivative(signal: np.ndarray, channel: int = 0):
    """Calculates half-rectified first-order derivative of a single channel of audio (left by default).

    Parameters
    ----------
    signal : array of float, shape (n_samples, channels)
        Input signal.
    channel : int
        Audio channel. 0 for left, 1 for right. Defaults to 0.

    Returns
    -------
    df_wave : array of float, shape (n_samples, )
        A half-rectified, first order derivative of signal
    """
    single_channel = signal[:, channel]
    df_wave = np.abs(np.diff(single_channel, n=1))
    return df_wave


def plot_peaks(signal, peaks):
    # get minima and maxima of signal
    min_amp, max_amp = np.min(signal), np.max(signal)

    # plot signal
    plt.plot(signal, c='blue')

    # plot transients
    plt.vlines(x=peaks, ymin=min_amp, ymax=max_amp, colors='red')

    plt.show()


def find_transients(signal, sample_rate):
    """Finds transients in wave."""
    peaks = onset_detect(y=signal, sr=sample_rate, units='samples', backtrack=True)
    min_interonset = float(np.min(np.diff(peaks))) // sample_rate
    print(f'{len(peaks)} detected, minimum interonset: {min_interonset} seconds.')
    return peaks


def cut_apart(signal, points, sample_rate, output_folder, output_file, crossfade_pad=0.05):
    s = 0
    i = 0
    pad_length = int(crossfade_pad*sample_rate) # extra-samples for crossfade
    for point in points:
        wave_cut = cut_samples(signal, s, point+pad_length)
        sf.write(f'{output_folder}{output_file}/{output_file}_{i:04d}.wav', wave_cut, sample_rate)
        s = point
        i += 1

def file_sorter(fname):
    fname_exploded = fname.split('_')
    num = fname_exploded[2][:-4]
    return int(num)

def files_to_glue(input_folder, control_folder=None):
    # get filenames
    files_in_input_folder = listdir(input_folder)
    # keep only wav files
    files_to_glue = [f for f in files_in_input_folder if f[-3:] == 'wav']
    files_to_glue.sort(key=file_sorter)

    if control_folder is not None:
        control_input_folder = listdir(control_folder)
        # keep only wav
        control_files = [f for f in control_input_folder if f[-3:] == 'wav']
        control_files.sort(key=file_sorter)
        control_lengths = []
        for f in control_files:
            w, _ = sf.read(control_folder + '/' + f)
            control_lengths.append(len(w))

    waves_to_glue = []
    for i,f in enumerate( files_to_glue):
        if f[-3:] == 'wav':
            w, _ = sf.read(input_folder + '/' + f)
            if control_folder is not None:
                if len(w) < control_lengths[i]:
                    w_trimmed = w[:control_lengths[i]]
                elif len(w) > control_lengths[i]:
                    w_trimmed = np.pad(w, (0, len(w) - control_lengths[i]))
                else:
                    w_trimmed = w
                waves_to_glue.append(w_trimmed)
            else:
                waves_to_glue.append(w[:])
    return waves_to_glue

def glue_together(waves_to_glue, sample_rate, output_folder, crossfades=True, xfade_length=None, apply_highpass=200):
    if crossfades:
        from maad.util import crossfade_list
        wave_out = crossfade_list(waves_to_glue, sample_rate, xfade_length)
    else:
        wave_out = np.concat(waves_to_glue)

    # filter to remove artifacts
    if apply_highpass is not None:
        sos = signal.butter(4, 200, 'hp', fs=sample_rate, output='sos')
        wave_out = signal.sosfilt(sos, wave_out)

    # check if song is longer than 120 sec
    # if so, cut length to 120 sec with fadeout
    max_length_samples = 90 * sample_rate
    if len(wave_out) > max_length_samples:
        wave_out = wave_out[:max_length_samples]
        # apply fade
        

    sf.write(output_folder + '_glued.wav', wave_out, sample_rate)