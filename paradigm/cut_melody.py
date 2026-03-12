from os import listdir, system
from os.path import isdir

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from librosa.onset import onset_detect




def cut_samples(signal, sample_start, sample_stop):
    return signal[sample_start:sample_stop, :]


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


def cut_apart(signal, points, sample_rate, output_folder, output_file):
    s = 0
    i = 0
    for point in points:
        wave_cut = cut_samples(signal, s, point)
        sf.write(f'{output_folder}{output_file}/{output_file}_{i:04d}.wav', wave_cut, sample_rate)
        s = point
        i += 1


def glue_together(input_folder, sample_rate):
    # get filenames
    files_to_glue = listdir(input_folder)
    files_to_glue.sort()

    waves_to_glue = []
    for f in files_to_glue:
        if f[-3:] == 'wav':
            w, _ = sf.read(input_folder + '/' + f)
            waves_to_glue.append(w)
    wave_out = np.concat(waves_to_glue)
    sf.write(input_folder + '_glued.wav', wave_out, sample_rate)


def main():
    # config
    input_folder = 'paradigm/soundpool/'
    input_files = [f'harm_{i}' for i in range(1, 33)]
    output_folder = 'paradigm/soundpool/cut_wav/'

    # for each song...
    for input_file in input_files:
        print(f"Processing input {input_file}.wav")

        # make folder if it doesn't exist
        create_dir(output_folder + input_file)

        # clear previous output
        empty_output_dir(output_folder + input_file)

        # read input
        wave, sr = sf.read(input_folder + input_file + '.wav')

        # calculate transients
        transients = find_transients(wave[:, 0], sr)
        # plot_peaks(wave[:, 0], transients)

        # cut apart
        cut_apart(wave, transients, sr, output_folder, input_file)

        # try to glue together (quality check)
        glue_together(output_folder+input_file, sr)


if __name__ == "__main__":
    main()
