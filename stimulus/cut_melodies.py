# %%
from os import listdir

import numpy as np
from mido import MidiFile
import soundfile as sf

# load file
midi_fname = 'stimulus/midi/Partita_for_Solo_Flute/fp-1all.mid'
wav_fname = 'paradigm/soundpool/harm_1.wav'
output_folder = 'stimulus/cut_wav/test'
output_folder2 = 'stimulus/cut_wav/test2'

mid = MidiFile(midi_fname)
wave, sr = sf.read(wav_fname)


def to_samp(t):
    return int(t * sr)

def cut(wave, t_start, t_stop):
    start, stop = to_samp(t_start), to_samp(t_stop)
    return wave[start:stop, :]

def cut_samples(wave, t_start, len):
    return wave[t_start:t_start+len, :]

t = 0
i = 0
for sec in range(len(wave)//4800):
    wave_cut = cut_samples(wave, sec*48000, 48000)
    sf.write(f'{output_folder2}/test{i}.wav', wave_cut, sr)
    # print(t)
    i += 1
# %%
t = 0
i = 0
for msg in mid:
    if not msg.is_meta and t < 120: # if not meta-message
        wave_cut = cut(wave, t, t+msg.time)
        sf.write(f'{output_folder}/test{i}.wav', wave_cut, sr)
        # print(t)
        t += msg.time
        i += 1


# %%
# glue together
# get filenames
files_to_glue = listdir(output_folder)

waves_to_glue = []
for f in files_to_glue:
    if f[-3:] == 'wav':
        w, _ = sf.read(output_folder+'/'+f)
        waves_to_glue.append(w)


# %%
wave_out = np.concat(waves_to_glue)
sf.write(f'{output_folder}/glued.wav', wave_out, sr)