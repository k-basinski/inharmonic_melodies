import os
import numpy as np
import slab
import mne
import matplotlib.pyplot as plt
from mtrf.model import TRF
import pandas as pd
from glob import glob

##DEFINE STIMULUS AND RESPONSE
subject = 'pilot_04'
new_fs = 64

dur_segment = 5
len_segment = dur_segment * new_fs

wav_path = 'C:/Users/kornelia/Desktop/new_*.wav'
path_to_fif = f"C:/Users/ASUS/Desktop/magisterka/data/{subject}_epo.fif"

wav_files = sorted(glob(wav_path))

#stimulus
mvar_stimulus = []

for wf in wav_files:

    sound = slab.Sound(wf).channel(0)
    sound = sound.resample(int(sound.samplerate / 3))

    envelope = sound.envelope().resample(new_fs)
    stim_data = envelope.data.reshape(-1, 1)

    onsets = np.diff(stim_data, axis=0, prepend=np.zeros((1, 1)))
    onsets[onsets < 0] = 0

    #melodic_features - do dodania
    mvar_epoch = np.concatenate([stim_data, onsets, melodic_features], axis=1)
    mvar_stimulus.append(mvar_epoch)

#response
response = []

all_epochs_files = sorted(glob(path_to_fif))

for file in all_epochs_files:

    epochs = mne.read_epochs(file, preload=True)

    #tylko EEG - wywal oczy eog=['EXG1', 'EXG2'] !!!! do naprawy
    epochs.pick_types(eeg=True)

    for epoch in epochs.get_data():
        response.append(epoch.T)

#kontrolka
print("WAV:", len(wav_files))
print("EEG:", len(response))

print("Stim shape:", mvar_stimulus[0].shape)
print("Resp shape:", response[0].shape)

##ENVELOPE AND ONSETS VISUALIZATION - potem drugi będzie ale ładniejszy jak u Giovanniego
times = np.linspace(0, len(mvar_stimulus[0])/new_fs, len(mvar_stimulus[0]))
plt.plot(times, mvar_stimulus[0][:,0], label='Envelope', alpha=0.7)
plt.plot(times, mvar_stimulus[0][:,1], label='Onsets', alpha=0.7)
plt.xlim(5, 10)
plt.xlabel('Time [s]')
plt.ylabel('Ammplitude [a.u.]')
plt.legend()

##STANDARIZATION
for i in range(len(mvar_stimulus)):

    s = mvar_stimulus[i]
    r = response[i]

    #stimulus
    s_std = s.std(axis=0)
    s_std[s_std == 0] = 1
    s = (s - s.mean(axis=0)) / s_std

    #eeg
    r_std = r.std(axis=0)
    r_std[r_std == 0] = 1
    r = (r - r.mean(axis=0)) / r_std

    #align
    min_len = min(len(s), len(r))

    mvar_stimulus[i] = s[:min_len]
    response[i] = r[:min_len]

#NaN cleanup
mvar_stimulus = [np.nan_to_num(s) for s in mvar_stimulus]
response = [np.nan_to_num(r) for r in response]

##PREPARING TRAINING AND TESTING DATASETS
stim_train = mvar_stimulus[:-2]
resp_train = response[:-2]

stim_test = mvar_stimulus[-2:]
resp_test = response[-2:]

stim_segments = []
resp_segments = []

for s_full, r_full in zip(stim_train, resp_train):

    n_seg = len(s_full) // len_segment

    if n_seg < 2:
        continue

    s_crop = s_full[:n_seg * len_segment]
    r_crop = r_full[:n_seg * len_segment]

    stim_segments.extend(np.array_split(s_crop, n_seg))
    resp_segments.extend(np.array_split(r_crop, n_seg))

print("Segments:", len(stim_segments))

##MODEL VALIDATION
m_fwd_trf = TRF()

tmin, tmax = 0, 0.35

regularization = [0.1, 1, 10, 100, 1000]  #lambda do testowania potem użyjemy np.logspace(-2, 6, 10)

m_fwd_trf.train(
    stim_segments,
    resp_segments,
    new_fs,
    tmin,
    tmax,
    regularization,
    k=3)

#PORÓWNANIE MODELI A I AM
#Model A == Tylko akustyka - pierwsze 2 kolumny
trf_A = TRF()
trf_A.train([s[:, :2] for s in stim_segments], resp_segments, new_fs, tmin, tmax, regularization)
_, r_A = trf_A.predict([s[:, :2] for s in stim_test], resp_test)

#Model AM == Akustyka + Melodyka - wszystkie 6 kolumn)
trf_AM = TRF()
trf_AM.train(stim_segments, resp_segments, new_fs, tmin, tmax, regularization)
_, r_AM = trf_AM.predict(stim_test, resp_test)

# Wynik
print(f"Średnia korelacja A: {np.nanmean(r_A)}")
print(f"Średnia korelacja AM: {np.nanmean(r_AM)}")
print(f"Zysk z oczekiwań melodycznych (Delta r): {np.nanmean(r_AM) - np.nanmean(r_A)}")

##VISUALIZATION
#Music score of a segment of auditory stimulus
def plot_figure_1a(stimulus_data, fs, start_sec=5, end_sec=10):
    features = ['Env', 'Onsets', 'So', 'Sp', 'Ho', 'Hp']
    times = np.linspace(0, len(stimulus_data) / fs, len(stimulus_data))

    # Wycięcie fragmentu czasu
    mask = (times >= start_sec) & (times <= end_sec)
    t_crop = times[mask]
    d_crop = stimulus_data[mask]

    fig, axes = plt.subplots(len(features), 1, figsize=(10, 8), sharex=True)
    for i, feat in enumerate(features):
        if i < d_crop.shape[3]:  # Sprawdzenie czy kolumna istnieje
            axes[i].plot(t_crop, d_crop[:, i], color='darkgreen' if i < 2 else 'gray')
            axes[i].set_ylabel(feat)
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)

    axes[-1].set_xlabel('Time (s)')
    plt.suptitle('Figure 1A: Music Representation')
    plt.tight_layout()
    plt.show()

#Prediction correlation AM + A !!! Uwaga to dla wszystkich głów
def plot_figure_2ab(results_A, results_AM):
    """
    results_A: lista korelacji dla każdego badanego (model akustyczny)
    results_AM: lista korelacji dla każdego badanego (model pełny)
    """
    # Figure 2A: Bar plot (Average)
    mean_A = np.mean(results_A)
    mean_AM = np.mean(results_AM)
    sem_A = np.std(results_A) / np.sqrt(len(results_A))
    sem_AM = np.std(results_AM) / np.sqrt(len(results_AM))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(['A', 'AM'], [mean_A, mean_AM], yerr=[sem_A, sem_AM], color=['green', 'coral'], capsize=5)
    ax1.set_title('Figure 2A: All subjects')
    ax1.set_ylabel('Prediction correlation (r)')

#Wzmocnienie predykcyjene AM dla każdego badanego
subjects = [f'S{i + 1}' for i in range(len(results_A))]
ax2.scatter(subjects, results_AM, color='coral', label='AM')
ax2.scatter(subjects, results_A, color='green', label='A', facecolors='none')

# Rysowanie słupków zysku (Delta r)
for i in range(len(results_A)):
    ax2.vlines(subjects[i], results_A[i], results_AM[i], color='gray', alpha=0.5)

ax2.set_title('Figure 2B: Single subject results')
ax2.set_ylabel('r')
ax2.legend()
plt.xticks(rotation=45)
plt.show()

#Wagi regresji grzbietowej dla TRFAM dod. i ujem. składowe TRF
def plot_figure_2e(trf_model, channels_idx=[7, 11]):  # Np. Fz, Cz, Pz
    # trf_model.weights ma kształt (features, lags, channels)
    features = ['Env', "Env'", 'Sp', 'Hp', 'So', 'Ho']
    lags = np.linspace(tmin * 1000, tmax * 1000, trf_model.weights.shape[3])

    fig, axes = plt.subplots(len(channels_idx), 1, figsize=(8, 10))

    for i, ch in enumerate(channels_idx):
        # Wyciągamy wagi dla danego kanału i normalizujemy
        w = trf_model.weights[:, :, ch]
        w_norm = w / np.max(np.abs(w))

        im = axes[i].imshow(w_norm, aspect='auto', origin='lower',
                            extent=[lags, lags[-1], 0, len(features)],
                            cmap='RdBu_r', vmin=-1, vmax=1)

        axes[i].set_yticks(np.arange(len(features)) + 0.5)
        axes[i].set_yticklabels(features)
        axes[i].set_title(f'Channel index: {ch}')
        axes[i].axvline(0, color='black', linewidth=0.5)

    plt.colorbar(im, ax=axes.ravel().tolist(), label='Normalised TRF weights')
    plt.xlabel('Latency (ms)')
    plt.suptitle('Figure 2E: TRF Weights Heatmap')
    plt.show()