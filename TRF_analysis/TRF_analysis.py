# %%
import os
import numpy as np
import slab
import mne
import matplotlib.pyplot as plt
from mtrf.model import TRF
import pandas as pd
from glob import glob
# %%
##DEFINE STIMULUS AND RESPONSE
subject = 'pilot_08'
new_fs = 64

dur_segment = 50 #100 za długie, trzeba sprawdzić wyższe niż 50?
len_segment = dur_segment * new_fs

wav_path = '/Users/zosiamikolajczak/ANL/ANL_inharmonic_melodies/data/sound/harmonic/*.wav'
path_to_fif = f"/Users/zosiamikolajczak/ANL/ANL_inharmonic_melodies/data/epochs_data/{subject}_epo.fif"
# path_to_fif = "/Users/zosiamikolajczak/ANL/ANL_inharmonic_melodies/data/*.fif"

wav_files = sorted(glob(wav_path))

# %%
#stimulus - calculate envelope and onset of the sound
mvar_stimulus = []

for wf in wav_files:

    sound = slab.Sound(wf).channel(0)
    sound = sound.resample(int(sound.samplerate / 3))

    envelope = sound.envelope().resample(new_fs) # returns a new signal containing the lowpass Hilbert envelopes of both channels
    stim_data = envelope.data.reshape(-1, 1)

    onsets = np.diff(stim_data, axis=0, prepend=np.zeros((1, 1))) # calculate the n-th discrete difference along the given axis
    onsets[onsets < 0] = 0

    mvar_epoch = np.concatenate([stim_data, onsets], axis=1)
    mvar_stimulus.append(mvar_epoch)

# %%
#response
#glob - zwraca listę ścieżek pasujących do podanego wzorca nazwy
response = []

all_epochs_files = sorted(glob(path_to_fif))

for file in all_epochs_files:

    epochs = mne.read_epochs(file, preload=True)

    #tylko EEG - wywal oczy eog=['EXG1', 'EXG2'] !!!! do naprawy
    epochs.pick_types(eeg=True)

    for epoch in epochs.get_data():
        response.append(epoch.T) # czemu T?

# %%
#kontrolka
print("WAV:", len(wav_files))
print("EEG:", len(response))

print("Stim shape:", mvar_stimulus[0].shape)
print("Resp shape:", response[0].shape)

# %%
##ENVELOPE AND ONSETS VISUALIZATION
times = np.linspace(0, len(mvar_stimulus[0])/new_fs, len(mvar_stimulus[0]))
plt.plot(times, mvar_stimulus[0][:,0], label='Envelope', alpha=0.7)
plt.plot(times, mvar_stimulus[0][:,1], label='Onsets', alpha=0.7)
plt.xlim(5, 10)
plt.xlabel('Time [s]')
plt.ylabel('Ammplitude [a.u.]')
plt.legend()
plt.show()

# %%
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

# %%
#NaN cleanup
mvar_stimulus = [np.nan_to_num(s) for s in mvar_stimulus]
response = [np.nan_to_num(r) for r in response]

# %%
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

# %%
##MODEL VALIDATION
m_fwd_trf = TRF()

tmin, tmax = -0.1, 0.4

regularization = [0.1, 1, 10] #lambda do testowania potem użyjemy np.logspace(-2, 6, 10)

m_fwd_trf.train(
    stim_segments,
    resp_segments,
    new_fs,
    tmin,
    tmax,
    regularization,
    k=3)

# %%
##VISUALIZATION
fig, ax = plt.subplots(3, sharex=True, figsize=(10, 10))

#envelope TRF
m_fwd_trf.plot(feature=0, axes=ax[0], show=False)
ax[0].set_title("TRF: Envelope")

#onsets TRF
m_fwd_trf.plot(feature=1, axes=ax[1], show=False)
ax[1].set_title("TRF: Onsets")

#global Field Power
m_fwd_trf.plot(channel='gfp', axes=ax[2], show=False)
ax[2].set_title("Global Field Power")

plt.tight_layout()
plt.show()

# %%
#VISUALIZATION WITH MNE - poprawić wyżej channels (wywala za dużo kanałów) !!!
from mne.channels import make_standard_montage
montage = make_standard_montage('biosemi64')
fwd_trf_evo = m_fwd_trf.to_mne_evoked(montage)[0]

fwd_trf_evo.plot_joint(
    [0.075, 0.13, 0.36],
    topomap_args={"scalings": 1},
    ts_args={"units": "a.u.", "scalings": dict(eeg=1)},
    )

# %%
##ESTIMATE MODEL'S ACCURACY (VISUALIZATION)
pred, r = m_fwd_trf.predict(mvar_stimulus, response, average=False)
idx = np.argmax(r)  # pick the channel with the best prediction

# Regularization changes the scale to stanardize plotting
pred[0] = (pred[0]-pred[0].mean(axis=0))/pred[0].std(axis=0)
times = np.linspace(0, len(response[0])/new_fs, len(response[0]))
plt.plot(times, response[0][:, idx], label='Obsereved EEG')
plt.plot(times, pred[0][:, idx], label='Predicted EEG')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [a.u.]')
plt.xlim(20, 30)  # zoom in on the x-axis
plt.legend()
plt.title(f'Correlation = {r.mean().round(3)}')

# %%
##LENGTH OF RECORDINGS - in progress !!! wykrzacza się na vizualizacji
#parameters
dur_segment = 50
new_fs = 64
len_segment = dur_segment * new_fs

tmin = -0.1
tmax = 0.4

regularization = [0.1, 1, 10]

train_durations = [30, 60, 120, 300, 600] #lambda do testowania potem użyjemy np.logspace(-2, 6, 10)
n_repeats = 10

#split data
stimulus_train = mvar_stimulus[:-2]
response_train = response[:-2]

stimulus_test = mvar_stimulus[-2:]
response_test = response[-2:]

#segmentation
stim_segments = []
resp_segments = []

for s_full, r_full in zip(stimulus_train, response_train):

    n_seg = len(s_full) // len_segment

    if n_seg < 2:
        continue

    s_crop = s_full[:n_seg * len_segment]
    r_crop = r_full[:n_seg * len_segment]

    stim_segments.extend(np.array_split(s_crop, n_seg))
    resp_segments.extend(np.array_split(r_crop, n_seg))

print("Liczba segmentów:", len(stim_segments))

total_dur = len(stim_segments) * dur_segment
print("Maksymalny czas treningu:", total_dur, "s")

#analysis

valid_durations = []
r_scores_mean = []
r_scores_std = []
best_lambdas = []

for t in train_durations:

    n_seg_to_use = int(t / dur_segment)

    if n_seg_to_use < 3 or n_seg_to_use > len(stim_segments):
        continue

    print(f"\nTraining duration: {t}s")

    valid_durations.append(t)

    r_tmp = []
    lambda_tmp = []

    for rep in range(n_repeats):

        idx = np.random.choice(
            len(stim_segments),
            n_seg_to_use,
            replace=False
        )

        stim_subset = [stim_segments[j] for j in idx]
        resp_subset = [resp_segments[j] for j in idx]

        trf = TRF()
        k_cv = min(3, n_seg_to_use)

        trf.train(
            stim_subset,
            resp_subset,
            new_fs,
            tmin,
            tmax,
            regularization,
            k=k_cv
        )

        _, r_val = trf.predict(stimulus_test, response_test)

        r_tmp.append(np.nanmean(r_val))
        lambda_tmp.append(trf.regularization)

    r_scores_mean.append(np.mean(r_tmp))
    r_scores_std.append(np.std(r_tmp))
    best_lambdas.append(np.median(lambda_tmp))

#visualization

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].errorbar(
    valid_durations,
    r_scores_mean,
    yerr=r_scores_std,
    marker='o'
)

ax[0].set_title("TRF performance vs training duration")
ax[0].set_xlabel("Training duration [s]")
ax[0].set_ylabel("Pearson r")
ax[0].grid(True)

ax[1].semilogy(
    valid_durations,
    best_lambdas,
    marker='s'
)

ax[1].set_title("Optimal regularization")
ax[1].set_xlabel("Training duration [s]")
ax[1].set_ylabel("Lambda")
ax[1].grid(True)

plt.tight_layout()
plt.show()

##HOW MANY PARTICIPANTS - in progress

