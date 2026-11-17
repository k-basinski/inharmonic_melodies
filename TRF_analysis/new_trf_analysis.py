# %%
import numpy as np
import pandas as pd
import mne
from mtrf.model import TRF
import pickle
import matplotlib.pyplot as plt
import soundfile as sf

# %%
# config
# subjects = [s for s in range(7, 66) if s not in [46, 47, 48,52]]
# subjects = [s for s in range(7, 20) if s not in [46, 47, 48,52]]
# subjects = [s for s in range(20, 40) if s not in [46, 47, 48,52]]
subjects = [s for s in range(40, 66) if s not in [46, 47, 48,52]]

# uncomment to do stuff with two people/saves time
# subjects = [7,8]

new_fs = 256
tmin, tmax = 0, 0.4
regularization = [0.1, 1, 10, 100]
epochs_path = ("/Users/kornelia/PycharmProjects/inharmonic_melodies/TRF_analysis/data/epochs/")
log_path = ("/Users/kornelia/PycharmProjects/inharmonic_melodies/TRF_analysis/data/logs/")
wav_path = ("/Users/kornelia/PycharmProjects/inharmonic_melodies/TRF_analysis/data/soundpool/")
idyom_keys = ['envelope', 'onsets', 'pitch_IC', 'length_IC', 'pitch_length_IC', 'pitch_H', 'length_H', 'pitch_length_H']

def fit_trf(stim_subset,response,cols):
    #only cols of interest
    stimulus = [x[:, cols] for x in stim_subset]
    #rachu ciachu
    # stimulus = [s[:len(r)] for s, r in zip(stimulus, response)]
    # normalize
    for i, (s, r) in enumerate(zip(stimulus, response)):
        stimulus[i] = (s - s.mean(axis=0)) / s.std(axis=0)
        response[i] = (r - r.mean(axis=0)) / r.std(axis=0)

    trf = TRF()
    trf.train(stimulus,response,new_fs,tmin,tmax,regularization, k=5)

    #prediction
    _, r = trf.predict(stimulus,response, average=False)
    return trf, r
#idyom
with open("/Users/kornelia/PycharmProjects/inharmonic_melodies/TRF_analysis/data/idyom/idyom_vectors.pickle", 'rb') as file:
    idyom_vectors = pickle.load(file)

# load epochs
eeg_data = {}
for s in subjects: # dodaj do pętli niżej żeby nie zasrać RAMu
    epochs_file = f'{epochs_path}im_{s}_epo.fif'
    e = mne.read_epochs(epochs_file)
    eeg_data[s] = e.filter(8,30,n_jobs=-1)

# %%
# słowniki zamiast list: klucz = numer badanego
eeg_signals = {p: [] for p in subjects}
idyom_signals = {p: [] for p in subjects}
eeg_signals_inh = {p: [] for p in subjects}
idyom_signals_inh = {p: [] for p in subjects}
signal_length = 28161 # sr * 110 seconds

for p in subjects:
    for e in eeg_data[p].events:
        trig = e[2]
        # for harmonic sounds
        if trig < 100:
            eeg_signal = eeg_data[p].get_data(picks=['eeg'], item = str(trig), tmin=0)
            idyom_signal = idyom_vectors[trig]
            eeg_signals[p].append(eeg_signal[0,:,:].T)
            idyom_signals[p].append(pd.DataFrame(idyom_signal).to_numpy().T[:,:signal_length].T)
        # for inharmonic
        else:
            trig_inh = trig - 100
            eeg_signal = eeg_data[p].get_data(picks=['eeg'], item = str(trig), tmin=0)
            idyom_signal = idyom_vectors[trig_inh]
            eeg_signals_inh[p].append(eeg_signal[0,:,:].T)
            idyom_signals_inh[p].append(pd.DataFrame(idyom_signal).to_numpy().T[:,:signal_length].T)



# %%
pd.DataFrame(idyom_vectors[14]).to_numpy().T[:,:28161]


# %%
model_cols = {
    'A':  [0, 1],                       # envelope, onsets
    'AO': [0,1,3, 6],   # acoustic + length
    'AP': [0, 1, 2, 5], # acoustic + pitch
    'APO': [0, 1, 2, 3, 5, 6],     # Acoustic + pitch + length
    'O': [3, 6],  #  length
    'P': [2, 5],  # pitch
    'PO': [2, 3, 5, 6],  # pitch + length
}

trfs = {}   # trfs[subject][(model, condition)] = obiekt TRF
rs = {}     # rs[subject][(model, condition)]   = korelacja

# %%
import os, pickle
results_path = "/Users/kornelia/PycharmProjects/inharmonic_melodies/TRF_analysis/data/results_nofilter/"
os.makedirs(results_path, exist_ok=True)

for p in subjects:
    trfs[p], rs[p] = {}, {}
    for name, cols in model_cols.items():
        trfs[p][(name, 'harm')], rs[p][(name, 'harm')] = fit_trf(
            idyom_signals[p], eeg_signals[p], cols)
        trfs[p][(name, 'inharm')], rs[p][(name, 'inharm')] = fit_trf(
            idyom_signals_inh[p], eeg_signals_inh[p], cols)

    with open(f"{results_path}sub_{p}_fit.pickle", 'wb') as f:
        pickle.dump({'trfs': trfs[p], 'rs': rs[p]}, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"subject {p} done")
