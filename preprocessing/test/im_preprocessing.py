# %%
import numpy as np
import pandas as pd
import matplotlib as plt
import mne

# %%
# DATA
ch_exclude = [f'EXG{i}' for i in range(3, 9)]
raw = mne.io.read_raw_bdf('/Users/zosiamikolajczak/ANL/ANL_inharmonic_melodies/eeg_data/im_1.bdf',
                          eog = ['EXG1', 'EXG2'],
                          exclude= ch_exclude,
                          stim_channel='Status',
                          preload=True)

#%%
raw.plot()

#%%
raw.set_montage('biosemi64')
raw.get_montage().plot()

#%% FILTERING
# low&high pass filter wygładza sygnał -> usuwa szum w postaci za niskich i za wysokich częstotliwości
# low pass usuwa oscylacje na niskich częstotliwościach, a high pass na wysokich
# notch filter wywala jedną konkretną częstotliwość; 50 bo prąd zmienny ma taką częstotliwość
raw_filtered = raw.copy().filter(l_freq=0.1, h_freq=30).notch_filter(50)

#%% EPOCH
# pierwsza kolumna -> nr sampla
# ostatnia kolumna -> wartość trigera
events = mne.find_events(raw_filtered, stim_channel='Status', initial_event=True)
# słownik identyfikatorów trigerów; zależne od badania
event_ids = {
    '1' : 1,
    '2' : 2,
    '3' : 3,
    '4' : 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    '10': 10,
    '11': 11,
    '12': 12,
    '13': 13,
    '14': 14,
    '15': 15,
    '16': 16,
    '17': 17,
    '18': 18,
    '19': 19,
    '20': 20,
    '21': 21,
    '22': 22,
    '23': 23,
    '24': 24,
    '25': 25,
    '26': 26,
    '27': 27,
    '28': 28,
    '29': 29,
    '30': 30,
    '31': 31,
    '32': 32,
    '101': 101,
    '102': 102,
    '103': 103,
    '104': 104,
    '105': 105,
    '106': 106,
    '107': 107,
    '108': 108,
    '109': 109,
    '110': 110,
    '111': 111,
    '112': 112,
    '113': 113,
    '114': 114,
    '115': 115,
    '116': 116,
    '117': 117,
    '118': 118,
    '119': 119,
    '120': 120,
    '121': 121,
    '122': 122,
    '123': 123,
    '124': 124,
    '125': 125,
    '126': 126,
    '127': 127,
    '128': 128,
    '129': 129,
    '130': 130,
    '131': 131,
    '132': 132,
}

# %%
# Interpolate bad channels
# This is not structly necessary for a single subject.
# However, when processing multiple subjects, it will allow comparing results across all sensors.
raw.interpolate_bads()

# %%
# Load the events embedded in the raw file as eelbrain.Dataset, a type of object that represents a data-table
events = eelbrain.load.mne.events(raw)

# %%
# Display the events table:
events





#%% ICA - INDEPENDENT COMPONENT ANALYSIS
# dobre do wyciągania artefaktów ocznych
# rozkłada sygnał na niezależne komponenty (u nas arbitralnie 30)

#compute ica
ica = mne.preprocessing.ICA(
    n_components=30,
    max_iter='auto',
    random_state=97,
)

# fit ica to ar-ed data
ica.fit(raw_filtered)
# wyrzuca błąd; " I can conclude that the behaviour was very likely due to the low signal-to-noise ratio"


#%%
def run_ica(method, fit_params=None):
    ica = ICA(
        n_components=20,
        method=method,
        fit_params=fit_params,
        max_iter="auto",
        random_state=0,
    )
    t0 = time()
    ica.fit(raw, reject=reject)
    fit_time = time() - t0
    title = f"ICA decomposition using {method} (took {fit_time:.1f}s)"
    ica.plot_components(title=title)

# %%
# średnie sygnały/napięcia w poszczególnych elektrodach
ica.plot_components()
ica.exclude = []

# %%
