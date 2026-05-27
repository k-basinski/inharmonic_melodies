# %%
# setup
import mne
import matplotlib

matplotlib.use('Qt5Agg')

# %%
# DATA
ch_exclude = [f'EXG{i}' for i in range(3,9)]
raw = mne.io.read_raw_bdf('/Users/zosiamikolajczak/ANL/ANL_inharmonic_melodies/data/eeg_data/pilot_09.bdf',
                          eog = (['EXG1', 'EXG2']),
                          exclude=ch_exclude,
                          stim_channel= 'Status',
                          preload=True
                          )
# %%
# MONTAGE
raw.set_montage('biosemi64')

# %%
raw.get_montage().plot()

# %%
raw.plot()

# %%
# FILTERING
filtered_raw = raw.copy()
filtered_raw.filter(l_freq=0.1, h_freq=30)
filtered_raw.notch_filter(50)

# %%
filtered_raw.plot() # zaznaczamy wadliwe kanały

# %%
# INTERPOLATION
filtered_raw.interpolate_bads()

# %%
filtered_raw.plot() 

# %%
# EPOCHS
events_ids = {
    '2' : 2,
    '6' : 6,
    '7' : 7,
    '8' : 8,
    '9' : 9,
    '16' : 16,
    '18' : 18,
    '20' : 20,
    '21' : 21,
    '23' : 23,
    '24' : 24,
    '25' : 25,
    '26' : 26,
    '27' : 27,
    '30' : 30,
    '101' : 101,
    '103' : 103,
    '104' : 104,
    '105' : 105,
    '110' : 110,
    '111' : 111,
    '112' : 112,
    '113' : 113,
    '114' : 114,
    '115' : 115,
    '117' : 117,
    '119' : 119,
    '122' : 122,
    '128' : 128,
    '129' : 129
}



events = mne.find_events(filtered_raw, stim_channel='Status')

epochs = mne.Epochs(filtered_raw,
                    events,
                    event_id=events_ids,
                    tmax= 100,
                    baseline=None, # nie robimy tu baseline correction
                    decim=8)

# %%
epochs.plot()

# %%
# ICA
ica = mne.preprocessing.ICA(
    n_components=30,
    max_iter='auto',
    random_state=666)
ica.fit(epochs)
ica.plot_components(inst=epochs)

# %%
epochs.load_data()
ica.apply(epochs,
          exclude=[0,5,6])

# %%
# automatyczne wyrzucanie oczek
eog_epochs = mne.preprocessing.create_eog_epochs(filtered_raw)
eog_evoked = eog_epochs.average()
eog_indices, eog_scores = ica.find_bads_eog(epochs)
ica.exclude = eog_indices

# %%
# OTHER PLOTS

eog_epochs = mne.preprocessing.create_eog_epochs(filtered_raw)
eog_evoked = eog_epochs.average()
eog_indices, eog_scores = ica.find_bads_eog(epochs)
ica.exclude = eog_indices
ica.plot_components(inst=epochs)

# %%
# plot overlay
ica.plot_overlay(raw, exclude=ica.exclude)
# %%
# barplot of ICA component "EOG match" scores
ica.plot_scores(eog_scores)
# %%
# # barplot of ICA component "ECG match" scores
# ica.plot_scores(ecg_scores, exclude=ecg_indices, title="ECG components")

# plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
ica.plot_sources(eog_evoked)

# %%
# SAVING
epochs.save(fname = '/Users/zosiamikolajczak/ANL/ANL_inharmonic_melodies/data/epochs_data/pilot_09_epo.fif',
            fmt='double',
            overwrite = True,
            )
