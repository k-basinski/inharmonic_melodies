# %%
# setup
import mne
import matplotlib
import pandas as pd

matplotlib.use('Qt5Agg')

# %%
# DATA
ch_exclude = [f'EXG{i}' for i in range(3,9)]
raw = mne.io.read_raw_bdf('/Users/zosiamikolajczak/ANL/ANL_inharmonic_melodies/data/eeg_data/pilot_17.bdf',
                          eog = (['EXG1', 'EXG2']),
                          exclude=ch_exclude,
                          stim_channel= 'Status',
                          preload=True
                          )

# %%
# BLOCKS FOR EPOCHS
blocks = pd.read_csv('/Users/zosiamikolajczak/ANL/ANL_inharmonic_melodies/git/inharmonic_melodies/paradigm/soundpool/pilot_17_logs.csv')

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
# FIND EPOCHS ID
trig = blocks['trig'].tolist()

string = []
for i in trig:
    s = str(i)
    string.append(s)

events_ids = {k: v for k, v in zip(string, trig)}

# %% REMOVE USUSED EPOCHS
events_ids.pop('3')
events_ids.pop('22')
print(events_ids)

# %% EPOCS
events = mne.find_events(filtered_raw, stim_channel='Status')

epochs = mne.Epochs(filtered_raw,
                    events,
                    event_id=events_ids,
                    tmax= 110,
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
          exclude=[0,1,2])

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
epochs.save(fname = '/Users/zosiamikolajczak/ANL/ANL_inharmonic_melodies/data/epochs_data/pilot_17_epo.fif',
            fmt='double',
            overwrite = True,
            )

# %%
