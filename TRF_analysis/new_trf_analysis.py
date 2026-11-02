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
subjects = list(range(12,52))
new_fs = 256
tmin, tmax = 0, 0.4
regularization = [0.1, 1, 10, 100]
epochs_path = (f"/Users/kbas/sci_data/inharmonic_melodies/epochs/")
log_path = (f"/Users/kbas/sci_data/inharmonic_melodies/logs/")
wav_path = ("/Users/kbas/sci_data/inharmonic_melodies/soundpool/")
idyom_keys = ['envelope', 'onsets', 'pitch_IC', 'length_IC', 'pitch_length_IC', 'pitch_H', 'length_H', 'pitch_length_H']

def fit_trf(stim_subset,response,cols):
    #only cols of interest
    stimulus = [x[:, cols] for x in stim_subset]
    # normalize
    for i, (s, r) in enumerate(zip(stimulus, response)):
        stimulus[i] = (s - s.mean(axis=0)) / s.std(axis=0)
        response[i] = (r - r.mean(axis=0)) / r.std(axis=0)

    trf = TRF()
    trf.train(stimulus,response,new_fs,tmin,tmax,regularization, k=5)

    #prediction
    _, r = trf.predict(stimulus,response)
    return trf, r

# load idyom results
with open('../idyom_modelling/idyom_vectors.pickle', 'rb') as file:
    idyom_vectors = pickle.load(file)

# load epochs
eeg_data = {}
for s in subjects: # dodaj do pętli niżej żeby nie zasrać RAMu
    epochs_file = f'{epochs_path}pilot_{s}_epo.fif'
    e = mne.read_epochs(epochs_file)
    eeg_data[s] = e


# %%
eeg_data[30].ch_names

# %%
eeg_signals = []
idyom_signals = []
eeg_signals_inh = []
idyom_signals_inh = []
signal_length = 28161 # sr * 110 seconds

for p in subjects:
    for e in eeg_data[p].events:
        # get trigger id from events table
        trig = e[2]
         # for harmonic sounds
        if trig < 100:
            eeg_signal = eeg_data[p].get_data(picks=['eeg'], item = str(trig), tmin=0)
            idyom_signal = idyom_vectors[trig]
            eeg_signals.append(eeg_signal[0,:,:].T)
            idyom_signals.append(pd.DataFrame(idyom_signal).to_numpy().T[:,:signal_length].T)
        # for inharmonic
        else:
            trig_inh = trig - 100
            eeg_signal = eeg_data[p].get_data(picks=['eeg'], item = str(trig), tmin=0)
            idyom_signal = idyom_vectors[trig_inh]
            eeg_signals_inh.append(eeg_signal[0,:,:].T)
            idyom_signals_inh.append(pd.DataFrame(idyom_signal).to_numpy().T[:,:signal_length].T)



# %%
pd.DataFrame(idyom_vectors[14]).to_numpy().T[:,:28161]


# %%
# MODEL A
#harmonic
trf_A_harm, r_A_harm = fit_trf(idyom_signals,eeg_signals,[0, 1])
#inharmonic
trf_A_inharm, r_A_inharm = fit_trf(idyom_signals_inh,eeg_signals_inh,[0, 1])

# %%
# MODEL AM
#harmonic
trf_AM_harm, r_AM_harm = fit_trf(idyom_signals,eeg_signals,[0, 1, 2])

#inharmonic
trf_AM_inharm, r_AM_inharm = fit_trf(idyom_signals_inh,eeg_signals_inh,[0, 1, 2])

# %%
# MODEL AMH
#harmonic
trf_AMH_harm, r_AMH_harm = fit_trf(idyom_signals,eeg_signals,[0, 1, 2, 3])
#inharmonic
trf_AMH_inharm, r_AMH_inharm = fit_trf(idyom_signals_inh,eeg_signals_inh,[0, 1, 2, 3])


# %%
feature_names = dict(zip(range(7), idyom_keys))

# %%
print(eeg_data[20].ch_names[37])
# %%
trf_AM_harm.weights.min()
# %%
yticks, ylabels = np.arange(6)+.5, ['amp', 'onsets', 'pitch (surprise)', 'onset S', 'pitch H', 'onset H']
fig, ax = plt.subplots(4, figsize=(8,10))
ch_index = 'gfp' # Fz
trf_AM_harm.plot(channel=ch_index, axes=ax[0], kind='image', show=False)
trf_AM_inharm.plot(channel=ch_index, axes=ax[1], kind='image', show=False)
ch_index = 47 # Change ! Cz
trf_AM_harm.plot(channel=ch_index, axes=ax[2], kind='image', show=False)
trf_AM_inharm.plot(channel=ch_index, axes=ax[3], kind='image', show=False)
for i in range(4):
    ax[i].set_yticks(yticks[:3], ylabels[:3])
    plt.set_cmap("coolwarm")
    ax[i].autoscale(False)

ax[0].set_title("Harmonic (Fz)")
ax[1].set_title("Inharmonic (Fz)")
ax[2].set_title("Harmonic (Cz)")
ax[3].set_title("Inharmonic (Cz)")
plt.tight_layout()
plt.show()
# %%
# %%
yticks, ylabels = np.arange(4)+.5, ['amp', 'onsets', 'pitch surprise', 'onset surprise']
fig, ax = plt.subplots(4, figsize=(8,14))
ch_index = 37 # Fz
trf_AMH_harm.plot(channel=ch_index, axes=ax[0], kind='image', show=False)
trf_AMH_inharm.plot(channel=ch_index, axes=ax[1], kind='image', show=False)
ch_index = 47 # Change ! Cz
trf_AMH_harm.plot(channel=ch_index, axes=ax[2], kind='image', show=False)
trf_AMH_inharm.plot(channel=ch_index, axes=ax[3], kind='image', show=False)
for i in range(4):
    ax[i].set_yticks(yticks[:4], ylabels[:4])
    plt.set_cmap("coolwarm")

ax[0].set_title("Harmonic (Fz)")
ax[1].set_title("Inharmonic (Fz)")
ax[2].set_title("Harmonic (Cz)")
ax[3].set_title("Inharmonic (Cz)")

plt.tight_layout()
plt.savefig('trfs.png', dpi=300)
plt.show()

# %%
montage = mne.channels.make_standard_montage('biosemi64')
evokeds_AMH_harm = trf_AMH_harm.to_mne_evoked(montage)
evokeds_AMH_inharm = trf_AMH_inharm.to_mne_evoked(montage)

# %%
evokeds_AMH_inharm[2].plot_joint([0.09, 0.145, 0.22])
# %%
import seaborn as sns
model_performance = [r_A_harm, r_AM_harm, r_AMH_harm, r_A_inharm, r_AM_inharm, r_AMH_inharm]
performance_labels = ['acoustic', 'acoustic+pitch', 'acoustic+pitch+onset', 'acoustic', 'acoustic+pitch', 'acoustic+pitch+onset']
sns.barplot(y=performance_labels, x=model_performance, hue=['harmonic','harmonic','harmonic','inharmonic','inharmonic','inharmonic'])
plt.tight_layout()
plt.savefig('predictions.png', dpi=300)





# %%
fig, ax = plt.subplots(2)
trf_A_harm.plot(channel=ch_index, axes=ax[0], kind='image', show=False)
trf_A_inharm.plot(channel=ch_index, axes=ax[1], kind='image', show=False)
ax[0].set_title("acoustic - harmonic")
ax[1].set_title("acoustic - inharmonic")
plt.tight_layout()
plt.show()
# %%

fig, ax = plt.subplots(2)
trf_AMH_harm.plot(channel=ch_index, axes=ax[0], kind='image', show=False)
trf_AMH_inharm.plot(channel=ch_index, axes=ax[1], kind='image', show=False)
ax[0].set_title("acoustic+music IC+entropy - harmonic")
ax[1].set_title("acoustic+music IC+entropy - inharmonic")

plt.tight_layout()
plt.show()
# %%
# normalized weights
# trf_AMH_harm.weights
harm = trf_AMH_harm.weights[:,:,ch_index]
inharm = trf_AMH_inharm.weights[:,:,ch_index]
normalized_AMH_harm = (harm - harm.mean(axis=0)) / harm.std()
normalized_AMH_inharm = (inharm - inharm.mean(axis=0)) / inharm.std()
vmin = np.min([normalized_AMH_harm, normalized_AMH_inharm])
vmax = np.max([normalized_AMH_harm, normalized_AMH_inharm])

fig, ax = plt.subplots(3,  figsize=(5, 10))
time_v = np.arange(104) / new_fs
ax[0].imshow(normalized_AMH_harm, 'coolwarm', aspect='auto', vmin=vmin, vmax=vmax, extent=(0, 104, 0, 6))
ax[1].imshow(normalized_AMH_inharm, 'coolwarm', aspect='auto', vmin=vmin, vmax=vmax)
ax[2].imshow(normalized_AMH_inharm - normalized_AMH_harm, 'coolwarm', aspect='auto', vmin=vmin, vmax=vmax)
ax[0].set(
    xlabel="Time lag [s]",
    # xlim=(time_v.min(), time_v.max()),
)
ax[0].set_yticks([.5, 1.5, 2.5, 3.5, 4.5, 5.5], ['a', 'b', 'c', 'd', 'e', 'f'])
plt.tight_layout()
plt.show()
# %%

# %%
def plot_group_trf(models_list, condition_label):
    n_features = len(feature_names)
    n_model_features = models_list[0].weights.shape[0]
    n_timepoints = min(
        model.weights.shape[2]
        for model in models_list)
    trimmed_weights = np.array([
        model.weights[:, :, :n_timepoints]
        for model in models_list])
    mean_weights = trimmed_weights.mean(axis=0)
    avg_model = models_list[0]
    avg_model.weights = mean_weights


    fig, ax = plt.subplots(n_features,1,figsize=(10, 4 * n_features))
    ax = np.atleast_1d(ax)
    for i in range(n_features):
        avg_model.plot(feature=i,axes=ax[i],show=False)
        ax[i].set_title(feature_names[i],fontsize=12)
        ax[i].set_ylabel("Amplitude (a.u.)")
        ax[i].grid(True, alpha=0.3)

    fig.suptitle(f"Forward TRF Weights Across Time ({condition_label})", fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig.subplots_adjust(top=0.93)
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    avg_model.plot(channel='gfp', axes=ax,kind='image', show=False)
    ax.set_yticks(np.arange(n_features))
    ax.set_yticklabels(list(feature_names.values()))
    ax.set_title( f"Spectrotemporal Receptive Field Across Time and Frequency ({condition_label})",fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    return avg_model

plot_group_trf([trf_AM_harm], "harmonic")
# %%
# how to test if smth is on list
dupa = [1, 2, 3]
3 in dupa


if 3 in dupa:
    print("jej!")
