# %%
import numpy as np
from numpy._core.fromnumeric import _3D
import pandas as pd
import matplotlib.pyplot as plt
import mne
import slab
import pickle
from glob import glob
from mtrf.model import TRF
import os
import re
import copy
# %%
!pwd
# %%
subjects = ["im_2", "im_3", "im_7"]
new_fs = 256
tmin, tmax = 0, 0.35
regularization = [0.1, 1, 10, 100]
# %%
def extract_id(fname):
    #pull melody num
    return int(re.findall(r'\d+', fname)[0])

def build_stimulus(wav_files, melody, log_df):
    mvar_stimulus = {}
    #dictonary: filename + trig
    trig_map = dict(zip(log_df["filename"],log_df["trig"]))

    for wf in wav_files:
        fname = os.path.basename(wf)
        trig = int(trig_map[fname])
         #harmonicity labels
        if 1 <= trig <= 30:
            melody_id = trig
            is_harm = 1
        elif 101 <= trig <= 130:
            melody_id = trig - 100
            is_harm = 0

    #reshape + rename idyom
        envelope = np.asarray(melody[melody_id]["envelope"]).reshape(-1, 1)
        onsets = np.asarray(melody[melody_id]["onsets"]).reshape(-1, 1)
        S_p = np.asarray(melody[melody_id]["pitch_IC"]).reshape(-1, 1)
        S_o = np.asarray(melody[melody_id]["length_IC"]).reshape(-1, 1)
        H_p = np.asarray(melody[melody_id]["pitch_H"]).reshape(-1, 1)
        H_o = np.asarray(melody[melody_id]["length_H"]).reshape(-1, 1)

    #length adjustment
        L = min(len(envelope),len(onsets),len(S_p),len(S_o),len(H_p),len(H_o))

        envelope = envelope[:L]
        onsets = onsets[:L]
        S_p = S_p[:L]
        S_o = S_o[:L]
        H_p = H_p[:L]
        H_o = H_o[:L]

    #harmonicity
        harm = np.full((L, 1),is_harm)

    #combine all to one
        stim = np.concatenate([envelope,onsets,S_p,S_o,H_p,H_o,harm],axis=1)
        mvar_stimulus[fname] = {"data": stim,"trig": trig,"melody_id": melody_id,"harmonicity": is_harm}

    return mvar_stimulus

def fit_trf(stim_subset,resp_subset,cols):
    #only cols of interest
    s = [x[:, cols] for x in stim_subset]
    trf = TRF()
    trf.train(s,resp_subset,new_fs,tmin,tmax,regularization,k=3)

    #prediction
    _, r = trf.predict(s,resp_subset)
    return trf, r

def plot_group_trf(models_list, condition_label, out_prefix):
    n_features = len(feature_names)
    n_model_features = models_list[0].weights.shape[0]
    n_timepoints = min(
        model.weights.shape[2]
        for model in models_list)
    trimmed_weights = np.array([
        model.weights[:, :, :n_timepoints]
        for model in models_list])
    mean_weights = trimmed_weights.mean(axis=0)
    avg_model = copy.deepcopy(models_list[0])
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

# IDyOM
idyom_path = "data/idyom"
melody = {}
for file in glob(os.path.join(idyom_path, "melody_*.csv")):
    melody_id = extract_id(os.path.basename(file))
    df = pd.read_csv(file)
    melody[melody_id] = {
        "envelope": df["envelope"].values,
        "onsets": df["onsets"].values,
        "pitch_IC": df["pitch_IC"].values,
        "length_IC": df["length_IC"].values,
        "pitch_H": df["pitch_H"].values,
        "length_H": df["length_H"].values}

# %%
melody
# %%
results_A_harm = []
results_A_inharm = []
results_AM_harm = []
results_AM_inharm = []

models_A_harm = []
models_A_inharm = []
models_AM_harm = []
models_AM_inharm = []

fif_path = (f"/Users/kbas/sci_data/inharmonic_melodies/epochs/{subject}_epo.fif")
log_path = (f"/Users/kbas/sci_data/inharmonic_melodies/logs/{subject}_log.csv")
wav_path = ("/Users/kbas/sci_data/inharmonic_melodies/soundpool/*.wav")

# %%
subject = subjects[3]
#LOGS: right order of wavs
log_df = pd.read_csv(log_path)
ordered_fnames_all = (log_df["filename"].values)
#EEG
epochs = (mne.read_epochs(fif_path,preload=True).pick_types(eeg=True).resample(new_fs))
#transposition samples x channels
response = [e.T for e in epochs.get_data(tmin=0)]

#WAVs
wav_dict = {os.path.basename(f): f for f in glob(wav_path)}
#if participant did not finish the procedure = print missing and continue
missing_wavs = [f for f in ordered_fnames_all if f not in wav_dict]
if missing_wavs:
    print(f"Missing WAV files for "f"{subject}: {missing_wavs}")
ordered_wavs = [wav_dict[f] for f in ordered_fnames_all if f in wav_dict]
# %%
epochs.events.shape
# %%
log_df
# %%
#build stim + checkout
stimulus = build_stimulus(ordered_wavs,melody,log_df)
print("Stimulus trials:",len(stimulus))
print("EEG trials:",len(response))
# %%
# ordered stimulus
ordered_fnames = [f for f in ordered_fnames_all if f in stimulus]
ordered_stimulus = [stimulus[fname]["data"] for fname in ordered_fnames]

# trials adjustment
n_trials = min(len(ordered_stimulus), len(response))
ordered_stimulus = ordered_stimulus[:n_trials]
response = response[:n_trials]
ordered_fnames = ordered_fnames[:n_trials]

print(f"Trials used for {subject}: {n_trials}")

#trials adjustment
n_trials = min(len(ordered_stimulus),len(response))
ordered_stimulus = (ordered_stimulus[:n_trials])
response = (response[:n_trials])
ordered_fnames = (ordered_fnames[:n_trials])
print(f"Trials used for {subject}: {n_trials}")

if n_trials == 0:
    raise ValueError(f"Ziutek {subject} cos kombinuje")
print("\nTRIAL ORDER:")

for i, fname in enumerate(ordered_fnames):
    info = stimulus[fname]
    print(
        f"trial {i + 1}: "
        f"{fname} | "
        f"trig={info['trig']} | "
        f"melody={info['melody_id']} | "
        f"harm={info['harmonicity']}")

#length adjustment
stim = []
resp = []
for s, r in zip(ordered_stimulus,response):
    L = min(s.shape[0],r.shape[0])
    stim.append(s[:L, :])
    resp.append(r[:L, :])

stim_concat = np.concatenate(stim,axis=0)
resp_concat = np.concatenate(resp,axis=0)

#normalization - idk tbh
stim_mean = (stim_concat.mean(axis=0))
stim_std = (stim_concat.std(axis=0))

resp_mean = (resp_concat.mean(axis=0))
resp_std = (resp_concat.std(axis=0))

stim_std[stim_std == 0] = 1
resp_std[resp_std == 0] = 1

stim = [(s - stim_mean) / stim_std for s in stim]
resp = [(r - resp_mean) / resp_std for r in resp]

#Harmonicity: division in two datasets
harmonicity_flags = [stimulus[fname]["harmonicity"] for fname in ordered_fnames]
harm_idx = [i for i, h in enumerate(harmonicity_flags)if h == 1]
inharm_idx = [i for i, h in enumerate(harmonicity_flags)if h == 0]
stim_harm = [stim[i] for i in harm_idx]
resp_harm = [resp[i] for i in harm_idx]
stim_inharm = [stim[i] for i in inharm_idx]
resp_inharm = [resp[i] for i in inharm_idx]

print(
    f"{subject}: "
    f"harmonic trials="
    f"{len(stim_harm)}, "
    f"inharmonic trials="
    f"{len(stim_inharm)}")

#NaN
for i in range(len(stim)):
    if np.isnan(stim[i]).any():
        raise ValueError(
            f"NaN detected in "
            f"stimulus trial {i}")
    if np.isnan(resp[i]).any():
        raise ValueError(
            f"NaN detected in "
            f"EEG trial {i}")

#TEMPORAL RESPONSE FUNCTION


# MODEL A
#harmonic
trf_A_harm, r_A_harm = fit_trf(stim_harm,resp_harm,[0, 1])
#inharmonic
trf_A_inharm, r_A_inharm = fit_trf(stim_inharm,resp_inharm,[0, 1])


# MODEL AM
#harmonic
trf_AM_harm, r_AM_harm = fit_trf(stim_harm,resp_harm,[0, 1, 2, 3, 4, 5])
#inharmonic
trf_AM_inharm, r_AM_inharm = fit_trf(stim_inharm,resp_inharm,[0, 1, 2, 3, 4, 5])


#save results
results_A_harm.append(r_A_harm)
results_A_inharm.append(r_A_inharm)
results_AM_harm.append(r_AM_harm)
results_AM_inharm.append(r_AM_inharm)

#save models
models_A_harm.append(trf_A_harm)
models_A_inharm.append(trf_A_inharm)
models_AM_harm.append(trf_AM_harm)
models_AM_inharm.append(trf_AM_inharm)

print("Obywatelu jest sukces:",subject)

# %%

#for every participant
for subject in subjects:
    print(f"\nProcessing {subject}")
    try:

    #LOGS: right order of wavs
        log_df = pd.read_csv(log_path)
        ordered_fnames_all = (log_df["filename"].values)
    #EEG
        epochs = (mne.read_epochs(fif_path,preload=True).pick_types(eeg=True).resample(new_fs))
        #transposition samples x channels
        response = [e.T for e in epochs.get_data(tmin=0)]

    #WAVs
        wav_dict = {os.path.basename(f): f for f in glob(wav_path)}
        #if participant did not finish the procedure = print missing and continue
        missing_wavs = [f for f in ordered_fnames_all if f not in wav_dict]
        if missing_wavs:
            print(f"Missing WAV files for "f"{subject}: {missing_wavs}")
        ordered_wavs = [wav_dict[f] for f in ordered_fnames_all if f in wav_dict]

    #build stim + checkout
        stimulus = build_stimulus(ordered_wavs,melody,log_df)
        print("Stimulus trials:",len(stimulus))
        print("EEG trials:",len(response))

    # ordered stimulus
        ordered_fnames = [f for f in ordered_fnames_all if f in stimulus]
        ordered_stimulus = [stimulus[fname]["data"] for fname in ordered_fnames]

        # trials adjustment
        n_trials = min(len(ordered_stimulus), len(response))
        ordered_stimulus = ordered_stimulus[:n_trials]
        response = response[:n_trials]
        ordered_fnames = ordered_fnames[:n_trials]

        print(f"Trials used for {subject}: {n_trials}")

    #trials adjustment
        n_trials = min(len(ordered_stimulus),len(response))
        ordered_stimulus = (ordered_stimulus[:n_trials])
        response = (response[:n_trials])
        ordered_fnames = (ordered_fnames[:n_trials])
        print(f"Trials used for {subject}: {n_trials}")

        if n_trials == 0:
            raise ValueError(f"Ziutek {subject} cos kombinuje")
        print("\nTRIAL ORDER:")

        for i, fname in enumerate(ordered_fnames):
            info = stimulus[fname]
            print(
                f"trial {i + 1}: "
                f"{fname} | "
                f"trig={info['trig']} | "
                f"melody={info['melody_id']} | "
                f"harm={info['harmonicity']}")

    #length adjustment
        stim = []
        resp = []
        for s, r in zip(ordered_stimulus,response):
            L = min(s.shape[0],r.shape[0])
            stim.append(s[:L, :])
            resp.append(r[:L, :])

        stim_concat = np.concatenate(stim,axis=0)
        resp_concat = np.concatenate(resp,axis=0)

    #normalization - idk tbh
        stim_mean = (stim_concat.mean(axis=0))
        stim_std = (stim_concat.std(axis=0))

        resp_mean = (resp_concat.mean(axis=0))
        resp_std = (resp_concat.std(axis=0))

        stim_std[stim_std == 0] = 1
        resp_std[resp_std == 0] = 1

        stim = [(s - stim_mean) / stim_std for s in stim]
        resp = [(r - resp_mean) / resp_std for r in resp]

    #Harmonicity: division in two datasets
        harmonicity_flags = [stimulus[fname]["harmonicity"] for fname in ordered_fnames]
        harm_idx = [i for i, h in enumerate(harmonicity_flags)if h == 1]
        inharm_idx = [i for i, h in enumerate(harmonicity_flags)if h == 0]
        stim_harm = [stim[i] for i in harm_idx]
        resp_harm = [resp[i] for i in harm_idx]
        stim_inharm = [stim[i] for i in inharm_idx]
        resp_inharm = [resp[i] for i in inharm_idx]

        print(
            f"{subject}: "
            f"harmonic trials="
            f"{len(stim_harm)}, "
            f"inharmonic trials="
            f"{len(stim_inharm)}")

    #NaN
        for i in range(len(stim)):
            if np.isnan(stim[i]).any():
                raise ValueError(
                    f"NaN detected in "
                    f"stimulus trial {i}")
            if np.isnan(resp[i]).any():
                raise ValueError(
                    f"NaN detected in "
                    f"EEG trial {i}")

    #TEMPORAL RESPONSE FUNCTION


        # MODEL A
        #harmonic
        trf_A_harm, r_A_harm = fit_trf(stim_harm,resp_harm,[0, 1])
        #inharmonic
        trf_A_inharm, r_A_inharm = fit_trf(stim_inharm,resp_inharm,[0, 1])


        # MODEL AM
        #harmonic
        trf_AM_harm, r_AM_harm = fit_trf(stim_harm,resp_harm,[0, 1, 2, 3, 4, 5])
        #inharmonic
        trf_AM_inharm, r_AM_inharm = fit_trf(stim_inharm,resp_inharm,[0, 1, 2, 3, 4, 5])


        #save results
        results_A_harm.append(r_A_harm)
        results_A_inharm.append(r_A_inharm)
        results_AM_harm.append(r_AM_harm)
        results_AM_inharm.append(r_AM_inharm)

        #save models
        models_A_harm.append(trf_A_harm)
        models_A_inharm.append(trf_A_inharm)
        models_AM_harm.append(trf_AM_harm)
        models_AM_inharm.append(trf_AM_inharm)

        print("Obywatelu jest sukces:",subject)

    #if error
    except Exception as e:
        print("ERROR:", subject, repr(e))
        import traceback
        traceback.print_exc()
        continue

    print("\nDONE")
# %%

feature_names = {
    0: 'Envelope',
    1: 'Onsets',
    2: 'Pitch surprisal (S_p)',
    3: 'Length surprisal (S_o)',
    4: 'Pitch entropy (H_p)',
    5: 'Length entropy (H_o)',}



avg_model_AM_harm = plot_group_trf(models_AM_harm, "Harmonic","harm")
avg_model_AM_inharm = plot_group_trf(models_AM_inharm,"Inharmonic","inharm")
