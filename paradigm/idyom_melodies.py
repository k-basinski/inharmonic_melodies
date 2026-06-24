# %%
import pickle

import matplotlib.pyplot as plt
import mido
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import envelope, resample

# input midi file
midi_fpath = "stimulus/midi/monophonic/"
output_fpath = "stimulus/midi/monophonic_midi/"
idyom_fpath = "stimulus/idyom/largeWestern_wt_bach_fits.pickle"
idyom_vectors_fpath = "stimulus/idyom/idyom_vectors.pickle"


def extract_track(poly_mid: mido.MidiFile, track_idx: int):
    mono_mid = poly_mid
    track = poly_mid.tracks[track_idx]
    mono_mid.tracks = []
    mono_mid.type = 0
    mono_mid.tracks.append(track)
    return mono_mid


def print_note_onsets(mono_mid):
    for msg in mono_mid:
        if msg.type == "note_on":
            print(msg)


def get_notes(midi_file):
    poly_midi = mido.MidiFile(f"{midi_fpath}{midi_file}.mid")

    timer = 0.0
    counter = 0
    rows = []
    for m in poly_midi:
        timer += m.time
        if m.type == "note_on":
            row = {
                "note_id": counter,
                "note_pitch": m.note,
                "note_time": timer,
                "midi_file": midi_file,
            }
            rows.append(row)
            counter += 1
    res_df = pd.DataFrame(rows)
    return res_df


def extract_idyom_results(idyom_object, midi_file):
    idyom_res = np.array(idyom_object[midi_file]).T
    idyom_df = pd.DataFrame(idyom_res, columns=["IC", "entropy"])
    note_df = get_notes(midi_file)
    res_df = pd.concat([note_df, idyom_df], axis=1)
    return res_df


def main():
    sr = 64
    sig_length = 120  # in seconds
    melodies = list(range(1, 33))
    idyom_vectors = {}

    with open(idyom_fpath, "rb") as file:
        idyom_results = pickle.load(file)

    for melody in melodies:
        print(f"Calculating melody {melody}...")
        idyr = extract_idyom_results(idyom_results, f"harm_{melody}_mono")
        idyr_trunc = idyr[idyr["note_time"] < 120]
        vect_IC = vect_entropy = np.zeros(sig_length * sr)
        note_indices = np.floor(idyr_trunc["note_time"] * sr).astype(int)
        vect_IC[note_indices] = idyr_trunc["IC"]
        vect_entropy[note_indices] = idyr_trunc["entropy"]
        idyom_vectors[melody] = {"IC": vect_IC, "entropy": vect_entropy}

    with open(idyom_vectors_fpath, "wb") as file:
        pickle.dump(idyom_vectors, file)


def reality_check(idyom_vectors):
    # load sound and reality check
    sr = 64
    sig_length = 120
    h2_wav, sr_wav = sf.read("../paradigm/soundpool/inh_2.wav")
    v_IC = idyom_vectors[2]["IC"]
    v_entropy = idyom_vectors[2]["entropy"]
    h2_trunc = h2_wav[: sr_wav * sig_length, 0]
    h2_env = envelope(h2_trunc, n_out=sr * sig_length, squared=True, residual=None)
    h2_resampled = resample(h2_trunc, sr * sig_length)
    # visualize
    tmin, tmax = 105 * sr, 108 * sr
    fig, axs = plt.subplots(nrows=4, ncols=1)
    axs[0].plot((np.abs(h2_resampled[tmin:tmax])))
    axs[1].plot((np.abs(h2_env[tmin:tmax])))
    # axs[0].set_ylim(0, 1e-9)
    # axs[0].set_yscale('log')
    axs[2].plot(v_IC[tmin:tmax])
    axs[3].plot(v_entropy[tmin:tmax])
    plt.show()


if __name__ == "__main__":
    main()
