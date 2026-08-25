# %%
# pyright: basic
import pickle

import matplotlib.pyplot as plt
import mido
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import envelope, resample

# input midi file
midi_fpath = "stimulus/midi/monophonic/"
idyom_pitch_length = "idyom_modelling/largeWestern_wt_bach_quantization_24_maxOrder_20_viewpoints_pitch_length.pickle"
idyom_pitch = "idyom_modelling/largeWestern_wt_bach_quantization_24_maxOrder_20_viewpoints_pitch.pickle"
idyom_length = "idyom_modelling/largeWestern_wt_bach_quantization_24_maxOrder_20_viewpoints_length.pickle"
idyom_vectors_fpath = "idyom_modelling/idyom_vectors.pickle"


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


def load_idyom_output(idyom_obj_path):
    with open(idyom_pitch_length, "rb") as file:
        idyom_results = pickle.load(file)

    return idyom_results


def main():
    sr = 64
    sig_length = 120  # in seconds
    melodies = list(range(1, 33))
    idyom_vectors = {}

    # open idyom results
    with open(idyom_pitch_length, "rb") as file:
        idyom_pitch_length_results = pickle.load(file)
    with open(idyom_pitch, "rb") as file:
        idyom_pitch_results = pickle.load(file)
    with open(idyom_length, "rb") as file:
        idyom_length_results = pickle.load(file)

    # for each melody
    for melody in melodies:
        print(f"Calculating melody {melody}...")
        # get idyom results from pickle
        idyr_pitch_length = extract_idyom_results(
            idyom_pitch_length_results, f"harm_{melody}_mono"
        )
        idyr_length = extract_idyom_results(idyom_length_results, f"harm_{melody}_mono")
        idyr_pitch = extract_idyom_results(idyom_pitch_results, f"harm_{melody}_mono")

        # truncate to 120 seconds
        idyr_trunc_pitch_length = idyr_pitch_length[
            idyr_pitch_length["note_time"] < 120
        ]
        idyr_trunc_length = idyr_length[idyr_pitch_length["note_time"] < 120]
        idyr_trunc_pitch = idyr_pitch[idyr_pitch["note_time"] < 120]

        # make empty vectors
        vect_IC_pitch_length = vect_entropy_pitch_length = np.zeros(sig_length * sr)
        vect_IC_length = vect_entropy_length = np.zeros(sig_length * sr)
        vect_IC_pitch = vect_entropy_pitch = np.zeros(sig_length * sr)

        # figure out note indices
        note_indices = np.floor(idyr_trunc_pitch["note_time"] * sr).astype(int)

        # place idyom results where notes are
        vect_IC_pitch_length[note_indices] = idyr_trunc_pitch_length["IC"]
        vect_IC_length[note_indices] = idyr_trunc_length["IC"]
        vect_IC_pitch[note_indices] = idyr_trunc_pitch["IC"]
        vect_entropy_pitch_length[note_indices] = idyr_trunc_pitch_length["entropy"]
        vect_entropy_length[note_indices] = idyr_trunc_length["entropy"]
        vect_entropy_pitch[note_indices] = idyr_trunc_pitch["entropy"]
        idyom_vectors[melody] = {
            "pitch_IC": vect_IC_pitch,
            "length_IC": vect_IC_pitch_length,
            "pitch_length_IC": vect_IC_pitch_length,
            "pitch_H": vect_entropy_pitch,
            "length_H": vect_entropy_pitch_length,
            "pitch_length_H": vect_entropy_pitch_length,
        }

    with open(idyom_vectors_fpath, "wb") as file:
        pickle.dump(idyom_vectors, file)

    visualise_model_results(idyom_vectors)


def visualise_model_results(idyom_vectors):
    # load sound and reality check
    sr = 64
    sig_length = 120
    melody_id = 2
    h2_wav, sr_wav = sf.read(f"paradigm/soundpool/harm_{melody_id}.wav")
    h2_trunc = h2_wav[: sr_wav * sig_length, 0]

    # calculate envelope and resample
    h2_env = envelope(h2_trunc, n_out=sr * sig_length, squared=True, residual=None)
    h2_resampled = resample(h2_trunc, sr * sig_length)

    # visualize
    tmin, tmax = 105 * sr, 108 * sr
    fig, axs = plt.subplots(nrows=6, ncols=1, sharex=True)

    # acoustic wave
    axs[0].plot(h2_resampled[tmin:tmax])
    axs[0].set_ylabel('Acoustic wave')

    # amplitude envelope
    axs[1].plot(np.abs(h2_env[tmin:tmax]))
    axs[1].set_yscale('log')
    axs[1].set_ylabel("Amplitude envelope")

    # Pitch entropy
    axs[2].plot(idyom_vectors[melody_id]["pitch_H"][tmin:tmax])
    axs[2].set_ylabel("Pitch entropy")

    # Length entropy
    axs[3].plot(idyom_vectors[melody_id]["length_H"][tmin:tmax])
    axs[3].set_ylabel("Length entropy")

    # Pitch IC
    axs[4].plot(idyom_vectors[melody_id]["pitch_IC"][tmin:tmax])
    axs[4].set_ylabel("Pitch IC")

    # Length IC
    axs[5].plot(idyom_vectors[melody_id]["length_IC"][tmin:tmax])
    axs[5].set_ylabel("Length IC")


    plt.tight_layout()
    plt.show()


# %%
if __name__ == "__main__":
    main()

    # iv
