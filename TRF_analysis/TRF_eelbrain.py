# https://elifesciences.org/articles/85012
# %%
import eelbrain
from matplotlib import pyplot
import mne

# %%
STIMULUS_DIR = '/Users/zosiamikolajczak/ANL/ANL_inharmonic_melodies/data/sound/harmonic/*.wav'
EEG_DIR = DATA_ROOT / 'eeg'

# Load one subject's raw EEG file
SUBJECT = 'S18'
LOW_FREQUENCY = 0.5
HIGH_FREQUENCY = 20