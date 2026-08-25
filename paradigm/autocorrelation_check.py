# %%
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import correlate

# open files
harm_full, sr = sf.read("soundpool/harm_1.wav")
inharm_full, sr = sf.read("soundpool/inh_1.wav")

# %%

# pick a window in sounds and cut
s_start, s_len = 6, 0.2
s_start_smpl, s_end_smpl = int(s_start * sr), int((s_start + s_len) * sr)

# plotting window
plotting_right, plotting_left = 0.1, 0.2

# calculate autocorrelations
harm = harm_full[s_start_smpl:s_end_smpl, 0]
inharm = inharm_full[s_start_smpl:s_end_smpl, 0]
xcorr_harm = correlate(harm, harm)[-len(harm) :]
xcorr_inharm = correlate(inharm, inharm)[-len(harm) :]
t_units = np.arange(len(xcorr_harm)) / sr

fig, ax = plt.subplots()

ax.plot(t_units, xcorr_harm, label="harmonic")
ax.plot(t_units, xcorr_inharm, label="inharmonic")
ax.set_xlim(plotting_right, plotting_left)
ax.legend()
plt.show()
# %%
xcorr_harm
# %%
harm.shape
