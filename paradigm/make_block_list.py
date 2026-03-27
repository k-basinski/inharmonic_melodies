# %%
import numpy as np
import pandas as pd


# %%

# %%
def generate_blocks(pid):
    melodies = list(range(1, 31))
    # losowo wybieramy wersję dla każdego sid (15/15)
    harmonicity = np.random.permutation(['harm'] * 15 + ['inh'] * 15)
    df = pd.DataFrame({
        'sid': melodies,
        'harmonicity': harmonicity
    })
    df['harmonicity_trigs'] = df['harmonicity'].map({'inh': 100, 'harm': 0})
    df['pid'] = pid
    df['trig'] = df['harmonicity_trigs'] + df['sid']
    df['filename'] = df['harmonicity'] + "_" + df['sid'].astype(str) + ".wav"
    df = df.sample(frac=1).reset_index(drop=True)

    return df


# %%
df = generate_blocks(1)
# %%
for p in range(60):
    d = generate_blocks(p)
    d.to_csv(f'soundpool/p{p}_blocks.csv')

