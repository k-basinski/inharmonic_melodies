# %%
import numpy as np
import pandas as pd

# %%

def generate_blocks(pid):
    sounds_per_block = 10

    harmonicity = ['harmonic', 'inharmonic']
    harm_trigs = [0, 100]
    
    # make long list of everything
    sounds = [(i, lh, lht) for i in range(sounds_per_block) for lh,lht in zip(harmonicity, harm_trigs)]
    df = pd.DataFrame(sounds, columns=['sid', 'harmonicity', 'harmonicity_trigs'])
    
    # append pid
    df['pid'] = pid

    # calculate trigger value
    df['trig'] = df.harmonicity_trigs + df.sid
    
    # format filename
    df['filename'] = df.harmonicity + "_" + df.sid.astype(str) + ".wav"

    # permute order and return
    dfp = df.sample(frac=1)

    
    return dfp

# %%
generate_blocks(1)
# %%
for p in range(40):
    d = generate_blocks(p)
    d.to_csv(f'soundpool/p{p}_blocks.csv')
