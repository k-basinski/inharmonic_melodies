#%%
from pydub import AudioSegment
#%%
# Open an wav file
wav_full_file = 'paradigm/soundpool/harm_1.wav'
song = AudioSegment.from_file(wav_full_file,
                              format="wav")
#%%
# start and end time
start_sec = 22
end_sec = 32

# pydub does things in milliseconds, so convert time
start = (start_sec)*1000
end = (end_sec)*1000

# song clip of 10 seconds from starting
first_10_seconds = song[start: end]

#%%
# save file
first_10_seconds.export("stimulus/cut_wav/test_2/test_3", format="wav")
print("New Audio file is created and saved")


class XD:
    def __init__(self):
        pass


obj = XD()

obj.json