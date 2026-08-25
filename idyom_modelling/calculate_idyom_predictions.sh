#!/bin/bash

# this script runs from within the `idyom_modelling` folder, on a Linux/MacOS system

# go to IDyOMpy folder
cd ../IDyOMpy

# point this to your conda install (miniforge3, miniconda3 or others)
source ~/miniforge3/etc/profile.d/conda.sh

# activate dedicated IDyOMpy conda environment (see IDyOMpy documentation)
conda activate idyompyenv

# train IDyOM on largeWestern_wt_bach corpus (large Western without Bach)
# test on 32 melodies used in this experiment
# run separately for pitch and onset viewpoints

# run for pitch
python App.py -t dataset/largeWestern_wt_bach -s ../stimulus/midi/monophonic_midi -v pitch


# run the same for onsets
python App.py -t dataset/largeWestern_wt_bach -s ../stimulus/midi/monophonic_midi -v length

# run the same for both pitch and onsets
python App.py -t dataset/largeWestern_wt_bach -s ../stimulus/midi/monophonic_midi -v pitch,length

# move the output to idyom_modelling folder
mv out/monophonic_midi/surprises/largeWestern_wt_bach/data/* ../idyom_modelling

# remove *.mat files as we don't need them
rm ../idyom_modelling/*.mat
