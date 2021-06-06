# Description
train model with obtained data from data_collector.py

model.py uses left, center, right images,

model_noise.py uses center image only.

# How to run
put all collected images in data/IMG/

put driving_log.csv in data/

to use all of three images, run
```
python model.py
```
if you want to use only center image, run 
```
python model_noise.py
```
