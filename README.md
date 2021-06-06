# end-to-end-self-driving-cars
nvidia's end to end learning for self-driving cars implemented on carla
![image](https://github.com/phoi5675/end-to-end-self-driving-cars/blob/master/image/preview.gif)

final project for deep learning class(EI4446)

## overall process
1. collecting data
```
python collector/data_collector.py
```
2. train model
```
python train/model.py
```
3. run model
```
python benchmark/run_imit_agent.py
```


## how to run carla
to run carla simulator, see [Carla Simulator tutorial](https://carla.readthedocs.io/en/0.9.11/start_quickstart/)

make sure $PYTHONPATH is set in right path, and unzipped carla_dist.egg file on $PYTHONPATH

## requirements
### software
python 3.7

there's compatibility issue with .csv file and h5py 3.x. use h5py 2.x instead
```
pip install -r requirements.txt
```
### hardware (recommended)
GTX1080 or higher (tested on RTX2080 super)

i7-9700 or higher

16GB ram

## simple descriptions
### to get more info, please see Readme.md in each folder
- benchmark : to run benchmark 
- carla : after copying files from CARLA/PythonAPI/carla/*, overwrite files in this carla folder
- collector : to collect data
- imit_agent : to test trained model
- train : train the model

## acknowledgments
- base code for benchmark, collector and imit_agent : [carla simulator](https://github.com/carla-simulator/carla), [imitation learning](https://github.com/phoi5675/carlaIL)
- data collecting method(insert noise to steer) : [End-to-end Driving via Conditional Imitation Learning](https://arxiv.org/abs/1710.02410)
- network architecture : [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316)
- train code : [Behavioral Cloning End to End Deep Learning Project](https://github.com/abhileshborode/Behavorial-Clonng-Self-driving-cars)
