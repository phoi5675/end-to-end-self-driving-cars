# Description
a test program for trained model.

base file for benchmark file, so method for running this code is same as benchmark program in this repo.

if you want to change model, you should edit :
1. camera angle in FrontCamera in game_imitation.py
2. image processing code in Imitagent._compute_action() in imit_agent.py

# How to run
1. run CARLA
2. change map; Town04 or Town06. you can do this by running config.py in ${carla_simulator}/PythonAPI/util/


for example,
```
python config -m Town04
```
3. put trained model.h5 in ./model/
4. run test program

```
python run_imit_agent.py
```
