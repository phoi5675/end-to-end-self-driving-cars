# Description
a simple benchmark program for trained model.

this benchmark will run test for 10 times, each test ends when the vehicle collides with obstacles;traffic lights, guardrails, etc.

it will calculate moved distance in meters, and the result will be printed when all test is finished.

the result will not be saved as a file. 

# How to run the benchmark
1. run CARLA
2. change map; Town04 or Town06. you can do this by running config.py in ${carla_simulator}/PythonAPI/util/


for example,
```
python config.py -m Town04
```
3. run benchmark

```
python run_imit_agent.py
```

# How to change model
change model.h5 file in ./model/
