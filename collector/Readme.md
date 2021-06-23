# Description
collecting data for training model

by default, this will collect image from front-facing camera only. to save left and right-facing image, uncomment

```
# left_save_img.save(path + left_img_name + ".jpg")
# right_save_img.save(path + right_img_name + ".jpg")
```
in Recorder.py

## how to change image size, image processing
change code FrontCamera._parse_image in game_collector.py

default image size is 400 * 200, maximum image size is 800 * 600

if you want to change maximum size, change sensor input size

## keyboard settings
- c : change weather
- r : toggle record
- n : toggle noise
- t : reset control and location
- Ctrl + Q or ESC : exit
- F1 : toggle help

# How to run
if you don't have "output" folder, make folder named "output" by
```
mkdir output
```


to collect data,
```
python data_collector.py
```
