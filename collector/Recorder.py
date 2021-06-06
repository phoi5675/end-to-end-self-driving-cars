import math
import numpy as np
import os
import threading
import datetime
from PIL import Image
import csv
import random


class Recorder(object):
    # 순서는 carla IL 의 datasheet 순서를 따름
    center_image = None
    left_image = None
    right_image = None
    noise = False

    @staticmethod
    def record(world, path='output/', agent=None):
        if agent is not None and agent.weird_reset_count >= 2:
            agent.weird_reset_count += 1
            print("reset")

        if agent is not None and (agent.get_high_level_command() == 5 or agent.get_high_level_command() == 6):
            return

        file_path = os.getcwd() + '/'
        # file name : driving_log.csv
        file_name = file_path + 'driving_log.csv'

        now = datetime.datetime.now()
        now_date = now.strftime('%Y_%m_%d_%H_%M_%f')
        left_img_name = "left" + now_date
        center_img_name = "center" + now_date
        right_img_name = "right" + now_date

        if Recorder.center_image is None or \
                Recorder.right_image is None or \
                Recorder.left_image is None:
            return
        c = world.player.get_control()
        if abs(c.steer) > 0.5 or (random.random() < 0.85 and abs(c.steer) < 0.005):
            return
        
        # save csv
        with open(file_name, 'a+', newline='') as f:
            csv_writer = csv.writer(f)

            v = world.player.get_velocity()

            speed = 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
            steer = c.steer
            throttle = c.throttle
            brake = c.brake

            csv_writer.writerow([center_img_name, left_img_name, right_img_name,
                                 steer, throttle, brake, speed])

        # save image
        center_save_img = Image.fromarray(Recorder.center_image)
        left_save_img = Image.fromarray(Recorder.left_image)
        right_save_img = Image.fromarray(Recorder.right_image)

        center_save_img.save(path + center_img_name + ".jpg")
        # left_save_img.save(path + left_img_name + ".jpg")
        # right_save_img.save(path + right_img_name + ".jpg")
