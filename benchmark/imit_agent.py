#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles.
The agent also responds to traffic lights. """

import tensorflow as tf

from tensorflow.keras.models import load_model
import h5py
from tensorflow.keras import __version__ as keras_version
import numpy as np
from PIL import Image
import os
import sys
import math
import cv2
import random

# ==============================================================================
# -- add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
from carla import ColorConverter as cc

from agents.navigation.agent import Agent, AgentState
from agents.navigation.local_planner import LocalPlanner
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO


class ImitAgent(Agent):
    """
    BasicAgent implements a basic agent that navigates scenes to reach a given
    target destination. This agent respects traffic lights and other vehicles.
    """
    front_image = None

    def __init__(self, vehicle, model_path="model/", target_speed=26, image_cut=[115, 510]):
        """

        :param vehicle: actor to apply to local planner logic onto
        """
        super(ImitAgent, self).__init__(vehicle)

        self._proximity_threshold = 10.0  # meters
        self._state = AgentState.NAVIGATING
        args_lateral_dict = {
            'K_P': 1,
            'K_D': 0.02,
            'K_I': 0,
            'dt': 1.0 / 20.0}
        self._local_planner = LocalPlanner(
            self._vehicle, opt_dict={'target_speed': target_speed,
                                     'lateral_control_dict': args_lateral_dict})
        self._hop_resolution = 1.5
        self._path_seperation_hop = 3
        self._path_seperation_threshold = 0.5
        self._target_speed = target_speed
        self._grp = None

        self._image_cut = image_cut
        self._image_size = (88, 200, 3)

        config_gpu = tf.ConfigProto()  # tf 설정 프로토콜인듯?
        config_gpu.gpu_options.per_process_gpu_memory_fraction = 0.25  # memory_fraction % 만큼만 gpu vram 사용

        file_name = model_path + "model.h5"
        self.model = load_model(file_name)

        self.is_collision = False
        self.speed = 0

    def set_destination(self, location, start_loc=None):
        """
        This method creates a list of waypoints from agent's position to destination location
        based on the route returned by the global router
        """

        if start_loc is not None:
            start_waypoint = self._map.get_waypoint(carla.Location(start_loc[0], start_loc[1], start_loc[2]))
        else:
            start_waypoint = self._map.get_waypoint(self._vehicle.get_location())

        end_waypoint = self._map.get_waypoint(
            carla.Location(location[0], location[1], location[2]))

        route_trace = self._trace_route(start_waypoint, end_waypoint)
        assert route_trace

        self._local_planner.set_global_plan(route_trace)

        self._local_planner.change_intersection_hcl()

    def _trace_route(self, start_waypoint, end_waypoint):
        """
        This method sets up a global router and returns the optimal route
        from start_waypoint to end_waypoint
        """

        # Setting up global router
        if self._grp is None:
            dao = GlobalRoutePlannerDAO(self._vehicle.get_world().get_map(), self._hop_resolution)
            grp = GlobalRoutePlanner(dao)
            grp.setup()
            self._grp = grp

        # Obtain route plan
        route = self._grp.trace_route(
            start_waypoint.transform.location,
            end_waypoint.transform.location)

        return route

    def run_step(self, debug=False):
        """
        Execute one step of navigation.
        :return: carla.VehicleControl
        """

        self._state = AgentState.NAVIGATING
        # standard local planner behavior
        self._local_planner.buffer_waypoints()

        direction = self.get_high_level_command()
        v = self._vehicle.get_velocity()
        speed = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)  # use m/s
        self.speed = speed * 3.6  # use km/s

        throttle_control = self._local_planner.run_step(debug=debug)  # steer 값은 기존의 local planner 이용

        control = self._compute_action(ImitAgent.front_image)

        control.throttle = throttle_control.throttle
        return control

    def _compute_action(self, rgb_image):
        """
        Calculate steer, gas, brake from image input
        :return: carla.VehicleControl
        """

        rgb_image.convert(cc.Raw)

        image_cut = [230, 480, 160, 640]
        image_resize = (200, 400, 3)
        w = image_resize[1]
        h = image_resize[0]

        src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dst = np.float32([[100, 0], [300, 0], [0, h], [w, h]])
        M = cv2.getPerspectiveTransform(dst, src)
        if not self:
            return

        # carla.Image 를 기존 manual_control.py.CameraManager._parse_image() 부분을 응용
        rgb_image.convert(cc.Raw)
        array = np.frombuffer(rgb_image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (rgb_image.height, rgb_image.width, 4))
        array = array[image_cut[0]:image_cut[1], image_cut[2]:image_cut[3], :3]  # 필요 없는 부분을 잘라내고
        array = array[:, :, ::-1]  # 채널 색상 순서 변경? 안 하면 색 이상하게 출력

        image_pil = Image.fromarray(array.astype('uint8'), 'RGB')
        image_pil = image_pil.resize((image_resize[1], image_resize[0]))  # 원하는 크기로 리사이즈
        # image_pil.save('output/%06d.png' % image.frame)
        np_image = np.array(image_pil, dtype=np.dtype("uint8"))

        # bird-eye view transform
        # https://nikolasent.github.io/opencv/2017/05/07/Bird%27s-Eye-View-Transformation.html
        image_input = cv2.warpPerspective(np_image, M, (w, h))

        # Control() 대신 VehicleControl() 으로 변경됨 (0.9.X 이상)
        control = carla.VehicleControl()
        control.steer = float(self.model.predict(image_input[None, :, :, :], batch_size=1))
        control.throttle = 0
        control.brake = 0

        control.hand_brake = 0
        control.reverse = 0

        return control

    def get_high_level_command(self):
        # convert new version of high level command to old version
        def hcl_converter(_hcl):
            from agents.navigation.local_planner import RoadOption
            REACH_GOAL = 0.0
            GO_STRAIGHT = 5.0
            TURN_RIGHT = 4.0
            TURN_LEFT = 3.0
            LANE_FOLLOW = 2.0

            if _hcl == RoadOption.STRAIGHT:
                return GO_STRAIGHT
            elif _hcl == RoadOption.LEFT:
                return TURN_LEFT
            elif _hcl == RoadOption.RIGHT:
                return TURN_RIGHT
            elif _hcl == RoadOption.LANEFOLLOW or _hcl == RoadOption.VOID:
                return LANE_FOLLOW
            else:
                return REACH_GOAL

        hcl = self._local_planner.get_high_level_command()
        return hcl_converter(hcl)

    def is_reached_goal(self):
        return self._local_planner.is_waypoint_queue_empty()

    def reset_destination(self):
        sp = self._map.get_spawn_points()
        rand_sp = random.choice(sp)

        control_reset = carla.VehicleControl()
        control_reset.steer, control_reset.throttle, control_reset.brake = 0.0, 0.0, 0.0
        self._vehicle.apply_control(control_reset)
        self._vehicle.set_transform(rand_sp)

        spawn_point = random.choice(self._map.get_spawn_points())
        self.set_destination((spawn_point.location.x, spawn_point.location.y, spawn_point.location.z),
                             start_loc=(rand_sp.location.x, rand_sp.location.y, rand_sp.location.z))
