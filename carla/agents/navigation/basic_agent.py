#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles.
The agent also responds to traffic lights. """

import carla
from agents.navigation.agent import Agent, AgentState
from agents.navigation.local_planner import LocalPlanner
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.navigation.local_planner import RoadOption

import math
import random
import time
import os
import time


class BasicAgent(Agent):
    """
    BasicAgent implements a basic agent that navigates scenes to reach a given
    target destination. This agent respects traffic lights and other vehicles.
    """

    def __init__(self, vehicle, target_speed=20):
        """

        :param vehicle: actor to apply to local planner logic onto
        """
        super(BasicAgent, self).__init__(vehicle)

        self._proximity_threshold = 10.0  # meters
        self._state = AgentState.NAVIGATING
        args_lateral_dict = {
            'K_P': 1,
            'K_D': 0,
            'K_I': 0.005,
            'dt': 0.2}
        # 0.65, 0, 0.05, 0.2
        # P : 꺾이는 정도, I : 수렴 지점으로 진동 정도
        self._local_planner = LocalPlanner(
            self._vehicle, opt_dict={'target_speed': target_speed,
                                     'lateral_control_dict': args_lateral_dict})
        self._hop_resolution = 1.0
        self._path_seperation_hop = 2
        self._path_seperation_threshold = 0.5
        self._target_speed = target_speed
        self._grp = None

        self._radar_data = None
        self._obstacle_ahead = False
        self._obstacle_far_ahead = False
        self._speed = 0.0

        self.noise_steer = 0
        self.noise_steer_max = 0
        self.noise_start_time = 0
        self.noise_active_duration = 0
        self.is_noise_increase = True
        self.noise_bias = 0
        self.noise_duration = 5
        self.steer = 0

        self.weird_steer_count = 0
        self.weird_reset_count = 0

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

        self._local_planner.change_intersection_hcl(enter_hcl_len=2, exit_hcl_len=3)

        self.weird_steer_count = 0
        self.weird_reset_count = 0

        print("set new waypoint")

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

        v = self._vehicle.get_velocity()
        c = self._vehicle.get_control()

        speed = 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        self._speed = speed

        control = self._local_planner.run_step(debug=debug)

        # check maneuvering
        if self.is_maneuvering_weird(control):
            self.weird_steer_count += 1

        if self.weird_steer_count >= 10:
            print("vehicle is steering in wrong way")
            self.weird_steer_count = 0
            self.weird_reset_count += 1

        if self.weird_reset_count > 2:
            self.reset_destination()

        if control.steer > 1:
            control.steer = 1
        elif control.steer < -1:
            control.steer = -1
        self.steer = control.steer

        return control

    # ====================================================================
    # ----- appended from original code ----------------------------------
    # ====================================================================
    # TODO changeleft/right 설정하기
    def get_high_level_command(self):
        # convert new version of high level command to old version
        def hcl_converter(_hcl):
            if _hcl == RoadOption.LEFT:
                return 1
            elif _hcl == RoadOption.RIGHT:
                return 2
            elif _hcl == RoadOption.STRAIGHT:
                return 3
            elif _hcl == RoadOption.LANEFOLLOW:
                return 4
            elif _hcl == RoadOption.CHANGELANELEFT:
                return 5
            elif _hcl == RoadOption.CHANGELANERIGHT:
                return 6

        # return self._local_planner.get_high_level_command()
        hcl = self._local_planner.get_high_level_command()
        return hcl_converter(hcl)

    def is_maneuvering_weird(self, control):
        hcl = self._local_planner.get_high_level_command()
        turn_threshold = 0.7
        change_lane_threshold = 0.65
        if hcl is RoadOption.STRAIGHT or hcl is RoadOption.LANEFOLLOW:
            turn_threshold = 0.5
            if abs(control.steer) >= turn_threshold:
                return True
        elif abs(control.steer) > turn_threshold:
            return True
        elif hcl is RoadOption.LEFT and control.steer >= turn_threshold:
            return True
        elif hcl is RoadOption.RIGHT and control.steer <= -turn_threshold:
            return True
        elif (hcl is RoadOption.CHANGELANERIGHT or hcl is RoadOption.CHANGELANELEFT) \
                and abs(control.steer) >= change_lane_threshold:
            return True
        else:
            return False

    def is_reached_goal(self):
        return self._local_planner.is_waypoint_queue_empty()

    def is_dest_far_enough(self):
        return self._local_planner.is_dest_far_enough()

    def set_radar_data(self, radar_data):
        self._radar_data = radar_data

    def set_stop_radar_range(self):
        hlc = self._local_planner.get_high_level_command()
        c = self._vehicle.get_control()
        sign = 1 if c.steer >= 0 else -1
        steer = abs(c.steer) * 50
        # 교차로 주행 시
        if hlc is RoadOption.RIGHT or hlc is RoadOption.LEFT:
            yaw_angle = 30
        elif hlc is RoadOption.STRAIGHT:
            yaw_angle = 20
        else:  # 교차로 아닌 경우
            yaw_angle = 15

        # right turn
        if sign > 0:
            left_offset = -steer * 0.7
            right_offset = steer
        else:
            left_offset = -steer
            right_offset = steer

        return -(yaw_angle + left_offset), (yaw_angle + right_offset)

    def set_target_speed(self, speed):
        self._local_planner.set_target_speed(speed)

    def is_obstacle_ahead(self, _rotation, _detect):
        left_radar_range, right_radar_range = self.set_stop_radar_range()

        threshold = max(self._speed * 0.25, 3)
        if -7.5 <= _rotation.pitch <= 3 and left_radar_range <= _rotation.yaw <= right_radar_range and \
                _detect.depth <= threshold:
            return True
        return False

    def is_obstacle_far_ahead(self, _rotation, _detect):
        radar_range = self.set_stop_radar_range()
        c = self._vehicle.get_control()
        steer = abs(c.steer) * 4
        left_margin = steer if c.steer < 0 else 0
        right_margin = steer if c.steer > 0 else 0
        if 1.0 <= _rotation.pitch <= 5.0 and -(5 + left_margin) <= _rotation.yaw <= (5 + right_margin) \
                and _detect.depth <= 15:
            return True
        return False

    def run_radar(self):
        if self._radar_data is None:
            return False
        current_rot = self._radar_data.transform.rotation
        self._obstacle_ahead = False
        self._obstacle_far_ahead = False

        for detect in self._radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)

            rotation = carla.Rotation(
                pitch=current_rot.pitch + alt,
                yaw=azi,
                roll=current_rot.roll)

            if self.is_obstacle_ahead(rotation, detect):
                self._obstacle_ahead = True
            elif self.is_obstacle_far_ahead(rotation, detect):
                self._obstacle_far_ahead = True

    def reset_destination(self):
        sp = self._map.get_spawn_points()
        rand_sp = random.choice(sp)

        control_reset = carla.VehicleControl()
        control_reset.steer, control_reset.throttle, control_reset.brake = 0.0, 0.0, 0.0
        self._vehicle.apply_control(control_reset)
        self._vehicle.set_transform(rand_sp)

        time.sleep(3.0)
        spawn_point = random.choice(self._map.get_spawn_points())
        self.set_destination((spawn_point.location.x, spawn_point.location.y, spawn_point.location.z),
                             start_loc=(rand_sp.location.x, rand_sp.location.y, rand_sp.location.z))

        self.weird_reset_count = 0
        self.weird_steer_count = 0

    def noisy_agent(self):
        cur_time = time.time()
        if self.noise_start_time == 0:
            signed = random.choice([-1, 1])
            self.noise_start_time = cur_time
            self.noise_steer_max = random.uniform(0.3, 0.35) * signed
            self.noise_steer = 0.1 * signed
            self.noise_bias = random.uniform(0.01, 0.015) * signed
            self.noise_active_duration = self.noise_duration * random.uniform(0.3, 0.5)
        elif cur_time - self.noise_start_time > self.noise_duration:
            self.noise_start_time = 0
            self.noise_steer_max = 0
            self.noise_steer = 0
            self.noise_bias = 0
            self.is_noise_increase = True
        elif self.noise_active_duration < cur_time - self.noise_start_time < self.noise_duration:
            return 0.0
        elif cur_time - self.noise_start_time < self.noise_active_duration:
            if abs(self.noise_steer) < abs(self.noise_steer_max) and self.is_noise_increase:
                self.noise_steer += random.uniform(0, self.noise_bias * random.uniform(0.5, 1.3))
            elif abs(self.noise_steer) > abs(self.noise_steer_max) * 0.9 or self.is_noise_increase is False:
                self.is_noise_increase = False
                self.noise_steer -= random.uniform(0, self.noise_bias * random.uniform(0.5, 1.3))
            '''
            if abs(self.noise_steer) < abs(self.noise_bias) * 1.5:
                self.noise_steer = 0
            '''
            return self.noise_steer

        return 0.0
