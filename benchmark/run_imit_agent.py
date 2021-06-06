#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
    Example of automatic vehicle control from client side.
"""

from __future__ import print_function

import imit_agent
import glob
import os
import sys

try:
    import pygame

except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

# ==============================================================================
# -- import controller ---------------------------------------------------------
# ==============================================================================

from game_imitation import *


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- game_loop() ---------------------------------------------------------
# ==============================================================================

def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    tick_double_time = False  # delta_seconds 의 두 배 시간으로 녹화하게 만듦
    fps = 20
    cur_count = 0
    test_count = 10
    distance_traveled = 0
    now = 0
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(4.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        server_world = client.get_world()
        settings = server_world.get_settings()
        settings.fixed_delta_seconds = 1 / fps

        # set synchronous mode
        # server_world.apply_settings(settings)

        hud = HUD(args.width, args.height)
        world = World(server_world, hud, args)
        controller = KeyboardControl(world, False)

        agent = ImitAgent(world.player)
        spawn_point = world.map.get_spawn_points()[0]
        agent.set_destination((spawn_point.location.x,
                               spawn_point.location.y,
                               spawn_point.location.z))

        world.agent = agent
        world.collision_sensor.agent = agent

        clock = pygame.time.Clock()
        while cur_count < test_count:
            # tick_busy_loop(FPS) : 수직동기화랑 비슷한 tick() 함수
            clock.tick_busy_loop(fps)

            if controller.parse_events(client, world, clock):
                return

            world.tick(clock)
            # server_world.tick()  # 서버 시간 tick

            world.render(display)
            pygame.display.flip()

            control = agent.run_step()

            control.manual_gear_shift = False
            world.player.apply_control(control)

            # 이동거리
            # get distance traveled
            if hud.simulation_time - now >= 1:
                now = hud.simulation_time
                distance_traveled += agent.speed / 3.6

            # 충돌 확인
            if agent.is_collision:
                agent.is_collision = False
                cur_count += 1
                print("set new waypoint")
                agent.reset_destination()
                time.sleep(3)

    finally:
        if world is not None:
            world.destroy()
        print(distance_traveled)
        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--path',
        default='output/',
        help='path for saving data')
    argparser.add_argument(
        '--avoid-stopping',
        default=True,
        action='store_false',
        help=' Uses the speed prediction branch to avoid unwanted agent stops'
    )
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
