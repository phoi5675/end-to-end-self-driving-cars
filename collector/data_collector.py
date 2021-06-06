#!/usr/bin/env python

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys

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

import carla
from agents.navigation.basic_agent import BasicAgent


from game_collector import *
from Recorder import *
import argparse
import pygame
import logging
# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    tick = True
    agent = None

    fps = 20

    try:
        # 서버 연결
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)

        # pygame 해상도 설정 / HWSURFACE, DOUBLEBUF 는 화면 플리커링 등 방지용
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        # TODO 본 테스트 할 때는 sync 모드 켜기
        # set synchronous mode
        server_world = client.get_world()
        settings = server_world.get_settings()
        # settings.synchronous_mode = True
        # settings.fixed_delta_seconds = 1 / fps

        # server_world.apply_settings(settings)

        world = World(server_world, hud, args)

        controller = KeyboardControl(world, args.autopilot)

        agent = BasicAgent(world.player, target_speed=26)

        spawn_point = world.map.get_spawn_points()[0]
        # agent.set_destination(agent.vehicle.get_location(), spawn_point.location, clean=True)
        agent.set_destination((spawn_point.location.x, spawn_point.location.y, spawn_point.location.z))
        world.agent = agent

        clock = pygame.time.Clock()

        while True:
            # tick_busy_loop(FPS) : 수직동기화랑 비슷한 tick() 함수
            clock.tick_busy_loop(fps)
            if controller.parse_events(client, world, clock):
                return
            world.tick(clock)
            # server_world.tick()  # 서버 시간 tick

            # 화면 제외 다른 데이터 녹화
            if world.recording_enabled and tick:
                Recorder.record(world, path=args.path, agent=agent)
                tick = False
            elif world.recording_enabled and tick is False:
                tick = True

            # agent.update_information()

            control = agent.run_step()
            control.manual_gear_shift = False


            # 노이즈 적용
            if Recorder.noise:
                control = noisy_agent(control, world.player, agent)

            world.player.apply_control(control)

            # 목표에 도착 한 경우, 새로운 WP 설정 -> random!
            if agent.is_reached_goal():
                '''
                spawn_point = world.map.get_spawn_points()
                goal = random.choice(spawn_point) if spawn_point else carla.Transform()
                agent.set_destination((goal.location.x, goal.location.y, goal.location.z))
                '''
                agent.reset_destination()

            world.render(display)
            pygame.display.flip()

    finally:
        if world and world.recording_enabled:
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()


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
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='800x600',
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
        '--path',
        default='output/',
        help='path for saving data')
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
