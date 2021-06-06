import math
import numpy as np
import h5py
import os


class Recorder(object):
    # High-level commands
    VOID = -1
    '''
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    CHANGELEFT = 4
    CHANGERIGHT = 5
    LANEFOLLOW = 6
    TRLIGHT_STOP = 7  # 이거 제외하고 나머지는 원래 있던 변수
    '''

    REACH_GOAL = 0.0
    GO_STRAIGHT = 5.0
    TURN_RIGHT = 4.0
    TURN_LEFT = 3.0
    LANE_FOLLOW = 2.0

    # 순서는 carla IL 의 datasheet 순서를 따름
    image = None

    steer = 0.0
    gas = 0.0
    brake = 0.0
    hand_brake = False
    reverse_gear = False
    steer_noise = 0.0  # noise 값은 학습에 영향 x
    gas_noise = 0.0
    brake_noise = 0.0
    x_pos = 0.0
    y_pos = 0.0
    speed = 0.0
    # 센서에서 얻는 데이터
    collision_other = 0.0
    collision_ped = 0.0
    collision_car = 0.0
    opposite_line_enter = 0.0
    sidewalk_intersect = 0.0
    acc_x = 0.0
    acc_y = 0.0
    acc_z = 0.0
    platform_time = 0.0
    game_time = 0.0  # not used
    orientation_x = 0.0
    orientation_y = 0.0
    orientation_z = 0.0

    high_level_com = 0
    noise = False  # not used
    camera = 0  # not used
    angle = 0  # not used

    # h5 파일에 저장 할 때, 데이터를 200개씩 모아서 한 번에 저장
    SAVE_IMG_ARY = []
    SAVE_DATA_ARY = []
    DATA_LEN = 200

    @staticmethod
    def record(world, path='output/', agent=None):
        def _save_h5_file():
            file_path = os.getcwd() + '/' + path
            c = len(os.listdir(file_path))
            # file name : data_#####.h5
            file_name = file_path + 'data_{0:05d}.h5'.format(c)

            with h5py.File(file_name, 'w') as f:
                # 이미지 저장 순서는 개수 * 세로 * 가로 * RGB (h5 파일은 행렬 방식으로 해석)
                save_img_ary = np.array(Recorder.SAVE_IMG_ARY)
                print(file_name)
                f.create_dataset('rgb', data=save_img_ary)
                # img_dset = f.create_dataset('rgb', shape=(200, 88, 200, 3), dtype='uint8')
                f.create_dataset('targets', data=Recorder.SAVE_DATA_ARY, dtype=np.float32)

        def _clear_save_ary():
            Recorder.SAVE_DATA_ARY.clear()
            Recorder.SAVE_IMG_ARY.clear()

        def _get_hcl_from_agent(_agent):
            Recorder.high_level_com = _agent.get_high_level_command()

        v = world.player.get_velocity()
        t = world.player.get_transform()
        c = world.player.get_control()
        acc = world.player.get_acceleration()

        # 순서는 carla IL 의 datasheet 순서를 따름
        Recorder.steer = c.steer
        Recorder.gas = c.throttle
        Recorder.brake = c.brake
        Recorder.hand_brake = c.hand_brake
        Recorder.reverse_gear = c.reverse

        Recorder.x_pos = t.location.x
        Recorder.y_pos = t.location.y
        Recorder.speed = 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)

        Recorder.acc_x = acc.x
        Recorder.acc_y = acc.y
        Recorder.acc_z = acc.z
        Recorder.platform_time = world.hud.simulation_time
        # Recorder.game_time
        Recorder.orientation_x = t.rotation.roll
        Recorder.orientation_y = t.rotation.pitch
        Recorder.orientation_z = t.rotation.yaw

        # automatic agent 를 이용하여 데이터를 수집하는 경우, agent 를 인자로 넣어서 high-level command 를 입력 받음
        if agent is not None:
            _get_hcl_from_agent(agent)

        DATA_ARY = [Recorder.steer, Recorder.gas, Recorder.brake, Recorder.hand_brake, Recorder.reverse_gear,
                    Recorder.steer_noise, Recorder.gas_noise, Recorder.brake_noise,
                    Recorder.x_pos, Recorder.y_pos, Recorder.speed,
                    Recorder.collision_other, Recorder.collision_ped, Recorder.collision_car,
                    Recorder.opposite_line_enter, Recorder.sidewalk_intersect,
                    Recorder.acc_x, Recorder.acc_y, Recorder.acc_z,
                    Recorder.platform_time, Recorder.game_time,
                    Recorder.orientation_x, Recorder.orientation_y, Recorder.orientation_z,
                    Recorder.high_level_com, Recorder.noise, Recorder.camera, Recorder.angle]

        # boolean 값은 0.0 or 1.0 으로 저장
        # 배열 내용 추가
        if len(Recorder.SAVE_IMG_ARY) < Recorder.DATA_LEN and Recorder.image is not None:
            Recorder.SAVE_IMG_ARY.append(Recorder.image)
        if len(Recorder.SAVE_DATA_ARY) < Recorder.DATA_LEN:
            Recorder.SAVE_DATA_ARY.append(DATA_ARY)

        if len(Recorder.SAVE_IMG_ARY) == Recorder.DATA_LEN \
                and len(Recorder.SAVE_DATA_ARY) == Recorder.DATA_LEN:
            _save_h5_file()
            _clear_save_ary()

        '''
        for _list in DATA_ARY:
            print(_list, end=' ')
        print('\n')
        if Recorder.image is not None:  # 동기화 문제 때문에 image 가 None 상태 유지되는 경우 예방
            Recorder.image.save('output/%.3f.png' % Recorder.platform_time)
        '''
