from controller import Supervisor
from controller import Field

import subprocess
from simple_pid import PID

import cProfile
import ray

import cv2 as cv
import numpy as np
import time
from controller import Robot
ray.init(num_cpus=8)


@ray.remote(num_cpus=2)
class SupervisorClass(object):
    def __init__(self):
        self.supervisor = Supervisor()
        # manager = self.supervisor.getFromDef("manager")
        # manager.restartController()
        self.root_node = self.supervisor.getRoot()
        self.child_field = self.root_node.getField('children')
        self.time_step = 32
        self.shutdown = False

    def step(self):
        while not self.shutdown:
            print ("steping")
            self.supervisor.step(32)

    def get_create_robot_str(self, name, model, translation, rotation):
        robot_str = 'DEF ' + name + ' ' + model + ' { translation ' + translation + ' rotation ' + rotation + \
                                        ' controller \"<extern>\" \
                                        controllerArgs "--name=' + name + '" extensionSlot [\
                                        Accelerometer { lookupTable [ -39.24 -39.24 0.005 39.24 39.24 0.005 ]}\
                                        Gyro {lookupTable [-50 -50 0.005 50 50 0.005 ]}\
                                        SickLms291 {translation 0 0.23 -0.136 noise 0.1}\
                                        GPS {}\
                                        InertialUnit {}\
                                        ]}'
        return robot_str

    def reset(self):
        # self.supervisor.simulationReset()
        self.shutdown = True
        # manager = self.supervisor.getFromDef("manager")
        # manager.restartController()
    # def init_services(self):
    #     self.services['create_object'] = Service(self.root_service_str + "field/import_node_from_string",
    #                                              field_import_node_from_string)
    #     self.services['get_root'] = Service(self.root_service_str + "get_root", get_uint64)
    #     self.services['get_field'] = Service(self.root_service_str + "node/get_field", node_get_field)
    #     self.services['get_from_deff'] = Service(self.root_service_str + "get_from_def", supervisor_get_from_def)
    #     self.services['remove_object'] = Service(self.root_service_str + "node/remove", node_remove)
    #     self.services['reset'] = Service(self.root_service_str + 'simulation_reset', get_bool)
    #     self.services['get_state'] = Service(self.root_service_str + 'simulation_get_mode', get_int)
    #     self.is_collided = False
    #     self.root_node = self.services['get_root'].call(0).value
    #     self.child_field = self.services['get_field'].call(
    #         node_get_fieldRequest(node=self.root_node, fieldName='children')).field
    #     while True:
    #         try:
    #             time.sleep(0.1)
    #             self.services['get_state'].call(True)
    #             break
    #         except Exception as e:
    #             rospy.loginfo_throttle(1, "still waiting for reset to finish", e )

    def remove_object(self, defName):
        object = self.supervisor.getFromDef(defName)
        if object is not None:
            object.remove()

    def create_robot(self, name, model, translation, rotation):
        self.remove_object(defName=name)
        robot_str = self.get_create_robot_str(name, model, translation, rotation)

        self.child_field.importMFNodeFromString(-1, robot_str)

    def reset(self):
        pass

class History():
    def __init__(self, window_size, update_time):
        self.data = [None for x in range(window_size)]
        self.idx = 0
        self.update_time = update_time
        self.last_update = 0
        self.window_size = window_size
        # to keep track of last element always in self.idx
        self.update_idx = False

    def add_element(self, element):
        if self.update_idx:
            self.idx = (self.idx + 1) % self.window_size
            self.update_idx = True

        if self.data[self.idx] is None:
            for idx in range (self.window_size):
                self.data[idx] = element
        self.data[self.idx] = element
        # TODO fix time
        # if rospy.Time.now().secs - self.last_update > self.update_time:
        #   self.last_update = rospy.Time.now().secs
        self.update_idx = True


    def get_elemets(self):
        return_data = []
        for data in (self.data[self.idx:] + self.data[:self.idx]):
            if data is not None:
                return_data.append(data)

        return return_data

@ray.remote(num_cpus=2)
class RobotClass(object):
    def __init__(self, name, angle=0,max_speed=6.3):
        self.init_done = False
        self.robot = Robot()
        self.model = 'Pioneer3at'
        self.name = name
        self.collision_distance = 0.5
        self.wheels_left = ['back left wheel', 'front left wheel']
        self.wheels_right = ['back right wheel', 'front right wheel']
        self.max_speed = max_speed
        self.max_laser_range = 5.0  # meter
        self.width_laser_image = 100
        self.height_laser_image = 50
        self.pos_history = History(5, 1)
        self.pos = (None, None)
        self.time_step = 32
        self.angular_pid = PID(0.75, 0, 0.01, setpoint=0)
        self.linear_pid = PID(4, 0, 0.05, setpoint=0)
        self.orientation = angle
        self.orientation_history = History(5, 1)
        self.scan_image = None
        self.scan_image_history = History(5, 1)
        self.is_collided = False
        self.sick_laser = self.robot.getLidar("Sick LMS 291")
        self.sick_laser.enable(self.time_step)
        self.sick_fov = self.sick_laser.getFov()
        self.sick_max_range = self.sick_laser.getMaxRange()
        self.gps = self.robot.getGPS("gps")
        self.gps.enable(self.time_step)
        self.imu = self.robot.getInertialUnit("inertial unit")
        self.imu.enable(self.time_step)
        self.left_motors = [self.robot.getMotor(motor) for motor in self.wheels_left]
        self.right_motors = [self.robot.getMotor(motor) for motor in self.wheels_right]
        self.is_pause = False
        self.reset = False
        self.velocity_window = 3
        self.velocity_history = np.zeros((self.velocity_window))
        self.last_velocity_idx = 0
        self.shutdown = False
        self.init_done = True

    def get_pos(self):
        return self.pos

    def get_orientation(self):
        return self.imu.getRollPitchYaw()

    def step(self):
        # while not self.shutdown:
        print("{} pos: {} oriention: {} time: {}".format(self.name, self.get_pos(), self.get_orientation(), time.time()))
        self.robot.step(self.time_step)
        laser = self.laser_cb()
        print (len(laser), self.sick_fov)
        time.sleep(0.1)
        # return laser
        # print (laser[0].x, laser[0].y, laser[0].z, laser[0].time)

    def is_init(self):
        return self.init_done

    def get_laser_data(self):
        return self.sick_laser.getRangeImage()

    def __del__(self):
        self.shutdown = True
        self.robot.__del__()

    def get_velocity(self):
        return np.mean(self.velocity_history)

    def pause(self):
        self.is_pause = True

    def resume(self):
        self.is_pause = False

    def take_action(self, action):
        if self.is_pause:
            return
        speed_left = 0
        speed_right = 0
        if action == 0:
            pass  # stop robot
        elif action == 1:
            speed_left = speed_right = self.max_speed
        elif action == 2:
            speed_left = speed_right = -self.max_speed
        elif action == 3:
            speed_left = self.max_speed
            speed_right = -self.max_speed
        elif action == 4:
            speed_left = -self.max_speed
            speed_right = self.max_speed
        elif action == 5:
            speed_left = self.max_speed / 2
            speed_right = self.max_speed / 2
        elif action == 6:
            speed_left = self.max_speed / 2
            speed_right = -self.max_speed / 2
        elif action == 7:
            speed_left = -self.max_speed / 2
            speed_right = self.max_speed / 2

        for wheel in self.wheels_left:
            self.services[wheel + "_vel"].call(speed_left)
        for wheel in self.wheels_right:
            self.services[wheel + "_vel"].call(speed_right)

    def stop_robot(self):
        for wheel in self.wheels_left:
            self.services[wheel + "_vel"].call(0)
        for wheel in self.wheels_right:
            self.services[wheel + "_vel"].call(0)

    def angle_distance_to_point(self, pos):
        current_pos = self.get_pos()
        if current_pos[0] is None:
            return None, None
        angle = math.atan2(pos[1] - current_pos[1], pos[0] - current_pos[0])
        distance = math.hypot(pos[0] - current_pos[0], pos[1] - current_pos[1])
        angle = (angle - self.orientation + math.pi) % (math.pi * 2) - math.pi
        return angle, distance

    def go_to_pos(self, pos, stop_after_getting=False):
        distance = 2
        # diff_angle_prev = (angle - self.orientation + math.pi) % (math.pi * 2) - math.pi
        while not distance < 1.5:
            if self.is_pause or self.reset:
                return
            diff_angle, distance = self.angle_distance_to_point(pos)
            if distance is None:
                return
            # if abs(diff_angle_prev - diff_angle) > math.pi*3/:
            #     diff_angle = diff_angle_prev
            # else:
            #     diff_angle_prev = diff_angle
            # angular_vel = min(max(self.angular_pid(math.atan2(math.sin(angle-self.orientation), math.cos(angle-self.orientation)))*200, -self.max_speed/3),self.max_speed)
            angular_vel = min(max(self.angular_pid(diff_angle) * 3, -self.max_speed / 2), self.max_speed / 2)
            linear_vel = min(max(self.linear_pid(-distance), -self.max_speed), self.max_speed)
            # set speed left wheels
            left_vel = linear_vel + angular_vel
            right_vel = linear_vel - angular_vel
            check_speed = max(abs(left_vel), abs(right_vel))
            if check_speed > self.max_speed:
                left_vel = left_vel * self.max_speed / check_speed
                right_vel = right_vel * self.max_speed / check_speed

            if self.reset:
                return
            for wheel in self.wheels_left:
                self.services[wheel + "_vel"].call(left_vel)
            for wheel in self.wheels_right:
                self.services[wheel + "_vel"].call(right_vel)
            # rospy.loginfo("angle: {} distance: {} vel: angular: {} linear: {} left: {} right: {} diff: {} orientation: {}"\
            #     .format(np.rad2deg(angle), distance, angular_vel, linear_vel, left_vel, right_vel, np.rad2deg(diff_angle), np.rad2deg(self.orientation)))
            time.sleep(0.1)
        if stop_after_getting:
            self.stop_robot()

    # def get_pos(self):
    #     counter_problem = 0
    #     while self.pos[0] is None:
    #         if self.reset:
    #             return (None, None)
    #         rospy.logwarn("waiting for pos to be available")
    #         time.sleep(0.1)
    #         counter_problem += 1
    #         if counter_problem > 300:
    #             raise Exception('Probable shared memory issue happend')

        # return self.pos[:2]

    def imu_cb(self):
        self.orientation = quat2euler(
            imo_msg.orientation.x, imo_msg.orientation.y, imo_msg.orientation.z, imo_msg.orientation.w)[0]
        self.orientation_history.add_element(self.orientation)

    def get_laser_image(self):
        while self.scan_image is None:
            time.sleep(0.1)
        return np.expand_dims(self.scan_image, axis=2)

    def laser_cb(self):
        if self.reset:
            return
        ranges = self.sick_laser.getRangeImage()
        if len(ranges) == 0:
            print("laser is empty")

        angle_min = -self.sick_fov/2
        angle_max = self.sick_fov/2
        angle_increment = self.sick_fov/len(ranges)
        angle_increments = np.arange(float(angle_min) - float(angle_min),
                                     float(angle_max) - 0.001 - float(angle_min),
                                     float(angle_increment))
        ranges = np.asarray(ranges, dtype=float)
        remove_index = np.append(np.argwhere(ranges >= self.max_laser_range),
                                 np.argwhere(ranges <= self.sick_laser.getMinRange()))
        angle_increments = np.delete(angle_increments, remove_index)
        ranges = np.delete(ranges, remove_index)
        min_ranges = float("inf") if len(ranges) == 0 else np.min(ranges)
        if min_ranges < self.collision_distance:
            self.is_collided = True
            print("robot collided:")
        print ("ranges:{} min_ranges:{} max_laser_range:{}".format(ranges, min_ranges, self.max_laser_range))
        x = np.floor((self.width_laser_image / 2.0 - (self.height_laser_image / self.max_laser_range) * np.multiply(
            np.cos(angle_increments), ranges))).astype(np.int)
        y = np.floor((self.height_laser_image - 1 - (self.height_laser_image / self.max_laser_range) * np.multiply(
            np.sin(angle_increments), ranges))).astype(np.int)
        if len(x) > 0 and (np.max(x) >= self.width_laser_image or np.max(y) >= self.height_laser_image):
            print("problem, max x:{} max y:{}".format(np.max(x), np.max(y)))
        scan_image = np.zeros((self.height_laser_image, self.width_laser_image), dtype=np.uint8)
        scan_image[y, x] = 255
        self.scan_image_history.add_element(scan_image)
        self.scan_image = scan_image

    def position_cb(self):
        new_pos = self.gps.getValues()
        prev_pos = self.pos
        self.pos = (-new_pos[2], -new_pos[0], time.time())

        self.pos_history.add_element(self.pos)

        # calculate velocity
        if prev_pos[0] is not None:
            self.velocity_history[self.last_velocity_idx] = math.hypot(self.pos[1] - prev_pos[1],
                                                                       self.pos[0] - prev_pos[0]) / (
                                                                        self.pos[2] - prev_pos[2])
            self.last_velocity_idx = (self.last_velocity_idx + 1) % self.velocity_window




def reset_webots():
    subprocess.call('pkill -9 webots-bin', shell=True)
    subprocess.call('tmux kill-session -t webots', shell=True)
    subprocess.Popen(['tmux', 'new-session', '-d', '-s', 'webots'])
    subprocess.Popen(['tmux', 'send-keys', '-t', 'webots', 'source ~/.bashrc', 'C-m'])
    subprocess.Popen(['tmux', 'send-keys', '-t', "webots", "webots --mode=fast", "C-m"])

def ex4():
    supervisor = SupervisorClass.remote()
    ray.get(supervisor.create_robot.remote('person', 'Pioneer3at', "10 0 0", "10 0 0 0"))
    person = RobotClass.remote("person")
    while not ray.get(person.is_init.remote()):
        time.sleep(0.1)
    ray.get(supervisor.create_robot.remote('robot', 'Pioneer3at', "0 0 0", "10 0 0 0"))
    robot = RobotClass.remote("robot")
    b = robot.laser_cb.remote()
    supervisor.step.remote()
    robot.step.remote()
    person.step.remote()
    laser = ray.get(b)
    robot.__del__.remote()
    person.__del__.remote()

    supervisor.reset.remote()
    ray.shutdown(robot)
    ray.shutdown(person)
    ray.shutdown(supervisor)

cProfile.run('ex4()')
reset_webots()
# supervisor.reset()
# robotNode = supervisor.getFromDef('person')

# print(robotNode)


#
# freq = 0
#
# while supervisor.step(32) != -1:
#     if freq == 10:
#         translation = translationField.getSFVec3f()
#         print('Yellow position: %g %g %g\n' % (translation[0], translation[1], translation[2]))
#         freq = 0
#     freq += 1