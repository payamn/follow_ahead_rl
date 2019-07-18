import gym
import gym_vrep
# def tele_robot():
#     img = np.zeros((500, 500, 1), np.uint8)
#     cv.imshow("control", img)
#
#     while True:
#
#         k = cv.waitKey(0)
#
#         if k==ord('w'):
#             v.robot_handler.move_robot(0)
#         elif k==ord('a'):
#             v.robot_handler.move_robot(1)
#         elif k==ord('d'):
#             v.robot_handler.move_robot(2)
#         elif k==ord('s'):
#             v.robot_handler.move_robot(3)
#         elif k == ord('q'):
#             v.robot_handler.move_robot(4)
#         elif k == ord('e'):
#             v.robot_handler.move_robot(5)
#         elif k == ord('z'):
#             v.robot_handler.move_robot(6)
#         time.sleep(0.1)


if __name__ == "__main__":
    env = gym.make('vrep-v0')
    # v.start_simulation()
    # v.remove_robot()
    # v.stop_simulation()
    # v.move_robot((-1., 1.2000, .14))

