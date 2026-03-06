
import cv2
import gym
import math
import rospy
import roslaunch
import time
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import Image
from time import sleep

from gym.utils import seeding


class Gazebo_Linefollow_Env(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        LAUNCH_FILE = '/home/fizzer/enph353_gym-gazebo-noetic/gym_gazebo/envs/ros_ws/src/linefollow_ros/launch/linefollow_world.launch'
        gazebo_env.GazeboEnv.__init__(self, LAUNCH_FILE)
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world',
                                              Empty)

        self.action_space = spaces.Discrete(3)  # F,L,R
        self.reward_range = (-np.inf, np.inf)
        self.episode_history = []

        self._seed()

        self.bridge = CvBridge()
        self.timeout = 0  # Used to keep track of images with no line detected


    def process_image(self, data):
        '''
            @brief Coverts data into a opencv image and displays it
            @param data : Image data from ROS

            @retval (state, done)
        '''
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # cv2.imshow("raw", cv_image)

        NUM_BINS = 10
        state = [0]*20
        done = False

        # TODO: Analyze the cv_image and compute the state array and
        # episode termination condition.
        #
        # The state array is a list of 3 elements indicating where in the
        # image the line is:
        # i.e.
        #    [1, 0, 0] indicates line is on the left
        #    [0, 1, 0] indicates line is in the center
        #
        # The episode termination condition should be triggered when the line
        # is not detected for more than 1 frame. In this case set the done
        # variable to True.
        #
        # You can use the self.timeout variable to keep track of which frames
        # have no line detected.

        height, width, _ =cv_image.shape

        def get_roi_state(roi, width):
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            _, thresh_img = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            row_state = [0] * 10
            cX = None
            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    bin_index = min(int(cX / (width / 10)), 9)
                    row_state[bin_index] = 1
            return row_state, cX
        
        lower_roi = cv_image[int(3*height/4):height, 0:width]
        lower_state, lower_cX = get_roi_state(lower_roi, width)

        mid_roi = cv_image[int(height/2):int(3*height/4), 0:width]
        mid_state, mid_cX = get_roi_state(mid_roi, width)

        state = lower_state + mid_state

        line_detected = 1 in lower_state
        if not line_detected:
            if self.timeout > 3:
                done = True
            else:
                self.timeout += 1
        else:
            self.timeout = 0

        if lower_cX is not None:
            cv2.circle(cv_image, (lower_cX, int(7*height/8)), 5, (0, 0, 255), -1)
        
        lower_text = str(lower_state)
        mid_text = str(mid_state)
        lower_size = cv2.getTextSize(lower_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        mid_size = cv2.getTextSize(mid_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.putText(cv_image, mid_text, (width - mid_size[0] - 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(cv_image, lower_text, (width - lower_size[0] - 10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        

        cv2.imshow("Original Feed", cv_image)
        cv2.waitKey(1)

        return state, done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        self.episode_history.append(action)

        vel_cmd = Twist()

        if action == 0:  # FORWARD
            vel_cmd.linear.x = 0.4
            vel_cmd.angular.z = 0.0
        elif action == 1:  # LEFT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.5
        elif action == 2:  # RIGHT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = -0.5

        self.vel_pub.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw', Image,
                                              timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state, done = self.process_image(data)

        # Set the rewards for your action
            ##try:
                ##    line_pos = state.index(1)
                ##except ValueError:
                ##    line_pos = 0
            ##if line_pos in [4, 5]:
            ##    reward = 20
            ##elif line_pos in [3,6]:
            ##    reward = 10
            ##elif line_pos in [2,7]:
            ##    reward = 2

            ##if action == 0:  # FORWARD
            ##    reward = 20
            ##elif action == 1:  # LEFT
            ##    reward = 5
            ##else:
            ##    reward = 5  # RIGHT

        if not done:
            line_bin = state[:10].index(1) if 1 in state[:10] else -1

            if line_bin == -1:
                reward = -50
            else:
                distance_from_center = abs(line_bin - 4.5)
                if action == 0: # Forward
                    reward = 20 - (distance_from_center * 4)
                elif action == 1: # LEFT
                    if line_bin < 4:
                        reward = 15
                    else:
                        reward = -10
                else:  # Right
                    if line_bin > 5:
                        reward = 15
                    else: reward = -10
            
        else:
            reward = -200

        return state, reward, done, {}

    def reset(self):

        print("Episode history: {}".format(self.episode_history))
        self.episode_history = []
        print("Resetting simulation...")
        # Resets the state of the environment and returns an initial
        # observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            # reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            # resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        # read image data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw',
                                              Image, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        self.timeout = 0
        state, done = self.process_image(data)

        return state
