#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import threading
import math
import numpy as np

def euclidean_distance(p1x, p1y, p2x, p2y):
    x_dist = p1x - p2x
    y_dist = p1y - p2y
    return math.sqrt(x_dist * x_dist + y_dist * y_dist)

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.image_lock = threading.RLock()
        self.lights = []

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()

        self.STATE_COUNT_THRESHOLD = rospy.get_param('~state_count_threshold')

        # Determine if using sim or real model. Edited the launch files in tl_detector
        use_simulator_classifier = rospy.get_param('~traffic_light_classifier_sim')
        self.light_classifier = TLClassifier(sim = use_simulator_classifier)
        self.listener = tf.TransformListener()

        self.pub1_deep_net_out = rospy.Publisher('/deep_net_out', Image, queue_size=1)
        self.TrafficLightState = rospy.Publisher('/traffic_light_state', Int32, queue_size=1)

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= self.STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        min_distance = float("inf")
        closest_index = 0
        if (self.waypoints):
            for wp_index in range(len(self.waypoints.waypoints)):
                waypoint_ps = self.waypoints.waypoints[wp_index].pose
                distance = euclidean_distance(pose.position.x,
                    pose.position.y,
                    waypoint_ps.pose.position.x,
                    waypoint_ps.pose.position.y)
                if (distance < min_distance):
                    min_distance = distance
                    closest_index = wp_index

        return closest_index

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if (not self.has_image):
            self.prev_light_loc = None
            return False

        state = TrafficLight.UNKNOWN  # Default state

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")

        state = self.light_classifier.get_classification(cv_image)

        #self.pub1_deep_net_out.publish(self.bridge.cv2_to_imgmsg(
        # self.light_classifier.image_np_deep, "rgb8"))

        self.TrafficLightState.publish(Int32(state))

        stop_line_positions = self.config['stop_line_positions']
        light_pose = Pose()
        closest_waypoint_to_light = -1

        if (self.pose and self.waypoints):
            if (self.pose and self.waypoints):
                closest_waypoint_index = self.get_closest_waypoint(self.pose.pose)
                closest_waypoint_ps = self.waypoints.waypoints[closest_waypoint_index].pose

                # find the closest visible traffic light (if one exists)
                closest_light_distance = float("inf")
                for light_position in stop_line_positions:
                    distance = euclidean_distance(light_position[0], light_position[1],
                                                  closest_waypoint_ps.pose.position.x,
                                                  closest_waypoint_ps.pose.position.y)
                    if distance < closest_light_distance:
                        closest_light_distance = distance
                        light_pose.position.x = light_position[0]
                        light_pose.position.y = light_position[1]

            closest_waypoint_to_light = self.get_closest_waypoint(light_pose)

        return closest_waypoint_to_light, state

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
