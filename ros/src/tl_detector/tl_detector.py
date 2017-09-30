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
import time

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

        self.best_waypoint = 0
        self.last_car_position = 0
        self.last_light_pos_wp = []

        self.FAR_LIGHT_DIST = 100.0

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

        #self.pub1_deep_net_out = rospy.Publisher('/deep_net_out', Image, queue_size=1)
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
        start_time = time.time()
        light_wp, state = self.process_traffic_lights()
        light_class = self.get_class_name(state)

        rospy.loginfo('image_cb - light_wp:: {} | light_class:: {} | state :: {} | time :: {}'.format(light_wp, light_class, Int32(state), (time.time() - start_time)))
        rospy.loginfo('\n')

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

    def get_class_name(self, code_l):
        if code_l == TrafficLight.RED:
            return "RED"
        elif code_l == TrafficLight.YELLOW:
            return "YELLOW"
        elif code_l == TrafficLight.GREEN:
            return "GREEN"
        else:
            return "UNKNOWN"

    def euclidean_distance(self, p1, p2):
        delta_x = p1.x - p2.x
        delta_y = p1.y - p2.y
        return math.sqrt(delta_x*delta_x + delta_y*delta_y)

    def euclidean_distance_light(self, light_p, waypoint_p):
        x_dist = light_p[0] - waypoint_p.x
        y_dist = light_p[1] - waypoint_p.y
        return math.sqrt(x_dist * x_dist + y_dist * y_dist)

    def get_closest_waypoint_light(self, way_point, light_pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            light_pose (Pose): position to match a waypoint to
            way_point : Waypoint

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        best_waypoint = None
        waypoints = way_point.waypoints
        min_dist = self.euclidean_distance_light(light_pose, waypoints[0].pose.pose.position)
        for i, point in enumerate(waypoints):
            dist = self.euclidean_distance_light(light_pose, point.pose.pose.position)
            if dist < min_dist:
                best_waypoint = i
                min_dist = dist
        return best_waypoint

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        best_waypoint = self.best_waypoint
        if self.waypoints is not None:
            waypoints = self.waypoints.waypoints
            min_dist = self.euclidean_distance(pose.position, waypoints[0].pose.pose.position)
            for i, point in enumerate(waypoints):
                dist = self.euclidean_distance(pose.position, point.pose.pose.position)
                if dist < min_dist:
                    best_waypoint = i
                    min_dist = dist
            self.best_waypoint = best_waypoint
            return best_waypoint

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")

        light_positions = self.config['stop_line_positions']

        light_pos_wp = []
        if self.waypoints is not None:
            wp = self.waypoints
            for i in range(len(light_positions)):
                l_pos = self.get_closest_waypoint_light(wp, light_positions[i])
                light_pos_wp.append(l_pos)
            self.last_light_pos_wp = light_pos_wp
        else:
            light_pos_wp = self.last_light_pos_wp

        if (self.pose):
            car_position = self.get_closest_waypoint(self.pose.pose)
            if car_position is not None:
                self.last_car_position = car_position

        if (self.last_car_position > max(light_pos_wp)):
            light_num_wp = min(light_pos_wp)
        else:
            light_delta = light_pos_wp[:]
            light_delta[:] = [x - self.last_car_position for x in light_delta]
            light_num_wp = min(i for i in light_delta if i > 0) + self.last_car_position

        light_idx = light_pos_wp.index(light_num_wp)
        light = light_positions[light_idx]

        light_distance = self.euclidean_distance_light(light, self.waypoints.waypoints[self.last_car_position].pose.pose.position)
        rospy.loginfo('light_distance:: {}'.format(light_distance))

        if light:
            if (light_distance >= self.FAR_LIGHT_DIST):
                return -1, TrafficLight.UNKNOWN
            else:
                state = self.light_classifier.get_classification(cv_image)
                return light_num_wp, state
        self.waypoints = None
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
