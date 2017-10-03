#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint

from math import cos, sin, sqrt
from tf.transformations import euler_from_quaternion

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200  # number of waypoints we will publish. You can change this number
MPH_TO_MPS = 0.44704  # simple conversion macro
NUM_SLOW_WPS = 100


class WaypointUpdater(object):
    def __init__(self):
        """
        Initializes the WaypointUpdater class
        """
        # initialize the node with ROS
        rospy.init_node('waypoint_updater')

        # Get the maximum speed
        self.max_speed = rospy.get_param('~max_speed') * MPH_TO_MPS
        assert self.max_speed is not None

        # subscribe to all relevant topics
        self.sub_cur_pose = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        self.sub_cur_vel  = rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb, queue_size=1)
        self.sub_base_wp  = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)
        self.sub_traffic_wp = rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb, queue_size=1)
        # TODO: Subscribe to /traffic_waypoint and /obstacle_waypoint

        # setup the publishers
        self.pub_final_waypoints = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # initialize the globals
        self.time = None
        self.pose = None
        self.lin_vel = 0.0
        self.ang_vel = 0.0
        self.base_wp = None
        self.base_wp_len = None
        self.traffic_wp = -1

        # start a permanent spin
        rospy.spin()

    def pose_cb(self, msg):
        """
        Processes the reception of the /current_pose message. This function is also responsible for generating the
        next set of waypoints required for the /final_waypoints and publish the message.

        :param msg: A PoseStamped object
        :return: None
        """
        if self.base_wp is None:
            pass

        self.time = msg.header.stamp
        self.pose = msg.pose

        # get closest waypoint
        index = self.__get_closest_waypoint()

        # Publish the new waypoints
        self.__publish_waypoints(index, msg.header.frame_id)

    def waypoints_cb(self, msg):
        """
        Processes the reception of the /base_waypoints message.

        :param msg: A Lane object
        :return: None
        """
        # Copy the waypoints
        self.base_wp = msg.waypoints
        self.base_wp_len = len(self.base_wp)

        # Unsubscribe, since this is no longer requried
        self.sub_base_wp.unregister()


    def velocity_cb(self, msg):
        """
        Processes the reception of the /current_velocity message

        :param msg: A TwistStamped object
        :return: None
        """
        self.lin_vel = msg.twist.linear.x
        self.ang_vel = msg.twist.angular.z

    def traffic_cb(self, msg):

        # Preconditions
        if self.base_wp is None:
            pass

        self.traffic_wp = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def __publish_waypoints(self, index, frame_id):
        # make list of n waypoints ahead of vehicle
        new_wps = self.__get_next_waypoints(index, LOOKAHEAD_WPS)
        assert len(new_wps) == LOOKAHEAD_WPS, 'Next waypoints error'

        # Find the last index
        last_index = index + LOOKAHEAD_WPS

        # Traffic waypoint is not set or is too far away
        if (self.traffic_wp == -1) or (self.traffic_wp > (last_index % self.base_wp_len)) or (index > self.traffic_wp):
            self.__set_all_wp(new_wps, self.max_speed)

        # Traffic waypoint is set
        else:
            diff = self.max_speed / float(NUM_SLOW_WPS)

            # For all the points beyond the traffic waypoint, set it to 0.
            for i in range(self.traffic_wp, last_index):
                new_wps[i - index].twist.twist.linear.x = 0

            # For all the points before the slow down, set it to max
            for i in range(index, (self.traffic_wp - NUM_SLOW_WPS)):
                new_wps[i - index].twist.twist.linear.x = self.max_speed

            # For all the points from slow down to the traffic waypoint,
            # set the value as it's required value
            for i in range((self.traffic_wp - NUM_SLOW_WPS), self.traffic_wp):
                if i >= index:
                    new_wps[i - index].twist.twist.linear.x = (self.traffic_wp - i - 1) * diff

        rospy.loginfo('{} :: {} - {} - {}'.format((self.traffic_wp - index),
                                                  new_wps[0].twist.twist.linear.x,
                                                  new_wps[LOOKAHEAD_WPS//2].twist.twist.linear.x,
                                                  new_wps[-1].twist.twist.linear.x))

        # make lane data structure to be published
        lane = self.__make_lane(frame_id, new_wps)

        # publish the subset of waypoints ahead
        self.pub_final_waypoints.publish(lane)
        

    @staticmethod
    def __set_all_wp(wps, vel):
        """
        Sets a constant velocity to all waypoints

        :param wps: A list of waypoints
        :param vel: The velocity to be set
        :return: None
        """
        for wp in wps:
            wp.twist.twist.linear.x = vel


    def __get_closest_waypoint(self):
        """
        :return: Returns the closest waypoint ahead of the car
        """
        best_gap = float('inf')
        best_index = 0
        my_position = self.pose.position

        for i, waypoint in enumerate(self.base_wp):
            other_position = waypoint.pose.pose.position
            gap = self.__distance(my_position, other_position)

            if gap < best_gap:
                best_index, best_gap = i, gap


        if self.__is_waypoint_behind(self.base_wp[best_index]):
            best_index += 1
        return best_index

    def __is_waypoint_behind(self, waypoint):
        """
        Checks if a given waypoint is behind the car's current position based on it's orientation

        :param waypoint: The waypoint
        :return: True / False
        """
        # Get the yaw from the quaternion
        _, _, yaw = euler_from_quaternion([self.pose.orientation.x,
                                           self.pose.orientation.y,
                                           self.pose.orientation.z,
                                           self.pose.orientation.w])

        origin_x = self.pose.position.x
        origin_y = self.pose.position.y

        shift_x = waypoint.pose.pose.position.x - origin_x
        shift_y = waypoint.pose.pose.position.y - origin_y

        x = (shift_x * cos(0.0 - yaw)) - (shift_y * sin(0.0 - yaw))

        if x > 0.0:
            return False
        return True

    def __get_next_waypoints(self, i, n):
        """
        Gets the next n waypoints starting from i
        :param i: The start location
        :param n: The number of waypoints to get
        :return: A list of waypoints from base_wp
        """
        if (i + n < len(self.base_wp)):
            return self.base_wp[i:(i + n)]
        else:
            return self.base_wp[i:] + self.base_wp[:(i + n) - len(self.base_wp)]

    @staticmethod
    def __make_lane(frame_id, waypoints):
        """
        Makes a lane object given the frame_id and waypoints

        :param frame_id: The sequence number to be used
        :param waypoints: A set of waypoints to be published
        :return: A Lane object
        """
        lane = Lane()
        lane.header.frame_id = frame_id
        lane.header.stamp = rospy.Time.now()
        lane.waypoints = waypoints
        return lane

    @staticmethod
    def __distance(a, b):
        """
        Returns the distance between two positions
        :param a: Position 1
        :param b: Position 2
        :return: Float32 distance between two positions
        """
        return sqrt(((a.x - b.x) ** 2) + ((a.y - b.y) ** 2))

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
