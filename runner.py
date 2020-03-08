#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import rospy
import random
import math
from common import *
from rrt_improved import *

# Robot motion commands:
# http://docs.ros.org/api/geometry_msgs/html/msg/Twist.html
from geometry_msgs.msg import Twist
# Laser scan message:
# http://docs.ros.org/api/sensor_msgs/html/msg/LaserScan.html
from sensor_msgs.msg import LaserScan
# For groundtruth information.
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion

import matplotlib.pyplot as plt
from geometry_msgs.msg import PoseArray, Pose, Point


X = 0
Y = 1
YAW = 2


class PathSubscriber(object):
  def __init__(self, name='c0'):
    rospy.Subscriber('/'+name+'/path', PoseArray, self.callback)
    self._path = None

  def callback(self, msg):
    self._path = np.array([[pose.position.x, pose.position.y] for pose in msg.poses])

  @property
  def ready(self):
    return not self._path is None

  @property
  def path(self):
    return self._path

def run(args):
  runner_id = args.id
  rname = 'r' + str(runner_id)
  rospy.init_node('runner' + str(runner_id))

  # Update control every 10 ms.
  rate_limiter = rospy.Rate(100)
  publisher = rospy.Publisher(rname + '/cmd_vel', Twist, queue_size=5)
  # Keep track of groundtruth position for plotting purposes.
  groundtruth = MultiGroundtruthPose(names=([rname]))

  path_sub = PathSubscriber(rname)
  
  while not rospy.is_shutdown():
    # Make sure all measurements are ready.
    if not groundtruth.ready or not path_sub.ready:
      rate_limiter.sleep()
      continue

    pose = groundtruth.poses[rname]

    v = get_velocity(pose[:2], path_sub.path)
    u, w = feedback_linearized(pose, v, 0.1)
  

    vel_msg = Twist()
    vel_msg.linear.x = u
    vel_msg.angular.z = w
    publisher.publish(vel_msg)

    rate_limiter.sleep()
    


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runners control')
  parser.add_argument('--id', action='store', default='-1', help='Method.')
  args, unknown = parser.parse_known_args()
  try:
    run(args)
  except rospy.ROSInterruptException:
    pass
