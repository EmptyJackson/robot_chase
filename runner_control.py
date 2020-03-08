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
from geometry_msgs.msg import PoseArray, Pose, Point

import matplotlib.pyplot as plt

X = 0
Y = 1
YAW = 2

def position_to_point(position):
  p = Point()
  p.x = position[X]
  p.y = position[Y]
  p.z = 0.
  return p

def create_pose_array(path):
  pose_path = []
  for position in path:
    p = Pose()
    p.position = position
    pose_path.append(p)
  path_msg = PoseArray(poses=pose_path)
  return path_msg

def run(args):
  runner_id = args.id
  rname = 'r' + str(runner_id)
  rospy.init_node('runner_control' + str(runner_id))

  runners = ['r0', 'r1', 'r2']
  chasers = ['c0', 'c1', 'c2']

  # Update control every 10 ms.
  rate_limiter = rospy.Rate(100)
  publisher = rospy.Publisher(rname+'/path', PoseArray, queue_size=1)
  # Keep track of groundtruth position for plotting purposes.
  groundtruth = MultiGroundtruthPose(names=(runners + chasers))

  occupancy_grid = get_occupancy_grid()

  pf_targets = {}
  for chaser in chasers:
    pf_targets[chaser] = [np.array([0, 0], dtype=np.float32), 1., 1.]
  
  potential_field = PotentialField(pf_targets)

  print('Runner control initialised.')
  while not rospy.is_shutdown():
    # Make sure all measurements are ready.
    if not groundtruth.ready:
      rate_limiter.sleep()
      continue

    for chaser in chasers:
      potential_field.update_target(chaser, groundtruth.poses[chaser][:2])

    pose = groundtruth.poses[rname]

    path, s, g = rrt_star_path(pose, np.array([6, 2]), occupancy_grid, potential_field, is_open=True)

    path_list = [position_to_point(node) for node in path]
    path_msg = create_pose_array(path_list)
    
    publisher.publish(path_msg)

    rate_limiter.sleep()
    


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runners control')
  parser.add_argument('--id', action='store', default='-1', help='Method.')
  args, unknown = parser.parse_known_args()
  try:
    run(args)
  except rospy.ROSInterruptException:
    pass