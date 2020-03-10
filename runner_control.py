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
import plots

# Robot motion commands:
# http://docs.ros.org/api/geometry_msgs/html/msg/Twist.html
from geometry_msgs.msg import Twist
# Laser scan message:
# http://docs.ros.org/api/sensor_msgs/html/msg/LaserScan.html
from sensor_msgs.msg import LaserScan
# For groundtruth information.
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import PoseArray, Pose, Point, PoseStamped
from nav_msgs.msg import Path

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

  other_runners = ['r0', 'r1', 'r2']
  other_runners.remove(rname)
  chasers = ['c0', 'c1', 'c2']

  # Update control every 10 ms.
  rate_limiter = rospy.Rate(100)
  publisher = rospy.Publisher(rname+'/path', PoseArray, queue_size=1)
  if RVIZ_PUBLISH:
    rviz_publisher = rospy.Publisher('/'+rname+'_path', Path, queue_size=1)
  # Keep track of groundtruth position for plotting purposes.
  groundtruth = MultiGroundtruthPose(names=(other_runners + chasers + [rname]))

  occupancy_grid = get_occupancy_grid()

  pf_targets = {}
  for chaser in chasers:
    pf_targets[chaser] = [np.array([0, 0], dtype=np.float32), 3., 8.]

  for runner in other_runners:
    pf_targets[runner] = [np.array([0, 0], dtype=np.float32), 1., 1]
  
  potential_field = PotentialField(pf_targets, use_walls=True)

  path = None

  frame_id = 0
  print('Runner control initialised.')
  while not rospy.is_shutdown():
    # Make sure all measurements are ready.
    if not groundtruth.ready:
      rate_limiter.sleep()
      continue

    for chaser in chasers:
      potential_field.update_target(chaser, groundtruth.poses[chaser][:2])

    for runner in other_runners:
      potential_field.update_target(runner, groundtruth.poses[runner][:2])

    #if runner_id == '0':
    #  plots.plot_field(potential_field, 8)

    pose = groundtruth.poses[rname]

    if path is None:
      start_pose = pose
    else:
      path_point = path[min(len(path)-1, 4)]
      start_pose = np.concatenate((path_point, [pose[2]]))
      print('!')

    path, s, g = rrt_star_path(start_pose, np.array([6, 2]), occupancy_grid, potential_field, is_open=True)

    path_list = [position_to_point(node) for node in path]
    path_msg = create_pose_array(path_list)
    
    publisher.publish(path_msg)


    if RVIZ_PUBLISH:
      path_msg = Path()
      path_msg.header.seq = frame_id
      path_msg.header.stamp = rospy.Time.now()
      path_msg.header.frame_id = '/odom'
      for position in path:
        pt = position_to_point(position)
        ps = PoseStamped()
        ps.header.seq = frame_id
        ps.header.stamp = rospy.Time.now()
        ps.header.frame_id = '/odom'
        p = Pose()
        p.position = pt
        ps.pose = p
        path_msg.poses.append(ps)
      rviz_publisher.publish(path_msg)

    rate_limiter.sleep()
    frame_id += 1
    


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runners control')
  parser.add_argument('--id', action='store', default='-1', help='Method.')
  args, unknown = parser.parse_known_args()
  try:
    run(args)
  except rospy.ROSInterruptException:
    pass
