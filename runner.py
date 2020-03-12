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
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion

import matplotlib.pyplot as plt
from geometry_msgs.msg import PoseArray, Pose, Point, PointStamped


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
  capture_publisher = rospy.Publisher('/captured_runners', String, queue_size=1)
  if RVIZ_PUBLISH:
    rviz_publisher = rospy.Publisher('/' + rname + '_position', PointStamped, queue_size=1)

  # Keep track of groundtruth position for plotting purposes.
  groundtruth = MultiGroundtruthPose(names=(['c0', 'c1', 'c2', rname]))

  rev_frames = 0
  had_path = False

  path_sub = PathSubscriber(rname)
  frame_id = 0
  init_publish = False
  while not rospy.is_shutdown():
    # Make sure all measurements are ready.
    if not groundtruth.ready:
      rate_limiter.sleep()
      continue

    pose = groundtruth.poses[rname]

    # Publish initial position
    if not init_publish or not path_sub.ready:
      if RVIZ_PUBLISH:
        position_msg = PointStamped()
        position_msg.header.seq = frame_id
        position_msg.header.stamp = rospy.Time.now()
        position_msg.header.frame_id = '/odom'
        pt = Point()
        pt.x = pose[X]
        pt.y = pose[Y]
        pt.z = .05
        position_msg.point = pt
        rviz_publisher.publish(position_msg)
      init_publish = True
      rate_limiter.sleep()
      continue

    for c in ['c0', 'c1', 'c2']:
      if np.linalg.norm(groundtruth.poses[c][:2] - pose[:2]) < CAPTURE_DIST:
        print('Runner ', rname, ' captured by chaser ', c, 'at time', rospy.get_time())
        s = String()
        s.data = rname
        capture_publisher.publish(s)
        return

    path = path_sub.path

    if path is None or len(path) == 0:
      if had_path:
        rev_frames = 50

    else:
      had_path = True

    v = get_velocity(pose[:2], path, RUNNER_SPEED)
    u, w = feedback_linearized(pose, v, 0.1)

    if rev_frames > 0:
      u = -CHASER_SPEED
      w = 0

    vel_msg = Twist()
    vel_msg.linear.x = u
    vel_msg.angular.z = w
    publisher.publish(vel_msg)
    if RVIZ_PUBLISH:
      position_msg = PointStamped()
      position_msg.header.seq = frame_id
      position_msg.header.stamp = rospy.Time.now()
      position_msg.header.frame_id = '/odom'
      pt = Point()
      pt.x = pose[X]
      pt.y = pose[Y]
      pt.z = .05
      position_msg.point = pt
      rviz_publisher.publish(position_msg)

    rate_limiter.sleep()
    frame_id += 1

    if rev_frames > 0:
      rev_frames -= 1


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runners control')
  parser.add_argument('--id', action='store', default='-1', help='Method.')
  args, unknown = parser.parse_known_args()
  try:
    run(args)
  except rospy.ROSInterruptException:
    pass
