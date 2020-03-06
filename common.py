#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import rospy
import random
import math

# Robot motion commands:
# http://docs.ros.org/api/geometry_msgs/html/msg/Twist.html
from geometry_msgs.msg import Twist
# Laser scan message:
# http://docs.ros.org/api/sensor_msgs/html/msg/LaserScan.html
from sensor_msgs.msg import LaserScan
# For groundtruth information.
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion

X = 0
Y = 1
YAW = 2

class MultiGroundtruthPose(object):
  def __init__(self, names):
    rospy.Subscriber('/gazebo/model_states', ModelStates, self.callback)
    poses = [np.array([np.nan, np.nan, np.nan], dtype=np.float32) for _ in range(len(names))]
    self._poses = dict(zip(names, poses))
    self._names = names

  def callback(self, msg):
    idx = [(i, n) for i, n in enumerate(msg.name) if n in self._names]
    if not idx:
      raise ValueError('Specified name "{}" does not exist.'.format(self._name))

    for id_, name in idx:
    
        self._poses[name][X] = msg.pose[id_].position.x
        self._poses[name][Y] = msg.pose[id_].position.y
        _, _, yaw = euler_from_quaternion([
            msg.pose[id_].orientation.x,
            msg.pose[id_].orientation.y,
            msg.pose[id_].orientation.z,
            msg.pose[id_].orientation.w])
        self._poses[name][YAW] = yaw

  @property
  def ready(self):
    return np.count_nonzero(np.isnan(self._poses.values())) == 0

  @property
  def poses(self):
    return self._poses

def get_velocity(position, path_points):
  v = np.zeros_like(position)
  if len(path_points) == 0:
    return v
  # Stop moving if the goal is reached.
  if np.linalg.norm(position - path_points[-1]) < .2:
    return v

  closestIndex = -1
  closest = 99999

  for pIndex in range(len(path_points)):
    pp = path_points[pIndex]

    d = np.linalg.norm(pp - position)
    if d < closest:
      closest = d
      closestIndex = pIndex

  if closestIndex == 0:
    d_prev = 9999
  else:
    d_prev = np.linalg.norm(path_points[closestIndex-1] - position)

  if closestIndex == len(path_points)-1:
    d_next = 9999
  else:
    d_next = np.linalg.norm(path_points[closestIndex+1] - position)

  if d_prev < d_next:
    v = path_points[closestIndex] - position
  else:
    v = path_points[closestIndex+1] - position

  v /= np.linalg.norm(v)
  v *= SPEED

  return v

