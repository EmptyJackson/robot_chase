#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as patches
import scipy.signal
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

ROBOT_RADIUS = 0.105 / 2.
CYLINDER_POSITIONS = np.array(
  [[ 6., 6.], 
   [ 2., 2.],
   [-2.,-6.],
   [-6.,-2.]], dtype=np.float32)
CYLINDER_RADIUS = 1. + ROBOT_RADIUS
WALL_WIDTH = ROBOT_RADIUS + 0.15 / 2.
DOOR_BOUNDS = np.array([1.125, 2.875], dtype=np.float32)
GRID_FREQ = 4.
ARENA_OFFSET = 8. - ROBOT_RADIUS
RESOLUTION = 0.05

FREE = 0
OCCUPIED = 2

SPEED = 1
CAPTURE_DIST = 0.25

class MultiGroundtruthPose(object):
  def __init__(self, names=['c0', 'c1', 'c2', 'r0', 'r1', 'r2']):
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

def feedback_linearized(pose, velocity, epsilon):
  u = 0.  # [m/s]
  w = 0.  # [rad/s] going counter-clockwise.

  # MISSING: Implement feedback-linearization to follow the velocity
  # vector given as argument. Epsilon corresponds to the distance of
  # linearized point in front of the robot.

  theta = pose[YAW]
  dxp = velocity[0]
  dyp = velocity[1]

  u = dxp * np.cos(theta) + dyp * np.sin(theta)
  w = (-dxp * np.sin(theta) + dyp * np.cos(theta)) / epsilon

  return u, w

# Defines an occupancy grid.
class OccupancyGrid(object):
  def __init__(self, values, origin, resolution):
    self._original_values = values.copy()
    self._values = values.copy()
    # Inflate obstacles (using a convolution).
    inflated_grid = np.zeros_like(values)
    inflated_grid[values == OCCUPIED] = 1.
    w = 2 * int(ROBOT_RADIUS / resolution) + 1
    inflated_grid = scipy.signal.convolve2d(inflated_grid, np.ones((w, w)), mode='same')
    self._values[inflated_grid > 0.] = OCCUPIED
    self._origin = np.array(origin[:2], dtype=np.float32)
    self._origin -= resolution / 2.
    assert origin[YAW] == 0.
    self._resolution = resolution

  @property
  def values(self):
    return self._values

  @property
  def resolution(self):
    return self._resolution

  @property
  def origin(self):
    return self._origin

  def draw(self):
    plt.imshow(self._original_values.T, interpolation='none', origin='lower',
               extent=[self._origin[X],
                       self._origin[X] + self._values.shape[0] * self._resolution,
                       self._origin[Y],
                       self._origin[Y] + self._values.shape[1] * self._resolution])
    plt.set_cmap('gray_r')

  def get_index(self, position):
    idx = ((position - self._origin) / self._resolution).astype(np.int32)
    if len(idx.shape) == 2:
      idx[:, 0] = np.clip(idx[:, 0], 0, self._values.shape[0] - 1)
      idx[:, 1] = np.clip(idx[:, 1], 0, self._values.shape[1] - 1)
      return (idx[:, 0], idx[:, 1])
    idx[0] = np.clip(idx[0], 0, self._values.shape[0] - 1)
    idx[1] = np.clip(idx[1], 0, self._values.shape[1] - 1)
    return tuple(idx)

  def get_position(self, i, j):
    return np.array([i, j], dtype=np.float32) * self._resolution + self._origin

  def is_occupied(self, position):
    return self._values[self.get_index(position)] == OCCUPIED

  def is_free(self, position):
    return self._values[self.get_index(position)] == FREE


def get_occupancy_grid():
  dim = int(2 * ARENA_OFFSET / RESOLUTION)
  grid = np.zeros((dim, dim), dtype=np.int8)

  for x in range(dim):
    for y in range(dim):
      position = np.array([x, y])
      position = (position - (dim/2)) * RESOLUTION
      if collision(position):
        grid[x, y] = OCCUPIED
  return OccupancyGrid(grid, [-ARENA_OFFSET, -ARENA_OFFSET, 0.], RESOLUTION)

def collision(position):
  # Arena
  if np.any(np.abs(position) > ARENA_OFFSET):
    return True

  # Cylinders
  for c in CYLINDER_POSITIONS:
    if np.linalg.norm(c - position) < CYLINDER_RADIUS:
      return True

  # Grid
  rel_pos = position % GRID_FREQ
  on_edge = np.logical_or(
    rel_pos < WALL_WIDTH,
    rel_pos > GRID_FREQ - WALL_WIDTH)
  in_door = np.logical_and(
    rel_pos > DOOR_BOUNDS[0],
    rel_pos < DOOR_BOUNDS[1])
  tmp = in_door[0]
  in_door[0] = in_door[1]
  in_door[1] = tmp
  return np.any(np.logical_and(on_edge, np.logical_not(in_door)))

if __name__=='__main__':
  get_occupancy_grid().draw()
  plt.show()
