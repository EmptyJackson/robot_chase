#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as patches
import scipy.signal
import scipy.stats
import rospy
import random
import math

from std_msgs.msg import Header
from geometry_msgs.msg import Twist, Pose, Point
from sensor_msgs.msg import LaserScan
import nav_msgs.msg as ros_nav
from nav_msgs.msg import MapMetaData
# For groundtruth information.
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion

RVIZ_PUBLISH = True  # Publish particles for RViz

X = 0
Y = 1
YAW = 2

ROBOT_RADIUS = 0.105 / 2.
CYLINDER_POSITIONS = np.array(
  [[ 6., 6.], 
   [ 2., 2.],
   [-2.,-6.],
   [-6.,-2.]], dtype=np.float32)
CYLINDER_RADIUS = 1. + ROBOT_RADIUS + 0.05
WALL_WIDTH = ROBOT_RADIUS + 0.15 / 2.
DOOR_BOUNDS = np.array([1.25, 2.75], dtype=np.float32)
GRID_FREQ = 4.
ARENA_OFFSET = 8. - ROBOT_RADIUS
RESOLUTION = 0.05

FREE = 0
OCCUPIED = 2

CHASER_SPEED = 0.1
RUNNER_SPEED = 0.1
CAPTURE_DIST = 0.35

def position_to_point(position):
  p = Point()
  p.x = position[X]
  p.y = position[Y]
  p.z = 0.
  return p

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

def get_velocity(position, path_points, speed):
  v = np.zeros_like(position)
  if len(path_points) == 0:
    return v
  # Stop moving if the goal is reached.
  if np.linalg.norm(position - path_points[-1]) < .2:
    return v

  # Estimate position along path
  min_dist = np.inf
  best_pt = -1
  for i, pt in enumerate(path_points):
    dist = np.linalg.norm(pt - position)
    if dist < min_dist:
      min_dist = dist
      best_pt = i

  # Return velocity to next point
  speed = 0.1
  next_pt = min(best_pt+1, len(path_points)-1)
  v = path_points[next_pt] - position
  return v * speed / np.linalg.norm(v)

def feedback_linearized(pose, velocity, epsilon):
  u = 0.  # [m/s]
  w = 0.  # [rad/s] going counter-clockwise.

  theta = pose[YAW]
  dxp = velocity[0]
  dyp = velocity[1]

  u = dxp * np.cos(theta) + dyp * np.sin(theta)
  w = (-dxp * np.sin(theta) + dyp * np.cos(theta)) / epsilon

  return u, w

# Defines an occupancy grid.
class OccupancyGrid(object):
  def __init__(self, values, origin, resolution):
    # Add border
    for i in range(values.shape[0]):
      for j in [0, 1, -2, -1]:
        values[i][j] = values[i][j] = OCCUPIED
    for j in range(values.shape[1]):
      for i in [0, 1, -2, -1]:
        values[i][j] = values[i][j] = OCCUPIED
    self._original_values = values.copy()
    self._values = values.copy()
    # Inflate obstacles (using a convolution).
    inflated_grid = np.zeros_like(values)
    inflated_grid[values == OCCUPIED] = 1.
    w = 2 * int(ROBOT_RADIUS / resolution)
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

  def get_ros_message(self):
    header = Header()
    header.seq = 0
    header.stamp = rospy.Time.now()
    header.frame_id = '/odom'

    meta_data = MapMetaData()
    meta_data.resolution = self._resolution
    meta_data.width = self._values.shape[X]
    meta_data.height = self._values.shape[Y]

    p = Pose()
    p.position.x = self._origin[X]
    p.position.y = self._origin[Y]
    p.position.z = 0.01
    p.orientation.x = 0.
    p.orientation.y = 0.
    p.orientation.z = 0.
    p.orientation.w = 1.    
    meta_data.origin = p

    grid_msg = ros_nav.OccupancyGrid(header=header, info=meta_data, data=[])
    for row in self._values:
      for v in row:
        grid_msg.data.append(int(v * (100/OCCUPIED)))
    return grid_msg


occ_grid = None
def get_occupancy_grid():
  global occ_grid
  if occ_grid != None:
    return occ_grid

  dim = int(2 * ARENA_OFFSET / RESOLUTION)
  grid = np.zeros((dim, dim), dtype=np.int8)

  for x in range(dim):
    for y in range(dim):
      position = np.array([x, y])
      position = (position - (dim/2)) * RESOLUTION
      if collision(position):
        grid[x, y] = OCCUPIED

  occ_grid = OccupancyGrid(grid, [-ARENA_OFFSET, -ARENA_OFFSET, 0.], RESOLUTION)
  return occ_grid

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

class PotentialField:
  def __init__(self, targets, is_path=False, use_walls=False):
    # targets = {name : [position, sigma, magnitude], ...}
    self.targets = targets
    self.is_path = is_path
    self.use_walls = use_walls

  def sample(self, position, target_names=None):
    weight = 0.

    # Targets
    if target_names is None:
      target_names = self.targets.keys()
    for target_name in target_names:
      if not target_name in self.targets.keys():
        continue
      target = self.targets[target_name]
      if self.is_path:
        for path_point in target[0]:
          weight += scipy.stats.norm.pdf(
            np.linalg.norm(position - path_point), 0, target[1]) * target[2]
        if len(target) < 5:
          weight *= 5. / len(target)
      else:
        weight += scipy.stats.norm.pdf(
          np.linalg.norm(position - target[0]), 0, target[1]) * target[2]

    # Walls
    if self.use_walls:
      local_dist = np.max(np.abs((position % GRID_FREQ) - GRID_FREQ/2)) ** 2
      weight += local_dist / 12.
        
    return weight

  def update_target(self, name, position):
    self.targets[name][0] = position

  def add_target(self, name, target):
    self.targets[name] = target

  def rand_sample(self):
    s = 0
    maximum = 0
    for _ in range(1000):
      p = (np.random.random(2) * 2 - 1) * 8
      sample = self.sample(p)
      s += sample
      if sample > maximum:
        maximum = sample
    s /= 1000.
    return s, maximum


class SimpleLaser(object):
  def __init__(self):
    rospy.Subscriber('/c0/scan', LaserScan, self.callback)
    self._angles = [0., np.pi / 4., -np.pi / 4., np.pi / 2., -np.pi / 2.]
    self._width = np.pi / 180. * 10.  # 10 degrees cone of view.
    self._measurements = [float('inf')] * len(self._angles)
    self._indices = None

  def callback(self, msg):
    # Helper for angles.
    def _within(x, a, b):
      pi2 = np.pi * 2.
      x %= pi2
      a %= pi2
      b %= pi2
      if a < b:
        return a <= x and x <= b
      return a <= x or x <= b;

    # Compute indices the first time.
    if self._indices is None:
      self._indices = [[] for _ in range(len(self._angles))]
      for i, d in enumerate(msg.ranges):
        angle = msg.angle_min + i * msg.angle_increment
        for j, center_angle in enumerate(self._angles):
          if _within(angle, center_angle - self._width / 2., center_angle + self._width / 2.):
            self._indices[j].append(i)

    ranges = np.array(msg.ranges)
    for i, idx in enumerate(self._indices):
      # We do not take the minimum range of the cone but the 10-th percentile for robustness.
      self._measurements[i] = np.percentile(ranges[idx], 10)

  @property
  def ready(self):
    return not np.isnan(self._measurements[0])

  @property
  def measurements(self):
    return self._measurements


def braitenberg(front, front_left, front_right, left, right):
  s = np.tanh([left, front_left, front, front_right, right])
  s = [1 - x for x in s]

  u_weights = [0, -0.5, -1, -0.5, 0]
  w_weights = [-0.5, -1, 0, 1, 0.5]

  u = np.dot(s, u_weights) + 1.
  w = np.dot(s, w_weights)

  return u, w


if __name__=='__main__':
  get_occupancy_grid().draw()
  plt.show()
