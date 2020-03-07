#!/usr/bin/env python
import rospy
import random
import argparse
import numpy as np
from common import *

from geometry_msgs.msg import PoseArray, Pose, Point

OCC_GRID = get_occupancy_grid()
NUM_CLOUD_POINTS = 25

def in_line_of_sight(p1, p2):
  sample_rate = 10 # Samples per meter
  n_samples = np.linalg.norm(p1 - p2) * sample_rate
  x = np.linspace(p1[X], p2[X], n_samples)
  y = np.linspace(p1[Y], p2[Y], n_samples)
  for coord in zip(x, y):
    if OCC_GRID.is_occupied(coord):
      return False
  return True

class Particle:
  def __init__(self, max_speed, chasers, position=None):
    self._max_speed = max_speed
    if not position is None:
      self._position = position
    else:
      self._position = np.random.uniform(-ARENA_OFFSET, ARENA_OFFSET, 2)
      while not self.is_valid(chasers):
        self._position = np.random.uniform(-ARENA_OFFSET, ARENA_OFFSET, 2)

  def is_valid(self, chasers):
    if not OCC_GRID.is_free(self._position):
      return False
    for c in chasers:
      if in_line_of_sight(self._position, c):
        return False
    return True

  def step(self):
    d = np.random.uniform(-1., 1., 2)
    speed = np.random.uniform(0., self._max_speed)
    return speed * d / np.linalg.norm(d)

  @property
  def position(self):
    return self._position
  

class ParticleCloud:
  def __init__(self, num_points, max_speed, chasers, start_pos=None):
    self._num_points = num_points
    self._max_speed = max_speed
    if start_pos is None:
      self.reset(chasers)
    else:
      self._particles = set()
      for _ in range(self._num_points):
        self._particles.append(Particle(self._max_speed, chasers, start_pos))

  def reset(self, chasers):
    self._particles = set()
    for _ in range(self._num_points):
      self._particles.append(Particle(self._max_speed, chasers))

  def update(self, chasers):
    invalid_ps = set()
    for particle in self._particles:
      invalid = particle.step(chasers)
      if invalid:
        invalid_ps.add(particle)

    n_invalid = len(invalid_ps)
    if n_invalid == self._num_points:
      # Reset cloud when all particles invalidated
      self.reset()
    else:
      self._particles.remove(invalid_ps)
      for _ in range(n_invalid):
        pos = random.sample(self._particles, 1)[0].position
        self._particles.add(particle(self._max_speed, chasers, pos))


def simple(poses):
  # Return paths as array of Point
  pass

def create_pose_array(path):
  pose_path = []
  for position in path:
    p = Pose()
    p.position = position
    pose_path.append(p)
  path_msg = PoseArray()
  path_msg.poses = pose_path
  return path_msg

def run(args):
  rospy.init_node('chaser_control')
  nav_method = globals()[args.mode]

  # Update paths every 100 ms.
  rate_limiter = rospy.Rate(10)
  publishers = [rospy.Publisher('/c'+str(i)+'/path', PoseArray, queue_size=1) for i in range(3)]

  gts = MultiGroundtruthPose(['c0', 'c1', 'c2', 'r0', 'r1', 'r2'])
  runner_ests = {'r0':None, 'r1':None, 'r2':None}
  last_seen = {'r0':None, 'r1':None, 'r2':None}
  while not rospy.is_shutdown():
    # Make sure all groundtruths are ready.
    if not gts.ready:
      rate_limiter.sleep()
      continue

    chaser_positions = []
    for c in ['c0', 'c1', 'c2']:
      chaser_positions = gts.poses[c][:2]

    for r in ['r0', 'r1', 'r2']:
      r_pos = gts.poses[r][:2]
      visible = False
      for c_pos in chaser_positions:
        if in_line_of_sight(r_pos, c_pos):
          runner_ests[r] = None
          visible = True
          break
      if visible:
        runner_ests[r] = None
      else:
        if runner_ests[r] is None:
          if last_seen[r] is None:
            runner_ests[r] = ParticleCloud(NUM_CLOUD_POINTS, CHASER_SPEED, chaser_positions)
          else:
            runner_ests[r] = ParticleCloud(
              NUM_CLOUD_POINTS, CHASER_SPEED, chaser_positions, last_seen[r])

    # todo: Calculate paths
    paths = nav_method(gts.poses)

    for path, publisher in zip(paths, publishers):
      path_msg = create_pose_array(path)
      publisher.publish(path_msg)

    rate_limiter.sleep()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runs centralised chaser control')
  parser.add_argument('--mode', action='store', default='simple', help='Routing method.', choices=['simple'])
  args, unknown = parser.parse_known_args()
  try:
    run(args)
  except rospy.ROSInterruptException:
    pass
