#!/usr/bin/env python
import rospy
import random
import argparse
import numpy as np
from common import *
from rrt_improved import *

from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import PoseArray, Pose, Point

OCC_GRID = get_occupancy_grid()
NUM_CLOUD_POINTS = 25
PUBLISH_PARTICLES = True  # Publish particles for RViz

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
        self._particles.add(Particle(self._max_speed, chasers, start_pos))

  def reset(self, chasers):
    self._particles = set()
    for _ in range(self._num_points):
      self._particles.add(Particle(self._max_speed, chasers))

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

  def get_positions(self):
    for particle in self._particles:
      yield particle.position

  def get_random_position(self):
    return random.sample(self._particles, 1)[0].position

  def get_central_position(self):
    mean_position = np.array([0., 0.], dtype=np.float32)
    for particle in self._particles:
      mean_position += particle.position
    mean_position /= self._num_points
    best_position = self._particles[0].position
    best_dist = np.linalg.norm(mean_position - best_position)
    for particle in self._particles[1:]:
      dist = np.linalg.norm(mean_position - particle.position)
      if best_dist > dist:
        best_position = particle.position
        best_dist = dist
    return best_position
  

def position_to_point(position):
  p = Point()
  p.x = position[X]
  p.y = position[Y]
  p.z = 0.
  return p

def simple(poses, allocations, runner_ests):
  paths = {}
  for c in ['c0', 'c1', 'c2']:
    goal = poses[allocations[c]][:2]
    paths[c] = [position_to_point(goal)]

  # Return paths as array of Point
  return paths

def rrt(poses, allocations, runner_ests):
  occupancy_grid = get_occupancy_grid()
  potential_field = PotentialField({}, is_path=True)
  paths = {}
  chasers = ['c0', 'c1', 'c2']
  for c_id, c in enumerate(chasers):
    start_pose = poses[c]
    # Estimate or get goal position
    target_runner = allocations[c]
    if runner_ests[target_runner] is None:
      goal_position = poses[target_runner][:2]
    else:
      # Sample random position from point cloud
      goal_position = runner_ests[target_runner].get_random_position()

      # Get potential field
      targets = []
      for other_c_id in range(c_id):
        if allocations[chasers[other_c_id]] == target_runner:
          targets.append(chasers[other_c_id])

    path, _, _ = rrt_star_path(start_pose, goal_position, occupancy_grid, potential_field, targets=targets)
    paths[c] = []
    potential_field.add_target(c, path)
    for point in path:
      paths[c].append(position_to_point(point))
  return paths

def create_pose_array(path):
  pose_path = []
  for position in path:
    p = Pose()
    p.position = position
    pose_path.append(p)
  path_msg = PoseArray(poses=pose_path)
  return path_msg

def run(args):
  rospy.init_node('chaser_control')
  nav_method = globals()[args.mode]

  # Update paths every 100 ms.
  rate_limiter = rospy.Rate(10)
  publishers = {'c'+str(i): rospy.Publisher('/c'+str(i)+'/path', PoseArray, queue_size=1) for i in range(3)}
  if PUBLISH_PARTICLES:
    particle_publisher = rospy.Publisher('/Particles', PointCloud, queue_size=1)

  gts = MultiGroundtruthPose(['c0', 'c1', 'c2', 'r0', 'r1', 'r2'])
  runner_ests = {'r0':None, 'r1':None, 'r2':None}
  last_seen = {'r0':None, 'r1':None, 'r2':None}
  
  potential_field = PotentialField(pf_targets)

  frame_id = 0
  print('Chaser controller initialized.')
  while not rospy.is_shutdown():
    # Make sure all groundtruths are ready
    if not gts.ready:
      rate_limiter.sleep()
      continue

    chaser_positions = []
    for c in ['c0', 'c1', 'c2']:
      chaser_positions.append(gts.poses[c][:2])

    # Update estimated runner positions
    r_positions = {}
    for r in ['r0', 'r1', 'r2']:
      r_pos = gts.poses[r][:2]
      visible = False
      for c_pos in chaser_positions:
        if in_line_of_sight(r_pos, c_pos):
          if not visible:
            print(r + ' at ' + str(r_pos) + 'found by chaser at ' + str(c_pos) + '.')
          runner_ests[r] = None
          last_seen[r] = r_pos
          visible = True
          break
      if not visible:
        if runner_ests[r] is None:
          if last_seen[r] is None:
            runner_ests[r] = ParticleCloud(NUM_CLOUD_POINTS, CHASER_SPEED, chaser_positions)
          else:
            runner_ests[r] = ParticleCloud(
              NUM_CLOUD_POINTS, CHASER_SPEED, chaser_positions, last_seen[r])
        else:
          runner_ests[r].update(chaser_positions)
        r_positions[r] = runner_ests[r].get_central_position()
      else:
        r_positions[r] = r_pos

    # Allocate chasers to runners
    least_dist = np.inf
    for r in ['r0', 'r1', 'r2']:
      for c in ['c0', 'c1', 'c2']:
        dist = np.linalg.norm(gts.poses[c][:2] - gts.poses[r][:2])
        if dist < least_dist:
          least_dist = dist
          target_runner = r

    allocations = {}
    for c in ['c0', 'c1', 'c2']:
      allocations[c] = target_runner

    # Calculate chaser paths
    paths = nav_method(gts.poses, allocations, runner_ests)

    # Publish chaser paths
    for c in ['c0', 'c1', 'c2']:
      path_msg = create_pose_array(paths[c])
      publishers[c].publish(path_msg)

    # Publish localization particles
    if PUBLISH_PARTICLES:
      particle_msg = PointCloud()
      particle_msg.header.seq = frame_id
      particle_msg.header.stamp = rospy.Time.now()
      particle_msg.header.frame_id = '/odom'
      for r_est in runner_ests:
        for p in r_est.get_positions:
          pt = Point32()
          pt.x = p[X]
          pt.y = p[Y]
          pt.z = .05
          particle_msg.points.append(pt)
      particle_publisher.publish(particle_msg)


    rate_limiter.sleep()
    frame_id += 1


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runs centralised chaser control')
  parser.add_argument('--mode', action='store', default='simple', help='Routing method.', choices=['simple', 'rrt'])
  args, unknown = parser.parse_known_args()
  try:
    run(args)
  except rospy.ROSInterruptException:
    pass
