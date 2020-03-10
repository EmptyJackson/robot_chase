#!/usr/bin/env python
import rospy
import random
import argparse
import numpy as np
from common import *
from rrt_improved import *
import plots

import nav_msgs.msg as ros_nav
from nav_msgs.msg import Path
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud, ChannelFloat32
from geometry_msgs.msg import PoseArray, Pose, Point, PoseStamped

OCC_GRID = get_occupancy_grid()
NUM_CLOUD_POINTS = 25

CHASERS = ['c0', 'c1', 'c2']
RUNNERS = ['r0', 'r1', 'r2']

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

  def step(self, chasers):
    d = np.random.uniform(-1., 1., 2)
    speed = np.random.uniform(0., self._max_speed)
    self._position += speed * d / np.linalg.norm(d)
    return self.is_valid(chasers)

  @property
  def position(self):
    return self._position
  

class ParticleCloud:
  def __init__(self, num_points, max_speed, chasers, runner, start_pos=None):
    self._runner = runner
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
      invalid = not particle.step(chasers)
      if invalid:
        invalid_ps.add(particle)

    n_invalid = len(invalid_ps)
    if n_invalid == self._num_points:
      # Reset cloud when all particles invalidated
      self.reset(chasers)
      print('Runner ' + self._runner + ' was lost by chasers.')
    else:
      self._particles -= invalid_ps
      for _ in range(n_invalid):
        pos = np.copy(random.sample(self._particles, 1)[0].position)
        self._particles.add(Particle(self._max_speed, chasers, pos))

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
    best_dist = np.inf
    for particle in self._particles:
      dist = np.linalg.norm(mean_position - particle.position)
      if best_dist > dist:
        best_position = particle.position
        best_dist = dist
    return best_position

  @property
  def runner(self):
    return self._runner


def simple(poses, allocations, runner_ests):
  paths = {}
  for c in ['c0', 'c1', 'c2']:
    goal = poses[allocations[c]][:2]
    paths[c] = [position_to_point(goal)]

  # Return paths as array of Point
  return paths

def rrt(poses, allocations, runner_ests):
  path_tail_max = 20
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

    #plots.plot_field(potential_field, 8)

    path, s, g = rrt_star_path(start_pose, goal_position, occupancy_grid, potential_field, targets=targets)

    path_tail = path[-min(len(path), path_tail_max):]
    potential_field.add_target(c, [path_tail, 1., 1.])
    paths[c] = []
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

def capture_runner(r_name_msg):
  RUNNERS.remove(r_name_msg.data)

def run(args):
  rospy.init_node('chaser_control')
  nav_method = globals()[args.mode]

  # Update paths every 100 ms.
  rate_limiter = rospy.Rate(10)
  rospy.Subscriber('/captured_runners', String, capture_runner)
  publishers = {'c'+str(i): rospy.Publisher('/c'+str(i)+'/path', PoseArray, queue_size=1) for i in range(3)}
  if RVIZ_PUBLISH:
    runner_est_publisher = rospy.Publisher('/runner_particles', PointCloud, queue_size=1)
    path_publishers = {}
    for c in CHASERS:
      path_publishers[c] = rospy.Publisher('/'+c+'_path', Path, queue_size=1)

    map_publisher = rospy.Publisher('/map', ros_nav.OccupancyGrid, queue_size=1)
    map_publisher.publish(OCC_GRID.get_ros_message())

  gts = MultiGroundtruthPose(CHASERS + RUNNERS)
  runner_ests = {'r0':None, 'r1':None, 'r2':None}
  last_seen = {'r0':None, 'r1':None, 'r2':None}
  last_target = ''
  
  frame_id = 0
  print('Chaser controller initialized.')
  while not rospy.is_shutdown():
    # Make sure all groundtruths are ready
    if not gts.ready:
      rate_limiter.sleep()
      continue

    chaser_positions = []
    for c in CHASERS:
      chaser_positions.append(gts.poses[c][:2])

    # Update estimated runner positions
    for r in RUNNERS:
      r_pos = gts.poses[r][:2]
      visible = False
      for c_pos in chaser_positions:
        if in_line_of_sight(r_pos, c_pos):
          if not runner_ests[r] is None:
            print('Runner ' + r + ' found at ' + str(r_pos) + ' by chaser at ' + str(c_pos) + '.')
          runner_ests[r] = None
          last_seen[r] = r_pos
          visible = True
          break
      if not visible:
        if runner_ests[r] is None:
          if last_seen[r] is None:
            runner_ests[r] = ParticleCloud(NUM_CLOUD_POINTS, CHASER_SPEED, chaser_positions, r)
          else:
            runner_ests[r] = ParticleCloud(
              NUM_CLOUD_POINTS, CHASER_SPEED, chaser_positions, r, last_seen[r])
        else:
          runner_ests[r].update(chaser_positions)

    # Allocate chasers to runners
    least_dist = np.inf
    for r in RUNNERS:
      for c in CHASERS:
        if not runner_ests[r] is None:
          # Max distance from a point in the point cloud
          dist = np.max([np.linalg.norm(gts.poses[c][:2] - r_pos) for r_pos in runner_ests[r].get_positions()])
        else:
          dist = np.linalg.norm(gts.poses[c][:2] - gts.poses[r][:2])
        if dist < least_dist:
          least_dist = dist
          target_runner = r
    if last_target != target_runner:
      print('New target runner allocated: ' + target_runner)
      last_target = target_runner

    allocations = {}
    for c in CHASERS:
      allocations[c] = target_runner

    # Calculate chaser paths
    path_tail_max = 15
    occupancy_grid = get_occupancy_grid()
    potential_field = PotentialField({}, is_path=True)
    paths = {}
    chasers = ['c0', 'c1', 'c2']

    if not runner_ests[target_runner] is None:
      chasers_ordered = sorted(chasers,
        key=lambda x: np.max([np.linalg.norm(gts.poses[x][:2] - r_pos) for r_pos in runner_ests[r].get_positions()]))
    else:
      chasers_ordered = sorted(chasers, key=lambda x: np.linalg.norm(gts.poses[x][:2] - gts.poses[allocations[x]][:2]))
    
    for c in chasers_ordered:
      #plots.plot_field(potential_field, ARENA_OFFSET)
      c_id = int(c[1])
      start_pose = gts.poses[c]
      # Estimate or get goal position
      target_runner = allocations[c]
      if runner_ests[target_runner] is None:
        goal_position = gts.poses[target_runner][:2]
      else:
        # Sample closest position in point cloud
        min_dist = np.inf
        for r_pos in runner_ests[target_runner].get_positions():
          dist = np.linalg.norm(start_pose[:2] - r_pos[:2])
          if dist < min_dist:
            goal_position = r_pos
            min_dist = dist

      # Get potential field
      targets = []
      for other_c_id in range(3):
        if other_c_id == c_id:
          continue
        
        if allocations[chasers[other_c_id]] == target_runner:
          targets.append(chasers[other_c_id])

      
      path, s, g = rrt_star_path(start_pose, goal_position, occupancy_grid, potential_field, targets=targets)    

      path_tail = path[-min(len(path), path_tail_max):]
      potential_field.add_target(c, [path_tail, .5, 10.])
      
      path_arr = [position_to_point(point) for point in path]
      paths[c] = path_arr

      # Publish chaser paths
      path_msg = create_pose_array(path_arr)
      publishers[c].publish(path_msg)

      # Update particle clouds
      chaser_positions = []
      for c_name in CHASERS:
        chaser_positions.append(gts.poses[c_name][:2])

      for r in RUNNERS:
        r_pos = gts.poses[r][:2]
        visible = False
        for c_pos in chaser_positions:
          if in_line_of_sight(r_pos, c_pos):
            if not runner_ests[r] is None:
              print('Runner ' + r + ' found at ' + str(r_pos) + ' by chaser at ' + str(c_pos) + '.')
            runner_ests[r] = None
            last_seen[r] = r_pos
            visible = True
            break
        if not visible:
          if runner_ests[r] is None:
            if last_seen[r] is None:
              runner_ests[r] = ParticleCloud(NUM_CLOUD_POINTS, CHASER_SPEED, chaser_positions, r)
            else:
              runner_ests[r] = ParticleCloud(
                NUM_CLOUD_POINTS, CHASER_SPEED, chaser_positions, r, last_seen[r])
          else:
            runner_ests[r].update(chaser_positions)

      # Publish localization particles and chaser positions
      if RVIZ_PUBLISH:
        particle_msg = PointCloud()
        particle_msg.header.seq = frame_id
        particle_msg.header.stamp = rospy.Time.now()
        particle_msg.header.frame_id = '/odom'
        intensity_channel = ChannelFloat32()
        intensity_channel.name = 'intensity'
        particle_msg.channels.append(intensity_channel)
        for r in RUNNERS:
          r_est = runner_ests[r]
          if not r_est is None:
            for p in r_est.get_positions():
              pt = Point()
              pt.x = p[X]
              pt.y = p[Y]
              pt.z = .05
              particle_msg.points.append(pt)
              intensity_channel.values.append(int(r[1]))
        runner_est_publisher.publish(particle_msg)

        path_msg = Path()
        path_msg.header.seq = frame_id
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = '/odom'
        for position in paths[c]:
          ps = PoseStamped()
          ps.header.seq = frame_id
          ps.header.stamp = rospy.Time.now()
          ps.header.frame_id = '/odom'
          p = Pose()
          p.position = position
          ps.pose = p
          path_msg.poses.append(ps)
        path_publishers[c].publish(path_msg)

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
